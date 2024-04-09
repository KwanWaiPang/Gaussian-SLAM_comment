""" This module includes the Mapper class, which is responsible scene mapping: Paragraph 3.2  """
import time
from argparse import ArgumentParser

import numpy as np
import torch
import torchvision

from src.entities.arguments import OptimizationParams
from src.entities.datasets import TUM_RGBD, BaseDataset, ScanNet
from src.entities.gaussian_model import GaussianModel
from src.entities.logger import Logger
from src.entities.losses import isotropic_loss, l1_loss, ssim
from src.utils.mapper_utils import (calc_psnr, compute_camera_frustum_corners,
                                    compute_frustum_point_ids,
                                    compute_new_points_ids,
                                    compute_opt_views_distribution,
                                    create_point_cloud, geometric_edge_mask,
                                    sample_pixels_based_on_gradient)
from src.utils.utils import (get_render_settings, np2ptcloud, np2torch,
                             render_gaussian_model, torch2np)
from src.utils.vis_utils import *  # noqa - needed for debugging


class Mapper(object):
    def __init__(self, config: dict, dataset: BaseDataset, logger: Logger) -> None:
        """ Sets up the mapper parameters
        Args:
            config: configuration of the mapper
            dataset: The dataset object used for extracting camera parameters and reading the data
            logger: The logger object used for logging the mapping process and saving visualizations
        """
        self.config = config
        self.logger = logger
        self.dataset = dataset
        self.iterations = config["iterations"]
        self.new_submap_iterations = config["new_submap_iterations"]
        self.new_submap_points_num = config["new_submap_points_num"]
        self.new_submap_gradient_points_num = config["new_submap_gradient_points_num"]
        self.new_frame_sample_size = config["new_frame_sample_size"]
        self.new_points_radius = config["new_points_radius"]
        self.alpha_thre = config["alpha_thre"]
        self.pruning_thre = config["pruning_thre"]
        self.current_view_opt_iterations = config["current_view_opt_iterations"]
        self.opt = OptimizationParams(ArgumentParser(description="Training script parameters"))
        self.keyframes = []

    # 计算一个二进制掩码，用于确定关键帧中应该生成新的高斯模型的区域
    def compute_seeding_mask(self, gaussian_model: GaussianModel, keyframe: dict, new_submap: bool) -> np.ndarray:
        """
        Computes a binary mask to identify regions within a keyframe where new Gaussian models should be seeded
        based on alpha masks or color gradient
        Args:
            gaussian_model: The current submap（当前子地图对应的高斯模型）
            keyframe (dict): Keyframe dict containing color, depth, and render settings （包含了颜色、深度和渲染设置的关键帧字典）
            new_submap (bool): A boolean indicating whether the seeding is occurring in current submap or a new submap
        Returns:
            np.ndarray: A binary mask of shpae (H, W) indicates regions suitable for seeding new 3D Gaussian models
        """
        # 初始化了一个变量 seeding_mask，用于存储生成的种子点掩码。
        seeding_mask = None
        
        # 如果是新的子地图，那么就使用几何边缘掩码
        if new_submap:
            # 将关键帧的颜色图像转换为 numpy 数组，并将值缩放到 [0, 255] 的范围内。
            color_for_mask = (torch2np(keyframe["color"].permute(1, 2, 0)) * 255).astype(np.uint8)

            # 调用 geometric_edge_mask 函数生成几何边缘掩码，用于确定适合生成新高斯模型的区域。
            # 函数中实现的为提取边缘轮廓,并对轮廓进行膨胀处理
            seeding_mask = geometric_edge_mask(color_for_mask, RGB=True)
        else: #如果不是新的子地图，那么就执行下面的操作
            # 调用 render_gaussian_model 函数渲染当前子地图，并得到渲染结果的字典。
            # 返回渲染后的颜色、深度、半径,2D均值与alpha等信息
            render_dict = render_gaussian_model(gaussian_model, keyframe["render_settings"])

            # 根据渲染的结果，生成 alpha 掩码和深度误差掩码。
            # 根据渲染结果中的 alpha 通道，生成了一个 alpha 掩码。这个掩码是根据设定的阈值 self.alpha_thre 对 alpha 值进行比较而得到的。
            alpha_mask = (render_dict["alpha"] < self.alpha_thre)

            # 从关键帧中获取了深度图像，并将其转换为张量格式。
            gt_depth_tensor = keyframe["depth"][None]
            # 计算了深度误差，即关键帧深度图像与渲染深度图像之间的差异。同时，根据关键帧深度图像中大于零的部分生成了一个掩码，以忽略无效深度值的影响。
            depth_error = torch.abs(gt_depth_tensor - render_dict["depth"]) * (gt_depth_tensor > 0)
            # 根据深度误差和阈值条件，生成了深度误差掩码。这个掩码用于确定渲染深度值大于关键帧深度值且深度误差大于阈值的区域。
            depth_error_mask = (render_dict["depth"] > gt_depth_tensor) * (depth_error > 40 * depth_error.median())

            # 将 alpha 掩码和深度误差掩码进行逻辑或操作，得到最终的种子点掩码。
            seeding_mask = alpha_mask | depth_error_mask

            # 将种子点掩码转换为 numpy 数组格式。
            seeding_mask = torch2np(seeding_mask[0])
        return seeding_mask

    # 基于给定的颜色和深度图像、相机内参、估计的相机到世界变换、种子点掩码以及一个标志（表示是否为新子地图），在图像中生成新的高斯模型的均值点。
    def seed_new_gaussians(self, gt_color: np.ndarray, gt_depth: np.ndarray, intrinsics: np.ndarray,
                           estimate_c2w: np.ndarray, seeding_mask: np.ndarray, is_new_submap: bool) -> np.ndarray:
        """
        Seeds means for the new 3D Gaussian based on ground truth color and depth, camera intrinsics,
        estimated camera-to-world transformation, a seeding mask, and a flag indicating whether this is a new submap.
        Args:
            gt_color: The ground truth color image as a numpy array with shape (H, W, 3).
            gt_depth: The ground truth depth map as a numpy array with shape (H, W).
            intrinsics: The camera intrinsics matrix as a numpy array with shape (3, 3).
            estimate_c2w: The estimated camera-to-world transformation matrix as a numpy array with shape (4, 4).
            seeding_mask: A binary mask indicating where to seed new Gaussians, with shape (H, W).
            is_new_submap: Flag indicating whether the seeding is for a new submap (True) or an existing submap (False).
        Returns(返回的就是初始化的3D高斯点的坐标,形状为(N,3)):
            np.ndarray: An array of 3D points where new Gaussians will be initialized, with shape (N, 3)

        """
        # 根据pose、颜色、深度、内参获取点云。该点云包含了在世界坐标系下的三维点坐标以及对应的点的颜色。
        pts = create_point_cloud(gt_color, 1.005 * gt_depth, intrinsics, estimate_c2w)

        # 将深度图像展平成一维数组。
        flat_gt_depth = gt_depth.flatten()
        # 生成一个非零深度值掩码，用于过滤掉深度图像中的零深度像素。
        non_zero_depth_mask = flat_gt_depth > 0.  # need filter if zero depth pixels in gt_depth

        # 获取了种子点掩码中为真的像素位置
        valid_ids = np.flatnonzero(seeding_mask)
        
        # 根据是否为新的子地图，进行不同的采样策略：
        if is_new_submap:
            # 如果是新的子地图，首先确定需要采样的点的数量，然后从点云中均匀采样和基于梯度的采样，最后将两者结合，并加上种子点掩码中为真的像素位置，去重并得到最终的采样点。

            # 首先确定需要采样的点的数量。如果 self.new_submap_points_num 小于 0，则采样所有点；否则，随机选择 self.new_submap_points_num 个点。
            if self.new_submap_points_num < 0:
                uniform_ids = np.arange(pts.shape[0]) #生成的所有的点
            else:
                uniform_ids = np.random.choice(pts.shape[0], self.new_submap_points_num, replace=False) #随机选择new_submap_points_num个点
            
            # 基于颜色梯度进行采样，使用函数 sample_pixels_based_on_gradient()，该函数从彩色图像中根据梯度（根据梯度值的大小）选择一定数量的像素。
            gradient_ids = sample_pixels_based_on_gradient(gt_color, self.new_submap_gradient_points_num)
            # 将两者的索引合并为一个数组 combined_ids（包含了需要采样的点以及梯度值较大的点）。
            combined_ids = np.concatenate((uniform_ids, gradient_ids))
            # 再结合掩码中值有效的点
            combined_ids = np.concatenate((combined_ids, valid_ids))
            # 去重得到最终的采样点的索引 sample_ids。
            sample_ids = np.unique(combined_ids)
        else:
            # 如果不是新的子地图，根据设定的采样点数量以及有效的种子点，进行随机采样。
            if self.new_frame_sample_size < 0 or len(valid_ids) < self.new_frame_sample_size:
                # 如果 self.new_frame_sample_size 小于 0 或者有效的种子点数量小于 self.new_frame_sample_size，则直接将有效的种子点作为采样点。
                sample_ids = valid_ids #掩码中值有效的点
            else:
                # 否则，从有效的种子点中随机选择 self.new_frame_sample_size 个点作为采样点。
                sample_ids = np.random.choice(valid_ids, size=self.new_frame_sample_size, replace=False)
        # 最终，根据采样点的索引，从点云中提取对应的坐标，并返回作为新的高斯模型的均值点。
        sample_ids = sample_ids[non_zero_depth_mask[sample_ids]]
        return pts[sample_ids, :].astype(np.float32)

    def optimize_submap(self, keyframes: list, gaussian_model: GaussianModel, iterations: int = 100) -> dict:
        """
        Optimizes the submap by refining the parameters of the 3D Gaussian based on the observations
        from keyframes observing the submap.
        Args:
            keyframes: A list of tuples consisting of frame id and keyframe dictionary
            gaussian_model: An instance of the GaussianModel class representing the initial state
                of the Gaussian model to be optimized.
            iterations: The number of iterations to perform the optimization process. Defaults to 100.
        Returns:
            losses_dict: Dictionary with the optimization statistics
        """

        iteration = 0
        losses_dict = {}

        current_frame_iters = self.current_view_opt_iterations * iterations
        distribution = compute_opt_views_distribution(len(keyframes), iterations, current_frame_iters)
        start_time = time.time()
        while iteration < iterations + 1:
            gaussian_model.optimizer.zero_grad(set_to_none=True)
            keyframe_id = np.random.choice(np.arange(len(keyframes)), p=distribution)

            frame_id, keyframe = keyframes[keyframe_id]
            render_pkg = render_gaussian_model(gaussian_model, keyframe["render_settings"])

            image, depth = render_pkg["color"], render_pkg["depth"]
            gt_image = keyframe["color"]
            gt_depth = keyframe["depth"]

            mask = (gt_depth > 0) & (~torch.isnan(depth)).squeeze(0)
            color_loss = (1.0 - self.opt.lambda_dssim) * l1_loss(
                image[:, mask], gt_image[:, mask]) + self.opt.lambda_dssim * (1.0 - ssim(image, gt_image))

            depth_loss = l1_loss(depth[:, mask], gt_depth[mask])
            reg_loss = isotropic_loss(gaussian_model.get_scaling())
            total_loss = color_loss + depth_loss + reg_loss
            total_loss.backward()

            losses_dict[frame_id] = {"color_loss": color_loss.item(),
                                     "depth_loss": depth_loss.item(),
                                     "total_loss": total_loss.item()}

            with torch.no_grad():

                if iteration == iterations // 2 or iteration == iterations:
                    prune_mask = (gaussian_model.get_opacity()
                                  < self.pruning_thre).squeeze()
                    gaussian_model.prune_points(prune_mask)

                # Optimizer step
                if iteration < iterations:
                    gaussian_model.optimizer.step()
                gaussian_model.optimizer.zero_grad(set_to_none=True)

            iteration += 1
        optimization_time = time.time() - start_time
        losses_dict["optimization_time"] = optimization_time
        losses_dict["optimization_iter_time"] = optimization_time / iterations
        return losses_dict

    def grow_submap(self, gt_depth: np.ndarray, estimate_c2w: np.ndarray, gaussian_model: GaussianModel,
                    pts: np.ndarray, filter_cloud: bool) -> int:
        """
        Expands the submap by integrating new points from the current keyframe
        （通过新输入的点pts来扩展子地图）
        Args:
            gt_depth: The ground truth depth map for the current keyframe, as a 2D numpy array.
            estimate_c2w: The estimated camera-to-world transformation matrix for the current keyframe of shape (4x4)
            gaussian_model (GaussianModel): The Gaussian model representing the current state of the submap.
            pts: The current set of 3D points in the keyframe of shape (N, 3)
            filter_cloud: A boolean flag indicating whether to apply filtering to the point cloud to remove
                outliers or noise before integrating it into the map.
        Returns:
            int: The number of points added to the submap
        """
        # 获取当前子地图中高斯点的坐标。
        gaussian_points = gaussian_model.get_xyz()

        # 计算相机视锥体的角点（就是相机可视的3D范围的点），使用地面真实深度图、估计的相机到世界变换矩阵和相机内参。
        camera_frustum_corners = compute_camera_frustum_corners(gt_depth, estimate_c2w, self.dataset.intrinsics)

        # 根据高斯点和相机视锥体的角点（就是相机可视的3D范围的点），计算可重用的点的索引，使用CUDA加速。
        # 当前的高斯点，在当前帧的视锥体内的点的索引
        reused_pts_ids = compute_frustum_point_ids(
            gaussian_points, np2torch(camera_frustum_corners), device="cuda")
        
        # 计算新的点的索引，使用新点pts、视觉范围内的高斯点gaussian_points[reused_pts_ids]以及radius。并放到CUDA上。
        # 根据新的点pts，以及视觉范围内以及存在的点，通过检测他们的距离（是否有邻居点）等策略，确定哪些点应该添加
        new_pts_ids = compute_new_points_ids(gaussian_points[reused_pts_ids], np2torch(pts[:, :3]).contiguous(),
                                             radius=self.new_points_radius, device="cuda")
        # 将新点的索引从PyTorch张量转换为NumPy数组
        new_pts_ids = torch2np(new_pts_ids)
        if new_pts_ids.shape[0] > 0:#如果有新的点被添加到子地图中：
            # 创建点云对象 cloud_to_add，其中包含新的点的坐标和颜色。
            cloud_to_add = np2ptcloud(pts[new_pts_ids, :3], pts[new_pts_ids, 3:] / 255.0)

            # 如果需要对点云进行过滤，则应用统计异常值移除过滤器。
            if filter_cloud:
                cloud_to_add, _ = cloud_to_add.remove_statistical_outlier(nb_neighbors=40, std_ratio=2.0)
            
            # 将点云集成到高斯模型中。
            gaussian_model.add_points(cloud_to_add)

        # 设置高斯模型的特征梯度为False，以防止在优化过程中对这些特征进行更新。
        gaussian_model._features_dc.requires_grad = False
        gaussian_model._features_rest.requires_grad = False
        # 打印当前高斯模型的大小。
        print("Gaussian model size", gaussian_model.get_size())
        
        # 返回添加到子地图中的新点的数量。
        return new_pts_ids.shape[0]

    # 用于执行地图构建的过程
    def map(self, frame_id: int, estimate_c2w: np.ndarray, gaussian_model: GaussianModel, is_new_submap: bool) -> dict:
        """ Calls out the mapping process described in paragraph 3.2
        The process goes as follows: seed new gaussians -> add to the submap -> optimize the submap
        Args:
            frame_id: current keyframe id
            estimate_c2w (np.ndarray): The estimated camera-to-world transformation matrix of shape (4x4)
            gaussian_model (GaussianModel): The current Gaussian model of the submap
            is_new_submap (bool): A boolean flag indicating whether the current frame initiates a new submap
        Returns:
            opt_dict: Dictionary with statistics about the optimization process （包含优化过程统计信息的词典）
        """

        # 从数据集中获取当前帧对应的的真实颜色和深度图像数据。
        _, gt_color, gt_depth, _ = self.dataset[frame_id]
        # 计算了相机到世界坐标系的逆变换矩阵
        estimate_w2c = np.linalg.inv(estimate_c2w)

        # 创建了一个颜色转换对象 color_transform，可以用于将图像转换为张量形式，方便后续在 PyTorch 模型中使用。
        color_transform = torchvision.transforms.ToTensor()

        # 生成关键帧
        # 所谓的关键帧是字典的形式，包含颜色、深度和渲染设置
        keyframe = {
            "color": color_transform(gt_color).cuda(), # 将真实颜色图像转换为张量形式
            "depth": np2torch(gt_depth, device="cuda"), # 将真实深度图像转换为张量形式
            "render_settings": get_render_settings( # 获取渲染设置参数
                self.dataset.width, self.dataset.height, self.dataset.intrinsics, estimate_w2c)}

        # 计算种子掩码，应该是用于标记新的高斯模型的位置
        seeding_mask = self.compute_seeding_mask(gaussian_model, keyframe, is_new_submap)

        # 初始化3D高斯点云
        # pts就是对应的初始化的3D高斯点的坐标
        pts = self.seed_new_gaussians(
            gt_color, gt_depth, self.dataset.intrinsics, estimate_c2w, seeding_mask, is_new_submap)
        # seed_new_gaussians函数实现的功能：
        # 1、根据颜色图像、深度图像、相机内参、估计的相机到世界变换、获取3D点坐标及其对应的颜色
        # 2、根据上面得到的seeding_mask，以及一定的规则，对1中获取的3D点进行采样，得到初始化的3D高斯点的坐标

        # 根据数据集的类型和是否为新的子地图，设置了 filter_cloud 变量。如果数据集是 TUM_RGBD 或者 ScanNet 类型，并且不是新的子地图，那么 filter_cloud 就为 True，否则为 False。
        filter_cloud = isinstance(self.dataset, (TUM_RGBD, ScanNet)) and not is_new_submap
        # A boolean flag indicating whether to apply filtering to the point cloud to remove outliers or noise before integrating it into the map.
        # filter_cloud是一个布尔标志，指示在将点云集成到地图之前是否对点云应用过滤以移除异常值或噪声。

        # 调用了 grow_submap() 函数，对子地图进行了增长。该函数根据提供的参数（深度图像、相机到世界变换、当前子地图的高斯模型、初始化的3D高斯点的坐标及颜色、是否过滤点云），对子地图进行了扩展和优化。
        new_pts_num = self.grow_submap(gt_depth, estimate_c2w, gaussian_model, pts, filter_cloud)

        max_iterations = self.iterations
        if is_new_submap: #如果是新的子地图，那么就跟新子地图的迭代次数
            max_iterations = self.new_submap_iterations
        start_time = time.time()
        # 调用了 optimize_submap() 函数，对子地图进行了优化。该函数根据提供的参数（关键帧、当前子地图的高斯模型、迭代次数），对子地图进行了优化。
        opt_dict = self.optimize_submap([(frame_id, keyframe)] + self.keyframes, gaussian_model, max_iterations)
        optimization_time = time.time() - start_time
        print("Optimization time: ", optimization_time)

        self.keyframes.append((frame_id, keyframe))

        # Visualise the mapping for the current frame
        with torch.no_grad():
            render_pkg_vis = render_gaussian_model(gaussian_model, keyframe["render_settings"])
            image_vis, depth_vis = render_pkg_vis["color"], render_pkg_vis["depth"]
            psnr_value = calc_psnr(image_vis, keyframe["color"]).mean().item()
            opt_dict["psnr_render"] = psnr_value
            print(f"PSNR this frame: {psnr_value}")
            self.logger.vis_mapping_iteration(
                frame_id, max_iterations,
                image_vis.clone().detach().permute(1, 2, 0),
                depth_vis.clone().detach().permute(1, 2, 0),
                keyframe["color"].permute(1, 2, 0),
                keyframe["depth"].unsqueeze(-1),
                seeding_mask=seeding_mask)

        # Log the mapping numbers for the current frame
        self.logger.log_mapping_iteration(frame_id, new_pts_num, gaussian_model.get_size(),
                                          optimization_time/max_iterations, opt_dict)
        return opt_dict

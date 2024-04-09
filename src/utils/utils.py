import os
import random

import numpy as np
import open3d as o3d
import torch
from gaussian_rasterizer import GaussianRasterizationSettings, GaussianRasterizer

# 目的是设置随机数生成器的种子，以确保在多次运行中产生的随机数是可重现的
def setup_seed(seed: int) -> None:
    """ Sets the seed for generating random numbers to ensure reproducibility across multiple runs.
    Args:
        seed: The seed value to set for random number generators in torch, numpy, and random.
    """
    torch.manual_seed(seed) #设置了PyTorch的随机数生成器的种子为给定的seed值。
    torch.cuda.manual_seed_all(seed) #设置了PyTorch的CUDA随机数生成器的种子为给定的seed值。如果你在使用GPU进行计算，这一步会确保GPU上的随机数生成也是可重现的。
    os.environ["PYTHONHASHSEED"] = str(seed) #设置了Python中哈希函数的种子
    np.random.seed(seed) #设置了NumPy的随机数生成器的种子为给定的seed值
    random.seed(seed) #设置了Python标准库中random模块的随机数生成器的种子
    torch.backends.cudnn.deterministic = True #设置了PyTorch的CuDNN模块的确定性模式为True。这个设置确保了在使用CuDNN时的一致性，尤其是在涉及卷积神经网络时。
    torch.backends.cudnn.benchmark = False #禁用了PyTorch的CuDNN模块的基准模式。在基准模式下，CuDNN会尝试找到最适合当前配置的卷积算法，但这可能会导致不同运行之间结果的差异。


def torch2np(tensor: torch.Tensor) -> np.ndarray:
    """ Converts a PyTorch tensor to a NumPy ndarray.
    Args:
        tensor: The PyTorch tensor to convert.
    Returns:
        A NumPy ndarray with the same data and dtype as the input tensor.
    """
    return tensor.detach().cpu().numpy()


def np2torch(array: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """Converts a NumPy ndarray to a PyTorch tensor.
    Args:
        array: The NumPy ndarray to convert.
        device: The device to which the tensor is sent. Defaults to 'cpu'.

    Returns:
        A PyTorch tensor with the same data as the input array.
    """
    return torch.from_numpy(array).float().to(device)


def np2ptcloud(pts: np.ndarray, rgb=None) -> o3d.geometry.PointCloud:
    """converts numpy array to point cloud
    Args:
        pts (ndarray): point cloud
    Returns:
        (PointCloud): resulting point cloud
    """
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    if rgb is not None:
        cloud.colors = o3d.utility.Vector3dVector(rgb)
    return cloud


def dict2device(dict: dict, device: str = "cpu") -> dict:
    """Sends all tensors in a dictionary to a specified device.
    Args:
        dict: The dictionary containing tensors.
        device: The device to send the tensors to. Defaults to 'cpu'.
    Returns:
        The dictionary with all tensors sent to the specified device.
    """
    for k, v in dict.items():
        if isinstance(v, torch.Tensor):
            dict[k] = v.to(device)
    return dict


def get_render_settings(w, h, intrinsics, w2c, near=0.01, far=100, sh_degree=0):
    """
    Constructs and returns a GaussianRasterizationSettings object for rendering,
    configured with given camera parameters.

    Args:
        width (int): The width of the image.
        height (int): The height of the image.
        intrinsic (array): 3*3, Intrinsic camera matrix.
        w2c (array): World to camera transformation matrix.
        near (float, optional): The near plane for the camera. Defaults to 0.01.
        far (float, optional): The far plane for the camera. Defaults to 100.

    Returns:
        GaussianRasterizationSettings: Configured settings for Gaussian rasterization.
    """
    # 从相机内参中提取fx, fy, cx, cy
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1,
                                                  1], intrinsics[0, 2], intrinsics[1, 2]
    # 相机到世界坐标系的变换矩阵
    w2c = torch.tensor(w2c).cuda().float()

    # 从相机外参中提取相机的中心点
    cam_center = torch.inverse(w2c)[:3, 3]
    viewmatrix = w2c.transpose(0, 1)
    opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                [0.0, 0.0, far /
                                    (far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]], device='cuda').float().transpose(0, 1)
    full_proj_matrix = viewmatrix.unsqueeze(
        0).bmm(opengl_proj.unsqueeze(0)).squeeze(0)
    
    # 返回一个GaussianRasterizationSettings对象
    return GaussianRasterizationSettings(
        image_height=h,
        image_width=w,
        tanfovx=w / (2 * fx),
        tanfovy=h / (2 * fy),
        bg=torch.tensor([0, 0, 0], device='cuda').float(),
        scale_modifier=1.0,
        viewmatrix=viewmatrix,
        projmatrix=full_proj_matrix,
        sh_degree=sh_degree,
        campos=cam_center,
        prefiltered=False,
        debug=False)


def render_gaussian_model(gaussian_model, render_settings,
                          override_means_3d=None, override_means_2d=None,
                          override_scales=None, override_rotations=None,
                          override_opacities=None, override_colors=None):
    """
    Renders a Gaussian model with specified rendering settings, allowing for
    optional overrides of various model parameters.
    (用于渲染高斯模型,并返回渲染的结果)

    Args:
        gaussian_model(高斯模型): A Gaussian model object that provides methods to get
            various properties like xyz coordinates, opacity, features, etc.
        render_settings(渲染的参数设置): Configuration settings for the GaussianRasterizer.
        (下面为可选的参数用于覆盖高斯模型的各种属性。)
        override_means_3d (Optional): If provided, these values will override(覆盖)
            the 3D mean values from the Gaussian model.
        override_means_2d (Optional): If provided, these values will override
            the 2D mean values. Defaults to zeros if not provided.
        override_scales (Optional): If provided, these values will override the
            scale values from the Gaussian model.
        override_rotations (Optional): If provided, these values will override
            the rotation values from the Gaussian model.
        override_opacities (Optional): If provided, these values will override
            the opacity values from the Gaussian model.
        override_colors (Optional): If provided, these values will override the
            color values from the Gaussian model.
    Returns:
        A dictionary containing the rendered color, depth, radii, and 2D means
        of the Gaussian model. The keys of this dictionary are 'color', 'depth',
        'radii', and 'means2D', each mapping to their respective rendered values.
    """
    # 创建了一个GaussianRasterizer的对象renderer，用于执行渲染操作
    renderer = GaussianRasterizer(raster_settings=render_settings)

    # 先检查是否有提供覆盖的3D均值(override_means_3d)，如果没有提供，则使用高斯模型的原始均值(gaussian_model.get_xyz())；否则使用提供的覆盖值。
    if override_means_3d is None:
        means3D = gaussian_model.get_xyz()
    else:
        means3D = override_means_3d

    # 检查是否有提供覆盖的2D均值(override_means_2d)，如果没有提供，则创建一个与means3D相同形状的全零张量；否则使用提供的覆盖值。
    if override_means_2d is None:
        means2D = torch.zeros_like(
            means3D, dtype=means3D.dtype, requires_grad=True, device="cuda")
        means2D.retain_grad()
    else:
        means2D = override_means_2d

    # 没有提供覆盖的不透明度(override_opacities)，则使用高斯模型的原始不透明度(gaussian_model.get_opacity())；否则使用提供的覆盖值。
    if override_opacities is None:
        opacities = gaussian_model.get_opacity()
    else:
        opacities = override_opacities

    # 如果提供了覆盖的颜色(override_colors)，则直接使用提供的值，否则通过调用gaussian_model.get_features()获取颜色。
    shs, colors_precomp = None, None
    if override_colors is not None:
        colors_precomp = override_colors
    else:
        shs = gaussian_model.get_features()

    # 构建渲染参数(render_args)，包括3D均值、2D均值、不透明度、颜色、缩放、旋转等。
    render_args = {
        "means3D": means3D,
        "means2D": means2D,
        "opacities": opacities,
        "colors_precomp": colors_precomp,
        "shs": shs,
        "scales": gaussian_model.get_scaling() if override_scales is None else override_scales,
        "rotations": gaussian_model.get_rotation() if override_rotations is None else override_rotations,
        "cov3D_precomp": None
    }
    # 调用renderer的方法，执行渲染操作，返回渲染结果。
    # 将渲染结果存储在color、depth、alpha和radii变量中，并将这些结果以字典形式返回。
    color, depth, alpha, radii = renderer(**render_args)

    return {"color": color, "depth": depth, "radii": radii, "means2D": means2D, "alpha": alpha}


def batch_search_faiss(indexer, query_points, k):
    """
    Perform a batch search on a IndexIVFFlat indexer to circumvent the search size limit of 65535.

    Args:
        indexer: The FAISS indexer object.
        query_points: A tensor of query points.
        k (int): The number of nearest neighbors to find.

    Returns:
        distances (torch.Tensor): The distances of the nearest neighbors.
        ids (torch.Tensor): The indices of the nearest neighbors.
    """
    split_pos = torch.split(query_points, 65535, dim=0)
    distances_list, ids_list = [], []

    for split_p in split_pos:
        distance, id = indexer.search(split_p.float(), k)
        distances_list.append(distance.clone())
        ids_list.append(id.clone())
    distances = torch.cat(distances_list, dim=0)
    ids = torch.cat(ids_list, dim=0)

    return distances, ids

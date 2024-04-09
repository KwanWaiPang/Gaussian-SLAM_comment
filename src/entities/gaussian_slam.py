""" This module includes the Gaussian-SLAM class, which is responsible for controlling Mapper and Tracker
    It also decides when to start a new submap and when to update the estimated camera poses.
"""
import os
import pprint
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src.entities.arguments import OptimizationParams
from src.entities.datasets import get_dataset
from src.entities.gaussian_model import GaussianModel
from src.entities.mapper import Mapper
from src.entities.tracker import Tracker
from src.entities.logger import Logger
from src.utils.io_utils import save_dict_to_ckpt, save_dict_to_yaml
from src.utils.mapper_utils import exceeds_motion_thresholds
from src.utils.utils import np2torch, setup_seed, torch2np
from src.utils.vis_utils import *  # noqa - needed for debugging


class GaussianSLAM(object):

    def __init__(self, config: dict) -> None:

        self._setup_output_path(config)
        self.device = "cuda"
        self.config = config

        self.scene_name = config["data"]["scene_name"]
        self.dataset_name = config["dataset_name"]
        self.dataset = get_dataset(config["dataset_name"])({**config["data"], **config["cam"]})# 读取数据集

        n_frames = len(self.dataset)
        frame_ids = list(range(n_frames))
        self.mapping_frame_ids = frame_ids[::config["mapping"]["map_every"]] + [n_frames - 1]

        self.estimated_c2ws = torch.empty(len(self.dataset), 4, 4)
        self.estimated_c2ws[0] = torch.from_numpy(self.dataset[0][3])

        save_dict_to_yaml(config, "config.yaml", directory=self.output_path)

        self.submap_using_motion_heuristic = config["mapping"]["submap_using_motion_heuristic"] ##基于运动的状态来决定子图的更新（tum中为true）

        self.keyframes_info = {}
        self.opt = OptimizationParams(ArgumentParser(description="Training script parameters"))

        if self.submap_using_motion_heuristic: # 如果基于运动的状态来决定子图的更新，new_submap_frame_ids=0
            self.new_submap_frame_ids = [0]
        else:#否则，则根据config["mapping"]["new_submap_every"]来决定
            self.new_submap_frame_ids = frame_ids[::config["mapping"]["new_submap_every"]] + [n_frames - 1]
            self.new_submap_frame_ids.pop(0)

        self.logger = Logger(self.output_path, config["use_wandb"])
        self.mapper = Mapper(config["mapping"], self.dataset, self.logger)
        self.tracker = Tracker(config["tracking"], self.dataset, self.logger)

        print('Tracking config')
        pprint.PrettyPrinter().pprint(config["tracking"])
        print('Mapping config')
        pprint.PrettyPrinter().pprint(config["mapping"])

    def _setup_output_path(self, config: dict) -> None:
        """ Sets up the output path for saving results based on the provided configuration. If the output path is not
        specified in the configuration, it creates a new directory with a timestamp.
        Args:
            config: A dictionary containing the experiment configuration including data and output path information.
        """
        if "output_path" not in config["data"]:
            output_path = Path(config["data"]["output_path"])
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_path = output_path / self.timestamp
        else:
            self.output_path = Path(config["data"]["output_path"])
        self.output_path.mkdir(exist_ok=True, parents=True)
        os.makedirs(self.output_path / "mapping_vis", exist_ok=True)
        os.makedirs(self.output_path / "tracking_vis", exist_ok=True)

    def should_start_new_submap(self, frame_id: int) -> bool:
        """ Determines whether a new submap should be started based on the motion heuristic or specific frame IDs.
        基于运动启发式或特定帧ID，确定是否应该启动新的子地图。
        Args:
            frame_id: The ID of the current frame being processed.
        Returns:
            A boolean indicating whether to start a new submap.（通过返回一个布尔值来决定是否新增submap）
        """
        if self.submap_using_motion_heuristic:
            if exceeds_motion_thresholds( #根据当前的帧和上一个submap的帧他们对应的运动是否超出阈值来判断是否需要更新submap
                self.estimated_c2ws[frame_id], self.estimated_c2ws[self.new_submap_frame_ids[-1]], #[-1] 表示取列表中的最后一个元素。
                    rot_thre=50, trans_thre=0.5):
                return True
        elif frame_id in self.new_submap_frame_ids:#如果不是基于运动的状态来决定子图的更新，则根据config["mapping"]["new_submap_every"]来决定，每隔多少帧更新一次
            return True
        return False

    def start_new_submap(self, frame_id: int, gaussian_model: GaussianModel) -> None:
        """ Initializes a new submap, saving the current submap's checkpoint and resetting the Gaussian model.
        This function updates the submap count and optionally marks the current frame ID for new submap initiation.
        Args:
            frame_id: The ID of the current frame at which the new submap is started.
            gaussian_model: The current GaussianModel instance to capture and reset for the new submap.
        Returns:
            A new, reset GaussianModel instance for the new submap.
        """
        # 获取当前高斯模型的参数
        gaussian_params = gaussian_model.capture_dict() 

        # 生成一个子地图的检查点名称，通过将当前子地图ID转换为6位的字符串，不足位数的用零填充
        submap_ckpt_name = str(self.submap_id).zfill(6)

        # 创建一个字典 submap_ckpt，其中包含了当前子地图的高斯参数以及与该子地图相关的关键帧信息。
        submap_ckpt = {
            "gaussian_params": gaussian_params,
            "submap_keyframes": sorted(list(self.keyframes_info.keys()))
        }

        # 调用一个函数来保存 submap_ckpt 字典到磁盘上。
        save_dict_to_ckpt(
            submap_ckpt, f"{submap_ckpt_name}.ckpt", directory=self.output_path / "submaps")
        
        # 重新初始化高斯模型（0应该是sh系数）
        gaussian_model = GaussianModel(0)

        # 进行高斯模型的训练设置，包括初始化高斯参数以及优化训练等参数
        gaussian_model.training_setup(self.opt)

        # 下面两行代码清空了与关键帧相关的信息。
        self.mapper.keyframes = []
        self.keyframes_info = {}

        # 根据是否使用运动启发式算法来决定是否将当前帧ID添加到新子地图的帧ID列表中，并将其添加到映射帧ID列表中。
        if self.submap_using_motion_heuristic:
            self.new_submap_frame_ids.append(frame_id)
            self.mapping_frame_ids.append(frame_id)
        
        # 递增子地图ID
        self.submap_id += 1

        # 返回新的高斯模型
        return gaussian_model

    # 执行Gaussian-SLAM的tracking与mapping
    def run(self) -> None:
        """ Starts the main program flow for Gaussian-SLAM, including tracking and mapping. """

        # 目的是设置随机数生成器的种子，以确保在多次运行中产生的随机数是可重现的
        setup_seed(self.config["seed"])
        
        # 初始化高斯模型
        gaussian_model = GaussianModel(0)
        # 设置高斯模型的训练参数
        gaussian_model.training_setup(self.opt)
        self.submap_id = 0 # 子地图的id，初始化为0
        
        # 遍历所有帧
        for frame_id in range(len(self.dataset)):
            
            # 进行tracker跟踪，得到估计的相机位姿（self.estimated_c2ws）
            if frame_id in [0, 1]: # 如果是第一帧或第二帧
                estimated_c2w = self.dataset[frame_id][-1] # 读取真实的相机位姿
            else:# 如果不是第一帧或第二帧，则使用Tracker跟踪
                estimated_c2w = self.tracker.track(
                    frame_id, gaussian_model,
                    torch2np(self.estimated_c2ws[torch.tensor([0, frame_id - 2, frame_id - 1])]))
            self.estimated_c2ws[frame_id] = np2torch(estimated_c2w) #Converts a NumPy ndarray to a PyTorch tensor

            # Reinitialize gaussian model for new segment
            # 如果需要开始新的子地图，则保存当前子地图的参数，并重置高斯模型
            # gaussian_model对应着当前的子地图，如果需要开始新的子地图，则重新初始化了，否则就一直优化当前的子地图
            if self.should_start_new_submap(frame_id):
                save_dict_to_ckpt(self.estimated_c2ws[:frame_id + 1], "estimated_c2w.ckpt", directory=self.output_path)#保存当前帧的估计相机位姿
                gaussian_model = self.start_new_submap(frame_id, gaussian_model)#创建新的子地图对应更新了gaussian_model，否则不更新gaussian_model

            # mapping
            # 如果当前帧存在于mapping_frame_ids列表中，则进行mapping
            if frame_id in self.mapping_frame_ids:
                print("\nMapping frame", frame_id) # 打印当前帧的ID，用于显示当前正在mapping的帧
                # 重新设置高斯模型的训练参数
                gaussian_model.training_setup(self.opt)
                # 将当前帧的估计相机位姿转换为numpy格式
                estimate_c2w = torch2np(self.estimated_c2ws[frame_id])
                
                # 根据条件判断 new_submap 是否为真，如果 self.keyframes_info 为空则为真，否则为假。
                new_submap = not bool(self.keyframes_info)
                
                # 将当前帧的ID、估计的相机位姿、高斯模型以及 是否新的submap的flag 传递给 Mapper 的 map 函数，得到优化参数字典 opt_dict。
                opt_dict = self.mapper.map(frame_id, estimate_c2w, gaussian_model, new_submap)
                # 输出的opt_dict为包含优化过程统计信息的词典

                # Keyframes info update
                self.keyframes_info[frame_id] = {
                    "keyframe_id": len(self.keyframes_info.keys()),
                    "opt_dict": opt_dict
                }
        # 保存最后一帧的估计相机位姿
        save_dict_to_ckpt(self.estimated_c2ws[:frame_id + 1], "estimated_c2w.ckpt", directory=self.output_path)

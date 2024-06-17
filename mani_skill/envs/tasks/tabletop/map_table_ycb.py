#!/usr/bin/env python
import os
import sys
from typing import Any, Dict, List, Union
import numpy as np
import yaml
import random
import matplotlib.pyplot as plt
import array

import torch
import sapien
import gymnasium as gym

current_dir = os.path.dirname(os.path.abspath(__file__))
from mani_skill.utils.project_utils import *

project_dir = find_project_root(current_dir, "splat_rl")
print(project_dir)
sys.path.append(project_dir)
# Append the gaussian_slam directory to the system path
gaussian_slam_path = os.path.join(project_dir, 'gaussian_slam')
sys.path.append(gaussian_slam_path)
mani_skill_path = os.path.join(project_dir, 'maniskill/mani_skill')
sys.path.append(mani_skill_path)


from mani_skill import ASSET_DIR
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.registration import register_env
from mani_skill.agents.robots import Panda, PandaWristCam, Fetch, MapperArm
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.building import actors
from mani_skill.envs.utils.randomization.pose import random_quaternions
from mani_skill.utils.building.actor_builder import ActorBuilder
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.structs import Actor, Pose
from mani_skill.utils import  sapien_utils, common
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

from gaussian_slam.src.entities.gaussian_slam_online import GaussianSLAMOnline
from gaussian_slam.src.entities.mapper import Mapper
from gaussian_slam.src.entities.base_dataset import CameraData
from gaussian_slam.src.entities.common_datasets import ManiSkillCameraStream

@register_env("MapTable-v1", max_episode_steps=100)
class MapTableEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda_wristcam", "mapper_arm", "fetch"]
    SUPPORTED_REWARD_MODES = ["sparse", "none", "dense", "normalized_dense"]
    agent: Union[PandaWristCam, MapperArm, Fetch]
    DEFAULT_EPISODE_JSON: str
    DEFAULT_ASSET_ROOT: str
    DEFAULT_MODEL_JSON: str

    def __init__(
            self,
            *args,
            robot_uids="mapper_arm",
            robot_init_qpos_noise=0.02,
            num_envs=1,
            reconfiguration_freq=None,
            episode_json: str = None,
            **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        if episode_json is None:
            episode_json = self.DEFAULT_EPISODE_JSON
        if not os.path.exists(episode_json):
            raise FileNotFoundError(
                f"Episode json ({episode_json}) is not found."
                "To download default json:"
                "`python -m mani_skill.utils.download_asset pick_clutter_ycb`."
            )
        self._episodes: List[Dict] = load_json(episode_json)
        self.model_id = None
        self.all_model_ids = np.array(
            list(
                load_json(ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json").keys()
            )
        )
        if reconfiguration_freq is None:
            if num_envs == 1:
                reconfiguration_freq = 1
            else:
                reconfiguration_freq = 0
        super().__init__(
            *args,
            robot_uids=robot_uids,
            num_envs=num_envs,
            reconfiguration_freq=reconfiguration_freq,
            **kwargs,
        )
        self.ycb_objects_model_ids = None
        # ...
    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_cfg=GPUMemoryConfig(
                max_rigid_contact_count=2 ** 21, max_rigid_patch_count=2 ** 19
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            )
        ]
    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )
    def _load_model(self, model_id: str) -> ActorBuilder:
        raise NotImplementedError()

    def _load_scene(self, options: dict):
        self.scene_builder = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.scene_builder.build()

        # sample some clutter configurations
        eps_idxs = np.arange(0, len(self._episodes))
        rand_idx = torch.randperm(len(eps_idxs), device=torch.device("cpu"))
        eps_idxs = eps_idxs[rand_idx]
        eps_idxs = np.concatenate(
            [eps_idxs] * np.ceil(self.num_envs / len(eps_idxs)).astype(int)
        )[: self.num_envs]

        self.selectable_target_objects: List[List[Actor]] = []
        """for each sub-scene, a list of objects that can be selected as targets"""
        all_objects = []

        for i, eps_idx in enumerate(eps_idxs):
            self.selectable_target_objects.append([])
            episode = self._episodes[eps_idx]
            for actor_cfg in episode["actors"]:
                builder = self._load_model(actor_cfg["model_id"])
                init_pose = actor_cfg["pose"]
                builder.initial_pose = sapien.Pose(p=init_pose[:3], q=init_pose[3:])
                builder.set_scene_idxs([i])
                obj = builder.build(name=f"set_{i}_{actor_cfg['model_id']}")
                all_objects.append(obj)
                if actor_cfg["rep_pts"] is not None:
                    # rep_pts is representative points, representing visible points
                    # we only permit selecting target objects that are visible
                    self.selectable_target_objects[-1].append(obj)

        self.all_objects = Actor.merge(all_objects, name="all_objects")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.scene_builder.initialize(env_idx)

            # reset objects to original poses
            if b == self.num_envs:
                # if all envs reset
                self.all_objects.pose = self.all_objects.initial_pose
            else:
                # if only some envs reset, we unfortunately still have to do some mask wrangling
                mask = torch.isin(self.all_objects._scene_idxs, env_idx)
                self.all_objects.pose = self.all_objects.initial_pose[mask]

    def _initalize_gslam(self):
        self.hand_camera_params = self.get_sensor_params()["hand_camera"]
        rgb_intrinsics = self.hand_camera_params["intrinsic_cv"]
        self.hand_camera_config = self._sensors["hand_camera"]
        self.camera_data = CameraData(rgb_intrinsics, self.hand_camera_config.width, self.hand_camera_config.height)
        gslam_config = os.path.join(gaussian_slam_path, "configs/maniskill/maniskill_camera_stream.yaml")
        self.gslam = GaussianSLAMOnline(gslam_config)


    def evaluate(self):
        # success is achieved when the cube's xy position on the table is within the
        # goal region's area (a circle centered at the goal region's xy position)

        return {}

    # ...
    def _get_obs_extra(self, info: Dict):
        # some useful observation info for solving the task includes the pose of the tcp (tool center point) which is the point between the
        # grippers of the robot
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            #goal_pos=self.goal_region.pose.p,
        )
        if self._obs_mode in ["state", "state_dict"]:
            # if the observation mode is state/state_dict, we provide ground truth information about where the cube is.
            # for visual observation modes one should rely on the sensed visual data to determine where the cube is
            obs.update(
                obj_pose=self.obj.pose.raw_pose,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action, info: Dict):
        reward = 0.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action, info: Dict):
        # this should be equal to compute_dense_reward / max possible reward
        max_reward = 3.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward

    @property
    def _default_sensor_configs(self):
        # registers one 128x128 camera looking at the robot, cube, and target
        # a smaller sized camera will be lower quality, but render faster
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [
            CameraConfig("base_camera", pose=pose, width=1920, height=1080, fov=np.pi / 2, near=0.01, far=100)
        ]

    @property
    def _default_human_render_camera_configs(self):
        # registers a more high-definition (512x512) camera used just for rendering when render_mode="rgb_array" or calling env.render_rgb_array()
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=5)

@register_env("MapTableYCBEnv-v1", max_episode_steps=100)
class MapTableYCBEnv(MapTableEnv):
    DEFAULT_EPISODE_JSON = f"{ASSET_DIR}/tasks/pick_clutter/ycb_train_5k.json.gz"

    def _load_model(self, model_id):
        builder = actors.get_actor_builder(self.scene, id=f"ycb:{model_id}")
        return builder
def main():
    print("Hello ")
    env = gym.make(id = "MapTableYCBEnv", render_mode="rgb_array")
    env.reset()
    while True:
        img = env.render_human()

if __name__ == "__main__":
    main()

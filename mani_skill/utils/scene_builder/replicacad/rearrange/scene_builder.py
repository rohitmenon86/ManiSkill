"""
Code for building scenes from the ReplicaCAD dataset https://aihabitat.org/datasets/replica_cad/

This code is also heavily commented to serve as a tutorial for how to build custom scenes from scratch and/or port scenes over from other datasets/simulators
"""

import json
import os
import os.path as osp
from typing import Dict, List, Tuple, Union
from pathlib import Path

import numpy as np
import sapien
import torch
import transforms3d

from mani_skill import ASSET_DIR
from mani_skill.agents.robots.fetch import (
    Fetch,
    FETCH_UNIQUE_COLLISION_BIT,
    FETCH_BASE_COLLISION_BIT,
)
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.scene_builder import SceneBuilder
from mani_skill.utils.scene_builder.registration import register_scene_builder
from mani_skill.utils.structs.articulation import Articulation
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.building.actors import (
    _load_ycb_dataset,
    build_actor_ycb,
)

from ..scene_builder import ReplicaCADSceneBuilder, DATASET_CONFIG_DIR


@register_scene_builder("ReplicaCADRearrange")
class ReplicaCADRearrangeSceneBuilder(ReplicaCADSceneBuilder):

    task_names: List[str] = ["set_table:train"]

    def __init__(self, env, robot_init_qpos_noise=0.02):
        _load_ycb_dataset()

        # init base replicacad scene builder first
        super().__init__(env, robot_init_qpos_noise, include_staging_scenes=True)

        # for ReplicaCAD we have saved the list of all scene configuration files from the dataset to a local json file
        self._rearrange_scene_configs = []
        for task_name in self.task_names:
            task, split = task_name.split(":")
            self._rearrange_scene_configs += [
                osp.join(split, task, f)
                for f in sorted(
                    os.listdir(
                        osp.join(
                            ASSET_DIR,
                            "scene_datasets/replica_cad_dataset/rearrange/v1_extracted",
                            split,
                            task,
                        ),
                    )
                )
                if f.endswith(".json")
            ]

    def build(self, scene: ManiSkillScene, scene_idx=0, **kwargs):
        with open(
            osp.join(
                ASSET_DIR,
                "scene_datasets/replica_cad_dataset/rearrange/v1_extracted",
                self._rearrange_scene_configs[scene_idx],
            ),
            "rb",
        ) as f:
            episode_json = json.load(f)

        self._rearrange_base_scene_config = Path(episode_json["scene_id"]).name

        super().build(
            scene,
            scene_idx=self._rcad_config_to_idx[self._rearrange_base_scene_config],
        )

        self._rearrange_default_object_poses: List[
            Tuple[Union[Actor, Articulation], Union[sapien.Pose, Pose]]
        ] = self._rcad_default_object_poses.copy()
        self._rearrange_objects: Dict[str, Actor] = self._rcad_objects.copy()
        self._rearrange_movable_objects: Dict[str, Actor] = (
            self._rcad_movable_objects.copy()
        )

        q = transforms3d.quaternions.axangle2quat(
            np.array([1, 0, 0]), theta=np.deg2rad(90)
        )

        for i, (actor_id, transformation) in enumerate(episode_json["rigid_objs"]):
            actor_id = actor_id.split(".")[0]
            actor_name = f"{actor_id}-{i}"
            builder, _ = build_actor_ycb(
                actor_id, scene, name=actor_name, return_builder=True
            )
            actor = builder.build(name=actor_name)

            aabb = (
                actor._objs[0]
                .find_component_by_type(sapien.render.RenderBodyComponent)
                .compute_global_aabb_tight()
            )
            height = aabb[1, 2] - aabb[0, 2]
            pose = sapien.Pose(q=q, p=[0, 0, height]) * sapien.Pose(
                matrix=transformation
            )

            self._rearrange_default_object_poses.append((actor, pose))
            self._rearrange_objects[actor_name] = actor
            self._rearrange_movable_objects[actor_name] = actor

    @property
    def scene_configs(self):
        return self._rearrange_scene_configs

    @property
    def navigable_positions(
        self,
    ) -> List[Tuple[Union[Actor, Articulation], Union[Pose, sapien.Pose]]]:
        assert isinstance(
            self._rcad_scene_idx, int
        ), "Must build scene before getting navigable positions"
        return self._rcad_navigable_positions[self._rearrange_base_scene_config]

    @property
    def default_object_poses(self) -> np.ndarray:
        return self._rearrange_default_object_poses

    @property
    def scene_objects(self) -> Dict[str, Actor]:
        return self._rearrange_objects

    @property
    def movable_objects(self) -> Dict[str, Actor]:
        return self._rearrange_movable_objects

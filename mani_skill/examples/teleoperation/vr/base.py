from typing import Literal
import numpy as np
import sapien
from sapien.render import (
    RenderVRDisplay,
)

import gymnasium as gym
from mani_skill.envs.sapien_env import BaseEnv
controller_id2names = {1: "left", 2: "right"}

class VRTeleopInterface:
    """Basic class for streaming to and from a VR headset communicating via ALVR and SteamVR"""
    def __init__(self, env: BaseEnv, visualize: bool = True):
        self.env = env
        self.visualize = visualize
        self.vr_display = RenderVRDisplay()
        self.controllers = self.vr_display.get_controller_ids()
        self.renderer_context = sapien.render.SapienRenderer()._internal_context
        self.reset()
        assert self.base_env.num_envs == 1, "can only do VR teleop when there is only one environment running"

    def __del__(self):
        del self.controllers
        del self.vr_display
        del self.renderer_context

    def reset(self):
        self._create_visual_models()
        self.controller_axes = None
        self.marker_spheres = None
        self.vr_display.set_scene(self.base_env._scene.sub_scenes[0])

    def set_scene(self, scene):
        """
        register the VR viewer to the scene
        """
        self.scene = scene
        self.vr_display.set_scene(scene)

    @property
    def base_env(self) -> BaseEnv:
        return self.env.unwrapped

    @property
    def root_pose(self):
        """
        return the root pose of the VR viewer
        see set_root_pose for more details
        """
        return self.vr_display.root_pose

    @root_pose.setter
    def root_pose(self, pose):
        """
        Set the root pose of the VR viewer
        pose: sapien.Pose, ([x, y, z], [qx, qy, qz, qw]), the position and quaternion of the root pose
        root_pose: the root pose of the VR viewer. which is the foot of the VR viewer
        """
        self.vr_display.root_pose = pose

    @property
    def controllers_names(self, controller_id):
        """
        return the controllers names
            1: left controller
            2: right controller
        """
        return controller_id2names[controller_id]

    # ---------------------------------------------------------------------------- #
    # Functions dependent on VR system used
    # ---------------------------------------------------------------------------- #
    def get_user_action(self) -> Literal["quit", "calibrate_ee", "reset"]:
        """
        Check if one of the given system actions is active (e.g. user is pressing a button on a meta quest controller or making a gesture while using vision pro)

        Implementing the sim teleop interface class requires implementing this function to check if the action type given is active on the controller
        - "quit": Action to stop data collection
        - "calibrate_ee": Action to calibrate 1 or 2 end-effectors
        - "reset": Action to stop the current episode data collection and begin collecting another episode.
        """
        raise NotImplementedError()

    @property
    def head_pose(self):
        """
        return the head pose of the VR viewer
        """
        return self.vr_display.get_hmd_pose()

    @property
    def controller_poses(self):
        """
        return the controller poses, [left_pose, right_pose]
        pose: in format of sapien.Pose, ([x, y, z], [qx, qy, qz, qw]), the position and quaternion of the root pose
        """
        return [self.vr_display.get_controller_pose(c) for c in self.controllers]

    @property
    def controller_left_poses(self):
        """
        return the left controller pose
        """
        return self.vr_display.get_controller_pose(self.controllers[0])

    @property
    def controller_right_poses(self):
        """
        return the right controller pose
        """
        return self.vr_display.get_controller_pose(self.controllers[1])

    @property
    def controller_hand_poses(self):
        """
        hit point of ray casted from the controller
        """
        t2c = sapien.Pose()
        t2c.rpy = [0, self.ray_angle, 0]
        t2ws = [self.root_pose * c2r * t2c for c2r in self.controller_poses]
        return t2ws

    @property
    def controller_left_hand_poses(self):
        """
        hit point of ray casted from the controller
        """
        t2c = sapien.Pose()
        t2c.rpy = [0, self.ray_angle, 0]
        c2r = self.controller_poses[0]
        r2w = self.root_pose
        t2w = r2w * c2r * t2c
        return t2w

    @property
    def controller_right_hand_poses(self):
        """
        hit point of ray casted from the controller
        """
        t2c = sapien.Pose()
        t2c.rpy = [0, self.ray_angle, 0]
        c2r = self.controller_poses[1]
        r2w = self.root_pose
        t2w = r2w * c2r * t2c
        return t2w

    @property
    def render_scene(self):
        return self.vr_display._internal_scene

    def render(self):
        """
        update the VR viewer
        """
        if self.visualize:
            self._update_controller_axes()
        self.vr_display.update_render()
        self.vr_display.render()

    @property
    def ray_angle(self):
        return np.pi / 4

    def pick(self, controller_id):
        """
        hit point of ray casted from the controller
        """
        t2c = sapien.Pose()
        t2c.rpy = [0, self.ray_angle, 0]
        c2r = self.controller_poses[controller_id]
        r2w = self.root_pose
        t2w = r2w * c2r * t2c
        d = t2w.to_transformation_matrix()[:3, 0]

        assert isinstance(self.scene.physx_system, sapien.physx.PhysxCpuSystem)
        px: sapien.physx.PhysxCpuSystem = self.scene.physx_system
        res = px.raycast(t2w.p, d, 50)  # hit of the ray casted from the controller
        return res

    def update_marker_sphere(self, i, hit):
        """
        update the marker sphere which is used to show the target of ray casted from the controller
        """
        if self.marker_spheres is None:
            self.marker_spheres = [
                self._create_marker_sphere() for c in self.controllers
            ]

        if hit is None:
            self.marker_spheres[i].transparency = 1  # total transparent, invisible
        else:
            self.marker_spheres[i].set_position(hit.position)
            self.marker_spheres[i].transparency = 0

    """help visual functions"""

    def _create_visual_models(self):
        self.cone = self.renderer_context.create_cone_mesh(16)
        self.capsule = self.renderer_context.create_capsule_mesh(0.1, 0.5, 16, 4)
        self.cylinder = self.renderer_context.create_cylinder_mesh(16)
        self.sphere = self.renderer_context.create_uvsphere_mesh()

        self.laser = self.renderer_context.create_line_set(
            [0, 0, 0, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0]
        )

        self.mat_red = self.renderer_context.create_material(
            [5, 0, 0, 1], [0, 0, 0, 1], 0, 1, 0
        )
        self.mat_green = self.renderer_context.create_material(
            [0, 1, 0, 1], [0, 0, 0, 1], 0, 1, 0
        )
        self.mat_blue = self.renderer_context.create_material(
            [0, 0, 1, 1], [0, 0, 0, 1], 0, 1, 0
        )
        self.mat_cyan = self.renderer_context.create_material(
            [0, 1, 1, 1], [0, 0, 0, 1], 0, 1, 0
        )
        self.mat_magenta = self.renderer_context.create_material(
            [1, 0, 1, 1], [0, 0, 0, 1], 0, 1, 0
        )
        self.mat_white = self.renderer_context.create_material(
            [1, 1, 1, 1], [0, 0, 0, 1], 0, 1, 0
        )
        self.red_cone = self.renderer_context.create_model([self.cone], [self.mat_red])
        self.green_cone = self.renderer_context.create_model(
            [self.cone], [self.mat_green]
        )
        self.blue_cone = self.renderer_context.create_model(
            [self.cone], [self.mat_blue]
        )
        self.red_capsule = self.renderer_context.create_model(
            [self.capsule], [self.mat_red]
        )
        self.green_capsule = self.renderer_context.create_model(
            [self.capsule], [self.mat_green]
        )
        self.blue_capsule = self.renderer_context.create_model(
            [self.capsule], [self.mat_blue]
        )
        self.cyan_capsule = self.renderer_context.create_model(
            [self.capsule], [self.mat_cyan]
        )
        self.magenta_capsule = self.renderer_context.create_model(
            [self.capsule], [self.mat_magenta]
        )
        self.white_cylinder = self.renderer_context.create_model(
            [self.cylinder], [self.mat_white]
        )
        self.red_sphere = self.renderer_context.create_model(
            [self.sphere], [self.mat_red]
        )

    def _create_coordiate_axes(self):
        render_scene = self.render_scene

        node = render_scene.add_node()
        obj = render_scene.add_object(self.red_cone, node)
        obj.set_scale([0.5, 0.2, 0.2])
        obj.set_position([1, 0, 0])
        obj.shading_mode = 0
        obj.cast_shadow = False

        obj = render_scene.add_object(self.red_capsule, node)
        obj.set_position([0.52, 0, 0])
        obj.shading_mode = 0
        obj.cast_shadow = False

        obj = render_scene.add_object(self.green_cone, node)
        obj.set_scale([0.5, 0.2, 0.2])
        obj.set_position([0, 1, 0])
        obj.set_rotation([0.7071068, 0, 0, 0.7071068])
        obj.shading_mode = 0
        obj.cast_shadow = False

        obj = render_scene.add_object(self.green_capsule, node)
        obj.set_position([0, 0.51, 0])
        obj.set_rotation([0.7071068, 0, 0, 0.7071068])
        obj.shading_mode = 0
        obj.cast_shadow = False

        obj = render_scene.add_object(self.blue_cone, node)
        obj.set_scale([0.5, 0.2, 0.2])
        obj.set_position([0, 0, 1])
        obj.set_rotation([0, 0.7071068, 0, 0.7071068])
        obj.shading_mode = 0
        obj.cast_shadow = False

        obj = render_scene.add_object(self.blue_capsule, node)
        obj.set_position([0, 0, 0.5])
        obj.set_rotation([0, 0.7071068, 0, 0.7071068])
        obj.shading_mode = 0
        obj.cast_shadow = False

        node.set_scale([0.025, 0.025, 0.025])

        return node

    def _update_controller_axes(self):
        if self.controller_axes is None:
            self.controller_axes = [
                self._create_coordiate_axes() for c in self.controllers
            ]

        for n, pose in zip(self.controller_axes, self.controller_poses):
            c2w = self.vr_display.root_pose * pose
            n.set_position(c2w.p)
            n.set_rotation(c2w.q)

    def _create_marker_sphere(self):
        node = self.render_scene.add_object(self.red_sphere)
        node.set_scale([0.05] * 3)
        node.shading_mode = 0
        node.cast_shadow = False
        node.transparency = 1
        return node

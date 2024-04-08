from copy import deepcopy

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils


@register_agent()
class FloatingPandaGripper(BaseAgent):
    uid = "floating_panda_gripper"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/panda/panda_v2_gripper.urdf"  # You can use f"{PACKAGE_ASSET_DIR}" to reference a urdf file in the mani_skill /assets package folder
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link=dict(
            panda_leftfinger=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            panda_rightfinger=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
        ),
    )

    def __init__(self, *args, **kwargs):
        self.root_joint_names = [
            "root_x_axis_joint",
            "root_y_axis_joint",
            "root_z_axis_joint",
            "root_x_rot_joint",
            "root_y_rot_joint",
            "root_z_rot_joint",
        ]
        self.gripper_joint_names = [
            "panda_finger_joint1",
            "panda_finger_joint2",
        ]
        self.gripper_stiffness = 1e3
        self.gripper_damping = 1e2
        self.gripper_force_limit = 100

        self.ee_link_name = "panda_hand_tcp"

        super().__init__(*args, **kwargs)

    @property
    def _controller_configs(self):
        pd_ee_pose = PDEEPoseControllerConfig(
            self.root_joint_names,
            None,
            None,
            self.gripper_stiffness,
            self.gripper_damping,
            self.gripper_force_limit,
            use_delta=False,
            frame="base",
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            normalize_action=False,
        )
        pd_ee_pose_quat = deepcopy(pd_ee_pose)
        pd_ee_pose_quat.rotation_convention = "quaternion"
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.root_joint_names,
            None,
            None,
            self.gripper_stiffness,
            self.gripper_damping,
            self.gripper_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.root_joint_names,
            -0.1,
            0.1,
            self.gripper_stiffness,
            self.gripper_damping,
            self.gripper_force_limit,
            use_delta=True,
        )

        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            -0.01,  # a trick to have force when the object is thin
            0.04,
            self.gripper_stiffness,
            self.gripper_damping,
            self.gripper_force_limit,
        )
        return dict(
            pd_joint_pos=dict(root=arm_pd_joint_pos, gripper=gripper_pd_joint_pos),
            pd_joint_delta_pos=dict(
                root=arm_pd_joint_delta_pos, gripper=gripper_pd_joint_pos
            ),
            pd_ee_pose=dict(root=pd_ee_pose, gripper=gripper_pd_joint_pos),
            pd_ee_pose_quat=dict(root=pd_ee_pose_quat, gripper=gripper_pd_joint_pos),
        )

    @property
    def _sensor_configs(self):
        return []

    def _after_init(self):
        self.finger1_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "panda_leftfinger"
        )
        self.finger2_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "panda_rightfinger"
        )
        self.finger1pad_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "panda_leftfinger_pad"
        )
        self.finger2pad_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "panda_rightfinger_pad"
        )
        self.tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_link_name
        )

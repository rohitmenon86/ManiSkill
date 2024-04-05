import sapien
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.examples.teleoperation.vr.base import VRSimTeleopInterface


class MetaQuest3SimTeleop(VRSimTeleopInterface):
    def __init__(self, env: BaseEnv) -> None:
        super().__init__(env)
        self.offset_pose: sapien.Pose = None

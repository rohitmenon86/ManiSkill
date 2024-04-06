from typing import Literal
import sapien
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.examples.teleoperation.vr.base import VRTeleopInterface


class MetaQuest3SimTeleopWrapper(VRTeleopInterface):
    def __init__(self, env: BaseEnv) -> None:
        super().__init__(env)
        self.offset_pose: sapien.Pose = None


    def get_user_action(self) -> Literal["quit", "reset", "trigger_1", "trigger_2"]:
        right_button_pressed = self.vr_display.get_controller_button_pressed(2)
        if right_button_pressed == 128:
            return "reset" # "A"
        elif right_button_pressed == 2:
            return "quit" # "B"
        elif right_button_pressed == 8589934592:
            return "trigger_1" # "up trigger"
        elif right_button_pressed == 17179869188:
            return "trigger_2" # "down trigger"
        elif right_button_pressed == 25769803780:
            return None # "both up and down"
        else:
            return None

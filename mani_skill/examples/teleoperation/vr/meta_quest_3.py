from typing import Literal
import sapien
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.examples.teleoperation.vr.base import VRTeleopInterface

BUTTONS = {
    "A": 2 << 6,
    "B": 2,
    "up_trigger": 2 << 32,
    "down_trigger": 2 << 33 | 2 << 1
}

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
        elif right_button_pressed == 2 << 32:
            return "trigger_1" # "up trigger"
        elif right_button_pressed == 2 << 33 | 2 << 1:
            return "trigger_2" # "down trigger"
        elif right_button_pressed == BUTTONS["up_trigger"] | BUTTONS["down_trigger"]:
            return None # "both up and down"
        else:
            return None

    # def get_mobile_base_action

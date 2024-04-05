from mani_skill.examples.teleoperation.vr.base_vr_interface import VRSimTeleopInterface


class MetaQuest3SimTeleop(VRSimTeleopInterface):
    def calibrate_ee(self):
        return super().calibrate_ee()

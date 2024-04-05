from mani_skill.examples.teleoperation.vr.base import VRSimTeleopInterface


class MetaQuest3SimTeleop(VRSimTeleopInterface):
    def calibrate_ee(self):
        global offset_pose
        vr.set_scene(env.unwrapped._scene.sub_scenes[0])
        vr.root_pose = sapien.Pose()
        while True:
            # env.render_human()
            vr.render()
            rp = vr.controller_right_poses
            if vr.pressed_button(2) == "down":
                offset_pose = rp
                break

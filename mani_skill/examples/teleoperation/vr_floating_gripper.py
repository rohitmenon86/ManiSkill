import copy
import gymnasium as gym
import numpy as np
import sapien
from scipy import linalg
import mani_skill.envs
import argparse
from mani_skill.utils import common
from mani_skill.utils.wrappers import RecordEpisode
from mani_skill.examples.teleoperation.vr import MetaQuest3SimTeleopWrapper
from transforms3d import euler
MOBILE_SPEED = 0.01

def collect_episode(env: gym.Env, vr: MetaQuest3SimTeleopWrapper):
    ### 1. Calibrate human into the scene and robot
    vr.reset()
    vr.root_pose = sapien.Pose()
    offset_poses = [None, None]
    vr_xy = np.array([0, 0])
    vr.root_pose = sapien.Pose(p=[env.unwrapped.agent.tcp.pose.sp.p[0]+0.2, env.unwrapped.agent.tcp.pose.sp.p[1], 0])
    while True:
        vr.render()
        rp = vr.controller_right_poses
        if vr.get_user_action() == "trigger_2":
            offset_poses = [vr.controller_left_hand_poses, vr.controller_right_poses]
            while vr.get_user_action() is not None:
                vr.render()
            break

    vr_xy = np.array([0, 0])
    init_vr_root_pose = sapien.Pose(-offset_poses[1].p + env.unwrapped.agent.tcp.pose.sp.p)
    vr.root_pose = init_vr_root_pose
    vr_xy = init_vr_root_pose.p[:2].copy()

    ### 2. Collect demonstration
    gripper_action = 1
    while True:
        joystick_xy = vr.vr_display.get_controller_axis_state(2, 0)
        if abs(joystick_xy[1]) > abs(joystick_xy[0]):
            vr_xy[0] += joystick_xy[1] * MOBILE_SPEED
        else:
            vr_xy[1] += joystick_xy[0] * MOBILE_SPEED
        vr.root_pose = sapien.Pose(p=[vr_xy[0], vr_xy[1], vr.root_pose.p[2]], q=vr.root_pose.q)


        user_action = vr.get_user_action()
        for obj in vr.base_env._hidden_objects:
            obj.show_visual()
        vr.render()

        if user_action == "quit":
            return "quit"
        elif user_action == "reset":
            return "reset"
        elif user_action == "trigger_1":
            gripper_action = gripper_action * -1
            while vr.get_user_action() is not None:
                vr.render()
        # elif user_action == "trigger_2":
        #     vr_xy = env.unwrapped.agent.tcp.pose.sp.p[:2]
        #     vr.root_pose = sapien.Pose(p=[vr_xy[0], vr_xy[1], vr.root_pose.p[2]], q=vr.root_pose.q)
        #     while vr.get_user_action() is not None:
        #         vr.render()
        action = env.action_space.sample() * 0
        action[-1] = gripper_action


        rp = vr.root_pose * vr.controller_right_poses


        q = (rp * sapien.Pose(q=euler.euler2quat(0, np.pi, np.pi))).q
        target_tcp_pose = sapien.Pose(p=rp.p, q=q)
        ee_action = np.zeros(7)
        ee_action[:3] = target_tcp_pose.p
        ee_action[3:7] = target_tcp_pose.q
        action_dict = dict(root=ee_action, gripper=gripper_action)
        action = env.agent.controller.from_action_dict(action_dict)
        env.step(action)

def main(args):
    env = gym.make(args.env_id, control_mode="pd_ee_pose_quat", robot_uids="floating_panda_gripper", enable_shadow=True, render_mode="rgb_array")
    # env = RecordEpisode(env, save_video=args.save_video, output_dir=args.record_dir, video_fps=60, source_type="meta-quest3-vr", source_desc="collected using the meta quest 3 headset to control the fetch arm and end-effector and the mobile base")
    # TODO (stao): Support other headset as interfaces in the future
    vr = MetaQuest3SimTeleopWrapper(env)

    # Maps trigger_1 action of controller to grasp action

    eps_id = 0
    while True:
        print(f"Collecting episode {eps_id}")
        env.reset(seed=eps_id)
        ret = collect_episode(env, vr)
        eps_id += 1
        if ret == "quit": break
        elif ret == "reset": continue


    env.close()
    del env


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="ReplicaCAD_SceneManipulation-v1")
    parser.add_argument("-o", "--obs-mode", type=str, default="none")
    parser.add_argument("--record-dir", type=str, default="demos/teleop/ReplicaCAD_SceneManipulation-v1", help="path to where trajectories and optionally videos of teleoperation are saved to")
    parser.add_argument("--save-video", action="store_true", help="whether to also save videos of the teleoperated results")
    return parser.parse_args()

if __name__ == "__main__":
    main(parse_args())

import gymnasium as gym
import numpy as np
import sapien
import mani_skill.envs
import argparse
from mani_skill.utils.wrappers import RecordEpisode
from mani_skill.examples.teleoperation.vr import MetaQuest3SimTeleopWrapper
from transforms3d import euler
def collect_episode(env: gym.Env, vr: MetaQuest3SimTeleopWrapper):
    ### 1. Calibrate human into the scene and robot
    vr.reset()
    vr.root_pose = sapien.Pose()
    offset_poses = [None, None]
    while True:
        vr.render()
        rp = vr.controller_right_poses
        if vr.get_user_action() == "trigger_2":
            offset_poses = [vr.controller_left_hand_poses, vr.controller_right_poses]
            break

    vr.root_pose = sapien.Pose(-offset_poses[1].p + env.unwrapped.agent.tcp.pose.sp.p)
    while True:
        user_action = vr.get_user_action()
        for obj in vr.base_env._hidden_objects:
            obj.show_visual()
        vr.render()

        rp = vr.root_pose * vr.controller_right_poses

        action = env.action_space.sample() * 0

        # generate the target tcp pose
        q = (rp * sapien.Pose(q=euler.euler2quat(0, np.pi, np.pi))).q
        target_tcp_pose = sapien.Pose(p=rp.p, q=q)

        action[:3] = target_tcp_pose.p
        action[3:7] = target_tcp_pose.q
        gripper_action = 1

        if user_action == "quit":
            return "quit"
        elif user_action == "reset":
            return "reset"
        elif user_action == "trigger_1":
            gripper_action = -1
        action[-1] = gripper_action
        env.step(action)

def main(args):
    env = gym.make(args.env_id, control_mode="pd_ee_pose_quat", enable_shadow=True, render_mode="rgb_array")
    env = RecordEpisode(env, save_video=args.save_video, output_dir=args.record_dir)
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
    parser.add_argument("-e", "--env-id", type=str, default="StackCube-v1")
    parser.add_argument("-o", "--obs-mode", type=str, default="none")
    parser.add_argument("-r", "--robot-uid", type=str, default="panda")
    parser.add_argument("--record-dir", type=str, default="demos", help="path to where trajectories and optionally videos of teleoperation are saved to")
    parser.add_argument("--save-video", action="store_true", help="whether to also save videos of the teleoperated results")
    return parser.parse_args()

if __name__ == "__main__":
    main(parse_args())

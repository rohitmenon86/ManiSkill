import gymnasium as gym
import numpy as np
import sapien
import mani_skill.envs
import argparse
from mani_skill.utils.wrappers import RecordEpisode
from mani_skill.examples.teleoperation.vr import MetaQuest3SimTeleopWrapper
from transforms3d import euler
MOBILE_SPEED = 0.25

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
    init_vr_root_pose = sapien.Pose(-offset_poses[1].p + env.unwrapped.agent.tcp.pose.sp.p)
    vr.root_pose = init_vr_root_pose
    while True:
        vr.root_pose = init_vr_root_pose * sapien.Pose(p=vr.base_env.agent.robot.root_pose.sp.p)

        user_action = vr.get_user_action()
        for obj in vr.base_env._hidden_objects:
            obj.show_visual()
        vr.render()

        rp = vr.root_pose * vr.controller_right_poses

        # TODO (stao): what is the axis int argument for? seems to only work when i set it to 0
        joystick_xy = vr.vr_display.get_controller_axis_state(2, 0)

        action = env.action_space.sample() * 0

        # generate the target tcp pose
        # q = (rp * sapien.Pose(q=euler.euler2quat(0, np.pi, np.pi))).q
        q = (rp * sapien.Pose(q=euler.euler2quat(0, -np.pi, np.pi))).q
        target_tcp_pose = sapien.Pose(p=rp.p, q=q)

        ee_action = np.zeros(7)
        ee_action[:3] = target_tcp_pose.p
        ee_action[3:7] = target_tcp_pose.q
        gripper_action = 1

        if user_action == "quit":
            return "quit"
        elif user_action == "reset":
            return "reset"
        elif user_action == "trigger_1":
            gripper_action = -1
        action[-1] = gripper_action

        base_action = np.zeros([3])

        # determine base action
        base_action[0] = joystick_xy[1] * MOBILE_SPEED
        base_action[2] = -joystick_xy[0] * MOBILE_SPEED


        body_action = np.zeros([3])
        action_dict = dict(base=base_action, arm=ee_action, body=body_action, gripper=gripper_action)
        action = env.agent.controller.from_action_dict(action_dict)
        env.step(action)

def main(args):
    env = gym.make(args.env_id, control_mode="pd_ee_pose_quat", robot_uids="fetch", enable_shadow=True, render_mode="rgb_array")
    env = RecordEpisode(env, save_video=args.save_video, output_dir=args.record_dir)
    # import ipdb;ipdb.set_trace()
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
    parser.add_argument("-r", "--robot-uid", type=str, default="panda")
    parser.add_argument("--record-dir", type=str, default="demos", help="path to where trajectories and optionally videos of teleoperation are saved to")
    parser.add_argument("--save-video", action="store_true", help="whether to also save videos of the teleoperated results")
    return parser.parse_args()

if __name__ == "__main__":
    main(parse_args())

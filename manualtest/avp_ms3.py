import argparse

import gymnasium as gym
import numpy as np
import sapien
from avp_stream import VisionProStreamer
from transforms3d import euler, quaternions

import mani_skill.envs


def avp_convention_to_sapien(xyz):
    res = xyz.copy()
    res[0] = xyz[1]
    res[1] = -xyz[0]
    return res


def mat_to_pose(mat):
    q = quaternions.mat2quat(mat[:3, :3])
    p = avp_convention_to_sapien(mat[:3, 3])
    return sapien.Pose(p=p, q=q)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--env-id",
        type=str,
        default="PushCube-v1",
        help="The environment ID of the task you want to simulate",
    )
    parser.add_argument("--ip", type=str, help="IP address of the Apple Vision Pro")
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        help="Seed the random actions and simulator. Default is no seed",
    )
    args = parser.parse_args()
    return args


def main(args):
    avp_ip = "10.31.181.201"  # example IP
    avp_ip = "2603:8000:8f00:baf6::a45"
    avp_ip = "172.20.8.255"
    avp_streamer = VisionProStreamer(ip=avp_ip, record=True)

    env = gym.make(
        args.env_id, num_envs=1, render_mode="human", control_mode="pd_ee_pose"
    )
    env.reset(seed=0)
    env.render()

    ### CALIBRATION ###

    right_hand_zero = None
    for _ in range(100):
        avp_result = avp_streamer.latest
        right_wrist = avp_result["right_wrist"]
        # avp_result does not have None for things that are not seen, they just track the last value seen
        # pos = right_wrist[0, :3, 3] # (3)
        right_hand_zero = mat_to_pose(right_wrist[0])
    # right_hand_zero = pos
    right_hand_tcp_zero = env.unwrapped.agent.tcp.pose.sp
    print("Calibration done!")
    print(right_hand_tcp_zero)

    # example of ee_pose control
    while True:
        action = env.action_space.sample() * 0

        avp_result = avp_streamer.latest
        right_wrist = avp_result["right_wrist"]
        right_pinch_distance = avp_result["right_pinch_distance"]
        right_wrist = mat_to_pose(right_wrist[0])
        # pos = right_wrist[0, :3, 3] # (3)
        # dorigin = avp_convention_to_sapien(pos - right_hand_zero)

        env.unwrapped.agent.tcp.pose.sp
        # target_tcp_pose = sapien.Pose(p=right_hand_tcp_zero.p + dorigin, q=right_hand_tcp_zero.q)
        target_tcp_pose = right_hand_tcp_zero * right_hand_zero.inv() * right_wrist
        target_tcp_pose = (
            sapien.Pose(q=right_hand_tcp_zero.q)
            * sapien.Pose(q=right_hand_zero.inv().q)
            * sapien.Pose(q=right_wrist.q)
        )
        target_tcp_pose.p = right_hand_tcp_zero.p + (right_wrist.p - right_hand_zero.p)
        # * right_hand_zero.inv() * mat_to_pose(right_wrist[0])
        target_tcp_pose.q

        # quaternions.qmult(right_hand_tcp_zero.q)
        action[:3] = target_tcp_pose.p
        # quaternions.mat2quat()

        eulerxyz = np.array(euler.quat2euler(target_tcp_pose.q))
        print(eulerxyz)
        eulerxyz_temp = eulerxyz.copy()
        eulerxyz[0] *= -1
        eulerxyz[1] = eulerxyz_temp[2]
        eulerxyz[2] = eulerxyz_temp[1]
        action[3:6] = eulerxyz

        # toggle grasping interface
        if right_pinch_distance < 0.02:
            action[-1] = -1  # grasp
        else:
            action[-1] = 1  # ungrasp
        # print(target_tcp_pose)
        # print(right_hand_tcp_zero.p, tcp.p, right_pinch_distance)
        env.step(action)
        env.render()


if __name__ == "__main__":
    main(parse_args())

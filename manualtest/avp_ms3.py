import gymnasium as gym
import numpy as np
import sapien
from avp_stream import VisionProStreamer

import mani_skill.envs


def main():
    avp_ip = "10.31.181.201"  # example IP
    avp_ip = "2603:8000:8f00:baf6::a45"
    # avp_streamer = VisionProStreamer(ip = avp_ip, record = True)

    env = gym.make(
        "StackCube-v1", num_envs=1, render_mode="human", control_mode="pd_ee_pose"
    )
    env.reset(seed=0)
    viewer = env.render()

    # calibration

    # right_hand_zero = np.zeros(3)
    # for _ in range(100):
    #     avp_result = avp_streamer.latest
    #     right_transform = avp_result["right_wrist"]
    #     # avp_result does not have None for things that are not seen, they just track the last value seen
    #     pos = right_transform[0, :3, 3] # (3)

    # right_hand_zero = pos
    right_hand_tcp_zero = env.unwrapped.agent.tcp.pose.sp
    # print("Calibration done!")
    print(right_hand_tcp_zero)

    # def avp_convention_to_sapien(xyz):
    #     res = xyz.copy()
    #     res[0] = xyz[1]
    #     res[1] = xyz[0]
    #     res[-1] = -res[-1]
    #     return res
    viewer.paused = True
    while True:
        # avp_result = avp_streamer.latest
        # right_transform = avp_result["right_wrist"]

        action = env.action_space.sample() * 0
        # # print(right_transform)
        # pos = right_transform[0, :3, 3] # (3)
        # # action[:3] =
        # dorigin = avp_convention_to_sapien(pos - right_hand_zero)

        tcp = env.unwrapped.agent.tcp.pose.sp
        # target_tcp_pose = sapien.Pose(p=right_hand_tcp_zero.p + dorigin, q=right_hand_tcp_zero.q)
        # dpos = target_tcp_pose.p - tcp.p
        # action[:3] = target_tcp_pose.p#dpos# / np.linalg.norm(dpos)
        action[:3] = right_hand_tcp_zero.p
        from transforms3d import euler

        eulerxyz = euler.quat2euler(right_hand_tcp_zero.q)
        action[3:6] = eulerxyz
        # action[0]+=0.1
        print(right_hand_tcp_zero.p, tcp.p)
        # print(target_tcp_pose.p, target_tcp_pose.p - tcp.p, right_hand_tcp_zero.p)
        env.step(action)
        env.render()
        # import time
        # time.sleep(0.1)

    while True:
        r = s.latest
        # print(r['head'], r['right_wrist'], r['right_fingers'])
        print(r["right_wrist"])


if __name__ == "__main__":
    main()

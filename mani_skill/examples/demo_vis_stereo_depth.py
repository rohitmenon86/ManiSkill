import argparse

import gymnasium as gym
import numpy as np

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
import trimesh
import trimesh.scene
def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="PushCube-v1", help="The environment ID of the task you want to simulate")
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        help="Seed the random actions and environment. Default is no seed",
    )
    args = parser.parse_args()
    return args


def main(args):
    if args.seed is not None:
        np.random.seed(args.seed)
    env: BaseEnv = gym.make(
        args.env_id,
        obs_mode="none",
        reward_mode="none",
        render_mode="rgb_array",
        # sensor_cfgs={"use_stereo_depth": True},
        shader_dir="rt-fast"
    )

    obs, _ = env.reset(seed=args.seed)
    import matplotlib.pyplot as plt
    while True:
        rgb=env.render()
        plt.imshow(rgb);plt.show()
    env.close()

if __name__ == "__main__":
    main(parse_args())

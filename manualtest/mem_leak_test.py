import gc

import gymnasium as gym
import psutil

import mani_skill.envs
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


def main():
    process = psutil.Process()
    print(f"{process.memory_info().rss / 1000 / 1000:0.02f}MB")
    env = gym.make("PushCube-v1", num_envs=1, obs_mode="none", reward_mode="sparse")
    env.reset(seed=0)
    i = 0
    while True:
        i += 1
        env.step(None)
        if i % 100 == 0:
            gc.collect()
            print(f"{i}, {process.memory_info().rss / 1000 / 1000:0.02f}MB")


if __name__ == "__main__":
    main()

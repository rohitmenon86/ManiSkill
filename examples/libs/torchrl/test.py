import gymnasium
import mani_skill
from wrapper import ManiSkillWrapper

env = ManiSkillWrapper(gymnasium.make("PushCube-v1", obs_mode="rgbd", num_envs=4))
obs = env.reset(seed=0)
env.rand_step()

import ipdb;ipdb.set_trace()

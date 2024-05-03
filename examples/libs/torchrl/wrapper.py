# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util

import itertools
import warnings
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch

from tensordict import TensorDictBase
from torchrl.envs.libs.gym import GymWrapper
from torchrl.envs.utils import _classproperty, make_composite_from_td

from mani_skill.envs.sapien_env import BaseEnv

# _has_isaac = importlib.util.find_spec("isaacgym") is not None


class ManiSkillWrapper(GymWrapper):
    """Wrapper for ManiSkill environments.

    The original library can be found `here <https://github.com/haosulab/ManiSkill>`
    """

    @property
    def lib(self):
        import mani_skill

        return mani_skill

    def __init__(
        self, env: BaseEnv, **kwargs  # noqa: F821
    ):
        super().__init__(
            env, device=torch.device(env.device), **kwargs
        )

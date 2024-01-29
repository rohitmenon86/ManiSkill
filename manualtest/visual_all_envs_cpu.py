import gymnasium as gym
import numpy as np
import sapien

import mani_skill2.envs
from mani_skill2.utils.sapien_utils import to_numpy
from mani_skill2.utils.wrappers import RecordEpisode

if __name__ == "__main__":
    # sapien.set_log_level("info")
    # , "StackCube-v1", "PickCube-v1", "PushCube-v1", "PickSingleYCB-v1", "OpenCabinet-v1"
    num_envs = 4
    sapien.physx.set_gpu_memory_config(
        found_lost_pairs_capacity=2**26,
        max_rigid_patch_count=2**19,
        max_rigid_contact_count=2**21,
    )
    for env_id in ["StackCube-v1"]:
        env = gym.make(
            env_id,
            num_envs=num_envs,
            enable_shadow=True,
            robot_uid="panda",
            reward_mode="normalized_dense",
            render_mode="rgb_array",
            # control_mode="base_pd_joint_vel_arm_pd_joint_delta_pos",
            control_mode="pd_ee_delta_pos",
            sim_freq=100,
            control_freq=20,
        )
        env = RecordEpisode(
            env,
            output_dir="videos/manual_test",
            trajectory_name=f"{env_id}",
            info_on_video=False,
            video_fps=30,
            save_trajectory=True,
        )
        env.reset(seed=2, options=dict(reconfigure=True))
        # env.reset(seed=1)

        done = False
        i = 0
        if num_envs == 1:
            viewer = env.render_human()
            viewer.paused = True
            env.render_human()
        while i < 50 or (i < 50000 and num_envs == 1):
            action = env.action_space.sample()
            if len(action.shape) == 1:
                action = action.reshape(1, -1)
            action[:] * 0
            # TODO (stao): on cpu sim, -1 here goes up, gpu sim -1 goes down?
            action[:, 2] = -1
            obs, rew, terminated, truncated, info = env.step(action)
            done = np.logical_or(to_numpy(terminated), to_numpy(truncated))
            if num_envs == 1:
                env.render_human()
            done = done.any()
            i += 1
        env.close()

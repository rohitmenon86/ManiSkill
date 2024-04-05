import gymnasium as gym
import mani_skill.envs
import argparse
from mani_skill.utils.wrappers import RecordEpisode
from mani_skill.examples.teleoperation.vr import MetaQuest3SimTeleop

def main(args):
    env = gym.make(args.env_id, control_mode="pd_ee_pose_quat", enable_shadow=True)
    env = RecordEpisode(env, save_video=True, output_dir=args.record_dir)
    env.reset(seed=0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="StackCube-v1")
    parser.add_argument("-o", "--obs-mode", type=str, default="none")
    parser.add_argument("-r", "--robot-uid", type=str, default="panda")
    parser.add_argument("--record-dir", type=str, default="demos", help="path to where trajectories and optionally videos of teleoperation are saved to")
    parser.add_argument("--save-video", action="store_true", help="whether to also save videos of the teleoperated results")
    args, opts = parser.parse_known_args()


if __name__ == "__main__":
    main(parse_args())

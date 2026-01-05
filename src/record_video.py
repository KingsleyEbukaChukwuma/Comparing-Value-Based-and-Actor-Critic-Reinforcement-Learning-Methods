from __future__ import annotations

import argparse
import os

import gymnasium as gym
from stable_baselines3 import DQN, PPO

from src.envs.make_env import make_env



ALGOS = {"dqn": DQN, "ppo": PPO}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_id", type=str, required=True)
    ap.add_argument("--algo", type=str, required=True, choices=list(ALGOS.keys()))
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--reward", type=str, default="v0", choices=["v0", "v1", "v2"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video_dir", type=str, default="videos")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=2000)
    args = ap.parse_args()

    os.makedirs(args.video_dir, exist_ok=True)

    base_env = make_env(args.env_id, seed=args.seed + 12345, reward_variant=args.reward, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        base_env,
        video_folder=args.video_dir,
        episode_trigger=lambda ep: True,
        name_prefix=f"{args.env_id}_{args.algo}_{args.reward}_seed{args.seed}".replace("/", "-"),
    )

    model = ALGOS[args.algo].load(args.model_path)

    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        steps = 0
        ep_return = 0.0
        while not done and steps < args.max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_return += float(reward)
            steps += 1
        print(f"Episode {ep+1}: return={ep_return:.2f}, steps={steps}")

    env.close()
    print("Videos saved to:", args.video_dir)


if __name__ == "__main__":
    main()

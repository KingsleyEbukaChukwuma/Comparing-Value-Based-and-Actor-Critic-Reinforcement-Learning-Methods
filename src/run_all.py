from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
from typing import List


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, default="results")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--reward_variants", type=str, nargs="+", default=["v0", "v1", "v2"])
    ap.add_argument("--envs", type=str, nargs="+", default=["CartPole-v1", "LunarLander-v3"])
    ap.add_argument("--algos", type=str, nargs="+", default=["dqn", "ppo"])
    ap.add_argument("--timesteps_cartpole", type=int, default=150_000)
    ap.add_argument("--timesteps_lunar", type=int, default=600_000)
    ap.add_argument("--configs_dir", type=str, default="configs")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--n_envs_ppo", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.runs_dir, exist_ok=True)

    for env_id, algo, reward, seed in itertools.product(args.envs, args.algos, args.reward_variants, args.seeds):
        env_key = "cartpole" if env_id.lower().startswith("cartpole") else "lunar"
        cfg_path = os.path.join(args.configs_dir, f"{algo}_{env_key}.yaml")

        timesteps = args.timesteps_cartpole if env_key == "cartpole" else args.timesteps_lunar

        cmd: List[str] = [
            sys.executable, "-m", "src.train",
            "--config", cfg_path,
            "--env_id", env_id,
            "--algo", algo,
            "--reward", reward,
            "--seed", str(seed),
            "--timesteps", str(timesteps),
            "--runs_dir", args.runs_dir,
            "--device", args.device,
        ]
        if algo == "ppo":
            cmd += ["--n_envs", str(args.n_envs_ppo)]

        print("\nRUN:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

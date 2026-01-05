from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from typing import Any, Dict, List

import numpy as np
import yaml
import torch

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from src.envs.make_env import make_env, make_vec_env


from src.utils import ensure_dir, get_versions, save_json

def sanitize_for_yaml(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_yaml(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_yaml(v) for v in obj]
    if hasattr(obj, "__module__") and obj.__module__.startswith("torch"):
        return getattr(obj, "__name__", str(obj))
    return obj



def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sample_from(values: List[Any]) -> Any:
    return random.choice(values)


def build_policy_kwargs_sample(space: Dict[str, Any]) -> Dict[str, Any]:
    pk: Dict[str, Any] = {}
    if not space:
        return pk

    if "net_arch" in space:
        pk["net_arch"] = sample_from(space["net_arch"])

    if "activation_fn" in space:
        act = sample_from(space["activation_fn"])
        if isinstance(act, str):
            a = act.lower()
            if a == "relu":
                pk["activation_fn"] = torch.nn.ReLU
            elif a == "tanh":
                pk["activation_fn"] = torch.nn.Tanh
            elif a == "elu":
                pk["activation_fn"] = torch.nn.ELU
            else:
                raise ValueError(f"Unknown activation_fn: {act}")
    return pk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_config", type=str, required=True)
    ap.add_argument("--env_id", type=str, required=True)
    ap.add_argument("--algo", type=str, required=True, choices=["dqn", "ppo"])
    ap.add_argument("--reward", type=str, default="v0", choices=["v0", "v1", "v2"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_trials", type=int, default=20)
    ap.add_argument("--train_timesteps", type=int, default=200_000)
    ap.add_argument("--eval_episodes", type=int, default=20)
    ap.add_argument("--eval_freq", type=int, default=10_000)
    ap.add_argument("--n_envs", type=int, default=4)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--out_dir", type=str, default="tuning")
    args = ap.parse_args()

    set_global_seeds(args.seed)

    cfg = load_yaml(args.base_config)
    base_kwargs: Dict[str, Any] = dict(cfg.get("model_kwargs") or {})
    policy_space: Dict[str, Any] = cfg.get("policy_kwargs_space") or {}
    dqn_space: Dict[str, List[Any]] = cfg.get("dqn_space") or {}
    ppo_space: Dict[str, List[Any]] = cfg.get("ppo_space") or {}

    out_dir = ensure_dir(args.out_dir)
    save_json(os.path.join(out_dir, "versions.json"), get_versions())
    with open(os.path.join(out_dir, "base_config_used.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    results_csv = os.path.join(out_dir, f"{args.env_id}_{args.algo}_{args.reward}_tuning.csv".replace("/", "-"))
    header = ["trial", "eval_mean", "eval_std", "train_time_seconds", "sampled_kwargs_json"]
    if not os.path.exists(results_csv):
        with open(results_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

    Algo = DQN if args.algo == "dqn" else PPO
    space = dqn_space if args.algo == "dqn" else ppo_space

    best_mean = -1e18
    best_kwargs = None

    for t in range(1, args.n_trials + 1):
        sampled = dict(base_kwargs)
        for k, vals in space.items():
            sampled[k] = sample_from(vals)

        pk = build_policy_kwargs_sample(policy_space)
        if pk:
            sampled["policy_kwargs"] = pk

        train_env = None
        eval_env = None
        mean_r = float("nan")
        std_r = float("nan")
        train_time = float("nan")

        try:
            trial_seed = args.seed + 1000 * t
            if args.algo == "ppo":
                train_env = make_vec_env(args.env_id, n_envs=args.n_envs, seed=trial_seed, reward_variant=args.reward)
                eval_env = make_env(args.env_id, seed=trial_seed + 9999, reward_variant=args.reward)
            else:
                train_env = make_env(args.env_id, seed=trial_seed, reward_variant=args.reward)
                eval_env = make_env(args.env_id, seed=trial_seed + 9999, reward_variant=args.reward)

            model = Algo(
                policy="MlpPolicy",
                env=train_env,
                verbose=0,
                seed=trial_seed,
                device=args.device,
                **sampled,
            )

            eval_cb = EvalCallback(
                eval_env,
                eval_freq=args.eval_freq,
                n_eval_episodes=args.eval_episodes,
                deterministic=True,
                render=False,
            )

            start = time.time()
            model.learn(total_timesteps=args.train_timesteps, callback=eval_cb, progress_bar=False)
            train_time = time.time() - start

            mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=args.eval_episodes, deterministic=True)

            with open(results_csv, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([t, float(mean_r), float(std_r), float(train_time), json.dumps(sampled, sort_keys=True, default=str)])

            if mean_r > best_mean:
                best_mean = float(mean_r)
                best_kwargs = sampled
                with open(
                    os.path.join(
                        out_dir,
                        f"best_{args.env_id}_{args.algo}_{args.reward}.yaml".replace("/", "-"),
                    ),
                    "w",
                    encoding="utf-8",
                ) as f:
                    safe_sampled = sanitize_for_yaml(sampled)
                    yaml.safe_dump({"model_kwargs": safe_sampled}, f, sort_keys=False)

        finally:
            try:
                if eval_env is not None:
                    eval_env.close()
            except Exception:
                pass
            try:
                if train_env is not None:
                    train_env.close()
            except Exception:
                pass

        print(f"[{t}/{args.n_trials}] eval={mean_r:.2f}±{std_r:.2f} time={train_time:.1f}s")

    print("Best mean:", best_mean)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import os
import random
import time
from typing import Any, Dict

import numpy as np
import yaml
import torch

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from src.envs.make_env import make_env, make_vec_env

from src.utils import ensure_dir, get_versions, save_json



ALGOS = {"dqn": DQN, "ppo": PPO}


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_policy_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    pk = cfg.get("model_kwargs", {}).get("policy_kwargs") or {}
    if not pk:
        return {}

    act = pk.get("activation_fn")
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
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--env_id", type=str, required=True)
    ap.add_argument("--algo", type=str, required=True, choices=list(ALGOS.keys()))
    ap.add_argument("--reward", type=str, default="v0", choices=["v0", "v1", "v2"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timesteps", type=int, required=True)
    ap.add_argument("--eval_episodes", type=int, default=20)
    ap.add_argument("--eval_freq", type=int, default=10_000)
    ap.add_argument("--n_envs", type=int, default=1)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--runs_dir", type=str, default="runs")
    ap.add_argument("--run_name", type=str, default=None)
    ap.add_argument("--save_replay", action="store_true")
    args = ap.parse_args()

    set_global_seeds(args.seed)

    cfg = load_yaml(args.config)
    run_name = args.run_name or f"{args.env_id.lower()}_{args.algo}_{args.reward}_seed{args.seed}_{int(time.time())}"
    run_dir = ensure_dir(os.path.join(args.runs_dir, run_name))
    tb_dir = ensure_dir(os.path.join(run_dir, "tb"))
    ensure_dir(os.path.join(run_dir, "eval"))
    ensure_dir(os.path.join(run_dir, "best_model"))
    ensure_dir(os.path.join(run_dir, "monitor_train"))
    ensure_dir(os.path.join(run_dir, "monitor_eval"))

    # Save reproducibility info
    save_json(os.path.join(run_dir, "versions.json"), get_versions())
    with open(os.path.join(run_dir, "config_used.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    AlgoCls = ALGOS[args.algo]

    # Build envs
    if args.algo == "ppo":
        train_env = make_vec_env(args.env_id, n_envs=args.n_envs, seed=args.seed, reward_variant=args.reward, monitor_dir=os.path.join(run_dir, "monitor_train"))
        eval_env = make_env(args.env_id, seed=args.seed + 10_000, reward_variant=args.reward, monitor_dir=os.path.join(run_dir, "monitor_eval"))
    else:
        train_env = make_env(args.env_id, seed=args.seed, reward_variant=args.reward, monitor_dir=os.path.join(run_dir, "monitor_train"))
        eval_env = make_env(args.env_id, seed=args.seed + 10_000, reward_variant=args.reward, monitor_dir=os.path.join(run_dir, "monitor_eval"))

    model_kwargs: Dict[str, Any] = dict(cfg.get("model_kwargs") or {})
    # Fix activation_fn string if provided
    pk = build_policy_kwargs(cfg)
    if pk:
        model_kwargs["policy_kwargs"] = pk

    model = AlgoCls(
        policy="MlpPolicy",
        env=train_env,
        tensorboard_log=tb_dir,
        verbose=0,
        seed=args.seed,
        device=args.device,
        **model_kwargs,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(run_dir, "best_model"),
        log_path=os.path.join(run_dir, "eval"),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )

    start = time.time()
    model.learn(total_timesteps=args.timesteps, callback=eval_cb, progress_bar=True)
    train_time = time.time() - start

    mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=args.eval_episodes, deterministic=True)

    model.save(os.path.join(run_dir, "final_model.zip"))
    if args.algo == "dqn" and args.save_replay:
        try:
            model.save_replay_buffer(os.path.join(run_dir, "replay_buffer.pkl"))
        except Exception as e:
            with open(os.path.join(run_dir, "replay_save_error.txt"), "w", encoding="utf-8") as f:
                f.write(str(e))

    summary = {
        "run_name": run_name,
        "env_id": args.env_id,
        "algo": args.algo,
        "reward": args.reward,
        "seed": args.seed,
        "timesteps": args.timesteps,
        "eval_episodes": args.eval_episodes,
        "eval_freq": args.eval_freq,
        "n_envs": args.n_envs if args.algo == "ppo" else 1,
        "final_eval_mean_return": float(mean_r),
        "final_eval_std_return": float(std_r),
        "train_time_seconds": float(train_time),
        "model_kwargs": model_kwargs,
    }
    save_json(os.path.join(run_dir, "summary.json"), summary)

    try:
        eval_env.close()
    except Exception:
        pass
    try:
        train_env.close()
    except Exception:
        pass

    print(f"Saved run to: {run_dir}")
    print(f"Final eval mean±std: {mean_r:.3f} ± {std_r:.3f}")
    print(f"Train time: {train_time:.1f}s")


if __name__ == "__main__":
    main()

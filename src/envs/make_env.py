from __future__ import annotations

from typing import Callable, Optional

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

from .reward_wrappers import maybe_wrap_reward



def make_env(
    env_id: str,
    seed: int,
    reward_variant: str = "v0",
    monitor_dir: Optional[str] = None,
    render_mode: Optional[str] = None,
) -> gym.Env:
    env = gym.make(env_id, render_mode=render_mode)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    env = maybe_wrap_reward(env, reward_variant)

    if monitor_dir is None:
        env = Monitor(env)
    else:
        env = Monitor(env, filename=f"{monitor_dir}/monitor.csv")

    return env


def make_vec_env(
    env_id: str,
    n_envs: int,
    seed: int,
    reward_variant: str = "v0",
    monitor_dir: Optional[str] = None,
):
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    def thunk(rank: int) -> Callable[[], gym.Env]:
        def _init():
            return make_env(
                env_id=env_id,
                seed=seed + rank,
                reward_variant=reward_variant,
                monitor_dir=None if monitor_dir is None else f"{monitor_dir}/env{rank}",
            )
        return _init

    fns = [thunk(i) for i in range(n_envs)]
    Vec = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    return Vec(fns)

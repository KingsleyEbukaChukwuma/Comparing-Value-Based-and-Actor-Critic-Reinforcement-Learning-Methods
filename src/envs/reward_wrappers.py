from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import gymnasium as gym


@dataclass(frozen=True)
class RewardConfig:
    variant: str = "v0"  # "v0" | "v1" | "v2"


class RewardShapingWrapper(gym.Wrapper):
    """
    Wraps an env and modifies rewards according to a named variant.

    IMPORTANT:
    - Keep 'v0' as the original reward (no changes).
    - For grading, treat reward shaping as an ablation: compare v0 vs v1 vs v2
      using the SAME best hyperparameters per (env, algo).

    Supported envs:
    - CartPole-v1 (minimal shaping; illustrative)
    - LunarLander-v3 (use caution: base env is already shaped)
    """

    def __init__(self, env: gym.Env, cfg: RewardConfig):
        super().__init__(env)
        self.cfg = cfg

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaped = float(reward)

        env_id = getattr(self.env.unwrapped.spec, "id", "")

        if self.cfg.variant == "v0":
            return obs, shaped, terminated, truncated, info

        if env_id.startswith("CartPole"):
            shaped = self._shape_cartpole(obs, shaped, variant=self.cfg.variant)
        elif env_id.startswith("LunarLander"):
            shaped = self._shape_lunarlander(obs, shaped, variant=self.cfg.variant)

        info = dict(info)
        info["reward_raw"] = float(reward)
        info["reward_shaped"] = float(shaped)
        info["reward_variant"] = self.cfg.variant
        return obs, shaped, terminated, truncated, info

    @staticmethod
    def _shape_cartpole(obs: np.ndarray, reward: float, variant: str) -> float:
        """
        v1: tiny penalty for large pole angle
        v2: v1 + tiny penalty for cart position away from center
        """
        x, x_dot, theta, theta_dot = obs
        if variant in ("v1", "v2"):
            reward -= 0.01 * abs(theta)
        if variant == "v2":
            reward -= 0.005 * abs(x)
        return float(reward)

    @staticmethod
    def _shape_lunarlander(obs: np.ndarray, reward: float, variant: str) -> float:
        """
        LunarLander base reward is already shaped. We only apply small, state-dependent shaping.

        Observation:
        [x, y, vx, vy, angle, angular_velocity, left_leg_contact, right_leg_contact]

        v1: penalize high horizontal speed + tilt near ground (y < 0.6)
        v2: v1 + mild penalty for angular velocity near ground
        """
        x, y, vx, vy, angle, ang_vel, left, right = obs
        near_ground = y < 0.60
        if not near_ground:
            return float(reward)

        reward -= 0.03 * abs(vx)
        reward -= 0.02 * abs(angle)

        if variant == "v2":
            reward -= 0.01 * abs(ang_vel)

        return float(reward)


def maybe_wrap_reward(env: gym.Env, reward_variant: str) -> gym.Env:
    reward_variant = (reward_variant or "v0").lower()
    if reward_variant == "v0":
        return env
    return RewardShapingWrapper(env, RewardConfig(variant=reward_variant))

from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime
from typing import Any, Dict

import gymnasium
import stable_baselines3


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def get_versions() -> Dict[str, str]:
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "gymnasium": getattr(gymnasium, "__version__", "unknown"),
        "stable_baselines3": getattr(stable_baselines3, "__version__", "unknown"),
    }


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

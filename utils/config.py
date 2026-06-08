from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyYAML is required: pip install pyyaml") from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PATH_KEYS = {"train_csv", "val_csv", "save_dir", "weights", "config", "repo_path"}


def _resolve_path(value: str, project_root: Path) -> str:
    expanded = Path(os.path.expanduser(os.path.expandvars(value)))
    if expanded.is_absolute():
        return str(expanded)
    return str(project_root / expanded)


def _resolve_paths(obj: Any, project_root: Path, key: str | None = None) -> Any:
    if isinstance(obj, dict):
        return {k: _resolve_paths(v, project_root, k) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_paths(v, project_root, key) for v in obj]
    if isinstance(obj, str):
        expanded = os.path.expanduser(os.path.expandvars(obj))
        if key in PATH_KEYS and expanded:
            return _resolve_path(expanded, project_root)
        return expanded
    return obj


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return _resolve_paths(cfg, PROJECT_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-branch metric learning trainer")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="Path to YAML config")
    parser.add_argument("--run_name", type=str, default="experiment",
                        help="Run name used for checkpoints")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device (cpu, cuda, or cuda:0)")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Override dataloader worker count")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--resume", type=str, default=None,
                        help="Optional checkpoint path to resume")
    return parser.parse_args()


def resolve_device(cfg_device: str | None, arg_device: str | None) -> str:
    if arg_device:
        return arg_device
    if cfg_device and cfg_device != "auto":
        return cfg_device
    return "cuda" if torch_cuda_available() else "cpu"


def torch_cuda_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    return torch.cuda.is_available()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

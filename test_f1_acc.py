from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.multiprocessing as mp
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

# Ensure project root (parent of sofia package) is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sofia.data.audio_dataset import AudioSampleDataset, MultiBranchCollate
from sofia.models.fusion import MultiBranchFusion
from sofia.train import build_encoders, select_head
from sofia.utils.config import load_config, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate model with ACC and F1")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint to load")
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu, cuda, or cuda:0)")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--num_workers", type=int, default=None, help="Override dataloader worker count")
    parser.add_argument("--csv", type=str, required=True, help="CSV file to evaluate (expects label/full_path columns)")
    parser.add_argument("--amp", action="store_true", help="Force amp on for eval (defaults to config value)")
    parser.add_argument("--no-amp", dest="no_amp", action="store_true", help="Force amp off for eval")
    parser.add_argument("--save_attn", type=str, default=None, help="Optional path to write per-sample attention weights as CSV")
    parser.add_argument("--print_attn_mean", action="store_true", help="Print mean attention weight per branch over the dataset")
    return parser.parse_args()


def build_loader(csv_path: str, data_cfg: Dict, collate: MultiBranchCollate, batch_size: int, num_workers: int) -> DataLoader:
    dataset = AudioSampleDataset(
        csv_path=csv_path,
        base_sample_rate=int(data_cfg["base_sample_rate"]),
        segment_seconds=float(data_cfg["segment_seconds"]),
        random_crop=False,
        mono=bool(data_cfg.get("mono", True)),
        normalize=bool(data_cfg.get("normalize", True)),
        demucs_enabled=bool(data_cfg.get("demucs_enabled", True)),
        demucs_model=str(data_cfg.get("demucs_model", "htdemucs")),
        demucs_device=str(data_cfg.get("demucs_device", "cpu")),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(1, num_workers),
        pin_memory=True,
        collate_fn=collate,
        multiprocessing_context="spawn",
        persistent_workers=True,
    )


def load_models(cfg: Dict, device: torch.device, checkpoint: str):
    branches_cfg = cfg.get("audio_branches", {})
    enabled_branches = {k: v for k, v in branches_cfg.items() if v.get("enabled", False)}
    if not enabled_branches:
        raise ValueError("No audio branches enabled in config")

    encoders = build_encoders(enabled_branches, device=device)

    fusion_cfg = cfg["fusion"]
    fusion = MultiBranchFusion(
        branch_input_dims={name: cfg_item["embed_dim"] for name, cfg_item in enabled_branches.items()},
        projector_dim=int(fusion_cfg["projector_dim"]),
        fusion_hidden=fusion_cfg.get("fusion_hidden", []),
        fusion_out=int(fusion_cfg["fusion_out"]),
        dropout=float(fusion_cfg.get("dropout", 0.1)),
        l2_norm=bool(fusion_cfg.get("l2_norm", True)),
        mode=str(fusion_cfg.get("mode", "concat")),
    ).to(device)

    head = select_head(cfg, embedding_dim=int(fusion_cfg["fusion_out"]), num_classes=int(cfg["training"]["num_classes"]))
    head = head.to(device)

    state = torch.load(checkpoint, map_location="cpu")
    fusion.load_state_dict(state["fusion"])
    head.load_state_dict(state["head"])

    fusion.eval()
    head.eval()
    for enc in encoders.values():
        enc.eval()

    return encoders, fusion, head


def evaluate_f1(
    encoders: Dict[str, torch.nn.Module],
    fusion: MultiBranchFusion,
    head: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    freeze_encoders: bool,
    collect_outputs: bool = False,
):
    total_loss = 0.0
    total_samples = 0
    correct = 0
    all_labels = []
    all_preds = []
    attn_weights = []
    attn_branch_names = fusion.branch_names if collect_outputs else None

    fusion.eval()
    head.eval()
    for enc in encoders.values():
        enc.eval()

    with torch.no_grad():
        for batch in loader:
            labels = batch["label"].to(device, non_blocking=True)
            audio = {k: v.to(device, non_blocking=True) for k, v in batch["audio"].items()}

            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=amp):
                embeddings = {}
                for name, encoder in encoders.items():
                    embeddings[name] = encoder(audio[name], sample_rate=encoder.target_sr)
                if collect_outputs:
                    fused, extras = fusion(embeddings, return_extras=True)
                    if extras.get("attn_weights") is not None:
                        attn_weights.append(extras["attn_weights"].detach().cpu())
                else:
                    fused = fusion(embeddings)
                logits = head(fused, labels)
                loss = torch.nn.functional.cross_entropy(logits, labels)

            preds = torch.argmax(logits, dim=1)
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
            correct += (preds == labels).sum().item()

            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())

    acc = correct / total_samples if total_samples > 0 else 0.0
    y_true = torch.cat(all_labels).numpy() if all_labels else []
    y_pred = torch.cat(all_preds).numpy() if all_preds else []
    f1 = f1_score(y_true, y_pred, average="binary", zero_division=0) if len(y_true) > 0 else 0.0

    stats = {
        "loss": total_loss / total_samples if total_samples > 0 else 0.0,
        "acc": acc,
        "f1": f1,
    }

    if collect_outputs and attn_weights:
        merged = torch.cat(attn_weights, dim=0)
        stats["attn_weights"] = merged
        stats["attn_branch_names"] = attn_branch_names
        stats["attn_mean"] = merged.mean(dim=0)
    return stats


def main() -> None:
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)

    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device(resolve_device(cfg.get("device"), args.device))

    data_cfg = cfg["data"]
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at {csv_path}")

    branches_cfg = cfg.get("audio_branches", {})
    enabled_branches = {k: v for k, v in branches_cfg.items() if v.get("enabled", False)}
    collate = MultiBranchCollate(
        branch_cfg=enabled_branches,
        segment_seconds=float(data_cfg["segment_seconds"]),
        random_crop=False,
    )

    batch_size = args.batch_size or int(data_cfg.get("batch_size", 16))
    num_workers = args.num_workers or int(data_cfg.get("num_workers", 2))

    eval_loader = build_loader(str(csv_path), data_cfg, collate, batch_size, num_workers)

    encoders, fusion, head = load_models(cfg, device, args.checkpoint)

    train_amp = bool(cfg["training"].get("amp", True))
    amp = train_amp
    if args.amp:
        amp = True
    if args.no_amp:
        amp = False

    freeze_encoders = True
    collect_attn = bool(args.save_attn or args.print_attn_mean)

    stats = evaluate_f1(
        encoders,
        fusion,
        head,
        eval_loader,
        device=device,
        amp=amp,
        freeze_encoders=freeze_encoders,
        collect_outputs=collect_attn,
    )
    print(f"[eval] loss {stats['loss']:.4f} | acc {stats['acc']:.4f} | f1 {stats['f1']:.4f}")

    if collect_attn and stats.get("attn_weights") is not None:
        branch_names = stats.get("attn_branch_names", fusion.branch_names)
        if args.print_attn_mean and stats.get("attn_mean") is not None:
            print("[attn] mean weight per branch:")
            for name, w in zip(branch_names, stats["attn_mean"].tolist()):
                print(f"  {name}: {w:.4f}")

        if args.save_attn:
            attn_weights = stats["attn_weights"].tolist()
            save_path = Path(args.save_attn)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with save_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["sample_idx", *branch_names])
                for idx, row in enumerate(attn_weights):
                    writer.writerow([idx, *row])
            print(f"[attn] saved per-sample attention weights to {save_path}")


if __name__ == "__main__":
    main()

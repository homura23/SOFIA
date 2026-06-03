from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sofia.data.audio_dataset import AudioSampleDataset, MultiBranchCollate
from sofia.models.fusion import MultiBranchFusion
from sofia.train import build_encoders
from sofia.utils.config import load_config, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-pair feature distance analysis")
    parser.add_argument("--config", type=str, default="config/moe.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--csv", type=str, nargs="+", default=None, help="Legacy combined CSVs")
    parser.add_argument("--ai_csv", type=str, nargs="+", default=None, help="AI CSVs (one or more)")
    parser.add_argument("--real_csv", type=str, default=None, help="Real CSV (single)")
    parser.add_argument("--ai_index", type=int, default=None)
    parser.add_argument("--real_index", type=int, default=None)
    parser.add_argument("--pairs_per_ai", type=int, default=None, help="Random pairs per AI CSV")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ai_label", type=int, default=None, help="Optional label check for AI sample")
    parser.add_argument("--real_label", type=int, default=None, help="Optional label check for real sample")
    parser.add_argument("--metric", type=str, choices=["cosine", "l2"], default="cosine")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--amp", action="store_true", help="Force AMP on")
    parser.add_argument("--no-amp", dest="no_amp", action="store_true", help="Force AMP off")
    parser.add_argument("--plot_path", type=str, default=None)
    parser.add_argument("--save_json", type=str, default=None)
    return parser.parse_args()


def load_models(cfg: Dict, device: torch.device, checkpoint: str) -> Tuple[Dict[str, torch.nn.Module], MultiBranchFusion]:
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
        moe_routing=str(fusion_cfg.get("moe_routing", "dense")),
        top_k=int(fusion_cfg.get("top_k", 1)),
    ).to(device)

    state = torch.load(checkpoint, map_location="cpu")
    fusion.load_state_dict(state["fusion"])

    fusion.eval()
    for enc in encoders.values():
        enc.eval()

    return encoders, fusion


def apply_out_proj(out_proj: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    if isinstance(out_proj, torch.nn.Identity):
        return x
    if x.dim() == 2:
        return out_proj(x)
    b, n, d = x.shape
    flat = x.reshape(b * n, d)
    proj = out_proj(flat)
    return proj.reshape(b, n, -1)


def compute_components(fusion: MultiBranchFusion, embeddings: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if fusion.mode not in {"moe", "mixture_of_experts", "complex_moe", "complex_mixture_of_experts"}:
        raise ValueError(f"Fusion mode {fusion.mode} does not use MOE routing")

    projected = []
    for name in fusion.branch_names:
        if name not in embeddings:
            raise KeyError(f"Missing embedding for branch '{name}'")
        # Encoder output per branch (raw feature vector)
        x = embeddings[name]
        # Project to a shared dimension for fusion
        x = F.normalize(x, dim=1, eps=1e-6)
        x = fusion.projectors[name](x)
        x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        projected.append(x)

    h = torch.cat(projected, dim=1)
    stacked = torch.stack(projected, dim=1)
    # Route to experts using the shared concatenated features
    gate_logits = fusion.model_gate(h)
    alpha, _ = fusion._compute_moe_weights(gate_logits)
    expert_out = torch.stack(
        [fusion.experts[name](x) for name, x in zip(fusion.branch_names, projected)],
        dim=1,
    )

    # MOE weighted sum plus residual mean
    z_moe = (alpha.unsqueeze(-1) * expert_out).sum(dim=1)
    residual = stacked.mean(dim=1)
    # Out projection + optional L2 normalization to match inference output
    fused_pre = apply_out_proj(fusion.out_proj, z_moe + residual)
    fused = torch.nan_to_num(fused_pre, nan=0.0, posinf=1e4, neginf=-1e4)
    if fusion.l2_norm:
        fused = F.normalize(fused, dim=1, eps=1e-6)

    return {
        "projected": stacked,
        "expert_out": expert_out,
        "z_moe": z_moe,
        "residual": residual,
        "fused": fused,
    }


def vector_distance(a: torch.Tensor, b: torch.Tensor, metric: str) -> float:
    if metric == "cosine":
        return float(1.0 - F.cosine_similarity(a, b, dim=0).item())
    if metric == "l2":
        return float(torch.norm(a - b).item())
    raise ValueError(f"Unsupported metric: {metric}")


def build_datasets(csv_paths: Iterable[str], data_cfg: Dict) -> List[AudioSampleDataset]:
    datasets: List[AudioSampleDataset] = []
    for path in csv_paths:
        datasets.append(
            AudioSampleDataset(
                csv_path=path,
                base_sample_rate=int(data_cfg["base_sample_rate"]),
                segment_seconds=float(data_cfg["segment_seconds"]),
                random_crop=False,
                mono=bool(data_cfg.get("mono", True)),
                normalize=bool(data_cfg.get("normalize", True)),
                demucs_enabled=bool(data_cfg.get("demucs_enabled", True)),
                demucs_model=str(data_cfg.get("demucs_model", "htdemucs")),
                demucs_device=str(data_cfg.get("demucs_device", "cpu")),
            )
        )
    return datasets


def get_item_by_index(datasets: List[AudioSampleDataset], csv_paths: List[str], index: int) -> Tuple[Dict, str]:
    offset = 0
    for dataset, path in zip(datasets, csv_paths):
        next_offset = offset + len(dataset)
        if index < next_offset:
            return dataset[index - offset], path
        offset = next_offset
    raise IndexError(f"Index {index} out of range for combined dataset length {offset}")


def build_collate(cfg: Dict) -> MultiBranchCollate:
    data_cfg = cfg["data"]
    branches_cfg = cfg.get("audio_branches", {})
    enabled_branches = {k: v for k, v in branches_cfg.items() if v.get("enabled", False)}
    return MultiBranchCollate(
        branch_cfg=enabled_branches,
        segment_seconds=float(data_cfg["segment_seconds"]),
        random_crop=False,
    )


def analyze_items(
    encoders: Dict[str, torch.nn.Module],
    fusion: MultiBranchFusion,
    collate: MultiBranchCollate,
    ai_item: Dict,
    real_item: Dict,
    ai_csv: str,
    real_csv: str,
    ai_label: int | None,
    real_label: int | None,
    device: torch.device,
    amp: bool,
    metric: str,
    plot_path: str | None,
    save_json: str | None,
) -> None:
    if ai_label is not None and int(ai_item["label"]) != ai_label:
        raise ValueError(f"AI label mismatch: expected {ai_label}, got {int(ai_item['label'])}")
    if real_label is not None and int(real_item["label"]) != real_label:
        raise ValueError(f"Real label mismatch: expected {real_label}, got {int(real_item['label'])}")

    ai_source = str(ai_item.get("source", ""))
    real_source = str(real_item.get("source", ""))
    # Build a 2-sample batch so distances are computed between AI and real samples
    batch = collate([ai_item, real_item])
    audio = {k: v.to(device) for k, v in batch["audio"].items()}

    with torch.no_grad(), torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=amp):
        embeddings = {name: enc(audio[name], sample_rate=enc.target_sr) for name, enc in encoders.items()}
        components = compute_components(fusion, embeddings)

    # Distance per stage: encoder -> projector -> expert -> fused
    out: Dict[str, Dict[str, float] | float] = {}
    out["encoder_raw"] = {}
    for name in fusion.branch_names:
        emb = embeddings[name].detach().cpu()
        out["encoder_raw"][name] = vector_distance(emb[0], emb[1], metric)

    out["projected"] = {}
    for idx, name in enumerate(fusion.branch_names):
        proj = components["projected"][:, idx, :].detach().cpu()
        out["projected"][name] = vector_distance(proj[0], proj[1], metric)

    out["expert_out"] = {}
    for idx, name in enumerate(fusion.branch_names):
        exp = components["expert_out"][:, idx, :].detach().cpu()
        out["expert_out"][name] = vector_distance(exp[0], exp[1], metric)

    z_moe = components["z_moe"].detach().cpu()
    residual = components["residual"].detach().cpu()
    fused = components["fused"].detach().cpu()
    out["z_moe"] = vector_distance(z_moe[0], z_moe[1], metric)
    out["residual"] = vector_distance(residual[0], residual[1], metric)
    out["fused"] = vector_distance(fused[0], fused[1], metric)

    result = {
        "metric": metric,
        "ai_csv": ai_csv,
        "real_csv": real_csv,
        "ai_csv": ai_csv,
        "real_csv": real_csv,
        "ai_source": ai_source,
        "real_source": real_source,
        "branch_names": fusion.branch_names,
        "distances": out,
    }

    if save_json:
        save_path = Path(save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"[pair] saved analysis to {save_path}")
    else:
        print(json.dumps(result, indent=2))

    if plot_path:
        if plt is None:
            raise RuntimeError("matplotlib is required for plotting")

        stages = ["encoder_raw", "projected", "expert_out"]
        fig, ax = plt.subplots(figsize=(10, 4))
        for name in fusion.branch_names:
            values = [out[stage][name] for stage in stages]
            ax.plot(stages, values, marker="o", label=name)
        ax.set_title(f"Per-branch distance across stages (AI: {ai_source}, Real: {real_source})")
        ax.set_ylabel(f"{metric} distance")
        ax.legend(loc="best", fontsize=8)

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.bar(["z_moe", "residual", "fused"], [out["z_moe"], out["residual"], out["fused"]], color="#4C72B0")
        ax2.set_title(f"Global distance components (AI: {ai_source}, Real: {real_source})")
        ax2.set_ylabel(f"{metric} distance")

        plot_path = Path(plot_path)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig2.tight_layout()
        fig.savefig(plot_path.with_suffix(".per_branch.png"), dpi=200)
        fig2.savefig(plot_path.with_suffix(".global.png"), dpi=200)
        plt.close(fig)
        plt.close(fig2)
        print(f"[pair] saved plots to {plot_path.with_suffix('.per_branch.png')} and {plot_path.with_suffix('.global.png')}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(resolve_device(cfg.get("device"), args.device))

    encoders, fusion = load_models(cfg, device, args.checkpoint)

    train_amp = bool(cfg["training"].get("amp", True))
    amp = train_amp
    if args.amp:
        amp = True
    if args.no_amp:
        amp = False

    if args.ai_csv or args.real_csv:
        if not args.ai_csv or not args.real_csv:
            raise ValueError("Both --ai_csv and --real_csv are required")
        ai_csv_paths = args.ai_csv
        real_csv_path = args.real_csv
    else:
        if not args.csv:
            raise ValueError("--csv is required when --ai_csv/--real_csv not provided")
        ai_csv_paths = args.csv
        real_csv_path = args.csv[0]

    collate = build_collate(cfg)

    if args.pairs_per_ai is not None:
        rng = random.Random(args.seed)
        ai_datasets = build_datasets(ai_csv_paths, cfg["data"])
        real_dataset = build_datasets([real_csv_path], cfg["data"])[0]
        if len(real_dataset) == 0:
            raise ValueError("Real CSV has no samples")

        for ai_csv, ai_dataset in zip(ai_csv_paths, ai_datasets):
            if len(ai_dataset) == 0:
                continue
            sample_count = min(args.pairs_per_ai, len(ai_dataset))
            ai_indices = rng.sample(range(len(ai_dataset)), k=sample_count)
            for offset, ai_idx in enumerate(ai_indices):
                real_idx = rng.randrange(len(real_dataset))
                suffix = f"ai{ai_idx}_real{real_idx}_p{offset}"
                if args.plot_path:
                    plot_path = f"{args.plot_path}_{suffix}"
                else:
                    plot_path = None
                analyze_items(
                    encoders=encoders,
                    fusion=fusion,
                    collate=collate,
                    ai_item=ai_dataset[ai_idx],
                    real_item=real_dataset[real_idx],
                    ai_csv=ai_csv,
                    real_csv=real_csv_path,
                    ai_label=args.ai_label,
                    real_label=args.real_label,
                    device=device,
                    amp=amp,
                    metric=args.metric,
                    plot_path=plot_path,
                    save_json=None,
                )
        return

    if args.ai_index is None or args.real_index is None:
        raise ValueError("--ai_index and --real_index are required without --pairs_per_ai")

    ai_datasets = build_datasets(ai_csv_paths, cfg["data"])
    ai_total = sum(len(ds) for ds in ai_datasets)
    if args.ai_index < 0 or args.ai_index >= ai_total:
        raise IndexError(f"AI index {args.ai_index} out of range")

    real_datasets = build_datasets([real_csv_path], cfg["data"])
    real_total = len(real_datasets[0])
    if args.real_index < 0 or args.real_index >= real_total:
        raise IndexError(f"Real index {args.real_index} out of range")

    ai_item, ai_csv = get_item_by_index(ai_datasets, ai_csv_paths, args.ai_index)
    real_item = real_datasets[0][args.real_index]

    analyze_items(
        encoders=encoders,
        fusion=fusion,
        collate=collate,
        ai_item=ai_item,
        real_item=real_item,
        ai_csv=ai_csv,
        real_csv=real_csv_path,
        ai_label=args.ai_label,
        real_label=args.real_label,
        device=device,
        amp=amp,
        metric=args.metric,
        plot_path=args.plot_path,
        save_json=args.save_json,
    )


if __name__ == "__main__":
    main()

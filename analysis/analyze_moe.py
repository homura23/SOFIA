from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
except Exception:
    pd = None
    sns = None
    plt = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sofia.data.audio_dataset import AudioSampleDataset, MultiBranchCollate
from sofia.models.fusion import MultiBranchFusion
from sofia.models.heads import SoftmaxHead
from sofia.train import build_encoders, select_head
from sofia.utils.config import load_config, resolve_device


# =========================================================
# Expert display configuration
# IMPORTANT:
# These keys must match the actual names in fusion.branch_names.
# Based on your current plot, we assume:
#   fxpp, muq, mert, wave, rawnet
# and display them as:
#   Fxplusplus, Muq, Mert, Wave2Vec, Rawnet_vocal
# =========================================================
EXPERT_ORDER = ["fxpp", "muq", "mert", "wave", "rawnet"]

EXPERT_LABELS = {
    "fxpp": "Fxplusplus",
    "muq": "Muq",
    "mert": "Mert",
    "wave": "Wave2Vec",
    "rawnet": "Rawnet_vocal",
}

EXPERT_COLORS = {
    "fxpp": "#F8D7DA",   # light pink
    "muq": "#D9EFD3",    # light green
    "mert": "#F6EDB2",   # light yellow
    "wave": "#CFE8F7",   # light blue
    "rawnet": "#E5D0B1", # light brown
}


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MOE expert attribution and weight analysis")
    parser.add_argument("--config", type=str, default="config/moe.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--amp", action="store_true", help="Force AMP on")
    parser.add_argument("--no-amp", dest="no_amp", action="store_true", help="Force AMP off")

    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser("single", help="Analyze a single sample")
    single.add_argument("--csv", type=str, nargs="+", required=True)
    single.add_argument("--index", type=int, required=True, help="Sample index in CSV")
    single.add_argument("--save_json", type=str, default=None)

    dataset = subparsers.add_parser("dataset", help="Analyze dataset-level expert weights")
    dataset.add_argument("--csv", type=str, nargs="+", required=True, help="One or more CSVs")
    dataset.add_argument("--output_dir", type=str, required=True)
    dataset.add_argument("--max_samples", type=int, default=None, help="Max samples per CSV")
    dataset.add_argument("--device", type=str, default=None)
    dataset.add_argument("--batch_size", type=int, default=None)
    dataset.add_argument("--num_workers", type=int, default=None)
    dataset.add_argument("--amp", action="store_true", help="Force AMP on")
    dataset.add_argument("--no-amp", dest="no_amp", action="store_true", help="Force AMP off")

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


class SourceAwareCollate:
    def __init__(self, base_collate: MultiBranchCollate) -> None:
        self.base_collate = base_collate

    def __call__(self, batch: List[Dict]) -> Dict:
        out = self.base_collate(batch)
        out["source"] = [item.get("source", "") for item in batch]
        return out


def load_models(cfg: Dict, device: torch.device, checkpoint: str) -> Tuple[Dict[str, torch.nn.Module], MultiBranchFusion, torch.nn.Module]:
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


def apply_out_proj(out_proj: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    if isinstance(out_proj, torch.nn.Identity):
        return x
    if x.dim() == 2:
        return out_proj(x)
    b, n, d = x.shape
    flat = x.reshape(b * n, d)
    proj = out_proj(flat)
    return proj.reshape(b, n, -1)


def compute_moe_components(fusion: MultiBranchFusion, embeddings: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if fusion.mode not in {"moe", "mixture_of_experts", "complex_moe", "complex_mixture_of_experts"}:
        raise ValueError(f"Fusion mode {fusion.mode} does not use MOE routing")

    projected = []
    for name in fusion.branch_names:
        if name not in embeddings:
            raise KeyError(f"Missing embedding for branch '{name}'")
        x = embeddings[name]
        x = F.normalize(x, dim=1, eps=1e-6)
        x = fusion.projectors[name](x)
        x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        projected.append(x)

    h = torch.cat(projected, dim=1)
    stacked = torch.stack(projected, dim=1)
    gate_logits = fusion.model_gate(h)
    alpha, topk_idx = fusion._compute_moe_weights(gate_logits)
    expert_out = torch.stack(
        [fusion.experts[name](x) for name, x in zip(fusion.branch_names, projected)],
        dim=1,
    )
    z_moe = (alpha.unsqueeze(-1) * expert_out).sum(dim=1)
    residual = stacked.mean(dim=1)
    fused_pre = apply_out_proj(fusion.out_proj, z_moe + residual)
    fused = torch.nan_to_num(fused_pre, nan=0.0, posinf=1e4, neginf=-1e4)
    if fusion.l2_norm:
        fused = F.normalize(fused, dim=1, eps=1e-6)

    return {
        "projected": stacked,
        "gate_logits": gate_logits,
        "alpha": alpha,
        "topk_indices": topk_idx,
        "expert_out": expert_out,
        "z_moe": z_moe,
        "residual": residual,
        "fused_pre": fused_pre,
        "fused": fused,
    }


def compute_expert_logits_and_contrib(
    fusion: MultiBranchFusion,
    head: torch.nn.Module,
    components: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    if not isinstance(head, SoftmaxHead):
        raise ValueError("This analysis assumes SoftmaxHead for linear logit decomposition")

    alpha = components["alpha"]
    expert_out = components["expert_out"]
    residual = components["residual"]

    weight = head.fc.weight
    bias = head.fc.bias

    # Unweighted per-expert logits (expert output only)
    expert_logits_raw = []
    for idx in range(expert_out.size(1)):
        expert_proj = apply_out_proj(fusion.out_proj, expert_out[:, idx, :])
        if fusion.l2_norm:
            expert_proj = F.normalize(expert_proj, dim=1, eps=1e-6)
        expert_logits_raw.append(F.linear(expert_proj, weight, bias))
    expert_logits_raw = torch.stack(expert_logits_raw, dim=1)

    # Contribution to final logits using the same normalization scalar as fused
    expert_components = apply_out_proj(fusion.out_proj, expert_out * alpha.unsqueeze(-1))
    residual_component = apply_out_proj(fusion.out_proj, residual)
    total = expert_components.sum(dim=1) + residual_component
    if fusion.l2_norm:
        denom = total.norm(dim=1, keepdim=True).clamp_min(1e-6)
    else:
        denom = torch.ones(total.size(0), 1, device=total.device)

    expert_components = expert_components / denom.unsqueeze(1)
    residual_component = residual_component / denom

    expert_contrib = torch.einsum("bnd,cd->bnc", expert_components, weight)
    residual_contrib = residual_component @ weight.t()

    return {
        "expert_logits_raw": expert_logits_raw,
        "expert_contrib": expert_contrib,
        "residual_contrib": residual_contrib,
        "bias": bias,
    }


def analyze_single_sample(
    cfg: Dict,
    encoders: Dict[str, torch.nn.Module],
    fusion: MultiBranchFusion,
    head: torch.nn.Module,
    csv_paths: List[str],
    index: int,
    device: torch.device,
    amp: bool,
    save_json: str | None,
) -> None:
    data_cfg = cfg["data"]
    branches_cfg = cfg.get("audio_branches", {})
    enabled_branches = {k: v for k, v in branches_cfg.items() if v.get("enabled", False)}
    collate = MultiBranchCollate(
        branch_cfg=enabled_branches,
        segment_seconds=float(data_cfg["segment_seconds"]),
        random_crop=False,
    )
    collate = SourceAwareCollate(collate)

    datasets = build_datasets(csv_paths, data_cfg)
    total_len = sum(len(ds) for ds in datasets)
    if index < 0 or index >= total_len:
        raise IndexError(f"Index {index} out of range for dataset length {total_len}")

    item, item_csv = get_item_by_index(datasets, csv_paths, index)
    batch = collate([item])
    labels = batch["label"].to(device)
    audio = {k: v.to(device) for k, v in batch["audio"].items()}

    with torch.no_grad(), torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=amp):
        embeddings = {name: enc(audio[name], sample_rate=enc.target_sr) for name, enc in encoders.items()}
        components = compute_moe_components(fusion, embeddings)
        logits = head(components["fused"], labels)

    pred = int(logits.argmax(dim=1).item())
    label = int(labels.item())
    correct = pred == label
    if not correct:
        print(f"[single] sample {index} predicted {pred} != label {label}; skip analysis")
        return

    detail = compute_expert_logits_and_contrib(fusion, head, components)

    out = {
        "sample_index": index,
        "label": label,
        "pred": pred,
        "csv_path": item_csv,
        "gating_weights": components["alpha"].squeeze(0).tolist(),
        "topk_indices": components["topk_indices"].squeeze(0).tolist() if components["topk_indices"] is not None else None,
        "logits": logits.squeeze(0).tolist(),
        "expert_logits_raw": detail["expert_logits_raw"].squeeze(0).tolist(),
        "expert_logit_contrib": detail["expert_contrib"].squeeze(0).tolist(),
        "residual_logit_contrib": detail["residual_contrib"].squeeze(0).tolist(),
        "bias": detail["bias"].tolist(),
        "branch_names": fusion.branch_names,
    }

    if save_json:
        save_path = Path(save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"[single] saved analysis to {save_path}")
    else:
        print(json.dumps(out, indent=2))


def plot_violin_box(df, output_path: Path, title: str) -> None:
    if pd is None or sns is None or plt is None:
        raise RuntimeError("pandas, seaborn, and matplotlib are required for plotting")

    # Keep experts present in the data and preserve fixed order
    present_experts = [e for e in EXPERT_ORDER if e in df["expert"].unique()]
    if not present_experts:
        raise ValueError(
            f"No known experts found in dataframe. "
            f"Expected one of {EXPERT_ORDER}, got {sorted(df['expert'].unique().tolist())}"
        )

    plot_df = df[df["expert"].isin(present_experts)].copy()
    plot_df["expert"] = pd.Categorical(
        plot_df["expert"],
        categories=present_experts,
        ordered=True
    )

    palette = {e: EXPERT_COLORS[e] for e in present_experts}

    # Minimal style closer to reference
    sns.set_theme(style="ticks", context="notebook")

    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)

    sns.violinplot(
        data=plot_df,
        x="expert",
        y="weight",
        order=present_experts,
        palette=palette,
        inner="box",
        cut=0,
        linewidth=1.2,
        saturation=1,
        scale="width",
        width=0.55,
        ax=ax,
    )

    # Add black borders to violin plots
    for coll in ax.collections:
        try:
            coll.set_edgecolor("black")
            coll.set_linewidth(1.0)
            coll.set_alpha(0.95)
        except Exception:
            pass

    # Compute and plot mean points (black diamonds)
    means = (
        plot_df.groupby("expert", observed=True)["weight"]
        .mean()
        .reindex(present_experts)
    )
    ax.scatter(
        range(len(present_experts)),
        means.values,
        marker="D",
        color="black",
        s=28,
        zorder=5
    )

    # Use display labels
    ax.set_xticklabels(
        [EXPERT_LABELS[e] for e in present_experts],
        rotation=25,
        ha="right"
    )

    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel("Expert", fontsize=12)
    ax.set_ylabel("Weight Distribution", fontsize=12)

    # Adaptive y-axis to match reference view
    y_min = max(0.0, float(plot_df["weight"].min()) - 0.02)
    y_max = min(1.0, float(plot_df["weight"].max()) + 0.02)
    if y_max - y_min < 0.05:
        center = (y_max + y_min) / 2.0
        y_min = max(0.0, center - 0.03)
        y_max = min(1.0, center + 0.03)
    ax.set_ylim(y_min, y_max)

    ax.grid(axis="y", linestyle="--", alpha=0.25)
    sns.despine(ax=ax)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def analyze_dataset_weights(
    cfg: Dict,
    encoders: Dict[str, torch.nn.Module],
    fusion: MultiBranchFusion,
    head: torch.nn.Module,
    csv_paths: Iterable[str],
    output_dir: str,
    device: torch.device,
    amp: bool,
    batch_size: int,
    num_workers: int,
    max_samples: int | None,
) -> None:
    data_cfg = cfg["data"]
    branches_cfg = cfg.get("audio_branches", {})
    enabled_branches = {k: v for k, v in branches_cfg.items() if v.get("enabled", False)}
    collate = MultiBranchCollate(
        branch_cfg=enabled_branches,
        segment_seconds=float(data_cfg["segment_seconds"]),
        random_crop=False,
    )

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    for csv_path in csv_paths:
        loader = build_loader(csv_path, data_cfg, collate, batch_size, num_workers)
        rows: List[Dict[str, object]] = []
        total = 0
        used = 0

        with torch.no_grad():
            for batch in loader:
                if max_samples is not None and total >= max_samples:
                    break

                labels = batch["label"].to(device)
                audio = {k: v.to(device) for k, v in batch["audio"].items()}
                sources = batch.get("source", [""] * labels.size(0))
                if len(sources) != labels.size(0):
                    sources = list(sources)[: labels.size(0)]
                    if len(sources) < labels.size(0):
                        sources += [""] * (labels.size(0) - len(sources))

                if max_samples is not None:
                    remaining = max_samples - total
                    if remaining <= 0:
                        break
                    if labels.size(0) > remaining:
                        labels = labels[:remaining]
                        audio = {k: v[:remaining] for k, v in audio.items()}
                        sources = sources[:remaining]

                total += labels.size(0)

                with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=amp):
                    embeddings = {name: enc(audio[name], sample_rate=enc.target_sr) for name, enc in encoders.items()}
                    components = compute_moe_components(fusion, embeddings)
                    logits = head(components["fused"], labels)

                pred = logits.argmax(dim=1)
                correct_mask = pred.eq(labels)
                if correct_mask.any():
                    alpha = components["alpha"][correct_mask].detach().cpu()
                    kept_indices = correct_mask.nonzero(as_tuple=False).squeeze(1).tolist()
                    kept_sources = [sources[idx] if idx < len(sources) else "" for idx in kept_indices]
                    used += int(correct_mask.sum().item())
                    for sample_idx in range(alpha.size(0)):
                        for expert_idx, weight in enumerate(alpha[sample_idx].tolist()):
                            rows.append(
                                {
                                    "dataset": Path(csv_path).stem,
                                    "source": kept_sources[sample_idx] if sample_idx < len(kept_sources) else "",
                                    "expert": fusion.branch_names[expert_idx],
                                    "weight": float(weight),
                                }
                            )

        if not rows:
            print(f"[dataset] {csv_path}: no correct samples; skip plotting")
            continue

        if pd is None:
            raise RuntimeError("pandas is required for dataset analysis")

        df = pd.DataFrame(rows)
        dataset_name = Path(csv_path).stem

        if "source" in df and df["source"].nunique() > 1:
            for source in sorted(df["source"].unique()):
                sub = df[df["source"] == source]
                safe_source = "unknown" if source == "" else source.replace("/", "_")
                plot_path = output_dir_path / f"{dataset_name}_moe_weights_{safe_source}.png"
                plot_violin_box(
                    sub,
                    plot_path,
                    title=f"{dataset_name} ({source}): expert weight distribution"
                )
        else:
            plot_path = output_dir_path / f"{dataset_name}_moe_weights.png"
            plot_violin_box(
                df,
                plot_path,
                title=f"{dataset_name}: expert weight distribution"
            )

        print(f"[dataset] {csv_path}: used {used}/{total} correct samples")
        if "source" in df and df["source"].nunique() > 1:
            print(f"[dataset] saved plots by source under {output_dir_path}")
        else:
            print(f"[dataset] saved plot to {plot_path}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device(resolve_device(cfg.get("device"), args.device))

    encoders, fusion, head = load_models(cfg, device, args.checkpoint)

    train_amp = bool(cfg["training"].get("amp", True))
    amp = train_amp
    if args.amp:
        amp = True
    if args.no_amp:
        amp = False

    if args.command == "single":
        analyze_single_sample(
            cfg=cfg,
            encoders=encoders,
            fusion=fusion,
            head=head,
            csv_paths=args.csv,
            index=args.index,
            device=device,
            amp=amp,
            save_json=args.save_json,
        )
    elif args.command == "dataset":
        batch_size = args.batch_size or int(cfg["data"].get("batch_size", 8))
        num_workers = args.num_workers or int(cfg["data"].get("num_workers", 2))
        analyze_dataset_weights(
            cfg=cfg,
            encoders=encoders,
            fusion=fusion,
            head=head,
            csv_paths=args.csv,
            output_dir=args.output_dir,
            device=device,
            amp=amp,
            batch_size=batch_size,
            num_workers=num_workers,
            max_samples=args.max_samples,
        )


if __name__ == "__main__":
    main()
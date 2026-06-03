from __future__ import annotations

import argparse
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
from sofia.train import build_encoders
from sofia.utils.config import load_config, resolve_device


FAKE_SOURCE_ORDER = [
    "ace",
    "acestep1.5",
    "heartmula",
    "minimax_2.6",
    "mureka",
    "mureka_v9",
    "suno_v5.5",
    "suno_v4",
    "sunov5_new",
    "chirp-v2",
    "chirp-v3",
    "chirp-v3.5",
    "udio-120s",
    "udio-30s",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset-level feature distribution analysis")
    parser.add_argument("--config", type=str, default="config/moe.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--csv", type=str, nargs="+", action="append", required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--amp", action="store_true", help="Force AMP on")
    parser.add_argument("--no-amp", dest="no_amp", action="store_true", help="Force AMP off")
    parser.add_argument("--method", type=str, choices=["pca", "tsne", "umap"], default="pca")
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    parser.add_argument("--umap_n_neighbors", type=int, default=30)
    parser.add_argument("--umap_metric", type=str, default="cosine")
    parser.add_argument("--umap_spread", type=float, default=1.0)
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--stage", type=str, choices=["fused", "encoder", "projected", "expert"], default="fused")
    parser.add_argument("--branch", type=str, default=None, help="Required for encoder/projected/expert stages")
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


def extract_features(
    fusion: MultiBranchFusion,
    embeddings: Dict[str, torch.Tensor],
    stage: str,
    branch: str | None,
) -> torch.Tensor:
    if stage == "fused":
        # Final fused embedding used by the classifier
        return fusion(embeddings)

    if stage == "encoder" and branch is None:
        # Concatenate raw encoder outputs from all branches
        parts = []
        for name in fusion.branch_names:
            parts.append(embeddings[name])
        return torch.cat(parts, dim=1)

    if stage == "projected" and branch is None:
        # Concatenate projected outputs from all branches
        parts = []
        for name in fusion.branch_names:
            x = embeddings[name]
            x = F.normalize(x, dim=1, eps=1e-6)
            x = fusion.projectors[name](x)
            x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
            parts.append(x)
        return torch.cat(parts, dim=1)

    if branch is None:
        raise ValueError("branch is required for expert stage")
    if branch not in fusion.branch_names:
        raise ValueError(f"Unknown branch: {branch}")

    # Encoder output for the selected branch
    x = embeddings[branch]
    if stage == "encoder":
        return x

    # Project to shared dimension before expert or fusion
    x = F.normalize(x, dim=1, eps=1e-6)
    x = fusion.projectors[branch](x)
    x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)

    if stage == "projected":
        return x

    if stage == "expert":
        # Expert output after branch-specific transformation
        x = fusion.experts[branch](x)
        x = apply_out_proj(fusion.out_proj, x)
        if fusion.l2_norm:
            x = F.normalize(x, dim=1, eps=1e-6)
        return x

    raise ValueError(f"Unsupported stage: {stage}")


def reduce_features(
    features,
    method: str,
    umap_min_dist: float,
    umap_n_neighbors: int,
    umap_metric: str,
    umap_spread: float,
):
    if method == "pca":
        try:
            from sklearn.decomposition import PCA
        except Exception as exc:
            raise RuntimeError("scikit-learn is required for PCA") from exc
        return PCA(n_components=2, random_state=42).fit_transform(features)

    if method == "tsne":
        try:
            from sklearn.manifold import TSNE
        except Exception as exc:
            raise RuntimeError("scikit-learn is required for t-SNE") from exc
        return TSNE(n_components=2, init="pca", random_state=42, perplexity=30).fit_transform(features)

    if method == "umap":
        try:
            import umap
        except Exception as exc:
            raise RuntimeError("umap-learn is required for UMAP") from exc
        return umap.UMAP(
            n_components=2,
            random_state=42,
            min_dist=umap_min_dist,
            n_neighbors=umap_n_neighbors,
            metric=umap_metric,
            spread=umap_spread,
        ).fit_transform(features)

    raise ValueError(f"Unsupported method: {method}")


def analyze_distribution(
    cfg: Dict,
    encoders: Dict[str, torch.nn.Module],
    fusion: MultiBranchFusion,
    csv_paths: Iterable[str],
    output_dir: str,
    device: torch.device,
    amp: bool,
    batch_size: int,
    num_workers: int,
    max_samples: int,
    method: str,
    stage: str,
    branch: str | None,
    umap_min_dist: float,
    umap_n_neighbors: int,
    umap_metric: str,
    umap_spread: float,
) -> None:
    if pd is None or sns is None or plt is None:
        raise RuntimeError("pandas, seaborn, and matplotlib are required for plotting")

    data_cfg = cfg["data"]
    branches_cfg = cfg.get("audio_branches", {})
    enabled_branches = {k: v for k, v in branches_cfg.items() if v.get("enabled", False)}
    collate = MultiBranchCollate(
        branch_cfg=enabled_branches,
        segment_seconds=float(data_cfg["segment_seconds"]),
        random_crop=False,
    )
    collate = SourceAwareCollate(collate)

    vectors: List[List[float]] = []
    labels: List[int] = []
    datasets: List[str] = []
    sources: List[str] = []

    print(f"[dist] csv_paths: {list(csv_paths)}")
    # Collect features from each dataset with a cap to avoid overplotting
    for csv_path in csv_paths:
        loader = build_loader(csv_path, data_cfg, collate, batch_size, num_workers)
        dataset_name = Path(csv_path).stem
        collected = 0
        dataset = loader.dataset
        total_items = len(dataset)
        source_counts: Dict[str, int] = {}
        label_counts: Dict[int, int] = {}
        for item in getattr(dataset, "items", []):
            src = str(item.get("source", ""))
            label = int(item.get("label", 0))
            source_counts[src] = source_counts.get(src, 0) + 1
            label_counts[label] = label_counts.get(label, 0) + 1
        print(f"[dist] {dataset_name}: total items {total_items}")
        if source_counts:
            print(f"[dist] {dataset_name}: sources {source_counts}")
        if label_counts:
            print(f"[dist] {dataset_name}: labels {label_counts}")

        with torch.no_grad():
            for batch in loader:
                audio = {k: v.to(device) for k, v in batch["audio"].items()}
                batch_labels = batch["label"].tolist()
                batch_sources = batch.get("source", [""] * len(batch_labels))

                with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=amp):
                    embeddings = {name: enc(audio[name], sample_rate=enc.target_sr) for name, enc in encoders.items()}
                    feats = extract_features(fusion, embeddings, stage=stage, branch=branch)

                feats = feats.detach().cpu().tolist()
                for vec, lab, src in zip(feats, batch_labels, batch_sources):
                    vectors.append(vec)
                    labels.append(int(lab))
                    datasets.append(dataset_name)
                    sources.append(str(src))
                    collected += 1
                    if collected >= max_samples:
                        break
                if collected >= max_samples:
                    break
            print(f"[dist] {dataset_name}: collected {collected} samples")

    # Reduce to 2D for visualization
    if not vectors:
        raise ValueError("No features collected. Check CSV paths and filters.")
    reduced = reduce_features(
        vectors,
        method,
        umap_min_dist=umap_min_dist,
        umap_n_neighbors=umap_n_neighbors,
        umap_metric=umap_metric,
        umap_spread=umap_spread,
    )
    df = pd.DataFrame({
        "x": reduced[:, 0],
        "y": reduced[:, 1],
        "label": labels,
        "dataset": datasets,
        "source": sources,
    })
    df["dataset_source"] = df["dataset"].astype(str) + ":" + df["source"].astype(str)
    count_table = df.groupby(["dataset_source", "label"]).size().reset_index(name="count")
    print("[dist] sample counts by dataset_source and label:")
    for _, row in count_table.iterrows():
        print(f"  {row['dataset_source']} | label {row['label']}: {row['count']}")
    if df["dataset_source"].nunique() == 1:
        print("[dist] warning: only one dataset_source found; check --csv inputs")

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    tab20 = plt.get_cmap("tab20")
    sources_present = sorted(set(df["source"].astype(str)))
    ordered_sources = [name for name in FAKE_SOURCE_ORDER if name in sources_present]
    extras = sorted(name for name in sources_present if name not in FAKE_SOURCE_ORDER)
    ordered_sources.extend(extras)
    source_color = {name: tab20(idx % tab20.N) for idx, name in enumerate(ordered_sources)}
    x_vals = df["x"].to_numpy()
    y_vals = df["y"].to_numpy()
    labels_arr = df["label"].to_numpy()
    sources_arr = df["source"].astype(str).to_numpy()

    real_mask = labels_arr == 0
    if real_mask.any():
        plt.scatter(
            x_vals[real_mask],
            y_vals[real_mask],
            c="#000000",
            s=8,
            alpha=0.6,
            marker="o",
        )

    ai_mask = ~real_mask
    if ai_mask.any():
        ai_colors = [source_color.get(src, tab20(0)) for src in sources_arr[ai_mask]]
        plt.scatter(
            x_vals[ai_mask],
            y_vals[ai_mask],
            c=ai_colors,
            s=12,
            alpha=0.75,
            marker="o",
        )

    stage_suffix = stage if stage == "fused" else f"{stage}_{branch}"
    plt.title(f"Feature distribution ({stage_suffix}, {method})")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    # Legend intentionally omitted for cleaner layout
    plot_path = output_dir_path / f"feature_dist_{stage_suffix}_{method}.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"[dist] saved plot to {plot_path}")


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

    batch_size = args.batch_size or int(cfg["data"].get("batch_size", 8))
    num_workers = args.num_workers or int(cfg["data"].get("num_workers", 2))

    if args.stage in {"expert"} and not args.branch:
        raise ValueError("--branch is required for expert stage")

    csv_paths = [path for group in args.csv for path in group]

    analyze_distribution(
        cfg=cfg,
        encoders=encoders,
        fusion=fusion,
        csv_paths=csv_paths,
        output_dir=args.output_dir,
        device=device,
        amp=amp,
        batch_size=batch_size,
        num_workers=num_workers,
        max_samples=args.max_samples,
        method=args.method,
        stage=args.stage,
        branch=args.branch,
        umap_min_dist=args.umap_min_dist,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_metric=args.umap_metric,
        umap_spread=args.umap_spread,
    )


if __name__ == "__main__":
    main()

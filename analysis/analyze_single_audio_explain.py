from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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
from sofia.models.heads import SoftmaxHead
from sofia.train import build_encoders, select_head
from sofia.utils.config import load_config, resolve_device


# Keep expert display consistent with analyze_moe.py
EXPERT_ORDER = ["fxpp", "muq", "mert", "wave", "rawnet"]
EXPERT_LABELS = {
    "fxpp": "Fxplusplus",
    "muq": "Muq",
    "mert": "Mert",
    "wave": "Wave2Vec",
    "rawnet": "Rawnet_vocal",
}
EXPERT_COLORS = {
    "fxpp": "#F8D7DA",
    "muq": "#D9EFD3",
    "mert": "#F6EDB2",
    "wave": "#CFE8F7",
    "rawnet": "#E5D0B1",
}


def resolve_branch_labels(branch_names: List[str]) -> List[str]:
    return [EXPERT_LABELS.get(name, name) for name in branch_names]


def resolve_branch_colors(branch_names: List[str]) -> List[str]:
    return [EXPERT_COLORS.get(name, "#CCCCCC") for name in branch_names]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single audio expert analysis with logits")
    parser.add_argument("--config", type=str, default="config/moe.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--amp", action="store_true", help="Force AMP on")
    parser.add_argument("--no-amp", dest="no_amp", action="store_true", help="Force AMP off")
    parser.add_argument("--save_path", type=str, required=True)
    return parser.parse_args()


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


def build_single_batch(cfg: Dict, csv_path: str, index: int):
    data_cfg = cfg["data"]
    branches_cfg = cfg.get("audio_branches", {})
    enabled_branches = {k: v for k, v in branches_cfg.items() if v.get("enabled", False)}
    collate = MultiBranchCollate(
        branch_cfg=enabled_branches,
        segment_seconds=float(data_cfg["segment_seconds"]),
        random_crop=False,
    )
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
    if index < 0 or index >= len(dataset):
        raise IndexError(f"Index {index} out of range for dataset length {len(dataset)}")
    item = dataset[index]
    return collate([item]), item


def apply_out_proj(out_proj: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    if isinstance(out_proj, torch.nn.Identity):
        return x
    return out_proj(x)


def compute_projected_and_expert(
    fusion: MultiBranchFusion,
    embeddings: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    projected = []
    for name in fusion.branch_names:
        x = embeddings[name]
        x = F.normalize(x, dim=1, eps=1e-6)
        x = fusion.projectors[name](x)
        x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        projected.append(x)

    h = torch.cat(projected, dim=1)
    gate_logits = fusion.model_gate(h)
    alpha, _ = fusion._compute_moe_weights(gate_logits)

    expert_out = torch.stack(
        [fusion.experts[name](x) for name, x in zip(fusion.branch_names, projected)],
        dim=1,
    )
    return torch.stack(projected, dim=1), expert_out, alpha


def logits_from_embeddings(
    head: SoftmaxHead,
    fusion: MultiBranchFusion,
    embeds: torch.Tensor,
) -> torch.Tensor:
    out = apply_out_proj(fusion.out_proj, embeds)
    if fusion.l2_norm:
        out = F.normalize(out, dim=1, eps=1e-6)
    return F.linear(out, head.fc.weight, head.fc.bias)


def plot_results(
    save_path: str,
    branch_names: List[str],
    projected_logits: torch.Tensor,
    expert_weights: torch.Tensor,
    final_logits: torch.Tensor,
    label: int,
    pred: int,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting")
    save_path = Path(save_path)
    stem = save_path.stem
    out_dir = save_path.parent

    # (a) Projected logits per branch (binary)
    fig_a, ax_a = plt.subplots(1, 1, figsize=(5, 4))
    x = torch.arange(len(branch_names)).numpy()
    width = 0.35
    tick_labels = resolve_branch_labels(branch_names)
    bar_colors = resolve_branch_colors(branch_names)
    proj_np = projected_logits.detach().cpu().numpy()
    if projected_logits.ndim == 1 or projected_logits.size(1) == 1:
        ax_a.bar(x, proj_np.reshape(-1), width=0.5, color=bar_colors, label="logit")
    else:
        ax_a.bar(x - width / 2, proj_np[:, 0], width=width, color=bar_colors, alpha=0.7, label="class0")
        ax_a.bar(x + width / 2, proj_np[:, 1], width=width, color=bar_colors, alpha=1.0, label="class1")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(tick_labels, rotation=20, ha="right", fontsize=9)
    ax_a.set_ylabel("Logit Values")
    ax_a.set_title("(a) Projected logits")
    fig_a.suptitle(f"label={label} pred={pred}", fontsize=10)
    fig_a.tight_layout()
    fig_a.savefig(out_dir / f"{stem}_a.png", dpi=200)
    plt.close(fig_a)

    # (b) Expert weights
    fig_b, ax_b = plt.subplots(1, 1, figsize=(5, 4))
    ax_b.bar(tick_labels, expert_weights.numpy(), color=bar_colors)
    ax_b.tick_params(axis="x", labelsize=9)
    ax_b.set_ylabel("Expert Weighting")
    ax_b.set_title("(b) Expert weights")
    fig_b.suptitle(f"label={label} pred={pred}", fontsize=10)
    fig_b.tight_layout()
    fig_b.savefig(out_dir / f"{stem}_b.png", dpi=200)
    plt.close(fig_b)

    # (c) Final logits
    fig_c, ax_c = plt.subplots(1, 1, figsize=(4, 4))
    final_np = final_logits.detach().cpu().numpy()
    if final_logits.ndim == 0 or final_logits.numel() == 1:
        ax_c.bar(["logit"], final_np.reshape(-1))
    else:
        ax_c.bar(["class0", "class1"], final_np)
    ax_c.set_ylabel("Logit")
    ax_c.set_title("(c) Final logits")
    fig_c.suptitle(f"label={label} pred={pred}", fontsize=10)
    fig_c.tight_layout()
    fig_c.savefig(out_dir / f"{stem}_c.png", dpi=200)
    plt.close(fig_c)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(resolve_device(cfg.get("device"), args.device))

    encoders, fusion, head = load_models(cfg, device, args.checkpoint)
    if not isinstance(head, SoftmaxHead):
        raise ValueError("This analysis requires SoftmaxHead for logit visualization")

    train_amp = bool(cfg["training"].get("amp", True))
    amp = train_amp
    if args.amp:
        amp = True
    if args.no_amp:
        amp = False

    batch, item = build_single_batch(cfg, args.csv, args.index)
    label = int(item["label"])
    audio = {k: v.to(device) for k, v in batch["audio"].items()}

    with torch.no_grad(), torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=amp):
        embeddings = {name: enc(audio[name], sample_rate=enc.target_sr) for name, enc in encoders.items()}
        projected, expert_out, alpha = compute_projected_and_expert(fusion, embeddings)
        fused = fusion(embeddings)
        logits = head(fused, torch.tensor([label], device=device))

    pred = int(logits.argmax(dim=1).item())

    # Projected logits per branch
    projected_logits = []
    for idx in range(projected.size(1)):
        logits_branch = logits_from_embeddings(head, fusion, projected[:, idx, :])
        projected_logits.append(logits_branch.squeeze(0))
    projected_logits = torch.stack(projected_logits, dim=0).cpu()

    # Expert weights (alpha)
    expert_weights = alpha.squeeze(0).detach().cpu()

    # Final logits
    final_logits = logits.squeeze(0).detach().cpu()

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plot_results(
        save_path=str(save_path),
        branch_names=fusion.branch_names,
        projected_logits=projected_logits,
        expert_weights=expert_weights,
        final_logits=final_logits,
        label=label,
        pred=pred,
    )
    print(f"[single] saved plot to {save_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sofia.data.audio_dataset import MultiBranchCollate
from sofia.models.fusion import MultiBranchFusion
from sofia.train import build_encoders, select_head
from sofia.utils.audio import crop_or_pad, ensure_channels, load_audio, separate_vocals
from sofia.utils.config import load_config, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a single audio file without a CSV.")
    parser.add_argument("--config", type=str, default="config/rawnet_only.yaml", help="Path to YAML config.")
    parser.add_argument("--audio", type=str, required=True, help="Path to the full audio file.")
    parser.add_argument("--vocal_audio", type=str, default=None, help="Optional precomputed vocal stem.")
    parser.add_argument("--checkpoint", type=str, required=True, help="SOFIA fusion/head checkpoint.")
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cpu, cuda, cuda:0.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for label 1.")
    parser.add_argument("--json", action="store_true", help="Print JSON with prediction details.")
    parser.add_argument("--quiet", action="store_true", help="Print only true/false.")
    return parser.parse_args()


def build_single_item(
    audio_path: str,
    vocal_audio_path: str | None,
    data_cfg: Dict,
    branch_cfg: Dict[str, Dict],
    device: torch.device,
) -> Dict:
    base_sr = int(data_cfg["base_sample_rate"])
    mono = bool(data_cfg.get("mono", True))
    normalize = bool(data_cfg.get("normalize", True))
    segment_seconds = float(data_cfg.get("segment_seconds", 0))
    target_len = int(segment_seconds * base_sr) if segment_seconds > 0 else None

    full_wave, _ = load_audio(audio_path, target_sr=base_sr, mono=mono, normalize=normalize)
    if target_len:
        full_wave = crop_or_pad(full_wave, target_len, random_crop=False)

    needs_vocals = any(cfg.get("enabled", True) and cfg.get("use_vocals", False) for cfg in branch_cfg.values())
    vocal_wave = None
    if needs_vocals and vocal_audio_path:
        vocal_wave, _ = load_audio(vocal_audio_path, target_sr=base_sr, mono=mono, normalize=normalize)
        if target_len:
            vocal_wave = crop_or_pad(vocal_wave, target_len, random_crop=False)
    elif needs_vocals and bool(data_cfg.get("demucs_enabled", True)):
        try:
            vocal_wave = separate_vocals(
                waveform=full_wave,
                sample_rate=base_sr,
                model_name=str(data_cfg.get("demucs_model", "htdemucs")),
                device=str(data_cfg.get("demucs_device", device)),
            )
            vocal_wave = ensure_channels(vocal_wave, 1 if mono else 2)
            if target_len:
                vocal_wave = crop_or_pad(vocal_wave, target_len, random_crop=False)
        except Exception:
            vocal_wave = None

    if needs_vocals and (vocal_wave is None or vocal_wave.numel() == 0):
        vocal_wave = torch.zeros_like(full_wave)

    return {
        "full": full_wave,
        "vocal": vocal_wave if vocal_wave is not None else torch.zeros_like(full_wave),
        "sample_rate": base_sr,
        "label": torch.tensor(0, dtype=torch.long),
        "path_full": audio_path,
        "path_vocal": vocal_audio_path or "",
        "source": "single_audio",
    }


def build_batch(item: Dict, data_cfg: Dict, branch_cfg: Dict[str, Dict]) -> Dict:
    collate = MultiBranchCollate(
        branch_cfg=branch_cfg,
        segment_seconds=float(data_cfg["segment_seconds"]),
        random_crop=False,
    )
    return collate([item])


def load_sofia_models(cfg: Dict, device: torch.device, checkpoint: str):
    enabled_branches = {k: v for k, v in cfg.get("audio_branches", {}).items() if v.get("enabled", False)}
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
    head = select_head(
        cfg,
        embedding_dim=int(fusion_cfg["fusion_out"]),
        num_classes=int(cfg["training"]["num_classes"]),
    ).to(device)

    state = torch.load(checkpoint, map_location="cpu")
    fusion.load_state_dict(state["fusion"])
    head.load_state_dict(state["head"])

    for module in [*encoders.values(), fusion, head]:
        module.eval()
    return encoders, fusion, head


@torch.no_grad()
def predict_with_sofia(cfg: Dict, batch: Dict, device: torch.device, checkpoint: str) -> torch.Tensor:
    encoders, fusion, head = load_sofia_models(cfg, device, checkpoint)
    audio = {k: v.to(device, non_blocking=True) for k, v in batch["audio"].items()}
    labels = torch.zeros(1, dtype=torch.long, device=device)
    embeddings = {name: encoder(audio[name], sample_rate=encoder.target_sr) for name, encoder in encoders.items()}
    fused = fusion(embeddings)
    return head(fused, labels)


def format_result(logits: torch.Tensor, threshold: float) -> Dict:
    probs = torch.softmax(logits, dim=1)
    label = int(torch.argmax(probs, dim=1).item())
    prob_label_1 = float(probs[0, 1].item()) if probs.shape[1] > 1 else float("nan")
    is_true = bool(prob_label_1 >= threshold) if probs.shape[1] > 1 else bool(label)
    return {
        "result": "true" if is_true else "false",
        "predicted_label": label,
        "prob_label_1": prob_label_1,
        "threshold": threshold,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(resolve_device(cfg.get("device"), args.device))
    branch_cfg = {k: v for k, v in cfg.get("audio_branches", {}).items() if v.get("enabled", False)}
    item = build_single_item(args.audio, args.vocal_audio, cfg["data"], branch_cfg, device)
    batch = build_batch(item, cfg["data"], branch_cfg)

    logits = predict_with_sofia(cfg, batch, device, args.checkpoint)

    result = format_result(logits.detach().cpu(), args.threshold)
    if args.quiet:
        print(result["result"])
    elif args.json:
        print(json.dumps(result, indent=2))
    else:
        print(result["result"])
        print(f"predicted_label={result['predicted_label']} prob_label_1={result['prob_label_1']:.6f}")


if __name__ == "__main__":
    main()

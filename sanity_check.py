
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PARENT = PROJECT_ROOT.parent
for p in (PROJECT_ROOT, PARENT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from sofia.utils.config import load_config, resolve_device
from sofia.data.audio_dataset import AudioSampleDataset, MultiBranchCollate
from sofia.encoders.fxpp import FxPPEncoder
from sofia.encoders.muq import MuQEncoder
from sofia.encoders.rawnet import RawNetEncoder
from sofia.models.fusion import MultiBranchFusion


def main():
    # Ensure CUDA works with DataLoader workers by using spawn start method
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)

    cfg_path = PROJECT_ROOT / "config" / "default.yaml"
    cfg = load_config(str(cfg_path))
    device = torch.device(resolve_device(cfg.get("device"), None))

    # 1) Dataset + collate
    data_cfg = cfg["data"]
    ds = AudioSampleDataset(
        csv_path=data_cfg["train_csv"],
        base_sample_rate=int(data_cfg["base_sample_rate"]),
        segment_seconds=float(data_cfg["segment_seconds"]),
        random_crop=False,
        mono=bool(data_cfg.get("mono", True)),
        normalize=bool(data_cfg.get("normalize", True)),
        demucs_enabled=bool(data_cfg.get("demucs_enabled", True)),
        demucs_model=str(data_cfg.get("demucs_model", "htdemucs")),
        demucs_device=str(data_cfg.get("demucs_device", device)),
    )

    branches_cfg = cfg.get("audio_branches", {})
    enabled_branches = {k: v for k, v in branches_cfg.items() if v.get("enabled", False)}

    collate = MultiBranchCollate(
        branch_cfg=enabled_branches,
        segment_seconds=float(data_cfg["segment_seconds"]),
        random_crop=False,
    )
    demucs_ready = bool(data_cfg.get("demucs_enabled", True))

    loader = DataLoader(
        ds,
        batch_size=int(data_cfg.get("batch_size", 2)),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
        collate_fn=collate,
        multiprocessing_context="spawn",
    )
    batch = next(iter(loader))
    print("[OK] Batch labels:", batch["label"].tolist())
    for name, audio in batch["audio"].items():
        print(f"[OK] Branch '{name}' audio shape: {tuple(audio.shape)}")

    # 2) Encoders
    encoders = {}
    if enabled_branches.get("fxpp", {}).get("enabled", False):
        encoders["fxpp"] = FxPPEncoder(
            weights=enabled_branches["fxpp"].get("weights"),
            repo_path=enabled_branches["fxpp"].get("repo_path"),
            device=device,
        )
    if enabled_branches.get("muq", {}).get("enabled", False):
        encoders["muq"] = MuQEncoder(
            weights=enabled_branches["muq"].get("weights"),
            repo_path=enabled_branches["muq"].get("repo_path"),
            device=device,
        )
    if enabled_branches.get("rawnet", {}).get("enabled", False):
        try:
            encoders["rawnet"] = RawNetEncoder(
                config_path=enabled_branches["rawnet"]["config"],
                weights=enabled_branches["rawnet"]["weights"],
                repo_path=enabled_branches["rawnet"].get("repo_path"),
                device=device,
            )
        except Exception as e:
            print("[WARN] RawNet failed to initialize:", e)

    # 3) Forward embeddings
    audio = {k: v.to(device) for k, v in batch["audio"].items()}
    embeddings = {}
    with torch.no_grad():
        for name, encoder in encoders.items():
            emb = encoder(audio[name], sample_rate=encoder.target_sr)
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)
            embeddings[name] = emb
            print(f"[OK] {name} embedding shape: {tuple(emb.shape)}")

    # 4) Fusion forward
    fusion_cfg = cfg["fusion"]
    branch_input_dims = {name: enabled_branches[name]["embed_dim"] for name in embeddings.keys()}
    fusion = MultiBranchFusion(
        branch_input_dims=branch_input_dims,
        projector_dim=int(fusion_cfg["projector_dim"]),
        fusion_hidden=fusion_cfg.get("fusion_hidden", []),
        fusion_out=int(fusion_cfg["fusion_out"]),
        dropout=float(fusion_cfg.get("dropout", 0.1)),
        l2_norm=bool(fusion_cfg.get("l2_norm", True)),
    ).to(device)

    # Align batch size across embeddings (trim to smallest B) to avoid shape mismatch in sanity check
    min_b = min(v.shape[0] for v in embeddings.values())
    embeddings = {k: v[:min_b] for k, v in embeddings.items()}
    fused = fusion({k: v.to(device) for k, v in embeddings.items()})
    print("[OK] Fused embedding shape:", tuple(fused.shape))

    print("[SUMMARY] demucs_ready:", demucs_ready)
    print("Sanity check completed.")


if __name__ == "__main__":
    main()

from __future__ import annotations

import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from sofia.data.audio_dataset import AudioSampleDataset, MultiBranchCollate
from sofia.encoders.fxpp import FxPPEncoder
from sofia.encoders.mert import MertEncoder
from sofia.encoders.muq import MuQEncoder
from sofia.encoders.rawnet import RawNetEncoder
from sofia.encoders.wave import Wave2Vec2Encoder
# from sofia.encoders.timbre import TimbreEncoder
from sofia.models.fusion import MultiBranchFusion
from sofia.models.heads import ArcFaceHead, SoftmaxHead
from sofia.utils.config import ensure_dir, load_config, parse_args, resolve_device
from sofia.utils.meter import AverageMeter, accuracy


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for p in module.parameters():
        p.requires_grad = requires_grad


def build_encoders(cfg: Dict, device: torch.device) -> Dict[str, nn.Module]:
    encoders: Dict[str, nn.Module] = {}
    if cfg.get("fxpp", {}).get("enabled", False):
        encoders["fxpp"] = FxPPEncoder(
            weights=cfg["fxpp"].get("weights"),
            repo_path=cfg["fxpp"].get("repo_path"),
            device=device,
        )
    if cfg.get("muq", {}).get("enabled", False):
        encoders["muq"] = MuQEncoder(
            weights=cfg["muq"].get("weights"),
            repo_path=cfg["muq"].get("repo_path"),
            device=device,
        )
    if cfg.get("mert", {}).get("enabled", False):
        encoders["mert"] = MertEncoder(
            weights=cfg["mert"].get("weights"),
            device=device,
            layer_pooling=str(cfg["mert"].get("layer_pooling", "layer_mean")),
        )
    if cfg.get("wave", {}).get("enabled", False):
        encoders["wave"] = Wave2Vec2Encoder(
            weights=cfg["wave"].get("weights"),
            device=device,
            layer_pooling=str(cfg["wave"].get("layer_pooling", "layer_mean")),
        )
    if cfg.get("rawnet", {}).get("enabled", False):
        encoders["rawnet"] = RawNetEncoder(
            config_path=cfg["rawnet"]["config"],
            weights=cfg["rawnet"]["weights"],
            repo_path=cfg["rawnet"].get("repo_path"),
            device=device,
        )
    # if cfg.get("timbre", {}).get("enabled", False):
    #     encoders["timbre"] = TimbreEncoder(
    #         weights=cfg["timbre"].get("weights"),
    #         device=device,
    #         cut_seconds=float(cfg["timbre"].get("cut_seconds", 60)),
    #     )
    if not encoders:
        raise ValueError("No encoders enabled in config")
    return encoders


def select_head(cfg: Dict, embedding_dim: int, num_classes: int) -> nn.Module:
    loss_cfg = cfg.get("loss", {})
    head_type = loss_cfg.get("type", "arcface").lower()
    if head_type == "arcface":
        return ArcFaceHead(
            in_features=embedding_dim,
            num_classes=num_classes,
            scale=float(loss_cfg.get("scale", 32.0)),
            margin=float(loss_cfg.get("margin", 0.3)),
        )
    if head_type == "softmax":
        return SoftmaxHead(in_features=embedding_dim, num_classes=num_classes)
    raise ValueError(f"Unsupported head type: {head_type}")


def make_optimizer(params, lr: float, weight_decay: float, betas) -> torch.optim.Optimizer:
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)


def save_checkpoint(path: str, fusion: nn.Module, head: nn.Module, optimizer: torch.optim.Optimizer, scaler: GradScaler, epoch: int) -> None:
    torch.save(
        {
            "fusion": fusion.state_dict(),
            "head": head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
        },
        path,
    )


def load_checkpoint(path: str, fusion: nn.Module, head: nn.Module, optimizer: torch.optim.Optimizer, scaler: GradScaler) -> int:
    state = torch.load(path, map_location="cpu")
    fusion.load_state_dict(state["fusion"])
    head.load_state_dict(state["head"])
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    return int(state.get("epoch", 0))


def train_epoch(
    encoders: Dict[str, nn.Module],
    fusion: MultiBranchFusion,
    head: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    amp: bool,
    log_interval: int,
    freeze_encoders: bool,
) -> Dict[str, float]:
    fusion.train()
    head.train()
    if freeze_encoders:
        for enc in encoders.values():
            enc.eval()

    loss_meter = AverageMeter("loss")
    acc_meter = AverageMeter("acc")

    for step, batch in enumerate(loader, 1):
        labels = batch["label"].to(device, non_blocking=True)
        audio = {k: v.to(device, non_blocking=True) for k, v in batch["audio"].items()}

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=amp):
            embeddings = {}
            with torch.set_grad_enabled(not freeze_encoders):
                for name, encoder in encoders.items():
                    embeddings[name] = encoder(audio[name], sample_rate=encoder.target_sr)
            fused = fusion(embeddings)
            logits = head(fused, labels)
            loss = F.cross_entropy(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(list(fusion.parameters()) + list(head.parameters()), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        # Per-iteration metrics
        step_acc = accuracy(logits, labels)
        loss_meter.update(loss.item(), labels.size(0))
        acc_meter.update(step_acc, labels.size(0))
        print(f"step {step:05d} | loss {loss.item():.4f} | acc {step_acc:.4f}")

    return {"loss": loss_meter.avg, "acc": acc_meter.avg}


def evaluate(
    encoders: Dict[str, nn.Module],
    fusion: MultiBranchFusion,
    head: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    freeze_encoders: bool,
    collect_outputs: bool = False,
) -> Dict[str, float]:
    fusion.eval()
    head.eval()
    for enc in encoders.values():
        enc.eval()

    loss_meter = AverageMeter("loss")
    acc_meter = AverageMeter("acc")
    attn_weights = []
    attn_branch_names = fusion.branch_names if collect_outputs else None

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
                loss = F.cross_entropy(logits, labels)

            loss_meter.update(loss.item(), labels.size(0))
            acc_meter.update(accuracy(logits, labels), labels.size(0))

    stats = {"loss": loss_meter.avg, "acc": acc_meter.avg}
    if collect_outputs and attn_weights:
        merged = torch.cat(attn_weights, dim=0)
        stats["attn_weights"] = merged
        stats["attn_branch_names"] = attn_branch_names
        stats["attn_mean"] = merged.mean(dim=0)
    return stats


def main() -> None:
    # Use spawn so workers can safely load models/code on demand
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)

    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device(resolve_device(cfg.get("device"), args.device))
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    data_cfg = cfg["data"]
    train_ds = AudioSampleDataset(
        csv_path=data_cfg["train_csv"],
        base_sample_rate=int(data_cfg["base_sample_rate"]),
        segment_seconds=float(data_cfg["segment_seconds"]),
        random_crop=bool(data_cfg.get("random_crop", True)),
        mono=bool(data_cfg.get("mono", True)),
        normalize=bool(data_cfg.get("normalize", True)),
        demucs_enabled=bool(data_cfg.get("demucs_enabled", True)),
        demucs_model=str(data_cfg.get("demucs_model", "htdemucs")),
        demucs_device=str(data_cfg.get("demucs_device", device)),
    )

    val_ds = None
    if data_cfg.get("val_csv"):
        val_ds = AudioSampleDataset(
            csv_path=data_cfg["val_csv"],
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
        random_crop=bool(data_cfg.get("random_crop", True)),
    )

    batch_size = args.batch_size or int(data_cfg.get("batch_size", 16))
    num_workers = args.num_workers or int(data_cfg.get("num_workers", 8))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate,
        multiprocessing_context="spawn",
        persistent_workers=False,
    )

    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=max(1, num_workers // 2),
            pin_memory=True,
            collate_fn=collate,
            multiprocessing_context="spawn",
            persistent_workers=True,
        )

    encoders = build_encoders(enabled_branches, device=device)

    branch_input_dims = {name: cfg_item["embed_dim"] for name, cfg_item in enabled_branches.items()}
    fusion_cfg = cfg["fusion"]
    fusion = MultiBranchFusion(
        branch_input_dims=branch_input_dims,
        projector_dim=int(fusion_cfg["projector_dim"]),
        fusion_hidden=fusion_cfg.get("fusion_hidden", []),
        fusion_out=int(fusion_cfg["fusion_out"]),
        dropout=float(fusion_cfg.get("dropout", 0.1)),
        l2_norm=bool(fusion_cfg.get("l2_norm", True)),
        mode=str(fusion_cfg.get("mode", "concat")),
        moe_routing=str(fusion_cfg.get("moe_routing", "dense")),
        top_k=int(fusion_cfg.get("top_k", 1)),
    ).to(device)

    # Training schedule: optional softmax warmup then ArcFace
    train_cfg = cfg["training"]
    num_classes = int(train_cfg["num_classes"])
    warmup_epochs = int(train_cfg.get("warmup_softmax_epochs", 0))
    base_lr = float(train_cfg["lr"])
    warmup_lr = float(train_cfg.get("warmup_lr", max(base_lr, 1e-3)))
    base_betas = tuple(train_cfg.get("betas", (0.9, 0.98)))
    base_wd = float(train_cfg.get("weight_decay", 0.0001))
    base_amp = bool(train_cfg.get("amp", True))
    warmup_amp = bool(train_cfg.get("warmup_amp", False))

    in_features = int(fusion_cfg["fusion_out"])
    if warmup_epochs > 0:
        head = SoftmaxHead(in_features=in_features, num_classes=num_classes).to(device)
        current_lr = warmup_lr
        current_amp = warmup_amp
        print(f"[Schedule] Using Softmax warmup for {warmup_epochs} epoch(s): lr={current_lr}, amp={current_amp}")
    else:
        head = select_head(cfg, embedding_dim=in_features, num_classes=num_classes).to(device)
        current_lr = base_lr
        current_amp = base_amp

    if cfg["training"].get("freeze_encoders", True):
        for enc in encoders.values():
            set_requires_grad(enc, False)

    params = list(fusion.parameters()) + list(head.parameters())
    optimizer = make_optimizer(params, lr=current_lr, weight_decay=base_wd, betas=base_betas)
    scaler = GradScaler('cuda' if device.type == 'cuda' else 'cpu', enabled=current_amp)

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, fusion, head, optimizer, scaler)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    ensure_dir(cfg["training"]["save_dir"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.resume:
        run_save_dir = os.path.dirname(args.resume)
    else:
        run_save_dir = os.path.join(cfg["training"]["save_dir"], f"{args.run_name}_{timestamp}")
    ensure_dir(run_save_dir)

    epochs = int(cfg["training"]["epochs"])
    log_interval = int(cfg["training"].get("log_interval", 20))
    freeze_encoders = bool(cfg["training"].get("freeze_encoders", True))
    amp = current_amp

    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        # Switch from warmup to target head after warmup_epochs
        if warmup_epochs > 0 and epoch == warmup_epochs:
            print("[Schedule] Switching to target head from warmup")
            # Rebuild head
            head = select_head(cfg, embedding_dim=in_features, num_classes=num_classes).to(device)

            # Rebuild optimizer while preserving fusion states to avoid re-starting its momentum/history
            old_state = optimizer.state_dict()
            fusion_param_ids = {id(p) for p in fusion.parameters()}
            params = list(fusion.parameters()) + list(head.parameters())
            optimizer = make_optimizer(params, lr=base_lr, weight_decay=base_wd, betas=base_betas)
            # restore optimizer.state for fusion params
            old_states = old_state.get("state", {})
            for group in optimizer.param_groups:
                for p in group["params"]:
                    pid = id(p)
                    if pid in fusion_param_ids and pid in old_states:
                        optimizer.state[p] = old_states[pid]

            scaler = GradScaler('cuda' if device.type == 'cuda' else 'cpu', enabled=base_amp)
            amp = base_amp

        train_stats = train_epoch(
            encoders=encoders,
            fusion=fusion,
            head=head,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp=amp,
            log_interval=log_interval,
            freeze_encoders=freeze_encoders,
        )
        print(f"Train: loss {train_stats['loss']:.4f}, acc {train_stats['acc']:.4f}")

        if val_loader is not None:
            val_stats = evaluate(encoders, fusion, head, val_loader, device=device, amp=amp, freeze_encoders=freeze_encoders)
            print(f"Val  : loss {val_stats['loss']:.4f}, acc {val_stats['acc']:.4f}")

        if (epoch + 1) % int(cfg["training"].get("save_every", 1)) == 0:
            ckpt_path = os.path.join(run_save_dir, f"{args.run_name}_epoch{epoch+1}.pt")
            save_checkpoint(ckpt_path, fusion, head, optimizer, scaler, epoch + 1)
            print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()

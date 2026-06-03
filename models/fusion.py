from __future__ import annotations

from typing import Dict, Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class BranchProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FusionNetwork(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: Iterable[int], out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        dims: List[int] = [in_dim, *hidden_dims, out_dim]
        layers: List[nn.Module] = []
        for idx in range(len(dims) - 1):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            if idx < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[idx + 1]))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiBranchFusion(nn.Module):
    def __init__(
        self,
        branch_input_dims: Dict[str, int],
        projector_dim: int,
        fusion_hidden: Iterable[int],
        fusion_out: int,
        dropout: float = 0.1,
        l2_norm: bool = True,
        mode: str = "concat",
        moe_routing: str = "dense",
        top_k: int = 1,
    ) -> None:
        super().__init__()
        self.branch_names = list(branch_input_dims.keys())
        self.projectors = nn.ModuleDict(
            {name: BranchProjector(in_dim=dim, out_dim=projector_dim, dropout=dropout) for name, dim in branch_input_dims.items()}
        )
        self.mode = mode.lower()
        self.l2_norm = l2_norm
        self.num_branches = len(self.projectors)
        self.moe_routing = moe_routing.lower()
        self.top_k = top_k

        if self.moe_routing not in {"dense", "topk", "top-k"}:
            raise ValueError(f"Unsupported moe_routing: {moe_routing}")
        if self.top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")

        # Shared concatenated dimension h = [h1; ...; hN]
        concat_dim = projector_dim * self.num_branches

        if self.mode == "concat":
            fused_input_dim = concat_dim
            self.fusion = FusionNetwork(fused_input_dim, fusion_hidden, fusion_out, dropout=dropout)
            self.model_gate = None
            self.feature_gate = None
            self.experts = None
            self.out_proj = None
        elif self.mode == "model_gating":
            # alpha = softmax(W_g h + b_g), z = sum_i alpha_i h_i
            self.model_gate = nn.Linear(concat_dim, self.num_branches)
            self.out_proj = nn.Linear(projector_dim, fusion_out) if fusion_out != projector_dim else nn.Identity()
            self.fusion = None
            self.feature_gate = None
            self.experts = None
        elif self.mode == "feature_gating":
            # g = sigmoid(W_g h + b_g), split g=[g1;...;gN], z = sum_i g_i ⊙ h_i
            self.feature_gate = nn.Linear(concat_dim, concat_dim)
            self.out_proj = nn.Linear(projector_dim, fusion_out) if fusion_out != projector_dim else nn.Identity()
            self.fusion = None
            self.model_gate = None
            self.experts = None
        elif self.mode in {"moe", "mixture_of_experts"}:
            # e_i = phi(W_i h_i + b_i), alpha = softmax(W_g h + b_g)
            # z = sum_i alpha_i e_i + mean_i(h_i)
            self.model_gate = nn.Linear(concat_dim, self.num_branches)
            self.experts = nn.ModuleDict(
                {
                    name: nn.Sequential(
                        nn.Linear(projector_dim, projector_dim),
                        nn.GELU(),
                    )
                    for name in self.branch_names
                }
            )
            self.out_proj = nn.Linear(projector_dim, fusion_out) if fusion_out != projector_dim else nn.Identity()
            self.fusion = None
            self.feature_gate = None
        elif self.mode in {"complex_moe", "complex_mixture_of_experts"}:
            # Complex experts: D -> 4D -> D
            # Routing: dense softmax or top-k sparse softmax
            self.model_gate = nn.Linear(concat_dim, self.num_branches)
            self.experts = nn.ModuleDict(
                {
                    name: nn.Sequential(
                        nn.Linear(projector_dim, 4 * projector_dim),
                        nn.LayerNorm(4 * projector_dim),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(4 * projector_dim, projector_dim),
                    )
                    for name in self.branch_names
                }
            )
            self.out_proj = nn.Linear(projector_dim, fusion_out) if fusion_out != projector_dim else nn.Identity()
            self.fusion = None
            self.feature_gate = None
        else:
            raise ValueError(f"Unsupported fusion mode: {mode}")

    def _compute_moe_weights(self, gate_logits: torch.Tensor):
        # Dense routing
        if self.moe_routing == "dense":
            return torch.softmax(gate_logits, dim=1), None

        # top-k / top-k routing
        k = min(self.top_k, self.num_branches)
        topk_vals, topk_idx = torch.topk(gate_logits, k=k, dim=1)
        sparse_logits = torch.full_like(gate_logits, float("-inf"))
        sparse_logits.scatter_(1, topk_idx, topk_vals)
        return torch.softmax(sparse_logits, dim=1), topk_idx

    def forward(self, embeddings: Dict[str, torch.Tensor], return_extras: bool = False):
        extras: Dict[str, torch.Tensor] = {}
        projected = []
        for name in self.branch_names:
            if name not in embeddings:
                raise KeyError(f"Missing embedding for branch '{name}'")
            x = embeddings[name]
            # L2-normalize branch embeddings to stabilize scale
            x = F.normalize(x, dim=1, eps=1e-6)
            # Project and guard against NaNs/Infs
            x = self.projectors[name](x)
            x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
            projected.append(x)

        h = torch.cat(projected, dim=1)  # [B, N*D]
        if self.mode == "concat":
            fused = self.fusion(h)
        elif self.mode == "model_gating":
            stacked = torch.stack(projected, dim=1)  # [B, N, D]
            alpha = torch.softmax(self.model_gate(h), dim=1)  # [B, N]
            fused = (alpha.unsqueeze(-1) * stacked).sum(dim=1)  # [B, D]
            fused = self.out_proj(fused)
            if return_extras:
                extras["gating_weights"] = alpha
        elif self.mode == "feature_gating":
            stacked = torch.stack(projected, dim=1)  # [B, N, D]
            gates = torch.sigmoid(self.feature_gate(h)).reshape(h.size(0), self.num_branches, -1)  # [B, N, D]
            fused = (gates * stacked).sum(dim=1)  # [B, D]
            fused = self.out_proj(fused)
            if return_extras:
                extras["feature_gates"] = gates
        elif self.mode in {"moe", "mixture_of_experts", "complex_moe", "complex_mixture_of_experts"}:
            stacked = torch.stack(projected, dim=1)  # [B, N, D]
            gate_logits = self.model_gate(h)  # [B, N]
            alpha, topk_idx = self._compute_moe_weights(gate_logits)
            expert_out = torch.stack([self.experts[name](x) for name, x in zip(self.branch_names, projected)], dim=1)  # [B, N, D]
            z_moe = (alpha.unsqueeze(-1) * expert_out).sum(dim=1)  # [B, D]
            residual = stacked.mean(dim=1)  # [B, D]
            fused = z_moe + residual
            fused = self.out_proj(fused)
            if return_extras:
                extras["gating_weights"] = alpha
                if topk_idx is not None:
                    extras["topk_indices"] = topk_idx
                extras["moe_residual"] = residual
        else:
            raise RuntimeError(f"Unhandled fusion mode: {self.mode}")

        fused = torch.nan_to_num(fused, nan=0.0, posinf=1e4, neginf=-1e4)
        if self.l2_norm:
            fused = F.normalize(fused, dim=1, eps=1e-6)
        if return_extras:
            return fused, extras
        return fused

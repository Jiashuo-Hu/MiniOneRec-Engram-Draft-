import math
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn

from sid_engram.sid_utils import PAD_SID_ID


@dataclass
class SidEngramConfig:
    hidden_size: int
    memory_dim: int = 512
    num_hash_buckets: int = 262144
    ngram_orders: List[int] = field(default_factory=lambda: [1])
    use_hierarchical_sid: bool = False
    injection_mode: str = "response_anchor"
    recency_decay: float = 0.2
    gate_min: float = 0.0
    warmup_steps: int = 0
    soft_constraint_steps: int = 0
    layer_ids: List[int] = field(default_factory=lambda: [8, 16])

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class SidEngram(nn.Module):
    def __init__(self, config: SidEngramConfig):
        super().__init__()
        self.config = config
        self.memory = nn.Embedding(config.num_hash_buckets, config.memory_dim)
        self.key_proj = nn.Linear(config.memory_dim, config.hidden_size)
        self.value_proj = nn.Linear(config.memory_dim, config.hidden_size)
        self.query_norm = nn.RMSNorm(config.hidden_size)
        self.key_norm = nn.RMSNorm(config.hidden_size)
        self.register_buffer("_global_step", torch.zeros((), dtype=torch.long), persistent=False)
        self.last_stats = {}

    def set_step(self, step: int):
        self._global_step.fill_(int(step))

    def _hash_1gram(self, sid_ids):
        return (sid_ids * 1000003 + 17).remainder(self.config.num_hash_buckets)

    def _hash_2gram(self, left, right):
        return (left * 1000003 + right * 9176 + 19260817).remainder(self.config.num_hash_buckets)

    def _hash_prefix(self, prefix_ids, level):
        return (prefix_ids * 1000003 + 314159 + int(level) * 271828).remainder(self.config.num_hash_buckets)

    def _build_hashes(self, sid_history_ids, sid_prefix_ids=None):
        valid = sid_history_ids.ge(0)
        hashes = []
        masks = []
        if 1 in self.config.ngram_orders:
            hashes.append(self._hash_1gram(sid_history_ids.clamp_min(0)))
            masks.append(valid)
        if 2 in self.config.ngram_orders and sid_history_ids.size(1) > 1:
            left = sid_history_ids[:, :-1].clamp_min(0)
            right = sid_history_ids[:, 1:].clamp_min(0)
            hashes.append(self._hash_2gram(left, right))
            masks.append(valid[:, :-1] & valid[:, 1:])
        if self.config.use_hierarchical_sid and sid_prefix_ids is not None:
            for level in range(min(3, sid_prefix_ids.size(-1))):
                prefix = sid_prefix_ids[:, :, level]
                prefix_valid = prefix.ge(0)
                hashes.append(self._hash_prefix(prefix.clamp_min(0), level))
                masks.append(prefix_valid)
        if not hashes:
            empty = sid_history_ids.new_zeros((sid_history_ids.size(0), 1))
            return empty, empty.bool()
        return torch.cat(hashes, dim=1), torch.cat(masks, dim=1)

    def forward(self, hidden_states, sid_history_ids=None, response_anchor_pos=None, sid_prefix_ids=None, attention_mask=None):
        if sid_history_ids is None or response_anchor_pos is None:
            return torch.zeros_like(hidden_states)
        if hidden_states.dim() != 3:
            raise ValueError("SidEngram expects hidden_states with shape [B, L, D].")

        B, L, D = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Ensure all params are on the correct device (accelerate device_map="auto"
        # may leave freshly-attached modules on CPU)
        if self.memory.weight.device != device:
            self.to(device)

        sid_history_ids = sid_history_ids.to(device)
        if sid_prefix_ids is not None:
            sid_prefix_ids = sid_prefix_ids.to(device)
        response_anchor_pos = response_anchor_pos.to(device).long()
        valid_batch = response_anchor_pos.ge(0) & response_anchor_pos.lt(L)
        if not valid_batch.any():
            return torch.zeros_like(hidden_states)

        hash_ids, mask = self._build_hashes(sid_history_ids, sid_prefix_ids=sid_prefix_ids)
        hash_ids = hash_ids.to(device)
        mask = mask.to(device)
        mem = self.memory(hash_ids)

        positions = torch.arange(mask.size(1), device=device).float()
        recency = torch.exp(-self.config.recency_decay * (mask.size(1) - 1 - positions))
        weights = recency.unsqueeze(0) * mask.float()
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        pooled = (mem * weights.unsqueeze(-1)).sum(dim=1) / denom

        key = self.key_norm(self.key_proj(pooled))
        value = self.value_proj(pooled)
        query = hidden_states[torch.arange(B, device=device), response_anchor_pos.clamp(0, L - 1)]
        query = self.query_norm(query)
        gate = (query * key).sum(dim=-1) / math.sqrt(D)
        gate = torch.sigmoid(gate).unsqueeze(-1)
        if self.config.gate_min > 0:
            gate = gate.clamp_min(self.config.gate_min)

        injection = (gate * value).to(hidden_states.dtype)
        injection = injection * valid_batch.float().unsqueeze(-1).to(hidden_states.dtype)
        output = torch.zeros_like(hidden_states)
        output[torch.arange(B, device=device), response_anchor_pos.clamp(0, L - 1)] = injection

        with torch.no_grad():
            self.last_stats = {
                "gate_mean": float(gate.mean().detach().cpu()),
                "gate_std": float(gate.std().detach().cpu()) if gate.numel() > 1 else 0.0,
                "memory_norm": float(pooled.norm(dim=-1).mean().detach().cpu()),
                "injection_norm": float(injection.norm(dim=-1).mean().detach().cpu()),
                "active_trigger_count": float(mask.sum(dim=1).float().mean().detach().cpu()),
            }
        return output

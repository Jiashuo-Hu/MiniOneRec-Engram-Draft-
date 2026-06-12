import json
import os
import types
from typing import Iterable

import torch

from sid_engram.sid_engram import SidEngram, SidEngramConfig
from sid_engram.sid_utils import deterministic_hash, load_json


def _get_layers(model):
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("Expected a Qwen2-style causal LM with model.layers.")
    return model.model.layers


def _sid_engram_pre_hook(module, args, kwargs):
    sid_history_ids = kwargs.get("sid_history_ids")
    response_anchor_pos = kwargs.get("response_anchor_pos")
    sid_prefix_ids = kwargs.get("sid_prefix_ids")
    if sid_history_ids is None or response_anchor_pos is None:
        return args, kwargs
    if not args:
        return args, kwargs
    hidden_states = args[0]
    if hidden_states.size(1) <= 1:
        return args, kwargs
    hidden_states = hidden_states + module.sid_engram(
        hidden_states,
            sid_history_ids=sid_history_ids,
            response_anchor_pos=response_anchor_pos,
            sid_prefix_ids=sid_prefix_ids,
            attention_mask=kwargs.get("attention_mask"),
        )
    return (hidden_states,) + args[1:], kwargs


def attach_sid_engram(model, config: SidEngramConfig):
    layers = _get_layers(model)
    model.sid_engram_config = config.to_dict()
    for layer_id in config.layer_ids:
        if layer_id < 0 or layer_id >= len(layers):
            raise ValueError(f"layer_id {layer_id} out of range for {len(layers)} layers")
        layer = layers[layer_id]
        if not hasattr(layer, "sid_engram"):
            layer.sid_engram = SidEngram(config)
            layer.register_forward_pre_hook(_sid_engram_pre_hook, with_kwargs=True)
    _patch_model_kwargs(model)
    return model


def _patch_model_kwargs(model):
    if getattr(model, "_sid_engram_kwargs_patched", False):
        return
    original_forward = model.forward
    original_prepare = model.prepare_inputs_for_generation

    def forward_with_sid(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        cache_position=None,
        logits_to_keep=0,
        sid_history_ids=None,
        sid_prefix_ids=None,
        response_anchor_pos=None,
        **kwargs,
    ):
        if sid_history_ids is not None:
            kwargs["sid_history_ids"] = sid_history_ids
        if sid_prefix_ids is not None:
            kwargs["sid_prefix_ids"] = sid_prefix_ids
        if response_anchor_pos is not None:
            kwargs["response_anchor_pos"] = response_anchor_pos
        return original_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

    def prepare_inputs_for_generation_with_sid(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        logits_to_keep=None,
        sid_history_ids=None,
        sid_prefix_ids=None,
        response_anchor_pos=None,
        **kwargs,
    ):
        model_inputs = original_prepare(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )
        model_inputs["sid_history_ids"] = sid_history_ids
        model_inputs["sid_prefix_ids"] = sid_prefix_ids
        model_inputs["response_anchor_pos"] = response_anchor_pos
        return model_inputs

    model.forward = types.MethodType(forward_with_sid, model)
    model.prepare_inputs_for_generation = types.MethodType(prepare_inputs_for_generation_with_sid, model)
    model._sid_engram_kwargs_patched = True


def iter_sid_engram_modules(model) -> Iterable[SidEngram]:
    for layer in _get_layers(model):
        if hasattr(layer, "sid_engram"):
            yield layer.sid_engram


def sid_engram_state_dict(model):
    return {f"sid_engram_layers.{i}": module.state_dict() for i, module in enumerate(iter_sid_engram_modules(model))}


def save_sid_engram_checkpoint(model, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    torch.save(sid_engram_state_dict(model), os.path.join(output_dir, "sid_engram.pt"))
    with open(os.path.join(output_dir, "sid_engram_config.json"), "w", encoding="utf-8") as f:
        json.dump(model.sid_engram_config, f, indent=2)


def load_sid_engram_checkpoint(model, checkpoint_dir, map_location="cpu"):
    config_path = os.path.join(checkpoint_dir, "sid_engram_config.json")
    state_path = os.path.join(checkpoint_dir, "sid_engram.pt")
    config = SidEngramConfig.from_dict(load_json(config_path))
    attach_sid_engram(model, config)
    state = torch.load(state_path, map_location=map_location)
    modules = list(iter_sid_engram_modules(model))
    for i, module in enumerate(modules):
        key = f"sid_engram_layers.{i}"
        if key in state:
            module.load_state_dict(state[key])
    return model


@torch.no_grad()
def _bucket_for_trigger(trigger, sid2idx, prefix2idx, num_hash_buckets):
    if trigger.startswith("sid:"):
        sid = trigger[len("sid:") :]
        if sid not in sid2idx:
            return None
        sid_id = int(sid2idx[sid])
        return int((sid_id * 1000003 + 17) % num_hash_buckets)
    if trigger.startswith("bigram:"):
        pair = trigger[len("bigram:") :]
        if "|" not in pair:
            return None
        left, right = pair.split("|", 1)
        if left not in sid2idx or right not in sid2idx:
            return None
        l_id, r_id = int(sid2idx[left]), int(sid2idx[right])
        return int((l_id * 1000003 + r_id * 9176 + 19260817) % num_hash_buckets)
    if trigger.startswith("prefix:"):
        prefix = trigger[len("prefix:") :]
        if prefix not in prefix2idx:
            return None
        level = max(0, prefix.count("<") - 1)
        p_id = int(prefix2idx[prefix])
        return int((p_id * 1000003 + 314159 + level * 271828) % num_hash_buckets)
    return deterministic_hash(trigger, num_hash_buckets)


def initialize_sid_engram_from_stats(model, tokenizer, sid_stats_dir, sid_vocab_dir, num_hash_buckets):
    trigger2targets_path = os.path.join(sid_stats_dir, "trigger2targets.json")
    trigger2scores_path = os.path.join(sid_stats_dir, "trigger2scores.json")
    if not os.path.exists(trigger2targets_path) or not os.path.exists(trigger2scores_path):
        return
    trigger2targets = load_json(trigger2targets_path)
    trigger2scores = load_json(trigger2scores_path)
    sid2idx = load_json(os.path.join(sid_vocab_dir, "sid2idx.json"))
    prefix2idx = load_json(os.path.join(sid_vocab_dir, "prefix2idx.json"))
    emb = model.get_input_embeddings().weight.detach()
    device = emb.device
    bucket_sum = {}
    bucket_weight = {}
    for trigger, targets in trigger2targets.items():
        scores = trigger2scores.get(trigger, [1.0] * len(targets))
        vecs = []
        weights = []
        for target, score in zip(targets, scores):
            ids = tokenizer(target, add_special_tokens=False).input_ids
            if not ids:
                continue
            ids = torch.tensor(ids, device=device)
            vecs.append(emb[ids].mean(dim=0))
            weights.append(float(score))
        if not vecs:
            continue
        w = torch.tensor(weights, device=device)
        v = (torch.stack(vecs) * w[:, None]).sum(dim=0) / w.sum().clamp_min(1e-6)
        bucket = _bucket_for_trigger(trigger, sid2idx, prefix2idx, num_hash_buckets)
        if bucket is None:
            continue
        bucket_sum[bucket] = bucket_sum.get(bucket, 0) + v
        bucket_weight[bucket] = bucket_weight.get(bucket, 0) + 1

    for module in iter_sid_engram_modules(model):
        mem_dim = module.memory.weight.size(1)
        for bucket, vec in bucket_sum.items():
            vec = vec / float(bucket_weight[bucket])
            if vec.numel() >= mem_dim:
                init = vec[:mem_dim]
            else:
                init = torch.zeros(mem_dim, device=device)
                init[: vec.numel()] = vec
            with torch.no_grad():
                module.memory.weight[bucket].copy_(init.to(module.memory.weight.device, dtype=module.memory.weight.dtype))

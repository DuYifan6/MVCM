import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class ConsistencyAwareMultiheadAttention(nn.Module):
    """Token-level multi-head self-attention with an additive MVCM bias."""

    def __init__(self, hidden_size, num_heads, dropout=0.1, lambda_init=1.0):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(dropout)
        self.lambda_bias = nn.Parameter(torch.tensor(float(lambda_init)))

    def _split_heads(self, tensor):
        batch, length, _ = tensor.shape
        tensor = tensor.view(batch, length, self.num_heads, self.head_dim)
        return tensor.transpose(1, 2)

    def forward(self, hidden_states, attention_mask, consistency_bias):
        query = self._split_heads(self.q_proj(hidden_states))
        key = self._split_heads(self.k_proj(hidden_states))
        value = self._split_heads(self.v_proj(hidden_states))
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        scores = scores + self.lambda_bias * consistency_bias.unsqueeze(1)

        key_mask = attention_mask[:, None, None, :].bool()
        scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
        probabilities = torch.softmax(scores, dim=-1)
        probabilities = probabilities.masked_fill(~key_mask, 0.0)
        dropped = self.attn_dropout(probabilities)

        context = torch.matmul(dropped, value)
        context = context.transpose(1, 2).contiguous().view(
            hidden_states.size(0), hidden_states.size(1), self.hidden_size
        )
        output = self.out_proj(context)
        output = output * attention_mask.unsqueeze(-1).to(output.dtype)
        return output, probabilities


class ConsistencyAwareTransformerLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        ff_multiplier=4,
        dropout=0.1,
        lambda_init=1.0,
    ):
        super().__init__()
        self.attention = ConsistencyAwareMultiheadAttention(
            hidden_size, num_heads, dropout=dropout, lambda_init=lambda_init
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ff_multiplier * hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_multiplier * hidden_size, hidden_size),
        )

    def forward(self, hidden_states, attention_mask, consistency_bias):
        attended, probabilities = self.attention(
            hidden_states, attention_mask, consistency_bias
        )
        hidden_states = self.norm1(hidden_states + self.dropout1(attended))
        hidden_states = self.norm2(hidden_states + self.dropout2(self.ffn(hidden_states)))
        hidden_states = hidden_states * attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        return hidden_states, probabilities


class MVCTTokenCATModel(nn.Module):
    """BERT followed by stacked token-level consistency-aware layers."""

    def __init__(
        self,
        bert_model_name="bert-base-uncased",
        mvc_weights: Sequence[float] = (0.4, 0.25, 0.2, 0.15),
        cat_layers=2,
        cat_heads=8,
        cat_ff_multiplier=4,
        dropout_p=0.1,
        lambda_init=1.0,
        learnable_mvc_weights=True,
        encoder: Optional[nn.Module] = None,
        mask_missing_evidence=False,
    ):
        super().__init__()
        self.mask_missing_evidence = bool(mask_missing_evidence)
        self.encoder = encoder or AutoModel.from_pretrained(bert_model_name)
        hidden_size = int(self.encoder.config.hidden_size)
        if not mvc_weights:
            raise ValueError("mvc_weights cannot be empty")

        raw_weights = torch.tensor(mvc_weights, dtype=torch.float)
        active = raw_weights > 0
        self.register_buffer("channel_mask", active)
        safe_weights = torch.where(active, raw_weights, torch.ones_like(raw_weights))
        safe_weights = safe_weights / safe_weights[active].sum().clamp_min(1e-12)
        logits = safe_weights.clamp_min(1e-8).log()
        if learnable_mvc_weights:
            self.channel_logits = nn.Parameter(logits)
        else:
            self.register_buffer("channel_logits", logits)

        self.cat_layers = nn.ModuleList(
            [
                ConsistencyAwareTransformerLayer(
                    hidden_size=hidden_size,
                    num_heads=cat_heads,
                    ff_multiplier=cat_ff_multiplier,
                    dropout=dropout_p,
                    lambda_init=lambda_init,
                )
                for _ in range(cat_layers)
            ]
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_p),
        )
        self.head_fake = nn.Linear(hidden_size, 2)
        self.head_ai = nn.Linear(hidden_size, 2)
        self.last_attn = None
        self.last_view_attention = None

    @property
    def mvc_weights(self):
        if not bool(self.channel_mask.any()):
            return torch.zeros_like(self.channel_logits)
        masked_logits = self.channel_logits.masked_fill(~self.channel_mask, -1e9)
        return torch.softmax(masked_logits, dim=0) * self.channel_mask.to(
            self.channel_logits.dtype
        )

    def combine_mvc(self, mvc_tensor):
        if mvc_tensor.dim() == 3:
            mvc_tensor = mvc_tensor.unsqueeze(0)
        if mvc_tensor.dim() != 4 or mvc_tensor.shape[1:3] != (3, 3):
            raise ValueError(f"Expected MVCM shape (B, 3, 3, C), got {mvc_tensor.shape}")
        channels = mvc_tensor.size(-1)
        if self.mask_missing_evidence and channels != 4:
            raise ValueError("Missing-evidence masking requires four MVC channels")
        if channels > self.mvc_weights.numel():
            raise ValueError("MVCM has more channels than configured weights")
        if not bool(self.channel_mask[:channels].any()):
            return torch.zeros_like(mvc_tensor[..., 0])
        weights = self.mvc_weights[:channels]
        centered = 2.0 * mvc_tensor - 1.0
        original_bias = (centered * weights.view(1, 1, 1, channels)).sum(dim=-1)
        if not self.mask_missing_evidence:
            return original_bias

        # A proxy for absent comparable evidence, not a proven OpenIE failure.
        # Keep the stored cache untouched and exclude only factual/logical
        # contributions for pairs with both sentinel values.
        missing = torch.isclose(mvc_tensor[..., 1], mvc_tensor.new_tensor(0.0), atol=1e-6, rtol=0)
        missing = missing & torch.isclose(mvc_tensor[..., 2], mvc_tensor.new_tensor(0.5), atol=1e-6, rtol=0)
        available = torch.ones_like(mvc_tensor)
        available[..., 1] = (~missing).to(mvc_tensor.dtype)
        available[..., 2] = (~missing).to(mvc_tensor.dtype)
        pair_weights = available * weights.view(1, 1, 1, channels)
        # If all configured channels are unavailable, use zero (neutral) bias.
        pair_weights = pair_weights / pair_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        masked_bias = (centered * pair_weights).sum(dim=-1)
        return torch.where(missing, masked_bias, original_bias)

    def build_token_bias(self, mvc_tensor, view_ids, attention_mask):
        view_bias = self.combine_mvc(mvc_tensor)
        batch_size, length = view_ids.shape
        if view_bias.size(0) == 1 and batch_size > 1:
            view_bias = view_bias.expand(batch_size, -1, -1)
        if view_bias.size(0) != batch_size:
            raise ValueError("MVCM batch size does not match token batch size")

        safe_views = view_ids.clamp(min=0, max=2)
        batch_index = torch.arange(batch_size, device=view_ids.device)[:, None, None]
        row_index = safe_views[:, :, None].expand(-1, -1, length)
        col_index = safe_views[:, None, :].expand(-1, length, -1)
        token_bias = view_bias[batch_index, row_index, col_index]
        valid_views = view_ids.ge(0)
        valid_pairs = valid_views[:, :, None] & valid_views[:, None, :]
        valid_tokens = attention_mask.bool()
        valid_pairs = valid_pairs & valid_tokens[:, :, None] & valid_tokens[:, None, :]
        return token_bias.masked_fill(~valid_pairs, 0.0)

    @staticmethod
    def aggregate_attention_by_view(attention, view_ids, attention_mask):
        """Average token attention over heads and token pairs into (B, 3, 3)."""
        if attention.dim() != 4:
            raise ValueError("attention must have shape (B, heads, N, N)")
        mean_attention = attention.mean(dim=1)
        valid_tokens = attention_mask.bool()
        result = mean_attention.new_zeros((attention.size(0), 3, 3))
        for source in range(3):
            source_mask = view_ids.eq(source) & valid_tokens
            for target in range(3):
                target_mask = view_ids.eq(target) & valid_tokens
                pair_mask = source_mask[:, :, None] & target_mask[:, None, :]
                numerator = (mean_attention * pair_mask).sum(dim=(1, 2))
                denominator = pair_mask.sum(dim=(1, 2)).clamp_min(1)
                result[:, source, target] = numerator / denominator
        return result

    def forward(self, sequence, mvc_tensor):
        input_ids = sequence["input_ids"]
        attention_mask = sequence["attention_mask"]
        view_ids = sequence["view_ids"]
        encoded = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden_states = encoded.last_hidden_state
        token_bias = self.build_token_bias(mvc_tensor, view_ids, attention_mask)

        attention = None
        for layer in self.cat_layers:
            hidden_states, attention = layer(hidden_states, attention_mask, token_bias)
        if attention is None:
            raise RuntimeError("At least one CAT layer is required")

        self.last_attn = attention
        self.last_view_attention = self.aggregate_attention_by_view(
            attention, view_ids, attention_mask
        )
        features = self.classifier(hidden_states[:, 0])
        return self.head_fake(features), self.head_ai(features), attention


def attention_kl_regularizer(attention, attention_mask, view_ids=None):
    """KL(A || uniform) over valid keys; zero for uniform attention."""
    if attention is None:
        return torch.tensor(0.0, device=attention_mask.device)
    probabilities = attention.clamp_min(1e-12)
    entropy = -(probabilities * probabilities.log()).sum(dim=-1)
    valid_key_count = attention_mask.sum(dim=-1).clamp_min(1).to(attention.dtype)
    uniform_entropy = valid_key_count.log()[:, None, None]
    kl = uniform_entropy - entropy
    valid_queries = attention_mask.bool()
    if view_ids is not None:
        valid_queries = valid_queries & view_ids.ge(0)
    expanded_mask = valid_queries[:, None, :].expand_as(kl)
    if not bool(expanded_mask.any()):
        return kl.new_tensor(0.0)
    return kl.masked_select(expanded_mask).mean().clamp_min(0.0)

"""Controlled dual-task baselines for the frozen MVCT data protocol."""
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModel


class BaselineNewsDataset(Dataset):
    """The same title/body/description token budget, without loading MVCM."""

    def __init__(self, frame, tokenizer, max_title=32, max_body=256, max_desc=64):
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_lengths = (max_title, max_body, max_desc)
        self.length = 4 + max_title + max_body + max_desc
        if any(value is None for value in (
            tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id
        )):
            raise ValueError("Tokenizer must define CLS, SEP, and PAD token IDs.")

    def __len__(self):
        return len(self.frame)

    def _tokens(self, value, limit):
        return self.tokenizer(
            str(value), add_special_tokens=False, truncation=True,
            max_length=limit, return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        ids = [self.tokenizer.cls_token_id]
        for value, limit in zip(
            (row.title, row.text, row.desc), self.max_lengths
        ):
            ids.extend(self._tokens(value, limit))
            ids.append(self.tokenizer.sep_token_id)
        attention = [1] * len(ids)
        padding = self.length - len(ids)
        if padding < 0:
            raise RuntimeError("Baseline sequence exceeds the fixed token budget.")
        ids.extend([self.tokenizer.pad_token_id] * padding)
        attention.extend([0] * padding)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "label_fake": torch.tensor(int(row.label_fake), dtype=torch.long),
            "label_ai": torch.tensor(int(row.label_ai), dtype=torch.long),
            "sample_id": torch.tensor(int(row.ID), dtype=torch.long),
        }


class DualHeadTransformer(nn.Module):
    """Plain pretrained encoder with the same shared projection and two heads."""

    def __init__(self, model_name, dropout=0.1, encoder: Optional[nn.Module] = None):
        super().__init__()
        self.encoder = encoder or AutoModel.from_pretrained(model_name)
        hidden = int(self.encoder.config.hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout)
        )
        self.head_fake = nn.Linear(hidden, 2)
        self.head_ai = nn.Linear(hidden, 2)

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0]
        hidden = self.classifier(hidden)
        return self.head_fake(hidden), self.head_ai(hidden)


class DualHeadTextCNN(nn.Module):
    """Kim-style TextCNN over the identical tokenized three-view sequence."""

    def __init__(
        self, vocab_size, pad_token_id, embedding_dim=128,
        filters=128, kernels=(3, 4, 5), dropout=0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_token_id
        )
        self.convolutions = nn.ModuleList([
            nn.Conv1d(embedding_dim, filters, kernel_size=kernel)
            for kernel in kernels
        ])
        width = filters * len(kernels)
        self.classifier = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout)
        )
        self.head_fake = nn.Linear(width, 2)
        self.head_ai = nn.Linear(width, 2)

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids).transpose(1, 2)
        features = [
            torch.relu(layer(embedded)).amax(dim=-1)
            for layer in self.convolutions
        ]
        hidden = self.classifier(torch.cat(features, dim=1))
        return self.head_fake(hidden), self.head_ai(hidden)

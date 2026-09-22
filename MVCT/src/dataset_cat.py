import hashlib
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm


VIEW_TITLE = 0
VIEW_BODY = 1
VIEW_DESCRIPTION = 2
VIEW_SPECIAL = -1


def _ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def _sanitize_mvc(mvc_arr: np.ndarray) -> np.ndarray:
    mvc_arr = np.asarray(mvc_arr, dtype=np.float32)
    if mvc_arr.ndim == 3 and mvc_arr.shape == (3, 3, 3):
        local = np.eye(3, dtype=np.float32)
        mvc_arr = np.concatenate([mvc_arr, local[..., None]], axis=-1)
    if mvc_arr.ndim != 3 or mvc_arr.shape[:2] != (3, 3):
        raise ValueError(f"Bad MVC shape: {mvc_arr.shape}; expected (3, 3, C)")
    mvc_arr = np.nan_to_num(mvc_arr, nan=0.0, posinf=1.0, neginf=0.0)
    mvc_arr = np.clip(mvc_arr, 0.0, 1.0)
    for channel in range(mvc_arr.shape[-1]):
        np.fill_diagonal(mvc_arr[:, :, channel], 1.0)
    return mvc_arr


def _cache_key(row, builder) -> str:
    signature = getattr(builder, "cache_signature", "mvcm-v2") if builder else "mvcm-v2"
    payload = json.dumps(
        {
            "title": str(row.title),
            "text": str(row.text),
            "desc": str(row.desc),
            "builder": signature,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


class MVCNewsDatasetCAT(Dataset):
    """Concatenated token sequence plus a cached four-channel MVCM."""

    def __init__(
        self,
        df,
        tokenizer,
        mvc_cache_dir="mvc_cache_v2",
        max_len_title=32,
        max_len_text=256,
        max_len_desc=64,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.mvc_cache_dir = mvc_cache_dir
        self.max_len_title = max_len_title
        self.max_len_text = max_len_text
        self.max_len_desc = max_len_desc
        self.sequence_length = 4 + max_len_title + max_len_text + max_len_desc

        special_ids = (
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.pad_token_id,
        )
        if any(token_id is None for token_id in special_ids):
            raise ValueError("Tokenizer must define CLS, SEP, and PAD token IDs")

    def __len__(self):
        return len(self.df)

    def _content_tokens(self, text, max_len):
        encoded = self.tokenizer(
            str(text),
            add_special_tokens=False,
            truncation=True,
            max_length=max_len,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return list(encoded["input_ids"])

    def _encode_three_views(self, title, body, description):
        view_tokens = (
            (self._content_tokens(title, self.max_len_title), VIEW_TITLE),
            (self._content_tokens(body, self.max_len_text), VIEW_BODY),
            (self._content_tokens(description, self.max_len_desc), VIEW_DESCRIPTION),
        )
        input_ids = [self.tokenizer.cls_token_id]
        view_ids = [VIEW_SPECIAL]
        for tokens, view_id in view_tokens:
            input_ids.extend(tokens)
            view_ids.extend([view_id] * len(tokens))
            input_ids.append(self.tokenizer.sep_token_id)
            view_ids.append(VIEW_SPECIAL)

        attention_mask = [1] * len(input_ids)
        pad_count = self.sequence_length - len(input_ids)
        if pad_count < 0:
            raise RuntimeError("Constructed sequence exceeds configured maximum length")
        input_ids.extend([self.tokenizer.pad_token_id] * pad_count)
        attention_mask.extend([0] * pad_count)
        view_ids.extend([VIEW_SPECIAL] * pad_count)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "view_ids": torch.tensor(view_ids, dtype=torch.long),
        }

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mvc_path = Path(row["mvc_path"])
        if not mvc_path.exists():
            raise FileNotFoundError(
                f"MVC cache is missing: {mvc_path}. Re-run with precompute=True."
            )
        return {
            "sequence": self._encode_three_views(row["title"], row["text"], row["desc"]),
            "mvc": torch.tensor(_sanitize_mvc(np.load(mvc_path)), dtype=torch.float),
            "label_fake": torch.tensor(int(row["label_fake"]), dtype=torch.long),
            "label_ai": torch.tensor(int(row["label_ai"]), dtype=torch.long),
            "sample_id": torch.tensor(int(row["ID"]), dtype=torch.long),
        }


def _compute_single_mvc(row, builder, mvc_cache_dir):
    if builder is None:
        raise ValueError("builder is required when precompute=True")
    path = Path(mvc_cache_dir) / f"mvc_{_cache_key(row, builder)}.npy"
    if not path.exists():
        mvc_tensor, _ = builder.build_mvc(str(row.title), str(row.text), str(row.desc))
        sanitized = _sanitize_mvc(mvc_tensor).astype(np.float32)
        temp_name = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=path.parent, prefix=path.stem, suffix=".tmp", delete=False
            ) as handle:
                temp_name = handle.name
                np.save(handle, sanitized)
            os.replace(temp_name, path)
        finally:
            if temp_name and os.path.exists(temp_name):
                os.remove(temp_name)
    return str(path)


def _expected_cache_path(row, builder, mvc_cache_dir):
    return str(Path(mvc_cache_dir) / f"mvc_{_cache_key(row, builder)}.npy")


def _stratify_or_none(labels):
    counts = pd.Series(labels).value_counts()
    return labels if len(counts) > 1 and int(counts.min()) >= 2 else None


def _clean_and_group_dataframe(
    path,
    drop_incomplete=True,
    max_body_chars=30000,
    group_column: Optional[str] = None,
    drop_conflicting_duplicates=True,
    drop_exact_duplicates=True,
):
    df = pd.read_csv(path).fillna("")
    required = ["ID", "title", "text", "desc", "label_fake", "label_ai"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    for column in ("title", "text", "desc"):
        df[column] = df[column].astype(str)
    if drop_incomplete:
        complete = np.logical_and.reduce(
            [df[column].str.strip().ne("") for column in ("title", "text", "desc")]
        )
        df = df.loc[complete].copy()
    if max_body_chars is not None:
        df = df.loc[df["text"].str.len() <= int(max_body_chars)].copy()

    normalized = df[["title", "text", "desc"]].apply(
        lambda col: col.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    )
    df["_content_hash"] = pd.util.hash_pandas_object(normalized, index=False).astype(str)
    label_key = df["label_fake"].astype(str) + "_" + df["label_ai"].astype(str)
    if drop_conflicting_duplicates:
        label_counts = df.assign(_label_key=label_key).groupby("_content_hash")[
            "_label_key"
        ].nunique()
        conflicting_hashes = set(label_counts[label_counts > 1].index)
        if conflicting_hashes:
            df = df.loc[~df["_content_hash"].isin(conflicting_hashes)].copy()
    if drop_exact_duplicates:
        df = df.drop_duplicates("_content_hash", keep="first").copy()

    if group_column:
        if group_column not in df.columns:
            raise ValueError(f"Configured group column does not exist: {group_column}")
        df["_split_group"] = df[group_column].astype(str)
    else:
        df["_split_group"] = df["_content_hash"]
    return df


def _group_stratified_split(df, test_size, val_size, random_state):
    work = df.assign(
        _label_key=df["label_fake"].astype(str) + "_" + df["label_ai"].astype(str)
    )
    group_labels = work.groupby("_split_group", sort=False)["_label_key"].agg(
        lambda values: values.mode().iloc[0]
    )
    groups = group_labels.index.to_numpy()
    labels = group_labels.to_numpy()
    dev_groups, test_groups = train_test_split(
        groups,
        test_size=test_size,
        random_state=random_state,
        stratify=_stratify_or_none(labels),
    )
    dev_labels = group_labels.loc[dev_groups].to_numpy()
    train_groups, val_groups = train_test_split(
        dev_groups,
        test_size=val_size / (1.0 - test_size),
        random_state=random_state,
        stratify=_stratify_or_none(dev_labels),
    )
    train_df = df[df["_split_group"].isin(set(train_groups))].copy()
    val_df = df[df["_split_group"].isin(set(val_groups))].copy()
    test_df = df[df["_split_group"].isin(set(test_groups))].copy()
    return train_df, val_df, test_df


def _save_split_manifests(train_df, val_df, test_df, split_dir):
    if not split_dir:
        return
    _ensure_dir(split_dir)
    columns = ["ID", "type", "label_fake", "label_ai", "_split_group"]
    for name, frame in (("train", train_df), ("val", val_df), ("test", test_df)):
        available = [column for column in columns if column in frame.columns]
        frame[available].to_csv(
            Path(split_dir) / f"{name}.csv", index=False, encoding="utf-8-sig"
        )


def load_and_split_with_mvc_cat(
    path,
    tokenizer,
    builder,
    test_size=0.1,
    val_size=0.1,
    mvc_cache_dir="mvc_cache_v2",
    precompute=True,
    random_state=42,
    max_len_title=32,
    max_len_text=256,
    max_len_desc=64,
    use_parallel=False,
    max_workers=4,
    drop_incomplete=True,
    max_body_chars=30000,
    group_column=None,
    split_dir=None,
    drop_conflicting_duplicates=True,
    drop_exact_duplicates=True,
):
    """Clean data, create group-safe splits, and attach versioned MVCM caches."""
    _ensure_dir(mvc_cache_dir)
    df = _clean_and_group_dataframe(
        path,
        drop_incomplete=drop_incomplete,
        max_body_chars=max_body_chars,
        group_column=group_column,
        drop_conflicting_duplicates=drop_conflicting_duplicates,
        drop_exact_duplicates=drop_exact_duplicates,
    )
    train_df, val_df, test_df = _group_stratified_split(
        df, test_size=test_size, val_size=val_size, random_state=random_state
    )
    _save_split_manifests(train_df, val_df, test_df, split_dir)

    def precompute_part(frame):
        rows = list(frame.itertuples(index=False))
        if not precompute:
            paths = [_expected_cache_path(row, builder, mvc_cache_dir) for row in rows]
            missing = [path for path in paths if not Path(path).exists()]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} MVC cache files are missing; run with precompute=True first"
                )
            return paths
        if not use_parallel:
            return [
                _compute_single_mvc(row, builder, mvc_cache_dir)
                for row in tqdm(rows, desc="Precompute MVC")
            ]

        workers = max(1, min(int(max_workers), 2))
        paths = [None] * len(rows)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_index = {
                executor.submit(_compute_single_mvc, row, builder, mvc_cache_dir): index
                for index, row in enumerate(rows)
            }
            for future in tqdm(
                as_completed(future_to_index), total=len(rows), desc="Precompute MVC"
            ):
                paths[future_to_index[future]] = future.result()
        return paths

    for frame in (train_df, val_df, test_df):
        frame["mvc_path"] = precompute_part(frame)

    dataset_args = (tokenizer, mvc_cache_dir, max_len_title, max_len_text, max_len_desc)
    return (
        MVCNewsDatasetCAT(train_df, *dataset_args),
        MVCNewsDatasetCAT(val_df, *dataset_args),
        MVCNewsDatasetCAT(test_df, *dataset_args),
    )

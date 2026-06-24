#dataset.py
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def _ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def _sanitize_mvc(mvc_arr: np.ndarray) -> np.ndarray:
    """
    强制清洗 mvc：
      - NaN/Inf -> 0
      - clip 到 [0,1]
      - 支持旧的 (3,3,3) -> 补 local 通道变 (3,3,4)
      - 对角线兜底为 1（每通道）
    """
    mvc_arr = np.array(mvc_arr, dtype=np.float32)

    # 兼容旧缓存： (3,3,3) -> (3,3,4)
    if mvc_arr.ndim == 3 and mvc_arr.shape == (3, 3, 3):
        local = np.eye(3, dtype=np.float32)
        mvc_arr = np.concatenate([mvc_arr, local[..., None]], axis=-1)

    if mvc_arr.ndim != 3 or mvc_arr.shape[0] != 3 or mvc_arr.shape[1] != 3:
        raise ValueError(f"Bad MVC shape: {mvc_arr.shape} (expect (3,3,C))")

    # NaN/Inf 清洗
    mvc_arr = np.nan_to_num(mvc_arr, nan=0.0, posinf=0.0, neginf=0.0)
    mvc_arr = np.clip(mvc_arr, 0.0, 1.0)

    # 对角线兜底 = 1
    C = mvc_arr.shape[-1]
    for k in range(C):
        np.fill_diagonal(mvc_arr[:, :, k], 1.0)

    return mvc_arr


class MVCNewsDataset(Dataset):
    """
    Dataset: title, text, desc + cached MVC
    mvc 缓存保存 (3,3,4): [semantic, fact, logic, local]
    """
    def __init__(self, df, tokenizer, mvc_cache_dir="mvc_cache",
                 max_len_title=32, max_len_text=256, max_len_desc=64):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.mvc_cache_dir = mvc_cache_dir
        self.max_len_title = max_len_title
        self.max_len_text = max_len_text
        self.max_len_desc = max_len_desc

    def __len__(self):
        return len(self.df)

    def _encode(self, text, max_len):
        return self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        t, x, d = str(row["title"]), str(row["text"]), str(row["desc"])

        t_enc = self._encode(t, self.max_len_title)
        x_enc = self._encode(x, self.max_len_text)
        d_enc = self._encode(d, self.max_len_desc)

        title = {
            "input_ids": t_enc["input_ids"].squeeze(0),
            "attention_mask": t_enc["attention_mask"].squeeze(0),
        }
        text = {
            "input_ids": x_enc["input_ids"].squeeze(0),
            "attention_mask": x_enc["attention_mask"].squeeze(0),
        }
        desc = {
            "input_ids": d_enc["input_ids"].squeeze(0),
            "attention_mask": d_enc["attention_mask"].squeeze(0),
        }

        mvc_arr = np.load(row["mvc_path"])
        mvc_arr = _sanitize_mvc(mvc_arr)             # ✅ 防 NaN/Inf/旧缓存
        mvc_tensor = torch.tensor(mvc_arr, dtype=torch.float)  # (3,3,4)

        return {
            "title": title,
            "text": text,
            "desc": desc,
            "mvc": mvc_tensor,
            "label_fake": torch.tensor(int(row["label_fake"]), dtype=torch.long),
            "label_ai": torch.tensor(int(row["label_ai"]), dtype=torch.long),
        }


def _compute_single_mvc(idx, row, builder, mvc_cache_dir):
    fname = os.path.join(mvc_cache_dir, f"mvc_{idx}.npy")
    if not os.path.exists(fname):
        mvc_tensor, _ = builder.build_mvc(str(row.title), str(row.text), str(row.desc))
        mvc_tensor = _sanitize_mvc(mvc_tensor)  # ✅ 写盘前也清洗一次
        np.save(fname, mvc_tensor.astype(np.float32))
    return fname


def load_and_split_with_mvc(path, tokenizer, builder,
                           test_size=0.1, val_size=0.1,
                           mvc_cache_dir="mvc_cache", precompute=True,
                           random_state=42,
                           max_len_title=32, max_len_text=256, max_len_desc=64,
                           use_parallel=False, max_workers=4):
    """
    方案A：use_parallel 默认 False（最稳）
    """
    _ensure_dir(mvc_cache_dir)
    df = pd.read_csv(path).fillna("")

    for c in ["title", "text", "desc", "label_fake", "label_ai"]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    strat = df["label_fake"].astype(str) + "_" + df["label_ai"].astype(str)
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=strat, random_state=random_state
    )

    strat2 = train_df["label_fake"].astype(str) + "_" + train_df["label_ai"].astype(str)
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_size / (1 - test_size),
        stratify=strat2,
        random_state=random_state,
    )

    def _precompute(df_part, start_idx=0):
        mvc_paths = []

        if not precompute:
            for i in range(len(df_part)):
                mvc_paths.append(os.path.join(mvc_cache_dir, f"mvc_{start_idx + i}.npy"))
            return mvc_paths

        print(f"\nComputing MVC ({len(df_part)} samples)...")
        rows = list(df_part.itertuples(index=False))

        # 方案A：单线程最稳
        if not use_parallel:
            for i, row in tqdm(list(enumerate(rows)), total=len(rows), desc="Precompute MVC"):
                idx = start_idx + i
                mvc_paths.append(_compute_single_mvc(idx, row, builder, mvc_cache_dir))
            return mvc_paths

        # 如果你以后一定要并行，最多 2（OpenIE+SBERT 很容易炸）
        real_workers = min(max_workers, 2)
        print(f"⚡ 并行预计算 (workers={real_workers})")
        with ThreadPoolExecutor(max_workers=real_workers) as ex:
            futures = []
            for i, row in enumerate(rows):
                idx = start_idx + i
                futures.append(ex.submit(_compute_single_mvc, idx, row, builder, mvc_cache_dir))
            for f in tqdm(as_completed(futures), total=len(futures), desc="Precompute MVC"):
                mvc_paths.append(f.result())

        return mvc_paths

    train_paths = _precompute(train_df, 0)
    val_paths = _precompute(val_df, len(train_paths))
    test_paths = _precompute(test_df, len(train_paths) + len(val_paths))

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_df["mvc_path"] = train_paths
    val_df["mvc_path"] = val_paths
    test_df["mvc_path"] = test_paths

    return (
        MVCNewsDataset(train_df, tokenizer, mvc_cache_dir, max_len_title, max_len_text, max_len_desc),
        MVCNewsDataset(val_df, tokenizer, mvc_cache_dir, max_len_title, max_len_text, max_len_desc),
        MVCNewsDataset(test_df, tokenizer, mvc_cache_dir, max_len_title, max_len_text, max_len_desc),
    )

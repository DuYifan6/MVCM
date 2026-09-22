import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


def _large_artifact_path(env_name: str, relative_path: str) -> str:
    """Keep large artifacts on the AutoDL data disk when it is available."""
    override = os.environ.get(env_name)
    if override:
        return override
    data_disk = Path("/root/autodl-tmp")
    if data_disk.is_dir():
        return str(data_disk / "MVCT_runtime" / relative_path)
    return relative_path


def _path_override(env_name: str, relative_path: str) -> str:
    """Allow portable data/model locations without editing source code."""
    return os.environ.get(env_name, relative_path)

@dataclass
class Config:
    DATA_PATH: str = _path_override("MVCT_DATA_PATH", "data/data_all.csv")
    # Keep the reconstructed CAT experiment isolated from the legacy outputs.
    OUTPUT_DIR: str = _large_artifact_path(
        "MVCT_OUTPUT_DIR", "outputs/cat_v1"
    )
    MVC_CACHE_DIR: str = _large_artifact_path(
        "MVCT_CACHE_DIR", "mvc_cache_v2"
    )

    # tokenizer / bert
    BERT_MODEL: str = _path_override(
        "MVCT_BERT_MODEL", "models/bert-base-uncased"
    )
    SBERT_MODEL: str = _path_override(
        "MVCT_SBERT_MODEL", "models/all-mpnet-base-v2"
    )
    NLI_MODEL: str = _path_override("MVCT_NLI_MODEL", "models/nli-model")
    USE_NLI: bool = False
    NLI_BATCH_SIZE: int = 16
    NLI_MAX_LENGTH: int = 256
    # MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli label order:
    # entailment=0, neutral=1, contradiction=2.
    NLI_CONTRADICTION_ID: Optional[int] = 2
    SBERT_BATCH_SIZE: int = 64
    OPENIE_WORKERS: int = 3
    GPU_BATCH_AUTO_REDUCE: bool = True

    # lengths
    MAX_LEN_TITLE: int = 32
    MAX_LEN_TEXT: int = 256
    MAX_LEN_DESC: int = 64

    # token-level Consistency-Aware Transformer (CAT)
    CAT_LAYERS: int = 2
    CAT_HEADS: int = 8
    CAT_FF_MULTIPLIER: int = 4
    CAT_DROPOUT: float = 0.1
    CAT_LAMBDA_INIT: float = 1.0
    LEARNABLE_MVC_WEIGHTS: bool = True

    # train
    BATCH_SIZE: int = 8
    LR: float = 2e-5
    EPOCHS: int = 10
    WEIGHT_DECAY: float = 0.01
    DEVICE: str = "cuda"

    # ===== MVC weights (S, F, L, Local) =====
    SEM_WEIGHT: float   = 0.4
    FACT_WEIGHT: float  = 0.25
    LOGIC_WEIGHT: float = 0.2
    LOCAL_WEIGHT: float = 0.15

    # Hybrid logical-consistency weights. They sum to one.
    LOGIC_NLI_WEIGHT: float = 0.55
    LOGIC_NEGATION_WEIGHT: float = 0.15
    LOGIC_TEMPORAL_WEIGHT: float = 0.15
    LOGIC_NUMERIC_WEIGHT: float = 0.15
    LOGIC_ALIGNMENT_THRESHOLD: float = 0.55
    LOGIC_COVERAGE_TARGET: int = 5
    MAX_LOGIC_PAIRS_PER_VIEW_PAIR: int = 12

    # StanfordCoreNLP server
    STANFORD_URL: str = "http://localhost:9000"

    # =========================
    # 方案A关键：预计算不并行
    # =========================
    USE_PARALLEL: bool = False     # ✅ 预计算阶段必须关
    MAX_WORKERS: int = 4           # 预留，方案A不使用

    # 训练阶段 DataLoader 并行（这才是“训练并行”）
    TRAIN_NUM_WORKERS: int = 4
    VAL_NUM_WORKERS: int = 2
    PIN_MEMORY: bool = True

    # 训练相关
    ATTN_REG_WEIGHT: float = 0.03
    ACCUM_STEPS: int = 1
    FOCAL_ALPHA: float = 0.8
    FOCAL_GAMMA: float = 2.0
    FAKE_CLASS_WEIGHT: Optional[Tuple[float, float]] = None
    EARLYSTOP_PATIENCE: int = 3
    FAKE_THRESHOLD: float = 0.55

    # Data quality and leakage controls.
    DROP_INCOMPLETE: bool = True
    DROP_CONFLICTING_DUPLICATES: bool = True
    DROP_EXACT_DUPLICATES: bool = True
    MAX_BODY_CHARS: int = 30000
    GROUP_COLUMN: Optional[str] = None
    SPLIT_DIR: str = "splits/cat_v1"

    # checkpoint saving
    SAVE_EPOCH_CHECKPOINTS: bool = False
    SAVE_BEST_CHECKPOINT: bool = True
    SAVE_LAST_CHECKPOINT: bool = True

cfg = Config()

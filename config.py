from dataclasses import dataclass

@dataclass
class Config:
    DATA_PATH: str = "data_all.csv"
    OUTPUT_DIR: str = "outputs"
    MVC_CACHE_DIR: str = "mvc_cache"

    # tokenizer / bert
    BERT_MODEL: str = "models/bert-base-uncased"
    SBERT_MODEL: str = "models/all-mpnet-base-v2"

    # lengths
    MAX_LEN_TITLE: int = 32
    MAX_LEN_TEXT: int = 256
    MAX_LEN_DESC: int = 64

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
    FAKE_CLASS_WEIGHT = None
    EARLYSTOP_PATIENCE: int = 3
    FAKE_THRESHOLD: float = 0.55

    # checkpoint saving
    SAVE_EPOCH_CHECKPOINTS: bool = False
    SAVE_BEST_CHECKPOINT: bool = False

cfg = Config()

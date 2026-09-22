# 检查点选择标准对照实验

## 本轮唯一研究改动

主要检查点改为：验证集 Fake/Real Macro-F1 最大，阈值固定为 0.55。
差异在 1e-12 内视为平分，保留较早轮次；不按 AI 指标或扫描阈值打破平分。
三个版本仍为 full、without_mvcm、missing_evidence，训练种子仍为 42、43、44。
保留原训练损失、优化器、学习率、批次顺序、最大轮数、模型结构、缓存与划分。

**早停仍按联合验证损失，patience 沿用原检查点，改善容差仍为 1e-6。**
因此本轮不是“改为按 F1 早停”，而是从原停止规则允许观察的轮次中按 F1 选模型。
新验证过程在同一次前向计算中额外汇总两个任务的 F1，不额外做一次每轮模型推理。

每次训练保存两个选择器的检查点，不做两次训练：

- `best_checkpoint.pt`：按验证 Fake/Real Macro-F1 选择。
- `loss_selected/best_checkpoint.pt`：同次训练按原联合验证损失选择。
- `selection_summary.json`：两种选择的轮次和实际训练轮数。
- `history.json`：原损失历史及两个任务每轮的验证 Macro-F1。

两个检查点均重新加载并验证损失；F1 检查点还校验选择分数是否复现。
两个诊断目录都只评估验证集。阈值扫描仍仅为附带输出，不用于本轮选择。
`within_run_selection` 给出同次训练切换选择标准的变化。
`paired_vs_without_mvcm` 给出新选择标准下模块对照差值。
这两种差值不是同一个问题；按验证 F1 选出的验证 F1 不下降属于选择规则本身的结果，
不是独立泛化证据。不要据此宣称测试集提升，也不要只展示最好的种子。

## 文件与兼容性

只需上传新增的 `run_checkpoint_selection.py` 和 `train_checkpoint_selection.py`。
不要重新上传或修改旧 `train_cat.py`、`model_cat.py`、`config.py` 等文件。
旧代码均未修改，以免破坏已完成实验的协议哈希。
没有新增依赖；继续使用既有 mvct 环境。训练需要 GPU，无需 OpenIE 或缓存重建。

默认读取：
`/root/autodl-tmp/MVCT_runtime/outputs/cat_v1/replications/missing_evidence_v1`

默认写入独立目录：
`/root/autodl-tmp/MVCT_runtime/outputs/cat_v1/replications/checkpoint_selection_v1`

启动前核对旧协议完整性、旧代码哈希、数据/划分哈希、参考配置、缓存签名、依赖版本以及九个完成记录。
新实验额外记录新文件哈希和 Python、Transformers、Tokenizers、scikit-learn 版本。
不一致时停止，不要通过编辑协议 JSON 或删除旧结果绕过检查。
模型权重文件只加载本项目可信的检查点。

## 云端操作

在项目目录准备环境：

```bash
cd /tmp/pycharm_project_173
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

/root/autodl-tmp/envs/mvct/bin/python -u run_checkpoint_selection.py --check-only
```

看到 `Preflight passed` 后运行一批（最多三个新实验）：

```bash
mkdir -p /root/autodl-tmp/MVCT_runtime/logs
nohup /root/autodl-tmp/envs/mvct/bin/python -u \
  run_checkpoint_selection.py --max-new-runs 3 \
  >> /root/autodl-tmp/MVCT_runtime/logs/checkpoint_selection.log 2>&1 &

tail -f /root/autodl-tmp/MVCT_runtime/logs/checkpoint_selection.log
```

不要同时启动多个实例。出现 `Invocation limit reached` 表示本批完成，
重复相同启动命令继续下一批，直到 `completed_runs` 为 9。
已完成实验会跳过；中断的未完成训练会创建新 attempt 从头训练，不做 epoch 续训，旧 attempt 保留。
因每个实验保留两个检查点，新增磁盘需求约为旧每批的两倍，启动前会按实际参考文件大小检查空间。
时间主要由重新训练决定，不能把这一步误认为只读评估。

查看汇总：

```bash
/root/autodl-tmp/envs/mvct/bin/python -m json.tool \
  /root/autodl-tmp/MVCT_runtime/outputs/cat_v1/replications/checkpoint_selection_v1/replication_summary.json

cat /root/autodl-tmp/MVCT_runtime/outputs/cat_v1/replications/checkpoint_selection_v1/replication_results.csv
```

## 本地验证

```text
python -m pytest tests/test_checkpoint_selection.py tests/test_repeated_validation.py tests/test_cat_core.py tests/test_validation_diagnostic.py tests/test_cache_audit.py -q
python run_checkpoint_selection.py --help
```

测试使用小型模型和模拟轮次，不代表云端九次训练已经完成，也不保证性能提高。

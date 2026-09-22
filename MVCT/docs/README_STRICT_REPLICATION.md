# 严格模式 MVCM 收益确认：6 次正式训练

这是一套新的验证集确认实验，不是重复短程检查，也不是最终测试集评估。
冻结旧训练函数、缓存、模型结构、划分及历史协议，不把新结果混入第一轮结果。
只有 `full` 和 `without_mvcm`，训练种子 42、43、44，顺序为每个种子先 full 再 without。

## 实验规则

- 从第一轮 `replications/missing_evidence_v1/protocol.json` 恢复配置；不拿本地 `config.py` 的训练参数替换云端历史配置。
- 复用原 `train_cat.train_model_cat`：按验证任务损失选择检查点，改进需超过 `1e-6`，保留原 patience 和训练轮数。
- Fake/Real 使用固定 `probability > 0.55`；报告两任务 Accuracy、Macro-F1。
- 每次独立进程从同一个本地预训练 BERT 初始化，而非从旧训练检查点接着训练。
- 两模型区别仍按第一轮定义：without_mvcm 的四通道权重为零；它保留原来的注意力 KL 正则。这不是移除全部 CAT 结构的消融。
- 不启用 missing-evidence 修正，不改损失、学习率、batch size 或 MVC 缓存。
- 严格环境在启动子进程前设置：Python hash seed 42、CUBLAS `:4096:8`、OMP/MKL/NUMEXPR 各 4；离线加载、tokenizer parallelism false。
- 所有训练子进程启用确定性算法，禁用 cuDNN benchmark 和 TF32；不支持确定性运算则报错，不静默降级。
- 测试集仅核对划分成员，不做预测或指标评估。旧诊断程序产生的阈值扫描只作附带记录，不参与本轮选择。
- 汇总所有预定种子，不挑最佳种子。均值、样本标准差以及配对差值不是显著性结论；通过诊断也不保证 MVCM 会提升。

## 上传：只需新增入口

把本地 `D:\PythonProject\MVCM_project\run_strict_replication.py` 上传到：

```text
/tmp/pycharm_project_173/run_strict_replication.py
```

可同时上传本说明。云端已有的 `runtime_guard.py`、`run_repro_check.py`、
`run_repeated_validation.py`、模型和训练等依赖继续使用，不要整项目覆盖上传。
不需要新增 Python 依赖，不需要启动 OpenIE，不重建缓存。

## 1. GPU 模式预检查

```bash
cd /tmp/pycharm_project_173
/root/autodl-tmp/envs/mvct/bin/python -u run_strict_replication.py --check-only
```

预检查核对第一轮 9 个完成记录、原协议完整性、旧源文件、原依赖版本（torch/numpy/pandas）、
数据 CSV 和三份划分。读取 BERT/分词器文件和全部训练/验证缓存的字节指纹。
不读取测试样本进入模型；不证明旧实验当时模型/缓存的字节与现在相同，因为旧协议未记录它们。

新增协议还记录当前依赖版本、CUDA/cuDNN、GPU UUID/驱动和严格设置。
后续换机器、换依赖、改代码、模型或缓存都会拒绝混入同一个新套件，不要手工改协议绕过。

成功输出类似：

```text
Preflight passed: strict, loss-selected, fixed 0.55; completed=0/6, pending=6.
Next runs: [(42, 'full'), (42, 'without_mvcm')]
Check only: no training or suite files written.
```

此命令会使用临时预检查文件及共享锁，但不创建实验套件、不训练、不生成检查点。
指纹读取会花一定时间；不是卡住就重复启动。

## 2. 先跑一对 seed 42

```bash
cd /tmp/pycharm_project_173
/root/autodl-tmp/envs/mvct/bin/python -u run_strict_replication.py --max-new-runs 2
```

默认一次最多 2 次训练，不是 2 个 epoch。运行日志会显示每次实际日志路径：

```text
Run full, seed=42; log: .../seed_42/full/attempt_时间戳/worker.log
```

详细训练进度写入对应 `worker.log`，主终端在该次训练结束前可能保持安静。
另开终端，复制实际打印的完整路径查看：

```bash
tail -f /完整实际路径/worker.log
```

`tail -f` 中按 Ctrl+C 只退出查看，不会停止训练；不要在训练终端误按。
先等这对完成，检查结果后继续种子 43、44，不因是否获益而挑选或删掉种子。

## 3. 续跑剩余种子

重复完全相同的命令：

```bash
cd /tmp/pycharm_project_173
/root/autodl-tmp/envs/mvct/bin/python -u run_strict_replication.py --max-new-runs 2
```

已完成且通过指纹和指标核验的实验自动跳过。通常三次启动分别完成种子 42、43、44。
若只剩一个未完成的种子内版本，配额按实际剩余的单次训练计算，不强制重新跑已完成版本。

- `Invocation limit reached`：本次额度用完，整套可能还没完成。
- `Replication completed`：六次全部完成。
- 中途报错：保留现场且停止，不自动反复重试，不生成该次 completed.json。
- 再次启动会保留未完成 attempt，创建新 attempt 从头训练。**不是从中断 epoch 自动恢复**。
- 如果恰好在模型已训练完但诊断/提交未完成时退出，也会重跑该次；先查看日志，避免不必要的成本。

不要手动删除运行锁。锁是协作式保护；旧脚本若绕过受保护入口在之后启动，仍可能并发。
运行期间不要上传修改代码或启动其他训练。Linux 子进程继承锁，主启动器意外结束时仍受保护。

## 可选：后台执行一对实验

前台/后台命令二选一，绝不要同时启动。

```bash
cd /tmp/pycharm_project_173
mkdir -p /root/autodl-tmp/MVCT_runtime/logs
nohup /root/autodl-tmp/envs/mvct/bin/python -u run_strict_replication.py --max-new-runs 2 >> /root/autodl-tmp/MVCT_runtime/logs/strict_loss_confirmation.log 2>&1 &
```

```bash
tail -n 30 /root/autodl-tmp/MVCT_runtime/logs/strict_loss_confirmation.log
```

主日志仍以启动/完成状态为主；训练批次进度在它给出的 worker.log。

## 结果和空间

默认新目录：

```text
/root/autodl-tmp/MVCT_runtime/outputs/cat_v1/replications/strict_loss_confirmation_v1
```

- `protocol.json`：冻结本轮方案、环境和输入指纹。
- `replication_summary.json`：完成数、各版本均值/样本 SD、逐种子配对差值。
- `replication_results.csv`：每次已核验结果。
- `seed_*/版本/completed.json`：完成身份、结果以及产物指纹。
- `seed_*/版本/attempt_*/`：配置、日志、单个最佳检查点、历史、验证报告和预测。

每次训练只保存一个最佳检查点，不保存 last 或逐 epoch 检查点。
按旧约 1.4 GiB/检查点估计，6 次检查点约 8.4 GiB；预检查按实际旧检查点大小和本次额度检查磁盘余量，并另留 1 GiB。
旧 attempt、模型和缓存不会自动删除。完整运行用时需以新机器第一对训练的实际耗时估计。

查看汇总（无需 GPU，可使用基础 Python）：

```bash
/root/miniconda3/bin/python -m json.tool /root/autodl-tmp/MVCT_runtime/outputs/cat_v1/replications/strict_loss_confirmation_v1/replication_summary.json
```

## 本地验证边界

```text
python -m pytest tests/test_strict_replication.py -q
```

本地测试验证调度、保护和汇总逻辑；没有本地真实 CUDA/BERT 全量训练。
云端预检查和第一对训练仍是必要验证，不应将单元测试通过当成实验成功。

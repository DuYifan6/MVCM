# 短程复现检查与运行锁

本次只新增入口，不修改冻结的训练代码、配置、缓存、协议、完成记录、历史结果。
请先保存已有结果；不要按分数挑选或删除 seed 43 的重复尝试。

## 上传文件

只需上传以下三个新文件到云端 `/tmp/pycharm_project_173/`：

- `run_repro_check.py`
- `runtime_guard.py`
- `run_locked.py`

继续使用现有 mvct 环境，无新增依赖。真实检查需要 GPU，不需启动 OpenIE，
不重建缓存，不生成测试集预测，不保存模型大检查点。

## 第一步：现有设置下的短程检查

```bash
cd /tmp/pycharm_project_173
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

/root/autodl-tmp/envs/mvct/bin/python -u run_repro_check.py --mode current --steps 10 --val-steps 2
```

该入口自动串行启动四个独立 Python 进程：old_a、old_b、new_a、new_b。
每个进程从相同 seed=42 初始化 full 模型，先训练 10 个批次，验证 2 个批次，
再按下一次数据加载迭代训练 10 个批次。总计 80 个训练批次和 8 个验证批次，
不是 9 个完整实验。保留原 batch size、优化器、学习率、损失和 DataLoader worker 数。
每一批更新后计算完整模型权重指纹，会有同步和哈希开销，不能直接按普通训练步速推算时间。

训练调用原 `train_cat.train_one_epoch`；旧验证调用 `evaluate_epoch`，
新验证调用 `evaluate_selection_epoch`，以观察新增 F1 统计及其后续训练是否改变结果。
此检查不复刻完整主入口、每轮 checkpoint 序列化及长程早停，因此不能单独证明整个历史实验可复现。
它也不恢复历史训练当时的随机数状态，而是检查当前环境下相同初始化过程是否可重复。

默认输出在数据盘：
`/root/autodl-tmp/MVCT_runtime/outputs/cat_v1/repro_checks/current_时间戳/`

- `plan.json`：检查范围、参数、新检查脚本哈希。
- `old_a/worker.log` 等：进程日志，包含短程步骤进度。
- `old_a/trace.json` 等：初始化权重及随机状态指纹，逐批样本 ID、输入与已使用缓存文件指纹、
  logits、任务损失、正则项、每步更新后的权重指纹，验证边界的随机数与优化器状态。
- `summary.json`：四组配对比较及第一个差异位置。

加载时核对旧协议完整性、数据/划分哈希、旧源文件哈希、缓存签名与划分成员。
另外记录当前 BERT/分词器文件内容指纹、当前依赖版本、GPU 身份和确定性相关选项。
**缓存内容指纹仅覆盖这次短程检查实际观察的样本；不是对整个缓存的全量验证。**
旧实验没有记录的模型/缓存指纹，不能事后证明历史文件与现在完全相同。

`--mode current` 保留继承的 cuBLAS/hash seed 设置及默认运算选择，只设置原入口已有的
cuDNN benchmark=False、deterministic=True，并记录实际环境。它不是历史硬件环境的恢复。

## 如何解读结果

- `trace_matches`：当前环境、被观察的短程执行痕迹完全一致。不是性能提高、模型正确或历史训练完全一致的证明。
- `differences_detected`：查看 `first_difference`。
  - `inventory/config/environment`：先核对文件、配置、运行环境。
  - `initial_weights`：初始化权重不同；报告首个不同参数名。
  - `initial_rng/rng_before`：随机状态不同。
  - `sample_ids/input_hash/cache_file_hashes`：顺序或输入内容不同。
  - `logits_hash/task_loss/attention_kl`：前向或损失计算已经不同。
  - `weights_after`：此前记录一致，但参数更新后不同，需要进一步查梯度与优化器。
  - `phase_end`：验证/训练阶段结束状态或阶段汇总不同。
- `worker_failed`：查看对应 `worker.log` / `error.json`，本次不能判定通过。

“第一个差异位置”是观测到的证据，不是自动确定的根因。
所有比较采用精确数值/字节指纹一致性；微小浮点差异也会标记，不等同于重大性能差异。
该工具插入的 CPU 读取、同步、哈希本身可能影响执行时序，因此结论仅限于被观测的执行。

## 第二步：必要时单独检查严格确定性

仅在分析第一步输出后再运行：

```bash
/root/autodl-tmp/envs/mvct/bin/python -u run_repro_check.py --mode strict --steps 10 --val-steps 2
```

strict 模式仅对子进程设置 PYTHONHASHSEED=42、CUBLAS_WORKSPACE_CONFIG=:4096:8，
启用 torch.use_deterministic_algorithms(True)，关闭 matmul/cuDNN TF32。
这是诊断控制条件，不应与原实验结果混合成同一方案。
若某个运算不支持确定性执行，会报错并保留日志，不会自动改模型或静默退回非确定性执行。

## 运行锁

短程检查自动取得同一运行锁，避免两个新检查或受保护的训练同时启动。
Linux 上还会检查已存在的旧脚本进程；发现时拒绝启动，不自动结束进程。
后续如需启动正式重复训练，使用下列入口代替直接运行旧脚本（当前先不要启动新训练）：

```bash
/root/autodl-tmp/envs/mvct/bin/python -u run_locked.py run_checkpoint_selection.py --max-new-runs 3
```

旧脚本文件和旧协议哈希不变。运行锁是协作式保护，**不能阻止有人随后绕过入口直接运行旧脚本**。
所有新的启动都应使用受保护入口，也不要在检查期间修改模型、缓存或代码。
锁文件保留属正常现象，退出后由系统释放锁，不要因为文件存在而删除它；
Linux 子进程继承锁句柄，父启动器退出时子进程未结束仍保持保护。
没有对旧完成记录进行恢复、挑选或重写；重复实验记录的修复仍需独立审计。

## 本地自测

```text
python -m pytest tests/test_repro_check.py -q
python run_repro_check.py --fixture --mode current --steps 2 --val-steps 1
```

`--fixture` 是 CPU 合成小模型自测；云端真实检查不要加此参数。
本地自测通过不表示真实 BERT/GPU 检查已经通过。

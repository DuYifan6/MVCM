# 跨服务器严格配对续跑

适用情况：seed 42 的 `full` 和 `without_mvcm` 已在旧服务器完成；新服务器接手 seed 43、44。
这不是把同一训练进程迁移到另一台机器，而是以**种子内成对对照**为单位继续实验：每个种子中，
full 和 without_mvcm 始终在同一服务器上训练；不同种子的服务器身份在最终汇总中保留。

不要将新服务器的 seed 43/44 直接写入旧服务器 `strict_loss_confirmation_v1`。旧目录的协议绑定了其
运行环境，强行混入会掩盖 GPU UUID 差异。新服务器使用独立目录，之后由只读汇总程序合并。

## 需要上传到新服务器 `_173`

- `run_strict_replication.py`：覆盖已有的同名脚本；这是唯一有意覆盖的脚本更新。
- `aggregate_crosshost_confirmation.py`：新文件，训练阶段暂不调用。

不上传 `config.py`，不重建缓存，不需要 OpenIE。

## 新服务器：seed 43、44 预检查与训练

```bash
cd /tmp/pycharm_project_173

/root/autodl-tmp/envs/mvct/bin/python -u run_strict_replication.py \
  --suite-dir /root/autodl-tmp/MVCT_runtime/outputs/cat_v1/replications/strict_loss_confirmation_host_b_v1 \
  --seeds 43 44 \
  --check-only
```

预检查通过后，以相同参数启动四次训练：

```bash
/root/autodl-tmp/envs/mvct/bin/python -u run_strict_replication.py \
  --suite-dir /root/autodl-tmp/MVCT_runtime/outputs/cat_v1/replications/strict_loss_confirmation_host_b_v1 \
  --seeds 43 44 \
  --max-new-runs 4
```

该运行依次执行 `(43, full)`、`(43, without_mvcm)`、`(44, full)`、`(44, without_mvcm)`。
目标目录是新目录；若该目录已经含有不匹配协议的文件，脚本会拒绝写入。中断 attempt 保留，完成记录才会被续跑跳过。

## 与旧服务器并行传输

训练新服务器种子 43/44 时，可从旧服务器传输已经稳定的 seed 42 内容到新服务器：

```text
strict_loss_confirmation_v1/protocol.json
strict_loss_confirmation_v1/seed_42/
```

旧目录的 `replication_summary.json` 和 `replication_results.csv` 在这里只有 2/6 的历史快照，
不参与最终跨主机汇总，传与不传均不影响训练。不要在新服务器执行旧目录中的训练命令。

## 两个源套件完成后的只读合并

```bash
cd /tmp/pycharm_project_173

/root/autodl-tmp/envs/mvct/bin/python -u aggregate_crosshost_confirmation.py \
  --host-a-suite /root/autodl-tmp/MVCT_runtime/outputs/cat_v1/replications/strict_loss_confirmation_v1 \
  --host-b-suite /root/autodl-tmp/MVCT_runtime/outputs/cat_v1/replications/strict_loss_confirmation_host_b_v1 \
  --output-dir /root/autodl-tmp/MVCT_runtime/outputs/cat_v1/replications/strict_loss_confirmation_crosshost_v1 \
  --check-only
```

预检查会核对两套实验的冻结数据、划分、缓存、预训练模型、参考配置、依赖、CUDA/cuDNN、驱动和严格设置。
它允许 GPU UUID 不同，但保留两个 UUID；若代码、数据、依赖、驱动或严格设置不同，会拒绝合并。

预检查通过后，去掉 `--check-only` 再执行一次，生成：

```text
strict_loss_confirmation_crosshost_v1/aggregate_protocol.json
strict_loss_confirmation_crosshost_v1/replication_results.csv
strict_loss_confirmation_crosshost_v1/replication_summary.json
```

最终结果是同一固定验证集上的 3 个种子配对比较；它仍不是最终测试集结果、显著性检验或跨硬件的独立外部验证。

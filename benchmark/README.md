# GLM-OCR 私有性能评测

本目录固定一套 12 页的小型 benchmark，用于比较同一张 RTX 4090 上不同 handler、推理引擎和并发参数的性能。数据覆盖 6 张真实图片和 6 个扫描文档页，包括中英日韩、公式、短文本和长截图。

二进制素材来自本机既有 OCR benchmark 语料，只允许发送到获授权的 Runpod 端点，不提交到公开仓库。`dataset/` 和 `results/` 已加入 `.gitignore`。

## 生成数据集

```bash
python3 benchmark/prepare_dataset.py
```

生成后的 `dataset/manifest.json` 记录每个输入的 SHA-256、人工或 PDF 文本 ground truth。只要哈希不变，各轮测试就在比较完全相同的输入。

## 运行

```bash
RUNPOD_API_KEY=... python3 benchmark/run_benchmark.py \
  --endpoint-id czu0b186rffoss \
  --variant baseline-ngram-c8 \
  --concurrency 8 \
  --rounds 3 \
  --max-image-side 1900 \
  --output benchmark/results/baseline-ngram-c8.json
```

使用 `--kind image` 或 `--kind document` 可独立测量图片或文档队列，避免两类请求混跑时互相争用 GPU，适合计算分类单页成本。

结果同时记录 Runpod `executionTime`、`delayTime`、客户端墙钟时间、吞吐、GPU execution-window 成本/页，以及与 ground truth 的字符顺序相似度。GPU execution window 完全由 Runpod 返回的 `delayTime + executionTime` 合并得到，不包含客户端轮询间隔和 worker 冷启动；默认按 Runpod 当前 RTX 4090 Serverless Flex 公示价 `$1.10/h` 换算。优化版本至少应满足：成功率不下降、质量指标无显著回退，并在 P50/P95、吞吐或单位成本上产生可重复收益。

# GLM-OCR / Runpod RTX 4090 部署评测（一页纸）

**结论：可以投入低流量试运行。** 保持单张 RTX 4090 24 GB、`workersMin=0`、`workersMax=1`，handler 并发设为 **8**，由现有任务队列把瞬时 10+ 用户请求排队并凑批；进入 Runpod 前将每页长边缩到 **1900 px**。这是本轮固定显卡条件下的最低稳态单页成本方案；如果将 P95 延迟优先于成本，则把 handler 并发降到 4。

## 决策数据

Runpod 当前公示 RTX 4090 Serverless 价格为 **$1.10/小时**，并按 worker 从启动到完全停止的秒数计费（[官方价格与计费口径](https://www.runpod.io/product/serverless)）。以下单价按用户要求，仅用 Runpod 返回的 `delayTime + executionTime` 合并 GPU execution window 换算，不含冷启动：

| 独立队列 | 正式测试 | Mean / P95 executionTime | 吞吐 | 稳态单价 | 每千页 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 图片 OCR | 6 个样本 × 3 轮，18/18 成功 | 2.043 s / 3.916 s | 1.294 页/s | **$0.000236/页** | **$0.236** |
| 文档 OCR | 6 个样本 × 3 轮，18/18 成功 | 5.400 s / 7.447 s | 0.768 页/s | **$0.000398/页** | **$0.398** |

并发阶梯的同口径结果：并发 2 为 2.560/4.694 秒、$0.000553/页；并发 4 为 2.580/4.697 秒、$0.000348/页；并发 8 为 3.839/6.242 秒、$0.000264/页（Mean/P95、混合队列）。**4 是延迟 Pareto 点，8 是成本/吞吐 Pareto 点。**

## 已验证的优化与否决项

- 依赖已固定为 `vllm/vllm-openai:v0.26.0`、Transformers 5.13.0、固定 GLM-OCR 源码与模型 revision，消除 nightly/main 漂移。
- ngram4 相对同镜像 ngram1 将 Mean/P95 约降低 6.8%，单位成本基本相同；保留 ngram4。
- 关闭 speculative decoding 虽启用 V2 Model Runner/异步调度，但两次冷启动均超过 Runpod 就绪窗口，新增 23 次失败；否决。
- `max_num_batched_tokens=8192` + `max_num_seqs=4` 使 Mean/P95/成本约恶化 29.6%/59.8%/16.7%；保留 vLLM 默认 batching。
- worker 内处理原图时，超长截图在 CPU 并发解码/缩放阶段耗时约 5.5 秒，图片 P95 升至 10.18 秒；1900 px 预处理必须前移到现有任务队列，handler 只保留兜底。

## 上线口径与风险

当前可用 HTTP 队列端点为 `POST https://api.runpod.ai/v2/czu0b186rffoss/run`（异步）或 `/runsync`。线上配置已留在 RTX 4090、并发 8、ngram4、默认 batching，队列为空且最终正式组全部成功。

稳态单价不是低流量真实账单：实测自定义镜像冷启动约 **90–120 秒**，按官方“worker 启动到停止”计费，每次冷启动约增加 **$0.0275–$0.0367/批**。10 页一批时仅冷启动就增加约 $0.00275–$0.00367/页；100 页一批降为 $0.000275–$0.000367/页。因此保持 scale-to-zero 的前提是任务队列凑批；若用户不能接受约两分钟首批等待，再评估 Active Worker，而不是继续微调 executionTime。

质量相似度：文档均值 0.9838、最低 0.9196；图片均值 0.8398，其中最低 0.2682 来自 ground truth 本身截断的超长截图，只可用于版本间回归比较，不代表绝对准确率。下一阶段应扩充人工校验 ground truth，再做按文档长度/语言分层的准确率验收。

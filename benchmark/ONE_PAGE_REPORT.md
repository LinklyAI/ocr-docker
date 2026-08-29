# GLM-OCR / Runpod RTX 4090 部署评测（一页纸）

**结论：自部署解决并发和可控性，不解决成本。** 对完全相同的 12 页、相同 1900 px 请求字节分别实测 3 轮后，智谱 GLM-OCR MaaS 的图片单价约为 Runpod 稳态单价的 **1/12**，文档单价约为 **1/7**。当前流量很低时，建议现有任务队列优先使用智谱的 2 并发额度；不要继续扩号池。Runpod 保留为可控容量方案，只有当并发、可用性或数据路径的价值高于 7–12 倍边际成本及冷启动成本时才启用。

## 同文件逐页实测

正式组共 36 个请求，智谱与 Runpod 均为 36/36 成功。每种内容 6 页 × 3 轮；12 个请求图 SHA-256 逐个一致。智谱按接口实际返回的 `total_tokens` 和官方输入输出同价 **0.2 元/百万 Tokens** 计算（[官方模型价格](https://docs.bigmodel.cn/cn/guide/models/vlm/glm-ocr)、[响应 usage 字段](https://docs.bigmodel.cn/api-reference/%E6%A8%A1%E5%9E%8B-api/%E6%96%87%E6%A1%A3%E8%A7%A3%E6%9E%90)）。Runpod 按 RTX 4090 Serverless **$1.10/小时**及 GPU execution window 计算（[官方价格](https://www.runpod.io/product/serverless)）；统一人民币使用 2026-08-28 美元兑人民币中间价 **6.7811**（[中国外汇交易中心授权数据](https://www.news.cn/20260828/8ded7ca354aa4c95b9a24a3cdd0b4fb8/c.html)）。

| 每页 | 智谱实际 Tokens | 智谱 MaaS | Runpod 稳态 | Runpod / 智谱 | 每千页：智谱 / Runpod |
| --- | ---: | ---: | ---: | ---: | ---: |
| 图片 OCR | 666.0 | **¥0.000133** | ¥0.001601（$0.000236） | **12.02×** | ¥0.133 / ¥1.601 |
| 文档 OCR | 1,926.3 | **¥0.000385** | ¥0.002699（$0.000398） | **7.00×** | ¥0.385 / ¥2.699 |

以上 Runpod 已是最有利的稳态口径，尚未计入实测 90–120 秒冷启动。每次 scale-to-zero 后重启约再付 ¥0.186–0.249：10 页一批会额外增加 ¥0.0186–0.0249/页，100 页一批增加 ¥0.00186–0.00249/页。因为 Runpod 的稳态边际成本本身已经更高，所以增加批量只能摊薄冷启动，不能反超智谱单价。

## 速度、质量与部署选择

| 类型 | 智谱墙钟 Mean / P95 | Runpod executionTime Mean / P95 | 智谱 / Runpod 相似度均值 |
| --- | ---: | ---: | ---: |
| 图片 | 1.441 / 3.661 秒 | 2.043 / 3.916 秒 | 0.6708 / 0.8398 |
| 文档 | 2.096 / 4.663 秒 | 5.400 / 7.447 秒 | 0.8739 / 0.9838 |

延迟口径不同：智谱是含网络的客户端墙钟，Runpod 是平台返回的纯执行时间，不能当作严格同口径性能排名；但可以确认智谱在 1–2 个并发下并不慢，问题是硬并发上限导致后续请求排队。相似度使用同一份 ground truth，但智谱布局解析 API 不开放自定义 prompt，而 Runpod 按样本使用 `Text Recognition:` / `Formula Recognition:`，因此质量值适合风险提示和版本回归，不是最终准确率裁决。

部署决策保持单张 RTX 4090 24 GB、handler 并发 **8**、任务队列预缩放长边到 **1900 px**、`workersMin=0`、`workersMax=1`。如果选择 Runpod，图片稳态吞吐为 1.294 页/秒，文档为 0.768 页/秒；混合并发阶梯中并发 4 是延迟 Pareto 点，并发 8 是吞吐/成本 Pareto 点。若需要无冷启动的 10 并发，必须接受 Active Worker 的持续账单；在当前低流量下不建议。

## 已验证的工程边界

- 依赖已固定为 `vllm/vllm-openai:v0.26.0`、Transformers 5.13.0、固定 GLM-OCR 源码与模型 revision，避免 nightly/main 漂移。
- 保留 ngram4 和 vLLM 默认 batching；关闭 speculative decoding以及手工限制 batch 均已实测失败或退化。
- 现有异步 HTTP 端点为 `POST https://api.runpod.ai/v2/czu0b186rffoss/run`；密钥只通过环境变量传入。
- 当前建议是“智谱低并发主路由 + Runpod 可控容量”，不是账号池。Runpod 冷启动较长，故应由任务队列按积压量提前预热，而不能把 scale-to-zero 当作即时故障切换。

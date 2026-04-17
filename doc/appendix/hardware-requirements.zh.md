# 附录 B: 计算资源与硬件需求

**[English](hardware-requirements.md)** | **[中文](hardware-requirements.zh.md)**

---

本附录提供 THETA 三个 Qwen3-Embedding 模型（0.6B/4B/8B）在不同硬件条件下的详细性能测试数据和使用建议。

## 执行摘要

### 关键发现

1. **GPU 显存占用**
   - 0.6B 模型：1.1-5.4 GB（所有配置均可运行）
   - 4B 模型：7.5-14.5 GB（所有配置均可运行）
   - 8B 模型：14.1-18.0 GB（batch_size=64 + seq_len=512 时 OOM）

2. **CPU vs GPU 性能对比**
   - CPU 推理速度：0.2-0.3 docs/s（**极慢，不推荐生产使用**）
   - GPU 推理速度：
     - 0.6B: 80-87 docs/s（**267-290倍于 CPU**）
     - 4B: 55-56 docs/s（**183倍于 CPU**）
     - 8B: 35 docs/s（**117倍于 CPU**）

3. **24GB GPU 可运行的最大配置**
   - 0.6B: batch_size=64, seq_len=512 ✅
   - 4B: batch_size=64, seq_len=512 ✅
   - 8B: batch_size=64, seq_len=256 ✅（seq_len=512 时 OOM）

4. **推荐生产配置**（留 20% 显存余量）
   - 0.6B: batch_size=32, seq_len=256（峰值显存 2.2 GB）
   - 4B: batch_size=32, seq_len=256（峰值显存 9.3 GB）
   - 8B: batch_size=16, seq_len=256（峰值显存 15.1 GB）

---

## B.1 模型显存占用参考表

固定 `seq_len=256`，展示各模型在不同 `batch_size` 下的峰值显存占用（GB）。

| Model Size | batch=1 | batch=4 | batch=8 | batch=16 | batch=32 | batch=64 |
|------------|---------|---------|---------|----------|----------|----------|
| **0.6B**   | 1.15 GB | 1.25 GB | 1.39 GB | 1.66 GB  | 2.20 GB  | 3.27 GB  |
| **4B**     | 7.61 GB | 7.79 GB | 7.98 GB | 8.42 GB  | 9.28 GB  | 11.01 GB |
| **8B**     | 14.16 GB| 14.35 GB| 14.59 GB| 15.07 GB | 16.04 GB | 17.98 GB |

### 显存占用规律

1. **模型本体显存**:
   - 0.6B: 1.12 GB
   - 4B: 7.55 GB (约 6.7倍于 0.6B)
   - 8B: 14.10 GB (约 1.9倍于 4B)

2. **显存增长规律**:
   - batch_size 翻倍 → 显存增加约 0.5-2 GB
   - seq_len 翻倍 → 显存增加约 0.3-1.5 GB
   - 显存占用 ≈ 模型本体 + batch_size × seq_len × 常数

---

## B.2 OOM 边界与安全配置

### 0.6B 模型

- **模型本体显存**: 1.12 GB
- **最大安全 batch_size**（seq_len=256）: 64（峰值显存 3.27 GB）
- **最大安全 seq_len**（batch_size=32）: 512（峰值显存 3.27 GB）
- **推荐生产配置**: batch_size=32, seq_len=256（峰值显存 2.20 GB，留 20% 余量）
- **OOM 情况**: 无 OOM（所有 18 个配置均通过）

### 4B 模型

- **模型本体显存**: 7.55 GB
- **最大安全 batch_size**（seq_len=256）: 64（峰值显存 11.01 GB）
- **最大安全 seq_len**（batch_size=32）: 512（峰值显存 11.01 GB）
- **推荐生产配置**: batch_size=32, seq_len=256（峰值显存 9.28 GB，留 20% 余量）
- **OOM 情况**: 无 OOM（所有 18 个配置均通过）

### 8B 模型

- **模型本体显存**: 14.10 GB
- **最大安全 batch_size**（seq_len=256）: 64（峰值显存 17.98 GB）
- **最大安全 seq_len**（batch_size=32）: 512（峰值显存 17.98 GB）
- **推荐生产配置**: batch_size=16, seq_len=256（峰值显存 15.07 GB，留 20% 余量）
- **OOM 情况**: 1 个 OOM（batch_size=64, seq_len=512 时超出 24GB 限制）

### OOM 边界总结

在 24GB GPU 上：
- **安全边界**: 峰值显存 < 19.2 GB（80% 利用率）
- **8B 模型 OOM 配置**: batch_size=64 + seq_len=512（预估需要 >24 GB）
- **避免 OOM 建议**: 
  - 8B 模型使用 batch_size ≤ 32 或 seq_len ≤ 256
  - 或使用梯度累积代替大 batch_size

---

## B.3 CPU vs GPU 决策指南

根据数据规模和模型大小选择合适的设备。

| 数据规模      | 0.6B 模型                       | 4B 模型        | 8B 模型        |
|--------------|--------------------------------|---------------|---------------|
| **1K docs**  | GPU 推荐 (CPU需67.5分钟)        | GPU (0.3分钟)  | GPU (0.5分钟)  |
| **5K docs**  | GPU 推荐 (CPU需314.6分钟)       | GPU (1.5分钟)  | GPU (2.4分钟)  |
| **10K docs** | GPU 推荐 (CPU需637.5分钟)       | GPU (3.0分钟)  | GPU (4.8分钟)  |
| **50K docs** | N/A                            | GPU (14.9分钟) | GPU (23.8分钟) |

### 设备选择建议

```
有 GPU (≥8GB)   → 强烈推荐 GPU (快 100-300 倍)
仅 CPU          → 仅用于 <100 docs 的测试
```

---

## B.4 实测性能汇总

综合所有测试结果的性能汇总表。

| Model | Device | n_docs | batch_size | docs/s | Peak Memory (GB) |
|-------|--------|--------|------------|--------|------------------|
| 0.6B  | CPU    | 100    | 32         | 0.2    | N/A              |
| 0.6B  | CPU    | 500    | 32         | 0.2    | N/A              |
| 0.6B  | CPU    | 1000   | 32         | 0.2    | N/A              |
| 0.6B  | CPU    | 2000   | 32         | 0.3    | N/A              |
| 0.6B  | CPU    | 5000   | 32         | 0.3    | N/A              |
| 0.6B  | CPU    | 10000  | 32         | 0.3    | N/A              |
| 0.6B  | GPU    | 1000   | 64         | 80.0   | 3.27             |
| 0.6B  | GPU    | 5000   | 64         | 85.8   | 3.27             |
| 0.6B  | GPU    | 10000  | 64         | 86.8   | 3.27             |
| 0.6B  | GPU    | 50000  | 64         | 87.0   | 3.27             |
| 4B    | GPU    | 1000   | 32         | 54.9   | 9.28             |
| 4B    | GPU    | 5000   | 32         | 55.9   | 9.28             |
| 4B    | GPU    | 10000  | 32         | 56.0   | 9.28             |
| 4B    | GPU    | 50000  | 32         | 56.0   | 9.28             |
| 8B    | GPU    | 1000   | 16         | 35.0   | 15.07            |
| 8B    | GPU    | 5000   | 16         | 35.0   | 15.07            |
| 8B    | GPU    | 10000  | 16         | 35.0   | 15.07            |
| 8B    | GPU    | 50000  | 16         | 35.0   | 15.07            |

---

## B.5 使用建议

### 选择模型规格

```
数据量 < 10K docs     → 0.6B (快速，低显存)
数据量 10K-50K docs   → 4B (平衡性能)
数据量 > 50K docs     → 8B (最高质量)
质量要求极高          → 8B (不论数据量)
```

### 配置 batch_size

```
显存充足 (>16GB)  → batch_size=32 或 64
显存有限 (8-16GB) → batch_size=16
显存紧张 (<8GB)   → batch_size=8 或使用梯度累积
```

### 不同显存容量的推荐配置

#### 8GB GPU
- 0.6B: batch_size=64 ✅
- 4B: batch_size=4-8 ⚠️
- 8B: 不推荐 ❌

#### 16GB GPU
- 0.6B: batch_size=64 ✅
- 4B: batch_size=32 ✅
- 8B: batch_size=8 ⚠️

#### 24GB GPU
- 0.6B: batch_size=64 ✅
- 4B: batch_size=64 ✅
- 8B: batch_size=32 ✅

---

## 测试环境

- **GPU**: NVIDIA GeForce RTX 3090, 24GB VRAM
- **CPU**: 多核 CPU（用于对比测试）
- **精度**: FP16 (GPU), FP32 (CPU)
- **框架**: PyTorch 2.7.0 + Transformers 5.3.0
- **模型来源**: ModelScope (Qwen/Qwen3-Embedding-{0.6B,4B,8B})
- **测试日期**: 2026-04-17

## 测试方法

### GPU 显存测试
- 测试矩阵: 3 模型 × 6 batch_sizes (1,4,8,16,32,64) × 3 seq_lens (128,256,512) = 54 个配置
- 每个配置运行一次前向传播，记录峰值显存
- 使用 `torch.cuda.max_memory_allocated()` 监控显存

### CPU/GPU 时间测试
- 每个配置运行 3 次，取后 2 次平均值（排除首次热身）
- 使用随机生成的文本数据
- CPU 测试使用所有可用核心
- GPU 测试使用推荐的 batch_size

## 关键结论

1. **强烈推荐使用 GPU**: GPU 推理速度是 CPU 的 **117-290 倍**
2. **24GB GPU 足够运行所有模型**: 除 8B 模型的极端配置外
3. **生产环境推荐配置**: 
   - 小规模任务 (<10K docs): 0.6B 模型, batch_size=32
   - 中等规模任务 (10K-50K docs): 4B 模型, batch_size=32
   - 大规模/高质量任务: 8B 模型, batch_size=16
4. **CPU 仅适用于**: 极小规模测试 (<100 docs) 或无 GPU 环境的离线处理

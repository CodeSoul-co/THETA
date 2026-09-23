# Appendix B: Hardware Requirements & Performance Benchmarks

---

This appendix provides detailed performance benchmarks and hardware requirements for THETA's three Qwen3-Embedding models (0.6B/4B/8B) across different hardware configurations.

## Executive Summary

### Key Findings

1. **GPU Memory Usage**
   - 0.6B model: 1.1-5.4 GB (all configurations runnable)
   - 4B model: 7.5-14.5 GB (all configurations runnable)
   - 8B model: 14.1-18.0 GB (OOM at batch_size=64 + seq_len=512)

2. **CPU vs GPU Performance**
   - CPU inference speed: 0.2-0.3 docs/s (**extremely slow, not recommended for production**)
   - GPU inference speed:
     - 0.6B: 80-87 docs/s (**267-290× faster than CPU**)
     - 4B: 55-56 docs/s (**183× faster than CPU**)
     - 8B: 35 docs/s (**117× faster than CPU**)

3. **Maximum Configurations on 24GB GPU**
   - 0.6B: batch_size=64, seq_len=512 (supported)
   - 4B: batch_size=64, seq_len=512 (supported)
   - 8B: batch_size=64, seq_len=256 (supported; OOM at seq_len=512)

4. **Recommended Production Configurations** (with 20% memory headroom)
   - 0.6B: batch_size=32, seq_len=256 (peak memory 2.2 GB)
   - 4B: batch_size=32, seq_len=256 (peak memory 9.3 GB)
   - 8B: batch_size=16, seq_len=256 (peak memory 15.1 GB)

---

## B.1 Model Memory Usage Reference

Peak GPU memory usage (GB) for different `batch_size` values at fixed `seq_len=256`.

| Model Size | batch=1 | batch=4 | batch=8 | batch=16 | batch=32 | batch=64 |
|------------|---------|---------|---------|----------|----------|----------|
| **0.6B**   | 1.15 GB | 1.25 GB | 1.39 GB | 1.66 GB  | 2.20 GB  | 3.27 GB  |
| **4B**     | 7.61 GB | 7.79 GB | 7.98 GB | 8.42 GB  | 9.28 GB  | 11.01 GB |
| **8B**     | 14.16 GB| 14.35 GB| 14.59 GB| 15.07 GB | 16.04 GB | 17.98 GB |

### Memory Usage Patterns

1. **Base Model Memory**:
   - 0.6B: 1.12 GB
   - 4B: 7.55 GB (~6.7× larger than 0.6B)
   - 8B: 14.10 GB (~1.9× larger than 4B)

2. **Memory Growth Rules**:
   - Doubling batch_size → +0.5-2 GB memory
   - Doubling seq_len → +0.3-1.5 GB memory
   - Memory usage ≈ Base model + batch_size × seq_len × constant

---

## B.2 OOM Boundaries & Safe Configurations

### 0.6B Model

- **Base model memory**: 1.12 GB
- **Max safe batch_size** (seq_len=256): 64 (peak 3.27 GB)
- **Max safe seq_len** (batch_size=32): 512 (peak 3.27 GB)
- **Recommended production config**: batch_size=32, seq_len=256 (peak 2.20 GB, 20% headroom)
- **OOM cases**: None (all 18 configurations passed)

### 4B Model

- **Base model memory**: 7.55 GB
- **Max safe batch_size** (seq_len=256): 64 (peak 11.01 GB)
- **Max safe seq_len** (batch_size=32): 512 (peak 11.01 GB)
- **Recommended production config**: batch_size=32, seq_len=256 (peak 9.28 GB, 20% headroom)
- **OOM cases**: None (all 18 configurations passed)

### 8B Model

- **Base model memory**: 14.10 GB
- **Max safe batch_size** (seq_len=256): 64 (peak 17.98 GB)
- **Max safe seq_len** (batch_size=32): 512 (peak 17.98 GB)
- **Recommended production config**: batch_size=16, seq_len=256 (peak 15.07 GB, 20% headroom)
- **OOM cases**: 1 OOM (batch_size=64, seq_len=512 exceeds 24GB limit)

### OOM Boundary Summary

On 24GB GPU:
- **Safe boundary**: Peak memory < 19.2 GB (80% utilization)
- **8B model OOM config**: batch_size=64 + seq_len=512 (requires >24 GB)
- **Avoiding OOM**: 
  - Use batch_size ≤ 32 or seq_len ≤ 256 for 8B model
  - Or use gradient accumulation instead of large batch_size

---

## B.3 CPU vs GPU Decision Guide

Choose appropriate device based on data scale and model size.

| Data Scale   | 0.6B Model                     | 4B Model       | 8B Model       |
|--------------|--------------------------------|----------------|----------------|
| **1K docs**  | GPU recommended (CPU: 67.5min) | GPU (0.3min)   | GPU (0.5min)   |
| **5K docs**  | GPU recommended (CPU: 314.6min)| GPU (1.5min)   | GPU (2.4min)   |
| **10K docs** | GPU recommended (CPU: 637.5min)| GPU (3.0min)   | GPU (4.8min)   |
| **50K docs** | N/A                            | GPU (14.9min)  | GPU (23.8min)  |

### Device Selection Guidelines

```
Have GPU (≥8GB)   → Strongly recommend GPU (100-300× faster)
CPU only          → Only for <100 docs testing
```

---

## B.4 Performance Summary

Comprehensive performance summary across all tested configurations.

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

## B.5 Usage Recommendations

### Choosing Model Size

```
Data < 10K docs       → 0.6B (fast, low memory)
Data 10K-50K docs     → 4B (balanced performance)
Data > 50K docs       → 8B (highest quality)
High quality required → 8B (regardless of data size)
```

### Configuring batch_size

```
Abundant memory (>16GB) → batch_size=32 or 64
Limited memory (8-16GB) → batch_size=16
Tight memory (<8GB)     → batch_size=8 or use gradient accumulation
```

### Recommended Configurations by GPU Memory

#### 8GB GPU
- 0.6B: batch_size=64 (supported)
- 4B: batch_size=4-8 (limited)
- 8B: Not recommended

#### 16GB GPU
- 0.6B: batch_size=64 (supported)
- 4B: batch_size=32 (supported)
- 8B: batch_size=8 (limited)

#### 24GB GPU
- 0.6B: batch_size=64 (supported)
- 4B: batch_size=64 (supported)
- 8B: batch_size=32 (supported)

---

## Test Environment

- **GPU**: NVIDIA GeForce RTX 3090, 24GB VRAM
- **CPU**: Multi-core CPU (for comparison)
- **Precision**: FP16 (GPU), FP32 (CPU)
- **Framework**: PyTorch 2.7.0 + Transformers 5.3.0
- **Model Source**: ModelScope (Qwen/Qwen3-Embedding-{0.6B,4B,8B})
- **Test Date**: April 17, 2026

## Test Methodology

### GPU Memory Testing
- Test matrix: 3 models × 6 batch_sizes (1,4,8,16,32,64) × 3 seq_lens (128,256,512) = 54 configurations
- Each configuration runs one forward pass, recording peak memory
- Memory monitored using `torch.cuda.max_memory_allocated()`

### CPU/GPU Time Testing
- Each configuration runs 3 times, averaging last 2 runs (excluding warmup)
- Uses randomly generated text data
- CPU tests use all available cores
- GPU tests use recommended batch_size

## Key Conclusions

1. **Strongly recommend GPU**: GPU inference is **117-290× faster than CPU**
2. **24GB GPU sufficient for all models**: Except extreme 8B configurations
3. **Production recommendations**: 
   - Small tasks (<10K docs): 0.6B model, batch_size=32
   - Medium tasks (10K-50K docs): 4B model, batch_size=32
   - Large/high-quality tasks: 8B model, batch_size=16
4. **CPU only suitable for**: Very small tests (<100 docs) or offline processing without GPU

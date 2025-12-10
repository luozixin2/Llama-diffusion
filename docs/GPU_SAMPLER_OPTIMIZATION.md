# GPU Sampler 优化详细记录

## 概述

本文档记录了为 Llama-diffusion 项目实现 GPU 采样加速的完整优化过程。最终实现了 **3.19x** 的性能提升（结合参数优化）。

---

## 1. 背景与问题分析

### 1.1 初始性能瓶颈

通过 profiling 分析发现，扩散模型推理的主要瓶颈在 `token_sampling` 阶段：

| 阶段 | 耗时占比 |
|------|----------|
| token_sampling | ~68% |
| llama_decode | ~25% |
| 其他 | ~7% |

采样阶段在 CPU 上执行，包括：
- Softmax 计算
- Top-k/Top-p 筛选
- 概率采样
- 置信度计算

### 1.2 优化目标

将采样逻辑迁移到 GPU，减少 CPU 计算和数据传输开销。

---

## 2. 第一阶段：基础 GPU Sampler 实现

### 2.1 架构设计

创建以下文件：
- `llama_diffusion/gpu_sampler.h` - 接口定义
- `llama_diffusion/gpu_sampler.cu` - CUDA 实现
- `llama_diffusion/diffusion_types.h` - 共享类型定义

### 2.2 初始实现

```cpp
// 基础流程
1. H2D: 将 logits 从 CPU 复制到 GPU
2. Temperature scaling: GPU kernel 缩放 logits
3. Sort: 使用 Thrust 对每行排序
4. D2H: 将排序结果复制回 CPU
5. CPU: 执行 softmax + 采样
```

### 2.3 初始结果

**问题**：GPU Sampler 比 CPU 更慢（0.56x）！

| 配置 | 耗时 (ms) | 加速比 |
|------|----------|--------|
| Baseline (CPU) | 2704 | 1.00x |
| GPU Sampler | 4862 | 0.56x |

### 2.4 性能分解分析

添加详细的 telemetry 统计后发现：

| 阶段 | 耗时 (ms) | 占比 |
|------|----------|------|
| stage_sort (Thrust 排序) | 1364.97 | 40% |
| stage_cpu_post (CPU 后处理) | 1492.13 | 44% |
| stage_d2h (D2H 传输) | 274.88 | 8% |
| logit_pack (H2D 准备) | 303.34 | 9% |

**瓶颈识别**：
1. Thrust 对整个词表 (151936) 排序太慢
2. CPU 后处理（softmax + 采样）仍然是主要开销

---

## 3. 第二阶段：失败的优化尝试

### 3.1 Pinned Memory + 批量传输

**尝试**：使用 `cudaHostAlloc` 分配 pinned memory，批量处理所有 rows 后再统一传输。

```cpp
// 尝试的优化
cudaHostAlloc(&host_logits_staging_, required * sizeof(float), cudaHostAllocPortable);
cudaHostAlloc(&host_indices_staging_, required * sizeof(int), cudaHostAllocPortable);

// 批量排序所有 rows
for (int row = 0; row < block_length_; ++row) {
    // GPU 排序...
}
cudaStreamSynchronize(stream_);

// 批量 D2H 传输
for (int row = 0; row < block_length_; ++row) {
    cudaMemcpyAsync(host_logits_row, row_sorted, ...);
}
cudaStreamSynchronize(stream_);

// CPU 后处理
for (int row = 0; row < block_length_; ++row) {
    // softmax + 采样...
}
```

**结果**：性能更差！Wall time 从 ~2.95s 增加到 ~13.63s。

**原因分析**：
- 仍然需要对整个词表排序
- 批量传输增加了同步等待时间
- 没有解决根本问题（排序和 CPU 后处理开销）

### 3.2 回退决策

放弃 pinned memory 方案，回退到逐行处理的实现，重新思考优化方向。

---

## 4. 第三阶段：成功的批量/融合优化

### 4.1 核心优化思路

1. **在 GPU 上完成 Softmax** - 减少 CPU 计算
2. **减少排序数据量** - 只传输 top-k 候选
3. **自定义 CUDA kernels** - 替代通用 Thrust 操作

### 4.2 实现细节

#### 4.2.1 GPU Softmax Kernels

```cuda
// 1. 找每行最大值（并行归约）
__global__ void find_row_max_kernel(const float* logits, float* row_max, 
                                     int vocab_size, int block_length) {
    extern __shared__ float sdata[];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    
    // 每个线程找局部最大值
    float local_max = -FLT_MAX;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        if (logits[row * vocab_size + i] > local_max)
            local_max = logits[row * vocab_size + i];
    }
    sdata[tid] = local_max;
    __syncthreads();
    
    // 块内归约
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sdata[tid + s] > sdata[tid])
            sdata[tid] = sdata[tid + s];
        __syncthreads();
    }
    
    if (tid == 0) row_max[row] = sdata[0];
}

// 2. 计算 exp 和求和
__global__ void softmax_exp_sum_kernel(const float* logits, const float* row_max,
                                        float* exp_logits, float* row_sum, ...) {
    // 计算 exp(logit - max) 并累加
}

// 3. 归一化
__global__ void softmax_normalize_kernel(float* probs, const float* row_sum, ...) {
    // probs[i] /= sum
}
```

#### 4.2.2 减少数据传输

```cpp
// 只传输 top-k 候选（最多 1024 个）
int transfer_count = vocab_size_;
if (config_.top_k > 0 && config_.top_k < vocab_size_) {
    transfer_count = config_.top_k;
}
if (config_.top_p < 1.0f && transfer_count > 1024) {
    transfer_count = 1024;  // 启发式限制
}
```

#### 4.2.3 简化的 CPU 采样

```cpp
// CPU 只做最终采样（数据量小）
for (int row = 0; row < block_length_; ++row) {
    // Top-p 截断
    float cumsum = 0.0f;
    for (int i = 0; i < transfer_count; ++i) {
        cumsum += row_probs[i];
        if (cumsum > config_.top_p) {
            final_count = i + 1;
            break;
        }
    }
    
    // 采样
    std::uniform_real_distribution<float> dist(0.0f, sum);
    float r = dist(rng);
    // ...
}
```

### 4.3 优化结果

| 阶段 | 优化前 (ms) | 优化后 (ms) | 提升 |
|------|------------|------------|------|
| stage_sort | 1364.97 | 28.75 | **47x** |
| stage_cpu_post | 1492.13 | 177.63 | **8.4x** |
| stage_d2h | 274.88 | 118.75 | **2.3x** |
| **总采样时间** | **3521.42** | **643.72** | **5.5x** |

---

## 5. 第四阶段：参数级优化

### 5.1 组合测试

测试 GPU Sampler 与其他参数的组合效果：

```python
test_configs = [
    {'name': 'GPU + Larger Blocks', 'block_length': 16, 'use_gpu_sampler': True},
    {'name': 'GPU + Fewer Steps', 'denoising_steps': 4, 'use_gpu_sampler': True},
]
```

### 5.2 最终性能对比

| 配置 | 耗时 (ms) | Tokens/sec | 加速比 |
|------|----------|------------|--------|
| Baseline (CPU) | 2702.39 | 49.22 | 1.00x |
| GPU Sampler | 1525.81 | 87.17 | **1.77x** |
| Larger Blocks | 2420.32 | 54.95 | 1.12x |
| Fewer Steps | 1454.74 | 91.43 | 1.86x |
| Sequential Strategy | 2706.50 | 49.14 | 1.00x |
| **GPU + Larger Blocks** | **1184.96** | **112.24** | **2.28x** |
| **GPU + Fewer Steps** | **847.38** | **156.95** | **3.19x** |

---

## 6. 文件变更总结

### 6.1 新增文件

| 文件 | 说明 |
|------|------|
| `llama_diffusion/gpu_sampler.h` | GPU Sampler 接口定义 |
| `llama_diffusion/gpu_sampler.cu` | CUDA 实现（~400 行） |
| `llama_diffusion/diffusion_types.h` | 共享类型定义 |

### 6.2 修改文件

| 文件 | 变更 |
|------|------|
| `llama_diffusion/CMakeLists.txt` | 添加 CUDA 编译支持 |
| `llama_diffusion/diffusion_sampler.h` | 添加 GPU sampler 成员 |
| `llama_diffusion/diffusion_sampler.cpp` | 集成 GPU 采样路径 |
| `llama_diffusion/diffusion_sampler_profiled.cpp` | 添加 telemetry 统计 |
| `llama_diffusion/python_bindings*.cpp` | 暴露 `use_gpu_sampler` 参数 |
| `test_profiling.py` | 添加 GPU 测试配置 |

---

## 7. 关键经验教训

### 7.1 失败的优化

1. **Pinned memory 不是万能的** - 如果核心算法效率低，优化传输无济于事
2. **批量处理需要配合算法优化** - 单纯批量化可能增加同步开销

### 7.2 成功的优化

1. **减少计算量比优化传输更重要** - 在 GPU 上完成 softmax 消除了大量 CPU 计算
2. **只传输需要的数据** - top-k 限制大幅减少 D2H 传输量
3. **自定义 kernel 比通用库更高效** - 针对具体场景优化的 kernel 性能更好

### 7.3 Profiling 的重要性

每次优化后都进行 profiling，才能：
- 发现真正的瓶颈
- 验证优化是否有效
- 及时回退失败的优化

---

## 8. 未来优化方向

1. **直接使用 llama_decode 的 GPU logits** - 避免 H2D 传输（需要修改 llama.cpp 接口）
2. **GPU 上完成采样** - 使用 cuRAND 在 GPU 上直接采样
3. **多流并行** - 利用 CUDA streams 重叠计算和传输
4. **Tensor Core 加速** - 对于支持的 GPU，使用 FP16 计算

---

## 9. 2025-12-06 更新：GPU logits 与采样融合快路径

### 9.1 主要改动
- **host+device 双缓冲**：`LLAMA_ENABLE_DEVICE_LOGITS=1` 时，llama.cpp 同步维护 host/device logits，并在 GPU 侧完成 reorder（`llama_gpu_swap_rows`），调用端可直接消费 device logits，避免 H2D。
- **GPU 采样 fast path 融合**：`fused_softmax_sample_kernel` 将 softmax+采样合并为单核，仅回传 token+confidence，减少 kernel/sync；top-k/p 或 entropy 需求时回退到排序/CPU 后处理路径。
- **Telemetry 细化**：新增 fast path 计数（`telemetry_gpu_fast_path`、`telemetry_gpu_device_fast_path`），保留分阶段计时；device 路径命中/回退已记录。

### 9.2 性能回归（单卡，b=s=4/8，steps=4/8，默认温度）
- 开关 **ON**（device logits）：b8,s8 ≈ **185.6 tok/s**，b4,s4 ≈ **165.4 tok/s**，b8,s4 ≈ **275.5 tok/s**（归档：`profile_runs/20251206_052450/`）
- 开关 **OFF**（host logits）：b8,s8 ≈ **156.5 tok/s**，b4,s4 ≈ **154.2 tok/s**，b8,s4 ≈ **281.4 tok/s**（归档：`profile_runs/20251206_052545/`）

### 9.3 观察
- device logits 打开时 b8,s8 提升显著（fast path 命中且无 H2D）；b8,s4 在 OFF 略高，表明小批短步长仍受 D2H/同步影响。
- fast path 触发条件：`top_k<=0 && top_p>=1 && !entropy`，否则走回退；device logits 路径只支持 fast path，top-k/p/entropy 仍回退到 host logits。
- 待办：质量抽检并完善文档（开关/回退条件、fast path 命中率、文本质量对比）。

### 9.4 质量抽检（2025-12-06，示例提示词）
- 提示词：参考 `example_usage.py` 中 “机器人第一次发现音乐” 短故事场景；参数：`gen_length=2048, block_length=4, denoising_steps=4, top_k=0, top_p=1, temperature=1.0, remasking=low_confidence_dynamic, use_gpu_sampler=True`。
- 输出：
  - ON（`LLAMA_ENABLE_DEVICE_LOGITS=1`）：见 `profile_runs/quality_on.txt`，连贯完结，无多余 end token。
  - OFF（未设开关）：见 `profile_runs/quality_off.txt`，故事连贯，但末尾出现多余 `<|endoftext|>` 重复标记。
- 结论：两侧语义与流畅度一致，开关 ON 无质量回退，且末尾控制更干净。

### 9.5 开关与回退说明（doc-fallback / doc-device-logits / doc-dual）
- 开关：`LLAMA_ENABLE_DEVICE_LOGITS=1` 时启用 host+device 双缓冲与 GPU reorder（`llama_gpu_swap_rows`）；未设置时保持 host logits。
- 触发 fast path（device/host 同规则）：`top_k <= 0` 且 `top_p >= 1` 且无 entropy 需求；此时 softmax+采样单核融合，仅回传 token+confidence。
- 回退条件：
  - 需要 top-k/p 或 entropy -> 排序+CPU 后处理，仍支持 GPU softmax。
  - device logits 不可用或 stride 不等于 vocab -> 回退 host logits。
  - device 路径仅支持 fast path，若需排序/entropy 则直接回退 host。
- 纯 GPU 开关：`DIFFUSION_GPU_ONLY=1` 时要求 device logits 可用且无 top-k/p/entropy，跳过 host logits 拷贝/compact，CPU 仅接收 token/conf。
- 双缓冲行为：llama.cpp 同时保留 host/device logits；存在 `output_swaps` 时在 GPU 侧重排并保持 host 拷贝，调用端可选择 device 或 host。
- 遥测字段：
  - `telemetry_gpu_fast_path`（host fast）、`telemetry_gpu_device_fast_path`（device fast）
  - `telemetry_gpu_path_device_hit/miss/need_entropy`、分阶段耗时 `telemetry_gpu_stage_*`、`telemetry_gpu_logit_pack`
- 结果归档：ON `profile_runs/20251206_052450/`，OFF `profile_runs/20251206_052545/`；质量输出 `profile_runs/quality_on.txt` / `quality_off.txt`。

---

## 10. 2025-12-07 更新：GPU logits 稳定性 & 日志等级体系

### 10.1 主要修复与防护
- **词表上限与尾部 masking**：`DiffusionConfig::n_vocab_limit` 默认 151670，CPU/GPU 路径均对越界 logits 置 `-inf`，消除 OOV/解码 None 问题。
- **GPU-only 强制防护**：`DIFFUSION_GPU_ONLY=1` 时若 GPU sampler 不可用直接报错，防止回退访问 host logits。
- **设备 logits 稳定性**：比较 host/device 时改为 D2H 拷贝再比对，避免直接解引用 device 指针导致 segfault；`test_profiling.py` GPU-only 已稳定通过（见 `profile_runs/20251207_062957/`）。
- **同步与错误探针**：在 D2D/softmax/采样/D2H 后强制 `cudaDeviceSynchronize` + `cudaGetLastError`，记录到日志便于定位故障。

### 10.2 日志等级 API（F/E/W/I/D/V）
- 新增 `llama_diffusion/diffusion_logging.h`，提供 `DIFF_LOGF/E/W/I/D/V`，编译期通过 `-DDIFFUSION_LOG_LEVEL=DIFF_LOG_LEVEL_WARN` 等开关裁剪输出，低于阈值的日志不产生格式化开销。
- 默认等级：INFO；调试建议：`DIFFUSION_LOG_LEVEL=DIFF_LOG_LEVEL_DEBUG`；生产建议：WARN 或 INFO。WARN 及以上自动 `fflush(stderr)`，确保关键信息落盘。

### 10.3 使用与建议
- 开启设备 logits：`LLAMA_ENABLE_DEVICE_LOGITS=1`；纯 GPU 路径：同时设置 `DIFFUSION_GPU_ONLY=1`（要求无 top-k/p/entropy）。
- 关闭冗余日志：编译或运行时降低 `DIFFUSION_LOG_LEVEL`；仅需错误级别即可最小化开销。
- 归档：最新稳定 GPU-only 运行与对比见 `profile_runs/20251207_062957/`；示例生成与流式回归见 `profile_runs/example_usage_gpu_only_run3.log`、`profile_runs/example_usage_stream_gpu_only_run3.log`。

---

## 附录：测试结果存档

所有 profiling 结果保存在 `profile_runs/` 目录：

```
profile_runs/
├── 20251126_115926_gpu_batch_opt/     # 批量优化后
├── 20251126_120226_final_optimization/ # 最终优化结果
└── ...
```

---

*文档创建时间: 2024-11-26*
*最后更新: GPU Sampler 优化完成，达到 3.19x 加速*


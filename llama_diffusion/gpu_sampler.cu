#include "gpu_sampler.h"
#include "diffusion_logging.h"
#include "diffusion_profiler.h"

#if defined(DIFFUSION_ENABLE_CUDA)

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <cfloat>
#include <algorithm>

namespace diffusion {

namespace {

// Number of streams for parallel processing
constexpr int NUM_STREAMS = 4;

// Scale logits by temperature for a single row
__global__ void scale_logits_row_kernel(float* logits, float inv_temp, int vocab_size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vocab_size) {
        logits[idx] *= inv_temp;
    }
}

// Batched kernels (for fallback single-stream mode)
__global__ void scale_logits_kernel(float* logits, float inv_temp, size_t size) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        logits[idx] *= inv_temp;
    }
}

// Mask tail logits (ids >= offset) to -inf across all rows
__global__ void mask_tail_kernel(float* logits, int vocab_size, int block_length, int offset) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    if (row >= block_length) return;
    const int base = row * vocab_size;
    for (int vid = offset + tid; vid < vocab_size; vid += stride) {
        logits[base + vid] = -INFINITY;
    }
}

__global__ void find_row_max_kernel(const float* logits, float* row_max, int vocab_size, int block_length) {
    extern __shared__ float sdata[];
    
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    
    if (row >= block_length) return;
    
    const float* row_logits = logits + row * vocab_size;
    
    float local_max = -FLT_MAX;
    for (int i = tid; i < vocab_size; i += stride) {
        float val = row_logits[i];
        if (val > local_max) local_max = val;
    }
    sdata[tid] = local_max;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sdata[tid + s] > sdata[tid]) {
            sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        row_max[row] = sdata[0];
    }
}

__global__ void softmax_exp_sum_kernel(const float* logits, const float* row_max, 
                                        float* exp_logits, float* row_sum,
                                        int vocab_size, int block_length) {
    extern __shared__ float sdata[];
    
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    
    if (row >= block_length) return;
    
    const float* row_logits = logits + row * vocab_size;
    float* row_exp = exp_logits + row * vocab_size;
    const float max_val = row_max[row];
    
    float local_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += stride) {
        float val = row_logits[i];
        float e = (val > -FLT_MAX + 100.0f) ? expf(val - max_val) : 0.0f;
        row_exp[i] = e;
        local_sum += e;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        row_sum[row] = sdata[0];
    }
}

__global__ void softmax_normalize_kernel(float* probs, const float* row_sum, 
                                          int vocab_size, int block_length) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    
    if (row >= block_length) return;
    
    float* row_probs = probs + row * vocab_size;
    const float inv_sum = 1.0f / (row_sum[row] + 1e-10f);
    
    for (int i = tid; i < vocab_size; i += stride) {
        row_probs[i] *= inv_sum;
    }
}

// Fill random values on GPU
__global__ void fill_random_kernel(float* out, uint64_t seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, 0, &state);
    out[idx] = curand_uniform(&state);
}

// Fused softmax + sampling for fast path (no top-k/p, no probs)
// One block per row; supports block_length <= gridDim.x
__global__ void fused_softmax_sample_kernel(
    const float* logits,       // [block_length, vocab_size]
    int vocab_size,
    int block_length,
    float inv_temp,            // 1.0f when no temperature
    const float* random_vals,  // [block_length] pre-generated uniform(0,1]
    int* sampled_tokens,       // [block_length]
    float* confidences         // [block_length]
) {
    const int row = blockIdx.x;
    if (row >= block_length) return;

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const float* row_logits = logits + row * vocab_size;

    // shared layout: [blockDim] max/tmp | [blockDim] chunk_sums | [1] prefix_before | [1] target_chunk (int)
    extern __shared__ unsigned char shared_raw[];
    float* sdata = reinterpret_cast<float*>(shared_raw);                    // blockDim
    float* chunk_sums = sdata + nthreads;                                   // blockDim
    float* prefix_before = chunk_sums + nthreads;                           // 1 float
    int* target_chunk_ptr = reinterpret_cast<int*>(prefix_before + 1);      // 1 int

    // 1) reduce max
    float local_max = -FLT_MAX;
    for (int i = tid; i < vocab_size; i += nthreads) {
        float val = row_logits[i] * inv_temp;
        if (val > local_max) local_max = val;
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s && sdata[tid + s] > sdata[tid]) {
            sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }
    const float row_max = sdata[0];

    // 2) reduce sum of exp(logit - max)
    float local_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += nthreads) {
        float e = expf(row_logits[i] * inv_temp - row_max);
        local_sum += e;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    const float row_sum = sdata[0] + 1e-10f;

    // 3) random threshold in [0, row_sum)
    if (tid == 0) {
        const float r = random_vals[row] * row_sum;
        prefix_before[0] = r;
        target_chunk_ptr[0] = -1;
    }
    __syncthreads();
    const float target_r = prefix_before[0];

    // 4) chunk sums
    const int chunk_size = (vocab_size + nthreads - 1) / nthreads;
    const int start_idx = tid * chunk_size;
    const int end_idx = min(start_idx + chunk_size, vocab_size);
    float chunk_sum = 0.0f;
    for (int i = start_idx; i < end_idx; ++i) {
        chunk_sum += expf(row_logits[i] * inv_temp - row_max);
    }
    chunk_sums[tid] = chunk_sum;
    __syncthreads();

    // 5) find target chunk
    if (tid == 0) {
        float prefix = 0.0f;
        float prefix_before_val = 0.0f;
        int tgt = -1;
        for (int c = 0; c < nthreads; ++c) {
            float old_prefix = prefix;
            prefix += chunk_sums[c];
            if (tgt < 0 && prefix >= target_r) {
                tgt = c;
                prefix_before_val = old_prefix; // reuse to store prefix_before target
            }
        }
        if (tgt < 0) {
            tgt = nthreads - 1;
            prefix_before_val = prefix - chunk_sums[tgt]; // avoid uninitialized prefix_before
        }
        prefix_before[0] = prefix_before_val;
        target_chunk_ptr[0] = tgt;
    }
    __syncthreads();

    const int target_chunk = target_chunk_ptr[0];
    if (tid == target_chunk) {
        float prefix = prefix_before[0]; // absolute mass before this chunk
        int sampled_idx = vocab_size - 1;
        float sampled_prob = 0.0f;
        for (int i = start_idx; i < end_idx; ++i) {
            float e = expf(row_logits[i] * inv_temp - row_max); // unnormalized mass
            prefix += e;
            if (prefix >= target_r) {
                sampled_idx = i;
                sampled_prob = e / row_sum; // normalized prob
                break;
            }
        }
        sampled_tokens[row] = sampled_idx;
        confidences[row] = sampled_prob;
    }
}

// Phase 2 优化: GPU 端并行采样 kernel（可用于融合路径）
// 使用分块并行累加 + warp-level 规约
__global__ void sample_tokens_kernel(
    const float* probs,           // [block_length, vocab_size] normalized probabilities
    const float* random_vals,     // [block_length] random values in [0, 1)
    int* sampled_tokens,          // [block_length] output token ids
    float* confidences,           // [block_length] output confidences
    int vocab_size,
    int block_length,
    int top_k,                    // 0 means disabled
    float top_p                   // 1.0 means disabled
) {
    // 每个 block 处理一个位置的采样
    // 使用并行分块策略：每个线程处理一个 chunk，找到第一个累积和超过阈值的 chunk
    extern __shared__ char shared_mem[];
    float* chunk_sums = (float*)shared_mem;           // [blockDim.x]
    int* chunk_first_idx = (int*)(chunk_sums + blockDim.x);  // [blockDim.x]
    
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    
    if (row >= block_length) return;
    
    const float* row_probs = probs + row * vocab_size;
    const float r = random_vals[row];
    
    // 每个线程处理的 chunk 大小
    const int chunk_size = (vocab_size + num_threads - 1) / num_threads;
    const int start_idx = tid * chunk_size;
    const int end_idx = min(start_idx + chunk_size, vocab_size);
    
    // Step 1: 每个线程计算其 chunk 的累积和和第一个超过阈值的局部索引
    float local_sum = 0.0f;
    int local_first_idx = -1;  // -1 表示此 chunk 中没有找到
    
    for (int i = start_idx; i < end_idx; ++i) {
        local_sum += row_probs[i];
    }
    
    chunk_sums[tid] = local_sum;
    chunk_first_idx[tid] = -1;
    __syncthreads();
    
    // Step 2: 计算 prefix sum 来确定哪个 chunk 包含目标
    // 使用简单的串行 prefix sum（线程数通常很小，256或512）
    if (tid == 0) {
        float prefix = 0.0f;
        int target_chunk = -1;
        
        for (int c = 0; c < num_threads; ++c) {
            float old_prefix = prefix;
            prefix += chunk_sums[c];
            if (target_chunk < 0 && prefix >= r) {
                target_chunk = c;
                // 存储目标 chunk 之前的累积和
                chunk_sums[c] = old_prefix;
            }
        }
        
        // 存储目标 chunk 索引在位置 0
        chunk_first_idx[0] = target_chunk;
    }
    __syncthreads();
    
    // Step 3: 目标 chunk 的线程进行精确搜索
    int target_chunk = chunk_first_idx[0];
    if (target_chunk < 0) target_chunk = num_threads - 1;  // fallback
    
    int sampled_idx = vocab_size - 1;
    float sampled_prob = 0.0f;
    
    if (tid == target_chunk) {
        float prefix_before = chunk_sums[tid];  // 之前 chunk 的累积和
        float cumsum = prefix_before;
        
        const int search_start = tid * chunk_size;
        const int search_end = min(search_start + chunk_size, vocab_size);
        
        for (int i = search_start; i < search_end; ++i) {
            cumsum += row_probs[i];
            if (cumsum >= r) {
                sampled_idx = i;
                sampled_prob = row_probs[i];
                break;
            }
        }
    }
    __syncthreads();
    
    // Step 4: 写入结果（只有目标线程写入）
    if (tid == target_chunk) {
        sampled_tokens[row] = sampled_idx;
        confidences[row] = sampled_prob;
    }
}

// 计算累积概率并采样（支持 top-p）
__global__ void sample_with_topp_kernel(
    const float* sorted_probs,    // [block_length, vocab_size] sorted probabilities (descending)
    const int* sorted_indices,    // [block_length, vocab_size] original indices
    const float* random_vals,     // [block_length] random values in [0, 1)
    int* sampled_tokens,          // [block_length] output token ids
    float* confidences,           // [block_length] output confidences
    int vocab_size,
    int block_length,
    int top_k,                    // 0 means disabled
    float top_p                   // 1.0 means disabled
) {
    const int row = blockIdx.x;
    
    if (row >= block_length) return;
    
    const float* row_probs = sorted_probs + row * vocab_size;
    const int* row_indices = sorted_indices + row * vocab_size;
    
    // 计算有效范围（top-k 和 top-p）
    int effective_k = vocab_size;
    if (top_k > 0 && top_k < vocab_size) {
        effective_k = top_k;
    }
    
    // 计算 top-p 截断点并归一化
    float cumsum = 0.0f;
    float normalization_sum = 0.0f;
    int cutoff_idx = effective_k;
    
    for (int i = 0; i < effective_k; ++i) {
        cumsum += row_probs[i];
        if (cumsum > top_p && i > 0) {
            cutoff_idx = i;
            break;
        }
    }
    
    // 计算归一化常数
    for (int i = 0; i < cutoff_idx; ++i) {
        normalization_sum += row_probs[i];
    }
    
    if (normalization_sum <= 0.0f) {
        sampled_tokens[row] = row_indices[0];
        confidences[row] = 1.0f;
        return;
    }
    
    // 采样
    float r = random_vals[row] * normalization_sum;
    cumsum = 0.0f;
    int sampled_idx = 0;
    
    for (int i = 0; i < cutoff_idx; ++i) {
        cumsum += row_probs[i];
        if (cumsum >= r) {
            sampled_idx = i;
            break;
        }
    }
    
    sampled_tokens[row] = row_indices[sampled_idx];
    confidences[row] = row_probs[sampled_idx] / normalization_sum;
}

bool check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        DIFF_LOGE("[GpuSampler] %s failed: %s\n", msg, cudaGetErrorString(err));
        return false;
    }
    return true;
}

} // namespace

class GpuSampler::Impl {
public:
    Impl(int block_length, int vocab_size, const DiffusionConfig& config)
        : block_length_(block_length),
          vocab_size_(vocab_size),
          config_(config),
          d_logits_(nullptr),
          d_probs_(nullptr),
          d_indices_(nullptr),
          d_row_max_(nullptr),
          d_row_sum_(nullptr),
          d_random_vals_(nullptr),
          d_sampled_tokens_(nullptr),
          d_confidences_(nullptr),
          h_pinned_logits_(nullptr),
          h_pinned_probs_(nullptr),
          h_pinned_indices_(nullptr),
          use_multi_stream_(block_length >= NUM_STREAMS && vocab_size >= 8192),
          use_gpu_sampling_(true),
          initialized_(false) {
        for (int i = 0; i < NUM_STREAMS; ++i) {
            streams_[i] = nullptr;
        }
        initialized_ = init();
        DIFF_LOGD("[GpuSampler][debug] init done block_len=%d vocab=%d use_multi_stream=%d use_gpu_sampling=%d initialized=%d\n",
                  block_length_, vocab_size_, use_multi_stream_ ? 1 : 0, use_gpu_sampling_ ? 1 : 0, initialized_ ? 1 : 0);
    }

    ~Impl() {
        // Free device memory
        if (d_logits_) cudaFree(d_logits_);
        if (d_probs_) cudaFree(d_probs_);
        if (d_indices_) cudaFree(d_indices_);
        if (d_row_max_) cudaFree(d_row_max_);
        if (d_row_sum_) cudaFree(d_row_sum_);
        if (d_random_vals_) cudaFree(d_random_vals_);
        if (d_sampled_tokens_) cudaFree(d_sampled_tokens_);
        if (d_confidences_) cudaFree(d_confidences_);
        
        // Free pinned host memory
        if (h_pinned_logits_) cudaFreeHost(h_pinned_logits_);
        if (h_pinned_probs_) cudaFreeHost(h_pinned_probs_);
        if (h_pinned_indices_) cudaFreeHost(h_pinned_indices_);
        
        // Destroy streams
        for (int i = 0; i < NUM_STREAMS; ++i) {
            if (streams_[i]) cudaStreamDestroy(streams_[i]);
        }
    }

    bool is_available() const {
        if (!initialized_) {
            DIFF_LOGW("[GpuSampler][warn] is_available: not initialized\n");
        }
        return initialized_;
    }

    // Core sampling implementation - works with raw pointer
    // Uses multi-stream parallelism when available
    bool sample_impl(
        const float* logits_ptr,
        size_t logits_size,
        RemaskingStrategy remasking_strategy,
        std::mt19937& rng,
        std::vector<llama_token>& sampled_tokens,
        std::vector<float>& confidences,
        std::vector<std::vector<float>>* token_probs,
        GpuSamplerStats* stats,
        bool logits_on_device = false,
        bool force_non_fused = false
    ) {
        DIFF_LOGD("[GpuSampler][debug] sample_impl start logits_on_device=%d force_non_fused=%d ptr=%p size=%zu block=%d vocab=%d\n",
                  logits_on_device ? 1 : 0, force_non_fused ? 1 : 0,
                  (const void*)logits_ptr, logits_size, block_length_, vocab_size_);
        if (!initialized_) {
            DIFF_LOGE("[GpuSampler] sample_impl not initialized\n");
            return false;
        }

        // Device logits: allow sampling fewer rows (<= block_length_) to support active-position sampling.
        if (logits_on_device) {
            if (vocab_size_ <= 0 || (logits_size % static_cast<size_t>(vocab_size_) != 0)) {
                DIFF_LOGE("[GpuSampler] device sample_impl invalid size: got=%zu vocab=%d\n",
                          logits_size, vocab_size_);
                return false;
            }
            const int num_rows = static_cast<int>(logits_size / static_cast<size_t>(vocab_size_));
            if (num_rows <= 0 || num_rows > block_length_) {
                DIFF_LOGE("[GpuSampler] device sample_impl rows out of range: rows=%d block=%d vocab=%d size=%zu\n",
                          num_rows, block_length_, vocab_size_, logits_size);
                return false;
            }
            return sample_impl_single_stream_device(logits_ptr, logits_size, remasking_strategy,
                                                    rng, sampled_tokens, confidences, token_probs, stats,
                                                    num_rows,
                                                    force_non_fused);
        }

        const size_t expected = static_cast<size_t>(block_length_) * vocab_size_;
        if (logits_size != expected) {
            DIFF_LOGE("[GpuSampler] sample_impl size mismatch: got=%zu expected=%zu (block=%d vocab=%d)\n",
                      logits_size, expected, block_length_, vocab_size_);
            return false;
        }

        // Use multi-stream path if available and block_length is large enough
        if (use_multi_stream_ && h_pinned_logits_ && block_length_ >= NUM_STREAMS) {
            return sample_impl_multi_stream(logits_ptr, logits_size, remasking_strategy, 
                                            rng, sampled_tokens, confidences, token_probs, stats);
        } else {
            return sample_impl_single_stream(logits_ptr, logits_size, remasking_strategy,
                                             rng, sampled_tokens, confidences, token_probs, stats);
        }
    }

    // Single-stream implementation (original)
    bool sample_impl_single_stream(
        const float* logits_ptr,
        size_t logits_size,
        RemaskingStrategy remasking_strategy,
        std::mt19937& rng,
        std::vector<llama_token>& sampled_tokens,
        std::vector<float>& confidences,
        std::vector<std::vector<float>>* token_probs,
        GpuSamplerStats* stats
    ) {
        const size_t expected = static_cast<size_t>(block_length_) * vocab_size_;
        cudaStream_t stream = streams_[0];

        const bool need_probs = (remasking_strategy == RemaskingStrategy::ENTROPY_BOUNDED) && token_probs;
        token_probs_cache_.clear();
        if (need_probs) {
            token_probs_cache_.reserve(block_length_);
        }

        sampled_tokens.assign(block_length_, 0);
        confidences.assign(block_length_, 0.0f);

        DiffusionProfiler& profiler = DiffusionProfiler::instance();
        struct PrepareEvents {
            cudaEvent_t start{};
            cudaEvent_t after_copy{};
            cudaEvent_t after_temp{};
            cudaEvent_t after_mask{};
            cudaEvent_t after_rng{};
            PrepareEvents() {
                cudaEventCreateWithFlags(&start, cudaEventDefault);
                cudaEventCreateWithFlags(&after_copy, cudaEventDefault);
                cudaEventCreateWithFlags(&after_temp, cudaEventDefault);
                cudaEventCreateWithFlags(&after_mask, cudaEventDefault);
                cudaEventCreateWithFlags(&after_rng, cudaEventDefault);
            }
            ~PrepareEvents() {
                cudaEventDestroy(start);
                cudaEventDestroy(after_copy);
                cudaEventDestroy(after_temp);
                cudaEventDestroy(after_mask);
                cudaEventDestroy(after_rng);
            }
        } prep_events;
        
        // ========== Stage 1: H2D transfer + temperature scaling ==========
        cudaEventRecord(prep_events.start, stream);

        const size_t total_bytes = expected * sizeof(float);
        if (!check_cuda(cudaMemcpyAsync(d_logits_, logits_ptr, total_bytes, cudaMemcpyHostToDevice, stream), "H2D logits")) {
            return false;
        }
        cudaEventRecord(prep_events.after_copy, stream);

        double prepare_ms = 0.0;

        // ========== Fast path: GPU sampling without sort/topk/topp ==========
        const bool fast_gpu_sample = use_gpu_sampling_ &&
                                     !need_probs &&
                                     config_.top_k <= 0 &&
                                     config_.top_p >= 1.0f;
        static bool fastpath_warned_host = false;
        if (!fast_gpu_sample && use_gpu_sampling_ && !need_probs && config_.top_k <= 0 && config_.top_p >= 1.0f && !fastpath_warned_host) {
            DIFF_LOGI("[GpuSampler][info] fused fast path skipped on host logits (maybe force_non_fused/top_k/p/entropy)\n");
            fastpath_warned_host = true;
        }
        if (fast_gpu_sample) {
            ProfilerTimer sample_timer;
            // No extra temperature step in fused fast path; mark temp event same as copy
            cudaEventRecord(prep_events.after_temp, stream);

            // Tail mask (same as non-fused path) to avoid sampling OOV ids
            const int safe_vocab = config_.n_vocab_limit > 0 ? config_.n_vocab_limit : vocab_size_;
            if (safe_vocab < vocab_size_) {
                const int threads_mask = 256;
                const int blocks_mask = block_length_;
                mask_tail_kernel<<<blocks_mask, threads_mask, 0, stream>>>(d_logits_, vocab_size_, block_length_, safe_vocab);
                if (stats) {
                    stats->n_vocab_limit = safe_vocab;
                }
            }
            cudaEventRecord(prep_events.after_mask, stream);

            // Fused softmax + sampling kernel (single pass)
            std::uniform_int_distribution<uint64_t> dist64;
            uint64_t seed = dist64(rng);
            int rand_threads = 256;
            int rand_blocks = (block_length_ + rand_threads - 1) / rand_threads;
            fill_random_kernel<<<rand_blocks, rand_threads, 0, stream>>>(d_random_vals_, seed, block_length_);
            cudaEventRecord(prep_events.after_rng, stream);
            int sample_threads = (vocab_size_ <= 4096) ? 128 : 256;
            const size_t sample_smem = (sample_threads * 2 + 1) * sizeof(float) + sizeof(int);
            const float inv_temp = (config_.temperature != 1.0f) ? 1.0f / config_.temperature : 1.0f;
            fused_softmax_sample_kernel<<<block_length_, sample_threads, sample_smem, stream>>>(
                d_logits_,
                vocab_size_,
                block_length_,
                inv_temp,
                d_random_vals_,
                d_sampled_tokens_,
                d_confidences_
            );

            // Copy back minimal results with event-based wait (avoid full stream sync)
            ProfilerTimer d2h_timer;
            if (!check_cuda(cudaMemcpyAsync(sampled_tokens.data(), d_sampled_tokens_,
                                            block_length_ * sizeof(int),
                                            cudaMemcpyDeviceToHost, stream),
                            "D2H sampled_tokens")) {
                return false;
            }
            if (!check_cuda(cudaMemcpyAsync(confidences.data(), d_confidences_,
                                            block_length_ * sizeof(float),
                                            cudaMemcpyDeviceToHost, stream),
                            "D2H confidences")) {
                return false;
            }
            cudaEvent_t d2h_event;
            if (cudaEventCreateWithFlags(&d2h_event, cudaEventDisableTiming) != cudaSuccess) {
                return false;
            }
            cudaEventRecord(d2h_event, stream);
            cudaEventSynchronize(d2h_event);
            cudaEventDestroy(d2h_event);
            double d2h_ms = d2h_timer.elapsed_ms();

            // Defensive clamp after copy back (should be redundant after mask_tail)
            if (safe_vocab < vocab_size_) {
                const int clamp_id = safe_vocab - 1;
                for (int i = 0; i < block_length_; ++i) {
                    if (sampled_tokens[i] >= safe_vocab) {
                        sampled_tokens[i] = clamp_id;
                        confidences[i] = 0.0f;
                    }
                }
            }

            // Optional debug: compare fused kernel vs non-fused GPU sampling on same logits/seed
            if (std::getenv("DIFFUSION_FUSED_VS_NONFUSED")) {
                std::vector<llama_token> fused_tokens = sampled_tokens;
                std::vector<float> fused_conf = confidences;

                // Reference path: non-fused GPU softmax + sampling (device logits)
                const int threads_per_block = 256;
                const size_t smem_size = threads_per_block * sizeof(float);
                find_row_max_kernel<<<block_length_, threads_per_block, smem_size, stream>>>(
                    d_logits_, d_row_max_, vocab_size_, block_length_);

                softmax_exp_sum_kernel<<<block_length_, threads_per_block, smem_size, stream>>>(
                    d_logits_, d_row_max_, d_probs_, d_row_sum_, vocab_size_, block_length_);

                softmax_normalize_kernel<<<block_length_, threads_per_block, 0, stream>>>(
                    d_probs_, d_row_sum_, vocab_size_, block_length_);

                // Use the same seed for deterministic comparison
                int threads = 256;
                int blocks = (block_length_ + threads - 1) / threads;
                fill_random_kernel<<<blocks, threads, 0, stream>>>(d_random_vals_, seed, block_length_);

                int ref_sample_threads = (vocab_size_ <= 4096) ? 128 : 256;
                const size_t ref_sample_smem = ref_sample_threads * sizeof(float) + ref_sample_threads * sizeof(int);
                sample_tokens_kernel<<<block_length_, ref_sample_threads, ref_sample_smem, stream>>>(
                    d_probs_, d_random_vals_,
                    d_sampled_tokens_, d_confidences_,
                    vocab_size_, block_length_,
                    config_.top_k, config_.top_p
                );

                std::vector<llama_token> ref_tokens(block_length_);
                std::vector<float> ref_conf(block_length_);
                cudaMemcpyAsync(ref_tokens.data(), d_sampled_tokens_,
                                block_length_ * sizeof(int),
                                cudaMemcpyDeviceToHost, stream);
                cudaMemcpyAsync(ref_conf.data(), d_confidences_,
                                block_length_ * sizeof(float),
                                cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);

                if (safe_vocab < vocab_size_) {
                    const int clamp_id_ref = safe_vocab - 1;
                    for (int i = 0; i < block_length_; ++i) {
                        if (ref_tokens[i] >= safe_vocab) {
                            ref_tokens[i] = clamp_id_ref;
                            ref_conf[i] = 0.0f;
                        }
                    }
                }

                int mism = 0;
                int first_idx = -1;
                for (int i = 0; i < block_length_; ++i) {
                    if (fused_tokens[i] != ref_tokens[i]) {
                        mism++;
                        if (first_idx < 0) first_idx = i;
                    }
                }
                if (mism > 0) {
                    DIFF_LOGD("[fused_vs_nonfused] mismatch=%d/%d first=%d fused=%d ref=%d\n",
                              mism, block_length_, first_idx,
                              fused_tokens[first_idx], ref_tokens[first_idx]);
                } else {
                    DIFF_LOGD("[fused_vs_nonfused] match all (%d tokens)\n", block_length_);
                }

                // Restore fused outputs as the return value
                sampled_tokens.swap(fused_tokens);
                confidences.swap(fused_conf);
            }

            float ms_copy = 0.0f, ms_temp = 0.0f, ms_mask = 0.0f, ms_rng = 0.0f, ms_prepare = 0.0f;
            cudaEventElapsedTime(&ms_copy, prep_events.start, prep_events.after_copy);
            cudaEventElapsedTime(&ms_temp, prep_events.after_copy, prep_events.after_temp);
            cudaEventElapsedTime(&ms_mask, prep_events.after_temp, prep_events.after_mask);
            cudaEventElapsedTime(&ms_rng, prep_events.after_mask, prep_events.after_rng);
            cudaEventElapsedTime(&ms_prepare, prep_events.start, prep_events.after_rng);
            prepare_ms = ms_prepare;

            double total_ms = sample_timer.elapsed_ms();
            double fused_ms = std::max(0.0, total_ms - d2h_ms);

            profiler.record_custom("sampler_gpu_stage_prepare_ms", prepare_ms);
            profiler.record_custom("sampler_gpu_stage_softmax_ms", fused_ms);
            profiler.record_custom("sampler_gpu_stage_sort_ms", 0.0);
            profiler.record_custom("sampler_gpu_stage_d2h_ms", d2h_ms);
            profiler.record_custom("sampler_gpu_stage_cpu_post_ms", 0.0);
            profiler.record_custom("sampler_gpu_prepare_copy_ms", ms_copy);
            profiler.record_custom("sampler_gpu_prepare_temp_ms", ms_temp);
            profiler.record_custom("sampler_gpu_prepare_mask_ms", ms_mask);
            profiler.record_custom("sampler_gpu_prepare_rng_ms", ms_rng);

            if (stats) {
                stats->stage_prepare_ms = prepare_ms;
                stats->stage_prepare_copy_ms = ms_copy;
                stats->stage_prepare_temp_ms = ms_temp;
                stats->stage_prepare_mask_ms = ms_mask;
                stats->stage_prepare_rng_ms = ms_rng;
                stats->stage_softmax_ms = fused_ms;
                stats->stage_sort_ms = 0.0;
                stats->stage_sample_ms = fused_ms;
                stats->stage_d2h_ms = d2h_ms;
                stats->stage_cpu_post_ms = 0.0;
                stats->fast_path = true;
                stats->device_logits = false;
            }

            // No token_probs in fast path
            return true;
        }
        if (config_.temperature != 1.0f) {
            const float inv_temp = 1.0f / config_.temperature;
            const size_t threads = 256;
            const size_t blocks = (expected + threads - 1) / threads;
            scale_logits_kernel<<<blocks, threads, 0, stream>>>(d_logits_, inv_temp, expected);
            cudaEventRecord(prep_events.after_temp, stream);
        } else {
            cudaEventRecord(prep_events.after_temp, stream);
        }

        // Tail mask to avoid sampling ids beyond vocab limit (needed before softmax and fast path)
        const int safe_vocab_device = config_.n_vocab_limit > 0 ? config_.n_vocab_limit : vocab_size_;
        if (safe_vocab_device < vocab_size_) {
            const int threads_mask = 256;
            const int blocks_mask = block_length_;
            mask_tail_kernel<<<blocks_mask, threads_mask, 0, stream>>>(d_logits_, vocab_size_, block_length_, safe_vocab_device);
            if (stats) {
                stats->n_vocab_limit = safe_vocab_device;
            }
        }
        cudaEventRecord(prep_events.after_mask, stream);
        cudaEventRecord(prep_events.after_rng, stream);

        // Rely on stream ordering; avoid extra sync before softmax
        // ========== Stage 2: GPU Softmax (batched for all rows) ==========
        ProfilerTimer softmax_timer;
        
        const int threads_per_block = 256;
        const size_t smem_size = threads_per_block * sizeof(float);
        
        find_row_max_kernel<<<block_length_, threads_per_block, smem_size, stream>>>(
            d_logits_, d_row_max_, vocab_size_, block_length_);
        
        softmax_exp_sum_kernel<<<block_length_, threads_per_block, smem_size, stream>>>(
            d_logits_, d_row_max_, d_probs_, d_row_sum_, vocab_size_, block_length_);
        
        softmax_normalize_kernel<<<block_length_, threads_per_block, 0, stream>>>(
            d_probs_, d_row_sum_, vocab_size_, block_length_);
        double softmax_ms = softmax_timer.elapsed_ms();

        // ========== Stage 3: Sort probabilities (only if needed for top-p/top-k) ==========
        ProfilerTimer sort_timer;
        double sort_ms = 0.0;
        
        auto policy = thrust::cuda::par.on(stream);
        const bool need_sort = (config_.top_p < 1.0f) || (config_.top_k > 0 && config_.top_k < vocab_size_);
        
        if (need_sort) {
            for (int row = 0; row < block_length_; ++row) {
                float* row_probs = d_probs_ + row * vocab_size_;
                int* row_indices = d_indices_ + row * vocab_size_;
                
                thrust::device_ptr<int> idx_ptr(row_indices);
                thrust::sequence(policy, idx_ptr, idx_ptr + vocab_size_);
                
                thrust::device_ptr<float> prob_ptr(row_probs);
                thrust::sort_by_key(policy, prob_ptr, prob_ptr + vocab_size_, idx_ptr, thrust::greater<float>());
            }
            
        }
        sort_ms = sort_timer.elapsed_ms();

        // ========== Stage 4: D2H transfer ==========
        ProfilerTimer d2h_timer;
        
        int transfer_count = vocab_size_;
        if (config_.top_k > 0 && config_.top_k < vocab_size_) {
            transfer_count = config_.top_k;
        }
        if (config_.top_p < 1.0f && transfer_count > 1024) {
            transfer_count = 1024;
        }
        
        host_probs_.resize(static_cast<size_t>(block_length_) * transfer_count);
        host_indices_.resize(static_cast<size_t>(block_length_) * transfer_count);
        
        for (int row = 0; row < block_length_; ++row) {
            float* src_probs = d_probs_ + row * vocab_size_;
            int* src_indices = d_indices_ + row * vocab_size_;
            float* dst_probs = host_probs_.data() + row * transfer_count;
            int* dst_indices = host_indices_.data() + row * transfer_count;
            
            if (!check_cuda(cudaMemcpyAsync(dst_probs, src_probs, transfer_count * sizeof(float), cudaMemcpyDeviceToHost, stream), "D2H probs")) {
                return false;
            }
            if (need_sort) {
                if (!check_cuda(cudaMemcpyAsync(dst_indices, src_indices, transfer_count * sizeof(int), cudaMemcpyDeviceToHost, stream), "D2H indices")) {
                    return false;
                }
            }
        }
        
        if (!check_cuda(cudaStreamSynchronize(stream), "sync after D2H")) {
            return false;
        }
        double d2h_ms = d2h_timer.elapsed_ms();

        // ========== Stage 5: CPU sampling (minimal work) ==========
        ProfilerTimer cpu_timer;
        
        sample_on_cpu(need_sort, need_probs, transfer_count, rng, sampled_tokens, confidences);
        
        double cpu_ms = cpu_timer.elapsed_ms();

        float ms_copy = 0.0f, ms_temp = 0.0f, ms_mask = 0.0f, ms_rng = 0.0f, ms_prepare = 0.0f;
        cudaEventElapsedTime(&ms_copy, prep_events.start, prep_events.after_copy);
        cudaEventElapsedTime(&ms_temp, prep_events.after_copy, prep_events.after_temp);
        cudaEventElapsedTime(&ms_mask, prep_events.after_temp, prep_events.after_mask);
        cudaEventElapsedTime(&ms_rng, prep_events.after_mask, prep_events.after_rng);
        cudaEventElapsedTime(&ms_prepare, prep_events.start, prep_events.after_rng);
        prepare_ms = ms_prepare;

        // Record stats
        profiler.record_custom("sampler_gpu_stage_prepare_ms", prepare_ms);
        profiler.record_custom("sampler_gpu_stage_softmax_ms", softmax_ms);
        profiler.record_custom("sampler_gpu_stage_sort_ms", sort_ms);
        profiler.record_custom("sampler_gpu_stage_d2h_ms", d2h_ms);
        profiler.record_custom("sampler_gpu_stage_cpu_post_ms", cpu_ms);
        profiler.record_custom("sampler_gpu_prepare_copy_ms", ms_copy);
        profiler.record_custom("sampler_gpu_prepare_temp_ms", ms_temp);
        profiler.record_custom("sampler_gpu_prepare_mask_ms", ms_mask);
        profiler.record_custom("sampler_gpu_prepare_rng_ms", ms_rng);
        
        if (stats) {
            stats->stage_prepare_ms = prepare_ms;
            stats->stage_prepare_copy_ms = ms_copy;
            stats->stage_prepare_temp_ms = ms_temp;
            stats->stage_prepare_mask_ms = ms_mask;
            stats->stage_prepare_rng_ms = ms_rng;
            stats->stage_softmax_ms = softmax_ms;
            stats->stage_sort_ms = sort_ms;
            stats->stage_sample_ms = 0.0;
            stats->stage_d2h_ms = d2h_ms;
            stats->stage_cpu_post_ms = cpu_ms;
        }

        if (need_probs && token_probs) {
            *token_probs = token_probs_cache_;
        }

        return true;
    }

    // Single-stream implementation when logits already reside on device (CUDA)
    bool sample_impl_single_stream_device(
        const float* logits_ptr,
        size_t logits_size,
        RemaskingStrategy remasking_strategy,
        std::mt19937& rng,
        std::vector<llama_token>& sampled_tokens,
        std::vector<float>& confidences,
        std::vector<std::vector<float>>* token_probs,
        GpuSamplerStats* stats,
        int num_rows,
        bool force_non_fused = false
    ) {
        DIFF_LOGD("[GpuSampler][debug] device path start ptr=%p size=%zu force_non_fused=%d block=%d vocab=%d\n",
                  (const void*)logits_ptr, logits_size, force_non_fused ? 1 : 0, num_rows, vocab_size_);
        const size_t expected = static_cast<size_t>(num_rows) * vocab_size_;
        cudaStream_t stream = streams_[0];

        const bool need_probs = (remasking_strategy == RemaskingStrategy::ENTROPY_BOUNDED) && token_probs;
        token_probs_cache_.clear();
        if (need_probs) {
            token_probs_cache_.reserve(block_length_);
        }

        sampled_tokens.assign(block_length_, 0);
        confidences.assign(block_length_, 0.0f);

        DiffusionProfiler& profiler = DiffusionProfiler::instance();

        // Use CUDA events for accurate async timing
        auto make_event = []() {
            cudaEvent_t ev;
            cudaEventCreateWithFlags(&ev, cudaEventDefault);
            return ev;
        };
        ProfilerTimer wall_timer;
        cudaEvent_t ev_whole_start = make_event();
        cudaEvent_t ev_whole_end = make_event();
        cudaEvent_t ev_start = make_event();
        cudaEvent_t ev_after_copy = make_event();
        cudaEvent_t ev_after_temp = make_event();
        cudaEvent_t ev_after_mask = make_event();
        cudaEvent_t ev_after_rng = make_event();
        cudaEvent_t ev_after_softmax = make_event();
        cudaEvent_t ev_after_sample = make_event();
        cudaEvent_t ev_after_d2h = make_event();

        // ========== Stage 1: D2D transfer + temperature scaling ==========
        cudaEventRecord(ev_whole_start, stream);
        cudaEventRecord(ev_start, stream);

        const size_t total_bytes = expected * sizeof(float);
        if (!check_cuda(cudaMemcpyAsync(d_logits_, logits_ptr, total_bytes, cudaMemcpyDeviceToDevice, stream), "D2D logits")) {
            cudaEventDestroy(ev_whole_start);
            cudaEventDestroy(ev_whole_end);
            cudaEventDestroy(ev_start);
            cudaEventDestroy(ev_after_copy);
            cudaEventDestroy(ev_after_temp);
            cudaEventDestroy(ev_after_mask);
            cudaEventDestroy(ev_after_rng);
            cudaEventDestroy(ev_after_softmax);
            cudaEventDestroy(ev_after_sample);
            cudaEventDestroy(ev_after_d2h);
            return false;
        }
        cudaEventRecord(ev_after_copy, stream);
        cudaDeviceSynchronize();
        auto err = cudaGetLastError();
        DIFF_LOGD("[GpuSampler][debug] device D2D logits err=%d\n", int(err));

        // ========== Fast path: GPU sampling without sort/topk/topp ==========
        const bool fast_gpu_sample = !force_non_fused &&
                                     use_gpu_sampling_ &&
                                     !need_probs &&
                                     config_.top_k <= 0 &&
                                     config_.top_p >= 1.0f;
        DIFF_LOGD("[GpuSampler][debug] device fast_gpu_sample=%d use_gpu_sampling=%d need_probs=%d top_k=%d top_p=%f\n",
                  fast_gpu_sample ? 1 : 0, use_gpu_sampling_ ? 1 : 0, need_probs ? 1 : 0, config_.top_k, config_.top_p);
        static bool fastpath_warned_device = false;
        if (!fast_gpu_sample && use_gpu_sampling_ && !need_probs && config_.top_k <= 0 && config_.top_p >= 1.0f && !fastpath_warned_device) {
            DIFF_LOGI("[GpuSampler][info] fused fast path skipped on device logits (force_non_fused=%d)\n", force_non_fused ? 1 : 0);
            fastpath_warned_device = true;
        }
        // Temperature scaling (applied before fused branch so fused uses inv_temp=1.0)
        if (config_.temperature != 1.0f) {
            const float inv_temp = 1.0f / config_.temperature;
            const size_t threads = 256;
            const size_t blocks = (expected + threads - 1) / threads;
            scale_logits_kernel<<<blocks, threads, 0, stream>>>(d_logits_, inv_temp, expected);
            cudaEventRecord(ev_after_temp, stream);
        } else {
            cudaEventRecord(ev_after_temp, stream);
        }

        // Tail mask for vocab limit on device logits
        const int safe_vocab_device = config_.n_vocab_limit > 0 ? config_.n_vocab_limit : vocab_size_;
        if (safe_vocab_device < vocab_size_) {
            const int threads_mask = 256;
            const int blocks_mask = block_length_;
            mask_tail_kernel<<<blocks_mask, threads_mask, 0, stream>>>(d_logits_, vocab_size_, num_rows, safe_vocab_device);
            if (stats) {
                stats->n_vocab_limit = safe_vocab_device;
            }
        }
        cudaEventRecord(ev_after_mask, stream);
        cudaEventRecord(ev_after_rng, stream);

        // Fused fast path on device logits (skip softmax kernels)
        if (fast_gpu_sample) {
            std::uniform_int_distribution<uint64_t> dist64;
            uint64_t seed = dist64(rng);
            int rand_threads = 256;
            int rand_blocks = (num_rows + rand_threads - 1) / rand_threads;
            fill_random_kernel<<<rand_blocks, rand_threads, 0, stream>>>(d_random_vals_, seed, num_rows);
            cudaEventRecord(ev_after_rng, stream);

            int sample_threads = (vocab_size_ <= 4096) ? 128 : 256;
            const size_t sample_smem = (sample_threads * 2 + 1) * sizeof(float) + sizeof(int);
            const float inv_temp_fused = 1.0f; // logits 已按需温度缩放

            fused_softmax_sample_kernel<<<num_rows, sample_threads, sample_smem, stream>>>(
                d_logits_,
                vocab_size_,
                block_length_,
                inv_temp_fused,
                d_random_vals_,
                d_sampled_tokens_,
                d_confidences_
            );
            cudaEventRecord(ev_after_sample, stream);

            cudaMemcpyAsync(sampled_tokens.data(), d_sampled_tokens_,
                            num_rows * sizeof(int),
                            cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(confidences.data(), d_confidences_,
                            num_rows * sizeof(float),
                            cudaMemcpyDeviceToHost, stream);
            cudaEventRecord(ev_after_d2h, stream);
            cudaEventRecord(ev_whole_end, stream);
            cudaStreamSynchronize(stream);

            // Clamp any out-of-range ids defensively
            if (safe_vocab_device < vocab_size_) {
                const int clamp_id = safe_vocab_device - 1;
                for (int i = 0; i < num_rows; ++i) {
                    if (sampled_tokens[i] >= safe_vocab_device) {
                        sampled_tokens[i] = clamp_id;
                        confidences[i] = 0.0f;
                    }
                }
            }

            float ms_copy = 0.0f, ms_temp = 0.0f, ms_mask = 0.0f, ms_rng = 0.0f, ms_prepare = 0.0f;
            float ms_fused = 0.0f, ms_d2h = 0.0f, ms_whole_gpu = 0.0f;
            cudaEventElapsedTime(&ms_copy, ev_start, ev_after_copy);
            cudaEventElapsedTime(&ms_temp, ev_after_copy, ev_after_temp);
            cudaEventElapsedTime(&ms_mask, ev_after_temp, ev_after_mask);
            cudaEventElapsedTime(&ms_rng, ev_after_mask, ev_after_rng);
            cudaEventElapsedTime(&ms_prepare, ev_start, ev_after_rng);
            cudaEventElapsedTime(&ms_fused, ev_after_rng, ev_after_sample);
            cudaEventElapsedTime(&ms_d2h, ev_after_sample, ev_after_d2h);
            cudaEventElapsedTime(&ms_whole_gpu, ev_whole_start, ev_whole_end);
            double wall_ms = wall_timer.elapsed_ms();
            double stage_total_ms = ms_prepare + ms_fused + ms_d2h;

            if (stats) {
                stats->stage_prepare_ms = ms_prepare;
                stats->stage_prepare_copy_ms = ms_copy;
                stats->stage_prepare_temp_ms = ms_temp;
                stats->stage_prepare_mask_ms = ms_mask;
                stats->stage_prepare_rng_ms = ms_rng;
                stats->stage_softmax_ms = ms_fused;
                stats->stage_sort_ms = 0.0;
                stats->stage_sample_ms = ms_fused;
                stats->stage_d2h_ms = ms_d2h;
                stats->stage_cpu_post_ms = 0.0;
                stats->stage_event_wait_ms = 0.0;
                stats->stage_total_ms = stage_total_ms;
                stats->stage_whole_gpu_ms = ms_whole_gpu;
                stats->stage_whole_wall_ms = wall_ms;
                stats->fast_path = true;
                stats->device_logits = true;
            }

            DiffusionProfiler& profiler = DiffusionProfiler::instance();
            profiler.record_custom("sampler_gpu_stage_prepare_ms", ms_prepare);
            profiler.record_custom("sampler_gpu_stage_softmax_ms", ms_fused);
            profiler.record_custom("sampler_gpu_stage_sort_ms", 0.0);
            profiler.record_custom("sampler_gpu_stage_sample_ms", ms_fused);
            profiler.record_custom("sampler_gpu_stage_d2h_ms", ms_d2h);
            profiler.record_custom("sampler_gpu_stage_cpu_post_ms", 0.0);
            profiler.record_custom("sampler_gpu_stage_event_wait_ms", 0.0);
            profiler.record_custom("sampler_gpu_stage_total_ms", stage_total_ms);
            profiler.record_custom("sampler_gpu_stage_whole_gpu_ms", ms_whole_gpu);
            profiler.record_custom("sampler_gpu_stage_whole_wall_ms", wall_ms);
            profiler.record_custom("sampler_gpu_prepare_copy_ms", ms_copy);
            profiler.record_custom("sampler_gpu_prepare_temp_ms", ms_temp);
            profiler.record_custom("sampler_gpu_prepare_mask_ms", ms_mask);
            profiler.record_custom("sampler_gpu_prepare_rng_ms", ms_rng);

            cudaEventDestroy(ev_whole_start);
            cudaEventDestroy(ev_whole_end);
            cudaEventDestroy(ev_start);
            cudaEventDestroy(ev_after_copy);
            cudaEventDestroy(ev_after_temp);
            cudaEventDestroy(ev_after_mask);
            cudaEventDestroy(ev_after_rng);
            cudaEventDestroy(ev_after_softmax);
            cudaEventDestroy(ev_after_sample);
            cudaEventDestroy(ev_after_d2h);

            return true;
        }

        // ========== Stage 2: GPU Softmax (batched for all rows) ==========

        const int threads_per_block = 256;
        const size_t smem_size = threads_per_block * sizeof(float);
        find_row_max_kernel<<<num_rows, threads_per_block, smem_size, stream>>>(
            d_logits_, d_row_max_, vocab_size_, block_length_);
        err = cudaGetLastError();
        DIFF_LOGD("[GpuSampler][debug] device find_row_max err=%d\n", int(err));

        softmax_exp_sum_kernel<<<num_rows, threads_per_block, smem_size, stream>>>(
            d_logits_, d_row_max_, d_probs_, d_row_sum_, vocab_size_, block_length_);
        err = cudaGetLastError();
        DIFF_LOGD("[GpuSampler][debug] device softmax_exp_sum err=%d\n", int(err));

        softmax_normalize_kernel<<<num_rows, threads_per_block, 0, stream>>>(
            d_probs_, d_row_sum_, vocab_size_, block_length_);
        err = cudaGetLastError();
        DIFF_LOGD("[GpuSampler][debug] device softmax_normalize err=%d\n", int(err));
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        DIFF_LOGD("[GpuSampler][debug] device softmax sync err=%d\n", int(err));
        cudaEventRecord(ev_after_softmax, stream);

        // Fast GPU sampling path (non-fused) on device logits
        if (fast_gpu_sample) {
            std::uniform_int_distribution<uint64_t> dist64;
            uint64_t seed = dist64(rng);
            int threads = 256;
            int blocks = (num_rows + threads - 1) / threads;
            fill_random_kernel<<<blocks, threads, 0, stream>>>(d_random_vals_, seed, num_rows);

            int sample_threads = (vocab_size_ <= 4096) ? 128 : 256;
            const size_t sample_smem = sample_threads * sizeof(float) + sample_threads * sizeof(int);
            sample_tokens_kernel<<<num_rows, sample_threads, sample_smem, stream>>>(
                d_probs_, d_random_vals_,
                d_sampled_tokens_, d_confidences_,
                vocab_size_, num_rows,
                config_.top_k, config_.top_p
            );

            cudaMemcpyAsync(sampled_tokens.data(), d_sampled_tokens_,
                            num_rows * sizeof(int),
                            cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(confidences.data(), d_confidences_,
                            num_rows * sizeof(float),
                            cudaMemcpyDeviceToHost, stream);
            cudaDeviceSynchronize();
            err = cudaGetLastError();
            DIFF_LOGD("[GpuSampler][debug] device fast path cudaGetLastError after sample/d2h err=%d\n", int(err));

            cudaEventRecord(ev_after_sample, stream);
            cudaEventRecord(ev_after_d2h, stream);
            cudaEventRecord(ev_whole_end, stream);
            cudaStreamSynchronize(stream);

            // Clamp any out-of-range ids defensively (should already be masked on device)
            const int clamp_limit = config_.n_vocab_limit > 0 ? config_.n_vocab_limit : vocab_size_;
            if (clamp_limit < vocab_size_) {
                const int clamp_id = clamp_limit - 1;
                for (int i = 0; i < num_rows; ++i) {
                    if (sampled_tokens[i] >= clamp_limit) {
                        sampled_tokens[i] = clamp_id;
                        confidences[i] = 0.0f;
                    }
                }
            }

            float ms_copy = 0.0f, ms_temp = 0.0f, ms_mask = 0.0f, ms_rng = 0.0f, ms_prepare = 0.0f;
            float ms_softmax = 0.0f, ms_sample = 0.0f, ms_d2h = 0.0f, ms_whole_gpu = 0.0f;
            cudaEventElapsedTime(&ms_copy, ev_start, ev_after_copy);
            cudaEventElapsedTime(&ms_temp, ev_after_copy, ev_after_temp);
            cudaEventElapsedTime(&ms_mask, ev_after_temp, ev_after_mask);
            cudaEventElapsedTime(&ms_rng, ev_after_mask, ev_after_rng);
            cudaEventElapsedTime(&ms_prepare, ev_start, ev_after_rng);
            cudaEventElapsedTime(&ms_softmax, ev_after_rng, ev_after_softmax);
            cudaEventElapsedTime(&ms_sample, ev_after_softmax, ev_after_sample);
            cudaEventElapsedTime(&ms_d2h, ev_after_sample, ev_after_d2h);
            cudaEventElapsedTime(&ms_whole_gpu, ev_whole_start, ev_whole_end);
            double wall_ms = wall_timer.elapsed_ms();
            double stage_total_ms = ms_prepare + ms_softmax + ms_sample + ms_d2h;

            if (stats) {
                stats->stage_prepare_ms = ms_prepare;
                stats->stage_prepare_copy_ms = ms_copy;
                stats->stage_prepare_temp_ms = ms_temp;
                stats->stage_prepare_mask_ms = ms_mask;
                stats->stage_prepare_rng_ms = ms_rng;
                stats->stage_softmax_ms = ms_softmax;
                stats->stage_sort_ms = 0.0;
                stats->stage_sample_ms = ms_sample;
                stats->stage_d2h_ms = ms_d2h;
                stats->stage_cpu_post_ms = 0.0;
                stats->stage_event_wait_ms = 0.0;
                stats->stage_total_ms = stage_total_ms;
                stats->stage_whole_gpu_ms = ms_whole_gpu;
                stats->stage_whole_wall_ms = wall_ms;
                stats->fast_path = false;          // non-fused path
                stats->device_logits = true;
            }

            DiffusionProfiler& profiler = DiffusionProfiler::instance();
            profiler.record_custom("sampler_gpu_stage_prepare_ms", ms_prepare);
            profiler.record_custom("sampler_gpu_stage_softmax_ms", ms_softmax);
            profiler.record_custom("sampler_gpu_stage_sort_ms", 0.0);
            profiler.record_custom("sampler_gpu_stage_sample_ms", ms_sample);
            profiler.record_custom("sampler_gpu_stage_d2h_ms", ms_d2h);
            profiler.record_custom("sampler_gpu_stage_cpu_post_ms", 0.0);
            profiler.record_custom("sampler_gpu_stage_event_wait_ms", 0.0);
            profiler.record_custom("sampler_gpu_prepare_copy_ms", ms_copy);
            profiler.record_custom("sampler_gpu_prepare_temp_ms", ms_temp);
            profiler.record_custom("sampler_gpu_prepare_mask_ms", ms_mask);
            profiler.record_custom("sampler_gpu_prepare_rng_ms", ms_rng);
            profiler.record_custom("sampler_gpu_stage_total_ms", stage_total_ms);
            profiler.record_custom("sampler_gpu_stage_whole_gpu_ms", ms_whole_gpu);
            profiler.record_custom("sampler_gpu_stage_whole_wall_ms", wall_ms);

            cudaEventDestroy(ev_whole_start);
            cudaEventDestroy(ev_whole_end);
            cudaEventDestroy(ev_start);
            cudaEventDestroy(ev_after_copy);
            cudaEventDestroy(ev_after_temp);
            cudaEventDestroy(ev_after_mask);
            cudaEventDestroy(ev_after_rng);
            cudaEventDestroy(ev_after_softmax);
            cudaEventDestroy(ev_after_sample);
            cudaEventDestroy(ev_after_d2h);

            return true;
        }

        // For need_probs/top-k/p we currently don't support device logits path; fall back.
        cudaEventDestroy(ev_whole_start);
        cudaEventDestroy(ev_whole_end);
        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_after_copy);
        cudaEventDestroy(ev_after_temp);
        cudaEventDestroy(ev_after_mask);
        cudaEventDestroy(ev_after_rng);
        cudaEventDestroy(ev_after_softmax);
        cudaEventDestroy(ev_after_sample);
        cudaEventDestroy(ev_after_d2h);
        return false;
    }

    // Multi-stream implementation - overlaps H2D, compute, and D2H for different rows
    bool sample_impl_multi_stream(
        const float* logits_ptr,
        size_t logits_size,
        RemaskingStrategy remasking_strategy,
        std::mt19937& rng,
        std::vector<llama_token>& sampled_tokens,
        std::vector<float>& confidences,
        std::vector<std::vector<float>>* token_probs,
        GpuSamplerStats* stats
    ) {
        const bool need_probs = (remasking_strategy == RemaskingStrategy::ENTROPY_BOUNDED) && token_probs;
        token_probs_cache_.clear();
        if (need_probs) {
            token_probs_cache_.reserve(block_length_);
        }

        sampled_tokens.assign(block_length_, 0);
        confidences.assign(block_length_, 0.0f);

        DiffusionProfiler& profiler = DiffusionProfiler::instance();
        ProfilerTimer total_timer;
        
        const bool need_sort = (config_.top_p < 1.0f) || (config_.top_k > 0 && config_.top_k < vocab_size_);
        const float inv_temp = 1.0f / config_.temperature;
        const bool apply_temp = (config_.temperature != 1.0f);
        
        // Determine transfer count per row
        int transfer_count = vocab_size_;
        if (config_.top_k > 0 && config_.top_k < vocab_size_) {
            transfer_count = config_.top_k;
        }
        if (config_.top_p < 1.0f && transfer_count > 1024) {
            transfer_count = 1024;
        }
        
        // Prepare pinned host buffers
        const size_t row_bytes = static_cast<size_t>(vocab_size_) * sizeof(float);
        const size_t total_floats = static_cast<size_t>(block_length_) * vocab_size_;
        
        // Copy input logits to pinned memory first
        ProfilerTimer prepare_timer;
        std::memcpy(h_pinned_logits_, logits_ptr, total_floats * sizeof(float));
        
        host_probs_.resize(static_cast<size_t>(block_length_) * transfer_count);
        host_indices_.resize(static_cast<size_t>(block_length_) * transfer_count);
        
        double prepare_ms = prepare_timer.elapsed_ms();
        
        // Process rows in parallel using multiple streams
        // Each stream handles a subset of rows
        ProfilerTimer compute_timer;
        
        const int threads_per_block = 256;
        const size_t smem_size = threads_per_block * sizeof(float);
        const size_t scale_threads = 256;
        const size_t scale_blocks = (vocab_size_ + scale_threads - 1) / scale_threads;
        
        // Launch all H2D transfers first (async, overlapped)
        for (int row = 0; row < block_length_; ++row) {
            int stream_idx = row % NUM_STREAMS;
            cudaStream_t stream = streams_[stream_idx];
            
            float* src = h_pinned_logits_ + row * vocab_size_;
            float* dst = d_logits_ + row * vocab_size_;
            
            cudaMemcpyAsync(dst, src, row_bytes, cudaMemcpyHostToDevice, stream);
        }
        
        // Launch temperature scaling kernels (overlapped with remaining H2D)
        if (apply_temp) {
            for (int row = 0; row < block_length_; ++row) {
                int stream_idx = row % NUM_STREAMS;
                cudaStream_t stream = streams_[stream_idx];
                
                float* row_logits = d_logits_ + row * vocab_size_;
                
                scale_logits_row_kernel<<<scale_blocks, scale_threads, 0, stream>>>(
                    row_logits, inv_temp, vocab_size_);
            }
        }
        
        // Synchronize all streams after H2D and temp scaling
        for (int i = 0; i < NUM_STREAMS; ++i) {
            cudaStreamSynchronize(streams_[i]);
        }
        
        // Now run softmax in batched mode (more efficient)
        cudaStream_t main_stream = streams_[0];
        
        find_row_max_kernel<<<block_length_, threads_per_block, smem_size, main_stream>>>(
            d_logits_, d_row_max_, vocab_size_, block_length_);
        
        softmax_exp_sum_kernel<<<block_length_, threads_per_block, smem_size, main_stream>>>(
            d_logits_, d_row_max_, d_probs_, d_row_sum_, vocab_size_, block_length_);
        
        softmax_normalize_kernel<<<block_length_, threads_per_block, 0, main_stream>>>(
            d_probs_, d_row_sum_, vocab_size_, block_length_);
        
        cudaStreamSynchronize(main_stream);
        double softmax_ms = compute_timer.elapsed_ms();

        // Fast GPU sampling path: no top-k/p, no entropy, skip D2H of full probs
        const bool fast_gpu_sample = use_gpu_sampling_ &&
                                     !need_probs &&
                                     config_.top_k <= 0 &&
                                     config_.top_p >= 1.0f;
        if (fast_gpu_sample) {
            ProfilerTimer sample_timer;

            std::uniform_int_distribution<uint64_t> dist64;
            uint64_t seed = dist64(rng);
            int threads = 256;
            int blocks = (block_length_ + threads - 1) / threads;
            fill_random_kernel<<<blocks, threads, 0, main_stream>>>(d_random_vals_, seed, block_length_);

            int sample_threads = (vocab_size_ <= 4096) ? 128 : 256;
            const size_t sample_smem = sample_threads * sizeof(float) + sample_threads * sizeof(int);
            sample_tokens_kernel<<<block_length_, sample_threads, sample_smem, main_stream>>>(
                d_probs_, d_random_vals_,
                d_sampled_tokens_, d_confidences_,
                vocab_size_, block_length_,
                config_.top_k, config_.top_p
            );

            cudaMemcpyAsync(sampled_tokens.data(), d_sampled_tokens_,
                            block_length_ * sizeof(int),
                            cudaMemcpyDeviceToHost, main_stream);
            cudaMemcpyAsync(confidences.data(), d_confidences_,
                            block_length_ * sizeof(float),
                            cudaMemcpyDeviceToHost, main_stream);

            cudaStreamSynchronize(main_stream);

            const int clamp_limit = config_.n_vocab_limit > 0 ? config_.n_vocab_limit : vocab_size_;
            if (clamp_limit < vocab_size_) {
                const int clamp_id = clamp_limit - 1;
                for (int i = 0; i < block_length_; ++i) {
                    if (sampled_tokens[i] >= clamp_limit) {
                        sampled_tokens[i] = clamp_id;
                        confidences[i] = 0.0f;
                    }
                }
            }
            double sample_ms = sample_timer.elapsed_ms();

            profiler.record_custom("sampler_gpu_stage_prepare_ms", prepare_ms);
            profiler.record_custom("sampler_gpu_stage_softmax_ms", softmax_ms);
            profiler.record_custom("sampler_gpu_stage_sort_ms", 0.0);
            profiler.record_custom("sampler_gpu_stage_d2h_ms", sample_ms);
            profiler.record_custom("sampler_gpu_stage_cpu_post_ms", 0.0);

            if (stats) {
                stats->stage_prepare_ms = prepare_ms;
                stats->stage_softmax_ms = softmax_ms;
                stats->stage_sort_ms = 0.0;
                stats->stage_sample_ms = sample_ms;
                stats->stage_d2h_ms = sample_ms;
                stats->stage_cpu_post_ms = 0.0;
            }

            return true;
        }
        
        // Sort if needed
        ProfilerTimer sort_timer;
        double sort_ms = 0.0;
        
        if (need_sort) {
            // Use multiple streams for sorting different rows
            for (int row = 0; row < block_length_; ++row) {
                int stream_idx = row % NUM_STREAMS;
                cudaStream_t stream = streams_[stream_idx];
                auto policy = thrust::cuda::par.on(stream);
                
                float* row_probs = d_probs_ + row * vocab_size_;
                int* row_indices = d_indices_ + row * vocab_size_;
                
                thrust::device_ptr<int> idx_ptr(row_indices);
                thrust::sequence(policy, idx_ptr, idx_ptr + vocab_size_);
                
                thrust::device_ptr<float> prob_ptr(row_probs);
                thrust::sort_by_key(policy, prob_ptr, prob_ptr + vocab_size_, idx_ptr, thrust::greater<float>());
            }
            
            // Sync all streams after sorting
            for (int i = 0; i < NUM_STREAMS; ++i) {
                cudaStreamSynchronize(streams_[i]);
            }
        }
        sort_ms = sort_timer.elapsed_ms();

        // GPU top-k/p sampling path when无需返回 full probs
        if (need_sort && !need_probs) {
            ProfilerTimer sample_timer;
            std::uniform_int_distribution<uint64_t> dist64;
            uint64_t seed = dist64(rng);
            int threads = 256;
            int blocks = (block_length_ + threads - 1) / threads;
            fill_random_kernel<<<blocks, threads, 0, main_stream>>>(d_random_vals_, seed, block_length_);

            sample_with_topp_kernel<<<block_length_, 1, 0, main_stream>>>(
                d_probs_, d_indices_, d_random_vals_,
                d_sampled_tokens_, d_confidences_,
                vocab_size_, block_length_,
                config_.top_k, config_.top_p
            );

            cudaMemcpyAsync(sampled_tokens.data(), d_sampled_tokens_,
                            block_length_ * sizeof(int),
                            cudaMemcpyDeviceToHost, main_stream);
            cudaMemcpyAsync(confidences.data(), d_confidences_,
                            block_length_ * sizeof(float),
                            cudaMemcpyDeviceToHost, main_stream);
            cudaStreamSynchronize(main_stream);
            double sample_ms = sample_timer.elapsed_ms();

            profiler.record_custom("sampler_gpu_stage_prepare_ms", prepare_ms);
            profiler.record_custom("sampler_gpu_stage_softmax_ms", softmax_ms);
            profiler.record_custom("sampler_gpu_stage_sort_ms", sort_ms);
            profiler.record_custom("sampler_gpu_stage_d2h_ms", sample_ms);
            profiler.record_custom("sampler_gpu_stage_cpu_post_ms", 0.0);

            if (stats) {
                stats->stage_prepare_ms = prepare_ms;
                stats->stage_softmax_ms = softmax_ms;
                stats->stage_sort_ms = sort_ms;
                stats->stage_sample_ms = sample_ms;
                stats->stage_d2h_ms = sample_ms;
                stats->stage_cpu_post_ms = 0.0;
            }

            return true;
        }
        
        // D2H transfer using multiple streams
        ProfilerTimer d2h_timer;
        
        for (int row = 0; row < block_length_; ++row) {
            int stream_idx = row % NUM_STREAMS;
            cudaStream_t stream = streams_[stream_idx];
            
            float* src_probs = d_probs_ + row * vocab_size_;
            int* src_indices = d_indices_ + row * vocab_size_;
            float* dst_probs = h_pinned_probs_ + row * transfer_count;
            int* dst_indices = h_pinned_indices_ + row * transfer_count;
            
            cudaMemcpyAsync(dst_probs, src_probs, transfer_count * sizeof(float), cudaMemcpyDeviceToHost, stream);
            if (need_sort) {
                cudaMemcpyAsync(dst_indices, src_indices, transfer_count * sizeof(int), cudaMemcpyDeviceToHost, stream);
            }
        }
        
        // Sync all streams
        for (int i = 0; i < NUM_STREAMS; ++i) {
            cudaStreamSynchronize(streams_[i]);
        }
        double d2h_ms = d2h_timer.elapsed_ms();
        
        // Copy from pinned to regular host memory
        std::memcpy(host_probs_.data(), h_pinned_probs_, 
                    static_cast<size_t>(block_length_) * transfer_count * sizeof(float));
        if (need_sort) {
            std::memcpy(host_indices_.data(), h_pinned_indices_,
                        static_cast<size_t>(block_length_) * transfer_count * sizeof(int));
        }
        
        // CPU sampling
        ProfilerTimer cpu_timer;
        sample_on_cpu(need_sort, need_probs, transfer_count, rng, sampled_tokens, confidences);
        double cpu_ms = cpu_timer.elapsed_ms();
        
        // Record stats
        profiler.record_custom("sampler_gpu_stage_prepare_ms", prepare_ms);
        profiler.record_custom("sampler_gpu_stage_softmax_ms", softmax_ms);
        profiler.record_custom("sampler_gpu_stage_sort_ms", sort_ms);
        profiler.record_custom("sampler_gpu_stage_d2h_ms", d2h_ms);
        profiler.record_custom("sampler_gpu_stage_cpu_post_ms", cpu_ms);
        
        if (stats) {
            stats->stage_prepare_ms = prepare_ms;
            stats->stage_softmax_ms = softmax_ms;
            stats->stage_sort_ms = sort_ms;
            stats->stage_sample_ms = 0.0;
            stats->stage_d2h_ms = d2h_ms;
            stats->stage_cpu_post_ms = cpu_ms;
        }

        if (need_probs && token_probs) {
            *token_probs = token_probs_cache_;
        }

        return true;
    }
    
    // Helper function for CPU sampling (shared by both paths)
    void sample_on_cpu(
        bool need_sort,
        bool need_probs,
        int transfer_count,
        std::mt19937& rng,
        std::vector<llama_token>& sampled_tokens,
        std::vector<float>& confidences
    ) {
        for (int row = 0; row < block_length_; ++row) {
            float* row_probs = host_probs_.data() + row * transfer_count;
            int* row_indices = need_sort ? (host_indices_.data() + row * transfer_count) : nullptr;
            
            // Apply top-p cutoff
            int final_count = transfer_count;
            if (config_.top_p < 1.0f) {
                float cumsum = 0.0f;
                for (int i = 0; i < transfer_count; ++i) {
                    cumsum += row_probs[i];
                    if (cumsum > config_.top_p && i > 0) {
                        final_count = i + 1;
                        break;
                    }
                }
            }
            
            // Apply top-k
            if (config_.top_k > 0 && config_.top_k < final_count) {
                final_count = config_.top_k;
            }
            
            if (final_count <= 0) final_count = 1;
            
            // Renormalize for sampling
            float sum = 0.0f;
            for (int i = 0; i < final_count; ++i) {
                sum += row_probs[i];
            }
            
            // Sample
            std::uniform_real_distribution<float> dist(0.0f, sum);
            float r = dist(rng);
            float cumsum = 0.0f;
            int sampled_idx = 0;
            
            for (int i = 0; i < final_count; ++i) {
                cumsum += row_probs[i];
                if (cumsum >= r) {
                    sampled_idx = i;
                    break;
                }
            }
            
            // Get token ID
            int token_id = need_sort ? row_indices[sampled_idx] : sampled_idx;
            float prob = row_probs[sampled_idx] / sum;
            
            sampled_tokens[row] = static_cast<llama_token>(token_id);
            confidences[row] = prob;
            
            // Store full probs if needed for entropy calculation
            if (need_probs) {
                std::vector<float> full_probs(vocab_size_, 0.0f);
                for (int i = 0; i < final_count; ++i) {
                    int tid = need_sort ? row_indices[i] : i;
                    full_probs[tid] = row_probs[i] / sum;
                }
                token_probs_cache_.push_back(std::move(full_probs));
            }
        }
    }

    bool sample(
        const std::vector<float>& logits,
        RemaskingStrategy remasking_strategy,
        std::mt19937& rng,
        std::vector<llama_token>& sampled_tokens,
        std::vector<float>& confidences,
        std::vector<std::vector<float>>* token_probs,
        GpuSamplerStats* stats
    ) {
        return sample_impl(logits.data(), logits.size(), remasking_strategy, rng,
                          sampled_tokens, confidences, token_probs, stats);
    }

    bool sample_from_ptr(
        const float* logits_ptr,
        size_t logits_size,
        RemaskingStrategy remasking_strategy,
        std::mt19937& rng,
        std::vector<llama_token>& sampled_tokens,
        std::vector<float>& confidences,
        std::vector<std::vector<float>>* token_probs,
        GpuSamplerStats* stats
    ) {
        return sample_impl(logits_ptr, logits_size, remasking_strategy, rng,
                          sampled_tokens, confidences, token_probs, stats, false, false);
    }

    bool sample_from_device_ptr(
        const float* logits_ptr,
        size_t logits_size,
        RemaskingStrategy remasking_strategy,
        std::mt19937& rng,
        std::vector<llama_token>& sampled_tokens,
        std::vector<float>& confidences,
        std::vector<std::vector<float>>* token_probs,
        GpuSamplerStats* stats,
        bool force_non_fused
    ) {
        DIFF_LOGD("[GpuSampler][debug] sample_from_device_ptr logits_ptr=%p size=%zu force_non_fused=%d\n",
                  (const void*)logits_ptr, logits_size, force_non_fused ? 1 : 0);
        return sample_impl(logits_ptr, logits_size, remasking_strategy, rng,
                          sampled_tokens, confidences, token_probs, stats, true, force_non_fused);
    }

    // Scatter pointer version - avoids CPU-side concatenation
    // Directly transfers from scattered logits pointers using async H2D with multiple streams
    bool sample_from_scatter_ptrs(
        const std::vector<float*>& logits_ptrs,
        int vocab_size,
        RemaskingStrategy remasking_strategy,
        std::mt19937& rng,
        std::vector<llama_token>& sampled_tokens,
        std::vector<float>& confidences,
        std::vector<std::vector<float>>* token_probs,
        GpuSamplerStats* stats
    ) {
        if (!initialized_) {
            return false;
        }

        const int num_rows = static_cast<int>(logits_ptrs.size());
        if (num_rows != block_length_ || vocab_size != vocab_size_) {
            return false;
        }

        const bool need_probs = (remasking_strategy == RemaskingStrategy::ENTROPY_BOUNDED) && token_probs;
        token_probs_cache_.clear();
        if (need_probs) {
            token_probs_cache_.reserve(block_length_);
        }

        sampled_tokens.assign(block_length_, 0);
        confidences.assign(block_length_, 0.0f);

        DiffusionProfiler& profiler = DiffusionProfiler::instance();
        ProfilerTimer total_timer;

        const bool need_sort = (config_.top_p < 1.0f) || (config_.top_k > 0 && config_.top_k < vocab_size_);
        const float inv_temp = 1.0f / config_.temperature;
        const bool apply_temp = (config_.temperature != 1.0f);

        // Determine transfer count per row
        int transfer_count = vocab_size_;
        if (config_.top_k > 0 && config_.top_k < vocab_size_) {
            transfer_count = config_.top_k;
        }
        if (config_.top_p < 1.0f && transfer_count > 1024) {
            transfer_count = 1024;
        }

        const size_t row_bytes = static_cast<size_t>(vocab_size_) * sizeof(float);
        const int threads_per_block = 256;
        const size_t smem_size = threads_per_block * sizeof(float);
        const size_t scale_threads = 256;
        const size_t scale_blocks = (vocab_size_ + scale_threads - 1) / scale_threads;

        host_probs_.resize(static_cast<size_t>(block_length_) * transfer_count);
        host_indices_.resize(static_cast<size_t>(block_length_) * transfer_count);

        // ========== Stage 1: Direct scatter H2D transfer ==========
        ProfilerTimer prepare_timer;

        // Launch async H2D transfers for each row using multiple streams
        // No CPU-side concatenation needed!
        for (int row = 0; row < block_length_; ++row) {
            int stream_idx = row % NUM_STREAMS;
            cudaStream_t stream = streams_[stream_idx];
            
            float* src = logits_ptrs[row];  // Direct source pointer
            float* dst = d_logits_ + row * vocab_size_;
            
            cudaMemcpyAsync(dst, src, row_bytes, cudaMemcpyHostToDevice, stream);
        }

        // Launch temperature scaling kernels (overlapped with remaining H2D)
        if (apply_temp) {
            for (int row = 0; row < block_length_; ++row) {
                int stream_idx = row % NUM_STREAMS;
                cudaStream_t stream = streams_[stream_idx];
                
                float* row_logits = d_logits_ + row * vocab_size_;
                
                scale_logits_row_kernel<<<scale_blocks, scale_threads, 0, stream>>>(
                    row_logits, inv_temp, vocab_size_);
            }
        }

        // Synchronize all streams after H2D and temp scaling
        for (int i = 0; i < NUM_STREAMS; ++i) {
            cudaStreamSynchronize(streams_[i]);
        }
        double prepare_ms = prepare_timer.elapsed_ms();

        // ========== Stage 2: GPU Softmax (batched) ==========
        ProfilerTimer softmax_timer;
        cudaStream_t main_stream = streams_[0];

        find_row_max_kernel<<<block_length_, threads_per_block, smem_size, main_stream>>>(
            d_logits_, d_row_max_, vocab_size_, block_length_);

        softmax_exp_sum_kernel<<<block_length_, threads_per_block, smem_size, main_stream>>>(
            d_logits_, d_row_max_, d_probs_, d_row_sum_, vocab_size_, block_length_);

        softmax_normalize_kernel<<<block_length_, threads_per_block, 0, main_stream>>>(
            d_probs_, d_row_sum_, vocab_size_, block_length_);

        cudaStreamSynchronize(main_stream);
        double softmax_ms = softmax_timer.elapsed_ms();

        // ========== Stage 3: Sort (if needed) ==========
        ProfilerTimer sort_timer;
        double sort_ms = 0.0;

        if (need_sort) {
            for (int row = 0; row < block_length_; ++row) {
                int stream_idx = row % NUM_STREAMS;
                cudaStream_t stream = streams_[stream_idx];
                auto policy = thrust::cuda::par.on(stream);

                float* row_probs = d_probs_ + row * vocab_size_;
                int* row_indices = d_indices_ + row * vocab_size_;

                thrust::device_ptr<int> idx_ptr(row_indices);
                thrust::sequence(policy, idx_ptr, idx_ptr + vocab_size_);

                thrust::device_ptr<float> prob_ptr(row_probs);
                thrust::sort_by_key(policy, prob_ptr, prob_ptr + vocab_size_, idx_ptr, thrust::greater<float>());
            }

            for (int i = 0; i < NUM_STREAMS; ++i) {
                cudaStreamSynchronize(streams_[i]);
            }
        }
        sort_ms = sort_timer.elapsed_ms();

        // Phase 2 优化: 使用 GPU 采样（如果不需要完整概率分布）
        ProfilerTimer sample_timer;
        double sample_ms = 0.0;
        double d2h_ms = 0.0;
        double cpu_ms = 0.0;

        if (use_gpu_sampling_ && !need_probs) {
            // ========== GPU 采样路径 ==========
            // 在 CPU 生成随机数并传输到 GPU
            host_random_vals_.resize(block_length_);
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            for (int i = 0; i < block_length_; ++i) {
                host_random_vals_[i] = dist(rng);
            }
            
            cudaMemcpyAsync(d_random_vals_, host_random_vals_.data(), 
                           block_length_ * sizeof(float), cudaMemcpyHostToDevice, main_stream);
            
            // 调用 GPU 采样 kernel
            if (need_sort) {
                sample_with_topp_kernel<<<block_length_, 1, 0, main_stream>>>(
                    d_probs_, d_indices_, d_random_vals_,
                    d_sampled_tokens_, d_confidences_,
                    vocab_size_, block_length_,
                    config_.top_k, config_.top_p
                );
            } else {
                // shared memory: float[256] + int[256]
                const size_t sample_smem = 256 * sizeof(float) + 256 * sizeof(int);
                sample_tokens_kernel<<<block_length_, 256, sample_smem, main_stream>>>(
                    d_probs_, d_random_vals_,
                    d_sampled_tokens_, d_confidences_,
                    vocab_size_, block_length_,
                    config_.top_k, config_.top_p
                );
            }
            
            cudaStreamSynchronize(main_stream);
            sample_ms = sample_timer.elapsed_ms();
            
            // 只传输采样结果（大大减少 D2H 数据量）
            ProfilerTimer d2h_timer;
            std::vector<int> h_tokens(block_length_);
            std::vector<float> h_confs(block_length_);
            
            cudaMemcpy(h_tokens.data(), d_sampled_tokens_, 
                      block_length_ * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_confs.data(), d_confidences_,
                      block_length_ * sizeof(float), cudaMemcpyDeviceToHost);
            
            for (int i = 0; i < block_length_; ++i) {
                sampled_tokens[i] = static_cast<llama_token>(h_tokens[i]);
                confidences[i] = h_confs[i];
            }
            d2h_ms = d2h_timer.elapsed_ms();
            
        } else {
            // ========== 原始 CPU 采样路径（需要完整概率时使用） ==========
            ProfilerTimer d2h_timer;
            
            host_probs_.resize(static_cast<size_t>(block_length_) * transfer_count);
            host_indices_.resize(static_cast<size_t>(block_length_) * transfer_count);

            for (int row = 0; row < block_length_; ++row) {
                int stream_idx = row % NUM_STREAMS;
                cudaStream_t stream = streams_[stream_idx];

                float* src_probs = d_probs_ + row * vocab_size_;
                int* src_indices = d_indices_ + row * vocab_size_;
                float* dst_probs = host_probs_.data() + row * transfer_count;
                int* dst_indices = host_indices_.data() + row * transfer_count;

                cudaMemcpyAsync(dst_probs, src_probs, transfer_count * sizeof(float), cudaMemcpyDeviceToHost, stream);
                if (need_sort) {
                    cudaMemcpyAsync(dst_indices, src_indices, transfer_count * sizeof(int), cudaMemcpyDeviceToHost, stream);
                }
            }

            for (int i = 0; i < NUM_STREAMS; ++i) {
                cudaStreamSynchronize(streams_[i]);
            }
            d2h_ms = d2h_timer.elapsed_ms();

            ProfilerTimer cpu_timer;
            sample_on_cpu(need_sort, need_probs, transfer_count, rng, sampled_tokens, confidences);
            cpu_ms = cpu_timer.elapsed_ms();
        }

        // Record stats
        profiler.record_custom("sampler_gpu_stage_prepare_ms", prepare_ms);
        profiler.record_custom("sampler_gpu_stage_softmax_ms", softmax_ms);
        profiler.record_custom("sampler_gpu_stage_sort_ms", sort_ms);
        profiler.record_custom("sampler_gpu_stage_sample_ms", sample_ms);
        profiler.record_custom("sampler_gpu_stage_d2h_ms", d2h_ms);
        profiler.record_custom("sampler_gpu_stage_cpu_post_ms", cpu_ms);

        if (stats) {
            stats->stage_prepare_ms = prepare_ms;
            stats->stage_softmax_ms = softmax_ms;
            stats->stage_sort_ms = sort_ms;
            stats->stage_sample_ms = sample_ms;
            stats->stage_d2h_ms = d2h_ms;
            stats->stage_cpu_post_ms = cpu_ms;
        }

        if (need_probs && token_probs) {
            *token_probs = token_probs_cache_;
        }

        return true;
    }

private:
    bool init() {
        // Create multiple streams for parallel processing
        for (int i = 0; i < NUM_STREAMS; ++i) {
            if (!check_cuda(cudaStreamCreateWithFlags(&streams_[i], cudaStreamNonBlocking), "cudaStreamCreate")) {
                use_multi_stream_ = false;
                // Fallback: at least create the first stream
                if (i == 0) return false;
                break;
            }
        }
        
        const size_t total_floats = static_cast<size_t>(block_length_) * vocab_size_;
        
        // Allocate device memory
        if (!check_cuda(cudaMalloc(&d_logits_, total_floats * sizeof(float)), "cudaMalloc logits")) {
            DIFF_LOGE("[GpuSampler] init failed: d_logits_\n");
            return false;
        }
        if (!check_cuda(cudaMalloc(&d_probs_, total_floats * sizeof(float)), "cudaMalloc probs")) {
            DIFF_LOGE("[GpuSampler] init failed: d_probs_\n");
            return false;
        }
        if (!check_cuda(cudaMalloc(&d_indices_, total_floats * sizeof(int)), "cudaMalloc indices")) {
            DIFF_LOGE("[GpuSampler] init failed: d_indices_\n");
            return false;
        }
        if (!check_cuda(cudaMalloc(&d_row_max_, block_length_ * sizeof(float)), "cudaMalloc row_max")) {
            DIFF_LOGE("[GpuSampler] init failed: d_row_max_\n");
            return false;
        }
        if (!check_cuda(cudaMalloc(&d_row_sum_, block_length_ * sizeof(float)), "cudaMalloc row_sum")) {
            DIFF_LOGE("[GpuSampler] init failed: d_row_sum_\n");
            return false;
        }
        
        // Phase 2 优化: 分配 GPU 采样相关内存
        if (!check_cuda(cudaMalloc(&d_random_vals_, block_length_ * sizeof(float)), "cudaMalloc random_vals")) {
            use_gpu_sampling_ = false;
            DIFF_LOGW("[GpuSampler] disable gpu_sampling: d_random_vals_ alloc failed\n");
        }
        if (!check_cuda(cudaMalloc(&d_sampled_tokens_, block_length_ * sizeof(int)), "cudaMalloc sampled_tokens")) {
            use_gpu_sampling_ = false;
            DIFF_LOGW("[GpuSampler] disable gpu_sampling: d_sampled_tokens_ alloc failed\n");
        }
        if (!check_cuda(cudaMalloc(&d_confidences_, block_length_ * sizeof(float)), "cudaMalloc confidences")) {
            use_gpu_sampling_ = false;
            DIFF_LOGW("[GpuSampler] disable gpu_sampling: d_confidences_ alloc failed\n");
        }
        
        // Allocate pinned host memory for async transfers
        if (!check_cuda(cudaMallocHost(&h_pinned_logits_, total_floats * sizeof(float)), "cudaMallocHost logits")) {
            use_multi_stream_ = false;  // Fallback to non-pinned
            DIFF_LOGW("[GpuSampler][init][warn] cudaMallocHost logits failed, disable multi_stream\n");
        }
        if (!check_cuda(cudaMallocHost(&h_pinned_probs_, total_floats * sizeof(float)), "cudaMallocHost probs")) {
            use_multi_stream_ = false;
            DIFF_LOGW("[GpuSampler][init][warn] cudaMallocHost probs failed, disable multi_stream\n");
        }
        if (!check_cuda(cudaMallocHost(&h_pinned_indices_, total_floats * sizeof(int)), "cudaMallocHost indices")) {
            use_multi_stream_ = false;
            DIFF_LOGW("[GpuSampler][init][warn] cudaMallocHost indices failed, disable multi_stream\n");
        }
        
        return true;
    }

    int block_length_;
    int vocab_size_;
    DiffusionConfig config_;

    // Multiple streams for parallel processing
    cudaStream_t streams_[NUM_STREAMS];
    
    // Device memory
    float* d_logits_;
    float* d_probs_;
    int* d_indices_;
    float* d_row_max_;
    float* d_row_sum_;
    
    // Phase 2: GPU 采样相关设备内存
    float* d_random_vals_;
    int* d_sampled_tokens_;
    float* d_confidences_;
    
    // Pinned host memory for async transfers
    float* h_pinned_logits_;
    float* h_pinned_probs_;
    int* h_pinned_indices_;
    
    bool use_multi_stream_;
    bool use_gpu_sampling_;
    bool initialized_;

    std::vector<float> host_probs_;
    std::vector<int> host_indices_;
    std::vector<float> host_random_vals_;  // CPU 生成的随机数
    std::vector<std::vector<float>> token_probs_cache_;
};

GpuSampler::GpuSampler(int block_length, int vocab_size, const DiffusionConfig& config)
    : impl_(std::make_unique<Impl>(block_length, vocab_size, config)) {}

GpuSampler::~GpuSampler() = default;

bool GpuSampler::is_available() const {
    return impl_ && impl_->is_available();
}

bool GpuSampler::sample(
    const std::vector<float>& logits,
    RemaskingStrategy remasking_strategy,
    std::mt19937& rng,
    std::vector<llama_token>& sampled_tokens,
    std::vector<float>& confidences,
    std::vector<std::vector<float>>* token_probs,
    Stats* stats
) {
    if (!impl_) {
        return false;
    }
    return impl_->sample(logits, remasking_strategy, rng, sampled_tokens, confidences, token_probs, stats);
}

bool GpuSampler::sample_from_ptr(
    const float* logits_ptr,
    size_t logits_size,
    RemaskingStrategy remasking_strategy,
    std::mt19937& rng,
    std::vector<llama_token>& sampled_tokens,
    std::vector<float>& confidences,
    std::vector<std::vector<float>>* token_probs,
    Stats* stats
) {
    if (!impl_) {
        return false;
    }
    return impl_->sample_from_ptr(logits_ptr, logits_size, remasking_strategy, rng, 
                                   sampled_tokens, confidences, token_probs, stats);
}

bool GpuSampler::sample_from_scatter_ptrs(
    const std::vector<float*>& logits_ptrs,
    int vocab_size,
    RemaskingStrategy remasking_strategy,
    std::mt19937& rng,
    std::vector<llama_token>& sampled_tokens,
    std::vector<float>& confidences,
    std::vector<std::vector<float>>* token_probs,
    Stats* stats
) {
    if (!impl_) {
        return false;
    }
    return impl_->sample_from_scatter_ptrs(logits_ptrs, vocab_size, remasking_strategy, rng,
                                            sampled_tokens, confidences, token_probs, stats);
}

bool GpuSampler::sample_from_device_ptr(
    const float* logits_ptr,
    size_t logits_size,
    RemaskingStrategy remasking_strategy,
    std::mt19937& rng,
    std::vector<llama_token>& sampled_tokens,
    std::vector<float>& confidences,
    std::vector<std::vector<float>>* token_probs,
    Stats* stats,
    bool force_non_fused
) {
    if (!impl_) {
        return false;
    }
    return impl_->sample_from_device_ptr(logits_ptr, logits_size, remasking_strategy, rng,
                                         sampled_tokens, confidences, token_probs, stats,
                                         force_non_fused);
}

} // namespace diffusion

#endif // DIFFUSION_ENABLE_CUDA

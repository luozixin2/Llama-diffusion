#include "gpu_sampler.h"
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

// Phase 2 优化: GPU 端并行采样 kernel
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
        fprintf(stderr, "[GpuSampler] %s failed: %s\n", msg, cudaGetErrorString(err));
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
          use_multi_stream_(true),
          use_gpu_sampling_(true),
          initialized_(false) {
        for (int i = 0; i < NUM_STREAMS; ++i) {
            streams_[i] = nullptr;
        }
        initialized_ = init();
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

    bool is_available() const { return initialized_; }

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
        GpuSamplerStats* stats
    ) {
        if (!initialized_) {
            return false;
        }

        const size_t expected = static_cast<size_t>(block_length_) * vocab_size_;
        if (logits_size != expected) {
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
        
        // ========== Stage 1: H2D transfer + temperature scaling ==========
        ProfilerTimer prepare_timer;
        
        const size_t total_bytes = expected * sizeof(float);
        if (!check_cuda(cudaMemcpyAsync(d_logits_, logits_ptr, total_bytes, cudaMemcpyHostToDevice, stream), "H2D logits")) {
            return false;
        }

        if (config_.temperature != 1.0f) {
            const float inv_temp = 1.0f / config_.temperature;
            const size_t threads = 256;
            const size_t blocks = (expected + threads - 1) / threads;
            scale_logits_kernel<<<blocks, threads, 0, stream>>>(d_logits_, inv_temp, expected);
        }
        
        if (!check_cuda(cudaStreamSynchronize(stream), "sync after prep")) {
            return false;
        }
        double prepare_ms = prepare_timer.elapsed_ms();

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
        
        if (!check_cuda(cudaStreamSynchronize(stream), "sync after softmax")) {
            return false;
        }
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
            
            if (!check_cuda(cudaStreamSynchronize(stream), "sync after sort")) {
                return false;
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
                          sampled_tokens, confidences, token_probs, stats);
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
            return false;
        }
        if (!check_cuda(cudaMalloc(&d_probs_, total_floats * sizeof(float)), "cudaMalloc probs")) {
            return false;
        }
        if (!check_cuda(cudaMalloc(&d_indices_, total_floats * sizeof(int)), "cudaMalloc indices")) {
            return false;
        }
        if (!check_cuda(cudaMalloc(&d_row_max_, block_length_ * sizeof(float)), "cudaMalloc row_max")) {
            return false;
        }
        if (!check_cuda(cudaMalloc(&d_row_sum_, block_length_ * sizeof(float)), "cudaMalloc row_sum")) {
            return false;
        }
        
        // Phase 2 优化: 分配 GPU 采样相关内存
        if (!check_cuda(cudaMalloc(&d_random_vals_, block_length_ * sizeof(float)), "cudaMalloc random_vals")) {
            use_gpu_sampling_ = false;
        }
        if (!check_cuda(cudaMalloc(&d_sampled_tokens_, block_length_ * sizeof(int)), "cudaMalloc sampled_tokens")) {
            use_gpu_sampling_ = false;
        }
        if (!check_cuda(cudaMalloc(&d_confidences_, block_length_ * sizeof(float)), "cudaMalloc confidences")) {
            use_gpu_sampling_ = false;
        }
        
        // Allocate pinned host memory for async transfers
        if (!check_cuda(cudaMallocHost(&h_pinned_logits_, total_floats * sizeof(float)), "cudaMallocHost logits")) {
            use_multi_stream_ = false;  // Fallback to non-pinned
        }
        if (!check_cuda(cudaMallocHost(&h_pinned_probs_, total_floats * sizeof(float)), "cudaMallocHost probs")) {
            use_multi_stream_ = false;
        }
        if (!check_cuda(cudaMallocHost(&h_pinned_indices_, total_floats * sizeof(int)), "cudaMallocHost indices")) {
            use_multi_stream_ = false;
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

} // namespace diffusion

#endif // DIFFUSION_ENABLE_CUDA

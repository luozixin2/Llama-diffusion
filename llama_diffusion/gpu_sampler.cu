#include "gpu_sampler.h"
#include "diffusion_profiler.h"

#if defined(DIFFUSION_ENABLE_CUDA)

#include <cuda_runtime.h>
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
          h_pinned_logits_(nullptr),
          h_pinned_probs_(nullptr),
          h_pinned_indices_(nullptr),
          use_multi_stream_(true),
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
    
    // Pinned host memory for async transfers
    float* h_pinned_logits_;
    float* h_pinned_probs_;
    int* h_pinned_indices_;
    
    bool use_multi_stream_;
    bool initialized_;

    std::vector<float> host_probs_;
    std::vector<int> host_indices_;
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

} // namespace diffusion

#endif // DIFFUSION_ENABLE_CUDA

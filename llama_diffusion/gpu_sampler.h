#ifndef DIFFUSION_GPU_SAMPLER_H
#define DIFFUSION_GPU_SAMPLER_H

#include "diffusion_types.h"
#include <memory>
#include <random>
#include <vector>

namespace diffusion {

struct GpuSamplerStats {
    double stage_prepare_ms = 0.0;
    double stage_prepare_copy_ms = 0.0;
    double stage_prepare_temp_ms = 0.0;
    double stage_prepare_mask_ms = 0.0;
    double stage_prepare_rng_ms = 0.0;
    double stage_softmax_ms = 0.0;
    double stage_sort_ms = 0.0;
    double stage_sample_ms = 0.0;
    double stage_d2h_ms = 0.0;
    double stage_cpu_post_ms = 0.0;
    double stage_event_wait_ms = 0.0;
    double stage_total_ms = 0.0;        // Sum of stages above
    double stage_whole_gpu_ms = 0.0;    // CUDA event elapsed over entire device path
    double stage_whole_wall_ms = 0.0;   // CPU wall time over entire device path
    int n_vocab_limit = 0;              // effective vocab cap for masking tail tokens
    bool fast_path = false;
    bool device_logits = false;
};

#if defined(DIFFUSION_ENABLE_CUDA)

class GpuSampler {
public:
    using Stats = GpuSamplerStats;

    GpuSampler(int block_length, int vocab_size, const DiffusionConfig& config);
    ~GpuSampler();

    bool is_available() const;

    bool sample(
        const std::vector<float>& logits,
        RemaskingStrategy remasking_strategy,
        std::mt19937& rng,
        std::vector<llama_token>& sampled_tokens,
        std::vector<float>& confidences,
        std::vector<std::vector<float>>* token_probs,
        Stats* stats
    );

    // Direct pointer version - avoids copying when logits are already contiguous
    bool sample_from_ptr(
        const float* logits_ptr,
        size_t logits_size,
        RemaskingStrategy remasking_strategy,
        std::mt19937& rng,
        std::vector<llama_token>& sampled_tokens,
        std::vector<float>& confidences,
        std::vector<std::vector<float>>* token_probs,
        Stats* stats
    );

    // Scatter pointer version - directly transfers from scattered logits pointers
    // Avoids CPU-side concatenation, uses async H2D with multiple streams
    bool sample_from_scatter_ptrs(
        const std::vector<float*>& logits_ptrs,  // Array of pointers to each position's logits
        int vocab_size,
        RemaskingStrategy remasking_strategy,
        std::mt19937& rng,
        std::vector<llama_token>& sampled_tokens,
        std::vector<float>& confidences,
        std::vector<std::vector<float>>* token_probs,
        Stats* stats
    );

    // Device pointer version - when logits already on GPU (D2D copy)
    bool sample_from_device_ptr(
        const float* device_logits_ptr,
        size_t logits_size,
        RemaskingStrategy remasking_strategy,
        std::mt19937& rng,
        std::vector<llama_token>& sampled_tokens,
        std::vector<float>& confidences,
        std::vector<std::vector<float>>* token_probs,
        Stats* stats,
        bool force_non_fused = false
    );

    // Device pointer (strided) version - when logits already on GPU but rows have a stride >= vocab_size.
    // Packs each row's first vocab_size elements into a contiguous buffer (D2D 2D copy) before sampling.
    bool sample_from_device_ptr_strided(
        const float* device_logits_ptr,
        int64_t stride_tokens,
        int num_rows,
        RemaskingStrategy remasking_strategy,
        std::mt19937& rng,
        std::vector<llama_token>& sampled_tokens,
        std::vector<float>& confidences,
        std::vector<std::vector<float>>* token_probs,
        Stats* stats,
        bool force_non_fused = false
    );

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

#else

class GpuSampler {
public:
    using Stats = GpuSamplerStats;

    GpuSampler(int, int, const DiffusionConfig&) {}
    bool is_available() const { return false; }
    bool sample(
        const std::vector<float>&,
        RemaskingStrategy,
        std::mt19937&,
        std::vector<llama_token>&,
        std::vector<float>&,
        std::vector<std::vector<float>>*,
        Stats*
    ) { return false; }
    bool sample_from_ptr(
        const float*,
        size_t,
        RemaskingStrategy,
        std::mt19937&,
        std::vector<llama_token>&,
        std::vector<float>&,
        std::vector<std::vector<float>>*,
        Stats*
    ) { return false; }
    bool sample_from_scatter_ptrs(
        const std::vector<float*>&,
        int,
        RemaskingStrategy,
        std::mt19937&,
        std::vector<llama_token>&,
        std::vector<float>&,
        std::vector<std::vector<float>>*,
        Stats*
    ) { return false; }
    bool sample_from_device_ptr_strided(
        const float*,
        int64_t,
        int,
        RemaskingStrategy,
        std::mt19937&,
        std::vector<llama_token>&,
        std::vector<float>&,
        std::vector<std::vector<float>>*,
        Stats*,
        bool
    ) { return false; }
};

#endif

} // namespace diffusion

#endif // DIFFUSION_GPU_SAMPLER_H


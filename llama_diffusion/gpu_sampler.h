#ifndef DIFFUSION_GPU_SAMPLER_H
#define DIFFUSION_GPU_SAMPLER_H

#include "diffusion_types.h"
#include <memory>
#include <random>
#include <vector>

namespace diffusion {

struct GpuSamplerStats {
    double stage_prepare_ms = 0.0;
    double stage_softmax_ms = 0.0;
    double stage_sort_ms = 0.0;
    double stage_sample_ms = 0.0;
    double stage_d2h_ms = 0.0;
    double stage_cpu_post_ms = 0.0;
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
};

#endif

} // namespace diffusion

#endif // DIFFUSION_GPU_SAMPLER_H


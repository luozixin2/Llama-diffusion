#ifndef DIFFUSION_GPU_SAMPLER_H
#define DIFFUSION_GPU_SAMPLER_H

#include "diffusion_types.h"
#include <memory>
#include <random>
#include <vector>

namespace diffusion {

struct GpuSamplerStats {
    double stage_prepare_ms = 0.0;
    double stage_sort_ms = 0.0;
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
};

#endif

} // namespace diffusion

#endif // DIFFUSION_GPU_SAMPLER_H


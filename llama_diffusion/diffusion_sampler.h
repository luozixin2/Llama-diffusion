#ifndef DIFFUSION_SAMPLER_H
#define DIFFUSION_SAMPLER_H

#include "diffusion_types.h"
#include "llama.h"
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace diffusion {

class GpuSampler;

struct SamplerTelemetry {
    double gpu_logit_pack_ms = 0.0;
    int gpu_logit_pack_calls = 0;
    double gpu_invoke_ms = 0.0;
    int gpu_invoke_calls = 0;
    double gpu_stage_prepare_ms = 0.0;
    double gpu_stage_sort_ms = 0.0;
    double gpu_stage_d2h_ms = 0.0;
    double gpu_stage_cpu_post_ms = 0.0;
    int gpu_success = 0;
    int gpu_fail = 0;

    double cpu_sampling_ms = 0.0;
    int cpu_sampling_calls = 0;
    double cpu_loop_ms = 0.0;
    int cpu_loop_calls = 0;

    void reset() {
        *this = SamplerTelemetry{};
    }
};

class DiffusionSampler {
public:
    DiffusionSampler(llama_context* ctx, llama_model* model, const DiffusionConfig& config);
    ~DiffusionSampler();

    std::vector<llama_token> generate(const std::vector<llama_token>& prompt);

    void generate_stream(
        const std::vector<llama_token>& prompt,
        std::function<void(const std::vector<int>&)> callback
    );

    void reset_sampler_metrics();
    const SamplerTelemetry& get_sampler_metrics() const { return sampler_metrics_; }

//private:
    llama_context* ctx_;
    llama_model* model_;
    DiffusionConfig config_;
    std::mt19937 rng_;
    std::unique_ptr<GpuSampler> gpu_sampler_;
    bool use_gpu_sampler_ = false;
    SamplerTelemetry sampler_metrics_;

    int get_vocab_size();

    void denoise_block(
        std::vector<llama_token>& current_block,
        int block_idx,
        const std::vector<int>& num_transfer_tokens_per_step
    );

    void finalize_block(
        const std::vector<llama_token>& current_block,
        int block_idx
    );

    std::vector<int> get_num_transfer_tokens(int block_length, int steps);

    void apply_top_k(std::vector<float>& logits, int k);
    void apply_top_p(std::vector<float>& logits, float p);
    llama_token sample_token(const std::vector<float>& logits, float& prob);

    std::vector<bool> get_transfer_indices_sequential(
        const std::vector<llama_token>& block,
        const std::vector<float>& confidences,
        int num_transfer
    );

    std::vector<bool> get_transfer_indices_low_conf_static(
        const std::vector<llama_token>& block,
        const std::vector<float>& confidences,
        int num_transfer
    );

    std::vector<bool> get_transfer_indices_low_conf_dynamic(
        const std::vector<llama_token>& block,
        const std::vector<float>& confidences,
        int num_transfer
    );

    std::vector<bool> get_transfer_indices_entropy_bounded(
        const std::vector<llama_token>& block,
        const std::vector<std::vector<float>>& token_probs
    );

    bool should_stop(const std::vector<llama_token>& tokens, size_t start_idx);

protected:
    bool sample_block_tokens(
        int n_vocab,
        bool need_entropy_probs,
        std::vector<llama_token>& sampled_tokens,
        std::vector<float>& confidences,
        std::vector<std::vector<float>>* entropy_probs_storage
    );

    void sample_block_on_cpu(
        int n_vocab,
        std::vector<llama_token>& sampled_tokens,
        std::vector<float>& confidences,
        std::vector<std::vector<float>>* entropy_probs_storage
    );

    bool try_sample_with_gpu(
        int n_vocab,
        bool need_entropy_probs,
        std::vector<llama_token>& sampled_tokens,
        std::vector<float>& confidences,
        std::vector<std::vector<float>>* entropy_probs_storage
    );
};

} // namespace diffusion

#endif // DIFFUSION_SAMPLER_H
#ifndef DIFFUSION_SAMPLER_H
#define DIFFUSION_SAMPLER_H

#include "diffusion_types.h"
#include "llama.h"
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace diffusion {

class GpuSampler;

class DiffusionSampler {
public:
    DiffusionSampler(llama_context* ctx, llama_model* model, const DiffusionConfig& config);
    ~DiffusionSampler();

    std::vector<llama_token> generate(const std::vector<llama_token>& prompt);

    void generate_stream(
        const std::vector<llama_token>& prompt,
        std::function<void(const std::vector<int>&)> callback
    );

//private:
    llama_context* ctx_;
    llama_model* model_;
    DiffusionConfig config_;
    std::mt19937 rng_;
    std::unique_ptr<GpuSampler> gpu_sampler_;
    bool use_gpu_sampler_ = false;

    int get_vocab_size();

    void denoise_block(
        std::vector<llama_token>& current_block,
        int block_idx,
        const std::vector<int>& num_transfer_tokens_per_step
    );

    void finalize_block(
        const std::vector<llama_token>& current_block,
        int block_idx
    );

    std::vector<int> get_num_transfer_tokens(int block_length, int steps);

    void apply_top_k(std::vector<float>& logits, int k);
    void apply_top_p(std::vector<float>& logits, float p);
    llama_token sample_token(const std::vector<float>& logits, float& prob);

    std::vector<bool> get_transfer_indices_sequential(
        const std::vector<llama_token>& block,
        const std::vector<float>& confidences,
        int num_transfer
    );

    std::vector<bool> get_transfer_indices_low_conf_static(
        const std::vector<llama_token>& block,
        const std::vector<float>& confidences,
        int num_transfer
    );

    std::vector<bool> get_transfer_indices_low_conf_dynamic(
        const std::vector<llama_token>& block,
        const std::vector<float>& confidences,
        int num_transfer
    );

    std::vector<bool> get_transfer_indices_entropy_bounded(
        const std::vector<llama_token>& block,
        const std::vector<std::vector<float>>& token_probs
    );

    bool should_stop(const std::vector<llama_token>& tokens, size_t start_idx);
};

} // namespace diffusion

#endif // DIFFUSION_SAMPLER_H


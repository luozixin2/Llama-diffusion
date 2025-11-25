#ifndef DIFFUSION_SAMPLER_H
#define DIFFUSION_SAMPLER_H

#include "llama.h"
#include <vector>
#include <string>
#include <random>
#include <functional>

namespace diffusion {

enum class RemaskingStrategy {
    SEQUENTIAL,
    LOW_CONFIDENCE_STATIC,
    LOW_CONFIDENCE_DYNAMIC,
    ENTROPY_BOUNDED
};

struct DiffusionConfig {
    int gen_length = 128;
    int block_length = 8;
    int denoising_steps = 8;
    float temperature = 1.0f;
    int top_k = 0;
    float top_p = 1.0f;
    float confidence_threshold = 0.85f;
    float eb_threshold = 0.35f;
    llama_token mask_token_id = 0;
    RemaskingStrategy remasking_strategy = RemaskingStrategy::LOW_CONFIDENCE_DYNAMIC;
    std::vector<llama_token> stop_token_ids;
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

//private:
    llama_context* ctx_;
    llama_model* model_;
    DiffusionConfig config_;
    std::mt19937 rng_;

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

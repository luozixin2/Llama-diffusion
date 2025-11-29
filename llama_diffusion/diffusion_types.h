#ifndef DIFFUSION_TYPES_H
#define DIFFUSION_TYPES_H

#include "llama.h"
#include <vector>

namespace diffusion {

enum class RemaskingStrategy {
    SEQUENTIAL,
    LOW_CONFIDENCE_STATIC,
    LOW_CONFIDENCE_DYNAMIC,
    ENTROPY_BOUNDED,
    ITERATIVE_REFINEMENT
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
    int refinement_rounds = 3;  // 迭代细化策略的细化轮数（2步并作1步=2轮，3步并作1步=3轮，4步并作1步=4轮）
    llama_token mask_token_id = 0;
    RemaskingStrategy remasking_strategy = RemaskingStrategy::LOW_CONFIDENCE_DYNAMIC;
    std::vector<llama_token> stop_token_ids;
    bool enable_gpu_sampler = false;
};

} // namespace diffusion

#endif // DIFFUSION_TYPES_H


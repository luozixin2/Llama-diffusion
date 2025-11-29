#ifndef DIFFUSION_TYPES_H
#define DIFFUSION_TYPES_H

#include "llama.h"
#include <vector>

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
    float repetition_penalty = 1.0f;  // 重复惩罚：>1.0降低重复token的概率，<1.0增加重复token的概率
    float confidence_threshold = 0.85f;
    float eb_threshold = 0.35f;
    llama_token mask_token_id = 0;
    RemaskingStrategy remasking_strategy = RemaskingStrategy::LOW_CONFIDENCE_DYNAMIC;
    std::vector<llama_token> stop_token_ids;
    bool enable_gpu_sampler = false;
};

} // namespace diffusion

#endif // DIFFUSION_TYPES_H


#include "diffusion_sampler.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cassert>
#include <vector>

namespace diffusion {

DiffusionSampler::DiffusionSampler(llama_context* ctx, llama_model* model, const DiffusionConfig& config)
    : ctx_(ctx), model_(model), config_(config) {
    std::random_device rd;
    rng_.seed(rd());
}

DiffusionSampler::~DiffusionSampler() {}

int DiffusionSampler::get_vocab_size() {
    const llama_vocab* vocab = llama_model_get_vocab(model_);
    return llama_vocab_n_tokens(vocab);
}

std::vector<llama_token> DiffusionSampler::generate(const std::vector<llama_token>& prompt) {
    const size_t prompt_length = prompt.size();
    const int num_blocks = static_cast<int>((prompt_length + config_.gen_length + config_.block_length - 1) / config_.block_length);
    const size_t total_length = static_cast<size_t>(num_blocks) * config_.block_length;
    
    std::vector<llama_token> sequence(total_length, config_.mask_token_id);
    if (prompt_length > total_length) {
        assert(false && "Prompt length is greater than total sequence length!");
    }
    std::copy(prompt.begin(), prompt.end(), sequence.begin());

    // Prefill phase
    const int prefill_blocks = static_cast<int>(prompt_length / config_.block_length);
    const size_t prefill_length = static_cast<size_t>(prefill_blocks) * config_.block_length;

    if (prefill_length > 0) {
        llama_batch batch = llama_batch_init(static_cast<int>(prefill_length), 0, 1);

        for (size_t i = 0; i < prefill_length; i++) {
            batch.token[i] = sequence[i];
            batch.pos[i] = static_cast<llama_pos>(i);
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = false;
        }
        batch.n_tokens = static_cast<int>(prefill_length);

        if (llama_decode(ctx_, batch) != 0) {
            assert(false && "llama_decode failed in prefill phase!");
            llama_batch_free(batch);
            return {};
        }
        
        llama_batch_free(batch);
    }

    // Get number of tokens to transfer per step
    std::vector<int> num_transfer_tokens_per_step = get_num_transfer_tokens(config_.block_length, config_.denoising_steps);

    // Generation phase - iterate through blocks
    for (int block_idx = prefill_blocks; block_idx < num_blocks; block_idx++) {
        const int block_start = block_idx * config_.block_length;
        const int block_end = block_start + config_.block_length;
        
        std::vector<llama_token> current_block(
            sequence.begin() + block_start,
            sequence.begin() + block_end
        );

        // Denoising loop
        denoise_block(current_block, block_idx, num_transfer_tokens_per_step);

        // Final pass: store the denoised block in KV cache
        finalize_block(current_block, block_idx);

        // Update the main sequence
        std::copy(current_block.begin(), current_block.end(), sequence.begin() + block_start);
        
        // Check for early stopping
        if (should_stop(sequence, prompt_length)) {
            break;
        }
    }

    // Trim to desired length
    size_t final_size = prompt_length + config_.gen_length;
    if (sequence.size() > final_size) {
        sequence.resize(final_size);
    }
    
    return sequence;
}

void DiffusionSampler::generate_stream(
    const std::vector<llama_token>& prompt,
    std::function<void(const std::vector<int>&)> callback
) {
    const size_t prompt_length = prompt.size();
    const int num_blocks = static_cast<int>((prompt_length + config_.gen_length + config_.block_length - 1) / config_.block_length);
    const size_t total_length = static_cast<size_t>(num_blocks) * config_.block_length;
    
    std::vector<llama_token> sequence(total_length, config_.mask_token_id);
    if (prompt_length > total_length) {
        assert(false && "Prompt length is greater than total sequence length!");
    }
    std::copy(prompt.begin(), prompt.end(), sequence.begin());

    const int prefill_blocks = static_cast<int>(prompt_length / config_.block_length);
    const size_t prefill_length = static_cast<size_t>(prefill_blocks) * config_.block_length;

    if (prefill_length > 0) {
        llama_batch batch = llama_batch_init(static_cast<int>(prefill_length), 0, 1);

        for (size_t i = 0; i < prefill_length; i++) {
            batch.token[i] = sequence[i];
            batch.pos[i] = static_cast<llama_pos>(i);
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = false;
        }
        batch.n_tokens = static_cast<int>(prefill_length);

        if (llama_decode(ctx_, batch) != 0) {
            assert(false && "llama_decode failed in prefill phase!");
            llama_batch_free(batch);
            return;
        }
        
        llama_batch_free(batch);
    }

    std::vector<int> num_transfer_tokens_per_step = get_num_transfer_tokens(config_.block_length, config_.denoising_steps);

    for (int block_idx = prefill_blocks; block_idx < num_blocks; block_idx++) {
        const int block_start = block_idx * config_.block_length;
        const int block_end = block_start + config_.block_length;
        
        std::vector<llama_token> current_block(
            sequence.begin() + block_start,
            sequence.begin() + block_end
        );

        denoise_block(current_block, block_idx, num_transfer_tokens_per_step);
        finalize_block(current_block, block_idx);
        
        std::copy(current_block.begin(), current_block.end(), sequence.begin() + block_start);
        
        // Stream callback
        callback(std::vector<int>(current_block.begin(), current_block.end()));

        if (should_stop(sequence, prompt_length)) {
            return;
        }
    }
}

void DiffusionSampler::denoise_block(
    std::vector<llama_token>& current_block,
    int block_idx,
    const std::vector<int>& num_transfer_tokens_per_step
) {
    const int block_start = block_idx * config_.block_length;
    llama_memory_t memory = llama_get_memory(ctx_);

    for (int step = 0; step < config_.denoising_steps; step++) {
        // Check if there are still masked tokens
        bool has_mask = false;
        for (llama_token token : current_block) {
            if (token == config_.mask_token_id) {
                has_mask = true;
                break;
            }
        }
        
        if (!has_mask) {
            break;  // Early exit if no masks remain
        }

        // ✅ 关键：每次 denoising 前清除当前 block 的 KV cache
        // 这样即使我们 decode 了噪声 token，也不会永久污染 cache
        llama_memory_seq_rm(memory, 0, block_start, block_start + config_.block_length);

        // Create batch for the current block
        llama_batch batch = llama_batch_init(config_.block_length, 0, 1);

        for (int i = 0; i < config_.block_length; i++) {
            batch.token[i] = current_block[i];
            batch.pos[i] = static_cast<llama_pos>(block_start + i);
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = true;  // Need logits for sampling
        }
        batch.n_tokens = config_.block_length;

        if (llama_decode(ctx_, batch) != 0) {
            assert(false && "llama_decode failed inside denoise_block!");
            llama_batch_free(batch);
            return;
        }

        // Sample new tokens
        std::vector<llama_token> sampled_tokens(config_.block_length);
        std::vector<float> confidences(config_.block_length);
        std::vector<std::vector<float>> all_probs;

        const int n_vocab = get_vocab_size();

        for (int i = 0; i < config_.block_length; i++) {
            float* logits = llama_get_logits_ith(ctx_, i);
            if (logits == nullptr) {
                assert(false && "llama_get_logits_ith returned nullptr!");
                sampled_tokens[i] = config_.mask_token_id;
                confidences[i] = 0.0f;
                continue;
            }
            std::vector<float> logits_vec(logits, logits + n_vocab);

            // Apply sampling strategies
            if (config_.temperature != 1.0f) {
                for (float& l : logits_vec) l /= config_.temperature;
            }
            if (config_.top_k > 0) {
                apply_top_k(logits_vec, config_.top_k);
            }
            if (config_.top_p < 1.0f) {
                apply_top_p(logits_vec, config_.top_p);
            }

            float prob;
            sampled_tokens[i] = sample_token(logits_vec, prob);
            confidences[i] = prob;

            // Store probabilities for entropy-based remasking
            if (config_.remasking_strategy == RemaskingStrategy::ENTROPY_BOUNDED) {
                float max_logit_val = -INFINITY;
                for(float l : logits_vec) {
                    if(!std::isinf(l)) max_logit_val = std::max(max_logit_val, l);
                }
                float sum_exp = 0.0f;
                std::vector<float> probs(logits_vec.size());
                for (size_t j = 0; j < logits_vec.size(); j++) {
                    if (!std::isinf(logits_vec[j])) {
                        probs[j] = std::exp(logits_vec[j] - max_logit_val);
                        sum_exp += probs[j];
                    } else {
                        probs[j] = 0.0f;
                    }
                }
                if (sum_exp > 0.0f) {
                    for (float& p : probs) p /= sum_exp;
                }
                all_probs.push_back(probs);
            }
        }

        llama_batch_free(batch);

        // Determine which tokens to transfer (unmask)
        if (step >= static_cast<int>(num_transfer_tokens_per_step.size())) {
            // Should not happen, but handle gracefully
            for (int i = 0; i < config_.block_length; i++) {
                if (current_block[i] == config_.mask_token_id) {
                    current_block[i] = sampled_tokens[i];
                }
            }
            continue;
        }
        
        int num_transfer = num_transfer_tokens_per_step[step];

        std::vector<bool> transfer_indices;
        switch (config_.remasking_strategy) {
            case RemaskingStrategy::SEQUENTIAL:
                transfer_indices = get_transfer_indices_sequential(current_block, confidences, num_transfer);
                break;
            case RemaskingStrategy::LOW_CONFIDENCE_STATIC:
                transfer_indices = get_transfer_indices_low_conf_static(current_block, confidences, num_transfer);
                break;
            case RemaskingStrategy::LOW_CONFIDENCE_DYNAMIC:
                transfer_indices = get_transfer_indices_low_conf_dynamic(current_block, confidences, num_transfer);
                break;
            case RemaskingStrategy::ENTROPY_BOUNDED:
                transfer_indices = get_transfer_indices_entropy_bounded(current_block, all_probs);
                break;
            default:
                transfer_indices = get_transfer_indices_low_conf_static(current_block, confidences, num_transfer);
                break;
        }

        // Update the block with sampled tokens at selected positions
        for (int i = 0; i < config_.block_length; i++) {
            if (transfer_indices[i]) {
                current_block[i] = sampled_tokens[i];
            }
        }
        
        // ✅ 关键：denoising step 结束后，再次清除当前 block 的 cache
        // 确保不会保留基于噪声 token 的 KV
        llama_memory_seq_rm(memory, 0, block_start, block_start + config_.block_length);
    }
}

void DiffusionSampler::finalize_block(
    const std::vector<llama_token>& current_block,
    int block_idx
) {
    const int block_start = block_idx * config_.block_length;
    llama_memory_t memory = llama_get_memory(ctx_);

    // ✅ 确保清除之前可能残留的 cache
    llama_memory_seq_rm(memory, 0, block_start, block_start + config_.block_length);

    // ✅ 用最终确定的干净 token 写入 KV cache
    llama_batch batch = llama_batch_init(config_.block_length, 0, 1);

    for (int i = 0; i < config_.block_length; i++) {
        batch.token[i] = current_block[i];  // Clean tokens only
        batch.pos[i] = static_cast<llama_pos>(block_start + i);
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = false;  // Don't need logits, just storing KV cache
    }
    batch.n_tokens = config_.block_length;

    if (llama_decode(ctx_, batch) != 0) {
        assert(false && "llama_decode failed in finalize_block!");
    }

    llama_batch_free(batch);
}

std::vector<int> DiffusionSampler::get_num_transfer_tokens(int block_length, int steps) {
    std::vector<int> result;
    if (steps <= 0) return result;
    int base = block_length / steps;
    int remainder = block_length % steps;
    result.reserve(steps);
    for (int i = 0; i < steps; i++) {
        result.push_back(base + (i < remainder ? 1 : 0));
    }
    return result;
}

void DiffusionSampler::apply_top_k(std::vector<float>& logits, int k) {
    if (k <= 0 || k >= static_cast<int>(logits.size())) {
        return;
    }
    std::vector<std::pair<float, size_t>> sorted_logits;
    sorted_logits.reserve(logits.size());
    for (size_t i = 0; i < logits.size(); i++) {
        sorted_logits.push_back({logits[i], i});
    }
    std::partial_sort(sorted_logits.begin(),
                     sorted_logits.begin() + k,
                     sorted_logits.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });
    float min_value = sorted_logits[k - 1].first;
    for (size_t i = 0; i < logits.size(); i++) {
        if (logits[i] < min_value) {
            logits[i] = -INFINITY;
        }
    }
}

void DiffusionSampler::apply_top_p(std::vector<float>& logits, float p) {
    if (p >= 1.0f) return;
    std::vector<std::pair<float, size_t>> sorted_logits;
    sorted_logits.reserve(logits.size());
    for (size_t i = 0; i < logits.size(); i++) {
        sorted_logits.push_back({logits[i], i});
    }
    std::sort(sorted_logits.begin(), sorted_logits.end(),
             [](const auto& a, const auto& b) { return a.first > b.first; });
    float max_logit_val = -INFINITY;
    if (!sorted_logits.empty()) {
        max_logit_val = sorted_logits[0].first;
    }
    float sum_exp = 0.0f;
    for (const auto& pair : sorted_logits) {
        if (!std::isinf(pair.first)) {
            sum_exp += std::exp(pair.first - max_logit_val);
        }
    }
    if (sum_exp == 0.0f) return;
    float cumsum = 0.0f;
    std::vector<bool> to_remove(logits.size(), false);
    for (size_t i = 0; i < sorted_logits.size(); i++) {
        if (!std::isinf(sorted_logits[i].first)) {
            float prob = std::exp(sorted_logits[i].first - max_logit_val) / sum_exp;
            cumsum += prob;
            if (cumsum > p && i > 0) {
                for(size_t j = i; j < sorted_logits.size(); ++j) {
                    to_remove[sorted_logits[j].second] = true;
                }
                break;
            }
        }
    }
    for (size_t i = 0; i < logits.size(); i++) {
        if (to_remove[i]) {
            logits[i] = -INFINITY;
        }
    }
}

llama_token DiffusionSampler::sample_token(const std::vector<float>& logits, float& prob) {
    float max_logit_val = -INFINITY;
    for(float l : logits) {
        if(!std::isinf(l)) max_logit_val = std::max(max_logit_val, l);
    }
    float sum_exp = 0.0f;
    std::vector<float> probs(logits.size());
    for (size_t i = 0; i < logits.size(); i++) {
        if (!std::isinf(logits[i])) {
            probs[i] = std::exp(logits[i] - max_logit_val);
            sum_exp += probs[i];
        } else {
            probs[i] = 0.0f;
        }
    }
    if (sum_exp == 0.0f) {
        prob = 0.0f;
        return 0;
    }
    for (float& p : probs) {
        p /= sum_exp;
    }
    std::discrete_distribution<> dist(probs.begin(), probs.end());
    llama_token token = dist(rng_);
    prob = probs[token];
    return token;
}

std::vector<bool> DiffusionSampler::get_transfer_indices_sequential(
    const std::vector<llama_token>& block,
    const std::vector<float>& confidences,
    int num_transfer
) {
    std::vector<bool> result(block.size(), false);
    int first_mask = -1;
    for (size_t i = 0; i < block.size(); i++) {
        if (block[i] == config_.mask_token_id) {
            first_mask = static_cast<int>(i);
            break;
        }
    }
    if (first_mask >= 0) {
        int count = 0;
        for (int i = first_mask; i < static_cast<int>(block.size()) && count < num_transfer; i++) {
            if (block[i] == config_.mask_token_id) {
                result[i] = true;
                count++;
            }
        }
    }
    return result;
}

std::vector<bool> DiffusionSampler::get_transfer_indices_low_conf_static(
    const std::vector<llama_token>& block,
    const std::vector<float>& confidences,
    int num_transfer
) {
    std::vector<bool> result(block.size(), false);
    std::vector<std::pair<float, size_t>> conf_indices;
    for (size_t i = 0; i < block.size(); i++) {
        if (block[i] == config_.mask_token_id) {
            conf_indices.push_back({confidences[i], i});
        }
    }
    // Sort by confidence ascending (lowest first)
    std::sort(conf_indices.begin(), conf_indices.end());
    for (int i = 0; i < std::min(num_transfer, static_cast<int>(conf_indices.size())); i++) {
        result[conf_indices[i].second] = true;
    }
    return result;
}

std::vector<bool> DiffusionSampler::get_transfer_indices_low_conf_dynamic(
    const std::vector<llama_token>& block,
    const std::vector<float>& confidences,
    int num_transfer
) {
    std::vector<bool> result(block.size(), false);
    std::vector<std::pair<float, size_t>> conf_indices;
    int high_conf_count = 0;
    
    for (size_t i = 0; i < block.size(); i++) {
        if (block[i] == config_.mask_token_id) {
            if (confidences[i] > config_.confidence_threshold) {
                result[i] = true;
                high_conf_count++;
            }
            conf_indices.push_back({confidences[i], i});
        }
    }
    
    if (high_conf_count < num_transfer) {
        // Reset and use static strategy
        std::fill(result.begin(), result.end(), false);
        std::sort(conf_indices.begin(), conf_indices.end());
        for (int i = 0; i < std::min(num_transfer, static_cast<int>(conf_indices.size())); i++) {
            result[conf_indices[i].second] = true;
        }
    }
    return result;
}

std::vector<bool> DiffusionSampler::get_transfer_indices_entropy_bounded(
    const std::vector<llama_token>& block,
    const std::vector<std::vector<float>>& token_probs
) {
    std::vector<bool> result(block.size(), false);
    std::vector<std::pair<float, size_t>> entropy_indices;
    const float eps = 1e-12f;
    
    for (size_t i = 0; i < block.size(); i++) {
        if (block[i] == config_.mask_token_id) {
            float entropy = 0.0f;
            if (i < token_probs.size()) {
                for (float p : token_probs[i]) {
                    if (p > eps) {
                        entropy -= p * std::log(p);
                    }
                }
            }
            entropy_indices.push_back({entropy, i});
        }
    }
    
    // Sort by entropy ascending (lowest first)
    std::sort(entropy_indices.begin(), entropy_indices.end());
    
    float cumsum = 0.0f;
    for (const auto& pair : entropy_indices) {
        cumsum += pair.first;
        result[pair.second] = true;
        if (cumsum >= config_.eb_threshold) {
            break;
        }
    }
    
    // Ensure at least one token is selected
    bool any_selected = false;
    for (bool b : result) {
        if (b) {
            any_selected = true;
            break;
        }
    }
    if (!any_selected && !entropy_indices.empty()) {
        result[entropy_indices[0].second] = true;
    }
    
    return result;
}

bool DiffusionSampler::should_stop(const std::vector<llama_token>& tokens, size_t start_idx) {
    if (config_.stop_token_ids.empty()) {
        return false;
    }
    for (size_t i = start_idx; i < tokens.size(); i++) {
        for (llama_token stop_token : config_.stop_token_ids) {
            if (tokens[i] == stop_token) {
                return true;
            }
        }
    }
    return false;
}

} // namespace diffusion

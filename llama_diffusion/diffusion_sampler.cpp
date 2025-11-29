#include "diffusion_sampler.h"
#include "gpu_sampler.h"
#include "diffusion_profiler.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cassert>
#include <vector>
#include <unordered_map>

namespace diffusion {

DiffusionSampler::DiffusionSampler(llama_context* ctx, llama_model* model, const DiffusionConfig& config)
    : ctx_(ctx), model_(model), config_(config) {
    std::random_device rd;
    rng_.seed(rd());

    reset_sampler_metrics();

    if (config_.enable_gpu_sampler) {
        const int vocab_size = get_vocab_size();
        gpu_sampler_ = std::make_unique<GpuSampler>(config_.block_length, vocab_size, config_);
        if (gpu_sampler_ && gpu_sampler_->is_available()) {
            use_gpu_sampler_ = true;
        }
    }
}

DiffusionSampler::~DiffusionSampler() {}

void DiffusionSampler::reset_sampler_metrics() {
    sampler_metrics_.reset();
}

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

        // 收集已生成的token作为repetition_penalty的历史（当前block中非mask的token）
        std::vector<llama_token> prev_tokens;
        for (llama_token token : current_block) {
            if (token != config_.mask_token_id) {
                prev_tokens.push_back(token);
            }
        }

        const int n_vocab = get_vocab_size();
        const bool need_entropy_probs = (config_.remasking_strategy == RemaskingStrategy::ENTROPY_BOUNDED);
        std::vector<std::vector<float>>* entropy_ptr = need_entropy_probs ? &all_probs : nullptr;
        sample_block_tokens(
            n_vocab,
            need_entropy_probs,
            sampled_tokens,
            confidences,
            entropy_ptr,
            prev_tokens
        );

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
            case RemaskingStrategy::ITERATIVE_REFINEMENT:
                // 迭代细化策略：在一步中进行多轮细化，相当于两步并作一步
                transfer_indices = get_transfer_indices_iterative_refinement_internal(
                    current_block, sampled_tokens, confidences, num_transfer, step, 
                    config_.denoising_steps, block_start, memory);
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

void DiffusionSampler::apply_repetition_penalty(std::vector<float>& logits, const std::vector<llama_token>& prev_tokens, float penalty) {
    if (penalty == 1.0f || prev_tokens.empty()) {
        return;  // 无惩罚或没有历史token
    }
    
    // 统计历史token的出现频率
    std::unordered_map<llama_token, int> token_counts;
    for (llama_token token : prev_tokens) {
        token_counts[token]++;
    }
    
    // 对出现过的token应用惩罚
    for (const auto& pair : token_counts) {
        llama_token token = pair.first;
        int count = pair.second;
        if (token >= 0 && static_cast<size_t>(token) < logits.size()) {
            // 惩罚公式：logits[token] = logits[token] / (penalty ^ count)
            // 如果penalty > 1.0，重复次数越多，logits降低越多
            if (logits[token] > 0.0f) {
                logits[token] = logits[token] / std::pow(penalty, count);
            } else {
                logits[token] = logits[token] * std::pow(penalty, count);
            }
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

bool DiffusionSampler::sample_block_tokens(
    int n_vocab,
    bool need_entropy_probs,
    std::vector<llama_token>& sampled_tokens,
    std::vector<float>& confidences,
    std::vector<std::vector<float>>* entropy_probs_storage,
    const std::vector<llama_token>& prev_tokens
) {
    diffusion::ProfilerTimer total_timer;

    if (try_sample_with_gpu(
            n_vocab,
            need_entropy_probs,
            sampled_tokens,
            confidences,
            entropy_probs_storage)) {
        DiffusionProfiler::instance().record_custom(
            "sampler_gpu_total_ms",
            total_timer.elapsed_ms()
        );
        return true;
    }

    sample_block_on_cpu(
        n_vocab,
        sampled_tokens,
        confidences,
        entropy_probs_storage,
        prev_tokens
    );
    DiffusionProfiler::instance().record_custom(
        "sampler_cpu_sampling_ms",
        total_timer.elapsed_ms()
    );
    sampler_metrics_.cpu_sampling_ms += total_timer.elapsed_ms();
    sampler_metrics_.cpu_sampling_calls++;
    return false;
}

void DiffusionSampler::sample_block_on_cpu(
    int n_vocab,
    std::vector<llama_token>& sampled_tokens,
    std::vector<float>& confidences,
    std::vector<std::vector<float>>* entropy_probs_storage,
    const std::vector<llama_token>& prev_tokens
) {
    diffusion::ProfilerTimer cpu_timer;
    if (entropy_probs_storage) {
        entropy_probs_storage->clear();
        entropy_probs_storage->reserve(config_.block_length);
    }

    for (int i = 0; i < config_.block_length; i++) {
        float* logits = llama_get_logits_ith(ctx_, i);
        if (logits == nullptr) {
            sampled_tokens[i] = config_.mask_token_id;
            confidences[i] = 0.0f;
            continue;
        }
        std::vector<float> logits_vec(logits, logits + n_vocab);

        if (config_.temperature != 1.0f) {
            for (float& l : logits_vec) l /= config_.temperature;
        }
        
        // 应用repetition_penalty（在top_k和top_p之前应用）
        if (config_.repetition_penalty != 1.0f && !prev_tokens.empty()) {
            apply_repetition_penalty(logits_vec, prev_tokens, config_.repetition_penalty);
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

        if (entropy_probs_storage) {
            float max_logit_val = -INFINITY;
            for (float l : logits_vec) {
                if (!std::isinf(l)) {
                    max_logit_val = std::max(max_logit_val, l);
                }
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
                for (float& p : probs) {
                    p /= sum_exp;
                }
            }
            entropy_probs_storage->push_back(std::move(probs));
        }
    }

    DiffusionProfiler::instance().record_custom(
        "sampler_cpu_loop_ms",
        cpu_timer.elapsed_ms()
    );
    sampler_metrics_.cpu_loop_ms += cpu_timer.elapsed_ms();
    sampler_metrics_.cpu_loop_calls++;
}

bool DiffusionSampler::try_sample_with_gpu(
    int n_vocab,
    bool need_entropy_probs,
    std::vector<llama_token>& sampled_tokens,
    std::vector<float>& confidences,
    std::vector<std::vector<float>>* entropy_probs_storage
) {
    if (!use_gpu_sampler_ || !gpu_sampler_) {
        return false;
    }

    // 使用直接指针访问，避免内存拷贝 - Phase 3 优化
    diffusion::ProfilerTimer pack_timer;
    std::vector<float*> logits_ptrs(config_.block_length);
    for (int i = 0; i < config_.block_length; ++i) {
        float* logits = llama_get_logits_ith(ctx_, i);
        if (logits == nullptr) {
            use_gpu_sampler_ = false;
            return false;
        }
        logits_ptrs[i] = logits;
    }
    DiffusionProfiler::instance().record_custom(
        "sampler_gpu_logit_pack_ms",
        pack_timer.elapsed_ms()
    );
    sampler_metrics_.gpu_logit_pack_ms += pack_timer.elapsed_ms();
    sampler_metrics_.gpu_logit_pack_calls++;

    // 使用 sample_from_ptr 直接传递指针数组
    std::vector<float> logits_flat;
    if (config_.block_length > 0) {
        // 将所有 logits 拼接成一个连续数组
        size_t total_size = static_cast<size_t>(config_.block_length) * n_vocab;
        logits_flat.resize(total_size);
        size_t offset = 0;
        for (int i = 0; i < config_.block_length; ++i) {
            std::copy(logits_ptrs[i], logits_ptrs[i] + n_vocab, logits_flat.begin() + offset);
            offset += n_vocab;
        }
    }

    std::vector<std::vector<float>> tmp_probs;
    std::vector<std::vector<float>>* probs_ptr = (need_entropy_probs && entropy_probs_storage)
        ? &tmp_probs
        : nullptr;

    diffusion::ProfilerTimer gpu_timer;
    GpuSampler::Stats gpu_stats{};
    bool sampled_with_gpu = gpu_sampler_->sample_from_ptr(
        logits_flat.data(),  // 直接传递连续内存指针
        static_cast<size_t>(config_.block_length) * n_vocab,
        config_.remasking_strategy,
        rng_,
        sampled_tokens,
        confidences,
        probs_ptr,
        &gpu_stats
    );
    
    DiffusionProfiler::instance().record_custom(
        "sampler_gpu_invoke_ms",
        gpu_timer.elapsed_ms()
    );
    sampler_metrics_.gpu_invoke_ms += gpu_timer.elapsed_ms();
    sampler_metrics_.gpu_invoke_calls++;

    if (!sampled_with_gpu) {
        use_gpu_sampler_ = false;
        sampler_metrics_.gpu_fail++;
        return false;
    }

    sampler_metrics_.gpu_success++;
    sampler_metrics_.gpu_stage_prepare_ms += gpu_stats.stage_prepare_ms;
    sampler_metrics_.gpu_stage_softmax_ms += gpu_stats.stage_softmax_ms;
    sampler_metrics_.gpu_stage_sort_ms += gpu_stats.stage_sort_ms;
    sampler_metrics_.gpu_stage_sample_ms += gpu_stats.stage_sample_ms;
    sampler_metrics_.gpu_stage_d2h_ms += gpu_stats.stage_d2h_ms;
    sampler_metrics_.gpu_stage_cpu_post_ms += gpu_stats.stage_cpu_post_ms;

    if (need_entropy_probs && entropy_probs_storage && probs_ptr) {
        *entropy_probs_storage = std::move(tmp_probs);
    }
    return true;
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

std::vector<bool> DiffusionSampler::get_transfer_indices_iterative_refinement_internal(
    const std::vector<llama_token>& block,
    std::vector<llama_token>& sampled_tokens,  // 非const，用于更新细化后的token
    const std::vector<float>& confidences,
    int num_transfer,
    int current_step,
    int total_steps,
    int block_start,
    llama_memory_t memory
) {
    std::vector<bool> result(block.size(), false);
    std::vector<llama_token> working_block = block;
    int total_transferred = 0;
    const int max_refinement_rounds = config_.refinement_rounds;  // 使用配置的细化轮数
    
    // 动态阈值：根据剩余步骤调整
    int remaining_steps = total_steps - current_step;
    float base_threshold = config_.confidence_threshold;
    float min_threshold = 0.5f;
    float dynamic_threshold = (remaining_steps == 1) ? min_threshold : 
                             base_threshold * (static_cast<float>(remaining_steps - 1) / static_cast<float>(total_steps - 1)) + 
                             min_threshold * (1.0f - static_cast<float>(remaining_steps - 1) / static_cast<float>(total_steps - 1));
    
    // 第一轮：转移高置信度token（基于初始预测）
    std::vector<std::pair<float, size_t>> initial_candidates;
    for (size_t i = 0; i < block.size(); i++) {
        if (block[i] == config_.mask_token_id) {
            initial_candidates.push_back({confidences[i], i});
        }
    }
    std::sort(initial_candidates.begin(), initial_candidates.end(), 
             [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // 第一轮转移：转移所有超过阈值的token
    for (const auto& pair : initial_candidates) {
        if (total_transferred >= num_transfer) break;
        if (pair.first >= dynamic_threshold) {
            result[pair.second] = true;
            working_block[pair.second] = sampled_tokens[pair.second];
            total_transferred++;
        }
    }
    
    // 如果第一轮已经转移了足够的token，直接返回
    if (total_transferred >= num_transfer) {
        return result;
    }
    
    // 多轮细化：每轮转移token后，重新评估剩余token
    std::vector<float> refined_confidences = confidences;  // 存储细化后的置信度
    for (int round = 1; round < max_refinement_rounds && total_transferred < num_transfer; round++) {
        // 检查是否还有masked token
        bool has_remaining = false;
        for (llama_token token : working_block) {
            if (token == config_.mask_token_id) {
                has_remaining = true;
                break;
            }
        }
        if (!has_remaining) break;
        
        // 进行refinement decode以更新剩余token的预测
        llama_memory_seq_rm(memory, 0, block_start, block_start + config_.block_length);
        
        llama_batch refinement_batch = llama_batch_init(config_.block_length, 0, 1);
        for (int i = 0; i < config_.block_length; i++) {
            refinement_batch.token[i] = working_block[i];
            refinement_batch.pos[i] = static_cast<llama_pos>(block_start + i);
            refinement_batch.n_seq_id[i] = 1;
            refinement_batch.seq_id[i][0] = 0;
            refinement_batch.logits[i] = (working_block[i] == config_.mask_token_id);
        }
        refinement_batch.n_tokens = config_.block_length;
        
        if (llama_decode(ctx_, refinement_batch) != 0) {
            llama_batch_free(refinement_batch);
            break;
        }
        
        // 重新计算剩余masked token的置信度并重新采样
        const int n_vocab = get_vocab_size();
        std::vector<std::pair<float, size_t>> refined_candidates;
        std::vector<llama_token> refined_tokens(working_block.size(), 0);  // 存储细化后重新采样的token
        
        // 收集已转移的token作为repetition_penalty的历史
        std::vector<llama_token> prev_tokens_refinement;
        for (size_t i = 0; i < working_block.size(); i++) {
            if (result[i] && working_block[i] != config_.mask_token_id) {
                prev_tokens_refinement.push_back(working_block[i]);
            }
        }
        
        for (size_t i = 0; i < working_block.size(); i++) {
            if (working_block[i] == config_.mask_token_id && !result[i]) {
                float* logits = llama_get_logits_ith(ctx_, i);
                if (logits != nullptr) {
                    std::vector<float> logits_vec(logits, logits + n_vocab);
                    if (config_.temperature != 1.0f) {
                        for (float& l : logits_vec) l /= config_.temperature;
                    }
                    
                    // 应用repetition_penalty（在top_k和top_p之前应用）
                    if (config_.repetition_penalty != 1.0f && !prev_tokens_refinement.empty()) {
                        apply_repetition_penalty(logits_vec, prev_tokens_refinement, config_.repetition_penalty);
                    }
                    
                    // 应用top_k和top_p
                    if (config_.top_k > 0 && config_.top_k < n_vocab) {
                        apply_top_k(logits_vec, config_.top_k);
                    }
                    if (config_.top_p < 1.0f) {
                        apply_top_p(logits_vec, config_.top_p);
                    }
                    
                    float max_logit = *std::max_element(logits_vec.begin(), logits_vec.end());
                    float sum_exp = 0.0f;
                    for (float l : logits_vec) {
                        sum_exp += std::exp(l - max_logit);
                    }
                    
                    // 重新采样token
                    float prob;
                    llama_token new_token = sample_token(logits_vec, prob);
                    refined_tokens[i] = new_token;
                    sampled_tokens[i] = new_token;  // 更新sampled_tokens，使其包含细化后的token
                    
                    int max_idx = std::max_element(logits_vec.begin(), logits_vec.end()) - logits_vec.begin();
                    float new_conf = std::exp(logits_vec[max_idx] - max_logit) / sum_exp;
                    refined_confidences[i] = new_conf;
                    refined_candidates.push_back({new_conf, i});
                } else {
                    refined_candidates.push_back({refined_confidences[i], i});
                    refined_tokens[i] = sampled_tokens[i];  // 回退到原始采样
                }
            }
        }
        
        llama_batch_free(refinement_batch);
        
        if (refined_candidates.empty()) break;
        
        // 按细化后的置信度降序排序
        std::sort(refined_candidates.begin(), refined_candidates.end(), 
                 [](const auto& a, const auto& b) { return a.first > b.first; });
        
        // 转移细化后高置信度的token（使用重新采样的token）
        int transferred_this_round = 0;
        for (const auto& pair : refined_candidates) {
            if (total_transferred >= num_transfer) break;
            // 使用更宽松的阈值，因为这是细化后的预测
            float round_threshold = dynamic_threshold * 0.8f;  // 稍微降低阈值
            if (pair.first >= round_threshold || transferred_this_round == 0) {
                result[pair.second] = true;
                working_block[pair.second] = refined_tokens[pair.second];  // 使用细化后重新采样的token
                total_transferred++;
                transferred_this_round++;
            }
        }
        
        // 如果这一轮没有转移token，退出循环
        if (transferred_this_round == 0) break;
    }
    
    // 确保至少转移了num_transfer个token（如果还有剩余）
    if (total_transferred < num_transfer) {
        std::vector<std::pair<float, size_t>> remaining;
        for (size_t i = 0; i < block.size(); i++) {
            if (block[i] == config_.mask_token_id && !result[i]) {
                // 使用细化后的置信度（如果可用），否则使用原始置信度
                float conf = (i < refined_confidences.size()) ? refined_confidences[i] : confidences[i];
                remaining.push_back({conf, i});
            }
        }
        std::sort(remaining.begin(), remaining.end(), 
                 [](const auto& a, const auto& b) { return a.first > b.first; });  // 按置信度降序
        int need_more = num_transfer - total_transferred;
        for (int i = 0; i < std::min(need_more, static_cast<int>(remaining.size())); i++) {
            result[remaining[i].second] = true;
        }
    }
    
    // 在最后一步，确保所有剩余的masked token都被转移
    if (remaining_steps == 1) {
        for (size_t i = 0; i < block.size(); i++) {
            if (block[i] == config_.mask_token_id && !result[i]) {
                result[i] = true;
            }
        }
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

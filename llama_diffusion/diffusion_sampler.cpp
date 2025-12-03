#include "diffusion_sampler.h"
#include "gpu_sampler.h"
#include "diffusion_profiler.h"
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
    PROFILE_SECTION("total_generation_stream");
    
    const size_t prompt_length = prompt.size();
    
    // Get context size limit
    const uint32_t ctx_size_u = llama_n_ctx(ctx_);
    const int ctx_size = static_cast<int>(ctx_size_u);
    if (ctx_size == 0) {
        return;  // Invalid context
    }
    
    const int num_blocks = static_cast<int>((prompt_length + config_.gen_length + config_.block_length - 1) / config_.block_length);
    const size_t total_length = static_cast<size_t>(num_blocks) * config_.block_length;
    
    // Ensure total length doesn't exceed context size
    if (total_length > static_cast<size_t>(ctx_size)) {
        return;  // Sequence too long for context
    }
    
    std::vector<llama_token> sequence(total_length, config_.mask_token_id);
    if (prompt_length > total_length) {
        assert(false && "Prompt length is greater than total sequence length!");
        return;
    }
    std::copy(prompt.begin(), prompt.end(), sequence.begin());

    const int prefill_blocks = static_cast<int>(prompt_length / config_.block_length);
    const size_t prefill_length = static_cast<size_t>(prefill_blocks) * config_.block_length;
    DiffusionProfiler::instance().record_custom("num_blocks", num_blocks);
    DiffusionProfiler::instance().record_custom("total_length", total_length);
    DiffusionProfiler::instance().record_custom("prompt_length", prompt_length);
    DiffusionProfiler::instance().record_custom("prefill_length", prefill_length);

    // Prefill: process all prompt tokens in one batch (llama.cpp will handle internal batching)
    if (prefill_length > 0) {
        PROFILE_SECTION("prefill_phase");
        
        {
            PROFILE_SECTION("prefill_batch_preparation");
            llama_batch batch = llama_batch_init(static_cast<int>(prefill_length), 0, 1);

            for (size_t i = 0; i < prefill_length; i++) {
                batch.token[i] = sequence[i];
                batch.pos[i] = static_cast<llama_pos>(i);
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = false;  // No logits needed for prefill
            }
            batch.n_tokens = static_cast<int>(prefill_length);

            {
                PROFILE_SECTION("prefill_llama_decode");
                if (llama_decode(ctx_, batch) != 0) {
                    llama_batch_free(batch);
                    return;  // Decode failed
                }
            }
            
            llama_batch_free(batch);
        }
        
        DiffusionProfiler::instance().record_custom("prefill_tokens", prefill_length);
    }

    std::vector<int> num_transfer_tokens_per_step = get_num_transfer_tokens(config_.block_length, config_.denoising_steps);

    // Generation phase
    {
        PROFILE_SECTION("generation_phase");
        
        for (int block_idx = prefill_blocks; block_idx < num_blocks; block_idx++) {
            std::string block_section = "block_" + std::to_string(block_idx);
            PROFILE_SECTION(block_section.c_str());
            
            const int block_start = block_idx * config_.block_length;
            const int block_end = block_start + config_.block_length;
            
            std::vector<llama_token> current_block(
                sequence.begin() + block_start,
                sequence.begin() + block_end
            );

            denoise_block(current_block, block_idx, num_transfer_tokens_per_step);
            
            {
                PROFILE_SECTION("finalize_block");
                finalize_block(current_block, block_idx);
            }
            
            std::copy(current_block.begin(), current_block.end(), sequence.begin() + block_start);
            
            {
                PROFILE_SECTION("stream_callback");
                callback(std::vector<int>(current_block.begin(), current_block.end()));
            }

            if (should_stop(sequence, prompt_length)) {
                DiffusionProfiler::instance().record_custom("early_stop_block", block_idx);
                break;
            }
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
        const bool need_entropy_probs = (config_.remasking_strategy == RemaskingStrategy::ENTROPY_BOUNDED);
        std::vector<std::vector<float>>* entropy_ptr = need_entropy_probs ? &all_probs : nullptr;
        sample_block_tokens(
            n_vocab,
            need_entropy_probs,
            sampled_tokens,
            confidences,
            entropy_ptr
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
        
        // ❌ 删除这里的 llama_memory_seq_rm
        // 下一次循环开始时会清除，或者函数结束后 finalize_block 会清除。
        // 这里清除是多余的。
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
        
        // ✅ 关键修改：改为 true
        // 即使我们不需要 logits，这也强制 llama.cpp 执行完整的生成路径计算，
        // 确保 KV Cache 的写入方式与 denoise 阶段完全一致，避免潜在的 Mask 或优化路径差异。
        batch.logits[i] = true;
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

bool DiffusionSampler::sample_block_tokens(
    int n_vocab,
    bool need_entropy_probs,
    std::vector<llama_token>& sampled_tokens,
    std::vector<float>& confidences,
    std::vector<std::vector<float>>* entropy_probs_storage
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
        entropy_probs_storage
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
    std::vector<std::vector<float>>* entropy_probs_storage
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
    // ✅ 修复：按置信度降序排序，选择最高置信度的 token（与 Python 版本一致）
    // Python 使用 torch.topk 返回最大的 k 个值
    std::sort(conf_indices.begin(), conf_indices.end(), 
              [](const auto& a, const auto& b) { return a.first > b.first; });
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
        // Reset and use static strategy with highest confidence first
        std::fill(result.begin(), result.end(), false);
        // ✅ 修复：按置信度降序排序，选择最高置信度的 token（与 Python 版本一致）
        std::sort(conf_indices.begin(), conf_indices.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });
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

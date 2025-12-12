// diffusion_sampler_profiled.cpp
#include "diffusion_sampler_profiled.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cassert>
#include "diffusion_logging.h"

namespace diffusion {

std::vector<llama_token> DiffusionSamplerProfiled::generate_internal_profiled(
    const std::vector<llama_token>& prompt
) {
    PROFILE_SECTION("setup_and_initialization");
    
    const size_t prompt_length = prompt.size();
    const int num_blocks = static_cast<int>((prompt_length + config_.gen_length + config_.block_length - 1) / config_.block_length);
    const size_t total_length = static_cast<size_t>(num_blocks) * config_.block_length;
    
    std::vector<llama_token> sequence(total_length, config_.mask_token_id);
    std::copy(prompt.begin(), prompt.end(), sequence.begin());
    
    DiffusionProfiler::instance().record_custom("num_blocks", num_blocks);
    DiffusionProfiler::instance().record_custom("total_length", total_length);
    DiffusionProfiler::instance().record_custom("prompt_length", prompt_length);

    auto env_flag = [](const char* key) {
        const char* v = std::getenv(key);
        if (!v) return false;
        if (v[0] == '0' || v[0] == 'f' || v[0] == 'F' || v[0] == 'n' || v[0] == 'N') return false;
        return true;
    };
    DiffusionProfiler::instance().record_custom("flag_disable_gpu_sampler", env_flag("DIFFUSION_DISABLE_GPU_SAMPLER") ? 1.0 : 0.0);
    DiffusionProfiler::instance().record_custom("flag_force_cpu_sampling", env_flag("DIFFUSION_FORCE_CPU_SAMPLING") ? 1.0 : 0.0);
    DiffusionProfiler::instance().record_custom("flag_disable_partial_kv", env_flag("DIFFUSION_DISABLE_PARTIAL_KV_REUSE") ? 1.0 : 0.0);
    DiffusionProfiler::instance().record_custom("flag_force_full_decode", env_flag("DIFFUSION_FORCE_FULL_BLOCK_DECODE") ? 1.0 : 0.0);
    
    const int prefill_blocks = static_cast<int>(prompt_length / config_.block_length);
    const size_t prefill_length = static_cast<size_t>(prefill_blocks) * config_.block_length;
    
    DiffusionProfiler::instance().end_section("setup_and_initialization");
    
    // Prefill phase
    if (prefill_length > 0) {
        PROFILE_SECTION("prefill_phase");
        
        llama_batch batch = llama_batch_init(static_cast<int>(prefill_length), 0, 1);
        
        {
            PROFILE_SECTION("prefill_batch_preparation");
            for (size_t i = 0; i < prefill_length; i++) {
                batch.token[i] = sequence[i];
                batch.pos[i] = static_cast<llama_pos>(i);
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = false;
            }
            batch.n_tokens = static_cast<int>(prefill_length);
        }
        
        {
            PROFILE_SECTION("prefill_llama_decode");
            if (llama_decode(ctx_, batch) != 0) {
                assert(false && "llama_decode failed in prefill phase!");
                llama_batch_free(batch);
                return {};
            }
        }
        
        llama_batch_free(batch);
        
        DiffusionProfiler::instance().record_custom("prefill_tokens", prefill_length);
    }
    
    // Get number of tokens to transfer per step
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
            
            // Denoising loop
            denoise_block_profiled(current_block, block_idx, num_transfer_tokens_per_step);
            
            // Final pass
            {
                PROFILE_SECTION("finalize_block");
                finalize_block(current_block, block_idx);
            }
            
            std::copy(current_block.begin(), current_block.end(), sequence.begin() + block_start);
            
            if (should_stop(sequence, prompt_length)) {
                DiffusionProfiler::instance().record_custom("early_stop_block", block_idx);
                break;
            }
        }
    }
    
    // Trim to desired length
    size_t final_size = prompt_length + config_.gen_length;
    if (sequence.size() > final_size) {
        sequence.resize(final_size);
    }
    
    return sequence;
}

void DiffusionSamplerProfiled::denoise_block_profiled(
    std::vector<llama_token>& current_block,
    int block_idx,
    const std::vector<int>& num_transfer_tokens_per_step
) {
    const int block_start = block_idx * config_.block_length;
    const int micro_size = config_.micro_block_size;
    const int micro_count = config_.block_length / micro_size;
    llama_memory_t memory = llama_get_memory(ctx_);

    int remaining_masks = 0;
    std::vector<int> masks_per_micro(micro_count, 0);
    for (int i = 0; i < config_.block_length; ++i) {
        if (current_block[i] == config_.mask_token_id) {
            remaining_masks++;
            masks_per_micro[i / micro_size]++;
        }
    }

    // partial KV/logits reuse 需要跨 step 追踪“哪些微块的 token 在上一轮被更新过”
    std::vector<char> dirty_micros(static_cast<size_t>(micro_count), 1);

    for (int step = 0; step < config_.denoising_steps; step++) {
        std::string step_section = "denoising_step_" + std::to_string(step);
        PROFILE_SECTION(step_section.c_str());
        if (remaining_masks == 0) {
            DiffusionProfiler::instance().record_custom("early_exit_step", step);
            break;
        }

        // 收集活跃微块和位置
        std::vector<int> active_positions;
        std::vector<int> active_micros;
        for (int m = 0; m < micro_count; ++m) {
            if (masks_per_micro[m] == 0) continue;
            active_micros.push_back(m);
            const int base = m * micro_size;
            for (int j = 0; j < micro_size; ++j) {
                const int pos = base + j;
                if (current_block[pos] == config_.mask_token_id) {
                    active_positions.push_back(pos);
                }
            }
        }
        if (active_positions.empty()) {
            DiffusionProfiler::instance().record_custom("early_exit_step", step);
            break;
        }

        const int active_count = static_cast<int>(active_positions.size());
        // active position -> index mapping (avoids O(block_len * active_count) scans later)
        std::vector<int> active_pos2idx(config_.block_length, -1);
        for (int i = 0; i < active_count; ++i) {
            const int pos = active_positions[i];
            if (pos >= 0 && pos < config_.block_length) active_pos2idx[pos] = i;
        }
        const bool need_entropy_probs = (config_.remasking_strategy == RemaskingStrategy::ENTROPY_BOUNDED);
        auto env_flag = [](const char* key) {
            const char* v = std::getenv(key);
            if (!v) return false;
            if (v[0] == '0' || v[0] == 'f' || v[0] == 'F' || v[0] == 'n' || v[0] == 'N') return false;
            return true;
        };

        // 需要清理/重算的微块：仍含 mask 的微块 + 上一轮 token 发生过更新的微块
        std::vector<int> micros_to_decode;
        micros_to_decode.reserve(static_cast<size_t>(micro_count));
        for (int m = 0; m < micro_count; ++m) {
            if (masks_per_micro[m] > 0 || dirty_micros[static_cast<size_t>(m)]) {
                micros_to_decode.push_back(m);
            }
        }
        std::vector<int> decode_positions;
        decode_positions.reserve(static_cast<size_t>(micros_to_decode.size()) * static_cast<size_t>(micro_size));
        for (int m : micros_to_decode) {
            const int base = m * micro_size;
            for (int j = 0; j < micro_size; ++j) {
                decode_positions.push_back(base + j);
            }
        }
        const int decode_count = static_cast<int>(decode_positions.size());

        bool do_partial_kv = env_flag("DIFFUSION_PARTIAL_KV_REUSE")
            && !env_flag("DIFFUSION_DISABLE_PARTIAL_KV_REUSE")
            && !env_flag("DIFFUSION_FORCE_FULL_BLOCK_DECODE")
            && !need_entropy_probs
            && config_.top_k <= 0
            && config_.top_p >= 1.0f
            && !use_gpu_sampler_
            && active_count < config_.block_length
            && decode_count > 0
            && decode_count < config_.block_length;

        // KV 清理（整块 or 仅需要重算的微块）
        {
            PROFILE_SECTION("kv_cache_clear");
            if (do_partial_kv) {
                sampler_metrics_.partial_kv_attempt++;
                for (int m : micros_to_decode) {
                    const int start_pos = block_start + m * micro_size;
                    llama_memory_seq_rm(memory, 0, start_pos, start_pos + micro_size);
                }
            } else {
                llama_memory_seq_rm(memory, 0, block_start, block_start + config_.block_length);
            }
        }

        llama_batch batch = llama_batch_init(do_partial_kv ? decode_count : config_.block_length, 0, 1);
        {
            PROFILE_SECTION("batch_preparation");
            if (do_partial_kv) {
                for (int i = 0; i < decode_count; i++) {
                    const int pos = decode_positions[i];
                    batch.token[i] = current_block[pos];
                    batch.pos[i] = static_cast<llama_pos>(block_start + pos);
                    batch.n_seq_id[i] = 1;
                    batch.seq_id[i][0] = 0;
                    batch.logits[i] = (current_block[pos] == config_.mask_token_id);
                }
                batch.n_tokens = decode_count;
                last_logits_count_ = decode_count;
            } else {
                for (int i = 0; i < config_.block_length; i++) {
                    batch.token[i] = current_block[i];
                    batch.pos[i] = static_cast<llama_pos>(block_start + i);
                    batch.n_seq_id[i] = 1;
                    batch.seq_id[i][0] = 0;
                    batch.logits[i] = true;
                }
                batch.n_tokens = config_.block_length;
                last_logits_count_ = config_.block_length;
            }
        }

        {
            PROFILE_SECTION("llama_decode");
            if (llama_decode(ctx_, batch) != 0) {
                llama_batch_free(batch);
                if (do_partial_kv) {
                    sampler_metrics_.partial_kv_fallback++;
                    do_partial_kv = false;

                    // fallback: full-block decode
                    llama_memory_seq_rm(memory, 0, block_start, block_start + config_.block_length);
                    llama_batch b2 = llama_batch_init(config_.block_length, 0, 1);
                    for (int i = 0; i < config_.block_length; i++) {
                        b2.token[i] = current_block[i];
                        b2.pos[i] = static_cast<llama_pos>(block_start + i);
                        b2.n_seq_id[i] = 1;
                        b2.seq_id[i][0] = 0;
                        b2.logits[i] = true;
                    }
                    b2.n_tokens = config_.block_length;
                    last_logits_count_ = config_.block_length;
                    if (llama_decode(ctx_, b2) != 0) {
                        llama_batch_free(b2);
                        assert(false && "llama_decode failed!");
                        return;
                    }
                    llama_batch_free(b2);

                    // 重新创建一个占位 batch，保证后续 llama_batch_free 逻辑不变
                    batch = llama_batch_init(config_.block_length, 0, 1);
                } else {
                    assert(false && "llama_decode failed!");
                    return;
                }
            }
        }

        const int n_vocab = get_vocab_size();

        std::vector<llama_token> sampled_tokens_active(active_count);
        std::vector<float> confidences_active(active_count);
        std::vector<std::vector<float>> entropy_active;
        bool sampled = false;

        {
            PROFILE_SECTION("token_sampling");
            if (active_count == config_.block_length) {
                std::vector<std::vector<float>>* entropy_ptr_full = need_entropy_probs ? &entropy_active : nullptr;
                sample_block_tokens(
                    n_vocab,
                    need_entropy_probs,
                    sampled_tokens_active,
                    confidences_active,
                    entropy_ptr_full
                );
                sampled = true;
            } else {
                if (do_partial_kv) {
                    // 建 pos -> batch_idx 映射，并校验每个活跃位置都有 logits
                    std::vector<int> pos2idx(config_.block_length, -1);
                    for (int i = 0; i < decode_count; ++i) {
                        const int pos = decode_positions[i];
                        if (pos >= 0 && pos < config_.block_length) pos2idx[pos] = i;
                    }
                    std::vector<int> logits_override(active_count);
                    bool ok = true;
                    for (int i = 0; i < active_count; ++i) {
                        const int pos = active_positions[i];
                        const int li = (pos >= 0 && pos < config_.block_length) ? pos2idx[pos] : -1;
                        if (li < 0) { ok = false; break; }
                        logits_override[i] = li;
                        if (!batch.logits[li] || llama_get_logits_ith(ctx_, li) == nullptr) { ok = false; break; }
                    }

                    if (!ok) {
                        sampler_metrics_.partial_kv_fallback++;
                        do_partial_kv = false;
                        // fallback: full-block decode
                        llama_memory_seq_rm(memory, 0, block_start, block_start + config_.block_length);
                        llama_batch b2 = llama_batch_init(config_.block_length, 0, 1);
                        for (int i = 0; i < config_.block_length; i++) {
                            b2.token[i] = current_block[i];
                            b2.pos[i] = static_cast<llama_pos>(block_start + i);
                            b2.n_seq_id[i] = 1;
                            b2.seq_id[i][0] = 0;
                            b2.logits[i] = true;
                        }
                        b2.n_tokens = config_.block_length;
                        last_logits_count_ = config_.block_length;
                        if (llama_decode(ctx_, b2) != 0) {
                            llama_batch_free(b2);
                            assert(false && "llama_decode failed!");
                            return;
                        }
                        llama_batch_free(b2);
                    } else {
                        diffusion::ProfilerTimer cpu_timer_total;
                        sample_active_tokens_cpu(
                            n_vocab,
                            active_positions,
                            sampled_tokens_active,
                            confidences_active,
                            nullptr,
                            &logits_override
                        );
                        DiffusionProfiler::instance().record_custom("sampler_cpu_sampling_ms", cpu_timer_total.elapsed_ms());
                        sampler_metrics_.cpu_sampling_ms += cpu_timer_total.elapsed_ms();
                        sampler_metrics_.cpu_sampling_calls++;
                        sampler_metrics_.partial_kv_used++;
                        sampled = true;
                    }
                } else if (use_gpu_sampler_) {
                    auto env_flag = [](const char* key) {
                        const char* v = std::getenv(key);
                        if (!v) return false;
                        if (v[0] == '0' || v[0] == 'f' || v[0] == 'F' || v[0] == 'n' || v[0] == 'N') return false;
                        return true;
                    };
                    const bool gpu_only_mode = env_flag("DIFFUSION_GPU_ONLY") || config_.gpu_only_mode;
                    const bool device_logits_env = env_flag("LLAMA_ENABLE_DEVICE_LOGITS");
                    const bool allow_gpu_sampling = !env_flag("DIFFUSION_DISABLE_GPU_SAMPLER") && !env_flag("DIFFUSION_FORCE_CPU_SAMPLING");

                    if (!allow_gpu_sampling && gpu_only_mode) {
                        llama_batch_free(batch);
                        throw std::runtime_error("[DiffusionSamplerProfiled] gpu_only_mode=true 但 GPU sampling 被禁用 (DIFFUSION_DISABLE_GPU_SAMPLER/DIFFUSION_FORCE_CPU_SAMPLING)");
                    }

                    if (static_cast<int>(gpu_sampled_block_buffer_.size()) != config_.block_length) {
                        gpu_sampled_block_buffer_.assign(config_.block_length, 0);
                    }
                    if (static_cast<int>(gpu_conf_block_buffer_.size()) != config_.block_length) {
                        gpu_conf_block_buffer_.assign(config_.block_length, 0.0f);
                    }
                    if (need_entropy_probs) {
                        gpu_entropy_block_buffer_.assign(config_.block_length, std::vector<float>{});
                    } else {
                        gpu_entropy_block_buffer_.clear();
                    }
                    std::vector<llama_token>& sampled_block = gpu_sampled_block_buffer_;
                    std::vector<float>& confidences_block = gpu_conf_block_buffer_;
                    std::vector<std::vector<float>>* entropy_ptr_block = need_entropy_probs ? &gpu_entropy_block_buffer_ : nullptr;

                    diffusion::ProfilerTimer total_timer;
                    double gpu_elapsed_ms = 0.0;
                    if (allow_gpu_sampling && try_sample_with_gpu(
                            n_vocab,
                            need_entropy_probs,
                            sampled_block,
                            confidences_block,
                            entropy_ptr_block,
                            &gpu_elapsed_ms)) {
                        DiffusionProfiler::instance().record_custom("sampler_gpu_total_ms", gpu_elapsed_ms);
                        double overhead = std::max(0.0, total_timer.elapsed_ms() - gpu_elapsed_ms);
                        DiffusionProfiler::instance().record_custom("sampler_gpu_overhead_ms", overhead);
                        sampler_metrics_.gpu_total_ms += gpu_elapsed_ms;
                        sampler_metrics_.gpu_overhead_ms += overhead;

                        if (need_entropy_probs) entropy_active.resize(static_cast<size_t>(active_count));
                        for (int i = 0; i < active_count; ++i) {
                            const int pos = active_positions[i];
                            sampled_tokens_active[i] = sampled_block[pos];
                            confidences_active[i] = confidences_block[pos];
                            if (need_entropy_probs && pos < static_cast<int>(gpu_entropy_block_buffer_.size())) {
                                entropy_active[static_cast<size_t>(i)] = std::move(gpu_entropy_block_buffer_[pos]);
                            }
                        }
                        sampled = true;
                    } else {
                        if (gpu_only_mode) {
                            llama_batch_free(batch);
                            throw std::runtime_error("[DiffusionSampler] gpu_only_mode=true 但 GPU sampler 不可用或执行失败，已禁止回退到 host logits。");
                        }
                        if (device_logits_env) {
                            DIFF_LOGW("[DiffusionSampler][warn] device logits 启用但 GPU 采样未命中，回退 CPU 采样，可能触发 host/device 混用。\n");
                        }
                    }
                }

                if (!sampled) {
                    diffusion::ProfilerTimer cpu_timer_total;
                    std::vector<std::vector<float>>* entropy_ptr = need_entropy_probs ? &entropy_active : nullptr;
                    sample_active_tokens_cpu(
                        n_vocab,
                        active_positions,
                        sampled_tokens_active,
                        confidences_active,
                        entropy_ptr
                    );
                    DiffusionProfiler::instance().record_custom("sampler_cpu_sampling_ms", cpu_timer_total.elapsed_ms());
                    sampler_metrics_.cpu_sampling_ms += cpu_timer_total.elapsed_ms();
                    sampler_metrics_.cpu_sampling_calls++;
                }
            }
        }

        llama_batch_free(batch);

        // 将采样结果映射回 block
        std::vector<float> block_confidences(config_.block_length, -INFINITY);
        std::vector<std::vector<float>> entropy_probs_full;
        if (need_entropy_probs) {
            entropy_probs_full.resize(config_.block_length);
        }
        for (int i = 0; i < active_count; ++i) {
            const int pos = active_positions[i];
            block_confidences[pos] = confidences_active[i];
            if (need_entropy_probs && i < static_cast<int>(entropy_active.size())) {
                entropy_probs_full[pos] = std::move(entropy_active[i]);
            }
        }

        // 选择转移位置
        std::vector<bool> transfer_indices;
        if (step >= static_cast<int>(num_transfer_tokens_per_step.size())) {
            transfer_indices.assign(config_.block_length, true);
        } else {
            int num_transfer = num_transfer_tokens_per_step[step];
            std::string strategy_name = "remasking_strategy_";
            switch (config_.remasking_strategy) {
                case RemaskingStrategy::SEQUENTIAL:
                    strategy_name += "sequential";
                    transfer_indices = get_transfer_indices_sequential(current_block, block_confidences, num_transfer);
                    break;
                case RemaskingStrategy::LOW_CONFIDENCE_STATIC:
                    strategy_name += "low_conf_static";
                    transfer_indices = get_transfer_indices_low_conf_static(current_block, block_confidences, num_transfer);
                    break;
                case RemaskingStrategy::LOW_CONFIDENCE_DYNAMIC:
                    strategy_name += "low_conf_dynamic";
                    transfer_indices = get_transfer_indices_low_conf_dynamic(current_block, block_confidences, num_transfer);
                    break;
                case RemaskingStrategy::ENTROPY_BOUNDED:
                    strategy_name += "entropy_bounded";
                    transfer_indices = get_transfer_indices_entropy_bounded(current_block, entropy_probs_full);
                    break;
            }
            PROFILE_SECTION(strategy_name.c_str());
        }

        // 更新 token 和 mask 计数
        {
            PROFILE_SECTION("update_block_tokens");
            std::vector<char> next_dirty(static_cast<size_t>(micro_count), 0);
            for (size_t i = 0; i < transfer_indices.size(); ++i) {
                if (!transfer_indices[i]) continue;
                if (current_block[i] == config_.mask_token_id) {
                    const int idx = (i < active_pos2idx.size()) ? active_pos2idx[i] : -1;
                    if (idx < 0) continue;
                    current_block[i] = sampled_tokens_active[static_cast<size_t>(idx)];
                    remaining_masks--;
                    masks_per_micro[i / micro_size] = std::max(0, masks_per_micro[i / micro_size] - 1);
                    next_dirty[i / micro_size] = 1;
                }
            }
            dirty_micros.swap(next_dirty);
        }
    }
}


} // namespace diffusion

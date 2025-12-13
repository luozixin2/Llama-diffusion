// diffusion_sampler_profiled.cpp
#include "diffusion_sampler_profiled.h"
#include "gpu_sampler.h"
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
    const bool disable_gpu_sampler = env_flag("DIFFUSION_DISABLE_GPU_SAMPLER");
    const bool force_cpu_sampling = env_flag("DIFFUSION_FORCE_CPU_SAMPLING");
    const bool disable_partial_kv = env_flag("DIFFUSION_DISABLE_PARTIAL_KV_REUSE");
    const bool force_full_decode = env_flag("DIFFUSION_FORCE_FULL_BLOCK_DECODE");
    
    DIFF_LOGI("[DiffusionSampler][info] flags: disable_gpu_sampler=%d force_cpu_sampling=%d disable_partial_kv=%d force_full_decode=%d\n",
              disable_gpu_sampler ? 1 : 0,
              force_cpu_sampling ? 1 : 0,
              disable_partial_kv ? 1 : 0,
              force_full_decode ? 1 : 0);
    
    DiffusionProfiler::instance().record_custom("flag_disable_gpu_sampler", disable_gpu_sampler ? 1.0 : 0.0);
    DiffusionProfiler::instance().record_custom("flag_force_cpu_sampling", force_cpu_sampling ? 1.0 : 0.0);
    DiffusionProfiler::instance().record_custom("flag_disable_partial_kv", disable_partial_kv ? 1.0 : 0.0);
    DiffusionProfiler::instance().record_custom("flag_force_full_decode", force_full_decode ? 1.0 : 0.0);
    
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

    // Reusable buffers to reduce per-step allocations (important when steps=block_length)
    std::vector<int> active_positions;
    active_positions.reserve(static_cast<size_t>(config_.block_length));
    std::vector<int> active_micros;
    active_micros.reserve(static_cast<size_t>(micro_count));
    std::vector<int> active_pos2idx(config_.block_length, -1);
    std::vector<int> micros_to_decode;
    micros_to_decode.reserve(static_cast<size_t>(micro_count));
    std::vector<int> decode_positions;
    decode_positions.reserve(static_cast<size_t>(config_.block_length));
    std::vector<char> next_dirty(static_cast<size_t>(micro_count), 0);
    std::vector<float> block_confidences(config_.block_length, -INFINITY);

    for (int step = 0; step < config_.denoising_steps; step++) {
        std::string step_section = "denoising_step_" + std::to_string(step);
        PROFILE_SECTION(step_section.c_str());
        if (remaining_masks == 0) {
            DiffusionProfiler::instance().record_custom("early_exit_step", step);
            break;
        }

        // 收集活跃微块和位置
        active_positions.clear();
        active_micros.clear();
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
        std::fill(active_pos2idx.begin(), active_pos2idx.end(), -1);
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
        micros_to_decode.clear();
        for (int m = 0; m < micro_count; ++m) {
            if (masks_per_micro[m] > 0 || dirty_micros[static_cast<size_t>(m)]) {
                micros_to_decode.push_back(m);
            }
        }
        decode_positions.clear();
        decode_positions.reserve(static_cast<size_t>(micros_to_decode.size()) * static_cast<size_t>(micro_size));
        for (int m : micros_to_decode) {
            const int base = m * micro_size;
            for (int j = 0; j < micro_size; ++j) {
                decode_positions.push_back(base + j);
            }
        }
        const int decode_count = static_cast<int>(decode_positions.size());

        // Telemetry: quantify work per denoising step (helps explain block scaling)
        sampler_metrics_.denoise_step_count++;
        sampler_metrics_.active_count_sum += active_count;
        sampler_metrics_.active_count_samples++;
        sampler_metrics_.active_count_min = std::min(sampler_metrics_.active_count_min, active_count);
        sampler_metrics_.active_count_max = std::max(sampler_metrics_.active_count_max, active_count);
        sampler_metrics_.decode_count_sum += decode_count;
        sampler_metrics_.decode_count_samples++;
        sampler_metrics_.decode_count_min = std::min(sampler_metrics_.decode_count_min, decode_count);
        sampler_metrics_.decode_count_max = std::max(sampler_metrics_.decode_count_max, decode_count);
        if (decode_count == config_.block_length) {
            sampler_metrics_.decode_full_steps++;
        } else if (decode_count < config_.block_length) {
            sampler_metrics_.decode_partial_steps++;
        }

        // Quality guard: disable partial-KV by default when GPU sampler is active unless
        // explicitly enabled via DIFFUSION_PARTIAL_KV_REUSE_GPU=1.
        const bool env_partial_kv_reuse = env_flag("DIFFUSION_PARTIAL_KV_REUSE");
        const bool env_partial_kv_reuse_gpu = env_flag("DIFFUSION_PARTIAL_KV_REUSE_GPU");
        const bool env_disable_partial_kv = env_flag("DIFFUSION_DISABLE_PARTIAL_KV_REUSE");
        const bool env_force_full_decode = env_flag("DIFFUSION_FORCE_FULL_BLOCK_DECODE");
        const bool allow_partial_kv = !use_gpu_sampler_ || env_partial_kv_reuse_gpu;
        
        bool do_partial_kv = env_partial_kv_reuse
            && allow_partial_kv
            && !env_disable_partial_kv
            && !env_force_full_decode
            && !need_entropy_probs
            && config_.top_k <= 0
            && config_.top_p >= 1.0f
            && active_count < config_.block_length
            && decode_count > 0
            && decode_count < config_.block_length;

        // Debug logging: why partial KV is disabled (only log for first few blocks to avoid spam)
        static int last_logged_block = -1;
        if (block_idx < 10 && block_idx != last_logged_block) {
            if (do_partial_kv) {
                DIFF_LOGI("[DiffusionSampler][partial_kv] Block %d: Partial KV reuse ENABLED: active=%d/%d, decode=%d/%d\n",
                         block_idx, active_count, config_.block_length, decode_count, config_.block_length);
            } else if (env_partial_kv_reuse) {
                DIFF_LOGI("[DiffusionSampler][partial_kv] Block %d: Partial KV reuse disabled. Reasons:\n", block_idx);
                if (!allow_partial_kv) {
                    DIFF_LOGI("  - GPU sampler is enabled; partial_kv is disabled by default (set DIFFUSION_PARTIAL_KV_REUSE_GPU=1 to allow)\n");
                }
                if (env_disable_partial_kv) {
                    DIFF_LOGI("  - DIFFUSION_DISABLE_PARTIAL_KV_REUSE is set\n");
                }
                if (env_force_full_decode) {
                    DIFF_LOGI("  - DIFFUSION_FORCE_FULL_BLOCK_DECODE is set\n");
                }
                if (need_entropy_probs) {
                    DIFF_LOGI("  - need_entropy_probs=true (entropy probs required)\n");
                }
                if (config_.top_k > 0) {
                    DIFF_LOGI("  - top_k=%d > 0 (top-k sampling enabled)\n", config_.top_k);
                }
                if (config_.top_p < 1.0f) {
                    DIFF_LOGI("  - top_p=%.2f < 1.0 (top-p sampling enabled)\n", config_.top_p);
                }
                if (use_gpu_sampler_) {
                    DIFF_LOGI("  - use_gpu_sampler_=true (note: partial_kv may still be used via active-subset sampling)\n");
                }
                if (active_count >= config_.block_length) {
                    DIFF_LOGI("  - active_count=%d >= block_length=%d (all positions active)\n", 
                             active_count, config_.block_length);
                }
                if (decode_count <= 0) {
                    DIFF_LOGI("  - decode_count=%d <= 0 (no positions to decode)\n", decode_count);
                }
                if (decode_count >= config_.block_length) {
                    DIFF_LOGI("  - decode_count=%d >= block_length=%d (full block decode needed)\n", 
                             decode_count, config_.block_length);
                }
            } else {
                DIFF_LOGI("[DiffusionSampler][partial_kv] Block %d: DIFFUSION_PARTIAL_KV_REUSE not set\n", block_idx);
            }
            last_logged_block = block_idx;
        }

        // KV 清理（整块 or 仅需要重算的微块）
        {
            PROFILE_SECTION("kv_cache_clear");
            if (do_partial_kv) {
                sampler_metrics_.partial_kv_attempt++;
                sampler_metrics_.kv_rm_calls += static_cast<int>(micros_to_decode.size());
                sampler_metrics_.kv_rm_partial_calls += static_cast<int>(micros_to_decode.size());
                sampler_metrics_.kv_rm_tokens += static_cast<long long>(micros_to_decode.size()) * micro_size;
                for (int m : micros_to_decode) {
                    const int start_pos = block_start + m * micro_size;
                    llama_memory_seq_rm(memory, 0, start_pos, start_pos + micro_size);
                }
            } else {
                llama_memory_seq_rm(memory, 0, block_start, block_start + config_.block_length);
                sampler_metrics_.kv_rm_calls++;
                sampler_metrics_.kv_rm_full_calls++;
                sampler_metrics_.kv_rm_tokens += config_.block_length;
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
            sampler_metrics_.llama_decode_calls++;
            sampler_metrics_.llama_decode_tokens += batch.n_tokens;
            if (llama_decode(ctx_, batch) != 0) {
                llama_batch_free(batch);
                if (do_partial_kv) {
                    sampler_metrics_.partial_kv_fallback++;
                    do_partial_kv = false;

                    // fallback: full-block decode
                    llama_memory_seq_rm(memory, 0, block_start, block_start + config_.block_length);
                    sampler_metrics_.kv_rm_calls++;
                    sampler_metrics_.kv_rm_full_calls++;
                    sampler_metrics_.kv_rm_tokens += config_.block_length;
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
                    sampler_metrics_.llama_decode_calls++;
                    sampler_metrics_.llama_decode_tokens += b2.n_tokens;
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
                    // llama.cpp output_ids: batch index -> logits row (or -1 if batch.logits[i] != true)
                    int out_count_check = 0; // n_outputs (not mapping length)
                    const int32_t* out_ids_check = llama_get_logits_output_ids(ctx_, &out_count_check);
                    for (int i = 0; i < active_count; ++i) {
                        const int pos = active_positions[i];
                        const int li = (pos >= 0 && pos < config_.block_length) ? pos2idx[pos] : -1;
                        if (li < 0) { ok = false; break; }
                        logits_override[i] = li;
                        if (!batch.logits[li] ||
                            !out_ids_check ||
                            // output_ids mapping length == batch.n_tokens (here: decode_count)
                            out_ids_check[li] < 0) {
                            ok = false;
                            break;
                        }
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
                        // Prefer GPU sampling for active subset if available; fall back to CPU subset sampling.
                        bool sampled_partial = false;
                        if (use_gpu_sampler_ && gpu_sampler_) {
                            auto env_flag = [](const char* key) {
                                const char* v = std::getenv(key);
                                if (!v) return false;
                                if (v[0] == '0' || v[0] == 'f' || v[0] == 'F' || v[0] == 'n' || v[0] == 'N') return false;
                                return true;
                            };
                            const bool gpu_only_mode = env_flag("DIFFUSION_GPU_ONLY") || config_.gpu_only_mode;
                            const bool allow_gpu_sampling = !env_flag("DIFFUSION_DISABLE_GPU_SAMPLER") && !env_flag("DIFFUSION_FORCE_CPU_SAMPLING");
                            if (!allow_gpu_sampling && gpu_only_mode) {
                                llama_batch_free(batch);
                                throw std::runtime_error("[DiffusionSamplerProfiled] gpu_only_mode=true 但 GPU sampling 被禁用 (DIFFUSION_DISABLE_GPU_SAMPLER/DIFFUSION_FORCE_CPU_SAMPLING)");
                            }
                            if (allow_gpu_sampling) {
                                diffusion::ProfilerTimer gpu_timer_total;
                                GpuSampler::Stats gpu_stats{};

                                // Fast path: use llama.cpp device logits directly (avoid per-row H2D of full vocab)
                                bool ok_gpu = false;
                                int64_t stride_tokens = 0;
                                const float* logits_dev = llama_get_logits_device(ctx_, &stride_tokens);
                                int out_count = 0;
                                const int32_t* out_ids = llama_get_logits_output_ids(ctx_, &out_count);

                                // llama.cpp: out_ids is a mapping from batch index -> logits row (or -1 when batch.logits[i] != true).
                                // Here batch.n_tokens == decode_count, and out_count is the number of logits rows.
                                if (logits_dev && out_ids && stride_tokens >= n_vocab && out_count > 0) {
                                    std::vector<llama_token> gpu_tokens_out;
                                    std::vector<float> gpu_confs_out;
                                    if (stride_tokens == n_vocab) {
                                        ok_gpu = gpu_sampler_->sample_from_device_ptr(
                                            logits_dev,
                                            static_cast<size_t>(out_count) * static_cast<size_t>(n_vocab),
                                            config_.remasking_strategy,
                                            rng_,
                                            gpu_tokens_out,
                                            gpu_confs_out,
                                            nullptr,
                                            &gpu_stats,
                                            /*force_non_fused=*/false
                                        );
                                    } else {
                                        ok_gpu = gpu_sampler_->sample_from_device_ptr_strided(
                                            logits_dev,
                                            stride_tokens,
                                            out_count,
                                            config_.remasking_strategy,
                                            rng_,
                                            gpu_tokens_out,
                                            gpu_confs_out,
                                            nullptr,
                                            &gpu_stats,
                                            /*force_non_fused=*/false
                                        );
                                    }
                                    if (ok_gpu && static_cast<int>(gpu_tokens_out.size()) == out_count && static_cast<int>(gpu_confs_out.size()) == out_count) {
                                        sampled_tokens_active.resize(active_count);
                                        confidences_active.resize(active_count);
                                        for (int i = 0; i < active_count; ++i) {
                                            const int li = logits_override[i];
                                            const int oi = (li >= 0 && li < decode_count) ? static_cast<int>(out_ids[li]) : -1;
                                            if (oi < 0 || oi >= out_count) {
                                                ok_gpu = false;
                                                break;
                                            }
                                            sampled_tokens_active[i] = gpu_tokens_out[oi];
                                            confidences_active[i] = gpu_confs_out[oi];
                                        }
                                    }
                                    if (ok_gpu) {
                                        sampled_partial = true;
                                        DiffusionProfiler::instance().record_custom("sampler_gpu_subset_device_logits", 1.0);
                                        DiffusionProfiler::instance().record_custom("sampler_gpu_subset_rows", static_cast<double>(out_count));
                                        DiffusionProfiler::instance().record_custom("sampler_gpu_subset_active_rows", static_cast<double>(active_count));
                                    }
                                }

                                // Fallback: host logits scatter H2D per row (slower, but keeps correctness)
                                if (!sampled_partial) {
                                    std::vector<float*> logits_ptrs;
                                    logits_ptrs.reserve(static_cast<size_t>(active_count));
                                    int out_count_host = 0; // n_outputs
                                    const int32_t* out_ids_host = llama_get_logits_output_ids(ctx_, &out_count_host);
                                    bool ok_host = (out_ids_host != nullptr);
                                    for (int i = 0; i < active_count; ++i) {
                                        const int li = logits_override[i];
                                        if (!ok_host || li < 0 || li >= decode_count || out_ids_host[li] < 0) {
                                            ok_host = false;
                                            break;
                                        }
                                        const float* lp = llama_get_logits_ith(ctx_, li);
                                        if (!lp) {
                                            ok_host = false;
                                            break;
                                        }
                                        logits_ptrs.push_back(const_cast<float*>(lp));
                                    }
                                    if (!ok_host) {
                                        ok_gpu = false;
                                        sampled_partial = false;
                                    } else {
                                    std::vector<llama_token> gpu_tokens;
                                    std::vector<float> gpu_confs;
                                    ok_gpu = gpu_sampler_->sample_from_scatter_ptrs(
                                        logits_ptrs,
                                        n_vocab,
                                        config_.remasking_strategy,
                                        rng_,
                                        gpu_tokens,
                                        gpu_confs,
                                        nullptr,
                                        &gpu_stats
                                    );
                                    DiffusionProfiler::instance().record_custom("sampler_gpu_subset_device_logits", 0.0);
                                    DiffusionProfiler::instance().record_custom("sampler_gpu_subset_rows", static_cast<double>(active_count));
                                    DiffusionProfiler::instance().record_custom("sampler_gpu_subset_active_rows", static_cast<double>(active_count));
                                    if (ok_gpu && static_cast<int>(gpu_tokens.size()) == active_count && static_cast<int>(gpu_confs.size()) == active_count) {
                                        sampled_tokens_active = std::move(gpu_tokens);
                                        confidences_active = std::move(gpu_confs);
                                        sampled_partial = true;
                                    }
                                    }
                                }

                                DiffusionProfiler::instance().record_custom("sampler_gpu_subset_total_ms", gpu_timer_total.elapsed_ms());

                                if (!sampled_partial && gpu_only_mode) {
                                    llama_batch_free(batch);
                                    throw std::runtime_error("[DiffusionSamplerProfiled] gpu_only_mode=true 但 active-subset GPU sampling 失败，已禁止回退到 CPU 采样。");
                                }
                            }
                        }

                        if (!sampled_partial) {
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
                        }

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
        std::fill(block_confidences.begin(), block_confidences.end(), -INFINITY);
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
            std::fill(next_dirty.begin(), next_dirty.end(), 0);
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

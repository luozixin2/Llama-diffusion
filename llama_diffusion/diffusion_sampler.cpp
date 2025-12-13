#include "diffusion_sampler.h"
#include "gpu_sampler.h"
#include "diffusion_profiler.h"
#include "diffusion_logging.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <vector>
#include <cstdlib>
#include <cerrno>
#if defined(DIFFUSION_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

namespace diffusion {

DiffusionSampler::DiffusionSampler(llama_context* ctx, llama_model* model, const DiffusionConfig& config)
    : ctx_(ctx), model_(model), config_(config) {
    if (config_.micro_block_size <= 0 || config_.micro_block_size > config_.block_length ||
        (config_.block_length % config_.micro_block_size) != 0) {
        throw std::runtime_error("[DiffusionSampler] micro_block_size must be >0, <= block_length, and divide block_length.");
    }

    auto env_u64 = [](const char* key, bool* ok_out) -> uint64_t {
        if (ok_out) *ok_out = false;
        const char* v = std::getenv(key);
        if (!v || !*v) return 0;
        errno = 0;
        char* end = nullptr;
        unsigned long long x = std::strtoull(v, &end, 10);
        if (errno != 0 || end == v) return 0;
        if (ok_out) *ok_out = true;
        return static_cast<uint64_t>(x);
    };

    bool ok_seed = false;
    const uint64_t seed = env_u64("DIFFUSION_SEED", &ok_seed);
    if (ok_seed) {
        rng_.seed(static_cast<uint32_t>(seed));
        DIFF_LOGI("[DiffusionSampler][info] DIFFUSION_SEED=%llu\n", (unsigned long long) seed);
    } else {
        std::random_device rd;
        rng_.seed(rd());
    }

    reset_sampler_metrics();

    if (config_.enable_gpu_sampler) {
        const int vocab_size = get_vocab_size();
        gpu_sampler_ = std::make_unique<GpuSampler>(config_.block_length, vocab_size, config_);
        if (gpu_sampler_ && gpu_sampler_->is_available()) {
            use_gpu_sampler_ = true;
        }
    }
}

DiffusionSampler::~DiffusionSampler() {
#if defined(DIFFUSION_ENABLE_CUDA)
    if (device_logits_compact_) {
        cudaFree(device_logits_compact_);
        device_logits_compact_ = nullptr;
        device_logits_compact_bytes_ = 0;
    }
#endif
}

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
    const int micro_size = config_.micro_block_size;
    const int micro_count = config_.block_length / micro_size;
    llama_memory_t memory = llama_get_memory(ctx_);

    // 当微块尺寸等于块长时，直接走整块去噪以保持与旧实现一致
    if (micro_size == config_.block_length) {
        int remaining_masks = 0;
        for (llama_token t : current_block) {
            if (t == config_.mask_token_id) remaining_masks++;
        }

        llama_batch batch = llama_batch_init(config_.block_length, 0, 1);
        for (int i = 0; i < config_.block_length; i++) {
            batch.pos[i] = static_cast<llama_pos>(block_start + i);
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
        }

        for (int step = 0; step < config_.denoising_steps; step++) {
            if (remaining_masks == 0) break;

            llama_memory_seq_rm(memory, 0, block_start, block_start + config_.block_length);
            for (int i = 0; i < config_.block_length; i++) {
                batch.token[i] = current_block[i];
                batch.logits[i] = true;
            }
            batch.n_tokens = config_.block_length;
            last_logits_count_ = config_.block_length;

            if (llama_decode(ctx_, batch) != 0) {
                llama_batch_free(batch);
                assert(false && "llama_decode failed inside denoise_block (full path)!");
                return;
            }

            const int n_vocab = get_vocab_size();
            const bool need_entropy_probs = (config_.remasking_strategy == RemaskingStrategy::ENTROPY_BOUNDED);
            std::vector<llama_token> sampled_tokens(config_.block_length);
            std::vector<float> confidences(config_.block_length);
            std::vector<std::vector<float>> all_probs;
            std::vector<std::vector<float>>* entropy_ptr = need_entropy_probs ? &all_probs : nullptr;
            sample_block_tokens(n_vocab, need_entropy_probs, sampled_tokens, confidences, entropy_ptr);

            if (step >= static_cast<int>(num_transfer_tokens_per_step.size())) {
                for (int i = 0; i < config_.block_length; i++) {
                    if (current_block[i] == config_.mask_token_id) {
                        current_block[i] = sampled_tokens[i];
                        remaining_masks--;
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

            for (int i = 0; i < config_.block_length; i++) {
                if (transfer_indices[i] && current_block[i] == config_.mask_token_id) {
                    current_block[i] = sampled_tokens[i];
                    remaining_masks--;
                }
            }
        }

        llama_batch_free(batch);
        return;
    }

    // 统计每个微块的 mask 数，便于按微块清理/重算 KV
    int remaining_masks = 0;
    std::vector<int> masks_per_micro(micro_count, 0);
    for (int i = 0; i < config_.block_length; ++i) {
        if (current_block[i] == config_.mask_token_id) {
            remaining_masks++;
            masks_per_micro[i / micro_size]++;
        }
    }

    // partial KV/logits reuse 需要跨 step 追踪“哪些微块的 token 在上一轮被更新过”（否则 KV 会停留在旧 token 上）
    // 初始设为全 dirty：如果第 0 步就触发 partial，则会保守地重算这些微块。
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
    std::vector<float> block_confidences(config_.block_length, -INFINITY);
    std::vector<char> next_dirty(static_cast<size_t>(micro_count), 0);

    for (int step = 0; step < config_.denoising_steps; step++) {
        if (remaining_masks == 0) {
            break;
        }

        // 找出仍有 mask 的微块及其位置
        active_positions.clear();
        active_micros.clear();
        active_positions.reserve(static_cast<size_t>(remaining_masks));
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

        // Experimental: guarded partial KV/logits reuse
        //
        // Quality guard:
        // In practice, partial-KV reuse together with GPU sampling can degrade text quality
        // (repetition / garbled tokens) under common remasking strategies.
        // Therefore: when GPU sampler is active, partial-KV is disabled by default unless
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
        
        // Debug logging: why partial KV is disabled (only log when env flag is set but disabled)
        // Log at INFO level to ensure visibility (can be filtered by setting DIFFUSION_LOG_LEVEL_RUNTIME=2 for WARN only)
        // Only log once per block to avoid spam (use static counter per block_start)
        static int last_logged_block = -1;
        // Always log partial KV status for first few blocks to debug
        if (block_start < 100 && block_start != last_logged_block) {
            if (do_partial_kv) {
                DIFF_LOGI("[DiffusionSampler][partial_kv] Block %d: Partial KV reuse ENABLED: active=%d/%d, decode=%d/%d\n",
                         block_start, active_count, config_.block_length, decode_count, config_.block_length);
            } else if (env_partial_kv_reuse) {
                DIFF_LOGI("[DiffusionSampler][partial_kv] Block %d: Partial KV reuse disabled. Reasons:\n", block_start);
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
                DIFF_LOGI("[DiffusionSampler][partial_kv] Block %d: DIFFUSION_PARTIAL_KV_REUSE not set\n", block_start);
            }
            last_logged_block = block_start;
        }

        auto decode_full_block = [&]() -> bool {
            llama_memory_seq_rm(memory, 0, block_start, block_start + config_.block_length);
            sampler_metrics_.kv_rm_calls++;
            sampler_metrics_.kv_rm_full_calls++;
            sampler_metrics_.kv_rm_tokens += config_.block_length;
            llama_batch b = llama_batch_init(config_.block_length, 0, 1);
            for (int i = 0; i < config_.block_length; i++) {
                b.token[i] = current_block[i];
                b.pos[i] = static_cast<llama_pos>(block_start + i);
                b.n_seq_id[i] = 1;
                b.seq_id[i][0] = 0;
                b.logits[i] = true;
            }
            b.n_tokens = config_.block_length;
            last_logits_count_ = config_.block_length;
            sampler_metrics_.llama_decode_calls++;
            sampler_metrics_.llama_decode_tokens += b.n_tokens;
            const int rc = llama_decode(ctx_, b);
            llama_batch_free(b);
            return rc == 0;
        };

        llama_batch batch = llama_batch_init(do_partial_kv ? decode_count : config_.block_length, 0, 1);

        // KV 清理（整块 or 仅需要重算的微块）
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

        // batch 构造（整块 or compact 需要重算的微块）
        if (do_partial_kv) {
            for (int i = 0; i < decode_count; i++) {
                const int pos = decode_positions[i];
                batch.token[i] = current_block[pos];
                batch.pos[i] = static_cast<llama_pos>(block_start + pos);
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                // 仅对 mask token 请求 logits（用于采样）；非 mask 仅为更新 KV
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

        // decode（失败则回退到整块）
        sampler_metrics_.llama_decode_calls++;
        sampler_metrics_.llama_decode_tokens += batch.n_tokens;
        if (llama_decode(ctx_, batch) != 0) {
            llama_batch_free(batch);
            if (do_partial_kv) {
                sampler_metrics_.partial_kv_fallback++;
                do_partial_kv = false;
                if (!decode_full_block()) {
                    assert(false && "llama_decode failed inside denoise_block (fallback full)!");
                    return;
                }
                // 重新创建一个占位 batch，保证后续 llama_batch_free 逻辑不变
                batch = llama_batch_init(config_.block_length, 0, 1);
            } else {
                assert(false && "llama_decode failed inside denoise_block!");
                return;
            }
        }

        // 采样：整块 logits 计算，只对活跃位置采样（优先 GPU）
        const int n_vocab = get_vocab_size();

        std::vector<llama_token> sampled_tokens_active(active_count);
        std::vector<float> confidences_active(active_count);
        std::vector<std::vector<float>> entropy_active;
        bool sampled = false;

        // 分支1：活跃=整块，直接复用整块采样（内含 GPU 路径）
        if (active_count == config_.block_length) {
            std::vector<std::vector<float>>* entropy_ptr_full = need_entropy_probs ? &entropy_active : nullptr;
            sample_block_tokens(
                n_vocab,
                need_entropy_probs,
                sampled_tokens_active,  // 大小等于 block_length
                confidences_active,
                entropy_ptr_full
            );
            sampled = true;
        } else {
            // 分支2：活跃为子集
            // 2A) partial KV/logits：compact decode（仅重算必要微块）后仅 CPU 子集采样（用 override 映射 logits index）
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
                int out_count_check = 0; // note: this is n_outputs, not mapping length
                const int32_t* out_ids_check = llama_get_logits_output_ids(ctx_, &out_count_check);
                for (int i = 0; i < active_count; ++i) {
                    const int pos = active_positions[i];
                    const int li = (pos >= 0 && pos < config_.block_length) ? pos2idx[pos] : -1;
                    if (li < 0) { ok = false; break; }
                    logits_override[i] = li;
                    if (!batch.logits[li] ||
                        !out_ids_check ||
                        // output_ids mapping length == batch.n_tokens (here: decode_count), so li is safe to index.
                        // out_count_check is n_outputs and can be < decode_count when only some rows request logits.
                        out_ids_check[li] < 0) {
                        ok = false;
                        break;
                    }
                }

                if (!ok) {
                    sampler_metrics_.partial_kv_fallback++;
                    do_partial_kv = false;
                    // 回退到整块 decode，确保 logits index == 块内位置
                    if (!decode_full_block()) {
                        llama_batch_free(batch);
                        assert(false && "llama_decode failed inside denoise_block (fallback full)!");
                        return;
                    }
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
                            throw std::runtime_error("[DiffusionSampler] gpu_only_mode=true 但 GPU sampling 被禁用 (DIFFUSION_DISABLE_GPU_SAMPLER/DIFFUSION_FORCE_CPU_SAMPLING)");
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
                                    // output_ids mapping length == batch.n_tokens (here: decode_count)
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
                                throw std::runtime_error("[DiffusionSampler] gpu_only_mode=true 但 active-subset GPU sampling 失败，已禁止回退到 CPU 采样。");
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
            }

            // 2B) 非 partial：先尝试 GPU 采样整块后回收活跃位置；失败则回退 CPU 子集采样
            if (use_gpu_sampler_) {
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
                    throw std::runtime_error("[DiffusionSampler] gpu_only_mode=true 但 GPU sampling 被禁用 (DIFFUSION_DISABLE_GPU_SAMPLER/DIFFUSION_FORCE_CPU_SAMPLING)");
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
                if (allow_gpu_sampling) {
                if (try_sample_with_gpu(
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

                    // 仅回收活跃位置
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
                }

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

        llama_batch_free(batch);

        // 将置信度/概率映射回 block 级别
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

        // 选择要转移的 token
        std::vector<bool> transfer_indices;
        if (step >= static_cast<int>(num_transfer_tokens_per_step.size())) {
            transfer_indices.assign(config_.block_length, true);
        } else {
            int num_transfer = num_transfer_tokens_per_step[step];
            switch (config_.remasking_strategy) {
                case RemaskingStrategy::SEQUENTIAL:
                    transfer_indices = get_transfer_indices_sequential(current_block, block_confidences, num_transfer);
                    break;
                case RemaskingStrategy::LOW_CONFIDENCE_STATIC:
                    transfer_indices = get_transfer_indices_low_conf_static(current_block, block_confidences, num_transfer);
                    break;
                case RemaskingStrategy::LOW_CONFIDENCE_DYNAMIC:
                    transfer_indices = get_transfer_indices_low_conf_dynamic(current_block, block_confidences, num_transfer);
                    break;
                case RemaskingStrategy::ENTROPY_BOUNDED:
                    transfer_indices = get_transfer_indices_entropy_bounded(current_block, entropy_probs_full);
                    break;
                default:
                    transfer_indices = get_transfer_indices_low_conf_static(current_block, block_confidences, num_transfer);
                    break;
            }
        }

        // 更新 block，减少掩码计数
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

        // 下一次循环开始时才再次清理 KV
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
    // 线性时间选取前 k：nth_element 找到第 k 大阈值
    std::vector<float> tmp(logits);
    std::nth_element(tmp.begin(), tmp.begin() + (k - 1), tmp.end(), std::greater<float>());
    float min_value = tmp[k - 1];
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
    auto env_flag = [](const char* key) {
        const char* v = std::getenv(key);
        if (!v) return false;
        if (v[0] == '0' || v[0] == 'f' || v[0] == 'F' || v[0] == 'n' || v[0] == 'N') return false;
        return true;
    };
    const bool gpu_only_mode = env_flag("DIFFUSION_GPU_ONLY") || config_.gpu_only_mode;
    const bool device_logits_env = env_flag("LLAMA_ENABLE_DEVICE_LOGITS");

    DIFF_LOGD("[DiffusionSampler][debug] sample_block_tokens use_gpu_sampler=%d gpu_only=%d need_entropy=%d block_len=%d n_vocab=%d\n",
              use_gpu_sampler_ ? 1 : 0,
              gpu_only_mode ? 1 : 0,
              need_entropy_probs ? 1 : 0,
              config_.block_length,
              n_vocab);

    diffusion::ProfilerTimer total_timer;
    double gpu_elapsed_ms = 0.0;

    const bool allow_gpu_sampling = use_gpu_sampler_
        && !env_flag("DIFFUSION_DISABLE_GPU_SAMPLER")
        && !env_flag("DIFFUSION_FORCE_CPU_SAMPLING");

    if (allow_gpu_sampling && try_sample_with_gpu(
            n_vocab,
            need_entropy_probs,
            sampled_tokens,
            confidences,
            entropy_probs_storage,
            &gpu_elapsed_ms)) {
        DiffusionProfiler::instance().record_custom(
            "sampler_gpu_total_ms",
            gpu_elapsed_ms
        );
        double overhead = std::max(0.0, total_timer.elapsed_ms() - gpu_elapsed_ms);
        DiffusionProfiler::instance().record_custom(
            "sampler_gpu_overhead_ms",
            overhead
        );
        sampler_metrics_.gpu_total_ms += gpu_elapsed_ms;
        sampler_metrics_.gpu_overhead_ms += overhead;
        return true;
    }

    if (!allow_gpu_sampling) {
        if (gpu_only_mode) {
            throw std::runtime_error("[DiffusionSampler] gpu_only_mode=true 但 GPU sampling 被禁用 (DIFFUSION_DISABLE_GPU_SAMPLER/DIFFUSION_FORCE_CPU_SAMPLING)");
        }
    }

    // GPU 失败时的处理：GPU-only 模式下禁止访问 host logits，直接报错
    if (gpu_only_mode) {
        throw std::runtime_error("[DiffusionSampler] gpu_only_mode=true 但 GPU sampler 不可用或执行失败，已禁止回退到 host logits。");
    }

    // 设备 logits 模式下回退 CPU，打印告警
    if (device_logits_env) {
        DIFF_LOGW("[DiffusionSampler][warn] device logits 启用但 GPU 采样未命中，回退 CPU 采样，可能触发 host/device 混用。\n");
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

    // 复用线程局部缓冲，减少重复分配与拷贝
    static thread_local std::vector<float> logits_buf;
    static thread_local std::vector<float> probs_buf;
    logits_buf.resize(static_cast<size_t>(n_vocab));
    probs_buf.resize(static_cast<size_t>(n_vocab));

    int out_count = 0; // n_outputs (packed logits rows; output_ids length == batch.n_tokens)
    const int32_t* out_ids = llama_get_logits_output_ids(ctx_, &out_count);
    for (int i = 0; i < config_.block_length; i++) {
        // Avoid llama.cpp "invalid logits id" by checking output_ids mapping first
        if (!out_ids || out_ids[i] < 0) {
            sampled_tokens[i] = config_.mask_token_id;
            confidences[i] = 0.0f;
            continue;
        }
        float* logits = llama_get_logits_ith(ctx_, i);
        if (logits == nullptr) {
            sampled_tokens[i] = config_.mask_token_id;
            confidences[i] = 0.0f;
            continue;
        }

        // 拷贝 logits 并在同一次遍历完成温度缩放与 max 统计
        float max_logit = -INFINITY;
        for (int j = 0; j < n_vocab; ++j) {
            float v = logits[j];
        if (config_.temperature != 1.0f) {
                v /= config_.temperature;
            }
            logits_buf[static_cast<size_t>(j)] = v;
            if (!std::isinf(v) && v > max_logit) {
                max_logit = v;
            }
        }

        if (config_.top_k > 0) {
            apply_top_k(logits_buf, config_.top_k);
        }
        if (config_.top_p < 1.0f) {
            apply_top_p(logits_buf, config_.top_p);
        }

        // softmax 一次完成归一化，并复用 probs_buf
        float sum_exp = 0.0f;
        for (int j = 0; j < n_vocab; ++j) {
            float l = logits_buf[static_cast<size_t>(j)];
                if (!std::isinf(l)) {
                float e = std::exp(l - max_logit);
                probs_buf[static_cast<size_t>(j)] = e;
                sum_exp += e;
            } else {
                probs_buf[static_cast<size_t>(j)] = 0.0f;
            }
        }

        if (sum_exp <= 0.0f) {
            sampled_tokens[i] = config_.mask_token_id;
            confidences[i] = 0.0f;
            if (entropy_probs_storage) {
                entropy_probs_storage->emplace_back(static_cast<size_t>(n_vocab), 0.0f);
            }
            continue;
        }

        const float inv_sum = 1.0f / sum_exp;
        for (int j = 0; j < n_vocab; ++j) {
            probs_buf[static_cast<size_t>(j)] *= inv_sum;
        }

        // 采样：前缀和扫描替代 std::discrete_distribution，减少分配
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float r = dist(rng_);
        float cdf = 0.0f;
        int chosen = n_vocab - 1;
        for (int j = 0; j < n_vocab; ++j) {
            cdf += probs_buf[static_cast<size_t>(j)];
            if (r <= cdf) {
                chosen = j;
                break;
            }
        }

        sampled_tokens[i] = static_cast<llama_token>(chosen);
        confidences[i] = probs_buf[static_cast<size_t>(chosen)];

        if (entropy_probs_storage) {
            entropy_probs_storage->emplace_back(probs_buf.begin(), probs_buf.end());
        }
    }

    DiffusionProfiler::instance().record_custom(
        "sampler_cpu_loop_ms",
        cpu_timer.elapsed_ms()
    );
    sampler_metrics_.cpu_loop_ms += cpu_timer.elapsed_ms();
    sampler_metrics_.cpu_loop_calls++;
}

void DiffusionSampler::sample_active_tokens_cpu(
    int n_vocab,
    const std::vector<int>& active_positions,
    std::vector<llama_token>& sampled_tokens,
    std::vector<float>& confidences,
    std::vector<std::vector<float>>* entropy_probs_storage,
    const std::vector<int>* logits_positions_override
) {
    diffusion::ProfilerTimer cpu_timer;
    const int active_count = static_cast<int>(active_positions.size());
    if (entropy_probs_storage) {
        entropy_probs_storage->clear();
        entropy_probs_storage->reserve(active_count);
    }

    static thread_local std::vector<float> logits_buf;
    static thread_local std::vector<float> probs_buf;
    logits_buf.resize(static_cast<size_t>(n_vocab));
    probs_buf.resize(static_cast<size_t>(n_vocab));

    int out_count = 0; // n_outputs (packed logits rows; output_ids length == batch.n_tokens)
    const int32_t* out_ids = llama_get_logits_output_ids(ctx_, &out_count);
    int mapping_len = config_.block_length;
    if (logits_positions_override && !logits_positions_override->empty()) {
        int max_idx = -1;
        for (int v : *logits_positions_override) max_idx = std::max(max_idx, v);
        mapping_len = std::max(mapping_len, max_idx + 1);
    }
    for (int idx = 0; idx < active_count; ++idx) {
        // llama_get_logits_ith 接受 batch 内序号，需与构造的 batch 顺序一致
        // - 默认：整块 decode，logits index == 块内位置
        // - 可选：compact decode（仅活跃 token 入 batch），logits index 由 override 提供
        const int logits_idx = logits_positions_override ? (*logits_positions_override)[idx] : active_positions[idx];
        // Avoid llama.cpp "invalid logits id" by checking output_ids mapping first
        if (!out_ids || logits_idx < 0 || logits_idx >= mapping_len || out_ids[logits_idx] < 0) {
            sampled_tokens[idx] = config_.mask_token_id;
            confidences[idx] = 0.0f;
            if (entropy_probs_storage) {
                entropy_probs_storage->emplace_back(static_cast<size_t>(n_vocab), 0.0f);
            }
            continue;
        }
        float* logits = llama_get_logits_ith(ctx_, logits_idx);
        if (logits == nullptr) {
            sampled_tokens[idx] = config_.mask_token_id;
            confidences[idx] = 0.0f;
            if (entropy_probs_storage) {
                entropy_probs_storage->emplace_back(static_cast<size_t>(n_vocab), 0.0f);
            }
            continue;
        }

        float max_logit = -INFINITY;
        for (int j = 0; j < n_vocab; ++j) {
            float v = logits[j];
            if (config_.temperature != 1.0f) {
                v /= config_.temperature;
            }
            logits_buf[static_cast<size_t>(j)] = v;
            if (!std::isinf(v) && v > max_logit) {
                max_logit = v;
            }
        }

        if (config_.top_k > 0) {
            apply_top_k(logits_buf, config_.top_k);
        }
        if (config_.top_p < 1.0f) {
            apply_top_p(logits_buf, config_.top_p);
        }

        float sum_exp = 0.0f;
        for (int j = 0; j < n_vocab; ++j) {
            float l = logits_buf[static_cast<size_t>(j)];
            if (!std::isinf(l)) {
                float e = std::exp(l - max_logit);
                probs_buf[static_cast<size_t>(j)] = e;
                sum_exp += e;
            } else {
                probs_buf[static_cast<size_t>(j)] = 0.0f;
            }
        }

        if (sum_exp <= 0.0f) {
            sampled_tokens[idx] = config_.mask_token_id;
            confidences[idx] = 0.0f;
            if (entropy_probs_storage) {
                entropy_probs_storage->emplace_back(static_cast<size_t>(n_vocab), 0.0f);
            }
            continue;
        }

        const float inv_sum = 1.0f / sum_exp;
        for (int j = 0; j < n_vocab; ++j) {
            probs_buf[static_cast<size_t>(j)] *= inv_sum;
        }

        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float r = dist(rng_);
        float cdf = 0.0f;
        int chosen = n_vocab - 1;
        for (int j = 0; j < n_vocab; ++j) {
            cdf += probs_buf[static_cast<size_t>(j)];
            if (r <= cdf) {
                chosen = j;
                break;
            }
        }

        sampled_tokens[idx] = static_cast<llama_token>(chosen);
        confidences[idx] = probs_buf[static_cast<size_t>(chosen)];

        if (entropy_probs_storage) {
            entropy_probs_storage->emplace_back(probs_buf.begin(), probs_buf.end());
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
    std::vector<std::vector<float>>* entropy_probs_storage,
    double* gpu_elapsed_ms
) {
    auto env_flag = [](const char* key) {
        const char* v = std::getenv(key);
        if (!v) return false;
        if (v[0] == '0' || v[0] == 'f' || v[0] == 'F' || v[0] == 'n' || v[0] == 'N') return false;
        return true;
    };
    const bool gpu_only_mode = env_flag("DIFFUSION_GPU_ONLY") || config_.gpu_only_mode;
    const bool device_logits_env = env_flag("LLAMA_ENABLE_DEVICE_LOGITS");

    const bool device_logits_async = env_flag("LLAMA_DEVICE_LOGITS_ASYNC");
    const bool skip_sync_after_output_ids = env_flag("DIFFUSION_SKIP_SYNC_AFTER_OUTPUT_IDS");
    DIFF_LOGD("[DiffusionSampler][debug] enter try_sample_with_gpu gpu_only=%d device_logits_env=%d block_len=%d need_entropy=%d\n",
              gpu_only_mode ? 1 : 0,
              device_logits_env ? 1 : 0,
              config_.block_length,
              need_entropy_probs ? 1 : 0);

    // Host 端分阶段计时，进一步拆分 overhead
    double host_pre_ms = 0.0;
    double host_after_output_ms = 0.0;
    double host_post_ms = 0.0;
    double host_pre_get_device_ms = 0.0;
    double host_pre_get_output_ids_ms = 0.0;
    double host_pre_debug_compare_ms = 0.0;
    double host_pre_compact_ms = 0.0;
    double host_pre_sync_before_gpu_ms = 0.0;
    double host_pre_sync_before_get_device_ms = 0.0;
    double host_pre_misc_ms = 0.0;
    double host_pre_after_device_ms = 0.0;
    double host_pre_after_output_ids_ms = 0.0;
    double host_pre_after_debug_ms = 0.0;
    double host_pre_after_compact_ms = 0.0;
    double host_pre_after_sync_ms = 0.0;
    double host_pre_elapsed_after_get_device = 0.0;
    double host_pre_elapsed_after_output_ids = 0.0;
    double host_pre_elapsed_after_debug = 0.0;
    double host_pre_elapsed_after_compact = 0.0;
    double host_pre_elapsed_after_sync = 0.0;
    double host_pre_misc_before_get_device_ms = 0.0;
    double host_pre_misc_between_device_output_ms = 0.0;
    double host_pre_misc_between_output_debug_ms = 0.0;
    double host_pre_misc_between_debug_compact_ms = 0.0;
    double host_pre_misc_between_compact_sync_ms = 0.0;
    double host_pre_misc_after_sync_ms = 0.0;
    double host_pre_checkpoint_ms[6] = {0, 0, 0, 0, 0, 0};
    diffusion::ProfilerTimer host_phase_timer;

    if (!gpu_sampler_ || !gpu_sampler_->is_available()) {
        sampler_metrics_.gpu_path_device_miss++;
        sampler_metrics_.gpu_sampler_unavailable++;
        if (device_logits_env) {
            DIFF_LOGW("[DiffusionSampler][warn] GPU sampler unavailable while device logits enabled; will fallback if allowed.\n");
        }
        if (gpu_only_mode) {
            DIFF_LOGE("[DiffusionSampler][error] gpu_only_mode=true 但 GPU sampler 不可用，终止以避免访问 host logits。\n");
        }
        return false;
    }

    // Experimental: device logits path (CUDA only, when enabled upstream)
    if (device_logits_env) {
        diffusion::ProfilerTimer sync_before_get_device_timer;
        llama_synchronize(ctx_);
        host_pre_sync_before_get_device_ms = sync_before_get_device_timer.elapsed_ms();
        DiffusionProfiler::instance().record_custom(
            "sampler_gpu_host_pre_sync_before_get_device_ms",
            host_pre_sync_before_get_device_ms
        );
    } else {
        host_pre_sync_before_get_device_ms = 0.0;
        DiffusionProfiler::instance().record_custom(
            "sampler_gpu_host_pre_sync_before_get_device_ms",
            host_pre_sync_before_get_device_ms
        );
    }
    diffusion::ProfilerTimer get_device_timer;
    int64_t logits_stride = 0;
    const float* device_logits = llama_get_logits_device(ctx_, &logits_stride);
    DIFF_LOGD("[DiffusionSampler][debug] device_logits ptr=%p stride=%lld\n",
              (const void*)device_logits, (long long)logits_stride);
    double get_device_ms = get_device_timer.elapsed_ms();
    host_pre_checkpoint_ms[0] = get_device_ms;
    host_pre_get_device_ms = get_device_ms;
    host_pre_elapsed_after_get_device = host_phase_timer.elapsed_ms();
    DiffusionProfiler::instance().record_custom(
        "sampler_gpu_get_device_logits_ms",
        get_device_ms
    );
    sampler_metrics_.gpu_overhead_get_device_logits_ms += get_device_ms;
    const bool device_available = device_logits != nullptr;
    const bool stride_ok = logits_stride == n_vocab;
    bool local_stride_ok = stride_ok; // 允许在压缩后更新
    const bool full_logits = last_logits_count_ == config_.block_length;
    int debug_output_count = -1;
    const int32_t* output_ids_ptr = nullptr;
    bool can_use_device_logits =
        device_logits_env &&
        device_available &&
        full_logits &&
        config_.block_length > 0 &&
        static_cast<size_t>(config_.block_length) * static_cast<size_t>(n_vocab) == static_cast<size_t>(config_.block_length) * n_vocab;
    DIFF_LOGD("[DiffusionSampler][debug] device_available=%d stride_ok=%d full_logits=%d can_use_device_logits=%d last_logits_count=%d n_vocab=%d\n",
              device_available ? 1 : 0,
              stride_ok ? 1 : 0,
              full_logits ? 1 : 0,
              can_use_device_logits ? 1 : 0,
              last_logits_count_,
              n_vocab);

    if (need_entropy_probs) {
        sampler_metrics_.gpu_path_need_entropy++;
    }

    double local_gpu_ms = 0.0;
    const bool debug_device = std::getenv("DIFFUSION_DEBUG_DEVICE_LOGITS") != nullptr;

#if defined(DIFFUSION_ENABLE_CUDA)
    auto ensure_compact_buffer = [&](size_t bytes) -> bool {
        if (device_logits_compact_bytes_ < bytes) {
            if (device_logits_compact_) {
                cudaFree(device_logits_compact_);
                device_logits_compact_ = nullptr;
                device_logits_compact_bytes_ = 0;
            }
            if (cudaMalloc(&device_logits_compact_, bytes) != cudaSuccess) {
                device_logits_compact_ = nullptr;
                device_logits_compact_bytes_ = 0;
                return false;
            }
            device_logits_compact_bytes_ = bytes;
        }
        return device_logits_compact_ != nullptr;
    };
#endif

    if ((can_use_device_logits || debug_device) && device_available) {
        diffusion::ProfilerTimer output_ids_timer;
        output_ids_ptr = llama_get_logits_output_ids(ctx_, &debug_output_count); // also triggers output_reorder/sync
        double get_output_ids_ms = output_ids_timer.elapsed_ms();
        host_pre_checkpoint_ms[1] = get_device_ms + get_output_ids_ms;
        host_pre_get_output_ids_ms = get_output_ids_ms;
        host_pre_elapsed_after_output_ids = host_phase_timer.elapsed_ms();
        DiffusionProfiler::instance().record_custom("sampler_gpu_get_output_ids_ms", get_output_ids_ms);
        sampler_metrics_.gpu_overhead_get_output_ids_ms += get_output_ids_ms;
#ifdef LLAMA_CUDA
        if (device_logits_env && !skip_sync_after_output_ids) {
            diffusion::ProfilerTimer sync_after_output_ids_timer;
            cudaError_t sync_err_after_output_ids = cudaDeviceSynchronize();
            host_pre_misc_between_device_output_ms += sync_after_output_ids_timer.elapsed_ms();
            DiffusionProfiler::instance().record_custom(
                "sampler_gpu_host_pre_sync_after_output_ids_ms",
                sync_after_output_ids_timer.elapsed_ms());
            if (sync_err_after_output_ids != cudaSuccess) {
                DIFF_LOGW("[DiffusionSampler][warn] cudaDeviceSynchronize after get_output_ids err=%d\n",
                          int(sync_err_after_output_ids));
            }
        } else if (device_logits_env && skip_sync_after_output_ids && device_logits_async) {
            // 在异步模式下可选择跳过同步，用于观察开销；风险：可能导致日志或生成乱码
            DiffusionProfiler::instance().record_custom(
                "sampler_gpu_host_pre_sync_after_output_ids_ms",
                0.0);
        }
#endif
        if (output_ids_ptr && debug_output_count > 0) {
            int log_n = std::min(debug_output_count, 8);
            DIFF_LOGD("[DiffusionSampler][debug] output_ids count=%d first=%d %d %d %d %d %d %d %d\n",
                      debug_output_count,
                      log_n > 0 ? output_ids_ptr[0] : -1,
                      log_n > 1 ? output_ids_ptr[1] : -1,
                      log_n > 2 ? output_ids_ptr[2] : -1,
                      log_n > 3 ? output_ids_ptr[3] : -1,
                      log_n > 4 ? output_ids_ptr[4] : -1,
                      log_n > 5 ? output_ids_ptr[5] : -1,
                      log_n > 6 ? output_ids_ptr[6] : -1,
                      log_n > 7 ? output_ids_ptr[7] : -1);
        } else {
            DIFF_LOGD("[DiffusionSampler][debug] output_ids missing count=%d ptr=%p\n",
                      debug_output_count, (const void*)output_ids_ptr);
        }
    }

#if defined(DIFFUSION_ENABLE_CUDA)
    // Debug: host vs device logits diff to catch reorder/stride issues
    if (debug_device && device_available && stride_ok && debug_output_count >= config_.block_length) {
#ifdef LLAMA_CUDA
        cudaDeviceSynchronize();
#endif
        diffusion::ProfilerTimer debug_compare_timer;
        double max_abs_diff = 0.0;
        int max_row = -1;
        int max_col = -1;
        for (int row = 0; row < config_.block_length; ++row) {
            const float* host_row = llama_get_logits_ith(ctx_, row);
            const float* dev_row = device_logits + static_cast<size_t>(row) * static_cast<size_t>(logits_stride);
            if (!host_row || !dev_row) continue;

            // 先把设备行复制到主机，再比较，避免在主机直接解引用设备指针
            std::vector<float> dev_row_host(n_vocab);
            cudaError_t err = cudaMemcpy(dev_row_host.data(), dev_row,
                                         static_cast<size_t>(n_vocab) * sizeof(float),
                                         cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                DIFF_LOGD("[device_logits_debug] cudaMemcpy row=%d err=%d\n", row, int(err));
                continue;
            }
            for (int col = 0; col < n_vocab; ++col) {
                float diff = std::fabs(host_row[col] - dev_row_host[col]);
                if (diff > max_abs_diff) {
                    max_abs_diff = diff;
                    max_row = row;
                    max_col = col;
                }
            }
        }
        if (max_abs_diff > 1e-3f) {
            DIFF_LOGD("[device_logits_debug] max_abs_diff=%f at row=%d col=%d\n",
                      max_abs_diff, max_row, max_col);
        } else {
            DIFF_LOGD("[device_logits_debug] device logits match host (max_abs_diff=%f)\n",
                      max_abs_diff);
        }
        double debug_compare_ms = debug_compare_timer.elapsed_ms();
        host_pre_checkpoint_ms[2] = host_pre_checkpoint_ms[1] + debug_compare_ms;
        host_pre_debug_compare_ms = debug_compare_ms;
        DiffusionProfiler::instance().record_custom("sampler_gpu_debug_compare_ms", debug_compare_ms);
        sampler_metrics_.gpu_overhead_debug_compare_ms += debug_compare_ms;
    }
#endif
    host_pre_elapsed_after_debug = host_phase_timer.elapsed_ms();

    const float* device_logits_ptr = device_logits;
    bool used_compact = false;
    if (can_use_device_logits) {
#if defined(DIFFUSION_ENABLE_CUDA)
        const int output_count = debug_output_count;
        const size_t expected_rows = static_cast<size_t>(config_.block_length);
        const size_t total_rows_available = static_cast<size_t>(output_count);
        // 始终按 output_ids 重新压缩/排序 device logits，保证与 host 顺序一致
        if (output_ids_ptr && output_count >= static_cast<int>(expected_rows)) {
            diffusion::ProfilerTimer compact_timer;
            const size_t needed_bytes = expected_rows * static_cast<size_t>(n_vocab) * sizeof(float);
            if (ensure_compact_buffer(needed_bytes)) {
                float* compact_ptr = device_logits_compact_;
                for (size_t row = 0; row < expected_rows; ++row) {
                    const int src_row = output_ids_ptr[row];
                    if (src_row < 0) { continue; }
                    const float* src = device_logits + static_cast<size_t>(src_row) * static_cast<size_t>(logits_stride);
                    float* dst = compact_ptr + row * static_cast<size_t>(n_vocab);
                    cudaMemcpy(dst, src, static_cast<size_t>(n_vocab) * sizeof(float), cudaMemcpyDeviceToDevice);
                }
                device_logits_ptr = compact_ptr;
                used_compact = true;
                logits_stride = n_vocab;
                local_stride_ok = true;
            } else {
                DIFF_LOGW("[device_logits] compact buffer alloc failed bytes=%zu\n", needed_bytes);
            }
            host_pre_compact_ms += compact_timer.elapsed_ms();
            host_pre_checkpoint_ms[3] = host_pre_checkpoint_ms[2] + host_pre_compact_ms;
        }

        if (output_count < static_cast<int>(expected_rows)) {
            can_use_device_logits = false;
            sampler_metrics_.gpu_fallback_compact_fail++;
            if (debug_device) {
                DIFF_LOGW("[device_logits] fallback: output_count=%d expected=%zu\n",
                          output_count, expected_rows);
            }
        } else if (!local_stride_ok) {
            // stride 不匹配且无法压缩，放弃设备 logits
            can_use_device_logits = false;
            if (debug_device) {
                DIFF_LOGW("[device_logits] fallback: stride mismatch logits_stride=%lld n_vocab=%d\n",
                          (long long)logits_stride, n_vocab);
            }
        } else {
            // stride 已匹配或已压缩，直接使用 device logits 指针，避免多余 H2D 拷贝
            if (!used_compact) {
                device_logits_ptr = device_logits;
            }
            if (debug_device) {
                DIFF_LOGD("[device_logits][debug] use device pointer directly rows=%zu available=%zu\n",
                          expected_rows, total_rows_available);
            }
        }
#else
        can_use_device_logits = false;
#endif
    }
#ifdef LLAMA_CUDA
    // 额外同步：压缩/重排后的 device logits 落稳，避免后续采样读取未完成数据
    if (device_logits_env && can_use_device_logits) {
        diffusion::ProfilerTimer sync_after_compact_timer;
        cudaError_t sync_err_after_compact = cudaDeviceSynchronize();
        DiffusionProfiler::instance().record_custom(
            "sampler_gpu_host_pre_sync_after_compact_ms",
            sync_after_compact_timer.elapsed_ms());
        host_pre_misc_between_debug_compact_ms += sync_after_compact_timer.elapsed_ms();
        if (sync_err_after_compact != cudaSuccess) {
            DIFF_LOGW("[DiffusionSampler][warn] cudaDeviceSynchronize after compact err=%d\n",
                      int(sync_err_after_compact));
        }
    }
#endif
    host_pre_elapsed_after_compact = host_phase_timer.elapsed_ms();

#ifdef LLAMA_CUDA
    // 进入 device 分支前再同步一次，避免残留异步写导致长序列乱码
    if (device_logits_env && can_use_device_logits) {
        diffusion::ProfilerTimer sync_before_device_branch_timer;
        cudaError_t sync_err_before_device_branch = cudaDeviceSynchronize();
        DiffusionProfiler::instance().record_custom(
            "sampler_gpu_host_pre_sync_before_device_branch_ms",
            sync_before_device_branch_timer.elapsed_ms());
        host_pre_misc_between_compact_sync_ms += sync_before_device_branch_timer.elapsed_ms();
        if (sync_err_before_device_branch != cudaSuccess) {
            DIFF_LOGW("[DiffusionSampler][warn] cudaDeviceSynchronize before device branch err=%d\n",
                      int(sync_err_before_device_branch));
        }
    }
#endif


    DIFF_LOGD("[DiffusionSampler][debug] device branch check can_use_device_logits=%d\n",
              can_use_device_logits ? 1 : 0);
#ifdef LLAMA_CUDA
    cudaError_t pre_err = cudaGetLastError();
    DIFF_LOGD("[DiffusionSampler][debug] pre-branch cudaGetLastError=%d\n", int(pre_err));
#endif

    if (can_use_device_logits) {
        DIFF_LOGD("[DiffusionSampler][debug] enter device branch before call\n");
        DIFF_LOGD("[DiffusionSampler][debug] calling sample_from_device_ptr\n");
        {
#ifdef LLAMA_CUDA
            // 确保上游 logits 写入完成，避免读取未完成的 device logits
            diffusion::ProfilerTimer sync_timer;
            cudaError_t sync_err = cudaDeviceSynchronize();
            host_pre_sync_before_gpu_ms += sync_timer.elapsed_ms();
            host_pre_checkpoint_ms[4] = host_pre_checkpoint_ms[3] + host_pre_sync_before_gpu_ms;
            if (sync_err != cudaSuccess) {
                DIFF_LOGW("[DiffusionSampler][warn] cudaDeviceSynchronize before device sample failed err=%d\n", int(sync_err));
            }
#endif
        }
        host_pre_elapsed_after_sync = host_phase_timer.elapsed_ms();
        diffusion::ProfilerTimer gpu_timer;
        GpuSampler::Stats gpu_stats{};
        // Fast-path验证：默认使用 device 非融合路径（GpuSampler 内部可选融合）
        bool sampled_with_gpu = false;
        auto rng_base = rng_;
        host_pre_ms = host_phase_timer.elapsed_ms();
        host_phase_timer = diffusion::ProfilerTimer();

        sampled_with_gpu = gpu_sampler_->sample_from_device_ptr(
            device_logits_ptr,
            static_cast<size_t>(config_.block_length) * static_cast<size_t>(n_vocab),
            config_.remasking_strategy,
            rng_,
            sampled_tokens,
            confidences,
            need_entropy_probs ? entropy_probs_storage : nullptr,
            &gpu_stats,
            /*force_non_fused=*/false);
        DIFF_LOGD("[DiffusionSampler][debug] sample_from_device_ptr finished call path\n");
        DIFF_LOGD("[DiffusionSampler][debug] sample_from_device_ptr returned=%d tokens=%zu\n",
                  sampled_with_gpu ? 1 : 0, sampled_tokens.size());
        // 重新计时，记录 GPU 调用后的主机处理阶段
#ifdef LLAMA_CUDA
        // 非调试路径补齐同步，避免残留异步写导致后续读写错乱
        if (device_logits_env && can_use_device_logits) {
            cudaError_t sync_err_after_device_sample = cudaDeviceSynchronize();
            if (sync_err_after_device_sample != cudaSuccess) {
                DIFF_LOGW("[DiffusionSampler][warn] cudaDeviceSynchronize after device sample err=%d\n",
                          int(sync_err_after_device_sample));
            }
        }
#endif
        host_phase_timer = diffusion::ProfilerTimer();

        // 可选：同时运行 fast-path（融合内核）对比
        const bool enable_fastpath_compare = std::getenv("DIFFUSION_FASTPATH_COMPARE") != nullptr;
        bool fastpath_ok = false;
        std::vector<llama_token> fast_tokens;
        std::vector<float> fast_conf;
        if (enable_fastpath_compare && !need_entropy_probs) {
            auto rng_fast = rng_base;
            GpuSampler::Stats fast_stats{};
            fast_tokens.resize(config_.block_length);
            fast_conf.resize(config_.block_length);
            fastpath_ok = gpu_sampler_->sample_from_device_ptr(
                device_logits_ptr,
                static_cast<size_t>(config_.block_length) * static_cast<size_t>(n_vocab),
                config_.remasking_strategy,
                rng_fast,
                fast_tokens,
                fast_conf,
                nullptr,
                &fast_stats,
                /*force_non_fused=*/false
            );
        }

        // 对比 fast-path 与非融合 device 路径的 token 结果，只记录日志
        if (enable_fastpath_compare && fastpath_ok && sampled_with_gpu && fast_tokens.size() == sampled_tokens.size()) {
            int mismatch = 0;
            for (size_t i = 0; i < fast_tokens.size(); ++i) {
                if (fast_tokens[i] != sampled_tokens[i]) {
                    mismatch++;
                }
            }
            DiffusionProfiler::instance().record_custom("sampler_gpu_fastpath_mismatch", mismatch);
            if (mismatch > 0) {
                DIFF_LOGD("[fastpath_compare] mismatches=%d / %zu\n", mismatch, fast_tokens.size());
            }
        }
        double invoke_ms = gpu_timer.elapsed_ms();
        local_gpu_ms += invoke_ms;
        DiffusionProfiler::instance().record_custom("sampler_gpu_invoke_ms", invoke_ms);
        DiffusionProfiler::instance().record_custom("sampler_gpu_stage_prepare_ms", gpu_stats.stage_prepare_ms);
        DiffusionProfiler::instance().record_custom("sampler_gpu_stage_softmax_ms", gpu_stats.stage_softmax_ms);
        DiffusionProfiler::instance().record_custom("sampler_gpu_stage_sort_ms", gpu_stats.stage_sort_ms);
        DiffusionProfiler::instance().record_custom("sampler_gpu_stage_sample_ms", gpu_stats.stage_sample_ms);
        DiffusionProfiler::instance().record_custom("sampler_gpu_stage_d2h_ms", gpu_stats.stage_d2h_ms);
        DiffusionProfiler::instance().record_custom("sampler_gpu_stage_cpu_post_ms", gpu_stats.stage_cpu_post_ms);
        DiffusionProfiler::instance().record_custom("sampler_gpu_event_wait_ms", gpu_stats.stage_event_wait_ms);
        DiffusionProfiler::instance().record_custom("sampler_gpu_prepare_copy_ms", gpu_stats.stage_prepare_copy_ms);
        DiffusionProfiler::instance().record_custom("sampler_gpu_prepare_temp_ms", gpu_stats.stage_prepare_temp_ms);
        DiffusionProfiler::instance().record_custom("sampler_gpu_prepare_mask_ms", gpu_stats.stage_prepare_mask_ms);
        DiffusionProfiler::instance().record_custom("sampler_gpu_prepare_rng_ms", gpu_stats.stage_prepare_rng_ms);
        const double stage_sum = gpu_stats.stage_prepare_ms + gpu_stats.stage_softmax_ms +
                                 gpu_stats.stage_sort_ms + gpu_stats.stage_sample_ms +
                                 gpu_stats.stage_d2h_ms + gpu_stats.stage_cpu_post_ms +
                                 gpu_stats.stage_event_wait_ms;
        const double gap_ms = std::max(0.0, invoke_ms - stage_sum);
        DiffusionProfiler::instance().record_custom("sampler_gpu_gap_ms", gap_ms);
        sampler_metrics_.gpu_invoke_ms += invoke_ms;
        sampler_metrics_.gpu_invoke_calls++;
        sampler_metrics_.gpu_path_device_hit++;

        if (sampled_with_gpu) {
            sampler_metrics_.gpu_success++;
            sampler_metrics_.gpu_stage_prepare_ms += gpu_stats.stage_prepare_ms;
            sampler_metrics_.gpu_prepare_copy_ms += gpu_stats.stage_prepare_copy_ms;
            sampler_metrics_.gpu_prepare_temp_ms += gpu_stats.stage_prepare_temp_ms;
            sampler_metrics_.gpu_prepare_mask_ms += gpu_stats.stage_prepare_mask_ms;
            sampler_metrics_.gpu_prepare_rng_ms += gpu_stats.stage_prepare_rng_ms;
            sampler_metrics_.gpu_stage_softmax_ms += gpu_stats.stage_softmax_ms;
            sampler_metrics_.gpu_stage_sort_ms += gpu_stats.stage_sort_ms;
            sampler_metrics_.gpu_stage_sample_ms += gpu_stats.stage_sample_ms;
            sampler_metrics_.gpu_stage_d2h_ms += gpu_stats.stage_d2h_ms;
            sampler_metrics_.gpu_stage_cpu_post_ms += gpu_stats.stage_cpu_post_ms;
            sampler_metrics_.gpu_stage_event_wait_ms += gpu_stats.stage_event_wait_ms;
            sampler_metrics_.gpu_gap_ms += gap_ms;
            sampler_metrics_.gpu_total_ms += local_gpu_ms;
            if (gpu_stats.fast_path) {
                sampler_metrics_.gpu_device_fast_path++;
            }
            if (need_entropy_probs && entropy_probs_storage) {
                // entropy_probs_storage is unused in device path fast sampling; keep empty
            }
            if (gpu_elapsed_ms) {
                *gpu_elapsed_ms = local_gpu_ms;
            }
        host_post_ms = host_phase_timer.elapsed_ms();
        host_pre_checkpoint_ms[5] = host_pre_ms;
        // 这里的 after_* 表示各阶段自身耗时（非前缀差），便于排查
        if (host_pre_elapsed_after_output_ids <= 0.0) host_pre_elapsed_after_output_ids = host_pre_elapsed_after_get_device;
        if (host_pre_elapsed_after_debug <= 0.0) host_pre_elapsed_after_debug = host_pre_elapsed_after_output_ids;
        if (host_pre_elapsed_after_compact <= 0.0) host_pre_elapsed_after_compact = host_pre_elapsed_after_debug;
        if (host_pre_elapsed_after_sync <= 0.0) host_pre_elapsed_after_sync = host_pre_elapsed_after_compact;

        host_pre_misc_before_get_device_ms = std::max(0.0, host_pre_elapsed_after_get_device - host_pre_get_device_ms - host_pre_sync_before_get_device_ms);
        host_pre_misc_between_device_output_ms = std::max(0.0, host_pre_elapsed_after_output_ids - host_pre_elapsed_after_get_device - host_pre_get_output_ids_ms);
        host_pre_misc_between_output_debug_ms = std::max(0.0, host_pre_elapsed_after_debug - host_pre_elapsed_after_output_ids - host_pre_debug_compare_ms);
        host_pre_misc_between_debug_compact_ms = std::max(0.0, host_pre_elapsed_after_compact - host_pre_elapsed_after_debug - host_pre_compact_ms);
        host_pre_misc_between_compact_sync_ms = std::max(0.0, host_pre_elapsed_after_sync - host_pre_elapsed_after_compact - host_pre_sync_before_gpu_ms);
        host_pre_misc_after_sync_ms = std::max(0.0, host_pre_ms - host_pre_elapsed_after_sync);

        // 兼容旧指标：after_* 复用为各阶段 gap
        host_pre_after_device_ms = host_pre_misc_before_get_device_ms;
        host_pre_after_output_ids_ms = host_pre_misc_between_device_output_ms;
        host_pre_after_debug_ms = host_pre_misc_between_output_debug_ms;
        host_pre_after_compact_ms = host_pre_misc_between_debug_compact_ms;
        host_pre_after_sync_ms = host_pre_misc_between_compact_sync_ms;
        host_pre_misc_ms = host_pre_misc_after_sync_ms;

        DiffusionProfiler::instance().record_custom("sampler_gpu_overhead_host_pre_ms", host_pre_ms);
        DiffusionProfiler::instance().record_custom("sampler_gpu_overhead_host_after_output_ms", host_after_output_ms);
        DiffusionProfiler::instance().record_custom("sampler_gpu_overhead_host_post_ms", host_post_ms);
        DiffusionProfiler::instance().record_custom("sampler_gpu_host_pre_get_device_ms", host_pre_get_device_ms);
        DiffusionProfiler::instance().record_custom("sampler_gpu_host_pre_get_output_ids_ms", host_pre_get_output_ids_ms);
        DiffusionProfiler::instance().record_custom("sampler_gpu_host_pre_debug_compare_ms", host_pre_debug_compare_ms);
        DiffusionProfiler::instance().record_custom("sampler_gpu_host_pre_compact_ms", host_pre_compact_ms);
        DiffusionProfiler::instance().record_custom("sampler_gpu_host_pre_sync_ms", host_pre_sync_before_gpu_ms);
        DiffusionProfiler::instance().record_custom("sampler_gpu_host_pre_sync_before_get_device_ms", host_pre_sync_before_get_device_ms);
        DiffusionProfiler::instance().record_custom("sampler_gpu_host_pre_misc_ms", host_pre_misc_ms);
        DiffusionProfiler::instance().record_custom("sampler_gpu_host_pre_after_device_ms", host_pre_after_device_ms);
        DiffusionProfiler::instance().record_custom("sampler_gpu_host_pre_after_output_ids_ms", host_pre_after_output_ids_ms);
        DiffusionProfiler::instance().record_custom("sampler_gpu_host_pre_after_debug_ms", host_pre_after_debug_ms);
        DiffusionProfiler::instance().record_custom("sampler_gpu_host_pre_after_compact_ms", host_pre_after_compact_ms);
        DiffusionProfiler::instance().record_custom("sampler_gpu_host_pre_after_sync_ms", host_pre_after_sync_ms);
        DiffusionProfiler::instance().record_custom("sampler_gpu_host_pre_misc_before_get_device_ms", host_pre_misc_before_get_device_ms);
        DiffusionProfiler::instance().record_custom("sampler_gpu_host_pre_misc_between_device_output_ms", host_pre_misc_between_device_output_ms);
        DiffusionProfiler::instance().record_custom("sampler_gpu_host_pre_misc_between_output_debug_ms", host_pre_misc_between_output_debug_ms);
        DiffusionProfiler::instance().record_custom("sampler_gpu_host_pre_misc_between_debug_compact_ms", host_pre_misc_between_debug_compact_ms);
        DiffusionProfiler::instance().record_custom("sampler_gpu_host_pre_misc_between_compact_sync_ms", host_pre_misc_between_compact_sync_ms);
        DiffusionProfiler::instance().record_custom("sampler_gpu_host_pre_misc_after_sync_ms", host_pre_misc_after_sync_ms);

        sampler_metrics_.gpu_overhead_host_pre_ms += host_pre_ms;
        sampler_metrics_.gpu_overhead_host_after_output_ms += host_after_output_ms;
        sampler_metrics_.gpu_overhead_host_post_ms += host_post_ms;
            return true;
        }
        // fall through to host path on failure
        sampler_metrics_.gpu_fail++;
        sampler_metrics_.gpu_path_device_miss++;
    } else {
        if (!device_available) {
            sampler_metrics_.gpu_fallback_device_unavail++;
        } else if (!stride_ok) {
            sampler_metrics_.gpu_fallback_stride++;
            sampler_metrics_.gpu_fallback_stride_mismatch++;
        } else if (!full_logits) {
            sampler_metrics_.gpu_fallback_stride++;
            sampler_metrics_.gpu_fallback_partial_logits++;
        }
        if (debug_device) {
            DIFF_LOGW("[device_logits] fallback: device=%d stride_ok=%d full=%d stride=%lld n_vocab=%d block=%d last=%d output_count=%d\n",
                      device_available ? 1 : 0,
                      stride_ok ? 1 : 0,
                      full_logits ? 1 : 0,
                      static_cast<long long>(logits_stride),
                      n_vocab,
                      config_.block_length,
                      last_logits_count_,
                      debug_output_count);
        }
        if (config_.top_k > 0) {
            sampler_metrics_.gpu_fallback_topk++;
        }
        if (config_.top_p < 1.0f) {
            sampler_metrics_.gpu_fallback_topp++;
        }
        if (need_entropy_probs) {
            sampler_metrics_.gpu_fallback_entropy++;
        }
    }

    // 使用 scatter 指针，直接传递 logits 指针数组，避免 CPU 拼接拷贝
    diffusion::ProfilerTimer pack_timer;
    std::vector<float*> logits_ptrs(config_.block_length);
    // Avoid llama.cpp "invalid logits id" by checking output_ids mapping first
    // NOTE: llama.cpp fills the out_count parameter with n_outputs (rows in packed logits buffer),
    // which is NOT the mapping length (mapping length == batch.n_tokens).
    int out_count_host = 0;
    const int32_t* out_ids_host = llama_get_logits_output_ids(ctx_, &out_count_host);
    if (!out_ids_host) {
        return false;
    }
    for (int i = 0; i < config_.block_length; ++i) {
        if (out_ids_host[i] < 0) {
            return false;
        }
        float* logits = llama_get_logits_ith(ctx_, i);
        if (logits == nullptr) {
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

    std::vector<std::vector<float>> tmp_probs;
    std::vector<std::vector<float>>* probs_ptr = (need_entropy_probs && entropy_probs_storage)
        ? &tmp_probs
        : nullptr;

    diffusion::ProfilerTimer gpu_timer;
    GpuSampler::Stats gpu_stats{};
    host_pre_ms = host_phase_timer.elapsed_ms();
    host_phase_timer = diffusion::ProfilerTimer();

    bool sampled_with_gpu = gpu_sampler_->sample_from_scatter_ptrs(
        logits_ptrs,
        n_vocab,
        config_.remasking_strategy,
        rng_,
        sampled_tokens,
        confidences,
        probs_ptr,
        &gpu_stats
    );
    
    double invoke_ms = gpu_timer.elapsed_ms();
    local_gpu_ms += invoke_ms;
    DiffusionProfiler::instance().record_custom("sampler_gpu_invoke_ms", invoke_ms);
    DiffusionProfiler::instance().record_custom("sampler_gpu_stage_prepare_ms", gpu_stats.stage_prepare_ms);
    DiffusionProfiler::instance().record_custom("sampler_gpu_stage_softmax_ms", gpu_stats.stage_softmax_ms);
    DiffusionProfiler::instance().record_custom("sampler_gpu_stage_sort_ms", gpu_stats.stage_sort_ms);
    DiffusionProfiler::instance().record_custom("sampler_gpu_stage_sample_ms", gpu_stats.stage_sample_ms);
    DiffusionProfiler::instance().record_custom("sampler_gpu_stage_d2h_ms", gpu_stats.stage_d2h_ms);
    DiffusionProfiler::instance().record_custom("sampler_gpu_stage_cpu_post_ms", gpu_stats.stage_cpu_post_ms);
    DiffusionProfiler::instance().record_custom("sampler_gpu_event_wait_ms", gpu_stats.stage_event_wait_ms);
    DiffusionProfiler::instance().record_custom("sampler_gpu_prepare_copy_ms", gpu_stats.stage_prepare_copy_ms);
    DiffusionProfiler::instance().record_custom("sampler_gpu_prepare_temp_ms", gpu_stats.stage_prepare_temp_ms);
    DiffusionProfiler::instance().record_custom("sampler_gpu_prepare_mask_ms", gpu_stats.stage_prepare_mask_ms);
    DiffusionProfiler::instance().record_custom("sampler_gpu_prepare_rng_ms", gpu_stats.stage_prepare_rng_ms);
    const double stage_sum = gpu_stats.stage_prepare_ms + gpu_stats.stage_softmax_ms +
                             gpu_stats.stage_sort_ms + gpu_stats.stage_sample_ms +
                             gpu_stats.stage_d2h_ms + gpu_stats.stage_cpu_post_ms +
                             gpu_stats.stage_event_wait_ms;
    const double gap_ms = std::max(0.0, invoke_ms - stage_sum);
    DiffusionProfiler::instance().record_custom("sampler_gpu_gap_ms", gap_ms);
    sampler_metrics_.gpu_invoke_ms += invoke_ms;
    sampler_metrics_.gpu_invoke_calls++;

    if (!sampled_with_gpu) {
        sampler_metrics_.gpu_fail++;
        return false;
    }
    host_phase_timer = diffusion::ProfilerTimer();

    sampler_metrics_.gpu_success++;
    sampler_metrics_.gpu_total_ms += local_gpu_ms;
    sampler_metrics_.gpu_stage_prepare_ms += gpu_stats.stage_prepare_ms;
    sampler_metrics_.gpu_prepare_copy_ms += gpu_stats.stage_prepare_copy_ms;
    sampler_metrics_.gpu_prepare_temp_ms += gpu_stats.stage_prepare_temp_ms;
    sampler_metrics_.gpu_prepare_mask_ms += gpu_stats.stage_prepare_mask_ms;
    sampler_metrics_.gpu_prepare_rng_ms += gpu_stats.stage_prepare_rng_ms;
    sampler_metrics_.gpu_stage_softmax_ms += gpu_stats.stage_softmax_ms;
    sampler_metrics_.gpu_stage_sort_ms += gpu_stats.stage_sort_ms;
    sampler_metrics_.gpu_stage_sample_ms += gpu_stats.stage_sample_ms;
    sampler_metrics_.gpu_stage_d2h_ms += gpu_stats.stage_d2h_ms;
    sampler_metrics_.gpu_stage_cpu_post_ms += gpu_stats.stage_cpu_post_ms;
    sampler_metrics_.gpu_stage_event_wait_ms += gpu_stats.stage_event_wait_ms;
    sampler_metrics_.gpu_gap_ms += gap_ms;
    if (gpu_stats.fast_path) {
        sampler_metrics_.gpu_fast_path++;
    }

    if (need_entropy_probs && entropy_probs_storage && probs_ptr) {
        *entropy_probs_storage = std::move(tmp_probs);
    }
    if (gpu_elapsed_ms) {
        *gpu_elapsed_ms = local_gpu_ms;
    }
    host_after_output_ms = host_phase_timer.elapsed_ms();
    host_phase_timer = diffusion::ProfilerTimer();
    host_post_ms = host_phase_timer.elapsed_ms();
    DiffusionProfiler::instance().record_custom("sampler_gpu_overhead_host_pre_ms", host_pre_ms);
    DiffusionProfiler::instance().record_custom("sampler_gpu_overhead_host_after_output_ms", host_after_output_ms);
    DiffusionProfiler::instance().record_custom("sampler_gpu_overhead_host_post_ms", host_post_ms);
    sampler_metrics_.gpu_overhead_host_pre_ms += host_pre_ms;
    sampler_metrics_.gpu_overhead_host_after_output_ms += host_after_output_ms;
    sampler_metrics_.gpu_overhead_host_post_ms += host_post_ms;
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

#ifndef DIFFUSION_SAMPLER_H
#define DIFFUSION_SAMPLER_H

#include "diffusion_types.h"
#include "llama.h"
#include <functional>
#include <limits>
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
    double gpu_stage_softmax_ms = 0.0;
    double gpu_stage_sort_ms = 0.0;
    double gpu_stage_sample_ms = 0.0;
    double gpu_stage_d2h_ms = 0.0;
    double gpu_stage_cpu_post_ms = 0.0;
    double gpu_stage_event_wait_ms = 0.0;
    double gpu_prepare_copy_ms = 0.0;
    double gpu_prepare_temp_ms = 0.0;
    double gpu_prepare_mask_ms = 0.0;
    double gpu_prepare_rng_ms = 0.0;
    double gpu_gap_ms = 0.0; // invoke_wall - sum(stages) 的残差
    double gpu_total_ms = 0.0;
    double gpu_overhead_ms = 0.0;
    double gpu_overhead_get_device_logits_ms = 0.0;
    double gpu_overhead_get_output_ids_ms = 0.0;
    double gpu_overhead_debug_compare_ms = 0.0;
    double gpu_overhead_host_pre_ms = 0.0;
    double gpu_overhead_host_after_output_ms = 0.0;
    double gpu_overhead_host_post_ms = 0.0;
    int gpu_success = 0;
    int gpu_fail = 0;
    int gpu_path_device_hit = 0;
    int gpu_path_device_miss = 0;
    int gpu_path_need_entropy = 0;
    int gpu_fast_path = 0;
    int gpu_device_fast_path = 0;
    int gpu_fallback_topk = 0;
    int gpu_fallback_topp = 0;
    int gpu_fallback_entropy = 0;
    int gpu_fallback_stride = 0;
    int gpu_fallback_stride_mismatch = 0;
    int gpu_fallback_partial_logits = 0;
    int gpu_fallback_compact_fail = 0;
    int gpu_fallback_device_unavail = 0;
    int gpu_sampler_unavailable = 0;

    double cpu_sampling_ms = 0.0;
    int cpu_sampling_calls = 0;
    double cpu_loop_ms = 0.0;
    int cpu_loop_calls = 0;

    // Partial KV/logits reuse telemetry (experimental)
    int partial_kv_attempt = 0;
    int partial_kv_used = 0;
    int partial_kv_fallback = 0;

    // Micro-block scheduling / KV reuse effectiveness telemetry (CPU+GPU)
    // These are counters (not ms) and are used to understand why block scaling regresses.
    int denoise_step_count = 0; // total denoising steps executed (across all blocks)

    long long active_count_sum = 0;
    int active_count_samples = 0;
    int active_count_min = (std::numeric_limits<int>::max)();
    int active_count_max = 0;

    long long decode_count_sum = 0;
    int decode_count_samples = 0;
    int decode_count_min = (std::numeric_limits<int>::max)();
    int decode_count_max = 0;
    int decode_full_steps = 0;    // decode_count == block_length
    int decode_partial_steps = 0; // decode_count <  block_length

    int kv_rm_calls = 0;
    int kv_rm_full_calls = 0;
    int kv_rm_partial_calls = 0;
    long long kv_rm_tokens = 0; // total tokens cleared via llama_memory_seq_rm

    int llama_decode_calls = 0;
    long long llama_decode_tokens = 0; // total tokens passed to llama_decode (batch.n_tokens)

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
    int last_logits_count_ = 0;
    // For the last llama_decode call, record which block positions had logits requested (batch.logits[i] == true).
    // This allows GPU sampling to work even when n_outputs < block_length (e.g. frozen micro-blocks).
    std::vector<int> last_logits_positions_;
    // The diffusion "mask token" may come from tokenizer specials and can be out-of-vocab for the gguf model.
    // We keep config_.mask_token_id as the logical marker, but map it to an in-vocab placeholder when feeding llama.cpp.
    llama_token mask_token_id_for_model_ = 0;
    int vocab_size_ = 0;
    llama_token oov_token_fallback_id_ = 0;
    // Reusable buffers for full-block GPU sampling in micro-block path
    std::vector<llama_token> gpu_sampled_block_buffer_;
    std::vector<float> gpu_conf_block_buffer_;
    std::vector<std::vector<float>> gpu_entropy_block_buffer_;
#if defined(DIFFUSION_ENABLE_CUDA)
    float* device_logits_compact_ = nullptr;
    size_t device_logits_compact_bytes_ = 0;
#endif

    int get_vocab_size();
    llama_token sanitize_token_for_model(llama_token t) const {
        if (t < 0 || t >= vocab_size_) return oov_token_fallback_id_;
        return t;
    }

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

    // 针对微块/子集的 CPU 采样（active_positions 是 block 内的下标）
    void sample_active_tokens_cpu(
        int n_vocab,
        const std::vector<int>& active_positions,
        std::vector<llama_token>& sampled_tokens,
        std::vector<float>& confidences,
        std::vector<std::vector<float>>* entropy_probs_storage,
        const std::vector<int>* logits_positions_override = nullptr
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
        std::vector<std::vector<float>>* entropy_probs_storage,
        double* gpu_elapsed_ms = nullptr
    );
};

} // namespace diffusion

#endif // DIFFUSION_SAMPLER_H

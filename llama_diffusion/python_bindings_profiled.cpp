// python_bindings_profiled.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <limits>
#include "diffusion_sampler_profiled.h"
#include "diffusion_profiler.h"
#include "llama.h"

namespace py = pybind11;

class LlamaDiffusionProfiledWrapper {
public:
    LlamaDiffusionProfiledWrapper(const std::string& model_path, int n_ctx = 32768, int n_gpu_layers = 0) 
        : n_ctx_(n_ctx), n_gpu_layers_(n_gpu_layers) {
        llama_backend_init();
        
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = n_gpu_layers;
        
        model_ = llama_model_load_from_file(model_path.c_str(), model_params);
        if (!model_) {
            throw std::runtime_error("Failed to load model from: " + model_path);
        }
    }
    
    ~LlamaDiffusionProfiledWrapper() {
        if (model_) llama_model_free(model_);
        llama_backend_free();
    }
    
    std::pair<std::vector<int>, py::dict> generate_with_profiling(
        const std::vector<int>& prompt,
        int mask_token_id,
        int gen_length = 128,
        int block_length = 8,
        int denoising_steps = 8,
        float temperature = 1.0f,
        int top_k = 0,
        float top_p = 1.0f,
        const std::string& remasking_strategy = "low_confidence_dynamic",
        float confidence_threshold = 0.85f,
        float eb_threshold = 0.35f,
        const std::vector<int>& stop_token_ids = {},
        bool use_gpu_sampler = false,
        int micro_block_size = -1
    ) {
        std::vector<llama_token> llama_prompt(prompt.begin(), prompt.end());
        
        diffusion::DiffusionConfig config;
        config.gen_length = gen_length;
        config.block_length = block_length;
        config.micro_block_size = (micro_block_size > 0) ? micro_block_size : block_length;
        config.denoising_steps = denoising_steps;
        config.temperature = temperature;
        config.top_k = top_k;
        config.top_p = top_p;
        config.confidence_threshold = confidence_threshold;
        config.eb_threshold = eb_threshold;
        config.mask_token_id = mask_token_id;
        config.stop_token_ids.assign(stop_token_ids.begin(), stop_token_ids.end());
        config.enable_gpu_sampler = use_gpu_sampler;
        
        if (remasking_strategy == "sequential") {
            config.remasking_strategy = diffusion::RemaskingStrategy::SEQUENTIAL;
        } else if (remasking_strategy == "low_confidence_static") {
            config.remasking_strategy = diffusion::RemaskingStrategy::LOW_CONFIDENCE_STATIC;
        } else if (remasking_strategy == "low_confidence_dynamic") {
            config.remasking_strategy = diffusion::RemaskingStrategy::LOW_CONFIDENCE_DYNAMIC;
        } else if (remasking_strategy == "entropy_bounded") {
            config.remasking_strategy = diffusion::RemaskingStrategy::ENTROPY_BOUNDED;
        }
        
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = n_ctx_;
        ctx_params.n_seq_max = 2;
        ctx_params.block_size = block_length;
        ctx_params.flash_attn_type = block_length > 0 ? LLAMA_FLASH_ATTN_TYPE_BLOCK_CAUSAL
                                                     : LLAMA_FLASH_ATTN_TYPE_ENABLED; // Enable flash attention for better performance
        
        llama_context* ctx = llama_init_from_model(model_, ctx_params);
        if (!ctx) {
            throw std::runtime_error("Failed to create context");
        }
        
        diffusion::DiffusionSamplerProfiled sampler(ctx, model_, config);
        std::vector<llama_token> result = sampler.generate_with_profiling(llama_prompt);
        
        auto profile_summary = sampler.get_profile_summary();
        const auto& custom_metrics = diffusion::DiffusionProfiler::instance().get_custom_metrics();
        
        llama_free(ctx);
        
        // Convert profile summary to Python dict
        py::dict py_profile;
        for (const auto& outer : profile_summary) {
            py::dict inner_dict;
            for (const auto& inner : outer.second) {
                inner_dict[py::str(inner.first)] = inner.second;
            }
            py_profile[py::str(outer.first)] = inner_dict;
        }
        // Append custom metrics (record_custom)
        //
        // NOTE: record_custom is used for both time (ms) and counters (rows/hits/etc).
        // For backward-compat with existing exporters:
        // - If the metric name ends with "_ms", export as {"total_ms","avg_ms","call_count"}.
        // - Otherwise export as {"count","avg","call_count"} so downstream code does not
        //   accidentally interpret it as milliseconds.
        auto ends_with = [](const std::string& s, const char* suffix) -> bool {
            const size_t n = s.size();
            const size_t m = std::strlen(suffix);
            if (n < m) return false;
            return std::memcmp(s.data() + (n - m), suffix, m) == 0;
        };
        for (const auto& kv : custom_metrics) {
            const auto& name = kv.first;
            const auto& values = kv.second;
            double total = 0.0;
            for (double v : values) total += v;
            double avg = values.empty() ? 0.0 : total / values.size();
            py::dict stats;
            if (ends_with(name, "_ms")) {
                stats["total_ms"] = total;
                stats["avg_ms"] = avg;
            } else {
                stats["count"] = total;
                stats["avg"] = avg;
            }
            stats["call_count"] = static_cast<int>(values.size());
            py_profile[py::str(name)] = stats;
        }

        const auto& telemetry = sampler.get_sampler_metrics();
        auto add_metric = [&](const char* key, double total, int count) {
            py::dict stats;
            stats["total_ms"] = total;
            stats["avg_ms"] = count > 0 ? total / count : 0.0;
            stats["call_count"] = count;
            py_profile[py::str(key)] = stats;
        };
        auto add_count = [&](const char* key, int count) {
            py::dict stats;
            stats["count"] = count;
            py_profile[py::str(key)] = stats;
        };
        auto add_count64 = [&](const char* key, long long count) {
            py::dict stats;
            stats["count"] = py::int_(count);
            py_profile[py::str(key)] = stats;
        };
        add_metric("telemetry_gpu_logit_pack", telemetry.gpu_logit_pack_ms, telemetry.gpu_logit_pack_calls);
        add_metric("telemetry_gpu_invoke", telemetry.gpu_invoke_ms, telemetry.gpu_invoke_calls);
        add_metric("telemetry_gpu_stage_prepare", telemetry.gpu_stage_prepare_ms, telemetry.gpu_success);
        add_metric("telemetry_gpu_stage_softmax", telemetry.gpu_stage_softmax_ms, telemetry.gpu_success);
        add_metric("telemetry_gpu_stage_sort", telemetry.gpu_stage_sort_ms, telemetry.gpu_success);
        add_metric("telemetry_gpu_stage_sample", telemetry.gpu_stage_sample_ms, telemetry.gpu_success);
        add_metric("telemetry_gpu_stage_d2h", telemetry.gpu_stage_d2h_ms, telemetry.gpu_success);
        add_metric("telemetry_gpu_stage_cpu_post", telemetry.gpu_stage_cpu_post_ms, telemetry.gpu_success);
        add_metric("telemetry_gpu_event_wait", telemetry.gpu_stage_event_wait_ms, telemetry.gpu_success);
        add_metric("telemetry_gpu_prepare_copy", telemetry.gpu_prepare_copy_ms, telemetry.gpu_success);
        add_metric("telemetry_gpu_prepare_temp", telemetry.gpu_prepare_temp_ms, telemetry.gpu_success);
        add_metric("telemetry_gpu_prepare_mask", telemetry.gpu_prepare_mask_ms, telemetry.gpu_success);
        add_metric("telemetry_gpu_prepare_rng", telemetry.gpu_prepare_rng_ms, telemetry.gpu_success);
        add_metric("telemetry_gpu_gap", telemetry.gpu_gap_ms, telemetry.gpu_success + telemetry.gpu_fail);
        add_metric("telemetry_gpu_total", telemetry.gpu_total_ms, telemetry.gpu_success + telemetry.gpu_fail);
        add_metric("telemetry_gpu_overhead", telemetry.gpu_overhead_ms, telemetry.gpu_success + telemetry.gpu_fail);
        add_metric("telemetry_gpu_overhead_get_device_logits", telemetry.gpu_overhead_get_device_logits_ms, telemetry.gpu_success + telemetry.gpu_fail);
        add_metric("telemetry_gpu_overhead_get_output_ids", telemetry.gpu_overhead_get_output_ids_ms, telemetry.gpu_success + telemetry.gpu_fail);
        add_metric("telemetry_gpu_overhead_debug_compare", telemetry.gpu_overhead_debug_compare_ms, telemetry.gpu_success + telemetry.gpu_fail);
        add_metric("telemetry_gpu_overhead_host_pre", telemetry.gpu_overhead_host_pre_ms, telemetry.gpu_success + telemetry.gpu_fail);
        add_metric("telemetry_gpu_overhead_host_after_output", telemetry.gpu_overhead_host_after_output_ms, telemetry.gpu_success + telemetry.gpu_fail);
        add_metric("telemetry_gpu_overhead_host_post", telemetry.gpu_overhead_host_post_ms, telemetry.gpu_success + telemetry.gpu_fail);
        add_metric("telemetry_cpu_sampling", telemetry.cpu_sampling_ms, telemetry.cpu_sampling_calls);
        add_metric("telemetry_cpu_loop", telemetry.cpu_loop_ms, telemetry.cpu_loop_calls);
        add_count("telemetry_gpu_success", telemetry.gpu_success);
        add_count("telemetry_gpu_fail", telemetry.gpu_fail);
        add_count("telemetry_gpu_path_device_hit", telemetry.gpu_path_device_hit);
        add_count("telemetry_gpu_path_device_miss", telemetry.gpu_path_device_miss);
        add_count("telemetry_gpu_path_need_entropy", telemetry.gpu_path_need_entropy);
        add_count("telemetry_gpu_fast_path", telemetry.gpu_fast_path);
        add_count("telemetry_gpu_device_fast_path", telemetry.gpu_device_fast_path);
        add_count("telemetry_gpu_fallback_topk", telemetry.gpu_fallback_topk);
        add_count("telemetry_gpu_fallback_topp", telemetry.gpu_fallback_topp);
        add_count("telemetry_gpu_fallback_entropy", telemetry.gpu_fallback_entropy);
        add_count("telemetry_gpu_fallback_stride", telemetry.gpu_fallback_stride);
        add_count("telemetry_gpu_fallback_stride_mismatch", telemetry.gpu_fallback_stride_mismatch);
        add_count("telemetry_gpu_fallback_partial_logits", telemetry.gpu_fallback_partial_logits);
        add_count("telemetry_gpu_fallback_compact_fail", telemetry.gpu_fallback_compact_fail);
        add_count("telemetry_gpu_fallback_device_unavail", telemetry.gpu_fallback_device_unavail);
        add_count("telemetry_gpu_sampler_unavailable", telemetry.gpu_sampler_unavailable);
        
        // Partial KV cache reuse telemetry
        add_count("telemetry_partial_kv_attempt", telemetry.partial_kv_attempt);
        add_count("telemetry_partial_kv_used", telemetry.partial_kv_used);
        add_count("telemetry_partial_kv_fallback", telemetry.partial_kv_fallback);

        // Micro-block scheduling / KV reuse effectiveness telemetry
        add_count("telemetry_denoise_step_count", telemetry.denoise_step_count);
        add_count("telemetry_active_count_samples", telemetry.active_count_samples);
        add_count("telemetry_active_count_min", telemetry.active_count_samples > 0 ? telemetry.active_count_min : 0);
        add_count("telemetry_active_count_max", telemetry.active_count_max);
        add_count64("telemetry_active_count_sum", telemetry.active_count_sum);

        add_count("telemetry_decode_count_samples", telemetry.decode_count_samples);
        add_count("telemetry_decode_count_min", telemetry.decode_count_samples > 0 ? telemetry.decode_count_min : 0);
        add_count("telemetry_decode_count_max", telemetry.decode_count_max);
        add_count64("telemetry_decode_count_sum", telemetry.decode_count_sum);
        add_count("telemetry_decode_full_steps", telemetry.decode_full_steps);
        add_count("telemetry_decode_partial_steps", telemetry.decode_partial_steps);

        add_count("telemetry_kv_rm_calls", telemetry.kv_rm_calls);
        add_count("telemetry_kv_rm_full_calls", telemetry.kv_rm_full_calls);
        add_count("telemetry_kv_rm_partial_calls", telemetry.kv_rm_partial_calls);
        add_count64("telemetry_kv_rm_tokens", telemetry.kv_rm_tokens);

        add_count("telemetry_llama_decode_calls", telemetry.llama_decode_calls);
        add_count64("telemetry_llama_decode_tokens", telemetry.llama_decode_tokens);
        
        std::vector<int> int_result(result.begin(), result.end());
        return std::make_pair(int_result, py_profile);
    }
    
    void print_last_profile_report() {
        diffusion::DiffusionProfiler::instance().print_report();
    }

private:
    llama_model* model_ = nullptr;
    int n_ctx_;
    int n_gpu_layers_;
};

PYBIND11_MODULE(llama_diffusion_profiled, m) {
    m.doc() = "Llama.cpp diffusion model with performance profiling";
    
    py::class_<LlamaDiffusionProfiledWrapper>(m, "LlamaDiffusionProfiled")
        .def(py::init<const std::string&, int, int>(),
             py::arg("model_path"),
             py::arg("n_ctx") = 8192,
             py::arg("n_gpu_layers") = 0)
        .def("generate_with_profiling", &LlamaDiffusionProfiledWrapper::generate_with_profiling,
             py::arg("prompt"),
             py::arg("mask_token_id"),
             py::arg("gen_length") = 128,
             py::arg("block_length") = 8,
             py::arg("denoising_steps") = 8,
             py::arg("temperature") = 1.0f,
             py::arg("top_k") = 0,
             py::arg("top_p") = 1.0f,
             py::arg("remasking_strategy") = "low_confidence_dynamic",
             py::arg("confidence_threshold") = 0.85f,
             py::arg("eb_threshold") = 0.35f,
             py::arg("stop_token_ids") = std::vector<int>(),
             py::arg("use_gpu_sampler") = false,
             py::arg("micro_block_size") = -1,
             "Generate with detailed performance profiling\n\n"
             "Returns:\n"
             "    tuple: (generated_tokens, profile_dict)")
        .def("print_last_profile_report", &LlamaDiffusionProfiledWrapper::print_last_profile_report,
             "Print detailed performance report to stdout");
}

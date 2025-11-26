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
        bool use_gpu_sampler = false
    ) {
        std::vector<llama_token> llama_prompt(prompt.begin(), prompt.end());
        
        diffusion::DiffusionConfig config;
        config.gen_length = gen_length;
        config.block_length = block_length;
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

        const auto& telemetry = sampler.get_sampler_metrics();
        auto add_metric = [&](const char* key, double total, int count) {
            py::dict stats;
            stats["total_ms"] = total;
            stats["avg_ms"] = count > 0 ? total / count : 0.0;
            stats["call_count"] = count;
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
        add_metric("telemetry_cpu_sampling", telemetry.cpu_sampling_ms, telemetry.cpu_sampling_calls);
        add_metric("telemetry_cpu_loop", telemetry.cpu_loop_ms, telemetry.cpu_loop_calls);
        add_metric("telemetry_gpu_success", static_cast<double>(telemetry.gpu_success), telemetry.gpu_success);
        add_metric("telemetry_gpu_fail", static_cast<double>(telemetry.gpu_fail), telemetry.gpu_fail);
        
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
             "Generate with detailed performance profiling\n\n"
             "Returns:\n"
             "    tuple: (generated_tokens, profile_dict)")
        .def("print_last_profile_report", &LlamaDiffusionProfiledWrapper::print_last_profile_report,
             "Print detailed performance report to stdout");
}

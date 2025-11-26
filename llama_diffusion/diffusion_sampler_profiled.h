// diffusion_sampler_profiled.h
#ifndef DIFFUSION_SAMPLER_PROFILED_H
#define DIFFUSION_SAMPLER_PROFILED_H

#include "diffusion_sampler.h"
#include "diffusion_profiler.h"

namespace diffusion {

class DiffusionSamplerProfiled : public DiffusionSampler {
public:
    DiffusionSamplerProfiled(llama_context* ctx, llama_model* model, const DiffusionConfig& config)
        : DiffusionSampler(ctx, model, config) {}
    
    std::vector<llama_token> generate_with_profiling(const std::vector<llama_token>& prompt) {
        PROFILE_SECTION("total_generation");
        
        DiffusionProfiler::instance().reset();
        reset_sampler_metrics();
        auto result = generate_internal_profiled(prompt);
        
        return result;
    }
    
    void print_profile_report() {
        DiffusionProfiler::instance().print_report();
    }
    
    std::unordered_map<std::string, std::unordered_map<std::string, double>> get_profile_summary() {
        return DiffusionProfiler::instance().get_summary();
    }

private:
    std::vector<llama_token> generate_internal_profiled(const std::vector<llama_token>& prompt);
    
    void denoise_block_profiled(
        std::vector<llama_token>& current_block,
        int block_idx,
        const std::vector<int>& num_transfer_tokens_per_step
    );
};

} // namespace diffusion

#endif // DIFFUSION_SAMPLER_PROFILED_H
        
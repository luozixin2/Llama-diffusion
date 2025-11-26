// diffusion_sampler_profiled.cpp
#include "diffusion_sampler_profiled.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cassert>

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
    llama_memory_t memory = llama_get_memory(ctx_);
    
    for (int step = 0; step < config_.denoising_steps; step++) {
        std::string step_section = "denoising_step_" + std::to_string(step);
        PROFILE_SECTION(step_section.c_str());
        
        // Check for masks
        {
            PROFILE_SECTION("check_remaining_masks");
            bool has_mask = false;
            for (llama_token token : current_block) {
                if (token == config_.mask_token_id) {
                    has_mask = true;
                    break;
                }
            }
            if (!has_mask) {
                DiffusionProfiler::instance().record_custom("early_exit_step", step);
                break;
            }
        }
        
        // Clear KV cache
        {
            PROFILE_SECTION("kv_cache_clear");
            llama_memory_seq_rm(memory, 0, block_start, block_start + config_.block_length);
        }
        
        // Create and decode batch
        llama_batch batch = llama_batch_init(config_.block_length, 0, 1);
        
        {
            PROFILE_SECTION("batch_preparation");
            for (int i = 0; i < config_.block_length; i++) {
                batch.token[i] = current_block[i];
                batch.pos[i] = static_cast<llama_pos>(block_start + i);
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = true;
            }
            batch.n_tokens = config_.block_length;
        }
        
        {
            PROFILE_SECTION("llama_decode");
            if (llama_decode(ctx_, batch) != 0) {
                assert(false && "llama_decode failed!");
                llama_batch_free(batch);
                return;
            }
        }
        
        // Sample tokens
        std::vector<llama_token> sampled_tokens(config_.block_length);
        std::vector<float> confidences(config_.block_length);
        std::vector<std::vector<float>> all_probs;
        
        {
            PROFILE_SECTION("token_sampling");
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
        }
        
        llama_batch_free(batch);
        
        // Determine transfer indices
        std::vector<bool> transfer_indices;
        {
            std::string strategy_name = "remasking_strategy_";
            switch (config_.remasking_strategy) {
                case RemaskingStrategy::SEQUENTIAL:
                    strategy_name += "sequential";
                    break;
                case RemaskingStrategy::LOW_CONFIDENCE_STATIC:
                    strategy_name += "low_conf_static";
                    break;
                case RemaskingStrategy::LOW_CONFIDENCE_DYNAMIC:
                    strategy_name += "low_conf_dynamic";
                    break;
                case RemaskingStrategy::ENTROPY_BOUNDED:
                    strategy_name += "entropy_bounded";
                    break;
            }
            
            PROFILE_SECTION(strategy_name.c_str());
            
            if (step < static_cast<int>(num_transfer_tokens_per_step.size())) {
                int num_transfer = num_transfer_tokens_per_step[step];
                
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
                
                // Count transferred tokens
                int num_transferred = std::count(transfer_indices.begin(), transfer_indices.end(), true);
                DiffusionProfiler::instance().record_custom("tokens_transferred_per_step", num_transferred);
            }
        }
        
        // Update block
        {
            PROFILE_SECTION("update_block_tokens");
            for (int i = 0; i < config_.block_length; i++) {
                if (i < static_cast<int>(transfer_indices.size()) && transfer_indices[i]) {
                    current_block[i] = sampled_tokens[i];
                }
            }
        }
        
        // Clear cache again
        {
            PROFILE_SECTION("kv_cache_clear_post");
            llama_memory_seq_rm(memory, 0, block_start, block_start + config_.block_length);
        }
    }
}

} // namespace diffusion

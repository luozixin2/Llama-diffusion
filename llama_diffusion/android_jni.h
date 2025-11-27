/**
 * SDAR Llama-Diffusion Android JNI Interface
 * 
 * This header provides C interface for Android JNI integration.
 * 
 * Usage in Kotlin/Java:
 *   System.loadLibrary("llama_diffusion_jni")
 *   
 *   external fun createModel(modelPath: String, nCtx: Int, nGpuLayers: Int): Long
 *   external fun destroyModel(handle: Long)
 *   external fun generate(handle: Long, prompt: IntArray, ...): IntArray
 */

#ifndef LLAMA_DIFFUSION_ANDROID_JNI_H
#define LLAMA_DIFFUSION_ANDROID_JNI_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

// Opaque handle to the diffusion model
typedef void* LlamaDiffusionHandle;

/**
 * Create a new diffusion model instance
 * 
 * @param model_path Path to the GGUF model file
 * @param n_ctx Context size (default: 4096)
 * @param n_threads Number of CPU threads (0 = auto)
 * @return Handle to the model, or NULL on failure
 */
LlamaDiffusionHandle llama_diffusion_create(
    const char* model_path,
    int n_ctx,
    int n_threads
);

/**
 * Destroy a diffusion model instance and free resources
 * 
 * @param handle Model handle returned by llama_diffusion_create
 */
void llama_diffusion_destroy(LlamaDiffusionHandle handle);

/**
 * Generation configuration
 */
typedef struct {
    int gen_length;          // Number of tokens to generate (default: 128)
    int block_length;        // Block length for diffusion (default: 4)
    int denoising_steps;     // Denoising steps per block (default: 4)
    float temperature;       // Sampling temperature (default: 1.0)
    int top_k;               // Top-k sampling (0 = disabled)
    float top_p;             // Top-p sampling (1.0 = disabled)
    int mask_token_id;       // MASK token ID from tokenizer
    const int* stop_token_ids;  // Array of stop token IDs
    int num_stop_tokens;     // Number of stop tokens
} LlamaDiffusionConfig;

/**
 * Get default configuration
 */
LlamaDiffusionConfig llama_diffusion_default_config(void);

/**
 * Generate tokens using diffusion
 * 
 * @param handle Model handle
 * @param prompt Input token IDs
 * @param prompt_len Number of prompt tokens
 * @param config Generation configuration
 * @param output_tokens Output buffer for generated tokens (must be pre-allocated)
 * @param output_capacity Size of output buffer
 * @return Number of tokens generated, or -1 on error
 */
int llama_diffusion_generate(
    LlamaDiffusionHandle handle,
    const int* prompt,
    int prompt_len,
    const LlamaDiffusionConfig* config,
    int* output_tokens,
    int output_capacity
);

/**
 * Streaming callback type
 * 
 * @param tokens Array of new token IDs
 * @param num_tokens Number of tokens in this chunk
 * @param user_data User-provided context pointer
 */
typedef void (*LlamaDiffusionStreamCallback)(
    const int* tokens,
    int num_tokens,
    void* user_data
);

/**
 * Generate tokens with streaming callback
 * 
 * @param handle Model handle
 * @param prompt Input token IDs
 * @param prompt_len Number of prompt tokens
 * @param config Generation configuration
 * @param callback Callback function for streaming tokens
 * @param user_data User data passed to callback
 * @return 0 on success, -1 on error
 */
int llama_diffusion_generate_stream(
    LlamaDiffusionHandle handle,
    const int* prompt,
    int prompt_len,
    const LlamaDiffusionConfig* config,
    LlamaDiffusionStreamCallback callback,
    void* user_data
);

/**
 * Get version string
 */
const char* llama_diffusion_version(void);

/**
 * Get last error message
 */
const char* llama_diffusion_get_error(void);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_DIFFUSION_ANDROID_JNI_H


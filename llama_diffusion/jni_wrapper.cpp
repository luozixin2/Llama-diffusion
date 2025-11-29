#include <jni.h>
#include <string>
#include <vector>
#include <android/log.h>
#include "diffusion_sampler.h"
#include "llama.h"

#define LOG_TAG "LlamaDiffusion"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// Helper function to convert jintArray to std::vector<int>
std::vector<int> jintArrayToVector(JNIEnv* env, jintArray array) {
    if (array == nullptr) {
        return std::vector<int>();
    }
    jsize length = env->GetArrayLength(array);
    jint* elements = env->GetIntArrayElements(array, nullptr);
    std::vector<int> result(elements, elements + length);
    env->ReleaseIntArrayElements(array, elements, JNI_ABORT);
    return result;
}

// Helper function to convert std::vector<int> to jintArray
jintArray vectorToJintArray(JNIEnv* env, const std::vector<int>& vec) {
    jintArray result = env->NewIntArray(vec.size());
    if (result == nullptr) {
        return nullptr;
    }
    env->SetIntArrayRegion(result, 0, vec.size(), reinterpret_cast<const jint*>(vec.data()));
    return result;
}

// Structure to hold model and context
struct LlamaDiffusionContext {
    llama_model* model;
    int n_ctx;
    int n_gpu_layers;
};

extern "C" {

// Initialize the model
JNIEXPORT jlong JNICALL
Java_com_yourpackage_LlamaDiffusion_nativeInit(
    JNIEnv* env,
    jobject /* this */,
    jstring modelPath,
    jint nCtx,
    jint nGpuLayers
) {
    const char* model_path_cstr = env->GetStringUTFChars(modelPath, nullptr);
    std::string model_path(model_path_cstr);
    env->ReleaseStringUTFChars(modelPath, model_path_cstr);
    
    LOGI("Initializing llama backend");
    llama_backend_init();
    
    LOGI("Loading model from: %s", model_path.c_str());
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = nGpuLayers;
    
    llama_model* model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        LOGE("Failed to load model from: %s", model_path.c_str());
        return 0;
    }
    
    LlamaDiffusionContext* ctx = new LlamaDiffusionContext();
    ctx->model = model;
    ctx->n_ctx = nCtx;
    ctx->n_gpu_layers = nGpuLayers;
    
    LOGI("Model loaded successfully");
    return reinterpret_cast<jlong>(ctx);
}

// Free the model
JNIEXPORT void JNICALL
Java_com_yourpackage_LlamaDiffusion_nativeFree(
    JNIEnv* /* env */,
    jobject /* this */,
    jlong handle
) {
    if (handle == 0) return;
    
    LlamaDiffusionContext* ctx = reinterpret_cast<LlamaDiffusionContext*>(handle);
    if (ctx->model) {
        llama_model_free(ctx->model);
    }
    delete ctx;
    llama_backend_free();
    LOGI("Model freed");
}

// Generate function
JNIEXPORT jintArray JNICALL
Java_com_yourpackage_LlamaDiffusion_nativeGenerate(
    JNIEnv* env,
    jobject /* this */,
    jlong handle,
    jintArray prompt,
    jint maskTokenId,
    jint genLength,
    jint blockLength,
    jint denoisingSteps,
    jfloat temperature,
    jint topK,
    jfloat topP,
    jstring remaskingStrategy,
    jfloat confidenceThreshold,
    jfloat ebThreshold,
    jintArray stopTokenIds
) {
    if (handle == 0) {
        LOGE("Invalid model handle");
        return nullptr;
    }
    
    LlamaDiffusionContext* ctx = reinterpret_cast<LlamaDiffusionContext*>(handle);
    
    // Convert prompt
    std::vector<int> prompt_vec = jintArrayToVector(env, prompt);
    std::vector<llama_token> llama_prompt(prompt_vec.begin(), prompt_vec.end());
    
    // Setup diffusion config
    diffusion::DiffusionConfig config;
    config.gen_length = genLength;
    config.block_length = blockLength;
    config.denoising_steps = denoisingSteps;
    config.temperature = temperature;
    config.top_k = topK;
    config.top_p = topP;
    config.confidence_threshold = confidenceThreshold;
    config.eb_threshold = ebThreshold;
    config.mask_token_id = maskTokenId;
    
    // Convert stop tokens
    std::vector<int> stop_tokens = jintArrayToVector(env, stopTokenIds);
    config.stop_token_ids.assign(stop_tokens.begin(), stop_tokens.end());
    
    // Parse remasking strategy
    const char* strategy_cstr = env->GetStringUTFChars(remaskingStrategy, nullptr);
    std::string strategy(strategy_cstr);
    env->ReleaseStringUTFChars(remaskingStrategy, strategy_cstr);
    
    if (strategy == "sequential") {
        config.remasking_strategy = diffusion::RemaskingStrategy::SEQUENTIAL;
    } else if (strategy == "low_confidence_static") {
        config.remasking_strategy = diffusion::RemaskingStrategy::LOW_CONFIDENCE_STATIC;
    } else if (strategy == "low_confidence_dynamic") {
        config.remasking_strategy = diffusion::RemaskingStrategy::LOW_CONFIDENCE_DYNAMIC;
    } else if (strategy == "entropy_bounded") {
        config.remasking_strategy = diffusion::RemaskingStrategy::ENTROPY_BOUNDED;
    } else {
        LOGE("Unknown remasking strategy: %s", strategy.c_str());
        return nullptr;
    }
    
    // Create context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = ctx->n_ctx;
    ctx_params.n_seq_max = 2;
    ctx_params.n_batch = 512;
    ctx_params.n_ubatch = 512;
    ctx_params.block_size = blockLength;
    
    llama_context* llama_ctx = llama_init_from_model(ctx->model, ctx_params);
    if (!llama_ctx) {
        LOGE("Failed to create llama context");
        return nullptr;
    }
    
    // Create sampler and generate
    LOGI("Starting generation");
    diffusion::DiffusionSampler sampler(llama_ctx, ctx->model, config);
    std::vector<llama_token> result = sampler.generate(llama_prompt);
    
    // Free context
    llama_free(llama_ctx);
    
    LOGI("Generation completed, tokens: %zu", result.size());
    
    // Convert result to jintArray
    std::vector<int> result_int(result.begin(), result.end());
    return vectorToJintArray(env, result_int);
}

// Stream generation with callback
JNIEXPORT void JNICALL
Java_com_yourpackage_LlamaDiffusion_nativeGenerateStream(
    JNIEnv* env,
    jobject thiz,
    jlong handle,
    jintArray prompt,
    jobject callback,
    jint maskTokenId,
    jint genLength,
    jint blockLength,
    jint denoisingSteps,
    jfloat temperature,
    jint topK,
    jfloat topP,
    jstring remaskingStrategy,
    jfloat confidenceThreshold,
    jfloat ebThreshold,
    jintArray stopTokenIds
) {
    if (handle == 0) {
        LOGE("Invalid model handle");
        return;
    }
    
    LlamaDiffusionContext* ctx = reinterpret_cast<LlamaDiffusionContext*>(handle);
    
    // Convert prompt
    std::vector<int> prompt_vec = jintArrayToVector(env, prompt);
    std::vector<llama_token> llama_prompt(prompt_vec.begin(), prompt_vec.end());
    
    // Setup diffusion config
    diffusion::DiffusionConfig config;
    config.gen_length = genLength;
    config.block_length = blockLength;
    config.denoising_steps = denoisingSteps;
    config.temperature = temperature;
    config.top_k = topK;
    config.top_p = topP;
    config.confidence_threshold = confidenceThreshold;
    config.eb_threshold = ebThreshold;
    config.mask_token_id = maskTokenId;
    
    // Convert stop tokens
    std::vector<int> stop_tokens = jintArrayToVector(env, stopTokenIds);
    config.stop_token_ids.assign(stop_tokens.begin(), stop_tokens.end());
    
    // Parse remasking strategy
    const char* strategy_cstr = env->GetStringUTFChars(remaskingStrategy, nullptr);
    std::string strategy(strategy_cstr);
    env->ReleaseStringUTFChars(remaskingStrategy, strategy_cstr);
    
    if (strategy == "sequential") {
        config.remasking_strategy = diffusion::RemaskingStrategy::SEQUENTIAL;
    } else if (strategy == "low_confidence_static") {
        config.remasking_strategy = diffusion::RemaskingStrategy::LOW_CONFIDENCE_STATIC;
    } else if (strategy == "low_confidence_dynamic") {
        config.remasking_strategy = diffusion::RemaskingStrategy::LOW_CONFIDENCE_DYNAMIC;
    } else if (strategy == "entropy_bounded") {
        config.remasking_strategy = diffusion::RemaskingStrategy::ENTROPY_BOUNDED;
    } else {
        LOGE("Unknown remasking strategy: %s", strategy.c_str());
        return;
    }
    
    // Create context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = ctx->n_ctx;
    ctx_params.n_seq_max = 2;
    ctx_params.n_batch = 512;
    ctx_params.n_ubatch = 512;
    ctx_params.block_size = blockLength;
    
    llama_context* llama_ctx = llama_init_from_model(ctx->model, ctx_params);
    if (!llama_ctx) {
        LOGE("Failed to create llama context");
        return;
    }
    
    // Get callback method
    jclass callbackClass = env->GetObjectClass(callback);
    jmethodID onTokensMethod = env->GetMethodID(callbackClass, "onTokens", "([I)V");
    
    if (onTokensMethod == nullptr) {
        LOGE("Failed to find onTokens method");
        llama_free(llama_ctx);
        return;
    }
    
    // Create global reference for callback (to use in lambda)
    jobject globalCallback = env->NewGlobalRef(callback);
    
    // Create sampler and generate with streaming
    LOGI("Starting stream generation");
    diffusion::DiffusionSampler sampler(llama_ctx, ctx->model, config);
    
    // C++ callback that calls Java callback
    auto cpp_callback = [env, globalCallback, onTokensMethod](const std::vector<int>& tokens) {
        jintArray jTokens = vectorToJintArray(env, tokens);
        env->CallVoidMethod(globalCallback, onTokensMethod, jTokens);
        env->DeleteLocalRef(jTokens);
    };
    
    sampler.generate_stream(llama_prompt, cpp_callback);
    
    // Cleanup
    env->DeleteGlobalRef(globalCallback);
    llama_free(llama_ctx);
    
    LOGI("Stream generation completed");
}

// Get vocabulary size
JNIEXPORT jint JNICALL
Java_com_yourpackage_LlamaDiffusion_nativeGetVocabSize(
    JNIEnv* /* env */,
    jobject /* this */,
    jlong handle
) {
    if (handle == 0) return 0;
    
    LlamaDiffusionContext* ctx = reinterpret_cast<LlamaDiffusionContext*>(handle);
    const llama_vocab* vocab = llama_model_get_vocab(ctx->model);
    return llama_vocab_n_tokens(vocab);
}

// Convert token to piece
JNIEXPORT jstring JNICALL
Java_com_yourpackage_LlamaDiffusion_nativeTokenToPiece(
    JNIEnv* env,
    jobject /* this */,
    jlong handle,
    jint tokenId
) {
    if (handle == 0) return env->NewStringUTF("");
    
    LlamaDiffusionContext* ctx = reinterpret_cast<LlamaDiffusionContext*>(handle);
    const llama_vocab* vocab = llama_model_get_vocab(ctx->model);
    
    char buf[256];
    int n = llama_token_to_piece(vocab, tokenId, buf, sizeof(buf), 0, false);
    if (n < 0) {
        return env->NewStringUTF("");
    }
    
    return env->NewStringUTF(std::string(buf, n).c_str());
}

} // extern "C"

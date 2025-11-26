#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "diffusion_sampler.h"
#include "llama.h"

namespace py = pybind11;

class LlamaDiffusionWrapper {
public:
    LlamaDiffusionWrapper(const std::string& model_path, int n_ctx = 32768, int n_gpu_layers = 0) 
        : n_ctx_(n_ctx), n_gpu_layers_(n_gpu_layers) {
        // Initialize llama backend
        llama_backend_init();
        
        // Load model
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = n_gpu_layers;
        
        model_ = llama_model_load_from_file(model_path.c_str(), model_params);
        if (!model_) {
            throw std::runtime_error("Failed to load model from: " + model_path);
        }
    }
    
    ~LlamaDiffusionWrapper() {
        if (model_) llama_model_free(model_);
        llama_backend_free();
    }
    
    std::vector<int> generate(
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
        // Convert prompt to llama_token
        std::vector<llama_token> llama_prompt(prompt.begin(), prompt.end());
        
        // Setup diffusion config
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
        
        // Parse remasking strategy
        if (remasking_strategy == "sequential") {
            config.remasking_strategy = diffusion::RemaskingStrategy::SEQUENTIAL;
        } else if (remasking_strategy == "low_confidence_static") {
            config.remasking_strategy = diffusion::RemaskingStrategy::LOW_CONFIDENCE_STATIC;
        } else if (remasking_strategy == "low_confidence_dynamic") {
            config.remasking_strategy = diffusion::RemaskingStrategy::LOW_CONFIDENCE_DYNAMIC;
        } else if (remasking_strategy == "entropy_bounded") {
            config.remasking_strategy = diffusion::RemaskingStrategy::ENTROPY_BOUNDED;
        } else {
            throw std::runtime_error("Unknown remasking strategy: " + remasking_strategy);
        }
        
        // Create context with block_size matching block_length
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = n_ctx_;
        ctx_params.n_seq_max = 2; // Allow multiple sequences for streaming
        ctx_params.block_size = block_length; // Set block_size from config
        
        llama_context* ctx = llama_init_from_model(model_, ctx_params);
        if (!ctx) {
            throw std::runtime_error("Failed to create context");
        }
        
        // Create sampler and generate
        diffusion::DiffusionSampler sampler(ctx, model_, config);
        std::vector<llama_token> result = sampler.generate(llama_prompt);
        
        // Free context after generation
        llama_free(ctx);
        
        // Convert back to int vector
        return std::vector<int>(result.begin(), result.end());
    }
    
    void generate_stream(
        const std::vector<int>& prompt,
        py::function callback,
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
        // Convert prompt to llama_token
        std::vector<llama_token> llama_prompt(prompt.begin(), prompt.end());
        
        // Setup diffusion config
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
        
        // Parse remasking strategy
        if (remasking_strategy == "sequential") {
            config.remasking_strategy = diffusion::RemaskingStrategy::SEQUENTIAL;
        } else if (remasking_strategy == "low_confidence_static") {
            config.remasking_strategy = diffusion::RemaskingStrategy::LOW_CONFIDENCE_STATIC;
        } else if (remasking_strategy == "low_confidence_dynamic") {
            config.remasking_strategy = diffusion::RemaskingStrategy::LOW_CONFIDENCE_DYNAMIC;
        } else if (remasking_strategy == "entropy_bounded") {
            config.remasking_strategy = diffusion::RemaskingStrategy::ENTROPY_BOUNDED;
        } else {
            throw std::runtime_error("Unknown remasking strategy: " + remasking_strategy);
        }
        
        // Create context with block_size matching block_length
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = n_ctx_;
        ctx_params.n_seq_max = 2; // Allow multiple sequences for streaming
        ctx_params.block_size = block_length; // Set block_size from config
        
        llama_context* ctx = llama_init_from_model(model_, ctx_params);
        if (!ctx) {
            throw std::runtime_error("Failed to create context");
        }
        
        // Create sampler and generate with streaming
        diffusion::DiffusionSampler sampler(ctx, model_, config);
        
        // Wrap Python callback function in a C++ std::function
        auto cpp_callback = [callback](const std::vector<int>& tokens) {
            py::gil_scoped_acquire acquire;
            callback(tokens);
        };
        
        sampler.generate_stream(llama_prompt, cpp_callback);
        
        // Free context after generation
        llama_free(ctx);
    }
    
    int get_vocab_size() const {
        const llama_vocab* vocab = llama_model_get_vocab(model_);
        return llama_vocab_n_tokens(vocab);
    }
    
    std::string token_to_piece(int token_id) const {
        const llama_vocab* vocab = llama_model_get_vocab(model_);
        char buf[256];
        int n = llama_token_to_piece(vocab, token_id, buf, sizeof(buf), 0, false);
        if (n < 0) {
            return "";
        }
        return std::string(buf, n);
    }

private:
    llama_model* model_ = nullptr;
    int n_ctx_;
    int n_gpu_layers_;
};

PYBIND11_MODULE(llama_diffusion, m) {
    m.doc() = "Llama.cpp diffusion language model bindings";
    
    py::class_<LlamaDiffusionWrapper>(m, "LlamaDiffusion")
        .def(py::init<const std::string&, int, int>(),
             py::arg("model_path"),
             py::arg("n_ctx") = 8192,
             py::arg("n_gpu_layers") = 0,
             "Initialize the diffusion model\n\n"
             "Args:\n"
             "    model_path: Path to the GGUF model file\n"
             "    n_ctx: Context length (default: 8192)\n"
             "    n_gpu_layers: Number of layers to offload to GPU (default: 0)")
        .def("generate", &LlamaDiffusionWrapper::generate,
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
             "Generate text using block diffusion\n\n"
             "Args:\n"
             "    prompt: List of token IDs for the prompt\n"
             "    mask_token_id: Token ID for the mask token\n"
             "    gen_length: Maximum length to generate\n"
             "    block_length: Length of each diffusion block\n"
             "    denoising_steps: Number of denoising iterations per block\n"
             "    temperature: Sampling temperature\n"
             "    top_k: Top-k sampling (0 to disable)\n"
             "    top_p: Top-p (nucleus) sampling\n"
             "    remasking_strategy: Strategy for selecting tokens ('sequential', 'low_confidence_static', 'low_confidence_dynamic', 'entropy_bounded')\n"
             "    confidence_threshold: Threshold for low_confidence_dynamic strategy\n"
             "    eb_threshold: Entropy budget threshold for entropy_bounded strategy\n"
             "    stop_token_ids: List of token IDs that stop generation\n"
             "    use_gpu_sampler: Enable CUDA sampling kernels when available\n\n"
             "Returns:\n"
             "    List of generated token IDs")
        .def("generate_stream", &LlamaDiffusionWrapper::generate_stream,
             py::arg("prompt"),
             py::arg("callback"),
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
             py::call_guard<py::gil_scoped_release>(),
             "Generate text using block diffusion with streaming output\n\n"
             "Args:\n"
             "    prompt: List of token IDs for the prompt\n"
             "    callback: Python function that receives list of new token IDs\n"
             "    mask_token_id: Token ID for the mask token\n"
             "    gen_length: Maximum length to generate\n"
             "    block_length: Length of each diffusion block\n"
             "    denoising_steps: Number of denoising iterations per block\n"
             "    temperature: Sampling temperature\n"
             "    top_k: Top-k sampling (0 to disable)\n"
             "    top_p: Top-p (nucleus) sampling\n"
             "    remasking_strategy: Strategy for selecting tokens\n"
             "    confidence_threshold: Threshold for low_confidence_dynamic strategy\n"
             "    eb_threshold: Entropy budget threshold for entropy_bounded strategy\n"
             "    stop_token_ids: List of token IDs that stop generation\n"
             "    use_gpu_sampler: Enable CUDA sampling kernels when available")
        .def("get_vocab_size", &LlamaDiffusionWrapper::get_vocab_size,
             "Get the vocabulary size")
        .def("token_to_piece", &LlamaDiffusionWrapper::token_to_piece,
             py::arg("token_id"),
             "Convert a token ID to its text representation");
}

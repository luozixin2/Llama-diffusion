#include <cuda_runtime.h>
#include <stdint.h>

struct llama_output_swap_device {
    uint64_t i0;
    uint64_t i1;
};

// Swap rows i0 and i1 in-place on device logits matrix [n_rows, n_vocab]
__global__ void llama_gpu_swap_rows_kernel(
    float * logits,
    uint64_t n_vocab,
    uint64_t n_rows,
    const llama_output_swap_device * swaps,
    int n_swaps
) {
    const int idx = blockIdx.x;
    if (idx >= n_swaps) return;
    const llama_output_swap_device s = swaps[idx];
    if (s.i0 >= n_rows || s.i1 >= n_rows) return;

    const uint64_t base0 = s.i0 * n_vocab;
    const uint64_t base1 = s.i1 * n_vocab;

    for (uint64_t k = threadIdx.x; k < n_vocab; k += blockDim.x) {
        const uint64_t o0 = base0 + k;
        const uint64_t o1 = base1 + k;
        float tmp = logits[o0];
        logits[o0] = logits[o1];
        logits[o1] = tmp;
    }
}

extern "C" bool llama_gpu_swap_rows(
    float * logits,
    uint64_t n_vocab,
    uint64_t n_rows,
    const void * swaps,
    int n_swaps
) {
    if (!logits || !swaps || n_swaps <= 0) {
        return true;
    }
    const int threads = 256;
    const int blocks = n_swaps;
    llama_gpu_swap_rows_kernel<<<blocks, threads>>>(
        logits,
        n_vocab,
        n_rows,
        reinterpret_cast<const llama_output_swap_device*>(swaps),
        n_swaps);
    cudaError_t err = cudaDeviceSynchronize();
    return err == cudaSuccess;
}



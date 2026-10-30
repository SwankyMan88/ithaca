// Vulkan compute runtime (AMD, Intel and NVIDIA): loaded from the driver's vulkan-1.dll at run time, no SDK needed.
// Kernels are generated as SPIR-V in C++ at startup, the same way the CUDA path generates PTX.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

class Vk {
public:
    bool init(std::string& info);                 // false when there is no usable Vulkan GPU
    uint64_t alloc(size_t bytes);                 // device-local storage buffer (returns a handle, 0 on failure)
    void free(uint64_t buf);
    bool upload(uint64_t buf, const void* data, size_t bytes);
    bool download(void* data, uint64_t buf, size_t bytes);
    // A compute pipeline: `bindings` storage buffers at set 0, `push_bytes` of push constants.
    uint64_t pipeline(const std::vector<uint32_t>& spirv, int bindings, int push_bytes);
    bool dispatch(uint64_t pipe, const std::vector<uint64_t>& bufs, const void* push, unsigned gx);   // runs and waits, or records when inside a batch
    // Everything dispatched between these runs in one submit, in order, each seeing the results of the one before.
    void begin_batch();
    bool end_batch();
    void shutdown();
    ~Vk() { shutdown(); }
private:
    struct Impl;
    Impl* p_ = nullptr;
};

// SPIR-V for the kernels
std::vector<uint32_t> vk_kernel_add();      // y[i] += a * x[i]; bindings y, x; push {uint n; float a}; local size 256
std::vector<uint32_t> vk_kernel_mv_q4();    // y[r] = sum over blocks of scale * sum((q - 8) * x); bindings S, Q, X, Y; push {uint rows, nb}; local size 64
std::vector<uint32_t> vk_kernel_mv_q8();    // the same with 32 int8 weights per block
std::vector<uint32_t> vk_kernel_rmsnorm();  // y = x / sqrt(mean(x^2) + eps) * w, per token; bindings x, w, y; push {uint n, dim; float eps}; local size 16
std::vector<uint32_t> vk_kernel_act(bool gelu);   // a[i] = act(a[i]) * b[i]; bindings a, b; push {uint n}; local size 256

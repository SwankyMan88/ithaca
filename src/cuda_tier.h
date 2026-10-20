// VRAM tier: talks to the NVIDIA driver (nvcuda.dll) directly. No CUDA toolkit needed:
// kernels are PTX text that the driver JIT-compiles for whatever GPU is installed.
#pragma once
#include <cstdint>
#include <initializer_list>
#include <mutex>
#include <string>

class CudaTier {
public:
    // GPU weight kinds: Q4 (scale, 4-bit), Q8 (scale, int8), Q4M (scale + min, 4-bit: exact for Q4_K)
    enum { K_Q4_0 = 0, K_Q8_0 = 1, K_Q4M = 2 };
    enum Op { ADD, BIAS, SILU, SILU_SPLIT, F16, COPY2D, RMSNORM, ROPE, ATTN, KVSTORE, ATTN_COMB, GELU, GATHER, SCATTER, N_OPS };
    bool init(std::string& info);   // safe to call when no NVIDIA GPU: returns false
    bool ok() const { return ok_; }
    void bind_thread();             // make context current on calling thread
    bool mem_info(size_t& free_b, size_t& total_b);
    uint64_t alloc(size_t bytes);                            // 0 on failure
    uint64_t alloc_upload(const void* host, size_t bytes);  // 0 on failure
    void free(uint64_t p);
    bool htod(uint64_t dst, const void* src, size_t bytes);
    bool dtoh(void* dst, uint64_t src, size_t bytes);       // synchronizes the compute stream
    bool htod_async(uint64_t dst, const void* src, size_t bytes);  // ordered on the compute stream
    // CUDA graphs: record a launch sequence once, replay it with one call.
    bool begin_capture();
    void* end_capture();           // returns executable graph or nullptr
    bool graph_launch(void* exec);
    void graph_destroy(void* exec);
    // Host round-trip matmul: Y[n][rows] = W[rows][cols] * X[n][cols]
    bool matmul(uint64_t W, uint64_t qoff, int kind, const float* X, int n, int cols, int rows, float* Y);
    // Device-resident matmul (async).
    bool matmul_dev(uint64_t W, uint64_t qoff, int kind, uint64_t dX, int n, int cols, int rows, uint64_t dY);
    // Async launch of an op kernel; args are pointers to the kernel parameters.
    bool launch(Op op, unsigned gx, unsigned gy, unsigned block, std::initializer_list<void*> args);
    // Elementwise helper: launches ceil(n/256) blocks and appends n as the last parameter.
    bool ew(Op op, unsigned n, std::initializer_list<void*> args);
    bool reserve_act(size_t elems);  // int8 activation scratch for n*cols elements
    // MoE decode without host round trips: Y[s][R] = W[ids[s]] * X[s*xstep] for each slot s.
    // W holds all experts stacked (R rows each); ids lives in device memory, so this is graph-capturable.
    bool matmul_experts(uint64_t W, uint64_t qoff, int kind, uint64_t ids, int slots, uint64_t dX, int xrows, int xstep, int cols, int R, uint64_t dY);
    // Router on device: softmax over E logits per token, top-k ids + weights (x scale, renormalized if norm)
    bool topk(uint64_t logits, int E, int K, bool norm, float scale, uint64_t ids, uint64_t w);
    // X[j] += sum_s w[s] * T[s*D + j]
    bool wsum(uint64_t X, uint64_t T, uint64_t w, int K, int D);
    // Generic access used by the image/video/audio engines.
    void* load_module(const std::string& ptx, std::string& log);
    void* function(void* mod, const char* name);
    bool launch_raw(void* fn, unsigned gx, unsigned gy, unsigned gz, unsigned block, unsigned shmem, void** args);
    bool dtod(uint64_t dst, uint64_t src, size_t bytes);   // async, ordered on the compute stream
    bool zero(uint64_t dst, size_t bytes);                 // async
    bool sync();
    // Fast weight load: reads a GGUF tensor straight from the file into pinned buffers, uploads on a
    // side stream and converts the raw blocks into the GPU layout with a kernel (no CPU repacking).
    // Returns the device pointer, or 0 if this quant type has no GPU converter (caller falls back).
    uint64_t load_weight(void* file, uint64_t off, size_t nbytes, int gtype, int rows, int cols, int& kind, uint64_t& qoff);
    static bool can_load(int gtype);
    static size_t gpu_bytes(int gtype, int rows, int cols);
    std::string name;
private:
    bool ok_ = false;
    void* ctx_ = nullptr;
    void* fn_[3] = {nullptr, nullptr, nullptr};
    void* fe_[3] = {nullptr, nullptr, nullptr};
    void* ftopk_ = nullptr; void* fwsum_ = nullptr;
    void* cv_[11] = {};
    void* up_ = nullptr;                 // upload stream
    void* pin_[2] = {}; void* ev_[2] = {};
    uint64_t raw_[2] = {};
    void* ops_[N_OPS] = {};
    void* st_ = nullptr;
    bool has_graphs_ = false;
    void* quant_ = nullptr;
    uint64_t xq_ = 0, xd_ = 0, xs_ = 0; size_t cap_q_ = 0;
    uint64_t dx_ = 0, dy_ = 0; size_t cap_x_ = 0, cap_y_ = 0;
    std::recursive_mutex m_;
};

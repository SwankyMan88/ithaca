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
    enum { K_Q4_0 = 0, K_Q8_0 = 1, K_Q4M = 2, K_Q6 = 3, K_Q4K = 4, N_KINDS = 5 };
    enum { K_IQ = 16 };   // K_IQ + i: IQ weights kept in their GGUF blocks (i: see iq_index), decoded inside the matmul
    // bytes of the scale area for `blocks` blocks of 32 (K_Q4K: blocks must be a multiple of 8)
    static size_t scale_bytes(int kind, size_t blocks) { return kind == K_Q4K ? blocks / 8 * 20 : blocks * (kind == K_Q4M || kind == K_Q6 ? 4 : 2); }
    enum Op { ADD, BIAS, SILU, SILU_SPLIT, F16, COPY2D, RMSNORM, ROPE, ATTN, KVSTORE, ATTN_COMB, GELU, GATHER, SCATTER, KVSTORE_Q8, OAI, XBIAS, ROWEXP, SSM_CONV, SSM_SCAN, SSM_GNORM, SCONV, DNET, SIGMUL, SSM_SCAN1, MLA, GADD, N_OPS };
    bool init(std::string& info);   // safe to call when no NVIDIA GPU: returns false
    bool ok() const { return ok_; }
    void bind_thread();             // make context current on calling thread
    bool mem_info(size_t& free_b, size_t& total_b);
    uint64_t alloc(size_t bytes);                            // 0 on failure
    uint64_t alloc_upload(const void* host, size_t bytes);  // 0 on failure
    void free(uint64_t p);
    size_t alloc_size(uint64_t p);                          // size of the allocation p starts (0 if unknown)
    void* host_alloc(size_t bytes);                         // page-locked RAM: full-speed copies to and from VRAM
    void host_free(void* p);
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
    // same_x: dX is unchanged since the previous call (its int8 copy is reused)
    bool matmul_dev(uint64_t W, uint64_t qoff, int kind, uint64_t dX, int n, int cols, int rows, uint64_t dY, bool same_x = false);
    // Async launch of an op kernel; args are pointers to the kernel parameters.
    bool launch(Op op, unsigned gx, unsigned gy, unsigned block, std::initializer_list<void*> args);
    // Elementwise helper: launches ceil(n/256) blocks and appends n as the last parameter.
    bool ew(Op op, unsigned n, std::initializer_list<void*> args);
    bool reserve_act(size_t elems);  // int8 activation scratch for n*cols elements
    // ---- fused decode steps (false: not available here, use the separate kernels) ----
    // X += res * T (T rmsnorm'd with Wp first when Wp != 0; no add when T == 0), XN = rmsnorm(X) * Wn, quantized as the
    // int8 input of the next matmul_dev(XN, ..., same_x = true). XN may be 0 (then only the int8 copy is made).
    bool norm_quant(uint64_t X, uint64_t T, uint64_t Wp, float res, uint64_t Wn, uint64_t XN, int n, int D, float eps);
    // + bias, per-head rmsnorm (q, k), RoPE (q, k), k and v into the cache (int8 when Ks != 0). head_dim <= 256.
    bool qk_post(uint64_t Q, uint64_t K, uint64_t V, uint64_t bq, uint64_t bk, uint64_t bv, uint64_t qn, uint64_t kn, uint64_t cs,
                 uint64_t Kc, uint64_t Vc, uint64_t Ks, uint64_t Vs, uint64_t pos, int n, int nh, int nkv, int hd, int half, bool neox, bool rope, float eps);
    // decode attention partials -> O, and O quantized as the next matmul's input
    bool attn_comb_q(uint64_t PM, uint64_t PL, uint64_t PO, uint64_t O, int nh, int hd, int spl, int n);
    // single token: 2-3 matrices of one kind over the same input in one launch
    bool matmul_multi(int kind, int cnt, const uint64_t* W, const uint64_t* qoff, const int* rows, const uint64_t* dY, uint64_t dX, int cols, bool same_x);
    // single token: Y = act(Wg x) * (Wu x), also quantized as the down projection's input. rows % 32 == 0.
    bool matmul_glu(int kind, uint64_t Wg, uint64_t qg, uint64_t Wu, uint64_t qu, uint64_t dX, int cols, int rows, uint64_t dY, bool gelu, bool same_x);
    // MoE decode without host round trips: Y[s][R] = W[ids[s]] * X[s*xstep] for each slot s.
    // W holds all experts stacked (R rows each); ids lives in device memory, so this is graph-capturable.
    bool matmul_experts(uint64_t W, uint64_t qoff, int kind, uint64_t ids, int slots, uint64_t dX, int xrows, int xstep, int cols, int R, uint64_t dY, bool same_x = false);
    // Router on device: softmax over E logits per token, top-k ids + weights (x scale, renormalized if norm)
    bool topk(uint64_t logits, int E, int K, bool norm, float scale, uint64_t ids, uint64_t w);
    // Prompt routing without a host round trip: top-K per token (ids/w token-major, n*K), then grouped by expert:
    // off[E+1] row ranges, sorted_ids = token of each row, sorted_w its weight (times scale).
    bool moe_route(uint64_t logits, int n, int E, int K, bool norm, float scale, uint64_t ids, uint64_t w,
                   uint64_t off, uint64_t sorted_ids, uint64_t sorted_w, uint64_t inv = 0);   // inv: row of each (token, k) pair
    // prompts: X[t] += sum_k w[t K + k] * T[inv[t K + k]] in k order (deterministic, unlike atomic adds)
    bool moe_combine(uint64_t X, uint64_t T, uint64_t inv, uint64_t w, int n, int K, int D);
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
    // IQ weights (kind >= K_IQ) decoded to the K_Q8_0 layout: f16 scales at S, int8 values at Q
    bool iq_to_q8(uint64_t W, int kind, int cols, int rows, uint64_t S, uint64_t Q);
    static size_t gpu_bytes(int gtype, int rows, int cols);
    std::string name;
private:
    bool ok_ = false;
    void* ctx_ = nullptr;
    void* fn_[N_KINDS] = {};
    void* fn1_[N_KINDS] = {};   // n == 1 variants: 4 rows per warp
    void* fns_[N_KINDS] = {};   // n == 1, 1 row per warp (small matrices)
    uint64_t qx_src_ = 0; int qx_n_ = 0, qx_cols_ = 0;   // input of the last activation quantization
    void* fe_[N_KINDS] = {};
    static const int N_IQ = 12;   // IQ types plus Q2_K, Q3_K (see iq_index)
    void* fiq_[N_IQ][3] = {};      // per type: 4 rows x 1 token, 1 row x 1 token, 2 rows x 8 tokens
    void* fiqq8_[N_IQ] = {};
    void* fiqx_[N_IQ] = {};        // mixture-of-experts form (decode)       // the matrix -> the Q8_0 layout, for prompts on the int8 tensor cores
    uint64_t iqtab_ = 0;     // their lookup grids
    void* ftopk_ = nullptr; void* fwsum_ = nullptr; void* fsort_ = nullptr;
    void* cv_[11] = {};
    void* up_ = nullptr;                 // upload stream
    static const int kSlots = 6;           // read-ahead depth of the weight loader
    void* pin_[kSlots] = {}; void* ev_[kSlots] = {};
    uint64_t raw_[kSlots] = {};
    void* ops_[N_OPS] = {};
    void* st_ = nullptr;
    bool has_graphs_ = false;
    void* quant_ = nullptr;
    uint64_t xq_ = 0, xd_ = 0, xs_ = 0, xq2_ = 0, xd2_ = 0, xs2_ = 0; size_t cap_q_ = 0;
    void* fmm_[N_KINDS][2] = {}; void* fglu_[N_KINDS] = {}; void* fmt_[N_KINDS][2][2] = {}; void* fnormq_[3] = {}; void* fqkpost_ = nullptr; void* fcombq_ = nullptr; void* fmcomb_ = nullptr;
    bool quantize_input(uint64_t dX, int n, int cols, bool same_x);
    uint64_t dx_ = 0, dy_ = 0; size_t cap_x_ = 0, cap_y_ = 0;
    std::recursive_mutex m_;
};

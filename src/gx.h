// GPU runtime for the media engines: safetensors files, a VRAM pool, tiered weights and tensor ops.
#pragma once
#include "common.h"
#include <map>
#include <unordered_map>

// ---- safetensors (memory-mapped, several files merged into one name space) ----
struct STensor {
    std::string dtype;            // F16, BF16, F32, F8_E4M3 ...
    std::vector<int64_t> shape;
    const uint8_t* data = nullptr;
    size_t nbytes = 0;
    int64_t numel() const { int64_t n = 1; for (auto d : shape) n *= d; return n; }
};
class SafeTensors {
public:
    ~SafeTensors() { close(); }
    bool add(const std::wstring& path, std::string& err, const std::string& strip_prefix = "");
    void close();
    const STensor* get(const std::string& n) const { auto it = t_.find(n); return it == t_.end() ? nullptr : &it->second; }
    bool has(const std::string& n) const { return t_.count(n) != 0; }
    void put(const std::string& n, const STensor& t) { t_[n] = t; }   // tensors that live elsewhere (e.g. a GGUF mapping)
    const std::map<std::string, STensor>& all() const { return t_; }
private:
    struct Map { HANDLE f, m; const uint8_t* base; size_t size; };
    std::vector<Map> maps_;
    std::map<std::string, STensor> t_;
};

// f32 device tensor
struct GT {
    uint64_t p = 0;
    size_t n = 0;
    explicit operator bool() const { return p != 0; }
};

class Gx {
public:
    bool init(std::string& err);
    bool ok() const { return ok_; }
    bool tensor_cores() const { return tc_; }
    // fp16 tensor-core math (2x TF32 throughput). Only for networks that stay inside fp16 range.
    void set_half(bool on) { half_ = on && f_linh_ && !getenv("ITHACA_NO_HALF"); }

    // memory pool for activations (f32)
    GT alloc(size_t n);
    void free(GT& t);
    void trim();                 // give cached blocks back to the driver
    size_t vram_free();

    // ---- weights: fp16 in VRAM; whatever doesn't fit the budget is streamed per use ----
    // q8: store big weight matrices as int8 + per-row scale (half the VRAM of fp16, same tensor-core speed)
    void weights_begin(const SafeTensors* st, int64_t vram_budget, bool q8 = false);
    uint64_t w16(const std::string& name);       // device pointer to fp16 weight
    GT wf32(const std::string& name);            // small fp32 params (norms, biases), always resident
    void weights_end();                          // free everything from this component
    int64_t resident_bytes() const { return res_bytes_; }
    int64_t streamed_bytes() const { return streamed_; }
    const SafeTensors* current() const { return st_; }   // whose weights are on the GPU now

    // ---- ops (all async on the compute stream) ----
    // y[M,N] = x[M,K] W^T + b   (W fp16 [N,K]); beta=1 adds into existing y
    void linear(const GT& x, int M, int K, uint64_t W, int N, const GT* bias, GT& y, float beta = 0.f);
    // conv2d NCHW, batch 1. W fp16 [Cout][Cin*kh*kw]
    void conv(const GT& x, int Cin, int H, int W, uint64_t Wt, int Cout, int kh, int kw, int stride, int pad, const GT* bias, GT& y, float beta = 0.f, int padw = -1);
    // Chat-engine prompt processing on tensor cores: y[M,N] = x[M,K] W^T with W in the LLM's GPU layout
    // (kind 0 = 4-bit + scale, 1 = int8 + scale, 2 = 4-bit + scale/min pairs). False if unsupported.
    bool linear_quant(uint64_t x, int M, int K, uint64_t S, uint64_t Q, int kind, int N, uint64_t y);
    // Chat-engine prompt attention from the fp16 KV cache (causal, grouped heads, optional window / non-causal block).
    bool attn_prefill(uint64_t Q, uint64_t Kc, uint64_t Vc, uint64_t O, int n, int p0, int ldq, int kvd, int heads, int grp, int hd,
                      float scale, int win, int nc);
    void gemm_nt(const GT& A, int M, int K, int lda, const GT& B, int N, int ldb, GT& C, int ldc, float alpha,
                 int batch, size_t sA, size_t sB, size_t sC, size_t offA = 0, size_t offB = 0, size_t offC = 0);
    void gemm_nn(const GT& A, int M, int K, int lda, const GT& B, int N, int ldb, GT& C, int ldc, float alpha,
                 int batch, size_t sA, size_t sB, size_t sC, size_t offA = 0, size_t offB = 0, size_t offC = 0);
    void groupnorm(const GT& x, int C, int HW, int groups, const GT& g, const GT& b, float eps, bool silu, GT& y);
    void layernorm(const GT& x, int rows, int D, const GT& g, const GT& b, float eps, GT& y);
    void softmax(GT& x, int rows, int cols, float scale, bool causal = false, int T = 1);
    enum Act { SILU = 0, GELU = 1, QUICK_GELU = 2, GELU_TANH = 3, MISH = 4 };
    void act(GT& x, size_t n, Act a);
    void axpy(GT& y, const GT& x, size_t n, float a = 1.f);
    void chanadd(GT& y, const GT& b, int C, int HW);
    void geglu(const GT& in, int T, int F, GT& out);
    void transpose(const GT& in, int R, int C, GT& out);   // in [R][C] -> out [C][R]
    void up2(const GT& in, int C, int H, int W, GT& out);
    void embed(uint64_t ids_u32, uint64_t table_f16, int T, int D, GT& out);
    void copy(GT& dst, const GT& src, size_t n, size_t dst_off = 0, size_t src_off = 0);
    // multi-head attention on token-major tensors. q [T][H*d], k/v [S][H*d] -> out [T][H*d]
    void attention(const GT& q, const GT& k, const GT& v, int T, int S, int heads, int d, GT& out, bool causal = false);

    // video engine ops
    void rmsrow(const GT& x, int rows, int D, const GT& w, float eps, GT& y);
    void rmsc(const GT& x, int C, size_t N, const GT& g, bool silu, GT& y);          // channel-first L2 norm * sqrt(C) * g
    void gadd(GT& x, const GT& y, const GT& gate, int C, size_t n);                 // x += y * gate[col]
    void mul(GT& y, const GT& x, size_t n);                                         // y *= x
    void rope3(GT& x, const GT& cs, int L, int heads, int hd);
    // causal-style 3D conv, stride 1, spatial pad = k/2: x [Ci][Tin][H][W] -> y [Co][Tout][H][W], Tout = Tin - KT + 1
    void conv3d(const GT& x, int Ci, int Tin, int H, int W, uint64_t Wt, int Co, int KT, int KS, const GT* bias, GT& y, float beta = 0.f);
    void dupup(const GT& x, int Cin, int Cout, int T, int H, int W, int ft, int fs, int drop, GT& out);
    void tinter(const GT& x, int C, int T, int HW, GT& out);

    void upload(GT& t, const float* h, size_t n);
    void download(float* h, const GT& t, size_t n);
    void sync();
    void print_profile();
private:
    void launch_ew(void* fn, size_t n, std::vector<void*> args);
    void gemm(void* fn, uint64_t A, uint64_t B, uint64_t C, uint64_t bias, unsigned M, unsigned N, unsigned K, unsigned lda, unsigned ldb, unsigned ldc,
              unsigned sA, unsigned sB, unsigned sC, float alpha, float beta, unsigned flags, unsigned batch,
              unsigned H = 0, unsigned W = 0, unsigned kh = 1, unsigned kw = 1, unsigned stride = 1, unsigned pad = 0, unsigned Wo = 1, unsigned padw = 0);
    bool ok_ = false, tc_ = false;
    void* f_linf_ = nullptr;
    void *f_im2col_ = nullptr, *f_tradd_ = nullptr;
    void *f_rmsrow_ = nullptr, *f_rmsc_ = nullptr, *f_gadd_ = nullptr, *f_rope3_ = nullptr, *f_im2col3_ = nullptr, *f_dupup_ = nullptr, *f_tinter_ = nullptr, *f_f8_ = nullptr, *f_mul_ = nullptr;
    void *f_lin_, *f_nt_, *f_nn_, *f_conv_, *f_gn_, *f_ln_, *f_sm_, *f_act_, *f_axpy_, *f_chan_, *f_geglu_, *f_tr_, *f_up2_, *f_embed_, *f_h2f_;
    std::multimap<size_t, uint64_t> pool_;
    std::unordered_map<uint64_t, size_t> live_;

    const SafeTensors* st_ = nullptr;
    int64_t budget_ = 0, res_bytes_ = 0, streamed_ = 0;
    std::unordered_map<std::string, uint64_t> res_;      // resident fp16 weights
    std::unordered_map<std::string, GT> f32_;
    uint64_t ring_ = 0; size_t ring_cap_ = 0, ring_pos_ = 0;
    std::vector<uint16_t> conv_buf_;
    uint64_t w16_f8(const std::string& name, const STensor& t);
    uint64_t f8_buf_ = 0; size_t f8_cap_ = 0;
    bool q8_ = false;
    std::unordered_map<uint64_t, uint64_t> q8scale_;   // int8 weight ptr -> f32 row scales
    uint64_t qtmp_ = 0; size_t qtmp_cap_ = 0;
    void *f_linq8_ = nullptr, *f_q8row_ = nullptr, *f_linh_ = nullptr, *f_linq8h_ = nullptr, *f_fa64_ = nullptr, *f_fa128_ = nullptr, *f_fa72_ = nullptr, *f_linq4_ = nullptr, *f_linq4m_ = nullptr, *f_fkv_[5] = {};
    bool half_ = false;
    uint64_t place(size_t bytes, bool& resident);
};

Gx& gx();

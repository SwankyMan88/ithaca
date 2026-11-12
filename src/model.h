#pragma once
#include "cuda_tier.h"
#include "gguf.h"
#include "quant.h"
#include "tokenizer.h"
#include <array>
#include <memory>

// Where a layer's weights live right now. Planned at load, migrated in the background.
enum Tier { TIER_DISK = 0, TIER_RAM = 1, TIER_GPU = 2 };

struct LoadOpts {
    int threads = 0;          // 0 = auto
    int ctx = 4096;
    int64_t ram_mb = -1;      // -1 = auto
    int64_t vram_mb = -1;     // -1 = auto, 0 = disable GPU
};

struct Lin {
    const GTensor* t = nullptr;
    int rows = 0, cols = 0;
    std::atomic<uint64_t> dptr{0};
    int gkind = 0;
    uint64_t gqoff = 0;
};

struct Layer {
    std::vector<float> attn_norm, ffn_norm, q_norm, k_norm, bq, bk, bv, post_attn_norm, post_ffn_norm;
    std::vector<float> bo, sinks, b_inp, b_gate, b_up, b_down;   // gpt-oss: output and expert biases, attention sinks
    Lin q, k, v, o, qkv, gate, up, gateup, down;
    Lin gate_inp, gate_exps, up_exps, down_exps;   // mixture of experts: router + stacked experts
    bool swa = false;                              // sliding-window attention layer (Gemma 3)
    bool nope = false;                             // no RoPE in this layer
    // Mamba-2 layers (Granite 4.0 H): projections, conv / scan parameters, and the recurrent state (host copy)
    Lin ssm_in, ssm_out;   // also LFM2 short-conv in / out projections (sconv)
    bool sconv = false;
    Lin dn_z, dn_a, dn_b;   // Qwen3.5 Gated DeltaNet: output gate, decay (alpha) and write strength (beta) projections
    bool dnet = false, qgate = false;   // qgate: the q projection carries a sigmoid output gate per head
    std::vector<float> conv_w, conv_b, ssm_a, ssm_d, ssm_dtb, ssm_norm, conv_st, ssm_st;
    // DeepSeek V2 multi-head latent attention: KV down-projection (latent + shared RoPE key), its up-projection,
    // and the optional low-rank q pair. MoE shared experts (Qwen2-MoE, DeepSeek) live in gate / up / down.
    Lin qa, qb, kva, kvb;
    std::vector<float> qa_norm, kva_norm, shg;   // shg: Qwen2-MoE's sigmoid gate on the shared expert
    bool shexp = false;
    std::array<Lin*, 22> lins() { return {&q, &k, &v, &o, &qkv, &gate, &up, &gateup, &down, &gate_inp, &gate_exps, &up_exps, &down_exps, &ssm_in, &ssm_out, &dn_z, &dn_a, &dn_b, &qa, &qb, &kva, &kvb}; }
    const uint8_t* lo = nullptr; const uint8_t* hi = nullptr;
    std::atomic<int> plan{TIER_DISK}, state{TIER_DISK};
    // VRAM-resident copies used when the whole layer runs on the GPU
    uint64_t d_an = 0, d_fn = 0, d_qn = 0, d_kn = 0, d_bq = 0, d_bk = 0, d_bv = 0, d_kc = 0, d_vc = 0, d_pan = 0, d_pfn = 0;
    uint64_t d_ks = 0, d_vs = 0;   // int8 KV cache: f32 scale per (position, kv head)
    uint64_t d_qan = 0, d_kvan = 0, d_shg = 0;
    uint64_t d_bo = 0, d_sinks = 0, d_binp = 0, d_bgate = 0, d_bup = 0, d_bdown = 0;   // gpt-oss
    uint64_t d_convw = 0, d_convb = 0, d_ssma = 0, d_ssmd = 0, d_dtb = 0, d_ssmnorm = 0, d_convst = 0, d_ssmst = 0;   // Mamba-2 (state: in VRAM while the layer is)
    int kv_dev = 0;  // device KV cache valid for positions [0, kv_dev)
    int kv_cap = 0;  // positions the device KV buffers can hold (grows with the conversation)
    size_t bytes() const { return hi > lo ? (size_t)(hi - lo) : 0; }
};

struct HParams {
    std::string arch, name;
    int n_embd = 0, n_layer = 0, n_head = 0, n_kv = 0, hd = 0, n_ff = 0, n_vocab = 0, n_rot = 0;
    int ctx_train = 0;
    float eps = 1e-5f, rope_base = 10000.f;
    bool neox = false, gelu = false;
    float emb_scale = 1.f, res_scale = 1.f, attn_scale = 0.f, logit_scale = 1.f;
    int n_expert = 0, n_expert_used = 0, n_ff_exp = 0; bool expert_norm = true;
    int swa_window = 0; float rope_base_swa = 10000.f;
    bool oai = false;
    bool nope = false;   // attention without positions (Granite 4.0 H)
    int nope_step = 0;   // SmolLM3: every nope_step-th layer has no RoPE
    bool no_prenorm = false;   // OLMo 2: only post-norms
    float attn_softcap = 0, final_softcap = 0;   // Gemma 2: cap * tanh(x / cap) on attention scores / output logits
    int mla_rank = 0, mla_nope = 0, mla_v = 0, mla_qlora = 0;   // DeepSeek V2: latent size, non-RoPE key part, value size, q rank
    float exp_scale = 1.f;   // routed expert weights are multiplied by this
    int ssm_inner = 0, ssm_state = 0, ssm_groups = 1, ssm_heads = 0, ssm_conv = 0;   // Mamba-2   // gpt-oss: clamped SwiGLU (gate * sigmoid(1.702 gate) * (up + 1)), biases, attention sinks
};

class Model {
public:
    ~Model() { unload(); }
    bool load(const std::wstring& path, const LoadOpts& o, std::string& err);
    void unload();
    bool loaded() const { return loaded_; }

    // Feed tokens after the current n_past; returns logits for the last one.
    // A negative id -1-k means "row k of ext_embd" (image embeddings); a run of them attends bidirectionally.
    const float* eval(const int* toks, int n);
    bool recurrent() const { return recurrent_; }   // state layers: rewinding restarts from zero
    // Same, but returns logits for every one of the n positions (n * n_vocab floats). Used to verify drafts.
    const float* eval_all(const int* toks, int n) { all_logits_ = true; const float* r = eval(toks, n); all_logits_ = false; return r; }
    std::vector<float> ext_embd;
    void rewind(int n);
    int n_past = 0;
    int ctx = 0;

    HParams hp;
    Tokenizer tok;
    std::string chat_template;
    std::string path_utf8;
    std::string tier_json();       // live tier map for UI
    std::string gpu_info;
    int threads() const { return pool_.size(); }
    bool migrating() const { return migrating_; }
    int64_t gpu_bytes() const { return gpu_bytes_; }
    bool fully_on_gpu() const { return dev_ok_ && all_gpu(); }
    // Parking: the weights move to page-locked RAM and their VRAM is freed; wake() copies them straight back
    // (no file reads or format conversion). eval() wakes a parked model by itself. Only for models fully in VRAM.
    bool park();
    bool wake();
    bool parked() const { return !parked_.empty(); }
    int64_t parked_bytes() const { return parked_bytes_; }
    // Called when growing the KV cache runs out of VRAM; returns true if it freed something (e.g. a draft model).
    std::function<bool()> on_vram_short;
    int64_t kv_bytes_gpu() const;
    void bench_gpu();

    static CudaTier& cuda();
private:
    void matmul(Lin& w, const float* X, int n, float* Y);
    void rmsnorm(const float* x, const float* w, float* y, int n, int dim);
    void rope(float* x, int nheads, int pos, bool swa = false);
    void matmul_raw(int type, const uint8_t* base, int rows, int cols, const float* X, int n, float* Y);
    void moe_cpu(Layer& L, int n);
    void moe_gpu(Layer& L, int n);
    void route(const float* logits, int n, std::vector<std::vector<std::pair<int, float>>>& per_expert);
    void ensure_buffers(int n);
    bool alloc_dev_buffers();
    void gpu_layer(Layer& L, int n, unsigned nc = 0);
    void mamba_cpu(Layer& L, int n);   // xn_ -> tmp_
    void sconv_cpu(Layer& L, int n);
    void dnet_cpu(Layer& L, int n);   // Qwen3.5 Gated DeltaNet: xn_ -> tmp_
    void dnet_gpu(Layer& L, int n, bool same_x);   // LFM2 short convolution: xn_ -> tmp_
    void post_logits(float* v, size_t n) const;   // logit scale and final soft-cap
    bool recurrent_ = false;   // has state layers: the cache can only grow, a rewind restarts from zero
    Layer* xn_ready_ = nullptr; int xn_n_ = 0, xn_past_ = -1;   // prompts: this layer's input norm + int8 input were made by the layer before
    bool ensure_kv(int need);
    void* f_gqa_ = nullptr;
    bool kvq8_ = false;      // VRAM KV cache stored as int8 + scales (half the memory of fp16)
    size_t kv_pos_bytes() const;   // device KV bytes per position per layer   // decode attention generated for this model's head layout
    std::mutex kv_m_;
    int kv_want_ = 0;   // largest KV size asked for (ensure_kv): layers reaching VRAM mid-request get at least this
    void mmq(uint64_t W, uint64_t qoff, int kind, uint64_t x, int n, int cols, int rows, uint64_t y, bool same_x = false);
    bool moe_grouped(Layer& L, int n);
    std::atomic<bool> tc_prefill_{false};
    uint64_t dev_vec(const std::vector<float>& v);
    void tier_thread();
    void prefetch_thread();
    bool bind(Lin& l, const std::string& name, std::string& err, bool required = true);
    std::vector<float> vec(const std::string& name);

    bool loaded_ = false;
    uint64_t dW16_ = 0; size_t w16_cap_ = 0; int64_t w16_plan_ = 0;   // Q8_0 copy of one IQ matrix while a prompt runs through it
    struct Parked { Lin* l; void* host; size_t bytes; };
    std::vector<Parked> parked_;
    int64_t parked_bytes_ = 0;
    GGUF g_;
    Pool pool_;
    std::vector<Layer> layers_;
    std::vector<float> out_norm_;
    const GTensor* embd_ = nullptr;
    Lin out_;
    std::atomic<int> out_state_{TIER_DISK};
    int out_plan_ = TIER_DISK;
    std::vector<float> inv_freq_, inv_freq_swa_;
    bool moe_ = false, all_logits_ = false;
    uint64_t dLOGA_ = 0; int loga_rows_ = 0;
    std::vector<float> moe_x_, router_;
    std::vector<uint16_t> kc_, vc_;

    int bufn_ = 0;
    std::vector<float> x_, xn_, q_, k_, v_, att_, ff1_, ff2_, tmp_, logits_, qg_;
    std::vector<BQ8> qbuf_;
    std::vector<std::vector<float>> scratch_;

    std::thread tier_th_, pre_th_;
    std::atomic<bool> stop_{false}, migrating_{false};
    std::mutex pre_m_; std::condition_variable pre_cv_; int pre_want_ = -1;
    std::atomic<int64_t> gpu_bytes_{0}, ram_bytes_{0};
    double load_sec_ = 0;
    // device activations (whole-layer GPU path)
    std::atomic<bool> dev_ok_{false};
    int gpu_batch_ = 256, sc_rows_ = 256;   // sc_rows_: token rows of the attention score buffer
    uint64_t dCS2_ = 0, dR_ = 0, dG_ = 0, dIdx_ = 0, dWt_ = 0;
    uint64_t dML1_ = 0, dML2_ = 0, dML3_ = 0;   // MLA: raw q (later the compacted output), KV down-projection, normed latent
    std::vector<float> ml1_, ml2_, ml3_, ml4_;
    void mla_gpu(Layer& L, int n, bool same_x);
    void mla_cpu(Layer& L, int n);
    uint64_t dZX_ = 0, dXC_ = 0, dSY_ = 0;   // Mamba-2 prompts: in-projection, conv output, scan output
    void mamba_gpu(Layer& L, int n, bool same_x);
    bool ssm_gpu_ok() const { return hp.ssm_conv == 4 && hp.ssm_state >= 32 && 256 % hp.ssm_state == 0 && hp.ssm_heads && hp.ssm_inner % (256 / hp.ssm_state) == 0; }
    uint64_t dMG_ = 0, dMF1_ = 0, dMF2_ = 0, dMT_ = 0, dOff_ = 0, dSI_ = 0, dSW_ = 0;   // MoE prompts: all routed rows, grouped by expert
    uint64_t dPM_ = 0, dPL_ = 0, dPO_ = 0, dPOS_ = 0, dX_ = 0, dXN_ = 0, dQ_ = 0, dK_ = 0, dV_ = 0, dATT_ = 0, dF1_ = 0, dF2_ = 0, dT_ = 0, dLOG_ = 0, dSC_ = 0, dCS_ = 0, d_onorm_ = 0;
    std::vector<uint64_t> dev_allocs_;
    std::vector<float> cs_, cs2_;
    void* graph_ = nullptr;
    std::unordered_map<int, void*> seg_graphs_;
    bool all_gpu() const;
    void fill_cs(int n);
};

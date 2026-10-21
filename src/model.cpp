#include "model.h"
#include "gx.h"
#include <algorithm>
#include <cmath>
#include <cstdio>

CudaTier& Model::cuda() {
    static CudaTier c;
    static std::string info;
    static bool inited = (c.init(info) || (fprintf(stderr, "[gpu] %s\n", info.c_str()), true), true);
    (void)inited;
    return c;
}

static std::string gpu_init_info() {
    Model::cuda();
    return Model::cuda().ok() ? "GPU: " + Model::cuda().name : "CPU only (no usable NVIDIA GPU)";
}

std::vector<float> Model::vec(const std::string& name) {
    const GTensor* t = g_.get(name);
    if (!t) return {};
    int64_t n = t->ne[0] * t->ne[1];
    std::vector<float> v(n);
    dequant_row(t->type, t->data, v.data(), n);
    return v;
}

bool Model::bind(Lin& l, const std::string& name, std::string& err, bool required) {
    const GTensor* t = g_.get(name);
    if (!t) { if (required) err = "missing tensor " + name; return !required; }
    if (!gtype_supported(t->type)) { err = std::string("unsupported quant type ") + gtype_name(t->type) + " in " + name; return false; }
    l.t = t; l.cols = (int)t->ne[0]; l.rows = (int)(t->ne[1] * t->ne[2]);   // 3-D expert stacks: rows of all experts
    return true;
}

static size_t gpu_cost(const Lin& l) {
    if (!l.t) return 0;
    if (CudaTier::can_load(l.t->type)) return CudaTier::gpu_bytes(l.t->type, l.rows, l.cols);
    return (size_t)l.rows * (l.cols / 32) * 36 + 256;  // CPU-repacked Q8 layout
}

bool Model::load(const std::wstring& path, const LoadOpts& o, std::string& err) {
    unload();
    double t0 = now_sec();
    if (!g_.open(path, err)) return false;
    path_utf8 = w_to_utf8(path);
    hp = HParams();
    moe_ = false;
    hp.arch = g_.gets("general.architecture");
    hp.name = g_.gets("general.name", "model");
    const std::string& a = hp.arch;
    static const char* ok_arch[] = {"llama", "qwen2", "qwen2vl", "qwen3", "phi3", "granite", "mistral", "gemma3", "granitemoe", "qwen3moe"};
    if (std::find_if(std::begin(ok_arch), std::end(ok_arch), [&](const char* s) { return a == s; }) == std::end(ok_arch)) {
        err = "architecture '" + a + "' is not supported yet (supported: llama, mistral, mixtral, qwen2, qwen2.5-vl text, qwen3, qwen3-moe, phi3/phi4, granite, granite-moe, gemma3)";
        g_.close(); return false;
    }
    std::string P = a + ".";
    hp.n_embd = (int)g_.geti(P + "embedding_length", 0);
    hp.n_layer = (int)g_.geti(P + "block_count", 0);
    hp.n_head = (int)g_.geti(P + "attention.head_count", 0);
    {
        const GValue* kvv = g_.find(P + "attention.head_count_kv");
        hp.n_kv = kvv ? (kvv->arr_n.empty() ? (int)kvv->i : (int)kvv->arr_n[0]) : hp.n_head;
    }
    hp.n_ff = (int)g_.geti(P + "feed_forward_length", 0);
    hp.eps = (float)g_.getf(P + "attention.layer_norm_rms_epsilon", 1e-5);
    hp.rope_base = (float)g_.getf(P + "rope.freq_base", 10000.0);
    hp.hd = (int)g_.geti(P + "attention.key_length", hp.n_head ? hp.n_embd / hp.n_head : 0);
    hp.n_rot = (int)g_.geti(P + "rope.dimension_count", hp.hd);
    hp.ctx_train = (int)g_.geti(P + "context_length", 4096);
    hp.neox = a == "qwen2" || a == "qwen2vl" || a == "qwen3" || a == "phi3" || a == "qwen3moe" || a == "gemma3";
    hp.n_expert = (int)g_.geti(P + "expert_count", 0);
    hp.n_expert_used = (int)g_.geti(P + "expert_used_count", 0);
    { const GValue* en = g_.find(P + "expert_weights_norm"); hp.expert_norm = en ? en->i != 0 : true; }
    if (a == "gemma3") {
        // Gemma 3: scaled embeddings, GELU, sandwich norms, 5 sliding-window layers per global one
        hp.gelu = true; hp.emb_scale = std::sqrt((float)hp.n_embd);
        hp.swa_window = (int)g_.geti(P + "attention.sliding_window", 0); if (getenv("ITHACA_NO_SWA")) hp.swa_window = 0;
        hp.rope_base_swa = (float)g_.getf(P + "rope.local.freq_base", 10000.0);
        if (hp.n_layer == 62) hp.attn_scale = 1.f / std::sqrt((float)hp.n_embd / hp.n_head);   // 27B uses query_pre_attn_scalar
    }
    if (a == "granite" || a == "granitemoe") {
        hp.emb_scale = (float)g_.getf(P + "embedding_scale", 1.0);
        hp.res_scale = (float)g_.getf(P + "residual_scale", 1.0);
        hp.attn_scale = (float)g_.getf(P + "attention.scale", 0.0);
        hp.logit_scale = (float)g_.getf(P + "logit_scale", 1.0);
    }
    if (!hp.n_embd || !hp.n_layer || !hp.n_head || !hp.hd) { err = "incomplete model metadata"; g_.close(); return false; }
    chat_template = g_.gets("tokenizer.chat_template");
    if (!tok.load(g_, err)) { g_.close(); return false; }

    embd_ = g_.get("token_embd.weight");
    if (!embd_ || !gtype_supported(embd_->type)) { err = "missing/unsupported token_embd.weight"; g_.close(); return false; }
    hp.n_vocab = (int)embd_->ne[1];
    if (!bind(out_, "output.weight", err, false) || (!out_.t && !bind(out_, "token_embd.weight", err))) { g_.close(); return false; }
    out_norm_ = vec("output_norm.weight");

    layers_ = std::vector<Layer>(hp.n_layer);
    for (int i = 0; i < hp.n_layer; i++) {
        Layer& L = layers_[i];
        std::string b = "blk." + std::to_string(i) + ".";
        L.attn_norm = vec(b + "attn_norm.weight");
        L.ffn_norm = vec(b + "ffn_norm.weight");
        L.q_norm = vec(b + "attn_q_norm.weight");
        L.k_norm = vec(b + "attn_k_norm.weight");
        L.bq = vec(b + "attn_q.bias"); L.bk = vec(b + "attn_k.bias"); L.bv = vec(b + "attn_v.bias");
        L.post_attn_norm = vec(b + "post_attention_norm.weight"); L.post_ffn_norm = vec(b + "post_ffw_norm.weight");
        L.swa = hp.swa_window > 0 && (i + 1) % 6 != 0;
        bool ok = true;
        if (g_.get(b + "attn_qkv.weight")) ok &= bind(L.qkv, b + "attn_qkv.weight", err);
        else ok &= bind(L.q, b + "attn_q.weight", err) && bind(L.k, b + "attn_k.weight", err) && bind(L.v, b + "attn_v.weight", err);
        ok = ok && bind(L.o, b + "attn_output.weight", err);
        if (ok && g_.get(b + "ffn_gate_exps.weight")) {
            if (hp.n_expert < 1 || hp.n_expert_used < 1) { err = "MoE model without expert counts"; ok = false; }
            ok = ok && bind(L.gate_inp, b + "ffn_gate_inp.weight", err) && bind(L.gate_exps, b + "ffn_gate_exps.weight", err) &&
                 bind(L.up_exps, b + "ffn_up_exps.weight", err) && bind(L.down_exps, b + "ffn_down_exps.weight", err);
            if (ok && g_.get(b + "ffn_gate_inp_shexp.weight")) { err = "MoE models with shared experts (Qwen2-MoE) aren't supported yet"; ok = false; }
        } else {
            ok = ok && bind(L.down, b + "ffn_down.weight", err);
            if (ok && g_.get(b + "ffn_gate.weight")) ok = bind(L.gate, b + "ffn_gate.weight", err) && bind(L.up, b + "ffn_up.weight", err);
            else if (ok) ok = bind(L.gateup, b + "ffn_up.weight", err);
        }
        if (!ok || L.attn_norm.empty() || L.ffn_norm.empty()) {
            if (err.empty()) err = "layer " + std::to_string(i) + " is incomplete (MoE / hybrid models aren't supported yet)";
            layers_.clear(); g_.close(); return false;
        }
        for (Lin* l : L.lins()) {
            if (!l->t) continue;
            if (!L.lo || l->t->data < L.lo) L.lo = l->t->data;
            if (l->t->data + l->t->nbytes > L.hi) L.hi = l->t->data + l->t->nbytes;
        }
        if (L.gate_exps.t) { hp.n_ff_exp = L.gate_exps.rows / hp.n_expert; moe_ = true; }
        else if (L.gateup.t) hp.n_ff = L.gateup.rows / 2;
        else hp.n_ff = L.gate.rows;
    }

    hp.n_ff = std::max(hp.n_ff, hp.n_ff_exp);
    inv_freq_.resize(hp.n_rot / 2); inv_freq_swa_.resize(hp.n_rot / 2);
    std::vector<float> factors = vec("rope_freqs.weight");
    float rscale = g_.gets(P + "rope.scaling.type") == "linear" ? (float)g_.getf(P + "rope.scaling.factor", 1.0) : 1.f;
    if (rscale <= 0) rscale = 1.f;
    for (int i = 0; i < hp.n_rot / 2; i++) {
        inv_freq_[i] = std::pow(hp.rope_base, -2.0f * i / hp.n_rot) / rscale;   // global layers carry the context scaling
        if (i < (int)factors.size() && factors[i] > 0) inv_freq_[i] /= factors[i];
        inv_freq_swa_[i] = std::pow(hp.rope_base_swa, -2.0f * i / hp.n_rot);
    }

    ctx = std::max(256, std::min(o.ctx, hp.ctx_train > 0 ? hp.ctx_train : o.ctx));
    size_t kvd = (size_t)hp.n_kv * hp.hd;
    kc_.assign((size_t)hp.n_layer * ctx * kvd, 0);
    vc_.assign((size_t)hp.n_layer * ctx * kvd, 0);
    n_past = 0;

    int nt = o.threads > 0 ? o.threads : std::max(1, (int)std::thread::hardware_concurrency() / 2);
    pool_.start(nt);
    int maxc = std::max({hp.n_embd, hp.n_ff, hp.n_head * hp.hd, ctx}) + 64;
    scratch_.assign(nt, std::vector<float>(maxc));
    bufn_ = 0;

    // ---- Tier plan: VRAM first (output head, then layers from the bottom), then pinned RAM, then disk streaming.
    gpu_info = gpu_init_info();
    int64_t vram_budget = 0;
    size_t vfree = 0, vtotal = 0;
    if (o.vram_mb != 0 && cuda().ok() && cuda().mem_info(vfree, vtotal)) {
        int64_t reserve = 768ll << 20;  // leave headroom for other apps (Blender, the desktop)
        vram_budget = o.vram_mb < 0 ? (int64_t)vfree - reserve : std::min<int64_t>(o.vram_mb << 20, (int64_t)vfree - (256ll << 20));
        // With tensor-core flash attention, prompt batches never materialize a score matrix, so the
        // score buffer only needs one token's worth (decode) and batches can stay large.
        std::string ge;
        const bool flash = !getenv("ITHACA_NO_FLASH") && !getenv("ITHACA_NO_TC_PREFILL") && (gx().ok() || gx().init(ge)) && gx().tensor_cores() &&
                           (hp.hd == 64 || hp.hd == 80 || hp.hd == 96 || hp.hd == 128 || hp.hd == 256);
        gpu_batch_ = flash ? 256 : (int)std::max<int64_t>(1, std::min<int64_t>(256, (128ll << 20) / ((int64_t)hp.n_head * ctx * 4)));
        sc_rows_ = flash ? 1 : gpu_batch_;
        int64_t act = (int64_t)gpu_batch_ * 4 * ((int64_t)hp.n_embd * 2 + hp.n_head * hp.hd * 2 + kvd * 2 + hp.n_ff * 3 +
                      std::max<int64_t>({hp.n_embd, hp.n_head * hp.hd + 2 * (int64_t)kvd, 2ll * hp.n_ff}) + hp.n_rot) +
                      (int64_t)sc_rows_ * hp.n_head * ctx * 4 + hp.n_vocab * 4ll;
        vram_budget -= act + (64ll << 20);
    }
    MEMORYSTATUSEX ms{sizeof(ms)}; GlobalMemoryStatusEx(&ms);
    int64_t kv_bytes = (int64_t)(kc_.size() + vc_.size()) * 2;
    int64_t ram_budget = o.ram_mb < 0 ? (int64_t)ms.ullAvailPhys - (2560ll << 20) - kv_bytes : (o.ram_mb << 20);
    if (ram_budget < 0) ram_budget = 0;

    int64_t vused = 0, rused = 0;
    out_plan_ = TIER_DISK;
    if (vram_budget > 0 && (int64_t)gpu_cost(out_) <= vram_budget) { out_plan_ = TIER_GPU; vused += gpu_cost(out_); }
    bool gpu_open = vram_budget > 0;
    for (auto& L : layers_) {
        size_t c = 0;
        for (Lin* l : L.lins()) c += gpu_cost(*l);
        c += (size_t)ctx * kvd * 4;  // f16 K+V cache in VRAM
        if (gpu_open && vused + (int64_t)c <= vram_budget) { L.plan = TIER_GPU; vused += c; continue; }
        gpu_open = false;
        if (rused + (int64_t)L.bytes() <= ram_budget) { L.plan = TIER_RAM; rused += L.bytes(); }
        else L.plan = TIER_DISK;
    }
    if (rused > 0) {
        SIZE_T mn = (SIZE_T)(rused + (512ll << 20)), mx = (SIZE_T)(rused + kv_bytes + (2048ll << 20));
        SetProcessWorkingSetSizeEx(GetCurrentProcess(), mn, mx, QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE);
    }
    stop_ = false;
    migrating_ = true;
    tier_th_ = std::thread([this] { tier_thread(); });
    pre_th_ = std::thread([this] { prefetch_thread(); });
    loaded_ = true;
    load_sec_ = now_sec() - t0;
    return true;
}

struct MemRange { PVOID va; SIZE_T bytes; };
typedef BOOL(WINAPI* PrefetchFn)(HANDLE, ULONG_PTR, MemRange*, ULONG);
// Ask the OS to read a mapped range with large async I/O instead of 4 KB page faults.
static void prefetch_range(const void* p, size_t n) {
    static PrefetchFn pf = (PrefetchFn)(void*)GetProcAddress(GetModuleHandleA("kernel32.dll"), "PrefetchVirtualMemory");
    if (!pf || !p || !n) return;
    MemRange e{(PVOID)p, n};
    pf(GetCurrentProcess(), 1, &e, 0);
}

void Model::tier_thread() {
    cuda().bind_thread();
    // Separate read handle: big positioned reads into pinned memory beat page-faulting the mapping.
    HANDLE fh = CreateFileW(utf8_to_w(path_utf8).c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);   // no SEQUENTIAL_SCAN: it evicts the file from the cache right behind the reader
    struct FH { HANDLE h; ~FH() { if (h != INVALID_HANDLE_VALUE) CloseHandle(h); } } fh_guard{fh};
    auto upload = [&](Lin& l) -> bool {
        if (!l.t) return true;
        if (fh != INVALID_HANDLE_VALUE && CudaTier::can_load(l.t->type) && !getenv("ITHACA_SLOW_LOAD")) {
            int kind = 0; uint64_t qoff = 0;
            uint64_t p = cuda().load_weight(fh, (uint64_t)(l.t->data - g_.base), l.t->nbytes, l.t->type, l.rows, l.cols, kind, qoff);
            if (p) {
                l.gkind = kind; l.gqoff = qoff;
                l.dptr.store(p, std::memory_order_release);
                gpu_bytes_ += CudaTier::gpu_bytes(l.t->type, l.rows, l.cols);
                return true;
            }
        }
        // Repack into the GPU layout: [f32 scales][pad to 256][quants]. Q4_0 stays 4-bit,
        // everything else becomes 8-bit. Done on a few side threads, not the inference pool.
        const bool q4 = l.t->type == GT_Q4_0;
        const size_t nb = (size_t)l.cols / 32, nblk = nb * l.rows, qb = q4 ? 16 : 32;
        const size_t qoff = (nblk * 4 + 255) / 256 * 256;
        std::vector<uint8_t> buf(qoff + nblk * qb);
        float* S = (float*)buf.data(); uint8_t* Q = buf.data() + qoff;
        const size_t rb_in = gtype_row_bytes(l.t->type, l.cols);
        int nth = std::max(2, (int)std::thread::hardware_concurrency() / 2);
        std::vector<std::thread> th;
        for (int w = 0; w < nth; w++) th.emplace_back([&, w] {
            std::vector<float> row(l.cols);
            for (int r = w; r < l.rows && !stop_; r += nth) {
                const uint8_t* src = l.t->data + r * rb_in;
                for (size_t b = 0; b < nb; b++) {
                    size_t i = r * nb + b;
                    if (q4) { uint16_t d; memcpy(&d, src + b * 18, 2); S[i] = h2f(d); memcpy(Q + i * 16, src + b * 18 + 2, 16); }
                    else if (l.t->type == GT_Q8_0) { uint16_t d; memcpy(&d, src + b * 34, 2); S[i] = h2f(d); memcpy(Q + i * 32, src + b * 34 + 2, 32); }
                }
                if (q4 || l.t->type == GT_Q8_0) continue;
                dequant_row(l.t->type, src, row.data(), l.cols);
                for (size_t b = 0; b < nb; b++) {
                    const float* x = row.data() + b * 32; float amax = 0;
                    for (int j = 0; j < 32; j++) amax = std::max(amax, std::fabs(x[j]));
                    float d = amax / 127.f, id = d ? 1.f / d : 0.f;
                    size_t i = r * nb + b; S[i] = d;
                    for (int j = 0; j < 32; j++) Q[i * 32 + j] = (uint8_t)(int8_t)std::lrintf(x[j] * id);
                }
            }
        });
        for (auto& t : th) t.join();
        if (stop_) return false;
        size_t bytes = buf.size();
        uint64_t p = cuda().alloc_upload(buf.data(), bytes);
        if (!p) return false;
        l.gkind = q4 ? CudaTier::K_Q4_0 : CudaTier::K_Q8_0;
        l.gqoff = qoff;
        l.dptr.store(p, std::memory_order_release);
        gpu_bytes_ += bytes;
        return true;
    };
    double tt0 = now_sec();
    struct Done { std::atomic<bool>& f; double t0; Model* m; ~Done() { f = false; fprintf(stderr, "[tier] placed in %.2f s (%lld MB VRAM, %lld MB RAM)\n", now_sec() - t0, (long long)(m->gpu_bytes_ >> 20), (long long)(m->ram_bytes_ >> 20)); } } done{migrating_, tt0, this};
    bool any_gpu = out_plan_ == TIER_GPU;
    for (auto& L : layers_) any_gpu |= L.plan == TIER_GPU;
    if (any_gpu) dev_ok_ = alloc_dev_buffers();
    if (dev_ok_ && !getenv("ITHACA_NO_TC_PREFILL")) { std::string e; tc_prefill_ = (gx().ok() || gx().init(e)) && gx().tensor_cores(); }
    const size_t kvd = (size_t)hp.n_kv * hp.hd;
    for (size_t li = 0; li < layers_.size(); li++) {
        Layer& L = layers_[li];
        if (stop_) return;
        if (L.plan != TIER_GPU) continue;
        bool ok = true;
        for (Lin* l : L.lins()) ok = ok && upload(*l);
        if (ok && dev_ok_) {
            L.d_an = dev_vec(L.attn_norm); L.d_fn = dev_vec(L.ffn_norm);
            L.d_qn = dev_vec(L.q_norm); L.d_kn = dev_vec(L.k_norm);
            L.d_bq = dev_vec(L.bq); L.d_bk = dev_vec(L.bk); L.d_bv = dev_vec(L.bv);
            L.d_pan = dev_vec(L.post_attn_norm); L.d_pfn = dev_vec(L.post_ffn_norm);
            {   // the KV cache starts small and grows with the conversation (ensure_kv); the tier plan
                // already budgets the full context, so growing never pushes this model out of VRAM
                std::lock_guard<std::mutex> kl(kv_m_);
                int cap = std::min(ctx, std::max(1024, (n_past + 256 + 511) / 512 * 512));   // room for one more batch
                L.d_kc = cuda().alloc((size_t)cap * kvd * 2); L.d_vc = cuda().alloc((size_t)cap * kvd * 2);
                L.kv_cap = cap;
            }
            ok = L.d_an && L.d_fn && L.d_kc && L.d_vc;
            L.kv_dev = 0;
        }
        if (ok) L.state = TIER_GPU; else L.plan = TIER_RAM;  // VRAM ran out: fall back
    }
    if (out_plan_ == TIER_GPU && !stop_ && upload(out_)) { d_onorm_ = dev_vec(out_norm_); out_state_ = TIER_GPU; }
    for (auto& L : layers_) {
        if (stop_) return;
        if (L.plan != TIER_RAM) continue;
        prefetch_range(L.lo, L.bytes());
        if (!VirtualLock((void*)L.lo, L.bytes())) {
            volatile uint8_t sink = 0;  // could not lock: at least fault it into the page cache
            for (const uint8_t* p = L.lo; p < L.hi && !stop_; p += 4096) sink ^= *p;
        }
        ram_bytes_ += L.bytes();
        L.state = TIER_RAM;
    }
}

void Model::prefetch_thread() {
    PrefetchFn pf = (PrefetchFn)(void*)GetProcAddress(GetModuleHandleA("kernel32.dll"), "PrefetchVirtualMemory");
    for (;;) {
        int want;
        {
            std::unique_lock<std::mutex> l(pre_m_);
            pre_cv_.wait(l, [&] { return stop_ || pre_want_ >= 0; });
            if (stop_) return;
            want = pre_want_; pre_want_ = -1;
        }
        // Read ahead the next layers that still live on disk so the SSD works while the CPU computes.
        for (int i = want; i < want + 3 && i < (int)layers_.size(); i++) {
            Layer& L = layers_[i];
            if (L.state != TIER_DISK || !L.bytes()) continue;
            if (pf) { MemRange e{(PVOID)L.lo, L.bytes()}; pf(GetCurrentProcess(), 1, &e, 0); }
        }
    }
}

void Model::unload() {
    if (!loaded_ && !g_.base) return;
    stop_ = true;
    pre_cv_.notify_all();
    if (tier_th_.joinable()) tier_th_.join();
    if (pre_th_.joinable()) pre_th_.join();
    pool_.stop();
    auto fr = [](Lin& l) { uint64_t p = l.dptr.exchange(0); if (p) cuda().free(p); };
    fr(out_);
    dev_ok_ = false;
    cuda().graph_destroy(graph_); graph_ = nullptr;
    for (auto& kv : seg_graphs_) cuda().graph_destroy(kv.second);
    seg_graphs_.clear();
    for (uint64_t p : dev_allocs_) cuda().free(p);
    dev_allocs_.clear();
    dX_ = dXN_ = dQ_ = dK_ = dV_ = dATT_ = dF1_ = dF2_ = dT_ = dLOG_ = dSC_ = dCS_ = d_onorm_ = 0;
    for (auto& L : layers_) {
        for (Lin* l : L.lins()) fr(*l);
        if (L.d_kc) cuda().free(L.d_kc);
        if (L.d_vc) cuda().free(L.d_vc);
        L.d_kc = L.d_vc = 0; L.kv_cap = 0;
    }
    layers_.clear();
    out_.t = nullptr; out_state_ = TIER_DISK;
    if (dLOGA_) { cuda().free(dLOGA_); dLOGA_ = 0; } loga_rows_ = 0;
    kc_.clear(); kc_.shrink_to_fit(); vc_.clear(); vc_.shrink_to_fit();
    x_.clear(); xn_.clear(); q_.clear(); k_.clear(); v_.clear(); att_.clear(); ff1_.clear(); ff2_.clear(); tmp_.clear();
    bufn_ = 0; gpu_bytes_ = 0; ram_bytes_ = 0; n_past = 0;
    SetProcessWorkingSetSizeEx(GetCurrentProcess(), (SIZE_T)-1, (SIZE_T)-1, 0);
    g_.close();
    loaded_ = false;
}

void Model::ensure_buffers(int n) {
    if (n <= bufn_) return;
    bufn_ = n;
    size_t d = hp.n_embd, qd = (size_t)hp.n_head * hp.hd, kvd = (size_t)hp.n_kv * hp.hd, ff = hp.n_ff;
    x_.resize(n * d); xn_.resize(n * d); q_.resize(n * qd); k_.resize(n * kvd); v_.resize(n * kvd);
    att_.resize(n * qd); ff1_.resize(n * ff * 2); ff2_.resize(n * ff);
    moe_x_.resize(n * d); router_.resize((size_t)n * std::max(1, hp.n_expert));
    tmp_.resize(n * std::max({d, qd + 2 * kvd, 2 * ff}));
    logits_.resize(hp.n_vocab);
    qbuf_.resize(n * (std::max({d, qd, ff}) / 32 + 1));
}

void Model::matmul(Lin& w, const float* X, int n, float* Y) {
    uint64_t dp = w.dptr.load(std::memory_order_acquire);
    if (dp && cuda().matmul(dp, w.gqoff, w.gkind, X, n, w.cols, w.rows, Y)) return;
    matmul_raw(w.t->type, w.t->data, w.rows, w.cols, X, n, Y);
}

void Model::matmul_raw(int type, const uint8_t* base, int rows, int cols, const float* X, int n, float* Y) {
    const size_t rb = gtype_row_bytes(type, cols);
    if (type == GT_Q4_0 || type == GT_Q8_0) {
        const int nb = cols / 32;
        for (int t = 0; t < n; t++) quantize_act(X + (size_t)t * cols, qbuf_.data() + (size_t)t * nb, cols);
        const BQ8* qb = qbuf_.data();
        pool_.run(rows, 16, [&](int64_t b, int64_t e, int) {
            for (int64_t r = b; r < e; r++) {
                const uint8_t* wr = base + r * rb;
                for (int t = 0; t < n; t++)
                    Y[(size_t)t * rows + r] = type == GT_Q4_0 ? dot_q4_0(wr, qb + (size_t)t * nb, nb) : dot_q8_0(wr, qb + (size_t)t * nb, nb);
            }
        });
        return;
    }
    pool_.run(rows, 16, [&](int64_t b, int64_t e, int tid) {
        float* buf = scratch_[tid].data();
        for (int64_t r = b; r < e; r++) {
            const float* wr;
            if (type == GT_F32) wr = (const float*)(base + r * rb);
            else { dequant_row(type, base + r * rb, buf, cols); wr = buf; }
            for (int t = 0; t < n; t++) Y[(size_t)t * rows + r] = dot_f32(wr, X + (size_t)t * cols, cols);
        }
    });
}

void Model::rmsnorm(const float* x, const float* w, float* y, int n, int dim) {
    for (int t = 0; t < n; t++) {
        const float* a = x + (size_t)t * dim; float* o = y + (size_t)t * dim;
        float ss = dot_f32(a, a, dim);
        float s = 1.0f / std::sqrt(ss / dim + hp.eps);
        for (int i = 0; i < dim; i++) o[i] = a[i] * s * w[i];
    }
}

void Model::rope(float* x, int nheads, int pos, bool swa) {
    int half = hp.n_rot / 2;
    float cs[512], sn[512];
    const std::vector<float>& f = swa ? inv_freq_swa_ : inv_freq_;
    for (int i = 0; i < half; i++) { float th = pos * f[i]; cs[i] = std::cos(th); sn[i] = std::sin(th); }
    for (int h = 0; h < nheads; h++) {
        float* v = x + (size_t)h * hp.hd;
        for (int i = 0; i < half; i++) {
            int i0 = hp.neox ? i : 2 * i, i1 = hp.neox ? i + half : 2 * i + 1;
            float a = v[i0], b = v[i1];
            v[i0] = a * cs[i] - b * sn[i];
            v[i1] = a * sn[i] + b * cs[i];
        }
    }
}


uint64_t Model::dev_vec(const std::vector<float>& v) {
    if (v.empty()) return 0;
    uint64_t p = cuda().alloc_upload(v.data(), v.size() * 4);
    if (p) dev_allocs_.push_back(p);
    return p;
}

bool Model::alloc_dev_buffers() {
    size_t B = gpu_batch_, d = hp.n_embd, qd = (size_t)hp.n_head * hp.hd, kvd = (size_t)hp.n_kv * hp.hd, ff = hp.n_ff;
    size_t tmax = std::max({d, qd + 2 * kvd, 2 * ff});
    struct { uint64_t* p; size_t n; } bufs[] = {
        {&dX_, B * d}, {&dXN_, B * d}, {&dQ_, B * qd}, {&dK_, B * kvd}, {&dV_, B * kvd}, {&dATT_, B * qd},
        {&dF1_, B * ff}, {&dF2_, B * ff}, {&dT_, B * tmax}, {&dLOG_, (size_t)hp.n_vocab},
        {&dSC_, (size_t)sc_rows_ * hp.n_head * (size_t)ctx}, {&dCS_, B * (size_t)hp.n_rot}, {&dCS2_, B * (size_t)hp.n_rot}, {&dPOS_, 4},
        {&dR_, B * (size_t)std::max(1, hp.n_expert)}, {&dG_, B * d}, {&dIdx_, B * (size_t)std::max(1, hp.n_expert_used)}, {&dWt_, B * (size_t)std::max(1, hp.n_expert_used)},
        {&dPM_, (size_t)hp.n_head * 16}, {&dPL_, (size_t)hp.n_head * 16}, {&dPO_, (size_t)hp.n_head * 16 * hp.hd}};
    for (auto& b : bufs) {
        *b.p = cuda().alloc(b.n * 4);
        if (!*b.p) return false;
        dev_allocs_.push_back(*b.p);
    }
    return cuda().reserve_act(B * std::max({d, qd, ff}));
}

bool Model::all_gpu() const {
    if (out_state_ != TIER_GPU) return false;
    for (auto& L : layers_) if (L.state != TIER_GPU) return false;
    return true;
}

void Model::fill_cs(int n) {
    int half = hp.n_rot / 2;
    cs_.resize((size_t)n * half * 2);
    cs2_.resize(hp.swa_window ? cs_.size() : 0);
    for (int t = 0; t < n; t++)
        for (int i = 0; i < half; i++) {
            float th = (n_past + t) * inv_freq_[i];
            cs_[((size_t)t * half + i) * 2] = std::cos(th);
            cs_[((size_t)t * half + i) * 2 + 1] = std::sin(th);
            if (!cs2_.empty()) {
                float t2 = (n_past + t) * inv_freq_swa_[i];
                cs2_[((size_t)t * half + i) * 2] = std::cos(t2);
                cs2_[((size_t)t * half + i) * 2 + 1] = std::sin(t2);
            }
        }
}

// Top-k routing: softmax over all experts, keep the k best (renormalized when the model says so).
void Model::route(const float* lg, int n, std::vector<std::vector<std::pair<int, float>>>& pe) {
    const int E = hp.n_expert, K = std::min(hp.n_expert_used, hp.n_expert);
    pe.assign(E, {});
    std::vector<std::pair<float, int>> p(E);
    for (int t = 0; t < n; t++) {
        const float* l = lg + (size_t)t * E;
        float mx = *std::max_element(l, l + E), s = 0;
        for (int e = 0; e < E; e++) { p[e] = {std::exp(l[e] - mx), e}; s += p[e].first; }
        std::partial_sort(p.begin(), p.begin() + K, p.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        float ws = 0;
        for (int k = 0; k < K; k++) ws += p[k].first;
        const float norm = hp.expert_norm ? ws : s;
        for (int k = 0; k < K; k++) pe[p[k].second].push_back({t, p[k].first / norm});
    }
}

static inline float act_silu(float x) { return x / (1.0f + std::exp(-x)); }
static inline float act_gelu(float x) { return 0.5f * x * (1.f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x))); }

// Mixture-of-experts FFN on the CPU: tokens are grouped per expert so each expert's weights are read once.
void Model::moe_cpu(Layer& L, int n) {
    const int d = hp.n_embd, ff = hp.n_ff_exp;
    matmul(L.gate_inp, xn_.data(), n, router_.data());
    std::vector<std::vector<std::pair<int, float>>> pe;
    route(router_.data(), n, pe);
    const size_t eg = gtype_row_bytes(L.gate_exps.t->type, d) * ff, eu = gtype_row_bytes(L.up_exps.t->type, d) * ff,
                 ed = gtype_row_bytes(L.down_exps.t->type, ff) * d;
    for (int e = 0; e < hp.n_expert; e++) {
        const int c = (int)pe[e].size();
        if (!c) continue;
        for (int i = 0; i < c; i++) memcpy(&moe_x_[(size_t)i * d], &xn_[(size_t)pe[e][i].first * d], d * 4);
        matmul_raw(L.gate_exps.t->type, L.gate_exps.t->data + e * eg, ff, d, moe_x_.data(), c, ff1_.data());
        matmul_raw(L.up_exps.t->type, L.up_exps.t->data + e * eu, ff, d, moe_x_.data(), c, ff2_.data());
        for (size_t i = 0; i < (size_t)c * ff; i++) ff1_[i] = (hp.gelu ? act_gelu(ff1_[i]) : act_silu(ff1_[i])) * ff2_[i];
        matmul_raw(L.down_exps.t->type, L.down_exps.t->data + e * ed, d, ff, ff1_.data(), c, tmp_.data());
        for (int i = 0; i < c; i++) {
            float w = pe[e][i].second * hp.res_scale;
            float* xr = &x_[(size_t)pe[e][i].first * d]; const float* yr = &tmp_[(size_t)i * d];
            for (int j = 0; j < d; j++) xr[j] += w * yr[j];
        }
    }
}

// Expert slice of a stacked GPU weight: rows [e*R, (e+1)*R).
static void expert_slice(const Lin& l, int e, int ne, uint64_t& W, uint64_t& qoff) {
    const size_t nb = l.cols / 32, R = l.rows / ne, boff = (size_t)e * R * nb;
    const size_t sb = l.gkind == CudaTier::K_Q4M ? 8 : 4, qb = l.gkind == CudaTier::K_Q8_0 ? 32 : 16;
    W = l.dptr.load() + boff * sb;
    qoff = l.gqoff + boff * qb - boff * sb;
}

// Mixture-of-experts FFN on the GPU. Routing is read back to the host (one small sync per layer),
// then each used expert runs on its gathered tokens and scatters the weighted result into x.
void Model::moe_gpu(Layer& L, int n) {
    CudaTier& cu = cuda();
    const int E = hp.n_expert;
    unsigned d = hp.n_embd, ff = hp.n_ff_exp;
    cu.matmul_dev(L.gate_inp.dptr.load(), L.gate_inp.gqoff, L.gate_inp.gkind, dXN_, n, L.gate_inp.cols, L.gate_inp.rows, dR_);
    if (n == 1) {
        // Decode: route on the device and run the K experts as slots of one launch each (graph-capturable).
        const int K = std::min(hp.n_expert_used, E);
        cu.topk(dR_, E, K, hp.expert_norm, hp.res_scale, dIdx_, dWt_);
        cu.matmul_experts(L.gate_exps.dptr.load(), L.gate_exps.gqoff, L.gate_exps.gkind, dIdx_, K, dXN_, 1, 0, d, ff, dF1_);
        cu.matmul_experts(L.up_exps.dptr.load(), L.up_exps.gqoff, L.up_exps.gkind, dIdx_, K, dXN_, 1, 0, d, ff, dF2_);
        cu.ew(hp.gelu ? CudaTier::GELU : CudaTier::SILU, K * ff, {&dF1_, &dF2_});
        cu.matmul_experts(L.down_exps.dptr.load(), L.down_exps.gqoff, L.down_exps.gkind, dIdx_, K, dF1_, K, 1, ff, d, dT_);
        cu.wsum(dX_, dT_, dWt_, K, d);
        return;
    }
    cu.dtoh(router_.data(), dR_, (size_t)n * E * 4);
    std::vector<std::vector<std::pair<int, float>>> pe;
    route(router_.data(), n, pe);
    std::vector<uint32_t> idx; std::vector<float> w; std::vector<size_t> off(E + 1);
    for (int e = 0; e < E; e++) {
        off[e] = idx.size();
        for (auto& p : pe[e]) { idx.push_back(p.first); w.push_back(p.second * hp.res_scale); }
    }
    off[E] = idx.size();
    if (idx.empty()) return;
    cu.htod_async(dIdx_, idx.data(), idx.size() * 4);
    cu.htod_async(dWt_, w.data(), w.size() * 4);
    for (int e = 0; e < E; e++) {
        unsigned c = (unsigned)(off[e + 1] - off[e]);
        if (!c) continue;
        uint64_t pI = dIdx_ + off[e] * 4, pW = dWt_ + off[e] * 4, W, q;
        cu.ew(CudaTier::GATHER, c * d, {&dXN_, &pI, &dG_, &d});
        expert_slice(L.gate_exps, e, E, W, q); mmq(W, q, L.gate_exps.gkind, dG_, c, d, ff, dF1_);
        expert_slice(L.up_exps, e, E, W, q); mmq(W, q, L.up_exps.gkind, dG_, c, d, ff, dF2_);
        cu.ew(hp.gelu ? CudaTier::GELU : CudaTier::SILU, c * ff, {&dF1_, &dF2_});
        expert_slice(L.down_exps, e, E, W, q); mmq(W, q, L.down_exps.gkind, dF1_, c, ff, d, dT_);
        cu.ew(CudaTier::SCATTER, c * d, {&dX_, &dT_, &pI, &pW, &d});
    }
}

// Grows every VRAM layer's KV buffers to hold `need` positions (doubling, capped at ctx).
// If VRAM is short, on_vram_short gets a chance to free memory (the app drops the draft model).
bool Model::ensure_kv(int need) {
    if (!dev_ok_) return true;
    std::lock_guard<std::mutex> kl(kv_m_);
    const size_t kvd = (size_t)hp.n_kv * hp.hd;
    bool grew = false;
    for (auto& L : layers_) {
        if (!L.d_kc || L.kv_cap >= need) continue;
        int cap = std::min(ctx, std::max(need, L.kv_cap * 2));
        cap = std::min(ctx, (cap + 255) / 256 * 256);
        uint64_t nk = 0, nv = 0;
        for (int attempt = 0; attempt < 2; attempt++) {
            nk = cuda().alloc((size_t)cap * kvd * 2); nv = nk ? cuda().alloc((size_t)cap * kvd * 2) : 0;
            if (nk && nv) break;
            if (nk) cuda().free(nk);
            nk = nv = 0;
            if (attempt || !on_vram_short || !on_vram_short()) return false;
        }
        cuda().dtod(nk, L.d_kc, (size_t)L.kv_cap * kvd * 2);
        cuda().dtod(nv, L.d_vc, (size_t)L.kv_cap * kvd * 2);
        cuda().sync();
        cuda().free(L.d_kc); cuda().free(L.d_vc);
        L.d_kc = nk; L.d_vc = nv; L.kv_cap = cap;
        grew = true;
    }
    if (grew) {   // captured graphs hold the old KV pointers
        cuda().graph_destroy(graph_); graph_ = nullptr;
        for (auto& kv : seg_graphs_) cuda().graph_destroy(kv.second);
        seg_graphs_.clear();
    }
    return true;
}

int64_t Model::kv_bytes_gpu() const {
    int64_t b = 0;
    for (auto& L : layers_) if (L.d_kc) b += (int64_t)L.kv_cap * hp.n_kv * hp.hd * 4;
    return b;
}

void Model::rewind(int n) {
    if (n < n_past) n_past = n;
    for (auto& L : layers_) if (L.kv_dev > n) L.kv_dev = n;
}

// Prompt batches go through tensor cores (weights decoded while loading tiles); single tokens use the dp4a kernels.
void Model::mmq(uint64_t W, uint64_t qoff, int kind, uint64_t x, int n, int cols, int rows, uint64_t y) {
    if (n >= 16 && tc_prefill_ && gx().linear_quant(x, n, cols, W, W + qoff, kind, rows, y)) return;
    cuda().matmul_dev(W, qoff, kind, x, n, cols, rows, y);
}

void Model::gpu_layer(Layer& L, int n, unsigned nc) {
    CudaTier& cu = cuda();
    unsigned d = hp.n_embd, hd = hp.hd, nh = hp.n_head, nkv = hp.n_kv, qd = nh * hd, kvd = nkv * hd, ff = hp.n_ff;
    unsigned half = hp.n_rot / 2, neox = hp.neox ? 1 : 0, grp = nh / nkv, ctxu = ctx, un = n;
    float eps = hp.eps, res = hp.res_scale, scale = hp.attn_scale > 0 ? hp.attn_scale : 1.0f / std::sqrt((float)hd);
    auto mm = [&](Lin& w, uint64_t x, uint64_t y) { mmq(w.dptr.load(), w.gqoff, w.gkind, x, n, w.cols, w.rows, y); };
    static bool pp = getenv("ITHACA_PREPROF") != nullptr;
    const bool ppn = pp && n > 1;
    double T0 = ppn ? (cu.sync(), now_sec()) : 0;
    cu.launch(CudaTier::RMSNORM, un, 1, 256, {&dX_, &L.d_an, &dXN_, &d, &eps});
    if (L.qkv.t) {
        mm(L.qkv, dXN_, dT_);
        unsigned W = qd + 2 * kvd;
        uint64_t sk = dT_ + qd * 4ull, sv = dT_ + (qd + kvd) * 4ull;
        cu.ew(CudaTier::COPY2D, un * qd, {&dT_, &W, &dQ_, &qd, &qd});
        cu.ew(CudaTier::COPY2D, un * kvd, {&sk, &W, &dK_, &kvd, &kvd});
        cu.ew(CudaTier::COPY2D, un * kvd, {&sv, &W, &dV_, &kvd, &kvd});
    } else {
        mm(L.q, dXN_, dQ_); mm(L.k, dXN_, dK_); mm(L.v, dXN_, dV_);
    }
    if (L.d_bq) cu.ew(CudaTier::BIAS, un * qd, {&dQ_, &L.d_bq, &qd});
    if (L.d_bk) cu.ew(CudaTier::BIAS, un * kvd, {&dK_, &L.d_bk, &kvd});
    if (L.d_bv) cu.ew(CudaTier::BIAS, un * kvd, {&dV_, &L.d_bv, &kvd});
    if (L.d_qn) cu.launch(CudaTier::RMSNORM, un * nh, 1, 256, {&dQ_, &L.d_qn, &dQ_, &hd, &eps});
    if (L.d_kn) cu.launch(CudaTier::RMSNORM, un * nkv, 1, 256, {&dK_, &L.d_kn, &dK_, &hd, &eps});
    uint64_t cs = L.swa ? dCS2_ : dCS_;
    unsigned win = L.swa ? (unsigned)hp.swa_window : 0;
    cu.launch(CudaTier::ROPE, un, 1, 256, {&dQ_, &cs, &nh, &hd, &half, &neox, &qd});
    cu.launch(CudaTier::ROPE, un, 1, 256, {&dK_, &cs, &nkv, &hd, &half, &neox, &kvd});
    cu.ew(CudaTier::KVSTORE, un * kvd, {&dK_, &L.d_kc, &dPOS_, &kvd});
    cu.ew(CudaTier::KVSTORE, un * kvd, {&dV_, &L.d_vc, &dPOS_, &kvd});
    // Decode splits each head's keys over 16 CTAs (flash-decoding) so long chats keep the GPU busy.
    unsigned spl = n == 1 ? 16 : 1;
    double Ta = ppn ? (cu.sync(), now_sec()) : 0;
    // Prompts: tensor-core flash attention straight from the KV cache; decode: the split-K kernel below.
    if (!(n > 1 && tc_prefill_ && gx().attn_prefill(dQ_, L.d_kc, L.d_vc, dATT_, n, n_past, qd, kvd, nh, grp, hd, scale, win, nc)))
        cu.launch(CudaTier::ATTN, nh, un * spl, 256, {&dQ_, &L.d_kc, &L.d_vc, &dATT_, &dSC_, &nh, &hd, &kvd, &grp, &dPOS_, &ctxu, &scale, &spl, &dPM_, &dPL_, &dPO_, &win, &nc});
    if (spl > 1) cu.launch(CudaTier::ATTN_COMB, nh, un, std::max(32u, (hd + 31) / 32 * 32), {&dPM_, &dPL_, &dPO_, &dATT_, &nh, &hd, &spl});
    static double t_attn = 0, t_all = 0; static int calls = 0;
    if (ppn) { cu.sync(); t_attn += now_sec() - Ta; }
    mm(L.o, dATT_, dT_);
    if (ppn && ++calls % 64 == 0) fprintf(stderr, "[prefill] attention %.0f ms of %.0f ms layer time\n", t_attn * 1e3, t_all * 1e3);
    struct PP { bool on; double t0; double& acc; ~PP() { if (on) { cuda().sync(); acc += now_sec() - t0; } } } pp_guard{ppn, T0, t_all};
    if (L.d_pan) cu.launch(CudaTier::RMSNORM, un, 1, 256, {&dT_, &L.d_pan, &dT_, &d, &eps});
    cu.ew(CudaTier::ADD, un * d, {&dX_, &dT_, &res});
    cu.launch(CudaTier::RMSNORM, un, 1, 256, {&dX_, &L.d_fn, &dXN_, &d, &eps});
    if (L.gate_exps.t) { moe_gpu(L, n); return; }
    if (L.gateup.t) {
        mm(L.gateup, dXN_, dT_);
        cu.ew(CudaTier::SILU_SPLIT, un * ff, {&dT_, &dF1_, &ff});
    } else {
        mm(L.gate, dXN_, dF1_); mm(L.up, dXN_, dF2_);
        cu.ew(hp.gelu ? CudaTier::GELU : CudaTier::SILU, un * ff, {&dF1_, &dF2_});
    }
    mm(L.down, dF1_, dT_);
    if (L.d_pfn) cu.launch(CudaTier::RMSNORM, un, 1, 256, {&dT_, &L.d_pfn, &dT_, &d, &eps});
    cu.ew(CudaTier::ADD, un * d, {&dX_, &dT_, &res});
}

void Model::bench_gpu() {
    if (!dev_ok_ || layers_.empty() || layers_[0].state != TIER_GPU) { printf("gpu bench: layer 0 not on GPU\n"); return; }
    CudaTier& cu = cuda();
    Layer& L = layers_[0];
    const int R = 200;
    auto timeit = [&](const char* name, double bytes, auto fn) {
        float sink[4]; fn(); cu.dtoh(sink, dX_, 4);
        double t0 = now_sec();
        for (int i = 0; i < R; i++) fn();
        cu.dtoh(sink, dX_, 4);
        double us = (now_sec() - t0) / R * 1e6;
        printf("  %-14s %8.1f us  %6.1f GB/s\n", name, us, bytes / (us * 1e-6) / 1e9);
    };
    Lin* big = L.gate.t ? &L.gate : &L.gateup;
    Lin* down = &L.down;
    Lin* kk = L.k.t ? &L.k : &L.qkv;
    auto cost = [](Lin* l) { return (double)l->rows * (l->cols / 32) * (l->gkind == CudaTier::K_Q4_0 ? 20 : 36); };
    timeit("mm gate/up", cost(big), [&] { cu.matmul_dev(big->dptr.load(), big->gqoff, big->gkind, dXN_, 1, big->cols, big->rows, dT_); });
    timeit("mm down", cost(down), [&] { cu.matmul_dev(down->dptr.load(), down->gqoff, down->gkind, dF1_, 1, down->cols, down->rows, dT_); });
    timeit("mm k", cost(kk), [&] { cu.matmul_dev(kk->dptr.load(), kk->gqoff, kk->gkind, dXN_, 1, kk->cols, kk->rows, dK_); });
    unsigned d = hp.n_embd; float eps = hp.eps, one = 1.f;
    timeit("rmsnorm", d * 8.0, [&] { cu.launch(CudaTier::RMSNORM, 1, 1, 256, {&dX_, &L.d_an, &dXN_, &d, &eps}); });
    timeit("add", d * 12.0, [&] { cu.ew(CudaTier::ADD, d, {&dXN_, &dT_, &one}); });
    unsigned hd = hp.hd, nh = hp.n_head, kvd = hp.n_kv * hp.hd, grp = hp.n_head / hp.n_kv, ctxu = ctx; float sc = 0.1f;
    for (unsigned pos : {64u, 1024u, 3000u}) {
        char nm[32]; snprintf(nm, sizeof nm, "attn @%u", pos);
        cu.htod(dPOS_, &pos, 4);
        unsigned spl = 16;
        unsigned z0 = 0;
        timeit(nm, (double)pos * kvd * 4, [&] {
            cu.launch(CudaTier::ATTN, nh, spl, 256, {&dQ_, &L.d_kc, &L.d_vc, &dATT_, &dSC_, &nh, &hd, &kvd, &grp, &dPOS_, &ctxu, &sc, &spl, &dPM_, &dPL_, &dPO_, &z0, &z0});
            cu.launch(CudaTier::ATTN_COMB, nh, 1, 128, {&dPM_, &dPL_, &dPO_, &dATT_, &nh, &hd, &spl});
        });
    }
    if (out_.dptr.load()) timeit("mm output", cost(&out_), [&] { cu.matmul_dev(out_.dptr.load(), out_.gqoff, out_.gkind, dXN_, 1, out_.cols, out_.rows, dLOG_); });
}

static inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

static double g_ev[4]; static int g_evn;
const float* Model::eval(const int* toks, int n_all) {
    struct EvT { bool on; double t0; ~EvT() { if (on && ++g_evn % 20 == 0) fprintf(stderr, "[eval_all] layers %.1f head %.1f total %.1f ms\n", g_ev[0] / g_evn * 1e3, g_ev[1] / g_evn * 1e3, g_ev[2] / g_evn * 1e3); } };
    const bool evp = all_logits_ && getenv("ITHACA_SPECPROF");
    double ev0 = now_sec();
    EvT evt{evp, ev0};
    if (!loaded_ || n_all <= 0 || n_past + n_all > ctx) return nullptr;
    if (!ensure_kv(n_past + n_all)) return nullptr;
    if (n_all == 1 && toks[0] >= 0 && dev_ok_ && !migrating_ && all_gpu()) {
        // Fully VRAM-resident decode: replay one CUDA graph for the whole forward pass.
        CudaTier& cu = cuda();
        const int d = hp.n_embd;
        bool kv_ok = true;
        for (auto& L : layers_) kv_ok &= L.kv_dev >= n_past;
        if (kv_ok) {
            ensure_buffers(1);
            size_t erb = gtype_row_bytes(embd_->type, d);
            int id = std::min(std::max(toks[0], 0), hp.n_vocab - 1);
            dequant_row(embd_->type, embd_->data + (size_t)id * erb, x_.data(), d);
            if (hp.emb_scale != 1.f) for (int i = 0; i < d; i++) x_[i] *= hp.emb_scale;
            fill_cs(1);
            unsigned p = n_past;
            cu.htod_async(dX_, x_.data(), (size_t)d * 4);
            cu.htod_async(dCS_, cs_.data(), cs_.size() * 4);
            if (!cs2_.empty()) cu.htod_async(dCS2_, cs2_.data(), cs2_.size() * 4);
            cu.htod_async(dPOS_, &p, 4);
            if (!graph_ && cu.begin_capture()) {
                for (auto& L : layers_) gpu_layer(L, 1);
                unsigned ud = d; float eps = hp.eps;
                cu.launch(CudaTier::RMSNORM, 1, 1, 256, {&dX_, &d_onorm_, &dXN_, &ud, &eps});
                cu.matmul_dev(out_.dptr.load(), out_.gqoff, out_.gkind, dXN_, 1, out_.cols, out_.rows, dLOG_);
                graph_ = cu.end_capture();
                if (!graph_) fprintf(stderr, "[gpu] graph capture failed, using direct launches\n");
            }
            if (graph_) {
                cu.graph_launch(graph_);
                cu.dtoh(logits_.data(), dLOG_, (size_t)hp.n_vocab * 4);
                if (hp.logit_scale != 1.f) for (auto& v : logits_) v /= hp.logit_scale;
                for (auto& L : layers_) L.kv_dev = n_past + 1;
                n_past += 1;
                return logits_.data();
            }
        }
    }
    const int B = dev_ok_ ? std::min(256, gpu_batch_) : 256;
    for (int c0 = 0, n = 0; c0 < n_all; c0 += n) {
        // batches never mix text and image tokens: an image run is one non-causal block
        const bool img = toks[c0] < 0;
        n = 1;
        while (c0 + n < n_all && n < B && (toks[c0 + n] < 0) == img) n++;
        const unsigned nc = img ? (unsigned)n : 0;
        const int* tk = toks + c0;
        ensure_buffers(n);
        const int d = hp.n_embd, hd = hp.hd, qd = hp.n_head * hd, kvd = hp.n_kv * hd, ff = hp.n_ff;
        size_t erb = gtype_row_bytes(embd_->type, d);
        for (int t = 0; t < n; t++) {
            if (tk[t] < 0) {   // precomputed embedding (image): used as is, no embedding scale
                size_t row = (size_t)(-1 - tk[t]);
                if ((row + 1) * d <= ext_embd.size()) memcpy(&x_[(size_t)t * d], &ext_embd[row * d], d * 4);
                else memset(&x_[(size_t)t * d], 0, d * 4);
                continue;
            }
            int id = std::min(std::max(tk[t], 0), hp.n_vocab - 1);
            dequant_row(embd_->type, embd_->data + (size_t)id * erb, &x_[(size_t)t * d], d);
            if (hp.emb_scale != 1.f) for (int i = 0; i < d; i++) x_[(size_t)t * d + i] *= hp.emb_scale;
        }
        const float scale = hp.attn_scale > 0 ? hp.attn_scale : 1.0f / std::sqrt((float)hd);
        const int grp = hp.n_head / hp.n_kv;
        const size_t kvdz = (size_t)kvd;
        bool on_dev = false, cs_up = false;
        for (int li = 0; li < hp.n_layer; li++) {
            Layer& L = layers_[li];
            { std::lock_guard<std::mutex> l(pre_m_); pre_want_ = li + 1; }
            pre_cv_.notify_one();
            if (dev_ok_ && L.state == TIER_GPU) {
                CudaTier& cu = cuda();
                if (!on_dev) { cu.htod_async(dX_, x_.data(), (size_t)n * d * 4); on_dev = true; }
                if (!cs_up) {
                    fill_cs(n);
                    cu.htod_async(dCS_, cs_.data(), cs_.size() * 4);
                    if (!cs2_.empty()) cu.htod_async(dCS2_, cs2_.data(), cs2_.size() * 4);
                    unsigned p = n_past; cu.htod_async(dPOS_, &p, 4);
                    cu.dtoh(&p, dPOS_, 4);  // small sync so the host buffers above are consumed
                    cs_up = true;
                }
                auto kv_sync = [&](int k) {  // layer just moved to VRAM: bring its CPU-computed KV along
                    Layer& M = layers_[k];
                    if (M.kv_dev >= n_past) return;
                    size_t off = ((size_t)k * ctx + M.kv_dev) * kvdz;
                    size_t cnt = (size_t)(n_past - M.kv_dev) * kvdz * 2;
                    cu.htod_async(M.d_kc + (uint64_t)M.kv_dev * kvdz * 2, &kc_[off], cnt);
                    cu.htod_async(M.d_vc + (uint64_t)M.kv_dev * kvdz * 2, &vc_[off], cnt);
                };
                if (n == 1 && !nc && !migrating_) {
                    // Decode: replay a cached CUDA graph for this whole run of VRAM layers.
                    int e = li;
                    while (e < hp.n_layer && layers_[e].state == TIER_GPU) kv_sync(e++);
                    void*& g = seg_graphs_[li * 4096 + e];
                    if (!g && cu.begin_capture()) {
                        for (int k = li; k < e; k++) gpu_layer(layers_[k], 1);
                        g = cu.end_capture();
                    }
                    if (g && cu.graph_launch(g)) {
                        for (int k = li; k < e; k++) layers_[k].kv_dev = n_past + 1;
                        li = e - 1;
                        continue;
                    }
                }
                kv_sync(li);
                gpu_layer(L, n, nc);
                L.kv_dev = n_past + n;
                continue;
            }
            if (on_dev) { cuda().dtoh(x_.data(), dX_, (size_t)n * d * 4); on_dev = false; }

            rmsnorm(x_.data(), L.attn_norm.data(), xn_.data(), n, d);
            if (L.qkv.t) {
                matmul(L.qkv, xn_.data(), n, tmp_.data());
                for (int t = 0; t < n; t++) {
                    const float* s = &tmp_[(size_t)t * (qd + 2 * kvd)];
                    memcpy(&q_[(size_t)t * qd], s, qd * 4);
                    memcpy(&k_[(size_t)t * kvd], s + qd, kvd * 4);
                    memcpy(&v_[(size_t)t * kvd], s + qd + kvd, kvd * 4);
                }
            } else {
                matmul(L.q, xn_.data(), n, q_.data());
                matmul(L.k, xn_.data(), n, k_.data());
                matmul(L.v, xn_.data(), n, v_.data());
            }
            for (int t = 0; t < n; t++) {
                if (!L.bq.empty()) for (int i = 0; i < qd; i++) q_[(size_t)t * qd + i] += L.bq[i];
                if (!L.bk.empty()) for (int i = 0; i < kvd; i++) k_[(size_t)t * kvd + i] += L.bk[i];
                if (!L.bv.empty()) for (int i = 0; i < kvd; i++) v_[(size_t)t * kvd + i] += L.bv[i];
            }
            if (!L.q_norm.empty()) rmsnorm(q_.data(), L.q_norm.data(), q_.data(), n * hp.n_head, hd);
            if (!L.k_norm.empty()) rmsnorm(k_.data(), L.k_norm.data(), k_.data(), n * hp.n_kv, hd);
            for (int t = 0; t < n; t++) {
                int pos = n_past + t;
                rope(&q_[(size_t)t * qd], hp.n_head, pos, L.swa);
                rope(&k_[(size_t)t * kvd], hp.n_kv, pos, L.swa);
                size_t off = ((size_t)li * ctx + pos) * kvd;
                f32_to_f16(&k_[(size_t)t * kvd], &kc_[off], kvd);
                f32_to_f16(&v_[(size_t)t * kvd], &vc_[off], kvd);
            }
            const uint16_t* kbase = &kc_[(size_t)li * ctx * kvd];
            const uint16_t* vbase = &vc_[(size_t)li * ctx * kvd];
            pool_.run((int64_t)n * hp.n_head, 1, [&](int64_t b, int64_t e, int tid) {
                float* sc = scratch_[tid].data();
                for (int64_t idx = b; idx < e; idx++) {
                    int t = (int)(idx / hp.n_head), h = (int)(idx % hp.n_head), kvh = h / grp;
                    // causal (or the whole block for image tokens), limited to the window on sliding layers
                    int pos = nc ? n_past + n - 1 : n_past + t;
                    int j0 = L.swa && hp.swa_window && pos + 1 > hp.swa_window ? pos + 1 - hp.swa_window : 0;
                    const float* qv = &q_[(size_t)t * qd + (size_t)h * hd];
                    float mx = -1e30f;
                    for (int j = j0; j <= pos; j++) {
                        float s = dot_f16(qv, kbase + (size_t)j * kvd + kvh * hd, hd) * scale;
                        sc[j] = s; if (s > mx) mx = s;
                    }
                    float sum = 0;
                    for (int j = j0; j <= pos; j++) { sc[j] = std::exp(sc[j] - mx); sum += sc[j]; }
                    float inv = 1.0f / sum;
                    float* o = &att_[(size_t)t * qd + (size_t)h * hd];
                    memset(o, 0, hd * 4);
                    for (int j = j0; j <= pos; j++) axpy_f16(o, sc[j] * inv, vbase + (size_t)j * kvd + kvh * hd, hd);
                }
            });
            matmul(L.o, att_.data(), n, tmp_.data());
            if (!L.post_attn_norm.empty()) rmsnorm(tmp_.data(), L.post_attn_norm.data(), tmp_.data(), n, d);
            for (size_t i = 0; i < (size_t)n * d; i++) x_[i] += tmp_[i] * hp.res_scale;

            rmsnorm(x_.data(), L.ffn_norm.data(), xn_.data(), n, d);
            if (L.gate_exps.t) { moe_cpu(L, n); continue; }
            if (L.gateup.t) {
                matmul(L.gateup, xn_.data(), n, tmp_.data());
                for (int t = 0; t < n; t++)
                    for (int i = 0; i < ff; i++)
                        ff1_[(size_t)t * ff + i] = silu(tmp_[(size_t)t * 2 * ff + i]) * tmp_[(size_t)t * 2 * ff + ff + i];
            } else {
                matmul(L.gate, xn_.data(), n, ff1_.data());
                matmul(L.up, xn_.data(), n, ff2_.data());
                for (size_t i = 0; i < (size_t)n * ff; i++) ff1_[i] = (hp.gelu ? act_gelu(ff1_[i]) : silu(ff1_[i])) * ff2_[i];
            }
            matmul(L.down, ff1_.data(), n, tmp_.data());
            if (!L.post_ffn_norm.empty()) rmsnorm(tmp_.data(), L.post_ffn_norm.data(), tmp_.data(), n, d);
            for (size_t i = 0; i < (size_t)n * d; i++) x_[i] += tmp_[i] * hp.res_scale;
        }
        if (c0 + n >= n_all && dev_ok_ && out_state_ == TIER_GPU && out_.dptr.load()) {
            CudaTier& cu = cuda();
            if (!on_dev) { cu.htod_async(dX_, x_.data(), (size_t)n * d * 4); on_dev = true; }
            unsigned ud = d; float eps = hp.eps;
            if (evp) { cuda().sync(); g_ev[0] += now_sec() - ev0; }
            double evh = now_sec();
            if (all_logits_ && n > 1) {   // every position (draft verification)
                if (n > loga_rows_) {
                    if (dLOGA_) cu.free(dLOGA_);
                    loga_rows_ = std::max(n, 16); dLOGA_ = cu.alloc((size_t)loga_rows_ * hp.n_vocab * 4);
                }
                unsigned un = n;
                cu.launch(CudaTier::RMSNORM, un, 1, 256, {&dX_, &d_onorm_, &dXN_, &ud, &eps});
                mmq(out_.dptr.load(), out_.gqoff, out_.gkind, dXN_, n, out_.cols, out_.rows, dLOGA_);
                logits_.resize((size_t)n * hp.n_vocab);
                cu.dtoh(logits_.data(), dLOGA_, logits_.size() * 4);
                if (hp.logit_scale != 1.f) for (auto& v : logits_) v /= hp.logit_scale;
                n_past += n;
                if (evp) { g_ev[1] += now_sec() - evh; g_ev[2] += now_sec() - ev0; }
                continue;
            }
            uint64_t last = dX_ + (uint64_t)(n - 1) * d * 4;
            cu.launch(CudaTier::RMSNORM, 1, 1, 256, {&last, &d_onorm_, &dXN_, &ud, &eps});
            cu.matmul_dev(out_.dptr.load(), out_.gqoff, out_.gkind, dXN_, 1, out_.cols, out_.rows, dLOG_);
            cu.dtoh(logits_.data(), dLOG_, (size_t)hp.n_vocab * 4);
            if (hp.logit_scale != 1.f) for (auto& v : logits_) v /= hp.logit_scale;
            n_past += n;
            continue;
        }
        if (on_dev) { cuda().dtoh(x_.data(), dX_, (size_t)n * d * 4); on_dev = false; }
        n_past += n;
        if (c0 + n >= n_all) {
            const int rows = all_logits_ ? n : 1;
            logits_.resize((size_t)rows * hp.n_vocab);
            rmsnorm(&x_[(size_t)(n - rows) * d], out_norm_.data(), xn_.data(), rows, d);
            matmul(out_, xn_.data(), rows, logits_.data());
            if (hp.logit_scale != 1.f) for (auto& v : logits_) v /= hp.logit_scale;
        }
    }
    return logits_.data();
}

std::string Model::tier_json() {
    std::string s = "{\"loaded\":";
    s += loaded_ ? "true" : "false";
    if (!loaded_) return s + ",\"gpu\":\"" + gpu_init_info() + "\"}";
    char buf[256];
    snprintf(buf, sizeof buf, ",\"gpu_mb\":%lld,\"ram_mb\":%lld,\"load_ms\":%.0f,\"threads\":%d,\"ctx\":%d,\"n_past\":%d,\"out\":[%d,%d],\"layers\":[",
             (long long)(gpu_bytes_ >> 20), (long long)(ram_bytes_ >> 20), load_sec_ * 1000, pool_.size(), ctx, n_past,
             out_plan_, out_state_.load());
    s += buf;
    for (size_t i = 0; i < layers_.size(); i++) {
        snprintf(buf, sizeof buf, "%s[%d,%d,%zu]", i ? "," : "", layers_[i].plan.load(), layers_[i].state.load(), layers_[i].bytes() >> 20);
        s += buf;
    }
    s += std::string("],\"dev\":") + (dev_ok_ ? "true" : "false") + ",\"gpu\":\"" + gpu_info + "\",\"arch\":\"" + hp.arch + "\"}";
    return s;
}

// --quanttest <file.gguf>: for each quant type in the file, load one tensor with the GPU converter
// and compare a GPU matmul against a CPU dequantized reference.
int quant_selftest(const std::wstring& path) {
    GGUF g; std::string err;
    if (!g.open(path, err)) { printf("open failed: %s\n", err.c_str()); return 1; }
    CudaTier& cu = Model::cuda();
    if (!cu.ok()) { printf("no GPU\n"); return 1; }
    HANDLE fh = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, 0, OPEN_EXISTING, 0, 0);
    std::vector<int> seen;
    for (auto& t : g.tensors) {
        if (t.nd != 2 || std::find(seen.begin(), seen.end(), t.type) != seen.end() || !CudaTier::can_load(t.type)) continue;
        seen.push_back(t.type);
        int cols = (int)t.ne[0], rows = (int)std::min<int64_t>(t.ne[1], 512);
        size_t nbytes = gtype_row_bytes(t.type, cols) * rows;
        int kind; uint64_t qoff;
        uint64_t W = cu.load_weight(fh, (uint64_t)(t.data - g.base), nbytes, t.type, rows, cols, kind, qoff);
        if (!W) { printf("  %-6s load failed\n", gtype_name(t.type)); continue; }
        std::vector<float> x(cols), y(rows), ref(rows), row(cols);
        uint32_t s = 1; for (auto& v : x) { s = s * 1664525 + 1013904223; v = ((s >> 8) & 0xFFFF) / 32768.f - 1.f; }
        cu.matmul(W, qoff, kind, x.data(), 1, cols, rows, y.data());
        double num = 0, den = 0;
        for (int r = 0; r < rows; r++) {
            dequant_row(t.type, t.data + r * gtype_row_bytes(t.type, cols), row.data(), cols);
            double a = 0; for (int c = 0; c < cols; c++) a += (double)row[c] * x[c];
            num += (y[r] - a) * (y[r] - a); den += a * a;
        }
        double e = std::sqrt(num / std::max(den, 1e-30));
        printf("  %-6s %-28s rel.err %.5f  %s\n", gtype_name(t.type), t.name.c_str(), e, e < 0.02 ? "OK" : "BAD");
        if (getenv("ITHACA_MMBENCH")) {   // full-size tensor, time n = 1..8 tokens
            int R = (int)t.ne[1]; size_t nb2 = gtype_row_bytes(t.type, cols) * R; int k2; uint64_t q2;
            uint64_t W2 = cu.load_weight(fh, (uint64_t)(t.data - g.base), nb2, t.type, R, cols, k2, q2);
            uint64_t X = cu.alloc((size_t)8 * cols * 4), Y = cu.alloc((size_t)8 * R * 4);
            for (int n : {1, 2, 4, 6, 8}) {
                cu.matmul_dev(W2, q2, k2, X, n, cols, R, Y); cu.sync();
                double t0 = now_sec();
                for (int it = 0; it < 50; it++) cu.matmul_dev(W2, q2, k2, X, n, cols, R, Y);
                cu.sync();
                double us = (now_sec() - t0) / 50 * 1e6;
                printf("      n=%d  %7.1f us  (%.0f GB/s of weights)\n", n, us, CudaTier::gpu_bytes(t.type, R, cols) / (us * 1e3));
            }
            cu.free(X); cu.free(Y); cu.free(W2);
        }
        cu.free(W);
    }
    CloseHandle(fh);
    return 0;
}

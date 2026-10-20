#include "gx.h"
#include "json.h"
#include "model.h"
#include "quant.h"
#include <cmath>

std::string gx_ptx();
std::string gx_ptx_tc();

Gx& gx() { static Gx g; return g; }
static void prof_launch(void* fn, unsigned gx_, unsigned gy, unsigned gz, unsigned b, unsigned sh, void** a);
static CudaTier& cu() { return Model::cuda(); }

// ---------------- safetensors ----------------
bool SafeTensors::add(const std::wstring& path, std::string& err, const std::string& strip) {
    Map m{};
    m.f = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (m.f == INVALID_HANDLE_VALUE) { err = "cannot open " + w_to_utf8(path); return false; }
    LARGE_INTEGER sz; GetFileSizeEx(m.f, &sz); m.size = (size_t)sz.QuadPart;
    m.m = CreateFileMappingW(m.f, nullptr, PAGE_READONLY, 0, 0, nullptr);
    m.base = m.m ? (const uint8_t*)MapViewOfFile(m.m, FILE_MAP_READ, 0, 0, 0) : nullptr;
    if (!m.base) { if (m.m) CloseHandle(m.m); CloseHandle(m.f); err = "cannot map " + w_to_utf8(path); return false; }
    if (m.size > 4 && m.base[0] == 'P' && m.base[1] == 'K') {   // PyTorch zip checkpoint
        extern bool torch_load_zip(const uint8_t*, size_t, std::map<std::string, STensor>&, std::string&);
        std::map<std::string, STensor> tt;
        std::string e2;
        torch_load_zip(m.base, m.size, tt, e2);
        if (tt.empty()) { err = "could not read PyTorch checkpoint: " + e2; UnmapViewOfFile(m.base); CloseHandle(m.m); CloseHandle(m.f); return false; }
        for (auto& kv : tt) {
            std::string name = kv.first;
            if (!strip.empty() && name.compare(0, strip.size(), strip) == 0) name = name.substr(strip.size());
            t_[name] = kv.second;
        }
        maps_.push_back(m);
        return true;
    }
    uint64_t hl; memcpy(&hl, m.base, 8);
    if (hl > m.size - 8 || hl > (512u << 20)) { err = "not a safetensors file"; UnmapViewOfFile(m.base); CloseHandle(m.m); CloseHandle(m.f); return false; }
    J h;
    if (!json_parse(std::string((const char*)m.base + 8, hl), h) || h.t != J::OBJ) { err = "bad safetensors header"; UnmapViewOfFile(m.base); CloseHandle(m.m); CloseHandle(m.f); return false; }
    const uint8_t* data = m.base + 8 + hl;
    for (auto& kv : h.o) {
        if (kv.first == "__metadata__") continue;
        STensor t;
        t.dtype = kv.second["dtype"].str();
        for (auto& d : kv.second["shape"].a) t.shape.push_back((int64_t)d.n);
        auto& off = kv.second["data_offsets"].a;
        if (off.size() != 2) continue;
        t.data = data + (size_t)off[0].n;
        t.nbytes = (size_t)(off[1].n - off[0].n);
        std::string name = kv.first;
        if (!strip.empty() && name.compare(0, strip.size(), strip) == 0) name = name.substr(strip.size());
        t_[name] = t;
    }
    maps_.push_back(m);
    return true;
}
void SafeTensors::close() {
    for (auto& m : maps_) { UnmapViewOfFile(m.base); CloseHandle(m.m); CloseHandle(m.f); }
    maps_.clear(); t_.clear();
}

// ---------------- init ----------------
bool Gx::init(std::string& err) {
    if (ok_) return true;
    if (!cu().ok()) { err = "image/video/audio generation needs an NVIDIA GPU"; return false; }
    std::string log;
    void* mod = cu().load_module(gx_ptx(), log);
    if (!mod) { err = "GPU kernel compile failed: " + log; return false; }
    struct { void** f; const char* n; } fs[] = {{&f_lin_, "g_lin"}, {&f_nt_, "g_nt"}, {&f_nn_, "g_nn"}, {&f_conv_, "g_conv"}, {&f_gn_, "k_gn"},
        {&f_ln_, "k_ln"}, {&f_sm_, "k_softmax"}, {&f_act_, "k_act"}, {&f_axpy_, "k_axpy"}, {&f_chan_, "k_chanadd"}, {&f_geglu_, "k_geglu"},
        {&f_tr_, "k_transpose"}, {&f_up2_, "k_up2"}, {&f_embed_, "k_embed"}, {&f_h2f_, "k_h2f"}, {&f_im2col_, "k_im2col"}, {&f_tradd_, "k_tradd"},
        {&f_rmsrow_, "k_rmsrow"}, {&f_rmsc_, "k_rmsc"}, {&f_gadd_, "k_gadd"}, {&f_rope3_, "k_rope3"}, {&f_im2col3_, "k_im2col3"}, {&f_dupup_, "k_dupup"},
        {&f_tinter_, "k_tinter"}, {&f_f8_, "k_f8"}, {&f_mul_, "k_mul"}, {&f_q8row_, "k_q8row"}};
    for (auto& f : fs) if (!(*f.f = cu().function(mod, f.n))) { err = std::string("missing kernel ") + f.n; return false; }
    // Tensor-core GEMMs (RTX 30-series and newer). ITHACA_NO_TC=1 forces the plain kernels.
    std::string tlog;
    void* tm = getenv("ITHACA_NO_TC") ? nullptr : cu().load_module(gx_ptx_tc(), tlog);
    if (tm) {
        void *a = cu().function(tm, "t_lin"), *b = cu().function(tm, "t_nt"), *c = cu().function(tm, "t_nn"), *d = cu().function(tm, "t_conv");
        if (a && b && c && d) { f_lin_ = a; f_nt_ = b; f_nn_ = c; f_conv_ = d; tc_ = true; f_linf_ = cu().function(tm, "t_linf"); f_linq8_ = cu().function(tm, "t_linq8"); f_linh_ = cu().function(tm, "t_linh"); f_linq8h_ = cu().function(tm, "t_linq8h"); f_fa64_ = cu().function(tm, "t_fa64"); f_fa128_ = cu().function(tm, "t_fa128"); f_fa72_ = cu().function(tm, "t_fa72"); f_linq4_ = cu().function(tm, "t_linq4"); f_linq4m_ = cu().function(tm, "t_linq4m");
            const char* fk[5] = {"t_fkv64", "t_fkv80", "t_fkv96", "t_fkv128", "t_fkv256"}; for (int i = 0; i < 5; i++) f_fkv_[i] = cu().function(tm, fk[i]); }
    }
    ok_ = true;
    return true;
}

// ---------------- memory pool ----------------
GT Gx::alloc(size_t n) {
    size_t bytes = ((n * 4 + 255) / 256) * 256;
    GT t; t.n = n;
    auto it = pool_.lower_bound(bytes);
    if (it != pool_.end() && it->first <= bytes * 2) { t.p = it->second; live_[t.p] = it->first; pool_.erase(it); return t; }
    t.p = cu().alloc(bytes);
    if (!t.p) { trim(); t.p = cu().alloc(bytes); }
    if (t.p) live_[t.p] = bytes;
    return t;
}
void Gx::free(GT& t) {
    if (!t.p) return;
    auto it = live_.find(t.p);
    if (it != live_.end()) { pool_.emplace(it->second, t.p); live_.erase(it); }
    t = GT();
}
void Gx::trim() {
    cu().sync();
    for (auto& kv : pool_) cu().free(kv.second);
    pool_.clear();
}
size_t Gx::vram_free() { size_t f = 0, t = 0; cu().mem_info(f, t); return f; }

// ---------------- weights ----------------
static void to_f16(const STensor& t, std::vector<uint16_t>& out) {
    size_t n = (size_t)t.numel();
    out.resize(n);
    if (t.dtype == "F16") { memcpy(out.data(), t.data, n * 2); return; }
    for (size_t i = 0; i < n; i++) {
        float f;
        if (t.dtype == "BF16") { uint32_t u = (uint32_t)((const uint16_t*)t.data)[i] << 16; memcpy(&f, &u, 4); }
        else if (t.dtype == "F32") f = ((const float*)t.data)[i];
        else if (t.dtype == "F8_E4M3") {
            uint8_t b = t.data[i]; int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
            f = e ? std::ldexp(1.f + m / 8.f, e - 7) : std::ldexp(m / 8.f, -6);
            if (s) f = -f;
        } else f = 0;
        out[i] = f2h(f);
    }
}

void Gx::weights_begin(const SafeTensors* st, int64_t budget, bool q8) {
    weights_end();
    st_ = st; budget_ = budget; res_bytes_ = 0; streamed_ = 0; q8_ = q8 && f_linq8_ && f_q8row_; half_ = false;
}
// Device memory for a weight: resident while the budget allows, otherwise a slot in the streaming ring.
uint64_t Gx::place(size_t bytes, bool& resident) {
    resident = false;
    if (res_bytes_ + (int64_t)bytes <= budget_) {
        uint64_t p = cu().alloc(bytes);
        if (p) { resident = true; return p; }
        budget_ = res_bytes_;
    }
    size_t need = (bytes + 255) / 256 * 256;
    if (need > ring_cap_) {
        cu().sync();
        if (ring_) cu().free(ring_);
        ring_cap_ = std::max<size_t>(need * 2, 256u << 20);
        ring_ = cu().alloc(ring_cap_); ring_pos_ = 0;
        if (!ring_) { ring_cap_ = 0; return 0; }
    }
    if (ring_pos_ + need > ring_cap_) { cu().sync(); ring_pos_ = 0; }
    uint64_t p = ring_ + ring_pos_; ring_pos_ += need;
    streamed_ += bytes;
    return p;
}
uint64_t Gx::w16(const std::string& name) {
    auto it = res_.find(name);
    if (it != res_.end()) return it->second;
    const STensor* t = st_ ? st_->get(name) : nullptr;
    if (!t) { fprintf(stderr, "[gx] missing weight %s\n", name.c_str()); return 0; }
    size_t bytes = (size_t)t->numel() * 2;
    if (q8_ && t->shape.size() >= 2 && t->shape.back() % 32 == 0 && t->numel() >= 65536 && t->shape.size() == 2) {
        // int8 path: expand to fp16 in a scratch buffer (fp8 decode included), then quantize rows on the GPU
        const int64_t N = t->shape[0], K = t->shape[1];
        if ((size_t)bytes > qtmp_cap_) { cu().sync(); if (qtmp_) cu().free(qtmp_); qtmp_ = cu().alloc(bytes); qtmp_cap_ = qtmp_ ? bytes : 0; }
        else cu().sync();
        if (!qtmp_) return 0;
        if (t->dtype == "F8_E4M3") {
            // reuse the fp8 expander, pointed at the scratch buffer
            bool saved = q8_; q8_ = false;
            int64_t bb = budget_; budget_ = res_bytes_;   // force the expansion into the ring, then copy
            uint64_t tmp = w16_f8(name, *t);
            budget_ = bb; q8_ = saved;
            cu().dtod(qtmp_, tmp, bytes);
        } else if (t->dtype == "F16") cu().htod_async(qtmp_, t->data, bytes);
        else { to_f16(*t, conv_buf_); cu().htod_async(qtmp_, conv_buf_.data(), bytes); cu().sync(); }
        bool resident;
        uint64_t q = place((size_t)N * K, resident);
        GT sc = alloc((size_t)N * (K / 32));
        launch_ew(f_q8row_, (size_t)N * (K / 32), {&qtmp_, &q, &sc.p});
        if (resident) { res_[name] = q; res_bytes_ += (int64_t)N * K; }
        live_.erase(sc.p);    // scales live until weights_end, not in the activation pool
        q8scale_[q] = sc.p;
        return q;
    }
    if (t->dtype == "F8_E4M3") return w16_f8(name, *t);
    const void* src = t->data;
    if (t->dtype != "F16") { to_f16(*t, conv_buf_); src = conv_buf_.data(); }
    if (res_bytes_ + (int64_t)bytes <= budget_) {
        uint64_t p = cu().alloc(bytes);
        // Uploads go through the compute stream so they are strictly ordered with queued kernels.
        if (p && cu().htod_async(p, src, bytes)) {
            if (src == conv_buf_.data()) cu().sync();
            res_[name] = p; res_bytes_ += bytes; return p;
        }
        if (p) cu().free(p);
        budget_ = res_bytes_;  // VRAM is tighter than planned: stream from here on
    }
    // Streamed tier: copy into a ring buffer just before use. Wrapping waits for earlier kernels.
    size_t need = (bytes + 255) / 256 * 256;
    if (need > ring_cap_) {
        cu().sync();
        if (ring_) cu().free(ring_);
        ring_cap_ = std::max<size_t>(need * 2, 256u << 20);
        ring_ = cu().alloc(ring_cap_);
        ring_pos_ = 0;
        if (!ring_) { ring_cap_ = 0; return 0; }
    }
    if (ring_pos_ + need > ring_cap_) { cu().sync(); ring_pos_ = 0; }
    uint64_t p = ring_ + ring_pos_;
    ring_pos_ += need;
    cu().htod_async(p, src, bytes);
    if (src == conv_buf_.data()) cu().sync();  // the host buffer is reused for the next conversion
    streamed_ += bytes;
    return p;
}
// fp8 (e4m3, optional per-tensor "scale_weight") is uploaded as raw bytes and expanded to fp16 on the GPU.
uint64_t Gx::w16_f8(const std::string& name, const STensor& t) {
    size_t n = (size_t)t.numel(), bytes = n * 2;
    float scale = 1.f;
    std::string sn = name;
    if (sn.size() > 7 && sn.compare(sn.size() - 7, 7, ".weight") == 0) sn = sn.substr(0, sn.size() - 7) + ".scale_weight";
    if (const STensor* s = st_->get(sn)) {
        if (s->dtype == "F32") memcpy(&scale, s->data, 4);
        else if (s->dtype == "F16") scale = h2f(*(const uint16_t*)s->data);
    }
    uint64_t dst = 0;
    bool resident = false;
    if (res_bytes_ + (int64_t)bytes <= budget_) { dst = cu().alloc(bytes); resident = dst != 0; if (!dst) budget_ = res_bytes_; }
    if (!dst) {
        size_t need = (bytes + 255) / 256 * 256;
        if (need > ring_cap_) {
            cu().sync();
            if (ring_) cu().free(ring_);
            ring_cap_ = std::max<size_t>(need * 2, 256u << 20);
            ring_ = cu().alloc(ring_cap_); ring_pos_ = 0;
            if (!ring_) { ring_cap_ = 0; return 0; }
        }
        if (ring_pos_ + need > ring_cap_) { cu().sync(); ring_pos_ = 0; }
        dst = ring_ + ring_pos_; ring_pos_ += need;
        streamed_ += bytes;
    }
    if (n > f8_cap_) { cu().sync(); if (f8_buf_) cu().free(f8_buf_); f8_buf_ = cu().alloc(n); f8_cap_ = f8_buf_ ? n : 0; }
    else cu().sync();   // previous conversion must be done before the staging buffer is reused
    cu().htod_async(f8_buf_, t.data, n);
    unsigned un = (unsigned)n;
    std::vector<void*> a = {&f8_buf_, &dst, &scale};
    launch_ew(f_f8_, n, a);
    (void)un;
    if (resident) { res_[name] = dst; res_bytes_ += bytes; }
    return dst;
}

GT Gx::wf32(const std::string& name) {
    auto it = f32_.find(name);
    if (it != f32_.end()) return it->second;
    const STensor* t = st_ ? st_->get(name) : nullptr;
    if (!t) { fprintf(stderr, "[gx] missing param %s\n", name.c_str()); return GT(); }
    std::vector<uint16_t> h;
    to_f16(*t, h);
    std::vector<float> f(h.size());
    for (size_t i = 0; i < h.size(); i++) f[i] = h2f(h[i]);
    if (t->dtype == "F32") memcpy(f.data(), t->data, f.size() * 4);
    GT g; g.n = f.size(); g.p = cu().alloc(f.size() * 4 + 4);
    cu().htod_async(g.p, f.data(), f.size() * 4);
    cu().sync();   // f is a temporary
    f32_[name] = g;
    return g;
}
void Gx::weights_end() {
    cu().sync();
    for (auto& kv : q8scale_) cu().free(kv.second);
    q8scale_.clear();
    for (auto& kv : res_) cu().free(kv.second);
    for (auto& kv : f32_) cu().free(kv.second.p);
    res_.clear(); f32_.clear();
    if (ring_) cu().free(ring_);
    ring_ = 0; ring_cap_ = ring_pos_ = 0;
    res_bytes_ = 0;
    st_ = nullptr;
}

// ---------------- ops ----------------
// ITHACA_GXPROF=1: time every kernel (synchronizes after each launch).
static bool g_gxprof = getenv("ITHACA_GXPROF") != nullptr;
static std::map<void*, std::pair<double, long>> g_gxt;
static void prof_launch(void* fn, unsigned gx_, unsigned gy, unsigned gz, unsigned b, unsigned sh, void** a) {
    double t0 = g_gxprof ? (cu().sync(), now_sec()) : 0;
    cu().launch_raw(fn, gx_, gy, gz, b, sh, a);
    if (g_gxprof) { cu().sync(); auto& e = g_gxt[fn]; e.first += now_sec() - t0; e.second++; }
}
void Gx::print_profile() {
    std::map<void*, const char*> nm = {{f_lin_, "linear"}, {f_nt_, "attn QK"}, {f_nn_, "attn PV"}, {f_conv_, "conv"}, {f_gn_, "groupnorm"}, {f_ln_, "layernorm"},
        {f_linf_, "linear*"}, {f_im2col_, "im2col"}, {f_tradd_, "tr-add"}, {f_sm_, "softmax"}, {f_act_, "act"}, {f_axpy_, "add"}, {f_chan_, "chanadd"}, {f_geglu_, "geglu"}, {f_tr_, "transpose"}, {f_up2_, "upsample"}, {f_embed_, "embed"}, {f_linh_, "linear16"}, {f_fa64_, "flash64"}, {f_fa72_, "flash72"}, {f_fa128_, "flash128"}};
    for (auto& kv : g_gxt) printf("    %-10s %8.1f ms  (%ld calls)\n", nm.count(kv.first) ? nm[kv.first] : "?", kv.second.first * 1000, kv.second.second);
    g_gxt.clear();
}
void Gx::launch_ew(void* fn, size_t n, std::vector<void*> args) {
    unsigned un = (unsigned)n;
    args.push_back(&un);
    unsigned blocks = (unsigned)((n + 255) / 256);
    unsigned gx_ = std::min(blocks, 65535u), gy = (blocks + gx_ - 1) / gx_;
    prof_launch(fn, gx_, gy, 1, 256, 0, args.data());
}
void Gx::gemm(void* fn, uint64_t A, uint64_t B, uint64_t C, uint64_t bias, unsigned M, unsigned N, unsigned K, unsigned lda, unsigned ldb, unsigned ldc,
              unsigned sA, unsigned sB, unsigned sC, float alpha, float beta, unsigned flags, unsigned batch,
              unsigned H, unsigned W, unsigned kh, unsigned kw, unsigned stride, unsigned pad, unsigned Wo, unsigned padw) {
    void* a[] = {&A, &B, &C, &bias, &M, &N, &K, &lda, &ldb, &ldc, &sA, &sB, &sC, &alpha, &beta, &flags, &H, &W, &kh, &kw, &stride, &pad, &Wo, &padw};
    prof_launch(fn, (N + 63) / 64, (M + 63) / 64, batch, tc_ ? 128 : 256, 0, a);
}
void Gx::linear(const GT& x, int M, int K, uint64_t W, int N, const GT* bias, GT& y, float beta) {
    auto q = q8scale_.find(W);
    if (q != q8scale_.end()) {
        unsigned M_ = M, N_ = N, K_ = K, z = 0, one = 1, flags = bias && bias->p ? 1 : 0;
        uint64_t A = x.p, Y = y.p, B = bias ? bias->p : 0, S = q->second;
        float alpha = 1.f, b = beta;
        void* a[] = {&A, &W, &Y, &B, &M_, &N_, &K_, &K_, &K_, &N_, &z, &z, &z, &alpha, &b, &flags, &z, &z, &one, &one, &one, &z, &one, &z, &S};
        prof_launch(half_ && f_linq8h_ ? f_linq8h_ : f_linq8_, (N + 63) / 64, (M + 63) / 64, 1, 128, 0, a);
        return;
    }
    if (f_linf_ && K % 32 && K > 64 && !getenv("ITHACA_NO_FASTLIN")) {
        // ragged K (e.g. 4304): tensor-core kernel on the first K/32*32 columns, generic kernel adds the tail
        unsigned K1 = K / 32 * 32;
        gemm(half_ ? f_linh_ : f_linf_, x.p, W, y.p, bias ? bias->p : 0, M, N, K1, K, K, N, 0, 0, 0, 1.f, beta, bias && bias->p ? 1 : 0, 1);
        gemm(f_lin_, x.p + K1 * 4ull, W + K1 * 2ull, y.p, 0, M, N, K - K1, K, K, N, 0, 0, 0, 1.f, 1.f, 0, 1);
        return;
    }
    gemm(f_linf_ && K % 32 == 0 && !getenv("ITHACA_NO_FASTLIN") ? (half_ ? f_linh_ : f_linf_) : f_lin_, x.p, W, y.p, bias ? bias->p : 0, M, N, K, K, K, N, 0, 0, 0, 1.f, beta, bias && bias->p ? 1 : 0, 1);
}
bool Gx::attn_prefill(uint64_t Q, uint64_t Kc, uint64_t Vc, uint64_t O, int n, int p0, int ldq, int kvd, int heads, int grp, int hd,
                      float scale, int win, int nc) {
    void* f = hd == 64 ? f_fkv_[0] : hd == 80 ? f_fkv_[1] : hd == 96 ? f_fkv_[2] : hd == 128 ? f_fkv_[3] : hd == 256 ? f_fkv_[4] : nullptr;
    if (!f || getenv("ITHACA_NO_FLASH")) return false;
    const unsigned DV = hd == 256 ? 128 : hd;
    float sc = scale * 1.4426950409f;
    unsigned T = n, P0 = p0, L = ldq, KV = kvd, G = grp, W = win, NC = nc;
    for (unsigned dv0 = 0; dv0 < (unsigned)hd; dv0 += DV) {
        void* a[] = {&Q, &Kc, &Vc, &O, &T, &P0, &L, &KV, &G, &sc, &W, &NC, &dv0};
        prof_launch(f, (n + 127) / 128, heads, 1, 256, 0, a);
    }
    return true;
}
bool Gx::linear_quant(uint64_t X, int M, int K, uint64_t S, uint64_t Q, int kind, int N, uint64_t Y) {
    void* fn = kind == 1 ? f_linq8_ : kind == 0 ? f_linq4_ : f_linq4m_;
    if (!fn || K % 32) return false;
    unsigned M_ = M, N_ = N, K_ = K, ldb = kind == 1 ? K : K / 2, z = 0, one = 1, flags = 0;
    float alpha = 1.f, b = 0.f; uint64_t B0 = 0;
    void* a[] = {&X, &Q, &Y, &B0, &M_, &N_, &K_, &K_, &ldb, &N_, &z, &z, &z, &alpha, &b, &flags, &z, &z, &one, &one, &one, &z, &one, &z, &S};
    prof_launch(fn, (N + 63) / 64, (M + 63) / 64, 1, 128, 0, a);
    return true;
}
void Gx::conv(const GT& x, int Cin, int H, int W, uint64_t Wt, int Cout, int kh, int kw, int stride, int pad, const GT* bias, GT& y, float beta, int padw) {
    if (padw < 0) padw = pad;
    int Ho = (H + 2 * pad - kh) / stride + 1, Wo = (W + 2 * padw - kw) / stride + 1;
    int K = Cin * kh * kw;
    if (f_linf_ && K % 32 == 0 && !getenv("ITHACA_NO_FASTCONV")) {
        // im2col in pixel chunks (<= 64 MB of columns), tensor-core linear, transpose back into NCHW.
        const int P = Ho * Wo;
        int chunk = (int)std::max<size_t>(64, std::min<size_t>(P, (16u << 20) / (size_t)K));
        chunk = chunk / 64 * 64 ? chunk / 64 * 64 : chunk;
        GT col = alloc((size_t)chunk * K), tmp = alloc((size_t)chunk * Cout);
        for (int p0 = 0; p0 < P; p0 += chunk) {
            int n = std::min(chunk, P - p0);
            unsigned h = H, w = W, khu = kh, kwu = kw, s = stride, pd = pad, wo = Wo, ku = K, pp = p0, pw = padw;
            launch_ew(f_im2col_, (size_t)n * K, {(void*)&x.p, &col.p, &h, &w, &khu, &kwu, &s, &pd, &wo, &ku, &pp, &pw});
            gemm(half_ ? f_linh_ : f_linf_, col.p, Wt, tmp.p, bias ? bias->p : 0, n, Cout, K, K, K, Cout, 0, 0, 0, 1.f, 0.f, bias && bias->p ? 1 : 0, 1);
            unsigned r = n, c = Cout, ldo = P, r0 = p0; float b = beta;
            launch_ew(f_tradd_, (size_t)n * Cout, {&tmp.p, &y.p, &r, &c, &ldo, &r0, &b});
        }
        free(col); free(tmp);
        return;
    }
    gemm(f_conv_, Wt, x.p, y.p, bias ? bias->p : 0, Cout, Ho * Wo, K, K, 0, Ho * Wo, 0, 0, 0, 1.f, beta, (bias && bias->p ? 1 : 0) | 2, 1,
         H, W, kh, kw, stride, pad, Wo, padw);
}
void Gx::gemm_nt(const GT& A, int M, int K, int lda, const GT& B, int N, int ldb, GT& C, int ldc, float alpha, int batch, size_t sA, size_t sB, size_t sC,
                 size_t oA, size_t oB, size_t oC) {
    gemm(f_nt_, A.p + oA * 4, B.p + oB * 4, C.p + oC * 4, 0, M, N, K, lda, ldb, ldc, (unsigned)sA, (unsigned)sB, (unsigned)sC, alpha, 0.f, 0, batch);
}
void Gx::gemm_nn(const GT& A, int M, int K, int lda, const GT& B, int N, int ldb, GT& C, int ldc, float alpha, int batch, size_t sA, size_t sB, size_t sC,
                 size_t oA, size_t oB, size_t oC) {
    gemm(f_nn_, A.p + oA * 4, B.p + oB * 4, C.p + oC * 4, 0, M, N, K, lda, ldb, ldc, (unsigned)sA, (unsigned)sB, (unsigned)sC, alpha, 0.f, 0, batch);
}
void Gx::groupnorm(const GT& x, int C, int HW, int groups, const GT& g, const GT& b, float eps, bool silu, GT& y) {
    unsigned cpg = C / groups, hw = HW, act = silu ? 1 : 0;
    void* a[] = {(void*)&x.p, (void*)&g.p, (void*)&b.p, &y.p, &cpg, &hw, &eps, &act};
    prof_launch(f_gn_, groups, 1, 1, 256, 0, a);
}
void Gx::layernorm(const GT& x, int rows, int D, const GT& g, const GT& b, float eps, GT& y) {
    unsigned d = D;
    void* a[] = {(void*)&x.p, (void*)&g.p, (void*)&b.p, &y.p, &d, &eps};
    prof_launch(f_ln_, rows, 1, 1, 256, 0, a);
}
void Gx::softmax(GT& x, int rows, int cols, float scale, bool causal, int T) {
    unsigned c = cols, cz = causal ? 1 : 0, t = T;
    void* a[] = {&x.p, &c, &scale, &cz, &t};
    // grid.x is limited to 2^31-1 on modern GPUs; rows here stay far below that
    prof_launch(f_sm_, rows, 1, 1, 256, 0, a);
}
void Gx::act(GT& x, size_t n, Act m) { unsigned mode = m; launch_ew(f_act_, n, {&x.p, &mode}); }
void Gx::axpy(GT& y, const GT& x, size_t n, float a) { launch_ew(f_axpy_, n, {&y.p, (void*)&x.p, &a}); }
void Gx::chanadd(GT& y, const GT& b, int C, int HW) { unsigned hw = HW; launch_ew(f_chan_, (size_t)C * HW, {&y.p, (void*)&b.p, &hw}); }
void Gx::geglu(const GT& in, int T, int F, GT& out) { unsigned f = F; launch_ew(f_geglu_, (size_t)T * F, {(void*)&in.p, &out.p, &f}); }
void Gx::transpose(const GT& in, int R, int C, GT& out) { unsigned r = R, c = C; launch_ew(f_tr_, (size_t)R * C, {(void*)&in.p, &out.p, &r, &c}); }
void Gx::up2(const GT& in, int C, int H, int W, GT& out) { unsigned h = H, w = W; launch_ew(f_up2_, (size_t)C * H * W * 4, {(void*)&in.p, &out.p, &h, &w}); }
void Gx::embed(uint64_t ids, uint64_t table, int T, int D, GT& out) { unsigned d = D; launch_ew(f_embed_, (size_t)T * D, {&ids, &table, &out.p, &d}); }
void Gx::copy(GT& dst, const GT& src, size_t n, size_t doff, size_t soff) { cu().dtod(dst.p + doff * 4, src.p + soff * 4, n * 4); }
void Gx::rmsrow(const GT& x, int rows, int D, const GT& w, float eps, GT& y) {
    unsigned d = D;
    void* a[] = {(void*)&x.p, (void*)&w.p, &y.p, &d, &eps};
    prof_launch(f_rmsrow_, rows, 1, 1, 256, 0, a);
}
void Gx::rmsc(const GT& x, int C, size_t N, const GT& g, bool silu, GT& y) {
    unsigned c = C, nn = (unsigned)N, act = silu ? 1 : 0;
    launch_ew(f_rmsc_, N, {(void*)&x.p, (void*)&g.p, &y.p, &c, &nn, &act});
}
void Gx::gadd(GT& x, const GT& y, const GT& gate, int C, size_t n) { unsigned c = C; launch_ew(f_gadd_, n, {&x.p, (void*)&y.p, (void*)&gate.p, &c}); }
void Gx::mul(GT& y, const GT& x, size_t n) { launch_ew(f_mul_, n, {&y.p, (void*)&x.p}); }
void Gx::rope3(GT& x, const GT& cs, int L, int heads, int hd) {
    unsigned h = heads, half = hd / 2;
    launch_ew(f_rope3_, (size_t)L * heads * half, {&x.p, (void*)&cs.p, &h, &half});
}
void Gx::conv3d(const GT& x, int Ci, int Tin, int H, int W, uint64_t Wt, int Co, int KT, int KS, const GT* bias, GT& y, float beta) {
    const int Tout = Tin - KT + 1, P = H * W, K = Ci * KT * KS * KS;
    const bool fast = f_linf_ && K % 32 == 0;
    int chunk = (int)std::max<size_t>(64, std::min<size_t>(P, (16u << 20) / (size_t)K));
    GT col = alloc((size_t)chunk * K), tmp = alloc((size_t)chunk * Co);
    for (int t = 0; t < Tout; t++)
        for (int p0 = 0; p0 < P; p0 += chunk) {
            int n = std::min(chunk, P - p0);
            unsigned tin = Tin, h = H, w = W, kt = KT, ks = KS, pad = KS / 2, t0 = t, ku = K, pp = p0;
            launch_ew(f_im2col3_, (size_t)n * K, {(void*)&x.p, &col.p, &tin, &h, &w, &kt, &ks, &ks, &pad, &t0, &ku, &pp});
            gemm(fast ? (half_ ? f_linh_ : f_linf_) : f_lin_, col.p, Wt, tmp.p, bias ? bias->p : 0, n, Co, K, K, K, Co, 0, 0, 0, 1.f, 0.f, bias && bias->p ? 1 : 0, 1);
            unsigned r = n, c = Co, ldo = (unsigned)(Tout * P), r0 = (unsigned)(t * P + p0); float b = beta;
            launch_ew(f_tradd_, (size_t)n * Co, {&tmp.p, &y.p, &r, &c, &ldo, &r0, &b});
        }
    free(col); free(tmp);
}
void Gx::dupup(const GT& x, int Cin, int Cout, int T, int H, int W, int ft, int fs, int drop, GT& out) {
    unsigned rep = (unsigned)(Cout * ft * fs * fs / Cin), t = T, h = H, w = W, uft = ft, ufs = fs, d = drop;
    size_t n = (size_t)Cout * (T * ft - drop) * H * fs * W * fs;
    launch_ew(f_dupup_, n, {(void*)&x.p, &out.p, &rep, &t, &h, &w, &uft, &ufs, &d});
}
void Gx::tinter(const GT& x, int C, int T, int HW, GT& out) {
    unsigned c = C, t = T, hw = HW;
    launch_ew(f_tinter_, (size_t)C * 2 * T * HW, {(void*)&x.p, &out.p, &c, &t, &hw});
}

// Ordered on the compute stream: pooled buffers may still be in use by queued kernels.
void Gx::upload(GT& t, const float* h, size_t n) { cu().htod_async(t.p, h, n * 4); }
void Gx::download(float* h, const GT& t, size_t n) { cu().dtoh(h, t.p, n * 4); }
void Gx::sync() { cu().sync(); }

void Gx::attention(const GT& q, const GT& k, const GT& v, int T, int S, int heads, int d, GT& out, bool causal) {
    // Scores for a group of heads at a time, capped at ~256 MB, so long sequences fit in memory.
    const int C = heads * d;
    void* fa = d == 64 ? f_fa64_ : d == 128 ? f_fa128_ : d == 72 ? f_fa72_ : nullptr;
    if (fa && !causal && C % 4 == 0 && !getenv("ITHACA_NO_FLASH")) {
        // fused flash attention: no score matrix, one pass over the keys
        float sc2 = 1.4426950409f / std::sqrt((float)d);
        void* a[] = {(void*)&q.p, (void*)&k.p, (void*)&v.p, (void*)&out.p, &T, &S, (void*)&C, &sc2};
        prof_launch(fa, (T + 127) / 128, heads, 1, 256, 0, a);
        return;
    }
    size_t per_head = (size_t)T * S;
    int hg = (int)std::max<size_t>(1, std::min<size_t>(heads, (64ull << 20) / std::max<size_t>(1, per_head)));
    int tq = T;
    if (per_head > (64ull << 20)) { hg = 1; tq = std::max(1, (int)((64ull << 20) / S)); }
    GT sc = alloc((size_t)hg * tq * S);
    float scale = 1.f / std::sqrt((float)d);
    for (int h0 = 0; h0 < heads; h0 += hg) {
        int nh = std::min(hg, heads - h0);
        for (int t0 = 0; t0 < T; t0 += tq) {
            int nt = std::min(tq, T - t0);
            // scores[h][t][s] = q[t0+t][h*d:] . k[s][h*d:]
            gemm_nt(q, nt, d, C, k, S, C, sc, S, scale, nh, d, d, (size_t)nt * S, (size_t)t0 * C + h0 * d, (size_t)h0 * d, 0);
            if (causal) {
                // causal rows need their absolute position: only used by the text encoders, where T == S and tq == T
                softmax(sc, nh * nt, S, 1.f, true, nt);
            } else softmax(sc, nh * nt, S, 1.f);
            gemm_nn(sc, nt, S, S, v, d, C, out, C, 1.f, nh, (size_t)nt * S, d, d, 0, (size_t)h0 * d, (size_t)t0 * C + h0 * d);
        }
    }
    free(sc);
}

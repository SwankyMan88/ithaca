#include "tts.h"
#include "quant.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <random>

TtsEngine& tts_engine() { static TtsEngine e; return e; }
std::map<std::string, std::vector<float>>* g_tts_dbg = nullptr;   // developer hook for the self-test
static void dbg(const char* n, const GT& t, size_t cnt) {
    if (!g_tts_dbg) return;
    std::vector<float> v(cnt); gx().download(v.data(), t, cnt); (*g_tts_dbg)[n] = v;
}
static const double kPi = 3.14159265358979323846;

// =============================================================================================
// WAV I/O and resampling
// =============================================================================================
bool read_wav(const std::wstring& path, std::vector<float>& mono, int& sr, std::string& err) {
    FILE* f = _wfopen(path.c_str(), L"rb");
    if (!f) { err = "cannot open voice clip"; return false; }
    std::vector<uint8_t> d;
    uint8_t b[65536]; size_t r;
    while ((r = fread(b, 1, sizeof b, f)) > 0) d.insert(d.end(), b, b + r);
    fclose(f);
    if (d.size() < 44 || memcmp(d.data(), "RIFF", 4) || memcmp(d.data() + 8, "WAVE", 4)) { err = "voice clip must be a WAV file"; return false; }
    int fmt = 0, ch = 0, bits = 0;
    const uint8_t* data = nullptr; size_t dlen = 0;
    for (size_t o = 12; o + 8 <= d.size();) {
        uint32_t len; memcpy(&len, &d[o + 4], 4);
        if (!memcmp(&d[o], "fmt ", 4) && len >= 16) {
            fmt = d[o + 8] | (d[o + 9] << 8); ch = d[o + 10] | (d[o + 11] << 8);
            memcpy(&sr, &d[o + 12], 4); bits = d[o + 22] | (d[o + 23] << 8);
            if (fmt == 0xFFFE && len >= 26) fmt = d[o + 32] | (d[o + 33] << 8);   // extensible: sub-format
        } else if (!memcmp(&d[o], "data", 4)) { data = &d[o + 8]; dlen = std::min<size_t>(len, d.size() - o - 8); }
        o += 8 + len + (len & 1);
    }
    if (!data || !ch || !bits) { err = "unsupported WAV file"; return false; }
    int bps = bits / 8; size_t frames = dlen / (bps * ch);
    mono.assign(frames, 0.f);
    for (size_t i = 0; i < frames; i++) {
        float s = 0;
        for (int c = 0; c < ch; c++) {
            const uint8_t* p = data + (i * ch + c) * bps;
            float v = 0;
            if (fmt == 3 && bits == 32) memcpy(&v, p, 4);
            else if (bits == 16) v = (int16_t)(p[0] | (p[1] << 8)) / 32768.f;
            else if (bits == 24) v = (int32_t)((p[0] << 8) | (p[1] << 16) | (p[2] << 24)) / 2147483648.f;
            else if (bits == 32) { int32_t q; memcpy(&q, p, 4); v = q / 2147483648.f; }
            else if (bits == 8) v = (p[0] - 128) / 128.f;
            s += v;
        }
        mono[i] = s / ch;
    }
    return true;
}

bool write_wav(const std::wstring& path, const std::vector<float>& w, int sr) {
    FILE* f = _wfopen(path.c_str(), L"wb");
    if (!f) return false;
    uint32_t n = (uint32_t)w.size(), bytes = n * 2, riff = 36 + bytes, sr4 = sr, br = sr * 2, fmtlen = 16;
    uint16_t pcm = 1, ch = 1, align = 2, bits = 16;
    fwrite("RIFF", 1, 4, f); fwrite(&riff, 4, 1, f); fwrite("WAVEfmt ", 1, 8, f); fwrite(&fmtlen, 4, 1, f);
    fwrite(&pcm, 2, 1, f); fwrite(&ch, 2, 1, f); fwrite(&sr4, 4, 1, f); fwrite(&br, 4, 1, f); fwrite(&align, 2, 1, f); fwrite(&bits, 2, 1, f);
    fwrite("data", 1, 4, f); fwrite(&bytes, 4, 1, f);
    for (float v : w) { int16_t s = (int16_t)std::lrint(std::max(-1.f, std::min(1.f, v)) * 32767.f); fwrite(&s, 2, 1, f); }
    fclose(f);
    return true;
}

// Band-limited (Hann-windowed sinc) resampler.
std::vector<float> resample(const std::vector<float>& x, int from, int to) {
    if (from == to || x.empty()) return x;
    const double ratio = (double)to / from, cutoff = std::min(1.0, ratio) * 0.99;
    const int width = 16;
    size_t n = (size_t)std::ceil(x.size() * ratio);
    std::vector<float> y(n);
    for (size_t i = 0; i < n; i++) {
        double c = i / ratio;
        long lo = (long)std::floor(c) - width / std::min(1.0, ratio), hi = (long)std::floor(c) + width / std::min(1.0, ratio) + 1;
        double s = 0, ws = 0;
        for (long k = lo; k <= hi; k++) {
            if (k < 0 || k >= (long)x.size()) continue;
            double t = (c - k) * cutoff, span = width / cutoff * cutoff;
            if (std::fabs(c - k) > width / std::min(1.0, ratio)) continue;
            double sinc = t == 0 ? 1 : std::sin(kPi * t) / (kPi * t);
            double win = 0.5 + 0.5 * std::cos(kPi * (c - k) / (width / std::min(1.0, ratio)));
            (void)span;
            s += x[k] * sinc * win * cutoff; ws += sinc * win * cutoff;
        }
        y[i] = (float)s;
    }
    return y;
}

// =============================================================================================
// FFT / STFT / mel
// =============================================================================================
static void fft(std::vector<std::complex<double>>& a, bool inv) {
    const size_t n = a.size();
    for (size_t i = 1, j = 0; i < n; i++) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        double ang = 2 * kPi / len * (inv ? 1 : -1);
        std::complex<double> wl(std::cos(ang), std::sin(ang));
        for (size_t i = 0; i < n; i += len) {
            std::complex<double> w(1);
            for (size_t j = 0; j < len / 2; j++) {
                auto u = a[i + j], v = a[i + j + len / 2] * w;
                a[i + j] = u + v; a[i + j + len / 2] = u - v;
                w *= wl;
            }
        }
    }
    if (inv) for (auto& v : a) v /= (double)n;
}

void TtsEngine::mel(const std::vector<float>& wave, std::vector<float>& out, int& frames) {
    const int nfft = 1024, hop = 256, nm = 100, nb = nfft / 2 + 1, sr = 24000;
    const int L = (int)wave.size(), pad = nfft / 2;
    frames = 1 + L / hop;
    auto at = [&](long i) -> float {   // reflect padding
        if (i < 0) i = -i;
        if (i >= L) i = 2 * (L - 1) - i;
        return i >= 0 && i < L ? wave[i] : 0.f;
    };
    static std::vector<double> fb;
    if (fb.empty()) {   // HTK mel filterbank, no normalization
        auto h2m = [](double f) { return 2595.0 * std::log10(1.0 + f / 700.0); };
        auto m2h = [](double m) { return 700.0 * (std::pow(10.0, m / 2595.0) - 1.0); };
        std::vector<double> fp(nm + 2), fd(nm + 1);
        for (int i = 0; i < nm + 2; i++) fp[i] = m2h(h2m(0) + (h2m(sr / 2.0) - h2m(0)) * i / (nm + 1));
        for (int i = 0; i <= nm; i++) fd[i] = fp[i + 1] - fp[i];
        fb.assign((size_t)nb * nm, 0);
        for (int k = 0; k < nb; k++) {
            double f = (double)sr / 2 * k / (nb - 1);
            for (int m = 0; m < nm; m++) {
                double down = -(fp[m] - f) / fd[m], up = (fp[m + 2] - f) / fd[m + 1];
                fb[(size_t)k * nm + m] = std::max(0.0, std::min(down, up));
            }
        }
    }
    out.assign((size_t)nm * frames, 0.f);
    std::vector<std::complex<double>> a(nfft);
    std::vector<double> mag(nb);
    for (int t = 0; t < frames; t++) {
        for (int n = 0; n < nfft; n++) {
            double w = 0.5 - 0.5 * std::cos(2 * kPi * n / nfft);
            a[n] = at((long)t * hop + n - pad) * w;
        }
        fft(a, false);
        for (int k = 0; k < nb; k++) mag[k] = std::abs(a[k]);
        for (int m = 0; m < nm; m++) {
            double s = 0;
            for (int k = 0; k < nb; k++) s += fb[(size_t)k * nm + m] * mag[k];
            out[(size_t)m * frames + t] = (float)std::log(std::max(s, 1e-5));
        }
    }
}

// =============================================================================================
// Model
// =============================================================================================
bool TtsEngine::load(const std::wstring& model, std::string& err) {
    unload();
    if (!gx().init(err)) return false;
    if (!f5_.add(model, err, "ema_model.transformer.")) return false;
    if (!f5_.has("transformer_blocks.0.attn_norm.linear.weight") || !f5_.has("text_embed.text_embed.weight")) {
        err = "not an F5-TTS model (expected F5TTS v1 safetensors)"; unload(); return false;
    }
    while (f5_.has("transformer_blocks." + std::to_string(depth_) + ".attn.to_q.weight")) depth_++;
    dim_ = (int)f5_.get("transformer_blocks.0.attn.to_q.weight")->shape[0];
    heads_ = dim_ / 64;
    text_dim_ = (int)f5_.get("text_embed.text_embed.weight")->shape[1];
    std::wstring dir = model.substr(0, model.find_last_of(L"\\/")), up = dir.substr(0, dir.find_last_of(L"\\/"));
    // vocab.txt next to the model
    FILE* f = _wfopen((dir + L"\\vocab.txt").c_str(), L"rb");
    if (!f) { err = "vocab.txt not found next to the F5-TTS model"; unload(); return false; }
    std::string s; char b[65536]; size_t r;
    while ((r = fread(b, 1, sizeof b, f)) > 0) s.append(b, r);
    fclose(f);
    int id = 0;
    for (size_t p = 0; p <= s.size();) {
        size_t e = s.find('\n', p); if (e == std::string::npos) e = s.size();
        std::string line = s.substr(p, e - p);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (p < s.size()) vocab_.emplace(line, id++);
        p = e + 1;
    }
    // Vocos vocoder: a "vocos" folder next to the model or one level up
    std::wstring voc;
    for (auto d : {dir + L"\\vocos", up + L"\\vocos", dir})
        for (auto n : {L"\\pytorch_model.bin", L"\\vocos.safetensors", L"\\model.safetensors"})
            if (voc.empty() && GetFileAttributesW((d + n).c_str()) != INVALID_FILE_ATTRIBUTES) voc = d + n;
    if (voc.empty()) { err = "Vocos vocoder not found (put the vocos folder next to the model)"; unload(); return false; }
    if (!voc_.add(voc, err)) { unload(); return false; }
    if (!voc_.has("backbone.embed.weight")) { err = "vocoder file is not Vocos"; unload(); return false; }
    path = model;
    loaded_ = true;
    return true;
}
void TtsEngine::unload() { f5_.close(); voc_.close(); vocab_.clear(); depth_ = 0; loaded_ = false; }
void TtsEngine::dit_begin() { gx().weights_begin(&f5_, 1ll << 40); }
void TtsEngine::voc_begin() { gx().weights_begin(&voc_, 1ll << 40); }
void TtsEngine::end() { gx().weights_end(); }

std::vector<int> TtsEngine::text_ids(const std::string& text) {
    std::vector<int> ids;
    for (size_t i = 0; i < text.size();) {
        unsigned char c = text[i];
        int L = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : 4;
        auto it = vocab_.find(text.substr(i, L));
        ids.push_back(it == vocab_.end() ? 0 : it->second);
        i += L;
    }
    return ids;
}

static std::vector<float> hostw(const SafeTensors& st, const std::string& n) {
    const STensor* t = st.get(n);
    std::vector<float> v(t ? (size_t)t->numel() : 0);
    for (size_t i = 0; i < v.size(); i++) v[i] = t->dtype == "F32" ? ((const float*)t->data)[i] : h2f(((const uint16_t*)t->data)[i]);
    return v;
}

// Character embedding + sinusoidal positions + ConvNeXt-V2 blocks (pads stay zero). Weights from f5_ (dit_begin first).
void TtsEngine::text_embed(const std::vector<int>& ids, int N, std::vector<float>& out) {
    Gx& g = gx();
    const int D = text_dim_;
    std::vector<float> E = hostw(f5_, "text_embed.text_embed.weight");
    std::vector<int> tok(N);
    for (int p = 0; p < N; p++) tok[p] = p < (int)ids.size() ? ids[p] + 1 : 0;
    std::vector<float> x((size_t)N * D);
    for (int p = 0; p < N; p++) {
        if (!tok[p]) continue;
        for (int c = 0; c < D; c++) {
            int i = c % (D / 2);
            double fr = 1.0 / std::pow(10000.0, 2.0 * i / D);
            x[(size_t)p * D + c] = E[(size_t)tok[p] * D + c] + (float)(c < D / 2 ? std::cos(p * fr) : std::sin(p * fr));
        }
    }
    const int F = (int)f5_.get("text_embed.text_blocks.0.pwconv1.weight")->shape[0];
    GT xg = g.alloc((size_t)N * D), hg = g.alloc((size_t)N * F), yg = g.alloc((size_t)N * D);
    std::vector<float> h((size_t)N * F), y((size_t)N * D);
    for (int b = 0;; b++) {
        std::string p = "text_embed.text_blocks." + std::to_string(b) + ".";
        if (!f5_.has(p + "dwconv.weight")) break;
        std::vector<float> dw = hostw(f5_, p + "dwconv.weight"), db = hostw(f5_, p + "dwconv.bias"), gm = hostw(f5_, p + "grn.gamma"), bt = hostw(f5_, p + "grn.beta");
        for (int t = 0; t < N; t++)   // depthwise conv, kernel 7
            for (int c = 0; c < D; c++) {
                float s = db[c];
                for (int k = 0; k < 7; k++) { int q = t + k - 3; if (q >= 0 && q < N) s += dw[(size_t)c * 7 + k] * x[(size_t)q * D + c]; }
                y[(size_t)t * D + c] = s;
            }
        g.upload(yg, y.data(), y.size());
        g.layernorm(yg, N, D, g.wf32(p + "norm.weight"), g.wf32(p + "norm.bias"), 1e-6f, yg);
        GT b1 = g.wf32(p + "pwconv1.bias");
        g.linear(yg, N, D, g.w16(p + "pwconv1.weight"), F, &b1, hg);
        g.act(hg, (size_t)N * F, Gx::GELU);
        g.download(h.data(), hg, h.size());
        std::vector<double> gx2(F, 0.0);   // GRN: L2 over the sequence per channel, normalized by the channel mean
        for (int t = 0; t < N; t++) for (int c = 0; c < F; c++) { double v = h[(size_t)t * F + c]; gx2[c] += v * v; }
        double mean = 0; for (auto& v : gx2) { v = std::sqrt(v); mean += v; } mean /= F;
        for (int t = 0; t < N; t++) for (int c = 0; c < F; c++) { float& v = h[(size_t)t * F + c]; v = (float)(gm[c] * (v * gx2[c] / (mean + 1e-6)) + bt[c] + v); }
        g.upload(hg, h.data(), h.size());
        GT b2 = g.wf32(p + "pwconv2.bias");
        g.linear(hg, N, F, g.w16(p + "pwconv2.weight"), D, &b2, yg);
        g.download(y.data(), yg, y.size());
        for (int t = 0; t < N; t++) for (int c = 0; c < D; c++) x[(size_t)t * D + c] = tok[t] ? x[(size_t)t * D + c] + y[(size_t)t * D + c] : 0.f;
    }
    g.free(xg); g.free(hg); g.free(yg);
    out = x;
}

void TtsEngine::dit(const std::vector<float>& xh, const std::vector<float>& cond, const std::vector<float>& temb, int N, float t, std::vector<float>& out) {
    Gx& g = gx();
    const int D = dim_, H = heads_, hd = 64, in = 100 * 2 + text_dim_;
    std::vector<float> cat((size_t)N * in);
    for (int p = 0; p < N; p++) {
        memcpy(&cat[(size_t)p * in], &xh[(size_t)p * 100], 400);
        memcpy(&cat[(size_t)p * in + 100], &cond[(size_t)p * 100], 400);
        memcpy(&cat[(size_t)p * in + 200], &temb[(size_t)p * text_dim_], text_dim_ * 4);
    }
    GT cg = g.alloc(cat.size()), X = g.alloc((size_t)N * D), T1 = g.alloc((size_t)N * D), T2 = g.alloc((size_t)N * D);
    g.upload(cg, cat.data(), cat.size());
    { GT b = g.wf32("input_embed.proj.bias"); g.linear(cg, N, in, g.w16("input_embed.proj.weight"), D, &b, X); }
    g.free(cg);
    // convolutional position embedding: 2 x (grouped conv1d k31, 16 groups, Mish), added to x
    g.transpose(X, N, D, T1);   // [D][N]
    const int groups = 16, gc = D / groups;
    for (int l = 0; l < 2; l++) {
        std::string n = "input_embed.conv_pos_embed.conv1d." + std::to_string(l * 2);
        uint64_t W = g.w16(n + ".weight"); GT B = g.wf32(n + ".bias");
        GT& src = l == 0 ? T1 : T2; GT& dst = l == 0 ? T2 : T1;
        for (int gi = 0; gi < groups; gi++) {
            GT xs = src, ys = dst, bs = B;
            xs.p += (uint64_t)gi * gc * N * 4; ys.p += (uint64_t)gi * gc * N * 4; bs.p += (uint64_t)gi * gc * 4;
            g.conv(xs, gc, 1, N, W + (uint64_t)gi * gc * gc * 31 * 2, gc, 1, 31, 1, 0, &bs, ys, 0.f, 15);
        }
        g.act(dst, (size_t)D * N, Gx::MISH);
    }
    g.transpose(T1, D, N, T2);   // back to [N][D]
    g.axpy(X, T2, (size_t)N * D);
    dbg("h_input", X, (size_t)N * D);
    // time embedding (sin first, scale 1000)
    std::vector<float> te(256);
    double emb = std::log(10000.0) / 127;
    for (int i = 0; i < 128; i++) { double a = 1000.0 * t * std::exp(-emb * i); te[i] = (float)std::sin(a); te[128 + i] = (float)std::cos(a); }
    GT tg = g.alloc(256), t1 = g.alloc(D), st = g.alloc(D), mod = g.alloc((size_t)6 * D);
    g.upload(tg, te.data(), 256);
    { GT b = g.wf32("time_embed.time_mlp.0.bias"); g.linear(tg, 1, 256, g.w16("time_embed.time_mlp.0.weight"), D, &b, t1); }
    g.act(t1, D, Gx::SILU);
    { GT b = g.wf32("time_embed.time_mlp.2.bias"); g.linear(t1, 1, D, g.w16("time_embed.time_mlp.2.weight"), D, &b, st); }
    dbg("h_time", st, D);
    g.act(st, D, Gx::SILU);   // every AdaLN applies SiLU to the time embedding
    // rotary table from the stored inverse frequencies
    std::vector<float> inv = hostw(f5_, "rotary_embed.inv_freq"), cs((size_t)N * 32 * 2);
    for (int p = 0; p < N; p++) for (int i = 0; i < 32; i++) { cs[((size_t)p * 32 + i) * 2] = std::cos(p * inv[i]); cs[((size_t)p * 32 + i) * 2 + 1] = std::sin(p * inv[i]); }
    GT csg = g.alloc(cs.size());
    g.upload(csg, cs.data(), cs.size());
    GT n = g.alloc((size_t)N * D), q = g.alloc((size_t)N * D), k = g.alloc((size_t)N * D), v = g.alloc((size_t)N * D), a = g.alloc((size_t)N * D);
    const int FF = (int)f5_.get("transformer_blocks.0.ff.ff.0.0.weight")->shape[0];
    GT ff = g.alloc((size_t)N * FF);
    std::vector<float> mh((size_t)6 * D);
    GT m[6]; for (int i = 0; i < 6; i++) { m[i] = mod; m[i].p += (uint64_t)i * D * 4; m[i].n = D; }
    for (int bi = 0; bi < depth_; bi++) {
        std::string p = "transformer_blocks." + std::to_string(bi) + ".";
        { GT b = g.wf32(p + "attn_norm.linear.bias"); g.linear(st, 1, D, g.w16(p + "attn_norm.linear.weight"), 6 * D, &b, mod); }
        g.download(mh.data(), mod, mh.size());
        for (int c = 0; c < D; c++) { mh[(size_t)1 * D + c] += 1.f; mh[(size_t)4 * D + c] += 1.f; }   // 1 + scale
        g.upload(mod, mh.data(), mh.size());
        g.layernorm(X, N, D, m[1], m[0], 1e-6f, n);
        if (bi == 0) dbg("b0_norm", n, (size_t)N * D);
        { GT b = g.wf32(p + "attn.to_q.bias"); g.linear(n, N, D, g.w16(p + "attn.to_q.weight"), D, &b, q); }
        { GT b = g.wf32(p + "attn.to_k.bias"); g.linear(n, N, D, g.w16(p + "attn.to_k.weight"), D, &b, k); }
        { GT b = g.wf32(p + "attn.to_v.bias"); g.linear(n, N, D, g.w16(p + "attn.to_v.weight"), D, &b, v); }
        g.rope3(q, csg, N, H, hd); g.rope3(k, csg, N, H, hd);
        g.attention(q, k, v, N, N, H, hd, a);
        { GT b = g.wf32(p + "attn.to_out.0.bias"); g.linear(a, N, D, g.w16(p + "attn.to_out.0.weight"), D, &b, q); }
        if (bi == 0) dbg("b0_attn", q, (size_t)N * D);
        g.gadd(X, q, m[2], D, (size_t)N * D);
        g.layernorm(X, N, D, m[4], m[3], 1e-6f, n);
        { GT b = g.wf32(p + "ff.ff.0.0.bias"); g.linear(n, N, D, g.w16(p + "ff.ff.0.0.weight"), FF, &b, ff); }
        g.act(ff, (size_t)N * FF, Gx::GELU_TANH);
        { GT b = g.wf32(p + "ff.ff.2.bias"); g.linear(ff, N, FF, g.w16(p + "ff.ff.2.weight"), D, &b, q); }
        g.gadd(X, q, m[5], D, (size_t)N * D);
        if (bi == 0) dbg("h_block0", X, (size_t)N * D);
        if (bi == 21) dbg("h_block21", X, (size_t)N * D);
    }
    // final AdaLN: chunks are (scale, shift)
    { GT b = g.wf32("norm_out.linear.bias"); g.linear(st, 1, D, g.w16("norm_out.linear.weight"), 2 * D, &b, mod); }
    g.download(mh.data(), mod, (size_t)2 * D);
    std::vector<float> ss((size_t)2 * D);
    for (int c = 0; c < D; c++) { ss[c] = mh[(size_t)D + c]; ss[(size_t)D + c] = 1.f + mh[c]; }   // m[0] = shift, m[1] = 1 + scale
    g.upload(mod, ss.data(), ss.size());
    g.layernorm(X, N, D, m[1], m[0], 1e-6f, n);
    GT o = g.alloc((size_t)N * 100);
    { GT b = g.wf32("proj_out.bias"); g.linear(n, N, D, g.w16("proj_out.weight"), 100, &b, o); }
    out.resize((size_t)N * 100);
    g.download(out.data(), o, out.size());
    for (GT* tt : {&X, &T1, &T2, &tg, &t1, &st, &mod, &csg, &n, &q, &k, &v, &a, &ff, &o}) g.free(*tt);
}

// Vocos: ConvNeXt backbone over mel frames, then an ISTFT head (magnitude + phase).
void TtsEngine::vocos(const std::vector<float>& melv, int M, std::vector<float>& wave) {
    Gx& g = gx();
    const int D = (int)voc_.get("backbone.embed.weight")->shape[0];
    const int F = (int)voc_.get("backbone.convnext.0.pwconv1.weight")->shape[0];
    GT mg = g.alloc(melv.size()), e = g.alloc((size_t)D * M), x = g.alloc((size_t)M * D), y = g.alloc((size_t)M * D), h = g.alloc((size_t)M * F);
    g.upload(mg, melv.data(), melv.size());
    { GT b = g.wf32("backbone.embed.bias"); g.conv(mg, 100, 1, M, g.w16("backbone.embed.weight"), D, 1, 7, 1, 0, &b, e, 0.f, 3); }
    g.transpose(e, D, M, x);
    g.layernorm(x, M, D, g.wf32("backbone.norm.weight"), g.wf32("backbone.norm.bias"), 1e-6f, x);
    std::vector<float> xh((size_t)M * D), yh((size_t)M * D);
    for (int b = 0;; b++) {
        std::string p = "backbone.convnext." + std::to_string(b) + ".";
        if (!voc_.has(p + "dwconv.weight")) break;
        std::vector<float> dw = hostw(voc_, p + "dwconv.weight"), db = hostw(voc_, p + "dwconv.bias");
        g.download(xh.data(), x, xh.size());
        for (int t = 0; t < M; t++)
            for (int c = 0; c < D; c++) {
                float s = db[c];
                for (int k = 0; k < 7; k++) { int q = t + k - 3; if (q >= 0 && q < M) s += dw[(size_t)c * 7 + k] * xh[(size_t)q * D + c]; }
                yh[(size_t)t * D + c] = s;
            }
        g.upload(y, yh.data(), yh.size());
        g.layernorm(y, M, D, g.wf32(p + "norm.weight"), g.wf32(p + "norm.bias"), 1e-6f, y);
        { GT b1 = g.wf32(p + "pwconv1.bias"); g.linear(y, M, D, g.w16(p + "pwconv1.weight"), F, &b1, h); }
        g.act(h, (size_t)M * F, Gx::GELU);
        { GT b2 = g.wf32(p + "pwconv2.bias"); g.linear(h, M, F, g.w16(p + "pwconv2.weight"), D, &b2, y); }
        g.gadd(x, y, g.wf32(p + "gamma"), D, (size_t)M * D);
    }
    g.layernorm(x, M, D, g.wf32("backbone.final_layer_norm.weight"), g.wf32("backbone.final_layer_norm.bias"), 1e-6f, x);
    const int NO = (int)voc_.get("head.out.weight")->shape[0], nb = NO / 2, nfft = (nb - 1) * 2, hop = 256;
    GT so = g.alloc((size_t)M * NO);
    { GT b = g.wf32("head.out.bias"); g.linear(x, M, D, g.w16("head.out.weight"), NO, &b, so); }
    std::vector<float> s((size_t)M * NO);
    g.download(s.data(), so, s.size());
    for (GT* tt : {&mg, &e, &x, &y, &h, &so}) g.free(*tt);
    std::vector<float> win = hostw(voc_, "head.istft.window");
    if ((int)win.size() != nfft) { win.resize(nfft); for (int n = 0; n < nfft; n++) win[n] = (float)(0.5 - 0.5 * std::cos(2 * kPi * n / nfft)); }
    const size_t full = (size_t)nfft + (size_t)hop * (M - 1);
    std::vector<double> yv(full, 0.0), env(full, 0.0);
    std::vector<std::complex<double>> a(nfft);
    for (int t = 0; t < M; t++) {
        for (int k = 0; k < nb; k++) {
            double mag = std::min(100.0, std::exp((double)s[(size_t)t * NO + k])), ph = s[(size_t)t * NO + nb + k];
            a[k] = std::polar(mag, ph);
        }
        a[0] = a[0].real(); a[nb - 1] = a[nb - 1].real();
        for (int k = 1; k < nb - 1; k++) a[nfft - k] = std::conj(a[k]);
        fft(a, true);
        for (int n = 0; n < nfft; n++) { yv[(size_t)t * hop + n] += a[n].real() * win[n]; env[(size_t)t * hop + n] += (double)win[n] * win[n]; }
    }
    const size_t len = (size_t)hop * (M - 1);
    wave.resize(len);
    for (size_t i = 0; i < len; i++) { size_t j = i + nfft / 2; wave[i] = (float)(env[j] > 1e-11 ? yv[j] / env[j] : 0.0); }
}

// =============================================================================================
// Full pipeline
// =============================================================================================
static std::vector<std::string> chunk_text(const std::string& text, size_t max_chars) {
    std::vector<std::string> sentences, chunks;
    std::string cur;
    for (size_t i = 0; i < text.size(); i++) {   // split after ; : , . ! ? followed by whitespace
        cur += text[i];
        if (strchr(";:,.!?", text[i]) && i + 1 < text.size() && isspace((unsigned char)text[i + 1])) {
            sentences.push_back(cur); cur.clear();
            while (i + 1 < text.size() && isspace((unsigned char)text[i + 1])) i++;
        }
    }
    if (!cur.empty()) sentences.push_back(cur);
    std::string c;
    for (auto& s : sentences) {
        if (c.size() + s.size() <= max_chars) c += s + " ";
        else { if (!c.empty()) chunks.push_back(c); c = s + " "; }
    }
    if (!c.empty()) chunks.push_back(c);
    for (auto& ch : chunks) while (!ch.empty() && ch.back() == ' ') ch.pop_back();
    return chunks;
}

bool TtsEngine::speak(const TtsJob& j, std::vector<float>& out, std::string& err, const std::function<bool(const char*, int, int)>& progress) {
    if (!loaded_) { err = "no speech model loaded"; return false; }
    std::vector<float> ref; int sr;
    if (!read_wav(j.voice_wav, ref, sr, err)) return false;
    ref = resample(ref, sr, 24000);
    if (ref.size() > 12 * 24000) ref.resize(12 * 24000);
    double rms = 0; for (float v : ref) rms += (double)v * v; rms = std::sqrt(rms / std::max<size_t>(1, ref.size()));
    if (rms < 0.1 && rms > 0) for (auto& v : ref) v = (float)(v * 0.1 / rms);
    std::string rt = j.voice_text;
    while (!rt.empty() && isspace((unsigned char)rt.back())) rt.pop_back();
    if (rt.empty()) { err = "the voice needs the exact words spoken in its clip"; return false; }
    if (rt.back() == '.') rt += " "; else rt += ". ";
    std::vector<float> rmel; int R;
    mel(ref, rmel, R);
    const double ref_sec = ref.size() / 24000.0;
    size_t max_chars = (size_t)std::max(20.0, rt.size() / ref_sec * (22 - ref_sec) * j.speed);
    auto chunks = chunk_text(j.text, max_chars);
    std::mt19937_64 rng(j.seed ? j.seed : std::random_device{}());
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> tsched(j.steps + 1);
    for (int i = 0; i <= j.steps; i++) { double t = (double)i / j.steps; tsched[i] = (float)(t - (std::cos(kPi / 2 * t) - 1 + t)); }
    dit_begin();
    const int total = (int)chunks.size() * j.steps;
    int done = 0;
    out.clear();
    for (auto& ch : chunks) {
        std::vector<int> ids = text_ids(rt + ch);
        double sp = ch.size() < 10 ? 0.3 : j.speed;
        int N = R + (int)(R / (double)rt.size() * ch.size() / sp);
        N = std::max(N, std::max((int)ids.size(), R) + 1);
        std::vector<float> cond((size_t)N * 100, 0.f), zc((size_t)N * 100, 0.f), te, tz((size_t)N * text_dim_, 0.f);
        for (int p = 0; p < R; p++) for (int c = 0; c < 100; c++) cond[(size_t)p * 100 + c] = rmel[(size_t)c * R + p];
        text_embed(ids, N, te);
        std::vector<float> x((size_t)N * 100), vc, vu;
        for (auto& v : x) v = nd(rng);
        for (int i = 0; i < j.steps; i++) {
            if (!progress("Speaking", done++, total)) { err = "cancelled"; end(); return false; }
            dit(x, cond, te, N, tsched[i], vc);
            dit(x, zc, tz, N, tsched[i], vu);
            float dt = tsched[i + 1] - tsched[i];
            for (size_t q = 0; q < x.size(); q++) x[q] += dt * (vc[q] + (vc[q] - vu[q]) * j.cfg);
        }
        const int G = N - R;
        std::vector<float> gm((size_t)100 * G);
        for (int p = 0; p < G; p++) for (int c = 0; c < 100; c++) gm[(size_t)c * G + p] = x[(size_t)(R + p) * 100 + c];
        end();
        voc_begin();
        std::vector<float> w;
        vocos(gm, G, w);
        end();
        dit_begin();
        if (rms < 0.1 && rms > 0) for (auto& v : w) v = (float)(v * rms / 0.1);
        // cross-fade 0.15 s between chunks
        size_t cf = std::min<size_t>({(size_t)(0.15 * 24000), out.size(), w.size()});
        if (cf && !out.empty()) {
            size_t o0 = out.size() - cf;
            for (size_t i = 0; i < cf; i++) { float a = (float)i / cf; out[o0 + i] = out[o0 + i] * (1 - a) + w[i] * a; }
            out.insert(out.end(), w.begin() + cf, w.end());
        } else out.insert(out.end(), w.begin(), w.end());
    }
    end();
    gx().trim();
    return true;
}

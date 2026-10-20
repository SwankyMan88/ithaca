#include "wan.h"
#include "model.h"
#include "quant.h"
#include <algorithm>
#include <cmath>
#include <random>

WanEngine& wan_engine() { static WanEngine e; return e; }

// Wan 2.2 VAE latent statistics (part of the model definition: z = raw * std + mean)
static const float kWan22Mean[48] = {-0.2289f, -0.0052f, -0.1323f, -0.2339f, -0.2799f, 0.0174f, 0.1838f, 0.1557f, -0.1382f, 0.0542f, 0.2813f, 0.0891f,
    0.157f, -0.0098f, 0.0375f, -0.1825f, -0.2246f, -0.1207f, -0.0698f, 0.5109f, 0.2665f, -0.2108f, -0.2158f, 0.2502f, -0.2055f, -0.0322f, 0.1109f, 0.1567f,
    -0.0729f, 0.0899f, -0.2799f, -0.123f, -0.0313f, -0.1649f, 0.0117f, 0.0723f, -0.2839f, -0.2083f, -0.052f, 0.3748f, 0.0152f, 0.1957f, 0.1433f, -0.2944f,
    0.3573f, -0.0548f, -0.1681f, -0.0667f};
static const float kWan22Std[48] = {0.4765f, 1.0364f, 0.4514f, 1.1677f, 0.5313f, 0.499f, 0.4818f, 0.5013f, 0.8158f, 1.0344f, 0.5894f, 1.0901f, 0.6885f,
    0.6165f, 0.8454f, 0.4978f, 0.5759f, 0.3523f, 0.7135f, 0.6804f, 0.5833f, 1.4146f, 0.8986f, 0.5659f, 0.7069f, 0.5338f, 0.4889f, 0.4917f, 0.4069f, 0.4999f,
    0.6866f, 0.4093f, 0.5709f, 0.6065f, 0.6415f, 0.4944f, 0.5726f, 1.2042f, 0.5458f, 1.6887f, 0.3971f, 1.06f, 0.3943f, 0.5537f, 0.5444f, 0.4089f, 0.7468f, 0.7744f};

static std::vector<float> host_f32(const STensor* t) {
    std::vector<float> v(t ? (size_t)t->numel() : 0);
    for (size_t i = 0; i < v.size(); i++) {
        if (t->dtype == "F32") v[i] = ((const float*)t->data)[i];
        else if (t->dtype == "BF16") { uint32_t u = (uint32_t)((const uint16_t*)t->data)[i] << 16; memcpy(&v[i], &u, 4); }
        else v[i] = h2f(((const uint16_t*)t->data)[i]);
    }
    return v;
}

// Finds a companion file in the usual ComfyUI layout (models/<sub>/...) or next to the model.
static std::wstring find_companion(const std::wstring& model, const wchar_t* sub, std::initializer_list<const wchar_t*> must) {
    std::wstring dir = model.substr(0, model.find_last_of(L"\\/"));
    std::vector<std::wstring> dirs = {dir, dir.substr(0, dir.find_last_of(L"\\/")) + L"\\" + sub};
    for (auto& d : dirs) {
        WIN32_FIND_DATAW fd;
        HANDLE h = FindFirstFileW((d + L"\\*.safetensors").c_str(), &fd);
        if (h == INVALID_HANDLE_VALUE) continue;
        do {
            std::wstring lo = fd.cFileName; for (auto& c : lo) c = towlower(c);
            bool ok = true;
            for (auto m : must) if (lo.find(m) == std::wstring::npos) ok = false;
            if (ok) { FindClose(h); return d + L"\\" + fd.cFileName; }
        } while (FindNextFileW(h, &fd));
        FindClose(h);
    }
    return L"";
}

bool WanEngine::load(const std::wstring& ditp, const std::wstring& tep, const std::wstring& vaep, std::string& err) {
    unload();
    if (!gx().init(err)) return false;
    std::string pfx;
    if (!dit_.add(ditp, err)) return false;
    if (!dit_.has("blocks.0.self_attn.q.weight")) {
        if (dit_.has("model.diffusion_model.blocks.0.self_attn.q.weight")) { dit_.close(); dit_.add(ditp, err, "model.diffusion_model."); }
        else { err = "not a Wan video model"; dit_.close(); return false; }
    }
    if (dit_.get("blocks.0.self_attn.q.weight")->dtype == "Q4_K" || !dit_.has("patch_embedding.weight")) { err = "unsupported Wan file"; unload(); return false; }
    const STensor* pe = dit_.get("patch_embedding.weight");
    dim = (int)pe->shape[0]; zc = (int)pe->shape[1];
    heads = dim / 128;
    layers = 0; while (dit_.has("blocks." + std::to_string(layers) + ".self_attn.q.weight")) layers++;
    ffn = (int)dit_.get("blocks.0.ffn.0.weight")->shape[0];
    if (dit_.has("blocks.0.cross_attn.k_img.weight") || (int)dit_.get("head.head.weight")->shape[0] != zc * 4) {
        err = "This Wan model is an image-to-video variant; text-to-video models (like Wan 2.2 TI2V-5B) are supported so far."; unload(); return false;
    }
    std::wstring te = tep.empty() ? find_companion(ditp, L"text_encoders", {L"umt5"}) : tep;
    std::wstring vae = vaep.empty() ? find_companion(ditp, L"vae", {zc == 48 ? L"wan2.2" : L"wan", L"vae"}) : vaep;
    if (te.empty()) { err = "umT5-XXL text encoder not found (put umt5_xxl_*.safetensors in models\\text_encoders)"; unload(); return false; }
    if (vae.empty()) { err = "Wan VAE not found (put wan2.2_vae.safetensors in models\\vae)"; unload(); return false; }
    if (!te_.add(te, err) || !vae_.add(vae, err)) { unload(); return false; }
    if (!te_.has("encoder.block.0.layer.0.SelfAttention.q.weight") || !te_.has("spiece_model")) { err = "text encoder file is not a umT5 with an embedded tokenizer"; unload(); return false; }
    if (!vae_.has("decoder.conv1.weight") || vae_.get("decoder.conv1.weight")->shape[1] != zc) { err = "VAE does not match the video model's latent channels"; unload(); return false; }
    path = ditp;
    loaded_ = true;
    return true;
}
void WanEngine::unload() { dit_.close(); te_.close(); vae_.close(); loaded_ = false; sp_loaded_ = false; pieces_.clear(); piece_id_.clear(); }

// =============================================================================================
// SentencePiece (unigram) tokenizer, model read from the protobuf embedded in the text encoder
// =============================================================================================
bool WanEngine::tokenize(const std::string& text, std::vector<int>& ids, std::string& err) {
    if (!sp_loaded_) {
        const STensor* sp = te_.get("spiece_model");
        if (!sp) { err = "no tokenizer"; return false; }
        const uint8_t* p = sp->data; const uint8_t* end = p + sp->nbytes;
        auto varint = [&](const uint8_t*& q) { uint64_t v = 0; int s = 0; while (q < end) { uint8_t b = *q++; v |= (uint64_t)(b & 127) << s; s += 7; if (!(b & 128)) break; } return v; };
        while (p < end) {
            uint64_t key = varint(p); int field = (int)(key >> 3), wt = (int)(key & 7);
            if (wt == 2) {
                uint64_t len = varint(p); const uint8_t* q = p; const uint8_t* qe = p + len; p += len;
                if (field != 1) continue;
                std::string piece; float score = 0; int type = 1;
                while (q < qe) {
                    uint64_t k2 = varint(q); int f2 = (int)(k2 >> 3), w2 = (int)(k2 & 7);
                    if (w2 == 2) { uint64_t l2 = varint(q); if (f2 == 1) piece.assign((const char*)q, l2); q += l2; }
                    else if (w2 == 5) { if (f2 == 2) memcpy(&score, q, 4); q += 4; }
                    else if (w2 == 0) { uint64_t v = varint(q); if (f2 == 3) type = (int)v; }
                    else if (w2 == 1) q += 8;
                    else break;
                }
                int id = (int)pieces_.size();
                pieces_.push_back(piece); scores_.push_back(score);
                if (type == 1 || type == 4) { piece_id_.emplace(piece, id); max_piece_ = std::max(max_piece_, piece.size()); }
            } else if (wt == 0) varint(p);
            else if (wt == 5) p += 4;
            else if (wt == 1) p += 8;
            else break;
        }
        sp_loaded_ = !pieces_.empty();
        if (!sp_loaded_) { err = "could not read the tokenizer"; return false; }
    }
    // normalize: collapse whitespace, prefix and replace spaces with U+2581
    std::string s = "\xE2\x96\x81";
    bool sp = true;
    for (unsigned char c : text) {
        if (isspace(c)) { if (!sp) s += "\xE2\x96\x81"; sp = true; }
        else { s += (char)c; sp = false; }
    }
    while (s.size() > 3 && s.compare(s.size() - 3, 3, "\xE2\x96\x81") == 0 && sp) { s.resize(s.size() - 3); break; }
    const int n = (int)s.size();
    float min_score = 0; for (float v : scores_) min_score = std::min(min_score, v);
    std::vector<float> best(n + 1, -1e30f); std::vector<int> from(n + 1, -1), tok(n + 1, -1);
    best[0] = 0;
    for (int i = 0; i < n; i++) {
        if (best[i] <= -1e29f) continue;
        bool any = false;
        for (size_t L = 1; L <= max_piece_ && i + (int)L <= n; L++) {
            auto it = piece_id_.find(s.substr(i, L));
            if (it == piece_id_.end()) continue;
            float sc = best[i] + scores_[it->second];
            if (sc > best[i + L]) { best[i + L] = sc; from[i + L] = i; tok[i + L] = it->second; }
            any = true;
        }
        if (!any) {   // unknown character: one UTF-8 char as <unk> (id 2)
            unsigned char c = s[i]; int L = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : 4;
            L = std::min(L, n - i);
            float sc = best[i] + min_score - 10;
            if (sc > best[i + L]) { best[i + L] = sc; from[i + L] = i; tok[i + L] = 2; }
        }
    }
    std::vector<int> rev;
    for (int i = n; i > 0 && from[i] >= 0; i = from[i]) rev.push_back(tok[i]);
    ids.assign(rev.rbegin(), rev.rend());
    ids.push_back(1);  // </s>
    if (ids.size() > 512) { ids.resize(511); ids.push_back(1); }
    return true;
}

// =============================================================================================
// umT5-XXL encoder
// =============================================================================================
static int t5_bucket(int rel) {
    int ret = rel > 0 ? 16 : 0, n = std::abs(rel);
    if (n < 8) return ret + n;
    int v = 8 + (int)(std::log(n / 8.0) / std::log(128 / 8.0) * 8);
    return ret + std::min(v, 15);
}

bool WanEngine::encode_text(const std::string& text, std::vector<float>& ctx, std::string& err) {
    std::vector<int> ids;
    if (!tokenize(text, ids, err)) return false;
    Gx& g = gx();
    const int T = (int)ids.size(), D = 4096;
    const STensor* emb = te_.get("shared.weight");
    if (!emb) { err = "text encoder has no embeddings"; return false; }
    const int Dm = (int)emb->shape[1];
    std::vector<float> xh((size_t)T * Dm);
    for (int t = 0; t < T; t++)
        for (int c = 0; c < Dm; c++) {
            size_t i = (size_t)ids[t] * Dm + c;
            xh[(size_t)t * Dm + c] = emb->dtype == "F16" ? h2f(((const uint16_t*)emb->data)[i]) : ((const float*)emb->data)[i];
        }
    int H = 0; while (te_.has("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight") && H == 0) H = (int)te_.get("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight")->shape[1];
    const int dk = Dm / H;
    int L = 0; while (te_.has("encoder.block." + std::to_string(L) + ".layer.0.SelfAttention.q.weight")) L++;
    const int F = (int)te_.get("encoder.block.0.layer.1.DenseReluDense.wi_0.weight")->shape[0];
    int64_t budget = (int64_t)g.vram_free() - (1024ll << 20);
    g.weights_begin(&te_, std::max<int64_t>(0, budget), true);   // fp8 weights kept as int8: same precision, half of fp16
    GT x = g.alloc((size_t)T * Dm), h = g.alloc((size_t)T * Dm), q = g.alloc((size_t)T * Dm), k = g.alloc((size_t)T * Dm), v = g.alloc((size_t)T * Dm),
       a = g.alloc((size_t)T * F), b = g.alloc((size_t)T * F);
    g.upload(x, xh.data(), xh.size());
    std::vector<float> qh((size_t)T * Dm), kh(qh.size()), vh(qh.size()), oh(qh.size()), sc(T);
    for (int l = 0; l < L; l++) {
        std::string p = "encoder.block." + std::to_string(l) + ".layer.";
        g.rmsrow(x, T, Dm, g.wf32(p + "0.layer_norm.weight"), 1e-6f, h);
        g.linear(h, T, Dm, g.w16(p + "0.SelfAttention.q.weight"), Dm, nullptr, q);
        g.linear(h, T, Dm, g.w16(p + "0.SelfAttention.k.weight"), Dm, nullptr, k);
        g.linear(h, T, Dm, g.w16(p + "0.SelfAttention.v.weight"), Dm, nullptr, v);
        g.download(qh.data(), q, qh.size()); g.download(kh.data(), k, kh.size()); g.download(vh.data(), v, vh.size());
        std::vector<float> rel = host_f32(te_.get(p + "0.SelfAttention.relative_attention_bias.weight"));   // [32][H]
        if (rel.empty()) rel = host_f32(te_.get("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"));
        // attention with relative position bias, no 1/sqrt(d) scaling (T5)
        for (int hh = 0; hh < H; hh++)
            for (int i = 0; i < T; i++) {
                float mx = -1e30f;
                for (int j = 0; j < T; j++) {
                    float s = rel[(size_t)t5_bucket(j - i) * H + hh];
                    const float* qi = &qh[(size_t)i * Dm + hh * dk]; const float* kj = &kh[(size_t)j * Dm + hh * dk];
                    for (int d = 0; d < dk; d++) s += qi[d] * kj[d];
                    sc[j] = s; mx = std::max(mx, s);
                }
                float sum = 0; for (int j = 0; j < T; j++) { sc[j] = std::exp(sc[j] - mx); sum += sc[j]; }
                float* o = &oh[(size_t)i * Dm + hh * dk];
                for (int d = 0; d < dk; d++) o[d] = 0;
                for (int j = 0; j < T; j++) { float w = sc[j] / sum; const float* vj = &vh[(size_t)j * Dm + hh * dk]; for (int d = 0; d < dk; d++) o[d] += w * vj[d]; }
            }
        g.upload(q, oh.data(), oh.size());
        g.linear(q, T, Dm, g.w16(p + "0.SelfAttention.o.weight"), Dm, nullptr, x, 1.f);
        g.rmsrow(x, T, Dm, g.wf32(p + "1.layer_norm.weight"), 1e-6f, h);
        g.linear(h, T, Dm, g.w16(p + "1.DenseReluDense.wi_0.weight"), F, nullptr, a);
        g.act(a, (size_t)T * F, Gx::GELU_TANH);
        g.linear(h, T, Dm, g.w16(p + "1.DenseReluDense.wi_1.weight"), F, nullptr, b);
        g.mul(a, b, (size_t)T * F);
        g.linear(a, T, F, g.w16(p + "1.DenseReluDense.wo.weight"), Dm, nullptr, x, 1.f);
    }
    g.rmsrow(x, T, Dm, g.wf32("encoder.final_layer_norm.weight"), 1e-6f, h);
    ctx.assign((size_t)512 * D, 0.f);
    std::vector<float> out((size_t)T * Dm);
    g.download(out.data(), h, out.size());
    for (int t = 0; t < T && t < 512; t++) memcpy(&ctx[(size_t)t * D], &out[(size_t)t * Dm], std::min(D, Dm) * 4);
    for (GT* t : {&x, &h, &q, &k, &v, &a, &b}) g.free(*t);
    g.weights_end();
    return true;
}

// =============================================================================================
// Diffusion transformer
// =============================================================================================
int64_t WanEngine::dit_bytes() const {
    int64_t b = 0;
    for (auto& kv : dit_.all()) b += kv.second.numel() * 2;
    return b;
}

GT WanEngine::text_embed(const std::vector<float>& ctx) {
    Gx& g = gx();
    GT c = g.alloc(ctx.size()), h = g.alloc((size_t)512 * dim), o = g.alloc((size_t)512 * dim);
    g.upload(c, ctx.data(), ctx.size());
    GT b0 = g.wf32("text_embedding.0.bias"), b2 = g.wf32("text_embedding.2.bias");
    g.linear(c, 512, 4096, g.w16("text_embedding.0.weight"), dim, &b0, h);
    g.act(h, (size_t)512 * dim, Gx::GELU_TANH);
    g.linear(h, 512, dim, g.w16("text_embedding.2.weight"), dim, &b2, o);
    g.free(c); g.free(h);
    return o;
}

void WanEngine::dit(const std::vector<float>& x, int F, int H, int W, float timestep, const GT& ctxe, std::vector<float>& out) {
    Gx& g = gx();
    const int Hp = H / 2, Wp = W / 2, L = F * Hp * Wp, hd = dim / heads, pk = zc * 4;
    // patchify (1x2x2): token vector index c*4 + i*2 + j
    std::vector<float> pt((size_t)L * pk);
    for (int f = 0; f < F; f++)
        for (int hy = 0; hy < Hp; hy++)
            for (int wx = 0; wx < Wp; wx++) {
                size_t l = ((size_t)f * Hp + hy) * Wp + wx;
                for (int c = 0; c < zc; c++)
                    for (int i = 0; i < 2; i++)
                        for (int j = 0; j < 2; j++)
                            pt[l * pk + c * 4 + i * 2 + j] = x[(((size_t)c * F + f) * H + hy * 2 + i) * W + wx * 2 + j];
            }
    GT xp = g.alloc(pt.size()), X = g.alloc((size_t)L * dim);
    g.upload(xp, pt.data(), pt.size());
    { GT b = g.wf32("patch_embedding.bias"); g.linear(xp, L, pk, g.w16("patch_embedding.weight"), dim, &b, X); }
    g.free(xp);
    // time embedding
    std::vector<float> te(256);
    for (int i = 0; i < 128; i++) { double f = std::pow(10000.0, -i / 128.0); te[i] = (float)std::cos(timestep * f); te[128 + i] = (float)std::sin(timestep * f); }
    GT t0 = g.alloc(256), e1 = g.alloc(dim), e = g.alloc(dim), e0 = g.alloc((size_t)6 * dim);
    g.upload(t0, te.data(), 256);
    { GT b = g.wf32("time_embedding.0.bias"); g.linear(t0, 1, 256, g.w16("time_embedding.0.weight"), dim, &b, e1); }
    g.act(e1, dim, Gx::SILU);
    { GT b = g.wf32("time_embedding.2.bias"); g.linear(e1, 1, dim, g.w16("time_embedding.2.weight"), dim, &b, e); }
    g.copy(e1, e, dim);
    g.act(e1, dim, Gx::SILU);
    { GT b = g.wf32("time_projection.1.bias"); g.linear(e1, 1, dim, g.w16("time_projection.1.weight"), 6 * dim, &b, e0); }
    std::vector<float> eh(dim), e0h((size_t)6 * dim);
    g.download(eh.data(), e, dim); g.download(e0h.data(), e0, e0h.size());
    // 3D rotary table: pairs split [time | height | width] = hd/2 - 2*(hd/6), hd/6, hd/6
    const int half = hd / 2, ph = hd / 6, ptm = half - 2 * ph;
    std::vector<float> cs((size_t)L * half * 2);
    for (int f = 0; f < F; f++)
        for (int hy = 0; hy < Hp; hy++)
            for (int wx = 0; wx < Wp; wx++) {
                size_t l = ((size_t)f * Hp + hy) * Wp + wx;
                for (int j = 0; j < half; j++) {
                    int axis = j < ptm ? 0 : j < ptm + ph ? 1 : 2;
                    int jj = axis == 0 ? j : axis == 1 ? j - ptm : j - ptm - ph;
                    int ad = axis == 0 ? 2 * ptm : 2 * ph;
                    double om = 1.0 / std::pow(10000.0, 2.0 * jj / ad), pos = axis == 0 ? f : axis == 1 ? hy : wx;
                    cs[(l * half + j) * 2] = (float)std::cos(pos * om);
                    cs[(l * half + j) * 2 + 1] = (float)std::sin(pos * om);
                }
            }
    GT csg = g.alloc(cs.size());
    g.upload(csg, cs.data(), cs.size());
    GT n = g.alloc((size_t)L * dim), q = g.alloc((size_t)L * dim), k = g.alloc((size_t)L * dim), v = g.alloc((size_t)L * dim), a = g.alloc((size_t)L * dim),
       kc = g.alloc((size_t)512 * dim), vc = g.alloc((size_t)512 * dim), ff = g.alloc((size_t)L * ffn), mods = g.alloc((size_t)8 * dim);
    std::vector<float> mh((size_t)8 * dim);
    GT m[8];
    for (int i = 0; i < 8; i++) { m[i] = mods; m[i].p += (uint64_t)i * dim * 4; m[i].n = dim; }
    for (int bi = 0; bi < layers; bi++) {
        std::string p = "blocks." + std::to_string(bi) + ".";
        std::vector<float> mod = host_f32(dit_.get(p + "modulation"));
        // m0 shift1, m1 1+scale1, m2 gate1, m3 shift2, m4 1+scale2, m5 gate2
        for (int c = 0; c < dim; c++) {
            for (int kk = 0; kk < 6; kk++) mh[(size_t)kk * dim + c] = mod[(size_t)kk * dim + c] + e0h[(size_t)kk * dim + c];
            mh[(size_t)1 * dim + c] += 1.f; mh[(size_t)4 * dim + c] += 1.f;
        }
        g.upload(mods, mh.data(), (size_t)6 * dim);
        // self attention
        g.layernorm(X, L, dim, m[1], m[0], 1e-6f, n);
        { GT b = g.wf32(p + "self_attn.q.bias"); g.linear(n, L, dim, g.w16(p + "self_attn.q.weight"), dim, &b, q); }
        { GT b = g.wf32(p + "self_attn.k.bias"); g.linear(n, L, dim, g.w16(p + "self_attn.k.weight"), dim, &b, k); }
        { GT b = g.wf32(p + "self_attn.v.bias"); g.linear(n, L, dim, g.w16(p + "self_attn.v.weight"), dim, &b, v); }
        g.rmsrow(q, L, dim, g.wf32(p + "self_attn.norm_q.weight"), 1e-6f, q);
        g.rmsrow(k, L, dim, g.wf32(p + "self_attn.norm_k.weight"), 1e-6f, k);
        g.rope3(q, csg, L, heads, hd);
        g.rope3(k, csg, L, heads, hd);
        g.attention(q, k, v, L, L, heads, hd, a);
        { GT b = g.wf32(p + "self_attn.o.bias"); g.linear(a, L, dim, g.w16(p + "self_attn.o.weight"), dim, &b, q); }
        g.gadd(X, q, m[2], dim, (size_t)L * dim);
        // cross attention
        g.layernorm(X, L, dim, g.wf32(p + "norm3.weight"), g.wf32(p + "norm3.bias"), 1e-6f, n);
        { GT b = g.wf32(p + "cross_attn.q.bias"); g.linear(n, L, dim, g.w16(p + "cross_attn.q.weight"), dim, &b, q); }
        { GT b = g.wf32(p + "cross_attn.k.bias"); g.linear(ctxe, 512, dim, g.w16(p + "cross_attn.k.weight"), dim, &b, kc); }
        { GT b = g.wf32(p + "cross_attn.v.bias"); g.linear(ctxe, 512, dim, g.w16(p + "cross_attn.v.weight"), dim, &b, vc); }
        g.rmsrow(q, L, dim, g.wf32(p + "cross_attn.norm_q.weight"), 1e-6f, q);
        g.rmsrow(kc, 512, dim, g.wf32(p + "cross_attn.norm_k.weight"), 1e-6f, kc);
        g.attention(q, kc, vc, L, 512, heads, hd, a);
        { GT b = g.wf32(p + "cross_attn.o.bias"); g.linear(a, L, dim, g.w16(p + "cross_attn.o.weight"), dim, &b, X, 1.f); }
        // feed forward
        g.layernorm(X, L, dim, m[4], m[3], 1e-6f, n);
        { GT b = g.wf32(p + "ffn.0.bias"); g.linear(n, L, dim, g.w16(p + "ffn.0.weight"), ffn, &b, ff); }
        g.act(ff, (size_t)L * ffn, Gx::GELU_TANH);
        { GT b = g.wf32(p + "ffn.2.bias"); g.linear(ff, L, ffn, g.w16(p + "ffn.2.weight"), dim, &b, q); }
        g.gadd(X, q, m[5], dim, (size_t)L * dim);
    }
    // head
    std::vector<float> hm = host_f32(dit_.get("head.modulation"));
    for (int c = 0; c < dim; c++) { mh[c] = hm[c] + eh[c]; mh[(size_t)dim + c] = 1.f + hm[(size_t)dim + c] + eh[c]; }
    g.upload(mods, mh.data(), (size_t)2 * dim);
    g.layernorm(X, L, dim, m[1], m[0], 1e-6f, n);
    GT o = g.alloc((size_t)L * pk);
    { GT b = g.wf32("head.head.bias"); g.linear(n, L, dim, g.w16("head.head.weight"), pk, &b, o); }
    std::vector<float> oh((size_t)L * pk);
    g.download(oh.data(), o, oh.size());
    out.assign((size_t)zc * F * H * W, 0.f);
    for (int f = 0; f < F; f++)
        for (int hy = 0; hy < Hp; hy++)
            for (int wx = 0; wx < Wp; wx++) {
                size_t l = ((size_t)f * Hp + hy) * Wp + wx;
                for (int i = 0; i < 2; i++)
                    for (int j = 0; j < 2; j++)
                        for (int c = 0; c < zc; c++)
                            out[(((size_t)c * F + f) * H + hy * 2 + i) * W + wx * 2 + j] = oh[l * pk + (i * 2 + j) * zc + c];
            }
    for (GT* t : {&X, &t0, &e1, &e, &e0, &csg, &n, &q, &k, &v, &a, &kc, &vc, &ff, &mods, &o}) g.free(*t);
}

// =============================================================================================
// Wan 2.x causal 3D VAE decoder (decoded one latent frame at a time with per-conv caches)
// =============================================================================================
namespace {
struct Cache { GT t; int frames = 0; bool rep = false, seen = false; };
struct VaeDec {
    SafeTensors* st;
    std::map<std::string, Cache> cache;
    int C = 0, T = 0, H = 0, W = 0;   // shape of x
    GT x;
    int shape(const std::string& n, int i) { return (int)st->get(n)->shape[i]; }
    bool has(const std::string& n) { return st->has(n); }
    // causal conv with temporal kernel KT (cache of the previous input frames)
    GT cconv(const std::string& name, const GT& in, int Ci, int Tn, int h, int w, int& Co) {
        Gx& g = gx();
        const STensor* wt = st->get(name + ".weight");
        Co = (int)wt->shape[0];
        int KT = wt->shape.size() == 5 ? (int)wt->shape[2] : 1, KS = (int)wt->shape[wt->shape.size() - 1];
        GT b = g.wf32(name + ".bias");
        GT y = g.alloc((size_t)Co * Tn * h * w);
        size_t plane = (size_t)h * w;
        Cache& c = cache[name];
        if (KT == 1) {
            g.conv3d(in, Ci, Tn, h, w, g.w16(name + ".weight"), Co, 1, KS, &b, y);
        } else {
            int pad = KT - 1, prev = std::min(c.frames, pad);
            GT xin = g.alloc((size_t)Ci * (pad + Tn) * plane);
            // channel-major: for each channel, [zeros][cache frames][input frames]
            for (int ci = 0; ci < Ci; ci++) {
                size_t base = (size_t)ci * (pad + Tn) * plane;
                if (pad - prev > 0) Model::cuda().zero(xin.p + base * 4, (size_t)(pad - prev) * plane * 4);
                if (prev) g.copy(xin, c.t, (size_t)prev * plane, base + (size_t)(pad - prev) * plane, ((size_t)ci * c.frames + (c.frames - prev)) * plane);
                g.copy(xin, in, (size_t)Tn * plane, base + (size_t)pad * plane, (size_t)ci * Tn * plane);
            }
            g.conv3d(xin, Ci, pad + Tn, h, w, g.w16(name + ".weight"), Co, KT, KS, &b, y);
            g.free(xin);
        }
        // update cache: last 2 input frames (or previous last frame + this one)
        if (KT > 1) {
            int keep = std::min(2, Tn + (Tn < 2 && c.frames > 0 ? 1 : 0));
            GT nc = g.alloc((size_t)Ci * keep * plane);
            for (int ci = 0; ci < Ci; ci++) {
                size_t base = (size_t)ci * keep * plane;
                if (Tn >= 2) g.copy(nc, in, 2 * plane, base, ((size_t)ci * Tn + Tn - 2) * plane);
                else if (c.frames > 0) {
                    g.copy(nc, c.t, plane, base, ((size_t)ci * c.frames + c.frames - 1) * plane);
                    g.copy(nc, in, plane, base + plane, (size_t)ci * Tn * plane);
                } else g.copy(nc, in, plane, base, (size_t)ci * Tn * plane);
            }
            if (c.t) g.free(c.t);
            c.t = nc; c.frames = keep;
        }
        return y;
    }
    void resblock(const std::string& p) {
        Gx& g = gx();
        size_t N = (size_t)T * H * W;
        int Co = shape(p + "residual.2.weight", 0), co2;
        GT a = g.alloc((size_t)C * N);
        g.rmsc(x, C, N, g.wf32(p + "residual.0.gamma"), true, a);
        GT h1 = cconv(p + "residual.2", a, C, T, H, W, Co);
        g.free(a); a = g.alloc((size_t)Co * N);
        g.rmsc(h1, Co, N, g.wf32(p + "residual.3.gamma"), true, a);
        GT h2 = cconv(p + "residual.6", a, Co, T, H, W, co2);
        g.free(a); g.free(h1);
        if (has(p + "shortcut.weight")) {
            int cs; GT s = cconv(p + "shortcut", x, C, T, H, W, cs);
            g.axpy(h2, s, (size_t)Co * N); g.free(s);
        } else g.axpy(h2, x, (size_t)Co * N);
        g.free(x); x = h2; C = Co;
    }
    void attn(const std::string& p) {   // per-frame single-head attention (T == 1 here)
        Gx& g = gx();
        const int Tk = H * W;
        for (int t = 0; t < T; t++) {
            GT fr = g.alloc((size_t)C * Tk);
            for (int c = 0; c < C; c++) g.copy(fr, x, Tk, (size_t)c * Tk, ((size_t)c * T + t) * Tk);
            GT nrm = g.alloc((size_t)C * Tk), qkv = g.alloc((size_t)3 * C * Tk);
            g.rmsc(fr, C, Tk, g.wf32(p + "norm.gamma"), false, nrm);
            GT b = g.wf32(p + "to_qkv.bias");
            g.conv(nrm, C, H, W, g.w16(p + "to_qkv.weight"), 3 * C, 1, 1, 1, 0, &b, qkv);
            GT q = g.alloc((size_t)C * Tk), k = g.alloc((size_t)C * Tk), v = g.alloc((size_t)C * Tk), o = g.alloc((size_t)C * Tk);
            GT qs = qkv, ks = qkv, vs = qkv; ks.p += (uint64_t)C * Tk * 4; vs.p += (uint64_t)2 * C * Tk * 4;
            g.transpose(qs, C, Tk, q); g.transpose(ks, C, Tk, k); g.transpose(vs, C, Tk, v);
            g.attention(q, k, v, Tk, Tk, 1, C, o);
            g.transpose(o, Tk, C, nrm);
            GT pb = g.wf32(p + "proj.bias");
            g.conv(nrm, C, H, W, g.w16(p + "proj.weight"), C, 1, 1, 1, 0, &pb, fr, 1.f);
            for (int c = 0; c < C; c++) g.copy(x, fr, Tk, ((size_t)c * T + t) * Tk, (size_t)c * Tk);
            for (GT* tt : {&fr, &nrm, &qkv, &q, &k, &v, &o}) g.free(*tt);
        }
    }
    void resample(const std::string& p, bool first) {
        Gx& g = gx();
        if (has(p + "time_conv.weight")) {
            Cache& c = cache[p + "time_conv"];
            if (first && !c.seen) {   // first chunk: no temporal upsampling; later chunks start from a zero frame
                c.seen = true;
                c.t = g.alloc((size_t)C * H * W);
                Model::cuda().zero(c.t.p, (size_t)C * H * W * 4);
                c.frames = 1;
            } else {
                int Co;
                GT y = cconv(p + "time_conv", x, C, T, H, W, Co);   // [2C][T][HW]
                GT z = g.alloc((size_t)C * 2 * T * H * W);
                g.tinter(y, C, T, H * W, z);
                g.free(y); g.free(x); x = z; T *= 2;
            }
        }
        GT u = g.alloc((size_t)C * T * H * W * 4);
        g.up2(x, C * T, H, W, u);
        H *= 2; W *= 2;
        int Co = shape(p + "resample.1.weight", 0);
        GT o = g.alloc((size_t)Co * T * H * W), b = g.wf32(p + "resample.1.bias");
        g.conv3d(u, C, T, H, W, g.w16(p + "resample.1.weight"), Co, 1, 3, &b, o);
        g.free(u); g.free(x); x = o; C = Co;
    }
};
}  // namespace

bool WanEngine::vae_decode(const std::vector<float>& z, int F, int Hl, int Wl, std::vector<float>& rgb, int& oF, int& oH, int& oW) {
    Gx& g = gx();
    g.weights_begin(&vae_, std::max<int64_t>(0, (int64_t)g.vram_free() - (3ll << 30)));
    VaeDec d; d.st = &vae_;
    const size_t plane = (size_t)Hl * Wl;
    GT zall = g.alloc(z.size()), zc2 = g.alloc(z.size());
    g.upload(zall, z.data(), z.size());
    { GT b = g.wf32("conv2.bias"); g.conv3d(zall, zc, F, Hl, Wl, g.w16("conv2.weight"), zc, 1, 1, &b, zc2); }
    std::vector<std::vector<float>> chunks;
    int ups = 0; while (vae_.has("decoder.upsamples." + std::to_string(ups) + ".upsamples.0.residual.0.gamma")) ups++;
    for (int i = 0; i < F; i++) {
        bool first = i == 0;
        d.C = zc; d.T = 1; d.H = Hl; d.W = Wl;
        GT xi = g.alloc((size_t)zc * plane);
        for (int c = 0; c < zc; c++) g.copy(xi, zc2, plane, (size_t)c * plane, ((size_t)c * F + i) * plane);
        int Co;
        d.x = d.cconv("decoder.conv1", xi, zc, 1, Hl, Wl, Co);
        g.free(xi);
        d.C = Co;
        d.resblock("decoder.middle.0.");
        d.attn("decoder.middle.1.");
        d.resblock("decoder.middle.2.");
        for (int u = 0; u < ups; u++) {
            std::string p = "decoder.upsamples." + std::to_string(u) + ".upsamples.";
            GT xin = d.x; int Cin = d.C, Tin = d.T, Hin = d.H, Win = d.W;
            // keep the block input for the DupUp shortcut
            GT keep = g.alloc((size_t)Cin * Tin * Hin * Win);
            g.copy(keep, xin, (size_t)Cin * Tin * Hin * Win);
            int r = 0;
            for (; vae_.has(p + std::to_string(r) + ".residual.0.gamma"); r++) d.resblock(p + std::to_string(r) + ".");
            bool up = vae_.has(p + std::to_string(r) + ".resample.1.weight");
            bool tup = up && vae_.has(p + std::to_string(r) + ".time_conv.weight");
            if (up) d.resample(p + std::to_string(r) + ".", first);
            if (up) {
                int ft = tup ? 2 : 1, drop = first ? ft - 1 : 0;
                GT sc = g.alloc((size_t)d.C * (Tin * ft - drop) * Hin * 2 * Win * 2);
                g.dupup(keep, Cin, d.C, Tin, Hin, Win, ft, 2, drop, sc);
                g.axpy(d.x, sc, (size_t)d.C * d.T * d.H * d.W);
                g.free(sc);
            }
            g.free(keep);
        }
        size_t N = (size_t)d.T * d.H * d.W;
        GT a = g.alloc((size_t)d.C * N);
        g.rmsc(d.x, d.C, N, g.wf32("decoder.head.0.gamma"), true, a);
        GT o = d.cconv("decoder.head.2", a, d.C, d.T, d.H, d.W, Co);
        g.free(a); g.free(d.x);
        std::vector<float> h((size_t)Co * N);
        g.download(h.data(), o, h.size());
        g.free(o);
        chunks.push_back(std::move(h));
        oH = d.H; oW = d.W;
        if (i == 0) oF = 0;
        oF += d.T;
    }
    for (auto& kv : d.cache) if (kv.second.t) g.free(kv.second.t);
    g.free(zall); g.free(zc2);
    g.weights_end();
    // unpatchify (12 channels = c*4 + r*2 + q; q = row offset, r = column offset) into RGB frames
    const int PH = oH * 2, PW = oW * 2;
    rgb.assign((size_t)oF * PH * PW * 3, 0.f);
    int f0 = 0;
    for (auto& ch : chunks) {
        int Tn = (int)(ch.size() / (12 * (size_t)oH * oW));
        for (int t = 0; t < Tn; t++)
            for (int y = 0; y < oH; y++)
                for (int x = 0; x < oW; x++)
                    for (int c = 0; c < 3; c++)
                        for (int r = 0; r < 2; r++)
                            for (int q = 0; q < 2; q++) {
                                float v = ch[(((size_t)(c * 4 + r * 2 + q) * Tn + t) * oH + y) * oW + x];
                                rgb[(((size_t)(f0 + t) * PH + y * 2 + q) * PW + x * 2 + r) * 3 + c] = std::min(1.f, std::max(0.f, (v + 1.f) * 0.5f));
                            }
        f0 += Tn;
    }
    oH = PH; oW = PW;
    return true;
}

// =============================================================================================
// Sampling (rectified flow, Euler)
// =============================================================================================
bool WanEngine::generate(const VideoJob& jb, std::vector<uint8_t>& out, int& oW, int& oH, int& oF, std::string& err,
                         const std::function<bool(const char*, int, int)>& progress) {
    if (!loaded_) { err = "no video model loaded"; return false; }
    VideoJob j = jb;
    j.width = std::max(64, j.width / 32 * 32); j.height = std::max(64, j.height / 32 * 32);
    j.frames = std::max(1, (j.frames - 1) / 4 * 4 + 1);
    const int F = (j.frames - 1) / 4 + 1, H = j.height / 16, W = j.width / 16;
    const size_t n = (size_t)zc * F * H * W;
    if (!progress("Reading the prompt", 0, j.steps)) { err = "cancelled"; return false; }
    std::vector<float> cc, cu;
    if (!encode_text(j.prompt, cc, err) || !encode_text(j.negative, cu, err)) return false;
    Gx& g = gx();
    int64_t act = (int64_t)F * (H / 2) * (W / 2) * (int64_t)(dim * 8 + ffn) * 4 + (1536ll << 20);
    int64_t vb = std::max<int64_t>(0, (int64_t)g.vram_free() - act);
    // If the fp16 weights don't fit in VRAM, store them as int8 (half the size) instead of streaming every step.
    g.weights_begin(&dit_, vb, dit_bytes() > vb || getenv("ITHACA_Q8"));
    g.set_half(true);
    GT ce = text_embed(cc), ue = text_embed(cu);
    std::mt19937_64 rng(j.seed ? j.seed : std::random_device{}());
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x(n), vc, vu;
    for (auto& v : x) v = nd(rng);
    std::vector<float> sig(j.steps + 1);
    for (int i = 0; i <= j.steps; i++) { double t = 1.0 - (double)i / j.steps; sig[i] = (float)(j.shift * t / (1 + (j.shift - 1) * t)); }
    for (int i = 0; i < j.steps; i++) {
        if (!progress("Generating", i, j.steps)) { err = "cancelled"; break; }
        float s = sig[i];
        dit(x, F, H, W, s * 1000.f, ce, vc);
        if (j.cfg != 1.f) dit(x, F, H, W, s * 1000.f, ue, vu);
        float dt = sig[i + 1] - s;
        for (size_t q = 0; q < n; q++) { float v = j.cfg != 1.f ? vu[q] + j.cfg * (vc[q] - vu[q]) : vc[q]; x[q] += dt * v; }
    }
    g.free(ce); g.free(ue);
    g.weights_end();
    if (!err.empty()) { g.trim(); return false; }
    progress("Decoding video", j.steps, j.steps);
    const size_t per = (size_t)F * H * W;
    for (int c = 0; c < zc && zc == 48; c++) for (size_t q = 0; q < per; q++) x[c * per + q] = x[c * per + q] * kWan22Std[c] + kWan22Mean[c];
    std::vector<float> rgb;
    vae_decode(x, F, H, W, rgb, oF, oH, oW);
    g.trim();
    out.resize(rgb.size());
    for (size_t q = 0; q < rgb.size(); q++) out[q] = (uint8_t)(rgb[q] * 255.f + 0.5f);
    return true;
}

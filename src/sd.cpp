#include "sd.h"
#include "json.h"
#include "quant.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_map>

SdEngine& sd_engine() { static SdEngine e; return e; }
extern std::wstring g_store_dir;

// =============================================================================================
// CLIP BPE tokenizer (vocab.json + merges.txt, looked up next to the exe or in a ComfyUI install)
// =============================================================================================
namespace {
struct ClipTok {
    std::unordered_map<std::string, int> vocab, ranks;
    std::string benc[256];
    bool ok = false;
    static std::string read(const std::wstring& p) {
        FILE* f = _wfopen(p.c_str(), L"rb");
        if (!f) return "";
        std::string s; char b[65536]; size_t r;
        while ((r = fread(b, 1, sizeof b, f)) > 0) s.append(b, r);
        fclose(f);
        return s;
    }
    static void put(std::string& s, uint32_t c) {
        if (c < 0x80) s += (char)c;
        else if (c < 0x800) { s += (char)(0xC0 | (c >> 6)); s += (char)(0x80 | (c & 63)); }
        else { s += (char)(0xE0 | (c >> 12)); s += (char)(0x80 | ((c >> 6) & 63)); s += (char)(0x80 | (c & 63)); }
    }
    bool load(std::string& err) {
        if (ok) return true;
        std::vector<std::wstring> dirs = {g_store_dir + L"\\models\\tokenizers\\clip", g_store_dir + L"\\tokenizers\\clip"};
        wchar_t buf[MAX_PATH];
        if (GetEnvironmentVariableW(L"USERPROFILE", buf, MAX_PATH)) {
            std::wstring u = buf;
            for (auto d : {L"\\Documents\\ComfyUI_windows_portable\\ComfyUI\\comfy\\sd1_tokenizer", L"\\ComfyUI\\comfy\\sd1_tokenizer", L"\\Desktop\\ComfyUI_windows_portable\\ComfyUI\\comfy\\sd1_tokenizer"})
                dirs.push_back(u + d);
        }
        std::string vj, mt;
        for (auto& d : dirs) { vj = read(d + L"\\vocab.json"); mt = read(d + L"\\merges.txt"); if (!vj.empty() && !mt.empty()) break; }
        if (vj.empty() || mt.empty()) { err = "CLIP tokenizer files not found (Settings > Download models > Image gets them with any image model, or put vocab.json and merges.txt in models\\tokenizers\\clip)"; return false; }
        J j; if (!json_parse(vj, j)) { err = "bad vocab.json"; return false; }
        for (auto& kv : j.o) vocab[kv.first] = (int)kv.second.n;
        size_t pos = mt.find('\n') + 1; int rank = 0;   // first line is a version header
        while (pos < mt.size()) {
            size_t e = mt.find('\n', pos); if (e == std::string::npos) e = mt.size();
            std::string line = mt.substr(pos, e - pos);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (!line.empty()) ranks[line] = rank++;
            pos = e + 1;
        }
        int n = 0;
        for (int b = 0; b < 256; b++) {
            bool keep = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
            put(benc[b], keep ? b : 256 + n++);
        }
        ok = true;
        return true;
    }
    static bool letter(uint32_t c) { return c < 128 ? isalpha((int)c) != 0 : !(c >= 0x2000 && c <= 0x206F) && !(c >= 0x3000 && c <= 0x303F) && c < 0x1F000; }
    void bpe(const std::string& word, std::vector<int>& out) {
        std::vector<std::string> sym;
        for (unsigned char b : word) sym.push_back(benc[b]);
        if (sym.empty()) return;
        sym.back() += "</w>";
        while (sym.size() > 1) {
            int best = -1, br = INT32_MAX;
            for (size_t i = 0; i + 1 < sym.size(); i++) {
                auto it = ranks.find(sym[i] + " " + sym[i + 1]);
                if (it != ranks.end() && it->second < br) { br = it->second; best = (int)i; }
            }
            if (best < 0) break;
            sym[best] += sym[best + 1];
            sym.erase(sym.begin() + best + 1);
        }
        for (auto& s : sym) { auto it = vocab.find(s); if (it != vocab.end()) out.push_back(it->second); }
    }
    std::vector<int> encode(const std::string& text_in) {
        std::string t;
        bool sp = true;
        for (unsigned char c : text_in) {   // lowercase + collapse whitespace
            if (isspace(c)) { if (!sp) t += ' '; sp = true; }
            else { t += (char)tolower(c); sp = false; }
        }
        std::vector<int> out;
        size_t i = 0;
        auto cp_at = [&](size_t k, int& len) -> uint32_t {
            unsigned char c = t[k];
            len = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
            if (k + len > t.size()) { len = 1; return c; }
            if (len == 1) return c;
            uint32_t v = c & (0xFF >> (len + 1));
            for (int q = 1; q < len; q++) v = (v << 6) | (t[k + q] & 63);
            return v;
        };
        while (i < t.size()) {
            int len; uint32_t c = cp_at(i, len);
            if (c == ' ') { i++; continue; }
            if (c == '\'' && i + 1 < t.size()) {
                const char* cs[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
                bool hit = false;
                for (auto x : cs) if (t.compare(i, strlen(x), x) == 0) { bpe(x, out); i += strlen(x); hit = true; break; }
                if (hit) continue;
            }
            size_t j = i;
            if (letter(c)) { while (j < t.size()) { int l; uint32_t d = cp_at(j, l); if (!letter(d)) break; j += l; } }
            else if (c >= '0' && c <= '9') j = i + 1;
            else { while (j < t.size()) { int l; uint32_t d = cp_at(j, l); if (d == ' ' || letter(d) || (d >= '0' && d <= '9')) break; j += l; } }
            bpe(t.substr(i, j - i), out);
            i = j;
        }
        return out;
    }
};
ClipTok& clip_tok() { static ClipTok t; return t; }

// =============================================================================================
// CLIP text encoder: OpenAI/HF naming ("text_model.encoder.layers.N...") or OpenCLIP naming
// ("transformer.resblocks.N...", packed in_proj). Layout read from the weights.
// =============================================================================================
struct Clip {
    std::string pfx;
    bool openclip = false;
    int layers = 0, D = 0, heads = 0;
    Gx::Act act = Gx::GELU;
    bool detect(const SafeTensors& st, const std::string& p) {
        pfx = p;
        if (st.has(p + "token_embedding.weight") && st.has(p + "transformer.resblocks.0.ln_1.weight")) openclip = true;
        else if (st.has(p + "embeddings.token_embedding.weight")) openclip = false;
        else return false;
        const STensor* e = st.get(p + (openclip ? "token_embedding.weight" : "embeddings.token_embedding.weight"));
        D = (int)e->shape[1];
        while (st.has(p + (openclip ? "transformer.resblocks." + std::to_string(layers) + ".ln_1.weight"
                                    : "encoder.layers." + std::to_string(layers) + ".layer_norm1.weight"))) layers++;
        heads = D / 64;
        act = (!openclip && D == 768) ? Gx::QUICK_GELU : Gx::GELU;   // OpenAI CLIP-L uses quick_gelu
        return layers > 0;
    }
    std::string L(int i, const char* hf, const char* oc) const {
        return pfx + (openclip ? "transformer.resblocks." + std::to_string(i) + "." + oc : "encoder.layers." + std::to_string(i) + "." + hf);
    }
    // ids: 77 tokens. Returns hidden states after `run` layers (optionally final-normed) and, when
    // pooled != null, also the projected pooled output (all layers + final norm, at the EOS position).
    void encode(const std::vector<int>& ids, int run, bool final_norm, std::vector<float>& hidden, std::vector<float>* pooled, int eos) {
        Gx& g = gx();
        const int T = (int)ids.size();
        GT idg = g.alloc(T);
        std::vector<uint32_t> u(ids.begin(), ids.end());
        g.upload(idg, (const float*)u.data(), T);
        GT x = g.alloc((size_t)T * D), h = g.alloc((size_t)T * D), q = g.alloc((size_t)T * D), k = g.alloc((size_t)T * D), v = g.alloc((size_t)T * D),
           a = g.alloc((size_t)T * D), m = g.alloc((size_t)T * D * 4);
        g.embed(idg.p, g.w16(pfx + (openclip ? "token_embedding.weight" : "embeddings.token_embedding.weight")), T, D, x);
        // position embedding: openclip stores a raw parameter, HF an embedding table (same shape)
        GT pos = g.wf32(pfx + (openclip ? "positional_embedding" : "embeddings.position_embedding.weight"));
        g.axpy(x, pos, (size_t)T * D);
        auto layer = [&](int i) {
            g.layernorm(x, T, D, g.wf32(L(i, "layer_norm1.weight", "ln_1.weight")), g.wf32(L(i, "layer_norm1.bias", "ln_1.bias")), 1e-5f, h);
            if (openclip) {
                uint64_t W = g.w16(L(i, "", "attn.in_proj_weight"));
                GT B = g.wf32(L(i, "", "attn.in_proj_bias"));
                GT bq = B, bk = B, bv = B; bk.p += D * 4; bv.p += 2 * D * 4;
                g.linear(h, T, D, W, D, &bq, q);
                g.linear(h, T, D, W + (uint64_t)D * D * 2, D, &bk, k);
                g.linear(h, T, D, W + (uint64_t)2 * D * D * 2, D, &bv, v);
            } else {
                GT bq = g.wf32(L(i, "self_attn.q_proj.bias", "")), bk = g.wf32(L(i, "self_attn.k_proj.bias", "")), bv = g.wf32(L(i, "self_attn.v_proj.bias", ""));
                g.linear(h, T, D, g.w16(L(i, "self_attn.q_proj.weight", "")), D, &bq, q);
                g.linear(h, T, D, g.w16(L(i, "self_attn.k_proj.weight", "")), D, &bk, k);
                g.linear(h, T, D, g.w16(L(i, "self_attn.v_proj.weight", "")), D, &bv, v);
            }
            auto dump = [&](const char* nm, const GT& t, size_t n) {
                if (i != 0 || D != 768 || !getenv("ITHACA_DBG")) return;
                std::vector<float> hb(n); g.download(hb.data(), t, n);
                FILE* f = fopen((std::string(getenv("ITHACA_DBG")) + "/mine_" + nm + ".bin").c_str(), "wb");
                if (f) { fwrite(hb.data(), 4, n, f); fclose(f); }
            };
            dump("h1", h, (size_t)T * D); dump("q", q, (size_t)T * D);
            g.attention(q, k, v, T, T, heads, 64, a, true);
            dump("a", a, (size_t)T * D);
            GT bo = g.wf32(L(i, "self_attn.out_proj.bias", "attn.out_proj.bias"));
            g.linear(a, T, D, g.w16(L(i, "self_attn.out_proj.weight", "attn.out_proj.weight")), D, &bo, x, 1.f);
            dump("x_attn", x, (size_t)T * D);
            g.layernorm(x, T, D, g.wf32(L(i, "layer_norm2.weight", "ln_2.weight")), g.wf32(L(i, "layer_norm2.bias", "ln_2.bias")), 1e-5f, h);
            GT b1 = g.wf32(L(i, "mlp.fc1.bias", "mlp.c_fc.bias")), b2 = g.wf32(L(i, "mlp.fc2.bias", "mlp.c_proj.bias"));
            g.linear(h, T, D, g.w16(L(i, "mlp.fc1.weight", "mlp.c_fc.weight")), 4 * D, &b1, m);
            g.act(m, (size_t)T * 4 * D, act);
            dump("m", m, (size_t)T * 4 * D);
            g.linear(m, T, 4 * D, g.w16(L(i, "mlp.fc2.weight", "mlp.c_proj.weight")), D, &b2, x, 1.f);
        };
        std::string fn = openclip ? "ln_final" : "final_layer_norm";
        for (int i = 0; i < run; i++) layer(i);
        if (final_norm) { g.layernorm(x, T, D, g.wf32(pfx + fn + ".weight"), g.wf32(pfx + fn + ".bias"), 1e-5f, h); hidden.resize((size_t)T * D); g.download(hidden.data(), h, hidden.size()); }
        else { hidden.resize((size_t)T * D); g.download(hidden.data(), x, hidden.size()); }
        if (pooled) {
            for (int i = run; i < layers; i++) layer(i);
            g.layernorm(x, T, D, g.wf32(pfx + fn + ".weight"), g.wf32(pfx + fn + ".bias"), 1e-5f, h);
            std::vector<float> row(D);
            {
                std::vector<float> all((size_t)T * D);
                g.download(all.data(), h, all.size());
                memcpy(row.data(), &all[(size_t)eos * D], D * 4);
            }
            // projection: OpenCLIP stores P[in][out] (x @ P); HF stores a Linear weight [out][in]
            const STensor* P = gx_st ? gx_st->get(openclip ? pfx + "text_projection" : proj_name) : nullptr;
            if (P) {
                std::vector<float> w((size_t)P->numel());
                for (size_t z = 0; z < w.size(); z++) {
                    if (P->dtype == "F16") w[z] = h2f(((const uint16_t*)P->data)[z]);
                    else if (P->dtype == "F32") w[z] = ((const float*)P->data)[z];
                    else { uint32_t uu = (uint32_t)((const uint16_t*)P->data)[z] << 16; memcpy(&w[z], &uu, 4); }
                }
                int out = (int)P->shape[openclip ? 1 : 0];
                pooled->assign(out, 0.f);
                for (int o = 0; o < out; o++) {
                    double s = 0;
                    for (int c = 0; c < D; c++) s += (double)row[c] * (openclip ? w[(size_t)c * out + o] : w[(size_t)o * D + c]);
                    (*pooled)[o] = (float)s;
                }
            } else *pooled = row;
        }
        for (GT* t : {&idg, &x, &h, &q, &k, &v, &a, &m}) g.free(*t);
    }
    const SafeTensors* gx_st = nullptr;
    std::string proj_name;
};
}  // namespace

// =============================================================================================
// Loading + architecture detection
// =============================================================================================
static bool vpred_from_name(const std::wstring& p) {
    std::wstring lo = p; for (auto& c : lo) c = towlower(c);
    return lo.find(L"v-pred") != std::wstring::npos || lo.find(L"vpred") != std::wstring::npos || lo.find(L"v_pred") != std::wstring::npos;
}

bool SdEngine::load(const std::wstring& ckpt, const std::wstring& vae, std::string& err) {
    unload();
    std::string e;
    if (!gx().init(e)) { err = e; return false; }
    if (!st_.add(ckpt, err)) return false;
    if (!vae.empty()) {
        // a standalone VAE file uses bare names ("decoder..."); map it under the checkpoint's prefix
        if (!st_.add(vae, err, "")) { st_.close(); return false; }
    }
    unet_pfx_ = st_.has("model.diffusion_model.input_blocks.0.0.weight") ? "model.diffusion_model." : (st_.has("input_blocks.0.0.weight") ? "" : "?");
    if (unet_pfx_ == "?" && (st_.has("double_blocks.0.img_attn.qkv.weight") || st_.has("model.diffusion_model.double_blocks.0.img_attn.qkv.weight"))) {
        // Flux: an all-in-one file (ComfyUI layout: model.diffusion_model., text_encoders.clip_l / t5xxl, vae.)
        unet_pfx_ = st_.has("model.diffusion_model.img_in.weight") ? "model.diffusion_model." : "";
        clip_pfx_ = "text_encoders.clip_l.transformer.text_model."; t5_pfx_ = "text_encoders.t5xxl.transformer.";
        vae_pfx_ = st_.has("vae.decoder.conv_in.weight") ? "vae." : !vae.empty() && st_.has("decoder.conv_in.weight") ? "" : "?";
        if (!st_.has(clip_pfx_ + "embeddings.token_embedding.weight") || !st_.has(t5_pfx_ + "shared.weight") || vae_pfx_ == "?") {
            err = "This Flux file has only the transformer. Ithaca needs the all-in-one Flux checkpoint (with CLIP-L, T5-XXL and the VAE inside), like flux1-schnell-fp8.safetensors.";
            st_.close(); return false;
        }
        info_ = SdInfo(); info_.arch = "flux"; info_.default_res = 1024;
        path = ckpt; loaded_ = true;
        return true;
    }
    if (unet_pfx_ == "?") {
        if (st_.has("model.diffusion_model.joint_blocks.0.x_block.attn.qkv.weight")) err = "This is an SD3 model, which isn't supported yet.";
        else err = "Unrecognized image model (expected a single-file SD 1.x / 2.x / XL checkpoint).";
        st_.close(); return false;
    }
    vae_pfx_ = !vae.empty() && st_.has("decoder.conv_in.weight") ? "" : "first_stage_model.";
    if (!st_.has(vae_pfx_ + "decoder.conv_in.weight")) { err = "Checkpoint has no VAE. Choose a separate VAE file."; st_.close(); return false; }
    const STensor* k = nullptr;
    for (auto& kv : st_.all())
        if (kv.first.compare(0, unet_pfx_.size(), unet_pfx_) == 0 && kv.first.find("attn2.to_k.weight") != std::string::npos) { k = &kv.second; break; }
    if (!k) { err = "checkpoint has no cross-attention layers"; st_.close(); return false; }
    info_ = SdInfo();
    info_.ctx_dim = (int)k->shape[1];
    info_.model_ch = (int)st_.get(unet_pfx_ + "input_blocks.0.0.weight")->shape[0];
    if (auto le = st_.get(unet_pfx_ + "label_emb.0.0.weight")) info_.adm = (int)le->shape[1];
    info_.arch = info_.ctx_dim == 768 ? "sd1" : info_.ctx_dim == 1024 ? "sd2" : "sdxl";
    info_.default_res = info_.arch == "sdxl" ? 1024 : info_.arch == "sd2" ? 768 : 512;
    info_.vpred = vpred_from_name(ckpt) || (info_.arch == "sd2" && st_.has("v_pred"));
    vae_scale_ = info_.arch == "sdxl" ? 0.13025f : 0.18215f;
    // discrete-time noise schedule (scaled linear betas, 1000 steps)
    sigmas_all_.resize(1000);
    double ac = 1;
    for (int i = 0; i < 1000; i++) {
        double b = std::sqrt(0.00085) + (std::sqrt(0.012) - std::sqrt(0.00085)) * i / 999.0;
        ac *= 1 - b * b;
        sigmas_all_[i] = (float)std::sqrt((1 - ac) / ac);
    }
    path = ckpt;
    loaded_ = true;
    return true;
}
void SdEngine::unload() { st_.close(); loaded_ = false; }

// =============================================================================================
// Text conditioning
// =============================================================================================
bool SdEngine::encode_text(const std::string& text, std::vector<float>& ctx, int& ntok, std::vector<float>& pooled, std::string& err) {
    ClipTok& tk = clip_tok();
    if (!tk.load(err)) return false;
    std::vector<int> ids = tk.encode(text);
    // Long prompts: split into 75-token windows and concatenate their encodings (like ComfyUI).
    std::vector<std::vector<int>> chunks;
    for (size_t i = 0; i < std::max<size_t>(1, ids.size()); i += 75)
        chunks.push_back(std::vector<int>(ids.begin() + std::min(i, ids.size()), ids.begin() + std::min(i + 75, ids.size())));
    Gx& g = gx();
    Clip l, gg;
    l.gx_st = gg.gx_st = &st_;
    std::vector<std::pair<Clip*, int>> encs;   // encoder, pad token
    if (info_.arch == "sdxl") {
        if (!l.detect(st_, "conditioner.embedders.0.transformer.text_model.") || !gg.detect(st_, "conditioner.embedders.1.model.")) { err = "SDXL text encoders not found in checkpoint"; return false; }
    } else if (info_.arch == "sd2") {
        if (!l.detect(st_, "cond_stage_model.model.")) { err = "SD2 text encoder not found"; return false; }
    } else {
        if (!l.detect(st_, "cond_stage_model.transformer.text_model.")) { err = "SD1 text encoder not found"; return false; }
    }
    const int T = 77;
    int width = info_.ctx_dim;
    ntok = (int)chunks.size() * T;
    ctx.assign((size_t)ntok * width, 0.f);
    pooled.clear();
    for (size_t c = 0; c < chunks.size(); c++) {
        auto seq = [&](int pad) {
            std::vector<int> s = {49406};
            s.insert(s.end(), chunks[c].begin(), chunks[c].end());
            s.push_back(49407);
            while ((int)s.size() < T) s.push_back(pad);
            return s;
        };
        int eos = (int)chunks[c].size() + 1;
        std::vector<float> h1, h2, pl;
        g.weights_begin(&st_, 1ll << 40); g.set_half(true);
        if (info_.arch == "sdxl") {
            l.encode(seq(49407), l.layers - 1, false, h1, nullptr, eos);            // CLIP-L penultimate
            g.weights_end(); g.weights_begin(&st_, 1ll << 40); g.set_half(true);
            gg.encode(seq(0), gg.layers - 1, false, h2, c == 0 ? &pl : nullptr, eos);  // CLIP-G penultimate + pooled
            for (int t = 0; t < T; t++) {
                memcpy(&ctx[((size_t)c * T + t) * width], &h1[(size_t)t * l.D], l.D * 4);
                memcpy(&ctx[((size_t)c * T + t) * width + l.D], &h2[(size_t)t * gg.D], gg.D * 4);
            }
        } else if (info_.arch == "sd2") {
            l.encode(seq(0), l.layers - 1, true, h1, nullptr, eos);
            memcpy(&ctx[(size_t)c * T * width], h1.data(), h1.size() * 4);
        } else {
            int skip = 1;
            l.encode(seq(49407), l.layers - (skip - 1), true, h1, nullptr, eos);
            memcpy(&ctx[(size_t)c * T * width], h1.data(), h1.size() * 4);
        }
        g.weights_end();
        if (c == 0) pooled = pl;
    }
    return true;
}

// Developer hook: CLIP-L hidden state after `run` layers for a prompt (used by --sdtest).
bool SdEngine::debug_clipl(const std::string& text, int run, std::vector<float>& out, std::vector<int>& ids) {
    std::string err;
    if (!clip_tok().load(err)) return false;
    ids = {49406};
    auto t = clip_tok().encode(text);
    ids.insert(ids.end(), t.begin(), t.end());
    ids.push_back(49407);
    while (ids.size() < 77) ids.push_back(49407);
    Clip l; l.gx_st = &st_;
    if (!l.detect(st_, "conditioner.embedders.0.transformer.text_model.")) return false;
    gx().weights_begin(&st_, 1ll << 40);
    l.encode(ids, run, false, out, nullptr, 0);
    gx().weights_end();
    return true;
}

static void timestep_embed(float t, int dim, float* out) {
    int half = dim / 2;
    for (int i = 0; i < half; i++) {
        double f = std::exp(-std::log(10000.0) * i / half);
        out[i] = (float)std::cos(t * f);
        out[half + i] = (float)std::sin(t * f);
    }
}

std::vector<float> SdEngine::adm_vector(const std::vector<float>& pooled, int w, int h) {
    std::vector<float> y(pooled);
    float vals[6] = {(float)h, (float)w, 0, 0, (float)h, (float)w};
    for (float v : vals) { float e[256]; timestep_embed(v, 256, e); y.insert(y.end(), e, e + 256); }
    return y;
}

// =============================================================================================
// UNet, built from whichever blocks the checkpoint contains
// =============================================================================================
void SdEngine::unet_begin(int64_t budget) { gx().weights_begin(&st_, budget); gx().set_half(true); }
void SdEngine::unet_end() { gx().weights_end(); }

void SdEngine::unet(const GT& xin, int H, int W, float timestep, const GT& ctx, int ntok, const GT& y, GT& out) {
    Gx& g = gx();
    const std::string P = unet_pfx_;
    auto has = [&](const std::string& n) { return st_.has(P + n); };
    auto shape = [&](const std::string& n, int i) { return (int)st_.get(P + n)->shape[i]; };
    const int mc = info_.model_ch, ed = mc * 4;
    const bool sd1 = info_.arch == "sd1";

    // ---- embeddings
    std::vector<float> te(mc);
    timestep_embed(timestep, mc, te.data());
    GT t0 = g.alloc(mc), e1 = g.alloc(ed), emb = g.alloc(ed);
    g.upload(t0, te.data(), mc);
    { GT b = g.wf32(P + "time_embed.0.bias"); g.linear(t0, 1, mc, g.w16(P + "time_embed.0.weight"), ed, &b, e1); }
    g.act(e1, ed, Gx::SILU);
    { GT b = g.wf32(P + "time_embed.2.bias"); g.linear(e1, 1, ed, g.w16(P + "time_embed.2.weight"), ed, &b, emb); }
    if (info_.adm) {
        GT b0 = g.wf32(P + "label_emb.0.0.bias"), b2 = g.wf32(P + "label_emb.0.2.bias");
        g.linear(y, 1, info_.adm, g.w16(P + "label_emb.0.0.weight"), ed, &b0, e1);
        g.act(e1, ed, Gx::SILU);
        g.linear(e1, 1, ed, g.w16(P + "label_emb.0.2.weight"), ed, &b2, emb, 1.f);
    }
    g.act(emb, ed, Gx::SILU);   // every ResBlock applies SiLU to emb first
    g.free(t0); g.free(e1);

    // ---- building blocks
    auto res = [&](const std::string& p, GT& x, int& C, int h, int w) {
        int Co = shape(p + "in_layers.2.weight", 0);
        size_t hw = (size_t)h * w;
        GT a = g.alloc((size_t)C * hw), b = g.alloc((size_t)Co * hw), e = g.alloc(Co), o = g.alloc((size_t)Co * hw);
        g.groupnorm(x, C, (int)hw, 32, g.wf32(P + p + "in_layers.0.weight"), g.wf32(P + p + "in_layers.0.bias"), 1e-5f, true, a);
        GT cb = g.wf32(P + p + "in_layers.2.bias");
        g.conv(a, C, h, w, g.w16(P + p + "in_layers.2.weight"), Co, 3, 3, 1, 1, &cb, b);
        GT eb = g.wf32(P + p + "emb_layers.1.bias");
        g.linear(emb, 1, ed, g.w16(P + p + "emb_layers.1.weight"), Co, &eb, e);
        g.chanadd(b, e, Co, (int)hw);
        g.free(a); a = g.alloc((size_t)Co * hw);
        g.groupnorm(b, Co, (int)hw, 32, g.wf32(P + p + "out_layers.0.weight"), g.wf32(P + p + "out_layers.0.bias"), 1e-5f, true, a);
        if (has(p + "skip_connection.weight")) {
            GT sb = g.wf32(P + p + "skip_connection.bias");
            g.conv(x, C, h, w, g.w16(P + p + "skip_connection.weight"), Co, 1, 1, 1, 0, &sb, o);
        } else g.copy(o, x, (size_t)C * hw);
        GT ob = g.wf32(P + p + "out_layers.3.bias");
        g.conv(a, Co, h, w, g.w16(P + p + "out_layers.3.weight"), Co, 3, 3, 1, 1, &ob, o, 1.f);
        g.free(a); g.free(b); g.free(e); g.free(x);
        x = o; C = Co;
    };
    auto attn = [&](const std::string& p, const GT& n, int T, const GT& kv, int S, int Dkv, int C, GT& outp) {
        int heads = sd1 ? 8 : C / 64, d = C / heads;
        GT q = g.alloc((size_t)T * C), k = g.alloc((size_t)S * C), v = g.alloc((size_t)S * C), a = g.alloc((size_t)T * C);
        g.linear(n, T, C, g.w16(P + p + "to_q.weight"), C, nullptr, q);
        g.linear(kv, S, Dkv, g.w16(P + p + "to_k.weight"), C, nullptr, k);
        g.linear(kv, S, Dkv, g.w16(P + p + "to_v.weight"), C, nullptr, v);
        g.attention(q, k, v, T, S, heads, d, a);
        GT ob = g.wf32(P + p + "to_out.0.bias");
        g.linear(a, T, C, g.w16(P + p + "to_out.0.weight"), C, &ob, outp, 1.f);
        g.free(q); g.free(k); g.free(v); g.free(a);
    };
    auto xformer = [&](const std::string& p, GT& x, int C, int h, int w) {
        const int T = h * w;
        GT a = g.alloc((size_t)C * T), t = g.alloc((size_t)T * C), n = g.alloc((size_t)T * C);
        g.groupnorm(x, C, T, 32, g.wf32(P + p + "norm.weight"), g.wf32(P + p + "norm.bias"), 1e-6f, false, a);
        g.transpose(a, C, T, t);   // [C][T] -> [T][C]
        GT pb = g.wf32(P + p + "proj_in.bias");
        g.linear(t, T, C, g.w16(P + p + "proj_in.weight"), C, &pb, n);   // 1x1 conv == linear on tokens
        std::swap(t, n);
        for (int bi = 0;; bi++) {
            std::string b = p + "transformer_blocks." + std::to_string(bi) + ".";
            if (!has(b + "norm1.weight")) break;
            g.layernorm(t, T, C, g.wf32(P + b + "norm1.weight"), g.wf32(P + b + "norm1.bias"), 1e-5f, n);
            attn(b + "attn1.", n, T, n, T, C, C, t);
            g.layernorm(t, T, C, g.wf32(P + b + "norm2.weight"), g.wf32(P + b + "norm2.bias"), 1e-5f, n);
            attn(b + "attn2.", n, T, ctx, ntok, info_.ctx_dim, C, t);
            g.layernorm(t, T, C, g.wf32(P + b + "norm3.weight"), g.wf32(P + b + "norm3.bias"), 1e-5f, n);
            int F2 = shape(b + "ff.net.0.proj.weight", 0);
            GT f = g.alloc((size_t)T * F2), gl = g.alloc((size_t)T * F2 / 2);
            GT fb = g.wf32(P + b + "ff.net.0.proj.bias"), f2b = g.wf32(P + b + "ff.net.2.bias");
            g.linear(n, T, C, g.w16(P + b + "ff.net.0.proj.weight"), F2, &fb, f);
            g.geglu(f, T, F2 / 2, gl);
            g.linear(gl, T, F2 / 2, g.w16(P + b + "ff.net.2.weight"), C, &f2b, t, 1.f);
            g.free(f); g.free(gl);
        }
        GT ob = g.wf32(P + p + "proj_out.bias");
        g.linear(t, T, C, g.w16(P + p + "proj_out.weight"), C, &ob, n);
        g.transpose(n, T, C, a);   // back to [C][T]
        g.axpy(x, a, (size_t)C * T);
        g.free(a); g.free(t); g.free(n);
    };
    auto down = [&](const std::string& p, GT& x, int C, int& h, int& w) {
        int h2 = (h + 1) / 2, w2 = (w + 1) / 2;
        GT o = g.alloc((size_t)C * h2 * w2);
        GT b = g.wf32(P + p + "op.bias");
        g.conv(x, C, h, w, g.w16(P + p + "op.weight"), C, 3, 3, 2, 1, &b, o);
        g.free(x); x = o; h = h2; w = w2;
    };
    auto up = [&](const std::string& p, GT& x, int C, int& h, int& w) {
        GT u = g.alloc((size_t)C * h * w * 4), o = g.alloc((size_t)C * h * w * 4);
        g.up2(x, C, h, w, u);
        h *= 2; w *= 2;
        GT b = g.wf32(P + p + "conv.bias");
        g.conv(u, C, h, w, g.w16(P + p + "conv.weight"), C, 3, 3, 1, 1, &b, o);
        g.free(u); g.free(x); x = o;
    };

    // ---- forward
    int h = H, w = W, C = mc;
    GT x = g.alloc((size_t)C * h * w);
    { GT b = g.wf32(P + "input_blocks.0.0.bias"); g.conv(xin, 4, h, w, g.w16(P + "input_blocks.0.0.weight"), C, 3, 3, 1, 1, &b, x); }
    struct Skip { GT t; int C, h, w; };
    std::vector<Skip> skips;
    { GT s = g.alloc((size_t)C * h * w); g.copy(s, x, (size_t)C * h * w); skips.push_back({s, C, h, w}); }
    for (int i = 1;; i++) {
        std::string b = "input_blocks." + std::to_string(i) + ".";
        if (!has(b + "0.in_layers.0.weight") && !has(b + "0.op.weight")) break;
        if (has(b + "0.op.weight")) down(b + "0.", x, C, h, w);
        else {
            res(b + "0.", x, C, h, w);
            if (has(b + "1.norm.weight")) xformer(b + "1.", x, C, h, w);
        }
        GT s = g.alloc((size_t)C * h * w); g.copy(s, x, (size_t)C * h * w); skips.push_back({s, C, h, w});
    }
    res("middle_block.0.", x, C, h, w);
    if (has("middle_block.1.norm.weight")) xformer("middle_block.1.", x, C, h, w);
    res("middle_block.2.", x, C, h, w);
    for (int i = 0;; i++) {
        std::string b = "output_blocks." + std::to_string(i) + ".";
        if (!has(b + "0.in_layers.0.weight")) break;
        Skip s = skips.back(); skips.pop_back();
        GT cat = g.alloc((size_t)(C + s.C) * h * w);
        g.copy(cat, x, (size_t)C * h * w);
        g.copy(cat, s.t, (size_t)s.C * h * w, (size_t)C * h * w);
        g.free(x); g.free(s.t);
        x = cat; C += s.C;
        res(b + "0.", x, C, h, w);
        int nx = 1;
        if (has(b + "1.norm.weight")) { xformer(b + "1.", x, C, h, w); nx = 2; }
        std::string ub = b + std::to_string(nx) + ".";
        if (has(ub + "conv.weight")) up(ub, x, C, h, w);
    }
    for (auto& s : skips) g.free(s.t);
    GT a = g.alloc((size_t)C * h * w);
    g.groupnorm(x, C, h * w, 32, g.wf32(P + "out.0.weight"), g.wf32(P + "out.0.bias"), 1e-5f, true, a);
    GT ob = g.wf32(P + "out.2.bias");
    g.conv(a, C, h, w, g.w16(P + "out.2.weight"), 4, 3, 3, 1, 1, &ob, out);
    g.free(a); g.free(x); g.free(emb);
}

// =============================================================================================
// VAE decoder
// =============================================================================================
bool SdEngine::vae_decode(const std::vector<float>& z, int h, int w, std::vector<float>& img) {
    Gx& g = gx();
    const std::string P = vae_pfx_;
    auto has = [&](const std::string& n) { return st_.has(P + n); };
    auto shape = [&](const std::string& n, int i) { return (int)st_.get(P + n)->shape[i]; };
    g.weights_begin(&st_, 1ll << 40);
    const int ZC = shape("decoder.conv_in.weight", 1);   // latent channels: 4 (SD), 16 (Flux)
    int C = ZC;
    GT x = g.alloc((size_t)ZC * h * w);
    g.upload(x, z.data(), z.size());
    if (has("post_quant_conv.weight")) {
        GT o = g.alloc((size_t)4 * h * w), b = g.wf32(P + "post_quant_conv.bias");
        g.conv(x, 4, h, w, g.w16(P + "post_quant_conv.weight"), 4, 1, 1, 1, 0, &b, o);
        g.free(x); x = o;
    }
    auto conv = [&](GT& in, int Ci, int Co, const std::string& n, int k, GT& o, float beta = 0.f) {
        GT b = g.wf32(P + n + ".bias");
        g.conv(in, Ci, h, w, g.w16(P + n + ".weight"), Co, k, k, 1, k / 2, &b, o, beta);
    };
    auto res = [&](const std::string& p) {
        int Co = shape(p + "conv1.weight", 0);
        size_t hw = (size_t)h * w;
        GT a = g.alloc((size_t)C * hw), b = g.alloc((size_t)Co * hw), o = g.alloc((size_t)Co * hw);
        g.groupnorm(x, C, (int)hw, 32, g.wf32(P + p + "norm1.weight"), g.wf32(P + p + "norm1.bias"), 1e-6f, true, a);
        conv(a, C, Co, p + "conv1", 3, b);
        g.free(a); a = g.alloc((size_t)Co * hw);
        g.groupnorm(b, Co, (int)hw, 32, g.wf32(P + p + "norm2.weight"), g.wf32(P + p + "norm2.bias"), 1e-6f, true, a);
        if (has(p + "nin_shortcut.weight")) conv(x, C, Co, p + "nin_shortcut", 1, o);
        else g.copy(o, x, (size_t)C * hw);
        conv(a, Co, Co, p + "conv2", 3, o, 1.f);
        g.free(a); g.free(b); g.free(x);
        x = o; C = Co;
    };
    auto attn = [&](const std::string& p) {
        const int T = h * w;
        GT n = g.alloc((size_t)C * T), q = g.alloc((size_t)C * T), k = g.alloc((size_t)C * T), v = g.alloc((size_t)C * T);
        GT qt = g.alloc((size_t)C * T), kt = g.alloc((size_t)C * T), vt = g.alloc((size_t)C * T);
        g.groupnorm(x, C, T, 32, g.wf32(P + p + "norm.weight"), g.wf32(P + p + "norm.bias"), 1e-6f, false, n);
        conv(n, C, C, p + "q", 1, q); conv(n, C, C, p + "k", 1, k); conv(n, C, C, p + "v", 1, v);
        g.transpose(q, C, T, qt); g.transpose(k, C, T, kt); g.transpose(v, C, T, vt);
        g.attention(qt, kt, vt, T, T, 1, C, q);   // q reused as [T][C] output
        g.transpose(q, T, C, n);
        conv(n, C, C, p + "proj_out", 1, x, 1.f);
        for (GT* t : {&n, &q, &k, &v, &qt, &kt, &vt}) g.free(*t);
    };
    int Cin = shape("decoder.conv_in.weight", 0);
    { GT o = g.alloc((size_t)Cin * h * w); conv(x, ZC, Cin, "decoder.conv_in", 3, o); g.free(x); x = o; C = Cin; }
    res("decoder.mid.block_1.");
    attn("decoder.mid.attn_1.");
    res("decoder.mid.block_2.");
    int levels = 0;
    while (has("decoder.up." + std::to_string(levels) + ".block.0.conv1.weight")) levels++;
    for (int i = levels - 1; i >= 0; i--) {
        for (int j = 0; has("decoder.up." + std::to_string(i) + ".block." + std::to_string(j) + ".conv1.weight"); j++)
            res("decoder.up." + std::to_string(i) + ".block." + std::to_string(j) + ".");
        std::string u = "decoder.up." + std::to_string(i) + ".upsample.conv";
        if (has(u + ".weight")) {
            GT t = g.alloc((size_t)C * h * w * 4), o = g.alloc((size_t)C * h * w * 4);
            g.up2(x, C, h, w, t);
            h *= 2; w *= 2;
            conv(t, C, C, u, 3, o);
            g.free(t); g.free(x); x = o;
        }
    }
    GT a = g.alloc((size_t)C * h * w), o = g.alloc((size_t)3 * h * w);
    g.groupnorm(x, C, h * w, 32, g.wf32(P + "decoder.norm_out.weight"), g.wf32(P + "decoder.norm_out.bias"), 1e-6f, true, a);
    conv(a, C, 3, "decoder.conv_out", 3, o);
    img.resize((size_t)3 * h * w);
    g.download(img.data(), o, img.size());
    g.free(a); g.free(o); g.free(x);
    g.weights_end();
    return true;
}

// =============================================================================================
// VAE encoder (for editing an existing picture): returns the latent mean, in VAE space
// =============================================================================================
bool SdEngine::vae_encode(const std::vector<float>& img, int H, int W, std::vector<float>& z) {
    Gx& g = gx();
    const std::string P = vae_pfx_;
    auto has = [&](const std::string& n) { return st_.has(P + n); };
    auto shape = [&](const std::string& n, int i) { return (int)st_.get(P + n)->shape[i]; };
    if (!has("encoder.conv_in.weight")) return false;
    g.weights_begin(&st_, 1ll << 40);
    int C = 3, h = H, w = W;
    GT x = g.alloc((size_t)3 * h * w);
    g.upload(x, img.data(), img.size());
    auto conv = [&](GT& in, int Ci, int Co, const std::string& n, int k, GT& o, float beta = 0.f) {
        GT b = g.wf32(P + n + ".bias");
        g.conv(in, Ci, h, w, g.w16(P + n + ".weight"), Co, k, k, 1, k / 2, &b, o, beta);
    };
    auto res = [&](const std::string& p) {
        int Co = shape(p + "conv1.weight", 0);
        size_t hw = (size_t)h * w;
        GT a = g.alloc((size_t)C * hw), b = g.alloc((size_t)Co * hw), o = g.alloc((size_t)Co * hw);
        g.groupnorm(x, C, (int)hw, 32, g.wf32(P + p + "norm1.weight"), g.wf32(P + p + "norm1.bias"), 1e-6f, true, a);
        conv(a, C, Co, p + "conv1", 3, b);
        g.free(a); a = g.alloc((size_t)Co * hw);
        g.groupnorm(b, Co, (int)hw, 32, g.wf32(P + p + "norm2.weight"), g.wf32(P + p + "norm2.bias"), 1e-6f, true, a);
        if (has(p + "nin_shortcut.weight")) conv(x, C, Co, p + "nin_shortcut", 1, o);
        else g.copy(o, x, (size_t)C * hw);
        conv(a, Co, Co, p + "conv2", 3, o, 1.f);
        g.free(a); g.free(b); g.free(x);
        x = o; C = Co;
    };
    auto attn = [&](const std::string& p) {
        const int T = h * w;
        GT n = g.alloc((size_t)C * T), q = g.alloc((size_t)C * T), k = g.alloc((size_t)C * T), v = g.alloc((size_t)C * T);
        GT qt = g.alloc((size_t)C * T), kt = g.alloc((size_t)C * T), vt = g.alloc((size_t)C * T);
        g.groupnorm(x, C, T, 32, g.wf32(P + p + "norm.weight"), g.wf32(P + p + "norm.bias"), 1e-6f, false, n);
        conv(n, C, C, p + "q", 1, q); conv(n, C, C, p + "k", 1, k); conv(n, C, C, p + "v", 1, v);
        g.transpose(q, C, T, qt); g.transpose(k, C, T, kt); g.transpose(v, C, T, vt);
        g.attention(qt, kt, vt, T, T, 1, C, q);
        g.transpose(q, T, C, n);
        conv(n, C, C, p + "proj_out", 1, x, 1.f);
        for (GT* t : {&n, &q, &k, &v, &qt, &kt, &vt}) g.free(*t);
    };
    // The VAE pads only right/bottom before its stride-2 convs: pad on the host, then convolve without padding.
    auto downsample = [&](const std::string& n) {
        std::vector<float> a((size_t)C * h * w), b((size_t)C * (h + 1) * (w + 1), 0.f);
        g.download(a.data(), x, a.size());
        for (int c = 0; c < C; c++)
            for (int y = 0; y < h; y++) memcpy(&b[((size_t)c * (h + 1) + y) * (w + 1)], &a[((size_t)c * h + y) * w], (size_t)w * 4);
        GT pd = g.alloc(b.size()); g.upload(pd, b.data(), b.size());
        int h2 = h / 2, w2 = w / 2;
        GT o = g.alloc((size_t)C * h2 * w2), bias = g.wf32(P + n + ".bias");
        g.conv(pd, C, h + 1, w + 1, g.w16(P + n + ".weight"), C, 3, 3, 2, 0, &bias, o);
        g.free(pd); g.free(x); x = o; h = h2; w = w2;
    };
    int Cin = shape("encoder.conv_in.weight", 0);
    { GT o = g.alloc((size_t)Cin * h * w); conv(x, 3, Cin, "encoder.conv_in", 3, o); g.free(x); x = o; C = Cin; }
    for (int i = 0; has("encoder.down." + std::to_string(i) + ".block.0.conv1.weight"); i++) {
        for (int j = 0; has("encoder.down." + std::to_string(i) + ".block." + std::to_string(j) + ".conv1.weight"); j++)
            res("encoder.down." + std::to_string(i) + ".block." + std::to_string(j) + ".");
        std::string d = "encoder.down." + std::to_string(i) + ".downsample.conv";
        if (has(d + ".weight")) downsample(d);
    }
    res("encoder.mid.block_1.");
    attn("encoder.mid.attn_1.");
    res("encoder.mid.block_2.");
    int Co = shape("encoder.conv_out.weight", 0);
    GT a = g.alloc((size_t)C * h * w), o = g.alloc((size_t)Co * h * w);
    g.groupnorm(x, C, h * w, 32, g.wf32(P + "encoder.norm_out.weight"), g.wf32(P + "encoder.norm_out.bias"), 1e-6f, true, a);
    conv(a, C, Co, "encoder.conv_out", 3, o);
    if (has("quant_conv.weight")) { GT q = g.alloc((size_t)Co * h * w); conv(o, Co, Co, "quant_conv", 1, q); g.free(o); o = q; }
    z.resize((size_t)4 * h * w);   // first 4 channels: the mean (the rest is the log-variance)
    g.download(z.data(), o, z.size());
    g.free(a); g.free(o); g.free(x);
    g.weights_end();
    return true;
}

// =============================================================================================
// Sampling
// =============================================================================================
bool SdEngine::generate(const ImageJob& jb, std::vector<uint8_t>& rgb, std::string& err,
                        const std::function<bool(const char*, int, int)>& progress) {
    if (!loaded_) { err = "no image model loaded"; return false; }
    if (info_.arch == "flux") {
        if (!jb.init_rgb.empty()) { err = "Editing pictures isn't supported with Flux yet; pick an SD or SDXL model for that."; return false; }
        return flux_generate(jb, rgb, err, progress);
    }
    ImageJob j = jb;
    j.width = std::max(64, j.width / 8 * 8); j.height = std::max(64, j.height / 8 * 8);
    const int lh = j.height / 8, lw = j.width / 8, n = 4 * lh * lw;
    Gx& g = gx();
    if (!progress("Reading the prompt", 0, j.steps)) { err = "cancelled"; return false; }
    std::vector<float> c, u, pc, pu;
    int nc, nu;
    if (!encode_text(j.prompt, c, nc, pc, err) || !encode_text(j.negative, u, nu, pu, err)) return false;
    GT cg = g.alloc(c.size()), ug = g.alloc(u.size()), yc, yu;
    g.upload(cg, c.data(), c.size()); g.upload(ug, u.data(), u.size());
    if (info_.adm) {
        auto a = adm_vector(pc, j.width, j.height), b = adm_vector(pu, j.width, j.height);
        yc = g.alloc(a.size()); yu = g.alloc(b.size());
        g.upload(yc, a.data(), a.size()); g.upload(yu, b.data(), b.size());
    }
    // schedule
    float smin = sigmas_all_.front(), smax = sigmas_all_.back();
    std::vector<float> sig;
    if (j.scheduler == "karras") {
        double rmin = std::pow(smin, 1 / 7.0), rmax = std::pow(smax, 1 / 7.0);
        for (int i = 0; i < j.steps; i++) sig.push_back((float)std::pow(rmax + i / (double)std::max(1, j.steps - 1) * (rmin - rmax), 7.0));
    } else {
        for (int i = 0; i < j.steps; i++) {
            double t = 999.0 - i * 999.0 / std::max(1, j.steps - 1);
            int lo = (int)std::floor(t); double fr = t - lo; int hi = std::min(999, lo + 1);
            sig.push_back((float)std::exp((1 - fr) * std::log(sigmas_all_[lo]) + fr * std::log(sigmas_all_[hi])));
        }
    }
    sig.push_back(0.f);
    auto sigma_to_t = [&](float s) {
        float ls = std::log(s);
        int i = 0;
        while (i < 998 && std::log(sigmas_all_[i + 1]) < ls) i++;
        float l0 = std::log(sigmas_all_[i]), l1 = std::log(sigmas_all_[i + 1]);
        float w = std::min(1.f, std::max(0.f, (l0 - ls) / (l0 - l1)));
        return (1 - w) * i + w * (i + 1);
    };
    std::mt19937_64 rng(j.seed ? j.seed : std::random_device{}());
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x(n), den(n), old(n), noise(n), eu(n), ec(n);
    for (auto& v : noise) v = nd(rng);
    // Editing: encode the start picture and begin part-way down the schedule from it, noised to that level.
    const bool edit = j.init_rgb.size() == (size_t)j.width * j.height * 3;
    int first = 0;
    std::vector<float> z0, mk;   // start latent (model space), latent-resolution mask
    if (edit) {
        if (!progress("Reading your picture", 0, j.steps)) { err = "cancelled"; return false; }
        const size_t px = (size_t)j.width * j.height;
        std::vector<float> im(3 * px);
        for (size_t p = 0; p < px; p++) for (int c = 0; c < 3; c++) im[c * px + p] = j.init_rgb[p * 3 + c] / 127.5f - 1.f;
        if (!vae_encode(im, j.height, j.width, z0)) { err = "this checkpoint's VAE has no encoder, so it can't edit pictures"; return false; }
        for (auto& v : z0) v *= vae_scale_;
        first = std::max(0, std::min(j.steps - 1, (int)std::lround(j.steps * (1.0 - std::max(0.f, std::min(1.f, j.strength))))));
        if (j.mask.size() == px) {   // mask at latent resolution: any repaint inside an 8x8 cell repaints the cell
            mk.assign((size_t)lh * lw, 0.f);
            for (int y = 0; y < j.height; y++) for (int xx = 0; xx < j.width; xx++)
                mk[(size_t)(y / 8) * lw + xx / 8] = std::max(mk[(size_t)(y / 8) * lw + xx / 8], j.mask[(size_t)y * j.width + xx] / 255.f);
        }
    }
    for (int q = 0; q < n; q++) x[q] = edit ? z0[q] + noise[q] * sig[first] : noise[q] * sig[0];
    // keeps the unmasked part on the start picture's own path through the schedule
    auto keep = [&](float s) {
        if (mk.empty()) return;
        const size_t hw = (size_t)lh * lw;
        for (int c = 0; c < 4; c++) for (size_t p = 0; p < hw; p++) { float m = mk[p]; size_t q = c * hw + p; x[q] = m * x[q] + (1 - m) * (z0[q] + noise[q] * s); }
    };

    // Leave room for activations; whatever doesn't fit stays in RAM and is streamed each step.
    int64_t act = (int64_t)lh * lw * 64 * 1024 * (info_.arch == "sdxl" ? 3 : 2) + (512ll << 20);
    g.trim();   // buffers pooled by earlier work (the text encoders, the last picture) would otherwise count as used
    const int64_t F_unet = (int64_t)g.vram_free();
    int64_t budget = F_unet - act;
    unet_begin(std::max<int64_t>(0, budget)); const double T_unet = now_sec();
    GT xg = g.alloc(n), og = g.alloc(n);
    std::vector<float> xin(n);
    auto model = [&](float s, std::vector<float>& out) {
        float cin = 1.f / std::sqrt(s * s + 1), t = sigma_to_t(s);
        for (int i = 0; i < n; i++) xin[i] = x[i] * cin;
        g.upload(xg, xin.data(), n);
        unet(xg, lh, lw, t, ug, nu, yu, og); g.download(eu.data(), og, n);
        unet(xg, lh, lw, t, cg, nc, yc, og); g.download(ec.data(), og, n);
        for (int i = 0; i < n; i++) {
            float e = eu[i] + j.cfg * (ec[i] - eu[i]);
            out[i] = info_.vpred ? x[i] / (s * s + 1) - e * s / std::sqrt(s * s + 1) : x[i] - s * e;
        }
    };
    bool have_old = false;
    for (int i = first; i < j.steps; i++) {
        if (!progress(edit ? "Editing" : "Generating", i - first, j.steps - first)) { err = "cancelled"; break; }
        float s = sig[i], sn = sig[i + 1];
        model(s, den);
        if (on_preview) {
            // Linear latent -> RGB approximation of the denoised prediction (the "latent2rgb" trick): free, no VAE.
            static const float xl[4][3] = {{0.3651f, 0.4232f, 0.4341f}, {-0.2533f, -0.0042f, 0.1068f}, {0.1076f, 0.1111f, -0.0362f}, {-0.3165f, -0.2492f, -0.2188f}};
            static const float v1[4][3] = {{0.3512f, 0.2297f, 0.3227f}, {0.3250f, 0.4974f, 0.2350f}, {-0.2829f, 0.1762f, 0.2721f}, {-0.2120f, -0.2616f, -0.7177f}};
            const bool sdxl = info_.arch == "sdxl";
            const float bias[3] = {sdxl ? 0.1084f : 0.f, sdxl ? -0.0175f : 0.f, sdxl ? -0.0011f : 0.f};
            const size_t hw = (size_t)lh * lw;
            std::vector<uint8_t> pv(hw * 3);
            for (size_t p = 0; p < hw; p++)
                for (int c = 0; c < 3; c++) {
                    float v = bias[c];
                    for (int k = 0; k < 4; k++) v += den[k * hw + p] * (sdxl ? xl[k][c] : v1[k][c]);
                    pv[p * 3 + c] = (uint8_t)std::min(255.f, std::max(0.f, (v + 1.f) * 127.5f));
                }
            on_preview(pv, lw, lh);
        }
        if (j.sampler == "euler" || j.sampler == "euler_a") {
            float sd = sn, su = 0;
            if (j.sampler == "euler_a" && sn > 0) { su = std::min(sn, std::sqrt(sn * sn * (s * s - sn * sn) / (s * s))); sd = std::sqrt(sn * sn - su * su); }
            for (int q = 0; q < n; q++) { float d = (x[q] - den[q]) / s; x[q] += d * (sd - s); if (su > 0) x[q] += nd(rng) * su; }
        } else {  // dpmpp_2m
            float t = -std::log(s), tn = sn > 0 ? -std::log(sn) : 0, h = tn - t;
            if (!have_old || sn == 0) {
                for (int q = 0; q < n; q++) x[q] = sn == 0 ? den[q] : (sn / s) * x[q] - std::expm1(-h) * den[q];
            } else {
                float hl = t + std::log(sig[i - 1]), r = hl / h;
                for (int q = 0; q < n; q++) {
                    float dd = (1 + 1 / (2 * r)) * den[q] - (1 / (2 * r)) * old[q];
                    x[q] = (sn / s) * x[q] - std::expm1(-h) * dd;
                }
            }
            old = den; have_old = true;
        }
        keep(sig[i + 1]);
    }
    fprintf(stderr, "[sd] UNet: budget %lld MB, %lld MB resident, %lld MB streamed, %.1f s; VRAM free %lld MB at start, %lld MB at end\n", (long long)(budget >> 20), (long long)(g.resident_bytes() >> 20), (long long)(g.streamed_bytes() >> 20), now_sec() - T_unet, (long long)(F_unet >> 20), (long long)(g.vram_free() >> 20));
    unet_end();
    for (GT* t : {&xg, &og, &cg, &ug, &yc, &yu}) g.free(*t);
    if (!err.empty()) { g.trim(); return false; }
    progress("Decoding", j.steps, j.steps);
    for (auto& v : x) v /= vae_scale_;
    std::vector<float> img;
    vae_decode(x, lh, lw, img);
    g.trim();
    const int W = lw * 8, H = lh * 8;
    rgb.resize((size_t)W * H * 3);
    for (int c = 0; c < 3; c++)
        for (int p = 0; p < W * H; p++) {
            float v = (img[(size_t)c * W * H + p] + 1.f) * 127.5f;
            rgb[(size_t)p * 3 + c] = (uint8_t)std::min(255.f, std::max(0.f, v + 0.5f));
        }
    if (edit && !mk.empty()) {   // outside the mask the original pixels come back exactly; a soft 12 px edge blends the seam
        const int R = 12;
        std::vector<float> m((size_t)W * H), t((size_t)W * H);
        for (size_t p = 0; p < m.size(); p++) m[p] = j.mask[p] / 255.f;
        // box blur of radius R along one line (prefix sums; the window is clipped at the image edge)
        auto blur = [R](const float* in, float* out, int len, size_t step) {
            std::vector<double> ps(len + 1, 0.0);
            for (int i = 0; i < len; i++) ps[i + 1] = ps[i] + in[i * step];
            for (int i = 0; i < len; i++) { int a = std::max(0, i - R), b = std::min(len, i + R + 1); out[i * step] = (float)((ps[b] - ps[a]) / (b - a)); }
        };
        for (int pass = 0; pass < 2; pass++) {   // rows then columns, twice: close to a Gaussian edge
            for (int y = 0; y < H; y++) blur(&m[(size_t)y * W], &t[(size_t)y * W], W, 1);
            for (int xx = 0; xx < W; xx++) blur(&t[xx], &m[xx], H, (size_t)W);
        }
        for (size_t p = 0; p < m.size(); p++) {
            float a = std::min(1.f, m[p] * 1.6f);
            for (int c = 0; c < 3; c++) rgb[p * 3 + c] = (uint8_t)std::lround(a * rgb[p * 3 + c] + (1 - a) * j.init_rgb[p * 3 + c]);
        }
    }
    return true;
}

// =============================================================================================
// Flux.1 (schnell / dev): T5-XXL + CLIP-L conditioning, a rectified-flow transformer (19 double-stream and 38
// single-stream blocks, 3D RoPE over [text | image] tokens), and a 16-channel VAE. Follows the reference
// implementation in black-forest-labs/flux (Apache 2.0). Weights may be fp8: they're stored as int8 on the GPU.
// =============================================================================================
namespace {
// SentencePiece unigram tokenizer from a spiece.model protobuf (T5 v1.1)
struct Spm {
    std::vector<std::string> pieces; std::vector<float> scores; std::unordered_map<std::string, int> id; size_t maxp = 0;
    bool load(const std::string& b) {
        const uint8_t* p = (const uint8_t*)b.data(); const uint8_t* end = p + b.size();
        auto varint = [&](const uint8_t*& q) { uint64_t v = 0; int s = 0; while (q < end) { uint8_t c = *q++; v |= (uint64_t)(c & 127) << s; s += 7; if (!(c & 128)) break; } return v; };
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
                int i = (int)pieces.size();
                pieces.push_back(piece); scores.push_back(score);
                if (type == 1 || type == 4) { id.emplace(piece, i); maxp = std::max(maxp, piece.size()); }
            } else if (wt == 0) varint(p);
            else if (wt == 5) p += 4;
            else if (wt == 1) p += 8;
            else break;
        }
        return !pieces.empty();
    }
    std::vector<int> encode(const std::string& text) {
        std::string s = "\xE2\x96\x81"; bool sp = true;
        for (unsigned char c : text) { if (isspace(c)) { if (!sp) s += "\xE2\x96\x81"; sp = true; } else { s += (char)c; sp = false; } }
        if (sp && s.size() > 3) s.resize(s.size() - 3);
        const int n = (int)s.size();
        float mn = 0; for (float v : scores) mn = std::min(mn, v);
        std::vector<float> best(n + 1, -1e30f); std::vector<int> from(n + 1, -1), tok(n + 1, -1);
        best[0] = 0;
        for (int i = 0; i < n; i++) {
            if (best[i] <= -1e29f) continue;
            bool any = false;
            for (size_t L = 1; L <= maxp && i + (int)L <= n; L++) {
                auto it = id.find(s.substr(i, L)); if (it == id.end()) continue;
                float sc = best[i] + scores[it->second];
                if (sc > best[i + L]) { best[i + L] = sc; from[i + L] = i; tok[i + L] = it->second; }
                any = true;
            }
            if (!any) { unsigned char c = s[i]; int L = std::min(c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : 4, n - i); float sc = best[i] + mn - 10; if (sc > best[i + L]) { best[i + L] = sc; from[i + L] = i; tok[i + L] = 2; } }
        }
        std::vector<int> rev;
        for (int i = n; i > 0 && from[i] >= 0; i = from[i]) rev.push_back(tok[i]);
        std::vector<int> ids(rev.rbegin(), rev.rend());
        ids.push_back(1);   // </s>
        return ids;
    }
};
int t5_bucket_(int rel) {
    int ret = rel > 0 ? 16 : 0, n = std::abs(rel);
    if (n < 8) return ret + n;
    return ret + std::min(8 + (int)(std::log(n / 8.0) / std::log(128 / 8.0) * 8), 15);
}
std::vector<float> host_vec(const STensor* t) {
    std::vector<float> v(t ? (size_t)t->numel() : 0);
    if (!t) return v;
    std::vector<uint16_t> h; to_f16(*t, h);
    for (size_t i = 0; i < v.size(); i++) v[i] = t->dtype == "F32" ? ((const float*)t->data)[i] : h2f(h[i]);
    return v;
}
// T5 v1.1 encoder (gated GELU, relative position bias from block 0, no attention scaling). ids -> [T][D]
void t5_encode(const SafeTensors& st, const std::string& P, const std::vector<int>& ids, std::vector<float>& out) {
    Gx& g = gx();
    const int T = (int)ids.size();
    const STensor* emb = st.get(P + "shared.weight");
    const int D = (int)emb->shape[1];
    std::vector<float> xh((size_t)T * D);
    {
        std::vector<uint16_t> row;
        for (int t = 0; t < T; t++) {
            STensor r = *emb; r.shape = {D}; r.data = emb->data + (size_t)ids[t] * D * (emb->nbytes / emb->numel()); r.nbytes = (size_t)D * (emb->nbytes / emb->numel());
            std::vector<float> v = host_vec(&r);
            memcpy(&xh[(size_t)t * D], v.data(), D * 4);
        }
    }
    const STensor* rb = st.get(P + "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight");
    const int H = (int)rb->shape[1], dk = D / H;
    std::vector<float> rel = host_vec(rb);
    std::vector<float> bias((size_t)H * T * T);   // [h][i][j], same for every layer
    for (int h = 0; h < H; h++) for (int i = 0; i < T; i++) for (int j = 0; j < T; j++) bias[((size_t)h * T + i) * T + j] = rel[(size_t)t5_bucket_(j - i) * H + h];
    int L = 0; while (st.has(P + "encoder.block." + std::to_string(L) + ".layer.0.SelfAttention.q.weight")) L++;
    const int F = (int)st.get(P + "encoder.block.0.layer.1.DenseReluDense.wi_0.weight")->shape[0];
    GT x = g.alloc((size_t)T * D), h = g.alloc((size_t)T * D), q = g.alloc((size_t)T * D), k = g.alloc((size_t)T * D), v = g.alloc((size_t)T * D),
       a = g.alloc((size_t)T * F), b = g.alloc((size_t)T * F), sc = g.alloc((size_t)H * T * T), bg = g.alloc(bias.size());
    g.upload(x, xh.data(), xh.size()); g.upload(bg, bias.data(), bias.size());
    for (int l = 0; l < L; l++) {
        std::string p = P + "encoder.block." + std::to_string(l) + ".layer.";
        g.rmsrow(x, T, D, g.wf32(p + "0.layer_norm.weight"), 1e-6f, h);
        g.linear(h, T, D, g.w16(p + "0.SelfAttention.q.weight"), D, nullptr, q);
        g.linear(h, T, D, g.w16(p + "0.SelfAttention.k.weight"), D, nullptr, k);
        g.linear(h, T, D, g.w16(p + "0.SelfAttention.v.weight"), D, nullptr, v);
        // scores[h][i][j] = q_i . k_j + bias, per head (batched gemm over heads: q/k rows strided by D)
        g.gemm_nt(q, T, dk, D, k, T, D, sc, T, 1.f, H, dk, dk, (size_t)T * T, 0, 0, 0);
        g.axpy(sc, bg, bias.size());
        g.softmax(sc, H * T, T, 1.f);
        g.gemm_nn(sc, T, T, T, v, dk, D, h, D, 1.f, H, (size_t)T * T, dk, dk, 0, 0, 0);
        g.linear(h, T, D, g.w16(p + "0.SelfAttention.o.weight"), D, nullptr, x, 1.f);
        g.rmsrow(x, T, D, g.wf32(p + "1.layer_norm.weight"), 1e-6f, h);
        g.linear(h, T, D, g.w16(p + "1.DenseReluDense.wi_0.weight"), F, nullptr, a);
        g.act(a, (size_t)T * F, Gx::GELU_TANH);
        g.linear(h, T, D, g.w16(p + "1.DenseReluDense.wi_1.weight"), F, nullptr, b);
        g.mul(a, b, (size_t)T * F);
        g.linear(a, T, F, g.w16(p + "1.DenseReluDense.wo.weight"), D, nullptr, x, 1.f);
    }
    g.rmsrow(x, T, D, g.wf32(P + "encoder.final_layer_norm.weight"), 1e-6f, h);
    out.resize((size_t)T * D);
    g.download(out.data(), h, out.size());
    for (GT* t : {&x, &h, &q, &k, &v, &a, &b, &sc, &bg}) g.free(*t);
}
GT view(const GT& t, size_t off, size_t n) { GT v = t; v.p += off * 4; v.n = n; return v; }
}  // namespace

bool SdEngine::flux_tokenizer(std::string& t5, std::string& err) {
    std::wstring dir = path.substr(0, path.find_last_of(L"\\/")), up = dir.substr(0, dir.find_last_of(L"\\/"));
    std::vector<std::wstring> c = {dir, up + L"\\text_encoders", g_store_dir + L"\\models\\text_encoders", g_store_dir + L"\\tokenizers\\t5"};
    for (auto& d : c) for (auto n : {L"\\t5_spiece.model", L"\\spiece.model"}) { t5 = ClipTok::read(d + n); if (!t5.empty()) return true; }
    err = "T5 tokenizer not found (put t5_spiece.model in models\\text_encoders; Settings > Download models > Image gets it with Flux)";
    return false;
}

// One transformer pass: img [Li][64] tokens at time t -> velocity [Li][64]
void SdEngine::flux_dit(const GT& img_in, int hl, int wl, float t, float guidance, const GT& txt_in, int Lt, const GT& pooled, GT& out) {
    Gx& g = gx();
    const std::string& P = unet_pfx_;
    const int D = 3072, NH = 24, HD = 128, Hp = hl / 2, Wp = wl / 2, Li = Hp * Wp, L = Lt + Li, FF = 4 * D;
    auto lin = [&](const GT& x, int M, int K, const std::string& n, int N, GT& y, float beta = 0.f) { GT b = g.wf32(P + n + ".bias"); g.linear(x, M, K, g.w16(P + n + ".weight"), N, &b, y, beta); };
    auto mlp_emb = [&](const std::string& n, const GT& x, int K, GT& acc, float beta) {
        GT h = g.alloc(D); lin(x, 1, K, n + ".in_layer", D, h); g.act(h, D, Gx::SILU); lin(h, 1, D, n + ".out_layer", D, acc, beta); g.free(h);
    };
    auto temb = [&](float v) { std::vector<float> e(256); for (int i = 0; i < 128; i++) { double f = std::exp(-std::log(10000.0) * i / 128.0); e[i] = (float)std::cos(v * f); e[128 + i] = (float)std::sin(v * f); } GT o = g.alloc(256); g.upload(o, e.data(), 256); return o; };
    // conditioning vector
    GT vec = g.alloc(D), sv = g.alloc(D);
    { GT te = temb(t * 1000.f); mlp_emb("time_in", te, 256, vec, 0.f); g.free(te); }
    if (st_.has(P + "guidance_in.in_layer.weight")) { GT ge = temb(guidance * 1000.f); mlp_emb("guidance_in", ge, 256, vec, 1.f); g.free(ge); }
    mlp_emb("vector_in", pooled, 768, vec, 1.f);
    g.copy(sv, vec, D); g.act(sv, D, Gx::SILU);
    // rotary table: text at (0,0,0), image token (0, row, col); axes 16 / 56 / 56 dims
    std::vector<float> cs((size_t)L * 64 * 2);
    for (int l = 0; l < L; l++) {
        const int pr = l < Lt ? 0 : (l - Lt) / Wp, pc = l < Lt ? 0 : (l - Lt) % Wp;
        for (int j = 0; j < 64; j++) {
            int ax = j < 8 ? 0 : j < 36 ? 1 : 2, jj = ax == 0 ? j : ax == 1 ? j - 8 : j - 36, dax = ax == 0 ? 16 : 56;
            double om = 1.0 / std::pow(10000.0, 2.0 * jj / dax), pos = ax == 0 ? 0 : ax == 1 ? pr : pc;
            cs[((size_t)l * 64 + j) * 2] = (float)std::cos(pos * om); cs[((size_t)l * 64 + j) * 2 + 1] = (float)std::sin(pos * om);
        }
    }
    GT csg = g.alloc(cs.size()); g.upload(csg, cs.data(), cs.size());
    GT img = g.alloc((size_t)Li * D), txt = g.alloc((size_t)Lt * D);
    lin(img_in, Li, 64, "img_in", D, img);
    g.copy(txt, txt_in, (size_t)Lt * D);
    GT q = g.alloc((size_t)L * D), k = g.alloc((size_t)L * D), v = g.alloc((size_t)L * D), a = g.alloc((size_t)L * D), n = g.alloc((size_t)L * D), tmp = g.alloc((size_t)L * D),
       big = g.alloc((size_t)L * (3 * D + FF)), bigT = g.alloc((size_t)L * (3 * D + FF)), modg = g.alloc((size_t)12 * D);
    std::vector<float> mh((size_t)12 * D);
    // modulation: returns device vectors [shift, 1+scale, gate] x count
    // modulation vectors [shift, 1+scale, gate] x cnt/3 into modg at slot `at` (each stream of a double block once)
    auto modulate = [&](const std::string& name, int cnt, int at = 0) {
        GT m = g.alloc((size_t)cnt * D);
        lin(sv, 1, D, name, cnt * D, m);
        g.download(mh.data(), m, (size_t)cnt * D); g.free(m);
        for (int s = 0; s < cnt / 3; s++) for (int c = 0; c < D; c++) mh[((size_t)s * 3 + 1) * D + c] += 1.f;
        GT dst = view(modg, (size_t)at * D, (size_t)cnt * D);
        g.upload(dst, mh.data(), (size_t)cnt * D);
    };
    auto M = [&](int i) { return view(modg, (size_t)i * D, D); };
    // split rows of x [R][W] (W = parts * D...) into [R][D] pieces via a transpose: X^T rows are contiguous columns
    auto qkv_split = [&](const GT& x, int R, int W, int col, GT dst) {   // dst [R][D] <- x[:, col:col+D]
        g.transpose(view(bigT, (size_t)col * R, (size_t)D * R), D, R, dst);
    };
    for (int bi = 0; st_.has(P + "double_blocks." + std::to_string(bi) + ".img_mod.lin.weight"); bi++) {
        std::string p = "double_blocks." + std::to_string(bi) + ".";
        for (int s = 0; s < 2; s++) {   // 0 = text stream (rows [0, Lt)), 1 = image stream (rows [Lt, L))
            const std::string sp = s ? "img" : "txt";
            GT& x = s ? img : txt; const int R = s ? Li : Lt; const size_t r0 = s ? Lt : 0; const int o = s * 6;
            modulate(p + sp + "_mod.lin", 6, o);
            g.layernorm(x, R, D, M(o + 1), M(o + 0), 1e-6f, n);
            lin(n, R, D, p + sp + "_attn.qkv", 3 * D, big);
            g.transpose(big, R, 3 * D, bigT);
            qkv_split(big, R, 3 * D, 0, view(q, r0 * D, (size_t)R * D));
            qkv_split(big, R, 3 * D, D, view(k, r0 * D, (size_t)R * D));
            qkv_split(big, R, 3 * D, 2 * D, view(v, r0 * D, (size_t)R * D));
            GT qs = view(q, r0 * D, (size_t)R * D), ks = view(k, r0 * D, (size_t)R * D);
            g.rmsrow(qs, R * NH, HD, g.wf32(P + p + sp + "_attn.norm.query_norm.scale"), 1e-6f, qs);
            g.rmsrow(ks, R * NH, HD, g.wf32(P + p + sp + "_attn.norm.key_norm.scale"), 1e-6f, ks);
        }
        g.rope3(q, csg, L, NH, HD); g.rope3(k, csg, L, NH, HD);
        g.attention(q, k, v, L, L, NH, HD, a);
        for (int s = 0; s < 2; s++) {
            const std::string sp = s ? "img" : "txt";
            GT& x = s ? img : txt; const int R = s ? Li : Lt; const size_t r0 = s ? Lt : 0; const int o = s * 6;
            lin(view(a, r0 * D, (size_t)R * D), R, D, p + sp + "_attn.proj", D, tmp);
            g.gadd(x, tmp, M(o + 2), D, (size_t)R * D);
            g.layernorm(x, R, D, M(o + 4), M(o + 3), 1e-6f, n);
            lin(n, R, D, p + sp + "_mlp.0", FF, big);
            g.act(big, (size_t)R * FF, Gx::GELU_TANH);
            lin(big, R, FF, p + sp + "_mlp.2", D, tmp);
            g.gadd(x, tmp, M(o + 5), D, (size_t)R * D);
        }
    }
    // single stream over [text | image]
    GT X = g.alloc((size_t)L * D);
    g.copy(X, txt, (size_t)Lt * D); g.copy(X, img, (size_t)Li * D, (size_t)Lt * D);
    const int W1 = 3 * D + FF;
    for (int bi = 0; st_.has(P + "single_blocks." + std::to_string(bi) + ".linear1.weight"); bi++) {
        std::string p = "single_blocks." + std::to_string(bi) + ".";
        modulate(p + "modulation.lin", 3);
        g.layernorm(X, L, D, M(1), M(0), 1e-6f, n);
        lin(n, L, D, p + "linear1", W1, big);
        g.transpose(big, L, W1, bigT);            // [W1][L]
        g.transpose(view(bigT, 0, (size_t)D * L), D, L, q);
        g.transpose(view(bigT, (size_t)D * L, (size_t)D * L), D, L, k);
        g.transpose(view(bigT, (size_t)2 * D * L, (size_t)D * L), D, L, v);
        g.rmsrow(q, L * NH, HD, g.wf32(P + p + "norm.query_norm.scale"), 1e-6f, q);
        g.rmsrow(k, L * NH, HD, g.wf32(P + p + "norm.key_norm.scale"), 1e-6f, k);
        g.rope3(q, csg, L, NH, HD); g.rope3(k, csg, L, NH, HD);
        g.attention(q, k, v, L, L, NH, HD, a);
        // [attn | gelu(mlp)] as columns: build the transposed concat in big (rows = features), then transpose back
        { GT at = view(big, 0, (size_t)D * L); g.transpose(a, L, D, at); }
        GT mlpT = view(big, (size_t)D * L, (size_t)FF * L);
        g.copy(mlpT, view(bigT, (size_t)3 * D * L, (size_t)FF * L), (size_t)FF * L);
        g.act(mlpT, (size_t)FF * L, Gx::GELU_TANH);
        g.transpose(view(big, 0, (size_t)(D + FF) * L), D + FF, L, bigT);   // [L][D + FF]
        lin(bigT, L, D + FF, p + "linear2", D, tmp);
        g.gadd(X, tmp, M(2), D, (size_t)L * D);
    }
    // final layer on the image tokens: (shift, scale) = adaLN(silu(vec))
    {
        GT m = g.alloc((size_t)2 * D);
        lin(sv, 1, D, "final_layer.adaLN_modulation.1", 2 * D, m);
        g.download(mh.data(), m, (size_t)2 * D); g.free(m);
        for (int c = 0; c < D; c++) mh[(size_t)D + c] += 1.f;
        g.upload(modg, mh.data(), (size_t)2 * D);
        GT xi = view(X, (size_t)Lt * D, (size_t)Li * D);
        g.layernorm(xi, Li, D, M(1), M(0), 1e-6f, n);
        lin(n, Li, D, "final_layer.linear", 64, out);
    }
    for (GT* tt : {&vec, &sv, &csg, &img, &txt, &q, &k, &v, &a, &n, &tmp, &big, &bigT, &modg, &X}) g.free(*tt);
}

bool SdEngine::flux_generate(const ImageJob& jb, std::vector<uint8_t>& rgb, std::string& err, const std::function<bool(const char*, int, int)>& progress) {
    ImageJob j = jb;
    j.width = std::max(64, j.width / 16 * 16); j.height = std::max(64, j.height / 16 * 16);
    const int hl = j.height / 8, wl = j.width / 8, Li = (hl / 2) * (wl / 2), C = 16;
    const bool dev = st_.has(unet_pfx_ + "guidance_in.in_layer.weight");
    const int Lt = dev ? 512 : 256;
    // schnell is distilled for 1-4 steps (more add nothing); dev takes a guidance strength (3.5 typical) instead of
    // CFG, so the studio's SD default of 6 maps to 3.5.
    if (!dev && j.steps > 8) j.steps = 4;
    const float guidance = j.cfg == 6.f ? 3.5f : j.cfg;
    Gx& g = gx();
    g.set_half(false);
    if (!progress("Reading the prompt", 0, j.steps)) { err = "cancelled"; return false; }
    // CLIP-L: pooled output (final norm at the end-of-text token, no projection)
    std::vector<float> pooled;
    {
        ClipTok& tk = clip_tok();
        if (!tk.load(err)) return false;
        std::vector<int> ids = tk.encode(j.prompt);
        if (ids.size() > 75) ids.resize(75);
        ids.insert(ids.begin(), 49406); ids.push_back(49407);
        const int eos = (int)ids.size() - 1;
        while (ids.size() < 77) ids.push_back(49407);
        Clip l;
        if (!l.detect(st_, clip_pfx_)) { err = "CLIP-L text encoder not found in the Flux file"; return false; }
        g.weights_begin(&st_, 1ll << 40);
        std::vector<float> hid;
        l.encode(ids, l.layers, true, hid, &pooled, eos);
        g.weights_end();
    }
    // T5-XXL, padded to the model's text length
    std::vector<float> t5;
    {
        std::string tb; if (!flux_tokenizer(tb, err)) return false;
        Spm sp; if (!sp.load(tb)) { err = "could not read the T5 tokenizer"; return false; }
        std::vector<int> ids = sp.encode(j.prompt);
        if ((int)ids.size() > Lt) { ids.resize(Lt - 1); ids.push_back(1); }
        while ((int)ids.size() < Lt) ids.push_back(0);
        g.trim();
        g.weights_begin(&st_, std::max<int64_t>(0, (int64_t)g.vram_free() - (1536ll << 20)), true);
        t5_encode(st_, t5_pfx_, ids, t5);
        g.weights_end();
    }
    if (!progress("Generating", 0, j.steps)) { err = "cancelled"; return false; }
    g.trim();
    const int L = Lt + Li, D = 3072;
    int64_t act = (int64_t)L * (D * 8 + (3 * D + 4 * D) * 2) * 4 + (int64_t)L * 4096 * 4 + (768ll << 20);
    const int64_t budget = std::max<int64_t>(0, (int64_t)g.vram_free() - act);
    g.weights_begin(&st_, budget, true);
    const double T0 = now_sec();
    GT txt = g.alloc((size_t)Lt * D), pg = g.alloc(768), xg = g.alloc((size_t)Li * 64), vg = g.alloc((size_t)Li * 64);
    {
        GT t5g = g.alloc(t5.size()); g.upload(t5g, t5.data(), t5.size());
        GT b = g.wf32(unet_pfx_ + "txt_in.bias");
        g.linear(t5g, Lt, 4096, g.w16(unet_pfx_ + "txt_in.weight"), D, &b, txt);
        g.free(t5g);
    }
    g.upload(pg, pooled.data(), 768);
    // noise in latent layout [16][hl][wl], packed to tokens (c*4 + i*2 + j over 2x2 cells)
    std::mt19937_64 rng(j.seed ? j.seed : std::random_device{}());
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> z((size_t)C * hl * wl), tok((size_t)Li * 64);
    for (auto& v : z) v = nd(rng);
    const int Wp = wl / 2;
    auto pack = [&](bool to_tok) {
        for (int y = 0; y < hl / 2; y++) for (int x = 0; x < Wp; x++) for (int c = 0; c < C; c++) for (int i = 0; i < 2; i++) for (int jj = 0; jj < 2; jj++) {
            float& a = tok[((size_t)y * Wp + x) * 64 + c * 4 + i * 2 + jj]; float& b = z[((size_t)c * hl + y * 2 + i) * wl + x * 2 + jj];
            if (to_tok) a = b; else b = a;
        }
    };
    pack(true);
    g.upload(xg, tok.data(), tok.size());
    // timesteps 1 -> 0; dev shifts them toward noise for bigger pictures
    std::vector<float> ts(j.steps + 1);
    for (int i = 0; i <= j.steps; i++) {
        double t = 1.0 - (double)i / j.steps;
        if (dev && t > 0 && t < 1) { double mu = 0.5 + (1.15 - 0.5) * (Li - 256) / (4096.0 - 256); t = std::exp(mu) / (std::exp(mu) + (1 / t - 1)); }
        ts[i] = (float)t;
    }
    for (int i = 0; i < j.steps; i++) {
        if (!progress("Generating", i, j.steps)) { err = "cancelled"; break; }
        const double ts0 = now_sec();
        flux_dit(xg, hl, wl, ts[i], guidance, txt, Lt, pg, vg);
        if (getenv("ITHACA_FLUX_T")) { g.sync(); fprintf(stderr, "[flux] step %d: %.1f s, %lld MB streamed so far\n", i, now_sec() - ts0, (long long)(g.streamed_bytes() >> 20)); }
        g.axpy(xg, vg, (size_t)Li * 64, ts[i + 1] - ts[i]);
    }
    fprintf(stderr, "[flux] %lld MB resident, %lld MB streamed, %.1f s\n", (long long)(g.resident_bytes() >> 20), (long long)(g.streamed_bytes() >> 20), now_sec() - T0);
    g.download(tok.data(), xg, tok.size());
    for (GT* t : {&txt, &pg, &xg, &vg}) g.free(*t);
    g.weights_end();
    if (!err.empty()) { g.trim(); return false; }
    progress("Decoding", j.steps, j.steps);
    pack(false);
    for (auto& v : z) v = v / 0.3611f + 0.1159f;
    g.trim();
    std::vector<float> img;
    vae_decode(z, hl, wl, img);
    g.trim();
    const int W = wl * 8, H = hl * 8;
    rgb.resize((size_t)W * H * 3);
    for (int c = 0; c < 3; c++) for (int p = 0; p < W * H; p++) rgb[(size_t)p * 3 + c] = (uint8_t)std::min(255.f, std::max(0.f, (img[(size_t)c * W * H + p] + 1.f) * 127.5f + 0.5f));
    return true;
}

// =============================================================================================
// PNG writer (zlib "stored" blocks, no external dependencies)
// =============================================================================================
static uint32_t crc32(const uint8_t* d, size_t n, uint32_t c = 0) {
    static uint32_t tab[256];
    static bool init = false;
    if (!init) { for (uint32_t i = 0; i < 256; i++) { uint32_t x = i; for (int k = 0; k < 8; k++) x = x & 1 ? 0xEDB88320 ^ (x >> 1) : x >> 1; tab[i] = x; } init = true; }
    c = ~c;
    for (size_t i = 0; i < n; i++) c = tab[(c ^ d[i]) & 255] ^ (c >> 8);
    return ~c;
}
bool write_png(const std::wstring& path, const uint8_t* rgb, int w, int h) {
    std::vector<uint8_t> raw;
    raw.reserve((size_t)(w * 3 + 1) * h);
    for (int y = 0; y < h; y++) {
        // "Up" filter makes the stored data more regular; cheap and lossless
        raw.push_back(y ? 2 : 0);
        for (int x = 0; x < w * 3; x++) raw.push_back(y ? (uint8_t)(rgb[(size_t)y * w * 3 + x] - rgb[(size_t)(y - 1) * w * 3 + x]) : rgb[(size_t)y * w * 3 + x]);
    }
    std::vector<uint8_t> z = {0x78, 0x01};
    uint32_t a = 1, b = 0;
    for (uint8_t v : raw) { a = (a + v) % 65521; b = (b + a) % 65521; }
    for (size_t off = 0; off < raw.size(); off += 65535) {
        size_t len = std::min<size_t>(65535, raw.size() - off);
        z.push_back(off + len == raw.size() ? 1 : 0);
        z.push_back(len & 255); z.push_back(len >> 8); z.push_back(~len & 255); z.push_back((~len >> 8) & 255);
        z.insert(z.end(), raw.begin() + off, raw.begin() + off + len);
    }
    uint32_t ad = (b << 16) | a;
    for (int k = 3; k >= 0; k--) z.push_back((ad >> (k * 8)) & 255);
    FILE* f = _wfopen(path.c_str(), L"wb");
    if (!f) return false;
    auto be = [&](uint32_t v) { uint8_t q[4] = {(uint8_t)(v >> 24), (uint8_t)(v >> 16), (uint8_t)(v >> 8), (uint8_t)v}; fwrite(q, 1, 4, f); };
    auto chunk = [&](const char* type, const std::vector<uint8_t>& d) {
        be((uint32_t)d.size()); fwrite(type, 1, 4, f); fwrite(d.data(), 1, d.size(), f);
        uint32_t c = crc32((const uint8_t*)type, 4); c = crc32(d.data(), d.size(), c); be(c);
    };
    const uint8_t sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    fwrite(sig, 1, 8, f);
    std::vector<uint8_t> ihdr = {(uint8_t)(w >> 24), (uint8_t)(w >> 16), (uint8_t)(w >> 8), (uint8_t)w, (uint8_t)(h >> 24), (uint8_t)(h >> 16), (uint8_t)(h >> 8), (uint8_t)h, 8, 2, 0, 0, 0};
    chunk("IHDR", ihdr);
    chunk("IDAT", z);
    chunk("IEND", {});
    fclose(f);
    return true;
}

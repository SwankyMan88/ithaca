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
        std::vector<std::wstring> dirs = {g_store_dir + L"\\tokenizers\\clip"};
        wchar_t buf[MAX_PATH];
        if (GetEnvironmentVariableW(L"USERPROFILE", buf, MAX_PATH)) {
            std::wstring u = buf;
            for (auto d : {L"\\Documents\\ComfyUI_windows_portable\\ComfyUI\\comfy\\sd1_tokenizer", L"\\ComfyUI\\comfy\\sd1_tokenizer", L"\\Desktop\\ComfyUI_windows_portable\\ComfyUI\\comfy\\sd1_tokenizer"})
                dirs.push_back(u + d);
        }
        std::string vj, mt;
        for (auto& d : dirs) { vj = read(d + L"\\vocab.json"); mt = read(d + L"\\merges.txt"); if (!vj.empty() && !mt.empty()) break; }
        if (vj.empty() || mt.empty()) { err = "CLIP tokenizer files not found (put vocab.json and merges.txt in tokenizers\\clip next to Ithaca.exe)"; return false; }
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
    if (unet_pfx_ == "?") {
        if (st_.has("double_blocks.0.img_attn.qkv.weight") || st_.has("model.diffusion_model.double_blocks.0.img_attn.qkv.weight"))
            err = "This is a Flux model. Flux support is coming next; for now pick an SD 1.5, SD 2.x or SDXL checkpoint.";
        else if (st_.has("model.diffusion_model.joint_blocks.0.x_block.attn.qkv.weight")) err = "This is an SD3 model, which isn't supported yet.";
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
    int C = 4;
    GT x = g.alloc((size_t)4 * h * w);
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
    { GT o = g.alloc((size_t)Cin * h * w); conv(x, 4, Cin, "decoder.conv_in", 3, o); g.free(x); x = o; C = Cin; }
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
// Sampling
// =============================================================================================
bool SdEngine::generate(const ImageJob& jb, std::vector<uint8_t>& rgb, std::string& err,
                        const std::function<bool(const char*, int, int)>& progress) {
    if (!loaded_) { err = "no image model loaded"; return false; }
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
    for (auto& v : x) v = nd(rng) * sig[0];

    // Leave room for activations; whatever doesn't fit stays in RAM and is streamed each step.
    int64_t act = (int64_t)lh * lw * 64 * 1024 * (info_.arch == "sdxl" ? 3 : 2) + (512ll << 20);
    int64_t budget = (int64_t)g.vram_free() - act;
    unet_begin(std::max<int64_t>(0, budget));
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
    for (int i = 0; i < j.steps; i++) {
        if (!progress("Generating", i, j.steps)) { err = "cancelled"; break; }
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
    }
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

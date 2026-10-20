#include "vision.h"
#include "quant.h"
#include <wincodec.h>
#include <algorithm>
#include <cmath>

// ---- image decode: Windows Imaging Component handles PNG, JPEG, GIF, BMP, TIFF, WebP/HEIC (with codecs) ----
static bool decode_rgb(const std::string& bytes, int W, int H, std::vector<uint8_t>& rgb, std::string& err) {
    HRESULT co = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    IWICImagingFactory* f = nullptr; IWICStream* s = nullptr; IWICBitmapDecoder* d = nullptr;
    IWICBitmapFrameDecode* fr = nullptr; IWICFormatConverter* cv = nullptr; IWICBitmapScaler* sc = nullptr;
    bool ok = false;
    do {
        if (FAILED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&f)))) { err = "image decoder unavailable"; break; }
        if (FAILED(f->CreateStream(&s)) || FAILED(s->InitializeFromMemory((BYTE*)bytes.data(), (DWORD)bytes.size()))) { err = "bad image data"; break; }
        if (FAILED(f->CreateDecoderFromStream(s, nullptr, WICDecodeMetadataCacheOnDemand, &d)) || FAILED(d->GetFrame(0, &fr))) { err = "unsupported image format"; break; }
        if (FAILED(f->CreateFormatConverter(&cv)) ||
            FAILED(cv->Initialize(fr, GUID_WICPixelFormat24bppRGB, WICBitmapDitherTypeNone, nullptr, 0, WICBitmapPaletteTypeCustom))) { err = "image conversion failed"; break; }
        // Fant = area averaging, close to an antialiased bilinear downscale
        if (FAILED(f->CreateBitmapScaler(&sc)) || FAILED(sc->Initialize(cv, W, H, WICBitmapInterpolationModeFant))) { err = "image resize failed"; break; }
        rgb.resize((size_t)W * H * 3);
        if (FAILED(sc->CopyPixels(nullptr, W * 3, (UINT)rgb.size(), rgb.data()))) { err = "image read failed"; break; }
        ok = true;
    } while (0);
    for (IUnknown* u : {(IUnknown*)sc, (IUnknown*)cv, (IUnknown*)fr, (IUnknown*)d, (IUnknown*)s, (IUnknown*)f}) if (u) u->Release();
    if (SUCCEEDED(co)) CoUninitialize();
    return ok;
}

bool Vision::load(const std::wstring& path, int llm_embd, std::string& err) {
    unload();
    if (!g_.open(path, err)) return false;
    if (g_.gets("general.architecture") != "clip") { err = "not a vision projector (mmproj) file"; unload(); return false; }
    kind = g_.gets("clip.projector_type");
    if (kind != "gemma3") { err = "vision projector '" + kind + "' isn't supported yet (supported: Gemma 3)"; unload(); return false; }
    img_ = (int)g_.geti("clip.vision.image_size", 896);
    patch_ = (int)g_.geti("clip.vision.patch_size", 14);
    dim_ = (int)g_.geti("clip.vision.embedding_length", 1152);
    ff_ = (int)g_.geti("clip.vision.feed_forward_length", 4304);
    layers_ = (int)g_.geti("clip.vision.block_count", 27);
    heads_ = (int)g_.geti("clip.vision.attention.head_count", 16);
    eps_ = (float)g_.getf("clip.vision.attention.layer_norm_epsilon", 1e-6);
    proj_dim_ = (int)g_.geti("clip.vision.projection_dim", 0);
    if (const GValue* m = g_.find("clip.vision.image_mean")) if (m->arr_n.size() == 3) for (int i = 0; i < 3; i++) mean_[i] = (float)m->arr_n[i];
    if (const GValue* m = g_.find("clip.vision.image_std")) if (m->arr_n.size() == 3) for (int i = 0; i < 3; i++) std_[i] = (float)m->arr_n[i];
    if (llm_embd && proj_dim_ != llm_embd) { err = "this vision file is for a different model size"; unload(); return false; }
    // Gemma 3: 64x64 patches average-pooled 4x4 -> 256 tokens
    n_out_ = 256;
    // Hand the tensors to the GPU runtime under torch-style shapes (GGUF lists dims fastest-first).
    for (auto& t : g_.tensors) {
        STensor s;
        for (int i = t.nd - 1; i >= 0; i--) s.shape.push_back(t.ne[i]);
        if (t.name == "mm.input_projection.weight") {
            // stored [in][out]; the runtime wants [out][in]
            int64_t in = t.ne[1], out = t.ne[0];
            std::vector<float> src((size_t)in * out);
            dequant_row(t.type, t.data, src.data(), (int64_t)src.size());
            owned_.emplace_back((size_t)in * out);
            float* dst = owned_.back().data();
            for (int64_t i = 0; i < in; i++) for (int64_t o = 0; o < out; o++) dst[o * in + i] = src[i * out + o];
            s.dtype = "F32"; s.shape = {out, in}; s.data = (const uint8_t*)dst; s.nbytes = (size_t)in * out * 4;
        } else if (t.type == GT_F16 || t.type == GT_F32) {
            s.dtype = t.type == GT_F16 ? "F16" : "F32"; s.data = t.data; s.nbytes = t.nbytes;
        } else {
            int64_t n = s.numel();
            owned_.emplace_back((size_t)n);
            dequant_row(t.type, t.data, owned_.back().data(), n);
            s.dtype = "F32"; s.data = (const uint8_t*)owned_.back().data(); s.nbytes = (size_t)n * 4;
        }
        st_.put(t.name, s);
    }
    const GTensor* f1 = g_.get("v.blk.0.ffn_down.weight");
    if (!f1 || !g_.get("v.patch_embd.weight") || !g_.get("mm.soft_emb_norm.weight")) { err = "vision file is missing tensors"; unload(); return false; }
    // some converters store fc1 under the name ffn_down: pick by shape (fc1 takes the model width in)
    if (f1->ne[0] == dim_) { fc1_ = "ffn_down"; fc2_ = "ffn_up"; } else { fc1_ = "ffn_up"; fc2_ = "ffn_down"; }
    path_utf8 = w_to_utf8(path);
    ok_ = true;
    return true;
}

void Vision::unload() {
    if (gx().current() == &st_) gx().weights_end();
    st_.close();
    owned_.clear();
    g_.close();
    ok_ = false;
}

bool Vision::encode(const std::string& file, std::vector<float>& out, std::string& err) {
    if (!ok_) { err = "no vision model loaded"; return false; }
    double t0 = now_sec();
    std::vector<uint8_t> rgb;
    if (!decode_rgb(file, img_, img_, rgb, err)) return false;
    const int P = img_ / patch_, NP = P * P, hd = dim_ / heads_;
    std::vector<float> chw((size_t)3 * img_ * img_);
    for (int c = 0; c < 3; c++)
        for (size_t i = 0; i < (size_t)img_ * img_; i++) chw[c * (size_t)img_ * img_ + i] = (rgb[i * 3 + c] / 255.f - mean_[c]) / std_[c];

    double td = now_sec();
    Gx& g = gx();
    if (!g.ok() && !g.init(err)) return false;
    double ti = now_sec();
    // Weights stay on the GPU between images; another engine taking the runtime frees them (weights_begin).
    if (g.current() != &st_) g.weights_begin(&st_, 1ll << 40);
    g.set_half(true);
    GT im = g.alloc(chw.size());
    g.upload(im, chw.data(), chw.size());
    GT pe = g.wf32("v.patch_embd.bias");
    GT cm = g.alloc((size_t)dim_ * NP), x = g.alloc((size_t)NP * dim_);
    g.conv(im, 3, img_, img_, g.w16("v.patch_embd.weight"), dim_, patch_, patch_, patch_, 0, &pe, cm);
    g.transpose(cm, dim_, NP, x);                 // [dim][patches] -> [patches][dim]
    g.free(cm); g.free(im);
    GT pos = g.wf32("v.position_embd.weight");
    g.axpy(x, pos, (size_t)NP * dim_);
    GT h = g.alloc((size_t)NP * dim_), q = g.alloc((size_t)NP * dim_), k = g.alloc((size_t)NP * dim_), v = g.alloc((size_t)NP * dim_),
       a = g.alloc((size_t)NP * dim_), f = g.alloc((size_t)NP * ff_);
    for (int i = 0; i < layers_; i++) {
        std::string b = "v.blk." + std::to_string(i) + ".";
        GT w1 = g.wf32(b + "ln1.weight"), b1 = g.wf32(b + "ln1.bias");
        g.layernorm(x, NP, dim_, w1, b1, eps_, h);
        GT bq = g.wf32(b + "attn_q.bias"), bk = g.wf32(b + "attn_k.bias"), bv = g.wf32(b + "attn_v.bias"), bo = g.wf32(b + "attn_out.bias");
        g.linear(h, NP, dim_, g.w16(b + "attn_q.weight"), dim_, &bq, q);
        g.linear(h, NP, dim_, g.w16(b + "attn_k.weight"), dim_, &bk, k);
        g.linear(h, NP, dim_, g.w16(b + "attn_v.weight"), dim_, &bv, v);
        g.attention(q, k, v, NP, NP, heads_, hd, a);
        g.linear(a, NP, dim_, g.w16(b + "attn_out.weight"), dim_, &bo, x, 1.f);       // x += attn
        GT w2 = g.wf32(b + "ln2.weight"), b2 = g.wf32(b + "ln2.bias");
        g.layernorm(x, NP, dim_, w2, b2, eps_, h);
        GT bf1 = g.wf32(b + fc1_ + ".bias"), bf2 = g.wf32(b + fc2_ + ".bias");
        g.linear(h, NP, dim_, g.w16(b + fc1_ + ".weight"), ff_, &bf1, f);
        g.act(f, (size_t)NP * ff_, Gx::GELU_TANH);
        g.linear(f, NP, ff_, g.w16(b + fc2_ + ".weight"), dim_, &bf2, x, 1.f);        // x += mlp
    }
    GT pw = g.wf32("v.post_ln.weight"), pb = g.wf32("v.post_ln.bias");
    g.layernorm(x, NP, dim_, pw, pb, eps_, h);
    std::vector<float> feat((size_t)NP * dim_);
    g.download(feat.data(), h, feat.size());
    double tc = now_sec();
    // Gemma 3 projector: 4x4 average pool over the patch grid, RMSNorm, linear into the LLM width
    const int S = 16, K = P / S;
    std::vector<float> pooled((size_t)S * S * dim_, 0.f), nw(dim_);
    const STensor* nt = st_.get("mm.soft_emb_norm.weight");
    for (int i = 0; i < dim_; i++) nw[i] = nt->dtype == "F32" ? ((const float*)nt->data)[i] : h2f(((const uint16_t*)nt->data)[i]);
    for (int py = 0; py < P; py++)
        for (int px = 0; px < P; px++) {
            float* dst = &pooled[((size_t)(py / K) * S + px / K) * dim_];
            const float* src = &feat[((size_t)py * P + px) * dim_];
            for (int c = 0; c < dim_; c++) dst[c] += src[c] / (K * K);
        }
    for (int t = 0; t < S * S; t++) {
        float* r = &pooled[(size_t)t * dim_]; double ss = 0;
        for (int c = 0; c < dim_; c++) ss += (double)r[c] * r[c];
        float sc = 1.f / std::sqrt((float)(ss / dim_) + eps_);
        for (int c = 0; c < dim_; c++) r[c] = r[c] * sc * nw[c];
    }
    GT pin = g.alloc(pooled.size()), po = g.alloc((size_t)S * S * proj_dim_);
    g.upload(pin, pooled.data(), pooled.size());
    g.linear(pin, S * S, dim_, g.w16("mm.input_projection.weight"), proj_dim_, nullptr, po);
    out.resize((size_t)S * S * proj_dim_);
    g.download(out.data(), po, out.size());
    for (GT* t : {&h, &q, &k, &v, &a, &f, &x, &pin, &po}) g.free(*t);
    g.set_half(false);
    g.trim();
    last_ms = (now_sec() - t0) * 1000;
    if (getenv("ITHACA_GXPROF")) g.print_profile();
    if (getenv("ITHACA_VTIME")) fprintf(stderr, "[vision] decode %.0f ms, gx init %.0f ms, tower %.0f ms, rest %.0f ms\n", (td - t0) * 1e3, (ti - td) * 1e3, (tc - ti) * 1e3, (now_sec() - tc) * 1e3);
    return true;
}

std::wstring find_mmproj(const std::wstring& model_path, int llm_embd) {
    size_t sl = model_path.find_last_of(L"\\/");
    std::wstring dir = sl == std::wstring::npos ? L"." : model_path.substr(0, sl);
    std::wstring mname = sl == std::wstring::npos ? model_path : model_path.substr(sl + 1);
    for (auto& c : mname) c = towlower(c);
    WIN32_FIND_DATAW fd;
    HANDLE h = FindFirstFileW((dir + L"\\*.gguf").c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) return L"";
    std::wstring best; size_t best_score = 0;
    do {
        std::wstring n = fd.cFileName, lo = n;
        for (auto& c : lo) c = towlower(c);
        if (lo.find(L"mmproj") == std::wstring::npos) continue;
        GGUF g; std::string err;
        if (!g.open(dir + L"\\" + n, err)) continue;
        if (g.geti("clip.vision.projection_dim", 0) != llm_embd) continue;
        size_t p = 0;   // shared name prefix ("gemma-3-4b-it...") breaks ties between projectors of the same width
        while (p < lo.size() && p < mname.size() && lo[p] == mname[p]) p++;
        if (best.empty() || p + 1 > best_score) { best = dir + L"\\" + n; best_score = p + 1; }
    } while (FindNextFileW(h, &fd));
    FindClose(h);
    return best;
}

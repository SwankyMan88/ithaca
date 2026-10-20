// Developer self-test: compares each stage of the image engine with reference tensors
// (raw float32 .bin files). Usage: Ithaca.exe --sdtest <checkpoint> <refdir>
#include "sd.h"
#include "json.h"
#include "model.h"
#include "quant.h"
#include "wan.h"
#include "tts.h"
#include <array>
#include <cmath>

static bool load_ref(const std::wstring& dir, const char* name, std::vector<float>& v) {
    FILE* f = _wfopen((dir + L"\\" + utf8_to_w(name) + L".bin").c_str(), L"rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    v.resize(n / 4);
    fread(v.data(), 4, v.size(), f);
    fclose(f);
    return true;
}
static void compare(const char* what, const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) { printf("  %-10s SIZE MISMATCH %zu vs %zu\n", what, a.size(), b.size()); return; }
    double num = 0, den = 0, maxd = 0;
    for (size_t i = 0; i < a.size(); i++) { double d = a[i] - b[i]; num += d * d; den += (double)b[i] * b[i]; maxd = std::max(maxd, std::fabs(d)); }
    double rel = std::sqrt(num / std::max(den, 1e-20));
    printf("  %-10s rel.err %.5f  max abs diff %.4f  %s\n", what, rel, maxd, rel < 0.02 ? "OK" : rel < 0.08 ? "close" : "WRONG");
}

int wan_selftest(const std::wstring& dit, const std::wstring& ref) {
    WanEngine& e = wan_engine();
    std::string err;
    if (!e.load(dit, L"", L"", err)) { printf("load failed: %s\n", err.c_str()); return 1; }
    printf("[wan] dim %d, heads %d, layers %d, ffn %d, latent channels %d\n", e.dim, e.heads, e.layers, e.ffn, e.zc);
    const char* pr = "a red fox running through fresh snow, cinematic";
    std::vector<int> ids;
    e.tokenize(pr, ids, err);
    printf("  tokens:"); for (int i : ids) printf(" %d", i); printf("\n");
    std::vector<float> r, ctx;
    double t0 = now_sec();
    if (getenv("WAN_SKIP_TE") == nullptr) {
        e.encode_text(pr, ctx, err);
        printf("  text encoder: %.1f s\n", now_sec() - t0);
        if (load_ref(ref, "cond", r)) compare("t5 cond", ctx, r);
    }
    std::vector<float> x, out, cr;
    if (load_ref(ref, "dit_x", x) && load_ref(ref, "dit_out", r) && load_ref(ref, "cond", cr)) {
        Gx& g = gx();
        e.dit_begin((int64_t)g.vram_free() - (2ll << 30), getenv("ITHACA_Q8") != nullptr);
        GT ce = e.text_embed(cr);
        t0 = now_sec();
        e.dit(x, 3, 8, 12, 700.f, ce, out);
        printf("  dit (288 tokens): %.2f s\n", now_sec() - t0);
        compare("dit out", out, r);
        g.free(ce);
        g.weights_end();
    }
    std::vector<float> z, img;
    if (load_ref(ref, "vae_z", z) && load_ref(ref, "vae_img", r)) {
        int F, H, W;
        t0 = now_sec();
        e.vae_decode(z, 3, 4, 6, img, F, H, W);
        printf("  vae: %d frames %dx%d in %.2f s\n", F, W, H, now_sec() - t0);
        compare("vae video", img, r);
    }
    return 0;
}

int wan_cli_generate(const std::wstring& dit, const std::string& prompt, const std::wstring& out, int w, int h, int frames, int steps) {
    WanEngine& e = wan_engine();
    std::string err;
    if (!e.load(dit, L"", L"", err)) { printf("load failed: %s\n", err.c_str()); return 1; }
    VideoJob j;
    j.prompt = prompt;
    j.negative = "blurry, low quality, distorted, static, worst quality, watermark, text";
    j.width = w; j.height = h; j.frames = frames; j.steps = steps; j.seed = 42;
    double t0 = now_sec(), tg = 0;
    std::vector<uint8_t> rgb;
    int W, H, F;
    bool ok = e.generate(j, rgb, W, H, F, err, [&](const char* ph, int s, int n) {
        if (s == 1 && !tg) tg = now_sec();
        printf("\r  %s %d/%d   (%.0f s)   ", ph, s, n, now_sec() - t0); fflush(stdout); return true; });
    if (!ok) { printf("\nfailed: %s\n", err.c_str()); return 1; }
    printf("\n  %d frames %dx%d in %.0f s total (%.1f s/step after the first)\n", F, W, H, now_sec() - t0, tg ? (now_sec() - tg) / std::max(1, steps - 1) : 0.0);
    if (!write_mp4(out, rgb, W, H, F, j.fps, err)) printf("mp4: %s\n", err.c_str());
    std::wstring png = out + L".mid.png";
    write_png(png, &rgb[(size_t)(F / 2) * W * H * 3], W, H);
    return 0;
}

int tts_selftest(const std::wstring& model, const std::wstring& ref) {
    TtsEngine& e = tts_engine();
    std::string err;
    if (!e.load(model, err)) { printf("load failed: %s\n", err.c_str()); return 1; }
    std::vector<float> wave, r, m;
    int frames = 0;
    if (load_ref(ref, "ref_wave", wave) && load_ref(ref, "ref_mel", r)) {
        e.mel(wave, m, frames);
        printf("  mel frames %d\n", frames);
        compare("mel", m, r);
    }
    std::vector<int> ids = e.text_ids("Some call me nature, others call me mother nature. Hello from Ithaca, your local assistant.");
    std::vector<float> rid;
    if (load_ref(ref, "text_ids", rid)) {
        size_t bad = 0; for (size_t i = 0; i < std::min(ids.size(), rid.size()); i++) bad += ids[i] != (int)rid[i];
        printf("  text ids: %zu vs %zu, %zu differ\n", ids.size(), rid.size(), bad);
    }
    std::vector<float> x, out;
    if (load_ref(ref, "dit_x", x) && load_ref(ref, "dit_out", r) && frames) {
        const int N = (int)(x.size() / 100), R = frames;
        std::vector<float> cond((size_t)N * 100, 0.f), z((size_t)N * 100, 0.f), te, tz;
        for (int p = 0; p < R && p < N; p++) for (int c = 0; c < 100; c++) cond[(size_t)p * 100 + c] = m[(size_t)c * R + p];
        e.dit_begin();
        e.text_embed(ids, N, te);
        tz.assign(te.size(), 0.f);
        extern std::map<std::string, std::vector<float>>* g_tts_dbg;
        std::map<std::string, std::vector<float>> dm;
        g_tts_dbg = &dm;
        if (load_ref(ref, "h_text", r)) compare("text emb", te, r);
        double t0 = now_sec();
        e.dit(x, cond, te, N, 0.3f, out);
        g_tts_dbg = nullptr;
        for (const char* nm : {"h_time", "h_input", "b0_norm", "b0_attn", "h_block0", "h_block21"}) if (load_ref(ref, nm, r) && dm.count(nm)) compare(nm, dm[nm], r);
        { FILE* f = _wfopen((ref + L"\\mine_b0_attn.bin").c_str(), L"wb"); if (f) { fwrite(dm["b0_attn"].data(), 4, dm["b0_attn"].size(), f); fclose(f); } }
        if (load_ref(ref, "dit_single", r)) compare("dit single", out, r);
        printf("  dit (%d frames): %.2f s\n", N, now_sec() - t0);
        compare("dit cond", out, std::vector<float>(r.begin(), r.begin() + out.size()));
        e.dit(x, z, tz, N, 0.3f, out);
        compare("dit uncond", out, std::vector<float>(r.begin() + out.size(), r.end()));
        e.end();
    }
    std::vector<float> w;
    if (load_ref(ref, "voc_wave", r) && frames) {
        e.voc_begin();
        e.vocos(m, frames, w);
        e.end();
        compare("vocos", w, r);
    }
    return 0;
}

int tts_cli(const std::wstring& model, const std::wstring& voice, const std::string& vtext, const std::string& text, const std::wstring& out) {
    TtsEngine& e = tts_engine();
    std::string err;
    if (!e.load(model, err)) { printf("load failed: %s\n", err.c_str()); return 1; }
    TtsJob j; j.text = text; j.voice_wav = voice; j.voice_text = vtext; j.seed = 1;
    std::vector<float> w;
    double t0 = now_sec();
    if (!e.speak(j, w, err, [](const char*, int s, int n) { printf("\r  %d/%d", s, n); fflush(stdout); return true; })) { printf("\nfailed: %s\n", err.c_str()); return 1; }
    double rms = 0; for (float v : w) rms += (double)v * v; rms = std::sqrt(rms / std::max<size_t>(1, w.size()));
    printf("\n  %.2f s of audio in %.1f s (rms %.3f)\n", w.size() / 24000.0, now_sec() - t0, rms);
    write_wav(out, w, 24000);
    return 0;
}

int sd_cli_generate(const std::wstring& ckpt, const std::string& prompt, const std::wstring& out, int steps, int size) {
    SdEngine& e = sd_engine();
    std::string err;
    if (!e.load(ckpt, L"", err)) { printf("load failed: %s\n", err.c_str()); return 1; }
    ImageJob j;
    j.prompt = prompt;
    j.negative = "blurry, low quality, deformed";
    j.width = j.height = size ? size : e.info().default_res;
    j.steps = steps;
    j.seed = 42;
    double t0 = now_sec(), tstep = 0;
    std::vector<uint8_t> rgb;
    bool ok = e.generate(j, rgb, err, [&](const char* ph, int s, int n) {
        if (s == 1) tstep = now_sec();
        printf("\r  %s %d/%d   ", ph, s, n); fflush(stdout); return true; });
    if (!ok) { printf("\nfailed: %s\n", err.c_str()); return 1; }
    double total = now_sec() - t0;
    printf("\n  %dx%d, %d steps in %.1f s (%.2f s/step)\n", j.width, j.height, steps, total, (now_sec() - tstep) / std::max(1, steps - 1));
    write_png(out, rgb.data(), j.width, j.height);
    return 0;
}

static void gemm_unit() {
    Gx& g = gx();
    std::string err;
    if (!g.init(err)) { printf("gx init: %s\n", err.c_str()); return; }
    printf("  tensor cores: %s\n", g.tensor_cores() ? "yes (TF32)" : "no");
    for (auto dims : {std::array<int, 3>{77, 768, 768}, {64, 64, 64}, {4096, 640, 640}, {77, 3072, 768}}) {
        int M = dims[0], N = dims[1], K = dims[2];
        std::vector<float> a((size_t)M * K), bias(N), ref((size_t)M * N), out((size_t)M * N);
        std::vector<uint16_t> w((size_t)N * K);
        uint32_t s = 1;
        auto rnd = [&] { s = s * 1664525 + 1013904223; return ((s >> 8) & 0xFFFF) / 32768.f - 1.f; };
        for (auto& v : a) v = rnd();
        for (auto& v : w) v = f2h(rnd());
        for (auto& v : bias) v = rnd();
        for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) {
            double acc = bias[n];
            for (int k = 0; k < K; k++) acc += (double)a[(size_t)m * K + k] * h2f(w[(size_t)n * K + k]);
            ref[(size_t)m * N + n] = (float)acc;
        }
        GT ag = g.alloc(a.size()), bg = g.alloc(N), og = g.alloc(out.size());
        uint64_t wg = Model::cuda().alloc(w.size() * 2);
        Model::cuda().htod(wg, w.data(), w.size() * 2);
        g.upload(ag, a.data(), a.size()); g.upload(bg, bias.data(), N);
        g.linear(ag, M, K, wg, N, &bg, og);
        g.download(out.data(), og, out.size());
        printf("  gemm %dx%dx%d: ", M, N, K);
        compare("lin", out, ref);
        Model::cuda().free(wg); g.free(ag); g.free(bg); g.free(og);
    }
}

int sd_selftest(const std::wstring& ckpt, const std::wstring& ref) {
    SdEngine& e = sd_engine();
    std::string err;
    gemm_unit();
    double t0 = now_sec();
    if (!e.load(ckpt, L"", err)) { printf("load failed: %s\n", err.c_str()); return 1; }
    printf("[%s] ctx %d, channels %d, adm %d  (%.0f ms)\n", e.info().arch.c_str(), e.info().ctx_dim, e.info().model_ch, e.info().adm, (now_sec() - t0) * 1000);
    std::vector<float> c, pooled, r;
    int nt;
    for (const char* nm : {"conditioner.embedders.0.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight",
                           "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_q.weight"}) {
        const STensor* t = e.tensors().get(nm);
        if (!t) continue;
        gx().weights_begin(&e.tensors(), 1ll << 40);
        uint64_t p = gx().w16(nm);
        std::vector<uint16_t> back(t->numel());
        Model::cuda().dtoh(back.data(), p, back.size() * 2);
        size_t bad = 0, first = SIZE_MAX;
        for (size_t i = 0; i < back.size(); i++) if (back[i] != ((const uint16_t*)t->data)[i]) { bad++; if (first == SIZE_MAX) first = i; }
        printf("  upload check %s: %s %zu/%zu differ (first at %zd)\n", nm + strlen(nm) - 30, t->dtype.c_str(), bad, back.size(), (ptrdiff_t)first);
        gx().weights_end();
    }
    {
        std::vector<float> o; std::vector<int> ids;
        const char* pr = "a photo of a cat sitting on a red sofa, highly detailed";
        e.debug_clipl(pr, 0, o, ids);
        printf("  tokens:"); for (int i = 0; i < 16; i++) printf(" %d", ids[i]); printf("\n");
        if (load_ref(ref, "clipl_emb", r)) compare("L emb", o, r);
        e.debug_clipl(pr, 1, o, ids);
        if (load_ref(ref, "clipl_layer1", r)) compare("L layer1", o, r);
        e.debug_clipl(pr, 11, o, ids);
        if (load_ref(ref, "clipl_l11", r)) compare("L layer11", o, r);
    }
    t0 = now_sec();
    if (!e.encode_text("a photo of a cat sitting on a red sofa, highly detailed", c, nt, pooled, err)) { printf("text: %s\n", err.c_str()); return 1; }
    printf("text encoders: %.2f s\n", now_sec() - t0);
    if (load_ref(ref, "cond", r)) {
        compare("cond", c, r);
        if (r.size() == c.size() && e.info().arch == "sdxl") {
            for (int part = 0; part < 2; part++)
                for (int t : {0, 1, 5, 14, 15, 40, 76}) {
                    std::vector<float> a, b;
                    for (int d = part ? 768 : 0; d < (part ? 2048 : 768); d++) { a.push_back(c[t * 2048 + d]); b.push_back(r[t * 2048 + d]); }
                    char nm[32]; snprintf(nm, sizeof nm, "%s t%d", part ? "G" : "L", t);
                    compare(nm, a, b);
                }
        }
    }
    if (load_ref(ref, "pooled", r)) compare("pooled", pooled, r);
    std::vector<float> y = e.adm_vector(pooled, 512, 512);
    if (load_ref(ref, "y", r) && e.info().adm) compare("adm", y, r);

    std::vector<float> x, eps;
    if (load_ref(ref, "unet_x", x) && load_ref(ref, "unet_eps", r)) {
        Gx& g = gx();
        std::vector<float> cr; load_ref(ref, "cond", cr);
        GT xg = g.alloc(x.size()), cg = g.alloc(cr.size()), yg = g.alloc(y.size()), og = g.alloc(x.size());
        std::vector<float> yr; load_ref(ref, "y", yr);
        g.upload(xg, x.data(), x.size()); g.upload(cg, cr.data(), cr.size()); g.upload(yg, yr.data(), yr.size());
        e.unet_begin(1ll << 40);
        e.unet(xg, 64, 64, 500.f, cg, 77, yg, og);
        g.sync();
        t0 = now_sec();
        e.unet(xg, 64, 64, 500.f, cg, 77, yg, og);
        eps.resize(x.size());
        g.download(eps.data(), og, eps.size());
        if (getenv("ITHACA_GXPROF")) { g.print_profile(); e.unet(xg, 64, 64, 500.f, cg, 77, yg, og); g.sync(); printf("  -- profile of one UNet call:\n"); g.print_profile(); }
        printf("unet 64x64 latent: %.2f s per call (weights resident %lld MB)\n", now_sec() - t0, (long long)(g.resident_bytes() >> 20));
        {   // full-size (1024x1024 image) timing
            std::vector<float> xb((size_t)4 * 128 * 128, 0.1f);
            GT xb_g = g.alloc(xb.size()), ob = g.alloc(xb.size());
            g.upload(xb_g, xb.data(), xb.size());
            e.unet(xb_g, 128, 128, 500.f, cg, 77, yg, ob); g.sync();
            double t1 = now_sec();
            e.unet(xb_g, 128, 128, 500.f, cg, 77, yg, ob); g.sync();
            printf("unet 128x128 latent (1024px): %.2f s per call\n", now_sec() - t1);
            if (getenv("ITHACA_GXPROF")) { g.print_profile(); e.unet(xb_g, 128, 128, 500.f, cg, 77, yg, ob); g.sync(); printf("  -- profile at 1024px:\n"); g.print_profile(); }
            g.free(xb_g); g.free(ob);
        }
        e.unet_end();
        compare("unet eps", eps, r);
        g.free(xg); g.free(cg); g.free(yg); g.free(og);
    }
    std::vector<float> z, img;
    if (load_ref(ref, "vae_z", z) && load_ref(ref, "vae_img", r)) {
        t0 = now_sec();
        e.vae_decode(z, 32, 32, img);
        printf("vae 256x256: %.2f s\n", now_sec() - t0);
        // reference is [H][W][3] in 0..1; ours is [3][H][W] in -1..1
        std::vector<float> mine(r.size());
        int HW = (int)(r.size() / 3);
        for (int p = 0; p < HW; p++) for (int ch = 0; ch < 3; ch++) mine[(size_t)p * 3 + ch] = std::min(1.f, std::max(0.f, (img[(size_t)ch * HW + p] + 1) / 2));
        compare("vae image", mine, r);
    }
    return 0;
}

// --fatest: fused flash attention vs. the score-matrix path, for each supported head size.
int fa_selftest() {
    Gx& g = gx(); std::string err;
    if (!g.init(err)) { printf("gx init: %s\n", err.c_str()); return 1; }
    for (int d : {64, 72, 128}) {
        const int T = 333, S = 517, H = 4, C = H * d;
        std::vector<float> q((size_t)T * C), k((size_t)S * C), v((size_t)S * C), a((size_t)T * C), b((size_t)T * C);
        uint32_t s = 7; auto rnd = [&] { s = s * 1664525 + 1013904223; return ((s >> 8) & 0xFFFF) / 32768.f - 1.f; };
        for (auto* vec : {&q, &k, &v}) for (auto& x : *vec) x = rnd();
        GT gq = g.alloc(q.size()), gk = g.alloc(k.size()), gv = g.alloc(v.size()), go = g.alloc(a.size());
        g.upload(gq, q.data(), q.size()); g.upload(gk, k.data(), k.size()); g.upload(gv, v.data(), v.size());
        _putenv("ITHACA_NO_FLASH=");
        g.attention(gq, gk, gv, T, S, H, d, go); g.download(a.data(), go, a.size());
        _putenv("ITHACA_NO_FLASH=1");
        g.attention(gq, gk, gv, T, S, H, d, go); g.download(b.data(), go, b.size());
        _putenv("ITHACA_NO_FLASH=");
        double num = 0, den = 0;
        for (size_t i = 0; i < a.size(); i++) { num += (a[i] - b[i]) * (a[i] - b[i]); den += b[i] * b[i]; }
        printf("  flash d=%-3d rel.err %.5f  %s\n", d, std::sqrt(num / den), std::sqrt(num / den) < 0.01 ? "OK" : "BAD");
        g.free(gq); g.free(gk); g.free(gv); g.free(go);
    }
    return 0;
}

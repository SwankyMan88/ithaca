// Media jobs served over HTTP: image generation (now), video/audio (engines to come).
#include "json.h"
#include "model.h"
#include "sd.h"
#include "tts.h"
#include "wan.h"
#include <algorithm>
#include <ctime>

extern std::wstring g_store_dir;

static std::mutex g_media;                 // one media job at a time
static std::mutex g_prog_m;
static std::string g_phase;
static int g_step = 0, g_total = 0;
static std::string g_preview;   // latest live preview as a BMP file
static int g_preview_n = 0;      // bumps on every new preview
static std::atomic<bool> g_media_busy{false}, g_media_cancel{false};

void chat_model_release();
void chat_model_restore();
// Frees the chat model's VRAM when a job needs more than is free; restores it when the job ends.
struct VramLease {
    bool released = false;
    explicit VramLease(int64_t need_mb) {
        size_t f = 0, t = 0;
        Model::cuda().mem_info(f, t);
        if ((int64_t)(f >> 20) < need_mb) { chat_model_release(); released = true; }
    }
    ~VramLease() { if (released) { gx().trim(); chat_model_restore(); } }
};

static std::string read_text(const std::wstring& p) {
    FILE* f = _wfopen(p.c_str(), L"rb");
    if (!f) return "";
    std::string s; char b[65536]; size_t r;
    while ((r = fread(b, 1, sizeof b, f)) > 0) s.append(b, r);
    fclose(f);
    return s;
}

std::string media_progress_json() {
    std::lock_guard<std::mutex> l(g_prog_m);
    return "{\"busy\":" + std::string(g_media_busy ? "true" : "false") + ",\"phase\":" + json_esc(g_phase) +
           ",\"step\":" + std::to_string(g_step) + ",\"total\":" + std::to_string(g_total) + ",\"preview\":" + std::to_string(g_preview_n) + "}";
}
std::string media_preview_bmp() { std::lock_guard<std::mutex> l(g_prog_m); return g_preview; }
static void set_preview(const std::vector<uint8_t>& rgb, int w, int h) {
    // 24-bit bottom-up BMP: every browser shows it, no encoder needed
    int row = (w * 3 + 3) & ~3; uint32_t sz = 54 + row * h;
    std::string b(sz, '\0');
    auto u32 = [&](int o, uint32_t v) { memcpy(&b[o], &v, 4); };
    b[0] = 'B'; b[1] = 'M'; u32(2, sz); u32(10, 54); u32(14, 40); u32(18, w); u32(22, h);
    b[26] = 1; b[28] = 24; u32(34, row * h);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            const uint8_t* s = &rgb[((size_t)(h - 1 - y) * w + x) * 3];
            char* d = &b[54 + (size_t)y * row + x * 3];
            d[0] = s[2]; d[1] = s[1]; d[2] = s[0];
        }
    std::lock_guard<std::mutex> l(g_prog_m);
    g_preview.swap(b); g_preview_n++;
}
void media_cancel() { g_media_cancel = true; }

// Returns JSON: {"url":..., "seed":..., "seconds":...} or {"error":...}
std::string media_generate_image(const J& req) {
    J prefs;
    json_parse(read_text(g_store_dir + L"\\prefs.json"), prefs);
    std::string model = req["model"].str(prefs["media_image"].str());
    if (model.empty()) return "{\"error\":" + json_esc("No image model is loaded or selected. Pick one in the Image studio (left sidebar).") + "}";
    std::lock_guard<std::mutex> lk(g_media);
    g_media_busy = true; g_media_cancel = false;
    struct Done { ~Done() { g_media_busy = false; } } done;
    VramLease lease(7000);
    SdEngine& e = sd_engine();
    std::string err;
    std::wstring vae = utf8_to_w(req["vae"].str(prefs["media_image_vae"].str()));
    if (!e.loaded() || e.path != utf8_to_w(model)) {
        { std::lock_guard<std::mutex> l(g_prog_m); g_phase = "Loading model"; g_step = 0; g_total = 0; }
        if (!e.load(utf8_to_w(model), vae, err)) return "{\"error\":" + json_esc("Could not load the image model: " + err) + "}";
    }
    ImageJob j;
    j.prompt = req["prompt"].str();
    j.negative = req["negative"].str(prefs["img_negative"].str("blurry, low quality, deformed, watermark, text"));
    int def = e.info().default_res;
    j.width = (int)req["width"].num(prefs["img_width"].num(def));
    j.height = (int)req["height"].num(prefs["img_height"].num(def));
    j.steps = std::max(1, std::min(150, (int)req["steps"].num(prefs["img_steps"].num(25))));
    j.cfg = (float)req["cfg"].num(prefs["img_cfg"].num(e.info().arch == "sdxl" ? 6.0 : 7.0));
    j.sampler = req["sampler"].str(prefs["img_sampler"].str("dpmpp_2m"));
    j.scheduler = req["scheduler"].str(prefs["img_scheduler"].str("karras"));
    double sd = req["seed"].num(-1);
    j.seed = sd >= 0 ? (uint64_t)sd : ((uint64_t)GetTickCount64() * 2654435761ull) & 0xFFFFFFFFull;
    if (j.prompt.empty()) return "{\"error\":\"empty prompt\"}";
    double t0 = now_sec();
    std::vector<uint8_t> rgb;
    e.on_preview = set_preview;
    { std::lock_guard<std::mutex> l(g_prog_m); g_preview.clear(); }
    bool ok = e.generate(j, rgb, err, [&](const char* ph, int s, int n) {
        std::lock_guard<std::mutex> l(g_prog_m);
        g_phase = ph; g_step = s; g_total = n;
        return !g_media_cancel.load();
    });
    if (!ok) return "{\"error\":" + json_esc(err == "cancelled" ? "Image generation was cancelled." : "Image generation failed: " + err) + "}";
    std::wstring dir = g_store_dir + L"\\outputs";
    CreateDirectoryW(dir.c_str(), nullptr);
    char name[64];
    time_t now = time(nullptr);
    tm lt; localtime_s(&lt, &now);
    snprintf(name, sizeof name, "img_%04d%02d%02d_%02d%02d%02d_%llu.png", lt.tm_year + 1900, lt.tm_mon + 1, lt.tm_mday, lt.tm_hour, lt.tm_min, lt.tm_sec,
             (unsigned long long)(j.seed % 100000));
    if (!write_png(dir + L"\\" + utf8_to_w(name), rgb.data(), j.width / 8 * 8, j.height / 8 * 8)) return "{\"error\":\"could not save the image\"}";
    // remember the prompt next to the image for the gallery
    FILE* f = _wfopen((dir + L"\\" + utf8_to_w(name) + L".json").c_str(), L"wb");
    if (f) {
        std::string meta = "{\"prompt\":" + json_esc(j.prompt) + ",\"negative\":" + json_esc(j.negative) + ",\"seed\":" + std::to_string(j.seed) +
                           ",\"steps\":" + std::to_string(j.steps) + ",\"cfg\":" + std::to_string(j.cfg) + ",\"model\":" + json_esc(model) + "}";
        fwrite(meta.data(), 1, meta.size(), f); fclose(f);
    }
    char out[256];
    snprintf(out, sizeof out, ",\"seed\":%llu,\"seconds\":%.1f,\"width\":%d,\"height\":%d}", (unsigned long long)j.seed, now_sec() - t0, j.width, j.height);
    return std::string("{\"url\":\"/outputs/") + name + "\"" + out;
}

std::string media_generate_video(const J& req) {
    J prefs;
    json_parse(read_text(g_store_dir + L"\\prefs.json"), prefs);
    std::string model = req["model"].str(prefs["media_video"].str());
    if (model.empty()) return "{\"error\":" + json_esc("No video model is loaded or selected. Pick one in the Video studio (left sidebar).") + "}";
    std::string lo = model; for (auto& c : lo) c = (char)tolower(c);
    if (lo.size() > 5 && lo.compare(lo.size() - 5, 5, ".gguf") == 0)
        return "{\"error\":" + json_esc("GGUF video models (like the Wan 2.2 14B files) aren't supported yet. Choose the Wan 2.2 TI2V 5B .safetensors model.") + "}";
    std::lock_guard<std::mutex> lk(g_media);
    g_media_busy = true; g_media_cancel = false;
    struct Done { ~Done() { g_media_busy = false; } } done;
    VramLease lease(11000);
    WanEngine& e = wan_engine();
    std::string err;
    if (!e.loaded() || e.path != utf8_to_w(model)) {
        { std::lock_guard<std::mutex> l(g_prog_m); g_phase = "Loading video model"; g_step = 0; g_total = 0; }
        if (!e.load(utf8_to_w(model), L"", L"", err)) return "{\"error\":" + json_esc("Could not load the video model: " + err) + "}";
    }
    VideoJob j;
    j.prompt = req["prompt"].str();
    if (j.prompt.empty()) return "{\"error\":\"empty prompt\"}";
    j.negative = req["negative"].str(prefs["vid_negative"].str("blurry, low quality, distorted, static, worst quality, jpeg artifacts, watermark, text, deformed"));
    j.width = (int)req["width"].num(prefs["vid_width"].num(640));
    j.height = (int)req["height"].num(prefs["vid_height"].num(352));
    double secs = req["seconds"].t == J::STR ? atof(req["seconds"].s.c_str()) : req["seconds"].num(prefs["vid_seconds"].num(2));
    j.fps = 24;
    j.frames = (int)req["frames"].num(std::max(1.0, secs) * 24 + 1);
    j.frames = std::min(j.frames, 241);
    j.steps = std::max(1, std::min(60, (int)req["steps"].num(prefs["vid_steps"].num(20))));
    j.cfg = (float)req["cfg"].num(prefs["vid_cfg"].num(5.0));
    double sd = req["seed"].num(-1);
    j.seed = sd >= 0 ? (uint64_t)sd : ((uint64_t)GetTickCount64() * 2654435761ull) & 0xFFFFFFFFull;
    double t0 = now_sec();
    std::vector<uint8_t> rgb;
    int W, H, F;
    bool ok = e.generate(j, rgb, W, H, F, err, [&](const char* ph, int s, int n) {
        std::lock_guard<std::mutex> l(g_prog_m);
        g_phase = ph; g_step = s; g_total = n;
        return !g_media_cancel.load();
    });
    if (!ok) return "{\"error\":" + json_esc(err == "cancelled" ? "Video generation was cancelled." : "Video generation failed: " + err) + "}";
    std::wstring dir = g_store_dir + L"\\outputs";
    CreateDirectoryW(dir.c_str(), nullptr);
    char name[64];
    time_t now = time(nullptr);
    tm lt; localtime_s(&lt, &now);
    snprintf(name, sizeof name, "vid_%04d%02d%02d_%02d%02d%02d.mp4", lt.tm_year + 1900, lt.tm_mon + 1, lt.tm_mday, lt.tm_hour, lt.tm_min, lt.tm_sec);
    if (!write_mp4(dir + L"\\" + utf8_to_w(name), rgb, W, H, F, j.fps, err)) return "{\"error\":" + json_esc("Could not save the video: " + err) + "}";
    FILE* f = _wfopen((dir + L"\\" + utf8_to_w(name) + L".json").c_str(), L"wb");
    if (f) {
        std::string meta = "{\"prompt\":" + json_esc(j.prompt) + ",\"seed\":" + std::to_string(j.seed) + ",\"model\":" + json_esc(model) + "}";
        fwrite(meta.data(), 1, meta.size(), f); fclose(f);
    }
    char out[256];
    snprintf(out, sizeof out, ",\"seed\":%llu,\"seconds\":%.1f,\"width\":%d,\"height\":%d,\"frames\":%d}", (unsigned long long)j.seed, now_sec() - t0, W, H, F);
    return std::string("{\"url\":\"/outputs/") + name + "\"" + out;
}

// ---- voices: voices\<name>.wav + voices\<name>.txt (the words spoken in the clip) ----
static std::wstring voices_dir() { std::wstring d = g_store_dir + L"\\voices"; CreateDirectoryW(d.c_str(), nullptr); return d; }
static bool safe_name(const std::string& n) {
    if (n.empty() || n.size() > 60) return false;
    for (char c : n) if (!(isalnum((unsigned char)c) || c == ' ' || c == '-' || c == '_')) return false;
    return true;
}
std::string media_voices_json() {
    std::string s = "[";
    WIN32_FIND_DATAW fd;
    HANDLE h = FindFirstFileW((voices_dir() + L"\\*.wav").c_str(), &fd);
    int n = 0;
    if (h != INVALID_HANDLE_VALUE) {
        do {
            std::wstring nm = fd.cFileName; nm.resize(nm.size() - 4);
            std::string txt = read_text(voices_dir() + L"\\" + nm + L".txt");
            s += (n++ ? "," : "") + std::string("{\"name\":") + json_esc(w_to_utf8(nm)) + ",\"text\":" + json_esc(txt) + ",\"url\":" + json_esc("/voices/" + w_to_utf8(fd.cFileName)) + "}";
        } while (FindNextFileW(h, &fd));
        FindClose(h);
    }
    return s + "]";
}
static int b64v(char c) { return c >= 'A' && c <= 'Z' ? c - 'A' : c >= 'a' && c <= 'z' ? c - 'a' + 26 : c >= '0' && c <= '9' ? c - '0' + 52 : c == '+' ? 62 : c == '/' ? 63 : -1; }
std::string media_voice_add(const J& req) {
    std::string name = req["name"].str(), text = req["text"].str(), b64 = req["wav"].str();
    if (!safe_name(name)) return "{\"error\":\"Use a short name with letters, numbers, spaces, - or _\"}";
    if (text.empty()) return "{\"error\":\"Type the exact words spoken in the clip\"}";
    size_t comma = b64.find(','); if (comma != std::string::npos && comma < 100) b64 = b64.substr(comma + 1);
    std::string bin; int val = 0, bits = -8;
    for (char c : b64) { int d = b64v(c); if (d < 0) continue; val = (val << 6) | d; bits += 6; if (bits >= 0) { bin += (char)((val >> bits) & 0xFF); bits -= 8; } }
    if (bin.size() < 44 || bin.compare(0, 4, "RIFF")) return "{\"error\":\"The clip must be a WAV file\"}";
    std::wstring base = voices_dir() + L"\\" + utf8_to_w(name);
    FILE* f = _wfopen((base + L".wav").c_str(), L"wb"); if (!f) return "{\"error\":\"could not save\"}";
    fwrite(bin.data(), 1, bin.size(), f); fclose(f);
    f = _wfopen((base + L".txt").c_str(), L"wb"); if (f) { fwrite(text.data(), 1, text.size(), f); fclose(f); }
    std::vector<float> w; int sr; std::string err;
    if (!read_wav(base + L".wav", w, sr, err)) { DeleteFileW((base + L".wav").c_str()); DeleteFileW((base + L".txt").c_str()); return "{\"error\":" + json_esc(err) + "}"; }
    return media_voices_json();
}
std::string media_voice_delete(const J& req) {
    std::string name = req["name"].str();
    if (safe_name(name)) { std::wstring b = voices_dir() + L"\\" + utf8_to_w(name); DeleteFileW((b + L".wav").c_str()); DeleteFileW((b + L".txt").c_str()); }
    return media_voices_json();
}
bool media_read_voice(const std::string& name, std::string& data) {
    if (name.find("..") != std::string::npos || name.find('/') != std::string::npos || name.find('\\') != std::string::npos) return false;
    data = read_text(voices_dir() + L"\\" + utf8_to_w(name));
    return !data.empty();
}

std::string media_generate_audio(const J& req) {
    J prefs;
    json_parse(read_text(g_store_dir + L"\\prefs.json"), prefs);
    std::string mode = req["mode"].str(req["kind2"].str("speech"));
    for (auto& c : mode) c = (char)tolower(c);
    if (mode.find("music") != std::string::npos || mode.find("sfx") != std::string::npos || mode.find("sound") != std::string::npos)
        return "{\"error\":" + json_esc("Music and sound-effect generation aren't available yet; only speech (voice cloning) is built in so far.") + "}";
    if (mode.find("transcri") != std::string::npos)
        return "{\"error\":" + json_esc("Speech-to-text isn't available yet (it needs a Whisper model).") + "}";
    std::string model = req["model"].str(prefs["media_audio"].str());
    if (model.empty()) return "{\"error\":" + json_esc("No speech model is loaded or selected. Pick one in the Audio studio (left sidebar).") + "}";
    std::string voice = req["voice"].str(prefs["voice"].str());
    if (voice.empty()) return "{\"error\":" + json_esc("No voice is set up yet. Add a short voice clip in the Audio studio and pick it as the default voice.") + "}";
    if (!safe_name(voice)) return "{\"error\":\"bad voice name\"}";
    std::wstring vb = voices_dir() + L"\\" + utf8_to_w(voice);
    std::string vtext = read_text(vb + L".txt");
    if (vtext.empty()) return "{\"error\":" + json_esc("The voice \"" + voice + "\" is missing its clip or transcript.") + "}";
    std::string text = req["prompt"].str(req["text"].str());
    if (text.empty()) return "{\"error\":\"nothing to say\"}";
    std::lock_guard<std::mutex> lk(g_media);
    g_media_busy = true; g_media_cancel = false;
    struct Done { ~Done() { g_media_busy = false; } } done;
    VramLease lease(2000);
    TtsEngine& e = tts_engine();
    std::string err;
    if (!e.loaded() || e.path != utf8_to_w(model)) {
        { std::lock_guard<std::mutex> l(g_prog_m); g_phase = "Loading speech model"; g_step = 0; g_total = 0; }
        if (!e.load(utf8_to_w(model), err)) return "{\"error\":" + json_esc("Could not load the speech model: " + err) + "}";
    }
    TtsJob j;
    j.text = text; j.voice_wav = vb + L".wav"; j.voice_text = vtext;
    j.speed = (float)req["speed"].num(prefs["tts_speed"].num(1.0));
    j.steps = std::max(8, std::min(64, (int)req["steps"].num(32)));
    double t0 = now_sec();
    std::vector<float> wave;
    bool ok = e.speak(j, wave, err, [&](const char* ph, int s, int n) {
        std::lock_guard<std::mutex> l(g_prog_m);
        g_phase = ph; g_step = s; g_total = n;
        return !g_media_cancel.load();
    });
    if (!ok) return "{\"error\":" + json_esc(err == "cancelled" ? "Speech generation was cancelled." : "Speech generation failed: " + err) + "}";
    std::wstring dir = g_store_dir + L"\\outputs";
    CreateDirectoryW(dir.c_str(), nullptr);
    char name[64];
    time_t now = time(nullptr);
    tm lt; localtime_s(&lt, &now);
    snprintf(name, sizeof name, "speech_%04d%02d%02d_%02d%02d%02d.wav", lt.tm_year + 1900, lt.tm_mon + 1, lt.tm_mday, lt.tm_hour, lt.tm_min, lt.tm_sec);
    write_wav(dir + L"\\" + utf8_to_w(name), wave, 24000);
    FILE* f = _wfopen((dir + L"\\" + utf8_to_w(name) + L".json").c_str(), L"wb");
    if (f) { std::string meta = "{\"prompt\":" + json_esc(text) + ",\"voice\":" + json_esc(voice) + "}"; fwrite(meta.data(), 1, meta.size(), f); fclose(f); }
    char out[200];
    snprintf(out, sizeof out, ",\"seconds\":%.1f,\"duration\":%.1f,\"seed\":0}", now_sec() - t0, wave.size() / 24000.0);
    return std::string("{\"url\":\"/outputs/") + name + "\"" + out;
}

std::string media_gallery_json() {
    std::vector<std::pair<std::wstring, FILETIME>> files;
    WIN32_FIND_DATAW fd;
    for (const wchar_t* pat : {L"*.png", L"*.mp4", L"*.wav"}) {
        HANDLE h = FindFirstFileW((g_store_dir + L"\\outputs\\" + pat).c_str(), &fd);
        if (h == INVALID_HANDLE_VALUE) continue;
        do files.push_back({fd.cFileName, fd.ftLastWriteTime}); while (FindNextFileW(h, &fd));
        FindClose(h);
    }
    std::sort(files.begin(), files.end(), [](auto& a, auto& b) { return CompareFileTime(&a.second, &b.second) > 0; });
    std::string s = "[";
    for (size_t i = 0; i < files.size() && i < 300; i++) {
        std::string meta = read_text(g_store_dir + L"\\outputs\\" + files[i].first + L".json");
        J m; json_parse(meta, m);
        std::wstring n = files[i].first;
        const char* kind = n.find(L".mp4") != std::wstring::npos ? "video" : n.find(L".wav") != std::wstring::npos ? "audio" : "image";
        s += (i ? "," : "") + std::string("{\"url\":\"/outputs/") + w_to_utf8(n) + "\",\"kind\":\"" + kind + "\",\"prompt\":" + json_esc(m["prompt"].str()) + "}";
    }
    return s + "]";
}

// Serves /outputs/<file>. Returns false when the name is not a plain file name.
bool media_read_output(const std::string& name, std::string& data, std::string& type) {
    if (name.empty() || name.find("..") != std::string::npos || name.find('/') != std::string::npos || name.find('\\') != std::string::npos) return false;
    data = read_text(g_store_dir + L"\\outputs\\" + utf8_to_w(name));
    if (data.empty()) return false;
    std::string lo = name; for (auto& c : lo) c = (char)tolower(c);
    type = lo.size() > 4 && lo.compare(lo.size() - 4, 4, ".png") == 0 ? "image/png" : lo.find(".mp4") != std::string::npos ? "video/mp4" :
           lo.find(".wav") != std::string::npos ? "audio/wav" : "application/octet-stream";
    return true;
}

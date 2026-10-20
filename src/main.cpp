// Ithaca: double-click to start. Serves a chat UI on http://127.0.0.1:7860 and opens your browser.
// CLI: Ithaca.exe --model path.gguf --prompt "hello" [-n 128] [--threads N] [--ctx N] [--vram MB] [--ram MB]
#include <winsock2.h>
#include <ws2tcpip.h>
#include <shellapi.h>
#include "chat.h"
#include "json.h"
#include <cstdio>
#include <algorithm>

extern const char* kIndexHtml;
extern std::wstring g_store_dir;
bool handle_store(const std::string& method, const std::string& path, const std::string& body, int& code, std::string& type, std::string& out);
bool handle_hub(const std::string& method, const std::string& path, const std::string& body, int& code, std::string& type, std::string& out);

static Model g_model;
static Vision g_vision;
static Chat g_chat(g_model, &g_vision);
static bool g_has_vision = false;   // an mmproj for the loaded model sits next to it

static std::string b64decode(const std::string& in) {
    static int T[256]; static bool init = false;
    if (!init) { for (int i = 0; i < 256; i++) T[i] = -1; const char* a = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"; for (int i = 0; i < 64; i++) T[(unsigned char)a[i]] = i; init = true; }
    size_t s = in.find(','); s = (in.compare(0, 5, "data:") == 0 && s != std::string::npos) ? s + 1 : 0;   // accept data: URLs
    std::string out; int v = 0, bits = 0;
    for (size_t i = s; i < in.size(); i++) {
        int c = T[(unsigned char)in[i]];
        if (c < 0) continue;
        v = (v << 6) | c; bits += 6;
        if (bits >= 8) { bits -= 8; out += (char)((v >> bits) & 0xFF); }
    }
    return out;
}
static std::vector<Msg> parse_msgs(const J& arr) {
    std::vector<Msg> msgs;
    for (auto& m : arr.a) {
        Msg x{m["role"].str("user"), m["content"].str(), {}};
        for (auto& im : m["images"].a) x.images.push_back(b64decode(im.str()));
        msgs.push_back(std::move(x));
    }
    return msgs;
}
static std::mutex g_gen;           // one generation / load at a time
static std::atomic<bool> g_stop{false};
std::atomic<bool> g_busy{false};
static std::string g_last_stats = "{}";
static std::mutex g_stats_m;
static LoadOpts g_opts;
static std::vector<std::wstring> g_extra_dirs;
static std::wstring g_cfg_path, g_exe_dir;

// ---------- config ----------
static void save_cfg() {
    std::string s = "{\"model\":" + json_esc(g_model.path_utf8) + ",\"threads\":" + std::to_string(g_opts.threads) +
                    ",\"ctx\":" + std::to_string(g_opts.ctx) + ",\"vram_mb\":" + std::to_string(g_opts.vram_mb) +
                    ",\"ram_mb\":" + std::to_string(g_opts.ram_mb) + ",\"dirs\":[";
    for (size_t i = 0; i < g_extra_dirs.size(); i++) s += (i ? "," : "") + json_esc(w_to_utf8(g_extra_dirs[i]));
    s += "]}";
    FILE* f = _wfopen(g_cfg_path.c_str(), L"wb");
    if (f) { fwrite(s.data(), 1, s.size(), f); fclose(f); }
}
static std::string load_cfg() {
    FILE* f = _wfopen(g_cfg_path.c_str(), L"rb");
    if (!f) return "";
    std::string txt; char b[4096]; size_t r;
    while ((r = fread(b, 1, sizeof b, f)) > 0) txt.append(b, r);
    fclose(f);
    J j; if (!json_parse(txt, j)) return "";
    g_opts.threads = (int)j["threads"].num(0); g_opts.ctx = (int)j["ctx"].num(4096);
    g_opts.vram_mb = (int64_t)j["vram_mb"].num(-1); g_opts.ram_mb = (int64_t)j["ram_mb"].num(-1);
    for (auto& d : j["dirs"].a) g_extra_dirs.push_back(utf8_to_w(d.str()));
    return j["model"].str();
}

// ---------- model discovery ----------
static void scan(const std::wstring& dir, int depth, std::vector<std::pair<std::wstring, uint64_t>>& out) {
    if (depth > 5) return;
    WIN32_FIND_DATAW fd;
    HANDLE h = FindFirstFileW((dir + L"\\*").c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) return;
    do {
        std::wstring n = fd.cFileName;
        if (n == L"." || n == L"..") continue;
        std::wstring p = dir + L"\\" + n;
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
            if (n[0] != L'.' && n != L"node_modules") scan(p, depth + 1, out);
            continue;
        }
        if (n.size() < 5 || _wcsicmp(n.c_str() + n.size() - 5, L".gguf") != 0) continue;
        std::wstring lo = n; for (auto& c : lo) c = towlower(c);
        if (lo.rfind(L"incomplete", 0) == 0 || lo.find(L"mmproj") != std::wstring::npos) continue;
        if (lo.find(L"-of-0000") != std::wstring::npos) continue;  // split shards
        out.push_back({p, ((uint64_t)fd.nFileSizeHigh << 32) | fd.nFileSizeLow});
    } while (FindNextFileW(h, &fd));
    FindClose(h);
}
static std::string models_json() {
    std::vector<std::wstring> dirs = {g_exe_dir + L"\\models"};
    wchar_t buf[MAX_PATH];
    if (GetEnvironmentVariableW(L"LOCALAPPDATA", buf, MAX_PATH)) dirs.push_back(std::wstring(buf) + L"\\nomic.ai\\GPT4All");
    if (GetEnvironmentVariableW(L"USERPROFILE", buf, MAX_PATH)) {
        std::wstring u = buf;
        for (auto s : {L"\\.lmstudio\\models", L"\\.cache\\lm-studio\\models", L"\\models", L"\\Documents\\models", L"\\Downloads"}) dirs.push_back(u + s);
    }
    for (auto& d : g_extra_dirs) dirs.push_back(d);
    std::vector<std::pair<std::wstring, uint64_t>> found;
    for (auto& d : dirs) scan(d, d.find(L"Downloads") != std::wstring::npos ? 4 : 0, found);
    std::sort(found.begin(), found.end());
    found.erase(std::unique(found.begin(), found.end()), found.end());
    std::string s = "[";
    for (size_t i = 0; i < found.size(); i++) {
        std::wstring name = found[i].first.substr(found[i].first.find_last_of(L'\\') + 1);
        s += (i ? "," : "") + std::string("{\"path\":") + json_esc(w_to_utf8(found[i].first)) + ",\"name\":" +
             json_esc(w_to_utf8(name)) + ",\"mb\":" + std::to_string(found[i].second >> 20) + "}";
    }
    return s + "]";
}

static bool do_load(const std::string& path, std::string& err) {
    std::lock_guard<std::mutex> l(g_gen);
    g_chat.reset();
    g_vision.unload();
    if (!g_model.load(utf8_to_w(path), g_opts, err)) return false;
    g_has_vision = !find_mmproj(utf8_to_w(path), g_model.hp.n_embd).empty();
    save_cfg();
    return true;
}

// VRAM handoff for media jobs: the chat model gives back its GPU memory while an image/video/speech
// job runs (Windows would otherwise page GPU memory over PCIe), then reloads in the background.
static std::string g_released_path;
static std::mutex g_release_m;
void chat_model_release() {
    std::lock_guard<std::mutex> rl(g_release_m);
    std::lock_guard<std::mutex> l(g_gen);
    if (!g_model.loaded() || g_model.gpu_bytes() == 0) return;
    g_released_path = g_model.path_utf8;
    g_chat.reset();
    g_model.unload();
}
void chat_model_restore() {
    std::lock_guard<std::mutex> rl(g_release_m);
    if (g_released_path.empty()) return;
    std::string p = g_released_path, err;
    g_released_path.clear();
    std::lock_guard<std::mutex> l(g_gen);
    if (!g_model.loaded()) g_model.load(utf8_to_w(p), g_opts, err);   // opens instantly; layers return to VRAM in the background
}

// ---------- HTTP ----------
static bool send_all(SOCKET s, const std::string& d) {
    size_t off = 0;
    while (off < d.size()) {
        int r = send(s, d.data() + off, (int)(d.size() - off), 0);
        if (r <= 0) return false;
        off += r;
    }
    return true;
}
static void respond(SOCKET s, int code, const std::string& type, const std::string& body) {
    std::string h = "HTTP/1.1 " + std::to_string(code) + (code == 200 ? " OK" : " Error") + "\r\nContent-Type: " + type +
                    "\r\nContent-Length: " + std::to_string(body.size()) + "\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n";
    send_all(s, h + body);
}

static void handle(SOCKET s) {
    std::string req;
    char buf[65536];
    size_t hdr_end = std::string::npos;
    while (hdr_end == std::string::npos) {
        int r = recv(s, buf, sizeof buf, 0);
        if (r <= 0) { closesocket(s); return; }
        req.append(buf, r);
        hdr_end = req.find("\r\n\r\n");
        if (req.size() > (1 << 20)) { closesocket(s); return; }
    }
    size_t clen = 0;
    {
        std::string lower = req.substr(0, hdr_end);
        for (auto& c : lower) c = (char)tolower(c);
        size_t p = lower.find("content-length:");
        if (p != std::string::npos) clen = strtoul(lower.c_str() + p + 15, nullptr, 10);
        // Reject cross-site requests: only our own page may drive the API.
        size_t o = lower.find("\r\norigin:");
        if (o != std::string::npos) {
            std::string origin = lower.substr(o + 9, lower.find("\r\n", o + 2) - o - 9);
            if (origin.find("127.0.0.1") == std::string::npos && origin.find("localhost") == std::string::npos) {
                respond(s, 403, "text/plain", "forbidden"); closesocket(s); return;
            }
        }
    }
    if (clen > (64u << 20)) { closesocket(s); return; }
    while (req.size() < hdr_end + 4 + clen) {
        int r = recv(s, buf, sizeof buf, 0);
        if (r <= 0) break;
        req.append(buf, r);
    }
    std::string body = req.substr(hdr_end + 4);
    std::string line = req.substr(0, req.find("\r\n"));
    std::string method = line.substr(0, line.find(' '));
    std::string path = line.substr(method.size() + 1, line.find(' ', method.size() + 1) - method.size() - 1);

    if (path == "/" || path == "/index.html") { respond(s, 200, "text/html; charset=utf-8", kIndexHtml); }
    else if (path == "/api/models") { respond(s, 200, "application/json", models_json()); }
    else if (path == "/api/status") {
        std::string st = g_model.tier_json();
        st.pop_back();
        {
            std::lock_guard<std::mutex> l(g_stats_m);
            st += ",\"busy\":" + std::string(g_busy ? "true" : "false") + ",\"stats\":" + g_last_stats +
                  ",\"model\":" + json_esc(g_model.path_utf8) + ",\"name\":" + json_esc(g_model.hp.name) +
                  ",\"opts\":{\"threads\":" + std::to_string(g_opts.threads) + ",\"ctx\":" + std::to_string(g_opts.ctx) +
                  ",\"vram_mb\":" + std::to_string(g_opts.vram_mb) + ",\"ram_mb\":" + std::to_string(g_opts.ram_mb) +
                  "},\"hw_threads\":" + std::to_string(std::thread::hardware_concurrency()) + ",\"vision\":" + (g_has_vision ? "true" : "false") + "}";
        }
        respond(s, 200, "application/json", st);
    }
    else if (path == "/api/count" && method == "POST") {
        // Tokens the conversation would occupy, for the context meter.
        J j; json_parse(body, j);
        std::vector<Msg> msgs;
        for (auto& m : j["messages"].a) msgs.push_back({m["role"].str("user"), m["content"].str()});
        int n = 0, ctx = 0;
        if (g_model.loaded()) { n = (int)g_model.tok.encode(g_chat.render(msgs), true).size(); ctx = g_model.ctx; }
        respond(s, 200, "application/json", "{\"tokens\":" + std::to_string(n) + ",\"ctx\":" + std::to_string(ctx) + "}");
    }
    else if (path == "/api/doc/extract" && method == "POST") {   // {name, data: base64} -> extracted text
        extern std::string doc_extract_json(const std::string&, const std::string&);
        J j; json_parse(body, j);
        respond(s, 200, "application/json", doc_extract_json(j["name"].str("file"), b64decode(j["data"].str())));
    }
    else if (path == "/api/stop" && method == "POST") { g_stop = true; respond(s, 200, "application/json", "{}"); }
    else if (path == "/api/adddir" && method == "POST") {
        J j; json_parse(body, j);
        std::string d = j["dir"].str();
        while (!d.empty() && (d.back() == '\\' || d.back() == '/' || d.back() == '"')) d.pop_back();
        if (!d.empty() && d[0] == '"') d.erase(0, 1);
        if (!d.empty()) { g_extra_dirs.push_back(utf8_to_w(d)); save_cfg(); }
        respond(s, 200, "application/json", models_json());
    }
    else if (path == "/api/load" && method == "POST") {
        J j; json_parse(body, j);
        g_opts.threads = (int)j["threads"].num(g_opts.threads);
        g_opts.ctx = (int)j["ctx"].num(g_opts.ctx);
        g_opts.vram_mb = (int64_t)j["vram_mb"].num((double)g_opts.vram_mb);
        g_opts.ram_mb = (int64_t)j["ram_mb"].num((double)g_opts.ram_mb);
        std::string p = j["path"].str(g_model.path_utf8);
        g_stop = true;
        std::string err;
        if (p.empty()) respond(s, 400, "application/json", "{\"error\":\"no model selected\"}");
        else if (do_load(p, err)) respond(s, 200, "application/json", g_model.tier_json());
        else respond(s, 400, "application/json", "{\"error\":" + json_esc(err) + "}");
    }
    else if (path == "/api/chat" && method == "POST") {
        J j; json_parse(body, j);
        std::vector<Msg> msgs = parse_msgs(j["messages"]);
        GenOpts o;
        o.temp = (float)j["temperature"].num(o.temp);
        o.top_p = (float)j["top_p"].num(o.top_p);
        o.top_k = (int)j["top_k"].num(o.top_k);
        o.rep_pen = (float)j["repeat_penalty"].num(o.rep_pen);
        o.max_tokens = (int)j["max_tokens"].num(o.max_tokens);
        send_all(s, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n");
        std::lock_guard<std::mutex> l(g_gen);
        if (!g_model.loaded()) { send_all(s, "data: {\"error\":\"no model loaded\"}\n\n"); closesocket(s); return; }
        g_stop = false; g_busy = true;
        GenStats st = g_chat.generate(msgs, o, [&](const std::string& t) {
            if (g_stop) return false;
            return send_all(s, "data: {\"t\":" + json_esc(t) + "}\n\n");
        });
        g_busy = false;
        char sb[512];
        snprintf(sb, sizeof sb, "{\"prompt\":%d,\"reused\":%d,\"gen\":%d,\"prefill_tps\":%.1f,\"tps\":%.2f}",
                 st.prompt_tokens, st.reused, st.gen_tokens, st.prefill_tps, st.gen_tps);
        { std::lock_guard<std::mutex> l2(g_stats_m); g_last_stats = sb; }
        if (!st.error.empty()) send_all(s, "data: {\"error\":" + json_esc(st.error) + "}\n\n");
        send_all(s, std::string("data: {\"done\":true,\"stats\":") + sb + "}\n\n");
    }
    else {
        int code; std::string type, out;
        if (handle_store(method, path, body, code, type, out) || handle_hub(method, path, body, code, type, out)) respond(s, code, type, out);
        else respond(s, 404, "text/plain", "not found");
    }
    shutdown(s, SD_SEND);
    closesocket(s);
}

static int serve(int port, bool open_browser) {
    WSADATA wd; WSAStartup(MAKEWORD(2, 2), &wd);
    SOCKET ls = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    sockaddr_in a{}; a.sin_family = AF_INET; a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    int bound = 0;
    for (int p = port; p < port + 20; p++) {
        a.sin_port = htons((u_short)p);
        if (bind(ls, (sockaddr*)&a, sizeof a) == 0) { bound = p; break; }
    }
    if (!bound || listen(ls, 16) != 0) { printf("could not open a local port\n"); return 1; }
    std::string url = "http://127.0.0.1:" + std::to_string(bound) + "/";
    printf("\n  Ithaca is running:  %s\n  (close this window to quit)\n\n", url.c_str());
    if (open_browser) ShellExecuteA(nullptr, "open", url.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
    for (;;) {
        SOCKET c = accept(ls, nullptr, nullptr);
        if (c == INVALID_SOCKET) continue;
        std::thread(handle, c).detach();
    }
}

int main() {
    SetConsoleOutputCP(CP_UTF8);
    int argc; wchar_t** wargv = CommandLineToArgvW(GetCommandLineW(), &argc);
    std::vector<std::string> args;
    for (int i = 0; i < argc; i++) args.push_back(w_to_utf8(wargv[i]));
    wchar_t exe[MAX_PATH]; GetModuleFileNameW(nullptr, exe, MAX_PATH);
    g_exe_dir = exe; g_exe_dir = g_exe_dir.substr(0, g_exe_dir.find_last_of(L'\\'));
    g_cfg_path = g_exe_dir + L"\\ithaca.json";
    g_store_dir = g_exe_dir;
    std::string last = load_cfg();

    std::string model, prompt, image;
    int n = 128, port = 7860; bool browser = true, wait = false, gpubench = false; float temp = 0.f;
    for (size_t i = 1; i < args.size(); i++) {
        auto nx = [&]() -> std::string { return i + 1 < args.size() ? args[++i] : ""; };
        const std::string& k = args[i];
        if (k == "--model" || k == "-m") model = nx();
        else if (k == "--prompt" || k == "-p") prompt = nx();
        else if (k == "--image") image = nx();
        else if (k == "-n") n = atoi(nx().c_str());
        else if (k == "--threads" || k == "-t") g_opts.threads = atoi(nx().c_str());
        else if (k == "--ctx" || k == "-c") g_opts.ctx = atoi(nx().c_str());
        else if (k == "--vram") g_opts.vram_mb = atoll(nx().c_str());
        else if (k == "--ram") g_opts.ram_mb = atoll(nx().c_str());
        else if (k == "--temp") temp = (float)atof(nx().c_str());
        else if (k == "--port") port = atoi(nx().c_str());
        else if (k == "--no-browser") browser = false;
        else if (k == "--wait") wait = true;
        else if (k == "--gpubench") { wait = true; gpubench = true; }
        else if (k == "--quanttest") { extern int quant_selftest(const std::wstring&); return quant_selftest(utf8_to_w(nx())); }
        else if (k == "--fatest") { extern int fa_selftest(); return fa_selftest(); }
        else if (k == "--extract") {   // --extract <file>: print the text Ithaca would read from a document
            extern std::string doc_extract_json(const std::string&, const std::string&);
            std::string p = nx(), b; FILE* f = _wfopen(utf8_to_w(p).c_str(), L"rb"); char buf[65536]; size_t r;
            if (f) { while ((r = fread(buf, 1, sizeof buf, f)) > 0) b.append(buf, r); fclose(f); }
            double t0 = now_sec();
            J j; json_parse(doc_extract_json(p, b), j);
            if (!j["error"].str().empty()) { printf("error: %s\n", j["error"].str().c_str()); return 1; }
            printf("%s: %d pages, %d chars in %.0f ms\n%s\n", j["kind"].str().c_str(), (int)j["pages"].num(0), (int)j["chars"].num(0), (now_sec() - t0) * 1000, j["text"].str().c_str());
            return 0;
        }
        else if (k == "--sdgen") {   // --sdgen <checkpoint> <prompt> <out.png> [steps] [size]
            extern int sd_cli_generate(const std::wstring&, const std::string&, const std::wstring&, int, int);
            std::string ck = nx(), pr = nx(), out = nx(), st = nx(), sz = nx();
            return sd_cli_generate(utf8_to_w(ck), pr, utf8_to_w(out), st.empty() ? 25 : atoi(st.c_str()), sz.empty() ? 0 : atoi(sz.c_str()));
        }
        else if (k == "--wangen") {   // --wangen <dit> <prompt> <out.mp4> [w h frames steps]
            extern int wan_cli_generate(const std::wstring&, const std::string&, const std::wstring&, int, int, int, int);
            std::string ck = nx(), pr = nx(), out = nx(), w = nx(), h = nx(), f = nx(), st = nx();
            return wan_cli_generate(utf8_to_w(ck), pr, utf8_to_w(out), w.empty() ? 832 : atoi(w.c_str()), h.empty() ? 480 : atoi(h.c_str()),
                                    f.empty() ? 49 : atoi(f.c_str()), st.empty() ? 20 : atoi(st.c_str()));
        }
        else if (k == "--ttsgen") {   // --ttsgen <model> <voice.wav> <voice text> <text> <out.wav>
            extern int tts_cli(const std::wstring&, const std::wstring&, const std::string&, const std::string&, const std::wstring&);
            std::string ck = nx(), vw = nx(), vt = nx(), tx = nx(), out = nx();
            return tts_cli(utf8_to_w(ck), utf8_to_w(vw), vt, tx, utf8_to_w(out));
        }
        else if (k == "--ttstest") {
            extern int tts_selftest(const std::wstring&, const std::wstring&);
            std::string ck = nx(), rd = nx();
            return tts_selftest(utf8_to_w(ck), utf8_to_w(rd));
        }
        else if (k == "--wantest") {
            extern int wan_selftest(const std::wstring&, const std::wstring&);
            std::string ck = nx(), rd = nx();
            return wan_selftest(utf8_to_w(ck), utf8_to_w(rd));
        }
        else if (k == "--sdtest") {
            extern int sd_selftest(const std::wstring&, const std::wstring&);
            std::string ck = nx(), rd = nx();
            return sd_selftest(utf8_to_w(ck), utf8_to_w(rd));
        }
        else if (k == "--help" || k == "-h") {
            printf("Ithaca [--model file.gguf] [--prompt text -n N] [--threads N] [--ctx N] [--vram MB|0] [--ram MB] [--port P] [--no-browser]\n");
            return 0;
        }
    }

    if (!prompt.empty()) {
        std::string err;
        double t0 = now_sec();
        if (!g_model.load(utf8_to_w(model.empty() ? last : model), g_opts, err)) { printf("load failed: %s\n", err.c_str()); return 1; }
        printf("[%s | %s | %d layers | %s | opened in %.0f ms]\n", g_model.hp.name.c_str(), g_model.hp.arch.c_str(),
               g_model.hp.n_layer, g_model.gpu_info.c_str(), (now_sec() - t0) * 1000);
        if (wait) { while (g_model.migrating()) Sleep(50); printf("[tiers settled after %.1f s]\n", now_sec() - t0); }
        if (gpubench) { g_model.bench_gpu(); g_model.unload(); return 0; }
        GenOpts o; o.max_tokens = n; o.temp = temp;
        Msg um{"user", prompt, {}};
        if (!image.empty()) {
            FILE* f = _wfopen(utf8_to_w(image).c_str(), L"rb"); std::string b; char buf[65536]; size_t r;
            if (f) { while ((r = fread(buf, 1, sizeof buf, f)) > 0) b.append(buf, r); fclose(f); }
            um.images.push_back(b);
        }
        GenStats st = g_chat.generate({um}, o, [](const std::string& t) { fwrite(t.data(), 1, t.size(), stdout); fflush(stdout); return true; });
        if (g_chat.image_ms > 0) printf("\n[image encoded in %.0f ms]", g_chat.image_ms);
        if (!st.error.empty()) printf("\nerror: %s\n", st.error.c_str());
        printf("\n\n[prompt %d tok @ %.1f tok/s | gen %d tok @ %.2f tok/s]\n%s\n", st.prompt_tokens, st.prefill_tps, st.gen_tokens, st.gen_tps,
               g_model.tier_json().c_str());
        g_model.unload();
        return 0;
    }

    std::string m = model.empty() ? last : model;
    if (!m.empty()) {
        std::string err;
        if (do_load(m, err)) printf("  loaded %s\n", m.c_str());
        else printf("  could not load %s: %s\n", m.c_str(), err.c_str());
    }
    return serve(port, browser);
}

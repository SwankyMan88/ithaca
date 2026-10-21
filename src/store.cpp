// Local persistence for the UI: chat sessions, memories, preferences (plain JSON files next to the exe),
// plus a user-triggered "run this Python block" helper.
#include "common.h"
#include "json.h"
#include <algorithm>

std::wstring g_store_dir;  // set by main: the exe folder

static std::string read_file(const std::wstring& p, bool& ok) {
    ok = false;
    FILE* f = _wfopen(p.c_str(), L"rb");
    if (!f) return "";
    std::string s; char b[65536]; size_t r;
    while ((r = fread(b, 1, sizeof b, f)) > 0) s.append(b, r);
    fclose(f);
    ok = true;
    return s;
}
static bool write_file(const std::wstring& p, const std::string& s) {
    std::wstring tmp = p + L".tmp";
    FILE* f = _wfopen(tmp.c_str(), L"wb");
    if (!f) return false;
    bool ok = fwrite(s.data(), 1, s.size(), f) == s.size();
    fclose(f);
    return ok && MoveFileExW(tmp.c_str(), p.c_str(), MOVEFILE_REPLACE_EXISTING);
}
static bool valid_id(const std::string& id) {
    if (id.empty() || id.size() > 64) return false;
    for (char c : id) if (!isalnum((unsigned char)c) && c != '-' && c != '_') return false;
    return true;
}
static std::wstring chats_dir() {
    std::wstring d = g_store_dir + L"\\chats";
    CreateDirectoryW(d.c_str(), nullptr);
    return d;
}

static std::string run_python(const std::string& code) {
    wchar_t tmpdir[MAX_PATH]; GetTempPathW(MAX_PATH, tmpdir);
    std::wstring dir = std::wstring(tmpdir) + L"ithaca_run";
    CreateDirectoryW(dir.c_str(), nullptr);
    std::wstring file = dir + L"\\snippet.py";
    if (!write_file(file, code)) return "could not write temp file";
    SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
    HANDLE rd, wr;
    if (!CreatePipe(&rd, &wr, &sa, 0)) return "pipe failed";
    SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);
    STARTUPINFOW si{}; si.cb = sizeof si; si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = wr; si.hStdError = wr; si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    PROCESS_INFORMATION pi{};
    std::wstring cmd = L"python -I -X utf8 \"" + file + L"\"";
    std::vector<wchar_t> cl(cmd.begin(), cmd.end()); cl.push_back(0);
    if (!CreateProcessW(nullptr, cl.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW | CREATE_SUSPENDED, nullptr, dir.c_str(), &si, &pi)) {
        CloseHandle(rd); CloseHandle(wr);
        return "Python was not found on PATH. Install Python to run code blocks.";
    }
    extern HANDLE g_job;   // dies with Ithaca
    if (g_job) AssignProcessToJobObject(g_job, pi.hProcess);
    ResumeThread(pi.hThread);
    CloseHandle(wr);
    std::string out; char b[4096]; DWORD n;
    double t0 = now_sec();
    for (;;) {
        DWORD avail = 0;
        if (!PeekNamedPipe(rd, nullptr, 0, nullptr, &avail, nullptr)) break;
        if (avail) { if (!ReadFile(rd, b, std::min<DWORD>(avail, sizeof b), &n, nullptr) || !n) break; out.append(b, n); }
        else if (WaitForSingleObject(pi.hProcess, 20) == WAIT_OBJECT_0) {
            while (PeekNamedPipe(rd, nullptr, 0, nullptr, &avail, nullptr) && avail && ReadFile(rd, b, std::min<DWORD>(avail, sizeof b), &n, nullptr) && n) out.append(b, n);
            break;
        }
        if (now_sec() - t0 > 30) { TerminateProcess(pi.hProcess, 1); out += "\n[stopped after 30 s]"; break; }
        if (out.size() > (256 << 10)) { TerminateProcess(pi.hProcess, 1); out += "\n[output truncated]"; break; }
    }
    DWORD code_ = 0; GetExitCodeProcess(pi.hProcess, &code_);
    CloseHandle(pi.hProcess); CloseHandle(pi.hThread); CloseHandle(rd);
    if (code_ && code_ != STILL_ACTIVE) out += "\n[exit code " + std::to_string(code_) + "]";
    return out;
}

// ---- media model discovery (image / video / audio / vision) by file name ----
static const char* classify_media(std::wstring lo) {
    for (auto& c : lo) c = towlower(c);
    auto has = [&](const wchar_t* k) { return lo.find(k) != std::wstring::npos; };
    if (has(L"vae") || has(L"text_encoder") || has(L"t5xxl") || has(L"clip_l") || has(L"umt5")) return "skip";
    if (has(L"mmproj") || has(L"clip_vision") || has(L"llava")) return "vision";
    if (has(L"foley")) return "audio";
    for (auto k : {L"wan2", L"wan_", L"ltx", L"hunyuan_video", L"hunyuanvideo", L"cogvideo", L"mochi", L"svd", L"animatediff", L"motion_module"}) if (has(k)) return "video";
    for (auto k : {L"whisper", L"tts", L"kokoro", L"chatterbox", L"bark", L"musicgen", L"s3gen", L"t3_cfg", L"piper", L"audioldm", L"stable-audio", L"stable_audio", L"rvc", L"xtts", L"f5", L"voice"}) if (has(k)) return "audio";
    for (auto k : {L"flux", L"sdxl", L"sd_xl", L"stable-diffusion", L"stable_diffusion", L"sd3", L"sd15", L"v1-5", L"sd_1", L"sd-1", L"pixart", L"kolors", L"playground", L"juggernaut", L"realvis", L"dreamshaper", L"pony", L"illustrious", L"hidream", L"qwen-image", L"qwen_image"}) if (has(k)) return "image";
    return nullptr;
}
static void scan_media(const std::wstring& dir, int depth, const char* hint, std::string& out, int& n) {
    if (depth > 6) return;
    WIN32_FIND_DATAW fd;
    HANDLE h = FindFirstFileW((dir + L"\\*").c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) return;
    do {
        std::wstring nm = fd.cFileName;
        if (nm == L"." || nm == L"..") continue;
        std::wstring p = dir + L"\\" + nm;
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
            std::wstring lo = nm; for (auto& c : lo) c = towlower(c);
            if (lo[0] == L'.' || lo == L"node_modules" || lo == L"__pycache__" || lo == L"loras" || lo == L"vae_approx" || lo == L"upscale_models") continue;
            const char* sub = hint;  // ComfyUI folder names tell us the model type
            if (lo == L"checkpoints" || lo == L"diffusion_models" || lo == L"unet") sub = "image";
            scan_media(p, depth + 1, sub, out, n);
            continue;
        }
        std::wstring lo = nm; for (auto& c : lo) c = towlower(c);
        bool ext = false;
        for (auto e : {L".safetensors", L".gguf", L".ckpt", L".pth", L".onnx", L".bin"})
            if (lo.size() > wcslen(e) && lo.compare(lo.size() - wcslen(e), wcslen(e), e) == 0) ext = true;
        uint64_t sz = ((uint64_t)fd.nFileSizeHigh << 32) | fd.nFileSizeLow;
        if (!ext || sz < (50ull << 20)) continue;
        const char* kind = classify_media(nm);
        if (!kind) kind = hint;
        // F5-TTS checkpoints sit next to their vocab.txt
        if (!kind && lo.size() > 12 && lo.compare(lo.size() - 12, 12, L".safetensors") == 0 &&
            GetFileAttributesW((dir + L"\\vocab.txt").c_str()) != INVALID_FILE_ATTRIBUTES) kind = "audio";
        if (!kind || !strcmp(kind, "skip")) continue;
        if (!strcmp(kind, "image") && lo.size() > 5 && lo.compare(lo.size() - 5, 5, L".gguf") == 0 && !classify_media(nm)) continue;  // LLM GGUFs
        out += (n++ ? "," : "") + std::string("{\"path\":") + json_esc(w_to_utf8(p)) + ",\"name\":" + json_esc(w_to_utf8(nm)) +
               ",\"mb\":" + std::to_string(sz >> 20) + ",\"kind\":\"" + kind + "\"}";
    } while (FindNextFileW(h, &fd));
    FindClose(h);
}
static std::string media_models_json() {
    std::string out = "[";
    int n = 0;
    scan_media(g_store_dir + L"\\models", 0, nullptr, out, n);
    {   // folders the user added in a studio
        bool ok; std::string pr = read_file(g_store_dir + L"\\prefs.json", ok);
        J p; json_parse(pr, p);
        for (auto& d : p["media_dirs"].a) scan_media(utf8_to_w(d.str()), 0, nullptr, out, n);
    }
    wchar_t buf[MAX_PATH];
    if (GetEnvironmentVariableW(L"USERPROFILE", buf, MAX_PATH)) {
        std::wstring u = buf;
        for (auto d : {L"\\Documents", L"\\Desktop", L"\\models"}) {
            // ComfyUI installs keep everything under .../ComfyUI/models
            WIN32_FIND_DATAW fd;
            HANDLE h = FindFirstFileW((u + d + L"\\*").c_str(), &fd);
            if (h == INVALID_HANDLE_VALUE) continue;
            do {
                std::wstring nm = fd.cFileName, lo = nm;
                for (auto& c : lo) c = towlower(c);
                if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) || nm[0] == L'.') continue;
                std::wstring base = u + d + L"\\" + nm;
                if (lo.find(L"comfy") != std::wstring::npos) {
                    scan_media(base + L"\\ComfyUI\\models", 0, nullptr, out, n);
                    scan_media(base + L"\\models", 0, nullptr, out, n);
                } else if (lo.find(L"model") != std::wstring::npos || lo.find(L"ai") != std::wstring::npos) {
                    scan_media(base, 2, nullptr, out, n);
                }
            } while (FindNextFileW(h, &fd));
            FindClose(h);
        }
    }
    return out + "]";
}

// Returns true when the request was handled.
bool handle_store(const std::string& method, const std::string& path, const std::string& body, int& code, std::string& type, std::string& out) {
    code = 200; type = "application/json";
    auto simple = [&](const wchar_t* name, const char* def) {
        std::wstring p = g_store_dir + L"\\" + name;
        if (method == "POST") {
            J j;
            if (!json_parse(body, j)) { code = 400; out = "{\"error\":\"bad json\"}"; return; }
            out = write_file(p, body) ? "{}" : "{\"error\":\"write failed\"}";
        } else {
            bool ok; out = read_file(p, ok);
            if (!ok || out.empty()) out = def;
        }
    };
    if (path == "/api/memories") { simple(L"memories.json", "[]"); return true; }
    if (path == "/api/prefs") { simple(L"prefs.json", "{}"); return true; }
    if (path == "/api/chats" && method == "GET") {
        std::vector<std::pair<double, std::string>> rows;
        WIN32_FIND_DATAW fd;
        HANDLE h = FindFirstFileW((chats_dir() + L"\\*.json").c_str(), &fd);
        if (h != INVALID_HANDLE_VALUE) {
            do {
                bool ok; std::string s = read_file(chats_dir() + L"\\" + fd.cFileName, ok);
                J j; if (!ok || !json_parse(s, j)) continue;
                std::string id = w_to_utf8(fd.cFileName); id.resize(id.size() - 5);
                double upd = j["updated"].num(0);
                rows.push_back({upd, "{\"id\":" + json_esc(id) + ",\"title\":" + json_esc(j["title"].str("New chat")) +
                                         ",\"updated\":" + std::to_string((long long)upd) + "}"});
            } while (FindNextFileW(h, &fd));
            FindClose(h);
        }
        std::sort(rows.begin(), rows.end(), [](auto& a, auto& b) { return a.first > b.first; });
        out = "[";
        for (size_t i = 0; i < rows.size(); i++) out += (i ? "," : "") + rows[i].second;
        out += "]";
        return true;
    }
    if (path.rfind("/api/chats/", 0) == 0) {
        std::string id = path.substr(11);
        bool del = false;
        if (id.size() > 7 && id.compare(id.size() - 7, 7, "/delete") == 0) { del = true; id.resize(id.size() - 7); }
        if (!valid_id(id)) { code = 400; out = "{\"error\":\"bad id\"}"; return true; }
        std::wstring p = chats_dir() + L"\\" + utf8_to_w(id) + L".json";
        if (del) { DeleteFileW(p.c_str()); out = "{}"; }
        else if (method == "POST") {
            J j;
            if (!json_parse(body, j)) { code = 400; out = "{\"error\":\"bad json\"}"; }
            else out = write_file(p, body) ? "{}" : "{\"error\":\"write failed\"}";
        } else {
            bool ok; out = read_file(p, ok);
            if (!ok) { code = 404; out = "{\"error\":\"not found\"}"; }
        }
        return true;
    }
    if (path == "/api/media/models") { out = media_models_json(); return true; }
    if (path == "/api/voices") { extern std::string media_voices_json(); out = media_voices_json(); return true; }
    if (path == "/api/voices/add" && method == "POST") { extern std::string media_voice_add(const J&); J j; json_parse(body, j); out = media_voice_add(j); return true; }
    if (path == "/api/voices/delete" && method == "POST") { extern std::string media_voice_delete(const J&); J j; json_parse(body, j); out = media_voice_delete(j); return true; }
    if (path.rfind("/voices/", 0) == 0) {
        extern bool media_read_voice(const std::string&, std::string&);
        if (media_read_voice(path.substr(8), out)) type = "audio/wav"; else { code = 404; type = "text/plain"; out = "not found"; }
        return true;
    }
    if (path == "/api/media/adddir" && method == "POST") {
        J j; json_parse(body, j);
        std::string d = j["dir"].str();
        while (!d.empty() && (d.back() == '\\' || d.back() == '/' || d.back() == '"')) d.pop_back();
        if (!d.empty() && d[0] == '"') d.erase(0, 1);
        bool ok; std::string pr = read_file(g_store_dir + L"\\prefs.json", ok);
        J p; json_parse(pr.empty() ? "{}" : pr, p);
        std::string dirs = "[";
        int n = 0;
        for (auto& x : p["media_dirs"].a) dirs += (n++ ? "," : "") + json_esc(x.str());
        if (!d.empty()) dirs += (n++ ? "," : "") + json_esc(d);
        dirs += "]";
        // rewrite prefs with the new list (keep everything else as-is)
        std::string outp = "{";
        int k = 0;
        for (auto& kv : p.o) {
            if (kv.first == "media_dirs") continue;
            std::string v;
            if (kv.second.t == J::STR) v = json_esc(kv.second.s);
            else if (kv.second.t == J::NUM) { char b[64]; snprintf(b, sizeof b, "%.10g", kv.second.n); v = b; }
            else if (kv.second.t == J::BOOL) v = kv.second.b ? "true" : "false";
            else continue;
            outp += (k++ ? "," : "") + json_esc(kv.first) + ":" + v;
        }
        outp += std::string(k ? "," : "") + "\"media_dirs\":" + dirs + "}";
        write_file(g_store_dir + L"\\prefs.json", outp);
        out = media_models_json();
        return true;
    }
    if (path.rfind("/api/media/preview", 0) == 0) {
        extern std::string media_preview_bmp();
        out = media_preview_bmp(); type = "image/bmp";
        if (out.empty()) { code = 404; type = "text/plain"; }
        return true;
    }
    if (path == "/api/media/progress") { extern std::string media_progress_json(); out = media_progress_json(); return true; }
    if (path == "/api/media/cancel") { extern void media_cancel(); media_cancel(); out = "{}"; return true; }
    if (path == "/api/media/gallery") { extern std::string media_gallery_json(); out = media_gallery_json(); return true; }
    if (path.rfind("/outputs/", 0) == 0) {
        extern bool media_read_output(const std::string&, std::string&, std::string&);
        if (!media_read_output(path.substr(9), out, type)) { code = 404; type = "text/plain"; out = "not found"; }
        return true;
    }
    if (path == "/api/media/generate" && method == "POST") {
        // The chat model calls this through tool tags. Engines for these model types are not built yet,
        // so report clearly what is missing; the chat model rephrases it for the user.
        J j; json_parse(body, j);
        std::string kind = j["kind"].str("image");
        if (kind != "image" && kind != "video" && kind != "audio") kind = "image";
        if (kind == "image") {
            extern std::string media_generate_image(const J&);
            out = media_generate_image(j);
            return true;
        }
        if (kind == "video") {
            extern std::string media_generate_video(const J&);
            out = media_generate_video(j);
            return true;
        }
        if (kind == "audio") {
            extern std::string media_generate_audio(const J&);
            out = media_generate_audio(j);
            return true;
        }
        bool ok; std::string pr = read_file(g_store_dir + L"\\prefs.json", ok);
        J p; json_parse(pr, p);
        std::string sel = p[("media_" + kind).c_str()].str();
        if (sel.empty()) out = "{\"error\":" + json_esc("No " + kind + " model is loaded or selected. Pick one in the " + std::string(1, (char)toupper(kind[0])) + kind.substr(1) + " studio (left sidebar).") + "}";
        else out = "{\"error\":" + json_esc("A " + kind + " model is selected, but this version of Ithaca cannot run " + kind + " models yet. The " + kind + " engine is planned for a future update.") + "}";
        return true;
    }
    if (path == "/api/run" && method == "POST") {
        J j; json_parse(body, j);
        out = "{\"output\":" + json_esc(run_python(j["code"].str())) + "}";
        return true;
    }
    return false;
}

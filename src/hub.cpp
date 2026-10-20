// Model hub and system page: the GPT4All model catalog and Hugging Face search, a resumable
// downloader (WinHTTP, nothing to install), a hardware report and a benchmark of this PC.
#include "common.h"
#include "json.h"
#include "model.h"
#include "gx.h"
#include <winhttp.h>
#include <winioctl.h>
#include <intrin.h>
#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <map>

extern std::wstring g_store_dir;
extern std::atomic<bool> g_busy;

// ---------------------------------------------------------------------------------------------
// HTTPS
struct Url { std::wstring host, path; INTERNET_PORT port = 443; bool tls = true; };
static bool parse_url(const std::string& u, Url& o) {
    URL_COMPONENTS c{}; c.dwStructSize = sizeof c;
    wchar_t host[256], path[4096];
    c.lpszHostName = host; c.dwHostNameLength = 256; c.lpszUrlPath = path; c.dwUrlPathLength = 4096;
    wchar_t extra[4096]; c.lpszExtraInfo = extra; c.dwExtraInfoLength = 4096;
    std::wstring w = utf8_to_w(u);
    if (!WinHttpCrackUrl(w.c_str(), 0, 0, &c)) return false;
    o.host = std::wstring(host, c.dwHostNameLength);
    o.path = std::wstring(path, c.dwUrlPathLength) + std::wstring(extra, c.dwExtraInfoLength);
    o.port = c.nPort; o.tls = c.nScheme == INTERNET_SCHEME_HTTPS;
    return true;
}
struct Http {
    HINTERNET s = 0, c = 0, r = 0;
    ~Http() { if (r) WinHttpCloseHandle(r); if (c) WinHttpCloseHandle(c); if (s) WinHttpCloseHandle(s); }
    // Opens a GET (redirects followed). Returns HTTP status, 0 on network error.
    int get(const std::string& url, const std::wstring& extra_headers = L"") {
        Url u; if (!parse_url(url, u)) return 0;
        s = WinHttpOpen(L"Ithaca/1.0", WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY, 0, 0, 0);
        if (!s) s = WinHttpOpen(L"Ithaca/1.0", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY, 0, 0, 0);
        if (!s) return 0;
        WinHttpSetTimeouts(s, 15000, 15000, 30000, 60000);
        c = WinHttpConnect(s, u.host.c_str(), u.port, 0);
        if (!c) return 0;
        r = WinHttpOpenRequest(c, L"GET", u.path.c_str(), 0, 0, 0, u.tls ? WINHTTP_FLAG_SECURE : 0);
        if (!r) return 0;
        if (!WinHttpSendRequest(r, extra_headers.empty() ? 0 : extra_headers.c_str(), (DWORD)-1L, 0, 0, 0, 0) || !WinHttpReceiveResponse(r, 0)) return 0;
        DWORD code = 0, sz = sizeof code;
        WinHttpQueryHeaders(r, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, 0, &code, &sz, 0);
        return (int)code;
    }
    int64_t length() {
        wchar_t b[64]; DWORD sz = sizeof b;
        if (!WinHttpQueryHeaders(r, WINHTTP_QUERY_CONTENT_LENGTH, 0, b, &sz, 0)) return -1;
        return _wtoi64(b);
    }
    int read(char* buf, int n) { DWORD got = 0; if (!WinHttpReadData(r, buf, n, &got)) return -1; return (int)got; }
};
static bool fetch(const std::string& url, std::string& out, std::string& err) {
    Http h; int code = h.get(url);
    if (code != 200) { err = code ? "HTTP " + std::to_string(code) : "network error (are you online?)"; return false; }
    char b[65536]; int n;
    while ((n = h.read(b, sizeof b)) > 0) out.append(b, n);
    if (n < 0) { err = "connection dropped"; return false; }
    return true;
}

// ---------------------------------------------------------------------------------------------
// Catalog: GPT4All's curated list (cached for an hour) and Hugging Face GGUF search.
static std::mutex g_cat_m;
static std::string g_cat_cache; static double g_cat_time = -1e9;
static std::string strip_html(std::string s) {
    std::string o; bool in = false;
    for (char ch : s) { if (ch == '<') in = true; else if (ch == '>') { in = false; o += ' '; } else if (!in) o += ch; }
    std::string r; for (char ch : o) if (!(ch == ' ' && !r.empty() && r.back() == ' ')) r += ch;
    while (!r.empty() && r[0] == ' ') r.erase(0, 1);
    return r;
}
static std::string catalog_json() {
    std::lock_guard<std::mutex> l(g_cat_m);
    if (!g_cat_cache.empty() && now_sec() - g_cat_time < 3600) return g_cat_cache;
    std::string txt, err;
    if (!fetch("https://gpt4all.io/models/models3.json", txt, err)) return "{\"error\":" + json_esc("Could not reach the GPT4All catalog: " + err) + "}";
    J j; if (!json_parse(txt, j) || j.t != J::ARR) return "{\"error\":\"catalog format changed\"}";
    std::string s = "{\"models\":[";
    bool first = true;
    for (auto& m : j.a) {
        std::string fn = m["filename"].str();
        if (fn.size() < 5 || fn.compare(fn.size() - 5, 5, ".gguf")) continue;   // skip API-only and embedding entries
        if (m["disableGUI"].str() == "true") continue;
        std::string url = m["url"].str("https://gpt4all.io/models/gguf/" + fn);
        auto num = [&](const char* k) { const J& v = m[k]; return v.t == J::STR ? atof(v.s.c_str()) : v.num(0); };
        s += std::string(first ? "" : ",") + "{\"name\":" + json_esc(m["name"].str(fn)) + ",\"file\":" + json_esc(fn) + ",\"url\":" + json_esc(url) +
             ",\"bytes\":" + std::to_string((int64_t)num("filesize")) + ",\"ram_gb\":" + std::to_string((int)num("ramrequired")) +
             ",\"params\":" + json_esc(m["parameters"].str()) + ",\"quant\":" + json_esc(m["quant"].str()) + ",\"type\":" + json_esc(m["type"].str()) +
             ",\"md5\":" + json_esc(m["md5sum"].str()) + ",\"desc\":" + json_esc(strip_html(m["description"].str())) + ",\"source\":\"GPT4All\"}";
        first = false;
    }
    g_cat_cache = s + "]}"; g_cat_time = now_sec();
    return g_cat_cache;
}
static std::string url_enc(const std::string& s) {
    std::string o; char b[4];
    for (unsigned char ch : s) { if (isalnum(ch) || strchr("-_.~", ch)) o += ch; else { snprintf(b, 4, "%%%02X", ch); o += b; } }
    return o;
}
static std::string hf_search_json(const std::string& q) {
    std::string txt, err;
    if (!fetch("https://huggingface.co/api/models?search=" + url_enc(q) + "&filter=gguf&sort=downloads&direction=-1&limit=30", txt, err))
        return "{\"error\":" + json_esc("Hugging Face search failed: " + err) + "}";
    J j; json_parse(txt, j);
    std::string s = "{\"repos\":[";
    for (size_t i = 0; i < j.a.size(); i++)
        s += (i ? "," : "") + std::string("{\"id\":") + json_esc(j.a[i]["id"].str()) + ",\"downloads\":" + std::to_string((int64_t)j.a[i]["downloads"].num(0)) +
             ",\"likes\":" + std::to_string((int64_t)j.a[i]["likes"].num(0)) + "}";
    return s + "]}";
}
static std::string hf_files_json(const std::string& repo) {
    std::string txt, err;
    if (repo.find("..") != std::string::npos || !fetch("https://huggingface.co/api/models/" + repo + "/tree/main?recursive=true", txt, err))
        return "{\"error\":" + json_esc("Could not list files: " + err) + "}";
    J j; json_parse(txt, j);
    std::string s = "{\"files\":["; bool first = true;
    for (auto& f : j.a) {
        std::string p = f["path"].str();
        if (p.size() < 5 || p.compare(p.size() - 5, 5, ".gguf") || f["type"].str() != "file") continue;
        if (p.find("-of-0000") != std::string::npos || p.find("mmproj") != std::string::npos) continue;   // split shards and vision projectors don't load on their own
        int64_t sz = (int64_t)f["size"].num(0); const J& lfs = f["lfs"]; if (lfs.t == J::OBJ) sz = (int64_t)lfs["size"].num((double)sz);
        s += std::string(first ? "" : ",") + "{\"path\":" + json_esc(p) + ",\"bytes\":" + std::to_string(sz) +
             ",\"url\":" + json_esc("https://huggingface.co/" + repo + "/resolve/main/" + p) + "}";
        first = false;
    }
    return s + "]}";
}

// ---------------------------------------------------------------------------------------------
// Downloads: one worker thread each, written to models\<file>.part and renamed when complete.
// A .part left behind by a crash or cancel resumes with an HTTP Range request.
struct Dl { std::string file, url, state = "queued", err; int64_t got = 0, total = 0; double speed = 0; std::atomic<bool> cancel{false}; };
static std::mutex g_dl_m;
static std::map<std::string, std::shared_ptr<Dl>> g_dls;
static std::wstring models_dir() { std::wstring d = g_store_dir + L"\\models"; CreateDirectoryW(d.c_str(), 0); return d; }
static bool safe_name(const std::string& f) { return !f.empty() && f.find_first_of("\\/:*?\"<>|") == std::string::npos && f.find("..") == std::string::npos; }

static void dl_run(std::shared_ptr<Dl> d) {
    std::wstring fin = models_dir() + L"\\" + utf8_to_w(d->file), part = fin + L".part";
    for (int attempt = 0; attempt < 4 && !d->cancel; attempt++) {
        int64_t have = 0;
        WIN32_FILE_ATTRIBUTE_DATA fa;
        if (GetFileAttributesExW(part.c_str(), GetFileExInfoStandard, &fa)) have = ((int64_t)fa.nFileSizeHigh << 32) | fa.nFileSizeLow;
        Http h;
        int code = h.get(d->url, have ? L"Range: bytes=" + std::to_wstring(have) + L"-" : L"");
        if (code == 416 && have) { d->total = have; break; }            // already complete
        if (code != 200 && code != 206) { d->err = code ? "HTTP " + std::to_string(code) : "network error"; Sleep(2000); continue; }
        if (code == 200) have = 0;                                       // server ignored the range: start over
        int64_t len = h.length();
        d->total = len > 0 ? have + len : 0; d->got = have; d->state = "downloading"; d->err.clear();
        HANDLE f = CreateFileW(part.c_str(), GENERIC_WRITE, 0, 0, code == 200 ? CREATE_ALWAYS : OPEN_ALWAYS, 0, 0);
        if (f == INVALID_HANDLE_VALUE) { d->err = "cannot write to the models folder"; break; }
        LARGE_INTEGER z; z.QuadPart = 0; SetFilePointerEx(f, z, 0, FILE_END);
        std::vector<char> buf(1 << 20);
        double t0 = now_sec(); int64_t g0 = d->got;
        int n; bool ok = true;
        while (!d->cancel && (n = h.read(buf.data(), (int)buf.size())) > 0) {
            DWORD w; if (!WriteFile(f, buf.data(), n, &w, 0) || w != (DWORD)n) { d->err = "disk full?"; ok = false; break; }
            d->got += n;
            double t = now_sec(); if (t - t0 > 1.0) { d->speed = (d->got - g0) / (t - t0); t0 = t; g0 = d->got; }
        }
        CloseHandle(f);
        if (d->cancel) break;
        if (ok && (d->total == 0 || d->got >= d->total)) { d->total = d->got; break; }
        if (ok) d->err = "connection dropped, resuming";
    }
    if (d->cancel) { d->state = "cancelled"; return; }
    if (d->total > 0 && d->got >= d->total && MoveFileExW(part.c_str(), fin.c_str(), MOVEFILE_REPLACE_EXISTING)) { d->state = "done"; d->err.clear(); }
    else { d->state = "error"; if (d->err.empty()) d->err = "download failed"; }
}
static std::string dl_start(const J& j) {
    std::string file = j["file"].str(), url = j["url"].str();
    if (!safe_name(file) || (url.rfind("https://gpt4all.io/", 0) && url.rfind("https://huggingface.co/", 0) && url.rfind("https://", 0)))
        return "{\"error\":\"bad file name or url\"}";
    if (GetFileAttributesW((models_dir() + L"\\" + utf8_to_w(file)).c_str()) != INVALID_FILE_ATTRIBUTES) return "{\"error\":\"already downloaded\"}";
    std::lock_guard<std::mutex> l(g_dl_m);
    auto it = g_dls.find(file);
    if (it != g_dls.end() && (it->second->state == "downloading" || it->second->state == "queued")) return "{}";
    auto d = std::make_shared<Dl>(); d->file = file; d->url = url; d->total = (int64_t)j["bytes"].num(0);
    g_dls[file] = d;
    std::thread(dl_run, d).detach();
    return "{}";
}
static std::string dl_json() {
    std::lock_guard<std::mutex> l(g_dl_m);
    std::string s = "["; bool first = true;
    for (auto& kv : g_dls) {
        auto& d = kv.second;
        s += std::string(first ? "" : ",") + "{\"file\":" + json_esc(d->file) + ",\"state\":" + json_esc(d->state) + ",\"err\":" + json_esc(d->err) +
             ",\"got\":" + std::to_string(d->got) + ",\"total\":" + std::to_string(d->total) + ",\"speed\":" + std::to_string((int64_t)d->speed) + "}";
        first = false;
    }
    return s + "]";
}

// ---------------------------------------------------------------------------------------------
// System report
static std::string reg_str(HKEY root, const wchar_t* key, const wchar_t* val) {
    wchar_t b[512]; DWORD sz = sizeof b;
    if (RegGetValueW(root, key, val, RRF_RT_REG_SZ, 0, b, &sz)) return "";
    std::string s = w_to_utf8(b); while (!s.empty() && s.back() == ' ') s.pop_back(); while (!s.empty() && s[0] == ' ') s.erase(0, 1);
    return s;
}
static std::string cpu_features() {
    int r[4]; __cpuid(r, 1);
    std::string f;
    if (r[2] & (1 << 28)) f += "AVX ";
    if (r[2] & (1 << 12)) f += "FMA ";
    if (r[2] & (1 << 29)) f += "F16C ";
    __cpuidex(r, 7, 0);
    if (r[1] & (1 << 5)) f += "AVX2 ";
    if (r[1] & (1 << 16)) f += "AVX-512 ";
    if (r[2] & (1 << 11)) f += "VNNI ";
    if (!f.empty()) f.pop_back();
    return f;
}
// RAM sticks from the SMBIOS memory-device table (type 17)
static std::string ram_sticks_json(int& mts) {
    mts = 0;
    UINT n = GetSystemFirmwareTable('RSMB', 0, 0, 0);
    std::vector<uint8_t> b(n);
    if (!n || GetSystemFirmwareTable('RSMB', 0, b.data(), n) != n) return "[]";
    std::string s = "["; bool first = true;
    size_t p = 8;
    while (p + 4 < b.size()) {
        uint8_t type = b[p], len = b[p + 1];
        if (type == 127 || len < 4) break;
        size_t strs = p + len, e = strs;
        while (e + 1 < b.size() && !(b[e] == 0 && b[e + 1] == 0)) e++;
        auto str = [&](int idx) { std::string r; if (!idx) return r; size_t q = strs; for (int i = 1; q < e + 1; i++) { std::string t((char*)&b[q]); if (i == idx) return t; q += t.size() + 1; } return r; };
        if (type == 17 && len >= 0x1B) {
            uint16_t size = *(uint16_t*)&b[p + 0x0C];
            if (size && size != 0xFFFF) {
                int64_t mb = size == 0x7FFF && len >= 0x20 ? *(uint32_t*)&b[p + 0x1C] : (size & 0x8000 ? (size & 0x7FFF) / 1024 : size);
                int speed = *(uint16_t*)&b[p + 0x15], conf = len >= 0x22 ? *(uint16_t*)&b[p + 0x20] : 0;
                static const char* kinds[] = {"", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "DDR", "", "DDR2", "", "", "", "DDR3", "", "DDR4", "LPDDR", "LPDDR2", "LPDDR3", "LPDDR4", "", "", "", "DDR5", "LPDDR5"};
                uint8_t mt = b[p + 0x12];
                std::string kind = mt < sizeof kinds / sizeof *kinds ? kinds[mt] : "";
                int sp = conf ? conf : speed; if (sp > mts) mts = sp;
                std::string part = str(b[p + 0x1A]), maker = str(b[p + 0x17]);
                while (!part.empty() && part.back() == ' ') part.pop_back();
                s += std::string(first ? "" : ",") + "{\"mb\":" + std::to_string(mb) + ",\"type\":" + json_esc(kind) + ",\"mts\":" + std::to_string(sp) +
                     ",\"rated\":" + std::to_string(speed) + ",\"maker\":" + json_esc(maker) + ",\"part\":" + json_esc(part) + ",\"slot\":" + json_esc(str(b[p + 0x10])) + "}";
                first = false;
            }
        }
        p = e + 2;
    }
    return s + "]";
}
static bool drive_is_ssd(wchar_t letter) {
    wchar_t dev[] = L"\\\\.\\X:"; dev[4] = letter;
    HANDLE h = CreateFileW(dev, 0, FILE_SHARE_READ | FILE_SHARE_WRITE, 0, OPEN_EXISTING, 0, 0);
    if (h == INVALID_HANDLE_VALUE) return false;
    STORAGE_PROPERTY_QUERY q{}; q.PropertyId = StorageDeviceSeekPenaltyProperty; q.QueryType = PropertyStandardQuery;
    DEVICE_SEEK_PENALTY_DESCRIPTOR r{}; DWORD got = 0;
    bool ok = DeviceIoControl(h, IOCTL_STORAGE_QUERY_PROPERTY, &q, sizeof q, &r, sizeof r, &got, 0) && got >= sizeof r;
    CloseHandle(h);
    return ok && !r.IncursSeekPenalty;
}
std::string sysinfo_json() {
    std::string s = "{";
    // OS
    typedef LONG(WINAPI * RtlGV)(OSVERSIONINFOW*);
    OSVERSIONINFOW ov{}; ov.dwOSVersionInfoSize = sizeof ov;
    if (auto f = (RtlGV)GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "RtlGetVersion")) f(&ov);
    std::string prod = reg_str(HKEY_LOCAL_MACHINE, L"SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion", L"ProductName");
    if (ov.dwBuildNumber >= 22000 && prod.find("Windows 10") == 0) prod.replace(8, 2, "11");
    s += "\"os\":" + json_esc(prod + " (build " + std::to_string(ov.dwBuildNumber) + ")");
    // CPU
    SYSTEM_INFO si; GetNativeSystemInfo(&si);
    DWORD len = 0; GetLogicalProcessorInformationEx(RelationProcessorCore, 0, &len);
    std::vector<char> lp(len); int cores = 0;
    if (GetLogicalProcessorInformationEx(RelationProcessorCore, (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)lp.data(), &len))
        for (DWORD off = 0; off < len; off += ((SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)&lp[off])->Size) cores++;
    DWORD mhz = 0, sz = sizeof mhz;
    RegGetValueW(HKEY_LOCAL_MACHINE, L"HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", L"~MHz", RRF_RT_REG_DWORD, 0, &mhz, &sz);
    s += ",\"cpu\":{\"name\":" + json_esc(reg_str(HKEY_LOCAL_MACHINE, L"HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", L"ProcessorNameString")) +
         ",\"cores\":" + std::to_string(cores) + ",\"threads\":" + std::to_string(std::thread::hardware_concurrency()) +
         ",\"mhz\":" + std::to_string(mhz) + ",\"features\":" + json_esc(cpu_features()) + "}";
    // RAM
    MEMORYSTATUSEX ms{}; ms.dwLength = sizeof ms; GlobalMemoryStatusEx(&ms);
    int mts = 0; std::string sticks = ram_sticks_json(mts);
    s += ",\"ram\":{\"total_mb\":" + std::to_string(ms.ullTotalPhys >> 20) + ",\"free_mb\":" + std::to_string(ms.ullAvailPhys >> 20) +
         ",\"mts\":" + std::to_string(mts) + ",\"sticks\":" + sticks + "}";
    // GPU
    CudaTier& cu = Model::cuda();
    if (!cu.ok()) { std::string info; cu.init(info); }
    s += ",\"gpu\":";
    if (cu.ok()) {
        size_t fr = 0, tot = 0; cu.mem_info(fr, tot);
        std::string drv = reg_str(HKEY_LOCAL_MACHINE, L"SOFTWARE\\NVIDIA Corporation\\Global\\NVTweak", L"DisplayDriverVersion");
        s += "{\"name\":" + json_esc(cu.name) + ",\"vram_mb\":" + std::to_string(tot >> 20) + ",\"free_mb\":" + std::to_string(fr >> 20) +
             ",\"tensor_cores\":" + (gx().ok() || [] { std::string e; return gx().init(e); }() ? (gx().tensor_cores() ? "true" : "false") : "false") +
             ",\"driver\":" + json_esc(drv) + "}";
    } else s += "null";
    // Drives
    s += ",\"drives\":[";
    DWORD mask = GetLogicalDrives(); bool first = true;
    for (int i = 0; i < 26; i++) {
        if (!(mask & (1u << i))) continue;
        wchar_t root[] = L"X:\\"; root[0] = L'A' + i;
        UINT t = GetDriveTypeW(root);
        if (t != DRIVE_FIXED && t != DRIVE_REMOVABLE) continue;
        ULARGE_INTEGER fb, tb;
        if (!GetDiskFreeSpaceExW(root, &fb, &tb, 0)) continue;
        s += std::string(first ? "" : ",") + "{\"letter\":\"" + (char)('A' + i) + "\",\"total_gb\":" + std::to_string(tb.QuadPart >> 30) +
             ",\"free_gb\":" + std::to_string(fb.QuadPart >> 30) + ",\"ssd\":" + (drive_is_ssd(root[0]) ? "true" : "false") + "}";
        first = false;
    }
    wchar_t md = models_dir()[0];
    s += "],\"models_drive\":\"" + std::string(1, (char)md) + "\"}";
    return s;
}

// ---------------------------------------------------------------------------------------------
// Benchmark. Runs in the background; the page polls /api/bench for progress and results.
static std::mutex g_bench_m;
static std::string g_bench_state = "{\"state\":\"idle\"}";
static std::atomic<bool> g_bench_running{false};
static std::map<std::string, double> g_bench_res;
static std::string g_bench_step;
static void bench_publish(const char* state) {
    std::lock_guard<std::mutex> l(g_bench_m);
    std::string s = "{\"state\":\"" + std::string(state) + "\",\"step\":" + json_esc(g_bench_step) + ",\"results\":{";
    bool first = true;
    for (auto& kv : g_bench_res) {
        char b[64]; snprintf(b, sizeof b, "%.3f", kv.second);
        s += std::string(first ? "" : ",") + json_esc(kv.first) + ":" + b; first = false;
    }
    g_bench_state = s + "}}";
}
static void bench_set(const std::string& k, double v) { { std::lock_guard<std::mutex> l(g_bench_m); g_bench_res[k] = v; } bench_publish("running"); }
static void bench_step(const char* s) { { std::lock_guard<std::mutex> l(g_bench_m); g_bench_step = s; } bench_publish("running"); }

__attribute__((target("avx2,fma"))) static double fma_loop(int64_t iters) {
    __m256 a0 = _mm256_set1_ps(1.0001f), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
    const __m256 m = _mm256_set1_ps(0.99999f), c = _mm256_set1_ps(0.00001f);
    for (int64_t i = 0; i < iters; i++) {
        a0 = _mm256_fmadd_ps(a0, m, c); a1 = _mm256_fmadd_ps(a1, m, c); a2 = _mm256_fmadd_ps(a2, m, c); a3 = _mm256_fmadd_ps(a3, m, c);
        a4 = _mm256_fmadd_ps(a4, m, c); a5 = _mm256_fmadd_ps(a5, m, c); a6 = _mm256_fmadd_ps(a6, m, c); a7 = _mm256_fmadd_ps(a7, m, c);
    }
    __m256 s = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)), _mm256_add_ps(_mm256_add_ps(a4, a5), _mm256_add_ps(a6, a7)));
    float o[8]; _mm256_storeu_ps(o, s);
    return o[0];
}
static double cpu_gflops(int threads) {
    const int64_t it = 40000000;   // 8 vectors x 8 lanes x 2 flops per iteration
    std::vector<std::thread> th; std::atomic<double> sink{0};
    double t0 = now_sec();
    for (int i = 0; i < threads; i++) th.emplace_back([&] { double v = fma_loop(it); double o = sink.load(); sink.store(o + v); });
    for (auto& t : th) t.join();
    return (double)threads * it * 128 / (now_sec() - t0) / 1e9;
}
static double ram_gbps() {
    const size_t n = (size_t)1 << 28;   // 1 GB of floats... in 4-byte units: 256M floats
    std::vector<float> buf;
    try { buf.assign(n / 2, 1.f); } catch (...) { return 0; }   // 512 MB
    int T = std::max(1u, std::thread::hardware_concurrency() / 2);
    double best = 0;
    for (int rep = 0; rep < 3; rep++) {
        std::vector<std::thread> th; std::atomic<double> sink{0};
        double t0 = now_sec();
        size_t per = buf.size() / T;
        for (int i = 0; i < T; i++) th.emplace_back([&, i] {
            const float* p = buf.data() + i * per; float a = 0, b = 0, c = 0, d = 0;
            for (size_t k = 0; k + 4 <= per; k += 4) { a += p[k]; b += p[k + 1]; c += p[k + 2]; d += p[k + 3]; }
            double o = sink.load(); sink.store(o + a + b + c + d);
        });
        for (auto& t : th) t.join();
        best = std::max(best, buf.size() * 4.0 / (now_sec() - t0) / 1e9);
    }
    return best;
}
// Sequential write and uncached read of a temp file in the models folder.
static void disk_bench(double& wr, double& rd) {
    wr = rd = 0;
    std::wstring p = models_dir() + L"\\ithaca_bench.tmp";
    const DWORD chunk = 8 << 20; const int chunks = 128;   // 1 GB
    void* buf = VirtualAlloc(0, chunk, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!buf) return;
    memset(buf, 0x5A, chunk);
    HANDLE f = CreateFileW(p.c_str(), GENERIC_WRITE, 0, 0, CREATE_ALWAYS, FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH, 0);
    if (f != INVALID_HANDLE_VALUE) {
        double t0 = now_sec(); DWORD w; bool ok = true;
        for (int i = 0; i < chunks && ok; i++) ok = WriteFile(f, buf, chunk, &w, 0) && w == chunk;
        CloseHandle(f);
        if (ok) wr = (double)chunk * chunks / (now_sec() - t0) / 1e9;
        f = CreateFileW(p.c_str(), GENERIC_READ, 0, 0, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN, 0);
        if (f != INVALID_HANDLE_VALUE) {
            t0 = now_sec(); DWORD r; ok = true;
            for (int i = 0; i < chunks && ok; i++) ok = ReadFile(f, buf, chunk, &r, 0) && r == chunk;
            CloseHandle(f);
            if (ok) rd = (double)chunk * chunks / (now_sec() - t0) / 1e9;
        }
        DeleteFileW(p.c_str());
    }
    VirtualFree(buf, 0, MEM_RELEASE);
}
static void gpu_bench() {
    CudaTier& cu = Model::cuda();
    Gx& g = gx();
    std::string e;
    if (!cu.ok() || (!g.ok() && !g.init(e))) return;
    // PCIe upload (pageable host memory, as the tiering streams weights)
    {
        const size_t n = 256u << 20;
        std::vector<char> h(n, 1);
        uint64_t d = cu.alloc(n);
        if (d) {
            cu.htod(d, h.data(), n); cu.sync();
            double t0 = now_sec();
            for (int i = 0; i < 4; i++) cu.htod_async(d, h.data(), n);
            cu.sync();
            bench_set("pcie_gbps", 4.0 * n / (now_sec() - t0) / 1e9);
            uint64_t d2 = cu.alloc(n);
            if (d2) {
                cu.dtod(d2, d, n); cu.sync();
                t0 = now_sec();
                for (int i = 0; i < 10; i++) cu.dtod(d2, d, n);
                cu.sync();
                bench_set("vram_gbps", 10.0 * 2 * n / (now_sec() - t0) / 1e9);   // read + write
                cu.free(d2);
            }
            cu.free(d);
        }
    }
    if (!g.tensor_cores()) return;
    // tensor-core matmul: y[M,N] = x[M,K] W^T, 4096^3
    const int M = 4096, N = 4096, K = 4096;
    GT x = g.alloc((size_t)M * K), y = g.alloc((size_t)M * N);
    uint64_t W = cu.alloc((size_t)N * K * 2);
    if (x.p && y.p && W) {
        cu.zero(x.p, (size_t)M * K * 4); cu.zero(W, (size_t)N * K * 2);
        for (int half = 0; half < 2; half++) {
            g.set_half(half == 1);
            g.linear(x, M, K, W, N, nullptr, y); cu.sync();
            double t0 = now_sec();
            for (int i = 0; i < 5; i++) g.linear(x, M, K, W, N, nullptr, y);
            cu.sync();
            bench_set(half ? "gpu_fp16_tflops" : "gpu_tf32_tflops", 5.0 * 2 * M * N * K / (now_sec() - t0) / 1e12);
        }
        g.set_half(false);
    }
    if (W) cu.free(W);
    g.free(x); g.free(y); g.trim();
}
static void bench_run() {
    { std::lock_guard<std::mutex> l(g_bench_m); g_bench_res.clear(); }
    bench_step("CPU, one core");
    bench_set("cpu_1t_gflops", cpu_gflops(1));
    bench_step("CPU, all cores");
    bench_set("cpu_mt_gflops", cpu_gflops(std::thread::hardware_concurrency()));
    bench_step("Memory bandwidth");
    bench_set("ram_gbps", ram_gbps());
    bench_step("Disk read and write (1 GB)");
    double wr, rd; disk_bench(wr, rd);
    bench_set("disk_write_gbps", wr); bench_set("disk_read_gbps", rd);
    if (Model::cuda().ok()) { bench_step("GPU"); gpu_bench(); }
    { std::lock_guard<std::mutex> l(g_bench_m); g_bench_step.clear(); }
    bench_publish("done");
    g_bench_running = false;
}

bool handle_hub(const std::string& method, const std::string& path, const std::string& body, int& code, std::string& type, std::string& out) {
    code = 200; type = "application/json";
    if (path == "/api/hub/catalog") { out = catalog_json(); return true; }
    if (path.rfind("/api/hub/search?q=", 0) == 0) {
        std::string q, e = path.substr(18);
        for (size_t i = 0; i < e.size(); i++) {
            if (e[i] == '%' && i + 2 < e.size()) { q += (char)strtol(e.substr(i + 1, 2).c_str(), 0, 16); i += 2; }
            else q += e[i] == '+' ? ' ' : e[i];
        }
        out = hf_search_json(q); return true;
    }
    if (path.rfind("/api/hub/files?repo=", 0) == 0) {
        std::string r = path.substr(20), q;
        for (size_t i = 0; i < r.size(); i++) { if (r[i] == '%' && i + 2 < r.size()) { q += (char)strtol(r.substr(i + 1, 2).c_str(), 0, 16); i += 2; } else q += r[i]; }
        out = hf_files_json(q); return true;
    }
    if (path == "/api/hub/download" && method == "POST") { J j; json_parse(body, j); out = dl_start(j); return true; }
    if (path == "/api/hub/downloads") { out = dl_json(); return true; }
    if (path == "/api/hub/cancel" && method == "POST") {
        J j; json_parse(body, j);
        std::lock_guard<std::mutex> l(g_dl_m);
        auto it = g_dls.find(j["file"].str());
        if (it != g_dls.end()) it->second->cancel = true;
        out = "{}"; return true;
    }
    if (path == "/api/sysinfo") { out = sysinfo_json(); return true; }
    if (path == "/api/bench" && method == "POST") {
        if (g_busy) { out = "{\"error\":\"wait for the chat reply to finish\"}"; return true; }
        if (!g_bench_running.exchange(true)) { bench_publish("running"); std::thread(bench_run).detach(); }
        out = "{}"; return true;
    }
    if (path == "/api/bench") { std::lock_guard<std::mutex> l(g_bench_m); out = g_bench_state; return true; }
    return false;
}

// Per-device privacy (see privacy.h). Crypto is Windows' own: BCrypt (AES-256-GCM, SHA-256, PBKDF2, random) and
// DPAPI for the "This PC" key. Nothing here is home-made cryptography.
#include "common.h"
#include "json.h"
#include "privacy.h"
#include <bcrypt.h>
#include <wincrypt.h>
#include <array>
#include <mutex>
#include <map>
#include <set>
#include <algorithm>

extern std::wstring g_store_dir;

namespace {
using Key = std::array<uint8_t, 32>;
const char kMagic[6] = "ITHE1";   // encrypted file: magic(5) + nonce(12) + tag(16) + ciphertext

// ---- primitives ----
bool rnd(void* p, size_t n) { return BCRYPT_SUCCESS(BCryptGenRandom(nullptr, (PUCHAR)p, (ULONG)n, BCRYPT_USE_SYSTEM_PREFERRED_RNG)); }
BCRYPT_ALG_HANDLE alg(int which) {   // 0 = AES-GCM, 1 = SHA-256, 2 = HMAC-SHA-256
    static BCRYPT_ALG_HANDLE h[3] = {};
    static std::once_flag once;
    std::call_once(once, [] {
        BCryptOpenAlgorithmProvider(&h[0], BCRYPT_AES_ALGORITHM, nullptr, 0);
        BCryptSetProperty(h[0], BCRYPT_CHAINING_MODE, (PUCHAR)BCRYPT_CHAIN_MODE_GCM, sizeof(BCRYPT_CHAIN_MODE_GCM), 0);
        BCryptOpenAlgorithmProvider(&h[1], BCRYPT_SHA256_ALGORITHM, nullptr, 0);
        BCryptOpenAlgorithmProvider(&h[2], BCRYPT_SHA256_ALGORITHM, nullptr, BCRYPT_ALG_HANDLE_HMAC_FLAG);
    });
    return h[which];
}
Key sha256(const std::string& d) {
    Key k{};
    BCRYPT_HASH_HANDLE hh = nullptr;
    if (BCRYPT_SUCCESS(BCryptCreateHash(alg(1), &hh, nullptr, 0, nullptr, 0, 0))) {
        BCryptHashData(hh, (PUCHAR)d.data(), (ULONG)d.size(), 0);
        BCryptFinishHash(hh, k.data(), 32, 0);
        BCryptDestroyHash(hh);
    }
    return k;
}
std::string bytes(const Key& k) { return std::string((const char*)k.data(), 32); }
std::string hex(const uint8_t* p, size_t n) { static const char* d = "0123456789abcdef"; std::string s; for (size_t i = 0; i < n; i++) { s += d[p[i] >> 4]; s += d[p[i] & 15]; } return s; }
bool unhex(const std::string& s, std::string& out) {
    if (s.size() % 2) return false;
    out.clear();
    for (size_t i = 0; i < s.size(); i += 2) {
        auto v = [](char c) { return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1; };
        int a = v(s[i]), b = v(s[i + 1]); if (a < 0 || b < 0) return false;
        out += (char)(a * 16 + b);
    }
    return true;
}
bool pbkdf2(const std::string& pin, const std::string& salt, int iters, Key& out) {
    return BCRYPT_SUCCESS(BCryptDeriveKeyPBKDF2(alg(2), (PUCHAR)pin.data(), (ULONG)pin.size(), (PUCHAR)salt.data(), (ULONG)salt.size(), iters, out.data(), 32, 0));
}
bool gcm(bool enc, const Key& k, const uint8_t* nonce, const uint8_t* in, size_t n, uint8_t* out, uint8_t* tag) {
    BCRYPT_KEY_HANDLE kh = nullptr;
    if (!BCRYPT_SUCCESS(BCryptGenerateSymmetricKey(alg(0), &kh, nullptr, 0, (PUCHAR)k.data(), 32, 0))) return false;
    BCRYPT_AUTHENTICATED_CIPHER_MODE_INFO ai; BCRYPT_INIT_AUTH_MODE_INFO(ai);
    ai.pbNonce = (PUCHAR)nonce; ai.cbNonce = 12; ai.pbTag = tag; ai.cbTag = 16;
    ULONG r = 0; uint8_t dummy = 0;
    NTSTATUS st = enc ? BCryptEncrypt(kh, n ? (PUCHAR)in : &dummy, (ULONG)n, &ai, nullptr, 0, n ? out : &dummy, (ULONG)n, &r, 0)
                      : BCryptDecrypt(kh, n ? (PUCHAR)in : &dummy, (ULONG)n, &ai, nullptr, 0, n ? out : &dummy, (ULONG)n, &r, 0);
    BCryptDestroyKey(kh);
    return BCRYPT_SUCCESS(st);
}
std::string seal(const Key& k, const std::string& plain) {
    std::string o(5 + 12 + 16 + plain.size(), '\0');
    memcpy(&o[0], kMagic, 5);
    uint8_t* nonce = (uint8_t*)&o[5]; uint8_t* tag = (uint8_t*)&o[17];
    if (!rnd(nonce, 12) || !gcm(true, k, nonce, (const uint8_t*)plain.data(), plain.size(), (uint8_t*)&o[33], tag)) return "";
    return o;
}
bool open_(const Key& k, const std::string& c, std::string& plain) {
    if (c.size() < 33 || memcmp(c.data(), kMagic, 5) != 0) return false;
    plain.assign(c.size() - 33, '\0');
    uint8_t tag[16]; memcpy(tag, &c[17], 16);
    return gcm(false, k, (const uint8_t*)&c[5], (const uint8_t*)&c[33], plain.size(), (uint8_t*)&plain[0], tag);
}
std::string b64url(const std::string& s) {
    static const char* t = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    std::string o; int v = 0, b = -6;
    for (unsigned char c : s) { v = (v << 8) | c; b += 8; while (b >= 0) { o += t[(v >> b) & 63]; b -= 6; } }
    if (b > -6) o += t[((v << 8) >> (b + 8)) & 63];
    return o;
}
bool unb64url(const std::string& s, std::string& out) {
    out.clear(); int v = 0, b = -8;
    for (char c : s) {
        int d = c >= 'A' && c <= 'Z' ? c - 'A' : c >= 'a' && c <= 'z' ? c - 'a' + 26 : c >= '0' && c <= '9' ? c - '0' + 52 : c == '-' ? 62 : c == '_' ? 63 : -1;
        if (d < 0) return false;
        v = (v << 6) | d; b += 6;
        if (b >= 0) { out += (char)((v >> b) & 0xFF); b -= 8; }
    }
    return true;
}
bool eq_ct(const std::string& a, const std::string& b) {   // constant-time compare
    if (a.size() != b.size()) return false;
    unsigned char d = 0; for (size_t i = 0; i < a.size(); i++) d |= (unsigned char)(a[i] ^ b[i]);
    return d == 0;
}
bool file_read(const std::wstring& p, std::string& out) {
    out.clear();
    FILE* f = _wfopen(p.c_str(), L"rb"); if (!f) return false;
    char b[65536]; size_t r; while ((r = fread(b, 1, sizeof b, f)) > 0) out.append(b, r);
    fclose(f); return true;
}
bool file_write(const std::wstring& p, const std::string& s) {
    std::wstring tmp = p + L".tmp";
    FILE* f = _wfopen(tmp.c_str(), L"wb"); if (!f) return false;
    bool ok = fwrite(s.data(), 1, s.size(), f) == s.size();
    ok = fclose(f) == 0 && ok;
    return ok && MoveFileExW(tmp.c_str(), p.c_str(), MOVEFILE_REPLACE_EXISTING);
}
void make_dirs(const std::wstring& path_to_file) {
    for (size_t i = g_store_dir.size() + 1; (i = path_to_file.find(L'\\', i)) != std::wstring::npos; i++) CreateDirectoryW(path_to_file.substr(0, i).c_str(), nullptr);
}
bool safe_rel(const std::wstring& r) {
    if (r.empty() || r.size() > 300 || r[0] == L'\\' || r.find(L"..") != std::wstring::npos || r.find(L':') != std::wstring::npos || r.find(L'/') != std::wstring::npos) return false;
    return true;
}
std::wstring devices_dir() { std::wstring d = g_store_dir + L"\\devices"; CreateDirectoryW(d.c_str(), nullptr); return d; }

// ---- devices and sessions ----
struct Dev { std::string id, devkey; Key base{}, data{}; std::wstring dir; bool pc = false; };
thread_local Dev t_dev;
thread_local int t_state = 0;   // 0 no device, 1 ready, 2 locked
thread_local std::string t_ses;
thread_local bool t_api = false;   // an API (/v1) request: uses the shared store, never a device's

std::mutex g_m;
struct Ses { std::string id; Key data; double last; };
std::map<std::string, Ses> g_ses;                                    // session token -> unlocked key
struct Fails { int n = 0; double until = 0; };
std::map<std::string, Fails> g_fails;                                // device id -> wrong PIN tries
std::set<std::string> g_tokens;                                      // launch tokens (hashed) made by this process
std::string g_pc_key, g_pc_id;
bool g_server_mode = false;

Dev make_dev(const std::string& devkey) {
    Dev d; d.devkey = devkey;
    Key idk = sha256("ithaca/id/" + devkey);
    d.id = hex(idk.data(), 12);
    d.base = sha256("ithaca/data/" + devkey);
    d.data = d.base;
    d.dir = devices_dir() + L"\\" + utf8_to_w(d.id);
    d.pc = !g_pc_id.empty() && d.id == g_pc_id;
    return d;
}
std::wstring pin_file(const Dev& d) { return d.dir + L"\\pin.json"; }
bool has_pin(const Dev& d) { return GetFileAttributesW(pin_file(d).c_str()) != INVALID_FILE_ATTRIBUTES; }
// The key a PIN unlocks: the device key alone isn't enough to read a PIN-protected device.
bool pin_key(const Dev& d, const std::string& pin, Key& out) {
    std::string s; J j;
    if (!file_read(pin_file(d), s) || !json_parse(s, j)) return false;
    std::string salt, check;
    if (!unhex(j["salt"].str(), salt) || !unhex(j["check"].str(), check)) return false;
    Key pk; if (!pbkdf2(pin, salt, (int)j["iters"].num(310000), pk)) return false;
    out = sha256("ithaca/pin/" + bytes(d.base) + bytes(pk));
    std::string plain;
    return open_(out, check, plain) && plain == "ithaca-ok:" + d.id;
}
std::string cookie(const std::string& name, const std::string& v, bool persist) {
    return "Set-Cookie: " + name + "=" + v + "; Path=/; " + (persist ? "Max-Age=315360000; " : "") + "HttpOnly; SameSite=Strict\r\n";
}
std::string new_session(const Dev& d, const Key& data) {
    char b[32]; rnd(b, 32);
    std::string tok = b64url(std::string(b, 32));
    std::lock_guard<std::mutex> l(g_m);
    for (auto it = g_ses.begin(); it != g_ses.end();) it = now_sec() - it->second.last > 12 * 3600 ? g_ses.erase(it) : std::next(it);
    g_ses[tok] = {d.id, data, now_sec()};
    return tok;
}
// Every file of a device, for re-encrypting when its PIN changes.
void walk(const std::wstring& dir, std::vector<std::wstring>& out) {
    WIN32_FIND_DATAW fd; HANDLE h = FindFirstFileW((dir + L"\\*").c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) return;
    do {
        std::wstring n = fd.cFileName; if (n == L"." || n == L"..") continue;
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) walk(dir + L"\\" + n, out);
        else if (n != L"pin.json" && !(n.size() > 4 && n.compare(n.size() - 4, 4, L".tmp") == 0)) out.push_back(dir + L"\\" + n);
    } while (FindNextFileW(h, &fd));
    FindClose(h);
}
// Re-encrypts all of a device's files from one key to another; checks they all open first, so it never half-fails
// because of a wrong key.
bool rekey(const Dev& d, const Key& from, const Key& to) {
    std::vector<std::wstring> files; walk(d.dir, files);
    std::string c, p;
    for (auto& f : files) if (!file_read(f, c) || !open_(from, c, p)) return false;
    for (auto& f : files) { if (!file_read(f, c) || !open_(from, c, p)) return false; if (!file_write(f, seal(to, p))) return false; }
    return true;
}
std::wstring tmp_dir() { wchar_t t[MAX_PATH]; GetTempPathW(MAX_PATH, t); std::wstring d = std::wstring(t) + L"ithaca_tmp"; CreateDirectoryW(d.c_str(), nullptr); return d; }
long long dir_size(const std::wstring& dir, int& files, FILETIME& newest) {
    long long s = 0; WIN32_FIND_DATAW fd; HANDLE h = FindFirstFileW((dir + L"\\*").c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) return 0;
    do {
        std::wstring n = fd.cFileName; if (n == L"." || n == L"..") continue;
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) s += dir_size(dir + L"\\" + n, files, newest);
        else { s += ((long long)fd.nFileSizeHigh << 32) | fd.nFileSizeLow; files++; if (CompareFileTime(&fd.ftLastWriteTime, &newest) > 0) newest = fd.ftLastWriteTime; }
    } while (FindNextFileW(h, &fd));
    FindClose(h); return s;
}
void remove_tree(const std::wstring& dir) {
    WIN32_FIND_DATAW fd; HANDLE h = FindFirstFileW((dir + L"\\*").c_str(), &fd);
    if (h != INVALID_HANDLE_VALUE) {
        do {
            std::wstring n = fd.cFileName; if (n == L"." || n == L"..") continue;
            if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) remove_tree(dir + L"\\" + n); else DeleteFileW((dir + L"\\" + n).c_str());
        } while (FindNextFileW(h, &fd));
        FindClose(h);
    }
    RemoveDirectoryW(dir.c_str());
}

// Old installs kept everything in plain files next to the exe: move them into "This PC", encrypted, and delete the
// plain copies only once each encrypted copy reads back identical.
void migrate_legacy(const Dev& pc) {
    Dev saved = t_dev; int st = t_state;
    t_dev = pc; t_state = 1;
    int moved = 0, kept = 0;
    auto one = [&](const std::wstring& rel) {
        std::wstring src = g_store_dir + L"\\" + rel; std::string a, b;
        if (!file_read(src, a)) return;
        if (store_write(rel, a) && store_read(rel, b) && b == a) { DeleteFileW(src.c_str()); moved++; } else kept++;
    };
    for (auto f : {L"prefs.json", L"memories.json"}) one(f);
    for (auto d : {L"chats", L"code_chats", L"orch_runs", L"outputs", L"voices"}) {
        WIN32_FIND_DATAW fd; HANDLE h = FindFirstFileW((g_store_dir + L"\\" + d + L"\\*").c_str(), &fd);
        if (h == INVALID_HANDLE_VALUE) continue;
        std::vector<std::wstring> names;
        do if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) names.push_back(fd.cFileName); while (FindNextFileW(h, &fd));
        FindClose(h);
        for (auto& n : names) one(std::wstring(d) + L"\\" + n);
        RemoveDirectoryW((g_store_dir + L"\\" + d).c_str());   // only when empty
    }
    if (moved || kept) printf("  Privacy: moved %d saved item(s) into \"This PC\", encrypted%s.\n", moved, kept ? " (some could not be moved and were left as they were)" : "");
    t_dev = saved; t_state = st;
}
std::string short_id(const std::string& id) { return id.substr(0, 6); }
}  // namespace

// ---------- storage ----------
bool store_has_device() { return t_state == 1; }
void privacy_api_request(bool on) { t_api = on; }
// A page request without an unlocked device gets no storage at all (never the shared folder): only API requests
// and the API-only server use the shared store.
static bool shared_ok() { return t_api || g_server_mode; }
static std::wstring full(const std::wstring& rel) { return t_state == 1 ? t_dev.dir + L"\\" + rel : shared_ok() ? g_store_dir + L"\\" + rel : L""; }
bool store_read(const std::wstring& rel, std::string& out) {
    out.clear();
    if (!safe_rel(rel)) return false;
    std::wstring p = full(rel); if (p.empty()) return false;
    std::string c; if (!file_read(p, c)) return false;
    if (t_state != 1) { out = c; return true; }
    return open_(t_dev.data, c, out);
}
bool store_write(const std::wstring& rel, const std::string& data) {
    if (!safe_rel(rel)) return false;
    std::wstring p = full(rel); if (p.empty()) return false;
    make_dirs(p);
    if (t_state != 1) return file_write(p, data);
    std::string c = seal(t_dev.data, data);
    return !c.empty() && file_write(p, c);
}
bool store_delete(const std::wstring& rel) { return safe_rel(rel) && !full(rel).empty() && DeleteFileW(full(rel).c_str()); }
bool store_exists(const std::wstring& rel) { return safe_rel(rel) && !full(rel).empty() && GetFileAttributesW(full(rel).c_str()) != INVALID_FILE_ATTRIBUTES; }
std::vector<std::wstring> store_list(const std::wstring& reldir, const std::wstring& ext) {
    std::vector<std::pair<FILETIME, std::wstring>> v;
    if (!safe_rel(reldir) || full(reldir).empty()) return {};
    WIN32_FIND_DATAW fd; HANDLE h = FindFirstFileW((full(reldir) + L"\\*" + ext).c_str(), &fd);
    if (h != INVALID_HANDLE_VALUE) {
        do if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) v.push_back({fd.ftLastWriteTime, fd.cFileName}); while (FindNextFileW(h, &fd));
        FindClose(h);
    }
    std::sort(v.begin(), v.end(), [](auto& a, auto& b) { return CompareFileTime(&a.first, &b.first) > 0; });
    std::vector<std::wstring> out; for (auto& x : v) out.push_back(x.second);
    return out;
}
std::wstring store_temp_path(const std::wstring& name) {
    char r[8]; rnd(r, 8);
    return tmp_dir() + L"\\" + utf8_to_w(hex((const uint8_t*)r, 8)) + L"_" + name;
}
bool store_put_file(const std::wstring& rel, const std::wstring& tmp) {
    std::string d; bool ok = file_read(tmp, d) && store_write(rel, d);
    DeleteFileW(tmp.c_str());
    return ok;
}
std::wstring store_temp_copy(const std::wstring& rel) {
    std::string d; if (!store_read(rel, d)) return L"";
    size_t s = rel.find_last_of(L'\\');
    std::wstring p = store_temp_path(s == std::wstring::npos ? rel : rel.substr(s + 1));
    return file_write(p, d) ? p : L"";
}

// ---------- requests ----------
void privacy_init(bool server_mode) {
    g_server_mode = server_mode;
    if (server_mode) return;   // API-only server: no app page, no devices
    std::wstring kf = devices_dir() + L"\\pc.key";
    std::string blob, key;
    if (file_read(kf, blob)) {
        DATA_BLOB in{(DWORD)blob.size(), (BYTE*)blob.data()}, out{};
        if (CryptUnprotectData(&in, nullptr, nullptr, nullptr, nullptr, CRYPTPROTECT_UI_FORBIDDEN, &out)) { key.assign((char*)out.pbData, out.cbData); SecureZeroMemory(out.pbData, out.cbData); LocalFree(out.pbData); }
    }
    bool fresh = false;
    if (key.size() != 32) {
        char b[32]; rnd(b, 32); key.assign(b, 32); SecureZeroMemory(b, 32);
        DATA_BLOB in{32, (BYTE*)key.data()}, out{};
        if (CryptProtectData(&in, L"Ithaca: This PC", nullptr, nullptr, nullptr, CRYPTPROTECT_UI_FORBIDDEN, &out)) { file_write(kf, std::string((char*)out.pbData, out.cbData)); LocalFree(out.pbData); }
        fresh = true;
    }
    g_pc_key = key;
    Dev pc = make_dev(key); g_pc_id = pc.id; pc.pc = true;
    CreateDirectoryW(pc.dir.c_str(), nullptr);
    (void)fresh;
    migrate_legacy(pc);   // a no-op once the old files are gone
}

// A launch link token. The running server knows the ones it made; a second launch of the exe writes its token's
// hash into store\devices\tokens, which only programs running as this Windows user can do.
std::string privacy_launch_token() {
    char b[32]; rnd(b, 32);
    std::string tok = hex((uint8_t*)b, 32);
    std::string h = hex(sha256("ithaca/token/" + tok).data(), 32);
    { std::lock_guard<std::mutex> l(g_m); g_tokens.insert(h); }
    std::wstring td = devices_dir() + L"\\tokens"; CreateDirectoryW(td.c_str(), nullptr);
    file_write(td + L"\\" + utf8_to_w(h), "1");
    return tok;
}
static bool take_token(const std::string& tok) {
    if (tok.size() != 64) return false;
    std::string h = hex(sha256("ithaca/token/" + tok).data(), 32);
    std::wstring f = devices_dir() + L"\\tokens\\" + utf8_to_w(h);
    bool ok = false;
    {
        std::lock_guard<std::mutex> l(g_m);
        if (g_tokens.erase(h)) ok = true;
    }
    WIN32_FILE_ATTRIBUTE_DATA fa;
    if (GetFileAttributesExW(f.c_str(), GetFileExInfoStandard, &fa)) {
        FILETIME now; GetSystemTimeAsFileTime(&now);
        ULARGE_INTEGER a{}, b{}; a.LowPart = now.dwLowDateTime; a.HighPart = now.dwHighDateTime; b.LowPart = fa.ftLastWriteTime.dwLowDateTime; b.HighPart = fa.ftLastWriteTime.dwHighDateTime;
        if (a.QuadPart - b.QuadPart < 10ull * 60 * 10000000) ok = true;   // younger than 10 minutes
        DeleteFileW(f.c_str());
    }
    return ok && !g_pc_key.empty();
}

static std::string header(const std::string& raw, const char* name) {   // case-insensitive header lookup
    std::string lo = raw; for (auto& c : lo) c = (char)tolower(c);
    std::string k = std::string("\r\n") + name + ":";
    size_t p = lo.find(k); if (p == std::string::npos) return "";
    size_t e = raw.find("\r\n", p + 2);
    std::string v = raw.substr(p + k.size(), e == std::string::npos ? std::string::npos : e - p - k.size());
    while (!v.empty() && v[0] == ' ') v.erase(0, 1);
    return v;
}
static std::string cookie_val(const std::string& cookies, const std::string& name) {
    size_t p = 0;
    while (p < cookies.size()) {
        size_t e = cookies.find(';', p); if (e == std::string::npos) e = cookies.size();
        std::string kv = cookies.substr(p, e - p);
        while (!kv.empty() && kv[0] == ' ') kv.erase(0, 1);
        if (kv.compare(0, name.size() + 1, name + "=") == 0) return kv.substr(name.size() + 1);
        p = e + 1;
    }
    return "";
}
int privacy_begin(const std::string& raw) {
    t_state = 0; t_dev = Dev(); t_ses.clear();
    if (g_server_mode) return 0;
    std::string cookies = header(raw, "cookie"), key;
    if (!unb64url(cookie_val(cookies, "ith_dev"), key) || key.size() != 32) return 0;
    t_dev = make_dev(key);
    if (has_pin(t_dev)) {
        t_ses = cookie_val(cookies, "ith_ses");
        std::lock_guard<std::mutex> l(g_m);
        auto it = g_ses.find(t_ses);
        if (it == g_ses.end() || it->second.id != t_dev.id || now_sec() - it->second.last > 12 * 3600) return t_state = 2;
        it->second.last = now_sec();
        t_dev.data = it->second.data;
    }
    CreateDirectoryW(t_dev.dir.c_str(), nullptr);
    return t_state = 1;
}
void privacy_end() { t_api = false; SecureZeroMemory(t_dev.data.data(), 32); SecureZeroMemory(t_dev.base.data(), 32); t_dev = Dev(); t_state = 0; t_ses.clear(); }

std::string privacy_page(const std::string& query, int state, bool& redirect) {
    redirect = false;
    if (g_server_mode) return "";
    size_t k = query.find("k=");
    if (k != std::string::npos) {
        std::string tok = query.substr(k + 2, 64);
        redirect = true;   // never leave a token in the address bar or history
        if (take_token(tok)) return cookie("ith_dev", b64url(g_pc_key), true) + cookie("ith_ses", "", false);
    }
    if (state == 0) {
        char b[32]; rnd(b, 32);
        std::string c = cookie("ith_dev", b64url(std::string(b, 32)), true);
        SecureZeroMemory(b, 32);
        return c;
    }
    return "";
}

bool handle_device(const std::string& method, const std::string& path, const std::string& body, int& code, std::string& out, std::string& set_cookie) {
    if (path.rfind("/api/device", 0) != 0) return false;
    code = 200;
    J j; json_parse(body.empty() ? "{}" : body, j);
    auto err = [&](int c, const std::string& m) { code = c; out = "{\"error\":" + json_esc(m) + "}"; return true; };
    if (path == "/api/device") {
        std::string name;
        if (t_state == 1) { std::string s; if (store_read(L"device.json", s)) { J d; json_parse(s, d); name = d["name"].str(); } }
        out = "{\"ready\":" + std::string(t_state == 1 ? "true" : "false") + ",\"locked\":" + (t_state == 2 ? "true" : "false") +
              ",\"pc\":" + (t_dev.pc ? "true" : "false") + ",\"id\":" + json_esc(t_dev.id.empty() ? "" : short_id(t_dev.id)) +
              ",\"pin\":" + (t_state && has_pin(t_dev) ? "true" : "false") + ",\"name\":" + json_esc(name) + "}";
        return true;
    }
    if (method != "POST" && path != "/api/device/others") return err(405, "POST only");
    if (path == "/api/device/unlock") {
        if (t_state != 2) return err(400, "This device isn't locked.");
        double now = now_sec();
        {
            std::lock_guard<std::mutex> l(g_m);
            Fails& f = g_fails[t_dev.id];
            if (now < f.until) return err(429, "Too many wrong PINs. Try again in " + std::to_string((int)(f.until - now) + 1) + " s.");
        }
        Key data;
        if (!pin_key(t_dev, j["pin"].str(), data)) {
            std::lock_guard<std::mutex> l(g_m);
            Fails& f = g_fails[t_dev.id];
            if (++f.n >= 5) f.until = now + std::min(3600.0, 30.0 * (1 << std::min(7, f.n - 5)));   // 30 s, 60 s, ... up to an hour
            return err(403, "Wrong PIN.");
        }
        { std::lock_guard<std::mutex> l(g_m); g_fails.erase(t_dev.id); }
        set_cookie = cookie("ith_ses", new_session(t_dev, data), false);
        out = "{}";
        return true;
    }
    if (path == "/api/device/link") {   // use another device's data on this one (its recovery code)
        std::string code_s = j["code"].str(), key;
        code_s.erase(std::remove_if(code_s.begin(), code_s.end(), [](char c) { return c == ' ' || c == '\n' || c == '\r' || c == '\t'; }), code_s.end());
        if (!unb64url(code_s, key) || key.size() != 32) return err(400, "That isn't a valid recovery code.");
        set_cookie = cookie("ith_dev", b64url(key), true) + cookie("ith_ses", "", false);
        out = "{}";
        return true;
    }
    if (path == "/api/device/new") {   // start over as a new, empty device (the old data stays, reachable by its code)
        char b[32]; rnd(b, 32);
        set_cookie = cookie("ith_dev", b64url(std::string(b, 32)), true) + cookie("ith_ses", "", false);
        out = "{}";
        return true;
    }
    if (t_state != 1) return err(423, "This device is locked.");
    if (path == "/api/device/name") {
        std::string n = j["name"].str().substr(0, 60);
        store_write(L"device.json", "{\"name\":" + json_esc(n) + "}");
        out = "{}";
        return true;
    }
    if (path == "/api/device/recovery") {
        if (has_pin(t_dev)) { Key k; if (!pin_key(t_dev, j["pin"].str(), k)) return err(403, "Enter this device's PIN to see its recovery code."); }
        out = "{\"code\":" + json_esc(b64url(t_dev.devkey)) + "}";
        return true;
    }
    if (path == "/api/device/pin") {
        const std::string pin = j["pin"].str();
        if (has_pin(t_dev)) { Key k; if (!pin_key(t_dev, j["current"].str(), k)) return err(403, "The current PIN is wrong."); }
        if (pin.empty()) {   // remove the PIN: back to the device key alone
            if (!has_pin(t_dev)) { out = "{}"; return true; }
            if (!rekey(t_dev, t_dev.data, t_dev.base)) return err(500, "Could not re-encrypt this device's data.");
            DeleteFileW(pin_file(t_dev).c_str());
            set_cookie = cookie("ith_ses", "", false);
            out = "{}";
            return true;
        }
        if (pin.size() < 4 || pin.size() > 64) return err(400, "Use 4 to 64 characters.");
        char salt[16]; rnd(salt, 16);
        Key pk; if (!pbkdf2(pin, std::string(salt, 16), 310000, pk)) return err(500, "Could not derive a key.");
        Key nd = sha256("ithaca/pin/" + bytes(t_dev.base) + bytes(pk));
        if (!rekey(t_dev, t_dev.data, nd)) return err(500, "Could not re-encrypt this device's data.");
        const std::string chk = seal(nd, "ithaca-ok:" + t_dev.id);
        std::string meta = "{\"salt\":\"" + hex((uint8_t*)salt, 16) + "\",\"iters\":310000,\"check\":\"" + hex((const uint8_t*)chk.data(), chk.size()) + "\"}";
        if (!file_write(pin_file(t_dev), meta)) return err(500, "Could not save the PIN.");
        set_cookie = cookie("ith_ses", new_session(t_dev, nd), false);
        out = "{}";
        return true;
    }
    if (path == "/api/device/others") {   // This PC only: other devices' folders (sizes; their contents can't be read)
        if (!t_dev.pc) return err(403, "Only This PC can manage other devices.");
        std::string s = "[";
        WIN32_FIND_DATAW fd; HANDLE h = FindFirstFileW((devices_dir() + L"\\*").c_str(), &fd);
        int n = 0;
        if (h != INVALID_HANDLE_VALUE) {
            do {
                std::wstring nm = fd.cFileName;
                if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) || nm == L"." || nm == L".." || nm == L"tokens" || w_to_utf8(nm) == g_pc_id) continue;
                int files = 0; FILETIME newest{};
                long long sz = dir_size(devices_dir() + L"\\" + nm, files, newest);
                ULARGE_INTEGER t; t.LowPart = newest.dwLowDateTime; t.HighPart = newest.dwHighDateTime;
                double ms = t.QuadPart ? (t.QuadPart / 10000.0 - 11644473600000.0) : 0;
                s += (n++ ? "," : "") + std::string("{\"id\":") + json_esc(w_to_utf8(nm)) + ",\"mb\":" + std::to_string(sz / 1048576.0) +
                     ",\"files\":" + std::to_string(files) + ",\"updated\":" + std::to_string((long long)ms) + ",\"pin\":" +
                     (GetFileAttributesW((devices_dir() + L"\\" + nm + L"\\pin.json").c_str()) != INVALID_FILE_ATTRIBUTES ? "true" : "false") + "}";
            } while (FindNextFileW(h, &fd));
            FindClose(h);
        }
        out = s + "]";
        return true;
    }
    if (path == "/api/device/others/delete") {
        if (!t_dev.pc) return err(403, "Only This PC can manage other devices.");
        std::string id = j["id"].str();
        if (id.size() != 24 || id.find_first_not_of("0123456789abcdef") != std::string::npos || id == g_pc_id) return err(400, "bad id");
        remove_tree(devices_dir() + L"\\" + utf8_to_w(id));
        out = "{}";
        return true;
    }
    return err(404, "not found");
}

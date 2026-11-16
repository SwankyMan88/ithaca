// Coding agent: file tools for one workspace folder. The model calls them through the web UI (which runs the
// loop and asks for approval); every path is resolved inside the workspace and every change can be undone.
// Office files (docx/pptx/xlsx) are read as addressed text and edited in place; PDFs are read-only.
#include "common.h"
#include "json.h"
#include <shellapi.h>
#include <shlobj.h>
#include <shobjidl.h>
#include <thread>
#include <atomic>
#include <algorithm>
#include <map>

extern std::wstring g_store_dir;
bool office_view(const std::string& ext, const std::string& z, std::string& out);
std::string office_replace(const std::string& ext, std::string& z, const std::string& old, const std::string& nw);
std::string xlsx_set_cell(std::string& z, const std::string& sheet, std::string ref, const std::string& value);
std::string doc_extract_json(const std::string& name, const std::string& bytes);

namespace {
bool rd(const std::wstring& p, std::string& out) {
    HANDLE h = CreateFileW(p.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, 0, OPEN_EXISTING, 0, 0);
    if (h == INVALID_HANDLE_VALUE) return false;
    LARGE_INTEGER sz; GetFileSizeEx(h, &sz);
    if (sz.QuadPart > (256ll << 20)) { CloseHandle(h); return false; }
    out.resize((size_t)sz.QuadPart);
    DWORD got = 0; bool ok = ReadFile(h, &out[0], (DWORD)out.size(), &got, 0) && got == out.size();
    CloseHandle(h);
    return ok;
}
bool wr(const std::wstring& p, const std::string& s) {
    // write to a temp file next to it, then swap in: a crash never leaves half a file
    std::wstring tmp = p + L".ithaca-tmp";
    HANDLE h = CreateFileW(tmp.c_str(), GENERIC_WRITE, 0, 0, CREATE_ALWAYS, 0, 0);
    if (h == INVALID_HANDLE_VALUE) return false;
    DWORD w = 0; bool ok = WriteFile(h, s.data(), (DWORD)s.size(), &w, 0) && w == s.size();
    CloseHandle(h);
    if (!ok || !MoveFileExW(tmp.c_str(), p.c_str(), MOVEFILE_REPLACE_EXISTING)) { DeleteFileW(tmp.c_str()); return false; }
    return true;
}
std::wstring lower(std::wstring s) { for (auto& c : s) c = towlower(c); return s; }
std::string ext_of(const std::wstring& p) {
    size_t d = p.find_last_of(L'.'), s = p.find_last_of(L"\\/");
    if (d == std::wstring::npos || (s != std::wstring::npos && d < s)) return "";
    std::string e = w_to_utf8(p.substr(d + 1)); for (auto& c : e) c = (char)tolower((unsigned char)c);
    return e;
}
bool is_dir(const std::wstring& p) { DWORD a = GetFileAttributesW(p.c_str()); return a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY); }
bool exists(const std::wstring& p) { return GetFileAttributesW(p.c_str()) != INVALID_FILE_ATTRIBUTES; }
bool binary(const std::string& s) { size_t n = std::min<size_t>(s.size(), 8192); for (size_t i = 0; i < n; i++) if (s[i] == 0) return true; return false; }
const char* kSkipDirs[] = {".git", "node_modules", "__pycache__", ".venv", "venv", "build", "dist", ".vs", ".idea", "target", "obj", "bin"};

// Resolves a workspace-relative path; empty on anything outside the workspace.
std::wstring resolve(const std::wstring& root, const std::string& rel) {
    std::wstring r = utf8_to_w(rel);
    for (auto& c : r) if (c == L'/') c = L'\\';
    while (!r.empty() && (r[0] == L'\\')) r.erase(0, 1);
    std::wstring joined = r.size() > 1 && r[1] == L':' ? r : root + L"\\" + r, full(32768, 0);
    DWORD n = GetFullPathNameW(joined.c_str(), (DWORD)full.size(), &full[0], nullptr);
    if (!n || n >= full.size()) return L"";
    full.resize(n);
    while (full.size() > 3 && full.back() == L'\\') full.pop_back();
    std::wstring lr = lower(root), lf = lower(full);
    if (lf != lr && lf.compare(0, lr.size() + 1, lr + L"\\") != 0) return L"";
    return full;
}
std::string rel_of(const std::wstring& root, const std::wstring& full) {
    std::string s = w_to_utf8(full.size() > root.size() ? full.substr(root.size() + 1) : L".");
    for (auto& c : s) if (c == '\\') c = '/';
    return s;
}

// ---- undo journal: store\code_undo\<id>.bin (old bytes) + .json {path, existed, renamed_from}
std::wstring undo_dir() { std::wstring d = g_store_dir + L"\\code_undo"; CreateDirectoryW(d.c_str(), 0); return d; }
std::string new_undo(const std::wstring& path, bool existed, const std::string& old, const std::wstring& renamed_from = L"") {
    static int seq = 0;
    std::string id = std::to_string((long long)GetTickCount64()) + "_" + std::to_string(++seq);
    std::wstring base = undo_dir() + L"\\" + utf8_to_w(id);
    if (existed) wr(base + L".bin", old);
    std::string meta = "{\"path\":" + json_esc(w_to_utf8(path)) + ",\"existed\":" + (existed ? "true" : "false") + ",\"from\":" + json_esc(w_to_utf8(renamed_from)) + "}";
    wr(base + L".json", meta);
    return id;
}

void list_rec(const std::wstring& root, const std::wstring& dir, int depth, int maxd, std::string& out, int& n) {
    std::vector<std::pair<bool, std::wstring>> ents;
    WIN32_FIND_DATAW fd; HANDLE h = FindFirstFileW((dir + L"\\*").c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) return;
    do {
        std::wstring nm = fd.cFileName;
        if (nm == L"." || nm == L"..") continue;
        ents.push_back({(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0, nm});
    } while (FindNextFileW(h, &fd));
    FindClose(h);
    std::sort(ents.begin(), ents.end(), [](auto& a, auto& b) { return a.first != b.first ? a.first > b.first : lower(a.second) < lower(b.second); });
    for (auto& e : ents) {
        if (n >= 400) { out += std::string(depth * 2, ' ') + "... (more not shown)\n"; return; }
        std::string nm = w_to_utf8(e.second);
        bool skip = false; for (auto s : kSkipDirs) if (e.first && nm == s) skip = true;
        out += std::string(depth * 2, ' ') + nm + (e.first ? "/" : "");
        n++;
        if (!e.first) {
            WIN32_FILE_ATTRIBUTE_DATA fa; if (GetFileAttributesExW((dir + L"\\" + e.second).c_str(), GetFileExInfoStandard, &fa)) {
                uint64_t sz = ((uint64_t)fa.nFileSizeHigh << 32) | fa.nFileSizeLow;
                out += sz >= 1024 * 1024 ? "  (" + std::to_string(sz >> 20) + " MB)" : sz >= 1024 ? "  (" + std::to_string(sz >> 10) + " KB)" : "  (" + std::to_string(sz) + " B)";
            }
        }
        out += skip ? "  (skipped)\n" : "\n";
        if (e.first && !skip && depth + 1 < maxd) list_rec(root, dir + L"\\" + e.second, depth + 1, maxd, out, n);
    }
}
void search_rec(const std::wstring& root, const std::wstring& dir, const std::string& pat, std::string& out, int& hits, int& files) {
    WIN32_FIND_DATAW fd; HANDLE h = FindFirstFileW((dir + L"\\*").c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) return;
    std::string lp = pat; for (auto& c : lp) c = (char)tolower((unsigned char)c);
    do {
        std::wstring nm = fd.cFileName, p = dir + L"\\" + nm;
        if (nm == L"." || nm == L".." || hits >= 120 || files > 5000) continue;
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
            bool skip = false; for (auto s : kSkipDirs) if (w_to_utf8(nm) == s) skip = true;
            if (!skip) search_rec(root, p, pat, out, hits, files);
            continue;
        }
        if ((((uint64_t)fd.nFileSizeHigh << 32) | fd.nFileSizeLow) > (4u << 20)) continue;
        files++;
        std::string s; if (!rd(p, s) || binary(s)) continue;
        int line = 1; size_t ls = 0;
        for (size_t i = 0; i <= s.size() && hits < 120; i++) {
            if (i == s.size() || s[i] == '\n') {
                std::string L = s.substr(ls, i - ls), ll = L; for (auto& c : ll) c = (char)tolower((unsigned char)c);
                if (ll.find(lp) != std::string::npos) {
                    if (!L.empty() && L.back() == '\r') L.pop_back();
                    if (L.size() > 200) L = L.substr(0, 200) + "...";
                    out += rel_of(root, p) + ":" + std::to_string(line) + ": " + L + "\n"; hits++;
                }
                line++; ls = i + 1;
            }
        }
    } while (FindNextFileW(h, &fd));
    FindClose(h);
}
std::string numbered(const std::string& s, int a, int b, int& total) {
    std::vector<std::string> lines; size_t p = 0;
    while (p <= s.size()) { size_t e = s.find('\n', p); if (e == std::string::npos) e = s.size(); std::string L = s.substr(p, e - p); if (!L.empty() && L.back() == '\r') L.pop_back(); lines.push_back(L); p = e + 1; if (e == s.size()) break; }
    if (!lines.empty() && lines.back().empty() && !s.empty() && s.back() == '\n') lines.pop_back();
    total = (int)lines.size();
    a = std::max(1, a); b = b <= 0 ? total : std::min(b, total);
    std::string out;
    for (int i = a; i <= b; i++) { out += std::to_string(i) + "\t" + lines[i - 1] + "\n"; if (out.size() > 60000) { out += "... (truncated: read a smaller range with start_line / end_line)\n"; break; } }
    return out;
}
// CRLF files stay CRLF: the model writes \n.
std::string match_eol(const std::string& old, std::string s) {
    if (old.find("\r\n") == std::string::npos) return s;
    std::string o; for (size_t i = 0; i < s.size(); i++) { if (s[i] == '\n' && (i == 0 || s[i - 1] != '\r')) o += '\r'; o += s[i]; }
    return o;
}
std::string lf(std::string s) { s.erase(std::remove(s.begin(), s.end(), '\r'), s.end()); return s; }
// Tool output may hold raw bytes (a binary file, a command's codepage text): keep it valid UTF-8 for the JSON.
std::string utf8_clean(const std::string& s) {
    std::string o; o.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        unsigned char c = (unsigned char)s[i]; int n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
        bool ok = n > 0 && i + n <= s.size() && !(n == 2 && c < 0xC2);
        for (int k = 1; ok && k < n; k++) ok = ((unsigned char)s[i + k] >> 6) == 2;
        if (ok && n == 1 && c < 0x20 && c != '\n' && c != '\r' && c != '\t') ok = false;
        if (ok) { o.append(s, i, n); i += n; } else { o += "\xEF\xBF\xBD"; i++; }
    }
    return o;
}
std::string res(bool ok, const std::string& output, const std::string& extra = "") {
    return "{\"ok\":" + std::string(ok ? "true" : "false") + ",\"output\":" + json_esc(utf8_clean(output)) + extra + "}";
}
}  // namespace

static std::atomic<bool> g_cmd_stop{false};   // the Stop button: ends a running command
// run_command: cmd /c in the workspace, no input, a time limit, the whole process tree killed on timeout or stop.
static std::string run_cmd(const std::wstring& root, const std::string& command, int secs) {
    SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
    HANDLE rd, wrp;
    if (!CreatePipe(&rd, &wrp, &sa, 0)) return "could not start the command";
    SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);
    HANDLE nul = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, OPEN_EXISTING, 0, nullptr);
    STARTUPINFOW si{}; si.cb = sizeof si; si.dwFlags = STARTF_USESTDHANDLES; si.hStdOutput = wrp; si.hStdError = wrp; si.hStdInput = nul;
    PROCESS_INFORMATION pi{};
    std::wstring cl = L"cmd.exe /d /s /c \"" + utf8_to_w(command) + L"\"";
    std::vector<wchar_t> buf(cl.begin(), cl.end()); buf.push_back(0);
    HANDLE job = CreateJobObjectW(nullptr, nullptr);
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION li{}; li.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    SetInformationJobObject(job, JobObjectExtendedLimitInformation, &li, sizeof li);
    if (!CreateProcessW(nullptr, buf.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW | CREATE_SUSPENDED, nullptr, root.c_str(), &si, &pi)) {
        CloseHandle(rd); CloseHandle(wrp); CloseHandle(nul); CloseHandle(job); return "could not start the command";
    }
    AssignProcessToJobObject(job, pi.hProcess);
    ResumeThread(pi.hThread);
    CloseHandle(wrp); CloseHandle(nul);
    std::string out; char b[8192]; DWORD n; bool timed = false;
    const double t0 = now_sec();
    for (;;) {
        DWORD avail = 0;
        if (!PeekNamedPipe(rd, nullptr, 0, nullptr, &avail, nullptr)) break;
        if (avail) { if (!ReadFile(rd, b, std::min<DWORD>(avail, sizeof b), &n, nullptr) || !n) break; out.append(b, n); if (out.size() > (1u << 20)) out.erase(0, out.size() - (512u << 10)); }
        else if (WaitForSingleObject(pi.hProcess, 25) == WAIT_OBJECT_0) {
            while (PeekNamedPipe(rd, nullptr, 0, nullptr, &avail, nullptr) && avail && ReadFile(rd, b, std::min<DWORD>(avail, sizeof b), &n, nullptr) && n) out.append(b, n);
            break;
        }
        if (now_sec() - t0 > secs) { timed = true; break; }
        if (g_cmd_stop) { out += "\n[stopped by the user]"; break; }
    }
    DWORD code = 0; GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(job);   // kills whatever is still running (the command and anything it started)
    CloseHandle(pi.hProcess); CloseHandle(pi.hThread); CloseHandle(rd);
    out.erase(std::remove(out.begin(), out.end(), '\r'), out.end());
    if (out.size() > 24000) out = out.substr(0, 6000) + "\n... (" + std::to_string(out.size() - 18000) + " characters cut) ...\n" + out.substr(out.size() - 12000);
    if (!timed && code == STILL_ACTIVE) return out;   // stopped by the user: already noted
    return out + (timed ? "\n[stopped after " + std::to_string(secs) + " s]" : "\n[exit code " + std::to_string((int)code) + "]");
}

// Folder picker for the workspace (the UI runs in a browser, which can't hand us a real path).
static std::string pick_folder() {
    std::string out;
    std::thread([&] {
        CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
        IFileDialog* d = nullptr;
        if (SUCCEEDED(CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&d)))) {
            DWORD o = 0; d->GetOptions(&o); d->SetOptions(o | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM);
            d->SetTitle(L"Choose the folder Ithaca may work in");
            HWND fg = GetForegroundWindow();
            if (SUCCEEDED(d->Show(fg))) {
                IShellItem* it = nullptr;
                if (SUCCEEDED(d->GetResult(&it))) { PWSTR p = nullptr; if (SUCCEEDED(it->GetDisplayName(SIGDN_FILESYSPATH, &p))) { out = w_to_utf8(p); CoTaskMemFree(p); } it->Release(); }
            }
            d->Release();
        }
        CoUninitialize();
    }).join();
    return out;
}

// One tool call. dry = true: nothing changes; the reply carries "before"/"after" (text files) for the approval view.
static std::string run_tool(const std::wstring& root, const std::string& name, const J& a, bool dry) {
    auto P = [&](const char* k) { return resolve(root, a[k].str()); };
    if (name == "list_dir") {
        std::wstring d = a["path"].str().empty() ? root : P("path");
        if (d.empty() || !is_dir(d)) return res(false, "not a folder inside the workspace: " + a["path"].str());
        std::string out = rel_of(root, d) + "/\n"; int n = 0;
        list_rec(root, d, 1, std::max(1, std::min(6, (int)a["depth"].num(2))) + 1, out, n);
        if (!n) out += "(empty: there are no files or folders here yet)\n";
        return res(true, out);
    }
    if (name == "read_file") {
        std::wstring p = P("path"); std::string s;
        if (p.empty()) return res(false, "path is outside the workspace");
        if (is_dir(p)) return res(false, "that is a folder; use list_dir");
        if (!rd(p, s)) return res(false, "cannot read " + a["path"].str() + " (missing, locked or over 256 MB)");
        const std::string e = ext_of(p);
        if (e == "docx" || e == "pptx" || e == "xlsx") {
            std::string v; if (!office_view(e, s, v)) return res(false, "could not open this " + e + " file");
            const char* how = e == "xlsx" ? "Edit cells with set_cell (cell like B4, value: number, text, or a formula starting with =)."
                                          : "Edit with edit_file: old = exact text from one paragraph (without the [..] label), new = replacement. Formatting is kept.";
            return res(true, "(" + e + " file shown as text. " + how + ")\n" + v);
        }
        if (e == "pdf") { J j; json_parse(doc_extract_json(w_to_utf8(p), s), j); return j["error"].t == J::STR ? res(false, j["error"].s) : res(true, "(PDF, read-only)\n" + j["text"].str()); }
        if (binary(s)) return res(false, "binary file (" + std::to_string(s.size()) + " bytes); it can't be read as text");
        int total = 0; std::string out = numbered(s, (int)a["start_line"].num(1), (int)a["end_line"].num(0), total);
        return res(true, out, ",\"lines\":" + std::to_string(total));
    }
    if (name == "search") {
        std::string pat = a["pattern"].str();
        if (pat.empty()) return res(false, "pattern is empty");
        std::wstring d = a["path"].str().empty() ? root : P("path");
        if (d.empty()) return res(false, "path is outside the workspace");
        std::string out; int hits = 0, files = 0;
        search_rec(root, d, pat, out, hits, files);
        return res(true, out.empty() ? "no matches in " + std::to_string(files) + " files" : out + (hits >= 120 ? "(stopped at 120 matches)\n" : ""));
    }
    // ---- changes
    auto done = [&](const std::string& msg, const std::string& undo, const std::string& before, const std::string& after, const std::string& path) {
        return res(true, msg, ",\"undo\":" + json_esc(undo) + ",\"path\":" + json_esc(path) + (before.size() + after.size() < (4u << 20) ? ",\"before\":" + json_esc(before) + ",\"after\":" + json_esc(after) : std::string()));
    };
    if (name == "write_file" || name == "create_file") {
        std::wstring p = P("path"); if (p.empty()) return res(false, "path is outside the workspace");
        const std::string e = ext_of(p);
        if (e == "docx" || e == "pptx" || e == "xlsx" || e == "pdf") return res(false, "write_file makes text files; use edit_file / set_cell to change " + e + " files");
        if (is_dir(p)) return res(false, "a folder has that name");
        std::string old; const bool had = rd(p, old);
        if (had && binary(old)) return res(false, "won't overwrite a binary file");
        if (name == "create_file" && had) return res(false, "file already exists; use edit_file or write_file");
        std::string nw = had ? match_eol(old, a["content"].str()) : a["content"].str();
        if (dry) return done("", "", lf(old), lf(nw), a["path"].str());
        // parent folders
        for (size_t i = root.size() + 1; (i = p.find(L'\\', i)) != std::wstring::npos; i++) CreateDirectoryW(p.substr(0, i).c_str(), 0);
        std::string u = new_undo(p, had, old);
        if (!wr(p, nw)) return res(false, "could not write the file");
        return done(std::string(had ? "updated " : "created ") + a["path"].str(), u, lf(old), lf(nw), a["path"].str());
    }
    if (name == "edit_file") {
        std::wstring p = P("path"); std::string s;
        if (p.empty()) return res(false, "path is outside the workspace");
        if (!rd(p, s)) return res(false, "cannot read " + a["path"].str());
        const std::string e = ext_of(p), old = a["old"].str(), nw = a["new"].str();
        if (old.empty()) return res(false, "old is empty; to add text, include a nearby line in old and repeat it in new");
        if (e == "docx" || e == "pptx") {
            // Models often copy the [p1] / [slide 2 p3] labels, or several paragraphs at once: drop the labels and
            // apply the paragraphs that changed one by one.
            auto lines = [](const std::string& t) {
                std::vector<std::string> v; size_t b = 0;
                for (;;) {
                    size_t n = t.find('\n', b); std::string l = t.substr(b, n == std::string::npos ? std::string::npos : n - b);
                    if (!l.empty() && l.back() == '\r') l.pop_back();
                    if (l.size() > 2 && l[0] == '[') { size_t r = l.find("] "); if (r != std::string::npos && r < 24) l = l.substr(r + 2); }
                    v.push_back(l);
                    if (n == std::string::npos) break; b = n + 1;
                }
                return v;
            };
            std::vector<std::string> ol = lines(old), nl = lines(nw);
            std::string z = s, err;
            if (ol.size() == nl.size()) {
                int changed = 0;
                for (size_t k = 0; k < ol.size() && err.empty(); k++)
                    if (ol[k] != nl[k]) { if (ol[k].empty()) err = "old has an empty paragraph"; else err = office_replace(e, z, ol[k], nl[k]); changed++; }
                if (!changed) err = "old and new are the same";
            } else err = office_replace(e, z, old, nw);
            if (!err.empty()) return res(false, err);
            if (dry) return done("", "", old, nw, a["path"].str());
            std::string u = new_undo(p, true, s);
            if (!wr(p, z)) return res(false, "could not write the file (is it open in Word or PowerPoint?)");
            return done("edited " + a["path"].str(), u, old, nw, a["path"].str());
        }
        if (e == "xlsx") return res(false, "use set_cell for spreadsheets");
        if (binary(s)) return res(false, "binary file");
        const std::string t = lf(s);
        std::string o = lf(old), n2 = lf(nw);
        size_t at = t.find(o);
        if (at == std::string::npos) {   // models often add or drop a blank line at the ends: trim them (from new too)
            while (!o.empty() && o[0] == '\n') { o.erase(0, 1); if (!n2.empty() && n2[0] == '\n') n2.erase(0, 1); }
            while (!o.empty() && o.back() == '\n') { o.pop_back(); if (!n2.empty() && n2.back() == '\n') n2.pop_back(); }
            at = o.empty() ? std::string::npos : t.find(o);
        }
        std::string nt = t;
        if (at == std::string::npos && !a["replace_all"].b) {   // last try: whole lines, ignoring trailing spaces
            auto split = [](const std::string& x) { std::vector<std::string> v; size_t p = 0; for (;;) { size_t e = x.find('\n', p); v.push_back(x.substr(p, e == std::string::npos ? std::string::npos : e - p)); if (e == std::string::npos) break; p = e + 1; } return v; };
            auto rs = [](std::string x) { while (!x.empty() && (x.back() == ' ' || x.back() == '\t')) x.pop_back(); return x; };
            std::vector<std::string> T = split(t), O = split(o);
            int found = -1, count = 0;
            for (size_t i = 0; !O.empty() && i + O.size() <= T.size(); i++) {
                bool eq = true;
                for (size_t k = 0; k < O.size() && eq; k++) eq = rs(T[i + k]) == rs(O[k]);
                if (eq) { count++; if (found < 0) found = (int)i; }
            }
            if (count > 1) return res(false, "old text occurs more than once; include more surrounding lines, or set replace_all");
            if (count == 0) return res(false, "old text not found. Copy it exactly from read_file (without the line numbers), including indentation.");
            std::string out;
            for (size_t i = 0; i < T.size(); i++) {
                if ((int)i == found) { out += n2; i += O.size() - 1; }
                else out += T[i];
                if (i + 1 < T.size()) out += '\n';
            }
            nt = out;
        } else {
            if (at == std::string::npos) return res(false, "old text not found. Copy it exactly from read_file (without the line numbers), including indentation.");
            if (t.find(o, at + 1) != std::string::npos && !a["replace_all"].b) return res(false, "old text occurs more than once; include more surrounding lines, or set replace_all");
            if (a["replace_all"].b) { size_t q = 0; while ((q = nt.find(o, q)) != std::string::npos) { nt.replace(q, o.size(), n2); q += n2.size(); } }
            else nt.replace(at, o.size(), n2);
        }
        if (dry) return done("", "", t, nt, a["path"].str());
        std::string u = new_undo(p, true, s);
        if (!wr(p, match_eol(s, nt))) return res(false, "could not write the file");
        return done("edited " + a["path"].str(), u, t, nt, a["path"].str());
    }
    if (name == "set_cell") {
        std::wstring p = P("path"); std::string s;
        if (p.empty() || ext_of(p) != "xlsx") return res(false, "set_cell works on .xlsx files in the workspace");
        if (!rd(p, s)) return res(false, "cannot read " + a["path"].str());
        // the value may come as a JSON number or boolean, not only a string
        const J& jv = a["value"]; std::string val = jv.str();
        if (jv.t == J::NUM) { char b[40]; snprintf(b, sizeof b, "%.15g", jv.n); val = b; }
        else if (jv.t == J::BOOL) val = jv.b ? "TRUE" : "FALSE";
        std::string z = s, err = xlsx_set_cell(z, a["sheet"].str(), a["cell"].str(), val);
        if (!err.empty()) return res(false, err);
        const std::string what = (a["sheet"].str().empty() ? "" : a["sheet"].str() + "!") + a["cell"].str() + " = " + val;
        if (dry) return done("", "", "", what, a["path"].str());
        std::string u = new_undo(p, true, s);
        if (!wr(p, z)) return res(false, "could not write the file (is it open in Excel?)");
        return done("set " + what, u, "", what, a["path"].str());
    }
    if (name == "make_dir") {
        std::wstring p = P("path"); if (p.empty()) return res(false, "path is outside the workspace");
        if (dry) return done("", "", "", "", a["path"].str());
        for (size_t i = root.size() + 1; i <= p.size(); i++) if (i == p.size() || p[i] == L'\\') CreateDirectoryW(p.substr(0, i).c_str(), 0);
        return is_dir(p) ? done("made folder " + a["path"].str(), "", "", "", a["path"].str()) : res(false, "could not make the folder");
    }
    if (name == "rename") {
        std::wstring f = P("from"), t = P("to");
        if (f.empty() || t.empty()) return res(false, "both paths must be inside the workspace");
        if (!exists(f)) return res(false, "nothing at " + a["from"].str());
        if (exists(t) && lower(f) != lower(t)) return res(false, "something already exists at " + a["to"].str());
        if (dry) return done("", "", a["from"].str(), a["to"].str(), a["from"].str());
        for (size_t i = root.size() + 1; (i = t.find(L'\\', i)) != std::wstring::npos; i++) CreateDirectoryW(t.substr(0, i).c_str(), 0);
        if (!MoveFileExW(f.c_str(), t.c_str(), 0)) return res(false, "could not rename (in use?)");
        std::string u = new_undo(t, false, "", f);
        return done("renamed " + a["from"].str() + " -> " + a["to"].str(), u, a["from"].str(), a["to"].str(), a["to"].str());
    }
    if (name == "delete") {
        std::wstring p = P("path");
        if (p.empty() || lower(p) == lower(root)) return res(false, "path is outside the workspace (or is the workspace itself)");
        if (!exists(p)) return res(false, "nothing at " + a["path"].str());
        if (dry) return done("", "", a["path"].str(), "", a["path"].str());
        std::wstring from = p; from.push_back(0); from.push_back(0);
        SHFILEOPSTRUCTW op{}; op.wFunc = FO_DELETE; op.pFrom = from.c_str(); op.fFlags = FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_SILENT | FOF_NOERRORUI;
        if (SHFileOperationW(&op) != 0 || exists(p)) return res(false, "could not delete (in use?)");
        return done("moved " + a["path"].str() + " to the Recycle Bin", "", a["path"].str(), "", a["path"].str());
    }
    if (name == "run_command") {
        const std::string c = a["command"].str();
        if (c.empty()) return res(false, "command is empty");
        if (dry) return res(true, "");
        const int secs = std::max(5, std::min(600, (int)a["timeout"].num(120)));
        g_cmd_stop = false;
        return res(true, run_cmd(root, c, secs));
    }
    return res(false, "unknown tool " + name);
}

static std::string undo(const std::string& id) {
    if (id.empty() || id.find_first_of("\\/:.") != std::string::npos) return "{\"error\":\"bad id\"}";
    std::wstring base = undo_dir() + L"\\" + utf8_to_w(id);
    std::string m; if (!rd(base + L".json", m)) return "{\"error\":\"nothing to undo\"}";
    J j; json_parse(m, j);
    std::wstring p = utf8_to_w(j["path"].str()), from = utf8_to_w(j["from"].str());
    bool ok;
    if (!from.empty()) ok = MoveFileExW(p.c_str(), from.c_str(), 0) != 0;
    else if (j["existed"].b) { std::string old; ok = rd(base + L".bin", old) && wr(p, old); }
    else ok = DeleteFileW(p.c_str()) != 0;
    if (ok) { DeleteFileW((base + L".json").c_str()); DeleteFileW((base + L".bin").c_str()); }
    return ok ? "{}" : "{\"error\":\"could not undo (the file changed or is in use)\"}";
}

bool handle_agent(const std::string& method, const std::string& path, const std::string& body, int& code, std::string& type, std::string& out) {
    if (path.rfind("/api/code/", 0) != 0) return false;
    code = 200; type = "application/json";
    J j; json_parse(body, j);
    if (path == "/api/code/pick" && method == "POST") { std::string p = pick_folder(); out = "{\"path\":" + json_esc(p) + "}"; return true; }
    if (path == "/api/code/undo" && method == "POST") { out = undo(j["id"].str()); return true; }
    if (path == "/api/code/stop" && method == "POST") { g_cmd_stop = true; out = "{}"; return true; }
    // the workspace: an existing folder, never a drive root or the Windows / Ithaca folders
    std::wstring root = utf8_to_w(j["root"].str()), full(32768, 0);
    DWORD n = root.empty() ? 0 : GetFullPathNameW(root.c_str(), (DWORD)full.size(), &full[0], nullptr);
    full.resize(n);
    while (full.size() > 3 && full.back() == L'\\') full.pop_back();
    if (full.empty() || !is_dir(full)) { out = "{\"error\":\"Choose a workspace folder first.\"}"; return true; }
    wchar_t win[MAX_PATH]; GetWindowsDirectoryW(win, MAX_PATH);
    if (full.size() <= 3 || lower(full).rfind(lower(win), 0) == 0 || lower(full) == lower(g_store_dir)) { out = "{\"error\":\"That folder is off limits. Pick a project folder.\"}"; return true; }
    if (path == "/api/code/check") { out = "{\"root\":" + json_esc(w_to_utf8(full)) + "}"; return true; }
    if (path == "/api/code/tool" && method == "POST") { out = run_tool(full, j["name"].str(), j["args"], j["dry"].t == J::BOOL && j["dry"].b); return true; }
    return false;
}

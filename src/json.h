// Tiny JSON reader/writer helpers (enough for the local API).
#pragma once
#include <string>
#include <vector>
#include <utility>
#include <cstdlib>
#include <cstdio>
#include <cstring>

struct J {
    enum T { NUL, BOOL, NUM, STR, ARR, OBJ } t = NUL;
    bool b = false; double n = 0; std::string s;
    std::vector<J> a; std::vector<std::pair<std::string, J>> o;
    const J& operator[](const char* k) const {
        static J none;
        for (auto& p : o) if (p.first == k) return p.second;
        return none;
    }
    double num(double d) const { return t == NUM ? n : d; }
    std::string str(const std::string& d = "") const { return t == STR ? s : d; }
};

namespace jdetail {
inline void ws(const char*& p) { while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') p++; }
inline void put_cp(std::string& s, unsigned c) {
    if (c < 0x80) s += (char)c;
    else if (c < 0x800) { s += (char)(0xC0 | (c >> 6)); s += (char)(0x80 | (c & 63)); }
    else if (c < 0x10000) { s += (char)(0xE0 | (c >> 12)); s += (char)(0x80 | ((c >> 6) & 63)); s += (char)(0x80 | (c & 63)); }
    else { s += (char)(0xF0 | (c >> 18)); s += (char)(0x80 | ((c >> 12) & 63)); s += (char)(0x80 | ((c >> 6) & 63)); s += (char)(0x80 | (c & 63)); }
}
inline bool str(const char*& p, std::string& out) {
    if (*p != '"') return false;
    p++;
    while (*p && *p != '"') {
        if (*p == '\\') {
            p++;
            switch (*p) {
            case 'n': out += '\n'; break; case 't': out += '\t'; break; case 'r': out += '\r'; break;
            case 'b': out += '\b'; break; case 'f': out += '\f'; break;
            case 'u': {
                unsigned c = (unsigned)strtoul(std::string(p + 1, 4).c_str(), nullptr, 16); p += 4;
                if (c >= 0xD800 && c < 0xDC00 && p[1] == '\\' && p[2] == 'u') {
                    unsigned lo = (unsigned)strtoul(std::string(p + 3, 4).c_str(), nullptr, 16);
                    c = 0x10000 + ((c - 0xD800) << 10) + (lo - 0xDC00); p += 6;
                }
                put_cp(out, c); break;
            }
            default: out += *p;
            }
            if (*p) p++;
        } else out += *p++;
    }
    if (*p == '"') p++;
    return true;
}
inline bool val(const char*& p, J& j, int depth = 0) {
    if (depth > 64) return false;
    ws(p);
    if (*p == '{') {
        j.t = J::OBJ; p++; ws(p);
        if (*p == '}') { p++; return true; }
        for (;;) {
            ws(p); std::string k; if (!str(p, k)) return false;
            ws(p); if (*p++ != ':') return false;
            J v; if (!val(p, v, depth + 1)) return false;
            j.o.emplace_back(std::move(k), std::move(v));
            ws(p); if (*p == ',') { p++; continue; }
            if (*p == '}') { p++; return true; }
            return false;
        }
    }
    if (*p == '[') {
        j.t = J::ARR; p++; ws(p);
        if (*p == ']') { p++; return true; }
        for (;;) {
            J v; if (!val(p, v, depth + 1)) return false;
            j.a.push_back(std::move(v));
            ws(p); if (*p == ',') { p++; continue; }
            if (*p == ']') { p++; return true; }
            return false;
        }
    }
    if (*p == '"') { j.t = J::STR; return str(p, j.s); }
    if (!strncmp(p, "true", 4)) { j.t = J::BOOL; j.b = true; p += 4; return true; }
    if (!strncmp(p, "false", 5)) { j.t = J::BOOL; p += 5; return true; }
    if (!strncmp(p, "null", 4)) { p += 4; return true; }
    char* e; j.n = strtod(p, &e);
    if (e == p) return false;
    j.t = J::NUM; p = e; return true;
}
}  // namespace jdetail

inline bool json_parse(const std::string& s, J& out) { const char* p = s.c_str(); return jdetail::val(p, out); }

inline std::string json_esc(const std::string& s) {
    std::string o = "\"";
    for (unsigned char c : s) {
        switch (c) {
        case '"': o += "\\\""; break; case '\\': o += "\\\\"; break;
        case '\n': o += "\\n"; break; case '\r': o += "\\r"; break; case '\t': o += "\\t"; break;
        default:
            if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
            else o += (char)c;
        }
    }
    return o + "\"";
}

// Document text extraction for "chat with your files": PDF, DOCX, PPTX, XLSX and plain text.
// Self-contained: own DEFLATE decoder, ZIP reader and a small PDF object/content-stream parser.
#include "common.h"
#include "json.h"
#include <algorithm>
#include <map>
#include <set>

// ---------------------------------------------------------------------------------------------
// DEFLATE (RFC 1951), zlib wrapper optional
namespace {
struct Bits {
    const uint8_t* p; size_t n, pos = 0; uint32_t buf = 0; int cnt = 0; bool err = false;
    Bits(const uint8_t* d, size_t len) : p(d), n(len) {}
    int need(int k) {
        while (cnt < k) { if (pos >= n) { err = true; return 0; } buf |= (uint32_t)p[pos++] << cnt; cnt += 8; }
        int v = buf & ((1u << k) - 1); buf >>= k; cnt -= k; return v;
    }
};
struct Huff { short count[16], sym[320]; };
static int build(Huff& h, const short* len, int n) {
    for (int i = 0; i < 16; i++) h.count[i] = 0;
    for (int i = 0; i < n; i++) h.count[len[i]]++;
    if (h.count[0] == n) return 0;
    int left = 1;
    for (int i = 1; i < 16; i++) { left <<= 1; left -= h.count[i]; if (left < 0) return left; }
    short offs[16]; offs[1] = 0;
    for (int i = 1; i < 15; i++) offs[i + 1] = offs[i] + h.count[i];
    for (int i = 0; i < n; i++) if (len[i]) h.sym[offs[len[i]]++] = (short)i;
    return left;
}
static int decode(Bits& s, const Huff& h) {
    int code = 0, first = 0, index = 0;
    for (int len = 1; len < 16; len++) {
        code |= s.need(1);
        int c = h.count[len];
        if (code - c < first) return h.sym[index + (code - first)];
        index += c; first += c; first <<= 1; code <<= 1;
        if (s.err) return -1;
    }
    return -1;
}
static const short kLBase[29] = {3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
static const short kLExt[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
static const short kDBase[30] = {1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
static const short kDExt[30] = {0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};
static bool codes(Bits& s, std::string& out, const Huff& lc, const Huff& dc, size_t limit) {
    for (;;) {
        int sym = decode(s, lc);
        if (sym < 0 || s.err) return false;
        if (sym < 256) { out += (char)sym; if (out.size() > limit) return false; continue; }
        if (sym == 256) return true;
        sym -= 257;
        if (sym >= 29) return false;
        int len = kLBase[sym] + s.need(kLExt[sym]);
        int ds = decode(s, dc);
        if (ds < 0 || ds >= 30) return false;
        size_t dist = kDBase[ds] + s.need(kDExt[ds]);
        if (dist > out.size()) return false;
        size_t from = out.size() - dist;
        for (int i = 0; i < len; i++) out += out[from + i];
        if (out.size() > limit) return false;
    }
}
}  // namespace

// Raw DEFLATE; returns false on corrupt data (whatever was decoded so far stays in out).
bool inflate_raw(const uint8_t* d, size_t n, std::string& out, size_t limit = 512u << 20) {
    Bits s(d, n);
    int last;
    do {
        last = s.need(1);
        int type = s.need(2);
        if (s.err) return false;
        if (type == 0) {
            s.buf = 0; s.cnt = 0;   // byte align
            if (s.pos + 4 > n) return false;
            unsigned len = d[s.pos] | d[s.pos + 1] << 8;
            s.pos += 4;
            if (s.pos + len > n) return false;
            out.append((const char*)d + s.pos, len); s.pos += len;
        } else if (type == 1) {
            static Huff lc, dc; static bool init = false;
            if (!init) {
                short l[288];
                for (int i = 0; i < 144; i++) l[i] = 8;
                for (int i = 144; i < 256; i++) l[i] = 9;
                for (int i = 256; i < 280; i++) l[i] = 7;
                for (int i = 280; i < 288; i++) l[i] = 8;
                build(lc, l, 288);
                for (int i = 0; i < 30; i++) l[i] = 5;
                build(dc, l, 30);
                init = true;
            }
            if (!codes(s, out, lc, dc, limit)) return false;
        } else if (type == 2) {
            static const short order[19] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};
            int nlen = s.need(5) + 257, ndist = s.need(5) + 1, ncode = s.need(4) + 4;
            if (nlen > 286 || ndist > 30) return false;
            short len[320] = {0};
            for (int i = 0; i < ncode; i++) len[order[i]] = (short)s.need(3);
            Huff cl; if (build(cl, len, 19) != 0) return false;
            int idx = 0;
            while (idx < nlen + ndist) {
                int sym = decode(s, cl);
                if (sym < 0) return false;
                if (sym < 16) len[idx++] = (short)sym;
                else {
                    short v = 0; int rep;
                    if (sym == 16) { if (!idx) return false; v = len[idx - 1]; rep = 3 + s.need(2); }
                    else if (sym == 17) rep = 3 + s.need(3);
                    else rep = 11 + s.need(7);
                    if (idx + rep > nlen + ndist) return false;
                    while (rep--) len[idx++] = v;
                }
            }
            Huff lc, dc;
            if (build(lc, len, nlen) < 0 || build(dc, len + nlen, ndist) < 0) return false;
            if (!codes(s, out, lc, dc, limit)) return false;
        } else return false;
    } while (!last);
    return true;
}
static bool inflate_zlib(const std::string& in, std::string& out) {
    if (in.size() < 2) return false;
    const uint8_t* d = (const uint8_t*)in.data();
    bool zl = (d[0] & 0x0F) == 8 && ((d[0] << 8) | d[1]) % 31 == 0;
    return inflate_raw(d + (zl ? 2 : 0), in.size() - (zl ? 2 : 0), out);
}

// ---------------------------------------------------------------------------------------------
// ZIP (DOCX/PPTX/XLSX are zip archives of XML)
static std::map<std::string, std::string> unzip(const std::string& z, const std::function<bool(const std::string&)>& want) {
    std::map<std::string, std::string> files;
    auto u16 = [&](size_t o) { return o + 2 <= z.size() ? (unsigned)(uint8_t)z[o] | (unsigned)(uint8_t)z[o + 1] << 8 : 0u; };
    auto u32 = [&](size_t o) { return o + 4 <= z.size() ? u16(o) | u16(o + 2) << 16 : 0u; };
    size_t eocd = std::string::npos;
    for (size_t i = z.size() >= 22 ? z.size() - 22 : 0; i + 1 > 0 && i + 65557 >= z.size(); i--) {
        if (u32(i) == 0x06054b50) { eocd = i; break; }
        if (!i) break;
    }
    if (eocd == std::string::npos) return files;
    size_t n = u16(eocd + 10), cd = u32(eocd + 16);
    for (size_t k = 0; k < n && cd + 46 <= z.size() && u32(cd) == 0x02014b50; k++) {
        unsigned method = u16(cd + 10), csize = u32(cd + 20), nlen = u16(cd + 28), elen = u16(cd + 30), clen = u16(cd + 32);
        size_t loc = u32(cd + 42);
        std::string name = z.substr(cd + 46, nlen);
        cd += 46 + nlen + elen + clen;
        if (!want(name) || loc + 30 > z.size() || u32(loc) != 0x04034b50) continue;
        size_t data = loc + 30 + u16(loc + 26) + u16(loc + 28);
        if (data + csize > z.size()) continue;
        std::string out;
        if (method == 0) out = z.substr(data, csize);
        else if (method == 8) inflate_raw((const uint8_t*)z.data() + data, csize, out);
        else continue;
        files[name] = std::move(out);
    }
    return files;
}

// Office XML: text runs, paragraphs/rows as newlines, basic entities.
static std::string xml_text(const std::string& x, const char* run_tag, const char* para_tag) {
    std::string out, rt = std::string("<") + run_tag, pt = std::string("</") + para_tag;
    size_t i = 0;
    while (i < x.size()) {
        size_t lt = x.find('<', i);
        if (lt == std::string::npos) break;
        if (x.compare(lt, pt.size(), pt) == 0) { out += '\n'; i = lt + 1; continue; }
        if (x.compare(lt, 6, "<w:tab") == 0 || x.compare(lt, 6, "<a:tab") == 0) { out += '\t'; i = lt + 1; continue; }
        if (x.compare(lt, 5, "<w:br") == 0 || x.compare(lt, 5, "<a:br") == 0) { out += '\n'; i = lt + 1; continue; }
        if (x.compare(lt, rt.size(), rt) == 0 && (x[lt + rt.size()] == '>' || x[lt + rt.size()] == ' ')) {
            size_t gt = x.find('>', lt);
            if (gt == std::string::npos || x[gt - 1] == '/') { i = lt + 1; continue; }
            size_t end = x.find('<', gt);
            std::string t = x.substr(gt + 1, end - gt - 1);
            for (auto [e, r] : std::initializer_list<std::pair<const char*, const char*>>{{"&lt;", "<"}, {"&gt;", ">"}, {"&quot;", "\""}, {"&apos;", "'"}, {"&amp;", "&"}}) {
                size_t p = 0; while ((p = t.find(e, p)) != std::string::npos) { t.replace(p, strlen(e), r); p += strlen(r); }
            }
            out += t;
            i = end;
            continue;
        }
        i = lt + 1;
    }
    return out;
}

static std::string docx_text(const std::string& z) {
    auto f = unzip(z, [](const std::string& n) { return n == "word/document.xml" || n.rfind("word/footnotes", 0) == 0; });
    std::string out;
    if (f.count("word/document.xml")) out = xml_text(f["word/document.xml"], "w:t", "w:p");
    return out;
}
static std::string pptx_text(const std::string& z) {
    auto f = unzip(z, [](const std::string& n) { return n.rfind("ppt/slides/slide", 0) == 0 && n.find(".xml") != std::string::npos && n.find("_rels") == std::string::npos; });
    std::vector<std::pair<int, std::string>> slides;
    for (auto& kv : f) slides.push_back({atoi(kv.first.c_str() + 16), kv.second});
    std::sort(slides.begin(), slides.end());
    std::string out;
    for (auto& s : slides) out += "[Slide " + std::to_string(s.first) + "]\n" + xml_text(s.second, "a:t", "a:p") + "\n";
    return out;
}
static std::string xlsx_text(const std::string& z) {
    auto f = unzip(z, [](const std::string& n) { return n == "xl/sharedStrings.xml" || (n.rfind("xl/worksheets/sheet", 0) == 0 && n.find("_rels") == std::string::npos); });
    std::vector<std::string> shared;
    {
        const std::string& x = f["xl/sharedStrings.xml"];
        size_t i = 0;
        while ((i = x.find("<si>", i)) != std::string::npos) {
            size_t e = x.find("</si>", i);
            shared.push_back(xml_text(x.substr(i, e - i), "t", "si"));
            i = e;
        }
    }
    std::string out;
    for (auto& kv : f) {
        if (kv.first.rfind("xl/worksheets/", 0) != 0) continue;
        out += "[" + kv.first.substr(14) + "]\n";
        const std::string& x = kv.second;
        size_t i = 0;
        while ((i = x.find("<row", i)) != std::string::npos) {
            size_t re = x.find("</row>", i);
            if (re == std::string::npos) break;
            std::string row = x.substr(i, re - i), line;
            size_t c = 0;
            while ((c = row.find("<c ", c)) != std::string::npos) {
                size_t ce = row.find("</c>", c), se = row.find("/>", c);
                if (ce == std::string::npos || (se != std::string::npos && se < ce && row.find('>', c) == se + 1)) { c++; continue; }
                std::string cell = row.substr(c, ce - c);
                bool str = cell.find("t=\"s\"") != std::string::npos;
                size_t v = cell.find("<v>");
                std::string val = v == std::string::npos ? xml_text(cell, "t", "is") : cell.substr(v + 3, cell.find("</v>") - v - 3);
                if (str) { int k = atoi(val.c_str()); val = k >= 0 && k < (int)shared.size() ? shared[k] : ""; }
                if (!line.empty()) line += "\t";
                line += val;
                c = ce;
            }
            out += line + "\n";
            i = re;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------------------------
// PDF: objects (including compressed object streams), page tree, fonts with ToUnicode maps,
// and the text operators of each page's content stream.
namespace pdf {
struct Obj {
    enum T { NUL, NUM, NAME, STR, ARR, DICT, REF, BOOL, OP } t = NUL;
    double n = 0; std::string s; std::vector<Obj> a; std::map<std::string, Obj> d; int ref = 0;
    const Obj* get(const char* k) const { auto it = d.find(k); return it == d.end() ? nullptr : &it->second; }
};
struct Lexer {
    const std::string& b; size_t i;
    Lexer(const std::string& s, size_t p = 0) : b(s), i(p) {}
    void ws() {
        while (i < b.size()) {
            char c = b[i];
            if (c == '%') { while (i < b.size() && b[i] != '\n' && b[i] != '\r') i++; }
            else if (c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\f' || c == 0) i++;
            else break;
        }
    }
    static bool delim(char c) { return strchr("()<>[]{}/% \n\r\t\f", c) != nullptr || c == 0; }
    bool parse(Obj& o, int depth = 0) {
        ws();
        if (i >= b.size() || depth > 64) return false;
        char c = b[i];
        if (c == '/') {
            i++; o.t = Obj::NAME; size_t s = i;
            while (i < b.size() && !delim(b[i])) i++;
            o.s = b.substr(s, i - s);
            std::string r;   // #xx escapes
            for (size_t k = 0; k < o.s.size(); k++) {
                if (o.s[k] == '#' && k + 2 < o.s.size() && isxdigit((unsigned char)o.s[k + 1]) && isxdigit((unsigned char)o.s[k + 2])) { r += (char)strtol(o.s.substr(k + 1, 2).c_str(), nullptr, 16); k += 2; }
                else r += o.s[k];
            }
            o.s = r;
            return true;
        }
        if (c == '(') {   // literal string with escapes and balanced parens
            i++; o.t = Obj::STR; int depthp = 1;
            while (i < b.size()) {
                char ch = b[i++];
                if (ch == '\\' && i < b.size()) {
                    char e = b[i++];
                    switch (e) {
                    case 'n': o.s += '\n'; break; case 'r': o.s += '\r'; break; case 't': o.s += '\t'; break;
                    case 'b': o.s += '\b'; break; case 'f': o.s += '\f'; break;
                    case '\r': if (i < b.size() && b[i] == '\n') i++; break;
                    case '\n': break;
                    default:
                        if (e >= '0' && e <= '7') {
                            int v = e - '0';
                            for (int k = 0; k < 2 && i < b.size() && b[i] >= '0' && b[i] <= '7'; k++) v = v * 8 + (b[i++] - '0');
                            o.s += (char)v;
                        } else o.s += e;
                    }
                } else if (ch == '(') { depthp++; o.s += ch; }
                else if (ch == ')') { if (--depthp == 0) break; o.s += ch; }
                else o.s += ch;
            }
            return true;
        }
        if (c == '<' && i + 1 < b.size() && b[i + 1] == '<') {
            i += 2; o.t = Obj::DICT;
            for (;;) {
                ws();
                if (i + 1 < b.size() && b[i] == '>' && b[i + 1] == '>') { i += 2; return true; }
                Obj k; if (!parse(k, depth + 1) || k.t != Obj::NAME) return false;
                Obj v; if (!parse(v, depth + 1)) return false;
                o.d[k.s] = std::move(v);
            }
        }
        if (c == '<') {   // hex string
            i++; o.t = Obj::STR; std::string h;
            while (i < b.size() && b[i] != '>') { if (isxdigit((unsigned char)b[i])) h += b[i]; i++; }
            i++;
            if (h.size() & 1) h += '0';
            for (size_t k = 0; k < h.size(); k += 2) o.s += (char)strtol(h.substr(k, 2).c_str(), nullptr, 16);
            return true;
        }
        if (c == '[') {
            i++; o.t = Obj::ARR;
            for (;;) {
                ws();
                if (i < b.size() && b[i] == ']') { i++; return true; }
                Obj v; if (!parse(v, depth + 1)) return false;
                o.a.push_back(std::move(v));
            }
        }
        if (isdigit((unsigned char)c) || c == '-' || c == '+' || c == '.') {
            size_t s = i; i++;
            while (i < b.size() && (isdigit((unsigned char)b[i]) || b[i] == '.')) i++;
            o.t = Obj::NUM; o.n = atof(b.substr(s, i - s).c_str());
            // "N G R" reference?
            size_t save = i; ws();
            size_t s2 = i;
            while (i < b.size() && isdigit((unsigned char)b[i])) i++;
            if (i > s2) {
                ws();
                if (i < b.size() && b[i] == 'R' && (i + 1 >= b.size() || delim(b[i + 1]))) { i++; o.t = Obj::REF; o.ref = (int)o.n; return true; }
            }
            i = save;
            return true;
        }
        if (c == ']' || c == '>' || c == ')' || c == '}' || c == '{') { i++; return false; }
        size_t s = i;
        while (i < b.size() && !delim(b[i])) i++;
        if (i == s) { i++; return false; }
        o.s = b.substr(s, i - s);
        if (o.s == "true" || o.s == "false") { o.t = Obj::BOOL; o.n = o.s == "true"; }
        else if (o.s == "null") o.t = Obj::NUL;
        else o.t = Obj::OP;
        return true;
    }
};

struct Doc {
    const std::string& f;
    std::map<int, size_t> off;                  // object number -> file offset of "N G obj"
    std::map<int, std::pair<int, int>> instm;   // object number -> (object stream, index)
    std::map<int, Obj> cache;
    std::map<int, std::string> stm_cache;
    explicit Doc(const std::string& file) : f(file) {}

    void scan() {
        // Find every "N G obj" directly: robust against broken xref tables. Later definitions win (incremental updates).
        size_t p = 0;
        while ((p = f.find(" obj", p)) != std::string::npos) {
            size_t q = p;
            while (q > 0 && (f[q - 1] == ' ')) q--;
            size_t g = q; while (g > 0 && isdigit((unsigned char)f[g - 1])) g--;
            size_t sp = g; while (sp > 0 && f[sp - 1] == ' ') sp--;
            size_t nn = sp; while (nn > 0 && isdigit((unsigned char)f[nn - 1])) nn--;
            if (nn < sp && g < q) off[atoi(f.c_str() + nn)] = nn;
            p += 4;
        }
        // compressed object streams: map their members
        for (auto& kv : std::map<int, size_t>(off)) {
            Obj o = load(kv.first);
            const Obj* t = o.get("Type");
            if (!t || t->s != "ObjStm") continue;
            std::string data = stream(kv.first);
            Lexer lx(data);
            int n = (int)num(o.get("N"));
            for (int k = 0; k < n; k++) {
                Obj a, b2; if (!lx.parse(a) || !lx.parse(b2)) break;
                if (!off.count((int)a.n)) instm[(int)a.n] = {kv.first, k};
            }
        }
    }
    double num(const Obj* o) { if (!o) return 0; if (o->t == Obj::REF) { Obj r = load(o->ref); return r.n; } return o->n; }
    Obj resolve(const Obj* o) { if (!o) return Obj(); if (o->t == Obj::REF) return load(o->ref); return *o; }
    Obj load(int id) {
        auto c = cache.find(id);
        if (c != cache.end()) return c->second;
        Obj o;
        cache[id] = o;   // guards cycles
        auto it = off.find(id);
        if (it != off.end()) {
            Lexer lx(f, f.find("obj", it->second) + 3);
            lx.parse(o);
        } else {
            auto s = instm.find(id);
            if (s != instm.end()) {
                Obj so = load(s->second.first);
                std::string data = stream(s->second.first);
                Lexer hx(data);
                int n = (int)num(so.get("N")), first = (int)num(so.get("First"));
                for (int k = 0; k < n; k++) {
                    Obj a, b2; if (!hx.parse(a) || !hx.parse(b2)) break;
                    if (k == s->second.second) { Lexer lx(data, first + (size_t)b2.n); lx.parse(o); break; }
                }
            }
        }
        cache[id] = o;
        return o;
    }
    std::string stream(int id) {
        auto c = stm_cache.find(id);
        if (c != stm_cache.end()) return c->second;
        std::string out;
        auto it = off.find(id);
        if (it == off.end()) return out;
        Obj d = load(id);
        size_t s = f.find("stream", f.find("obj", it->second));
        if (s == std::string::npos) return out;
        s += 6;
        if (s < f.size() && f[s] == '\r') s++;
        if (s < f.size() && f[s] == '\n') s++;
        size_t len = (size_t)num(d.get("Length"));
        if (!len || s + len > f.size() || f.compare(s + len, 20, "endstream") > 0) {
            size_t e = f.find("endstream", s);
            if (e == std::string::npos) return out;
            if (!len || s + len > e) len = e - s;
        }
        std::string raw = f.substr(s, len);
        Obj filt = resolve(d.get("Filter"));
        std::vector<std::string> fl;
        if (filt.t == Obj::NAME) fl.push_back(filt.s);
        for (auto& x : filt.a) fl.push_back(x.s);
        for (auto& name : fl) {
            if (name == "FlateDecode" || name == "Fl") { std::string o2; inflate_zlib(raw, o2); raw.swap(o2); }
            else if (name == "ASCII85Decode" || name == "A85") {
                std::string o2; uint32_t v = 0; int n = 0;
                for (size_t k = 0; k < raw.size(); k++) {
                    char ch = raw[k];
                    if (ch == '~') break;
                    if (ch == 'z' && n == 0) { o2.append(4, '\0'); continue; }
                    if (ch < '!' || ch > 'u') continue;
                    v = v * 85 + (ch - '!');
                    if (++n == 5) { for (int q = 3; q >= 0; q--) o2 += (char)(v >> (8 * q)); v = 0; n = 0; }
                }
                if (n > 1) { for (int q = n; q < 5; q++) v = v * 85 + 84; for (int q = 3; q >= 5 - n; q--) o2 += (char)(v >> (8 * q)); }
                raw.swap(o2);
            } else if (name == "ASCIIHexDecode" || name == "AHx") {
                std::string o2, h;
                for (char ch : raw) { if (ch == '>') break; if (isxdigit((unsigned char)ch)) h += ch; }
                if (h.size() & 1) h += '0';
                for (size_t k = 0; k < h.size(); k += 2) o2 += (char)strtol(h.substr(k, 2).c_str(), nullptr, 16);
                raw.swap(o2);
            }
            else return stm_cache[id] = "";   // images (DCT/JPX/CCITT) and rare filters: no text
        }
        return stm_cache[id] = raw;
    }
};

// ToUnicode CMap: code (as bytes) -> UTF-16 string
struct CMap { std::map<std::string, std::u16string> m; int cbytes = 1; };
static std::u16string hex16(const std::string& s) {
    std::u16string u;
    for (size_t i = 0; i + 1 < s.size(); i += 2) u += (char16_t)(((uint8_t)s[i] << 8) | (uint8_t)s[i + 1]);
    return u;
}
static CMap parse_cmap(const std::string& t) {
    CMap c;
    Lexer lx(t);
    std::vector<Obj> st;
    for (;;) {
        Obj o; size_t before = lx.i;
        if (!lx.parse(o)) { if (lx.i >= t.size() || lx.i == before) break; continue; }
        if (o.t == Obj::OP) {
            if (o.s == "endbfchar") {
                for (size_t k = 0; k + 1 < st.size(); k += 2) if (st[k].t == Obj::STR) { c.m[st[k].s] = hex16(st[k + 1].s); c.cbytes = (int)st[k].s.size(); }
            } else if (o.s == "endbfrange") {
                for (size_t k = 0; k + 2 < st.size(); k += 3) {
                    std::string lo = st[k].s, hi = st[k + 1].s;
                    if (lo.empty() || lo.size() != hi.size()) continue;
                    c.cbytes = (int)lo.size();
                    uint32_t a = 0, b = 0;
                    for (char ch : lo) a = a << 8 | (uint8_t)ch;
                    for (char ch : hi) b = b << 8 | (uint8_t)ch;
                    if (b < a || b - a > 65535) continue;
                    for (uint32_t v = a; v <= b; v++) {
                        std::string key(lo.size(), 0);
                        for (size_t q = 0; q < lo.size(); q++) key[lo.size() - 1 - q] = (char)(v >> (8 * q));
                        if (st[k + 2].t == Obj::ARR) { if (v - a < st[k + 2].a.size()) c.m[key] = hex16(st[k + 2].a[v - a].s); }
                        else { std::u16string u = hex16(st[k + 2].s); if (!u.empty()) u.back() = (char16_t)(u.back() + (v - a)); c.m[key] = u; }
                    }
                }
            }
            st.clear();
        } else st.push_back(std::move(o));
    }
    return c;
}
struct Font { CMap cmap; bool two = false, has = false; uint32_t enc[256] = {}; bool has_enc = false; };

static const uint16_t kMacRoman[128] = {0x00C4, 0x00C5, 0x00C7, 0x00C9, 0x00D1, 0x00D6, 0x00DC, 0x00E1, 0x00E0, 0x00E2, 0x00E4, 0x00E3, 0x00E5, 0x00E7, 0x00E9, 0x00E8, 0x00EA, 0x00EB, 0x00ED, 0x00EC, 0x00EE, 0x00EF, 0x00F1, 0x00F3, 0x00F2, 0x00F4, 0x00F6, 0x00F5, 0x00FA, 0x00F9, 0x00FB, 0x00FC, 0x2020, 0x00B0, 0x00A2, 0x00A3, 0x00A7, 0x2022, 0x00B6, 0x00DF, 0x00AE, 0x00A9, 0x2122, 0x00B4, 0x00A8, 0x2260, 0x00C6, 0x00D8, 0x221E, 0x00B1, 0x2264, 0x2265, 0x00A5, 0x00B5, 0x2202, 0x2211, 0x220F, 0x03C0, 0x222B, 0x00AA, 0x00BA, 0x03A9, 0x00E6, 0x00F8, 0x00BF, 0x00A1, 0x00AC, 0x221A, 0x0192, 0x2248, 0x2206, 0x00AB, 0x00BB, 0x2026, 0x00A0, 0x00C0, 0x00C3, 0x00D5, 0x0152, 0x0153, 0x2013, 0x2014, 0x201C, 0x201D, 0x2018, 0x2019, 0x00F7, 0x25CA, 0x00FF, 0x0178, 0x2044, 0x20AC, 0x2039, 0x203A, 0xFB01, 0xFB02, 0x2021, 0x00B7, 0x201A, 0x201E, 0x2030, 0x00C2, 0x00CA, 0x00C1, 0x00CB, 0x00C8, 0x00CD, 0x00CE, 0x00CF, 0x00CC, 0x00D3, 0x00D4, 0xF8FF, 0x00D2, 0x00DA, 0x00DB, 0x00D9, 0x0131, 0x02C6, 0x02DC, 0x00AF, 0x02D8, 0x02D9, 0x02DA, 0x00B8, 0x02DD, 0x02DB, 0x02C7};
// Common glyph names used in /Differences arrays
static uint32_t glyph_unicode(const std::string& n) {
    if (n.size() == 1) return (unsigned char)n[0];
    if (n.size() == 7 && n.compare(0, 3, "uni") == 0) return (uint32_t)strtoul(n.c_str() + 3, nullptr, 16);
    static const std::map<std::string, uint32_t> m = {
        {"space", ' '}, {"exclam", '!'}, {"quotedbl", '"'}, {"numbersign", '#'}, {"dollar", '$'}, {"percent", '%'}, {"ampersand", '&'},
        {"quotesingle", '\''}, {"parenleft", '('}, {"parenright", ')'}, {"asterisk", '*'}, {"plus", '+'}, {"comma", ','}, {"hyphen", '-'},
        {"period", '.'}, {"slash", '/'}, {"zero", '0'}, {"one", '1'}, {"two", '2'}, {"three", '3'}, {"four", '4'}, {"five", '5'},
        {"six", '6'}, {"seven", '7'}, {"eight", '8'}, {"nine", '9'}, {"colon", ':'}, {"semicolon", ';'}, {"less", '<'}, {"equal", '='},
        {"greater", '>'}, {"question", '?'}, {"at", '@'}, {"bracketleft", '['}, {"backslash", '\\'}, {"bracketright", ']'},
        {"asciicircum", '^'}, {"underscore", '_'}, {"grave", '`'}, {"braceleft", '{'}, {"bar", '|'}, {"braceright", '}'}, {"asciitilde", '~'},
        {"bullet", 0x2022}, {"fi", 0xFB01}, {"fl", 0xFB02}, {"ff", 0xFB00}, {"ffi", 0xFB03}, {"ffl", 0xFB04}, {"quoteright", 0x2019},
        {"quoteleft", 0x2018}, {"quotedblleft", 0x201C}, {"quotedblright", 0x201D}, {"quotesinglbase", 0x201A}, {"quotedblbase", 0x201E},
        {"endash", 0x2013}, {"emdash", 0x2014}, {"ellipsis", 0x2026}, {"minus", 0x2212}, {"degree", 0xB0}, {"copyright", 0xA9},
        {"registered", 0xAE}, {"trademark", 0x2122}, {"section", 0xA7}, {"paragraph", 0xB6}, {"dagger", 0x2020}, {"daggerdbl", 0x2021},
        {"multiply", 0xD7}, {"divide", 0xF7}, {"plusminus", 0xB1}, {"mu", 0xB5}, {"periodcentered", 0xB7}, {"Euro", 0x20AC},
        {"guillemotleft", 0xAB}, {"guillemotright", 0xBB}, {"nbspace", 0xA0}, {"germandbls", 0xDF}, {"eacute", 0xE9}, {"egrave", 0xE8},
        {"aacute", 0xE1}, {"agrave", 0xE0}, {"udieresis", 0xFC}, {"odieresis", 0xF6}, {"adieresis", 0xE4}, {"ccedilla", 0xE7}};
    auto it = m.find(n);
    return it == m.end() ? 0 : it->second;
}

static void put_utf8(std::string& o, uint32_t cp) {
    if (cp < 0x80) o += (char)cp;
    else if (cp < 0x800) { o += (char)(0xC0 | cp >> 6); o += (char)(0x80 | (cp & 63)); }
    else if (cp < 0x10000) { o += (char)(0xE0 | cp >> 12); o += (char)(0x80 | ((cp >> 6) & 63)); o += (char)(0x80 | (cp & 63)); }
    else { o += (char)(0xF0 | cp >> 18); o += (char)(0x80 | ((cp >> 12) & 63)); o += (char)(0x80 | ((cp >> 6) & 63)); o += (char)(0x80 | (cp & 63)); }
}
static void u16_to_utf8(std::string& o, const std::u16string& u) {
    for (size_t i = 0; i < u.size(); i++) {
        uint32_t c = u[i];
        if (c >= 0xD800 && c < 0xDC00 && i + 1 < u.size()) { c = 0x10000 + ((c - 0xD800) << 10) + (u[i + 1] - 0xDC00); i++; }
        if (c) put_utf8(o, c);
    }
}
// WinAnsi bytes 0x80-0x9F (the rest is Latin-1)
static const uint16_t kWinAnsi[32] = {0x20AC, 0, 0x201A, 0x0192, 0x201E, 0x2026, 0x2020, 0x2021, 0x02C6, 0x2030, 0x0160, 0x2039, 0x0152, 0, 0x017D, 0,
                                      0, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2013, 0x2014, 0x02DC, 0x2122, 0x0161, 0x203A, 0x0153, 0, 0x017E, 0x0178};
static void decode_str(std::string& o, const std::string& s, const Font* f) {
    if (f && f->has) {
        int nb = f->two ? 2 : f->cmap.cbytes;
        for (size_t i = 0; i < s.size(); i += nb) {
            auto it = f->cmap.m.find(s.substr(i, nb));
            if (it != f->cmap.m.end()) u16_to_utf8(o, it->second);
            else if (nb == 1) put_utf8(o, (uint8_t)s[i]);
        }
        return;
    }
    if (f && f->two) return;   // CID font without a map: glyph ids, no text
    for (unsigned char c : s) {
        if (f && f->has_enc && f->enc[c]) { put_utf8(o, f->enc[c]); continue; }
        put_utf8(o, c >= 0x80 && c < 0xA0 ? (kWinAnsi[c - 0x80] ? kWinAnsi[c - 0x80] : ' ') : c);
    }
}

static void run_content(Doc& doc, const std::string& cs, std::map<std::string, Font>& fonts, std::string& out, int depth,
                        const std::map<std::string, Obj>& xobjs) {
    Lexer lx(cs);
    std::vector<Obj> st;
    const Font* cur = nullptr;
    double last_y = 1e9, ty = 0;
    for (;;) {
        Obj o; size_t before = lx.i;
        if (!lx.parse(o)) { if (lx.i >= cs.size() || lx.i == before) break; st.clear(); continue; }
        if (o.t != Obj::OP) { st.push_back(std::move(o)); continue; }
        const std::string& op = o.s;
        if (op == "BI") { size_t e = cs.find("EI", lx.i); lx.i = e == std::string::npos ? cs.size() : e + 2; }   // inline image
        else if (op == "Tf" && st.size() >= 2) { auto it = fonts.find(st[st.size() - 2].s); cur = it == fonts.end() ? nullptr : &it->second; }
        else if (op == "Tj" && !st.empty()) decode_str(out, st.back().s, cur);
        else if ((op == "'" || op == "\"") && !st.empty()) { out += '\n'; decode_str(out, st.back().s, cur); }
        else if (op == "TJ" && !st.empty()) {
            for (auto& e : st.back().a) {
                if (e.t == Obj::STR) decode_str(out, e.s, cur);
                else if (e.t == Obj::NUM && e.n < -180 && !out.empty() && out.back() != ' ') out += ' ';   // wide kerning gap = space
            }
        } else if (op == "Td" || op == "TD" || op == "Tm" || op == "T*") {
            double y = op == "Tm" && st.size() >= 6 ? st[5].n : (op == "T*" ? ty - 1 : (st.size() >= 2 ? ty + st[1].n : ty));
            if (op == "Tm") ty = y; else if (op != "T*") ty = y; else ty -= 1;
            if (std::abs(y - last_y) > 0.5 || op == "T*") { if (!out.empty() && out.back() != '\n') out += '\n'; }
            else if (!out.empty() && out.back() != ' ' && out.back() != '\n') out += ' ';
            last_y = y;
        } else if (op == "ET") { if (!out.empty() && out.back() != ' ' && out.back() != '\n') out += ' '; }
        else if (op == "Do" && !st.empty() && depth < 4) {   // form XObjects can hold text too
            auto x = xobjs.find(st.back().s);
            if (x != xobjs.end() && x->second.t == Obj::REF) {
                Obj xo = doc.load(x->second.ref);
                const Obj* sub = xo.get("Subtype");
                if (sub && sub->s == "Form") run_content(doc, doc.stream(x->second.ref), fonts, out, depth + 1, xobjs);
            }
        }
        st.clear();
    }
}

static std::string extract(const std::string& file, int& pages) {
    Doc doc(file);
    doc.scan();
    // page tree
    std::vector<int> page_ids;
    std::function<void(const Obj&, int)> walk = [&](const Obj& node, int depth) {
        if (depth > 32) return;
        const Obj* kids = node.get("Kids");
        Obj k = doc.resolve(kids);
        for (auto& r : k.a) {
            if (r.t != Obj::REF) continue;
            Obj ch = doc.load(r.ref);
            const Obj* t = ch.get("Type");
            if (t && t->s == "Page") page_ids.push_back(r.ref);
            else walk(ch, depth + 1);
        }
    };
    for (auto& kv : doc.off) {
        Obj o = doc.load(kv.first);
        const Obj* t = o.get("Type");
        if (t && t->s == "Catalog") { Obj pages_root = doc.resolve(o.get("Pages")); walk(pages_root, 0); break; }
    }
    if (page_ids.empty())   // no usable catalog: take page objects in file order
        for (auto& kv : doc.off) { Obj o = doc.load(kv.first); const Obj* t = o.get("Type"); if (t && t->s == "Page") page_ids.push_back(kv.first); }
    for (auto& kv : doc.instm) if (page_ids.empty()) { Obj o = doc.load(kv.first); const Obj* t = o.get("Type"); if (t && t->s == "Page") page_ids.push_back(kv.first); }
    pages = (int)page_ids.size();
    std::string out;
    std::map<int, Font> font_cache;
    for (size_t pi = 0; pi < page_ids.size(); pi++) {
        Obj page = doc.load(page_ids[pi]);
        // resources may be inherited from the parent
        Obj res = doc.resolve(page.get("Resources"));
        for (Obj p = page; res.t != Obj::DICT && p.get("Parent");) { p = doc.resolve(p.get("Parent")); res = doc.resolve(p.get("Resources")); }
        std::map<std::string, Font> fonts;
        Obj fd = doc.resolve(res.get("Font"));
        for (auto& kv : fd.d) {
            if (kv.second.t != Obj::REF) continue;
            auto fc = font_cache.find(kv.second.ref);
            if (fc == font_cache.end()) {
                Font F;
                Obj fo = doc.load(kv.second.ref);
                const Obj* sub = fo.get("Subtype");
                F.two = sub && sub->s == "Type0";
                const Obj* tu = fo.get("ToUnicode");
                if (tu && tu->t == Obj::REF) { F.cmap = parse_cmap(doc.stream(tu->ref)); F.has = !F.cmap.m.empty(); }
                // simple fonts: base encoding + /Differences
                Obj e = doc.resolve(fo.get("Encoding"));
                std::string base = e.t == Obj::NAME ? e.s : (e.get("BaseEncoding") ? e.get("BaseEncoding")->s : "");
                if (base == "MacRomanEncoding") { for (int k = 0; k < 128; k++) F.enc[128 + k] = kMacRoman[k]; F.has_enc = true; }
                if (const Obj* dif = e.get("Differences")) {
                    int code = 0;
                    for (auto& x : dif->a) {
                        if (x.t == Obj::NUM) code = (int)x.n;
                        else if (x.t == Obj::NAME && code >= 0 && code < 256) { if (uint32_t u = glyph_unicode(x.s)) { F.enc[code] = u; F.has_enc = true; } code++; }
                    }
                }
                fc = font_cache.emplace(kv.second.ref, F).first;
            }
            fonts[kv.first] = fc->second;
        }
        Obj xo = doc.resolve(res.get("XObject"));
        std::string content;
        Obj c = *(page.get("Contents") ? page.get("Contents") : &page);
        if (c.t == Obj::REF) {
            Obj co = doc.load(c.ref);
            if (co.t == Obj::ARR) c = co;
            else content = doc.stream(c.ref);
        }
        for (auto& r : c.a) if (r.t == Obj::REF) content += doc.stream(r.ref) + "\n";
        std::string txt;
        run_content(doc, content, fonts, txt, 0, xo.d);
        out += "[Page " + std::to_string(pi + 1) + "]\n" + txt + "\n\n";
    }
    return out;
}
}  // namespace pdf

static std::string lower_ext(const std::string& name) {
    size_t d = name.find_last_of('.');
    std::string e = d == std::string::npos ? "" : name.substr(d + 1);
    for (auto& c : e) c = (char)tolower(c);
    return e;
}
static bool valid_utf8_text(const std::string& s) {
    size_t bad = 0;
    for (size_t i = 0; i < std::min<size_t>(s.size(), 65536); i++) { unsigned char c = s[i]; if (c == 0 || (c < 9)) bad++; }
    return bad * 100 < std::min<size_t>(s.size(), 65536) + 1;
}

// Returns {"name","text","pages","chars"} or {"error"}.
std::string doc_extract_json(const std::string& name, const std::string& bytes) {
    std::string ext = lower_ext(name), text, kind;
    int pages = 0;
    if (ext == "pdf" || bytes.compare(0, 5, "%PDF-") == 0) { text = pdf::extract(bytes, pages); kind = "pdf"; }
    else if (ext == "docx") { text = docx_text(bytes); kind = "docx"; }
    else if (ext == "pptx") { text = pptx_text(bytes); kind = "pptx"; }
    else if (ext == "xlsx") { text = xlsx_text(bytes); kind = "xlsx"; }
    else if (ext == "doc" || ext == "ppt" || ext == "xls") return "{\"error\":\"Old Office formats (.doc/.ppt/.xls) aren't supported. Save it as .docx/.pptx/.xlsx or PDF.\"}";
    else if (valid_utf8_text(bytes)) { text = bytes; kind = "text"; if (text.compare(0, 3, "\xEF\xBB\xBF") == 0) text.erase(0, 3); }
    else return "{\"error\":\"This file type isn't supported. Try PDF, Word, PowerPoint, Excel or a text/code file.\"}";
    // tidy: collapse runs of spaces and blank lines
    std::string t; int nl = 0;
    for (char c : text) {
        if (c == '\r') continue;
        if (c == '\n') { if (++nl <= 2) t += c; continue; }
        if (c == ' ' && !t.empty() && t.back() == ' ') continue;
        nl = 0; t += c;
    }
    size_t letters = 0;   // real text, not our "[Page N]" labels
    for (size_t i = 0; i < t.size(); i++) {
        if (t[i] == '[' && (i == 0 || t[i - 1] == '\n') && t.compare(i, 6, "[Page ") == 0) { i = t.find('\n', i); if (i == std::string::npos) break; continue; }
        if (isalnum((unsigned char)t[i]) || (unsigned char)t[i] >= 0x80) letters++;
    }
    if (letters < 20) {
        if (kind == "pdf") return "{\"error\":\"No text found in this PDF. It may be scanned images; try attaching the pages as pictures instead.\"}";
        return "{\"error\":\"No text found in this file.\"}";
    }
    return "{\"name\":" + json_esc(name) + ",\"kind\":\"" + kind + "\",\"text\":" + json_esc(t) + ",\"pages\":" + std::to_string(pages) +
           ",\"chars\":" + std::to_string(t.size()) + "}";
}

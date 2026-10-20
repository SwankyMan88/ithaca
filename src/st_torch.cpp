// PyTorch checkpoint reader (.pt / .pth / .bin zip format): a small zip directory reader plus a
// pickle interpreter that understands enough to rebuild a state_dict of tensors (no Python needed).
#include "gx.h"
#include <memory>

namespace {
struct Obj;
using P = std::shared_ptr<Obj>;
struct Obj {
    enum K { NONE, INT, FLOAT, STR, TUPLE, LIST, DICT, GLOBAL, STORAGE, TENSOR, OTHER } k = NONE;
    int64_t i = 0; double f = 0; std::string s, s2;      // s/s2: str or module/name, storage dtype/key
    std::vector<P> a;                                       // tuple/list items
    std::vector<std::pair<P, P>> d;                         // dict
    STensor t;
};
P mk(Obj::K k) { auto p = std::make_shared<Obj>(); p->k = k; return p; }

const char* storage_dtype(const std::string& n, int& es) {
    struct M { const char* n; const char* d; int es; } m[] = {{"FloatStorage", "F32", 4}, {"HalfStorage", "F16", 2}, {"BFloat16Storage", "BF16", 2},
        {"LongStorage", "I64", 8}, {"IntStorage", "I32", 4}, {"ByteStorage", "U8", 1}, {"BoolStorage", "BOOL", 1}, {"DoubleStorage", "F64", 8},
        {"CharStorage", "I8", 1}, {"ShortStorage", "I16", 2}};
    for (auto& x : m) if (n == x.n) { es = x.es; return x.d; }
    es = 4; return "F32";
}
}  // namespace

// Returns false when the data is not a zip-format torch checkpoint.
bool torch_load_zip(const uint8_t* base, size_t size, std::map<std::string, STensor>& out, std::string& err) {
    if (size < 22 || base[0] != 'P' || base[1] != 'K') return false;
    auto u16 = [&](size_t o) { return (uint32_t)base[o] | ((uint32_t)base[o + 1] << 8); };
    auto u32 = [&](size_t o) { return u16(o) | (u16(o + 2) << 16); };
    auto u64 = [&](size_t o) { return (uint64_t)u32(o) | ((uint64_t)u32(o + 4) << 32); };
    size_t eocd = SIZE_MAX;
    for (size_t o = size - 22; o + 1 > 0 && o + 65558 > size; o--) { if (u32(o) == 0x06054b50) { eocd = o; break; } if (o == 0) break; }
    if (eocd == SIZE_MAX) { err = "bad zip"; return false; }
    uint64_t cd = u32(eocd + 16), n = u16(eocd + 10);
    if ((cd == 0xFFFFFFFF || n == 0xFFFF) && eocd >= 20 && u32(eocd - 20) == 0x07064b50) {   // zip64
        uint64_t loc = u64(eocd - 20 + 8);
        n = u64(loc + 32); cd = u64(loc + 48);
    }
    std::map<std::string, std::pair<const uint8_t*, size_t>> files;
    size_t o = (size_t)cd;
    for (uint64_t e = 0; e < n && o + 46 <= size && u32(o) == 0x02014b50; e++) {
        uint32_t method = u16(o + 10), nl = u16(o + 28), xl = u16(o + 30), cl = u16(o + 32);
        uint64_t csz = u32(o + 20), lho = u32(o + 42);
        std::string name((const char*)base + o + 46, nl);
        // zip64 extra field
        size_t x = o + 46 + nl, xe = x + xl;
        while (x + 4 <= xe) {
            uint32_t id = u16(x), sz = u16(x + 2);
            if (id == 1) { size_t q = x + 4; if (u32(o + 24) == 0xFFFFFFFF) q += 8; if (csz == 0xFFFFFFFF) { csz = u64(q); q += 8; } if (lho == 0xFFFFFFFF) lho = u64(q); }
            x += 4 + sz;
        }
        if (method == 0 && lho + 30 <= size) {
            size_t data = (size_t)lho + 30 + u16((size_t)lho + 26) + u16((size_t)lho + 28);
            files[name] = {base + data, (size_t)csz};
        }
        o += 46 + nl + xl + cl;
    }
    std::string pkl, prefix;
    for (auto& f : files) {
        size_t s = f.first.rfind("data.pkl");
        if (s != std::string::npos && s + 8 == f.first.size()) { pkl = f.first; prefix = f.first.substr(0, s); break; }
    }
    if (pkl.empty()) { err = "no data.pkl in checkpoint"; return true; }
    const uint8_t* p = files[pkl].first; const uint8_t* end = p + files[pkl].second;
    std::vector<P> st;
    std::vector<size_t> marks;
    std::map<uint64_t, P> memo;
    auto pop = [&]() { P v = st.empty() ? mk(Obj::NONE) : st.back(); if (!st.empty()) st.pop_back(); return v; };
    auto pop_mark = [&]() { size_t m = marks.empty() ? 0 : marks.back(); if (!marks.empty()) marks.pop_back(); std::vector<P> v(st.begin() + m, st.end()); st.resize(m); return v; };
    auto rd = [&](size_t k) { uint64_t v = 0; for (size_t q = 0; q < k && p < end; q++) v |= (uint64_t)*p++ << (8 * q); return v; };
    auto line = [&]() { std::string s; while (p < end && *p != '\n') s += (char)*p++; if (p < end) p++; return s; };
    auto rebuild = [&](P callable, P args) -> P {
        std::string fn = callable->s + "." + callable->s2;
        if (fn.find("_rebuild_tensor") != std::string::npos && args->a.size() >= 4 && args->a[0]->k == Obj::STORAGE) {
            P st0 = args->a[0];
            int es; const char* dt = storage_dtype(st0->s, es);
            auto it = files.find(prefix + "data/" + st0->s2);
            P t = mk(Obj::TENSOR);
            t->t.dtype = dt;
            for (auto& d : args->a[2]->a) t->t.shape.push_back(d->i);
            if (it != files.end()) { t->t.data = it->second.first + args->a[1]->i * es; t->t.nbytes = (size_t)t->t.numel() * es; }
            return t;
        }
        if (fn.find("OrderedDict") != std::string::npos) return mk(Obj::DICT);
        return mk(Obj::OTHER);
    };
    while (p < end) {
        uint8_t op = *p++;
        switch (op) {
        case 0x80: p++; break;                              // PROTO
        case 0x95: p += 8; break;                           // FRAME
        case '.': p = end; break;                           // STOP
        case '(': marks.push_back(st.size()); break;        // MARK
        case '}': st.push_back(mk(Obj::DICT)); break;
        case ']': st.push_back(mk(Obj::LIST)); break;
        case ')': st.push_back(mk(Obj::TUPLE)); break;
        case 'N': st.push_back(mk(Obj::NONE)); break;
        case 0x88: case 0x89: { P v = mk(Obj::INT); v->i = op == 0x88; st.push_back(v); break; }
        case 'J': { P v = mk(Obj::INT); v->i = (int32_t)rd(4); st.push_back(v); break; }
        case 'K': { P v = mk(Obj::INT); v->i = rd(1); st.push_back(v); break; }
        case 'M': { P v = mk(Obj::INT); v->i = rd(2); st.push_back(v); break; }
        case 0x8a: { size_t n1 = rd(1); P v = mk(Obj::INT); v->i = (int64_t)rd(n1); st.push_back(v); break; }
        case 'G': { uint64_t b = 0; for (int q = 0; q < 8; q++) b = (b << 8) | *p++; P v = mk(Obj::FLOAT); memcpy(&v->f, &b, 8); st.push_back(v); break; }
        case 'X': { size_t n1 = rd(4); P v = mk(Obj::STR); v->s.assign((const char*)p, n1); p += n1; st.push_back(v); break; }
        case 0x8c: { size_t n1 = rd(1); P v = mk(Obj::STR); v->s.assign((const char*)p, n1); p += n1; st.push_back(v); break; }
        case 'B': { size_t n1 = rd(4); P v = mk(Obj::STR); v->s.assign((const char*)p, n1); p += n1; st.push_back(v); break; }
        case 'C': { size_t n1 = rd(1); P v = mk(Obj::STR); v->s.assign((const char*)p, n1); p += n1; st.push_back(v); break; }
        case 'q': memo[rd(1)] = st.empty() ? mk(Obj::NONE) : st.back(); break;
        case 'r': memo[rd(4)] = st.empty() ? mk(Obj::NONE) : st.back(); break;
        case 0x94: memo[memo.size()] = st.empty() ? mk(Obj::NONE) : st.back(); break;
        case 'h': st.push_back(memo[rd(1)]); break;
        case 'j': st.push_back(memo[rd(4)]); break;
        case 'c': { P g = mk(Obj::GLOBAL); g->s = line(); g->s2 = line(); st.push_back(g); break; }
        case 0x93: { P nm = pop(), md = pop(); P g = mk(Obj::GLOBAL); g->s = md->s; g->s2 = nm->s; st.push_back(g); break; }
        case 't': { P v = mk(Obj::TUPLE); v->a = pop_mark(); st.push_back(v); break; }
        case 0x85: { P v = mk(Obj::TUPLE); v->a = {pop()}; st.push_back(v); break; }
        case 0x86: { P b = pop(), a = pop(); P v = mk(Obj::TUPLE); v->a = {a, b}; st.push_back(v); break; }
        case 0x87: { P c3 = pop(), b = pop(), a = pop(); P v = mk(Obj::TUPLE); v->a = {a, b, c3}; st.push_back(v); break; }
        case 'R': { P args = pop(), fn = pop(); st.push_back(rebuild(fn, args)); break; }
        case 0x81: { P args = pop(), cls = pop(); st.push_back(rebuild(cls, args)); break; }   // NEWOBJ
        case 'b': pop(); break;                                                              // BUILD: ignore state
        case 'Q': {                                                                          // BINPERSID
            P pid = pop(); P s = mk(Obj::STORAGE);
            if (pid->a.size() >= 3) { s->s = pid->a[1]->s2; s->s2 = pid->a[2]->s; }
            st.push_back(s); break;
        }
        case 's': { P v = pop(), k = pop(); if (!st.empty() && st.back()->k == Obj::DICT) st.back()->d.push_back({k, v}); break; }
        case 'u': { auto items = pop_mark(); if (!st.empty() && st.back()->k == Obj::DICT) for (size_t q = 0; q + 1 < items.size(); q += 2) st.back()->d.push_back({items[q], items[q + 1]}); break; }
        case 'a': { P v = pop(); if (!st.empty()) st.back()->a.push_back(v); break; }
        case 'e': { auto items = pop_mark(); if (!st.empty()) for (auto& v : items) st.back()->a.push_back(v); break; }
        case '0': pop(); break;
        default: err = "unsupported pickle opcode " + std::to_string(op); p = end; break;
        }
    }
    if (st.empty()) { err = "empty checkpoint"; return true; }
    std::function<void(const P&, const std::string&)> walk = [&](const P& o2, const std::string& pre) {
        if (o2->k == Obj::TENSOR && o2->t.data) out[pre] = o2->t;
        else if (o2->k == Obj::DICT) for (auto& kv : o2->d) walk(kv.second, pre.empty() ? kv.first->s : pre + "." + kv.first->s);
    };
    walk(st.back(), "");
    return true;
}

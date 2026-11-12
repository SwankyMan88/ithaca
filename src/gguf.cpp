#include "gguf.h"

struct TypeInfo { int blck; int bytes; const char* name; };
static TypeInfo tinfo(int t) {
    switch (t) {
    case GT_F32: return {1, 4, "F32"};
    case GT_F16: return {1, 2, "F16"};
    case GT_BF16: return {1, 2, "BF16"};
    case GT_Q4_0: return {32, 18, "Q4_0"};
    case GT_Q4_1: return {32, 20, "Q4_1"};
    case GT_Q5_0: return {32, 22, "Q5_0"};
    case GT_Q5_1: return {32, 24, "Q5_1"};
    case GT_Q8_0: return {32, 34, "Q8_0"};
    case GT_Q2_K: return {256, 84, "Q2_K"};
    case GT_Q3_K: return {256, 110, "Q3_K"};
    case GT_Q4_K: return {256, 144, "Q4_K"};
    case GT_Q5_K: return {256, 176, "Q5_K"};
    case GT_Q6_K: return {256, 210, "Q6_K"};
    case GT_IQ2_XXS: return {256, 66, "IQ2_XXS"};
    case GT_IQ2_XS: return {256, 74, "IQ2_XS"};
    case GT_IQ2_S: return {256, 82, "IQ2_S"};
    case GT_IQ3_XXS: return {256, 98, "IQ3_XXS"};
    case GT_IQ3_S: return {256, 110, "IQ3_S"};
    case GT_IQ1_S: return {256, 50, "IQ1_S"};
    case GT_IQ1_M: return {256, 56, "IQ1_M"};
    case GT_IQ4_NL: return {32, 18, "IQ4_NL"};
    case GT_IQ4_XS: return {256, 136, "IQ4_XS"};
    case GT_MXFP4: return {32, 17, "MXFP4"};
    default: return {0, 0, "?"};
    }
}
const char* gtype_name(int t) { return tinfo(t).name; }
bool gtype_supported(int t) {
    switch (t) {
    case GT_F32: case GT_F16: case GT_BF16: case GT_Q4_0: case GT_Q4_1: case GT_Q5_0:
    case GT_Q5_1: case GT_Q8_0: case GT_Q2_K: case GT_Q3_K: case GT_Q4_K: case GT_Q5_K: case GT_Q6_K: return true;
    case GT_IQ2_XXS: case GT_IQ2_XS: case GT_IQ2_S: case GT_IQ3_XXS: case GT_IQ3_S: case GT_IQ1_S: case GT_IQ1_M:
    case GT_IQ4_NL: case GT_IQ4_XS: case GT_MXFP4: return true;
    default: return false;
    }
}
size_t gtype_row_bytes(int t, int64_t n) {
    TypeInfo i = tinfo(t);
    if (!i.blck) return 0;
    return (size_t)(n / i.blck) * i.bytes;
}

namespace {
struct Reader {
    const uint8_t* p; const uint8_t* end; bool bad = false;
    template <class T> T rd() {
        T v{};
        if (p + sizeof(T) > end) { bad = true; return v; }
        memcpy(&v, p, sizeof(T)); p += sizeof(T);
        return v;
    }
    std::string str() {
        uint64_t n = rd<uint64_t>();
        if (bad || p + n > end) { bad = true; return {}; }
        std::string s((const char*)p, n); p += n;
        return s;
    }
};
bool read_scalar(Reader& r, int t, GValue& v) {
    switch (t) {
    case 0: v.i = r.rd<uint8_t>(); break;
    case 1: v.i = r.rd<int8_t>(); break;
    case 2: v.i = r.rd<uint16_t>(); break;
    case 3: v.i = r.rd<int16_t>(); break;
    case 4: v.i = r.rd<uint32_t>(); break;
    case 5: v.i = r.rd<int32_t>(); break;
    case 6: v.f = r.rd<float>(); v.i = (int64_t)v.f; return !r.bad;
    case 7: v.i = r.rd<uint8_t>(); break;
    case 8: v.s = r.str(); return !r.bad;
    case 10: v.i = (int64_t)r.rd<uint64_t>(); break;
    case 11: v.i = r.rd<int64_t>(); break;
    case 12: v.f = r.rd<double>(); v.i = (int64_t)v.f; return !r.bad;
    default: return false;
    }
    v.f = (double)v.i;
    return !r.bad;
}
}  // namespace

// Header of one GGUF file: metadata (when kvout) and tensors, appended to `out` with data pointers into the mapping.
static bool parse_gguf(const uint8_t* base, size_t size, std::unordered_map<std::string, GValue>* kvout, std::vector<GTensor>& out, int shard, std::string& err);

bool GGUF::open(const std::wstring& path, std::string& err) {
    close();
    paths.push_back(path);
    hfile = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                        FILE_ATTRIBUTE_NORMAL, nullptr);
    if (hfile == INVALID_HANDLE_VALUE) { err = "cannot open file"; return false; }
    LARGE_INTEGER sz; GetFileSizeEx(hfile, &sz); size = (size_t)sz.QuadPart;
    hmap = CreateFileMappingW(hfile, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!hmap) { err = "cannot map file"; close(); return false; }
    base = (const uint8_t*)MapViewOfFile(hmap, FILE_MAP_READ, 0, 0, 0);
    if (!base) { err = "cannot map view (out of address space?)"; close(); return false; }

    if (!parse_gguf(base, size, &kv, tensors, 0, err)) { close(); return false; }
    // Split GGUF (name-00001-of-0000N.gguf): open the other parts too; their tensors join this one's list.
    const int nsplit = (int)geti("split.count", 1);
    if (nsplit > 1) {
        std::wstring p = path; size_t at = p.rfind(L"-00001-of-");
        if (at == std::wstring::npos) { err = "this is part of a split model: open its first part (-00001-of-...)"; close(); return false; }
        for (int i = 1; i < nsplit; i++) {
            wchar_t num[8]; swprintf(num, 8, L"%05d", i + 1);
            std::wstring q = p.substr(0, at + 1) + num + p.substr(at + 6);
            Shard s;
            s.hfile = CreateFileW(q.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
            if (s.hfile == INVALID_HANDLE_VALUE) { err = "missing split part " + w_to_utf8(q.substr(q.find_last_of(L'\\') + 1)); close(); return false; }
            LARGE_INTEGER z; GetFileSizeEx(s.hfile, &z); s.size = (size_t)z.QuadPart;
            s.hmap = CreateFileMappingW(s.hfile, nullptr, PAGE_READONLY, 0, 0, nullptr);
            s.base = s.hmap ? (const uint8_t*)MapViewOfFile(s.hmap, FILE_MAP_READ, 0, 0, 0) : nullptr;
            extra.push_back(s); paths.push_back(q);
            if (!s.base) { err = "cannot map split part"; close(); return false; }
            if (!parse_gguf(s.base, s.size, nullptr, tensors, i, err)) { close(); return false; }
        }
    }
    for (size_t i = 0; i < tensors.size(); i++) index[tensors[i].name] = (int)i;
    return true;
}

static bool parse_gguf(const uint8_t* base, size_t size, std::unordered_map<std::string, GValue>* kvout, std::vector<GTensor>& out, int shard, std::string& err) {
    std::unordered_map<std::string, GValue> kvl;
    std::unordered_map<std::string, GValue>& kv = kvout ? *kvout : kvl;
    Reader r{base, base + size};
    if (r.rd<uint32_t>() != 0x46554747) { err = "not a GGUF file"; return false; }
    uint32_t ver = r.rd<uint32_t>();
    if (ver < 2) { err = "GGUF v1 not supported"; return false; }
    uint64_t nt = r.rd<uint64_t>(), nk = r.rd<uint64_t>();
    for (uint64_t i = 0; i < nk && !r.bad; i++) {
        std::string k = r.str();
        GValue v; v.type = (int)r.rd<uint32_t>();
        if (v.type == 9) {
            v.arr_type = (int)r.rd<uint32_t>();
            uint64_t n = r.rd<uint64_t>();
            if (v.arr_type == 8) {
                v.arr_s.reserve(n);
                for (uint64_t j = 0; j < n && !r.bad; j++) v.arr_s.push_back(r.str());
            } else {
                v.arr_n.reserve(n);
                for (uint64_t j = 0; j < n && !r.bad; j++) {
                    GValue e; if (!read_scalar(r, v.arr_type, e)) { r.bad = true; break; }
                    v.arr_n.push_back(e.f);
                }
            }
        } else if (!read_scalar(r, v.type, v)) { r.bad = true; }
        kv[k] = std::move(v);
    }
    std::vector<uint64_t> offs;
    const size_t first = out.size();
    out.resize(first + nt);
    for (uint64_t i = 0; i < nt && !r.bad; i++) {
        GTensor& t = out[first + i];
        t.name = r.str();
        t.nd = (int)r.rd<uint32_t>();
        for (int d = 0; d < t.nd && d < 4; d++) t.ne[d] = (int64_t)r.rd<uint64_t>();
        t.type = (int)r.rd<uint32_t>();
        offs.push_back(r.rd<uint64_t>());
    }
    if (r.bad) { err = "truncated / corrupt GGUF header"; return false; }
    auto ai = kv.find("general.alignment");
    uint64_t align = ai != kv.end() && ai->second.i > 0 ? (uint64_t)ai->second.i : 32;
    uint64_t data0 = (uint64_t)(r.p - base);
    data0 = (data0 + align - 1) / align * align;
    for (uint64_t i = 0; i < nt; i++) {
        GTensor& t = out[first + i];
        t.data = base + data0 + offs[i]; t.shard = shard; t.foff = data0 + offs[i];
        t.nbytes = gtype_row_bytes(t.type, t.ne[0]) * t.ne[1] * t.ne[2] * t.ne[3];
        if (data0 + offs[i] + t.nbytes > size) { err = "tensor " + t.name + " out of file bounds (incomplete download?)"; return false; }
    }
    return true;
}

void GGUF::close() {
    if (base) UnmapViewOfFile(base);
    if (hmap) CloseHandle(hmap);
    if (hfile != INVALID_HANDLE_VALUE) CloseHandle(hfile);
    base = nullptr; hmap = nullptr; hfile = INVALID_HANDLE_VALUE; size = 0;
    for (auto& s : extra) { if (s.base) UnmapViewOfFile(s.base); if (s.hmap) CloseHandle(s.hmap); if (s.hfile != INVALID_HANDLE_VALUE) CloseHandle(s.hfile); }
    extra.clear(); paths.clear();
    kv.clear(); tensors.clear(); index.clear();
}

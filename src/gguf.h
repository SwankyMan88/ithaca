// Memory-mapped GGUF reader. Opening is O(metadata): tensor data is never copied,
// the OS pages it in on demand (the "storage" tier).
#pragma once
#include "common.h"
#include <unordered_map>

enum GType {
    GT_F32 = 0, GT_F16 = 1, GT_Q4_0 = 2, GT_Q4_1 = 3, GT_Q5_0 = 6, GT_Q5_1 = 7, GT_Q8_0 = 8,
    GT_Q2_K = 10, GT_Q3_K = 11, GT_Q4_K = 12, GT_Q5_K = 13, GT_Q6_K = 14, GT_BF16 = 30,
    GT_IQ2_XXS = 16, GT_IQ2_XS = 17, GT_IQ3_XXS = 18, GT_IQ1_S = 19, GT_IQ4_NL = 20, GT_IQ3_S = 21, GT_IQ2_S = 22,
    GT_IQ4_XS = 23, GT_IQ1_M = 29, GT_MXFP4 = 39,
};

struct GValue {
    int type = -1;
    int64_t i = 0;
    double f = 0;
    std::string s;
    int arr_type = -1;
    std::vector<std::string> arr_s;
    std::vector<double> arr_n;
};

struct GTensor {
    std::string name;
    int type = 0;
    int nd = 0;
    int64_t ne[4] = {1, 1, 1, 1};
    const uint8_t* data = nullptr;
    size_t nbytes = 0;
    int shard = 0;        // split GGUF: which file (0 = the one opened)
    uint64_t foff = 0;    // byte offset of the data in that file
};

const char* gtype_name(int t);
bool gtype_supported(int t);
size_t gtype_row_bytes(int t, int64_t n);

struct GGUF {
    HANDLE hfile = INVALID_HANDLE_VALUE, hmap = nullptr;
    const uint8_t* base = nullptr;
    size_t size = 0;
    std::unordered_map<std::string, GValue> kv;
    std::vector<GTensor> tensors;
    std::unordered_map<std::string, int> index;
    struct Shard { HANDLE hfile = INVALID_HANDLE_VALUE, hmap = nullptr; const uint8_t* base = nullptr; size_t size = 0; };
    std::vector<Shard> extra;              // split GGUF: parts 2..N, opened with the first
    std::vector<std::wstring> paths;       // every part's path, the opened one first

    ~GGUF() { close(); }
    bool open(const std::wstring& path, std::string& err);
    void close();
    const GTensor* get(const std::string& n) const {
        auto it = index.find(n);
        return it == index.end() ? nullptr : &tensors[it->second];
    }
    const GValue* find(const std::string& k) const {
        auto it = kv.find(k);
        return it == kv.end() ? nullptr : &it->second;
    }
    int64_t geti(const std::string& k, int64_t d) const { auto v = find(k); return v ? v->i : d; }
    double getf(const std::string& k, double d) const { auto v = find(k); return v ? v->f : d; }
    std::string gets(const std::string& k, const std::string& d = "") const { auto v = find(k); return v ? v->s : d; }
};

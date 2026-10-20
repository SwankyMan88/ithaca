#include "tokenizer.h"
#include <algorithm>

static void put_utf8(std::string& s, uint32_t c) {
    if (c < 0x80) s += (char)c;
    else if (c < 0x800) { s += (char)(0xC0 | (c >> 6)); s += (char)(0x80 | (c & 63)); }
    else if (c < 0x10000) { s += (char)(0xE0 | (c >> 12)); s += (char)(0x80 | ((c >> 6) & 63)); s += (char)(0x80 | (c & 63)); }
    else { s += (char)(0xF0 | (c >> 18)); s += (char)(0x80 | ((c >> 12) & 63)); s += (char)(0x80 | ((c >> 6) & 63)); s += (char)(0x80 | (c & 63)); }
}
static int utf8_len(unsigned char c) { return c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1; }
static uint32_t decode_cp(const std::string& s, size_t i, int& len) {
    unsigned char c = s[i]; len = utf8_len(c);
    if (i + len > s.size()) { len = 1; return c; }
    if (len == 1) return c;
    uint32_t cp = c & (0xFF >> (len + 1));
    for (int k = 1; k < len; k++) cp = (cp << 6) | (s[i + k] & 63);
    return cp;
}

enum { TT_NORMAL = 1, TT_UNKNOWN = 2, TT_CONTROL = 3, TT_USER = 4, TT_UNUSED = 5, TT_BYTE = 6 };

bool Tokenizer::load(const GGUF& g, std::string& err) {
    *this = Tokenizer();  // a previously loaded model's vocabulary must not leak into this one
    const GValue* tv = g.find("tokenizer.ggml.tokens");
    if (!tv || tv->arr_s.empty()) { err = "model has no embedded tokenizer"; return false; }
    tokens_ = tv->arr_s;
    std::string model = g.gets("tokenizer.ggml.model", "gpt2");
    gpt2_ = model != "llama";
    std::string pre = g.gets("tokenizer.ggml.pre", "");
    digits_ = (pre.find("qwen") != std::string::npos || pre.find("deepseek") != std::string::npos) ? 1 : 3;
    if (auto sv = g.find("tokenizer.ggml.scores")) for (double d : sv->arr_n) scores_.push_back((float)d);
    if (auto ty = g.find("tokenizer.ggml.token_type")) for (double d : ty->arr_n) types_.push_back((int)d);
    types_.resize(tokens_.size(), TT_NORMAL);
    scores_.resize(tokens_.size(), 0.f);
    vocab_.reserve(tokens_.size() * 2);
    for (int i = 0; i < (int)tokens_.size(); i++) vocab_.emplace(tokens_[i], i);
    if (auto mv = g.find("tokenizer.ggml.merges")) {
        merges_.reserve(mv->arr_s.size() * 2);
        for (int i = 0; i < (int)mv->arr_s.size(); i++) merges_.emplace(mv->arr_s[i], i);
    }
    bos = (int)g.geti("tokenizer.ggml.bos_token_id", -1);
    eos = (int)g.geti("tokenizer.ggml.eos_token_id", -1);
    add_bos = g.geti("tokenizer.ggml.add_bos_token", gpt2_ ? 0 : 1) != 0;

    // GPT-2 byte <-> unicode table
    int n = 0;
    for (int b = 0; b < 256; b++) {
        bool keep = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
        uint32_t cp = keep ? b : 256 + n++;
        put_utf8(byte_enc_[b], cp);
        byte_dec_[cp] = (uint8_t)b;
    }
    for (int b = 0; b < 256; b++) {
        char buf[8]; snprintf(buf, sizeof buf, "<0x%02X>", b);
        auto it = vocab_.find(buf);
        byte_tok_[b] = it != vocab_.end() ? it->second : -1;
    }
    specials_by_byte_.assign(256, {});
    for (int i = 0; i < (int)tokens_.size(); i++)
        if ((types_[i] == TT_CONTROL || types_[i] == TT_USER) && tokens_[i].size() > 1)
            specials_by_byte_[(unsigned char)tokens_[i][0]].push_back(i);
    for (auto& v : specials_by_byte_)
        std::sort(v.begin(), v.end(), [&](int a, int b) { return tokens_[a].size() > tokens_[b].size(); });
    return true;
}

bool Tokenizer::is_stop(int id) const {
    if (id == eos) return true;
    if (id < 0 || id >= (int)types_.size()) return false;
    if (types_[id] != TT_CONTROL) {
        const std::string& t = tokens_[id];
        return t == "<|im_end|>" || t == "<|eot_id|>" || t == "<|end|>" || t == "<|endoftext|>" || t == "<|end_of_text|>";
    }
    return id != bos;
}

std::string Tokenizer::piece(int id) const {
    if (id < 0 || id >= (int)tokens_.size()) return "";
    int t = types_[id];
    if (t == TT_CONTROL || t == TT_UNUSED) return "";
    const std::string& s = tokens_[id];
    if (t == TT_USER) return s;
    if (gpt2_) {
        std::string out;
        for (size_t i = 0; i < s.size();) {
            int len; uint32_t cp = decode_cp(s, i, len);
            auto it = byte_dec_.find(cp);
            if (it != byte_dec_.end()) out += (char)it->second; else out.append(s, i, len);
            i += len;
        }
        return out;
    }
    if (t == TT_BYTE && s.size() == 6) return std::string(1, (char)strtol(s.c_str() + 3, nullptr, 16));
    std::string out;
    for (size_t i = 0; i < s.size();) {
        if (i + 3 <= s.size() && (unsigned char)s[i] == 0xE2 && (unsigned char)s[i + 1] == 0x96 && (unsigned char)s[i + 2] == 0x81) { out += ' '; i += 3; }
        else out += s[i++];
    }
    return out;
}

// ---- GPT-2 style BPE ----
static bool is_letter(uint32_t c) {
    if (c < 128) return (c | 32) >= 'a' && (c | 32) <= 'z';
    if (c >= 0x2000 && c <= 0x206F) return false;   // general punctuation
    if (c >= 0x3000 && c <= 0x303F) return false;   // CJK punctuation
    if (c >= 0xFF00 && c <= 0xFF0F) return false;
    if (c >= 0xFF1A && c <= 0xFF20) return false;
    if (c == 0xA0 || c == 0xA7 || c == 0xB7 || c == 0xAB || c == 0xBB || (c >= 0xA1 && c <= 0xBF && c != 0xAA && c != 0xB5 && c != 0xBA)) return false;
    if (c >= 0x2190 && c <= 0x2BFF) return false;   // arrows, math, symbols
    if (c >= 0x1F000) return false;                 // emoji
    return true;
}
static bool is_digit_cp(uint32_t c) { return c >= '0' && c <= '9'; }
static bool is_space_cp(uint32_t c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f' || c == 0x3000 || c == 0xA0; }

void Tokenizer::bpe_word(const std::string& w, std::vector<int>& out) const {
    std::vector<std::string> sym;
    for (unsigned char b : w) sym.push_back(byte_enc_[b]);
    {
        auto it = vocab_.find(std::string());
        (void)it;
    }
    std::string whole; for (auto& s : sym) whole += s;
    auto direct = vocab_.find(whole);
    if (direct != vocab_.end()) { out.push_back(direct->second); return; }
    while (sym.size() > 1) {
        int best = -1, best_rank = INT32_MAX;
        for (size_t i = 0; i + 1 < sym.size(); i++) {
            auto it = merges_.find(sym[i] + " " + sym[i + 1]);
            if (it != merges_.end() && it->second < best_rank) { best_rank = it->second; best = (int)i; }
        }
        if (best < 0) break;
        sym[best] += sym[best + 1];
        sym.erase(sym.begin() + best + 1);
    }
    for (auto& s : sym) {
        auto it = vocab_.find(s);
        if (it != vocab_.end()) { out.push_back(it->second); continue; }
        for (size_t i = 0; i < s.size();) {  // fall back to single mapped bytes
            int len; decode_cp(s, i, len);
            auto jt = vocab_.find(s.substr(i, len));
            if (jt != vocab_.end()) out.push_back(jt->second);
            i += len;
        }
    }
}

void Tokenizer::encode_text(const std::string& s, std::vector<int>& out) const {
    if (!gpt2_) { spm_text(s, out); return; }
    // Hand-written equivalent of the GPT-4 / Qwen2 / Llama-3 pre-tokenizer regex.
    std::vector<uint32_t> cp; std::vector<size_t> off;
    for (size_t i = 0; i < s.size();) { int len; cp.push_back(decode_cp(s, i, len)); off.push_back(i); i += len; }
    off.push_back(s.size());
    size_t n = cp.size(), i = 0;
    auto emit = [&](size_t a, size_t b) { if (b > a) bpe_word(s.substr(off[a], off[b] - off[a]), out); };
    while (i < n) {
        uint32_t c = cp[i];
        // contractions
        if (c == '\'' && i + 1 < n) {
            auto lc = [&](size_t k) { return k < n ? (cp[k] | 32) : 0u; };
            uint32_t a = lc(i + 1), b = lc(i + 2);
            if (a == 's' || a == 't' || a == 'm' || a == 'd') { emit(i, i + 2); i += 2; continue; }
            if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) { emit(i, i + 3); i += 3; continue; }
        }
        // [^\r\n\p{L}\p{N}]?\p{L}+
        if (is_letter(c) || (c != '\r' && c != '\n' && !is_digit_cp(c) && i + 1 < n && is_letter(cp[i + 1]))) {
            size_t j = i + 1;
            while (j < n && is_letter(cp[j])) j++;
            emit(i, j); i = j; continue;
        }
        if (is_digit_cp(c)) {
            size_t j = i;
            while (j < n && is_digit_cp(cp[j]) && j - i < (size_t)digits_) j++;
            emit(i, j); i = j; continue;
        }
        //  ?[^\s\p{L}\p{N}]+[\r\n]*
        {
            size_t j = i;
            if (cp[j] == ' ' && j + 1 < n) j++;
            if (!is_space_cp(cp[j]) && !is_letter(cp[j]) && !is_digit_cp(cp[j])) {
                while (j < n && !is_space_cp(cp[j]) && !is_letter(cp[j]) && !is_digit_cp(cp[j])) j++;
                while (j < n && (cp[j] == '\r' || cp[j] == '\n')) j++;
                emit(i, j); i = j; continue;
            }
        }
        // whitespace
        size_t j = i;
        while (j < n && is_space_cp(cp[j])) j++;
        size_t last_nl = SIZE_MAX;
        for (size_t k = i; k < j; k++) if (cp[k] == '\r' || cp[k] == '\n') last_nl = k;
        if (last_nl != SIZE_MAX) { emit(i, last_nl + 1); i = last_nl + 1; continue; }
        if (j < n && j - i > 1) { emit(i, j - 1); i = j - 1; continue; }
        emit(i, j); i = j;
    }
}

// ---- SentencePiece (llama) ----
void Tokenizer::spm_text(const std::string& s, std::vector<int>& out) const {
    std::string t;
    for (char c : s) { if (c == ' ') t += "\xE2\x96\x81"; else t += c; }
    std::vector<std::string> sym;
    for (size_t i = 0; i < t.size();) { int len = utf8_len(t[i]); sym.push_back(t.substr(i, len)); i += len; }
    while (sym.size() > 1) {
        int best = -1; float bs = -1e30f; int bid = -1;
        for (size_t i = 0; i + 1 < sym.size(); i++) {
            auto it = vocab_.find(sym[i] + sym[i + 1]);
            if (it != vocab_.end() && scores_[it->second] > bs) { bs = scores_[it->second]; best = (int)i; bid = it->second; }
        }
        if (best < 0) break;
        (void)bid;
        sym[best] += sym[best + 1];
        sym.erase(sym.begin() + best + 1);
    }
    for (auto& x : sym) {
        auto it = vocab_.find(x);
        if (it != vocab_.end()) { out.push_back(it->second); continue; }
        for (unsigned char b : x) if (byte_tok_[b] >= 0) out.push_back(byte_tok_[b]);
    }
}

std::vector<int> Tokenizer::encode(const std::string& text, bool with_bos) const {
    std::vector<int> out;
    if (with_bos && add_bos && bos >= 0) out.push_back(bos);
    size_t i = 0, seg = 0;
    bool first = true;
    auto flush = [&](size_t a, size_t b) {
        if (b <= a) return;
        std::string part = text.substr(a, b - a);
        if (!gpt2_ && first) part = " " + part;  // SPM space prefix
        first = false;
        encode_text(part, out);
    };
    while (i < text.size()) {
        int hit = -1;
        for (int id : specials_by_byte_[(unsigned char)text[i]]) {
            const std::string& sp = tokens_[id];
            if (text.compare(i, sp.size(), sp) == 0) { hit = id; break; }
        }
        if (hit >= 0) {
            flush(seg, i);
            out.push_back(hit);
            i += tokens_[hit].size(); seg = i;
            first = false;
        } else i++;
    }
    flush(seg, text.size());
    return out;
}

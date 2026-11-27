#include "chat.h"
#include <algorithm>
#include <cmath>
#include <ctime>
#include <immintrin.h>

static bool has(const std::string& s, const char* x) { return s.find(x) != std::string::npos; }

static std::string strip_think(const std::string& s) {
    size_t e = s.find("</think>");
    if (e == std::string::npos) return s;
    size_t i = e + 8;
    while (i < s.size() && (s[i] == '\n' || s[i] == ' ')) i++;
    return s.substr(i);
}

std::string Chat::render(const std::vector<Msg>& msgs) const {
    const std::string& t = m_.chat_template;
    std::string sys, out;
    for (auto& m : msgs) if (m.role == "system") sys = m.content;
    auto each = [&](auto fn) { for (auto& m : msgs) if (m.role != "system") fn(m.role, m.role == "assistant" ? strip_think(m.content) : m.content); };

    if (has(t, "<\xEF\xBD\x9C" "User\xEF\xBD\x9C>")) {  // DeepSeek-R1 (fullwidth bars)
        out = sys;
        each([&](const std::string& r, const std::string& c) {
            if (r == "user") out += "<\xEF\xBD\x9CUser\xEF\xBD\x9C>" + c;
            else out += "<\xEF\xBD\x9C" "Assistant\xEF\xBD\x9C>" + c + "<\xEF\xBD\x9C" "end\xE2\x96\x81of\xE2\x96\x81sentence\xEF\xBD\x9C>";
        });
        return out + "<\xEF\xBD\x9C" "Assistant\xEF\xBD\x9C><think>\n";
    }
    if (has(t, "<|channel|>")) {  // gpt-oss (harmony): reasoning goes to the analysis channel, the answer to final
        char date[16]; time_t now = time(nullptr); strftime(date, sizeof date, "%Y-%m-%d", localtime(&now));
        out = std::string("<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: ") + date +
              "\n\nReasoning: medium\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>";
        if (!sys.empty()) out += "<|start|>developer<|message|># Instructions\n\n" + sys + "<|end|>";
        each([&](const std::string& r, const std::string& c) {
            if (r == "user") out += "<|start|>user<|message|>" + c + "<|end|>";
            else out += "<|start|>assistant<|channel|>final<|message|>" + c + "<|end|>";
        });
        return out + "<|start|>assistant";
    }
    if (has(t, "<|start_header_id|>")) {  // Llama 3
        auto blk = [&](const std::string& r, const std::string& c) { out += "<|start_header_id|>" + r + "<|end_header_id|>\n\n" + c + "<|eot_id|>"; };
        if (!sys.empty()) blk("system", sys);
        each(blk);
        return out + "<|start_header_id|>assistant<|end_header_id|>\n\n";
    }
    if (has(t, "<|start_of_role|>")) {  // Granite
        auto blk = [&](const std::string& r, const std::string& c) { out += "<|start_of_role|>" + r + "<|end_of_role|>" + c + "<|end_of_text|>\n"; };
        if (!sys.empty()) blk("system", sys);
        each(blk);
        return out + "<|start_of_role|>assistant<|end_of_role|>";
    }
    if (has(t, "<|im_sep|>")) {  // Phi-4
        auto blk = [&](const std::string& r, const std::string& c) { out += "<|im_start|>" + r + "<|im_sep|>" + c + "<|im_end|>"; };
        if (!sys.empty()) blk("system", sys);
        each(blk);
        return out + "<|im_start|>assistant<|im_sep|>";
    }
    if (has(t, "[INST]") || (!has(t, "<|im_start|>") && m_.hp.arch == "llama")) {  // Mistral / Llama 2
        bool first = true;
        each([&](const std::string& r, const std::string& c) {
            if (r == "user") { out += "[INST] " + (first && !sys.empty() ? sys + "\n\n" : "") + c + " [/INST]"; first = false; }
            else out += " " + c + "</s>";
        });
        return out;
    }
    if (has(t, "<|user|>") && !has(t, "<|im_start|>")) {  // Phi-3
        if (!sys.empty()) out += "<|system|>\n" + sys + "<|end|>\n";
        each([&](const std::string& r, const std::string& c) { out += "<|" + r + "|>\n" + c + "<|end|>\n"; });
        return out + "<|assistant|>\n";
    }
    if (has(t, "<start_of_turn>")) {  // Gemma: no system role, it is folded into the first user turn
        bool first = true;
        each([&](const std::string& r, const std::string& c) {
            std::string body = first && r == "user" && !sys.empty() ? sys + "\n\n" + c : c;
            if (r == "user") first = false;
            out += "<start_of_turn>" + std::string(r == "assistant" ? "model" : "user") + "\n" + body + "<end_of_turn>\n";
        });
        return out + "<start_of_turn>model\n";
    }
    if (has(t, "<|begin_of_sentence|>") && has(t, "Assistant: ")) {  // ERNIE 4.5
        out = "<|begin_of_sentence|>" + (sys.empty() ? std::string() : sys + "\n");
        each([&](const std::string& r, const std::string& c) {
            if (r == "user") out += "User: " + c + "\n";
            else out += "Assistant: " + c + "<|end_of_sentence|>";
        });
        return out + "Assistant: ";
    }
    if (has(t, "User: ") && has(t, "Assistant:")) {  // DeepSeek V2 (BOS comes from the tokenizer)
        if (!sys.empty()) out += sys + "\n\n";
        each([&](const std::string& r, const std::string& c) {
            if (r == "user") out += "User: " + c + "\n\n";
            else out += "Assistant: " + c + "<\xEF\xBD\x9C" "end\xE2\x96\x81of\xE2\x96\x81sentence\xEF\xBD\x9C>";
        });
        return out + "Assistant:";
    }
    // ChatML default (Qwen 2/2.5/3 and most others)
    auto blk = [&](const std::string& r, const std::string& c) { out += "<|im_start|>" + r + "\n" + c + "<|im_end|>\n"; };
    if (!sys.empty()) blk("system", sys);
    each(blk);
    return out + "<|im_start|>assistant\n";
}

bool Chat::token_text(int id, std::string& out) {
    out.clear();
    if (!harmony()) { if (m_.tok.is_stop(id)) return true; out = m_.tok.piece(id); return false; }
    const std::string& t = m_.tok.text(id);
    if (t == "<|return|>" || t == "<|call|>" || id == m_.tok.eos) { if (hs_.think) { out = "\n</think>\n\n"; hs_.think = false; } return true; }
    if (t == "<|start|>" || t == "<|channel|>") { hs_.header = true; if (t == "<|channel|>") hs_.chan.clear(); return false; }
    if (t == "<|message|>") {
        hs_.header = false;
        const bool fin = hs_.chan.compare(0, 5, "final") == 0;
        if (!fin && !hs_.think) { out = "<think>\n"; hs_.think = true; }
        else if (fin && hs_.think) { out = "\n</think>\n\n"; hs_.think = false; }
        return false;
    }
    if (t == "<|end|>" || t == "<|constrain|>") return false;
    if (m_.tok.is_stop(id)) return true;
    if (hs_.header) { hs_.chan += m_.tok.piece(id); return false; }
    out = m_.tok.piece(id);
    return false;
}

// The best M logits (value high to low, then id low to high) after the repetition penalty, in one pass over the
// vocabulary. The penalty only ever lowers a logit, so the penalized top M are among the raw top M + R, R = the
// number of distinct recent tokens: only those few get penalized and re-sorted. Same picks as building, penalizing
// and sorting the whole vocabulary, without touching 2 MB of memory per token.
static void top_logits(const float* lg, int n, int M, float pen, const std::vector<int>& recent, std::vector<std::pair<float, int>>& out) {
    std::vector<int> ids;   // distinct recent tokens
    if (pen != 1.f) for (int id : recent) if (id >= 0 && id < n && std::find(ids.begin(), ids.end(), id) == ids.end()) ids.push_back(id);
    const int W = std::min(n, M + (int)ids.size());
    auto better = [](const std::pair<float, int>& a, const std::pair<float, int>& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); };
    std::vector<std::pair<float, int>>& h = out;   // heap with the worst kept candidate on top
    h.clear(); h.reserve(W + 1);
    int i = 0;
    for (; i < n && (int)h.size() < W; i++) { h.push_back({lg[i], i}); std::push_heap(h.begin(), h.end(), better); }
    float thr = h.empty() ? 0.f : h.front().first;
    auto take = [&](int j) {   // ids arrive in increasing order, so an equal value is never better than the kept one
        if (!(lg[j] > thr)) return;
        std::pop_heap(h.begin(), h.end(), better); h.back() = {lg[j], j}; std::push_heap(h.begin(), h.end(), better);
        thr = h.front().first;
    };
    for (; i < n && (i & 7); i++) take(i);
    for (; i + 8 <= n; i += 8) {
        int m = _mm256_movemask_ps(_mm256_cmp_ps(_mm256_loadu_ps(lg + i), _mm256_set1_ps(thr), _CMP_GT_OQ));
        while (m) { const int b = __builtin_ctz(m); m &= m - 1; take(i + b); }
    }
    for (; i < n; i++) take(i);
    for (auto& c : h)   // the penalty once per appearance, in order, as before
        if (std::find(ids.begin(), ids.end(), c.second) != ids.end())
            for (int id : recent) if (id == c.second) c.first = c.first > 0 ? c.first / pen : c.first * pen;
    std::sort(h.begin(), h.end(), better);
    if ((int)h.size() > M) h.resize(M);
}

// Prompt lookup: when the last few tokens already appeared earlier in the conversation, guess that what followed
// them then follows again (code being edited, quoted text, lists, repeated names). Longest match, most recent first.
static int lookup_draft(const std::vector<int>& h, int K, std::vector<int>& d) {   // returns the match length (0: none)
    d.clear();
    const int n = (int)h.size();
    for (int N = 4; N >= 2; N--) {
        if (n < N + 1) continue;
        const int* tail = h.data() + n - N;
        for (int p = n - N - 1; p >= 0; p--)
            if (h[p] == tail[0] && std::equal(tail, tail + N, h.data() + p)) {
                for (int k = 0; k < K && p + N + k < n; k++) d.push_back(h[p + N + k]);
                return N;
            }
    }
    return 0;
}

int Chat::sample(const float* logits, const GenOpts& o, const std::vector<int>& recent) {
    const int n = m_.hp.n_vocab;
    std::vector<std::pair<float, int>> c;
    if (o.temp <= 0.f) { top_logits(logits, n, 1, o.rep_pen, recent, c); return c.empty() ? 0 : c[0].second; }
    top_logits(logits, n, std::min(n, std::max(1, o.top_k)), o.rep_pen, recent, c);
    if (c.empty()) return 0;
    float mx = c[0].first, sum = 0;
    for (auto& p : c) { p.first = std::exp((p.first - mx) / o.temp); sum += p.first; }
    float cum = 0; size_t cut = c.size();
    for (size_t i = 0; i < c.size(); i++) { c[i].first /= sum; cum += c[i].first; if (cum >= o.top_p) { cut = i + 1; break; } }
    c.resize(cut);
    float r = std::uniform_real_distribution<float>(0.f, cum)(rng_), acc = 0;
    for (auto& p : c) { acc += p.first; if (r <= acc) return p.second; }
    return c.back().second;
}

// modes: 0 value expected, 1 after '{' (key or '}'), 2 key expected, 3 ':' expected, 4 ',' or close expected,
// 5 in string, 6 after '\' in string, 7 \u hex digits, 11 number, 12 true/false/null, 13 complete, 14 after '[' (value or ']')
bool JsonCheck::feed(char c) {
    const bool ws = c == ' ' || c == '\n' || c == '\r' || c == '\t';
    if (mode != 5 && mode != 6 && mode != 7 && mode != 13) {   // indentation is fine; pages of padding are not
        if (ws) { if (++wsrun > 24) return false; } else wsrun = 0;
    }
    auto after_value = [&] { mode = stack.empty() ? 13 : 4; };
    switch (mode) {
    case 13: return true;
    case 0: case 14:
        if (ws) return true;
        if (mode == 14 && c == ']') { stack.pop_back(); after_value(); return true; }
        if (c == '{') { stack += '{'; mode = 1; return true; }
        if (c == '[') { stack += '['; mode = 14; return true; }
        if (c == '"') { key = false; mode = 5; return true; }
        if (c == '-' || (c >= '0' && c <= '9')) { mode = 11; return true; }
        if (c == 't' || c == 'f' || c == 'n') { lit = c; mode = 12; return true; }
        return false;
    case 1:
        if (ws) return true;
        if (c == '}') { stack.pop_back(); after_value(); return true; }
        if (c == '"') { key = true; mode = 5; return true; }
        return false;
    case 2:
        if (ws) return true;
        if (c == '"') { key = true; mode = 5; return true; }
        return false;
    case 3:
        if (ws) return true;
        if (c == ':') { mode = 0; return true; }
        return false;
    case 4:
        if (ws) return true;
        if (c == ',') { mode = stack.back() == '{' ? 2 : 0; return true; }
        if ((c == '}' && stack.back() == '{') || (c == ']' && stack.back() == '[')) { stack.pop_back(); after_value(); return true; }
        return false;
    case 5:
        if (c == '"') { if (key) mode = 3; else after_value(); return true; }
        if (c == '\\') { mode = 6; return true; }
        return (unsigned char)c >= 0x20;   // raw line breaks and tabs are not allowed inside a JSON string
    case 6:
        if (strchr("\"\\/bfnrt", c) && c) { mode = 5; return true; }
        if (c == 'u') { hex = 4; mode = 7; return true; }
        return false;
    case 7:
        if (!isxdigit((unsigned char)c)) return false;
        if (--hex == 0) mode = 5;
        return true;
    case 11:
        if ((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') return true;
        after_value(); return feed(c);
    case 12: {
        lit += c;
        for (const char* w : {"true", "false", "null"}) {
            if (lit == w) { after_value(); return true; }
            if (strncmp(w, lit.c_str(), lit.size()) == 0) return true;
        }
        return false;
    }
    }
    return false;
}

int Chat::sample_json(const float* logits, const GenOpts& o, const std::vector<int>& recent, const JsonCheck& jc) {
    const int n = m_.hp.n_vocab;
    // look at the most likely tokens first; keep the ones whose text keeps the JSON valid
    const int scan = std::min(n, 4096);
    std::vector<std::pair<float, int>> c;
    top_logits(logits, n, scan, o.rep_pen, recent, c);
    std::vector<std::pair<float, int>> ok;
    const int want = o.temp <= 0.f ? 1 : std::max(1, o.top_k);
    for (int i = 0; i < (int)c.size() && (int)ok.size() < want; i++) {
        const int id = c[i].second;
        if (m_.tok.is_stop(id)) continue;
        const std::string p = m_.tok.piece(id);
        if (p.empty()) continue;
        JsonCheck t = jc; bool good = true;
        for (char ch : p) if (!t.feed(ch)) { good = false; break; }
        if (good) ok.push_back(c[i]);
    }
    if (ok.empty()) return -1;
    if (o.temp <= 0.f) return ok[0].second;
    float mx = ok[0].first, sum = 0;
    for (auto& p : ok) { p.first = std::exp((p.first - mx) / o.temp); sum += p.first; }
    float cum = 0; size_t cut = ok.size();
    for (size_t i = 0; i < ok.size(); i++) { ok[i].first /= sum; cum += ok[i].first; if (cum >= o.top_p) { cut = i + 1; break; } }
    ok.resize(cut);
    float r = std::uniform_real_distribution<float>(0.f, cum)(rng_), acc = 0;
    for (auto& p : ok) { acc += p.first; if (r <= acc) return p.second; }
    return ok.back().second;
}

static const char kImgMark = '\x01';   // never produced by the tokenizer's text

std::vector<int> Chat::tokenize(const std::vector<Msg>& msgs_in, std::string& err) {
    bool any = false;
    for (auto& m : msgs_in) any |= !m.images.empty();
    if (!any) return m_.tok.encode(render(msgs_in), true);
    if (!vis_) { err = "image input is not available"; return {}; }
    if (!vis_->loaded()) {
        std::wstring mp = find_mmproj(utf8_to_w(m_.path_utf8), m_.hp.n_embd);
        if (mp.empty()) { err = "This model can't see images. Use a vision model such as Gemma 3, with its mmproj file in the same folder."; return {}; }
        if (!vis_->load(mp, m_.hp.n_embd, err)) return {};
    }
    // Images go in front of their message's text as markers; the rendered prompt is split at the markers.
    std::vector<Msg> msgs = msgs_in;
    std::vector<const std::string*> imgs;
    for (size_t i = 0; i < msgs.size(); i++) {
        std::string pre;
        for (auto& im : msgs_in[i].images) { pre += kImgMark; imgs.push_back(&im); }
        msgs[i].content = pre + msgs[i].content;
    }
    std::string text = render(msgs);
    std::vector<int> out;
    size_t a = 0; int k = 0; bool first = true;
    for (;;) {
        size_t b = text.find(kImgMark, a);
        std::vector<int> seg = m_.tok.encode(text.substr(a, b == std::string::npos ? std::string::npos : b - a), first);
        first = false;
        out.insert(out.end(), seg.begin(), seg.end());
        if (b == std::string::npos) break;
        const std::string& im = *imgs[k++];
        size_t h = std::hash<std::string>()(im);
        auto it = img_rows_.find(h);
        if (it == img_rows_.end()) {
            std::vector<float> e;
            if (!vis_->encode(im, e, err)) return {};
            image_ms += vis_->last_ms;
            int base = (int)(m_.ext_embd.size() / m_.hp.n_embd);
            m_.ext_embd.insert(m_.ext_embd.end(), e.begin(), e.end());
            it = img_rows_.emplace(h, base).first;
        }
        std::vector<int> pre = m_.tok.encode(vis_->before, false), post = m_.tok.encode(vis_->after, false);
        out.insert(out.end(), pre.begin(), pre.end());
        for (int r = 0; r < vis_->tokens(); r++) out.push_back(-1 - (it->second + r));
        out.insert(out.end(), post.begin(), post.end());
        a = b + 1;
    }
    return out;
}

GenStats Chat::generate(const std::vector<Msg>& msgs_in, const GenOpts& o, const std::function<bool(const std::string&)>& on_text) {
    GenStats st;
    std::vector<Msg> msgs = msgs_in;
    int budget = std::max(64, std::min(o.max_tokens, m_.ctx / 2));
    image_ms = 0;
    std::vector<int> toks = tokenize(msgs, st.error);
    if (toks.empty()) { if (st.error.empty()) st.error = "empty prompt"; return st; }
    // Drop the oldest turns until prompt + reply fits the context window.
    while ((int)toks.size() + budget > m_.ctx) {
        auto it = std::find_if(msgs.begin(), msgs.end(), [](const Msg& m) { return m.role != "system"; });
        if (it == msgs.end() || msgs.end() - it <= 1) break;
        msgs.erase(it);
        toks = tokenize(msgs, st.error);
        if (toks.empty()) return st;
    }
    if ((int)toks.size() + 8 > m_.ctx) { st.error = "prompt is longer than the context window"; return st; }
    budget = std::min(budget, m_.ctx - (int)toks.size());
    st.prompt_tokens = (int)toks.size();

    // Reuse the KV cache for the shared prefix: follow-up turns only process the new text.
    size_t common = 0;
    while (common < cache_.size() && common < toks.size() && cache_[common] == toks[common]) common++;
    if (common == toks.size()) common--;
    m_.rewind((int)common);
    common = std::min(common, (size_t)m_.n_past);   // recurrent models restart from zero when rewound
    cache_.resize(common);
    st.reused = (int)common;

    double t0 = now_sec();
    const float* logits = m_.eval(toks.data() + common, (int)(toks.size() - common));
    if (!logits) { st.error = "eval failed"; return st; }
    cache_ = toks;
    double t1 = now_sec();
    st.prefill_tps = (toks.size() - common) / std::max(1e-6, t1 - t0);

    std::string pending;
    {   // Templates that pre-open a reasoning block: show it to the client so it can fold the thoughts.
        std::string p = render(msgs);
        if (p.size() >= 8 && p.compare(p.size() - 8, 8, "<think>\n") == 0) on_text("<think>\n");
    }
    // The repetition penalty covers only what the model writes. Including the prompt penalized the user's own words,
    // so a reply about "sky" and "scattering" drifted to the no-space tokens ("theskyappearing").
    std::vector<int> recent;
    // Emits one sampled token; false = stop (end token, budget, client gone).
    bool done = false;
    hs_ = {};
    std::string pc;
    // JSON-constrained stretches (tool calls): after a trigger (json_after, several separated by '|', e.g.
    // "<json>|```json") each token must keep the JSON valid until the value is complete.
    JsonCheck jc; bool json_on = false; std::string tail;
    auto track = [&](const std::string& t) {
        if (o.json_after.empty()) return;
        if (json_on) {   // advance the checker; the value is complete: free text again
            for (char ch : t) if (!jc.feed(ch)) { json_on = false; break; }
            if (jc.done()) json_on = false;
            return;
        }
        tail += t; if (tail.size() > 64) tail.erase(0, tail.size() - 64);
        size_t e = tail.find_last_not_of(" \t\r\n");
        bool hit = false;
        for (size_t b = 0; e != std::string::npos && b <= o.json_after.size() && !hit;) {
            size_t z = o.json_after.find('|', b); if (z == std::string::npos) z = o.json_after.size();
            const std::string tr = o.json_after.substr(b, z - b);
            hit = !tr.empty() && e + 1 >= tr.size() && tail.compare(e + 1 - tr.size(), tr.size(), tr) == 0;
            b = z + 1;
        }
        if (hit) { json_on = true; jc = JsonCheck(); tail.clear(); }   // what follows the tag is only whitespace
    };
    auto pick = [&](const float* lg) { int id = json_on ? sample_json(lg, o, recent, jc) : -1; return id >= 0 ? id : sample(lg, o, recent); };
    auto emit = [&](int id) -> bool {
        if (st.gen_tokens >= budget) { done = true; return false; }   // the budget is spent: this token isn't shown (as without drafting)
        if (token_text(id, pc)) { pending += pc; done = true; return false; }
        pending += pc;
        track(pc);
        size_t ok = pending.size();
        for (size_t back = 1; back <= 4 && back <= pending.size(); back++) {
            unsigned char c = pending[pending.size() - back];
            if ((c & 0xC0) == 0x80) continue;
            int need = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
            if ((int)back < need) ok = pending.size() - back;
            break;
        }
        st.gen_tokens++;
        recent.push_back(id); if (recent.size() > 64) recent.erase(recent.begin());
        if (ok > 0) {
            if (!on_text(pending.substr(0, ok))) { pending.clear(); done = true; return false; }
            pending.erase(0, ok);
        }
        return true;
    };
    // Drafting works in JSON stretches too: each checked position is sampled under the constraint, as plain steps are.
    const bool use_draft = draft_ && draft_->loaded() && m_.hp.n_expert == 0, use_lookup = lookup && !m_.recurrent() && m_.hp.n_expert == 0;   // both may run: each step takes the better guess
    if (use_draft || use_lookup) {
        const int V = m_.hp.n_vocab, DV = use_draft ? draft_->hp.n_vocab : V;
        int next = pick(logits);   // emitted, not yet in the main model's cache
        if (emit(next)) {
            // Guessing pays when the tokens a check gains outweigh what drafting and the bigger check cost, so all of it is
            // measured as it goes: a plain step, one drafted token (none for lookup), what each extra token adds to a check,
            // and how often guesses are right (per source: the draft model, or lookup by match length). Each step takes the
            // guess length with the best expected tokens per second, or a plain step when none beats it; plain stretches
            // keep probing now and then, so a change from prose to code (or back) is noticed.
            double tp = 0, td = 0, beta = 0.1;               // plain step (s), drafted token (s), check cost per extra token (x tp)
            double an[5] = {1, 1, 1, 1, 1}, ad[5] = {2, 2, 2, 2, 2};   // accepted / drafted, decaying: [0] draft model, [N] lookup match length N
            int probe = 0, every = 12;                        // plain steps since the last check; probe spacing (doubles while checks lose)
            const int Kmax = std::max(1, std::min(7, draft_k + 2));
            auto rate = [&](double a, int K, double tdk) {   // expected tokens per second of a check of K guesses
                const double tok = a > 0.999 ? K + 1 : (1 - std::pow(a, K + 1)) / (1 - a);
                return tok / (K * tdk + tp * (1 + beta * K));
            };
            auto best_k = [&](double a, int kmax, double tdk, double& br) {   // 0: a plain step is faster
                int best = 0; br = 1.02 / std::max(tp, 1e-6);
                for (int k = 1; k <= kmax; k++) { const double r = rate(a, k, tdk); if (r > br) { br = r; best = k; } }
                return best;
            };
            while (!done) {
                if (m_.n_past + Kmax + 2 >= m_.ctx) break;
                const double ts = now_sec();
                std::vector<int> hist = cache_; hist.push_back(next);
                std::vector<int> d;
                int src = 0, kl = 0, km = 0;
                double rl = 0, rm = 0;
                if (tp > 0 && use_lookup) {
                    const int N = lookup_draft(hist, Kmax, d);
                    for (size_t k = 0; k < d.size(); k++)   // never guess image slots or stop tokens
                        if (d[k] < 0 || d[k] >= V || m_.tok.is_stop(d[k])) { d.resize(k); break; }
                    if (!d.empty()) { src = N; kl = best_k(an[N] / ad[N], (int)d.size(), 0, rl); }
                }
                if (tp > 0 && use_draft) {
                    if (td == 0) td = 0.25 * tp;   // a first guess until it is measured
                    km = best_k(an[0] / ad[0], Kmax, td, rm);
                }
                if (!kl && !km && probe >= every) { if (!d.empty()) kl = 1; else if (use_draft) km = 1; }   // probe now and then
                if (kl && (!km || rl >= rm)) d.resize(kl);   // the lookup guess (free)
                else {
                    d.clear(); src = 0;
                    const int K = km;
                    if (K > 0) {
                        // bring the draft up to the same history (dcache_ mirrors hist from doff_ on), then let it guess
                        // K tokens greedily. Far behind (a long new prompt) or out of room, it restarts on the last 512
                        // tokens: guessing the next few words needs little context, and reading it all would cost more.
                        if (doff_ >= hist.size()) { doff_ = 0; dcache_.clear(); }   // a shorter conversation than the last one
                        size_t c = 0;
                        while (c < dcache_.size() && doff_ + c < hist.size() && dcache_[c] == hist[doff_ + c]) c++;
                        if (doff_ + c >= hist.size()) c = hist.size() - doff_ - 1;
                        if (hist.size() - (doff_ + c) > 512 || hist.size() - doff_ + Kmax + 2 >= (size_t)draft_->ctx) {
                            doff_ = hist.size() > 512 ? hist.size() - 512 : 0; c = 0;
                        }
                        draft_->rewind((int)c); dcache_.assign(hist.begin() + doff_, hist.begin() + doff_ + c);
                        const size_t catchup = hist.size() - doff_ - c;
                        const float* dl = draft_->eval(hist.data() + doff_ + c, (int)catchup);
                        if (!dl) break;
                        dcache_.assign(hist.begin() + doff_, hist.end());
                        const double tg = now_sec();   // a long catch-up (the draft reading the prompt) is a one-off, not a per-token cost
                        for (int k = 0; k < K; k++) {
                            int t = (int)(std::max_element(dl, dl + DV) - dl);
                            if (t >= V || m_.tok.is_stop(t)) break;
                            d.push_back(t);
                            if (k + 1 < K) { dl = draft_->eval(&t, 1); if (!dl) break; dcache_.push_back(t); }
                        }
                        td = 0.7 * td + 0.3 * (now_sec() - (catchup <= 8 ? ts : tg)) / K;
                    }
                }
                const double tb = now_sec();
                if (d.empty()) {   // a plain step (and the measure of one)
                    cache_.push_back(next);
                    const float* lg = m_.eval(&next, 1);
                    if (!lg) { done = true; break; }
                    int id = pick(lg);
                    if (!emit(id)) break;
                    next = id;
                    const double t = now_sec() - tb;
                    tp = tp ? 0.8 * tp + 0.2 * t : t;
                    probe++;
                    continue;
                }
                probe = 0;
                // the main model scores [next, d...] in one pass
                std::vector<int> batch(1, next); batch.insert(batch.end(), d.begin(), d.end());
                const int base = m_.n_past;
                const float* L = m_.eval_all(batch.data(), (int)batch.size());
                static double sd = 0, sv = 0; static int steps = 0;
                sd += tb - ts; sv += now_sec() - tb;
                if (getenv("ITHACA_SPECPROF") && ++steps % 20 == 0)
                    fprintf(stderr, "[spec] draft %.1f ms, check %.1f ms per step; plain %.1f ms, drafted token %.1f ms, extra token +%.0f%%, accepted %.0f%% (model) %.0f/%.0f/%.0f%% (lookup 2/3/4)\n",
                            sd / steps * 1e3, sv / steps * 1e3, tp * 1e3, td * 1e3, beta * 100, 100 * an[0] / ad[0], 100 * an[2] / ad[2], 100 * an[3] / ad[3], 100 * an[4] / ad[4]);
                if (!L) { done = true; break; }
                st.drafted += (int)d.size();
                size_t j = 0; int s = -1;
                for (; j <= d.size(); j++) {
                    s = pick(L + j * (size_t)V);
                    if (j < d.size() && s == d[j]) { st.accepted++; if (!emit(s)) break; continue; }
                    break;
                }
                // keep [next, accepted drafts] in the cache; s becomes the new pending token
                m_.rewind(base + 1 + (int)std::min(j, d.size()));
                cache_ = hist; cache_.insert(cache_.end(), d.begin(), d.begin() + std::min(j, d.size()));
                if (done) break;
                if (!emit(s)) break;
                next = s;
                an[src] = 0.85 * an[src] + (double)std::min(j, d.size());
                ad[src] = 0.85 * ad[src] + (double)d.size();
                const double tv = now_sec() - tb;   // the first check of each size also records its CUDA graph: not a fair sample
                every = (1.0 + std::min(j, d.size())) / std::max(1e-6, now_sec() - ts) < 1.0 / tp ? std::min(256, every * 2) : 12;
                if (tv < 3 * tp * (1 + d.size())) beta = 0.8 * beta + 0.2 * std::min(1.0, std::max(0.0, (tv / tp - 1) / d.size()));
            }
            if (!done && !(m_.n_past >= m_.ctx)) {   // context nearly full: finish without drafting
                cache_.push_back(next);
                logits = m_.eval(&next, 1);
                for (; logits && !done && m_.n_past < m_.ctx;) {
                    int id = pick(logits);
                    if (!emit(id)) break;
                    cache_.push_back(id);
                    logits = m_.eval(&id, 1);
                }
            }
        }
    }
    else for (int i = 0; i < budget; i++) {
        const int id = pick(logits);
        if (token_text(id, pc)) { pending += pc; break; }
        pending += pc;
        track(pc);
        // Emit only complete UTF-8 sequences.
        size_t ok = pending.size();
        for (size_t back = 1; back <= 4 && back <= pending.size(); back++) {
            unsigned char c = pending[pending.size() - back];
            if ((c & 0xC0) == 0x80) continue;
            int need = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
            if ((int)back < need) ok = pending.size() - back;
            break;
        }
        st.gen_tokens++;
        cache_.push_back(id);
        recent.push_back(id); if (recent.size() > 64) recent.erase(recent.begin());
        if (ok > 0) {
            if (!on_text(pending.substr(0, ok))) { pending.clear(); break; }
            pending.erase(0, ok);
        }
        if (m_.n_past >= m_.ctx) break;
        logits = m_.eval(&id, 1);
        if (!logits) break;
    }
    if (!pending.empty()) on_text(pending);
    if (m_.n_past < (int)cache_.size()) cache_.resize(m_.n_past);  // last sampled token wasn't evaluated
    st.gen_tps = st.gen_tokens / std::max(1e-6, now_sec() - t1);
    return st;
}

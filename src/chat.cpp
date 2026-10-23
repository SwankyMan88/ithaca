#include "chat.h"
#include <algorithm>
#include <cmath>

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
    // ChatML default (Qwen 2/2.5/3 and most others)
    auto blk = [&](const std::string& r, const std::string& c) { out += "<|im_start|>" + r + "\n" + c + "<|im_end|>\n"; };
    if (!sys.empty()) blk("system", sys);
    each(blk);
    return out + "<|im_start|>assistant\n";
}

int Chat::sample(const float* logits, const GenOpts& o, const std::vector<int>& recent) {
    int n = m_.hp.n_vocab;
    std::vector<std::pair<float, int>> c(n);
    for (int i = 0; i < n; i++) c[i] = {logits[i], i};
    if (o.rep_pen != 1.f)
        for (int id : recent) if (id >= 0 && id < n) {
            float& v = c[id].first;
            v = v > 0 ? v / o.rep_pen : v * o.rep_pen;
        }
    if (o.temp <= 0.f) return std::max_element(c.begin(), c.end())->second;
    int k = std::min(n, std::max(1, o.top_k));
    std::partial_sort(c.begin(), c.begin() + k, c.end(), [](auto& a, auto& b) { return a.first > b.first; });
    c.resize(k);
    float mx = c[0].first, sum = 0;
    for (auto& p : c) { p.first = std::exp((p.first - mx) / o.temp); sum += p.first; }
    float cum = 0; size_t cut = c.size();
    for (size_t i = 0; i < c.size(); i++) { c[i].first /= sum; cum += c[i].first; if (cum >= o.top_p) { cut = i + 1; break; } }
    c.resize(cut);
    float r = std::uniform_real_distribution<float>(0.f, cum)(rng_), acc = 0;
    for (auto& p : c) { acc += p.first; if (r <= acc) return p.second; }
    return c.back().second;
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
    auto emit = [&](int id) -> bool {
        if (m_.tok.is_stop(id) || st.gen_tokens >= budget) { done = true; return false; }
        pending += m_.tok.piece(id);
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
    if (draft_ && draft_->loaded()) {
        const int V = m_.hp.n_vocab, DV = draft_->hp.n_vocab;
        int next = sample(logits, o, recent);   // emitted, not yet in the main model's cache
        if (emit(next)) {
            // Self-tuning: K follows the recent acceptance rate, and drafting pauses whenever it measures
            // slower than plain decoding (prose often is; code and repetitive text usually aren't).
            double acc = 0.6, spec_tps = 0, plain_tps = 0;
            int plain_left = 0;
            while (!done) {
                if (plain_left > 0) {   // plain decoding stretch, also measures the baseline speed
                    double tp = now_sec();
                    cache_.push_back(next);
                    const float* lg = m_.eval(&next, 1);
                    if (!lg) { done = true; break; }
                    int id = sample(lg, o, recent);
                    if (!emit(id)) break;
                    next = id;
                    double r = 1.0 / std::max(1e-4, now_sec() - tp);
                    plain_tps = plain_tps ? 0.8 * plain_tps + 0.2 * r : r;
                    plain_left--;
                    if (m_.n_past + 10 >= m_.ctx) break;
                    continue;
                }
                double ts = now_sec();
                std::vector<int> hist = cache_; hist.push_back(next);
                const int K = std::max(2, std::min(draft_k + 2, (int)std::lround(1 + 6 * acc)));
                if (m_.n_past + K + 2 >= m_.ctx || draft_->n_past + K + 2 >= draft_->ctx) break;
                // bring the draft up to the same history, then let it guess K tokens greedily
                size_t c = 0;
                while (c < dcache_.size() && c < hist.size() && dcache_[c] == hist[c]) c++;
                if (c == hist.size()) c--;
                draft_->rewind((int)c); dcache_.assign(hist.begin(), hist.begin() + c);
                double ta = now_sec();
                const float* dl = draft_->eval(hist.data() + c, (int)(hist.size() - c));
                if (!dl) break;
                dcache_ = hist;
                std::vector<int> d;
                for (int k = 0; k < K; k++) {
                    int t = (int)(std::max_element(dl, dl + DV) - dl);
                    if (t >= V || m_.tok.is_stop(t)) break;
                    d.push_back(t);
                    if (k + 1 < K) { dl = draft_->eval(&t, 1); if (!dl) break; dcache_.push_back(t); }
                }
                // the main model scores [next, d...] in one pass
                std::vector<int> batch(1, next); batch.insert(batch.end(), d.begin(), d.end());
                const int base = m_.n_past;
                double tb = now_sec();
                const float* L = m_.eval_all(batch.data(), (int)batch.size());
                static double sd = 0, sv = 0; static int steps = 0;
                sd += tb - ta; sv += now_sec() - tb;
                if (getenv("ITHACA_SPECPROF") && ++steps % 20 == 0) fprintf(stderr, "[spec] draft %.1f ms, verify %.1f ms per step\n", sd / steps * 1e3, sv / steps * 1e3);
                if (!L) { done = true; break; }
                st.drafted += (int)d.size();
                size_t j = 0; int s = -1;
                for (; j <= d.size(); j++) {
                    s = sample(L + j * (size_t)V, o, recent);
                    if (j < d.size() && s == d[j]) { st.accepted++; if (!emit(s)) break; continue; }
                    break;
                }
                // keep [next, accepted drafts] in the cache; s becomes the new pending token
                m_.rewind(base + 1 + (int)std::min(j, d.size()));
                cache_ = hist; cache_.insert(cache_.end(), d.begin(), d.begin() + std::min(j, d.size()));
                if (done) break;
                if (!emit(s)) break;
                next = s;
                acc = 0.7 * acc + 0.3 * (d.empty() ? 0.0 : (double)std::min(j, d.size()) / d.size());
                double r = (1 + std::min(j, d.size())) / std::max(1e-4, now_sec() - ts);
                spec_tps = spec_tps ? 0.7 * spec_tps + 0.3 * r : r;
                if (plain_tps && spec_tps < plain_tps * 0.97) { plain_left = 24; spec_tps = 0; }   // retry drafting later
                else if (!plain_tps && st.gen_tokens > 12) plain_left = 8;                         // measure the baseline once
            }
            if (!done && !(m_.n_past >= m_.ctx)) {   // context nearly full: finish without drafting
                cache_.push_back(next);
                logits = m_.eval(&next, 1);
                for (; logits && !done && m_.n_past < m_.ctx;) {
                    int id = sample(logits, o, recent);
                    if (!emit(id)) break;
                    cache_.push_back(id);
                    logits = m_.eval(&id, 1);
                }
            }
        }
    }
    else for (int i = 0; i < budget; i++) {
        int id = sample(logits, o, recent);
        if (m_.tok.is_stop(id)) break;
        pending += m_.tok.piece(id);
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

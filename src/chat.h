// Chat session: prompt templates, KV-prefix reuse, sampling, streaming.
#pragma once
#include "model.h"
#include "vision.h"
#include <random>
#include <unordered_map>

struct Msg { std::string role, content; std::vector<std::string> images; };   // images: encoded files (PNG/JPEG...)

struct GenOpts {
    float temp = 0.7f, top_p = 0.9f, rep_pen = 1.1f;
    int top_k = 40, max_tokens = 1024;
    // After the model writes this text (e.g. "<tool_call>"), only tokens that keep a valid JSON value are allowed,
    // until that value is complete. Small models then can't break their tool calls.
    std::string json_after;
};

// Incremental JSON checker: feed() returns false as soon as the text can no longer become valid JSON.
struct JsonCheck {
    std::string stack;   // open containers: '{' or '['
    int mode = 0;        // see feed()
    bool key = false; int hex = 0, wsrun = 0; std::string lit;   // wsrun: whitespace in a row outside strings
    bool done() const { return mode == 13; }
    bool feed(char c);
};

struct GenStats { int prompt_tokens = 0, reused = 0, gen_tokens = 0, drafted = 0, accepted = 0; double prefill_tps = 0, gen_tps = 0; std::string error; };

class Chat {
public:
    explicit Chat(Model& m, Vision* v = nullptr) : m_(m), vis_(v), rng_(std::random_device{}()) {}
    std::string render(const std::vector<Msg>& msgs) const;
    // on_text returns false to stop.
    GenStats generate(const std::vector<Msg>& msgs, const GenOpts& o, const std::function<bool(const std::string&)>& on_text);
    void reset() { cache_.clear(); img_rows_.clear(); m_.ext_embd.clear(); }
    // Prompt tokens with images expanded into embedding rows (negative ids). Empty + err on failure.
    std::vector<int> tokenize(const std::vector<Msg>& msgs, std::string& err);
    double image_ms = 0;
    // Speculative decoding: a small model with the same tokenizer drafts tokens, the main model checks them
    // in one batch. Every emitted token is still sampled from the main model, so output quality is unchanged.
    void set_draft(Model* d) { draft_ = d; dcache_.clear(); }
    Model* draft() const { return draft_; }
    int draft_k = 5;
private:
    int sample(const float* logits, const GenOpts& o, const std::vector<int>& recent);
    // sample() limited to tokens whose text keeps jc valid (-1: none of the likely tokens does)
    int sample_json(const float* logits, const GenOpts& o, const std::vector<int>& recent, const JsonCheck& jc);
    // Text of a generated token; true = stop. gpt-oss (harmony format): the analysis channel becomes a <think> block,
    // the final channel the answer; the channel headers themselves are not shown.
    bool token_text(int id, std::string& out);
    bool harmony() const { return m_.chat_template.find("<|channel|>") != std::string::npos; }
    struct { bool header = false, think = false; std::string chan; } hs_;
    Model& m_;
    Vision* vis_;
    Model* draft_ = nullptr;
    std::vector<int> dcache_;   // tokens in the draft model's KV cache
    std::unordered_map<size_t, int> img_rows_;   // image hash -> first row in m_.ext_embd
    std::vector<int> cache_;  // tokens currently in the KV cache
    std::mt19937 rng_;
};

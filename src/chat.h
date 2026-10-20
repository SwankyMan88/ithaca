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
};

struct GenStats { int prompt_tokens = 0, reused = 0, gen_tokens = 0; double prefill_tps = 0, gen_tps = 0; std::string error; };

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
private:
    int sample(const float* logits, const GenOpts& o, const std::vector<int>& recent);
    Model& m_;
    Vision* vis_;
    std::unordered_map<size_t, int> img_rows_;   // image hash -> first row in m_.ext_embd
    std::vector<int> cache_;  // tokens currently in the KV cache
    std::mt19937 rng_;
};

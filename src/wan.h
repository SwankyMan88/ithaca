// Video engine for the Wan family (Wan 2.2 TI2V-5B text-to-video; layout read from the files).
#pragma once
#include "gx.h"
#include <functional>

struct VideoJob {
    std::string prompt, negative;
    int width = 832, height = 480, frames = 49, fps = 24, steps = 20;
    float cfg = 5.0f, shift = 8.0f;
    uint64_t seed = 0;
};

class WanEngine {
public:
    // dit: diffusion model file; te: umT5 text encoder; vae: Wan VAE. Empty te/vae = auto-find next to dit.
    bool load(const std::wstring& dit, const std::wstring& te, const std::wstring& vae, std::string& err);
    void unload();
    bool loaded() const { return loaded_; }
    std::wstring path;
    bool generate(const VideoJob& job, std::vector<uint8_t>& rgb_frames, int& outW, int& outH, int& outF, std::string& err,
                  const std::function<bool(const char*, int, int)>& progress);

    // pieces (public for the self-test)
    bool tokenize(const std::string& text, std::vector<int>& ids, std::string& err);
    bool encode_text(const std::string& text, std::vector<float>& ctx, std::string& err);   // [512][4096]
    void dit(const std::vector<float>& x, int F, int H, int W, float timestep, const GT& ctx_emb, std::vector<float>& out);
    GT text_embed(const std::vector<float>& ctx);                                            // [512][dim] on GPU
    void dit_begin(int64_t vram_budget, bool q8 = false) { gx().weights_begin(&dit_, vram_budget, q8); gx().set_half(true); }
    int64_t dit_bytes() const;
    bool vae_decode(const std::vector<float>& z, int F, int H, int W, std::vector<float>& rgb, int& oF, int& oH, int& oW);
    int dim = 0, heads = 0, layers = 0, zc = 48, ffn = 0;
private:
    SafeTensors dit_, te_, vae_;
    bool loaded_ = false;
    // sentencepiece
    std::vector<std::string> pieces_;
    std::vector<float> scores_;
    std::unordered_map<std::string, int> piece_id_;
    size_t max_piece_ = 0;
    bool sp_loaded_ = false;
};

WanEngine& wan_engine();
bool write_mp4(const std::wstring& path, const std::vector<uint8_t>& rgb, int w, int h, int frames, int fps, std::string& err);

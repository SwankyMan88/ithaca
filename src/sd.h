// Stable Diffusion family image engine (SD 1.x, SD 2.x, SDXL and their fine-tunes).
// The network layout is read from the checkpoint's tensor names, so any single-file checkpoint of
// these families works, with optional separate VAE and text-encoder files.
#pragma once
#include "gx.h"
#include <functional>

struct ImageJob {
    std::string prompt, negative;
    int width = 1024, height = 1024, steps = 25;
    float cfg = 6.0f;
    uint64_t seed = 0;
    std::string sampler = "dpmpp_2m";   // euler, euler_a, dpmpp_2m
    std::string scheduler = "karras";   // normal, karras
    int clip_skip = 0;                  // 0 = model default
    // Editing: start from a picture (RGB8, width x height) instead of noise. strength = how much of it is redrawn
    // (0 keeps it, 1 ignores it). With a mask (width x height, 255 = repaint) only the masked part changes.
    std::vector<uint8_t> init_rgb, mask;
    float strength = 0.6f;
};

struct SdInfo {
    std::string arch;                   // "sd1", "sd2", "sdxl"
    bool vpred = false;
    int ctx_dim = 0, model_ch = 0, adm = 0, default_res = 512;
};

class SdEngine {
public:
    // ckpt: single-file checkpoint. vae / te: optional override files ("" = use the checkpoint's).
    bool load(const std::wstring& ckpt, const std::wstring& vae, std::string& err);
    void unload();
    bool loaded() const { return loaded_; }
    const SdInfo& info() const { return info_; }
    std::wstring path;
    // Called after each step with a cheap RGB preview of the current prediction (latent resolution).
    std::function<void(const std::vector<uint8_t>& rgb, int w, int h)> on_preview;
    // Runs the whole pipeline; writes RGB8 pixels. progress(phase, step, total)
    bool generate(const ImageJob& job, std::vector<uint8_t>& rgb, std::string& err,
                  const std::function<bool(const char*, int, int)>& progress);

    // pieces (public for the reference tests)
    bool encode_text(const std::string& text, std::vector<float>& ctx, int& ntok, std::vector<float>& pooled, std::string& err);
    std::vector<float> adm_vector(const std::vector<float>& pooled, int w, int h);
    void unet(const GT& x, int H, int W, float timestep, const GT& ctx, int ntok, const GT& y, GT& out);
    bool vae_decode(const std::vector<float>& z, int h, int w, std::vector<float>& img);  // z: VAE-space latent [4][h][w]
    bool vae_encode(const std::vector<float>& img, int H, int W, std::vector<float>& z);  // img [3][H][W] in -1..1 -> latent mean [4][H/8][W/8]
    const SafeTensors& tensors() const { return st_; }
    bool debug_clipl(const std::string& text, int run, std::vector<float>& out, std::vector<int>& ids);
    void unet_begin(int64_t budget);
    void unet_end();
private:
    SafeTensors st_;
    bool loaded_ = false;
    SdInfo info_;
    std::string unet_pfx_, vae_pfx_;
    float vae_scale_ = 0.18215f;
    std::vector<float> sigmas_all_;
};

SdEngine& sd_engine();
bool write_png(const std::wstring& path, const uint8_t* rgb, int w, int h);

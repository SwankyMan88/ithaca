// Image input for chat models: a vision encoder (mmproj GGUF) that turns a picture into embeddings
// the language model reads like tokens. Runs on the GPU runtime (fp16 tensor cores + flash attention).
#pragma once
#include "common.h"
#include "gguf.h"
#include "gx.h"
#include <deque>

class Vision {
public:
    bool load(const std::wstring& mmproj, int llm_embd, std::string& err);
    void unload();
    bool loaded() const { return ok_; }
    // Encoded image file (PNG/JPEG/...) -> tokens() rows of out_dim floats.
    bool encode(const std::string& file_bytes, std::vector<float>& out, std::string& err);
    int tokens() const { return n_out_; }
    int out_dim() const { return proj_dim_; }
    std::string path_utf8, kind;
    // Text that brackets the image embeddings in the prompt, for this projector family.
    std::string before = "\n\n<start_of_image>", after = "<end_of_image>\n\n";
    double last_ms = 0;
private:
    bool ok_ = false;
    GGUF g_;
    SafeTensors st_;
    std::deque<std::vector<float>> owned_;
    int img_ = 0, patch_ = 0, dim_ = 0, ff_ = 0, layers_ = 0, heads_ = 0, proj_dim_ = 0, n_out_ = 0;
    float eps_ = 1e-6f, mean_[3] = {0.5f, 0.5f, 0.5f}, std_[3] = {0.5f, 0.5f, 0.5f};
    std::string fc1_, fc2_;   // which of the ffn tensors is the up projection (names vary by converter)
};

// Finds an mmproj file next to the model whose projection size matches it ("" if none).
std::wstring find_mmproj(const std::wstring& model_path, int llm_embd);

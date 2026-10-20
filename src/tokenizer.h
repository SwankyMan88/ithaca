#pragma once
#include "gguf.h"
#include <unordered_map>

class Tokenizer {
public:
    bool load(const GGUF& g, std::string& err);
    std::vector<int> encode(const std::string& text, bool add_bos) const;
    std::string piece(int id) const;   // raw bytes this token produces ("" for control tokens)
    bool is_stop(int id) const;
    int bos = -1, eos = -1;
    bool add_bos = false;
    int n_vocab() const { return (int)tokens_.size(); }
private:
    void encode_text(const std::string& s, std::vector<int>& out) const;
    void bpe_word(const std::string& w, std::vector<int>& out) const;
    void spm_text(const std::string& s, std::vector<int>& out) const;
    bool gpt2_ = true;
    int digits_ = 3;
    std::vector<std::string> tokens_;
    std::vector<float> scores_;
    std::vector<int> types_;
    std::unordered_map<std::string, int> vocab_;
    std::unordered_map<std::string, int> merges_;
    std::vector<std::vector<int>> specials_by_byte_;  // first byte -> special ids, longest first
    std::string byte_enc_[256];
    std::unordered_map<uint32_t, uint8_t> byte_dec_;
    int byte_tok_[256];
};

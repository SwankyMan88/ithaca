// Speech engine: F5-TTS (flow-matching DiT, voice cloning from a short reference clip) + Vocos vocoder.
#pragma once
#include "gx.h"
#include <functional>

struct TtsJob {
    std::string text;
    std::wstring voice_wav;   // reference clip (any WAV)
    std::string voice_text;   // exact words spoken in the clip
    float speed = 1.f, cfg = 2.f;
    int steps = 32;
    uint64_t seed = 0;
};

class TtsEngine {
public:
    bool load(const std::wstring& model, std::string& err);   // finds vocab.txt and the vocos folder next to it
    void unload();
    bool loaded() const { return loaded_; }
    std::wstring path;
    bool speak(const TtsJob& job, std::vector<float>& wave, std::string& err, const std::function<bool(const char*, int, int)>& progress);

    // pieces (public for the self-test)
    std::vector<int> text_ids(const std::string& text);
    void mel(const std::vector<float>& wave, std::vector<float>& out, int& frames);          // [100][frames], log mel
    void text_embed(const std::vector<int>& ids, int N, std::vector<float>& out);            // [N][512]
    void dit(const std::vector<float>& x, const std::vector<float>& cond, const std::vector<float>& temb, int N, float t, std::vector<float>& out);
    void vocos(const std::vector<float>& mel, int frames, std::vector<float>& wave);
    void dit_begin();
    void voc_begin();
    void end();
private:
    SafeTensors f5_, voc_;
    bool loaded_ = false;
    std::unordered_map<std::string, int> vocab_;
    int dim_ = 1024, heads_ = 16, depth_ = 0, text_dim_ = 512;
};

TtsEngine& tts_engine();
bool read_wav(const std::wstring& path, std::vector<float>& mono, int& sr, std::string& err);
bool write_wav(const std::wstring& path, const std::vector<float>& wave, int sr);
std::vector<float> resample(const std::vector<float>& x, int from, int to);

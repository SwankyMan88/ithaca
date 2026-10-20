// MP4 (H.264) writer using Windows Media Foundation, which ships with Windows.
#include "common.h"
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <mferror.h>

bool write_mp4(const std::wstring& path, const std::vector<uint8_t>& rgb, int w, int h, int frames, int fps, std::string& err) {
    HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    bool co = SUCCEEDED(hr);
    if (FAILED(MFStartup(MF_VERSION))) { err = "Media Foundation is not available"; if (co) CoUninitialize(); return false; }
    IMFSinkWriter* wr = nullptr;
    IMFMediaType *out = nullptr, *in = nullptr;
    DWORD stream = 0;
    bool ok = false;
    do {
        if (FAILED(MFCreateSinkWriterFromURL(path.c_str(), nullptr, nullptr, &wr))) { err = "cannot create video file"; break; }
        MFCreateMediaType(&out);
        out->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
        out->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_H264);
        out->SetUINT32(MF_MT_AVG_BITRATE, (UINT32)std::min<int64_t>(40000000, (int64_t)w * h * fps / 4));
        out->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive);
        MFSetAttributeSize(out, MF_MT_FRAME_SIZE, w, h);
        MFSetAttributeRatio(out, MF_MT_FRAME_RATE, fps, 1);
        MFSetAttributeRatio(out, MF_MT_PIXEL_ASPECT_RATIO, 1, 1);
        if (FAILED(wr->AddStream(out, &stream))) { err = "H.264 encoder not available"; break; }
        MFCreateMediaType(&in);
        in->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
        in->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_RGB32);
        in->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive);
        MFSetAttributeSize(in, MF_MT_FRAME_SIZE, w, h);
        MFSetAttributeRatio(in, MF_MT_FRAME_RATE, fps, 1);
        MFSetAttributeRatio(in, MF_MT_PIXEL_ASPECT_RATIO, 1, 1);
        if (FAILED(wr->SetInputMediaType(stream, in, nullptr))) { err = "video input format rejected"; break; }
        if (FAILED(wr->BeginWriting())) { err = "cannot start writing video"; break; }
        const LONGLONG dur = 10000000LL / fps;
        bool fail = false;
        for (int f = 0; f < frames && !fail; f++) {
            IMFMediaBuffer* buf = nullptr;
            const DWORD bytes = (DWORD)w * h * 4;
            if (FAILED(MFCreateMemoryBuffer(bytes, &buf))) { fail = true; break; }
            BYTE* dst = nullptr;
            buf->Lock(&dst, nullptr, nullptr);
            // RGB32 is bottom-up BGRX
            for (int y = 0; y < h; y++) {
                const uint8_t* s = &rgb[(((size_t)f * h) + (h - 1 - y)) * w * 3];
                BYTE* d = dst + (size_t)y * w * 4;
                for (int x = 0; x < w; x++) { d[x * 4] = s[x * 3 + 2]; d[x * 4 + 1] = s[x * 3 + 1]; d[x * 4 + 2] = s[x * 3]; d[x * 4 + 3] = 255; }
            }
            buf->Unlock();
            buf->SetCurrentLength(bytes);
            IMFSample* smp = nullptr;
            MFCreateSample(&smp);
            smp->AddBuffer(buf);
            smp->SetSampleTime(dur * f);
            smp->SetSampleDuration(dur);
            if (FAILED(wr->WriteSample(stream, smp))) fail = true;
            smp->Release(); buf->Release();
        }
        if (fail) { err = "video encoding failed"; break; }
        ok = SUCCEEDED(wr->Finalize());
        if (!ok) err = "could not finish the video file";
    } while (false);
    if (in) in->Release();
    if (out) out->Release();
    if (wr) wr->Release();
    MFShutdown();
    if (co) CoUninitialize();
    return ok;
}

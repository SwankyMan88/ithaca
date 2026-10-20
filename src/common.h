// Ithaca - self-contained local LLM runner. Shared helpers.
#pragma once
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

inline std::wstring utf8_to_w(const std::string& s) {
    if (s.empty()) return L"";
    int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), nullptr, 0);
    std::wstring w(n, 0);
    MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), &w[0], n);
    return w;
}
inline std::string w_to_utf8(const std::wstring& w) {
    if (w.empty()) return "";
    int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), nullptr, 0, nullptr, nullptr);
    std::string s(n, 0);
    WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), &s[0], n, nullptr, nullptr);
    return s;
}

inline double now_sec() {
    static LARGE_INTEGER f = [] { LARGE_INTEGER x; QueryPerformanceFrequency(&x); return x; }();
    LARGE_INTEGER c; QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}

// Persistent worker pool. The calling thread participates as thread 0.
// Workers sleep on a condition variable so idle CPU goes to other apps (e.g. Blender).
class Pool {
public:
    using Fn = std::function<void(int64_t, int64_t, int)>;
    ~Pool() { stop(); }
    int size() const { return n_; }
    void start(int n) {
        stop();
        n_ = n < 1 ? 1 : n;
        quit_ = false;
        for (int i = 1; i < n_; i++) th_.emplace_back([this, i] { worker(i); });
    }
    void stop() {
        { std::lock_guard<std::mutex> l(m_); quit_ = true; }
        cv_.notify_all();
        for (auto& t : th_) t.join();
        th_.clear();
        n_ = 1;
    }
    void run(int64_t total, int64_t chunk, const Fn& f) {
        if (total <= 0) return;
        if (n_ == 1 || total <= chunk) { f(0, total, 0); return; }
        {
            std::lock_guard<std::mutex> l(m_);
            job_ = &f; total_ = total; chunk_ = chunk; next_ = 0; pending_ = n_ - 1; gen_++;
        }
        cv_.notify_all();
        work(0);
        int spins = 0;
        while (pending_.load(std::memory_order_acquire) > 0) {
            if (++spins < 4000) _mm_pause(); else std::this_thread::yield();
        }
    }
private:
    void worker(int id) {
        uint64_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> l(m_);
                cv_.wait(l, [&] { return quit_ || gen_ != seen; });
                if (quit_) return;
                seen = gen_;
            }
            work(id);
            pending_.fetch_sub(1, std::memory_order_release);
        }
    }
    void work(int id) {
        for (;;) {
            int64_t b = next_.fetch_add(chunk_);
            if (b >= total_) break;
            (*job_)(b, b + chunk_ < total_ ? b + chunk_ : total_, id);
        }
    }
    std::vector<std::thread> th_;
    std::mutex m_;
    std::condition_variable cv_;
    uint64_t gen_ = 0;
    bool quit_ = false;
    const Fn* job_ = nullptr;
    std::atomic<int64_t> next_{0};
    int64_t total_ = 0, chunk_ = 1;
    std::atomic<int> pending_{0};
    int n_ = 1;
};

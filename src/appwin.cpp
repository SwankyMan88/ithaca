// Ithaca's own app window: a native Win32 window with the page drawn inside it by WebView2, the web engine that is
// part of Windows 10/11. No browser opens, there is no address bar or tabs, and the window's profile (cookies) is
// kept in the store, separate from any browser. The runtime is loaded directly from its install folder (the same
// thing Microsoft's small loader library does), so Ithaca stays a single exe with no extra DLL to ship.
#include "common.h"
#include <objbase.h>
#include <string>
#include <atomic>

// ---- the few WebView2 COM interfaces Ithaca uses, in their published vtable order ----
// (in a named namespace on purpose: inside an anonymous one, GCC assumes nothing outside this file implements them
// and turns the calls into jumps to nowhere)
namespace wv2 {
struct EventRegistrationToken_ { int64_t value; };
struct ICoreWebView2;
struct ICoreWebView2Controller;
struct ICoreWebView2Environment;
struct ICoreWebView2CreateCoreWebView2ControllerCompletedHandler : IUnknown {
    virtual HRESULT STDMETHODCALLTYPE Invoke(HRESULT err, ICoreWebView2Controller* controller) = 0;
};
struct ICoreWebView2CreateCoreWebView2EnvironmentCompletedHandler : IUnknown {
    virtual HRESULT STDMETHODCALLTYPE Invoke(HRESULT err, ICoreWebView2Environment* env) = 0;
};
struct ICoreWebView2Environment : IUnknown {
    virtual HRESULT STDMETHODCALLTYPE CreateCoreWebView2Controller(HWND parent, ICoreWebView2CreateCoreWebView2ControllerCompletedHandler* handler) = 0;
};
struct ICoreWebView2Controller : IUnknown {
    virtual HRESULT STDMETHODCALLTYPE get_IsVisible(BOOL*) = 0;
    virtual HRESULT STDMETHODCALLTYPE put_IsVisible(BOOL) = 0;
    virtual HRESULT STDMETHODCALLTYPE get_Bounds(RECT*) = 0;
    virtual HRESULT STDMETHODCALLTYPE put_Bounds(RECT) = 0;
    virtual HRESULT STDMETHODCALLTYPE get_ZoomFactor(double*) = 0;
    virtual HRESULT STDMETHODCALLTYPE put_ZoomFactor(double) = 0;
    virtual HRESULT STDMETHODCALLTYPE add_ZoomFactorChanged(IUnknown*, EventRegistrationToken_*) = 0;
    virtual HRESULT STDMETHODCALLTYPE remove_ZoomFactorChanged(EventRegistrationToken_) = 0;
    virtual HRESULT STDMETHODCALLTYPE SetBoundsAndZoomFactor(RECT, double) = 0;
    virtual HRESULT STDMETHODCALLTYPE MoveFocus(int) = 0;
    virtual HRESULT STDMETHODCALLTYPE add_MoveFocusRequested(IUnknown*, EventRegistrationToken_*) = 0;
    virtual HRESULT STDMETHODCALLTYPE remove_MoveFocusRequested(EventRegistrationToken_) = 0;
    virtual HRESULT STDMETHODCALLTYPE add_GotFocus(IUnknown*, EventRegistrationToken_*) = 0;
    virtual HRESULT STDMETHODCALLTYPE remove_GotFocus(EventRegistrationToken_) = 0;
    virtual HRESULT STDMETHODCALLTYPE add_LostFocus(IUnknown*, EventRegistrationToken_*) = 0;
    virtual HRESULT STDMETHODCALLTYPE remove_LostFocus(EventRegistrationToken_) = 0;
    virtual HRESULT STDMETHODCALLTYPE add_AcceleratorKeyPressed(IUnknown*, EventRegistrationToken_*) = 0;
    virtual HRESULT STDMETHODCALLTYPE remove_AcceleratorKeyPressed(EventRegistrationToken_) = 0;
    virtual HRESULT STDMETHODCALLTYPE get_ParentWindow(HWND*) = 0;
    virtual HRESULT STDMETHODCALLTYPE put_ParentWindow(HWND) = 0;
    virtual HRESULT STDMETHODCALLTYPE NotifyParentWindowPositionChanged() = 0;
    virtual HRESULT STDMETHODCALLTYPE Close() = 0;
    virtual HRESULT STDMETHODCALLTYPE get_CoreWebView2(ICoreWebView2**) = 0;
};
struct ICoreWebView2 : IUnknown {
    virtual HRESULT STDMETHODCALLTYPE get_Settings(IUnknown**) = 0;
    virtual HRESULT STDMETHODCALLTYPE get_Source(LPWSTR*) = 0;
    virtual HRESULT STDMETHODCALLTYPE Navigate(LPCWSTR uri) = 0;
};
// ICoreWebView2EnvironmentOptions: the runtime reads these while it starts (it doesn't accept a null options object)
struct ICoreWebView2EnvironmentOptions_ : IUnknown {
    virtual HRESULT STDMETHODCALLTYPE get_AdditionalBrowserArguments(LPWSTR*) = 0;
    virtual HRESULT STDMETHODCALLTYPE put_AdditionalBrowserArguments(LPCWSTR) = 0;
    virtual HRESULT STDMETHODCALLTYPE get_Language(LPWSTR*) = 0;
    virtual HRESULT STDMETHODCALLTYPE put_Language(LPCWSTR) = 0;
    virtual HRESULT STDMETHODCALLTYPE get_TargetCompatibleBrowserVersion(LPWSTR*) = 0;
    virtual HRESULT STDMETHODCALLTYPE put_TargetCompatibleBrowserVersion(LPCWSTR) = 0;
    virtual HRESULT STDMETHODCALLTYPE get_AllowSingleSignOnUsingOSPrimaryAccount(BOOL*) = 0;
    virtual HRESULT STDMETHODCALLTYPE put_AllowSingleSignOnUsingOSPrimaryAccount(BOOL) = 0;
};
}  // namespace wv2
using namespace wv2;
namespace {
HWND g_win = nullptr;
ICoreWebView2Controller* g_ctl = nullptr;
std::wstring g_url;

// the handler interfaces' IIDs (WebView2.h)
const IID IID_EnvDone = {0x4e8a3389, 0xc9d8, 0x4bd2, {0xb6, 0xb5, 0x12, 0x4f, 0xee, 0x6c, 0xc1, 0x4d}};
const IID IID_CtlDone = {0x6c4819f3, 0xc9b7, 0x4260, {0x81, 0x27, 0xc9, 0xf5, 0xbd, 0xe7, 0xf6, 0x8c}};
template <class Base> struct Handler : Base {   // a tiny COM object: the runtime calls Invoke once
    ULONG refs = 1;
    const IID* iid = nullptr;
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID r, void** p) override {
        if (IsEqualIID(r, IID_IUnknown) || (iid && IsEqualIID(r, *iid))) { *p = this; AddRef(); return S_OK; }
        *p = nullptr; return E_NOINTERFACE;   // never pretend to be IMarshal, IAgileObject and the like
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return ++refs; }
    ULONG STDMETHODCALLTYPE Release() override { return --refs; }   // static lifetime: never deleted
};
struct OnController : Handler<ICoreWebView2CreateCoreWebView2ControllerCompletedHandler> {
    OnController() { iid = &IID_CtlDone; }
    HRESULT STDMETHODCALLTYPE Invoke(HRESULT err, ICoreWebView2Controller* c) override {
        if (FAILED(err) || !c) { MessageBoxW(g_win, L"The app window could not start its web view.", L"Ithaca", MB_ICONERROR); PostQuitMessage(1); return S_OK; }
        g_ctl = c; c->AddRef();
        RECT r; GetClientRect(g_win, &r); c->put_Bounds(r); c->put_IsVisible(TRUE);
        ICoreWebView2* wv = nullptr;
        if (SUCCEEDED(c->get_CoreWebView2(&wv)) && wv) { wv->Navigate(g_url.c_str()); wv->Release(); }
        return S_OK;
    }
} g_on_controller;
struct OnEnv : Handler<ICoreWebView2CreateCoreWebView2EnvironmentCompletedHandler> {
    OnEnv() { iid = &IID_EnvDone; }
    HRESULT STDMETHODCALLTYPE Invoke(HRESULT err, ICoreWebView2Environment* env) override {
        if (FAILED(err) || !env) { MessageBoxW(g_win, L"WebView2 (part of Windows) could not start. Install the \"Microsoft Edge WebView2 Runtime\" and try again.", L"Ithaca", MB_ICONERROR); PostQuitMessage(1); return S_OK; }
        env->CreateCoreWebView2Controller(g_win, &g_on_controller);
        return S_OK;
    }
} g_on_env;

const IID IID_EnvOptions = {0x2fde08a8, 0x1e9a, 0x4766, {0x8c, 0x05, 0x95, 0xa9, 0xce, 0xb9, 0xd1, 0xc5}};
LPWSTR co_str(const wchar_t* s) { size_t n = (wcslen(s) + 1) * sizeof(wchar_t); auto* p = (LPWSTR)CoTaskMemAlloc(n); if (p) memcpy(p, s, n); return p; }
struct Options : ICoreWebView2EnvironmentOptions_ {
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID r, void** p) override {
        if (IsEqualIID(r, IID_IUnknown) || IsEqualIID(r, IID_EnvOptions)) { *p = this; return S_OK; }
        *p = nullptr; return E_NOINTERFACE;
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return 2; }
    ULONG STDMETHODCALLTYPE Release() override { return 1; }
    HRESULT STDMETHODCALLTYPE get_AdditionalBrowserArguments(LPWSTR* v) override { *v = co_str(L""); return S_OK; }
    HRESULT STDMETHODCALLTYPE put_AdditionalBrowserArguments(LPCWSTR) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE get_Language(LPWSTR* v) override { *v = co_str(L""); return S_OK; }
    HRESULT STDMETHODCALLTYPE put_Language(LPCWSTR) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE get_TargetCompatibleBrowserVersion(LPWSTR* v) override { *v = co_str(L"86.0.616.0"); return S_OK; }
    HRESULT STDMETHODCALLTYPE put_TargetCompatibleBrowserVersion(LPCWSTR) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE get_AllowSingleSignOnUsingOSPrimaryAccount(BOOL* v) override { *v = FALSE; return S_OK; }
    HRESULT STDMETHODCALLTYPE put_AllowSingleSignOnUsingOSPrimaryAccount(BOOL) override { return S_OK; }
} g_options;

LRESULT CALLBACK wnd_proc(HWND h, UINT m, WPARAM w, LPARAM l) {
    switch (m) {
    case WM_SIZE: if (g_ctl) { RECT r; GetClientRect(h, &r); g_ctl->put_Bounds(r); } return 0;
    case WM_MOVE: if (g_ctl) g_ctl->NotifyParentWindowPositionChanged(); return 0;
    case WM_GETMINMAXINFO: { auto* mm = (MINMAXINFO*)l; mm->ptMinTrackSize = {380, 480}; return 0; }
    case WM_CLOSE: if (g_ctl) { g_ctl->Close(); g_ctl->Release(); g_ctl = nullptr; } DestroyWindow(h); return 0;
    case WM_DESTROY: PostQuitMessage(0); return 0;
    }
    return DefWindowProcW(h, m, w, l);
}
// The installed WebView2 runtime: its version from the registry (machine-wide, then per-user), then the folder.
std::wstring runtime_dll() {
    const wchar_t* key = L"SOFTWARE\\WOW6432Node\\Microsoft\\EdgeUpdate\\Clients\\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}";
    const wchar_t* key2 = L"Software\\Microsoft\\EdgeUpdate\\Clients\\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}";
    wchar_t ver[64]; DWORD n = sizeof ver;
    std::wstring base;
    if (RegGetValueW(HKEY_LOCAL_MACHINE, key, L"pv", RRF_RT_REG_SZ, nullptr, ver, &n) == ERROR_SUCCESS) {
        wchar_t pf[MAX_PATH]; if (GetEnvironmentVariableW(L"ProgramFiles(x86)", pf, MAX_PATH)) base = std::wstring(pf) + L"\\Microsoft\\EdgeWebView\\Application\\" + ver;
    } else if ((n = sizeof ver, RegGetValueW(HKEY_CURRENT_USER, key2, L"pv", RRF_RT_REG_SZ, nullptr, ver, &n)) == ERROR_SUCCESS) {
        wchar_t la[MAX_PATH]; if (GetEnvironmentVariableW(L"LOCALAPPDATA", la, MAX_PATH)) base = std::wstring(la) + L"\\Microsoft\\EdgeWebView\\Application\\" + ver;
    }
    return base.empty() ? L"" : base + L"\\EBWebView\\x64\\EmbeddedBrowserWebView.dll";
}
}  // namespace

// Opens the window on `url` and runs it until it is closed. Returns 0 when closed normally.
int run_app_window(const std::string& url, const std::wstring& title, const std::wstring& profile_dir) {
    g_url = utf8_to_w(url);
    {   // sharp text on high-DPI screens (Windows 10 1703+): per-monitor v2
        typedef BOOL(WINAPI * SetCtx)(HANDLE);
        if (auto f = (SetCtx)GetProcAddress(GetModuleHandleW(L"user32.dll"), "SetProcessDpiAwarenessContext")) f((HANDLE)-4);
    }
    CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    HINSTANCE hi = GetModuleHandleW(nullptr);
    WNDCLASSEXW wc{sizeof wc};
    wc.lpfnWndProc = wnd_proc; wc.hInstance = hi; wc.lpszClassName = L"IthacaApp";
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW); wc.hbrBackground = CreateSolidBrush(RGB(15, 15, 17));
    wc.hIcon = LoadIconW(hi, MAKEINTRESOURCEW(1)); if (!wc.hIcon) wc.hIcon = LoadIcon(nullptr, IDI_APPLICATION);
    wc.hIconSm = wc.hIcon;
    RegisterClassExW(&wc);
    const int sw = GetSystemMetrics(SM_CXSCREEN), sh = GetSystemMetrics(SM_CYSCREEN);
    const int w = std::min(1360, sw * 4 / 5), h = std::min(900, sh * 4 / 5);
    g_win = CreateWindowExW(0, wc.lpszClassName, title.c_str(), WS_OVERLAPPEDWINDOW, (sw - w) / 2, (sh - h) / 2, w, h, nullptr, nullptr, hi, nullptr);
    if (!g_win) return 1;
    // dark title bar on Windows 10 20H1+/11, like the page
    BOOL dark = TRUE; typedef HRESULT(WINAPI * DwmSet)(HWND, DWORD, LPCVOID, DWORD);
    if (HMODULE dwm = LoadLibraryW(L"dwmapi.dll")) if (auto f = (DwmSet)GetProcAddress(dwm, "DwmSetWindowAttribute")) f(g_win, 20, &dark, sizeof dark);
    ShowWindow(g_win, SW_SHOWNORMAL); UpdateWindow(g_win);

    const std::wstring dll = runtime_dll();
    HMODULE m = dll.empty() ? nullptr : LoadLibraryExW(dll.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    typedef HRESULT(STDMETHODCALLTYPE * CreateEnv)(bool, int, PCWSTR, IUnknown*, ICoreWebView2CreateCoreWebView2EnvironmentCompletedHandler*);
    auto create = m ? (CreateEnv)GetProcAddress(m, "CreateWebViewEnvironmentWithOptionsInternal") : nullptr;
    const bool dbg = getenv("ITHACA_WV_DEBUG") != nullptr;
    if (dbg) fprintf(stderr, "wv dll=%ls module=%p create=%p\n", dll.c_str(), (void*)m, (void*)create);
    HRESULT hr = create ? create(true, 0, profile_dir.c_str(), &g_options, &g_on_env) : E_FAIL;
    if (dbg) fprintf(stderr, "wv create hr=%08lx\n", (unsigned long)hr);
    if (FAILED(hr)) {
        MessageBoxW(g_win, L"This window needs the Microsoft Edge WebView2 Runtime, which comes with Windows 10 and 11. "
                           L"It wasn't found. Install it from Microsoft, or run Ithaca.exe to use your browser instead.", L"Ithaca", MB_ICONERROR);
        return 1;
    }
    MSG msg;
    while (GetMessageW(&msg, nullptr, 0, 0) > 0) { TranslateMessage(&msg); DispatchMessageW(&msg); }
    return 0;
}

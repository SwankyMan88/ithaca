// Per-device privacy: every browser or app window that opens Ithaca is a device with its own secret key (a cookie
// the page can't read). Its chats, coding chats, orchestration runs, memories, settings, voices and media live in
// store\devices\<id>\ encrypted with AES-256-GCM under that key, which the server never saves, so one device can't
// read another's data and the files on disk are unreadable without the key. "This PC" is the device the exe opens
// itself: its key is kept protected by Windows (DPAPI) so it survives a browser change. A device can add a PIN.
#pragma once
#include <string>
#include <vector>

// ---- storage: paths are relative to the current device's folder (or the shared store when there is none) ----
bool store_read(const std::wstring& rel, std::string& out);
bool store_write(const std::wstring& rel, const std::string& data);
bool store_delete(const std::wstring& rel);
bool store_exists(const std::wstring& rel);
// File names (not paths) in a folder of the current device that end with `ext` (e.g. L".json"), newest first.
std::vector<std::wstring> store_list(const std::wstring& reldir, const std::wstring& ext);
// Moves a finished file (an encoded image, video or clip written by an encoder) into the device's store.
bool store_put_file(const std::wstring& rel, const std::wstring& tmp_path);
// A plain temporary copy of a stored file, for engines that read from a path. Delete it when done.
std::wstring store_temp_copy(const std::wstring& rel);
// A temporary path for an encoder to write to (then store_put_file).
std::wstring store_temp_path(const std::wstring& name);
bool store_has_device();

// ---- requests (main.cpp) ----
void privacy_init(bool server_mode);   // loads or creates the "This PC" key; moves old plain data into it
std::string privacy_launch_token();     // one-time link token for the window or browser this exe opens
// Reads the device and session cookies of a request and makes that device current for this thread.
// Returns 0 = no device yet, 1 = ready, 2 = locked (PIN needed).
int privacy_begin(const std::string& raw_headers);
void privacy_end();
void privacy_api_request(bool on);   // /v1 requests use the shared store, never a device's
// The page itself: takes a launch token (?k=...) or makes a new device. Returns Set-Cookie header lines to send,
// and sets `redirect` when the page should reload without the token in its URL.
std::string privacy_page(const std::string& query, int state, bool& redirect);
bool handle_device(const std::string& method, const std::string& path, const std::string& body, int& code, std::string& out, std::string& set_cookie);

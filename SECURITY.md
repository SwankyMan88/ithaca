# Security Policy

## Supported versions

Only the latest release gets security fixes. Please update before reporting.

## Reporting a vulnerability

Please **don't open a public issue** for a security problem. Use GitHub's private reporting instead:
the repository's **Security** tab → **Report a vulnerability**.

Include what you can: the Ithaca version, what you did, what happened, and a proof of concept if you have one.
I'll reply as soon as I can, and I'll credit you in the release notes unless you'd rather I didn't.

## What Ithaca exposes

- **Local web server.** Ithaca serves its page and API on `127.0.0.1` by default. Anything that makes it listen on
  other addresses is your choice, and anyone who can reach that port can use the loaded model.
- **Coding section (`/api/code/*`).** These endpoints read and change files, so they only answer requests from this
  PC (loopback), and only inside the workspace folder you picked. Paths that leave the workspace are refused.
  Deletes go to the Recycle Bin and every change is in the undo journal (`store\code_undo`).
- **`run_command`.** Always asks you before it runs, even with auto-apply on. It runs with your user rights,
  in the workspace, with no input, a time limit, and the whole process tree is ended on timeout or Stop.
  Read the command before you approve it: a model can be wrong, or be steered by text in the files it reads.
- **Web pages.** Requests that carry another site's `Origin` are refused, so a page you visit can't drive Ithaca
  through your browser. (The `/v1` API accepts other origins only when you set `--api-key`.)
- **Each device's data.** Chats, coding chats, runs, memories, settings, voices and generated media are stored per
  device, encrypted with AES-256-GCM (Windows BCrypt) under a 256-bit key kept in an HttpOnly, SameSite=Strict
  cookie that the server never saves. "This PC" gets its key through a single-use launch link and keeps it under
  DPAPI. An optional PIN (PBKDF2-SHA256, 310,000 rounds) is needed on top of the key; wrong PINs are slowed down.
  Without an unlocked device, the app's endpoints answer 423 and return no data.
- **Tunnels and `--host`.** Requests from other devices are ordinary devices with their own data. But a tunnel or
  proxy that forwards to `127.0.0.1` makes its visitors look like this PC to the coding endpoints, which only check
  the address. Keep anything that forwards to Ithaca behind its own password.
- **Model files.** GGUF, safetensors and ComfyUI checkpoints are parsed as data, never run as code. PyTorch pickle
  files (`.ckpt`, `.pth`, `.bin`) go through a small built-in reader that only rebuilds tensors and never runs
  Python. Downloads come from the Hugging Face URLs you choose.

Issues in these areas — a way to read another device's data, to reach `/api/code/*` from another machine or a web
page, to escape the workspace, to run a command without approval, or to crash Ithaca with a crafted model file —
are exactly what I want to hear about.

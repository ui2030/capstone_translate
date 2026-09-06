# Cocktail

> **The shortest path to seeing your computer in your native language.**
> Recognition and translation both run locally — screen content and translations never leave your machine.
> Two things do need the outside world: a **one-time model download** on first run,
> and a **separate Tesseract OCR install**.

---

## Vision

Started as an overlay-translation tool, but the long-term goal is
**making your entire OS UI appear in your native language naturally**.
Start menu, settings, notifications, video subtitles, game NPC dialogue —
anywhere. See [VISION_AND_ROADMAP.md](VISION_AND_ROADMAP.md).

The Korean README is at [README.md](README.md).

---

## Current Status (Phase 0~5 — beta-ready)

- ✅ Region-box OCR + native-language overlay
- ✅ Tesseract (default) + PaddleOCRv5 (optional)
- ✅ Translation: opus-mt (en→ko fast-path) + m2m100 (multilingual fallback)
- ✅ Unicode-script multilingual auto-routing
- ✅ Self-capture feedback prevention (`SetWindowDisplayAffinity`)
- ✅ System tray + global hotkeys (Ctrl+Shift+Q/A/H/P)
- ✅ OverlayWindow (full-screen, mouse-through, multi-monitor)
- ✅ UIA adapter for active window text (Phase 2/3)
- ✅ DPAPI-encrypted persistent cache (Phase 4)
- ✅ Sensitive-area auto-pause (Phase 5)
- ✅ Auto-launch on Windows startup (Phase 6)
- ✅ Deterministic auto routing (SL-1) — OCR fixed to `eng+kor`, line language decided by character script only (no guessing)
- ✅ Display gate (DG-1) — low-confidence translations are hidden instead of drawn; hidden-line count shown in the status bar
- ✅ Physical-pixel coordinate unification (DP-1) — no overlay drift on high-DPI / scaled displays
- ✅ Monitor hot-plug — overlay refits on screen add / remove / geometry change

---

## Install / Run

First-time setup:

1. Python 3.10 / 3.11 + `pip install -r requirements.txt`
2. Install the Tesseract OCR engine (not bundled): https://github.com/UB-Mannheim/tesseract/wiki
3. On first launch the translation models (opus-mt / m2m100) are downloaded from
   Hugging Face (~1–2GB, internet required). Later runs load from cache and work offline.

```bash
run_cocktail.bat       # double-click: picks a usable Python, diagnoses, then starts
                       # order: COCKTAIL_PYTHON -> conda "cd" env -> py -3.11 -> python
python diagnose_environment.py
debug_run.bat          # timestamped log to debug_log.txt when the app dies silently
```

See [ENVIRONMENT.md](ENVIRONMENT.md) for verified versions and environment variables.

---

## Build / Distribute

See [BUILD.md](BUILD.md). 4-step workflow:
```bash
pip install -r requirements-dev.txt
python download_tessdata.py
pyinstaller build.spec
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer.iss
```

GitHub Actions release workflow at [.github/workflows/release.yml](.github/workflows/release.yml).

---

## Tech Stack

- UI: PySide6 (LGPL)
- OCR: Tesseract (Apache 2.0) / PaddleOCRv5 (optional, Apache 2.0)
- Translation: Helsinki-NLP/opus-mt-tc-big-en-ko (CC-BY-4.0, attribution required) + facebook/m2m100_418M (MIT)
- Language detection: `unicodedata` script fast-path only (Latin/unknown = en). Pick the source combo for other languages
- Persistent cache: Windows DPAPI (per-user encrypted)
- Tray + global hotkeys: PySide6 `QSystemTrayIcon` + `keyboard` (MIT, optional)

See [TECH_STACK.md](TECH_STACK.md).

---

## License & Commercial Use

- This project: **MIT** (see [LICENSE](LICENSE))
- Third-party components: each retains its own license — see [LICENSE-3RDPARTY.md](LICENSE-3RDPARTY.md)
- All bundled components are **commercial-friendly** (Apache 2.0 / MIT / LGPL).
- ❌ NLLB (CC-BY-NC) / PyQt5 (GPL) are **excluded** to keep redistribution rights clean.

---

## Security & Ethics (5 Principles)

See [VISION_AND_ROADMAP.md §5](VISION_AND_ROADMAP.md):
1. Local inference — no screen content or translation is sent to any server
   (the only network access is the one-time model download from Hugging Face)
2. No plain-text translation persistence — DPAPI-encrypted, hashed keys
3. Sensitive-area auto-pause — password / payment / banking windows
4. Open core — core logic is open source
5. Instant toggle — global hotkey to disable anytime

---

## Documentation Index

- ⭐ [STORY.md](STORY.md) — full project journey (best entry point for collaborators)
- [VISION_AND_ROADMAP.md](VISION_AND_ROADMAP.md) — long-term vision + roadmap
- [ENVIRONMENT.md](ENVIRONMENT.md) — verified Python/package versions
- [BUILD.md](BUILD.md) — build & distribute
- [CODE_SIGNING.md](CODE_SIGNING.md) — Windows code signing guide
- [ICONS.md](ICONS.md) — icon assets and integration
- [TECH_STACK.md](TECH_STACK.md) — technology overview
- [UPDATE_LOG.md](UPDATE_LOG.md) — change log
- [OCR_TEST_PLAN.md](OCR_TEST_PLAN.md) — pre-beta OCR/translation test plan
- [errors.md](errors.md) — defect log + 5-second checklist (read before code edits)

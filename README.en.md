# Cocktail

> **The shortest path to seeing your computer in your native language.**
> 100% local processing. No data leaves your machine.

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
- ✅ System tray + global hotkeys (Ctrl+Shift+T/A/H/P)
- ✅ OverlayWindow (full-screen, mouse-through, multi-monitor)
- ✅ UIA adapter for active window text (Phase 2/3)
- ✅ DPAPI-encrypted persistent cache (Phase 4)
- ✅ Sensitive-area auto-pause (Phase 5)
- ✅ Auto-launch on Windows startup (Phase 6)

---

## Run

```bash
run_cocktail.bat       # double-click to diagnose then start
python diagnose_environment.py
python "Cocktail완성본.py"
```

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
- Translation: Helsinki-NLP/opus-mt-tc-big-en-ko (Apache 2.0) + facebook/m2m100_418M (MIT)
- Language detection: `unicodedata` script fast-path + langdetect fallback
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
1. 100% local processing — no cloud transmission
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

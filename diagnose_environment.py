import importlib.util
import os
import shutil
import subprocess
import sys


REQUIRED_MODULES = [
    "numpy",
    "cv2",
    "torch",
    "pytesseract",
    "PIL",
    "transformers",
    "PySide6",
]

OPTIONAL_MODULES = [
    "paddleocr",
    "paddle",
]


def has_module(name):
    return importlib.util.find_spec(name) is not None


def find_tesseract():
    candidates = [
        os.environ.get("TESSERACT_CMD"),
        shutil.which("tesseract"),
        r"C:\tesseract\tesseract.exe",
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def main():
    ok = True
    print("== Cocktail environment diagnostics ==")
    print(f"Python: {sys.version.split()[0]}")

    missing = [name for name in REQUIRED_MODULES if not has_module(name)]
    if missing:
        ok = False
        print("[ERROR] Missing required Python modules:", ", ".join(missing))
        print("        Run: pip install -r requirements.txt")
    else:
        print("[OK] Required Python modules installed")

    optional_missing = [name for name in OPTIONAL_MODULES if not has_module(name)]
    if optional_missing:
        print("[INFO] Optional PaddleOCR backend not installed:", ", ".join(optional_missing))
        print("       Tesseract backend can still run.")
    else:
        print("[OK] Optional PaddleOCR backend installed")

    # 앱과 같은 규칙으로 ./tessdata 를 먼저 본다. 안 맞추면 앱은 kor을 보는데
    # 진단만 "eng, osd"라고 보고해서 사용자를 헷갈리게 한다.
    local_tessdata = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tessdata")
    if os.path.isdir(local_tessdata):
        os.environ["TESSDATA_PREFIX"] = local_tessdata

    tess = find_tesseract()
    if not tess:
        ok = False
        print("[ERROR] Tesseract executable not found")
        print("        Install Tesseract or set TESSERACT_CMD")
    else:
        print(f"[OK] Tesseract: {tess}")
        try:
            out = subprocess.check_output([tess, "--list-langs"], stderr=subprocess.STDOUT, text=True)
            langs = [line.strip() for line in out.splitlines() if line.strip() and "List of available" not in line]
            print("[OK] Tesseract languages:", ", ".join(langs) if langs else "(none)")
            if "eng" not in langs:
                ok = False
                print("[ERROR] eng.traineddata is missing")
        except Exception as exc:
            ok = False
            print(f"[ERROR] Could not query Tesseract languages: {exc}")

    try:
        import torch
        print("[OK] Translation device:", "CUDA" if torch.cuda.is_available() else "CPU")
    except Exception as exc:
        ok = False
        print(f"[ERROR] Torch check failed: {exc}")

    print("== Result:", "OK" if ok else "ACTION NEEDED", "==")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())


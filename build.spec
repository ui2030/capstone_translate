# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for Cocktail Overlay Translator.

Default release builds keep PaddleOCR/PaddlePaddle out of the app bundle.
Set COCKTAIL_BUNDLE_PADDLE=1 only for a separate full OCR build.
"""
import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all


datas = []
binaries = []
hiddenimports = [
    "scipy",
    "scipy._lib.array_api_compat",
]

project_docs = [
    "README.md",
    "README.en.md",
    "LICENSE",
    "LICENSE-3RDPARTY.md",
    "BUILD.md",
    "CODE_SIGNING.md",
    "UPDATE_LOG.md",
    "업데이트.md",
    "errors.md",
    "TECH_STACK.md",
    "VISION_AND_ROADMAP.md",
    "COMMERCIAL_SOTA_REVIEW.md",
]
for doc in project_docs:
    if Path(doc).exists():
        datas.append((doc, "."))

tessdata_dir = Path("tessdata")
if tessdata_dir.exists():
    for traineddata in tessdata_dir.glob("*.traineddata"):
        datas.append((str(traineddata), "tessdata"))

# PK-1: Tesseract 엔진 자체를 동봉한다. 별도 설치를 요구하면 "준비 5분"(합격기준 6)이 깨진다.
# 라이선스 Apache 2.0 — doc/LICENSE, doc/AUTHORS 를 같이 넣어야 조건을 지킨다.
# datas 로 넣는 이유: PyInstaller 가 이 DLL 들을 분석/재배치하면 tesseract.exe 옆에서
# 흩어진다. 통째로 `tesseract/` 한 폴더에 두는 것이 이 엔진이 뜨는 유일한 배치다.
# COCKTAIL_TESSERACT_DIR 로 소스 경로를 바꿀 수 있다(다른 PC 빌드).
tess_src = Path(os.environ.get("COCKTAIL_TESSERACT_DIR", r"C:\tesseract"))
if (tess_src / "tesseract.exe").exists():
    datas.append((str(tess_src / "tesseract.exe"), "tesseract"))
    for dll in tess_src.glob("*.dll"):
        datas.append((str(dll), "tesseract"))
    for lic in ("LICENSE", "AUTHORS"):
        if (tess_src / "doc" / lic).exists():
            datas.append((str(tess_src / "doc" / lic), "tesseract/doc"))
    print(f"[BUILD INFO] Tesseract bundled from {tess_src}")
else:
    print(f"[BUILD WARN] Tesseract not found at {tess_src} — "
          "설치본이 별도 Tesseract 설치를 요구하게 된다 (합격기준 6 위반).")

# PK-2: 기본 언어쌍(en→ko) 모델 동봉 → 첫 실행이 오프라인으로 끝난다.
# safetensors 만 넣는다: 같은 가중치의 pytorch_model.bin(399MB)/tf_model.h5(806MB) 사본과
# 벤치마크 zip 은 transformers 가 쓰지 않는다. 그대로 넣으면 1.6GB → 402MB.
# COCKTAIL_BUNDLE_MODEL=0 이면 건너뛴다(모델 없는 경량 빌드 = 첫 실행에 다운로드).
BUNDLE_MODEL = os.environ.get("COCKTAIL_BUNDLE_MODEL", "1").strip() != "0"
BUNDLED_MODELS = ["Helsinki-NLP/opus-mt-tc-big-en-ko"]
_MODEL_SKIP = {"pytorch_model.bin", "tf_model.h5", "benchmark_translations.zip",
               ".gitattributes"}
if BUNDLE_MODEL:
    from huggingface_hub import snapshot_download
    for repo in BUNDLED_MODELS:
        leaf = repo.split("/")[-1]
        try:
            snap = Path(snapshot_download(repo, local_files_only=True))
        except Exception as e:
            print(f"[BUILD WARN] {repo} 로컬 캐시 없음 — 동봉 생략 ({type(e).__name__}). "
                  "먼저 앱을 한 번 실행해 캐시를 채워라.")
            continue
        n = 0
        for f in snap.iterdir():
            if f.is_file() and f.name not in _MODEL_SKIP:
                datas.append((str(f.resolve()), f"models/{leaf}"))
                n += 1
        print(f"[BUILD INFO] model bundled: {leaf} ({n} files)")

# IC-2: exe/인스톨러 아이콘. `python cocktail.py --make-icon` 으로 코드에서 굽는다.
icon_file = Path("assets/icon.ico")
app_icon = str(icon_file) if icon_file.exists() else None
if app_icon is None:
    print("[BUILD WARN] assets/icon.ico 없음 — `python cocktail.py --make-icon` 을 먼저 실행하라.")
else:
    datas.append((app_icon, "assets"))

BUNDLE_PADDLE = os.environ.get("COCKTAIL_BUNDLE_PADDLE", "0").strip() == "1"


def _is_dev_artifact(entry):
    source = str(entry[0]).replace("/", "\\").lower()
    dest = str(entry[1]).replace("/", "\\").lower() if len(entry) > 1 else ""
    if source.endswith(".lib"):
        return True
    if "\\include\\" in source or dest.startswith("torch\\include"):
        return True
    return False


def _collect_runtime(pkg):
    d, b, h = collect_all(pkg)
    if pkg == "torch":
        d = [item for item in d if not _is_dev_artifact(item)]
        b = [item for item in b if not _is_dev_artifact(item)]
    return d, b, h


packages = [
    "transformers",
    "tokenizers",
    "sentencepiece",
    "sacremoses",
    "PIL",
    "cv2",
    "torch",
]
if BUNDLE_PADDLE:
    packages += ["paddleocr", "paddle"]
else:
    print("[BUILD INFO] PaddleOCR/PaddlePaddle excluded. Set COCKTAIL_BUNDLE_PADDLE=1 for full OCR build.")

for pkg in packages:
    try:
        d, b, h = _collect_runtime(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception as e:
        print(f"[BUILD WARN] collect_all({pkg}) failed: {e}")

block_cipher = None

a = Analysis(
    # 엔트리포인트는 ASCII 파일명이다. 예전 `Cocktail완성본.py`는 비-ASCII라
    # 런처 배치파일이 glob으로 찾아야 했고 일부 CI/압축 도구에서 깨졌다.
    ["cocktail.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "PyQt5",
        "PyQt6",
        *([] if BUNDLE_PADDLE else ["paddle", "paddleocr", "modelscope"]),
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# PK-3: PyInstaller 는 `tesseract.exe` 의 의존 DLL 을 분석해 **`_internal` 루트에도** 복사한다.
# 그 중 `libcrypto-3-x64.dll` / `libssl-3-x64.dll` 이 파이썬 `_ssl.pyd` 가 요구하는 OpenSSL 을
# 덮어써서, 패키지 앱에서 `import transformers` 가
#   "DLL load failed while importing _ssl: 지정된 프로시저를 찾을 수 없습니다"
# 로 죽는다 = **배포본에서 번역이 통째로 안 된다**(2026-09-07 재빌드에서 실측, self-test 실패).
# tesseract 의 DLL 은 `tesseract/` 폴더 사본만 있으면 그 엔진이 뜬다(같은 폴더에서 로드).
# 그래서 **루트로 흘러든 tesseract 사본만** 걷어내고, 파이썬이 실제로 쓰는 OpenSSL 을 되돌린다.
_tess_root = str(tess_src).lower()
a.binaries = [e for e in a.binaries
              if not (str(e[1]).lower().startswith(_tess_root)
                      and "\\" not in str(e[0]) and "/" not in str(e[0]))]
_py_dll_dir = Path(sys.base_prefix) / "Library" / "bin"
for _name in ("libcrypto-3-x64.dll", "libssl-3-x64.dll"):
    _src = _py_dll_dir / _name
    if _src.exists() and not any(str(e[0]).lower() == _name for e in a.binaries):
        a.binaries.append((_name, str(_src), "BINARY"))
        print(f"[BUILD INFO] PK-3: restored python OpenSSL {_name}")

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Cocktail",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    # UPX는 끈다. PyInstaller + Qt/torch DLL 조합에서 백신 오탐(휴리스틱 "packed executable")의
    # 대표 원인이고, 서명 없는 exe와 겹치면 SmartScreen까지 3연타로 맞는다.
    # 압축률 대비 배포 사고 위험이 크다 — 용량은 torch CPU 휠 쪽에서 줄이는 게 맞다.
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=app_icon,   # IC-2
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,   # EXE와 동일 사유
    upx_exclude=[],
    name="Cocktail",
)

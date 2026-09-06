"""Cocktail 릴리즈 게이트 — **배포 위생**만 검사한다.

역할 분담:
  - 앱이 올바르게 **동작하는가** → `test_cocktail.py` (실제로 함수를 호출해 결과를 본다)
  - 배포물이 올바르게 **구성됐는가** → 이 파일 (파일 누락/버전/라이선스/워크플로 순서)

예전 이 파일에는 앱 소스를 문자열로 뒤져 "이 구현이 들어 있는가"를 확인하는 검사가
9개 있었다(SL-1/B-15/BG-3~7 등). 코드 지문 검사라 리팩터링만 해도 깨지면서 실제 버그는
못 잡았다 — 그 역할은 `test_cocktail.py`로 옮겼다.
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

REQUIRED_FILES = [
    "cocktail.py",
    "cocktail_platform.py",
    "cocktail_capture.py",
    "cocktail_ocr.py",
    "cocktail_translate.py",
    "cocktail_engine.py",
    "cocktail_ui.py",
    "test_cocktail.py",
    "build.spec",
    "installer.iss",
    "requirements.txt",
    "requirements-dev.txt",
    "README.md",
    "README.en.md",
    "LICENSE",
    "LICENSE-3RDPARTY.md",
    "BUILD.md",
    "CODE_SIGNING.md",
    "UPDATE_LOG.md",
    "TECH_STACK.md",
    "VISION_AND_ROADMAP.md",
    "COMMERCIAL_SOTA_REVIEW.md",
    "errors.md",
    "diagnose_environment.py",
    ".github/workflows/release.yml",
]

# 상용 배포에 들어가면 안 되는 것들. NLLB는 CC-BY-NC(비상용), PyQt5/6은 GPL.
FORBIDDEN_DEPENDENCY_MARKERS = ["PyQt5", "PyQt6", "facebook/nllb", "nllb-200"]

# 소스 트리에 남으면 안 되는 대용량/라이선스 위험 디렉터리.
FORBIDDEN_DIRS = ["nllb_ct2"]


def read(path):
    return (ROOT / path).read_text(encoding="utf-8")


def fail(msg, failures):
    failures.append(msg)
    print(f"[FAIL] {msg}")


def ok(msg):
    print(f"[OK] {msg}")


def check_required_files(failures):
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).exists()]
    if missing:
        fail("missing required files: " + ", ".join(missing), failures)
    else:
        ok("required files present")


def check_ascii_source_filenames(failures):
    """비-ASCII 파일명은 배치 런처/CI/압축 도구에서 깨진다 (구 `Cocktail완성본.py`)."""
    bad = [p.name for p in ROOT.glob("*.py") if not p.name.isascii()]
    if bad:
        fail("non-ASCII source filenames: " + ", ".join(bad), failures)
    else:
        ok("source filenames are ASCII")


def check_installer_version(failures):
    m = re.search(r"#define\s+AppVersion\s+\"([^\"]+)\"", read("installer.iss"))
    if not m:
        fail("installer.iss AppVersion not found", failures)
    elif not re.fullmatch(r"\d+\.\d+\.\d+", m.group(1)):
        fail(f"installer AppVersion is not semver: {m.group(1)}", failures)
    else:
        ok(f"installer version {m.group(1)}")


def check_build_spec(failures):
    spec = read("build.spec")
    before = len(failures)
    for doc in ["README.md", "README.en.md", "LICENSE", "LICENSE-3RDPARTY.md",
                "UPDATE_LOG.md", "업데이트.md", "errors.md", "COMMERCIAL_SOTA_REVIEW.md"]:
        if doc not in spec:
            fail(f"build.spec does not include {doc}", failures)
    if '"cocktail.py"' not in spec:
        fail("build.spec entrypoint must be cocktail.py", failures)
    if "console=False" not in spec:
        fail("build.spec must keep console=False for GUI release", failures)
    if "COCKTAIL_BUNDLE_PADDLE" not in spec:
        fail("build.spec must gate PaddleOCR bundling behind COCKTAIL_BUNDLE_PADDLE", failures)
    if 'source.endswith(".lib")' not in spec or "torch\\\\include" not in spec:
        fail("build.spec must filter PyTorch development artifacts", failures)
    if "tessdata" not in spec or "*.traineddata" not in spec:
        fail("build.spec must include downloaded tessdata", failures)
    if "upx=True" in spec:
        # UPX 압축은 백신 오탐의 대표 원인 — 서명 없는 exe와 겹치면 배포 사고가 난다.
        fail("build.spec must not use upx=True (antivirus false positives)", failures)
    if len(failures) == before:
        ok("build.spec release basics")


def check_release_workflow(failures):
    wf = read(".github/workflows/release.yml")
    before = len(failures)
    for token in ["Install Tesseract OCR", "python test_cocktail.py",
                  "python pre_release_check.py", "Sign EXE (optional)",
                  "Build installer", "Packaged self-test",
                  "Output/Cocktail-Setup-*.exe"]:
        if token not in wf:
            fail(f"release workflow missing: {token}", failures)
    sign_pos, zip_pos = wf.find("Sign EXE (optional)"), wf.find("Zip artifacts")
    if sign_pos < 0 or zip_pos < 0 or sign_pos > zip_pos:
        fail("release workflow must sign before zip", failures)
    if len(failures) == before:
        ok("release workflow ordering")


def check_dependency_policy(failures):
    req = read("requirements.txt")
    haystack = req + "\n" + read("LICENSE-3RDPARTY.md")
    before = len(failures)
    for marker in FORBIDDEN_DEPENDENCY_MARKERS:
        if marker in req:
            fail(f"forbidden runtime dependency in requirements.txt: {marker}", failures)
    for required in ["PySide6", "transformers", "torch", "pytesseract"]:
        if required not in req:
            fail(f"requirements.txt missing runtime dependency: {required}", failures)
    for licensed in ["PySide6", "Tesseract", "m2m100", "keyboard"]:
        if licensed not in haystack:
            fail(f"license docs missing dependency note: {licensed}", failures)
    if len(failures) == before:
        ok("dependency policy checked")


def check_model_license_attribution(failures):
    """opus-mt-tc-big-en-ko는 **CC-BY-4.0**이다 (모델 카드 실측, 2026-08-03).

    Apache 2.0으로 잘못 적혀 있던 것을 정정했다. CC-BY는 상용 OK지만 **출처 표시 의무**가
    있어, 배포물에 저작자 표기가 반드시 동봉돼야 한다. 라이선스 표기는 배포 사고 중
    되돌리기가 가장 어려운 축이라 게이트로 잡는다.
    """
    doc = read("LICENSE-3RDPARTY.md")
    before = len(failures)
    if "cc-by-4.0" not in doc.lower() and "CC BY 4.0" not in doc:
        fail("LICENSE-3RDPARTY.md must state opus-mt's CC-BY-4.0 license", failures)
    if "University of Helsinki" not in doc:
        fail("LICENSE-3RDPARTY.md must carry the opus-mt attribution notice", failures)
    for path in ["README.md", "README.en.md"]:
        text = read(path)
        if "opus-mt-tc-big-en-ko (Apache 2.0)" in text:
            fail(f"{path} still claims opus-mt is Apache 2.0", failures)
    if len(failures) == before:
        ok("model license attribution (CC-BY-4.0)")


def check_docs_cross_links(failures):
    readme = read("README.md")
    before = len(failures)
    for link in ["test_cocktail.py", "BUILD.md", "LICENSE-3RDPARTY.md",
                 "COMMERCIAL_SOTA_REVIEW.md", "errors.md"]:
        if link not in readme:
            fail(f"README.md missing link: {link}", failures)
    build = read("BUILD.md")
    for command in ["python test_cocktail.py",
                    "python -m PyInstaller --noconfirm build.spec", "installer.iss"]:
        if command not in build:
            fail(f"BUILD.md missing command/reference: {command}", failures)
    if len(failures) == before:
        ok("documentation links checked")


def check_no_generated_dirs(failures, allow_build_artifacts=False):
    pycache = ROOT / "__pycache__"
    if pycache.exists():
        shutil.rmtree(pycache, ignore_errors=True)
        print("[OK] cleaned __pycache__")
    for name in FORBIDDEN_DIRS:
        if (ROOT / name).exists():
            fail(f"forbidden directory present (license risk): {name}/", failures)
    generated = [] if allow_build_artifacts else [
        p.name for p in ROOT.iterdir() if p.name == "build"]
    if generated:
        fail("generated directories present: " + ", ".join(generated), failures)
    else:
        ok("no blocking generated dirs in project root")


CHECKS = [
    check_required_files,
    check_ascii_source_filenames,
    check_installer_version,
    check_build_spec,
    check_release_workflow,
    check_dependency_policy,
    check_model_license_attribution,
    check_docs_cross_links,
]


def main():
    parser = argparse.ArgumentParser(description="Cocktail release gate")
    parser.add_argument("--allow-build-artifacts", action="store_true",
                        help="Allow local build/ after a manual PyInstaller build.")
    args = parser.parse_args()

    failures = []
    for check in CHECKS:
        check(failures)
    check_no_generated_dirs(failures, allow_build_artifacts=args.allow_build_artifacts)

    if failures:
        print(f"== Result: FAIL ({len(failures)}) ==")
        return 1
    print("== Result: OK ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())

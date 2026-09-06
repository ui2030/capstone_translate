# Cocktail — Build & Distribute

## 사전 준비

1. Python 3.10+ 설치
2. Tesseract OCR 엔진 설치 (개발 시): https://github.com/UB-Mannheim/tesseract/wiki
3. Inno Setup 6 설치 (인스톨러 빌드용): `winget install -e --id JRSoftware.InnoSetup`
   (또는 https://jrsoftware.org/isdl.php — winget 설치본은 `%LOCALAPPDATA%\Programs\Inno Setup 6`에 들어간다)

## 빌드 절차

```bash
# 1. 의존성 설치
pip install -r requirements-dev.txt

# 2. 빠른 회귀/배포 전 검사
python test_cocktail.py
python pre_release_check.py

# 3. traineddata 다운로드 (인스톨러 컴포넌트용)
python download_tessdata.py
# → tessdata/eng.traineddata, kor.traineddata, jpn.traineddata, chi_sim.traineddata, fra.traineddata, deu.traineddata
# build.spec는 존재하는 tessdata/*.traineddata를 dist/Cocktail/tessdata에도 포함한다.

# 3-1. 아이콘 굽기 (IC-2) — 코드로 그린다. 외부 이미지 금지(ICONS.md §6)
python cocktail.py --make-icon
# → assets/icon.ico (16/32/48/64/128/256 멀티사이즈). build.spec / installer.iss 가 참조.

# 4. .exe 빌드 (PyInstaller)
python -m PyInstaller --noconfirm build.spec
# → dist/Cocktail/Cocktail.exe (+ 의존성 dlls)
# 동봉물(PK-1/PK-2): _internal/tesseract (엔진 exe+DLL 56개, Apache 2.0 LICENSE 포함) /
#   _internal/tessdata / _internal/models/opus-mt-tc-big-en-ko (safetensors 399MB) / _internal/assets
# 레버: COCKTAIL_TESSERACT_DIR(기본 C:	esseract) · COCKTAIL_BUNDLE_MODEL=0(모델 미동봉) ·
#   COCKTAIL_BUNDLE_PADDLE=1(PaddleOCR 포함 풀 빌드)
# 실측 크기(2026-09-05): dist 5,318MB — torch CUDA 3,851MB가 대부분이다.
#   CPU 전용 torch로 빌드하면 ~1.4GB로 줄지만 번역이 348ms → 1780ms(5.1배) 느려진다.

# 4-1. 빌드된 exe 자기검증 (GUI를 띄우지 않음)
dist\Cocktail\Cocktail.exe --self-test

# 5. 인스톨러 빌드 (Inno Setup)
#    Inno Setup Compiler에서 installer.iss 열고 Build → Compile
#    또는 CLI (winget으로 깔면 사용자 폴더에 들어간다 — 아래 경로가 이 PC의 실제 위치):
#    "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" installer.iss
#    (`winget install -e --id JRSoftware.InnoSetup` 으로 설치. 관리자 권한 불필요)
# → Output\Cocktail-Setup-1.0.0.exe
```

## 인스톨러 사용자 경험

- 컴포넌트 선택 화면: 영어(필수) + 한국어/일본어/중국어/프랑스어/독일어(선택)
- 사용자가 체크한 언어의 traineddata만 `{app}\tessdata\` 에 설치
- 앱 시작 시 자동으로 `TESSDATA_PREFIX` 를 그쪽으로 설정 (cocktail_ocr.py)
- **Tesseract 엔진은 동봉된다** (PK-1, 2026-09-05). 사용자가 따로 설치할 것이 없다.
  `_internal	essdata` 는 인스톨러에서 제외되고, 사용자가 고른 언어만 `{app}	essdata` 에 깔린다.

## 모델 다운로드

- **기본 언어쌍 en→ko 모델은 동봉된다** (PK-2, 2026-09-05) — `_internal/models/opus-mt-tc-big-en-ko`.
  첫 실행이 오프라인으로 끝난다. `_bundled_model_dir()` 이 repo id 대신 이 폴더를 쓴다.
- `opus-mt-zh-en`(300MB) · `m2m100_418M` 폴백(1.9GB)은 그 언어를 실제로 번역할 때 자동 다운로드
  (`~/.cache/huggingface`). 전부 동봉하면 인스톨러가 3GB를 넘는다.
- 동봉 모델을 늘리려면 `build.spec` 의 `BUNDLED_MODELS` 에 repo id를 추가한다
  (로컬 HF 캐시에 이미 받아져 있어야 한다).

## GitHub Actions 릴리즈

- `v1.0.0` 같은 태그를 push하면 `.github/workflows/release.yml`이 Windows 빌드를 수행합니다.
- 워크플로우는 Tesseract OCR 엔진을 설치한 뒤 `diagnose_environment.py`, `test_cocktail.py`, `pre_release_check.py`를 실행합니다.
- PyInstaller 빌드 후, 인증서 secret이 있으면 `Cocktail.exe`를 서명하고 그 다음 zip을 만듭니다.
- Inno Setup을 설치해 `Output/Cocktail-Setup-*.exe` 인스톨러도 함께 생성합니다.
- 릴리즈 산출물: `Cocktail-<version>-windows.zip`, `Cocktail-Setup-*.exe`.

## 선택 OCR 백엔드: PaddleOCRv5

앱의 OCR 콤보에서 `PaddleOCRv5`를 선택하면 Tesseract 대신 PP-OCRv5 파이프라인을 사용합니다.
기본 설치에는 포함하지 않습니다. 용량과 CUDA/CPU 설치 차이가 커서, 배포 채널별로 별도 패키지로 다루는 편이 안전합니다.

```bash
# CPU 예시. GPU/CUDA 환경은 PaddlePaddle 공식 설치 명령 확인.
pip install "paddleocr>=3.0.0" "paddlepaddle>=3.0.0"
```

- 첫 사용 시 PaddleOCR 모델을 다운로드합니다.
- source 언어가 `auto`이면 `COCKTAIL_PADDLE_LANG` 환경변수 또는 기본 `ch` 모델을 사용합니다.
- 한국어/일본어/프랑스어처럼 특정 OCR 언어가 중요하면 source 콤보를 해당 언어로 명시하는 편이 정확합니다.
- PaddleOCR 초기화/추론 실패 시 앱은 자동으로 Tesseract로 폴백합니다.
- Windows CPU 환경에서는 `paddle_static`/oneDNN 경로가 실패할 수 있어 앱 기본값은 `COCKTAIL_PADDLE_ENGINE=paddle`에 해당하는 동적 엔진입니다.

## 라이선스 체크리스트 (배포 전)

- [ ] PySide6 (LGPL): closed-source 상용 OK ✓
- [ ] Helsinki-NLP/opus-mt-* (CC-BY-4.0): 상용 OK, 단 LICENSE-3RDPARTY.md 저작자 표기 동봉 필수 ✓
- [ ] facebook/m2m100_418M (MIT): 상용 OK ✓
- [ ] Tesseract (Apache 2.0): 상용 OK ✓
- [ ] tessdata (Apache 2.0): 상용 OK ✓
- [ ] PaddleOCR/PaddlePaddle 선택 패키지: Apache 2.0 계열 라이선스와 CUDA 런타임 재배포 조건 확인
- [ ] keyboard (MIT, 글로벌 단축키 — Phase 1A): 상용 OK, optional 처리 ✓
- [ ] uiautomation (Apache 2.0, Phase 2 PoC `uia_explore.py` 전용): 본 앱 의존성 X ✓
- [ ] NLLB / PyQt5 / 비상용 모델: 사용 안 함 ✓ (project_commercial_decisions.md 참고)
- [x] **`nllb_ct2/` 폴더 삭제 완료 (2026-08-03)** — M-5에서 폐기된 NLLB(CC-BY-NC) 변환 모델 잔재(약 600MB). `.gitignore`에 재추가 차단 등록.
      코드는 참조하지 않고, `build.spec`은 `datas`에 명시한 것만 담으므로 `dist/`에 들어가지
      않으며 `installer.iss`도 `dist\Cocktail\*`만 담는다(= 산출물 오염 없음).
      그래도 소스 배포/저장소 공개 전에는 반드시 지울 것. `.claudeignore`에도 등록돼 있다.

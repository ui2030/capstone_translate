# Cocktail — 개발/실행 환경 (검증된 버전)

> 본 문서는 사용자 머신에서 **실제로 검증된** Python/패키지 버전을 기록합니다.
> 다른 환경/머신으로 옮길 때 이 버전을 기준선으로 사용하세요.

마지막 검증: 2026-08-02 (8차 · 배포 준비 패스에서 실측)

---

## 1. Python

| 항목 | 검증된 값 | 권장 |
|---|---|---|
| Python | **3.11.15** (conda-forge) | 3.10.x ~ 3.11.x |
| pip | 26.0.1 | 24.0+ |
| 실행 인터프리터 경로 | `C:\Users\ui2030\anaconda3\envs\cd\python.exe` | — |
| conda env | `cd` (Anaconda) | venv도 OK |

```bash
# Python/pip 버전 확인
"C:\Users\ui2030\anaconda3\envs\cd\python.exe" --version   # Python 3.11.15
"C:\Users\ui2030\anaconda3\envs\cd\python.exe" -m pip --version
```

> **시스템 Python 주의**: PATH에 잡히는 시스템 Python(3.13)에는 이 앱의 패키지가 **없습니다**.
> 그냥 `python Cocktail완성본.py`로 실행하면 실패합니다. `run_cocktail.bat`이 올바른
> 인터프리터를 찾아주므로 그것을 쓰세요 (§6).

> **3.12+ 주의**: PaddleOCR/PaddlePaddle, 일부 transformer 의존성이 3.12+에서 패키징 이슈 가능.
> Cocktail은 **3.10/3.11에서만 검증**되었습니다.

---

## 2. GPU/CUDA (선택)

| 항목 | 검증된 값 |
|---|---|
| Torch | **2.11.0+cu126** (CUDA 12.6 빌드) |
| GPU 모드 | 이 머신: **사용 가능** (`torch.cuda.is_available() == True`, fp16 자동) |

CPU 환경에서도 동작합니다. CPU 시 fp16 없이 자동 분기됩니다.

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# → 2.11.0+cu126 True
```

---

## 3. 핵심 패키지 (2026-08-02 `cd` env 실측)

```
PySide6         6.11.1
transformers    5.8.1
torch           2.11.0+cu126
tokenizers      0.22.2
sentencepiece   0.2.1
sacremoses      0.1.1
pytesseract     0.3.13
Pillow          12.1.1
opencv-python   4.13.0.92
numpy           1.26.0
keyboard        0.13.5
scipy           1.17.1
```

미설치(선택 백엔드): `paddleocr`, `paddlepaddle` — 없어도 Tesseract 백엔드로 정상 동작.

> **langdetect는 더 이상 쓰지 않습니다.** SL-1에서 언어 추측을 퇴역시키고
> `unicodedata` 스크립트 판정만 씁니다. requirements.txt/build.spec/진단 스크립트에서 제거됨.

### 한 번에 설치
```bash
pip install -r requirements.txt
```

---

## 4. 시스템 의존성 (Python 외)

| 항목 | 필요 | 이 머신의 값 |
|---|---|---|
| **Tesseract OCR 엔진** | 필수 (별도 설치) | `C:\tesseract\tesseract.exe` (v5.5.0) |
| traineddata | OCR 언어별 | 프로젝트 `./tessdata/` (eng/kor/jpn/chi_sim/chi_tra/fra/deu/rus/ara) |
| Inno Setup 6 (빌드만) | 인스톨러 빌드 시 | https://jrsoftware.org/isdl.php |

Tesseract Windows 인스톨러: https://github.com/UB-Mannheim/tesseract/wiki

앱이 자동 탐색하는 경로 (위에서부터 순서대로):
1. 환경변수 `TESSERACT_CMD`
2. `C:\tesseract\tesseract.exe`
3. `C:\Program Files\Tesseract-OCR\tesseract.exe`
4. `C:\Program Files (x86)\Tesseract-OCR\tesseract.exe`

traineddata는 **프로젝트 폴더의 `./tessdata/`가 있으면 그쪽을 우선** 사용합니다
(`TESSDATA_PREFIX`를 앱이 자동 설정). 없으면 Tesseract 설치 폴더의 tessdata를 씁니다.
추가 언어는 `python download_tessdata.py`로 받습니다.

---

## 5. 환경변수

| 변수 | 용도 | 예 |
|---|---|---|
| `COCKTAIL_PYTHON` | `run_cocktail.bat`이 쓸 Python을 직접 지정 (최우선) | `set COCKTAIL_PYTHON=C:\envs\cd\python.exe` |
| `TESSERACT_CMD` | Tesseract 실행 파일 경로 오버라이드 | `set TESSERACT_CMD=D:\Tesseract\tesseract.exe` |
| `COCKTAIL_OCR_LANGS_AUTO` | auto 모드 OCR 언어 (기본 `eng+kor`) | `set COCKTAIL_OCR_LANGS_AUTO=eng+kor+jpn` |
| `COCKTAIL_PADDLE_LANG` | PaddleOCR 언어 | `set COCKTAIL_PADDLE_LANG=korean` |
| `COCKTAIL_PADDLE_ENGINE` / `COCKTAIL_PADDLE_DEVICE` | PaddleOCR 엔진/디바이스 | — |
| `COCKTAIL_SHOW_SETTINGS_ON_START` | 시작 시 설정창 강제 표시 | `set COCKTAIL_SHOW_SETTINGS_ON_START=1` |

---

## 6. 실행과 진단

### 일반 실행 — `run_cocktail.bat` 더블클릭
배치 파일이 아래 순서로 **처음 동작하는 Python**을 골라 실행합니다:

1. `COCKTAIL_PYTHON` 환경변수 (지정돼 있으면 무조건 우선)
2. `%USERPROFILE%\anaconda3\envs\cd\python.exe` / `...\miniconda3\envs\cd\python.exe`
3. `py -3.11` (Windows Python 런처)
4. PATH의 `python`

2~4번 후보는 `import PySide6`가 되는지로 판별합니다. 패키지 없는 시스템 Python을
집어서 실패하던 문제(8차 수정)를 이 판별이 막습니다. 전부 실패하면 안내 후 `pause`.

> `run_cocktail.bat`은 **ASCII 전용**입니다. 배치 파일에 한글을 넣으면 콘솔 코드페이지
> (CP949) 대 UTF-8 충돌로 파일이 깨집니다. 한국어 안내는 이 문서가 담당합니다.

### 진단만
```bash
python diagnose_environment.py
```
진단 항목: 필수 Python 모듈 / Optional PaddleOCR / Tesseract 실행 파일 /
traineddata 목록(앱과 동일한 `TESSDATA_PREFIX` 규칙 적용) / Torch device.

### 문제 추적 — `debug_run.bat`
앱이 뜨다 말거나 조용히 죽을 때 씁니다. 로그 각 줄에 타임스탬프를 붙여
`debug_log.txt`로 남깁니다.

```bash
debug_run.bat        # 더블클릭
# → debug_log.txt 확인. 마지막 줄이 어디서 멈췄는지 알려준다.
```

`debug_run.bat`은 인터프리터 경로가 **하드코딩**된 진단 전용 파일입니다
(`C:\Users\ui2030\anaconda3\envs\cd\python.exe`). 다른 PC에서는 그 경로만 바꾸세요.

### 회귀 검사 (코드 수정 후)
```bash
python smoke_tests.py
python pre_release_check.py --allow-build-artifacts
python "Cocktail완성본.py" --self-test
```

---

## 7. 새 환경에서 처음 시작하는 사람을 위한 빠른 가이드

```bash
# 1. Python 3.10/3.11 설치
# 2. (선택) conda 환경 생성
conda create -n cocktail python=3.11 -y
conda activate cocktail

# 3. CUDA 사용자: torch 별도 설치 권장 (공식 명령 — CUDA 버전에 맞춰서)
pip install torch --index-url https://download.pytorch.org/whl/cu126

# 4. 나머지
pip install -r requirements.txt

# 5. Tesseract 엔진 설치
# https://github.com/UB-Mannheim/tesseract/wiki

# 6. 진단
python diagnose_environment.py

# 7. 실행 — 새 환경 이름이 'cd'가 아니면 인터프리터를 알려준다
set COCKTAIL_PYTHON=C:\path\to\python.exe
run_cocktail.bat
```

> **최초 1회 인터넷 필요**: 번역 모델(opus-mt / m2m100)을 Hugging Face에서 내려받습니다.
> 다운로드 후에는 캐시에서 로드하므로 오프라인으로 동작합니다.

---

## 8. 라이선스 호환 점검 (배포 전)

설치되는 모든 패키지가 상용 호환인지 [BUILD.md](BUILD.md) 라이선스 체크리스트 참고.
요약:
- ✅ PySide6 (LGPL), opus-mt (Apache 2.0), m2m100 (MIT), Tesseract (Apache 2.0), tessdata (Apache 2.0)
- ✅ keyboard (MIT), uiautomation (Apache 2.0)
- ✅ PaddleOCR/PaddlePaddle (Apache 2.0 계열)
- ❌ NLLB (CC-BY-NC), PyQt5 (GPL) — 절대 추가 금지
- ⚠️ 프로젝트 폴더의 `nllb_ct2/` (약 600MB)는 폐기된 NLLB(CC-BY-NC) 잔재입니다.
  코드는 참조하지 않고 `.claudeignore`·`build.spec`에서 제외되지만 **배포 전 삭제해야 합니다.**

---

## 9. 알려진 환경 이슈

| 이슈 | 회피 |
|---|---|
| 시스템 Python(3.13)에는 패키지가 없음 | `run_cocktail.bat` 사용 또는 `COCKTAIL_PYTHON` 지정 |
| 배치 파일에 한글 → CP949/UTF-8 충돌로 깨짐 | `.bat`은 ASCII만 |
| PaddleOCR 3.x + Windows CPU 기본 `paddle_static` 실패 | `engine="paddle"`, `enable_mkldnn=False` (코드 기본값) |
| chardet 7.x → requests 경고 | `chardet<6` 고정 |
| Python 3.12+ 일부 패키지 휠 미배포 | 3.10/3.11 권장 |
| transformers 5.x API 변경 | 현재 5.8.1에서 검증. 마이너 변경 시 `from_pretrained` 시그니처 확인 |
| 전체화면 독점 앱 위에는 오버레이가 못 올라감 | 테두리 없는 창 모드로 전환 (TM-1 한계) |
| 일부 게임/관리자 권한 앱은 캡처 거부 | 알려진 한계 |
| `keyboard` 글로벌 단축키 시 일부 환경 관리자 권한 필요 | optional 처리 — 미동작 시 로컬 단축키만 사용 |

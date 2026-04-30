# Cocktail — 개발/실행 환경 (검증된 버전)

> 본 문서는 사용자 머신에서 **실제로 검증된** Python/패키지 버전을 기록합니다.
> 다른 환경/머신으로 옮길 때 이 버전을 기준선으로 사용하세요.

마지막 검증: 2026-04-30

---

## 1. Python

| 항목 | 검증된 값 | 권장 |
|---|---|---|
| Python | **3.10.18** | 3.10.x ~ 3.11.x |
| pip | 26.0.1 | 24.0+ |
| conda env (선택) | `aigugbi` (Anaconda) | venv도 OK |

```bash
# Python/pip 버전 확인
python --version    # Python 3.10.18
pip --version       # pip 26.0.1
```

> **3.12+ 주의**: PaddleOCR/PaddlePaddle, 일부 transformer 의존성이 3.12에서 패키징 이슈 가능.
> Cocktail은 **3.10/3.11에서만 검증**되었습니다.

---

## 2. GPU/CUDA (선택)

| 항목 | 검증된 값 |
|---|---|
| Torch | **2.7.1+cu118** (CUDA 11.8 빌드) |
| CUDA Toolkit | 11.8 호환 |
| GPU 모드 | 사용자 머신: **사용 가능** (fp16 자동) |

CPU 환경에서도 동작합니다. CPU 시 `USE_FP16=False`로 자동 분기됩니다.

```bash
# Torch + CUDA 확인
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# → 2.7.1+cu118 True
```

---

## 3. 핵심 패키지 (사용자 환경 기준)

### 이미 설치됨 ✅
```
numpy==2.2.6
pillow==12.1.1
tokenizers==0.22.2
torch==2.7.1+cu118
transformers==5.3.0
```

### 누락 — 설치 필요 ❌
본 앱 실행에 필요한데 사용자 환경에 빠진 것:
```
PySide6>=6.6
pytesseract>=0.3.10
opencv-python>=4.8.0
langdetect>=1.0.9
sentencepiece>=0.2.0
sacremoses>=0.1.1
keyboard>=0.13.5
```

### 한 번에 설치
```bash
pip install -r requirements.txt
```

---

## 4. 시스템 의존성 (Python 외)

| 항목 | 필요 | 확인 |
|---|---|---|
| **Tesseract OCR 엔진** | 필수 | `tesseract --version` |
| 한국어 traineddata (kor.traineddata) | 한국어 OCR 시 | `tesseract --list-langs` |
| Inno Setup 6 (빌드만) | 인스톨러 빌드 시 | https://jrsoftware.org/isdl.php |

Tesseract Windows 인스톨러: https://github.com/UB-Mannheim/tesseract/wiki

기본 설치 경로 (`C:\Program Files\Tesseract-OCR\`)를 우리 코드가 자동 탐색합니다.
환경변수 `TESSERACT_CMD`로 오버라이드 가능.

---

## 5. 선택 의존성 (optional)

### Phase 5/6 빌드용
```
pip install -r requirements-dev.txt
# pyinstaller>=6.6
# uiautomation>=2.0   (Phase 2 PoC `uia_explore.py` 전용)
```

### PaddleOCRv5 (선택 OCR 백엔드 — O-1)
```bash
# CPU
pip install "paddleocr>=3.0.0" "paddlepaddle>=3.0.0"
# 그리고
pip install "chardet<6"   # requests 경고 방지
```
GPU/CUDA는 PaddlePaddle 공식 설치 명령 참조.

### CTranslate2 (가속, 미사용 — 미래 옵션)
```bash
pip install ctranslate2
```

---

## 6. 환경 진단 자동화

```bash
# 더블클릭으로 진단 후 앱 시작
run_cocktail.bat

# 진단만
python diagnose_environment.py
```

진단 항목:
- 필수 Python 모듈
- Optional PaddleOCR
- Tesseract 실행 파일
- traineddata 목록
- Torch device (CPU/CUDA)

---

## 7. 새 환경에서 처음 시작하는 사람을 위한 빠른 가이드

```bash
# 1. Python 3.10/3.11 설치
# 2. (선택) conda 환경 생성
conda create -n cocktail python=3.10 -y
conda activate cocktail

# 3. CUDA 사용자: torch 별도 설치 권장 (공식 명령)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# 4. 나머지
pip install -r requirements.txt

# 5. Tesseract 엔진 설치
# https://github.com/UB-Mannheim/tesseract/wiki

# 6. 진단
python diagnose_environment.py

# 7. 실행
python "Cocktail완성본.py"
# 또는
run_cocktail.bat
```

---

## 8. 라이선스 호환 점검 (배포 전)

설치되는 모든 패키지가 상용 호환인지 [BUILD.md](BUILD.md) 라이선스 체크리스트 참고.
요약:
- ✅ PySide6 (LGPL), opus-mt (Apache 2.0), m2m100 (MIT), Tesseract (Apache 2.0), tessdata (Apache 2.0)
- ✅ keyboard (MIT), uiautomation (Apache 2.0)
- ✅ PaddleOCR/PaddlePaddle (Apache 2.0 계열)
- ❌ NLLB (CC-BY-NC), PyQt5 (GPL) — 절대 추가 금지

---

## 9. 알려진 환경 이슈

| 이슈 | 회피 |
|---|---|
| PaddleOCR 3.x + Windows CPU 기본 `paddle_static` 실패 | `engine="paddle"`, `enable_mkldnn=False` (코드 기본값) |
| chardet 7.x → requests 경고 | `chardet<6` 고정 |
| Python 3.12+ 일부 패키지 휠 미배포 | 3.10/3.11 권장 |
| transformers 5.x API 변경 | 본 코드는 5.3 기준. 마이너 변경 시 `from_pretrained` 시그니처 확인 |
| 일부 게임/관리자 권한 앱은 캡처 거부 | 알려진 한계. 게임 모드는 별도 |
| `keyboard` 글로벌 단축키 시 일부 환경 관리자 권한 필요 | optional 처리 — 미동작 시 로컬 단축키만 사용 |

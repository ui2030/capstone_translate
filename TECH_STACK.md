# Cocktail 기술 스택

## Runtime

- Python 3.10+
- PySide6: 투명 오버레이 UI, 버튼, 콤보박스, 상태 표시, Signal/Slot
- Windows API `SetWindowDisplayAffinity`: 오버레이가 자기 자신을 OCR 캡처하지 않도록 제외

## OCR

- 기본 백엔드: Tesseract OCR + pytesseract
- 선택 백엔드: PaddleOCR v3 / PP-OCRv5
- 화면 캡처: Pillow `ImageGrab`
- 전처리: OpenCV grayscale, resize, Otsu threshold, morphology
- OCR 언어:
  - Tesseract: `LANG_TO_TESS`와 설치된 traineddata 기준
  - PaddleOCR: source 언어 콤보 또는 `COCKTAIL_PADDLE_LANG` 기준

## 언어 감지

- Unicode script fast-path: `unicodedata`로 Hangul, Hiragana, Katakana, CJK, Arabic, Cyrillic 등 결정
- 짧은 라틴 표현 힌트: `Hvala`, `Merci`, `Hola`, `Danke` 등 langdetect 오판 보정
- YOLO식 bbox region smoothing: OCR 라인을 좌우 column/세로 island로 묶고, 같은 region의 강한 언어 신호로 짧은 라인을 안정화
- 폴백: langdetect

## 번역

- 1차 백엔드: Helsinki-NLP/opus-mt-tc-big-en-ko
- 폴백 백엔드: facebook/m2m100_418M
- 공통 최적화:
  - lazy load
  - batched generate
  - GPU fp16
  - `torch.inference_mode()`
  - LRU translation cache

## UI/UX 안정화

- 모델 로드 진행 표시
- OCR 백엔드 선택 표시
- 환경 진단 상태 라벨
- OCR 실패/백엔드 폴백/모델 준비 상태를 UI에 표시
- `run_cocktail.bat`와 `diagnose_environment.py`로 일반 사용자 진단 경로 제공

## 배포

- PyInstaller `build.spec`
- Inno Setup `installer.iss`
- Tesseract traineddata 선택 컴포넌트
- PaddleOCR는 optional dependency로 유지

## 라이선스 정책 (상용 호환만)

- ✅ PySide6 (LGPL), opus-mt (Apache 2.0), m2m100 (MIT), Tesseract (Apache 2.0), tessdata (Apache 2.0), PaddleOCR (Apache 2.0 계열)
- ❌ NLLB (CC-BY-NC), PyQt5 (GPL), 비상용 라이선스 모델

상세: `BUILD.md` 라이선스 체크리스트, `VISION_AND_ROADMAP.md` §4 차별 포인트.

## 미래 기술 스택 (Phase 1+, 계획 단계)

비전(`VISION_AND_ROADMAP.md`)에 따른 다음 도입 후보:

### Phase 1 — 백그라운드 + 모드 다양화
- `QSystemTrayIcon` / `QMenu` (PySide6 내장) — 트레이 상주
- 글로벌 단축키:
  - `keyboard` 패키지 (MIT, 라이선스 사전 확인 후) 또는
  - Windows API `RegisterHotKey` + `QAbstractNativeEventFilter` (의존성 없음)
- 활성 윈도우 좌표: `GetForegroundWindow` + `GetWindowRect` (Win32 ctypes)
- 마우스 hover: `QCursor.pos()` + 주기적 캡처
- 자동 실행: 레지스트리 `HKCU\Software\Microsoft\Windows\CurrentVersion\Run` 또는 시작 폴더 `.lnk`

### Phase 2 — UIA (UI Automation) PoC
- 후보: `uiautomation` (가벼움) 또는 `pywinauto` (기능 풍부)
- Windows 내장 접근성 API. 표준 UI 텍스트를 OCR 없이 직접 읽기.
- 별도 실험 스크립트 `uia_explore.py`로 검증 후 본 코드 통합 결정.

### Phase 3+ — 통합 및 확장
- UIA + OCR 하이브리드 라우팅
- 학습형 캐시 (디스크 캐시는 hash key 기반 — 평문 비저장, 보안 원칙)
- 민감 영역 자동 OFF (패스워드 입력 창 / 결제 페이지 감지)

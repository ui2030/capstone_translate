# Cocktail 기술 스택

## Runtime

- Python 3.10+
- PySide6: 투명 오버레이 UI, 버튼, 콤보박스, 상태 표시, Signal/Slot
- Windows API `SetWindowDisplayAffinity`: 오버레이가 자기 자신을 OCR 캡처하지 않도록 제외
- `CocktailAppController`: QApplication/설정창/오버레이 생성과 연결을 한 곳에서 관리

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
- 라틴/불명 스크립트: 추측하지 않고 전부 `en` (SL-1). 다른 라틴 언어는 사용자가 source 콤보에서 명시
- 폴백 없음 — langdetect 추측, region smoothing(M-7), 지배 언어 스냅(LG-1)은 SL-1에서 퇴역

## auto 모드 OCR 라우팅 (SL-1 — UX-2 SmartOCR OSD 레이어 대체)

합집합 OCR(8~10개 언어, 27초)은 UX-2에서 폐기했고, 그 자리를 채우던 OSD script 분류는
오판(영어 PDF → Arabic conf 3.95)이 잦아 SL-1에서 함께 퇴역했다. 지금 남은 레이어:
1. UIA 자동 라우팅 — 활성 창 + 기본 ocr_combo → UIA 우선 (OCR 불필요)
2. 화면 가시 영역 클립 — 가상 데스크톱 geometry intersect
3. auto OCR 언어 = `eng+kor` 고정 (`OCR_LANGS_AUTO_DEFAULT`) — 판별 비용 0

`COCKTAIL_OCR_LANGS_AUTO` 환경변수로 사용자 오버라이드 가능 (예: `eng+kor+jpn`).
`osd.traineddata`는 더 이상 쓰지 않는다. 인스톨러 동봉 목록과 `download_tessdata.py`에서도
제외했다(이미 설치돼 있어도 무해).

## 백그라운드 앱 구조

- `CocktailAppController`: 앱 생명주기 소유. `QApplication`, settings window, overlay window를 생성/연결.
- `BackgroundController`: 워커 thread, 엔진 시그널 5개, OCR/UIA 라우팅, 모델 lifecycle, 자동 일시정지,
  PersistentCache 타이머를 소유. `region`/`options_provider`/`is_self_hwnd` 3개 의존만 주입받음 (BG-4).
- `CaptureRegionController`: 캡처 모드, hover 크기, red rect(절대 좌표) 상태와 활성 창/hover 영역 계산을 소유.
  BG-6에서 owner 위젯 의존을 제거하고 `is_self_hwnd` 콜백만 받음.
- `SettingsWindow`: settings/control panel. UI 위젯, 트레이, 글로벌 단축키만 보유.
  엔진 시그널 5개를 `BackgroundController`로부터 connect. (BG-6 — `TransparentWindow`에서 리네이밍됨.)
- `OverlayWindow`: 전체 가상 데스크톱 위에 뜨는 투명 렌더러. 번역 박스만 그리며 마우스는 통과.
- `RuntimeOptions(@dataclass(frozen=True))`: 워커가 매 iteration 콤보 상태를 immutable 스냅샷으로 읽음 (BG-4).
- `QSystemTrayIcon`: 닫아도 앱이 종료되지 않고 트레이에 상주.
- `QSettings("Cocktail", "OverlayTranslator")`: 최초 1회 설정창 표시 여부 저장.

좌표계 정착 (BG-5 완료):
- `red_rect`는 화면 절대 좌표. settings window 이동/숨김에 영향받지 않음.
- worker는 `region.red_rect` 그대로 사용. OverlayWindow는 절대→overlay-내 변환 한 번만 수행.

구조 정착 (BG-6 완료):
- `TransparentWindow` → `SettingsWindow` 리네이밍 완료.
- `CaptureRegionController.owner` 의존 제거. `is_self_hwnd` 콜백 통합.
- `SettingsWindow`의 빈 `paintEvent` override 제거. Qt 기본이 위젯 트리를 그림.

남은 부채 (별도 사이클):
- 영역 편집 UX: 추후 `RegionEditor` 위젯이 같은 `CaptureRegionController.red_rect`(절대 좌표)를 조작.

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

상세: `COMMERCIAL_SOTA_REVIEW.md`, `BUILD.md` 라이선스 체크리스트, `VISION_AND_ROADMAP.md` §4 차별 포인트.

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

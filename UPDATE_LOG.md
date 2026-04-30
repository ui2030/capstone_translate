# Cocktail 업데이트 로그

## 2026-04-30

### U-1. 일반 사용자 진입 실패 방지

- Tesseract 실행 파일이 없어도 앱 import/startup 단계에서 즉시 죽지 않도록 변경.
- 시작 진단 요약을 UI 상태 라벨에 표시.
- `Tesseract`, `PaddleOCR`, 번역 device 상태를 한 줄로 표시.
- OCR 결과가 없을 때 영역/언어/OCR 백엔드 확인 메시지를 UI에 표시.

### U-2. OCR 백엔드 상태 표시

- `Tesseract` / `PaddleOCRv5` 선택 콤보 추가.
- PaddleOCR 미설치, 초기화 실패, 추론 실패 시 Tesseract 폴백 메시지 표시.
- 번역 성공 시 OCR 줄 수와 번역 소요 시간을 상태 라벨에 표시.
- 현재 Windows CPU 환경에서 PaddleOCR 3.5.0 / PaddlePaddle 3.3.1 설치 후 smoke test 수행.
- 기본 `paddle_static` 경로는 oneDNN 오류가 발생해, 앱 기본 Paddle 엔진을 `paddle` + `enable_mkldnn=False`로 변경.
- PaddleOCR 설치 후 `chardet 7.x` 때문에 `requests` 경고가 발생해 `chardet<6` 호환 범위로 정리.

### U-3. 일반 사용자 실행 보조

- `run_cocktail.bat` 추가.
- `diagnose_environment.py` 추가.
- 필수 Python 모듈, optional PaddleOCR, Tesseract 실행 파일, traineddata, Torch device를 점검.

### U-4. 기술 문서화

- `TECH_STACK.md` 추가.
- 사용 기술, OCR/번역/언어감지/UI/배포 구조를 별도 문서로 정리.

### M-7. YOLO식 공간 region 언어 안정화

- OCR 라인 bbox를 좌우 column/세로 island 기준으로 묶는 `_cluster_text_regions()` 추가.
- 라인별 언어 신뢰도 `_language_confidence()` 추가.
- `assign_region_languages()`로 같은 region 안의 강한 언어 신호를 짧고 불안정한 라인에 전파.
- `batch_translate()`가 라인별 source language list를 받을 수 있게 확장.
- 상태 라벨에 이번 OCR 프레임의 감지 언어 요약을 표시.

### V-1. 장기 비전 정의 — 컴퓨터를 모국어로

- 영역 박스 도구에서 **OS 전반 자동 번역 환경**으로의 pivot 비전 합의.
- 사용자 표현: "내가 사용하는 언어로만 볼 수 있는 세상".
- 시장 분석: PC OS 통합 번역기 메이저 제품 없음 (Apple/Google은 모바일만, MS PowerToys는 번역 없음, QTranslate/Capture2Text는 노후화) → 빈 자리.
- 기술 접근: **UIA + OCR 하이브리드**.
  - 1차 UI Automation으로 표준 UI 텍스트 직접 읽기 (OCR 불필요, 정확).
  - 2차 OCR 폴백 (게임/Custom GDI/영상 자막).
- 보안 5원칙: 100% 로컬 / 번역 데이터 비저장 / 민감 영역 자동 OFF / 오픈 코어 / 즉시 토글.
- 단계별 로드맵 Phase 0~6 정의 — 상세는 `VISION_AND_ROADMAP.md`.
- README.md 갱신 — 비전 + 현재 상태 + 학술 배경 보존.
- 메모리에 `project_vision.md` 영구 기록 (다른 세션 컨텍스트용).
- 코드 작업은 다음 단계로 미룸 (사용자 검증 + 단계별 견고 진행 원칙).

### 다음 단계 (Phase 1 — 계획)

- 시스템 트레이 + 글로벌 단축키 + 모드 다양화 (박스/활성 창/마우스 hover/전체 화면).
- 의존성 후보: `keyboard` (글로벌 단축키, MIT) — 라이선스 사전 확인 후 도입.
- Phase 2 PoC: `uia_explore.py` 별도 실험 스크립트 — `uiautomation` 또는 `pywinauto` 사용.

### Documentation. STORY.md 신규 — 프로젝트 여정 내러티브

- `STORY.md` 추가 — 0장(시작) ~ 9장(비전 도약)까지 결정의 흐름과 *왜*를 정교하게 기록.
- 다른 Claude 세션 또는 협업자가 이 문서 하나만 읽어도 전체 맥락을 이해할 수 있도록 자기-완결적 작성.
- README.md에 진입점으로 추가.
- 핵심 결정 8개를 한 줄 요약으로 마무리.

### T-1. 시스템 트레이 + 글로벌 단축키 (Phase 1A)

- `QSystemTrayIcon` + `QMenu` 추가 — 열기 / 번역 토글 / 캡처 모드 / 종료.
- 닫기 시 hide → 트레이로 minimize. 진짜 종료는 트레이 메뉴 또는 `_is_quitting` 플래그.
- `keyboard` 패키지(MIT) optional 도입 — 미설치 시 앱 정상 동작, 단축키만 비활성.
- 글로벌 단축키: `Ctrl+Shift+T`(번역 토글), `Ctrl+Shift+A`(창 다시 열기).
- keyboard 콜백은 별도 스레드 → 시그널 `global_toggle_signal` / `global_show_signal` 로 메인 스레드 진입.

### T-2. 캡처 모드 다양화 — 영역 박스 / 활성 창 / 마우스 hover (Phase 1A)

- UI 모드 콤보 추가 + 트레이 메뉴에도 동기화.
- `_foreground_window_rect()` — `GetForegroundWindow` + `GetWindowRect` (Win32 ctypes).
  우리 창 hwnd면 None 반환 (B-10 피드백 차단).
- `_mouse_hover_rect()` — `QCursor.pos()` ± 400×200.
- 워커 루프가 매 사이클 `_update_red_rect_for_mode()` 호출 → 절대 좌표를 위젯 내 red_rect로 변환.
- 모드 전환 시 `last_frame_hash = None` 으로 frame-skip 캐시 무효화.

### Phase 2 PoC. uia_explore.py 신규

- Phase 3(UIA + OCR 하이브리드) 진입 전 검증용.
- `uiautomation` (Apache 2.0) 사용. `requirements-dev.txt`에만 추가 — 본 앱 의존성 X.
- 5초 지연 후 활성 창 UIA 트리 출력 (`--depth`, `--watch` 옵션).
- 결과를 보고 어떤 앱이 어디까지 읽히는지 매트릭스화 → Phase 3 통합 결정.

### W-1. 마우스 통과 토글 (Phase 1B)

- `set_mouse_through(hwnd, on)` 헬퍼 — Win32 `WS_EX_TRANSPARENT` 스타일 토글.
- `WS_EX_LAYERED`는 항상 유지 (반투명 그리기 보존).
- 단축키 `Ctrl+Shift+P` + 트레이 메뉴 checkable로 노출.
- 활성 창/hover 모드에서 사용자가 다른 창을 자유롭게 클릭/조작할 수 있도록 함.
- 진정한 매끄러움(컨트롤 윈도우 분리)은 Phase 1C에서 진행 예정.

### S-1. 민감 영역 자동 OFF (Phase 5, 보안 원칙 #3)

- `SENSITIVE_KEYWORDS` 사전 — 영문/한국어 보편 + 한국 주요 은행/금융앱 제목.
- `is_sensitive_window(hwnd)` — `GetWindowText` + `GetClassName` 검사.
- `_check_sensitive_pause()` 워커 진입부에 추가 — 박스 모드 외에서 민감 창 활성 시 OCR/번역 일시 중지.
- 해제 시 상태 라벨에 "민감 창 해제 — 번역 재개" 명확 표기.
- VISION_AND_ROADMAP.md §5 보안 원칙 #3 첫 구현.

### W-2. OverlayWindow 분리 (Phase 1C)

- 신규 `OverlayWindow(QMainWindow)` — 전체 화면 + 투명 + 항상 위 + `Qt.Tool`(작업표시줄 안 뜸) +
  `WA_ShowWithoutActivating`(포커스 도둑질 X) + 마우스 통과 영구 ON + B-10 캡처 차단.
- 빨간 박스 + 번역 텍스트 모두 OverlayWindow가 그림. 좌표는 화면 절대.
- TransparentWindow는 컨트롤(콤보/버튼/상태/트레이/워커/시그널) 보유 + paintEvent 비움.
- 워커의 `adjusted_bbox`를 위젯 좌표 → 화면 절대 좌표로 변경.
- main에서 `window.overlay = overlay`로 연결. 상태/모드 변경 시 `overlay.update()` / `overlay.update_for_mode()`.
- 종료 시 ControlWindow가 OverlayWindow도 close.
- 사용자가 마우스 통과 토글 걱정 없이 다른 창을 자유롭게 클릭 가능 — 비전 V-1의 매끄러운 작동.

### Documentation. ENVIRONMENT.md 신규

- 사용자 머신에서 검증된 Python 3.10.18 / pip 26.0.1 / Torch 2.7.1+cu118 / CUDA 11.8 기록.
- 이미 설치된 패키지(numpy/pillow/tokenizers/torch/transformers) vs 누락(PySide6/pytesseract/opencv/langdetect/sentencepiece/sacremoses/keyboard) 명시.
- 새 환경 빠른 시작 가이드 + Tesseract 엔진 설치 안내 + 알려진 환경 이슈 표.
- 라이선스 호환 점검 체크리스트로 BUILD.md 연결.

### W-3. 멀티 모니터 대응 (Phase 1D)

- `OverlayWindow._fit_to_screen()` — `QApplication.screens()` 모두 합집합 bounding rect.
- paintEvent: 화면 절대 좌표 → 윈도우 내 좌표 변환 (`bbox - geometry().topLeft()`).
- 보조 모니터의 활성 창에서도 빨간 박스/번역 정상 표시.

### UIA-1. UIA 어댑터 + 워커 통합 (Phase 2/3)

- `UIATextProvider` — `uiautomation` lazy import. 미설치 시 graceful 비활성.
- ocr_combo에 "UIA (활성 창)" 옵션 추가 (uiautomation 사용 가능할 때만).
- `_uia_collect_and_translate()` 메서드 — 활성 창에서 UIA 트리 walk → bbox+텍스트 직접 추출.
- 워커 루프가 UIA 모드면 OCR 건너뛰고 UIA 결과로 emit. 빈 결과면 OCR 폴백.
- 박사 자체 리뷰: UIA 모드 항상 민감 창 검사 (보안 원칙 #3 일관 적용). 결과 100개 상한.
- 비전 V-1 핵심 — Apple Live Text 같은 정확/빠른 직접 읽기 첫 발걸음.

### D-1. DPAPI 영구 캐시 (Phase 4, 보안 원칙 #2)

- `dpapi_protect/unprotect()` — Windows DPAPI 암호화 (사용자 계정으로만 복호화).
- `PersistentCache` — `~/.cocktail/translation_cache.bin`에 암호화 저장.
- 키는 `sha256(src|tgt|text)`. 평문 source 디스크 미저장.
- batch_translate 통합: 메모리 LRU miss → 디스크 캐시 조회 → 메모리 승격.
- 박사 자체 리뷰: __init__에서 디스크 I/O 제거 → `preload_async()`로 백그라운드 로드 (P-9 정신).
- closeEvent에서 atomic replace로 save.
- 자주 쓰는 UI 라벨이 앱 재시작 후에도 즉시 표시 (학습형 효과).

### Documentation. ENVIRONMENT.md 신규

(이전 entry — 환경 정보 영구 기록)

### 박사 자체 비판 리뷰 도입

- 사용자 약속: 코드 수정 후 사용자에게 보내기 전 박사 페르소나로 자체 비판 리뷰 실시.
- 발견 결함 즉시 수정 + 결과 보고에 명시.
- 이번 사이클(Phase 1C/1D/2/3/4 동시) 자체 리뷰 발견 결함 6건 — 즉시 수정 완료.

### A-1. Phase 6 마무리 (주기적 save + DPAPI 보강 + 자동 실행)

- **주기적 save**: `QTimer` 30초 인터벌로 `PERSIST_CACHE.save()` 호출 (변경 없으면 no-op).
  비정상 종료 시 손실 폭 ≤ 30초.
- **DPAPI argtypes**: `_setup_crypt32()`로 `CryptProtectData`/`CryptUnprotectData`의
  argtypes/restype 명시 — 64비트 호출 규약 정확성 보장.
- **자동 실행**: `winreg`로 `HKCU\...\Run` 토글. 트레이 메뉴 "Windows 시작 시 자동 실행" checkable.
- **마켓 진입 자산**:
  - `LICENSE` (MIT — 박사 결정, 사용자가 다른 라이선스 원하면 즉시 교체)
  - `LICENSE-3RDPARTY.md` (의존성 라이선스 정리 + 상용 호환 체크리스트)
  - `CODE_SIGNING.md` (Authenticode + signtool + GitHub Actions 통합 가이드)
  - `ICONS.md` (아이콘 자산 가이드 + 코드 통합)
  - `.github/workflows/release.yml` (태그 push → PyInstaller 빌드 → GitHub Release 자동)
  - `README.en.md` (영어 진입점)

박사 자체 비판 리뷰 발견 잠재 결함 4건은 errors.md A-1 한계 절에 기록 (즉시 수정 X, 다음 턴 보강 후보).

### 다음 단계 (사용자 결정)

**사용자에게 정직 보고** — 다음 사이클 후보:
- 자동 실행 시 콘솔 창 뜨지 않게 `pythonw.exe`/`.exe` 우선 등록
- 주기적 save를 백그라운드 스레드로 (메인 스레드 블로킹 회피)
- GitHub Actions에 Inno Setup 인스톨러 빌드 단계 추가
- LICENSE를 다른 라이선스로 교체할지 사용자 결정 (현재 MIT)
- UIA 결과 매트릭스 수집 (`uia_explore.py` + 실 사용)
- 아이콘 자산 디자인 / 외주
- 베타 테스터 모집

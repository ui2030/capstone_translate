# Cocktail 프로젝트 핸드오프 — 2026-05-04

> 집에서 다시 시작할 때 이 문서 하나만 보고 따라가면 됨.
> 박사 모드 재개 멘트, 검증 명령, 빌드 명령, 다음 단계 옵션까지 전부 들어 있음.

---

## 0. 목표 한 줄 요약

> **"컴퓨터를 모국어로 보는 가장 짧은 길"** — 로컬 추론 OS-전반 자동 번역기.

자세히:
- 영역 박스로 한 곳을 번역하는 도구로 시작 → 궁극 목표는 OS 전반의 외국어를 사용자 모국어로 자동 표시.
- 인식·번역 전 과정 로컬 처리. 화면/번역 데이터 외부 전송 없음.
  단 최초 1회 모델 다운로드(HF) + Tesseract 별도 설치 필요.
- 상용 closed-source 배포 가능 라이선스만 사용 (PySide6 LGPL + opus-mt CC-BY-4.0 + m2m100 MIT + Tesseract Apache).
- 비전 / 로드맵 원본: [VISION_AND_ROADMAP.md](VISION_AND_ROADMAP.md)
- 구조 설계: [ARCHITECTURE.md](ARCHITECTURE.md)

---

## 1. 현재 상태 한눈에 (2026-08-02 기준)

- ✅ Phase 0~6 (기능) 완료. 베타 가능 + 마켓 진입 자산 갖춤.
- ✅ Phase 0 분리 사이클 (BG-1~BG-6) 완료. 5개 컴포넌트로 책임 정착.
- ✅ 1~7차 수정 패스 완료: DP-1 물리 픽셀 좌표 통일, SL-1 auto 결정론 라우팅,
  DG-1 표시 게이트, LM-1 라인/폰트 이상치, RC-1 워커 race, SV-1 민감 창 OCR 누출, 멀티모니터 핫플러그.
- ✅ 8차 배포 준비 패스: `run_cocktail.bat` 인터프리터 자동 탐색, 문서 현행화, 사어 의존 제거.
- ✅ smoke 35건 / pre-release 17건 / --self-test 통과.
- ✅ PyInstaller 빌드 786MB / 6372 files / `dist\Cocktail\Cocktail.exe --self-test exit 0` (5/4 기준 — 이후 재빌드 필요).
- 🟡 사용자 GUI 검증 / 실 환경 베타 / 인증서 서명 / RegionEditor UX는 미진행.
- ✅ `nllb_ct2/` (600MB, CC-BY-NC 잔재) 2026-08-03 삭제 완료. `.gitignore`로 재추가 차단.

코드 구조: 역할별 6개 모듈 (cocktail.py 엔트리 + platform/capture/ocr/translate/engine/ui). 2026-08-03 단일 파일 3,310줄에서 분할.

---

## 2. Phase별 진행 정리

### 📗 Chapter A — Phase 0: 영역 OCR + 오버레이 (✅ 완료)

**목표**: 사용자가 그린 빨간 박스 안의 외국어를 OCR → 번역 → 오버레이로 띄우는 핵심 루프.

**완료 내역**:
- Tesseract OCR + PaddleOCRv5 선택 백엔드.
- opus-mt(en→ko 전용) + m2m100(any-to-any 폴백) 하이브리드 번역.
- Unicode script 기반 다국어 자동 라우팅 (Hangul / Hiragana / CJK / Arabic / Cyrillic).
- 라인 언어는 문자 스크립트로만 결정, 라틴/불명은 en (SL-1). M-7 region smoothing은 퇴역.
- 자기 캡처 피드백 차단 (`SetWindowDisplayAffinity` + B-15 자동 가드).
- 한국어 박스 자동 폰트 축소 + height 동적 확장 (B-12/B-13).
- 모델 lazy load (P-9): UI 즉시 표시, 모델은 백그라운드 준비.
- 진단 스크립트 + status label로 일반 사용자 환경 문제 표면화 (U-1).

---

### 📗 Chapter B — Phase 1: 백그라운드 + 모드 다양화 (✅ 완료)

| 단계 | 내용 | 코드 식별자 |
|---|---|---|
| 1A | 시스템 트레이(`QSystemTrayIcon`) + 글로벌 단축키(`keyboard` 패키지) + 모드 콤보 | T-1, T-2 |
| 1B | 마우스 통과 토글 (`Ctrl+Shift+P`, `WS_EX_TRANSPARENT`/`LAYERED`) | W-1 |
| 1C | OverlayWindow 분리 (전체 화면 투명 + 항상 위 + 마우스 통과 + 캡처 차단) | W-2 |
| 1D | 멀티 모니터 (가상 데스크톱 전체 영역으로 OverlayWindow 확장) | W-3 |

**모드**:
- `MODE_WINDOW` — 활성 창 자동 (UX-1 기본값)
- `MODE_HOVER` — 마우스 커서 주변
- `MODE_BOX` — 빨간 박스로 사용자가 정한 정확 영역

---

### 📗 Chapter C — Phase 2/3: UIA 어댑터 (✅ 1차 완료)

**목표**: OCR 없이 표준 UI 텍스트를 직접 추출 (Windows UI Automation API).

**완료 내역**:
- `UIATextProvider` 어댑터 (`uiautomation` lazy import — 미설치 시 graceful 비활성).
- ocr_combo에 "UIA (활성 창)" 옵션.
- 워커 자동 라우팅 (UX-2 Layer 1) — 활성 창 모드 + ocr_combo 디폴트일 때 UIA 우선 → 실패 시 OCR 폴백.
- 사용자 명시 선택("Tesseract"/"PaddleOCRv5") 시 UIA 우회 (의도 존중).
- 결과 100개 상한 (paintEvent 부담 회피).
- 민감 창 검사 일관 적용 (UIA-1).

**미진행**: `uia_explore.py`로 어떤 앱이 어디까지 읽히는지 매트릭스 수집 — 별도 실험 단계.

---

### 📗 Chapter D — Phase 4: DPAPI 영구 캐시 (✅ 완료)

**목표**: 자주 보는 자막/UI 텍스트는 디스크 캐시로 즉시 표시 (보안 원칙 #2 — 평문 비저장).

**완료 내역**:
- `PersistentCache` 클래스 (`~/.cocktail/translation_cache.bin`).
- sha256 hash key + Windows DPAPI(`CryptProtectData`)로 전체 dict 암호화.
- Lazy load (`preload_async` — UI 시작 차단 X) + import 시점 디스크 I/O 금지 (P-9).
- 30초 주기 자동 save (`_cache_save_timer`, A-1) + closeEvent에서 최종 save.
- 백그라운드 스레드 save + `_mem_lock`으로 race 방지 (A-2).
- DPAPI helper `argtypes/restype` 명시 (64비트 호출 규약).

---

### 📗 Chapter E — Phase 5: 민감 영역 자동 OFF (✅ 완료)

**목표**: 패스워드 / 로그인 / 결제 / 은행 창에선 OCR/번역을 자동 중지 (보안 원칙 #3).

**완료 내역**:
- `SENSITIVE_KEYWORDS` 사전 (한국어 + 영어 키워드).
- `is_sensitive_window(hwnd)` — `GetWindowText` + `GetClassName` 검사 (S-1).
- 워커 + UIA 경로 둘 다 적용. 박스 모드 외에서만 적용 (사용자 명시 영역은 책임이 사용자에게).
- 민감 창 감지 시 `state_ready.emit([])` + status_msg로 사용자 알림.

---

### 📗 Chapter F — Phase 6: 마켓 진입 자산 (✅ 1차 완료)

**완료 내역**:
- `LICENSE` (MIT) + `LICENSE-3RDPARTY.md` (모든 의존성 라이선스 정리).
- `CODE_SIGNING.md` (Windows Authenticode 서명 가이드).
- `ICONS.md` (아이콘 자산 가이드).
- `.github/workflows/release.yml` — 태그 push → PyInstaller 빌드 + 서명 + Inno Setup 인스톨러 + GitHub Release 자동.
- Windows 시작 자동 실행 옵션 (`winreg` HKCU\Run, 트레이 메뉴 토글).
- 자동 실행 명령은 `pythonw.exe` 우선 (콘솔 창 회피, A-2).
- `README.en.md` 영어 진입점.
- `pre_release_check.py` 자동 게이트.

**미진행**:
- 실제 코드 서명 인증서 발급/적용.
- 디자이너 아이콘 자산.
- 베타 테스터 모집 / 피드백 사이클.

---

### 📗 Chapter G — Phase 0 분리 사이클: BG-1 ~ BG-6 (✅ 종결)

**목표**: 단일 거대 클래스 `TransparentWindow`(약 2400줄)에 모든 책임이 몰린 구조를 5개 컴포넌트로 분리.

| 사이클 | 추출 대상 | 결과 |
|---|---|---|
| **BG-1** | Tray-first 백그라운드 패턴 | 설정창은 최초 1회만 표시. `setQuitOnLastWindowClosed(False)`. `QSettings("Cocktail", "OverlayTranslator")` |
| **BG-2** | `CocktailAppController` | Qt 앱 생명주기 일원화. `__main__`에 흩어진 생성 코드 정리. `--self-test` 분기는 QApplication 생성 전 |
| **BG-3** | `CaptureRegionController` | 캡처 모드 / hover 크기 / `red_rect` 상태 분리. settings panel resize ≠ 영역 편집 |
| **BG-4** | `BackgroundController(QObject)` | 워커 thread + 엔진 시그널 5개 + OCR/UIA 라우팅 + 모델 lifecycle + 캐시 타이머 분리. `RuntimeOptions(@dataclass(frozen=True))` immutable 스냅샷 |
| **BG-5** | `red_rect` 절대 좌표화 | settings 창 위치 의존 제거. worker / OverlayWindow 좌표 변환 코드 단순화 |
| **BG-6** | `SettingsWindow` 리네이밍 + region 콜백 통합 + dead code 정리 | `TransparentWindow` → `SettingsWindow`. region.owner 제거 → `is_self_hwnd` 콜백. 빈 paintEvent override 제거 |

**산출물**:
- [ARCHITECTURE.md](ARCHITECTURE.md) 신규 — As-Is / To-Be / 마이그레이션 / 좌표계 결정.
- smoke 테스트 21건 (BG-1~6 모두 검사).
- pre_release_check 14건 (BG-1~6 게이트).

---

## 3. 현재 아키텍처 (BG-6 완료 시점)

### 5개 컴포넌트

```
┌─────────────────────────────────────┐
│  CocktailAppController              │ 앱 생명주기 / 컴포넌트 생성·연결
└──────────────┬──────────────────────┘
               │
        ┌──────┴──────┬─────────────────┬──────────────────┐
        ▼             ▼                 ▼                  ▼
┌──────────────┐ ┌─────────────┐ ┌──────────────┐ ┌──────────────┐
│ Settings     │ │ Background  │ │ CaptureRegion│ │  Overlay     │
│ Window       │ │ Controller  │ │ Controller   │ │  Window      │
│              │ │             │ │              │ │              │
│ • UI widgets │ │ • worker    │ │ • capture    │ │ • paintEvent │
│ • tray       │ │   thread    │ │   mode       │ │ • virtual    │
│ • 단축키     │ │ • signals×5 │ │ • red_rect   │ │   desktop    │
│ • combo 변경 │ │ • OCR/UIA   │ │   (절대 좌표)│ │ • mouse-thru │
│ • mode 토글  │ │ • 모델 lc   │ │ • 영역 계산  │ │ • capture    │
│              │ │ • cache 타이│ │              │ │   excluded   │
└──────────────┘ └──────┬──────┘ └──────────────┘ └──────────────┘
                        │
        signals ────────┘
        - state_ready(list)         worker → UI redraw + state apply
        - failure_alert(str)        worker → run_button text
        - model_ready()             preload 끝 → run_button enable
        - progress_msg(str)         preload 진행 → run_button text
        - status_msg(str)           worker/UIA → status_label
```

### 의존 주입 패턴

```python
# CocktailAppController.start()
self.settings_window = SettingsWindow()       # 자체적으로 region + controller 생성
self.overlay_window = OverlayWindow(self.settings_window)
self.settings_window.overlay = self.overlay_window
```

```python
# SettingsWindow.__init__
self.region = CaptureRegionController(is_self_hwnd=self._is_self_hwnd)
self.controller = BackgroundController(
    region=self.region,
    options_provider=self._snapshot_options,  # → RuntimeOptions(frozen)
    is_self_hwnd=self._is_self_hwnd,           # 동일 콜백을 region/BC가 공유
)
```

### 좌표계 (BG-5 정착)

- `red_rect` — **화면 절대 좌표** (`CaptureRegionController` 소유)
- `OCR bbox` — screenshot 내 좌표 → worker가 절대로 변환
- `state.adjusted_bbox` — 화면 절대 (worker → UI)
- `OverlayWindow paintEvent` — 절대 → overlay-내 변환 (`rect.x() - wx`) 한 번만

---

## 4. 검증 명령 (원터치 — 복붙)

```powershell
cd "c:\capstone_translate-main"
python -m py_compile cocktail.py cocktail_platform.py cocktail_capture.py cocktail_ocr.py cocktail_translate.py cocktail_engine.py cocktail_ui.py
python test_cocktail.py
python pre_release_check.py --allow-build-artifacts
python cocktail.py --self-test
```

**기대 결과**:
- smoke: `== Result: OK ==` (21 fast tests)
- pre-release: `== Result: OK ==` (14 OK)
- self-test: `[SELFTEST] OK`

---

## 5. 빌드 명령 (PyInstaller + Inno Setup)

### 1) Tesseract traineddata 다운로드 (한 번만)
```powershell
cd "c:\capstone_translate-main"
python download_tessdata.py
```

### 2) PyInstaller 빌드
```powershell
python -m PyInstaller --noconfirm build.spec
dist\Cocktail\Cocktail.exe --self-test
```

기대: `[SELFTEST] OK` + 786MB / 6372 files.

### 3) Inno Setup 인스톨러 (옵션)
```powershell
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer.iss
```

`Output/Cocktail-Setup-1.0.0.exe` 생성됨.

### 4) PaddleOCR 포함 빌드 (옵션)
```powershell
$env:COCKTAIL_BUNDLE_PADDLE = "1"
python -m PyInstaller --noconfirm build.spec
```

---

## 6. 다음 사이클 후보 (집에서 고를 옵션)

박사 권장 우선순위 순서:

### 🅰️ Option A — RegionEditor 위젯 (영역 편집 UX 보강) ★ 추천
**이유**: BG-5에서 MODE_BOX UX가 "settings 창 드래그로 박스 이동" → "박스가 settings와 분리된 절대 영역"으로 바뀌었음. 사용자가 박스를 마우스로 직접 편집할 UI가 필요.

**작업 요지**:
- 새 위젯 `RegionEditor(QWidget)` — 화면 위에 떠 있는 반투명 창.
- 드래그로 이동, 모서리 드래그로 리사이즈.
- 저장 시 `CaptureRegionController.red_rect` (절대 좌표)로 반영.
- `SettingsWindow`에 "영역 편집..." 버튼 추가 → `RegionEditor.show()`.
- MODE_BOX일 때만 의미 있음.

**예상 사이클**: BG-7 또는 RE-1.

---

### 🅱️ Option B — UIA 매트릭스 자동 수집 (Phase 2 PoC 마무리)
**이유**: VISION_AND_ROADMAP.md Phase 2 잔여 항목. 어떤 앱(작업 관리자, 설정, Slack, Chrome, VS Code 등)이 UIA로 어디까지 읽히는지 매트릭스 데이터 부재.

**작업 요지**:
- `uia_explore.py`를 자동 모드로 확장 — 표준 앱 목록을 띄우고 트리 dump.
- 결과를 `OCR_TEST_PLAN.md`에 표 형태로 기록.
- 잘 안 읽히는 앱은 OCR 폴백 의도 명시.

**효과**: 사용자에게 "어떤 앱에서 잘 동작하는지" 명확한 보장. 베타 출시 전 신뢰도 자료.

---

### 🅲️ Option C — 코드 서명 인증서 발급 + 적용
**이유**: Windows SmartScreen이 미서명 .exe를 차단함. 베타 사용자 진입 장벽.

**작업 요지**:
- DigiCert / Sectigo 등에서 EV 또는 OV 코드 서명 인증서 발급 (개인 ~$100/년).
- `CODE_SIGNING.md`에 적힌 절차로 GitHub Actions secret에 인증서 등록.
- `release.yml`의 "Sign EXE (optional)" 단계 활성화.

**효과**: SmartScreen 경고 제거. 인스톨러 신뢰도 ↑.

---

### 🅳️ Option D — 사용자 GUI 검증 사이클 (실 환경 회귀)
**이유**: 박사는 GUI 검증 불가능. UX-1 / UX-2 / B-15 / BG-1~6의 실제 동작은 사용자 검증 필요.

**작업 요지**:
- 박스 모드 / 활성 창 모드 / hover 모드 각각 실 사용.
- 한국어/일본어/중국어 자막 OCR 정확도 확인.
- 트레이 / 글로벌 단축키 / 자동 실행 토글 동작.
- 민감 창(은행/카드사) 자동 OFF 동작.
- 발견된 결함은 errors.md에 P-/B-/UX- 등으로 누적.

---

### 🅴️ Option E — Phase 1+ 비전 추가 작업
**이유**: VISION_AND_ROADMAP.md에는 다음 후보 기술이 적혀 있으나 미실현.

**후보**:
- 학습형 디스크 캐시 (사용자별 자주 보는 텍스트 우선 표시).
- macOS / Linux 이식 (AT-SPI / AX API).
- 비전 V-2 모델 통합 (멀티모달 Llava 등). 단, 라이선스 사전 확인 필수.
- 결제/은행 제외 외에 더 정교한 민감 영역 검출 (정규식 + ML 분류기).

---

## 7. 주요 파일 / 위치 가이드

### 코드
- [cocktail.py](cocktail.py) — 엔트리포인트 (앱 생명주기) + `cocktail_{platform,capture,ocr,translate,engine,ui}.py`
- [test_cocktail.py](test_cocktail.py) — 실동작 회귀 테스트 24건 (+ `--full` 모델 통합 1건)
- [pre_release_check.py](pre_release_check.py) — 배포 전 게이트 17건
- [diagnose_environment.py](diagnose_environment.py) — 사용자 환경 진단
- [download_tessdata.py](download_tessdata.py) — traineddata 다운로드 (eng/kor/jpn/chi_sim/chi_tra/fra/deu/rus/ara)
- [uia_explore.py](uia_explore.py) — Phase 2 UIA PoC

### 빌드 / 배포
- [build.spec](build.spec) — PyInstaller 설정 (PaddleOCR 게이트, .lib/include 필터)
- [installer.iss](installer.iss) — Inno Setup (traineddata 컴포넌트 선택)
- [.github/workflows/release.yml](.github/workflows/release.yml) — 자동 빌드/서명/릴리즈
- [requirements.txt](requirements.txt) / [requirements-dev.txt](requirements-dev.txt)

### 문서 (현재 상태)
- [README.md](README.md) / [README.en.md](README.en.md) — 진입점
- [ARCHITECTURE.md](ARCHITECTURE.md) — **As-Is/To-Be/마이그레이션** 설계 ⭐ (BG-N 진행 시 매번 갱신)
- [TECH_STACK.md](TECH_STACK.md) — 사용 기술 한 페이지
- [COMMERCIAL_SOTA_REVIEW.md](COMMERCIAL_SOTA_REVIEW.md) — SOTA / 라이선스 기준선
- [VISION_AND_ROADMAP.md](VISION_AND_ROADMAP.md) — 장기 비전 + Phase 0~6 + 보안 원칙
- [ENVIRONMENT.md](ENVIRONMENT.md) — Python/패키지 검증 버전
- [BUILD.md](BUILD.md) — 빌드 절차
- [CODE_SIGNING.md](CODE_SIGNING.md) — Authenticode 가이드
- [ICONS.md](ICONS.md) — 아이콘 자산 가이드
- [STORY.md](STORY.md) — 프로젝트 여정 내러티브
- [LICENSE](LICENSE) (MIT) / [LICENSE-3RDPARTY.md](LICENSE-3RDPARTY.md) — 라이선스 정리

### 문서 (히스토리)
- [errors.md](errors.md) — **결함 누적 로그** ⭐ (코드 수정 전 5초 체크리스트 필독)
- [업데이트.md](업데이트.md) — 한국어 사이클별 업데이트
- [UPDATE_LOG.md](UPDATE_LOG.md) — 영어 업데이트 요약

### 박사 자체 노트 (`.claudeignore`로 다음 세션 자동 컨텍스트 제외)
- [SESSION_PROGRESS.md](SESSION_PROGRESS.md)
- [CODEX_SESSION_RECHECK_2026-05-04.md](CODEX_SESSION_RECHECK_2026-05-04.md)
- [목록.txt](목록.txt) / [박사_템플릿.md](박사_템플릿.md)
- [HANDOFF.md](HANDOFF.md) — **이 파일** (다음 세션에 자동으로 안 읽힘. 사용자가 명시 요청 시 Read 가능)

---

## 8. 박사 모드 재개 (다음 세션 시작 멘트 — 복붙)

### 🅰️ Option A로 가려면
```
박사 안녕. RegionEditor 위젯 추가 시작하자. 영역 편집 UI 잡혀 있는 거 BG-7 사이클로 가면 돼. 아키텍처 먼저 ARCHITECTURE.md에 적고 시작해. HANDOFF.md 7번 항목 참고.
```

### 🅱️ Option B로 가려면
```
박사 안녕. UIA 매트릭스 수집 시작하자. uia_explore.py 자동 모드 확장 + OCR_TEST_PLAN.md 갱신. HANDOFF.md 7번 항목 참고.
```

### 🅲️ Option C로 가려면
```
박사 안녕. 코드 서명 인증서 적용 진행. CODE_SIGNING.md + release.yml 보고 절차 정리. 인증서 발급은 내가 했고 .pfx 파일 위치만 알려줄게.
```

### 🅳️ Option D로 가려면
```
박사 안녕. GUI 검증해보고 결함 발견했어. errors.md에 P-/B-/UX- 누적해줘. 발견 내용은: [내용 기재]
```

### 🅴️ Option E로 가려면
```
박사 안녕. Phase 1+ 비전 작업 시작하자. VISION_AND_ROADMAP.md 8번 한계 보고 우선순위 정하자.
```

### 🆗 어디까지 했는지부터 보고 결정하려면
```
박사 안녕. HANDOFF.md 읽고 어디까지 됐는지 다시 정리해줘. 그리고 다음 단계 추천 해줘.
```

---

## 9. 박사 작업 규칙 요약 (재확인)

memory에 저장돼 있어 자동 적용됨:
- **코드 수정 전 errors.md 읽기** (5초 체크리스트, ~80개 항목)
- **새 결함은 P-/B-/E-/M-/O-/U-/V-/T-/W-/S-/D-/A-/UX-/RD-/BG- 코드로 누적 기록**
- **상용화 우선** — NLLB(CC-BY-NC) / PyQt5(GPL) / 비상용 모델 금지
- **역할별 모듈 분리 완료** — 새 코드는 해당 계층 파일에. 의존 방향: platform < capture/ocr/translate < engine < ui
- **자체 비판 리뷰 후 보고** — 결함 발견 시 즉시 수정
- **GUI 검증 불가 명시** — 박사는 코드만 검증 가능. 사용자 GUI 검증 필요한 항목 명시 보고

---

## 10. 트러블슈팅 (자주 겪는 함정)

### "Tesseract not found"
- `C:\Program Files\Tesseract-OCR\tesseract.exe` 설치 + 환경변수 PATH 또는 `TESSERACT_CMD` env var.
- 인스톨러 빌드면 `C:\Program Files\Cocktail\tessdata\`에서 자동 인식.

### "PaddleOCR 초기화 실패"
- 정상. PaddleOCR는 optional. Tesseract로 자동 폴백됨.
- 강제 사용하려면 `pip install paddleocr paddlepaddle` + ocr_combo에서 PaddleOCRv5 선택.

### "smoke 테스트 실패 — method not found"
- BG-N 사이클로 메서드가 다른 클래스로 이동했을 수 있음.
- `python test_cocktail.py`가 통과하는지 확인 (소스 문자열이 아니라 실제 동작을 본다).
- 코루틴(`async def`)이면 `_method` 헬퍼가 `(FunctionDef, AsyncFunctionDef)` 둘 다 받는지 확인 (BG-5 자체 리뷰).

### "PyInstaller 빌드 실패 — torch include too large"
- `build.spec`의 `source.endswith(".lib")` / `torch\\include` 필터가 빠졌는지 확인 (R-2).
- 기본 빌드에선 PaddleOCR 제외 (`COCKTAIL_BUNDLE_PADDLE` 미설정).

### "릴리즈 워크플로우 — Tesseract 못 찾음"
- `.github/workflows/release.yml`에 "Install Tesseract OCR" 단계 있는지 확인 (R-1).
- 서명은 zip 만들기 전에 (sign_pos < zip_pos).

---

## 11. 한 줄 핸드오프 요약

> Phase 0 분리 사이클(BG-1~6) + 1~7차 품질 패스 + 8차 배포 준비 패스 완료. smoke 35건 / pre-release 17건 통과. 다음 단계는 사용자 GUI 실기 검증(D) 또는 코드 서명(C) 또는 RegionEditor(A) 또는 UIA 매트릭스(B) 중 하나 선택. `nllb_ct2/`는 2026-08-03 삭제 완료.

집에서 §8 멘트 복붙 → 끝.

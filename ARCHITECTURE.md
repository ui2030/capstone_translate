# Cocktail Architecture

Last updated: 2026-05-16 (BG-7 완료 시점 — RegionEditor 추가)

이 문서는 Cocktail Translator의 **현재 클래스 구조 / 시그널 흐름 / 데이터 흐름**과
**다음 분리 단계(BG-4~)의 목표 아키텍처**를 한 곳에 모은 설계 문서다.

코드 리팩토링(BG-N 사이클) 직전에 이 문서를 먼저 갱신한다. 작업 도중 책임 경계를 즉흥
결정하지 않기 위함이다.

---

## 1. As-Is — 현재 (BG-6 완료 시점)

**참고**: 아래 §1.1~§1.5는 BG-3 완료 시점의 분석이다. BG-4~BG-6이 끝난 시점의 To-Be(§2)가 실제 현재 구조를 반영한다. 분리 사이클 회고 자료로 보존.

### 1.1 클래스 의존 관계

```
                ┌─────────────────────────────────┐
                │    CocktailAppController        │  생명주기/연결만
                │  (BG-2)                         │
                └──────────────┬──────────────────┘
                               │ 생성
              ┌────────────────┴────────────────┐
              ▼                                 ▼
   ┌──────────────────────┐       ┌────────────────────────┐
   │  SettingsWindow   │◀──────│   OverlayWindow        │
   │  (settings panel)    │ ref   │   (transparent render) │
   │                      │       │   - paintEvent         │
   │  ▼ 보유              │       │   - mouse-through      │
   │  ┌────────────────┐  │       │   - 가상 데스크톱 전체 │
   │  │ CaptureRegion  │  │       └────────────────────────┘
   │  │ Controller     │  │
   │  │ (BG-3)         │  │
   │  └────────────────┘  │
   │                      │
   │  ▼ 사용              │
   │  - HybridTranslator  │       (모듈 글로벌 싱글톤)
   │  - PADDLE_OCR        │
   │  - UIA_PROVIDER      │
   │  - PERSIST_CACHE     │
   │  - LRU CACHE         │
   └──────────────────────┘
```

### 1.2 현재 `SettingsWindow`가 보유한 책임 (과적재)

`SettingsWindow`는 이름이 시사하는 것보다 훨씬 많은 일을 한다. 설정 패널이지만
백그라운드 번역기의 **거의 모든 상태와 루프**를 가지고 있다.

| 책임 카테고리 | 보유 항목 | 다음 분리 대상 |
|---|---|---|
| **Settings UI** | `initUI`, run_button, src/tgt/mode/ocr combo, status_label, capture_btn | `SettingsWindow` (유지 또는 분리) |
| **Capture region** | `region` (BG-3 분리 완료) | 그대로 유지, RegionEditor 추가 예정 |
| **Worker loop** | `translation_thread`, `run_asyncio_loop`, `capture_and_translate_async` | **`BackgroundController` (BG-4)** |
| **Translation state** | `translated_text`, `last_frame_hash`, `fail_streak`, `no_ocr_streak` | **`BackgroundController` (BG-4)** |
| **Model lifecycle** | `_preload_model`, `_model_ok`, `model_loaded` | **`BackgroundController` (BG-4)** |
| **Persistent cache timer** | `_cache_save_timer`, `_on_cache_save_tick` | **`BackgroundController` (BG-4)** |
| **Auto-pause flags** | `_capturable_paused`, `_sensitive_paused` | **`BackgroundController` (BG-4)** |
| **OCR routing** | `_src_hint`, `_ocr_langs`, `perform_ocr_with_boxes` (SL-1에서 `_smart_lang_for_image` / `_detect_script_via_osd` / `_script_cache` 퇴역) | **`BackgroundController` (BG-4)** |
| **UIA route** | `_uia_collect_and_translate`, `_check_sensitive_pause` | **`BackgroundController` (BG-4)** |
| **Drawing (legacy)** | `paintEvent`, `draw_translated_text`, `_fit_font`, `_lines_from_tess` | OverlayRenderer로 이미 위임됨, settings 잔재만 남음 |
| **Tray + hotkeys** | `_setup_tray`, `_register_global_hotkeys`, `tray`, `_autorun_action` | `SettingsWindow` 유지 (UI 진입점) |
| **Mode switching** | `_on_mode_combo_changed`, `set_capture_mode`, `toggle_mouse_through` | `SettingsWindow` 유지 (UI binding) |
| **Lifecycle UX** | `_maybe_show_initial_settings`, `_mark_settings_seen`, `_show_normal`, `_really_quit`, `closeEvent`, `_is_quitting` | `SettingsWindow` 유지 |

### 1.3 시그널/슬롯 흐름

```
SettingsWindow가 정의한 Signal:
  state_ready(list)         worker → 메인 → _apply_state → translated_text 갱신, OverlayWindow.update()
  failure_alert(str)        worker → 메인 → _on_failure → run_button 텍스트 변경
  model_ready()             _preload_model 끝 → _on_model_ready → run_button enable
  progress_msg(str)         preload 진행 → _on_progress → run_button 텍스트
  status_msg(str)           worker/UIA → _set_status → status_label
  global_toggle_signal()    keyboard 콜백 (별도 스레드) → _on_global_toggle → toggle_running
  global_show_signal()      keyboard 콜백 (별도 스레드) → _on_global_show → _show_normal
```

**문제**: 위 7개 시그널 중 **앞 5개는 백그라운드 엔진 → UI 통신**이다. UI 클래스가
엔진 시그널을 정의하는 구조라 BG-4에서 반드시 옮긴다.

뒤 2개(`global_*_signal`)는 **외부 입력 → 메인 스레드 진입용**이라 UI 측에 남아도 된다.

### 1.4 데이터 흐름 (capture loop, MODE_WINDOW 기준)

```
1. capture_and_translate_async (worker thread, asyncio loop)
   ├─ Auto-pause checks
   │   ├─ B-15: capturable=True → emit empty state, sleep 1s
   │   └─ S-1: sensitive window detected → emit empty state, sleep 1s
   │
   ├─ region.update_for_mode() ─────────────────────┐
   │                                                ▼
   │   CaptureRegionController                      │
   │     ├─ MODE_WINDOW → GetForegroundWindow rect  │
   │     ├─ MODE_HOVER  → QCursor.pos() rect        │
   │     └─ MODE_BOX    → 사용자 수동 영역 그대로   │
   │                                                │
   ├─ red_rect → 화면 절대 좌표 변환 (window+widget)│
   │   (※ 좌표계 의존: settings window geometry)    │
   │                                                ▼
   ├─ UX-2 Layer 2: primary screen으로 클립
   │
   ├─ Layer 1 라우팅
   │   ├─ UIA 가능 + ocr_combo 기본 → _uia_collect_and_translate → emit, continue
   │   └─ 그 외 → OCR 경로
   │
   ├─ ImageGrab.grab(bbox) → screenshot_np
   ├─ frame-skip hash 비교 (P-4)
   │
   ├─ OCR
   │   ├─ _ocr_langs() → auto면 eng+kor 기본 (SL-1, OSD 판별 퇴역)
   │   ├─ _zh_rescan_lang() → 근거 있으면 +chi_sim 재OCR, 8프레임 sticky (ZH-1)
   │   ├─ drop_pinyin_lines() → 한자 위 병음 줄 제외 (PY-1, 백엔드 공통)
   │   └─ perform_ocr_with_boxes → [(text, bbox, fsz), ...]
   │
   ├─ 언어 라우팅
   │   ├─ _src_hint() → "auto" 또는 ISO
   │   └─ assign_region_languages (SL-1: 스크립트 확정만, 라틴은 en)
   │
   ├─ batch_translate(src_texts, src_langs, tgt_lang)
   │   ├─ LRU CACHE
   │   ├─ PERSIST_CACHE (DPAPI)
   │   └─ HybridTranslator (opus-mt 전용쌍 → 피벗 zh→en→ko (M-9) → m2m100 폴백)
   │
   ├─ B-3: new_state 재구성 (누적 X), bbox는 화면 절대 좌표
   │
   └─ self.state_ready.emit(new_state) → 메인 스레드
                                       ↓
2. SettingsWindow._apply_state (메인 스레드 슬롯)
   ├─ self.translated_text = new_state
   ├─ self.update()                    (settings panel)
   └─ self.overlay.update()            ── OverlayWindow.paintEvent → _draw_one로 그림
```

### 1.5 좌표계 (BG-5 화면 절대 좌표 통일 + DP-1 물리 픽셀 통일)

**단위 규약 (DP-1, 2026-08-02)**: 앱의 모든 좌표는 **물리 픽셀 통일
(`QT_ENABLE_HIGHDPI_SCALING=0`)**. 모듈 최상단에서 `os.environ.setdefault`로 설정하며,
QApplication 생성(`CocktailAppController.__init__`)보다 반드시 앞서야 한다.
이로써 Qt 논리 좌표 == 물리 픽셀(devicePixelRatio 1)이 되어
`GetWindowRect` / `ImageGrab` / OCR bbox / `screens().geometry()` / `QCursor.pos()` /
`paintEvent`가 전부 같은 좌표계에서 만난다. DPI awareness(퍼모니터 인식) 자체는
Qt가 유지하므로 OS 블러 스트레칭은 없다.

| 데이터 | 좌표계 | 소유자 |
|---|---|---|
| `red_rect` | **화면 절대 좌표 / 물리 픽셀** (BG-5 + DP-1) | `CaptureRegionController` |
| `OCR bbox` (Tesseract 결과) | screenshot 내 좌표 (0,0=캡처 영역 좌상단), 물리 픽셀 | `_lines_from_tess` |
| `adjusted_bbox` (state에 저장) | 화면 절대 / 물리 픽셀 | worker → state |
| `QApplication.screens().geometry()` | 화면 절대 / 물리 픽셀 (DP-1) | Qt |
| `OverlayWindow.paintEvent` 그릴 때 | 화면 절대 → overlay-내 변환 (`rect.x() - wx` 한 번만) | OverlayWindow |

**완료**: `red_rect`가 절대 좌표라 settings window 이동/숨김에 영향받지 않는다.
worker는 `region.red_rect`를 그대로 읽으며, OverlayWindow는 절대 → overlay-window-내
변환 한 번만 수행한다. `OverlayWindow._fit_to_screen` / `RegionEditor._fit_to_virtual_desktop`의
`screens()` 합집합도 DP-1 이후 물리 픽셀 rect라 별도 변환 없이 정합한다.

**트레이드오프**: 스케일링 OFF라 설정 창(880x120)과 트레이 메뉴는 고배율 모니터로
옮기면 물리 픽셀 그대로 작게 보인다. 오버레이 좌표 정합을 우선한 의도된 선택.

### 1.6 스레드 경계와 화면 rect 캐시 (GT-1 / MS-1, 7차)

Qt의 화면·커서 API는 **GUI 스레드 전용**인데 워커 루프가 `region.update_for_mode()`와
W-4 클립에서 그것들을 직접 불렀다(잠재 크래시). 7차에서 경계를 명시했다.

| 데이터 | 누가 쓰나 (write) | 누가 읽나 (read) |
|---|---|---|
| `CaptureRegionController.screen_rects` / `primary_rect` | **메인 스레드만** — 생성 시 + `screenAdded`/`screenRemoved`/`geometryChanged` | 워커(`screen_rect_at`, `virtual_desktop_rect`), OverlayWindow(CL-1 클립) |
| 커서 좌표 | — | 워커 (`cursor_pos_physical()` = win32 `GetCursorPos`, 물리 픽셀) |
| `BackgroundController._last_emitted` | **워커만** (RC-1) | 워커만 |
| `BackgroundController.translated_text` | 메인 스레드 슬롯 (`_apply_state_to_self`) | OverlayWindow.paintEvent |

- `screen_rects` 튜플은 `(left, top, right_exclusive, bottom_exclusive)`. Qt의 `right()/bottom()`
  = "마지막 픽셀" 규약은 **캐시를 만들 때 한 번만** 흡수한다 → 호출부에서 `+1` 금지.
- 화면 변경 시그널의 연결/전파는 `CocktailAppController._wire_screen_changes()`가 소유한다
  (§2.1 "생성/연결은 앱 컨트롤러" 규약). 전파 대상: region 캐시 → OverlayWindow `_fit_to_screen`
  → 열려 있는 RegionEditor `_fit_to_virtual_desktop`.
- 워커는 메인 스레드가 갱신하는 상태를 **읽지도 않는다**(RC-1). 오버레이 emit은
  `_emit_state()` 단일 경로이며 워커 로컬 사본을 같은 자리에서 갱신한다.

---

## 2. To-Be — 타깃 아키텍처 (BG-4~ 종료 시점)

### 2.1 클래스 책임 경계

```
              ┌──────────────────────────────────┐
              │   CocktailAppController          │
              │   - QApplication 소유            │
              │   - setQuitOnLastWindowClosed    │
              │   - 4개 컴포넌트 생성/연결       │
              └──────────────┬───────────────────┘
                             │ 생성/주입
       ┌────────────┬────────┴────────┬────────────────┐
       ▼            ▼                 ▼                ▼
 ┌──────────┐ ┌─────────────┐  ┌──────────────┐ ┌──────────────┐
 │ Settings │ │ Background  │  │ CaptureRegion│ │  Overlay     │
 │ Window   │ │ Controller  │  │ Controller   │ │  Renderer    │
 │ (UI 만)  │ │ (엔진 만)   │  │ (BG-3 유지)  │ │ (그리기 만)  │
 │          │ │             │  │              │ │              │
 │ - tray   │ │ - worker    │  │ - capture_   │ │ - paintEvent │
 │ - combo  │ │   thread    │  │   mode       │ │ - 가상 데스크│
 │ - run_btn│ │ - signals   │  │ - red_rect   │ │   탑 영역    │
 │ - mode   │ │ - state     │  │ - update_for │ │ - 마우스 통과│
 │   toggle │ │ - cache     │  │   _mode      │ │              │
 │ - global │ │   timer     │  │              │ │              │
 │   hotkey │ │ - OCR/UIA   │  │              │ │              │
 │          │ │ - auto      │  │              │ │              │
 │          │ │   pause     │  │              │ │              │
 └────┬─────┘ └──────┬──────┘  └──────────────┘ └──────────────┘
      │              │
      └──────signal/slot─────┐
             control commands │
             (start/stop/mode)│
                              │
             worker → state ──┘
             (state_ready, status, failure, model_ready)
```

(주: `RegionEditor`는 BG-3에서는 추가하지 않는다. `CaptureRegionController`가 이미
상태를 갖고 있고, settings panel resize 잔재만 제거됐다. 추후 영역 편집 UI가 필요할 때
별도 사이클에서 `RegionEditor` 위젯을 추가해 같은 controller를 조작한다.)

### 2.2 `BackgroundController` 책임 (BG-4 신설)

**소유 상태**
- worker thread (`run_asyncio_loop`)
- `translated_text`, `last_frame_hash`, `fail_streak`, `no_ocr_streak`
- `model_loaded`, `_model_ok`
- `_capturable_paused`, `_sensitive_paused`
- `_script_cache` (UX-2 — SL-1에서 제거됨)
- `_cache_save_timer`
- `_last_emitted` (RC-1, 7차) — 워커가 마지막으로 emit한 상태의 **워커 로컬 사본**.
  메인 스레드가 갱신하는 `translated_text`를 워커 판정에 쓰지 않기 위한 것.
- `_ocr_line_confs` (DG-1, 7차) — 직전 OCR의 라인별 평균 confidence(표시 게이트용 병렬 리스트)

**소유 시그널 (UI로 emit)**
- `state_ready(list)` — 화면 절대 좌표 bbox 포함 새 상태
- `failure_alert(str)`
- `model_ready()`
- `progress_msg(str)`
- `status_msg(str)`

**제공 메서드 (UI에서 호출)**
- `start()` — 모델 preload + worker 시작
- `stop()` — `stop_thread=True` + cache flush
- `set_running(bool)` — `model_loaded` 토글 (현재 toggle_running)
- `set_capture_state(bool)` — `capturable` 토글
- `request_state_clear()` — 모드 변경 시 `last_frame_hash=None`

**주입 의존**
- `region: CaptureRegionController` — 캡처 영역 읽기 전용 접근
- `settings: SettingsWindow` (또는 더 좁은 인터페이스) — `src_combo`, `tgt_combo`,
  `ocr_combo` 현재 값 read. 박사 권장: 직접 widget 참조 금지, 가벼운 `RuntimeOptions`
  객체로 좁혀 받기

### 2.3 `SettingsWindow`(=현 `SettingsWindow` 슬림 버전) 책임

**남기는 것**
- 위젯/레이아웃 (initUI)
- 트레이 + 글로벌 단축키 (UI 진입점)
- 콤보 변경 → controller에 명령 전달
- controller signal → status_label / run_button 텍스트 갱신
- 자기 자신 capture affinity 토글

**버리는 것**
- 모든 worker / OCR / 번역 / 캐시 관련 메서드
- `state_ready` 등 엔진 시그널 정의
- `translated_text` 등 번역 상태

### 2.4 시그널 흐름 (To-Be)

```
SettingsWindow              BackgroundController         OverlayRenderer
   │                              │                          │
   ├── start() ─────────────────▶│                          │
   │                              │                          │
   │                              ├── _preload_model         │
   │                              │     (background thread)  │
   │                              │                          │
   │◀── model_ready ──────────────┤                          │
   │   (run_button enable)        │                          │
   │                              │                          │
   │── set_running(True) ────────▶│                          │
   │                              │                          │
   │                              ├── worker loop ──┐        │
   │                              │                 │        │
   │◀── status_msg ───────────────┤   capture/OCR   │        │
   │   (status_label)             │   /translate    │        │
   │                              │                 ▼        │
   │                              ├── state_ready ──────────▶│
   │                              │                          ├── update()
   │                              │                          │   paintEvent
   │                              │                          │
   │◀── failure_alert ────────────┤                          │
   │   (run_button text)          │                          │
```

---

## 3. 마이그레이션 계획

### BG-4: `BackgroundController` 추출 (✅ 완료 — 2026-05-04)

**완료 내역**
1. `BackgroundController(QObject)` 신설.
2. 5개 엔진 시그널 이동: `state_ready`, `failure_alert`, `model_ready`, `progress_msg`, `status_msg`.
3. `run_asyncio_loop`, `capture_and_translate_async`, `_uia_collect_and_translate`,
   `_check_sensitive_pause`, `_preload_model`, `_on_cache_save_tick` 이동.
4. `_smart_lang_for_image`, `_detect_script_via_osd`, `_ocr_langs`, `perform_ocr_with_boxes`,
   `_lines_from_tess`, `_src_hint`, `_ocr_backend_name` 이동.
5. SettingsWindow는 `self.controller = BackgroundController(region, options_provider, is_self_hwnd)`로
   소유. 시그널 5개 connect를 SettingsWindow에서 수행.
6. `translated_text`, `model_loaded`, `capturable`은 SettingsWindow proxy property로 유지
   (OverlayWindow.paintEvent / closeEvent 회귀 위험 0).
7. `RuntimeOptions(@dataclass(frozen=True))`로 콤보 immutable 스냅샷 → 워커.
8. `is_self_hwnd: Callable[[int], bool]` 콜백으로 settings/overlay hwnd를 BC가 검사.
9. `closeEvent` 단순화 — `self.controller.stop()`이 stop_thread + thread join + PERSIST_CACHE.save() 통합.

**테스트**
- `smoke_tests.py`: `test_background_controller_owns_engine` 신규 + 기존 5건 갱신.
- `pre_release_check.py`: `check_background_controller` 신규.

**박사 자체 리뷰 결함 (즉시 수정 1건)**
- forward-ref 결함: `_snapshot_options`의 `-> RuntimeOptions` 어노테이션이 클래스 정의 시점에
  평가되어 NameError. RuntimeOptions는 같은 파일 후반에 정의됨 → 문자열 어노테이션 `-> "RuntimeOptions"`로 처리.

**남은 부채 (BG-5로 이월)**
- 워커가 `self.region.owner.geometry()`로 settings 창 좌표에 의존. red_rect가 위젯 좌표라서.
  BG-5에서 절대 좌표로 통일하면 owner 의존 제거 가능.

### BG-5: 좌표계 통일 (red_rect 절대 좌표화) (✅ 완료 — 2026-05-04)

**완료 내역**
- `CaptureRegionController.red_rect` 기본값이 `QRect(240, 215, 600, 375)` 절대 좌표.
- `update_for_mode`에서 `owner.geometry()` 변환 제거. 절대 좌표를 그대로 저장.
- `BackgroundController.capture_and_translate_async`에서 `region.owner.geometry()` 4줄 변환 제거.
  `x1 = int(rr.x())` 직접 사용으로 단순화.
- `OverlayWindow.paintEvent` MODE_BOX 박스: `ctl.geometry().x()` 의존 제거. 절대→overlay 변환 한 번만.
- `SettingsWindow.initUI` size_grip의 widget-coord 위치 계산 코드 제거(grip은 hide 유지).

**테스트**
- `smoke_tests.py`: `test_red_rect_absolute_coordinates` 신규.
- `pre_release_check.py`: `check_red_rect_absolute` 신규.
- `_method` 헬퍼 보강: `AsyncFunctionDef` 처리 추가 (코루틴 검사 시 필요).

**박사 자체 리뷰 결함 (즉시 수정 1건)**
- smoke `_method` 헬퍼가 `async def`(AsyncFunctionDef)를 처리하지 못해 `capture_and_translate_async`
  검사 시 method-not-found. `(FunctionDef, AsyncFunctionDef)` 둘 다 받도록 수정.

**UX 영향**
- MODE_BOX에서 settings window 이동이 더 이상 빨간 박스를 끌고 가지 않는다 (절대 좌표 의도에 부합).
- 영구 영역 편집은 BG-6+ `RegionEditor` 사이클에서 추가.

### BG-6: `SettingsWindow` 리네이밍 + region 콜백 통합 + dead code 정리 (✅ 완료 — 2026-05-04)

**완료 내역**
- `TransparentWindow` → `SettingsWindow` 클래스/메서드 시그니처/문서 일괄 리네이밍 (12 files updated).
- `CaptureRegionController(owner)` → `CaptureRegionController(is_self_hwnd)` — UI 클래스 내부 구조 의존 제거.
  `BackgroundController`와 같은 콜백을 공유.
- `SettingsWindow.paintEvent` 빈 override 제거 — Qt 기본이 위젯 트리 자동 처리.
- `_on_state_ready`에서 `self.update()` 호출 제거 — settings window는 그릴 게 없음.
- `OverlayWindow` paintEvent helper(`_fit_font`, `_draw_one`)는 그대로 유지 — 자기 책임 내 정리됨.

**테스트**
- `smoke_tests.py`: `test_bg6_settings_rename_and_region_callback` 신규 + 기존 2건 갱신.
- `pre_release_check.py`: `check_bg6_rename_and_region_callback` 신규.

**Phase 0 분리 사이클 종결**
이로써 BG-1~BG-6의 백그라운드 분리 사이클이 끝난다. 다음 단계는:
- 영역 편집 UX (`RegionEditor` 위젯) — MODE_BOX UX 보강. (✅ BG-7에서 완료)
- Phase 1+ 비전 추가 (UIA 매트릭스, 글로벌 단축키 강화 등).

### BG-7: `RegionEditor` 위젯 추가 (✅ 완료 — 2026-05-16)

**문제**: BG-5에서 `red_rect`가 settings window 위젯 좌표 → 화면 절대 좌표로 전환되며,
"settings 창 리사이즈/이동으로 박스 조정" 기존 UX가 사라졌다. MODE_BOX는 절대 좌표
박스를 사용하지만 사용자가 마우스로 박스를 직접 편집할 UI가 없었다 — 기능적 부채.

**완료 내역**
- `RegionEditor(QMainWindow)` 신설. 가상 데스크톱 전체를 덮는 frameless + translucent +
  Qt.Tool + WA_TranslucentBackground + always-on-top + capture-excluded(B-10) 창.
- 영역 박스 윤곽 + 8개 핸들(네 모서리 + 네 변 중앙)을 paintEvent로 그림.
- 핸들 hit-test → 좌클릭 드래그로 resize, 박스 본체 좌클릭 드래그로 move.
- Esc 또는 우클릭/"취소" → 변경 사항 폐기. Enter 또는 "저장" → `region.red_rect`에 절대 좌표
  반영 + `set_capture_mode(MODE_BOX)`로 모드 전환 + `controller.reset_frame_hash()` 호출.
- 진입점 2개: SettingsWindow에 "영역 편집..." 버튼(`edit_region_btn`) + 트레이 메뉴 동명 항목.
- `_is_self_hwnd` 콜백이 RegionEditor의 winId도 자기 hwnd로 인식 → 워커가 편집 창 안의
  텍스트를 캡처/번역 시도하지 않음 (B-10 / 자기 캡처 피드백 가드).
- 안전 원칙 준수: 편집 중에도 BackgroundController의 민감 창 감지(S-1) 및 capturable
  guard(B-15)는 그대로 동작 — RegionEditor는 geometry만 바꾸고 번역 파이프라인에는 손대지 않음.
- Lazy 생성: 사용자가 "영역 편집..."을 처음 누를 때 인스턴스 생성. 닫혀도 인스턴스는 보존
  (재오픈 시 현재 `region.red_rect`로 초기화).

**좌표계**
- RegionEditor 자체는 가상 데스크톱 절대 좌표 (OverlayWindow와 동일 geometry).
- 내부 박스 상태는 항상 절대 좌표 `QRect`. paintEvent에서만 `rect.x() - wx` 한 번 변환.
- 저장 시 absolute QRect 그대로 `region.red_rect`에 대입.

**테스트**
- `smoke_tests.py`: `test_bg7_region_editor` 신규 (클래스 존재, capture-excluded 적용,
  is_self_hwnd 콜백 RegionEditor 포함, commit이 `region.red_rect` 대입 + MODE_BOX 전환 호출).
- `pre_release_check.py`: `check_bg7_region_editor` 신규 (코드 토큰 + 트레이 메뉴 항목).

---

## 4. 좌표계 결정 (BG-5에서 적용 완료)

**결정**: `red_rect`는 **화면 절대 좌표**로 통일 (✅ BG-5).

**근거 (적용됨)**
- OCR bbox 결과는 screenshot 내 상대 좌표 → 절대 좌표 변환은 worker 안에서 1회만.
- OverlayWindow는 가상 데스크톱 → 절대 → overlay-내 변환 한 번만 수행.
- settings window 이동/숨김이 캡처 영역에 영향 없음 (백그라운드 번역기 UX에 부합).
- `update_for_mode`가 `owner.geometry()` 의존 제거 → 기능적 owner 의존은 `foreground_window_rect`의
  자기 hwnd 가드(self.owner.winId() / overlay.winId())만 남음.

**BG-7에서 채택**
- RegionEditor 자체가 가상 데스크톱 절대 좌표 영역에 떠 있는 frameless 위젯이라
  내부 박스 상태도 처음부터 절대 좌표 `QRect`로 유지. paintEvent의 absolute → editor-내
  변환(`rect.x() - wx`)이 OverlayWindow와 동일 패턴.
- 저장 시점에 추가 변환 없이 `region.red_rect = QRect(...)` 그대로 대입.
  worker / OverlayWindow는 즉시 새 박스를 반영.

---

## 5. 비결정 사항 (다음 사이클에서 결정)

**BG-4에서 결정된 사항 (✅)**
- `BackgroundController`는 `QObject` 상속. 시그널/스레드 affinity 명확.
- `RuntimeOptions`는 `@dataclass(frozen=True)`. 워커가 매 iteration immutable 스냅샷 읽음.
- `_script_cache` TTL 30s 그대로 유지.
- `keyboard` 글로벌 단축키 콜백은 SettingsWindow에서 받아서 `controller.set_running()` 호출.

**BG-5에서 결정된 사항 (✅)**
- `update_for_mode`는 owner 참조 없이 동작 (절대 좌표 직접 저장).
- `region.owner`는 `foreground_window_rect`의 자기 hwnd 가드(winId/overlay)에만 남아 있음.
  BG-6에서 SettingsWindow 분리 시 `is_self_hwnd` 콜백으로 통합 검토.
- MODE_BOX 영역 편집 UX는 BG-6+ RegionEditor 사이클로 이월. 현재 size_grip은 hide 유지.

**BG-6에서 결정된 사항 (✅)**
- `TransparentWindow` → `SettingsWindow` 즉시 리네이밍 완료.
- `region.owner` 의존을 `is_self_hwnd` 콜백으로 통합 (BackgroundController와 같은 콜백 공유).
- `OverlayWindow` paintEvent helper는 그대로 유지 — paintEvent 본연의 책임이라 외부화 불필요.

**BG-7에서 결정된 사항 (✅)**
- RegionEditor는 별도 최상위 윈도우(`QMainWindow`). overlay와 별개로 떠야 사용자 마우스
  이벤트를 받을 수 있다 (overlay는 마우스 통과 영구 ON).
- 저장 전엔 `region.red_rect`를 건드리지 않는다 (편집 취소 시 원본 박스 보존).
- 모드 자동 전환: 저장 시 `set_capture_mode(MODE_BOX)`. 사용자가 "영역을 편집했다"는 의도는
  곧 "이 박스를 사용하고 싶다"이므로 자동으로 MODE_BOX로 보낸다.
- RegionEditor의 winId를 `_is_self_hwnd` 콜백에 포함시킨다 — 편집 중에도 worker가 활성
  창 모드 / 민감 창 가드를 정상 동작시키되, 편집 창 자체는 캡처/번역하지 않는다.

**다음 사이클 후보**
- UIA 결과 매트릭스 자동 수집 (Phase 2 PoC 마무리).
- 코드 서명 인증서 발급/적용.
- Phase 1+ 비전 작업 (`VISION_AND_ROADMAP.md` 참고).

---

## 6. 박사 자체 비판 리뷰 (이 문서 자체)

- 이 문서는 As-Is 클래스 다이어그램까지만 ASCII로 적었다. 실제 시그널 connection
  graph는 본문에 풀어쓴 정도. BG-4 구현 후 to-be 다이어그램과 실제 코드가 어긋나면
  먼저 이 문서를 고치고 나서 코드를 고친다.
- BG-5 좌표계 결정을 미리 박은 이유: BG-4 진행 중 `controller.red_rect`를 어떤 좌표계로
  받을지 즉흥 결정하면 BG-5에서 또 옮기게 된다. **결정을 한 단계 앞당겨 BG-4가
  미래 좌표계 기준으로 작성되도록** 한다.
- `SettingsWindow`/`OverlayRenderer` 리네이밍은 BG-6로 미뤘다. 이름 변경은 import 경로,
  smoke 테스트, pre_release_check, build.spec, installer.iss 검사 토큰까지 다 흔들어서
  내용 분리가 끝난 뒤 마지막 단계에서 한 번에 처리하는 게 안전.

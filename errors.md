# 에러 & 수정 로그 (반드시 코드 수정 전에 한 번 읽기)

이 파일은 "같은 실수를 두 번 하지 않기 위한" 체크리스트다.
새 수정에 들어가기 전에 ⬇ 의 항목을 훑어 보고, 위반하지 않는지 확인한다.
새로 발견한 버그/병목은 아래 형식으로 누적해서 기록한다.

---

## 0. 코드를 만지기 전 5초 체크리스트

- [ ] 모델 추론 코드에 `torch.inference_mode()` 또는 `torch.no_grad()`가 있는가?
- [ ] CUDA 사용 가능 시 모델이 `half()` (fp16) 로 떠 있는가?
- [ ] 번역이 라인별 루프가 아니라 **배치 1회 generate** 인가?
- [ ] OCR 루프가 화면 변화 없을 때도 돌고 있지 않은가? (frame-skip 해시 확인)
- [ ] 프레임 해시 격자를 **고정 크기**로 잡고 있지 않은가? (셀 픽셀 수를 고정해야 해상도 불변) — FD-1
- [ ] 변화 감지 임계를 정할 때 **실제 글자 크기 12~60px**로 실측했는가? (60px만 되면 자막은 못 잡는다) — FD-1
- [ ] `ImageGrab.grab(all_screens=True)`는 bbox와 무관하게 가상 데스크톱 전체를 찍는다 — 주 모니터 안이면 False인가? — CP-1
- [ ] 워커의 대기가 `time.sleep(<상수>)` 고정이 아니라 `poll_interval()`인가? 고정이면 캡처가 빨라져도 체감 지연은 그대로다 — PL-1
- [ ] 폴 간격을 내렸다면 **조용할 때는 예전 값으로 물러나는가**? (유휴 CPU 24%→8%를 지킨 것이 `POLL_IDLE_S=0.25`다) — PL-1
- [ ] 속도를 '한 사이클 시간'으로 재고 있지 않은가? 사용자 기준은 **화면 변화 → 자막**이고 둘은 최대 3배 벌어진다 — BM-1
- [ ] 한 사이클 시간이 화면이 바뀌는 주기보다 길면 지연이 **누적**된다. p95를 볼 때 큐잉인지 확인했는가? — QU-1
- [ ] 캡처 경로를 건드렸으면 ①주 ②보조 ③걸침 ④음수 좌표 4케이스를 **캡처 이미지로** 확인했는가? (W-4/W-5 회귀) — CP-1
- [ ] 사용자 화면에서 온 무언가를 **디스크에 남기는가**? 그렇다면 ①지우는 UI ②만료 ③끄는 옵션이 셋 다 있는가? 하나라도 없으면 그건 영구 기록이다 — PV-1
- [ ] 캐시 축출이 **LRU**인가? "앞쪽 절반 삭제"는 자주 쓰는 항목을 임의로 날린다 — PV-1
- [ ] DPAPI 암호화를 "안전하다"의 근거로 쓰고 있지 않은가? `CurrentUser` 범위는 **같은 계정의 아무 프로세스나** 푼다 — PV-1
- [ ] `print`/로그에 화면 원문을 찍고 있지 않은가? **사유는 남기고 텍스트는 길이+해시**로 (무음 실패로는 돌아가지 않는다) — PV-2
- [ ] 전역 키 후킹처럼 권한이 큰 기능이 **기본 켜짐**인가? 프라이버시를 내세우는 앱에서 그건 opt-in 이어야 한다 — PV-3
- [ ] 기본값을 끄로 바꿨으면 그 사실을 **첫 실행 안내나 설정 문구**로 알렸는가? 말없이 끄면 "고장났다"로 읽힌다 — PV-3
- [ ] 문서(README/VISION)가 약속한 프라이버시 문구가 **코드와 일치**하는가? 지키지 못하는 약속이 제일 나쁘다 — PV-1
- [ ] 중복 텍스트 캐시(LRU)에서 먼저 lookup 하는가?
- [ ] 핫 패스(paintEvent, OCR 루프, generate 루프)에 `print()`가 있지 않은가?
- [ ] 예외 발생 시 `stop_thread=True`로 전체 루프를 죽이지 않는가? (continue가 정답)
- [ ] 배치/그룹 루프의 예외가 **그 프레임 전체**를 날리지 않는가? (그룹 격리 + 실패 로그) — TE-1
- [ ] 모델을 `.to(device)` 후에 `.half()` 하고 있지 않은가? (로드 시점 fp16 = VRAM 피크 절반) — VR-1
- [ ] OCR 비용 예산을 **픽셀 수**로 잡고 있지 않은가? (비용은 면적이 아니라 글자 양이다) — UB-1
- [ ] 언어팩을 뺄 때 **그 문자가 쓰인 화면**에서 오독이 표시되는지 봤는가? (안 읽히는 것보다 자신 있게 틀리는 게 나쁘다) — LC-1b
- [ ] 중복 dedup 시 OCR 원문 vs OCR 원문을 비교하는가? (번역문과 비교 X)
- [ ] 언어 코드를 만질 때 `LANG_MAP` + `_LANGDETECT_TO_ISO` + `M2M100_LANGS` + 콤보박스 4곳을 동시에 확인했는가?
- [ ] "이 traineddata 가 이 문자를 읽는다"를 추측하지 않고 실측했는가? (`kor`은 한자를 **안** 읽는다) — ZH-1
- [ ] OCR 언어팩을 추가할 때 **다른 언어 화면의 OCR 시간**을 재고 조건부로 켰는가? (상시 추가 = +38~68%) — ZH-1
- [ ] 새 라인 필터 임계를 **반대편 코퍼스**(영어 6909줄 등)로 거짓양성 세어 정했는가? — PY-1
- [ ] 분기 조건에 `time.perf_counter()`(실측 시간)를 넣지 않았는가? 같은 입력이 실행마다 달라진다 — RP-1
- [ ] 비싼 재시도의 진입 조건이 **한 종류의 신호**에만 걸려 있지 않은가? — ZH-2
- [ ] 비싼 재시도(확대 재OCR)의 **진입 조건**이 1차 패스가 만든 값(라인 높이·라인 수)에만
      걸려 있지 않은가? 1차가 무너지면 그 값도 같이 오염돼 **가장 필요할 때만 꺼진다** — OU-2
- [ ] 1차가 **0줄**일 때 재시도가 `if lines:` 류 가드에 막히지 않는가? 0줄이 최악의 경우다 — OU-2
- [ ] 확대 보간을 고를 때 **링잉(음수 로브)**을 의심했는가? 특정 배율에서만 0줄이 나오면 배율이
      아니라 커널 문제다 — OU-3
- [ ] 두 OCR 패스 중 하나를 고르는 판정이 **한 종류의 신호**(평균 conf)에만 걸려 있지 않은가?
      conf 는 줄을 버리면 오히려 오른다 — UA-2
- [ ] 문단 병합이 **읽기 순서를 보존**하는가? 인식률 100%여도 순서가 틀리면 오역으로 보인다 — PM-2
- [ ] 표시 게이트에 **원문을 보는 규칙**이 하나라도 있는가? 번역문만 보면 그럴듯하게 깨진 원문을
      절대 못 잡는다 — DG-1c
- [ ] 번역 붕괴를 "쓰레기 입력 탓"으로 짐작하지 않았는가? 반복 붕괴는 **깨끗한 입력**에서 나는
      greedy 탐색 실패다(오독 주입 600문장에서 0건) — RP-2
- [ ] 반복 검출을 **단어** n-gram으로 하려 하는가? 붕괴는 조사·어미를 바꿔 반복해서 단어로는
      안 잡힌다. 문자 n-gram + **원문 자신의 반복률을 뺀** 값이어야 한다 — RP-2
- [ ] 전면 `num_beams>1`을 켜기 전에 기준 4를 재봤는가? 배치 12문장 +78%다 — RP-2
- [ ] 라틴 단어를 셀 때 `user1`·`mp3` 같은 **식별자**를 단어로 세고 있지 않은가?
      `(?![0-9])` 부정 전방탐색은 되짚기 때문에 소용없다 — CO-3
- [ ] 채점 지표가 "형태"만 보고 있지 않은가? 미번역·글자비율·길이는 붕괴/누락을 전부 놓친다 — BM-2
- [ ] 게이트 판정에 OCR 라인 높이(`line_px`)를 쓰지 않는가? 1차 오독에 오염된다 — DG-1c
- [ ] 병음 판정은 붙여쓰기·띄어쓰기 **둘 다** 상정했는가? — PY-2
- [ ] 게이트 임계를 올리기 전에 "이 임계가 실제로 무엇을 잡았는가"를 시료 전체에서 세어 봤는가? (쓰레기 0건 / 정상만 잡는 임계가 있다) — DG-1b
- [ ] 라인 분할을 가로 간격으로 하려 한다면 **양끝맞춤 본문**에서 재봤는가? (메뉴 간격 1.83배 = justified 위험 구간) — LM-1c
- [ ] 한자/가나 word 를 이을 때 `join_ocr_words()`를 쓰는가? (`" ".join` 은 문장을 바꾼다) — ZH-1
- [ ] auto 라인 언어는 `identify_language()`의 Unicode script 판정 **만** 쓰는가? 라틴/불명을 추측(langdetect·region 투표·지배 언어 스냅)하지 말 것 — SL-1 (M-6/M-7/LG-1 대체)
- [ ] 언어 추측 장치를 되살리려면 errors.md SL-1의 '복원 조건'을 먼저 확인했는가? — SL-1
- [ ] CT2 hypothesis에서 첫 토큰(언어 토큰)을 잘라냈는가?
- [ ] PIL ImageGrab 결과를 cv2에 넘길 때 `cv2.COLOR_RGB2GRAY` (BGR 아님) 인가?
- [ ] paintEvent에서 bbox를 red_rect에 클립할 때 `QRect.intersected()`를 쓰는가? (left만 보정 후 원본 width로 그리지 말 것)
- [ ] `Qt.AA_EnableHighDpiScaling=True`이면 위젯 좌표에 devicePixelRatio를 또 곱하지 말 것
- [ ] Windows에선 `SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE)`로 자기 윈도우를 캡처에서 제외했는가? (오버레이 → OCR 피드백 루프 차단)
- [ ] `self.translated_text` 갱신은 메인 스레드(시그널/슬롯)에서만 하는가? (워커 스레드 직접 할당 금지)
- [ ] 연속 실패 시 워커 break 대신 backoff continue 인가? (자동 복구)
- [ ] 번역 박스의 alpha가 충분히 높은가? (영문이 비치지 않게 alpha ≥ 230 권장)
- [ ] 한국어가 영문보다 길어 박스를 벗어날 수 있는가? `QFontMetrics.boundingRect`로 height 동적 확장 했는가?
- [ ] height가 너무 폭증할 때 폰트 자동 축소(initial→9px) fallback이 적용됐는가? (B-13)
- [ ] 캡처 차단(B-10)을 사용자가 일시적으로 풀 수단(버튼/단축키)을 제공했는가? (E-9 — 스크린샷 가능성)
- [ ] 추가/교체하는 모델/라이브러리의 라이선스가 상용 호환인가? (NLLB CC-BY-NC, PyQt5 GPL 금지) — M-5/E-10
- [ ] 모델은 lazy load 인가? (import 시점에 동기 로드 금지, UI는 즉시 떠야 함) — P-9
- [ ] **창이 뜨는 경로에 `import torch`/`import transformers`가 없는가?** lazy load라고 적어 놓고 최상단 import를 남기면 시작이 10초가 된다 — LZ-1
- [ ] 진단 한 줄(`environment_summary`)이 무거운 모듈을 끌고 오지 않는가? 창 뜨기 전 메인 스레드에서 돈다 — LZ-1
- [ ] 지연 초기화한 모듈 전역은 **완전히 채운 뒤 마지막에** 대입했는가? (다른 스레드가 `is not None`으로 판정한다) — LZ-1
- [ ] 배포물이 Tesseract·기본 모델을 **동봉**하는가? 사용자가 따로 설치할 게 하나라도 있으면 "준비 5분"이 아니다 — PK-1/PK-2
- [ ] 앱 폴더 탐색에 `sys._MEIPASS`도 넣었는가? PyInstaller onedir는 datas를 `_internal/`에 넣는다 — PK-1
- [ ] 외부 프로그램(tesseract 등)을 동봉했으면 **그 DLL이 `_internal` 루트로도 복사되지 않았는지** 확인했는가? 같은 이름의 시스템 DLL(OpenSSL 등)을 덮어써 파이썬 import가 죽는다 — PK-3
- [ ] 재빌드했으면 `dist/.../Cocktail.exe --self-test`를 **실제로 돌렸는가**? 빌드 성공 ≠ 동작 — PK-3
- [ ] ctypes로 Win32를 부를 때 `argtypes/restype`를 명시했는가? 리소스 타입/이름처럼 **정수 ID일 수 있는 인자**는 `c_void_p`로 받는가? — IC-2 / D-1
- [ ] 등록된 언어쌍은 opus-mt(전용), 그 외는 m2m100(폴백)으로 라우팅 되는가? — M-5
- [ ] opus-mt sepvoc 모델의 **입력**은 `source.spm`으로 인코딩하는가? (AutoTokenizer는 타깃 vocab으로 인코딩해 `<unk>` 범벅) — M-8
- [ ] 시그널/슬롯 데코레이터가 PySide6 Signal/Slot인가? (PyQt5 pyqtSignal/pyqtSlot 흔적 없음) — E-10
- [ ] OCR 언어가 하드코딩(`"eng+kor"`)이 아니라 사용자 src 콤보에 따라 동적 결정되는가? — B-14
- [ ] 사용 가능한 traineddata를 시작 시 진단(`pytesseract.get_languages()`)했는가? — B-14
- [ ] 배포 빌드(.exe)에서 `./tessdata/` 가 있으면 `TESSDATA_PREFIX` 자동 설정되는가? — 인스톨러 호환
- [ ] PaddleOCRv5는 optional lazy import인가? 설치/초기화 실패 시 Tesseract 폴백이 있는가? — O-1
- [ ] 일반 사용자 환경 문제가 앱 시작 예외가 아니라 UI 상태 라벨/진단 스크립트로 드러나는가? — U-1
- [ ] 기본 캡처 모드는 비전 V-1에 맞춰 `MODE_WINDOW`(활성 창)인가? — UX-1
- [ ] auto OCR 언어가 `eng+kor` 고정인가? (`OCR_LANGS_AUTO_DEFAULT`, `COCKTAIL_OCR_LANGS_AUTO` 오버라이드 가능) — UX-1 / SL-1
- [ ] OverlayWindow 빨간 박스는 `MODE_BOX`일 때만 그리는가? — UX-1
- [ ] `self.capturable=True`(스샷OK) + 번역 동작 동시 시 자동 일시 중지 + 사용자 경고 표시되는가? — B-15
- [ ] mode_combo 표시와 `self.capture_mode` 초기값이 일치하는가? (init에서 명시 동기화) — UX-1 자체 리뷰
- [ ] auto OCR 경로에 OSD(`pytesseract.image_to_osd`) 호출이 **없는가**? (script 오판 → 쓰레기 OCR) — SL-1 (UX-2 Layer 3-5 퇴역)
- [ ] 활성 창 모드 + UIA 사용 가능 + ocr_combo 기본값일 때 UIA 우선 시도, 실패 시 OCR 폴백 자동 라우팅 작동하는가? — UX-2
- [ ] OCR 영역이 화면 가시 영역(primary screen)으로 클립되는가? — UX-2
- [ ] mutable 클래스 변수 X, 인스턴스 변수(`self._...`)인가? — UX-2 자체 리뷰
- [ ] 트레이 상주 시 closeEvent는 hide()로 가고, 진짜 종료는 별도 경로(`_is_quitting` 플래그 등)인가? — T-1
- [ ] QApplication/settings/overlay 생성은 `CocktailAppController` 한 곳에서 관리되는가? `__main__`에 직접 생성 코드를 흩뿌리지 말 것 — BG-2
- [ ] `keyboard` 같은 글로벌 단축키 라이브러리는 optional 처리 + 시그널/슬롯으로 메인 스레드 진입하는가? — T-1
- [ ] 캡처 모드 추가 시 `_update_red_rect_for_mode()`로 red_rect를 갱신해 기존 OCR/렌더링 파이프라인을 재사용하는가? — T-2
- [ ] 캡처 모드/hover/red_rect 상태는 `CaptureRegionController`가 소유하는가? settings panel resize/initUI가 직접 캡처 영역을 만들지 말 것 — BG-3
- [ ] 워커 / 엔진 시그널 / OCR/UIA 라우팅 / 모델 lifecycle / 캐시 타이머는 `BackgroundController`가 소유하는가? `TransparentWindow`에 잔재가 남으면 안 됨 — BG-4
- [ ] `BackgroundController` 생성 시 `region`, `options_provider`, `is_self_hwnd` 3개 의존만 주입했는가? widget 직접 참조 금지 — BG-4
- [ ] 콤보 상태는 매 iteration마다 `RuntimeOptions(@dataclass(frozen=True))` 스냅샷으로 워커가 읽는가? — BG-4
- [ ] forward-ref가 필요한 어노테이션(예: TransparentWindow에서 RuntimeOptions 참조)은 문자열로 처리했는가? — BG-4 자체 리뷰
- [ ] `red_rect`는 화면 절대 좌표인가? `owner.geometry()` / `region.owner.geometry()` 변환 코드가 잔재로 남으면 안 됨 — BG-5
- [ ] OverlayWindow MODE_BOX 그리기는 settings 창 위치(`ctl.geometry().x()`)에 의존하지 않는가? `rect.x() - wx`만 사용 — BG-5
- [ ] AST 헬퍼(`_method`)는 `async def`(AsyncFunctionDef)도 처리하는가? `capture_and_translate_async` 같은 코루틴 검사 시 필요 — BG-5 자체 리뷰
- [ ] settings UI 클래스 이름은 `SettingsWindow`인가? (`TransparentWindow`는 BG-6에서 리네이밍 완료) — BG-6
- [ ] `CaptureRegionController.__init__`는 `is_self_hwnd` 콜백 시그니처인가? owner 위젯 직접 참조 금지 — BG-6
- [ ] `region.foreground_window_rect`는 `self._is_self_hwnd(int(hwnd))` 콜백을 사용하는가? `self.owner.winId()` 직접 호출 금지 — BG-6
- [ ] `SettingsWindow`에 빈 `paintEvent` override가 잔재로 남으면 안 됨 (Qt 기본이 자동 처리) — BG-6
- [ ] 분리 사이클(BG-1~BG-6) 완료 후 새 책임을 추가할 땐 어느 컴포넌트(SettingsWindow/BackgroundController/CaptureRegionController/OverlayWindow/CocktailAppController)인지 ARCHITECTURE.md를 먼저 갱신했는가?
- [ ] MODE_BOX 영역 박스를 마우스로 편집할 진입점이 SettingsWindow + 트레이 메뉴에 모두 존재하는가? — BG-7
- [ ] `RegionEditor`는 가상 데스크톱 전체 frameless/translucent/always-on-top 위젯이고, 내부 박스 상태는 처음부터 절대 좌표인가? (BG-5 좌표계 결정과 일관) — BG-7
- [ ] `RegionEditor` 위젯 자체에 `set_window_capture_affinity(self.winId(), exclude=True)`가 적용됐는가? (B-10 자기 캡처 피드백 차단) — BG-7
- [ ] `SettingsWindow._is_self_hwnd`가 `region_editor.winId`도 자기 hwnd로 인식하여 worker가 편집 창을 캡처/번역 시도하지 않는가? — BG-7
- [ ] `RegionEditor` 저장 시 `region.red_rect` 절대 좌표 대입 + `set_capture_mode(MODE_BOX)` 자동 전환 + `controller.reset_frame_hash()` 호출되는가? — BG-7
- [ ] `RegionEditor`는 lazy 생성(첫 클릭 시) + closeEvent를 hide-instead-of-close로 처리하고, 앱 종료 시점만 `force_close()`로 실제 close 하는가? — BG-7
- [ ] `RegionEditor`가 추가한다고 `BackgroundController`의 민감 창 가드(S-1)/capturable 가드(B-15)를 건드리지 않았는가? (RegionEditor는 geometry만 바꾸고 번역 파이프라인 미접근) — BG-7
- [ ] 활성 창 모드에서 `GetForegroundWindow` 결과가 우리 자신 hwnd면 캡처를 건너뛰는가? (B-10 피드백 재발 방지) — T-2
- [ ] 활성 창/hover 모드에서 사용자가 다른 창 클릭하려면 마우스 통과 토글이 있는가? (Ctrl+Shift+P) — W-1
- [ ] 박스 모드 외에서 민감 창(패스워드/로그인/결제) 활성 시 OCR/번역을 자동 중지하는가? — S-1
- [ ] 민감 키워드 사전(`SENSITIVE_KEYWORDS`)에 사용자 환경의 은행/금융 앱 제목이 포함됐는가? — S-1
- [ ] 마우스 통과(`WS_EX_TRANSPARENT`) 적용 시 `WS_EX_LAYERED`가 유지되어 반투명 그리기가 깨지지 않는가? — W-1
- [ ] OverlayWindow와 ControlWindow(=TransparentWindow) 둘 다 `WDA_EXCLUDEFROMCAPTURE`로 캡처에서 제외했는가? — W-2
- [ ] 워커가 저장하는 `adjusted_bbox`는 화면 절대 좌표인가? (OverlayWindow가 그대로 그림) — W-2
- [ ] OverlayWindow는 `Qt.Tool` 플래그로 작업표시줄에 안 뜨는가? `WA_ShowWithoutActivating`으로 포커스 도둑질 안 하는가? — W-2
- [ ] 종료 시 ControlWindow가 OverlayWindow도 같이 닫는가? — W-2
- [ ] OverlayWindow는 가상 데스크톱(모든 모니터) 전체 영역으로 확장됐는가? 화면 절대 → 윈도우 내 좌표 변환 적용했는가? — W-3
- [ ] 윈도우 이동/리사이즈/드래그 시 OverlayWindow.update()가 호출되는가? — W-2 자체 리뷰
- [ ] 워커 스레드 생성/시작은 `__init__` 본문에 있는가? QTimer slot 안으로 밀려 들어가지 않았는가? — CR-1
- [ ] 캡처 허용/차단 토글은 ControlWindow와 OverlayWindow 둘 다 같은 상태로 바꾸는가? — CR-1
- [ ] 활성 창/민감 창/자기 캡처 가드는 ControlWindow hwnd와 OverlayWindow hwnd를 모두 제외하는가? — CR-1
- [ ] UIA 번역 경로도 OCR 경로와 동일하게 `assign_region_languages()`를 적용하는가? — CR-1
- [ ] PersistentCache는 import 시점이 아니라 lazy load(첫 get/put 또는 preload_async)인가? — D-1
- [ ] PersistentCache의 키는 sha256 hash이고, 값(번역)은 DPAPI로 전체 dict 암호화되어 디스크에 저장되는가? — D-1 / 보안 원칙 #2
- [ ] UIA 모드는 mode_combo와 무관하게 항상 활성 창 민감 검사를 적용하는가? — UIA-1
- [ ] uiautomation은 optional lazy import이고, 미설치 시 앱은 정상 동작하는가? — UIA-1
- [ ] DPAPI helper는 `argtypes/restype` 명시되어 64비트 호출 규약이 정확한가? — D-1 보강
- [ ] PERSIST_CACHE는 비정상 종료 손실 방지를 위해 주기적 save(QTimer 30초)가 동작하는가? — A-1
- [ ] 자동 실행 등록 시 `pythonw.exe` (콘솔 안 뜸) 또는 PyInstaller `.exe`를 우선 사용하는가? — A-1
- [ ] 새 의존성/모델 추가 시 LICENSE-3RDPARTY.md에 동시 기록했는가? (CC-BY-NC, GPL, 상용 전용 금지)
- [ ] 다음 step 이동 전 `python smoke_tests.py`를 통과했는가? 치명 회귀(CR-1/M-7/OverlayWindow)를 자동 확인했는가? — ST-1
- [ ] 자동 실행 명령은 `.py` 실행 시 `pythonw.exe`를 우선해 콘솔 창을 피하는가? — A-2
- [ ] 주기적 PersistentCache save는 백그라운드 스레드이며 `_mem_lock`으로 저장 중 put/get 경쟁을 막는가? — A-2
- [ ] 릴리즈 워크플로우는 진단 전 Tesseract를 설치하고, 서명 후 zip/installer를 생성하는가? — R-1
- [ ] 기본 배포 빌드는 PaddleOCR/PaddlePaddle을 제외하고, PyTorch `.lib`/`include` 개발 산출물을 제거하며, `Cocktail.exe --self-test`를 통과하는가? — R-2
- [ ] 화면 좌표 클립은 `primaryScreen()` 단독이 아니라 `QApplication.screens()` 합집합(가상 데스크톱)인가? — W-4
- [ ] 일시정지/스킵으로 `continue` 하기 전에 오버레이를 비웠는가? (`if self.translated_text: self.state_ready.emit([])`) — W-4 / UX-3
- [ ] `ImageGrab.grab`에 `all_screens=True`를 줬는가? (False면 보조 모니터가 검은 이미지) — W-5
- [ ] 창 상태(마우스 통과 등)를 강제로 바꾸는 경로에서 토글 플래그를 같이 동기화했는가? — W-6
- [ ] 번역 정지(`set_running(False)`) 시 오버레이 자막을 지우는가? — UX-3
- [ ] frame-skip 해시가 내용 + 캡처 영역 좌표 둘 다 반영하고, 평균이 아니라 셀 최대 변화량 기준인가? — F-1
- [ ] OCR 픽셀 크기를 폰트에 넘길 때 `QFont(family, pt)` 생성자가 아니라 `setPixelSize()`인가? — F-2
- [ ] MODE_WINDOW 캡처 영역이 "활성 창이 있는 모니터 1개"로 클램프되는가? (바탕화면/전 화면 덮는 창 = 전 모니터 캡처) — SC-1
- [ ] 모델/스냅샷 로드는 `local_files_only=True` 우선 → 실패 시에만 네트워크인가? 전역 `HF_HUB_OFFLINE` 강제 금지 — LD-1
- [ ] 번역 캐시 키는 `_cache_key_text()`(공백 collapse+strip) 정규화 텍스트인가? 표시용 원문은 원본 유지 — RT-1
- [ ] 오버레이 상태 emit은 `_emit_state_if_changed()`를 통하는가? (직전 상태와 실질 동일하면 재그리기 생략) — RT-2
- [ ] OCR 라인은 평균 confidence + 실문자 비율 필터를 통과한 것만 그리는가? (UI 부스러기/시계 제외) — NZ-1
- [ ] `os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "0")`이 모듈 최상단(QApplication 생성 이전)에 있는가? — DP-1
- [ ] 앱의 모든 좌표를 **물리 픽셀** 하나로 다루는가? (Qt 논리 좌표 가정·devicePixelRatio 곱셈 금지) — DP-1 / E-7
- [ ] OCR 업스케일 판정이 이미지 크기만이 아니라 **1차 OCR 라인 높이 중앙값**도 보는가? (큰 화면 속 작은 캡션) — OU-1 / E-3
- [ ] 재OCR은 전처리+`_lines_from_tess(data, scale)` 기존 경로를 재사용하는가? 새 좌표 환원 코드 금지 — OU-1
- [ ] m2m100 폴백이 사용자 명시 src(en/ko 외) 또는 스크립트 확정 언어(ja/zh/ru…)에서만 켜지는가? 라틴 오판으로 새는 경로가 없는가? — SL-1
- [ ] 모든 `model.generate(...)`에 `no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE`가 있는가? — RP-1
- [ ] 오버레이에 그리기 직전 `display_gate_reject()`를 통과시켰는가? (확신 없으면 숨긴다) — DG-1
- [ ] OCR 라인의 `font_size`는 word 높이 **중앙값**인가? (`max`는 이상치 하나에 끌려간다) — LM-1
- [ ] 한 라인 안에서 높이 이상치 word 제거 + 큰 가로 간격 분할(`_line_segments`)을 거쳤는가? — LM-1
- [ ] 오버레이 폰트는 한글 우선 스택(`OVERLAY_FONT_FAMILIES`)인가? `QFont("Arial")` 하드코딩 금지 — FT-1
- [ ] 오버레이는 show() 후 `raise_()` + 주기적 `set_window_topmost()`로 항상-위를 되찾는가? — TM-1
- [ ] 워커가 "비울 게 있나"를 판정할 때 `self.translated_text`(메인 스레드 갱신)가 아니라
      워커 로컬 `self._last_emitted`를 보는가? emit은 `_emit_state`/`_clear_overlay_if_any` 단일 경로인가? — RC-1
- [ ] 워커 스레드에서 Qt GUI 전용 API(`QApplication.screens()`, `QCursor.pos()`, `screenAt`)를 부르지 않는가?
      화면 rect는 `region.screen_rects` 캐시, 커서는 `cursor_pos_physical()` — GT-1
- [ ] 모니터 추가/제거/해상도 변경에 `screenAdded`/`screenRemoved`/`geometryChanged`가 연결돼
      오버레이 재적합 + 화면 rect 캐시 갱신이 도는가? — MS-1
- [ ] 자막 클립은 가상 데스크톱 전체가 아니라 "그 박스가 속한 모니터 ∩ 창 rect"인가? — CL-1
- [ ] UIA가 민감 창에서 돌려주는 값이 빈 리스트가 아니라 `UIA_SKIP_FRAME` sentinel이고,
      호출부가 OCR 폴백 없이 프레임을 건너뛰는가? — SV-1
- [ ] `.bat` 파일에 한글(주석·경로·메시지)을 넣지 않았는가? 콘솔 OEM 코드페이지(CP949) 대
      UTF-8 충돌로 파일이 깨진다. 한국어 안내는 `.md`가 담당 — RB-1
- [ ] 의존성을 뺄 때 코드만 지우고 끝내지 않았는가? `requirements.txt` / `build.spec` /
      `diagnose_environment.py` / `pre_release_check.py` 4곳을 같이 훑었는가? — DR-1
- [ ] 문서에 "100% 로컬"처럼 사실보다 센 문구를 쓰지 않았는가? 최초 1회 모델 다운로드와
      Tesseract 별도 설치는 반드시 함께 적는다 — DR-1
- [ ] 표시 게이트의 "개수" 임계가 **가변 길이 단위**(PM-1 병합 문단)에 적용되고 있지 않은가?
      길이에 비례시켰는가? 절대 임계는 한 줄 기준으로만 유효하다 — CO-2
- [ ] 줄을 **숨기는** 모든 경로가 사유를 로그로 남기는가? 숨김은 무음 실패라 로그가 없으면
      사용자도 개발자도 원인을 못 찾는다. 단, 라인당 1회로 억제 — CO-2
- [ ] 표시 게이트를 새 규칙으로 강화할 때 "정상 문장이 사라지지 않는가"를 **먼저** 실측했는가?
      거짓 양성은 게이트의 실패다 — DG-1 / CO-2
- [ ] 설정창 위젯을 `setGeometry` 절대 좌표로 놓고 있지 않은가? 좌표는 상수여도 **글자 크기는
      상수가 아니다**(고DPI에서 폰트만 1.75배). 레이아웃 + `sizeHint`로 — UI-1
- [ ] 문구가 실행 중에 바뀌는 버튼은 `_FitButton`인가? (폭을 호출부마다 재면 빠뜨린 경로가 생긴다) — UI-1
- [ ] 창보다 길어질 수 있는 라벨은 `QSizePolicy.Ignored` + `_elide()`인가? (긴 상태 문구가
      창 최소 폭을 밀어 올리면 안 된다) — UI-1
- [ ] Qt 애플리케이션을 만드는 테스트/하네스는 `QCoreApplication`이 아니라 `QApplication`인가?
      나중에 위젯을 만드는 순간 프로세스가 **에러 없이** 죽는다 — UI-1
- [ ] 고DPI 대응을 좌표에 배율을 곱해 하려 하고 있지 않은가? DP-1을 깨지 말고 설정창 글꼴을
      **픽셀 크기**로 (px = pt × 그 화면 ldpi / 72). 스케일링을 꺼도 **글자는** 화면 DPI로 그려진다 — UI-2
- [ ] 글꼴을 바꿨으면 **자식 위젯에 직접** 줬는가? 부모 setFont는 자식 `font()`에 안 온다.
      `layout.invalidate()` 없이 `activate()`만 하면 옛 크기가 그대로 나온다 — UI-2
- [ ] 아이콘/이미지를 외부에서 받아오지 않았는가? 코드로 그리면 라이선스가 깨끗하다. 16px은
      크게 그려 축소 + `NoSubpixelAntialias`로 색 번짐을 막았는가? — IC-1

---

## 1. 성능 병목 (Cocktail완성본.py 1차 리뷰 — 2026-04-30)

### P-1. 이중 번역 (src → en → target)
- **증상**: 모든 번역이 2번 generate 호출 → CPU에서 ~2초/라인
- **원인**: src와 tgt 사이에 영어를 무조건 끼워 넣음. en→ko인데도 en→en→ko로 동작
- **수정**: src≠tgt면 1회 generate (`tokenizer.src_lang=src`, `forced_bos_token_id=tgt`).
  m2m100은 any-to-any 직접 번역을 지원함. 영어 피벗은 비표준 언어쌍에서 품질 보강용으로만 옵션 처리.
- **재발 방지**: 번역 함수에 `# single-pass: any-to-any` 주석을 남김.

### P-2. 라인별 순차 번역
- **증상**: `asyncio.gather([self.async_translate_text(t) for t in texts])` 가 병렬로 보이지만,
  내부의 `model.generate`는 동기 GPU 호출이라 이벤트 루프에서 **직렬 실행**됨
- **수정**: 모든 라인을 한 번에 `tokenizer(texts_list, padding=True, return_tensors='pt')` 로 토크나이즈하고
  `model.generate(**batch)` **단 1회** 호출. 이후 `tokenizer.batch_decode(...)`로 디코드.
- **이득**: 라인 5~10개 환경에서 약 3~5배 가속 (GPU 점유율 상승, kernel launch 오버헤드 분할상환).

### P-3. fp16/inference_mode 미적용
- **수정**: CUDA 가용 시 `model = model.half()`, generate 블록을 `with torch.inference_mode():` 로 감쌈.
- **주의**: CPU 환경(half 미지원)이면 `model.half()` 생략. `device.type == 'cuda'`로 분기.

### P-4. 매 프레임 OCR (변화 없어도)
- **수정**: 캡처 직후 numpy 배열의 빠른 해시(다운샘플 후 hash) 또는 `cv2.absdiff` mean 비교.
  직전 프레임과 거의 같으면 OCR/번역 모두 스킵.
- **임계값**: 8x8 평균 다운샘플 후 byte-equal, 또는 mean diff < 2.0 정도.

### P-5. 번역 캐시 부재
- **수정**: `functools.lru_cache(maxsize=512)` 또는 dict로 `(src_lang, tgt_lang, text) → translation` 캐시.
  같은 자막/UI 텍스트는 0ms 처리.

### P-6. 핫 패스의 print 폭주
- **수정**: `image_to_data` 결과 단어별 print 제거, paintEvent의 "출력 중:" print 제거.
  필요하면 `logging.DEBUG` 로 격하.

### P-7. langdetect 매 호출
- **수정**: combo box로 source language 선택 추가. "Auto"인 경우에만 langdetect 호출.
  langdetect는 짧은 텍스트에서 결과가 흔들려서 정확도도 낮음.

### P-9. 모델이 import 시점에 동기 로드 → GUI 시작 지연 (UX 결함)
- **증상**: 프로그램 시작 시 ~5-30초 동안 GUI 안 뜸 (모델 로드/워밍업 끝날 때까지).
  사용자는 "왜 안 떠?" 의심하고 종료할 수 있음.
- **원인**: `TRANSLATOR = init_translator()` 가 모듈 레벨에서 동기 호출 → import가 완료될 때까지 QApplication 생성 안 됨.
- **수정**:
  1. `HybridTranslator.__init__`은 가벼움 (모델 로드 X). 첫 `translate_batch` 호출에서 lazy load.
  2. UI 띄운 직후 백그라운드 스레드로 `TRANSLATOR.preload_default()` 호출 — en→ko opus-mt 미리 로드.
  3. 로드 중에 `run_button` 비활성화 + "모델 준비 중…" 텍스트.
  4. 로드 완료 시 `model_ready` 시그널 → 메인 슬롯에서 버튼 enable + "번역 시작".
- **재발 방지**: 5초 체크리스트에 항목 추가. 모듈 레벨 동기 모델 로드 절대 금지.
  새 모델 추가 시 lazy load 패턴 유지.

### P-8. max_new_tokens 미지정
- **수정**: `max_new_tokens=min(256, int(len(input_ids[0]) * 1.5) + 16)` 정도로 상한 명시.
  m2m100 기본값이 너무 크면 generate가 길어짐.

---

## 2. 정확성 버그

### B-1. dedup 비교 대상 오류
- **증상**: 같은 OCR 텍스트가 11px만 움직여도 재번역됨
- **원인**: `existing_lines = [item[0] for item in self.translated_text]` 인데
  `item[0]`은 **번역 결과(한국어)**. OCR한 영어와 텍스트 유사도 비교는 항상 0 → bbox만으로 판단
- **수정**: `translated_text`의 튜플을 `(src_text, tgt_text, bbox, font_size)` 4-tuple로 확장하고,
  dedup은 src_text끼리 비교.

### B-2. 예외 1번에 전체 루프 종료
- **수정**: `except Exception as e:` 블록에서 `self.stop_thread = True; break`를 제거. `continue`로 변경하고
  연속 실패 카운터를 두어 N회 연속 실패 시에만 종료.

### B-3. `translated_text` 무한 누적
- **수정**: 매 프레임 새로 OCR된 라인을 기준으로 리스트를 **재구성**한다.
  옛 번역은 화면에서 사라진 텍스트면 즉시 제거. (현재 누적 → 화면이 옛 번역으로 가득 참)

### B-4. `line_height_threshold = 1`
- **수정**: 1px → 평균 글자 높이의 절반(`max(8, height_median // 2)`).
  Tesseract `line_num` 필드를 직접 사용하는 게 더 정확함 (`tesseract_result['line_num']`).

### B-5. paintEvent에서 매 프레임 print
- **수정**: paintEvent의 모든 print 제거. paintEvent는 초당 수십 회 호출되므로 IO가 병목이 됨.

### B-6. langdetect 빈 문자열 예외
- **수정**: try/except 가 있긴 하지만 fallback이 "[ERROR]" 문자열을 그대로 화면에 그림.
  fallback은 src 추정 실패 시 combo의 source 또는 "en" 으로 기본값.

### B-8. PIL ImageGrab(RGB) → cv2.COLOR_BGR2GRAY 채널 순서 오류
- **증상**: 컬러 자막/배경에서 OCR 정확도가 낮거나, Otsu 임계 결과가 의도와 달라 글자가 잘림.
- **원인**: `ImageGrab.grab(...)`은 PIL → **RGB** 순서. 이를 numpy로 변환 후 `cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)`에 넣으면
  R 채널이 B 가중치(0.114)로, B 채널이 R 가중치(0.299)로 처리됨. 빨간색/파란색 자막이 회색조에서 위치가 뒤바뀜.
- **수정**: `cv2.COLOR_RGB2GRAY` 사용. 또는 PIL 이미지를 `screenshot.convert("RGB")` 후 `np.array`.
- **재발 방지**: 캡처 소스(PIL vs OpenCV vs mss)별 채널 순서를 한 번 더 확인.
  주석으로 "PIL → RGB" 명시. 5초 체크리스트에도 항목 추가.

### B-9. paintEvent QRect 클립 부정확 + word wrap 누락
- **증상**: 텍스트 박스가 red_rect 모서리 근처에 있을 때 일부가 빨간 사각형 바깥으로 그려짐.
  + 한국어 번역이 영문보다 길어 박스 안에서 잘림 (논문 초록 등 긴 문장에서 두드러짐).
- **원인**:
  1. `min(self.red_rect.width(), right - left)`: `left`는 `max(red_rect.left(), 원본_left)`로 보정됐지만
     width는 **원본 left 기준** `right - left`로 계산. 보정된 left + 원본 width → 오른쪽이 삐져나감.
  2. `drawText`에 `Qt.TextWordWrap`이 빠져 있음.
- **수정**: `QRect(int(left), int(top), int(right-left), int(bottom-top)).intersected(self.red_rect)`로
  안전하게 교차. drawText에 `Qt.TextWordWrap` 플래그 추가. `rect.isEmpty()`면 그리지 않음.
- **재발 방지**: 모서리 클립이 필요한 경우 항상 `QRect.intersected()` 사용.
  좌표 보정과 크기 계산을 분리하지 말 것.

### B-14. OCR 언어 하드코딩 → 다국어 입력 OCR 불가 + 인스톨러 비효율
- **증상**:
  1. `TESS_LANG = "eng+kor"` 하드코딩이라 일본어/중국어/프랑스어 자막은 OCR이 읽지 못하거나 깨짐.
  2. 한국인 아닌 사용자에게도 `kor.traineddata`가 무조건 깔리는 비효율.
- **수정**:
  1. `LANG_TO_TESS = {"en":"eng", "ko":"kor", "fr":"fra", "de":"deu", "zh":"chi_sim", "ja":"jpn"}` 매핑 추가.
  2. 시작 시 `pytesseract.get_languages()`로 사용 가능한 traineddata 진단 → `AVAILABLE_TESS_LANGS` 캐시.
  3. `_ocr_langs()` 메서드: src 콤보가 명시 언어면 해당 traineddata 단일 선택, "auto"면 사용 가능한 모든 언어 합집합 (`"eng+kor+jpn"` 등).
  4. 당시에는 `perform_ocr_with_boxes(image, lang_str)`로 Tesseract 언어를 인수로 넘겼으나,
     O-1 이후에는 `perform_ocr_with_boxes(image)` 내부에서 선택 OCR 백엔드와 언어를 결정한다.
  5. 배포 시 앱 폴더 `./tessdata/` 가 있으면 `TESSDATA_PREFIX` 자동 설정 → 인스톨러 컴포넌트로 설치된 traineddata 인식.
  6. Inno Setup `installer.iss`: traineddata는 `[Components]` 의 옵션으로 — eng 필수, kor/jpn/chi_sim/fra/deu 선택.
- **재발 방지**: 5초 체크리스트에 항목 3개 추가. 새 언어 지원 시 `LANG_TO_TESS` + `LANG_MAP` + 콤보 + 인스톨러 컴포넌트 4곳 동시 갱신.

### B-13. 박스 height 폭증 → 인접 줄 겹침 (폰트 자동 축소 결합)
- **증상**: B-12로 height 동적 확장만 적용하면, 좁은 영문 박스(예: 한 줄에 단어 3개)에 긴 한국어가 들어갈 때
  height가 영문 한 줄의 4~5배까지 늘어나 다음 영문 줄을 완전히 덮음.
- **수정**: `_fit_font(painter, base_rect, text, initial_size, min_size=9)` 추가.
  initial_size부터 1pt씩 내리면서 needed.height() ≤ base_rect.height()*2 (또는 base+4) 가
  될 때까지 폰트 축소. 그래도 안 맞으면 min_size(9pt)에서 멈추고 B-12의 height 확장으로 처리.
- **이득**: 인접 줄 겹침을 대폭 완화하면서 폰트가 너무 작아지는 가독성 저하도 9pt에서 차단.
- **재발 방지**: paintEvent의 height/font 측정 비용은 텍스트당 ~1ms. 라인 10개여도 ~10ms — 무시 가능.
  단, paintEvent를 빈번하게 트리거하지 말 것 (매 OCR 사이클 1회로 제한).

### B-12. 한국어 번역이 영문 박스를 넘어가 잘림 + 영문이 박스 너머 비침
- **증상**:
  1. 영문 한 줄(예: 80자)이 한국어로 번역되면 길이가 더 길어 박스 안에서 잘림. word wrap만으론 height가 모자람.
  2. 박스 alpha=200(78%)이라 뒤 영문 글자가 비쳐 한국어와 겹쳐 보임. 두 글자가 같은 위치에서 충돌.
- **수정**:
  1. `painter.fontMetrics().boundingRect(x, y, width, 10000, Qt.AlignLeft | Qt.TextWordWrap, text)`로
     실제 필요한 height를 미리 측정. width는 영문 박스 그대로, height만 `max(base, needed + 4)`로 확장.
  2. 클립 영역을 `self.red_rect` → `self.rect()` (위젯 전체)로 변경. 박스가 빨간 사각형 살짝 밖으로 나가도 허용.
     자기 자신은 B-10으로 캡처에서 제외되므로 OCR 피드백 위험 없음.
  3. 박스 alpha 200 → 235로 상향. 영문이 뒤로 비치지 않게 차단.
  4. `painter.setFont(font)`를 boundingRect 호출 **전**에 둘 것 (글자 폭 측정 정확도 보장).
- **재발 방지**:
  - 5초 체크리스트에 항목 추가.
  - 인접 줄 간격이 좁은 경우(예: 자막 두 줄이 거의 붙어 있음) height 확장이 다음 줄과 겹칠 수 있음.
    겹침이 빈번하면 폰트 자동 축소(0.8x) fallback을 추가 — 단, 현재는 보류.

### B-10. 자기 자신을 OCR하는 피드백 루프 (오버레이 → OCR 입력 오염) ⭐
- **증상**: 첫 캡처는 정상 번역되지만 약 1초 후 화면이 깨진 한국어/영어 혼합 문자열로 변함.
  "=A: 암 치료에 대한 약속", "기존 치료법 4 Syne AA?", "ICESp", "WAALLAEAAH ori" 같은
  명백히 번역 가능한 영어가 아닌 텍스트로 채워짐.
- **원인**: `ImageGrab.grab()`은 **데스크톱 합성 결과**를 캡처하므로 우리 오버레이 윈도우(빨간 박스 + 한국어 텍스트)도 포함됨.
  1차 캡처 → OCR/번역 → paintEvent로 화면 표시 → 2차 캡처에 한국어 오버레이 포함 →
  OCR이 영어 + 한국어 + 반투명 박스 가장자리를 모두 읽음 → 깨진 문자열을 다시 번역 → 더 깨진 결과 → ...
- **수정**: Windows `SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE=0x11)`을 통해
  우리 윈도우를 ImageGrab/mss/PrintScreen 등 모든 화면 캡처 API에서 제외.
  Windows 10 build 2004+ / Windows 11에서 동작.
- **재발 방지**:
  - 5초 체크리스트에 항목 추가.
  - 캡처 백엔드를 바꿀 때(mss/dxgi 등) 동일하게 제외 메커니즘이 필요함을 주석에 명시.
  - 비-Windows 플랫폼에선 hide()/show() 토글이나 별도 스크린 분리 등 대안 필요.

### B-11. 워커 스레드의 self.translated_text 직접 할당 (race condition)
- **증상**: 잠재적. paintEvent 도중 list rebinding이 일어나면 일관성 깨질 수 있음.
  파이썬 list 할당 자체는 atomic이지만, 새 list로 갈아치우는 사이 paintEvent의 for 루프가
  중간 상태를 볼 가능성. 추가로 `self.update()` 호출도 워커에서 직접 함.
- **원인**: PyQt 위젯의 상태 변경/렌더링 트리거는 메인(UI) 스레드에서만 안전.
  `QWidget.update()`는 thread-safe로 문서화되어 있으나, 그 외 위젯 상태 변경(self.run_button.setText 등)은 위험.
- **수정**: `pyqtSignal(list)` (state_ready) + `pyqtSignal(str)` (failure_alert)을 정의하고,
  워커는 `self.state_ready.emit(new_state)` / `self.failure_alert.emit(msg)` 만 호출.
  메인 슬롯 `_apply_state`/`_on_failure`에서 위젯 상태 갱신.
- **재발 방지**: 워커 스레드는 위젯 속성에 절대 직접 쓰지 않는다. 모든 위젯 변경은 시그널/슬롯 경유.

### B-7. 타이핑 효과 + 0.25초 워커 주기 충돌 → 줄들이 번갈아 깜빡임
- **증상**: 화면에 한국어 번역이 한 줄씩 나타났다 사라지며, 모든 줄이 동시에 보이지 않음.
- **원인**:
  1. `update_typing`이 `typing_index` 한 줄만 글자 단위로 타이핑 (한 줄 ~2.8초 소요).
  2. 워커 루프가 `await asyncio.sleep(0.25)` 마다 OCR/번역 후 `typing_index = 0`으로 리셋.
  → 첫 줄도 채 끝나기 전에 처음으로 돌아감. paintEvent의 "완료된 줄"이 영원히 0개에 머무름.
- **수정**: 타이핑 효과 자체를 제거. `translated_text`가 갱신되면 모든 줄을 paintEvent에서 동시에 그림.
  - 제거 대상: `typing_index`, `current_line`, `typing_timer`, `update_typing`, `start_typing_timer`,
    그리고 그에 따라 죽은 import (`QTimer`, `QMetaObject`, `pyqtSlot`).
  - paintEvent는 `for _, tgt, bbox, fsz in self.translated_text: draw_translated_text(...)` 한 줄로 단순화.
- **재발 방지**: UI 애니메이션 상태(progress index)를 데이터 갱신 주기(워커 sleep)보다 짧게 설계할 것.
  애니메이션을 다시 도입한다면 OCR src_text 동일 여부를 확인해 진행도를 보존하거나, 줄별 독립 cursor를 가져야 함.

---

## 3. 환경/부수 이슈

### U-1. 일반 사용자 진입 실패 방지: 앱 시작 진단 + 상태 라벨
- **증상**: 일반 사용자는 Tesseract/Python 모듈/traineddata/optional PaddleOCR 중 무엇이 빠졌는지 알기 어렵다.
  기존 구조처럼 Tesseract 실행 파일 탐색 실패가 import 단계 예외로 터지면 앱 자체가 뜨지 않아 복구 경로가 없다.
- **수정**:
  1. `_resolve_tesseract()`는 예외를 던지지 않고 `None`을 반환.
  2. `TESSERACT_ERROR`, `environment_summary()` 추가.
  3. UI 상단 상태 라벨 추가. 모델 준비, 환경 요약, OCR 없음, PaddleOCR 폴백, 번역 완료 시간을 표시.
  4. `diagnose_environment.py` 추가 — 필수 모듈, optional PaddleOCR, Tesseract, traineddata, Torch device 점검.
  5. `run_cocktail.bat` 추가 — 일반 사용자가 더블클릭으로 진단 후 앱 실행 가능.
- **재발 방지**:
  - 일반 사용자 환경 문제는 시작 예외로 종료하지 말고, 앱이 뜬 뒤 상태 라벨과 진단 도구로 설명한다.
  - 필수 기능이 완전히 불가능한 경우에도 어떤 설치가 필요한지 구체 메시지를 남긴다.
  - optional dependency는 import 시점에 강제하지 않는다.

### E-1. Tesseract 경로 하드코딩
- 현재: `C:\tesseract\tesseract.exe`
- **수정**: `TESSERACT_CMD` 환경변수 우선, 없으면 기본 경로 후보 리스트(`C:\tesseract\...`,
  `C:\Program Files\Tesseract-OCR\...`)를 순회. 사용자 PC마다 다르므로.

### E-2. Tesseract PSM 기본값 (3)
- **수정**: 오버레이 자막은 `--psm 6`(균일 블록) 또는 `--psm 7`(한 줄)이 더 정확하고 빠름.
  `config="--psm 6 --oem 1"` 추가.

### E-3. 작은 글자 OCR 정확도
- **수정**: 캡처 영역이 너무 작으면 `cv2.resize(scale_x=2, scale_y=2, interpolation=INTER_CUBIC)`
  upscale 후 OCR. Tesseract는 ~30px 글자 높이에서 정확도가 가장 좋음.
- **미해결분 → OU-1 (2026-08-02, 4차)**: 이 조건은 **이미지 크기**만 본다. 큰 화면(1920x1080) 안의
  작은 캡션(7~9px)은 트리거되지 않아 그대로 무너졌다. OU-1이 "1차 OCR 라인 높이 중앙값"
  기준의 적응 업스케일로 이 구멍을 막는다.

### E-4. `mousePressEvent` 들여쓰기
- 현재: 7-space (다른 메서드는 4-space). 동작은 하지만 가독성 ↓
- **수정**: 4-space로 통일.

### E-5. "모델 로드" 버튼이 실제로 모델을 로드하지 않음
- 모델은 import 시점에 이미 로드됨. 버튼은 단지 `model_loaded=True` 토글.
- **수정**: 버튼 라벨을 "번역 시작/정지"로 변경, 또는 모델 로드를 lazy하게 옮김.

### E-10. PyQt5 GPL → PySide6 LGPL 마이그레이션 (상용 라이선스) ⭐
- **이유**: PyQt5는 GPL — 상용 closed-source 배포 시 (a) 소스공개 강제 (b) 상용 라이선스 구매 ($550/dev/yr).
  PySide6은 LGPL — closed-source 상용 OK, 무료.
- **마이그레이션 핵심 변경**:
  - `from PyQt5 import QtWidgets` → `from PySide6 import QtWidgets`
  - `pyqtSignal` / `pyqtSlot` → `Signal` / `Slot` (모듈: `PySide6.QtCore`)
  - `QShortcut`은 PyQt5 `QtWidgets`였으나 PySide6에선 **`QtGui`로 이동**
  - `event.globalPos()` (deprecated in Qt6) → `event.globalPosition().toPoint()`
  - `Qt.AA_EnableHighDpiScaling` 호출 제거 — PySide6/Qt6은 자동 활성. (E-7 정책과 일관)
- **호환성 유지된 것 (변경 불필요)**: `Qt.AlignLeft|AlignVCenter|TextWordWrap`, `Qt.NoPen`,
  `Qt.WA_TranslucentBackground`, `QPainter.Antialiasing`, `QFontMetrics.boundingRect`,
  `QShortcut(QKeySequence, parent, activated=...)` 시그니처, `event.button()/buttons()`.
- **재발 방지**: 5초 체크리스트에 "PyQt5 흔적 없음" 항목. StackOverflow에서 PyQt5 코드를 그대로 복사 금지.
  `pyqtSignal` / `pyqtSlot` 단어가 코드에 남아 있으면 즉시 `Signal` / `Slot`으로 교체.

### E-9. 캡처 차단의 부작용 — 사용자도 스크린샷 못 찍음
- **증상**: B-10으로 `WDA_EXCLUDEFROMCAPTURE` 적용 후 사용자가 PrintScreen / Win+Shift+S /
  스크린샷 도구로 번역창을 찍으면 검은 영역만 나오거나 윈도우가 빠진 화면이 캡처됨.
  화면 공유(Zoom, Discord 등)에서도 동일.
- **원인**: `WDA_EXCLUDEFROMCAPTURE`는 OS가 데스크톱 합성에서 윈도우를 빼므로
  ImageGrab(우리 OCR용)뿐 아니라 사용자가 직접 쓰는 모든 캡처에서도 빠짐.
- **수정**: 토글 UI 추가
  1. 헬퍼를 일반화: `set_window_capture_affinity(hwnd, exclude: bool)` (WDA_NONE / WDA_EXCLUDEFROMCAPTURE).
  2. `self.capturable` 상태와 `capture_btn` 버튼("스샷차단" ↔ "스샷OK") 추가.
  3. 단축키 `Ctrl+Shift+H` (QShortcut) 으로 토글 가능.
  4. 캡처 허용 + 번역 동작 중이면 OCR 피드백(B-10) 위험을 사용자에게 경고 print.
- **운용 규칙**: 캡처 허용은 스크린샷 찍는 짧은 순간만. 찍은 후 즉시 다시 차단.
  (캡처 허용 + 번역 동작이 동시에 1초 이상 지속되면 B-10 증상 재발 가능.)
- **재발 방지**: 5초 체크리스트에 항목 추가. 캡처 차단 같은 OS 레벨 부작용이 있는 기능을 추가할 땐
  반드시 사용자가 풀 수 있는 경로(UI 또는 단축키)를 함께 제공.

### E-7. DPI scaling 이중 적용 (정책 확정 + 수정 완료 / DP-1로 보강)
- **⚠ 후속**: 2026-08-02 DP-1에서 정책이 갱신됐다 — **Qt 논리 스케일링 OFF
  (`QT_ENABLE_HIGHDPI_SCALING=0`), 앱 전 좌표 = 물리 픽셀**. 아래 "배율을 수동으로 곱하지 않는다"
  (`_scale=1.0`)는 그대로 유효하고, 이제 Qt도 곱하지 않는다. 상세는 §DP-1.
- **결정(당시)**: `Qt.AA_EnableHighDpiScaling=True` 정책을 신뢰. devicePixelRatio를 위젯 좌표에 추가로 곱하지 않음.
- **수정**: `self._scale = 1.0` 로 고정. `self.px = lambda v: int(v)`. 이로써 200% DPI에서도
  Qt가 logical→device 변환을 알아서 처리. font_size 등 다른 곳의 `self._scale` 곱셈은 1을 곱하므로 변화 없음.
- **재발 방지**: Qt high-dpi 정책은 "AA_EnableHighDpiScaling 신뢰" 또는 "수동 scaling" 둘 중 하나만 선택.
  혼용 금지. 5초 체크리스트에 항목 명시.

### E-8. 연속 실패 시 워커 silent 종료 (UX 결함)
- **증상**: `fail_streak >= 20` 도달 시 print 한 줄만 남기고 워커가 break → 죽은 상태.
  사용자는 왜 번역이 멈췄는지 모르고, 토글 버튼을 눌러도 새 워커가 부활하지 않음.
- **수정**: break 제거. 5초 backoff 후 fail_streak를 0으로 리셋하고 continue → 자동 복구.
  `failure_alert` 시그널로 메인 스레드의 버튼 텍스트를 잠시 "⚠ 일시 정지(5s)"로 변경,
  다음 사이클에서 정상 메시지로 복원.
- **재발 방지**: 데몬 워커 스레드는 가능한 한 살려둔다. 영구 종료가 필요하면 명시적 사용자 알림 + 재시작 경로 제공.

### E-6. kor.traineddata 누락 (한국어 OCR 불가)
- **증상**: `tesseract --list-langs` 결과가 `eng`, `osd` 둘 뿐. 그러나 코드는 `TESS_LANG="eng+kor"`로 호출.
- **원인**: Tesseract 본체 설치 시 한국어 언어팩을 같이 받지 않음. (Windows 인스톨러 기본값)
- **수정**:
  1. `tesseract-ocr/tessdata_fast` 또는 `tessdata_best` 저장소에서 `kor.traineddata` 다운로드
  2. `C:\Program Files\Tesseract-OCR\tessdata\` 에 관리자 권한으로 복사
  3. 검증: `tesseract --list-langs` 출력에 `kor` 포함 여부 확인
- **재발 방지**: README 또는 setup 스크립트에 traineddata 체크 추가.
  코드 측에서도 시작 시 `pytesseract.get_languages()`로 `'kor' in langs` 검증 후 없으면 명확한 에러 출력.

---

## 4. 수정 후 검증 체크리스트

- [ ] CPU 환경에서도 import/실행이 되는가? (CUDA 분기 누락 X)
- [ ] 같은 화면을 5초 두면 GPU 점유율이 거의 0인가? (frame-skip 작동)
- [ ] 같은 텍스트 두 번 출현 → 두 번째는 즉시 표시되는가? (캐시 작동)
- [ ] OCR이 일시 실패해도 다음 프레임에 다시 시도하는가? (예외 격리)
- [ ] 화면에 표시되는 자막이 사라지면 오버레이 한국어도 사라지는가? (dedup 재구성)

---

## 4-A. 모델 (Translator backend)

### O-1. 선택 OCR 백엔드: PaddleOCR v3 / PP-OCRv5
- **이유**: 2026년 기준 실시간 오버레이 OCR은 Tesseract 단독보다 PP-OCRv5 계열이 더 현실적인 상향 옵션.
  PP-OCRv5는 다국어 인식 모델을 제공하고, 공식 문서 기준 `PaddleOCR(...).predict(image)` 결과에
  `rec_texts`, `rec_scores`, `rec_boxes`가 포함된다.
- **수정**:
  1. UI에 OCR 백엔드 콤보 추가: `Tesseract` / `PaddleOCRv5`.
  2. `PaddleOcrManager` 추가. 사용자가 `PaddleOCRv5`를 선택했을 때만 `from paddleocr import PaddleOCR`.
  3. 모델은 언어별 lazy cache. `src=auto`는 `COCKTAIL_PADDLE_LANG` 또는 기본 `ch` 사용.
  4. PaddleOCR 미설치/초기화/추론 실패 시 Tesseract로 자동 폴백.
  5. `requirements.txt`에는 optional 주석으로만 기록. 기본 설치를 무겁게 만들지 않음.
  6. Windows CPU에서 기본 `paddle_static`/oneDNN 추론이
     `ConvertPirAttribute2RuntimeAttribute not support ... ArrayAttribute`로 실패하는 것을 확인.
     `engine="paddle"`, `enable_mkldnn=False`를 기본값으로 지정해 smoke test 통과.
- **주의**:
  - PaddleOCR v3는 Tesseract처럼 여러 traineddata를 `"eng+kor+jpn"` 형태로 한 번에 묶는 구조가 아니다.
    특정 언어 OCR 품질이 중요하면 source 콤보를 명시 언어로 선택한다.
  - `auto` 번역 감지는 OCR 이후 텍스트에 적용된다. OCR 백엔드가 잘못된 언어 모델로 글자를 못 읽으면
    후단 `identify_language()`가 복구할 수 없다.
  - PaddlePaddle 설치는 CPU/GPU/CUDA 조합별 차이가 크다. 기본 requirements에 강제하지 말 것.
  - `COCKTAIL_PADDLE_ENGINE`으로 엔진을 바꿀 수 있으나, Windows CPU 기본은 `paddle`이 가장 안전했다.
- **재발 방지**:
  - PaddleOCR import를 모듈 상단에 두지 말 것. UI 시작 지연/미설치 ImportError로 앱이 죽는다.
  - 결과 파싱은 `res.json["res"]`와 직접 dict 양쪽을 모두 처리한다. PaddleOCR minor 버전 차이를 흡수하기 위함.
  - 배포 빌드에 포함하려면 `build.spec`에서 optional `collect_all("paddleocr")` / `collect_all("paddle")`가 실제 포함되는지 확인한다.

### M-1. 모델 교체: m2m100_418M → NLLB-200-distilled-600M
- **이유**: en→ko 품질에서 NLLB가 m2m100_418M보다 확연히 우수. 파라미터 600M로 살짝 크지만
  GPU fp16 + 그리디에서 throughput 차이는 무시 가능. CPU 단독 환경이라면 CT2(M-2) 필수.
- **API 차이**:
  - 토크나이저 클래스: `M2M100Tokenizer` → `AutoTokenizer`
  - 언어 코드: `"en"`, `"ko"` → `"eng_Latn"`, `"kor_Hang"` (NLLB는 BCP47-like)
  - forced_bos: `tokenizer.get_lang_id("ko")` → `tokenizer.convert_tokens_to_ids("kor_Hang")`
  - **src_lang은 반드시 토크나이즈 호출 직전에 설정**해야 함 (NLLB도 m2m100과 동일)
- **재발 방지**: 이 항목은 과거 NLLB 실험 기록이다. 현재 구조(M-5 이후)는 ISO 639-1 코드와
  `_LANGDETECT_TO_ISO`/`M2M100_LANGS`를 기준으로 관리한다.

### M-2. CTranslate2 backend (선택, 강력 권장)
- **이유**: CT2 + int8은 GPU에서 ~2-3×, CPU에서 ~3-4× 가속.
  품질 손실은 보통 BLEU 0.5 미만으로 무시할 수 있음.
- **활성화 절차**:
  ```bash
  pip install ctranslate2
  ct2-transformers-converter \
      --model facebook/nllb-200-distilled-600M \
      --output_dir ./nllb_ct2 \
      --quantization int8
  ```
  코드는 `./nllb_ct2/` 폴더 존재 + ctranslate2 import 성공 시 자동으로 CT2 백엔드 선택.
- **fallback**: CT2 초기화 실패 시 HF로 자동 폴백 (try/except로 감쌈).
- **재발 방지**:
  - `Translator` 추상 클래스로 두 백엔드를 동일 인터페이스(`translate_batch`)로 통일.
  - 모델/CT2 디렉터리는 `TRANSLATOR_MODEL` / `TRANSLATOR_CT2_DIR` env로 오버라이드 가능.

### M-3. 토큰 디코드 시 첫 토큰 스킵 (CT2)
- **함정**: CT2의 `target_prefix=[[tgt_nllb]]`는 hypothesis 첫 토큰으로 언어 토큰을 그대로 포함.
  이걸 안 잘라내면 출력 앞에 `kor_Hang` 같은 문자열이 붙음.
- **수정**: `r.hypotheses[0][1:]` 로 첫 토큰 제거 후 decode.

### M-5. NLLB(CC-BY-NC) → opus-mt + m2m100 하이브리드 (상용 라이선스 정리) ⭐
- **이유**: NLLB-200-distilled-600M은 **CC-BY-NC-4.0** 라이선스 → 상용 배포 금지.
  Meta가 takedown 걸 수 있고, 영리 목적 사용은 라이선스 위반.
- **해결**: `HybridTranslator` 도입.
  - 1차: `Helsinki-NLP/opus-mt-tc-big-en-ko` (**CC-BY-4.0** — 2026-08-03 모델카드 실측 정정. 출처 표시 의무). en→ko에서 NLLB-600M과 품질 비슷하고 사이즈 절반, 더 빠름.
  - 2차 폴백: `facebook/m2m100_418M` (MIT). any-to-any 100언어. opus-mt에 등록 안 된 쌍에 사용.
  - 등록 dict: `BILINGUAL_MODELS = {("en","ko"): "Helsinki-NLP/opus-mt-tc-big-en-ko", ...}`.
    자주 쓰는 새 언어쌍은 여기에 opus-mt 모델로 등록 — 자동 fast-path.
- **API 차이 (NLLB → 새 구조)**:
  - 언어 코드: `"eng_Latn"` → `"en"`, `"kor_Hang"` → `"ko"` (ISO 639-1).
  - `LANG_MAP`/`_LANGDETECT_TO_ISO` 둘 다 ISO로 통일.
  - opus-mt: 언어쌍 전용이라 `forced_bos_token_id` 불필요.
  - m2m100: `tok.src_lang = "en"` + `forced_bos_token_id = tok.get_lang_id("ko")`.
- **라이선스 audit 정책**: 새 모델 추가 시 라이선스 먼저 확인. CC-BY-NC, 상용 전용, 미공개 라이선스 금지.
- **재발 방지**: 5초 체크리스트에 항목 추가. `project_commercial_decisions.md` 메모리에 결정사항 영구 기록.

### M-4. 언어 코드 추가/제거 시 관련 매핑 동시 갱신
- **함정**: `LANG_MAP`(UI 라벨→ISO)만 바꾸고 `_LANGDETECT_TO_ISO`/`M2M100_LANGS`/콤보박스 갱신을 빼먹으면
  auto 감지에서 silently fallback="en"으로 떨어지거나 m2m100 라우팅 시점에 실패한다.
- **재발 방지**: 언어 추가 시 `LANG_MAP`, `_LANGDETECT_TO_ISO`, `M2M100_LANGS`, 콤보박스 라벨,
  필요하면 `LANG_TO_TESS`와 인스톨러 traineddata 컴포넌트를 한 번에 검토한다.

### M-6. auto 언어 감지: Unicode script fast-path + langdetect 폴백
- **증상**: `_safe_detect()`가 모든 텍스트를 `langdetect.detect()`에 바로 넘기면 짧은 자막/단어에서 출렁임.
  특히 "감사합니다", "ありがとう", "謝謝", "شكرا", "Спасибо"처럼 Unicode script만으로 식별 가능한 문자열도
  확률 모델을 거쳐 불필요하게 느리고 부정확해질 수 있음.
- **수정**:
  1. `detect_script(text)` 추가 — `unicodedata.name/category`로 HANGUL/HIRAGANA/KATAKANA/CJK/ARABIC/CYRILLIC 등 집계.
  2. `identify_language(text)` 추가 — 비라틴 script는 결정론적으로 `ko/ja/zh/ar/ru/he/hi/th/...` 로 매핑.
  3. `Hvala`, `Merci`, `Hola`, `Danke`처럼 `langdetect`가 자주 틀리는 짧은 라틴 인사/감사 표현은
     `_LATIN_SHORT_HINTS`로 좁게 보정.
  4. 나머지 라틴/불명/기타 script만 `langdetect` 폴백.
  5. `_safe_detect()`는 호환 래퍼로 남기고 내부에서 `identify_language()` 호출.
- **주의**:
  - 순수 한자 라인(예: `謝謝`)은 기본 `zh`. 일본어 한자-only 문장은 오분류 가능.
  - 라틴 짧은 단어는 여전히 본질적으로 어렵다. 흔한 표현은 `_LATIN_SHORT_HINTS`로 보정하지만,
    사전이 커지면 오히려 오분류가 늘 수 있으므로 짧은 인사/감사 표현 위주로만 추가한다.
  - auto 감지가 `ar/ru/hi/th`를 반환해도 OCR 자체는 해당 Tesseract `traineddata`가 설치되어 있어야 읽힌다.
  - 반환 언어는 m2m100이 지원하는 코드여야 한다. 새 매핑을 추가할 때 `M2M100_LANGS` 확인 필수.
- **재발 방지**:
  - 비라틴 문자를 `langdetect`로 먼저 보내지 말 것.
  - 언어 매핑 변경 시 `LANG_MAP`, `_LANGDETECT_TO_ISO`, `M2M100_LANGS`, Tesseract `LANG_TO_TESS`/인스톨러 언어팩을 함께 검토.

### M-7. YOLO식 공간 region 언어 smoothing
- **증상**: OCR 라인은 개별 객체처럼 나오지만, auto 언어 감지를 각 라인에 완전히 독립 적용하면
  짧은 단어/자막 조각이 흔들린다. 좌우에 다른 언어가 동시에 있을 때 한 화면 전체 문맥으로 묶으면
  반대쪽 언어가 오염될 수 있다.
- **수정**:
  1. `_cluster_text_regions(ocr_results)` 추가. bbox 중심/간격으로 좌우 column과 세로 island를 분리.
  2. `_language_confidence(text)` 추가. 비라틴 script, 라틴 힌트, 긴 라틴, 짧은 라틴 순으로 신뢰도 부여.
  3. `assign_region_languages(ocr_results, src_hint)` 추가.
     `src=auto`일 때만 region 안의 강한 언어 신호를 짧고 불안정한 라인에 전파한다.
  4. `batch_translate()`가 라인별 source language list를 받을 수 있게 확장.
  5. 상태 라벨에 이번 프레임에서 감지된 source 언어 요약을 표시.
- **주의**:
  - 사용자가 source 언어를 명시하면 region smoothing은 적용하지 않고 해당 언어를 강제한다.
  - 서로 다른 언어가 같은 bbox region 안에 실제로 섞인 한 줄(`Hello 안녕`)은 여전히 다수 script 기준이다.
  - OCR 자체가 잘못된 모델로 글자를 깨뜨리면 후단 smoothing으로 복구할 수 없다.
- **재발 방지**:
  - 다국어 auto 처리 개선은 화면 전체 전역 추정이 아니라 bbox region 단위로 해야 한다.
  - 짧은 라틴 단어를 무조건 region 언어로 덮어쓰지 말고, `_LATIN_SHORT_HINTS`처럼 확실한 힌트는 보존한다.

### M-8. opus-mt sepvoc 모델의 소스 인코딩 (번역문이 통째로 깨지던 원인) ⭐
- **증상**: en→ko 번역 결과가 문장과 무관한 단어 나열. 예)
  `"Please save your work before closing the window."` → `"☆ 팀 방콕 세그먼트 pest☆ 잘 Autograph."`
- **원인**: `Helsinki-NLP/opus-mt-tc-big-en-ko`는 소스/타깃 vocab이 분리된(sepvoc) Marian 모델인데,
  repo의 `tokenizer_config.json`에 `"separate_vocabs": false`로 **잘못** 적혀 있다.
  그래서 transformers 5.x의 `AutoTokenizer`가 영어 입력을 **타깃(한국어) vocab**으로 인코딩하고,
  `source.spm` 조각 32,000개 중 25,330개가 `<unk>`로 뭉개진다. 모델은 그 쓰레기 입력을 그대로 번역한다.
  디코딩은 타깃 vocab이 맞으므로 정상 — **입력 경로만** 잘못됐다.
- **수정** (`OpusMtTranslator`):
  1. `_load_source_spm()` — `huggingface_hub.snapshot_download(model_name)`(캐시 히트, 재다운로드 없음)로
     스냅샷 경로를 얻고 `source.spm`을 `sentencepiece.SentencePieceProcessor`로 직접 로드.
  2. sepvoc 판정은 설정 파일이 아니라 실측: `source.spm` 조각을 샘플링해 20% 이상이
     `tok.get_vocab()`에 없으면 sepvoc → spm 경로. 공유 vocab 모델이면 `None`(기존 경로 유지).
  3. `_encode_with_spm()` — 라인별 `sp.encode(text)[:255] + [eos]`, `pad_token_id`로 패딩,
     실토큰 1 / 패딩 0의 `attention_mask`를 만들어 device로 올린다(입력은 int64라 fp16과 무관).
  4. `source.spm` 부재/로드 실패는 `[WARN]` 후 기존 토크나이저 경로로 degrade (크래시 금지).
- **주의**:
  - `BILINGUAL_MODELS`에 새 opus-mt 모델을 등록하면 sepvoc 여부가 모델마다 다르다.
    실측 판정 로직이 자동 처리하지만, 첫 로드 로그에 `sepvoc 모델 감지`가 찍히는지 확인할 것.
  - m2m100 폴백 경로는 이 문제와 무관하다(단일 vocab). 건드리지 말 것.
  - 콘솔이 cp949인 Windows에서는 `print`에 em dash(—)를 쓰면 `UnicodeEncodeError`가 난다.
    모델 로드 로그처럼 예외 처리 안쪽에서 터지면 원인 추적이 어려워지므로 ASCII 문장부호를 쓴다.
- **재발 방지**:
  - HF repo의 토크나이저 메타데이터를 믿지 말 것. 번역 품질 이상은 **입력 토큰 id를 먼저 덤프**해
    `<unk>` 비율을 본다 (모델/샘플링 탓으로 돌리기 전에).
  - `smoke_tests.py::test_m8_opus_mt_source_encoding`이 실제 모델로 한 문장을 번역해
    한글 포함 + `<unk>` 부재를 검사한다 (모델이 로컬 HF 캐시에 없으면 자동 skip).

---

### T-1. 시스템 트레이 + 글로벌 단축키 도입 (Phase 1A)
- **목적**: 비전 V-1의 "백그라운드 상주" 기반 — 트레이 상주 + 어떤 창에서든 토글 가능.
- **수정**:
  1. `QSystemTrayIcon` + `QMenu` (열기 / 번역 시작·정지 / 캡처 모드 / 종료).
  2. closeEvent: `_is_quitting=False`면 `event.ignore()` + `hide()` + 트레이 메시지. 트레이 "종료"가 진짜 종료 경로.
  3. `keyboard` 패키지(MIT) optional import — 미설치/등록 실패도 앱은 정상 동작.
  4. 글로벌 단축키 콜백은 keyboard 자체 스레드 → `global_toggle_signal` / `global_show_signal` 시그널로
     메인 스레드 진입 (B-11과 동일 원칙).
  5. 단축키: `Ctrl+Shift+Q` (번역 토글, 2026-08-02 T→Q 변경), `Ctrl+Shift+A` (창 다시 열기).
  6. 진짜 종료 경로에서 `_keyboard.unhook_all_hotkeys()` 호출.
- **재발 방지**:
  - 글로벌 단축키 라이브러리(keyboard, pynput 등)는 모듈 상단 `import` 강제 X.
    설치 실패가 앱 시작을 막으면 안 됨 (U-1과 일관).
  - 트레이 종료 vs 닫기 분기 항상 명시적 플래그(`_is_quitting`)로.
  - QSystemTrayIcon이 동작하지 않는 환경(예: Linux Wayland 일부)에서도 앱이 죽지 않도록 hasattr/isVisible 가드.

### A-1. Phase 6 마무리 — 주기적 save + DPAPI argtypes + 자동 실행 + 마켓 진입 자산
- **목적**: VISION_AND_ROADMAP §7 Phase 6 핵심: LICENSE / 마켓 진입 / 사용자 편의 마무리.
- **수정**:
  1. **주기적 save**: `QTimer` 30초 인터벌 → `PERSIST_CACHE.save()` (변경 없으면 no-op).
     비정상 종료 시 손실 폭 ≤ 30초로 제한.
  2. **DPAPI argtypes**: `_setup_crypt32()` 모듈 레벨 — `CryptProtectData`/`CryptUnprotectData`의
     `argtypes`/`restype` 명시. 64비트 호출 규약 정확성 보장.
  3. **자동 실행**: `winreg`로 `HKCU\Software\Microsoft\Windows\CurrentVersion\Run` 토글.
     트레이 메뉴 "Windows 시작 시 자동 실행" checkable. `_autorun_command()`로 현재 실행 형태에 맞게 등록.
- **자산** (신규 파일):
  - `LICENSE` (MIT)
  - `LICENSE-3RDPARTY.md` (의존성 라이선스 정리, 상용 호환 검증 체크리스트)
  - `CODE_SIGNING.md` (Authenticode 인증서, signtool, GitHub Actions 통합 가이드)
  - `ICONS.md` (아이콘 자산 가이드 + 코드 통합 + 변환 워크플로우)
  - `.github/workflows/release.yml` (태그 push → PyInstaller 빌드 + GitHub Release 생성)
  - `README.en.md` (영어 진입점)
- **한계 (정직)**:
  - 자동 실행 시 `python.exe`로 등록되면 콘솔 창 뜸 — `pythonw.exe` 또는 .exe 빌드 권장.
  - 주기적 save가 메인 스레드 (캐시 작아 ms 단위, 다음 턴 백그라운드 권장).
  - GitHub Actions에 Inno Setup 빌드 단계 없음 — Zip만 자동, 인스톨러는 수동.
  - LICENSE는 MIT로 박사가 결정 — 사용자가 다른 라이선스(Apache 2.0 등) 원하면 즉시 교체 가능.
- **재발 방지**:
  - 자동 실행은 `winreg` Windows 전용 — graceful 비활성 (sys.platform 가드).
  - DPAPI는 정책상 비활성 환경(VM, 도메인)에서도 try/except로 앱이 죽지 않게.
  - 새 의존성 추가 시 LICENSE-3RDPARTY.md 즉시 갱신.

### W-3. 멀티 모니터 대응 (Phase 1D)
- **문제**: W-2의 OverlayWindow가 primary screen만 덮음. 보조 모니터에 활성 창이 있으면 빨간 박스/번역이
  안 그려지거나 음수 좌표로 잘못 표시.
- **수정**:
  1. `OverlayWindow._fit_to_screen()` 변경 — `QApplication.screens()` 모두 합집합 bounding rect로 setGeometry.
  2. paintEvent에서 화면 절대 좌표(`bbox`) → 윈도우 내 좌표 변환 추가:
     `local = bbox - self.geometry().topLeft()`.
  3. ControlWindow.moveEvent / resizeEvent / mouseMoveEvent에서 `overlay.update()` 호출 — W-2 자체 리뷰 보강.
- **재발 방지**:
  - 새 윈도우 추가 시 가상 데스크톱 좌표(보조 모니터 음수 가능) 전제로 설계.
  - 화면 절대 좌표를 그릴 때는 항상 `self.geometry().topLeft()` 빼기.

### CR-1. 2026-05-04 크리티컬 리뷰 수정 — 워커 시작/OverlayWindow 일관성/UIA M-7
- **증상**:
  1. `translation_thread` 생성/시작 코드가 `TransparentWindow.__init__` 본문이 아니라
     `_on_cache_save_tick()` 내부로 잘못 들어가 있었다. 앱 시작 후 30초까지 워커가 없고,
     그 전에 진짜 종료하면 `self.translation_thread` 접근에서 예외 가능.
  2. `스샷OK/스샷차단` 토글이 ControlWindow에만 적용되고 실제 렌더링 담당인 OverlayWindow에는 적용되지 않았다.
     사용자는 스크린샷 허용이라고 보지만 번역 오버레이가 캡처에서 빠질 수 있었다.
  3. 활성 창/민감 창 가드가 ControlWindow hwnd만 제외하고 OverlayWindow hwnd는 일부 경로에서 제외하지 않았다.
  4. UIA 경로는 OCR 경로와 달리 `assign_region_languages()`를 거치지 않아 M-7 region smoothing이 빠져 있었다.
- **수정**:
  1. `translation_thread` 생성/시작과 `old_pos` 초기화를 `__init__` 본문으로 복귀.
     `_on_cache_save_tick()`은 `PERSIST_CACHE.save()`만 수행.
  2. `toggle_capturable()`이 ControlWindow와 OverlayWindow 둘 다 `SetWindowDisplayAffinity`를 동일 상태로 변경.
  3. `_foreground_window_rect()`와 `_check_sensitive_pause()`에서 OverlayWindow hwnd도 자기 창으로 간주해 제외.
  4. `_uia_collect_and_translate()`에서 `assign_region_languages(uia_results, src_hint)`를 적용 후
     `batch_translate(texts, src_langs, tgt_lang)` 호출.
- **재발 방지**:
  - QTimer slot 아래에 새 초기화 코드를 추가할 때 들여쓰기 검토. 특히 스레드 시작, 단축키 등록, 상태 초기화는
    `__init__` 본문에 있어야 한다.
  - 컨트롤/오버레이 2윈도우 구조에서는 OS 레벨 상태(capture affinity, hwnd guard, mouse-through)를 항상 양쪽 기준으로 검토한다.
  - 새 입력 경로(OCR/Paddle/UIA 등)를 추가하면 언어 감지 후처리(M-6/M-7), 캐시, 번역 라우팅이 같은 정책을 타는지 확인한다.

### ST-1. 다음 step 전 자동 smoke test
- **목적**: CR-1 같은 들여쓰기/윈도우 분리 일관성 결함은 수동 리뷰만으로 재발 가능하다.
  다음 개발 단계로 넘어가기 전 빠른 자동 검사를 고정한다.
- **수정**:
  1. `smoke_tests.py` 추가.
  2. 빠른 기본 검사:
     - `translation_thread`가 `__init__`에 있고 `_on_cache_save_tick()`에는 없는지 AST 검증.
     - `toggle_capturable()`이 ControlWindow와 OverlayWindow 둘 다 바꾸는지 검증.
     - 활성 창/민감 창 가드가 OverlayWindow hwnd도 제외하는지 검증.
     - UIA 경로가 `assign_region_languages()`를 쓰는지 검증.
     - M-7 region smoothing 샘플 검증.
     - `PersistentCache._key()`가 sha256 hash이고 평문을 담지 않는지 검증.
  3. `--paddle` 옵션으로 느린 PaddleOCR 실제 추론 smoke test 제공.
- **재발 방지**:
  - 다음 step 시작/종료 전 `python smoke_tests.py`를 실행한다.
  - PaddleOCR 모델/엔진을 건드린 날에는 `python smoke_tests.py --paddle`까지 실행한다.
  - smoke test가 실패하면 기능 추가를 중단하고 회귀부터 수정한다.

### A-2. Phase 6 보강 — 자동 실행 콘솔 창 회피 + 캐시 save 백그라운드화
- **증상/위험**:
  1. 개발 환경에서 Windows 자동 실행을 등록하면 `python.exe Cocktail완성본.py` 형태가 되어 부팅 시 콘솔 창이 뜰 수 있다.
  2. `QTimer` 30초 save가 메인 스레드에서 DPAPI 암호화/파일 write를 수행하면 캐시가 커질 때 UI 끊김이 생길 수 있다.
  3. 워커 스레드가 `PersistentCache.put/get` 중일 때 save가 동시에 `_mem`을 직렬화하면 경쟁 조건 가능성이 있다.
- **수정**:
  1. `_autorun_command()`가 `.py` 실행 형태에서는 같은 Python 폴더의 `pythonw.exe`를 우선 사용.
     없을 때만 현재 `python.exe`로 폴백. PyInstaller `.exe`는 기존대로 exe 자체 등록.
  2. `_on_cache_save_tick()`은 `_cache_save_in_progress` 가드 후 daemon thread에서 `PERSIST_CACHE.save()` 수행.
  3. `PersistentCache`에 `_mem_lock = threading.RLock()` 추가. `_load/save/get/put`의 `_mem` 접근을 보호.
  4. save 중 새 put이 들어오면 `snapshot == self._mem`일 때만 `_dirty=False`로 내려 새 변경 손실을 막음.
  5. `smoke_tests.py`에 `test_autorun_prefers_pythonw_for_python_script`,
     `test_cache_timer_saves_in_background` 추가.
- **재발 방지**:
  - UI 타이머에서 파일 I/O/암호화/모델 작업을 직접 하지 말 것. 짧은 상태 갱신만 허용.
  - 자동 실행/시작 프로그램 등록은 사용자가 보는 창 개수를 기준으로 검토한다. 개발 실행이면 `pythonw.exe`, 배포면 `.exe`.
  - 공유 dict를 저장/수정하는 코드는 lock과 dirty semantics를 함께 검토한다.

### R-1. 릴리즈 워크플로우 배포 순서 보강
- **증상/위험**:
  1. GitHub Actions Windows runner에는 Tesseract OCR 엔진이 기본 설치되어 있지 않을 수 있다.
     `diagnose_environment.py`를 먼저 실행하면 CI 릴리즈가 진단 단계에서 실패한다.
  2. 코드 서명 step이 zip 이후면 릴리즈 zip 안에 서명 전 `Cocktail.exe`가 들어갈 수 있다.
  3. PyInstaller dist에 README/LICENSE/라이선스 고지 문서가 없으면 배포 zip 사용자가 라이선스/사용법을 확인하기 어렵다.
- **수정**:
  1. `.github/workflows/release.yml`에 `Install Tesseract OCR` step 추가.
     `choco install tesseract` 후 `TESSERACT_CMD`를 `GITHUB_ENV`에 기록.
  2. 릴리즈 워크플로우 순서: 진단/smoke/pre-release → tessdata 다운로드 → PyInstaller → 서명(optional) →
     Inno Setup 설치/인스톨러 빌드 → zip → artifact/release 업로드.
  3. `build.spec`에 README, LICENSE, LICENSE-3RDPARTY, BUILD, CODE_SIGNING, UPDATE_LOG, TECH_STACK,
     VISION_AND_ROADMAP 문서를 datas로 포함.
  4. `pre_release_check.py` 추가. 필수 파일, installer version, build.spec docs, release workflow 순서,
     의존성/라이선스 정책, 문서 링크, 생성 디렉터리 여부를 검사.
  5. ST-1 실행이 만드는 `__pycache__` 때문에 pre-release가 실패하지 않도록
     `pre_release_check.py`에서 `__pycache__`는 자동 정리하고 `build/` 같은 blocking 생성물만 실패 처리.
  6. 로컬에서 `pyinstaller` 명령이 PATH의 다른 Python(예: Python 3.12)을 사용할 수 있음을 확인.
     BUILD.md와 release workflow를 `python -m PyInstaller --noconfirm build.spec` 기준으로 변경해
     현재 앱 의존성이 설치된 Python 환경에서 빌드하도록 고정.
- **재발 방지**:
  - 릴리즈 zip을 만들기 전 서명/문서 포함 여부를 확인한다.
  - CI에서 실행하는 진단이 요구하는 외부 바이너리(Tesseract/Inno 등)는 workflow에서 먼저 설치한다.
  - 배포 관련 파일을 바꾸면 `python pre_release_check.py`를 반드시 실행한다.

### R-2. 배포 용량/패키지 자기검증 보강
- **증상/위험**:
  1. 개발 PC에 PaddleOCR/PaddlePaddle이 설치되어 있으면 기본 `build.spec`가 선택 OCR 백엔드까지 자동 번들링한다.
     일반 배포본이 너무 커지고, 사용자가 원하지 않는 선택 엔진까지 포함될 수 있다.
  2. PyInstaller의 `collect_all("torch")`가 런타임에 불필요한 `.lib` 및 `include` 개발 산출물을 포함해
     Windows 배포 용량이 크게 증가한다.
  3. zip 배포물은 `dist/Cocktail`만 압축하므로, 다운로드된 `tessdata/*.traineddata`가 PyInstaller datas에 없으면
     인스톨러가 아닌 zip 사용자에게 OCR 언어 데이터가 빠질 수 있다.
  4. CI가 소스 smoke test까지만 실행하면 실제 `dist/Cocktail/Cocktail.exe`가 기본 의존성을 갖고 시작 가능한지 확인하지 못한다.
- **수정**:
  1. `build.spec`에 `COCKTAIL_BUNDLE_PADDLE` 게이트 추가. 기본값은 `0`이며, `1`일 때만 PaddleOCR/PaddlePaddle을 포함한다.
  2. 기본 빌드에서는 `paddle`, `paddleocr`, `modelscope`를 PyInstaller excludes에 추가한다.
  3. PyTorch 수집 결과에서 `.lib`와 `torch/include`를 필터링한다. 런타임 DLL은 유지하고 개발용 파일만 제거한다.
  4. `tessdata/*.traineddata`가 존재하면 PyInstaller dist의 `tessdata/`에 포함해 zip과 installer가 같은 OCR 언어 데이터를 갖게 한다.
  5. `Cocktail완성본.py --self-test` 추가. GUI/QApplication 및 번역 모델 로드 없이 필수 모듈, Tesseract, `eng.traineddata`, 캐시 키 규칙을 검사한다.
  6. GitHub Actions가 PyInstaller 후 `dist\Cocktail\Cocktail.exe --self-test`를 실행하도록 변경한다.
  7. `smoke_tests.py`와 `pre_release_check.py`에 R-2 정책 검사를 추가한다.
- **재발 방지**:
  - 선택 백엔드를 기본 배포물에 넣을 때는 환경변수/릴리즈 이름/라이선스 문서까지 함께 바꾼다.
  - 빌드 용량이 튀면 `_internal`의 상위 디렉터리와 `torch/lib` 파일 크기를 먼저 확인한다.
  - zip 배포도 installer와 동일하게 `tessdata/` 포함 여부를 확인한다.
  - 배포 전에는 소스 smoke test와 패키지 self-test를 둘 다 실행한다.

### UIA-1. UIA 어댑터 + 활성 창 텍스트 직접 추출 (Phase 2/3)
- **목적**: 비전 V-1 — OCR 없이 UI 텍스트 직접 읽기. 시작 메뉴/설정/Office/브라우저 등 정확/빠름.
- **수정**:
  1. `UIATextProvider` — `uiautomation` lazy import + 미설치/실패 시 graceful 비활성.
  2. `_TEXTUAL_CONTROL_TYPES` 화이트리스트 (Text/Document/Edit/Button/Menu/TreeItem 등).
  3. ocr_combo에 "UIA (활성 창)" 항목 (`UIA_PROVIDER.is_available()`일 때만).
  4. 워커 루프 진입부에서 UIA 모드면 `_uia_collect_and_translate()` 호출 → 결과 있으면 OCR 건너뛰고 emit.
  5. UIA bbox는 화면 절대 좌표(BoundingRectangle) — OverlayWindow가 그대로 그림.
  6. 박사 자체 리뷰 보강: UIA 모드는 mode_combo와 무관하게 항상 민감 창 검사 (보안 원칙 #3).
  7. UIA 결과 100개 상한 — paintEvent 부담 회피.
  8. `requirements.txt`에는 추가 X (선택). `requirements-dev.txt`에 PoC용으로만 명시.
- **한계 (정직)**:
  - 게임/Custom GDI는 UIA 노출 X — Tesseract/PaddleOCR로 폴백.
  - UIA 트리 walk 50~200ms — 워커 스레드라 UI 응답엔 영향 X, 단 활성 창 변화 빈도 낮춤(0.5초 sleep).
  - 동적 UI(애니메이션) 따라잡기 어려움.
- **재발 방지**:
  - UIA 결과 수 상한 (현재 100). 폰트 측정 비용 × N 폭증 방지.
  - 비-Windows / uiautomation 미설치 → graceful 비활성. 절대 시작 예외 X.

### D-1. DPAPI 암호화 영구 캐시 (Phase 4, 보안 원칙 #2 강화)
- **목적**: 메모리 LRU(P-5)는 앱 재시작하면 사라짐. 자주 쓰는 라벨이 매번 새 번역 → UX 저하.
  디스크 캐시를 두되 평문 저장 금지 (보안 원칙 #2).
- **수정**:
  1. `dpapi_protect/unprotect()` — Windows DPAPI(`crypt32.CryptProtectData`/`CryptUnprotectData`).
     사용자 윈도우 계정으로만 복호화 가능. 다른 계정/머신 X.
  2. `PersistentCache` — 메모리 dict + 디스크 파일.
     - 키: `sha256(src|tgt|text)`의 hex (평문 source 미저장).
     - 값: 번역 결과 (전체 dict를 DPAPI로 암호화해서 저장).
     - 위치: `~/.cocktail/translation_cache.bin`.
     - max_entries 4096. 초과 시 절반 비움 (LRU는 메모리 LRU에 위임).
  3. `batch_translate`: 메모리 캐시 miss 시 디스크 캐시 조회 → hit이면 메모리로 승격.
     번역 성공 시 메모리+디스크 둘 다 put.
  4. `closeEvent`에서 `PERSIST_CACHE.save()` (DPAPI 암호화 후 atomic replace).
  5. **lazy load** (박사 자체 리뷰 보강) — `__init__`에서 디스크 I/O 안 함. `preload_async()`로 백그라운드 또는 첫 get/put 시 동기 로드.
- **한계 (정직)**:
  - 비-Windows: `_enabled=False`로 무동작 (앱 정상). DPAPI는 Windows 전용.
  - 비정상 종료 시 save 호출 안 됨 — 다음 단계에 주기적 save (Timer로 30초마다 등) 또는 N건마다.
  - DPAPI는 사용자 계정 단위 — 같은 PC라도 다른 계정에서 복호화 불가. 정책상 OK (비전 V-1).
- **재발 방지**:
  - 키는 항상 hash. 평문 source/target 단어 디스크 저장 금지.
  - DPAPI 호출은 try/except로 감싸기 — 정책상 비활성된 환경(VM, 도메인 정책)에서 앱이 죽지 않게.
  - `__init__`에서 디스크 I/O 절대 금지 (P-9 정신).

### W-2. OverlayWindow 분리 (Phase 1C) — 컨트롤 ↔ 오버레이 윈도우 분리
- **문제**: Phase 1A/1B의 한 윈도우 구조에선 마우스 통과 ON 시 컨트롤도 클릭 못 함.
  사용자가 토글을 잊으면 작동 X. 진짜 매끄러움은 윈도우 분리.
- **수정**:
  1. `OverlayWindow(QMainWindow)` 신규 — 전체 화면 + 투명 + 항상 위 + `Qt.Tool`(작업표시줄 안 뜸) +
     `WA_ShowWithoutActivating`(포커스 도둑질 X) + 마우스 통과 영구 ON.
  2. B-10 캡처 차단(`WDA_EXCLUDEFROMCAPTURE`)을 OverlayWindow에도 적용 — 자기 캡처 피드백 차단.
  3. `paintEvent`에서 빨간 박스 + 번역 모두 그림. 좌표는 화면 절대(OverlayWindow는 (0,0)이 화면 (0,0)).
  4. `_fit_font` + `_draw_one`을 OverlayWindow로 이전 (B-12/B-13 로직 그대로).
  5. 워커의 `adjusted_bbox`를 위젯 좌표 → **화면 절대 좌표**로 변경 (`bbox + x1, bbox + y1`).
     x1/y1은 워커가 이미 화면 절대로 계산하던 값.
  6. TransparentWindow.paintEvent는 빈 super 호출만 (그리기 위임 완료).
  7. ControlWindow ↔ OverlayWindow 연결: `window.overlay = overlay` + 모드/상태 변경 시 `overlay.update()`.
  8. 종료: ControlWindow.closeEvent가 진짜 종료 경로일 때 OverlayWindow도 close.
- **이득**:
  - 사용자가 마우스 통과 걱정 없이 다른 창 자유롭게 클릭. 컨트롤은 ControlWindow에서 클릭.
  - 비전 V-1의 매끄러운 작동에 한 발 더 — Phase 3 UIA 통합 시 OverlayWindow가 표시 책임 일관 유지.
- **한계 (정직)**:
  - 멀티 모니터: 현재 primary screen에만. 보조 모니터 영역에 활성 창이 있으면 오버레이가 따라가지 않음.
    → Phase 1D에서 멀티 모니터 대응 (각 모니터별 OverlayWindow 또는 통합 가상 데스크톱).
  - W-1 마우스 통과 토글은 그대로 둠. 사용자가 OverlayWindow가 아닌 ControlWindow의 통과를 잘못 토글할 수 있음 — 다음 정리.
- **재발 방지**:
  - OverlayWindow 추가 시 `setWindowFlags`에 `Qt.Tool` + `WindowStaysOnTopHint`를 항상 같이.
  - `WA_ShowWithoutActivating` 빠뜨리면 시작 시 포커스 도둑질 발생.
  - 워커가 저장하는 좌표계를 변경할 때 모든 소비자(현재는 OverlayWindow만)를 동시에 갱신.

### W-1. 마우스 통과 토글 (Phase 1B) — 활성 창/hover 모드의 매끄러운 작동
- **문제**: T-2로 활성 창/hover 모드를 도입했으나, 우리 윈도우가 그 영역 위에 있으면 사용자 클릭이
  우리에게 갇혀 다른 창을 조작할 수 없음. 윈도우를 분리하지 않고 한 윈도우 안에서 해결 필요.
- **수정**: Win32 `WS_EX_TRANSPARENT` 스타일을 토글하는 `set_mouse_through(hwnd, on)` 헬퍼 추가.
  - ON: 우리 윈도우 영역 전체에서 마우스 이벤트가 밑 창으로 통과. 윈도우는 여전히 그려짐.
  - OFF: 다시 일반 클릭 가능 (컨트롤바/모드 콤보 등 클릭).
  - `WS_EX_LAYERED`는 항상 유지 — 반투명 렌더링이 깨지면 안 됨.
- **UI**: 트레이 메뉴 "마우스 통과" checkable + 단축키 `Ctrl+Shift+P`.
- **운용 규칙**: 활성 창 모드에서 ON으로 두고 사용. 컨트롤 만지려면 잠깐 OFF.
  진정한 매끄러움은 Phase 1C에서 컨트롤 윈도우 분리 시 달성될 예정.
- **재발 방지**:
  - 마우스 통과 토글이 활성된 상태로 사용자가 잊고 종료하지 않도록 트레이 메뉴 checkable로 노출.
  - `WS_EX_LAYERED`를 항상 OR로 유지. 빼면 PySide6 `WA_TranslucentBackground`와 충돌해 검정 배경 발생 위험.

### S-1. 민감 영역 자동 OFF (Phase 5, 보안 원칙 #3)
- **목적**: VISION_AND_ROADMAP.md §5 "민감 영역 자동 OFF". 패스워드 입력 창/결제 페이지/은행 앱 활성 시
  화면을 OCR로 읽지 않음. 100% 로컬 처리라도 비밀번호/카드번호를 OCR/캐시에 노출하지 않는 게 안전.
- **수정**:
  1. `SENSITIVE_KEYWORDS` 키워드 사전 — 영문(password/login/checkout/payment/cvv/2fa 등) +
     한국어(비밀번호/로그인/결제/공인인증/카드번호 등) + 한국 주요 은행/금융앱 이름.
  2. `is_sensitive_window(hwnd)` — `GetWindowText` + `GetClassName` 결과의 lowercase haystack에서
     키워드 부분 일치(`in` 검색).
  3. `_check_sensitive_pause()` — 메서드. 박스 모드는 적용 X (사용자가 박스를 직접 지정한 경우는 의도가 명확).
  4. 워커 루프 진입부에서 `_check_sensitive_pause()` 호출 → True 면 OCR/번역 건너뛰고 1초 sleep + 상태 라벨로 알림.
  5. 해제 시 "민감 창 해제 — 번역 재개" 메시지로 사용자에게 명확히.
- **한계 (정직)**:
  - 키워드 부분 일치는 false positive 가능 (예: "login" 단어가 다른 맥락의 글에 있는 창).
  - 키워드 사전이 사용자 환경 은행/앱을 다 커버하지 못함 — 사용자 추가 가능하게 추후 config 파일 분리.
  - 활성 창 안의 비밀번호 입력 *필드* 단위 감지가 아님 — 창 단위. 더 정밀하게 가려면 UIA로
    `IsPassword` 컨트롤 속성 검사 (Phase 3에서 추가 가능).
- **재발 방지**:
  - 새 보안 휴리스틱 추가 시 박스 모드 영향 검토. 사용자가 이미 명시적으로 박스 지정한 케이스는 보수적으로.
  - `SENSITIVE_KEYWORDS` 갱신 시 lowercase 통일 + 한/영 보편 표현 우선.
  - 보안 5원칙 #3 위반 의심 시 즉시 사용자 보고.

### T-2. 캡처 모드 다양화 — 영역 박스 / 활성 창 / 마우스 hover (Phase 1A)
- **목적**: 비전 V-1 — "어디든 자동 번역"의 첫 발. 빨간 박스 한 곳 고정에서 벗어남.
- **수정**:
  1. 모드 상수 `MODE_BOX`, `MODE_WINDOW`, `MODE_HOVER` + UI 콤보 + 트레이 메뉴.
  2. `_foreground_window_rect()`: `GetForegroundWindow` + `GetWindowRect`. 우리 창(`int(self.winId())`) 이면 None 반환 (B-10 피드백 차단).
  3. `_mouse_hover_rect()`: `QCursor.pos()` ± `hover_size` (기본 400×200).
  4. `_update_red_rect_for_mode()`: 워커 루프 매 사이클 호출. 모드별 절대 좌표 → 위젯 내 red_rect 좌표로 변환.
     박스 모드는 그대로 (사용자 수동).
  5. 모드 전환 시 `last_frame_hash = None` 으로 frame-skip 캐시 무효화.
- **한계 (정직)**:
  - 활성 창 모드: 우리 창 영역을 벗어난 활성 창이면 오버레이가 우리 창 안에서만 보임.
    완벽히 매끄러우려면 윈도우를 전체 화면으로 키우고 `Qt.WA_TransparentForMouseEvents` 부분 적용 필요 (Phase 1B).
  - 마우스 hover 모드: 마우스가 우리 창 밖이면 동일 한계.
- **재발 방지**:
  - 새 모드 추가 시 `_MODE_LABELS` + 트레이 메뉴 + `_update_red_rect_for_mode()` 분기 4곳을 한 번에.
  - 활성 창/hover의 절대좌표 → 위젯 좌표 변환은 항상 `self.geometry().x()/y()` 빼기.

### W-4. MODE_WINDOW 캡처 영역을 primaryScreen으로만 클립 (보조 모니터 영구 정지)
- **증상**: 활성 창이 보조 모니터에 있으면 번역이 아예 안 나옴. 상태 메시지도 없음.
- **원인**: UX-2 Layer 2 클립이 `QApplication.primaryScreen()` **한 화면**만 기준.
  보조 모니터 창의 절대 좌표는 primary 화면 밖 → 교집합 0 → 매 사이클 `continue` 무한 반복.
  게다가 `continue` 전에 오버레이를 비우지 않아 직전 자막이 화면에 그대로 남음.
- **수정**: `QApplication.screens()` 전체의 합집합(가상 데스크톱)으로 클립.
  `right()/bottom()`은 Qt 규약상 마지막 픽셀이라 `+1` 해서 exclusive 경계로 맞춤.
  교집합 8px 미만이면 `continue` 전에 `if self.translated_text: self.state_ready.emit([])`
  (B-15 / S-1 등 다른 일시정지 경로와 같은 패턴).
- **재발 방지**: 화면 좌표를 클립할 땐 항상 가상 데스크톱 기준. `primaryScreen()` 단독 사용 금지.
  일시정지/스킵 경로를 추가할 땐 오버레이 비우기를 반드시 동반 (잔상 방지).

### W-5. ImageGrab이 보조 모니터를 검은 이미지로 캡처
- **증상**: 보조 모니터 영역 OCR 결과가 0줄. 저장해 보면 전부 검정.
- **원인**: `ImageGrab.grab(bbox=...)`의 `all_screens` 기본값이 False.
  PIL win32 경로(`PIL/ImageGrab.py`)는 `grabscreen_win32(..., all_screens, ...)`가 돌려준
  `offset` 기준으로 `im.crop((left-x0, top-y0, ...))` 한다. False면 primary만 캡처 + `offset=(0,0)`
  이라 primary 밖 좌표는 이미지 밖 → 검정.
- **수정**: `ImageGrab.grab(bbox=(x1,y1,x2,y2), all_screens=True)`.
  True면 가상 데스크톱 전체를 찍고 `offset`이 가상 화면 원점이라 **절대 좌표 bbox가 그대로 맞는다**.
- **재발 방지**: 화면 캡처는 항상 `all_screens=True`. 호출처가 늘면 한 곳으로 모을 것 (현재 워커 1곳).

### W-6. 마우스 통과 토글 플래그 ↔ 실제 창 상태 불일치 (첫 토글 무효)
- **증상**: `Ctrl+Shift+P` 첫 번째 입력이 아무 효과 없음. 두 번째부터 동작.
- **원인**: `SettingsWindow._mouse_through` 초기값이 `False`인데, 실제로는
  `OverlayWindow.update_for_mode()`가 시작 시점과 모드 전환 때마다 통과를 **항상 ON**으로 강제.
  플래그와 창 상태가 어긋나 첫 토글이 "OFF→ON"(이미 ON)으로 낭비됨.
- **수정**: 상태 단일 출처화.
  1. 초기값 `True` (실제 적용 상태와 일치) + 트레이 액션 `setChecked(self._mouse_through)`.
  2. `_sync_mouse_through(on)` 헬퍼 — 플래그 + 트레이 체크 동시 갱신.
  3. `set_capture_mode()`에서 `overlay.update_for_mode()`(통과 ON 강제) 직후 `_sync_mouse_through(True)`.
- **재발 방지**: 창 상태를 강제로 바꾸는 경로를 추가하면 반드시 같은 자리에서 토글 플래그를 동기화.
  통과 OFF는 오버레이가 가상 데스크톱 전체 클릭을 삼키는 상태라 사용자가 인지 가능해야 함(트레이 체크 표시).

### UX-3. "번역 정지" 후 오버레이 자막 잔상
- **증상**: 정지 버튼/`Ctrl+Shift+T`/트레이 토글로 멈춰도 마지막 번역 박스가 화면에 계속 떠 있음.
- **원인**: `set_running(False)`는 워커 루프만 멈출 뿐, `translated_text`를 비우지 않음.
  워커가 멈췄으니 `state_ready`도 더는 안 나와 오버레이가 마지막 프레임 상태로 고정.
- **수정**: `BackgroundController.set_running()`에서 `value=False`이고 남은 상태가 있으면
  `self.state_ready.emit([])`. 버튼/글로벌 단축키/트레이 메뉴가 모두 `toggle_running` →
  `set_running` 단일 경로라 한 곳 수정으로 전 경로 커버.
- **재발 방지**: 파이프라인을 멈추는 새 경로를 만들면 오버레이 비우기를 같이 넣을 것.

### F-1. frame-skip 해시가 둔감 (자막 한 줄 변화 / 창 이동 놓침)
- **증상**: 자막이 바뀌었는데 번역이 안 갱신됨. 활성 창을 옮기면 옛 좌표에 자막이 남음.
- **원인 2가지**:
  1. `hash_distance`가 16×16 그레이의 **평균** 차이 < 2면 스킵. 256셀 중 1~2셀만 바뀌는
     자막 한 줄 변화는 평균 ≈1이라 통째로 삼켜짐.
  2. 해시가 **내용만** 반영. MODE_WINDOW에서 창 위치만 바뀌면 내용 해시가 같아 스킵 →
     자막이 예전 좌표에 그대로 렌더링.
- **수정**:
  - (a) `cheap_hash(img, rect)` — 캡처 영역 좌표 4개(int32, 16바이트)를 해시 앞에 붙임.
    헤더가 다르면 `hash_distance`가 즉시 `1<<30` 반환(무조건 재처리).
  - (b) 평균 → **셀 단위 최대 변화량**(`.max()`)으로 교체. 임계는 상수 `FRAME_DIFF_THRESHOLD = 8`
    (튜닝 레버 — 노이즈로 과잉 재처리면 올리고, 반응이 둔하면 내림).
  - 부수: `cheap_hash`의 `cv2.COLOR_BGR2GRAY` → `COLOR_RGB2GRAY` (입력이 PIL RGB, B-8 규약 위반이었음).
- **가드**: `smoke_tests.py::test_f1_frame_skip_local_change` — 1셀만 바뀐 이미지(구 평균 기준 <2)와
  영역만 이동한 경우 둘 다 임계 이상으로 판정되는지 확인.
- **재발 방지**: frame-skip 판정은 "내용 + 캡처 영역" 둘 다. 평균 기반 지표는 국소 변화를 못 본다.

### F-2. 폰트 크기 픽셀/포인트 혼동 (자막 박스 1.2배 과대)
- **증상**: 번역 박스 글자가 원문보다 크게 나와 인접 줄과 겹침. 축소 루프가 자주 발동.
- **원인**: OCR bbox에서 얻은 값은 **픽셀** 높이인데 `QFont("Arial", size)`의 두 번째 인자는 **pointSize**.
  96dpi에서 1pt ≈ 1.333px이라 실제로 약 1.2~1.33배 커짐.
- **수정**: `_fit_font`에서 `QFont("Arial")` 생성 후 `font.setPixelSize(size)`.
  초기 크기 계산(`font_size * 0.9`)과 9까지 줄이는 축소 루프 로직은 그대로 유지.
- **재발 방지**: OCR/화면 좌표에서 온 수치를 Qt 폰트에 넘길 땐 항상 `setPixelSize`. 생성자 2번째 인자는 pt.

### SC-1. 활성 창 모드가 전 모니터를 캡처 (오버레이가 모든 모니터에 꽉 참)
- **증상**: 오버레이 박스가 모든 모니터를 뒤덮고 중구난방으로 뜸. W-5(all_screens=True)로
  보조 모니터가 실제로 찍히기 시작하면서 표면화.
- **원인**: `MODE_WINDOW`는 `GetWindowRect(GetForegroundWindow())`를 그대로 캡처 영역으로 쓴다.
  바탕화면(Progman/WorkerW)이나 가상 데스크톱 전체를 덮는 창이 활성이면 그 rect가
  **모든 모니터 합집합** → 전 화면 OCR(수십~수백 줄) + 전 화면 렌더링.
  W-4 클립도 "가상 데스크톱 합집합" 기준이라 이 경우 아무것도 잘라 내지 못했다.
- **수정**: `CaptureRegionController.update_for_mode()`에서 MODE_WINDOW일 때 창 rect를
  **활성 창 중심점이 속한 모니터 1개**(`CaptureRegionController.screen_rect_at()` →
  `QGuiApplication.screenAt()`, 실패 시 `primaryScreen()`)로 교집합.
  Qt `right()/bottom()`은 마지막 픽셀이라 `+1`(W-4와 같은 규약).
  `MODE_BOX`(사용자 영역)는 클램프 전에 return하므로 무영향, `MODE_HOVER`도 무영향.
  W-5(`all_screens=True`)와 W-4(합집합 클립)는 그대로 — 캡처 능력이 아니라 **영역**만 좁힌 것.
- **튜닝 레버**: `CLAMP_WINDOW_TO_ACTIVE_SCREEN = True`. 창을 두 모니터에 걸쳐 쓰는
  사용자는 False로 구 동작 복귀.
- **가드**: `smoke_tests.py::test_sc1_active_screen_clamp`.
- **재발 방지**: "활성 창"류 영역은 창 rect를 그대로 믿지 말 것. 셸/전체화면 창이 가상
  데스크톱 전체를 반환한다.

### LD-1. 모델 로드 69초 — 로컬 캐시가 있는데 매번 HF Hub 네트워크 체크
- **증상**: 앱 시작 → 첫 번역까지 78초. 로그에서 `opus-mt 로드 중` 직후
  `Warning: You are sending unauthenticated requests to the HF Hub` → 긴 정지,
  `Fetching 14 files` (snapshot_download)까지 매번 발생. 모델은 이미 로컬 캐시에 있었다.
- **원인**: `from_pretrained` / `snapshot_download` 기본 동작이 캐시가 있어도 Hub에
  파일 목록·리비전을 물어본다. 네트워크가 느리거나 rate-limit이면 그대로 대기.
- **수정**: `_load_local_first(loader, model_name, **kw)` 헬퍼 —
  `local_files_only=True`로 먼저 시도, 캐시 없음/불완전으로 예외가 날 때만 네트워크 재시도.
  적용처 4곳: opus-mt `AutoTokenizer`/`AutoModelForSeq2SeqLM`, m2m100 `M2M100Tokenizer`/모델,
  M-8 `snapshot_download`. **M-8 sepvoc 판정/인코딩 로직 자체는 미변경**(경로만 로컬 우선).
- **금지**: `HF_HUB_OFFLINE` 등 전역 오프라인 강제 — 첫 설치(캐시 없음)를 통째로 망가뜨린다.
  반드시 try/except 폴백으로.
- **가드**: `smoke_tests.py::test_ld1_local_first_model_load`.
- **재발 방지**: 모델/파일 로드를 새로 추가하면 `_load_local_first`를 경유시킬 것.

### RT-1. OCR 미세 흔들림으로 캐시 miss → 재번역 폭주
- **증상**: 같은 정적 문서인데 1~3초마다 36~57줄을 계속 재번역(300~1300 ms/사이클). GPU/CPU 상시 점유.
- **원인**: 번역 캐시 키가 OCR 원문 **그대로**. Tesseract는 같은 화면도 프레임마다 공백/줄바꿈이
  미세하게 달라져 (`"A  B"` ↔ `"A B"`) 매번 새 키 → LRU/디스크 캐시 전부 miss.
- **수정**: `_cache_key_text()` — 연속 공백 collapse + strip만 하는 보수적 정규화.
  메모리 LRU와 `PERSIST_CACHE` 키를 모두 정규화 텍스트로 통일(`pending_keys`).
  **모델 입력·표시용 원문/번역문은 원본 유지** (배치/라우팅/sepvoc 경로 미변경).
  문자 자체를 손대는 정규화(소문자화, 유니코드 NFKC 등)는 다른 언어를 망칠 수 있어 하지 않음.
- **재발 방지**: OCR에서 온 텍스트를 키로 쓸 땐 반드시 정규화 텍스트. 단, 정규화는 공백 수준까지만.

### RT-2. 실질 동일한 상태를 매 사이클 재그리기 (깜빡임)
- **증상**: 화면이 안 바뀌었는데 자막 박스가 계속 다시 그려져 깜빡임.
- **원인**: bbox가 1~2px 흔들리면 `state_ready.emit(new_state)`가 무조건 나가고 오버레이 전체 repaint.
- **수정**: `_state_signature(state, round_px)` — (원문, 번역문, `bbox // round_px`) 지문.
  `BackgroundController._emit_state_if_changed()`가 직전 `translated_text`와 지문이 같으면 emit 생략.
  OCR 경로와 UIA 경로 둘 다 이 헬퍼를 통과(비우기 emit `state_ready.emit([])`는 기존 가드 유지).
- **튜닝 레버**: `STATE_BBOX_ROUND_PX = 4`. 올리면 둔감(자막이 조금 움직여도 갱신 안 함),
  0/1이면 사실상 무효.
- **가드**: `smoke_tests.py::test_rt1_cache_key_and_state_dedup` (RT-1과 공용).
- **재발 방지**: 상태 emit 경로를 새로 만들면 `_emit_state_if_changed`를 경유시킬 것.

### NZ-1. OCR 노이즈 라인까지 박스로 그림 (중구난방)
- **증상**: 아이콘 글자, 시계, 테두리 잡음까지 번역 박스가 붙어 화면이 지저분함.
  게다가 잡음 라인이 엉뚱한 언어로 감지되면 m2m100(418M) 폴백 로드를 유발한다.
- **원인**: 필터가 word 단위 `conf < 30` 하나뿐. 라인 전체의 신뢰도/문자 구성은 안 봤다.
- **수정**: `_lines_from_tess`에 라인 단위 2단 필터 추가.
  1. 라인 평균 word confidence < `OCR_LINE_MIN_CONF`(35.0) → 제외 (conf가 전부 -1이면 통과).
  2. `_ocr_line_is_noise()` — 공백 제외 길이 < `OCR_LINE_MIN_CHARS`(2),
     영숫자/한글/CJK 비율 < `OCR_LINE_MIN_LETTER_RATIO`(0.5),
     `OCR_LINE_REQUIRE_ALPHA`(True)일 때 글자(`isalpha`) 0개면 제외.
- **튜닝 레버**: 위 4개 상수. 부스러기가 남으면 `OCR_LINE_MIN_CONF`를 50~60으로,
  진짜 문장이 사라지면 내린다. 숫자만 있는 원문도 번역하려면 `OCR_LINE_REQUIRE_ALPHA=False`.
- **실측**: 렌더 이미지 E2E — 정상 문장/"OK"/"Settings"는 유지, `10:30`·`~ | * _`는 제거.
- **가드**: `smoke_tests.py::test_nz1_ocr_noise_line_filter`.
- **재발 방지**: 임계는 "정상 짧은 문장이 죽지 않는 선"에서 낮게 시작하고 상수로 노출할 것.

### DP-1. Qt 논리 좌표 vs 물리 픽셀 혼용 → 주 모니터에 오버레이가 아예 안 뜸 ⭐
- **증상**: 100% 배율 주 모니터에서는 오버레이 자막이 **전혀** 안 보이고,
  175% 배율 보조 4K 모니터에서는 자막이 원문 위치에서 어긋난 채 뜬다.
- **실측 근거 (개발 PC, 2026-08-02)**:
  - 주 모니터 1920x1080 @100% → 물리/논리 모두 `(0, 0, 1920, 1080)`
  - 보조 4K @175% → 물리 `(-3840, -535, 0, 1625)` / Qt 논리 `(-3840, -535, -1646, 699)`
  - 즉 Qt 논리 공간에는 `-1646 ~ 0` 이라는, 어느 모니터에도 속하지 않는 **틈**이 존재한다.
- **원인**: 좌표계 혼용. `GetWindowRect` / `ImageGrab` / OCR bbox는 전부 **물리 픽셀**인데
  Qt 지오메트리(`screens().geometry()`, `setGeometry`, `paintEvent`)와 `QCursor.pos()`는 **논리 좌표**.
  가상 데스크톱 전체를 덮는 단일 투명 창인 OverlayWindow가 보조 모니터의 1.75 배율로 스케일되면서,
  주 모니터에 그려야 할 자막이 1.75배 우측으로 밀려 창 밖으로 나가 버린다.
  E-7은 "Qt 자동 high-dpi를 신뢰"까지만 정했을 뿐, 캡처 API와 단위가 다른 문제는 다루지 않았다.
- **수정**: Qt의 논리 스케일링을 끄고 **앱 전 좌표를 물리 픽셀로 통일**한다.
  ```python
  import os
  os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "0")   # 모듈 최상단, QApplication 생성 이전
  ```
  - 논리 == 물리(devicePixelRatio 1)가 되어 `GetWindowRect` · `ImageGrab` · OCR bbox ·
    `paintEvent` · `QCursor.pos()` · W-4 가상 데스크톱 클립 · SC-1 `screenAt` 판정이 한 좌표계로 정합.
  - `OverlayWindow._fit_to_screen` / `RegionEditor._fit_to_virtual_desktop`의 `screens()` 합집합도
    자동으로 물리 rect가 되므로 **코드 변경 불필요** (확인 완료).
  - DPI awareness(퍼모니터 인식) 자체는 Qt가 유지 → OS가 창을 늘려 뭉개는 블러 없음.
- **E-7과의 관계**: E-7(`_scale=1.0` 고정, 수동 배율 곱셈 금지)은 **그대로 유효**.
  새 규약은 그 위에 얹힌다 — "Qt 논리 스케일링 OFF, 전 좌표 = 물리 픽셀".
  즉 여전히 배율을 **곱하지 않는다**. 다만 이제는 Qt도 곱하지 않는다.
- **한계 (의도된 트레이드오프)**: 스케일링이 꺼져 있으므로 설정 창(880x120)과 트레이 메뉴를
  175% 모니터로 옮기면 물리 픽셀 그대로 **작게** 보인다. 100% 주 모니터에서는 기존과 동일.
  오버레이 좌표 정합(핵심 기능)을 UI 크기(편의)보다 우선한 선택.
  UI 크기까지 되찾으려면 스케일링을 다시 켜는 게 아니라, 위젯 크기에만 모니터별
  `logicalDotsPerInch/96`을 곱하는 별도 작업이 필요하다.
- **가드**: `smoke_tests.py::test_dp1_highdpi_scaling_disabled_before_qapp`
  (모듈 레벨 문장인지 + `QApplication(...)` 호출보다 앞선 줄인지 AST로 검사).
- **재발 방지**: 좌표를 다루는 새 코드는 "이 값이 물리인가 논리인가"를 묻지 말고
  **전부 물리**로 간주한다. Qt 스케일링을 다시 켜는 변경은 캡처/OCR 전 경로를 같이 바꿔야 하므로 금지.

### OU-1. 작은 글씨(캡션/각주) OCR 붕괴 — 이미지 크기 기준 업스케일의 사각지대 ⭐
- **증상**: 사용자 보고 "작은 글씨는 번역이 아예 안 나온다". 본문은 정상인데 이미지 캡션·각주만 통째로 누락.
- **실측 근거 (개발 PC, 2026-08-02, `doc_55.png` 724x954)**:

  | | 라인 수 | 라인 높이 중앙값 | 캡션 블록 | 감지 언어 |
  |---|---|---|---|---|
  | 수정 전(1차 OCR) | 33 | 13px | **전멸** (`Sr eae aa! Feng JV` conf 60대 잔재 1줄) | `{en:32, af:1}` |
  | 2배 확대 재OCR | 36 | 12px | 복구 (`aur mail pilot flying` conf 79.2, `the 1920s.` conf 92.7, 본문 캡션 conf 95.7) | `{en:34, no:1, de:1}` |

- **원인**: Tesseract는 글자 높이 12px 아래에서 급격히 무너진다(7~9px 캡션은 conf 60대 + 철자 붕괴).
  E-3의 업스케일은 **이미지 min(h,w) < 400** 이라는 크기 조건이라, 화면 전체를 캡처하는 실사용에서는
  절대 발동하지 않았다. 즉 "작은 이미지"는 커버했지만 "큰 이미지 속 작은 글자"는 못 봤다.
- **수정**: `perform_ocr_with_boxes` — 1차 OCR 후 **통과 라인들의 높이 중앙값**이
  `OCR_UPSCALE_TRIGGER_PX`(18) 미만이면 `OCR_UPSCALE_FACTOR`(2)배 확대해 재OCR하고 그 결과를 채택.
  - 전처리+OCR을 `_tess_lines(gray, lang_str, scale)`로 묶어 1차/재OCR가 **같은 경로**를 탄다.
    좌표 환원은 기존 `_lines_from_tess(data, scale)`의 `// scale` 인프라를 그대로 재사용 — 새 좌표 경로 없음.
  - 확대는 `cv2.INTER_LANCZOS4`로 통일(기존 소형 이미지 경로 포함). 실측상 텍스트에서 CUBIC보다 철자가 정확.
  - **비용**: 재OCR은 트리거가 걸린 작은 글씨 화면에서만 발생한다. 본문 글자가 18px 이상인 일반 화면은
    1차 OCR 한 번으로 끝나므로 추가 비용 0. (평균이 아니라 **중앙값**을 쓰는 이유: 제목 한 줄이 커도 끌려가지 않게)
  - PaddleOCR 경로는 무접촉(자체 detector가 스케일을 다룸).
- **가드**: `smoke_tests.py::test_ou1_adaptive_upscale`
  (합성 렌더 11px 텍스트 → 트리거 조건 성립 + 확대 후 정확 인식 라인 증가 + bbox가 원본 좌표계(x≈20)로 환원 + `_tess_lines` 2회 호출 AST 검사).
- **튜닝 레버**: `OCR_UPSCALE_TRIGGER_PX`(↑ 더 자주 확대=느림 / ↓ 덜 확대), `OCR_UPSCALE_FACTOR`(3 이상은 이득 미미).

### LG-1. 언어 오판 파편화 → m2m100 폴백 낭비 + 쓰레기 번역
- **증상**: 영어 한 화면인데 상태줄에 `[de, en, pt, so]`처럼 언어가 파편화되고, 그 줄들은 번역이
  느리거나(5초대) 엉뚱하게 나오거나 아예 안 뜬다.
- **원인**: OU-1의 작은 글씨 붕괴로 생긴 철자 쓰레기(`Sir mail pot fying aw`)를 langdetect가
  cy/hu/de/pt/so로 판정 → M-5 라우팅이 opus-mt 등록 쌍을 못 찾아 **m2m100 폴백**으로 새고,
  배치가 언어별로 쪼개져 generate 호출이 늘어난다. M-7 region 투표는 **region 안**만 보므로
  같은 region의 이웃도 같이 오판되면 못 잡는다.
- **수정**: `assign_region_languages` — M-7 region 투표 **뒤에** 페이지 전역 스냅을 얹는다.
  1. 확신(`_language_confidence` ≥ 2) 있는 **라틴 스크립트 라인만**의 가중 다수결로 지배 언어를 구한다.
  2. 지배 비중이 `DOMINANT_LANG_MIN_SHARE`(0.6) 이상이고 **지배 언어가 라틴 계열**일 때만,
  3. 확신 `≤ DOMINANT_LANG_SNAP_MAX_CONF`(2)인 **라틴 스크립트 라인**의 언어를 지배 언어로 스냅.
  - 한/일/중/러/아랍 등 **비라틴 스크립트 라인은 절대 스냅하지 않는다** — 진짜 다국어 화면 보존(UX-2).
    지배 언어가 비라틴(`_SCRIPT_TO_LANG.values()`)이면 스냅 자체를 건너뛴다(영어 줄이 한국어로 끌려가지 않게).
  - **투표 모집단도 라틴 라인으로 한정한다 (자체 검수 보강)**. 초안은 비라틴 라인까지 투표에 넣어,
    영문 3줄(conf 2) + 파편(conf 2) + 한국어 1줄(conf 4)이면 en 비중이 `6/12 = 0.5`로 희석되어
    임계 미달 → **스냅이 통째로 꺼졌다**. 한국어 UI가 섞이는 실사용 조건에서 기능이 상시 무력화되는 구조.
    비라틴은 애초에 스냅 대상이 아니므로 지배 판정에 거부권을 가져선 안 된다.
  - 확신 3(라틴 짧은 힌트 적중, 예: "Merci"/"Hvala")은 건드리지 않아 M-7 기존 동작이 그대로 유지된다.
  - `batch_translate` / M-8 sepvoc 인코딩은 무접촉 — 입력 언어 리스트만 정돈된다.
- **효과 (doc_55 실측)**: `{en:34, no:1, de:1}` → `{en:36}`. 전 라인이 opus-mt 단일 경로로 묶여
  배치 1회로 끝나고, 파편 줄의 번역 품질도 회복.
- **가드**: `smoke_tests.py::test_lg1_dominant_language_snap`
  (영문 8줄 + 붕괴 파편 2줄 → 전부 en 수렴 / 한국어 줄은 ko 유지 / src 명시 시 강제 유지 /
  **영문 3줄 + 파편 2줄 + 고확신 한국어 1줄** → 파편이 en으로 스냅되고 ko는 보존 = 희석 회귀 가드).
  기존 `test_m7_region_smoothing`(다국어 4개 언어 혼재)이 회귀 가드로 같이 돈다.
- **튜닝 레버**: `DOMINANT_LANG_MIN_SHARE`(↑ 보수적), `DOMINANT_LANG_SNAP_MAX_CONF`(3으로 올리면 라틴 힌트까지 스냅 — 권장 안 함).

### OS-1. OSD script 오판 1회가 세션 전체를 오염 (영어 문서 → 아랍어 번역) ⭐
- **증상**: 영어 PDF를 띄웠는데 상태줄 언어가 `[ar]`, 오버레이에는 "엘리자베스 엘리자베스 …"(×12),
  "억원, 억원 …"(×16) 같은 반복 쓰레기가 뜬다. 창을 바꿔도 한동안 안 돌아온다.
- **원인 사슬 (사용자 실기 재현 + 검증 실험으로 확정)**:
  1. `_detect_script_via_osd`가 `pytesseract.image_to_osd`의 `script`를 **신뢰도 검사 없이** 채택.
     OSD는 근거가 빈약해도 항상 어떤 script를 돌려준다.
  2. 영어 PDF에서 한 번 `Arabic` 오판 → `_script_cache[hwnd]`(TTL 30초)에 **고착**.
  3. 이후 모든 프레임이 `ara` 데이터로 OCR → 아랍 글자 쓰레기(`5 لاع هنالاعا- )5310`).
  4. `identify_language`가 전 줄을 `ar`로 판정 → M-5 라우팅이 m2m100 `ar→ko` 폴백 →
     의미 없는 입력에 대해 신경망이 같은 구를 무한 반복(RP-1)해 오버레이에 뿌림.
  5. **성공 시 로그가 없어서**(실패만 `print`) 어느 단계에서 샜는지 추적 불가 — 진단이 오래 걸린 이유.
- **수정 (4겹, 앞 단계일수록 근본)**:
  1. **신뢰도 게이트** — `osd["script_conf"]`가 `OSD_SCRIPT_MIN_CONF`(1.0) 미만이면 `script=None` 반환
     → 기존 eng+kor 폴백 경로. `script_conf` 키가 없거나 파싱 실패면 기존 동작(신뢰) 유지(구 tesseract 호환).
     반환형이 `script` → `(script, conf)` 튜플로 바뀜(호출부는 `_smart_lang_for_image` 하나).
  2. **eng 상시 합류** — `_SCRIPT_TO_TESS` 후보에 `eng`가 없으면 항상 추가(Arabic → `ara+eng`).
     라틴 글리프는 eng 패턴이 이기므로 **OSD가 틀려도 영어 단어는 영어로 인식**되어 사슬이 1단계에서 끊긴다.
     이번 수정의 핵심 안전망 — 게이트를 뚫는 고확신 오판까지 커버한다.
  3. **관측성** — 성공 경로에도 `[INFO] OSD script=… conf=… → langs=…` 1줄.
     OSD는 캐시 miss(hwnd별 TTL)에서만 도니 핫 패스 print 아님(P-6 위반 아님).
  4. 게이트를 통과한 결과만 `_script_cache`에 들어가므로 오판 고착 자체가 사라진다.
- **가드**: `smoke_tests.py::test_os1_osd_confidence_gate_and_eng_fallback`
  (저확신 Arabic → `ara` 미사용 + eng 폴백 / 고확신 Arabic → `ara+eng` / `script_conf` 키 없음 → 기존 동작).
  `pytesseract.image_to_osd`를 monkeypatch해 **실동작**으로 검사(AST 문자열 검사 아님).
- **튜닝 레버**: `OSD_SCRIPT_MIN_CONF`(↑ 엄격 = 폴백 잦음/안전, ↓ 관대 = 오판 통과).
  실측상 확신 있는 판정은 2~40대, 근거 빈약이면 1 미만.

### RP-1. 잡음 입력에서 번역이 같은 구를 무한 반복 (반복 붕괴)
- **증상**: "엘리자베스 엘리자베스 …"(×12), "억원, 억원 …"(×16)처럼 한 단어/구가 박스를 채운다.
- **원인**: greedy decoding(`num_beams=1, do_sample=False`)의 고전적 degeneration.
  의미 없는 입력(OS-1의 아랍 쓰레기)에서 모델이 같은 구를 반복하는 상태로 빠지면 EOS까지 못 간다.
- **수정**: `OpusMtTranslator.translate_batch` / `M2M100Translator.translate_batch`의 `generate(...)`에
  `no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE` **한 줄씩만** 추가. 번역 파이프라인 구조
  (M-8 sepvoc 인코딩·배치·M-5 라우팅·캐시)는 무접촉.
- **값 선택 근거 (4, 3 아님)**: 주기 p의 반복은 길이 `n+p`만 넘으면 n-gram이 중복되므로
  **n=4도 모든 반복 루프를 잡는다**. 반면 n=3은 한 라인에 두 문장이 들어올 때
  정상 한국어의 합법적 반복까지 깬다 — 실측: `"이 회사는 올해 새로운 정책을 발표했다. 이 회사는 모든
  사무실에 적용했다."`에서 `"이 회사는"`이 두 번 나오는데, 이는 3-gram 금지에 걸릴 수 있는 패턴이다.
  n=4에서 이 문장이 그대로 보존됨을 실모델로 확인. 억제를 더 세게 하려면 3으로 내릴 수 있다.
- **가드**: `smoke_tests.py::test_rp1_no_repeat_ngram_guard`
  (두 translate_batch에 파라미터 존재 확인 + 실모델이 있으면 정상 영어 문장 1개를 번역해
  한글이 나오고 같은 토큰이 3연속 반복되지 않는지 = 품질 회귀 1차 가드. 모델 없으면 skip).
- **튜닝 레버**: `NO_REPEAT_NGRAM_SIZE`(↓3 강한 억제/정상 문장 왜곡 위험, ↑5 관대).

### SL-1. auto 모드 언어 라우팅 결정론화 — "다국어 자동 추측" 퇴역 (설계 결정) ⭐
- **성격**: 버그 수정이 아니라 **범위를 줄이는 설계 결정**. 무엇을 버렸는지·왜·어떻게 되돌리는지를 남긴다.
- **버린 것 (auto 경로 한정)**:
  1. OSD script 판별 — `_detect_script_via_osd` / `_smart_lang_for_image` / `_script_cache` /
     `_SCRIPT_TO_TESS` / `OSD_SCRIPT_MIN_CONF` **삭제**. auto OCR 언어는 `eng+kor` 고정.
  2. langdetect 추측 — `from langdetect import ...` / `DetectorFactory.seed` / `_LANGDETECT_TO_ISO` /
     `_LATIN_SHORT_HINTS` / `_latin_hint` / `_safe_detect` **삭제**. `identify_language`는
     `_SCRIPT_TO_LANG.get(script, "en")` 한 줄 — 스크립트로 확정되는 언어만 인정한다.
  3. M-7 region 투표(`_cluster_text_regions`, `_language_confidence`)와 LG-1 지배 언어 스냅
     (`DOMINANT_LANG_MIN_SHARE`, `DOMINANT_LANG_SNAP_MAX_CONF`) — 둘 다 "라틴 추측의 오차를
     사후 보정"하는 장치였다. 추측이 사라지면 보정할 대상도 없어 함께 **삭제**(dead code 방지).
- **버린 이유 (실증)**: 하루 실사용에서 오판 6종(cy/hu/de/pt/so/ar) 확인. 결정타는 영어 PDF를
  OSD가 `Arabic` **conf 3.95**로 판정한 사례 — OS-1의 신뢰도 게이트(1.0)를 정면으로 통과했다.
  대가는 일부 줄 `ara` OCR 쓰레기 + m2m100 `ar→ko` 5초 배치. 즉 **게이트로는 못 막는다**.
  기대 이득(진짜 다국어 화면 자동 대응)은 사용자 실사용(영어 화면 → 한국어)에 존재하지 않았고,
  비용(오판 1건 = 그 프레임 전체 붕괴 + 사이클 지연)만 상시 발생했다.
- **새 규약 (auto)**:
  1. OCR 언어 = `OCR_LANGS_AUTO_DEFAULT`(`eng`+`kor`, 가용한 것만). 판별 호출 자체가 없다.
     `kor` traineddata가 없으면 `eng` 단독 폴백(기존 패턴).
  2. 라인 언어 = 문자 스크립트가 결정적인 것만(`_SCRIPT_TO_LANG`: 한글→ko, 가나→ja, CJK→zh,
     키릴→ru …). **라틴·불명은 전부 `en`.** 오판이 원리적으로 불가능한 판정만 남긴 것.
  3. m2m100 폴백은 사용자가 src 콤보에서 en/ko 외를 명시했거나, 스크립트로 확정된 비라틴
     (ja/zh/ru…)에서만 켜진다. 라틴 오판으로 새는 경로가 없어졌다.
  4. tgt=Korean이고 라인이 ko로 판정되면 `src == tgt` → 번역 스킵(기존 동작 유지).
- **트레이드오프(명시)**: 프랑스어/독일어/스페인어 화면을 auto가 **더 이상 자동 인식하지 않는다**.
  전부 en으로 번역을 시도한다. 그런 화면은 **source 콤보에서 언어를 명시**하면 기존 경로가 그대로
  동작한다(OCR 언어도 그 언어로, 번역도 m2m100/opus-mt 정상 라우팅). 수동 탈출구:
  `COCKTAIL_OCR_LANGS_AUTO=eng+jpn` 같은 환경변수 오버라이드도 유지.
- **속도**: 사이클에서 OSD 호출(실측 111ms/회, 캐시 miss마다)과 langdetect(15라인 프레임 실측
  ~130ms, 라인당 ~9ms)가 사라진다. 더 큰 이득은 오판이 유발하던 m2m100 폴백 배치(1건당 ~5초)와
  모델 로드가 통째로 없어지는 것.
- **가드**: `smoke_tests.py::test_sl1_deterministic_line_languages`(영/한/일 혼합 라인이
  `[en, ko, ja]`로만 나오고, 라틴은 무엇이 와도 en / src 명시는 그대로),
  `::test_sl1_no_guessing_engines_in_auto_path`(AST로 langdetect import 잔재 검사 +
  `image_to_osd`/`_script_cache`/`_SCRIPT_TO_TESS` 문자열 잔재 검사 + 탈출구 유지 확인).
  `pre_release_check.py::check_sl1_auto_language_routing`도 같은 잔재를 막는다.
- **남겨둔 것**: `langdetect` 패키지 자체는 requirements/build.spec에 그대로 둔다(복원 시 import 한 줄).
  앱은 더 이상 import하지 않으므로 `--self-test` 필수 모듈 목록에서만 뺐다.
  `osd.traineddata`도 설치본에 남지만 미사용(무해).
- **복원 조건 (되살릴 때 이 3개를 먼저 확인)**:
  1. 사용자 시나리오가 실제로 다국어가 되었는가? (영어 단일이면 복원 이득 0)
  2. 새 판별기가 **오판율을 근거로** 검증되었는가? conf 게이트는 이미 실패했다 —
     신뢰도 수치가 아니라 "오판 시 무엇이 무너지는가"로 평가할 것.
  3. 오판이 나도 **그 프레임만** 손상되는가? (캐시 고착·배치 분할처럼 번지는 구조면 채택 금지)
  복원 방법: 이 커밋의 삭제분을 되살리거나, 더 나은 판별기(예: 라인 단위 fastText LID)를
  `identify_language`의 라틴 분기에만 얹는다. 나머지 파이프라인은 손댈 필요 없다.

### DG-1. 표시 직전 최종 게이트 — "확신 없으면 숨긴다" ⭐
- **증상**: 초소형 캡션(7px대)에서 OCR이 **말이 되는 오독**을 하면 NZ-1(원문 문자 구성)을
  그대로 통과해 "마우 냄비 thang에"류 그럴싸한 쓰레기가 오버레이에 뜬다. OU-1의 2배 확대로도
  살아나지 않는 영역이 남는다(확대해도 원본 정보가 없으면 복구 불가).
- **원인**: 필터가 전부 **번역 이전**에만 있다. NZ-1은 원문의 길이/문자 비율/평균 conf만 보는데,
  철자가 붕괴해도 "그럴듯한 단어열"이면 전부 정상으로 보인다. 번역 결과 자체를 본 적이 없었다.
- **수정**: `display_gate_reject(src, tgt, tgt_lang, line_px, conf)` — 표시 직전(new_state 구성
  시점) 마지막으로 "이 줄을 믿을 근거가 있는가"를 묻고, 없으면 **그 줄만 그리지 않는다**.
  1. **극소 라인 + 저 confidence** — `line_px < DISPLAY_TINY_LINE_PX`(12) **그리고**
     `conf < DISPLAY_TINY_LINE_MIN_CONF`(70). 두 조건 동시 성립일 때만이라 큰 글씨는 conf가 낮아도 산다.
  2. **목표 언어 스크립트 실종** — 번역문 글자 중 목표 언어 스크립트 비율이
     `DISPLAY_MIN_TGT_SCRIPT_RATIO`(0.3) 미만이면 번역이 실패한 것(한국어 목표인데 한글 없음).
     고유명사가 라틴으로 남는 정상 문장은 0.3 임계를 여유 있게 넘는다.
  3. **길이 붕괴** — 원문이 `DISPLAY_LEN_COLLAPSE_MIN_SRC`(20)자 이상인데 번역문이
     `DISPLAY_LEN_COLLAPSE_RATIO`(0.15)배 미만 = 모델이 입력을 버린 것.
- **conf 배선**: 라인 튜플 계약 `(text, bbox, font_size)`은 Paddle/UIA와 공용이라 **바꾸지 않았다**.
  `_lines_from_tess(data, scale, conf_out)` 선택 인수로 라인별 평균 conf를 병렬 리스트에 채워
  `BackgroundController._ocr_line_confs`로 전달한다. UIA는 conf 없음(None) → 규칙 1이 자동 무효.
- **관측**: 숨긴 줄 수를 상태줄에 `· 숨김 N줄(DG-1)`로 표시(핫 패스 print 아님).
- **가드**: `smoke_tests.py::test_dg1_display_gate` — 정상 5종이 **통과**하는 것까지 검사
  (거짓 양성이 곧 기능 상실이라 이쪽이 더 중요), 숨겨야 하는 6종은 거부 사유 반환.
- **튜닝 레버**: 위 5개 상수 + `DISPLAY_GATE_ENABLED`(False면 게이트 전체 무효, 디버그용).
- **재발 방지**: 표시 경로를 새로 만들면(예: 새 OCR 백엔드) 이 게이트를 반드시 경유시킨다.
  임계를 올릴 땐 "정상 줄이 사라지지 않는가"를 먼저 본다 — 숨김은 무음 실패라 사용자가 원인을 못 찾는다.

### CO-2. DG-1 미번역 잔류 규칙(CO-1)이 정상 문단을 통째로 숨김 ⭐
- **증상**: 영어 PDF(어린왕자 영문판) 번역 중 **특정 문단만** 자막 박스가 아예 안 뜬다.
  위아래 문단은 정상. 안 뜨는 건 삽화 바로 아래의 3줄짜리 본문 문단들이었다.
  숨김은 **무음 실패**라 로그에도 아무것도 안 남아 "번역이 안 됐나 OCR이 안 됐나"조차 알 수 없었다.
- **진단 (재현 하네스, 2026-08-11)**: 해당 PDF 4쪽을 화면과 같은 배율로 렌더해 실제
  파이프라인(OCR → LM-1 세그먼트 → PM-1 문단 병합 → SE-1 문장 분할 번역 → DG-1)을 그대로 태우고
  단계별 생존을 표로 찍었다. 결과 — **OCR도 병합도 번역도 전부 정상**이었다:

  | 단계 | 대상 문단 상태 | 수치 |
  |---|---|---|
  | OCR 라인 | 생존 | conf 94.8 / 라인 높이 17~24px |
  | PM-1 병합 | 3줄 → 1문단, 문장 온전 | 346자 / 204자 |
  | 번역(SE-1) | 문장 5개 전부 번역됨 | 번역/원문 길이비 0.49, 0.37 (붕괴 임계 0.15의 3배) |
  | `max_new_tokens` | 잘림 없음 | 문장별 in_len×1.6+16, 실측 24~99 (상한 256) |
  | **DG-1** | **숨김** | `carryover(...)` — 잔류 소문자 라틴 단어 **1개** |

  즉 용의자 2(번역 길이 상한)·3(LM-1)·4(NZ-1/conf)·5(dedup)는 전부 **배제**됐고, 원인은 DG-1 하나였다.
- **원인**: CO-1의 `DISPLAY_MAX_CARRYOVER_WORDS = 0`(무관용)은 게임 툴팁 **한 줄**(30~60자)
  기준으로 잡은 값인데, 그 뒤 **PM-1(문단 병합)이 게이트의 판정 단위를 200~350자 문단으로 키웠다**.
  40단어 문단에서 OCR이 단어 하나만 틀리면(긴 희귀어일수록 잘 틀린다) 번역기가 모르는 그 단어를
  **그대로 복사**하고, 나머지 39단어가 완벽해도 문단 전체가 사라진다.
  1글자 오독 주입 실측 8종 중 3종이 문단 전멸을 유발:
  `constrictor→consttictor`(346자 전멸) / `elephant→elephanl`(346자 전멸) /
  `disheartened→disheattened`(204자 전멸). 삽화 옆 문단이 걸린 건 우연이 아니다 —
  그 문단들이 하필 `constrictor`, `elephant`, `disheartened` 같은 **길고 희귀한 단어**를 담고 있다.
  절대 개수 임계를 **가변 길이 단위**에 적용한 것이 결함의 본질이다.
- **수정**: 허용치를 번역문 길이에 비례시킨다 —
  `carryover_allowance(tgt) = max(DISPLAY_MAX_CARRYOVER_WORDS, 어절수 × DISPLAY_CARRYOVER_MAX_RATIO)`
  (0.12 → 9어절 미만 0, 9~16 1, 17~25 2 …). 짧은 줄은 지금과 **똑같이 무관용**이라
  CO-1이 막던 "반쯤 번역된 헛소리"는 그대로 걸린다(실측 fixture: 10어절/잔류 2 > 허용 1 → 숨김 유지).
  긴 문단은 오독 한둘을 견디고 살아남는다.
- **관측성**: 숨길 때 `[INFO] DG-1 숨김: <사유> "<원문 머리 60자>"`를 찍는다.
  `display_gate_reject`(공개 진입점)가 `_display_gate_reason`(규칙 본체)을 감싸는 구조라
  **OCR 경로와 UIA 경로 양쪽이 자동으로 로그를 탄다**. 정적 화면이 같은 줄을 초당 몇 번 재판정해도
  (사유 종류, 원문 머리) 키로 최근 64건을 기억해 **라인당 1회**만 찍는다 — 핫 패스 print 금지 규약 유지.
- **가드**: `test_carryover_gate_allowance_scales_with_paragraph_length`(빠른 테스트, 실측 문자열 그대로),
  `test_long_paragraph_survives_to_display`(`--full`, batch_translate를 실제로 태워 문단이 표시까지 생존).
  비율을 0.0으로 되돌리는 뮤테이션에서 전자가 실패하는 것까지 확인했다.
- **비용**: 0. 40줄 게이트 1회 2.87ms → 2.87ms(측정 노이즈 내). `carryover_allowance`는 잔류 단어가
  **있을 때만** 호출된다.
- **재발 방지**: 표시 게이트에 "개수" 임계를 넣을 땐 그 임계가 **어떤 크기의 단위**에 적용되는지
  먼저 확인한다. 판정 단위를 키우는 변경(PM-1 같은 병합)을 하면 기존 절대 임계를 전부 재검토한다.
  거짓 양성(정상 문장 숨김)은 게이트의 **실패**다 — 깨진 자막 1줄보다 사라진 문단 1개가 더 나쁘다.

### LM-1. 대형 폰트 뭉침 라인 — word 높이 이상치 + 두 단 병합 ⭐
- **증상**: 특정 라인이 화면 1/3을 덮는 거대 폰트로 뭉쳐 나온다(예: "공군. 나치 군대가 1940에서…").
- **원인 (렌더 이미지 재현으로 확정, 개발 PC 2026-08-02)** — 둘 다 `_lines_from_tess`에 있었다:
  1. `font_size = max(word height)`. 라인에 장식 글자나 **도형**이 섞이면 그 하나가 라인 전체의
     폰트·bbox를 끌고 간다. 실측: 세로 기둥(그래프 축/이미지 테두리)이 `"|"` height **81px**로 잡혀
     본문 14px 라인이 `font=81, bbox 높이 81` → 오버레이 초기 폰트 **73px**.
     60px 장식 "W"가 본문에 붙은 경우도 `font=43`.
  2. `--psm 6`(균일 블록)은 **좌우 두 단**을 한 라인으로 묶는다. 실측: 255px 간격을 건너뛰어
     `'left column sentence here right column other text'` 한 줄, bbox 폭 **645px**.
- **수정**: `BackgroundController._line_segments(words)` 신설 — 라인의 word들을
  (a) 높이 중앙값의 `OCR_WORD_HEIGHT_OUTLIER_RATIO`(2.5)배를 넘는 word 제거,
  (b) 가로 간격이 `중앙값 높이 × OCR_LINE_SPLIT_GAP_RATIO`(3.0)를 넘으면 별개 라인으로 분할.
  이후 세그먼트별로 기존 NZ-1 필터(평균 conf / 문자 구성)를 그대로 적용하고,
  `font_size`는 `max` → **중앙값**으로 교체한다.
- **실측(수정 후)**: `font 81 → 11`, `bbox 높이 81 → 14`, 두 단은 2줄로 분리(폭 645 → 167/144).
- **부수 영향**: OU-1의 업스케일 트리거는 라인 높이 **중앙값**을 보는데, font_size가 max에서
  중앙값으로 내려가 트리거가 약간 더 자주 걸린다(작은 글씨를 더 잘 살리는 방향이라 무해).
- **가드**: `smoke_tests.py::test_lm1_line_merge_and_font_outlier`
  (합성 word 데이터로 이상치 제거·간격 분할 + 실제 렌더 이미지 E2E로 폰트/폭/높이 상한 검사).
- **튜닝 레버**: `OCR_WORD_HEIGHT_OUTLIER_RATIO`(↓ 공격적, 큰 제목 글자가 잘릴 위험),
  `OCR_LINE_SPLIT_GAP_RATIO`(↓ 더 잘게 쪼갬 — 넓은 자간 문서에서 문장이 끊길 위험).
- **재발 방지**: OCR 라인 통계를 쓸 때 `max`/`평균`은 이상치에 무방비다. 중앙값을 기본으로 하고,
  "한 라인"을 Tesseract의 line_num만 믿지 말 것 — psm에 따라 물리적으로 떨어진 텍스트가 묶인다.

### TM-1. 오버레이가 다른 항상-위 창에 가려짐
- **증상**: 번역 자막이 떠 있다가 런처/미디어 플레이어 컨트롤 등이 뜨면 그 뒤로 숨는다.
- **원인**: `Qt.WindowStaysOnTopHint`는 창을 topmost **밴드에 넣을 뿐**이고, 그 밴드 안의
  Z-order는 나중에 올라온 창이 이긴다. 우리는 생성 이후 한 번도 Z-order를 되찾지 않았다.
- **수정**: `set_window_topmost(hwnd)`(`SetWindowPos(HWND_TOPMOST, SWP_NOSIZE|NOMOVE|NOACTIVATE)`)
  + `OverlayWindow` show() 직후 `raise_()` 1회 + `QTimer`로 `OVERLAY_TOPMOST_INTERVAL_MS`(2000ms)마다 재적용.
  `SWP_NOACTIVATE`라 포커스를 뺏지 않는다(사용자가 치던 창을 방해하면 안 됨).
  D-1 교훈대로 `argtypes`를 명시했다 — 64비트에서 `HWND_TOPMOST(-1)`을 32비트 int로 넘기면 상위 비트가 쓰레기가 된다.
- **한계(문서화)**: **전체화면 독점(exclusive fullscreen) 앱 위에는 원리상 어떤 창도 못 올라간다.**
  게임/영상 전체화면은 테두리 없는 창 모드(borderless windowed)로 바꿔야 자막이 보인다.
- **튜닝 레버**: `OVERLAY_TOPMOST_INTERVAL_MS`(0이면 주기 재적용 OFF, 수동 raise_만).

### FT-1. 오버레이 폰트 "Arial" 하드코딩 (한글 글리프 없음)
- **증상**: 한국어 자막의 자간/굵기가 화면마다 들쭉날쭉하고 글자가 어색하게 섞여 보인다.
- **원인**: `QFont("Arial")`은 한글 글리프가 없어 Qt의 임의 폴백 폰트에 맡겨진다(환경마다 다름).
- **수정**: `OVERLAY_FONT_FAMILIES = ("Malgun Gothic", "Noto Sans KR", "Segoe UI", "Arial")`
  + `OverlayWindow._overlay_font(px)` 헬퍼(`setFamilies` + F-2의 `setPixelSize`)를 신설해
  `_fit_font`/`_draw_one`이 **같은 헬퍼 하나**를 쓰게 했다.
- **실측**: 개발 PC에서 `QFontInfo` 해석 결과 `Malgun Gothic`.
- **재발 방지**: 폰트를 만드는 자리가 둘 이상이면 반드시 헬퍼로 모을 것. 크기는 항상 `setPixelSize`(F-2).

### RC-1. 비우기 판정이 지연된 메인 스레드 상태를 읽음 (잔상 영구화 race)
- **증상**: 잠재 결함(감사 #8). 타이밍에 따라 자막이 사라져야 할 때 화면에 영구히 남는다.
- **원인**: 워커의 `if self.translated_text: self.state_ready.emit([])` 판정 5곳이,
  **메인 스레드 큐드 슬롯**(`_apply_state_to_self`)이 갱신하는 `translated_text`를 읽는다.
  워커가 상태를 emit한 직후 슬롯이 아직 안 돌았으면 `translated_text`는 여전히 옛 값(빈 리스트)이라
  "비울 게 없다"고 오판해 `emit([])`을 건너뛴다 → 오버레이는 마지막 상태로 고정.
  RT-2의 `_emit_state_if_changed` 비교 기준도 같은 값이라 같은 함정을 공유했다.
- **수정**: 워커가 "마지막으로 emit한 상태"를 자기 로컬 `self._last_emitted`로 추적한다.
  emit 경로를 `_emit_state(new_state)` 하나로 모으고(로컬 갱신 + 시그널 방출을 같은 자리에서),
  일시정지/스킵 5곳은 `_clear_overlay_if_any()`, dedup은 `_emit_state_if_changed()`가 담당한다.
  `set_running(False)`(UX-3)도 같은 헬퍼를 쓰므로 정지 → 재개 시 dedup이 잘못 걸려
  화면이 빈 채 고정되는 2차 함정도 같이 막힌다.
- **가드**: `smoke_tests.py::test_rc1_emit_state_uses_worker_local`
  (메인 슬롯이 지연된 상태를 재현해 비우기가 나가는지 + 비운 뒤 같은 상태가 다시 그려지는지 +
  워커 소스에 `if self.translated_text:` 잔재가 없는지).
- **재발 방지**: 워커는 메인 스레드가 쓰는 상태를 **읽지도 않는다**. 워커 판정은 워커 로컬로.

### GT-1. 워커 스레드에서 GUI 전용 Qt API 호출 (Qt 규약 위반)
- **증상**: 잠재 결함(감사 #13). 모니터 핫플러그/해상도 변경 시점에 크래시 가능.
- **원인**: `region.update_for_mode()`가 워커 루프에서 호출되는데 그 안에서
  `QCursor.pos()`(MODE_HOVER)와 `QGuiApplication.screenAt/primaryScreen`(SC-1)을 부르고,
  W-4 클립도 워커에서 `QApplication.screens()`를 열거했다. Qt 화면/커서 API는 GUI 스레드 전용이다.
- **수정** (최소 변경):
  1. 화면 rect는 **메인 스레드가 캐시**한다 — `CaptureRegionController.screen_rects`
     (`(l, t, r_excl, b_excl)` 튜플, 절대/물리 픽셀). `refresh_screen_rects()`는 생성 시 +
     MS-1 시그널에서만 호출. 워커는 `screen_rect_at()` / `virtual_desktop_rect()`로 **읽기만** 한다.
     Qt의 `right()/bottom()` = 마지막 픽셀 규약은 캐시를 만들 때 한 번만 흡수한다(호출부에서 +1 금지).
  2. 커서는 `cursor_pos_physical()` — win32 `GetCursorPos`. 스레드 무관이고, 프로세스가
     per-monitor DPI aware라 반환값이 **물리 픽셀** → DP-1 규약과 그대로 정합.
- **가드**: `test_w4_w5_multi_monitor_capture`(워커 소스에 `QApplication.screens()` 잔재 0),
  `test_sc1_active_screen_clamp`(판정이 캐시 기반인지 + 캐시 갱신이 screens()/primaryScreen()인지).
- **재발 방지**: 워커에 Qt 객체를 새로 끌어들이지 말 것. 필요하면 메인 스레드가 계산해 **값**으로 넘긴다.

### MS-1. 모니터 구성 변경에 오버레이/캡처 좌표가 안 따라옴
- **증상**: 감사 #5 잔여. 모니터를 꽂거나 빼거나 해상도를 바꾸면 오버레이 창이 옛 가상 데스크톱
  크기 그대로라 자막이 화면 밖에 그려지거나 잘린다.
- **원인**: `OverlayWindow._fit_to_screen` / `RegionEditor._fit_to_virtual_desktop`이 **생성 시 1회**만 돈다.
- **수정**: `CocktailAppController._wire_screen_changes()` — `screenAdded` / `screenRemoved` /
  각 화면의 `geometryChanged`를 `_on_screens_changed`에 연결하고, 거기서
  (a) `region.refresh_screen_rects()`(GT-1 캐시), (b) `overlay._fit_to_screen()` + `update()`,
  (c) 열려 있는 `RegionEditor._fit_to_virtual_desktop()`을 함께 수행한다.
  새로 붙은 화면의 `geometryChanged`도 매번 다시 훑어 연결한다(연결한 QScreen을 집합으로 기억 —
  `Qt.UniqueConnection`은 수신자가 QObject일 때만 확실하다).
- **책임 위치**: 생성/연결은 `CocktailAppController`(ARCHITECTURE.md §2.1 규약대로), 상태는 `region`.
- **재발 방지**: "생성 시 1회 계산하는 화면 의존 값"을 추가하면 이 시그널 3종에도 같이 물릴 것.

### CL-1. 그리기 클립이 가상 데스크톱 전체 기준
- **증상**: 감사 #11. 모니터 경계에 걸친 자막이 옆 모니터로 흘러넘치고, 모니터가 없는
  가상 데스크톱의 빈 영역(비대칭 배치에서 생기는 틈)에도 박스가 그려질 수 있다.
- **원인**: `OverlayWindow._draw_one`의 클립이 `self.rect()` = 가상 데스크톱 전체.
  B-12에서 red_rect → 위젯 전체로 넓힌 뒤 모니터 경계는 고려된 적이 없다.
- **수정**: `_clip_for_abs_bbox(abs_bbox, wx, wy)` — 박스 **중심이 속한 모니터**(GT-1 캐시)를
  창 rect와 교차해 overlay-내 좌표로 돌려주고, `_draw_one(..., clip_rect)`가 그걸로 클립한다.
  어느 모니터에도 안 속하는 좌표는 primary로 폴백되어 결과적으로 교집합이 비어 **안 그려진다**.
- **재발 방지**: 오버레이는 가상 데스크톱 전체를 덮는 단일 창이다 — "창 안"이 "화면 안"을 뜻하지 않는다.

### SV-1. UIA 민감 창 폴백이 OCR 경로를 열어 둠 (보안 결함) ⭐
- **증상**: 감사 #12. 민감 창(패스워드/결제/은행)이 UIA 경로에서 차단됐는데도 그 프레임이
  **OCR로 계속 진행**되어 캡처·OCR·번역되고, 번역 결과가 DPAPI 영구 캐시에까지 남는다.
- **원인**: `_uia_collect_and_translate`의 반환 의미가 뭉뚱그려져 있었다. 민감 창에서 `[]`를
  돌려주는데 호출부는 `if uia_state is not None and uia_state:`로 판정하므로 `[]` = "UIA 실패"
  → 아래 OCR 경로로 그대로 흘러간다. S-1 가드는 `capture_mode != MODE_BOX`에서만 도니
  **MODE_BOX + UIA 명시 선택** 조합에서는 두 번째 방어선도 없었다.
- **수정**: 반환 의미를 3종으로 분리 — `UIA_SKIP_FRAME`(민감 창 = **이 프레임 전체 스킵**),
  `None`(UIA 불가/추출 실패 → OCR 폴백 OK), `list`(정상). 호출부는 sentinel이면
  1초 대기 후 `continue` — OCR 캡처(`ImageGrab.grab`)에 **도달하지 않는다**.
  오버레이 비우기/상태 메시지/`_sensitive_paused` 플래그는 S-1 경로와 동일하게 유지.
- **가드**: `smoke_tests.py::test_sv1_uia_sensitive_blocks_ocr_fallback`
  (민감 창을 흉내 내 sentinel 반환 실동작 확인 + 호출부 분기 존재 + sentinel 처리가
  `ImageGrab.grab`보다 **앞선 줄**인지 소스 순서 검사).
- **재발 방지**: 보안 가드의 반환값에 "실패"와 "차단"을 같은 값으로 쓰지 말 것.
  차단은 반드시 **호출부가 구분할 수 있는 별도 타입**으로 표현한다.

### RB-1. `run_cocktail.bat`이 패키지 없는 시스템 Python을 잡아 실패 ⭐
- **증상**: 사용자가 `run_cocktail.bat`을 더블클릭하면 앱이 안 뜬다. "제품"의 첫 관문이 막힘.
- **원인**: 배치가 그냥 `python`을 호출했다. PATH에 먼저 잡히는 건 시스템 Python 3.13이고
  거기엔 PySide6/torch/transformers가 **하나도 없다**. 앱 코드는 멀쩡한데 실행이 안 되는 상태.
  진단(`diagnose_environment.py`)도 같은 잘못된 인터프리터로 돌아 원인 파악을 늦춘다.
- **수정**: 우선순위 탐색으로 교체. ① `COCKTAIL_PYTHON`(사용자 지정 — 있으면 무조건 존중)
  → ② `%USERPROFILE%\anaconda3|miniconda3\envs\cd\python.exe` → ③ `py -3.11` → ④ PATH `python`.
  ②~④ 후보는 **`import PySide6`가 되는지**로 판별한다(단순 "python이 실행되는가"로는
  바로 이 버그를 못 거른다). 전부 실패하면 조치 2가지를 찍고 `pause`.
  `diagnose_environment.py`도 앱과 같은 규칙으로 `./tessdata`를 `TESSDATA_PREFIX`에 걸어,
  앱은 kor을 보는데 진단만 "eng, osd"라고 보고하던 불일치를 없앴다.
- **함정(실증)**: 배치 파일에 **한글을 넣으면 안 된다**. cmd는 배치를 콘솔 OEM 코드페이지
  (한국어 Windows = CP949)로 읽는데 파일이 UTF-8이면 주석·경로·메시지가 깨져 파싱까지 무너진다.
  그래서 `run_cocktail.bat`은 ASCII 전용이고 한국어 안내는 `ENVIRONMENT.md`가 맡는다.
  (cmd 자신이 찍는 `pause` 프롬프트는 OS가 한국어로 출력하므로 문제 없음.)
- **검증**: 3분기 실측 — 정상(anaconda `cd` env 선택 → 앱 기동),
  `COCKTAIL_PYTHON`을 패키지 없는 3.13으로 강제(지정 존중 + 진단이 결손 모듈 보고 + 에러 pause),
  후보 전멸(`PATH`/`USERPROFILE` 무력화 → 안내 메시지 + pause).

### DR-1. 사어(死語) 의존과 과장 문구가 문서·빌드에 남음
- **증상**: SL-1에서 코드에서 걷어낸 `langdetect`가 `requirements.txt`, `build.spec`,
  `diagnose_environment.py`, `pre_release_check.py` 4곳에 살아 있었다. 쓰지도 않는 패키지를
  설치 요구하고, PyInstaller가 번들에 넣고, 진단은 없으면 "필수 모듈 누락"이라고 겁을 준다.
  같은 결로 `osd.traineddata`(OSD script 판별 퇴역)도 인스톨러 컴포넌트·다운로드 목록에 남아 있었다.
  문서는 "100% 로컬 처리, 어떤 데이터도 외부로 나가지 않습니다"라고 했지만 실제로는
  최초 1회 모델 다운로드 + Tesseract 별도 설치가 필요하다.
- **수정**: 4곳 전수 제거(가드인 `smoke_tests.py`/`pre_release_check.py`의 **잔재 검사**는 유지 —
  그건 되살아나는 것을 막는 장치다). `LICENSE-3RDPARTY.md` §6은 "제3자 컴포넌트 없음"으로 교체.
  `download_tessdata.py`/`installer.iss`에서 osd 제외(파일 자체는 보존).
  문구는 "인식·번역 전 과정 로컬 처리(화면/번역 데이터 외부 전송 없음), 단 최초 1회 모델
  다운로드 + Tesseract 설치 필요"로 사실화(README/README.en/VISION/HANDOFF/release.yml).
- **재발 방지**: 의존성은 **코드에서 지웠다고 끝이 아니다.** 위 4곳 + 라이선스 문서 + 인스톨러가
  한 세트다. 그리고 마케팅 문구는 네트워크를 한 번이라도 쓰면 "100%"라고 쓰지 않는다.

### PY-1 / ZH-1 / M-9. 중국어 화면 번역 전면 붕괴 — 한자는 안 읽고 병음을 영어로 읽음 ⭐
- **증상**: 중국어 학습 앱(한 문장 = 병음 / 한자 / 앱 자체 한국어 번역 3층 구조)에서
  오버레이가 뜻이 안 통하는 문장을 뱉었다.
  - `"작은 미는 강 옆에 서서 소에 붉은 공을 들고, 차가운 왈로 가득 찬 쓰었다."`
  - `"4주간을 보며, 그녀는 "첫 번째 물 공은 누구에게 쏟아져야 하는가?""`
  정답은 `"샤오메이는 강가에 서서 손에 빨간 국자를 들고 있었습니다…"` /
  `"모두들 웃으며 뛰어다니며 서로에게 물을 뿌렸습니다…"` 였다.
- **원인 (fixture 재현으로 확정, 개발 PC 2026-08-11)** — **한자는 애초에 읽히지 않았다.**
  통념("kor traineddata 가 한자를 읽어 준다")이 실측에서 틀렸다.

  | 화면 | `eng+kor` (auto 기본) | `eng+kor+chi_sim` |
  |---|---|---|
  | 순수 중국어(합성) | **0줄**, CJK 0자 | 4줄, CJK 55자 |
  | 학습앱 스샷 | 15줄이지만 **CJK 0자** (병음 줄만) | 16줄, 한자 정상 |

  그래서 살아남은 건 한자 위의 **병음 줄**뿐이고, SL-1은 라틴을 전부 `en`으로 확정하므로
  병음이 영어로 취급돼 en→ko 모델에 들어갔다. `"xiǎoměi"`→`"작은 미"`, `"sìzhōu"`→`"4주간"`이
  그 흔적이다. 즉 번역 모델 문제가 아니라 **OCR 언어 라우팅 문제가 1차 원인**이었다.

  뒤이어 두 개가 더 있었다.
  1. Tesseract 는 한자를 글자마다 별개 word 로 준다. `" ".join` 이 `"大 家 笑 着"` 를 만들고
     번역기는 이걸 다른 문장으로 읽는다 — `"큰 집은 웃고"` vs `"모두가 웃고"`.
  2. zh→ko 전용 opus-mt 는 **없다**(`Helsinki-NLP/opus-mt-zh-ko`, `opus-mt-tc-big-zh-ko`
     둘 다 RepositoryNotFound). 그래서 m2m100_418M 폴백으로 흘렀고, 그 모델은 주어를
     통째로 흘렸다 — `"小美站在河边…"` → `"**은** 강 옆에 서서…"`.
- **수정**: 세 갈래.
  - **PY-1** `cocktail_ocr.is_pinyin_line()` / `drop_pinyin_lines()` — 병음 줄을 번역 입력에서
    제외. 판정은 (a) 강성 성조부호(마크론/캐런/ǖǘǚǜ) 또는 (b) 병음 음절 분해율 + 최장
    분해 토큰 길이. 어큐트/그레이브(áéíóú, àèìòù)는 프랑스어·스페인어에 흔해 **일부러 뺐다**.
    영어 6909줄 스윕으로 임계 확정: 비율 0.8 + 최장 7 → 거짓양성 6줄(0.09%).
  - **ZH-1** `BackgroundController._zh_rescan_lang()` + `_tess_scan()` — 근거가 있을 때만
    `chi_sim` 을 덧붙여 재OCR. 진입 근거는 "병음 줄이 잡혔다" 또는 "한 줄도 못 읽었다".
    한 번 켜지면 8프레임 동안 1차 패스를 건너뛰어 왕복 2회를 1회로 줄인다. 글자 없는
    화면에서 헛수고하면 20프레임 쿨다운(`OCR_ZH_EMPTY_COOLDOWN`)으로 재탐색을 쉰다.
    한자↔한자 경계는 `join_ocr_words()` 로 공백 없이 잇는다(라인 안·문단 병합 양쪽).
  - **M-9** `HybridTranslator.PIVOT_ROUTES` — zh→ko 를 zh→en→ko 피벗으로 라우팅.
    후보 실측(중국어 12문장, chrF): m2m100 직행 24.5 / 3rd-party `shun89/opus-mt-zh-ko` 28.3 /
    **피벗 32.0** / `opus-mt-tc-bible-big-mul-mul` 4.5(성경 코퍼스 전용이라 사용 불가).
    피벗을 고른 이유는 품질 1위 + 라이선스가 이미 배포 중인 en-ko 와 같은 **CC-BY-4.0**
    (Helsinki 공식)이라서다. shun89 는 신고 라이선스가 apache-2.0 이지만 베이스가
    CC-BY-4.0 opus-mt 라 신고를 믿을 수 없고 학습 데이터도 문서화돼 있지 않다.
  - 피벗 모델의 첫 로드는 **백그라운드**(`_ensure_bilingual(..., block=False)`)다. 로드가
    끝날 때까지 그 프레임은 m2m100 폴백이 처리한다 — 워커가 수십 초 얼지 않는다.
- **속도 (같은 fixture, 레버 off/on 실측)**: 영어는 **회귀 없음**.

  | fixture | before | after |
  |---|---|---|
  | 영어 문서 doc_55 (10문단) | 3183 ms | 3207 ms (+0.8%) |
  | 영어 UI live_primary (52줄) | 3974 ms | 4027 ms (+1.3%) |
  | 중국어 학습앱(합성) | 987 ms | 1659 ms |
  | 중국어 학습앱(실스샷) | 2098 ms | 3357 ms |

  중국어 화면에서만 늘어난다(chi_sim 패스 + 피벗 1회). 이게 ZH-1을 조건부로 만든 이유다.
  영어 쪽 ±1~5%는 Tesseract 실행 편차다 — PY-1 필터 자체는 52줄 영어 기준 **0.3 ms**
  (`_pinyin_segmentable`은 `lru_cache(4096)`), 진입 근거가 없으면 2차 패스는 아예 안 돈다.
- **재발 방지**: 5초 체크리스트에 4개 항목 추가.
  - "이 언어팩이 이 문자를 정말 읽는가"는 **믿지 말고 실측한다.** kor→한자 통념이 틀렸다.
  - 언어팩을 상시 추가하기 전에 **다른 언어 화면의 OCR 시간**을 먼저 잰다(+38~68%).
  - 새 휴리스틱의 임계는 **반대편 코퍼스**(여기서는 영어 6909줄)로 거짓양성을 세고 정한다.
  - 새 번역 모델은 라이선스뿐 아니라 **출처(공식/3rd party)와 학습 코퍼스**까지 본다.
    성경만 학습한 mul-mul 은 라이선스가 깨끗해도 쓸 수 없었다.

### TE-1 / VR-1. 번역 그룹 하나가 실패하면 **그 프레임 전체**가 사라짐 ⭐
- **증상**: 앱을 켜고 영어+중국어가 섞인 화면을 보면 오버레이가 **한 줄도** 안 떴다.
  영어 50줄은 멀쩡히 OCR 됐는데도 화면이 통째로 비었다. 로그에도 아무것도 안 남았다.
- **원인 (재현 3/3, 2026-09-05)**: `batch_translate`는 소스 언어별로 그룹을 만들어 루프를 돈다.
  중국어 1줄이 m2m100 폴백 로드에서 `CUDA out of memory`(16~20 MiB 요청 실패)를 냈고,
  그 예외가 루프 밖으로 그대로 올라가 **이미 번역된 영어 결과까지 함께 버려졌다**.
  OOM 자체는 용량 문제가 아니다 — 단독 프로세스에서는 m2m100이 4.0초에 정상 로드된다(982MB).
  다만 로드 순서가 `from_pretrained()`(fp32) → `.to(cuda)` → `.half()` 라서 **최종 크기의 2배**를
  순간 점유했다(m2m100 약 1.9GB 피크). en-ko + zh-en 이 이미 올라간 상태에서 그 피크가 터진 것.
  덤으로 그 4.0초 동기 로드가 워커 스레드를 통째로 얼렸다.
- **수정**: 세 갈래.
  - **TE-1** 그룹 격리 — 언어 그룹 루프를 `try/except`로 감싼다. 실패한 그룹의 줄만 `None`이
    되고(호출부의 기존 빈값 가드가 걸러 준다) 나머지 그룹은 그대로 표시된다.
    실패는 삼키지 않는다(CO-2): `[WARN]` 로그 + `LAST_GROUP_ERRORS` → 상태줄에 `"zh→ko 1줄 실패: …"`.
  - **VR-1** 로드 시점 fp16 — `from_pretrained(..., torch_dtype=torch.float16)`로 **처음부터**
    반정밀도로 올린다. 실측: en-ko(412MB) → +zh-en(561MB) → +m2m100(1488MB), 피크가 최종값과
    같다(=fp32 사본 없음). OOM은 이 수정 뒤 재현되지 않았지만 **환경 의존이라 해결로 단정하지
    않는다** — 다른 앱이 VRAM을 쓰는 상태에서 다시 볼 수 있다. 그때도 TE-1 덕에 화면은 안 죽는다.
  - **P-9b 확장** m2m100 비동기 로드 — `_ensure_multilingual(block=False)`. 첫 프레임은 그 그룹만
    포기하고(상태줄에 "폴백 모델 로드 중") 백그라운드 로드가 끝나면 다음 프레임부터 정상.
    실측: 폴백 필요 첫 프레임 4000 ms 정지 → **95 ms**, 6초 뒤 프레임에서 아랍어 줄 정상 번역.
- **재발 방지**: 5초 체크리스트에 "언어 그룹/배치 루프의 예외가 프레임 전체를 죽이지 않는가"
  추가. 테스트 `test_translation_group_failure_does_not_kill_other_groups`(실패 그룹만 빠지고
  나머지는 살아남는지 + 실패가 기록되는지).

### UB-1. 전체화면에서 확대 재OCR(OU-1)이 **구조적으로 한 번도** 발동하지 않음 ⭐
- **증상**: 영어 문서(어린왕자 p4, 1920x1080) 단어 인식률 **42%**(287개 중 120개), 유사도 0.541.
  못 읽은 58%는 번역될 기회조차 없다. 화면 밖에서는 "OCR이 원래 그렇다"로 보였다.
- **원인**: PF-1이 걸어 둔 예산이 `h*w*up*up ≤ 4M`. 1920x1080의 2배는 8.3M이라 **항상 거짓**이다.
  즉 전체화면 경로에서는 OU-1이 코드로만 존재했다. 밴드(PF-2)로 잘린 작은 이미지에서만 돌았다.
  더 근본적으로 **픽셀 수는 비용의 대리지표가 아니다** — Tesseract 비용은 면적이 아니라 글자 양을
  따른다. 실측(eng 단독, 3840x2160 같은 픽셀 수): 여백 많은 문서 1차 436 ms / 글자 빽빽 4787 ms.
- **수정**: 예산을 **1차 패스 실측 시간**으로 바꿨다(`OCR_UPSCALE_TIME_BUDGET_MS = 900`).
  ⚠️ **이 부분은 RP-1(2026-09-05)에서 대체됐다** — 실측 시간을 쓰면 같은 입력이 실행마다 다르게
  읽힌다. 지금은 `OCR_UPSCALE_CHAR_BUDGET(2700) / 1차 패스 글자 수`. "픽셀은 대리지표가 아니다"는
  아래 근거는 그대로 유효하고, 대리지표만 시간 → 글자 수로 바뀌었다.
  `배율 = min(2, √(예산 / 1차시간))`, `OCR_UPSCALE_MIN_FACTOR(1.3)` 미만이면 확대를 생략한다.
  비용이 배율²에 비례한다는 보수적 가정이라 실제보다 크게 잡는다(2배 실측은 2.3~2.7배).
  채택 기준은 기존 UA-1(평균 conf 개선) 그대로 — 확대해서 나빠지면 버린다.

  | 1920x1080 배율 | 1.0(구동작) | 1.3 | 1.5 | 1.75 | 2.0 |
  |---|---|---|---|---|---|
  | OCR ms | 296 | 796 | 892 | 957 | 1097 |
  | 단어 인식률 / 유사도 | 42% / 0.541 | 99% / 0.997 | 100% / 0.995 | 99% / 0.997 | 99% / 0.995 |

  → 1.3배면 정확도는 이미 포화한다. 900ms 예산은 1080p에서 1.7배, 4K 여백형에서 1.4배를 주고,
  글자 빽빽한 4K에서는 배율이 0.4로 떨어져 자동 생략된다(느린 화면을 더 느리게 만들지 않는다).
- **재발 방지**: 테스트 `test_upscale_fires_on_fullscreen_sized_input`(1080p 크기 입력에서 2패스가
  도는지). 체크리스트에 "OCR 비용 예산을 픽셀 수로 잡지 마라 — 글자 양이 비용이다" 추가.

### LC-1 / LC-1b. 목표 언어와 같은 언어팩을 상시 켜 OCR 시간 2.5배, 그런데 빼면 한글이 오독됨 ⭐
- **증상/원인**: auto 모드는 `eng+kor`을 고정으로 썼다. 실측 1920x1080 문서 기준
  `eng` 313 ms / `eng+kor` 771 ms(+146%) / `eng+kor+chi_sim` 1201 ms.
  그런데 목표가 한국어면 인식된 한국어 줄은 SK-1이 **전부** 버린다(실스샷 51줄 중 38줄).
  값어치 없이 2.5배를 내고 있었다.
- **수정 LC-1**: `OCR_LANGS_TGT_REDUNDANT = {"ko": "kor"}` — 목표 언어와 같은 언어팩만 뺀다.
  한→외 번역(목표가 한국어가 아님)에서는 `kor`이 원문을 읽는 유일한 수단이라 그대로 둔다.
  `COCKTAIL_OCR_LANGS_AUTO` 오버라이드는 최우선(빼는 규칙도 건너뛴다).
- **수정 LC-1b (LC-1이 만든 2차 결함, 실측으로 발견)**: `kor`을 빼면 한글 화면에서
  **라틴 오독이 자막으로 뜬다.** `"인벤토리"` → `"Be] Ast"` (conf 49), `"체력 회복 물약"` →
  `"Alp 28 5U"` (conf 41) 처럼 Tesseract가 자신 있게 틀리고, 그 오독이 en→ko로 번역돼
  15줄 중 **6줄이 엉뚱한 박스로 표시**됐다(기존 동작은 0줄). 프레임 conf 실측이 갈랐다:

  | 화면 | 한국어 UI | 한국어 게임 스샷 | 4K 빽빽 문서 | 중국어 앱 | 영어 UI/문서 |
  |---|---|---|---|---|---|
  | 평균 conf | 54.8 | 57.1 | 75.6 | 83.8 | 91.7~95.4 |
  | conf<72 라인 비율 | 100% | 80% | 29% | 20% | 0~6% |

  → 평균 conf < 68 **또는** 저conf 라인 ≥ 50%면 `kor`을 붙여 한 번 더 읽고, 평균 conf가 실제로
  오를 때만 채택한다(ZH-1과 같은 모양, UA-1과 같은 채택 규칙). 한글이 확인되면
  12프레임 sticky로 `kor`을 처음부터 켜 두어 왕복은 진입 프레임 1회뿐이고,
  헛수고였으면 20프레임 쿨다운. 실측: 진입 프레임 1353 ms, 이후 552 ms(구동작 573 ms와 동일).
- **재발 방지**: 테스트 2건 — `test_ocr_langs_keeps_kor_when_target_is_not_korean`(한→외에서 kor
  유지), `test_low_conf_latin_screen_rescans_with_kor`(저conf 라틴 프레임에서 kor 재스캔·채택·sticky).
  체크리스트: "언어팩을 뺄 때 **그 문자가 쓰인 화면**에서 오독이 표시되는지 반드시 본다.
  안 읽히는 것보다 **자신 있게 틀리는 것**이 나쁘다."

### FD-1. 프레임 변화 감지가 **본문 크기 글자 전부**를 놓침 (실시간 자막 번역이 구조적으로 불가) ⭐
- **증상**: 1920x1080에서 자막/본문 한 줄만 바뀌면 프레임 스킵에 걸려 OCR이 아예 안 돈다.
  실측 최대 셀 변화량(임계 `FRAME_DIFF_THRESHOLD=8`): 28px 글자 **3**, 34px **3**, 44px **7**
  (전부 스킵) / 60px 16, 80px 18 (통과). 본문·자막·채팅·툴팁은 전부 60px 미만이다.
  PF-2 변경 밴드도 이 게이트를 통과한 프레임에서만 발동하므로 같이 죽어 있었다.
- **원인**: `cheap_hash`가 화면을 **16x16 고정 격자**로 평균(INTER_AREA)냈다. 1920x1080이면
  셀 하나가 120x67px라 28px 한 줄의 기여가 셀 평균에 희석된다. 화면이 클수록 더 희석되는
  구조적 결함이라 4K에서는 60px 글자도 위험했다(실측 g16에서 12px→2, 20px→6).
- **수정**: 격자를 **셀 픽셀 수 기준으로 적응**시킨다 — `FRAME_HASH_CELL_PX=16`,
  격자 변 = `clamp(max(w,h)/16, 16, 256)`(정사각이라 해시 길이만으로 복원 가능, F-1 헤더 계약 유지).
  1920x1080→120, 3840x2160→240. 해상도가 달라도 검출력이 같다(아래 표).
  같이 생긴 캐럿(텍스트 커서) 오검출은 `FRAME_DIFF_MIN_ROW_CELLS=2`로 막았다 —
  캐럿은 폭 2~3px라 **어느 행에서도 셀 1개만** 변하고(실측 항상 1), 12px 한 줄 교체는 8셀이 변한다.

  | 글자 크기 (한 줄 교체, 최소 변화량) | 12px | 20px | 28px | 34px | 44px | 60px |
  |---|---|---|---|---|---|---|
  | 구동작 16x16 (1920x1080) | 2 | 5 | **3** | **3** | **7** | 16 |
  | 신동작 셀16px=g120 (1920x1080) | 27 | 79 | 64 | 80 | 109 | 111 |
  | 신동작 셀16px=g240 (3840x2160) | 27 | 75 | 63 | 80 | 109 | 118 |

  → 임계 8 대비 최소 마진 3.4배. 비용은 오히려 줄었다(1920x1080 해시 5.7 ms → **1.7 ms**;
  극단적 다운스케일 16x16보다 정수배에 가까운 축소가 INTER_AREA에 유리하다).
- **재발 방지**: 테스트 2건 — `test_small_text_line_change_is_detected`(1080p·4K × 12~60px
  한 줄 교체가 전부 검출되고 밴드 위치도 맞는지), `test_static_screen_is_skipped`
  (정적 50프레임 오검출 0 + 캐럿 깜빡임 스킵). 체크리스트에 "프레임 해시 격자를 고정 크기로
  잡지 마라 — 셀 픽셀 수를 고정해야 해상도 불변" 추가.

### CP-1. 캡처가 **항상 가상 데스크톱 전체**를 찍어 유휴 CPU를 먹음 ⭐
- **증상**: 정적 화면에서도 프로세스가 1코어 기준 **24%**를 계속 쓴다. 유휴 폴 1회 372 ms 중
  캡처가 116 ms(600x300 hover 영역도 113 ms — bbox 크기와 무관).
- **원인**: `ImageGrab.grab(all_screens=True)`는 win32 경로에서 가상 데스크톱 전체
  (여기선 5760x2160)를 BitBlt 한 뒤 bbox로 crop한다. W-5에서 "보조 모니터가 검게 나온다"는
  이유로 무조건 True로 박아 둔 것이 모든 프레임의 고정 비용이 됐다.
- **수정**: `needs_all_screens(x1,y1,x2,y2)` — 캡처 영역이 **주 모니터 안에 완전히** 들어갈 때만
  `all_screens=False`. 주 모니터는 정의상 가상 좌표계 원점(0,0)이라 bbox가 그대로 맞는다.
  크기는 `GetSystemMetrics(SM_CXSCREEN/SM_CYSCREEN)`로 읽는다 — PIL의 win32 캡처가 보는 값과
  같아서, DPI unaware 프로세스(DP-1)에서 Qt 지오메트리를 쓰는 것보다 안전하다.
  실측(주1920x1080@100% + 보조3840x2160@175% (-3840,-535)): ①주 모니터 안 ②보조 모니터 안
  ③두 모니터에 걸침 ④음수 원점 — 4케이스 모두 검은 픽셀 없이 정상 픽셀(캡처 이미지 육안 확인),
  분기 결과와 `all_screens=True` 기준의 차이는 **시간차 잡음보다 작다**(7.6% vs 18.3%).

  | | 캡처 | 해시 | 유휴 폴 1회 | 유휴 CPU(1코어) |
  |---|---|---|---|---|
  | before | 116 ms | 5.7 ms | 372 ms | 24.2% |
  | after | 32 ms | 1.7 ms | 284 ms | **8.0%** |
- **재발 방지**: 테스트 `test_capture_uses_primary_fast_path_only_inside_primary`
  (주 모니터 전체/내부는 빠른 경로, 음수 좌표·좌우 걸침·아래 걸침은 전부 전체 캡처).
  레버 `CAPTURE_PRIMARY_FAST_PATH=False`로 즉시 구동작 복귀 가능.
  체크리스트: "보조 모니터 회귀(W-4/W-5)가 의심되면 4케이스를 **캡처 이미지로** 확인한다."

### UI-1. 설정창이 고정 픽셀 배치 — 창을 줄이면 버튼이 창 밖으로, 고DPI면 글자가 잘림 ⭐
- **증상**(채점판 실측 2026-09-05): 창 밖으로 나간 위젯 **14개**, 글자 잘림 **16개**.
  640x400에서 "스샷차단 / 영역 편집… / 확인" 3개, 480x220에서 "번역 시작"까지 4개가 창 밖.
  폰트 DPI 168에서는 창 크기와 무관하게 버튼 3개 + 상태 라벨의 글자가 폭을 넘겼다
  (예: "모델 준비 중…" 147px 필요 / 100px 확보).
- **원인**: `initUI()`가 위젯을 전부 `setGeometry(x, y, w, h)` 절대 좌표로 놓았다.
  좌표는 상수인데 **글자 크기는 상수가 아니다** — 고DPI에서 폰트만 1.75배가 되면 폭은 그대로다.
  창 크기도 마찬가지로 위젯이 따라가지 않는다.
- **수정**: `QVBoxLayout(가로 한 줄 + 상태줄 + 안내줄)`로 교체. 배치·기능·순서는 그대로다.
  - 레이아웃이 자식의 `sizeHint`를 최소 폭으로 삼기 때문에 **창 최소 크기가 자동으로 생긴다**
    (실측 772x104, 폰트 DPI 168에서 1146x128). 그 아래로는 창이 줄지 않으므로 이탈이 원천 봉쇄된다.
  - 버튼은 `_FitButton` — `setText()` 한 곳에서 `최소폭 = 글자폭 + TEXT_PAD_PX(20)`를 갱신한다.
    버튼 문구는 실행 중에 바뀌므로("번역 시작"→"모델 준비 중…"→"⚠ …") 호출부마다 폭을 재면
    빠뜨린 경로가 생긴다. 글꼴이 바뀌면(`QEvent.FontChange`) 다시 잰다.
  - 상태 라벨·안내 라벨은 `QSizePolicy.Ignored` + `_elide()`(말줄임 + 전체는 툴팁).
    긴 상태 문구가 창 최소 폭을 밀어 올리는 것을 막고, 잘린 채 보이는 것도 막는다.
    폭은 라벨이 아니라 **창**에서 뽑는다 — `resizeEvent` 시점의 라벨 폭은 아직 옛 값이다.
- **실측 before→after**: 창 밖 위젯 14→**0**, 글자 잘림 16→**0** (`python bench/score.py --only 5`,
  4가지 창 크기 × 기본/`QT_FONT_DPI=168`). 기준 5 실패→**통과**.
- **재발 방지**: 테스트 `test_settings_window_keeps_widgets_inside_at_any_size`
  — 창을 실제로 띄워 880x120 / 1400x900 / 640x400 / 480x220에서 말단 위젯의 이탈과 글자 잘림을 본다.
  (레이아웃을 떼고 옛 좌표로 되돌리는 회귀를 실제로 잡는 것까지 확인했다.)
- **부수 발견**: 테스트 헬퍼 `_make_controller`가 `QCoreApplication`을 만들고 있어서, 뒤에 오는
  창 테스트가 `QWidget`을 만드는 순간 프로세스가 **조용히 죽었다**(에러 없이 exit 127).
  애플리케이션 객체는 나중에 GUI용으로 바꿔 끼울 수 없다 → `_qt_app()`이 `QApplication`으로 만든다.

### UI-2. DP-1의 대가 — 175% 모니터에서 **글자만** 커져 설정창이 잘림 ⭐
- **증상(실측으로 정정됨)**: 인수인계 문서에는 "175% 모니터에서 설정창만 60% 크기"로 적혀 있었다.
  실제로 창을 보조 모니터(-3840, 3840x2160@175%)로 옮겨 `grab()`으로 렌더해 보니 **상자는 그대로,
  글자만 1.75배**가 되어 "Chine…", "델 준비 중", "역 편집"처럼 잘렸다. 상대적으로 창이 작아 보인 것.
- **원인**: DP-1(`QT_ENABLE_HIGHDPI_SCALING=0`)은 캡처·오버레이 좌표를 물리 픽셀로 통일하려고 켠 것이라
  **되돌리면 안 된다**. 그런데 스케일링을 꺼도 **글자는** 창이 놓인 모니터의 DPI로 그려진다
  (포인트 크기 → 화면 DPI로 환산). 반면 레이아웃이 쓰는 `fontMetrics()`는 주 모니터 DPI(96) 그대로다.
  → 그리기와 측정이 서로 다른 DPI를 본다.
- **수정**: `_apply_screen_font_scale()` — 설정창 글꼴을 **포인트 크기가 아니라 픽셀 크기**로 준다
  (`픽셀 = 9pt × 그 화면 ldpi / 72` → 96dpi에서 12px, 168dpi에서 21px). 픽셀 크기 글꼴은 어느 화면에서든
  그 픽셀이라 그리기와 측정이 다시 일치하고, UI-1의 레이아웃이 그만큼 창을 키운다.
  `moveEvent`에서 화면이 바뀔 때만 재적용. **좌표는 하나도 건드리지 않는다.**
- **함정 2개(둘 다 실측으로 걸림)**:
  - 부모(`QMainWindow`)에 `setFont`를 해도 자식의 `font()`는 안 따라온다(스타일시트가 걸린 창에서
    확인). 레이아웃이 옛 크기로 계산하므로 **자식에 직접** 줘야 한다.
  - `layout.invalidate()` 없이 `activate()`만 하면 **옛 글꼴로 캐시한** 크기를 그대로 돌려준다.
- **실측**: 보조 모니터로 이동 → 글꼴 12px→21px, 창 872→**1241px**, 이탈/잘림 **0**.
  주 모니터로 되돌리면 12px로 복귀. 주 모니터에서는 12px = 기존 9pt@96dpi와 같은 값이라 무변화.
  `QT_FONT_DPI` 강제 환경에서도 같은 픽셀 값이 나와 이중 적용(E-7 재발)이 없다.
- 좌표를 곱하는 방식으로는 글자를 못 키운다 — 그래서 항등 함수로 남아 있던 `self.px`는 삭제했다.
- **한계**: 오버레이·캡처는 그대로 물리 픽셀이다. "175% 모니터에서 **자막** 위치가 맞는가"는
  여전히 사람이 확인할 항목이며 이 수정의 범위가 아니다.

### UI-3. 첫 실행에 설정창만 뜨고 "무엇을 눌러야 자막이 나오는지"가 없음
- **증상**: 첫 실행 분기는 있는데(`_maybe_show_initial_settings`) 창이 뜨는 것으로 끝난다.
  처음 쓰는 사람은 콤보 4개와 버튼 4개를 보고 무엇부터 눌러야 할지 알 수 없다.
- **수정**: 상태줄 아래 한 줄짜리 안내 라벨(`hint_label`) — 첫 실행에만 보인다.
  "처음이신가요? 번역할 창을 띄운 뒤 [번역 시작]을 누르면 자막이 뜹니다."
  툴팁에 3단계 전체(창 띄우기 → 캡처 모드 → 번역 시작 / 닫아도 트레이 상주, Ctrl+Shift+A).
  `toggle_running()`(안내대로 눌렀다)과 `_mark_settings_seen()`(창을 닫았다)에서 숨긴다.
  마법사·다이얼로그를 만들지 않은 이유: 상주형 앱에서 첫 실행 모달은 제거 대상이 되기 쉽고,
  안내가 필요한 시점은 **그 버튼을 보고 있을 때**다.

### IC-1. 트레이 아이콘이 빨간 사각형 placeholder
- **증상**: `QPixmap(16,16).fill(QColor(220,60,60))` — 트레이에 빨간 네모가 뜬다. 창 아이콘도 없다.
- **수정**: `make_app_icon()`이 코드로 그린다(QPainter). **외부 이미지를 받지 않는다** —
  ICONS.md §6의 라이선스 오염을 피하기 위해서고, 그래서 이 아이콘의 저작권은 이 프로젝트에 있다.
  도안은 ICONS.md §2 그대로: 따뜻한 오렌지(#E06C2B) 둥근 사각형 + 원문/번역 글자 쌍 `A` → `가`
  (번역기 아이콘의 관용 구도). 트레이·창 아이콘 공용, 16/32/64/256 4종.
- **실측 함정**: 16px에 직접 그리면 글자에 **서브픽셀 색 번짐**(빨강·파랑 테두리)이 남고
  "가"가 오른쪽에서 잘린다. 렌더 PNG를 8배로 확대해 눈으로 확인하고 두 가지를 고쳤다 —
  8배로 크게 그린 뒤 `SmoothTransformation`으로 축소, 글자는 `NoSubpixelAntialias`.
- **남은 것**: `assets/icon.ico`(exe/인스톨러용)는 아직 없다. ICONS.md §3의 변환은 B4(설치본)에서.

### PY-2. 띄어쓰기된 병음을 못 알아본다 — 중국어 화면 전체가 무너지는 첫 도미노 ⭐
- **증상**: 세 가지가 한 사슬로 일어난다.
  1. `bench/fixtures/zh_pinyin.png` 의 한자 5문단 중 **0개** 인식(합격기준 3 "인식실패로 측정불가").
  2. 병음 줄이 영어로 확정돼 번역기에 들어가 자막으로 뜬다 — `"ni hao, huan ying shi yong
     zhe ge ying yong"` → `"니 하오, 후안 이 쯔 옌"` (합격기준 2 확정 오역 1건).
     나머지 두 줄은 DG-1 게이트에 우연히 걸려 가려졌을 뿐 설계로 막힌 게 아니다.
  3. Tesseract 자체는 같은 이미지를 `chi_sim` 으로 **완벽히 읽는다** — 라우팅 문제다.
- **원인**: PY-1 판정 (b)가 **붙여쓴 병음**(`xiǎoměizhànzàihébiān`)만 상정하고
  `PINYIN_MIN_LONG_TOKEN=7`(7자 이상 토큰 하나)을 요구했다. 실제 학습앱은 음절을 띄어 쓰므로
  토큰이 전부 2~4자라 (b)가 통째로 False가 된다. 판정 (a)(강성 성조부호)도 13px 병음에서는
  OCR이 성조를 날려 먹어 안 걸린다. 그래서 ZH-1 재OCR 진입 근거 (a)가 사라지고, 병음 줄이
  영어로 읽히니 근거 (b)("1차 패스 0줄")도 성립하지 않아 **`chi_sim` 패스가 영원히 안 돈다.**
- **수정**: PY-1에 판정 (c)를 추가한다 — 최장 토큰 대신 **분해되는 토큰의 개수**로 건다.
  `PINYIN_MIN_SPACED_TOKENS=6` + `PINYIN_SPACED_MIN_RATIO=0.90`(중국어 확정 화면에서는
  `PINYIN_ZH_MIN_SPACED_TOKENS=4` — chi_sim 패스가 병음 한 줄을 두 토막으로 쪼개 놓는다).
- **거짓양성 실측**(부정 코퍼스 55,156조각 = 영어 54,993 + 불어 91 + 독어 72, 10/4단어 분할):
  | 규칙 | 실제 병음 재현 | 거짓양성 |
  |---|---|---|
  | 현행((c) 없음) | 5/16 | 92 (0.167%) |
  | 개수 4 / 비율 0.90 | 14/16 | 135 (0.245%) ← 영어를 먹기 시작 |
  | **개수 6 / 비율 0.90** | **14/16** | **93 (0.169%)** ← 채택 |
  | 개수 7 / 비율 0.90 | 13/16 | 92 (0.167%) |
  채택값이 새로 만든 거짓양성은 전 코퍼스에서 **1건**(`to change a[i:ai] into`)이고
  **불어·독어 거짓양성은 늘지 않았다**(1→1). `café` / `Saint-Exupéry` 회귀 없음.
- **남은 것**: 3토막짜리 병음(`ni hao ma`, `xie xie ni`)은 여전히 못 잡는다. 개수를 3까지
  낮추면 영어 거짓양성이 0.167% → 1.6%로 뛴다(zh 모드 기준). 4토막 이상만 잡는 것이 균형점.
- **재발 방지**: 새 라인 필터 임계는 반드시 **반대편 코퍼스**(영/불/독)로 거짓양성을 세어 정한다.
  그리고 "병음처럼 생긴 것"의 형태를 하나만 상정하지 마라 — 붙여쓰기·띄어쓰기 둘 다 나온다.

### ZH-2. ZH-1 진입이 병음 판정 하나에만 걸려 있었다 (구조적 취약) ⭐
- **증상**: PY-2를 고치면 `zh_pinyin` 시료는 살아나지만, **병음이 없는 순수 한자 화면**에
  영어 UI 라벨이 한 줄만 있어도 진입 근거 (b)("1차 패스 0줄")가 깨져 `chi_sim` 이 안 돈다.
  근거 (a)(병음)와 (b)(0줄)가 사실상 같은 실패를 공유한다.
- **원인**: 진입 신호가 둘 다 "1차 패스의 **텍스트 출력**"에서만 나온다. 1차 패스가 못 읽은
  글자에 대해서는 아무 정보도 없다.
- **수정**: 근거 (c) — `unread_ink_ratio(gray, boxes)`. Otsu 이진화의 소수 클래스를 잉크로 잡고,
  인식된 라인 박스가 덮지 못한 잉크 비율이 `OCR_ZH_UNREAD_INK_RATIO=0.6` 이상이면 진입한다.
  병음 판정과 **완전히 독립**이고 언어에 무관하며 결정론적이다.
- **실측**(진입 판정 지점 그대로 = 1차 패스 + LC-1b kor 재스캔 후):
  영어 문서/UI/다중문맥/크기계단 **0.000~0.197** · 한국어 화면 **0.000**(kor 재스캔이 먼저 읽는다)
  · 중국어 화면 **1.000**. 임계 0.6이 두 무리 한가운데다.
- **실측 함정**: 판정 전에 512px로 축소해 값싸게 하려다 되돌렸다. 다크 테마 UI의 가는 획이
  `INTER_AREA` 에 뭉개져 Otsu가 패널 경계를 잉크로 잡고, `ui_en_dense` 가 0.041 → **0.659**로
  튀어 임계를 넘겼다(영어 화면에서 `chi_sim` 패스 +1.1초). 원본 해상도 그대로가 1080p에서 8 ms다.
  **축소는 비용이 아니라 정확도를 깎는 쪽이었다.**
- **쿨다운**: 진입해서 `chi_sim` 을 태웠는데 한자가 안 나오면(사진/게임 화면) `_zh_frames` 를
  `-OCR_ZH_EMPTY_COOLDOWN` 으로 떨어뜨린다. 구현은 "0줄일 때만" 걸려 있었는데 (c)로 들어온
  경우를 못 덮어서, 진입 판정 자체를 `entered_zh` 플래그로 바꿔 근거 3종 전부에 적용한다.
- **재발 방지**: 비싼 재시도의 진입 조건을 **한 종류의 신호**에만 걸지 마라. 그 신호가 실패하는
  화면이 곧 그 재시도가 가장 필요한 화면이다.

### RP-1(재현성). 같은 입력이 실행마다 다르게 읽힌다 — 채점판의 신뢰가 깨진다 ⭐
- **증상**: 같은 이미지를 5번 넣으면 5번 다른 결과가 나온다. 채점판 판정이 "확정 오역 1건 ↔
  0건"까지 뒤집힌다. 실측(전 시료 5회, 2~5회차는 CPU 부하 4프로세스):
  **구 규칙 서로 다른 결과 5종**(`ui_en_dense` 줄 수 20/21/22, `fullhd_doc` 12/10), 갈린 시료 5개.
- **원인**: UB-1 확대 배율이 `min(2, √(900ms / 1차 패스 **실측 시간**))` 이라 PC 부하가 그대로
  인식 결과로 샌다. `ui_en_dense` 는 1차 패스가 223 ms인데 배율 2배의 경계가 225 ms라
  **부하에 따라 배율이 2.0과 1.9 사이를 오갔다.**
- **수정**: 시간 대신 **1차 패스가 실제로 읽어낸 글자 수**를 예산 단위로 쓴다.
  `배율 = min(2, √(OCR_UPSCALE_CHAR_BUDGET(2700) / 1차 글자 수))`. 이미지에서 결정론적으로
  나오는 값이고, "픽셀 수는 비용의 대리지표가 아니다"라는 UB-1의 원래 근거도 그대로 지킨다
  (비용은 면적이 아니라 글자 양을 따르므로).
- **대리지표 실측**(프로세스 기동 고정비 130 ms를 뺀 순수 작업시간 / 글자 수):
  `fullhd_doc` 1320자 0.236 ms/자 · `doc_en_column` 766자 0.210 · `multi_context` 654자 0.207.
  예산 2700은 구 시간 규칙과 같은 지점에 선다 — 1080p 문서 1.43배(구 1.43), 문서/다중문맥
  1.9~2.0배(구 1.76~1.84), 4K 빽빽(≈2만 자)은 0.37로 떨어져 자동 생략.
- **검증**: 전 시료 5회 반복 해시 완전 일치(2~5회차는 CPU 부하 하). 구 규칙 대조군은 5회 전부 불일치.
- **재발 방지**: **분기 조건에 `time.perf_counter()` 를 넣지 마라.** 실측 시간을 예산으로 쓰고
  싶으면 그 시간을 이미지에서 결정론적으로 예측하는 대리지표로 바꿔라. 재현 불가능한
  채점판은 채점판이 아니다.

### DG-1b. 극소 라인 conf 임계 70이 **정상 줄만** 잡고 있었다
- **증상**: 다크 테마 UI의 `4 spaces`(11px 라벨)가 **정확히 읽혔는데도**(정답과 완전 일치)
  DG-1 규칙 1에 걸려 숨는다 → 채점판 오탐 1/68 = 1.5%(기준 2 실패). RP-1을 고치기 전에는
  실행마다 뜨락말락해서 이 결함이 노이즈에 가려져 있었다.
- **원인**: `DISPLAY_TINY_LINE_MIN_CONF=70` 은 근거 없이 잡은 값이다.
- **실측**(bench 전 시료 중 `line_px < 16` 라인 36개 + 6~12px 합성 캡션 코퍼스):
  | 부류 | conf |
  |---|---|
  | 정상으로 읽힌 극소 라인 | **64.0**(`4 spaces`) · 나머지 전부 81~96 |
  | 오독/부분 인식 극소 라인 | **46.8** (`ae provisional November ean`, 정답유사 0.66) |
  | 6~7px 오독 쓰레기 | 49.7~73.2 — 단 라인 높이 14px 이상이라 이 규칙 밖이다 |
  px<12 구간에서 실측된 **유일한** 쓰레기가 46.8이고 **유일한** 정상 탈락이 64.0이다.
- **수정**: `DISPLAY_TINY_LINE_MIN_CONF` 70 → **60**. 46.8은 여전히 막고 64.0은 통과시킨다.
- **재발 방지**: 게이트 임계를 감으로 잡지 마라. "이 임계가 실제로 무엇을 잡았는가"를
  시료 전체에서 세어 보면, 쓰레기를 한 건도 안 잡고 정상만 잡고 있는 경우가 있다.
  숨김은 무음 실패라 사용자가 원인을 못 찾는다(CO-2와 같은 교훈).

### LM-1c(기록만). 메뉴바/툴바 항목이 한 줄로 뭉친다 — 간격 규칙으로는 못 가른다
- **증상**: `ui_en_dense` 에서 메뉴바 6항목이 한 줄(`File Edit Selection View Terminal Help`),
  툴바 5항목이 한 줄(`Save Q Search Holy Settings G Refresh AN Warnings` — 아이콘 글리프가
  `Q`·`Holy`·`G`·`AN` 으로 오인식돼 섞인다)로 합쳐진다. 기준 3 채점 대상은 아니지만 8건.
- **시도**: `OCR_LINE_SPLIT_GAP_RATIO` 를 3.0에서 낮추면 갈린다. 실측 —
  임계 1.5에서 **전 시료 문단 경계 오류 0건**(ui_en_dense 8→0), 10/12/14px CER 0.0% 유지.
- **왜 안 했나 (실측 근거)**: 같은 임계가 **양끝맞춤(justified) 본문을 산산조각 낸다.**
  단어 간격을 늘려가며 잰 파괴 지점:
  | 단어 간격(글자 높이 배수) | 임계 1.5 | 임계 2.0 | 임계 3.0(현행) |
  |---|---|---|---|
  | 1.0배 | 1줄 | 1줄 | 1줄 |
  | 1.2배 | **10줄로 파괴** | 1줄 | 1줄 |
  | 1.5배 | 10줄 | **8줄로 파괴** | 1줄 |
  | 2.2배 | 10줄 | 10줄 | 1줄 |
  그리고 메뉴바 항목 간격은 **1.83배**다 — 정확히 justified 본문의 위험 구간 안에 있다.
  즉 **가로 간격이라는 신호 하나로는 두 경우를 원리적으로 못 가른다.** 실제 시료의 정상
  단어 간격 최대치는 0.89배(전 시료)이지만, 그건 시료가 전부 왼쪽 정렬이라 그렇다.
  errors.md에 이미 같은 종류의 사고 기록이 있다(간격 규칙 0.7에서 본문 14줄→8줄).
- **올리는 길**: 레이아웃 분석(B2, PP-OCRv6)이나 "이 줄이 문장으로 이어지는가"의 언어적 신호가
  필요하다. 간격 상수를 만지는 것은 한쪽을 고치고 다른 쪽을 깨는 이동이지 개선이 아니다.

### LZ-1. 창이 뜨기까지 12~14초 — 최상단 `import torch` 하나가 시작 전체를 인질로 잡음 ⭐
- **증상**: 프로세스 시작 → 설정 창 12~14초, → 첫 오버레이 22초. 사용자가 아이콘을 두 번
  누르는(중복 실행) 종류의 지연이다. 합격기준 6("준비 5분")의 절반이 여기서 사라진다.
- **실측 분해(2026-09-05)**: `import cocktail_ui` **10.24s**. 그중 `cocktail_translate` 10.5s —
  torch 2.3s + transformers 2.0s(그 안에서 `torch._dynamo` 2.0s + `sklearn.metrics` 1.4s를
  또 끌고 온다) + 자체 1.0s. **CUDA 초기화 자체는 12ms로 무관하다** — 범인은 import다.
- **근본 원인**: 모델 로드는 이미 백그라운드 스레드(`BackgroundController._preload_model`)로
  옮겨 놨는데(P-9), **모듈 최상단 import**가 그 앞에 그대로 남아 있었다. 즉 "lazy load"라고
  적어 놓고 실제로는 `import` 한 줄이 창보다 먼저 4.3초를 썼다.
  숨은 두 번째 경로도 있었다 — `environment_summary()`가 진단 한 줄("번역 CUDA")을 찍으려고
  torch를 import했고, 이 함수는 `BackgroundController.__init__` = **창이 뜨기 전 메인 스레드**에서 돈다.
- **수정**:
  - `cocktail_translate`: 최상단 `import torch` / `from transformers import ...` 삭제.
    `_ensure_torch()`(lock + 1회 캐시)가 `device`/`USE_FP16`/`_LOAD_DTYPE` 전역을 채우고,
    transformers 클래스는 `OpusMtTranslator`/`M2M100Translator.__init__` 안에서 import한다.
    두 생성자는 어차피 모델 로드 스레드에서만 불린다 — 워커/메인은 얼지 않는다.
  - `environment_summary(probe_torch=False)`: 기본은 `sys.modules`만 보고 미로드면 "준비 중".
    모델 로드가 끝나면 엔진이 `_refresh_env_status()`로 다시 불러 CUDA/CPU를 확정한다.
  - `--self-test`는 반대로 **일부러 무겁게** 간다(`probe_torch=True` + `import transformers`).
    앱 시작 경로에서 빠진 만큼, "spec은 있는데 DLL이 깨져 실제 import가 실패"하는 배포 사고를
    자가진단이 놓치면 안 된다. 자가진단은 사람이 기다리는 명령이 아니다.
- **실측 before→after**: `import cocktail_ui` 10.24s → **1.05s**.
  프로세스 시작 → **창 보임 12~14s → 1.19~1.41s**(패키지 exe 1.41s / 소스 1.19~1.8s).
  모델 준비 완료는 18~20s → **15.3s**(소스). 창이 뜬 뒤에 도는 일이라 체감에서 빠진다.
- **회귀 가드**: `test_window_opens_without_importing_torch` — **별도 프로세스**에서
  `SettingsWindow()`를 만든 뒤 `sys.modules`에 torch/transformers가 없음을 assert한다.
  같은 프로세스에서 재면 다른 테스트가 이미 torch를 올려놔 아무것도 증명하지 못한다.
  `bench/score.py` 기준 6도 `torch_imported_at_ui_import`를 판정 조건에 넣었다.
- **함정**: 모듈 전역을 지연 초기화할 때 `torch = _torch` 대입을 **맨 마지막에** 둔다.
  다른 스레드는 `torch is not None`을 "준비 완료"로 읽기 때문에, device/dtype보다 먼저
  대입하면 절반만 초기화된 상태가 노출된다.

### PK-1. 배포본이 Tesseract를 사용자에게 따로 설치시킴 — "준비 5분"이 깨진다
- **증상**: 인스톨러는 traineddata만 깔고 `tesseract.exe`는 `C:\tesseract` 또는
  `C:\Program Files\Tesseract-OCR`에 사용자가 미리 설치돼 있기를 기대했다.
  깨끗한 PC에서는 OCR이 통째로 죽고, 사용자는 "설치 파일 하나"가 아닌 두 개를 받아야 한다.
- **수정**: `build.spec`이 `C:\tesseract`(`COCKTAIL_TESSERACT_DIR`로 변경 가능)에서
  `tesseract.exe` + DLL 56개 + `doc/LICENSE`·`doc/AUTHORS`를 `tesseract/` 한 폴더로 동봉한다.
  `binaries`가 아니라 `datas`로 넣는다 — PyInstaller가 이 DLL들을 분석해 재배치하면
  tesseract.exe 옆에서 흩어져 엔진이 안 뜬다. 통째로 한 폴더에 두는 것이 유일한 배치다.
- **탐색 순서**: `TESSERACT_CMD` → **동봉본**(`<앱폴더>/tesseract/`, `sys._MEIPASS/tesseract/`)
  → `C:\tesseract` → Program Files. 우리가 버전을 아는 쪽이 이긴다.
- **함정**: PyInstaller 6의 onedir는 `datas`를 `dist/<app>/_internal/`에 넣는다.
  `sys.argv[0]` 폴더만 보던 기존 tessdata 탐색은 그래서 동봉본을 못 본다 →
  `_SEARCH_DIRS = [앱폴더, sys._MEIPASS]` 둘 다 본다(앱폴더 우선 — 인스톨러가 사용자 선택
  언어팩을 `{app}\tessdata`에 깔기 때문).
- **라이선스**: Tesseract는 Apache 2.0. 재배포는 라이선스 원문·NOTICE 동봉이 조건이라
  원본 배포판의 `doc/LICENSE`·`doc/AUTHORS`를 같이 넣는다(`LICENSE-3RDPARTY.md` §3).
- **검증**: `dist/Cocktail/Cocktail.exe --self-test` → `[INFO] Tesseract: ...\_internal\tesseract\tesseract.exe`,
  언어 10종 인식, `[SELFTEST] OK`. **파이썬 없이 도는 것을 실제로 확인한 유일한 테스트다.**

### PK-2. 첫 실행에 모델 1.6GB를 받게 할 것인가 — 기본 언어쌍만 동봉하기로 결정
- **결정**: `Helsinki-NLP/opus-mt-tc-big-en-ko`(safetensors 399MB)**만** 동봉한다.
  `opus-mt-zh-en`(300MB)과 `m2m100_418M` 폴백(1.9GB)은 그 언어를 실제로 번역할 때 내려받는다.
- **근거**: ① 이 앱의 강점이 오프라인·프라이버시인데 첫 실행에만 인터넷을 요구하면 그 약속이
  첫인상에서 깨진다. ② "준비 5분"이 회선 상태에 걸리면 안 된다. ③ 반대로 전부 동봉하면
  인스톨러가 3GB를 넘고, 대부분의 사용자가 쓰지도 않을 100언어 폴백이 그 대부분을 차지한다.
- **크기 함정**: HF 캐시의 en-ko 스냅샷은 **1.6GB**지만 실제로 필요한 건 399MB다.
  같은 가중치의 `pytorch_model.bin`(399MB)·`tf_model.h5`(806MB) 사본과 벤치마크 zip을
  transformers는 쓰지 않는다. `build.spec`의 `_MODEL_SKIP`이 그것들을 뺀다.
- **로드 경로**: `_bundled_model_dir()`이 `<탐색폴더>/models/<모델 leaf>/config.json`을 찾으면
  repo id 대신 그 폴더를 `from_pretrained`에 넘긴다. `_load_source_spm`의 `snapshot_download`도
  같이 우회해야 한다 — snapshot_download는 repo id만 받으므로 동봉 폴더를 그대로 스냅샷으로 쓴다.
  실측 확인: 동봉 폴더에서 sepvoc(M-8) 감지까지 정상, `"Storage almost full"` → `"저장 거의 가득"`.
- **라이선스**: 이제 우리가 CC-BY-4.0 가중치를 **재배포**한다(전에는 사용자 PC가 HF에서 받아
  갔으므로 배포가 아니었다). 저작자 표시 문구 동봉이 선택이 아니라 의무가 됐다 —
  `LICENSE-3RDPARTY.md`가 `build.spec`의 `project_docs`로 `_internal/`에 함께 설치되는 것을 확인.
- **남은 것**: 동봉 안 된 언어쌍은 첫 사용 시 다운로드가 도는데 **진행률(%)이 없다.**
  상태줄에 "모델 준비 중: 모델 다운로드/로드 중"만 뜬다(멈춘 것처럼 보이지는 않지만 남은 시간을 모른다).

### IC-2. exe/인스톨러 아이콘이 없다 — 그런데 아이콘은 이미 코드 안에 있었다
- **증상**: `build.spec`의 `icon=None`, `installer.iss`에 `SetupIconFile` 없음.
  탐색기·작업표시줄·설치 마법사에서 전부 기본 아이콘.
- **수정**: 외부에서 이미지를 받지 않는다(ICONS.md §6 라이선스 오염 금지).
  `cocktail_ui.make_app_icon()`(IC-1에서 QPainter로 그린 'A→가')을 그대로 써서
  `python cocktail.py --make-icon` → `assets/icon.ico`(16/32/48/64/128/256 멀티사이즈, 16KB).
  QIcon의 각 크기 pixmap을 PNG로 뽑아 Pillow `append_images`로 합친다. build.spec·installer.iss가 참조.
- **검증 함정**: 채점판에 "exe에 아이콘 리소스가 박혔는가"를 넣었더니 처음엔 항상 false였다.
  `LoadLibraryExW`의 restype을 안 정해 64비트 HMODULE이 c_int로 잘렸다(D-1과 같은 함정).
  고치자 이번엔 프로세스가 조용히 죽었다 — `EnumResourceNamesW` 콜백의 타입/이름 인자는
  **정수 ID일 수 있는데**(MAKEINTRESOURCE) `LPCWSTR`로 선언하면 ctypes가 그 정수를 포인터로
  알고 역참조한다. 둘 다 `c_void_p`로 받아야 한다. 양성(Cocktail.exe=True)·음성
  (PyInstaller 부트로더 runw.exe=False) 대조로 확인했다.

### PL-1. 워커 폴 간격이 `time.sleep(0.25)` 고정 — 캡처를 3.6배 빠르게 만들고도 체감은 그대로 ⭐
- **증상**: 사용자 제보 "자막이 2~3초, 견딜 만하지만 답답". 그런데 채점판 기준 4는 통과(p50 1.05s)였다.
- **원인**: 워커 루프의 대기가 세 군데 모두 상수였다(`0.25` / `0.25` / `0.3`).
  B0-2에서 캡처를 116→32 ms로 줄였는데 폴 주기는 그대로라, 화면이 바뀌고 나서
  **다음 캡처가 시작될 때까지 매번 250 ms**가 그대로 붙었다. 게다가 한 사이클이 끝난 뒤에도
  무조건 250 ms를 더 쉬어서 사이클 **주기**가 그만큼 길어졌고, 그것이 QU-1의 큐잉을 만들었다.
- **실측 분해**(1920x1080 실기 하네스, 자막이 2초마다 바뀌는 화면):
  | 구간 | 영상 자막 | 채팅(전체 스크롤) |
  |---|---|---|
  | 앞 사이클 대기(직렬) | 155 ms | 806 ms |
  | **폴 대기** | **203 ms** | **251 ms** |
  | 캡처 | 27 ms | 28 ms |
  | OCR | 371 ms | 1287 ms |
  | 번역 | 252 ms | 254 ms |
  | 그리기(신호→paintEvent) | 68 ms | 196 ms |
- **수정**: `BackgroundController.poll_interval()` — 최근에 일거리가 있었으면 `POLL_ACTIVE_S=0.06`,
  아무 일 없이 `POLL_ACTIVE_WINDOW_S=1.2`가 지나면 `POLL_IDLE_S=0.25`(**예전 값 그대로**).
  루프의 세 대기 지점이 전부 이 함수를 부른다.
- **유휴 CPU 가드**: 정적 화면 20초, 워커 스레드 CPU 시간만 실측 —
  before 5.23% / after 5.93%(코어 1개 기준), 20초 폴 횟수 70 vs 69, 평균 sleep 250.0 ms로 동일.
  조용할 때는 예전 동작과 **같은 코드 경로**라 유휴 CPU는 오르지 않는다.
- **효과**(새 기준 4, 같은 PC·같은 시료): p50 **1.54→1.36초**, p95 **3.10→2.03초**,
  채팅 화면 p50 **2.94→1.77초**. 정확도 관련 값(기준 1·2·3·5)은 전부 그대로.
- **재발 방지**: 테스트 3건 — `test_poll_backs_off_to_idle_when_screen_is_quiet`(유휴 CPU 가드,
  `POLL_IDLE_S >= 0.25` 포함), `test_poll_speeds_up_right_after_a_change`,
  `test_worker_loop_has_no_hardcoded_poll_sleep`(루프 본문에 상수 sleep이 다시 박히는 것을 잡는다).

### PL-2. "최근에 변화가 있었나"를 **변화를 잡은 시각**으로 재면 긴 사이클에서 창이 도중에 만료된다
- **증상**: PL-1을 넣었는데 채팅 화면에서만 폴 대기가 252 ms 그대로였다(자막 화면은 62 ms로 줄었다).
- **원인**: 활동 창을 "화면 변화를 감지한 시각"부터 쟀다. 빽빽한 화면은 한 사이클이 1.9~2.2초라
  OCR·번역이 끝나기 **전에** 1.2초 창이 만료되고, 곧바로 다음 자막을 250 ms 늦게 잡았다.
  빠른 폴이 가장 필요한 화면에서만 정확히 안 듣는 구조였다.
- **수정**: 기준 시각을 **"직전 사이클이 끝난 순간"**으로 바꿨다(전체 사이클 종료 시 갱신 +
  '변화는 있는데 글자가 없다' 경로에서도 갱신). 채팅 poll 252→70 ms, 앞 사이클 대기 1106→305 ms.
- **교훈**: 시간 기반 히스테리시스의 기준점은 "일이 시작된 시각"이 아니라 **"일이 끝난 시각"**이다.

### BM-1. 채점판 기준 4가 사용자 기준과 다른 것을 재고 있었다 (통과인데 사용자는 답답)
- **증상**: 기준 4 "화면변화→자막 p50 <1.5s"가 통과(1.05s)인데 실사용은 2~3초.
- **원인**: 채점기가 잰 것은 **한 사이클 시간**(캡처+OCR+번역+게이트)이었다. 빠진 것이 둘:
  ① 폴 간격(최대 250~300 ms) ② 화면이 앞 사이클 처리 도중에 바뀌면 그것이 끝날 때까지의 대기.
  시료도 1120x780 이하 정지 이미지 1장이라, 사용자가 실제로 쓰는 **계속 바뀌는 글**이 없었다.
- **수정**: `bench/score.py`의 기준 4를 **실제 앱을 띄워** 재는 프로브로 교체했다(`--_latency_probe`,
  `_ui_probe`와 같은 별도 프로세스 패턴). 화면의 글자를 바꾼 시각 t0과 그 번역이
  오버레이 `paintEvent`에 그려진 시각 t1을 잰다. 시나리오 3종: **자막**(배경 고정, 한 줄 교체) /
  **채팅**(새 줄이 붙고 전체가 위로 밀림) / **정적 문서**(한 장을 한 번 띄움). 목표는 그대로
  p50 <1.5s · p95 <3s. 자막이 아예 안 뜬 건 `missed`로 세어 **통과를 막는다** —
  덜 띄워서 빨라지는 것은 개선이 아니기 때문이다.
- **검증 함정**: 오버레이는 `WDA_EXCLUDEFROMCAPTURE`라 스크린샷에 안 잡힌다 → `paintEvent` 훅이
  유일한 계측점이다. 그리고 **자극 화면의 배경 문장이 시료 문장과 겹치면 안 된다** —
  처음 하네스가 채팅 시드를 시료 문장 풀에서 뽑아 쓰는 바람에 "이미 떠 있던 자막"에 즉시
  매칭돼 지연 340 ms가 나왔다(실제로는 2.8초).

### QU-1. 한 사이클이 화면 변화 주기보다 길면 지연이 **누적**된다 (미해결 — 병목은 OCR)
- **증상**: 채팅 화면(2초마다 새 줄)에서 p50 2.7초인데 p95가 5~8.5초까지 튄다.
- **원인**: 큐잉이다. 사이클 주기(캡처+OCR+번역+폴)가 화면이 바뀌는 주기보다 길면 대기가 쌓인다.
  PL-1로 주기를 250 ms 줄여 2.15→1.85초가 되자 2초 주기를 따라잡아 앞 사이클 대기가
  1106→305 ms로 무너졌다(실측). **주기를 변화 주기 아래로 내리는 것이 유일한 해법**이다.
- **남은 병목**: 1920x1080 빽빽한 텍스트 화면의 OCR **1.4~1.5초**. 확인한 것:
  - 이미지 직렬화는 범인이 아니다 — PNG 인코딩 4~8 ms, 컨테이너를 BMP로 바꿔도 이득 없음(실측).
  - 확대 재OCR(OU-1/UB-1)은 낭비가 아니다 — 시도 **100회 중 99회 채택**(평균 conf 개선). 기각은 글자가 거의 없는 영상 프레임 1회뿐.
    끄면 빨라지지만 그건 정확도를 파는 것이라 하지 않는다.
  - 언어 재스캔(ZH-1/LC-1b)도 안 돌았다 — 사이클당 tesseract 패스는 1회(영상 화면 52/52).
  → 즉 OCR 시간은 **정직한 비용**이고, 더 줄이려면 인식 엔진 교체(B2)다.
- **덤으로 처음 측정한 것**: 5760x2160 투명 오버레이의 재그리기 비용은 `paintEvent` 본체
  **0.6~3.5 ms**(최초 1회만 폰트 로드로 ~230 ms). 지연에 유의미하지 않다 — 손대지 마라.


### PK-3. 동봉한 Tesseract의 OpenSSL이 파이썬 `_ssl`을 덮어써 **배포본에서 번역이 통째로 죽었다** ⭐
- **증상**: 2026-09-07 재빌드 직후 `dist/Cocktail/Cocktail.exe --self-test` 가
  `[SELFTEST] transformers import failed: ImportError: DLL load failed while importing _ssl:
  지정된 프로시저를 찾을 수 없습니다` 로 실패. 채점판 기준 6도 같이 실패했다.
  Tesseract·OCR·UI는 멀쩡해서 **번역만 안 되는** 형태라 눈으로는 놓치기 쉽다.
- **원인**: `build.spec` 은 Tesseract를 `datas` 로 `tesseract/` 폴더에만 넣는다(PK-1). 그런데
  PyInstaller가 `tesseract.exe` 의 의존성을 분석해 같은 DLL들을 **`_internal` 루트에도** BINARY로
  복사한다. `C:\tesseract\libcrypto-3-x64.dll`(5.3MB)이 conda 파이썬의
  `Library/bin/libcrypto-3-x64.dll`(7.4MB)과 이름이 같아 그 자리를 차지했고,
  `_ssl.pyd` 가 필요한 프로시저를 못 찾는다 → `huggingface_hub` → `transformers` import 실패.
- **확인 방법**(재빌드 없이 5초): `_internal/libcrypto-3-x64.dll` 을 파이썬 환경의 것으로
  덮어쓰고 self-test를 다시 돌리면 통과한다. 크기가 다르면 이 사고를 의심하라.
- **수정**: `build.spec` 에서 **루트로 흘러든 tesseract 사본만** 걷어내고(`tesseract/` 폴더 사본은
  그대로 — tesseract.exe 는 자기 폴더에서 DLL을 로드한다), 파이썬이 실제로 쓰는
  `libcrypto/libssl` 을 명시적으로 되돌린다.
- **교훈**: 외부 실행 파일을 동봉하면 그 의존 DLL이 **앱 루트의 이름 공간을 침범**한다.
  "폴더에만 넣었다"는 것은 PyInstaller에서 보장이 아니다.
  그리고 **빌드 성공은 동작이 아니다** — 재빌드 뒤 self-test를 반드시 실제로 돌린다
  (채점판 기준 6이 패키지 exe로 이걸 자동으로 한다).


### OU-2. 확대 재OCR 트리거가 **1차 오독에 오염**돼 있었다 — 확대가 가장 필요할 때만 꺼진다 ⭐
- **증상**: 같은 문장·같은 크기인데 **폰트만 바꾸면** CER이 0%↔100%로 갈렸다(실측 448건).
  Courier New 10px 100%(라인 0개) · Impact 11px 100% · Georgia/Calibri/Times New Roman 10px 10.67%
  · Calibri Light 10px 23.32%. 반면 Segoe UI/Arial/Verdana/Consolas/Tahoma는 0~0.8%.
  배경(어두움·저대비·무늬·반투명)은 **거의 무관**(0.4~3.2%p) — "배경 탓" 가설은 기각됐다.
- **원인**: `_tess_scan` 의 확대 조건이 `median_px < OCR_UPSCALE_TRIGGER_PX(18)` 하나뿐인데,
  그 `median_px` 를 **1차 패스가 읽은 word 높이**에서 가져온다. 1차가 무너지면 여러 글자 행을
  한 덩어리로 묶어 **10px 글자를 28px로 보고**한다 → 조건 불성립 → 확대 꺼짐. 자기충족적 실패다.
  게다가 조건 전체가 `if lines:` 안에 있어서 **1차가 0줄이면 확대를 아예 건너뛰었다** —
  0줄이야말로 확대가 필요한 상황이다.

  | 조합 | 1차 라인수 | 보고된 높이 | 확대 | 최종 CER |
  |---|---|---|---|---|
  | Segoe UI 10px | 6 | 7.0 | O | 0.00% |
  | Georgia 10px | 3 | **28.0** | **X** | 10.67% |
  | Calibri 10px | 3 | **28.0** | **X** | 10.67% |
  | Courier New 10px | **0** | — | **X** | 100.00% |
- **수정**: 판정을 `_wants_upscale(gray, lines)` 로 빼고, 높이 규칙이 "확대 불필요"라고 할 때만
  **이미지에서 직접 잰 글리프 높이**로 두 번째 의견을 묻는다(`cocktail_ocr.ink_glyph_px`,
  Otsu 잉크 마스크의 연결 성분 높이 중앙값). OCR 결과를 전혀 안 쓰므로 1차 오독에 안 흔들린다.
  실측 대응: 글꼴 10px→4~8 · 12px→6~9 · 14px→7 · 18px→9~10 · 24px→12~13 →
  임계 `OCR_UPSCALE_TRIGGER_INK_PX=11`. 잉크가 거의 없는 화면은 `None` 이라 **빈 화면에서는
  확대가 안 켜진다**(헛돈 방지). `if lines:` 가드도 없앴다 — 0줄이면 무조건 재시도한다.
  **결정론 유지**(RP-1): 두 신호 다 같은 이미지에서 항상 같은 값이 나온다(같은 시료 4회 동일).
- **비용**: 이미지 측정은 1920x1080에서 8~9 ms이고, **높이 규칙이 이미 확대를 켠 프레임에서는
  아예 안 부른다**(싼 조건인 배율 예산을 먼저 본다). 기준 4 시나리오 화면 실측 차이는
  자막 +5 ms · 채팅 -13 ms · 정적 0 ms · 빈 화면 +11 ms · 사진 +38 ms.
  채점판 기준 4 A/B(각 reps 20, 같은 PC): 구 규칙 p50 1.44~1.69s vs 신 규칙 1.39~1.43s — **차이 없음**.

### OU-3. 확대 보간 LANCZOS4의 **링잉**이 좁은 글꼴을 뭉갠다 — Impact 11px은 정확히 2.0배에서만 0줄
- **증상**: OU-2로 확대를 켰는데도 Impact 11px이 여전히 CER 100%(2배 확대 후에도 0줄).
  그런데 1.5배·2.5배·3.0배에서는 1.58%로 멀쩡했다. **배율의 문제가 아니었다.**
- **원인**: `cv2.INTER_LANCZOS4` 의 큰 음수 로브가 가는 획 둘레에 halo(링잉)를 만든다.
  해상도 한계에 있는 좁은 글꼴(Impact·Courier New)에서는 그 halo가 이웃 글자를 붙여
  Tesseract의 라인 분석이 통째로 실패한다. 특정 배율에서 위상이 맞아떨어질 때만 터지므로
  "배율을 올려 보자"로는 절대 안 잡힌다.
- **수정**: **1차 패스가 한 줄도 못 읽었을 때만**(`blind=True`) 링잉이 없는 `cv2.INTER_LINEAR`로
  확대한다. 1차가 뭐라도 읽었다면 글자가 이미 분해되는 해상도이고, 그때는 Lanczos의 선명함이
  실제로 이득이다(Calibri Light 10px 실화면 0.40% vs Cubic 8.30%).
  실측(bench `size_*`/`screen_*` 23종, 앱 전체 경로, 14px 이하 CER 최악):
  전부 LANCZOS4 **100.00%** · 전부 CUBIC 8.30% · 전부 LINEAR 5.53% · **1차 0줄일 때만 LINEAR 2.77%**.
  0줄 재시도 개별: Courier New 10px 4.35→0.00 · 같은 이탤릭 3.56→0.00 · Impact 11px 100→1.19.
- **교훈**: 확대 배율을 의심하기 전에 **보간 커널**을 의심하라. 그리고 "선명하게 만드는" 보간이
  항상 OCR에 좋은 것이 아니다 — Tesseract 5는 자체 적응형 이진화를 하므로, 해상도 한계에서는
  선명함보다 **가짜 획을 만들지 않는 것**이 중요하다.

### UA-2 / UA-3. 확대 채택 규칙이 **글자를 버린 패스**를 채택하고, **더 정확한 패스**를 거부했다
- **증상**: (UA-2) Impact 12px 실화면에서 확대 패스가 가운데 줄을 통째로 흘렸는데 채택되어
  CER 7.51%→**45.85%**. (UA-3) Calibri italic 12px은 확대 패스가 더 정확한데(11.46%→5.53%)
  평균 conf가 3.7점 낮아 거부돼 11.46%가 그대로 나갔다.
- **원인**: 채택 기준이 평균 conf 하나뿐이었다(UA-1). 평균 conf는 **줄을 버리면 오히려 오른다**
  (남은 줄만 깨끗하니까). 반대로 conf는 절대 눈금이 아니라 몇 점 차이는 노이즈다.
- **수정**: 두 축을 더했다.
  - `OCR_UPSCALE_MIN_COVERAGE=0.8` — 2차가 읽은 글자 수가 1차의 80% 이상일 때만 채택.
    실측 분포: 정상 조합의 글자수 비는 전부 1.00~2.07, 이 사고만 **0.56**. 겹치지 않는다.
  - `OCR_UPSCALE_CONF_MARGIN=4.0` — conf가 4점까지 낮아도 채택. UA-1의 근거였던
    "쓰레기 줄이 늘어난 확대"는 conf가 71.9→60.4로 **11.5점** 떨어지므로 여전히 거부된다.
- **교훈**: 한 패스를 고르는 판정에 **한 종류의 신호만** 쓰지 마라(ZH-2와 같은 교훈).
  "정확도의 대리 지표"는 방향이 반대인 실패 모드를 하나씩 갖고 있다.

### PM-2. 문단 병합이 **읽기 순서를 뒤섞어** 문장을 재배열했다 ⭐
- **증상**: 인식은 잘 됐는데 자막 문장이 뒤죽박죽. 실측 CER +30~37pt:
  Comic Sans 10px 무늬배경 8.70→**45.45** · 같은 폰트 반투명 10.67→**45.06** ·
  Georgia 10px 실화면 4.35→**33.99**.
- **원인**: `merge_paragraph_lines` 는 각 줄을 **열려 있는 모든 문단**과 대본다(PM-1: 옆 단이나
  아이콘 조각이 사이에 껴도 문단이 안 끊기게 하려는 의도). 그런데 같은 행이 좌/우로 쪼개진
  화면에서는 1·2·4번 줄이 한 문단이 되고 3번이 따로 남아, 출력이 **"1 2 4 3"** 순서가 된다.
- **수정**: 건너뛰려는 줄이 **합쳐질 문단의 가로 구간 안**에 있으면 그 병합을 포기한다.
  옆 단(column)·구석 아이콘은 그 구간 밖이라 PM-1의 원래 의도는 그대로 산다
  (2단 문서 병합 회귀 테스트 포함, 채점판 기준 3의 `doc_en_column` 0건 유지).
- **교훈**: "문단을 잇는" 규칙은 **순서를 보존한다는 계약**을 같이 가져야 한다.
  인식률이 100%여도 순서가 틀리면 사용자에게는 오역으로 보인다.

### DG-1c. 표시 게이트가 **원문이 깨졌는지를 못 봤다** — 정책 위반 ⭐
- **증상**: 원문이 그럴듯하게 깨지면 번역문도 그럴듯해서 게이트 4규칙을 전부 통과하고
  **틀린 자막이 그대로 뜬다**. 실측:
  ```
  Times New Roman 10px (CER 10.67%) -> shown
    "마라는 내가 거짓말을하고 묻지 않고 로 컵을 부어. Ousie는 샐리가 멈추고
     달콤한 램프가 젖은 길을 따라 하나씩 나옵니다."   (lifted->lied, finally->Sally, street lamps->sweet lamps)
  Calibri 10px (CER 10.67%, conf 52.0) -> shown   <- conf가 임계 60 미만인데도 통과
  ```
- **원인**: DG-1의 4규칙 중 셋(carryover / tgt-script / len-collapse)은 **번역문만** 본다.
  원문을 보는 유일한 규칙(tiny-line)은 `line_px < 12` **그리고** `conf < 60` 의 논리곱인데,
  그 `line_px` 가 **OU-2와 같은 뿌리로 28px까지 부풀어** 조건이 영원히 성립하지 않았다.
- **수정**: 규칙 1을 **conf 단독**으로 바꿨다(`DISPLAY_SRC_MIN_CONF=60.0`). `line_px` 는 판정에서
  뺐다 — 1차 OCR이 만든 값이라 1차가 무너지면 같이 오염된다(인자는 호출부 계약 유지용으로 남김).
  임계 근거(bench 시료 전체를 앱 경로로 태워 정답 유사도 0.80으로 정상/깨짐 분류):
  정상 줄 78개 min **64.0** · p05 86.1 · 중앙값 95.9 / 깨진 줄 43.7·52.0·58.4·62.6·63.9.
  **conf<60 → 정상 숨김 0/78(0.0%)**, conf<65 → 1/78(1.3%, 기준 2의 오탐 한도 1% 초과).
  그래서 60이 상한이다. 채점판 기준 2 오탐률은 **0/79 = 0.0%** 로 유지됐다.
- **한계(정직하게)**: Georgia/Times 급(conf 62~64)은 conf로는 못 막는다. 그건 **OU-2/OU-3가
  애초에 제대로 읽게 해서** 막는다(둘 다 지금 0.00%). "사전 밖 토큰 비율" 같은 원문 언어 신호도
  검토했지만 **넣지 않았다** — A~D 수정 후 시료 코퍼스에 "그럴듯하게 깨진 줄"이 남지 않아
  임계를 실측으로 정할 근거가 없고, 사전 없는 붙은단어 판정은 정상 문장을 숨길 위험(CO-2)이 크다.


### RP-2. OCR이 **완벽한** 문장에서 번역이 혼자 무너진다 — greedy 디코딩의 탐색 실패 ⭐
- **증상**: CER 0.00% 인 원문이
  ```
  "The kettle on the far shelf had been whistling for a while before anyone noticed it."
  -> "그 때, 그 때, 한 때의 그 시절, 그 시절, 그때의 그 시절을 떠올리는 사람이 있었다."
  ```
  로 나왔다. 표시 게이트 4규칙을 전부 통과한다 — 한글 비율 1.00, 길이비 0.60, 미번역 잔류 0.
  채점판 기준 2도 못 봤다(대리 지표 셋이 전부 이 형태를 놓친다). 시료 23장 중 **17장**에
  같은 문장이 들어 있어 사실상 채점판에서 가장 자주 뜨는 자막이었다.
- **원인 규명(후보를 하나씩 실측으로 배제)**:
  | 후보 | 판정 |
  |---|---|
  | SE-1 문장 분할이 문장을 놓친다 | **기각**. 원문 3문장 → 조각 3개, 전부 번역됨. 문단을 통째로 넣으면 오히려 **정상 번역**이 나온다 |
  | PM-1 병합으로 입력이 길어져 앞 문장을 흘린다 | **기각**. 문장 누락 0/189 · 0/1940. 종결부호 뒤 공백이 없어 분할이 실패한 입력(`...dawn.Nobody...`)에서도 누락 0 |
  | `no_repeat_ngram_size=4` 가 부족하다/해롭다 | **둘 다 아니다**. 0으로 두면 `"그 때," ×16` 무한 루프가 되고(더 나쁘다), 3·2로 낮추면 반복만 줄고 뜻은 여전히 틀린다. 유지가 맞다 |
  | **greedy(`num_beams=1`)** | **원인 확정**. 같은 문장·같은 모델을 `num_beams=4` 로 뽑으면 `"이윽고 찻잔은 누군가가 눈치 채기 전에 잠시 동안 휘파람을 불고 있었다."` 로 정상. 원문에서 **한 글자만 달라도**(`for a while`→`for 2 while`, OCR이 실제로 그렇게 읽은 시료) 붕괴가 사라진다 = 입력 특성이 아니라 탐색 실패다 |
- **빈도(영어 1,940문장 · en→ko)**: 초과 반복률 ≥0.10 이 **8건 = 0.41%**.
  OCR 오독을 주입한 600문장에서는 **0/600** — 붕괴는 "쓰레기 입력" 현상이 **아니라**
  깨끗한 입력에서 나는 탐색 실패다(게이트의 전제와 반대라 게이트로는 못 잡는다).
- **왜 전면 `num_beams=4` 가 안 되나**: 배치 12문장 p50 **311ms → 553ms(+78%)**.
  기준 4는 p50 1.46s / 한도 1.5s 라 마진이 40~140ms 뿐이다. 채택 불가.
- **수정**: `_retry_collapsed()` — 붕괴 징후가 있는 **조각만** `num_beams=4` 로 다시 뽑고,
  **더 나아졌을 때만** 바꿔치기한다(재시도가 더 반복적이면 원본 유지). 상한
  `RETRY_MAX_PIECES=8`(한 프레임 최악 지연 방어), 실패하면 조용히 원본 유지(TE-1 정신).
- **판정값**: `repetition_excess()` = 글자만 남긴 **문자 3-gram** 중복률의 *원문 대비 초과분*.
  - **단어** n-gram이 아니라 **문자** n-gram인 이유: 붕괴는 조사·어미를 바꿔 가며 반복해서
    (`"그 때, 그 때, 한 때의 그 시절, 그 시절"`) 단어 단위로는 중복이 안 잡힌다.
  - **원문 반복률을 빼는** 이유: 원문이 이미 반복이면(`----------`, `:py:data:` 나열,
    `"a factor of 1.0 ... a factor of 2.0 ..."`) 정상 번역도 반복이라 절대값으로는 못 가른다.
    빼기 전에는 상위 10건 중 8건이 이런 **정상** 줄이었다.
  - 임계 0.10 근거: 정상 p50 −0.027 / p90 0.000 / **p99 +0.064** vs 붕괴 +0.10~+0.33.
  - 짧은 문자열 하한 5글자: 7이면 짧은 라벨을 통째로 면제해
    `"Council meeting minutes" → "회의록 회의록"`(초과 +0.250)을 못 본다.
    5로 내려도 2,129건에서 새 거짓양성 **0건**(p99 +0.062 불변).
- **효과(같은 코퍼스 재측정)**: 붕괴 8건 중 **6건이 정상 번역으로 교체**, 악화 **0건**.
  초과 반복률 max +0.333 → **+0.100**. 채점판 시료의 그 문장도 전 시료에서 정상 번역이 됐다.
- **속도 비용**: 재시도 1회 = generate 1회 = **약 190ms**(빔 수·`max_new_tokens` 를 줄여도
  안 줄어든다 — 비용은 빔이 아니라 "호출이 한 번 더 도는 것"이다). 발생률이 낮아 코퍼스
  전체 시간은 48.5s → 48.0s(변화 없음).
  그러나 채점판 기준 4의 **채팅 시나리오는 QU-1(큐잉 누적) 구간**이라 한 번의 190ms가
  이후 표본 전부에 남는다: 채팅 p95 2.48s → **2.9s대**(3회 실측 2.97/2.90/2.72, 한도 3.0).
  **통과는 하지만 마진이 얇아졌다.** 근본 해결은 B2(OCR 시간을 줄여 큐잉을 없애는 것)다.

### CO-3. 미번역 잔류 규칙이 **채팅 줄의 92%를 숨기고 있었다** ⭐
- **증상**: `"user1: did anyone look at the log file"` → 정상 번역이 나오는데도 게이트가
  `carryover(user)` 로 숨긴다. 채팅 12줄 중 **11줄**이 이렇게 사라졌다.
  사용자가 지목한 두 주 용도 중 하나(채팅)가 통째로 막힌 것이라 CO-2와 같은 종류의 사고다.
- **원인**: `carryover_words()` 가 `[A-Za-z]+` 로 자른다. `user1` 은 식별자인데 `user` 로
  잘려 "번역 안 된 영어 단어"로 세어졌다. `mp3`, `PyQt5`, `3D` 도 같다.
  채점판이 못 잡은 이유: 시료에 이런 줄이 하나도 없었다(오탐 분모 79줄 전부 문서·UI 문장).
- **수정**: 영숫자 토큰을 먼저 자른 뒤, **숫자가 끝에 붙은 것은 식별자로 보고 세지 않는다.**
  반대로 **글자 사이에** 숫자가 낀 것(`cre2ted`, `de5cription`, `applicati0n`)은 OCR 오독이
  만든 깨진 단어이므로 지금까지처럼 센다. 이 구분이 없으면 둘 중 하나를 반드시 잃는다
  (실측: OCR 오독을 주입한 코퍼스의 잔류 사례 절반이 숫자 치환형이었다).
- **함정**: `(?![0-9])` 만 붙인 부정 전방탐색은 **소용없다** — 정규식이 되짚어 `user1` 에서
  `use` 를 뽑는다. 실제로 처음 그렇게 짜서 아무것도 안 고쳐졌다.
- **한계**: `utf8mb4` 처럼 숫자가 **가운데** 든 진짜 식별자는 여전히 잔류로 오인된다.
  그런 화면이 잦으면 예외 목록이 아니라 `DISPLAY_MAX_CARRYOVER_WORDS` 를 올려라.

### BM-2. 채점판 기준 2가 **"오번역을 하는가"를 재지 않고 있었다**
- **증상**: 기준 2가 통과인데 실제로는 위 RP-2 붕괴가 시료 17장에서 매 실행 표시되고 있었다.
- **원인**: "치명적 오역"을 대리 지표 셋(미번역 잔류 / 목표언어 글자 비율 / 길이 붕괴)으로만
  봤다. 셋 다 **번역문의 형태**만 본다. 붕괴·반복·문장 누락은 형태가 멀쩡해서 전부 통과한다.
- **수정**: `bench/score.py` 기준 2에 두 가지를 **정확한 측정**으로 추가했다(대리 지표 아님).
  - **반복 붕괴**: 표시된 영어 줄의 `repetition_excess` ≥ `COLLAPSE_EXCESS(0.12)`.
    임계는 RP-2 재시도 이후 분포에서 잡았다(정상 p99 +0.055 / max +0.100 → 오검출 0/1940).
  - **문장 누락**: 원문이 2문장 이상인 줄을 **문장 단위로 실제로 다시 번역**해
    빈 조각·극단 축약 조각(길이비 <0.15)을 센다. 추측이 아니라 실행이다.
  둘 다 `exact_critical` 에 들어가므로 발생하면 기준 2가 **실패**한다.
- **결과**: 오탐률은 0/79 = 0.0% 로 **그대로**, 확정 오역은 0건 → **1건**이 됐다.
  `"Council meeting minutes" → "회의록 회의록"` — `num_beams=4` 로도 같은 출력이라
  탐색 실패가 아니라 **모델 한계**다(§ 아래 "남는 것"). 회귀가 아니라 **원래 있던 결함이
  이제 보이는 것**이다. 점수를 좋게 만들려고 임계를 되돌리지 않았다.
- **검토했지만 안 넣은 것 — 출력 중복 제거 post-processing**: 인접 중복 어절 제거를
  2,129건에 적용해 보니 7줄이 바뀌는데 **명백한 개선은 1줄**이고 나머지는 원문 자신이
  중복인 경우(`size size / 2`, `" "`)라 오히려 문장을 훼손했다. 한 사례를 위한 과적합이라 기각.
- **게이트에는 붕괴 규칙을 넣지 않았다 — 근거**: RP-2 재시도 이후 붕괴가 남는 사례가
  깨끗한 입력 1,940건 + OCR 오독 600건 = **2,540건에서 0건**이었다. 한 번도 안 걸릴 규칙을
  게이트(= 정상 문장을 숨길 위험이 있는 곳)에 넣지 않는다. 채점판이 세고 있으므로
  다시 나타나면 그때 근거를 갖고 넣는다.


### PV-1. 영구 번역 캐시 = **화면에서 본 모든 문장의 영구 기록**이었다 ⭐
- **증상**: `PersistentCache`가 번역한 모든 문장을 `~/.cocktail/translation_cache.bin`에
  계속 쌓았다. **지우는 UI도, 만료도, 끄는 옵션도 없었다.** 실측(2026-09-07, 개발 PC):
  **2,574항목 / 265KB**, 값(번역문)은 블롭 안에서 평문. 이 앱은 "전부 로컬"을 강점으로
  내세우는데 사용자는 이 기록의 존재도, 지우는 법도 알 수 없었다.
- **원인 셋**:
  ① **끌 수 없음** — `_enabled`는 `sys.platform == "win32"` 하나로 정해졌다(윈도우면 무조건 켬).
  ② **DPAPI를 안전의 근거로 착각** — `CryptProtectData`를 플래그 없이 부르면 `CurrentUser`
     범위다. **같은 윈도우 계정의 아무 프로세스나** 그대로 복호화한다. "암호화됨"은
     "다른 사용자·다른 PC로 새지 않음"일 뿐 "이 PC의 다른 프로그램이 못 읽음"이 아니다.
  ③ **축출이 LRU가 아님** — `put()`이 상한(4096)에 닿으면 `keys[:len//2]`, 즉 **dict 앞쪽
     절반을 무작정** 버렸다. 자주 쓰는 항목이 임의로 날아가고, 상한은 **개수만** 묶어
     기록의 **나이**는 아무도 안 묶었다(가끔 쓰는 사용자면 가장 오래된 기록이 반 년을 산다).
- **수정** (`cocktail_translate.py` / `cocktail_engine.py` / `cocktail_ui.py`):
  - **기본값 = 끔** (`PERSIST_CACHE_DEFAULT_ON = False`). 근거는 아래 "기본값 근거".
  - **트레이 "번역 기록 삭제"** — 메모리 + 디스크를 즉시 지우고 **삭제 후 다시 확인한**
    결과를 숫자로 보여준다(`메모리 N건 · 디스크 M건 · 파일 삭제 확인`) + 트레이 풍선.
  - **트레이 "번역 기록을 디스크에 저장"** 체크 항목. QSettings(`ui/persist_cache`, PS-1)로 유지.
    **끄면 디스크 기록도 같이 지운다** — 끈 상태로 예전 기록이 남으면 그건 "끔"이 아니다.
    앱 시작에서도 같은 경로를 타므로 이번 업데이트 후 첫 실행에서 옛 기록이 사라진다.
  - **만료 `PERSIST_CACHE_TTL_DAYS = 7`** — 시각은 **마지막으로 쓴 때**. 로드 시와 저장 시
    (30초 주기)에 쓸어 내고, 개별 항목은 `get`이 거른다.
  - **LRU 축출** — `OrderedDict` + `move_to_end`, 넘치면 `popitem(last=False)`.
  - 파일은 **삭제**한다(빈 파일로 덮지 않는다). `.tmp`도 같이 지운다.
  - 옛 형식(`{key: 번역문}`)은 **나이를 모르므로 로드에서 버린다.** 한 번 느려지는 대신
    만료를 걸 수 없는 기록을 남기지 않는 쪽을 택했다.
- **기본값 근거 (끔으로 정한 이유, 실측)**:
  ① 같은 세션의 반복은 메모리 LRU(`TRANS_CACHE`, 1024)가 이미 전부 잡는다. 영구 캐시가
     **추가로** 버는 것은 "지난 실행에서 본 문장을 다시 볼 때"뿐이다.
  ② 그 이득 실측 = 22줄 화면 하나에 **162 ms**(줄당 7.4 ms). 재방문 화면당 한 번뿐이다.
  ③ 기준 4는 애초에 영구 캐시를 **끈 상태로** 잰다(`bench/score.py`가 명시적으로 끈다).
     즉 채점되는 속도에 영향이 **0**이다. 실측: 켬/끔 어느 쪽이든 p50 1.42s / p95 2.82s.
  ④ 은행·로그인 화면이 지나갈 수 있는 앱에서 영구 기록은 opt-in 이어야 한다.
     민감 창 가드(SB-1)는 창 제목 기반이라 78%를 놓친다(SB-2).
- **TTL 7일 근거**: 상한(4096)은 개수만 묶는다 — 실화면 한 장이 12~22줄(bench 시료 실측)이라
  자막을 한 시간 보면 ~600항목이 쌓여 **많이 쓰는 사용자는 상한이 며칠 안에 도는데**,
  가끔 쓰는 사용자는 하루 수십 항목이라 상한이 **영원히 안 걸린다**. 상한이 지켜 주지
  못하는 쪽이 오히려 위험하므로 나이로 한 번 더 자른다. 7일 = 영구 캐시가 실제로 버는
  재사용 창(어제 보던 문서·게임을 오늘 다시 연다)이고, 그 너머는 다시 번역해도 줄당 7.4 ms다.
- **회귀 가드**(test_cocktail.py): 꺼진 상태에서 디스크에 안 쓴다 / 삭제·끄기가 파일을
  실제로 지운다 / 만료가 동작한다 / 축출이 LRU다 / 기본값이 꺼짐이다.
- **벤치 보호**: 앱이 "꺼짐 = 파일 삭제"가 됐으므로 `bench/score.py`의 세 진입점
  (`make_ctrl`, `_latency_probe`, `_ui_probe`)이 캐시 **경로**를 임시 파일로 돌린다.
  `_enabled = False` 만으로는 사용자 파일이 지워지는 것을 못 막는다.

### PV-2. 진단 로그가 화면 원문을 그대로 찍고 있었다
- **증상**: `[INFO] DG-1 숨김: low-src-conf(41) "Transfer 4,200,000 KRW to account 110-..."`.
  RP-2 재번역 로그도 같은 방식으로 원문 60자를 찍었다. 은행 화면의 글자가 콘솔과
  리다이렉트된 로그 파일(`debug_log.txt`, `debug_run.bat`)에 남는다.
- **원인**: CO-2("숨김은 무음 실패다")를 고치면서 **사유와 원문을 같이** 찍었다. 사유는
  필요하지만 원문은 필요 없었다 — 필요한 건 "**어떤 줄인가**"를 구분할 식별자뿐이다.
- **수정**: `log_text()` 한 곳으로 모았다. 기본은 `길이자 #sha1앞8자리`,
  `COCKTAIL_LOG_TEXT=1` 일 때만 원문. 게이트 반복 억제 키도 이 식별자를 쓰므로
  "같은 줄은 한 번만 찍는다"는 성질이 그대로 산다. **사유는 그대로 남긴다** —
  무음 실패로 되돌아가지 않는 것이 이 수정의 제약이었다(CO-2).
- **전수 확인**: 7개 모듈의 `print` 를 전부 훑었다. 사용자 텍스트를 찍던 곳은 이 둘뿐이고
  나머지는 경로·언어 목록·예외 타입·해상도·소요 시간이다. `LAST_GROUP_ERRORS`도
  예외 메시지만 담는다.

### PV-3. 전역 키보드 후킹이 기본 켜짐이었다 (키로거와 같은 API)
- **증상**: `keyboard` 패키지를 모듈 최상단에서 import 하고 시작할 때 무조건
  `add_hotkey` 를 걸었다. 이 라이브러리는 저수준 후킹으로 **모든 키 입력을 가로챈다** —
  키로거와 같은 API다. 프라이버시를 내세우는 앱의 기본값으로는 모순이고,
  백신 오탐의 단골이라 합격기준 6("백신 경고 없이 실행되는가")과도 직결된다.
- **수정**: `GLOBAL_HOTKEYS_DEFAULT = False`. 저장된 설정(`ui/global_hotkeys`, PS-1)으로 유지하고
  트레이 "전역 단축키 (Ctrl+Shift+Q/A)" 체크 항목으로 켠다. **라이브러리 import 자체도
  켤 때로 미뤘다**(`_load_keyboard()`) — 안 쓰는 후킹 라이브러리를 프로세스에 올릴 이유가 없다.
  등록/해제 코드는 지우지 않았다. 껐다 켰다 하는 것이다(`_register_/_unregister_global_hotkeys`).
- **끈 상태에서 잃는 것 / 잃지 않는 것**: 잃는 것은 "창이 안 보일 때 **키만으로** 토글"
  하나뿐이다. 트레이 메뉴 13항목(열기 / 번역 시작·정지 / 캡처 모드 / 영역 편집 / 마우스 통과 /
  자동 실행 / 전역 단축키 / 기록 저장 / 기록 삭제 / 라이선스 / 종료)과 창 단축키
  (Ctrl+Shift+H, Ctrl+Shift+P)로 모든 기능에 닿는다. 트레이 아이콘 더블클릭으로 창이 열린다.
  첫 실행 안내 한 줄 + 툴팁으로 "기본 꺼짐, 트레이에서 켜기"를 알린다.
- **채점 영향 없음(확인함)**: 기준 5가 세는 "단축키 2개"는 `win.findChildren(QShortcut)` —
  **창 안** 단축키(Ctrl+Shift+H/P)이지 전역 후킹이 아니다. 채점판을 고칠 필요가 없었다.
  실측 후 그대로 통과(트레이 10→13개, 단축키 2개). 트레이 툴팁의 단축키 안내는
  실제 등록 상태를 따라가게 했다 — **안 되는 키를 광고하지 않는다.**

### SB-2(기록만). 민감 창 가드(SB-1)가 현실의 78%를 놓친다
- **측정**(스크래치패드, 32개 실사용 시료): `is_sensitive_window`는 창 제목/클래스에
  `SENSITIVE_KEYWORDS` 가 들어 있는지만 본다. **잡음 7 / 놓침 25 = 78% 미검출.**
- **놓치는 부류**: 영문 뱅킹(`Internet Banking`, `Chase`, `Wells Fargo`) · 결제(`PayPal`,
  `Stripe`, `네이버페이`) · 송금(`Wise`, `송금하기`) · 증권(`삼성증권`, `키움증권 영웅문`) ·
  가상자산(`업비트`) · 행정/세무(`홈택스`, `정부24`) · **비밀번호 관리자**(`Bitwarden`, `KeePass`) ·
  **OTP**(`Google Authenticator`, `OTP 인증`) · **윈도우 자격증명 창**(`Windows 보안`,
  `사용자 계정 컨트롤` — 클래스 `Credential Dialog Xaml Host`) · 공동인증서(`인증서 선택`) ·
  의료/메일/메신저. 그리고 **제목이 빈 창은 무조건 통과**한다(`if not haystack.strip(): return False`)
  — 제목 없는 전체화면 앱이 여기 걸린다.
- **왜 구조적인가**: 브라우저 창 제목 = **페이지 제목**이다. 은행 사이트가 자기 이름을
  한글로 안 쓰면 어떤 키워드 목록도 못 잡는다. 키워드를 늘리는 것은 이 사슬을 못 끊는다.
- **고치지 않았다(별건)**: 방향만 적어 둔다 — ① `Credential Dialog Xaml Host` 같은
  **클래스 화이트리스트**(제목과 무관하게 확실) ② 브라우저는 UIA로 주소창 URL을 읽어
  도메인으로 판정 ③ 화면에 카드번호/계좌번호 **패턴**이 보이면 그 프레임을 버리는
  출력측 가드. ①이 가장 싸고 확실하다. 지금은 PV-1(기본 끔)이 "그래도 기록은 안 남는다"로
  피해를 줄여 준다.


## 5. 변경 이력

- 2026-09-07: PV-1(영구 캐시 = 영구 기록 — 기본 끔·삭제 UI·7일 만료·LRU 축출), PV-2(로그가 원문을 찍음), PV-3(전역 키 후킹 기본 켜짐), SB-2(민감 창 가드 78% 미검출, 기록만).
- 2026-09-07: RP-2(greedy 붕괴 → 붕괴 조각만 빔 서치 재시도), CO-3(식별자를 미번역 잔류로 오인해 채팅 11/12줄 숨김), BM-2(기준 2가 반복 붕괴·문장 누락을 정확히 재게 함).
- 2026-09-07: OU-2/OU-3(폰트별 인식 붕괴 — 확대 트리거가 1차 오독에 오염 + 확대 보간 링잉), UA-2/UA-3(확대 채택 규칙), PM-2(문단 병합 읽기 순서), DG-1c(게이트가 원문을 봄).
- 2026-09-07: PK-3(동봉 tesseract OpenSSL이 파이썬 _ssl을 덮음 — 배포본 번역 사망). PL-1/PL-2(적응형 폴 간격), BM-1(기준 4를 '화면 변화→자막'으로 재정의), QU-1(큐잉·OCR 병목 기록).
- 2026-04-30: 초기 작성 (Cocktail완성본.py 1차 리뷰). P-1 ~ P-8, B-1 ~ B-6, E-1 ~ E-5 식별.
- 2026-04-30: 모델 교체 — m2m100_418M → NLLB-200-distilled-600M.
  CTranslate2 자동 감지 백엔드 추가 (HF / CT2 둘 다 지원). M-1 ~ M-4 추가.
- 2026-04-30: E-6 추가 — Tesseract 한국어 traineddata 누락 확인 (`--list-langs`에 kor 없음).
- 2026-04-30: B-7 추가 + 수정 — 타이핑 효과 제거. 워커 0.25초 주기와 충돌해 줄들이 깜빡이던 문제 해소.
  죽은 import (QTimer/QMetaObject/pyqtSlot) 동시 정리.
- 2026-04-30: 박사 2차 리뷰 — B-8(채널 순서 RGB/BGR), B-9(QRect 클립 + word wrap) 식별 및 수정.
  E-7(DPI 이중 적용) 잠재 결함으로 기록만. 5초 체크리스트에 3개 항목 추가.
- 2026-04-30: 박사 3차 리뷰 — "1초 후 깨지는" 증상 진단.
  B-10(자기 캡처 피드백 루프, SetWindowDisplayAffinity로 차단), B-11(워커-paint race, 시그널/슬롯으로 해결),
  E-7 정책 확정(_scale=1.0), E-8(silent 종료 → backoff continue) 수정. 5초 체크리스트에 3개 항목 추가.
- 2026-04-30: B-12 추가 + 수정 — 한국어 박스 height 동적 확장(`QFontMetrics.boundingRect`),
  alpha 200→235 상향, 클립 영역을 red_rect→위젯 전체로 변경. 5초 체크리스트에 2개 항목 추가.
- 2026-04-30: B-13 추가 + 수정 — 폰트 자동 축소(`_fit_font`) 결합. height 폭증 시 폰트를
  initial→9pt까지 단계적으로 줄여 인접 줄 겹침 완화.
- 2026-04-30: E-9 추가 + 수정 — 캡처 차단 토글 UI 도입. 헬퍼를 `set_window_capture_affinity(exclude=)`로
  일반화. "스샷차단/스샷OK" 버튼 + Ctrl+Shift+H 단축키. 사용자가 스크린샷 찍을 수 있게 됨.
  5초 체크리스트에 2개 항목 추가.
- 2026-04-30: 상용화 1단계 — M-5(NLLB→opus-mt+m2m100 하이브리드, 라이선스 정리),
  E-10(PyQt5→PySide6 마이그레이션, GPL→LGPL), P-9(모델 lazy load + UX 진행 표시) 동시 적용.
  `HybridTranslator` 도입. 5초 체크리스트에 5개 항목 추가. 메모리에 `project_commercial_decisions.md` 기록.
- 2026-04-30: 상용화 3+5+6단계 — B-14(OCR 언어 동적 + traineddata 진단) 추가/수정.
  모델 로드 진행 표시(progress_msg 시그널) 추가. 빌드 자산 신규: `requirements-dev.txt`,
  `build.spec`(PyInstaller), `download_tessdata.py`, `installer.iss`(Inno Setup, traineddata 컴포넌트), `BUILD.md`.
  배포 시 `./tessdata/` → `TESSDATA_PREFIX` 자동 설정 코드 추가. 5초 체크리스트에 4개 항목 추가.
- 2026-04-30: OCR 상향 옵션 — O-1(PaddleOCR v3 / PP-OCRv5 선택 백엔드) 추가/수정.
  UI OCR 콤보 추가, `PaddleOcrManager` lazy import/cache, 실패 시 Tesseract 폴백, optional dependency/build 문서화.
- 2026-04-30: 일반 사용자 UX 안정화 — U-1 추가/수정.
  Tesseract 미설치 시 시작 예외 제거, 환경 요약/상태 라벨 추가, OCR 없음/폴백/번역 완료 상태 표시,
  `run_cocktail.bat`, `diagnose_environment.py`, `TECH_STACK.md`, `UPDATE_LOG.md`, `OCR_TEST_PLAN.md` 추가.
- 2026-04-30: 다국어 auto 안정화 — M-7 추가/수정.
  OCR bbox를 좌우 column/세로 island region으로 묶고, region 내 강한 언어 신호로 짧은 라인의 감지를 안정화.
  `batch_translate()` 라인별 source list 지원, 상태 라벨 언어 요약 표시.
- 2026-04-30: Phase 1A 시작 — V-1 비전 기반.
  T-1(시스템 트레이 + `keyboard` 글로벌 단축키 Ctrl+Shift+T/A) +
  T-2(캡처 모드 콤보: 영역 박스/활성 창/마우스 hover, `GetForegroundWindow` + `QCursor.pos()`) 추가/수정.
  closeEvent를 hide(트레이) vs `_is_quitting` 분기.
  Phase 2 PoC: `uia_explore.py` 신규 (uiautomation, Apache 2.0, 본 앱 의존성 X).
  새 의존성 라이선스 검증: keyboard MIT / uiautomation Apache 2.0.
  5초 체크리스트에 5개 항목 추가.
- 2026-04-30: Phase 1B + Phase 5 시작 — V-1 비전의 매끄러운 작동 + 보안 원칙 #3.
  W-1(마우스 통과 토글, `WS_EX_TRANSPARENT`/`WS_EX_LAYERED`, Ctrl+Shift+P + 트레이 checkable) 추가/수정.
  S-1(민감 영역 자동 OFF, `SENSITIVE_KEYWORDS` 사전, `GetWindowText`+`GetClassName` 검사,
  박스 모드 외에서만 적용) 추가/수정.
  5초 체크리스트에 5개 항목 추가.
- 2026-04-30: Phase 1C — 컨트롤 ↔ 오버레이 윈도우 분리.
  W-2(`OverlayWindow` 신규 — 전체 화면 + 투명 + 항상 위 + `Qt.Tool` + 마우스 통과 영구 ON + B-10 캡처 차단,
  paintEvent로 빨간 박스/번역 그림, 좌표는 화면 절대) 추가/수정.
  워커 adjusted_bbox를 위젯 좌표 → 화면 절대 좌표로 변경. TransparentWindow.paintEvent 위임.
  사용자 환경 검증 + ENVIRONMENT.md 신규 (Python 3.10.18 / torch 2.7.1+cu118 / 누락 패키지 명시).
  5초 체크리스트에 4개 항목 추가.
- 2026-04-30: Phase 1D + 2/3 + 4 동시 진행 — 박사 자체 비판 리뷰 거친 후.
  W-2 자체 리뷰 보강: W-1 토글이 OverlayWindow에 적용되도록 수정, moveEvent/resizeEvent/mouseMoveEvent에서 `overlay.update()`.
  W-3(멀티 모니터: 가상 데스크톱 전체로 OverlayWindow 확장, paintEvent 좌표 변환).
  UIA-1(uiautomation lazy import + 어댑터 + ocr_combo 옵션 + `_uia_collect_and_translate` 워커 통합 + 민감 창 검사 + 결과 100개 상한).
  D-1(DPAPI 암호화 영구 캐시, sha256 hash key, lazy load, batch_translate 통합, closeEvent에서 save).
  5초 체크리스트에 7개 항목 추가.
- 2026-04-30: Phase 6 마무리 + 보강 (사이클 옵션 C — 박사 자체 비판 리뷰 강화).
  A-1: 주기적 PERSIST_CACHE.save() (QTimer 30초) + DPAPI argtypes 명시 + Windows 자동 실행 옵션(`winreg`) +
  트레이 메뉴 "Windows 시작 시 자동 실행" checkable.
  마켓 진입 자산 신규: `LICENSE`(MIT), `LICENSE-3RDPARTY.md`, `CODE_SIGNING.md`, `ICONS.md`,
  `.github/workflows/release.yml`(태그 push → PyInstaller 빌드 → GitHub Release), `README.en.md`.
  박사 자체 리뷰 잠재 결함 4건 기록(콘솔 창 / 메인 스레드 save / Inno Setup 자동화 / LICENSE 결정).
  5초 체크리스트에 4개 항목 추가.
- 2026-05-04: CR-1 크리티컬 리뷰 수정.
  워커 스레드 시작 코드가 QTimer slot 내부로 들어간 치명적 들여쓰기 결함 수정.
  캡처 affinity 토글을 ControlWindow+OverlayWindow 양쪽에 적용.
  활성 창/민감 창 자기 hwnd 가드에 OverlayWindow 포함.
  UIA 번역 경로에도 M-7 region smoothing 적용.
- 2026-05-04: ST-1 + A-2 보강.
  `smoke_tests.py` 추가로 다음 step 전 자동 회귀 검사 체계 확립.
  자동 실행 명령은 `.py` 실행 시 `pythonw.exe` 우선으로 콘솔 창 회피.
  주기적 PersistentCache save를 백그라운드 스레드로 이동하고 `_mem_lock`으로 저장/조회/갱신 경쟁 조건 완화.
- 2026-05-04: R-1 릴리즈 워크플로우 보강.
  GitHub Actions에서 Tesseract 설치 후 진단 실행, 서명 후 zip 생성, Inno Setup 인스톨러 자동 빌드.
  build.spec에 사용자/라이선스 문서 포함. `pre_release_check.py` 추가.
- 2026-05-04: 사용자 GUI 검증 피드백 — UX-1 (기본 모드 활성 창, OCR 합집합 → eng+kor, 빨간 박스 박스 모드만) +
  B-15 (자기 캡처 자동 가드 — 운용 규칙 → 코드 자동화). 박사 자체 리뷰 보강 3건.
- 2026-05-04: 사용자 통찰 "AI 시대인데 주먹구구는 구식" — UX-2 SmartOCR (M³ Multiplex 영감, 5-Layer).
  Layer 1 UIA 자동 라우팅 / Layer 2 화면 클립 / Layer 3 OSD script 분류 / Layer 4 단일언어 OCR / Layer 5 hwnd 캐시.
  예측 OCR 27초 → ~2초. Tesseract OSD 활용 (추가 의존성 0).
  박사 자체 리뷰 결함 2건 즉시 수정 (mutable 클래스 변수, UIA 자동 라우팅이 명시 선택 무시).
  settings.local.json 권한 강화로 박사 자율 진행 가능.
- 2026-05-04: BG-2 앱 생명주기 컨트롤러 추가.
  `__main__`에서 QApplication/설정창/오버레이를 직접 만들던 배포 구조 결함을 줄이고,
  `CocktailAppController`가 생성/연결/실행을 관리하도록 변경.
  `COMMERCIAL_SOTA_REVIEW.md` 신규 작성으로 SOTA/상용 라이선스 기준선을 문서화.
  smoke/pre-release gate에 BG-2 구조 검사를 추가.
- 2026-05-04: BG-3 CaptureRegionController 분리.
  캡처 모드/hover 크기/red_rect와 활성 창/hover 영역 계산을 `TransparentWindow`에서 분리.
  기존 호출부는 proxy property로 유지해 회귀 위험을 줄이고, 추후 `RegionEditor`가 같은 상태를 조작할 수 있게 준비.
  smoke/pre-release gate에 BG-3 구조 검사를 추가.
- 2026-05-04: ARCHITECTURE.md 신규 — As-Is/To-Be 클래스 다이어그램, 시그널/데이터 흐름,
  BG-4~BG-6 마이그레이션 계획, BG-5 좌표계 결정(red_rect 절대 좌표화)을 한 곳에 모음.
  목적: BG-N 사이클 직전에 책임 경계와 좌표계를 미리 결정해 즉흥 결정 비용 회피.
- 2026-05-04: BG-4 BackgroundController 분리.
  워커 thread, 엔진 시그널 5개(state_ready/failure_alert/model_ready/progress_msg/status_msg),
  worker loop, OCR/SmartOCR/UIA 라우팅, 모델 lifecycle, 자동 일시정지 플래그,
  PersistentCache 타이머를 `TransparentWindow`에서 분리.
  의존 주입: `region`(CaptureRegionController), `options_provider`(callable→`RuntimeOptions`),
  `is_self_hwnd`(callable→bool). widget 직접 참조 금지.
  `RuntimeOptions(@dataclass(frozen=True))`로 콤보 스냅샷을 immutable로 워커에 전달.
  TransparentWindow는 proxy property(translated_text/model_loaded/capturable)와
  UI slot(_on_state_ready/_on_progress/_on_failure/_on_model_ready/_set_status)만 보유.
  박사 자체 리뷰 결함 1건 즉시 수정: forward-ref 어노테이션
  (`-> RuntimeOptions`이 클래스 정의 시점에 평가되어 NameError) → 문자열 어노테이션으로 변경.
  smoke/pre-release gate에 BG-4 구조 검사 추가, 기존 5개 테스트 BG-4 분리에 맞춰 갱신.
- 2026-05-04: BG-5 red_rect 좌표계 통일.
  `red_rect`를 settings window 위젯 좌표 → **화면 절대 좌표**로 전환.
  `CaptureRegionController.__init__` 기본값이 `QRect(240, 215, 600, 375)`(절대).
  `update_for_mode`가 활성 창/hover 절대 좌표를 그대로 저장 (owner.geometry() 변환 제거).
  `BackgroundController.capture_and_translate_async`에서 `region.owner.geometry()` 의존 제거.
  `OverlayWindow.paintEvent` MODE_BOX 박스 그리기에서 `ctl.geometry().x()` 의존 제거.
  `TransparentWindow.initUI`의 size_grip widget-coord 위치 코드 정리(grip은 hidden 상태 그대로).
  설정 창 이동/숨김이 캡처 영역에 영향 주지 않는 구조로 정착 (백그라운드 번역기 UX에 부합).
  smoke/pre-release gate에 BG-5 검사 추가.
  박사 자체 리뷰 결함 1건 즉시 수정: smoke `_method` 헬퍼가 `async def`(AsyncFunctionDef)를
  처리하지 못해 `capture_and_translate_async` 검사 실패 → `(FunctionDef, AsyncFunctionDef)` 둘 다 처리.
- 2026-05-16: BG-7 RegionEditor 위젯 추가 (MODE_BOX 영역 마우스 편집).
  BG-5에서 사라진 "마우스로 박스 편집" UX를 전용 위젯으로 복원. 5개 컴포넌트 분리 구조 유지.
  `RegionEditor(QWidget)` — 가상 데스크톱 전체 frameless/translucent/always-on-top + B-10 capture 제외.
  내부 박스 상태도 절대 좌표 QRect (BG-5 결정과 일관). paintEvent에서만 editor-내 좌표 1회 변환.
  진입점: SettingsWindow "영역 편집..." 버튼 + 트레이 메뉴 동명 항목.
  저장 시 `region.red_rect` 절대 대입 + `set_capture_mode(MODE_BOX)` 자동 전환 + `reset_frame_hash`.
  취소 시 변경 사항 폐기. Esc/우클릭/취소 버튼 = cancel, Enter/저장 버튼 = commit.
  `_is_self_hwnd`가 region_editor.winId도 자기 hwnd로 인식 → worker가 편집 창 캡처/번역 안 함 (B-10 가드).
  민감 창 가드(S-1) / capturable 가드(B-15)는 BackgroundController에 그대로 — RegionEditor는 geometry만 변경.
  SettingsWindow 폭 740→880 — 첫 행에 "영역 편집..." 버튼 공간 확보.
  closeEvent에 `region_editor.force_close()` 정리.
  smoke 19→20건 (`test_bg7_region_editor`) / pre-release 14→15건 (`check_bg7_region_editor`) 게이트 추가.
  사용자가 `pip install -r requirements.txt` 승인 → 박사 환경에서 전체 검증 통과:
    smoke 20/20 OK, pre-release 15/15 OK, `--self-test` OK (Tesseract + CUDA env).
  실 GUI 회귀(드래그/리사이즈/저장/취소/MODE_BOX 자동 전환/B-10 가드)는 사용자 PC 검증 필요.
- 2026-05-04: BG-6 SettingsWindow 리네이밍 + region 콜백 통합 + dead code 정리.
  Phase 0 분리 사이클 종결.
  `TransparentWindow` → `SettingsWindow` 일괄 리네이밍 (Cocktail완성본.py / smoke_tests.py /
  pre_release_check.py / ARCHITECTURE.md / TECH_STACK.md / COMMERCIAL_SOTA_REVIEW.md).
  `CaptureRegionController(owner)` → `CaptureRegionController(is_self_hwnd)` —
  UI 클래스 내부 구조 의존(self.owner.winId / self.owner.overlay) 완전 제거.
  `BackgroundController._is_self_hwnd`와 같은 콜백을 region도 공유.
  `SettingsWindow.paintEvent` 빈 override 제거 (Qt 기본이 위젯 트리 자동 처리).
  `_on_state_ready`에서 `self.update()` 호출 제거 (settings window는 그릴 게 없음 — overlay만 갱신).
  smoke/pre-release gate에 BG-6 검사 추가, 기존 2건(test_overlay_hwnd_guards, test_capture_region_controller) BG-6에 맞춰 갱신.
  ARCHITECTURE.md "Phase 0 분리 사이클 종결" 마킹.
- 2026-08-02: 오버레이 표시 결함 6건 수정 — W-4(가상 데스크톱 클립 + 스킵 시 오버레이 비우기),
  W-5(`ImageGrab.grab(all_screens=True)`), W-6(마우스 통과 플래그 단일 출처화),
  UX-3(정지 시 자막 잔상 제거), F-1(frame-skip: 영역 좌표 포함 + 셀 최대 변화량 기준, RGB2GRAY 정정),
  F-2(`setPixelSize`로 폰트 픽셀 단위 지정). smoke_tests 21→23건, 5초 체크리스트 7줄 추가.
  번역 파이프라인(M-8 sepvoc, batch_translate) 미변경.
- 2026-08-02(2차): 실사용 성능/표시 결함 4건 — SC-1(활성 창 모드를 활성 모니터 1개로 클램프),
  LD-1(HF 로컬 캐시 우선 로드, 전역 오프라인 강제 금지), RT-1(캐시 키 텍스트 정규화로 재번역 폭주 차단)
  + RT-2(실질 동일 상태 emit 생략), NZ-1(OCR 라인 confidence/실문자 비율 필터).
  smoke_tests 23→27건, 5초 체크리스트 5줄 추가. 번역 파이프라인(M-8 sepvoc, batch_translate 라우팅) 미변경.
- 2026-08-02(3차): DP-1 — DPI 좌표계 물리 픽셀 통일(`QT_ENABLE_HIGHDPI_SCALING=0`).
  100%/175% 혼합 멀티 모니터에서 주 모니터에 오버레이가 아예 안 뜨던 근본 원인 해소.
  E-7 정책 문구 갱신(수동 곱셈 금지는 유지, Qt 논리 스케일링도 OFF), ARCHITECTURE.md §1.5 단위 규약 명시.
  smoke_tests 27→28건, 5초 체크리스트 2줄 추가. 번역 파이프라인 미변경, 1·2차 수정 유지.
- 2026-08-02(4차): OU-1(작은 글씨 적응 업스케일 — 1차 OCR 라인 높이 중앙값 < 18px이면 2배 확대 재OCR,
  E-3의 이미지 크기 조건이 못 잡던 "큰 화면 속 작은 캡션" 커버), LG-1(페이지 지배 언어 스냅 —
  라틴+저확신 라인만 지배 언어로 수렴, 비라틴 보존). doc_55 실측: 캡션 블록 전멸 → 복구(33→36 라인),
  언어 `{en:32, af:1}`/`{en:34, no:1, de:1}` → `{en:36}`.
  smoke_tests 28→30건, 5초 체크리스트 4줄 추가. 번역 파이프라인(M-8 sepvoc, batch_translate) 미변경,
  1~3차 수정 유지.
- 2026-08-02(5차): OS-1(OSD script 오판이 `_script_cache`에 고착돼 세션 전체를 오염 —
  `script_conf` 신뢰도 게이트 + 후보에 eng 상시 합류 + 성공 경로 로그),
  RP-1(잡음 입력 반복 붕괴 — 두 `generate`에 `no_repeat_ngram_size=4`).
  영어 PDF → OSD "Arabic" 오판 → ara OCR 쓰레기 → `[ar]` → m2m100 ar→ko 반복 쓰레기 사슬을
  1단계(OCR 언어 선택)와 4단계(디코딩)에서 각각 차단.
  smoke_tests 30→32건, 5초 체크리스트 5줄 추가. 번역 파이프라인 구조 무접촉
  (generate 파라미터 1개 추가만 예외 허용), 1~4차 수정 유지.
- 2026-08-02(6차): SL-1 — auto 모드 언어 라우팅 결정론화(설계 결정, 오판 장치 퇴역).
  OSD script 판별(UX-2 Layer 3-5) + langdetect 추측 + M-7 region 투표 + LG-1 지배 언어 스냅을
  **전부 삭제**하고, auto = "OCR eng+kor 고정 / 라인 언어는 스크립트 확정만, 라틴·불명은 en"으로 교체.
  계기: 영어 PDF를 OSD가 conf 3.95로 Arabic 판정 — OS-1 신뢰도 게이트(1.0)를 통과했다(게이트로는 못 막음).
  하루 실사용 오판 6종(cy/hu/de/pt/so/ar) 실증. 다른 언어는 src 콤보 명시로(탈출구 유지:
  `COCKTAIL_OCR_LANGS_AUTO`). 사이클에서 OSD(실측 111ms/회)·langdetect(~130ms/15라인) 제거,
  오판발 m2m100 폴백(~5초/배치)도 소멸. Cocktail완성본.py 3274→2949줄(-325).
  smoke_tests 32→31건(test_m7/test_lg1/test_os1 → test_sl1 2건으로 교체, test_ux2는 UIA 라우팅만 유지),
  pre-release `check_smartocr_wiring` → `check_sl1_auto_language_routing`.
  5초 체크리스트 갱신(OSD/LG-1/M-7 항목 → SL-1 항목). 문서 동기화(README/README.en/TECH_STACK/
  ARCHITECTURE/HANDOFF). 번역 파이프라인(M-8 sepvoc, batch_translate, RP-1) 무접촉, 1~5차 수정 유지.
- 2026-08-02(7차 · 코드 완성 패스): 남은 품질·안정·보안 결함 9건 일괄 마감.
  **표시 품질** — DG-1(표시 직전 최종 게이트: 극소 라인+저conf / 목표 스크립트 실종 / 길이 붕괴는
  아예 안 그림), LM-1(거대 폰트 뭉침 근본 수정: word 높이 이상치 제거 + 큰 간격 분할 +
  font_size max→중앙값. 렌더 재현 실측 font 81→11, bbox 폭 645→167/144), FT-1(한글 우선 폰트 스택),
  TM-1(show 후 raise_ + 2초 주기 topmost 재적용, SWP_NOACTIVATE), CL-1(클립을 박스가 속한 모니터로 축소).
  **안정성** — RC-1(워커가 메인 스레드 갱신 상태를 읽던 race → `_last_emitted` 로컬 + emit 단일 경로),
  GT-1(워커의 GUI 전용 Qt API 호출 제거: 화면 rect 캐시 + `GetCursorPos`),
  MS-1(모니터 추가/제거/해상도 변경에 오버레이·캐시·RegionEditor 재적합).
  **보안** — SV-1(민감 창 UIA 차단이 OCR 폴백으로 새던 구멍을 `UIA_SKIP_FRAME` sentinel로 봉쇄).
  smoke_tests 31→35건(신규 4 + 기존 3건을 새 배선에 맞춰 갱신), 5초 체크리스트 11줄 추가.
  DP-1 물리 픽셀 규약·SL-1 결정론 라우팅·M-8 sepvoc·`batch_translate` 무접촉, 1~6차 수정 유지.
- 2026-08-02(8차 · 배포 준비 패스): 코드 무접촉, "받아서 더블클릭으로 쓰는 제품"이 되기 위한
  실행·문서·의존성 정합 마감. RB-1(`run_cocktail.bat` 인터프리터 우선순위 탐색 —
  `COCKTAIL_PYTHON` → conda `cd` env → `py -3.11` → `python`, `import PySide6`로 판별,
  ASCII 전용. 3분기 실측 검증), DR-1(사어 의존 전수 제거: `langdetect` 4곳 +
  `osd.traineddata` 동봉 목록 2곳 + 라이선스 문서, 과장 문구 사실화).
  `diagnose_environment.py`가 앱과 같은 `TESSDATA_PREFIX` 규칙을 쓰도록 수정(진단/실행 불일치 해소).
  `ENVIRONMENT.md` 현행화(존재하지 않는 `aigugbi` 환경 → 실측 `cd` env: Python 3.11.15 /
  PySide6 6.11.1 / transformers 5.8.1 / torch 2.11.0+cu126, Tesseract 5.5.0 `C:\tesseract`,
  환경변수 6종, `debug_run.bat` 진단 절차). `nllb_ct2/`(600MB, CC-BY-NC)는 **삭제하지 않고**
  배포 산출물에 안 들어감을 확인 + README/BUILD/ENVIRONMENT에 "배포 전 삭제 필요"로 명시.
  문서 정합(HANDOFF 상태·테스트 건수 21/14→35/17·코드 3,310줄, ARCHITECTURE OCR 라우팅 표,
  OCR_TEST_PLAN 기대 감지값 SL-1 반영, README 현재 상태에 DG-1/DP-1/멀티모니터 추가).
  smoke_tests 35건 / pre-release 17건 / `--self-test` 전건 통과. 1~7차 수정 전부 유지.
- 2026-08-11: CO-2 — DG-1 미번역 잔류 규칙이 **정상 문단을 통째로 숨기던** 거짓 양성 수정.
  증상은 "영어 PDF의 특정 3줄 문단만 자막이 안 뜸". 재현 하네스로 단계별 생존을 수치화해
  OCR(conf 94.8)·PM-1 병합·SE-1 번역(길이비 0.49/0.37)·`max_new_tokens`(실측 24~99, 상한 256)를
  **전부 배제**하고 원인을 DG-1 CO-1 하나로 확정했다. 근본 원인은 한 줄(30~60자) 기준으로 잡은
  절대 임계 0이 PM-1 이후 200~350자 문단에 그대로 적용된 것 — 40단어 중 OCR 오독 1단어가 문단 전멸.
  허용치를 번역문 어절 수에 비례(`DISPLAY_CARRYOVER_MAX_RATIO=0.12`)시키고,
  숨김 사유+원문 머리를 라인당 1회 로그로 남기도록 `display_gate_reject`를 단일 진입점으로 정리.
  테스트 41→43건(빠른 1 + `--full` 1). 속도 영향 0(40줄 게이트 2.87ms 동일).
  CO-1이 막던 반쯤 번역된 툴팁은 그대로 차단됨을 실측 fixture로 확인.
- 2026-09-05(B4 · 접근성 패스): 합격기준 6 통과. LZ-1(torch/transformers 지연 import —
  창 보임 12~14s → **1.2~1.4s**, `import cocktail_ui` 10.24→1.05s), PK-1(Tesseract 엔진 동봉),
  PK-2(en-ko 모델 399MB 동봉 결정 — zh-en·m2m100은 필요 시 다운로드), IC-2(코드로 그린
  `assets/icon.ico`를 exe/인스톨러에 연결). `dist/`·`build/` 2026-05-04 구산출물을 현재 코드로
  재빌드(dist 5,318MB — torch CUDA 3.9GB가 대부분). 채점판 기준 6을 import 시간 대신
  **창이 보일 때까지의 시간**으로 바꾸고(외부 프로세스 측정), 자가진단·시작 측정을
  **패키지 exe**로 돌리게 했다("파이썬 불필요"를 실제로 재는 유일한 방법).
  테스트 57→58건(LZ-1 회귀 가드). 인스톨러(.exe) 컴파일은 Inno Setup 미설치로 못 했다 — 사용자 승인 사항.

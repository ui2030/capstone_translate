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
- [ ] 중복 텍스트 캐시(LRU)에서 먼저 lookup 하는가?
- [ ] 핫 패스(paintEvent, OCR 루프, generate 루프)에 `print()`가 있지 않은가?
- [ ] 예외 발생 시 `stop_thread=True`로 전체 루프를 죽이지 않는가? (continue가 정답)
- [ ] 중복 dedup 시 OCR 원문 vs OCR 원문을 비교하는가? (번역문과 비교 X)
- [ ] 언어 코드를 만질 때 `LANG_MAP` + `_LANGDETECT_TO_ISO` + `M2M100_LANGS` + 콤보박스 4곳을 동시에 확인했는가?
- [ ] auto 언어 감지는 `identify_language()`의 Unicode script fast-path를 먼저 타는가? (비라틴을 langdetect에 맡기지 말 것) — M-6
- [ ] 다국어 auto 번역은 라인 단독 감지가 아니라 bbox region smoothing을 적용하는가? — M-7
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
- [ ] 등록된 언어쌍은 opus-mt(전용), 그 외는 m2m100(폴백)으로 라우팅 되는가? — M-5
- [ ] 시그널/슬롯 데코레이터가 PySide6 Signal/Slot인가? (PyQt5 pyqtSignal/pyqtSlot 흔적 없음) — E-10
- [ ] OCR 언어가 하드코딩(`"eng+kor"`)이 아니라 사용자 src 콤보에 따라 동적 결정되는가? — B-14
- [ ] 사용 가능한 traineddata를 시작 시 진단(`pytesseract.get_languages()`)했는가? — B-14
- [ ] 배포 빌드(.exe)에서 `./tessdata/` 가 있으면 `TESSDATA_PREFIX` 자동 설정되는가? — 인스톨러 호환
- [ ] PaddleOCRv5는 optional lazy import인가? 설치/초기화 실패 시 Tesseract 폴백이 있는가? — O-1
- [ ] 일반 사용자 환경 문제가 앱 시작 예외가 아니라 UI 상태 라벨/진단 스크립트로 드러나는가? — U-1
- [ ] 트레이 상주 시 closeEvent는 hide()로 가고, 진짜 종료는 별도 경로(`_is_quitting` 플래그 등)인가? — T-1
- [ ] `keyboard` 같은 글로벌 단축키 라이브러리는 optional 처리 + 시그널/슬롯으로 메인 스레드 진입하는가? — T-1
- [ ] 캡처 모드 추가 시 `_update_red_rect_for_mode()`로 red_rect를 갱신해 기존 OCR/렌더링 파이프라인을 재사용하는가? — T-2
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
- [ ] PersistentCache는 import 시점이 아니라 lazy load(첫 get/put 또는 preload_async)인가? — D-1
- [ ] PersistentCache의 키는 sha256 hash이고, 값(번역)은 DPAPI로 전체 dict 암호화되어 디스크에 저장되는가? — D-1 / 보안 원칙 #2
- [ ] UIA 모드는 mode_combo와 무관하게 항상 활성 창 민감 검사를 적용하는가? — UIA-1
- [ ] uiautomation은 optional lazy import이고, 미설치 시 앱은 정상 동작하는가? — UIA-1
- [ ] DPAPI helper는 `argtypes/restype` 명시되어 64비트 호출 규약이 정확한가? — D-1 보강
- [ ] PERSIST_CACHE는 비정상 종료 손실 방지를 위해 주기적 save(QTimer 30초)가 동작하는가? — A-1
- [ ] 자동 실행 등록 시 `pythonw.exe` (콘솔 안 뜸) 또는 PyInstaller `.exe`를 우선 사용하는가? — A-1
- [ ] 새 의존성/모델 추가 시 LICENSE-3RDPARTY.md에 동시 기록했는가? (CC-BY-NC, GPL, 상용 전용 금지)

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

### E-7. DPI scaling 이중 적용 (정책 확정 + 수정 완료)
- **결정**: `Qt.AA_EnableHighDpiScaling=True` 정책을 신뢰. devicePixelRatio를 위젯 좌표에 추가로 곱하지 않음.
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
  - 1차: `Helsinki-NLP/opus-mt-tc-big-en-ko` (Apache 2.0). en→ko에서 NLLB-600M과 품질 비슷하고 사이즈 절반, 더 빠름.
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

---

### T-1. 시스템 트레이 + 글로벌 단축키 도입 (Phase 1A)
- **목적**: 비전 V-1의 "백그라운드 상주" 기반 — 트레이 상주 + 어떤 창에서든 토글 가능.
- **수정**:
  1. `QSystemTrayIcon` + `QMenu` (열기 / 번역 시작·정지 / 캡처 모드 / 종료).
  2. closeEvent: `_is_quitting=False`면 `event.ignore()` + `hide()` + 트레이 메시지. 트레이 "종료"가 진짜 종료 경로.
  3. `keyboard` 패키지(MIT) optional import — 미설치/등록 실패도 앱은 정상 동작.
  4. 글로벌 단축키 콜백은 keyboard 자체 스레드 → `global_toggle_signal` / `global_show_signal` 시그널로
     메인 스레드 진입 (B-11과 동일 원칙).
  5. 단축키: `Ctrl+Shift+T` (번역 토글), `Ctrl+Shift+A` (창 다시 열기).
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

## 5. 변경 이력

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

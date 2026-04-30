# Cocktail — 프로젝트 여정 (Story)

> 이 문서는 **다른 Claude 세션 또는 협업자가 이 문서 하나만 읽어도** 프로젝트의 전체 흐름,
> 결정의 이유, 현재 위치, 다음 방향을 이해할 수 있도록 작성되었습니다.
>
> 더 짧은 안내는 [README.md](README.md), 5초 체크리스트는 [errors.md](errors.md),
> 미래 비전은 [VISION_AND_ROADMAP.md](VISION_AND_ROADMAP.md), 변경 일지는 [UPDATE_LOG.md](UPDATE_LOG.md).
>
> 이 문서는 시간 순으로 읽기를 권장합니다. 결정 하나하나에 *왜* 그렇게 했는지가 담겨 있습니다.

---

## 0장. 시작 — 캡스톤 학술 프로젝트 (초기)

**처음의 모습**:
- 한국 대학 캡스톤 프로젝트로 출발한 "**실시간 화면 번역 오버레이**"
- 사용 사례: 영어 논문 PDF / 영어 자막을 화면에서 직접 한국어로 봐야 하는 한국 학생
- 기술 선택:
  - **OCR**: Tesseract (eng+kor 하드코딩)
  - **번역 모델**: `facebook/m2m100_418M` (Hugging Face, MIT)
  - **UI**: PyQt5 — 빨간 사각형으로 캡처 영역 지정 → 그 안에 한국어 오버레이
  - **목표**: en → ko 한 방향
- 코드 베이스: `Cocktail완성본.py` (단일 파일, ~500줄 시점)

**초기 한계 (학생 본인이 README에 적은 것)**:
> "원래는 Tesseract, Paddle, EasyOCR을 모두 사용하려 했지만 활용을 제대로 하지 못해
> 1개 쓰는 것보다 못한 성능으로 결국 하나만 사용했다.
> ... 아직 최적화를 거치지 못해 상당히 느리다."

이 솔직한 자기 평가가 박사 리뷰의 시작점이 됩니다.

---

## 1장. 박사 1차 리뷰 — 성능 병목 8건

**계기**: 사용자가 박사 페르소나로 "비판적 리뷰 + 자체 수정"을 요청 (2026-04-30).
워크플로우 합의: 코드 수정 전 `errors.md`를 먼저 읽고, 새 결함을 P-/B-/E- 코드로 누적 기록.

### 발견한 성능 병목

| ID | 무엇이 잘못되었나 | 어떻게 고쳤나 |
|---|---|---|
| **P-1** | 모든 번역이 src→en→tgt로 2번 generate (en→ko인데 en→en→ko) | src≠tgt면 1회 generate. m2m100는 any-to-any 직접 번역 가능. 영어 피벗은 비표준 쌍에서만 옵션 |
| **P-2** | `asyncio.gather`로 라인별 번역 — 보이는 건 병렬, 실제는 GPU 동기로 직렬 | 모든 라인을 한 번에 토크나이즈 + `model.generate(**batch)` 1회. 5~10라인에서 3~5× 가속 |
| **P-3** | `inference_mode` / `fp16` 미적용 | CUDA 가용 시 `model.half()`, `torch.inference_mode()` 블록 |
| **P-4** | 화면 변화 없어도 매 프레임 OCR | 캡처 직후 8×8 다운샘플 해시. 이전 프레임과 거의 같으면 OCR/번역 모두 스킵 |
| **P-5** | 같은 자막을 매번 번역 (캐시 부재) | LRU 캐시 `(src, tgt, text) → translation`. 같은 텍스트 두 번째는 0ms |
| **P-6** | 핫 패스(`paintEvent`, OCR 루프)에 `print()` 폭주 | 모두 제거. 필요하면 `logging.DEBUG`로 격하 |
| **P-7** | 매 라인마다 `langdetect` 호출 | source 콤보 추가 — "auto"일 때만 detect 호출 |
| **P-8** | `max_new_tokens` 미지정 → m2m100 디폴트 너무 큼 | `min(256, int(in_len * 1.6) + 16)` 상한 |

**배운 것 (재발 방지)**: 앞으로 모델 추가/교체할 때 inference_mode/fp16/batch generate 세 가지를
"5초 체크리스트"의 첫 항목으로 두기. errors.md에 항목 추가.

---

## 2장. 박사 2차 — 정확성 결함

성능 다음은 정확성. 같은 문서를 정밀 정독.

| ID | 결함 | 핵심 |
|---|---|---|
| **B-1** | dedup 비교 대상 오류 | 번역문(한국어)으로 비교 → 항상 0% 유사도. **OCR 원문끼리** 비교로 변경. `(src, tgt, bbox, font_size)` 4-tuple로 확장 |
| **B-2** | 예외 1번에 전체 워커 종료 | `stop_thread=True; break` 제거 → `continue` + 연속 실패 카운터 |
| **B-3** | `translated_text` 무한 누적 → 옛 번역이 화면에 쌓임 | 매 프레임 OCR 결과 기준으로 **재구성** (누적 X) |
| **B-4** | `line_height_threshold = 1`px → 라인 묶음 깨짐 | Tesseract `block/par/line_num` 직접 사용 |
| **B-5** | `paintEvent`에 매 프레임 print | 모두 제거. paintEvent는 초당 수십 회 호출 |
| **B-6** | langdetect 빈 문자열 → "[ERROR]" 그대로 화면 출력 | fallback은 콤보 source 또는 "en" |
| **B-7** | **타이핑 효과 + 0.25초 워커 주기 충돌**: 한 줄도 못 끝낸 채 typing_index가 0으로 리셋되어 깜빡임 | 타이핑 효과 자체 제거. paintEvent에서 모든 줄 동시 표시 |

**B-7의 교훈**: UI 애니메이션 상태(progress index)가 데이터 갱신 주기(워커 sleep)보다 길면 충돌한다.
이 원칙을 5초 체크리스트에 추가.

---

## 3장. **1초 후 깨지는 미스터리** — 자기 캡처 피드백 루프 (B-10) ⭐

이 장이 프로젝트에서 가장 결정적인 발견입니다.

### 증상
사용자가 스크린샷을 보냄:
- **첫 번째 화면**: 정상 한국어 번역 ("초석: 약물 결합 치료는 약물...")
- **두 번째 화면 (1초 후)**: 깨진 문자열 — "=A: 암 치료에 대한 약속", "기존 치료법 4 Syne AA?",
  "ICESp", "WAALLAEAAH ori"

명백히 영어로 번역 가능한 텍스트가 아닌, **번역의 번역의 번역** 같은 손상.

### 박사 진단
`ImageGrab.grab()`은 **데스크톱 합성 결과**를 캡처한다. 우리 오버레이 윈도우(빨간 박스 + 한국어 텍스트) **자기 자신**도 데스크톱에 그려져 있으므로 캡처에 포함된다.

```
1차 캡처 (정상) → OCR/번역 → paintEvent로 화면에 한국어 오버레이
       ↓ 0.25초 후
2차 캡처 → 영어 + 한국어 + 반투명 박스 가장자리 모두 포함
       ↓
OCR이 깨진 혼합 문자열을 읽음 → 그걸 다시 번역 → 더 깨진 결과
       ↓
paintEvent로 더 깨진 한국어 표시 → 3차 캡처에서 더 오염 → ...
```

자기 출력을 입력으로 받는 **피드백 루프**.

### 해결 — Windows API `SetWindowDisplayAffinity`

```c
SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)  // 0x11
```

OS가 데스크톱 합성에서 우리 윈도우를 빼므로 ImageGrab/mss/PrintScreen/화면 공유
**모든 캡처 API에서 우리 창이 보이지 않게** 된다. Windows 10 build 2004+ / 11에서 동작.

이건 다른 OCR 오버레이 도구들이 거의 다루지 않는 함정 — **Cocktail의 차별 포인트 1**.

### 부작용 (E-9)
사용자가 자기 화면을 스크린샷 찍을 때도 우리 창이 안 보임. 그래서 **토글 UI** 추가:
- "스샷차단" / "스샷OK" 버튼
- 단축키 `Ctrl+Shift+H`
- 캡처 허용 + 번역 동작 동시 시 OCR 피드백 위험 경고

**배운 것**: OS 레벨 부작용 있는 기능 도입 시 사용자가 풀 수 있는 경로(UI/단축키) 필수.

---

## 4장. 한국어 박스 자동 적응 (B-12, B-13)

**증상**: 영어 논문 한 줄(~80자)이 한국어로 번역되면 더 길어짐. 박스 안에서 잘리거나 영문이 비쳐서 겹쳐 보임.

### B-12 — height 동적 확장 + alpha 강화
- `QFontMetrics.boundingRect`로 실제 필요한 height 미리 측정
- `expanded_h = max(base, needed + 4)` 로 박스 height만 확장 (width는 영문 원본)
- 클립 영역을 `red_rect` → `widget.rect()` 로 (자기 캡처는 B-10이 막아주므로 박스가 빨간 사각형 살짝 밖이어도 안전)
- 박스 alpha 200 → **235** (영문이 뒤로 비치지 않게)

### B-13 — 폰트 자동 축소
좁은 영문 박스에 긴 한국어를 넣으면 height가 4~5배 폭증해서 다음 영문 줄을 덮음.
- `_fit_font()` 함수: initial_size부터 1pt씩 내림 → `needed.height ≤ base*2` 가 될 때까지
- 최소 9pt에서 멈춤 (가독성 한계)
- 그래도 안 맞으면 B-12의 height 확장으로 처리

이 결합으로 **논문 초록 한 페이지 통째로 번역**해도 박스가 자연스럽게 적응.

---

## 5장. 상용화 결정 — 두 BLOCKER (M-5, E-10)

**계기**: 사용자가 *"배포까지 갈거니까 상용화 잘 시켜놔"* 명시.

박사 라이선스 감사에서 **즉시 차단해야 할 두 결함** 발견.

### BLOCKER 1: NLLB CC-BY-NC (비상용)
- 그 시점 모델은 `facebook/nllb-200-distilled-600M` (CC-BY-NC-4.0).
- **상용 사용 금지**. Meta가 takedown 가능. 영리 목적 사용은 라이선스 위반.

### BLOCKER 2: PyQt5 GPL
- PyQt5는 GPL — 상용 closed-source 배포 시 (a) 소스공개 강제 (b) 상용 라이선스 구매 ($550/dev/yr).

### 사용자 결정 + 박사 추천 → 동시 적용

**M-5 — `HybridTranslator` 도입 (NLLB → opus-mt + m2m100)**:
- 1차: `Helsinki-NLP/opus-mt-tc-big-en-ko` (Apache 2.0). en→ko에서 NLLB-600M과 품질 비슷, 사이즈 절반.
- 2차 폴백: `facebook/m2m100_418M` (MIT, any-to-any 100언어).
- `BILINGUAL_MODELS` dict에 등록된 (src, tgt) → opus-mt fast-path. 그 외 → m2m100.
- 언어 코드: NLLB BCP47 (`kor_Hang`) → ISO 639-1 (`ko`)로 통일.
- 사용자 아이디어 채택: "en→ko opus가 있다면 그대로 사용하다 다른 언어쌍 만나면 별도 모델로"

**E-10 — PyQt5 → PySide6 마이그레이션**:
- PySide6 LGPL → closed-source 상용 OK, 무료.
- 핵심 변경: `pyqtSignal`/`pyqtSlot` → `Signal`/`Slot`. `QShortcut`은 `QtGui`로 이동.
  `globalPos()` → `globalPosition().toPoint()`. `AA_EnableHighDpiScaling` 자동 활성.

**P-9 — 모델 lazy load + UX**:
- `TRANSLATOR = init_translator()` 모듈 레벨 동기 로드 → GUI 시작 5~30초 지연.
- `HybridTranslator.__init__`은 가벼움. 첫 호출/`preload_default()`에서 lazy load.
- UI는 즉시 뜨고 백그라운드 스레드로 모델 다운/로드.
- `progress_msg` 시그널로 버튼 텍스트가 "준비: 토크나이저 로드 중" → "준비: 모델 다운로드 중" → "준비: 워밍업 중" → "번역 시작"

**메모리 영구 기록**: `project_commercial_decisions.md` — 라이선스 정책 향후 모든 변경에서 참조.

---

## 6장. 다국어 진화 (B-14) + 빌드 자산 (Phase 5/6)

### 사용자의 통찰

> "kor.traineddata가 뭐야? 한국인 아닌 사람한테 필요 없는 거 아닌가?"

박사 답: **완전히 맞다.** 두 결함이 있다.
1. `TESS_LANG = "eng+kor"` 하드코딩 → 일본어/중국어 자막 OCR 안 됨.
2. 인스톨러가 모든 traineddata를 강제 설치 → 비효율.

### B-14 — OCR 언어 동적
- `LANG_TO_TESS` 매핑 추가: ISO ↔ Tesseract lang.
- 시작 시 `pytesseract.get_languages()` 진단 → `AVAILABLE_TESS_LANGS` 캐시.
- `_ocr_langs()`: src 콤보 명시면 단일 언어, "auto"면 사용 가능한 모든 언어 합집합.
- 배포 시 앱 폴더 `./tessdata/` 있으면 `TESSDATA_PREFIX` 자동 설정.

### 빌드 자산 신규
- `requirements-dev.txt` — PyInstaller 등 빌드 도구
- `build.spec` — `transformers/torch/sentencepiece` `collect_all`
- `download_tessdata.py` — `tessdata_fast`에서 6개 언어 다운로드
- `installer.iss` — Inno Setup, **traineddata 옵션 컴포넌트** (eng 필수, kor/jpn/chi_sim/fra/deu 선택)
- `BUILD.md` — 4단계 빌드 절차 + 라이선스 체크리스트

이걸로 **"나에게 맞는 언어팩만 깔리는 인스톨러"** 완성.

---

## 7장. YOLO식 region smoothing (M-6, M-7)

### 사용자의 비전

여러 언어가 한 화면에 있는 사진(Thank you / Gracias / 감사합니다 / 謝謝 / شكرا)을 보내며:
> "YOLO에 있는 개별 인식 기능을 착안해서 글이 있는 부분을 나누고,
> 그 언어들을 지정 언어로 바꾸는 형태는 어떻게 생각해?"

박사 평가: **매우 좋다.** 현재 `_safe_detect`(langdetect)는 짧은 텍스트와 비라틴 script에 약함.
사실 라인 단위 region 분리는 이미 Tesseract block/par/line이 함 — 문제는 **언어 식별 정확도**.

### M-6 — Unicode script fast-path

```python
"감사합니다"   → HANGUL    → ko    (즉시, langdetect X)
"ありがとう"   → HIRAGANA  → ja    (즉시)
"謝謝"         → CJK only  → zh    (script 분포)
"شكرا"         → ARABIC    → ar    (즉시)
"Спасибо"      → CYRILLIC  → ru    (즉시)
"Gracias"      → LATIN     → langdetect 폴백 → es
"Hvala"        → LATIN     → _LATIN_SHORT_HINTS 사전으로 → hr
```

- `detect_script()` + `identify_language()` 함수 신설
- 비라틴 script는 결정론적 매핑 (langdetect 호출 감소 90%+)
- 짧은 라틴 인사/감사 표현(`Hvala`, `Merci`, `Hola`, `Danke`)은 `_LATIN_SHORT_HINTS`로 보정
- 외부 의존성 0 — `unicodedata` 표준 라이브러리만

### M-7 — bbox region smoothing

라인 단독 감지가 흔들리는 케이스를 안정화.
- `_cluster_text_regions()`: bbox 중심/간격으로 좌우 column / 세로 island 분리
- `_language_confidence()`: 비라틴 > 라틴 힌트 > 긴 라틴 > 짧은 라틴 순 신뢰도
- `assign_region_languages()`: src=auto일 때 region 안의 강한 언어 신호를 짧고 불안정한 라인에 전파
- `batch_translate()`가 라인별 source list 받을 수 있게 확장
- 상태 라벨에 이번 프레임 감지 언어 요약 표시

**효과**: 좌측 column이 영어, 우측 column이 한국어인 화면에서 좌측이 우측을 오염시키지 않음.

---

## 8장. 사용자 진입 안정화 (U-1 ~ U-4) + PaddleOCRv5 (O-1)

이 시점부터 박사 페르소나가 자체적으로 "일반 사용자 환경"을 점검하기 시작.

### U-1 — 시작 예외 X, 상태 라벨로 표출
- Tesseract 미설치도 import 단계에서 죽지 않게: `_resolve_tesseract()`가 None 반환
- `environment_summary()` 함수 + UI 상단 상태 라벨
- OCR 결과 0건 시 "영역/언어/OCR 백엔드 확인" 메시지

### U-2 — OCR 백엔드 콤보 + 폴백
- `Tesseract` / `PaddleOCRv5` 선택
- PaddleOCR 미설치/초기화/추론 실패 시 자동 Tesseract 폴백 + 사유 표시
- 번역 성공 시 OCR 줄 수와 소요 시간 표시

### U-3 — `run_cocktail.bat` + `diagnose_environment.py`
- 일반 사용자가 더블클릭으로 진단 후 앱 시작
- 필수 모듈, optional PaddleOCR, Tesseract 실행파일, traineddata, Torch device 점검

### U-4 — `TECH_STACK.md`
- 사용 기술 한 페이지로 정리

### O-1 — PaddleOCRv5 선택 백엔드
- `from paddleocr import PaddleOCR`은 사용자가 PaddleOCRv5 선택 시에만 lazy import (모듈 상단 X)
- 결과 파싱은 `res.json["res"]`와 직접 dict 양쪽 처리 (minor 버전 차이 흡수)
- Windows CPU 기본 `paddle_static`/oneDNN 실패 → `engine="paddle"` + `enable_mkldnn=False`로 회피
- PaddleOCR 설치 후 chardet 7.x로 인한 requests 경고 → `chardet<6` 호환 권장

**배운 것**: optional dependency는 모듈 상단 import 금지 (앱 시작 자체가 죽음).
일반 사용자 환경 문제는 시작 예외가 아니라 UI 상태 라벨/진단 도구로 표출.

---

## 9장. 비전 도약 — "컴퓨터를 모국어로" (V-1)

이 장이 프로젝트의 **2막 시작**입니다.

### 사용자의 큰 그림

작업 관리자/시작 앱 목록 스크린샷을 공유하며:
> "저는 제가 사용하는 언어를 사랑해요.
> 그러니 이 언어로만 볼 수 있는 세상 또는 다른 사용자의 언어로만 볼 수 있는 세상을 원해요."

영역 박스 도구가 아니라, **OS 전반을 모국어로 보이게 하는 환경**.

### 박사 시장 분석

| 영역 | 상태 |
|---|---|
| 모바일 OS 통합 번역 | Apple Live Text, Google Lens — 포화 |
| **PC OS 통합 번역** | **메이저 제품 없음 — 빈 자리** |
| 클립보드 번역 | DeepL, Mate Translate — OCR 없음 |
| PC 화면 OCR | PowerToys Text Extractor (번역 X), QTranslate (2018 멈춤), Capture2Text |
| 게임 특화 | LunaTranslator, Textractor — 일반 데스크톱 부적합 |

→ Apple/Google이 모바일에서 한 일을 PC에서 할 자리가 비어 있음.

### 박사 기술 접근 — UIA + OCR 하이브리드

```
1차 (UIA): Windows UI Automation → 표준 UI 텍스트 직접 읽기 (OCR 불필요, 정확)
            시작 메뉴, 설정, Office, 브라우저, Explorer, 작업 관리자 → 100%
            (스크린 리더 NVDA/JAWS 가 쓰는 동일 기술)

2차 (OCR): UIA로 못 읽는 영역(게임, Custom GDI, 영상 자막)만 폴백
           현재 Tesseract / PaddleOCR 그대로

번역: HybridTranslator (opus-mt + m2m100) 그대로
렌더링: 원래 위치 위 모국어 풍선 — 현재 Cocktail 방식 그대로
```

### 보안 5원칙 (절대 위반 금지)
1. **100% 로컬 처리** — 클라우드 전송 없음
2. **번역 데이터 비저장** — 메모리만, 디스크 캐시는 hash key
3. **민감 영역 자동 OFF** — 패스워드/결제/은행 감지 시 일시 중지
4. **오픈 코어** — 핵심 로직 공개로 신뢰 확보
5. **즉시 토글** — 단축키 한 번으로 OFF

### 단계별 로드맵
- **Phase 0** (현재 ✅): 영역 박스 OCR + 한국어 오버레이
- **Phase 1** (계획): 시스템 트레이 + 글로벌 단축키 + 모드 다양화 (박스/활성 창/마우스 hover/전체 화면)
- **Phase 2** (PoC): UIA 탐색 — `pywinauto` 또는 `uiautomation`
- **Phase 3**: UIA + OCR 하이브리드 통합
- **Phase 4**: 학습형 캐시 (디스크는 hash key, 평문 비저장)
- **Phase 5**: 민감 영역 자동 OFF
- **Phase 6**: 마켓 진입 (무료 + 프리미엄)

상세 — [VISION_AND_ROADMAP.md](VISION_AND_ROADMAP.md), 메모리 `project_vision.md`.

---

## 현재 위치

| 영역 | 상태 |
|---|---|
| 코어 기능 | ✅ 영역 박스 OCR + 한국어/다국어 오버레이 |
| 성능 최적화 | ✅ batched generate, fp16, frame-skip, LRU cache, lazy load |
| 라이선스 | ✅ 상용 호환 (PySide6 + opus-mt + m2m100 + Tesseract) |
| 빌드/배포 | ✅ PyInstaller + Inno Setup 인스톨러 (traineddata 컴포넌트화) |
| OCR 백엔드 | ✅ Tesseract 기본 + PaddleOCRv5 선택 |
| 다국어 처리 | ✅ Unicode script fast-path + region smoothing |
| 사용자 안정화 | ✅ 진단 스크립트 + 상태 라벨 + .bat 런처 |
| 비전 정의 | ✅ "컴퓨터를 모국어로" + Phase 0~6 로드맵 |
| Phase 1 (트레이/단축키/모드) | 🔵 다음 |
| Phase 2 (UIA PoC) | 🔵 다음 |

**베타 가능 상태**. Phase 1 시작 가능.

---

## 다음 세션을 위한 안내

이 문서를 처음 보는 세션에게 — 다음 순서로 컨텍스트를 잡으세요.

1. **이 문서 (`STORY.md`)** — 전체 여정. (지금 읽는 것)
2. **`errors.md`** — 5초 체크리스트 + P/B/E/M/O/U/V 결함 인덱스. 코드 수정 전 필독.
3. **`VISION_AND_ROADMAP.md`** — 비전 + Phase 0~6 + 보안 5원칙.
4. 메모리:
   - `project_cocktail_translator.md` — 프로젝트 정체성 (성능 우선)
   - `project_commercial_decisions.md` — 라이선스 정책 (NLLB/PyQt5 금지)
   - `project_vision.md` — 장기 비전
   - `feedback_workflow.md` — errors.md 누적 규칙
5. **`TECH_STACK.md`** — 현재 기술 + 미래 기술 후보 (UIA 등)
6. **`UPDATE_LOG.md`** — 최근 변경 일지
7. **`BUILD.md`** — 빌드/배포 절차

코드를 만질 때는 반드시:
- `errors.md` 5초 체크리스트 위반 없는지 확인
- 새 결함은 코드 prefix(P-/B-/E-/M-/O-/U-/V-)로 누적 기록
- 라이선스 audit (CC-BY-NC, GPL, 상용 전용 금지)
- 보안 5원칙 위반 시 즉시 중단 + 사용자 보고

---

## 핵심 결정의 한 줄 요약

1. **성능 우선**: greedy + batch + fp16 + cache + frame-skip
2. **상용 라이선스만**: PySide6 + opus-mt + m2m100 + Tesseract (NLLB·PyQt5 금지)
3. **자기 캡처 차단**: SetWindowDisplayAffinity (다른 도구 안 막는 함정)
4. **lazy load**: UI 즉시, 모델은 백그라운드
5. **다국어 자동 라우팅**: Unicode script fast-path + region smoothing
6. **사용자 안정화**: 시작 예외 X, 진단 + 상태 라벨로 표출
7. **비전**: 영역 박스 → OS 전반 자동 번역 (UIA + OCR 하이브리드)
8. **보안**: 100% 로컬 + 비저장 + 민감 영역 OFF + 오픈 코어 + 즉시 토글

---

*이 STORY가 다음 세션의 출발점이 되기를. 박사로서 정직하게 정리했습니다.*
*— 2026-04-30*

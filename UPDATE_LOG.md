# Cocktail 업데이트 로그

## 2026-09-07 — 폰트별 인식 붕괴 근본 수정 (OU-2 / OU-3 / UA-2 / UA-3 / PM-2 / DG-1c)

사용자 제보: "폰트에 따라 문자 인식이 무너진다". 실측 448건에서 같은 문장·같은 크기인데
**폰트만 바꾸면 CER이 0%↔100%로 갈렸다.** 배경(어두움·저대비·무늬·반투명)은 거의 무관(가설 기각).

| 결함 | 무엇 |
|---|---|
| OU-2 | 확대 재OCR 트리거가 1차가 읽은 라인 높이만 봤다 → 1차가 무너지면 10px을 28px로 보고해 확대가 꺼진다. 1차 0줄이면 아예 건너뛴다 |
| OU-3 | 확대 보간 LANCZOS4의 링잉이 좁은 글꼴을 뭉갠다 (Impact 11px은 **정확히 2.0배에서만** 0줄) |
| UA-2 | 채택 기준이 평균 conf 하나뿐 → **줄을 통째로 흘린 패스**가 이긴다 (CER 7.51→45.85) |
| UA-3 | 반대로 3~4점 낮다고 **더 정확한 패스**를 거부한다 (11.46 vs 5.53) |
| PM-2 | 문단 병합이 비인접 줄을 묶어 문장을 "1 2 4 3" 순서로 내보낸다 (CER +30~37pt) |
| DG-1c | 표시 게이트 4규칙이 전부 번역문만 본다 → 원문이 그럴듯하게 깨지면 **틀린 자막이 그대로 뜬다** |

### 폰트별 CER before → after (앱 진짜 경로, 37조합)

| 조합 | before | after | | 조합 | before | after |
|---|---|---|---|---|---|---|
| Courier New 10px | 100.00% | **0.00%** | | Georgia 10px 실화면 | 33.99% | **0.00%** |
| Courier New 10px italic | 100.00% | **0.00%** | | Calibri 10px 실화면 | 16.60% | **0.00%** |
| Courier New 10px bold-italic | 100.00% | **0.40%** | | Times 10px 실화면 | 22.92% | **1.19%** |
| Courier New 10px 실화면 | 100.00% | **0.00%** | | Comic Sans 10px 무늬 | 45.45% | **0.00%** |
| Impact 11px | 100.00% | **1.19%** | | Comic Sans 10px 반투명 | 45.06% | **0.00%** |
| Impact 11px 실화면 | 4.74% | **0.40%** | | Comic Sans 10px 실화면 | 17.39% | **0.00%** |
| Impact 12px | 5.53% | **2.77%** | | Calibri Light 10px | 23.32% | **0.00%** |
| Impact 12px 실화면 | 7.51% | **2.37%** | | Calibri italic 12px | 30.04% | **5.53%** |
| Georgia 10px | 10.67% | **0.00%** | | Trebuchet 10px 실화면 | 3.95% | **0.00%** |
| Calibri 10px | 10.67% | **0.40%** | | **평균 (37조합)** | **21.72%** | **0.49%** |
| Times New Roman 10px | 10.67% | **0.00%** | | **최악** | **100.00%** | **5.53%** |

회귀 확인 — Segoe UI 10/11/12/13/14/18px(밝은·어두운·무늬 배경) · Arial · Verdana · Consolas ·
Tahoma: **한 조합도 나빠지지 않았다**(전부 0.00%, Segoe UI 11px만 원래대로 3.16%).
OCR 총 시간은 12,137 → **11,520 ms**(-5%) — 오히려 빨라졌다.

### 채점판

| # | before | after |
|---|---|---|
| 1 작은 글씨 | **실패** 14px 이하 CER 최대 **100.0%** | **통과 2.8%** (23시료 평균 19.40% → **0.22%**) |
| 2 오번역 | 통과 오탐 0/68 = 0.0% | **통과** 오탐 **0/79 = 0.0%** |
| 3 다중 문맥 | 통과 | 통과 (graded 3화면 전부 0건) |
| 4 속도 | 통과 p50 1.46 / p95 2.38 | **통과 p50 1.45 / p95 2.59** (오염=아니오) |
| 5 UI | 통과 | 통과 |
| 6 접근성 | 통과 | **실패 — 배포물이 소스보다 낡음**(재빌드 대기, 이번 라운드에서 안 함) |

기준 1의 before가 100%인 것은 이번에 추가된 폰트 축 시료(`size_impact_11`, `size_courier_10`,
`screen_courier_10` 등) 때문이다. 사용자가 본 그 증상을 시료가 처음으로 잡아낸 것이다.

### 코드

- `cocktail_ocr.py` — `ink_mask()` / `ink_glyph_px()` 신규(OCR 무관 글리프 높이),
  레버 `OCR_UPSCALE_TRIGGER_INK_PX=11` · `OCR_INK_GLYPH_MIN/MAX_PX` · `OCR_INK_MIN_GLYPHS` ·
  `OCR_UPSCALE_MIN_COVERAGE=0.8` · `OCR_UPSCALE_CONF_MARGIN=4.0`.
  `unread_ink_ratio()` 는 `ink_mask()` 를 공유하게 정리(동작 동일).
- `cocktail_engine.py` — `_wants_upscale()` 신규, `_tess_scan` 재구성(싼 조건 먼저 → 이미지 측정),
  `_tess_lines(..., blind=)` 로 확대 보간 분기, `merge_paragraph_lines` 에 읽기 순서 가드.
- `cocktail_translate.py` — `DISPLAY_TINY_LINE_PX`/`DISPLAY_TINY_LINE_MIN_CONF` →
  `DISPLAY_SRC_MIN_CONF=60.0`, 게이트 규칙 1이 `line_px` 를 안 본다.
- `test_cocktail.py` — 61 → **65**. 신규 가드 4개 + 기존 게이트 테스트를 새 계약으로 교체.

### 안 고친 것

- **Calibri italic 12px 5.53%** — 남은 최악값. 1차/2차 어느 쪽도 5% 아래로 안 내려간다
  (배율 1.3~3.0 × 보간 3종을 훑어 확인). 임계를 이 한 조합에 맞춰 흔들지 않았다.
- **한국어 폰트(Malgun Gothic·Gulim·Batang) CER 20~60%** — 목표 언어가 한국어면 SK-1이 어차피
  버리는 줄이라 이번 판정 대상이 아니다. 한→외 번역을 볼 때 다시 봐야 한다.
- **원문 언어 신호(사전 밖 토큰·붙은 단어 비율)** — DG-1c 항목의 "한계" 참고. 근거 없이 임계를
  잡느니 안 넣었다.


## 2026-09-07 — 체감 지연: 채점은 통과인데 사용자는 답답했던 이유 (PL-1 · PL-2 · BM-1 · QU-1)

`cocktail_capture.py` / `cocktail_engine.py` / `bench/score.py` / `test_cocktail.py`.
증상·원인·실측은 `errors.md` PL-1 / PL-2 / BM-1 / QU-1.

### 1. 채점기가 사용자 기준과 다른 것을 재고 있었다 — BM-1

기준 4는 **한 사이클 시간**(캡처+OCR+번역)을 재고 있었다. 사용자 기준은 원래
**"화면이 바뀐 순간 → 자막이 뜨는 순간"**이다. 빠진 것이 폴 간격(250~300 ms)과
"앞 사이클이 끝나기를 기다린 시간"이었고, 그 둘이 실제 지연의 25~45%였다.

이제 `bench/score.py`가 **실제 앱을 띄워** t0(화면 글자 교체) ~ t1(오버레이 paintEvent에
그 번역이 그려짐)을 잰다. 시나리오 3종 — 자막(배경 고정, 한 줄 교체) / 채팅(전체가 위로 밀림) /
정적 문서(한 장을 한 번). 오버레이는 `WDA_EXCLUDEFROMCAPTURE`라 스크린샷으로는 못 잡으므로
`paintEvent` 훅이 유일한 계측점이다. 자막이 안 뜬 건 `missed`로 세어 통과를 막는다.

### 2. 폴 간격이 고정 0.25초였다 — PL-1 / PL-2

캡처를 116→32 ms로 줄여 놓고도 폴 주기가 그대로라 매 사이클 250 ms가 지연에 그대로 붙었고,
사이클 **주기**가 길어져 채팅처럼 계속 바뀌는 화면에서는 대기가 **누적**됐다(QU-1).

`poll_interval()` — 최근에 일거리가 있었으면 0.06s, 조용해지면 1.2초 뒤 0.25s(예전 값)로 복귀.
기준 시각은 "변화를 잡은 순간"이 아니라 **"직전 사이클이 끝난 순간"**이다(PL-2: 시작 시각으로
재면 사이클이 2초인 화면에서 창이 도중에 만료돼, 빠른 폴이 가장 필요한 곳에서만 안 들었다).

| 새 기준 4 (같은 PC·같은 시료) | before | after |
|---|---|---|
| 전체 p50 / p95 | 1.54s / 3.10s → **실패** | **1.46s / 2.38s → 통과** |
| 자막(배경 고정) p50 / p95 | 0.75s / 1.62s | 0.76s / 1.96s |
| **채팅(전체 스크롤) p50 / p95** | 2.94s / 3.11s | **1.91s / 2.61s** |
| 정적 문서 p50 | 1.51s | 1.44s |
| 유휴 워커 CPU(코어 1개 기준, 20초) | 5.23% | 5.93% (폴 횟수 70→69, sleep 250.0 ms 동일) |

**마진은 얇다**: 실행 4회의 p50이 1.36/1.38/1.46초로 합격선 1.5초에 겨우 붙는다.
다른 앱이 CPU 37%를 쓰던 회차는 p50 2.15초로 **실패**했다 — 환경이 깨끗할 때만 통과다.

### 3. 재빌드가 드러낸 배포 사고 — PK-3

소스를 고쳤으니 `dist/` 를 재빌드했더니 패키지 exe의 self-test가 실패했다:
`transformers import failed: DLL load failed while importing _ssl`.
**동봉한 Tesseract의 `libcrypto-3-x64.dll`(5.3MB)이 `_internal` 루트에도 복사돼
파이썬이 쓰는 OpenSSL(7.4MB)을 덮어쓰고 있었다** — OCR·UI는 멀쩡하고 번역만 죽는 형태다.
`build.spec` 에서 루트로 흘러든 tesseract DLL 사본만 걷어내고 파이썬 OpenSSL을 되돌렸다.
`tesseract/` 폴더 사본은 그대로다(그 엔진은 자기 폴더에서 DLL을 로드한다).

### 4. 남은 병목은 OCR이고, 그건 정직한 비용이다 — QU-1

1920x1080 빽빽한 텍스트 화면의 OCR 1.4~1.5초. 세 가지를 확인해 전부 기각했다:
PNG 직렬화(4~8 ms, 범인 아님) / 확대 재OCR이 낭비인가(**시도 100회 중 99회 채택** — 끄는 건 정확도를
파는 것) / 언어 재스캔 중복(사이클당 tesseract 1패스, 안 돌았다). 더 줄이려면 B2(인식 엔진 교체)다.

처음 측정한 것: 5760x2160 투명 오버레이 `paintEvent` 본체 **0.6~3.5 ms** — 지연에 무관하다.

## 2026-09-05 (5) — 합격기준 6(접근성) 통과 · **6/6 전부 통과** (LZ-1 · PK-1 · PK-2 · IC-2)

`cocktail.py` / `cocktail_ocr.py` / `cocktail_engine.py` / `cocktail_translate.py` /
`build.spec` / `installer.iss` / `bench/score.py` / `test_cocktail.py`.
증상·원인·실측은 `errors.md` LZ-1 / PK-1 / PK-2 / IC-2.

### 1. 시작이 느린 게 아니라 **창이 늦게 뜨는** 것이었다 — LZ-1

| | before | after |
|---|---|---|
| 프로세스 → 창 보임 | 12~14s | **1.19~1.41s** |
| `import cocktail_ui` | 10.24s | **1.05s** |
| → 모델 준비 완료 | 18~20s | **15.4s** (창 뜬 뒤 백그라운드) |
| → 첫 자막 | 22s | **≈16.5s** |

- `cocktail_translate` 최상단의 `import torch`(2.3s) / `from transformers import ...`(2.0s)를
  삭제하고 `_ensure_torch()` + 생성자 내부 import로 옮겼다. 모델 로드는 **이미** 백그라운드
  스레드였는데 import가 그 앞에 남아 있었다 — "lazy load"라고 적어 놓고 아니었던 셈.
- `environment_summary(probe_torch=False)`: 진단 한 줄 때문에 창 뜨기 전 메인 스레드가
  torch를 import하던 두 번째 경로를 막았다. 모델 준비 후 엔진이 다시 불러 CUDA/CPU를 확정.
- `--self-test`는 반대로 일부러 무겁게(torch + transformers 실제 import) — 배포 사고를
  자가진단이 놓치면 안 된다. 자가진단은 사람이 기다리는 명령이 아니다.
- 버튼 `모델 준비 중…` → 준비 완료 흐름과 워커 스레드는 무접촉.

### 2. 배포 — 별도로 설치할 것이 하나도 없게 (PK-1 / PK-2 / IC-2)

- **Tesseract 엔진 동봉**: `tesseract.exe` + DLL 56개 + Apache 2.0 `LICENSE`/`AUTHORS`(68MB).
  탐색 순서 `TESSERACT_CMD` → 동봉본 → `C:\tesseract` → Program Files.
  PyInstaller onedir가 datas를 `_internal/`에 넣으므로 `sys._MEIPASS`도 탐색 경로에 추가.
- **기본 모델 동봉 결정**: `opus-mt-tc-big-en-ko` safetensors 399MB만.
  중복 사본(`pytorch_model.bin` 399MB, `tf_model.h5` 806MB)은 뺐다 → 1.6GB가 402MB로.
  zh-en·m2m100(합 2.2GB)은 그 언어를 실제로 쓸 때 다운로드.
  근거: 오프라인·프라이버시가 강점인데 첫 실행에만 인터넷을 요구하면 첫인상에서 그 약속이 깨진다.
- **아이콘**: `python cocktail.py --make-icon` → `assets/icon.ico`(멀티사이즈 16KB).
  `cocktail_ui.make_app_icon()`(코드로 그린 'A→가')이 유일한 원본 — 외부 이미지 없음.
- **재빌드**: `dist/` 2026-05-04 구산출물 → 현재 코드로. **5,318MB**
  (torch CUDA 3,851 / 모델 401 / cv2 109 / PySide6 92 / tesseract 68 / tessdata 30).
- `dist/Cocktail/Cocktail.exe --self-test` **OK** — 파이썬 없이 도는 것을 실제로 확인.

### 3. 채점판이 재는 것을 바꿨다 (`bench/score.py` 기준 6)

`import 시간` → **`창 보임 시간`(≤2.0s, 프로세스 밖에서 창 폴링)**.
무거운 것을 백그라운드로 옮기면 import 시간과 체감이 갈라지고, 사람이 기다리는 건 창이다.
자가진단·시작 측정은 **패키지 exe**로 돌린다("파이썬 불필요"를 재는 유일한 방법).
판정에 동봉 여부(tesseract/tessdata/모델) · exe 아이콘 리소스 ·
`torch_imported_at_ui_import == False`(LZ-1 회귀 가드)를 추가.

### 4. 검증

`py_compile` OK · `test_cocktail.py` **58/58**(신규 `test_window_opens_without_importing_torch`) ·
`pre_release_check.py --allow-build-artifacts` OK · `cocktail.py --self-test` OK ·
`bench/score.py` **6/6 통과**.

### 5. 못 한 것 (사용자 결정/승인 필요)

- **인스톨러 .exe 미생성** — Inno Setup(ISCC.exe)이 이 PC에 없다. 설치는 사용자 승인 사항이라
  설치하지 않았다. 설치 후 `ISCC installer.iss` → `Output\Cocktail-Setup-1.0.0.exe`.
- **dist 5.3GB 중 3.85GB가 torch CUDA.** CPU 휠이면 ~1.4GB지만 번역 348ms→1780ms(5.1배).
  HANDOFF_NEXT §9 "최소 사양" 결정과 같은 질문.
- 동봉 안 된 언어쌍의 첫 다운로드에 **진행률(%) 없음**(상태줄 문구만).
- 깨끗한 PC 설치·백신(코드 서명 인증서 미보유)·실제 5분 확인은 사람 몫(채점판 체크리스트).

## 2026-09-05 (4) — 합격기준 2(오번역 없음) 통과 (PY-2 · ZH-2 · RP-1 · DG-1b)

`cocktail_ocr.py` / `cocktail_engine.py` / `cocktail_translate.py`.
자세한 증상/원인/실측은 `errors.md` PY-2 / ZH-2 / RP-1 / DG-1b / LM-1c.

**채점판 before → after** (`python bench/score.py --only 1,2,3,4`, 두 측정 다 GPU 35~38% 점유 상태)

| # | 기준 | before | after |
|---|---|---|---|
| 1 | 작은 글씨 | 통과 · CER 0.0% | 통과 · CER 0.0% |
| 2 | 오번역 없음 | **실패** · 오탐 0/62 · **확정 오역 1건** · 대리지표 3건 · 병음 게이트누수 2건 · 인식실패 화면 1개 | **통과** · 오탐 **0/68** · **확정 오역 0건** · 대리지표 2건 · 병음 누수 **0** · 인식실패 화면 **0** |
| 3 | 다중 문맥 | 통과 · zh_pinyin **인식실패로 측정불가** · ui_en_dense 9건 | 통과 · zh_pinyin **0건**(측정 가능해짐) · ui_en_dense 8건 |
| 4 | 속도 | 통과 · p50 1.05s · p95 1.58s | 통과 · p50 1.08s · p95 1.58s |

### 1) PY-2 — 띄어쓰기된 병음을 못 알아봐 중국어 화면이 통째로 무너졌다
`zh_pinyin` 시료의 한자 5문단 중 **0개** 인식. Tesseract 는 `chi_sim` 으로 같은 이미지를
완벽히 읽으므로 순수 라우팅 문제였다. PY-1 판정이 **붙여쓴 병음**만 상정하고 "7자 이상 토큰
하나"를 요구해서, 음절을 띄어 쓴 실제 학습앱 병음(`ni hao huan ying …`)에서 전부 False가 났다.
→ ZH-1 진입 근거 (a)가 사라지고, 병음이 영어로 읽히니 근거 (b)("0줄")도 안 서서 `chi_sim`
패스가 영원히 안 돌았다. 그리고 그 병음 줄이 그대로 번역돼 자막이 됐다
(`"ni hao, huan ying shi yong zhe ge ying yong"` → `"니 하오, 후안 이 쯔 옌"`).

판정 (c) 추가 — 최장 토큰 대신 **분해되는 토큰의 개수**(`PINYIN_MIN_SPACED_TOKENS=6`,
비율 0.90; 중국어 확정 화면은 4). 거짓양성 스윕(부정 코퍼스 55,156조각 = 영 54,993 + 불 91 +
독 72): 재현 5/16 → **14/16**, 거짓양성 92 → **93**(신규 1건뿐), **불어·독어 거짓양성 증가 0**
(`café` / `Saint-Exupéry` 회귀 없음).

### 2) ZH-2 — 진입 근거를 병음 판정과 무관한 신호로 하나 더 (구조 보강)
(a)(병음)와 (b)(0줄)는 사실상 같은 실패를 공유한다 — 순수 한자 화면에 영어 라벨 한 줄만
있어도 둘 다 깨진다. 근거 (c) `unread_ink_ratio()` 추가: Otsu 이진화 잉크 중 인식된 라인
박스가 못 덮은 비율이 0.6 이상이면 진입. 실측(진입 판정 지점 그대로) — 영어 화면 **0.000~0.197**,
한국어 화면 0.000(LC-1b kor 재스캔이 먼저 읽는다), 중국어 화면 **1.000**.
축소 후 판정은 **되돌렸다**(다크 UI가 0.041→0.659로 튐, errors.md ZH-2 함정).
진입해서 한자를 못 찾으면 `entered_zh` 플래그로 쿨다운에 떨어뜨려 헛 왕복을 막는다.

### 3) RP-1 — 확대 배율을 실측 시간 → 1차 패스 글자 수로 (재현성)
같은 입력이 실행마다 다르게 읽혔다. 원인은 UB-1 배율이 `√(900ms / 1차 패스 실측 시간)` 이라
PC 부하가 인식 결과로 새는 것. `√(2700 / 1차 패스 글자 수)` 로 바꿨다 — 이미지에서
결정론적으로 나오는 값이고 "비용은 면적이 아니라 글자 양"이라는 UB-1 근거도 그대로다.
**검증**: 전 시료 5회(2~5회차는 CPU 부하 4프로세스) 결과 해시 **완전 일치**.
구 규칙 대조군은 같은 조건에서 **5회 전부 불일치**(서로 다른 결과 5종, 갈린 시료 5개).
예산 2700은 구 규칙과 같은 지점에 선다(1080p 1.43배 = 구 1.43배, 4K 빽빽은 자동 생략).

### 4) DG-1b — 극소 라인 conf 임계 70 → 60
정확히 읽힌 11px 라벨(`4 spaces`, 정답 완전 일치, conf 64)이 게이트에 숨어 오탐 1/68 = 1.5%.
실측(시료 전체 `line_px<16` 36개 + 6~12px 합성 캡션): px<12 구간에서 잡힌 쓰레기는 conf 46.8
하나, 탈락한 정상은 64.0 하나. 60은 그 사이이고 46.8을 여전히 막는다.

### 안 고친 것
- **LM-1c** 메뉴바/툴바 8건 합침. 임계 1.5면 전 시료 경계 오류 0건이 되지만, 같은 임계가
  **양끝맞춤 본문을 산산조각 낸다**(단어 간격 1.2배에서 1줄 → 10줄). 메뉴 항목 간격이 1.83배라
  가로 간격 신호 하나로는 원리적으로 못 가른다 → 레이아웃 분석(B2) 몫. 실측표는 errors.md LM-1c.
- 3토막 병음(`ni hao ma`)은 여전히 못 잡는다(개수를 3으로 낮추면 영어 거짓양성 10배).

**검증**: `py_compile` OK · `test_cocktail.py` **56/57**(신규 가드 2건 추가, 실패 1건은 동시 작업 중인
`cocktail_ui.py` 쪽 신규 테스트의 cp949 인코딩 문제로 본 변경과 무관) · `pre_release_check
--allow-build-artifacts` OK · `cocktail.py --self-test` OK.
**신규 회귀 가드**: 띄어쓰기 병음 판정 · 영/불/독 오판 금지(6문장 추가) ·
`test_chinese_screen_is_read_and_result_is_reproducible`(한자 인식 + 같은 입력 5회 동일) ·
`test_upscale_factor_ignores_wall_clock`(1차 패스를 0.4초 늦춰도 배율 불변).

## 2026-09-05 (3) — 합격기준 5(UI) 통과 (UI-1 · UI-2 · UI-3 · IC-1)

`cocktail_ui.py` 단독 수정. 자세한 증상/원인은 `errors.md` UI-1 / UI-2 / UI-3 / IC-1.

**채점판 before → after** (`python bench/score.py --only 5`)

| | 창 밖 위젯 | 글자 잘림 | 트레이 | 단축키 | 첫 실행 | 판정 |
|---|---|---|---|---|---|---|
| before | 14 | 16 | 10 | 2 | 표시 | **실패** |
| after | **0** | **0** | 10 | 2 | 표시 | **통과** |

### 1) UI-1 — 고정 픽셀 배치를 레이아웃으로 (기준 5 실패의 직접 원인)
위젯이 전부 `setGeometry` 절대 좌표였다. 좌표는 상수인데 **글자 크기는 상수가 아니다** —
폰트 DPI 168이면 폰트만 1.75배가 되고 폭은 그대로라 버튼 글자가 잘렸고, 창을 640x400으로 줄이면
버튼 3개(480x220이면 4개)가 창 밖으로 나갔다.
`QVBoxLayout`(가로 한 줄 + 상태줄 + 안내줄)으로 교체 — **배치·기능·순서는 그대로**다.
레이아웃이 자식 `sizeHint`를 최소 폭으로 삼으므로 창 최소 크기가 자동으로 생기고
(실측 772x104 / 폰트 DPI 168에서 1146x128) 그 아래로는 창이 줄지 않는다.
버튼은 `_FitButton`(문구가 바뀌는 `setText` 한 곳에서 최소 폭 = 글자폭+20 갱신, 글꼴 변경도 추적),
길어질 수 있는 라벨은 `QSizePolicy.Ignored` + 말줄임(`_elide`, 전체 문구는 툴팁).

### 2) UI-2 — DP-1을 깨지 않고 고DPI 보정 (글꼴을 픽셀 크기로)
DP-1(`QT_ENABLE_HIGHDPI_SCALING=0`)은 캡처·오버레이 좌표를 물리 픽셀로 통일하려고 켠 것이라
그대로 둔다. **인수인계의 "175%에서 설정창만 60% 크기"는 실측으로 정정됐다** — 실제 증상은
"상자는 그대로, **글자만** 1.75배가 되어 잘림"이다. 스케일링을 꺼도 글자는 창이 놓인 모니터
DPI로 그려지는데, 레이아웃이 쓰는 `fontMetrics()`는 주 모니터 DPI(96) 그대로라 둘이 어긋난다.
`_apply_screen_font_scale()`이 설정창 글꼴을 **포인트가 아니라 픽셀 크기**로 준다
(px = 9pt × 그 화면 ldpi / 72 → 96dpi 12px, 168dpi 21px). 픽셀 글꼴은 어디서나 그 픽셀이라
그리기와 측정이 일치하고 레이아웃이 창을 그만큼 키운다. 좌표는 하나도 안 건드린다.
실측: 보조 모니터 이동 시 창 872→**1241px**, 이탈·잘림 0, 주 모니터 복귀 시 12px로 원복.
좌표에 배율을 곱하는 방식은 글자를 못 키우므로, 항등 함수로 남아 있던 `self.px`는 삭제했다.
함정 2개(errors.md UI-2): 부모 `setFont`는 자식 `font()`에 안 온다 / `layout.invalidate()` 없이
`activate()`만 하면 옛 글꼴로 캐시한 크기가 나온다.

### 3) UI-3 — 첫 실행 안내 한 줄
첫 실행 분기는 있었지만 창이 뜨는 것으로 끝이라 "무엇을 눌러야 자막이 나오는지"가 없었다.
상태줄 아래 안내 라벨(첫 실행에만): "처음이신가요? 번역할 창을 띄운 뒤 [번역 시작]을 누르면
자막이 뜹니다." 툴팁에 3단계 전체 + 트레이 상주 안내. 번역을 시작하거나 창을 닫으면 사라진다.
마법사·모달은 만들지 않았다.

### 4) IC-1 — 트레이 아이콘 (빨간 사각형 placeholder 교체)
`make_app_icon()`이 QPainter로 그린다. **외부 이미지를 받지 않는다**(ICONS.md §6 라이선스 오염 회피
→ 이 아이콘의 저작권은 이 프로젝트에 있다). 도안은 ICONS.md §2 그대로 —
따뜻한 오렌지 둥근 사각형 + 원문/번역 글자 쌍 `A` → `가`. 트레이·창 아이콘 공용, 16/32/64/256.
16px에 직접 그리면 서브픽셀 색 번짐이 남아, 8배로 그려 축소 + `NoSubpixelAntialias`로 잡았다
(렌더 PNG 8배 확대 육안 확인).

### 검증
`py_compile` OK / `test_cocktail.py` **55/55**(신규 가드 1건 추가, 기존 54건 유지) /
`pre_release_check.py --allow-build-artifacts` OK / `cocktail.py --self-test` OK /
`bench/score.py` 기준 5 **통과**(1·3·4 통과 유지). 창 4크기 × 기본/DPI168 + 주/보조(175%) 모니터
렌더 PNG 육안 확인.
신규 가드: `test_settings_window_keeps_widgets_inside_at_any_size`.
부수 수정: 테스트 헬퍼가 만들던 `QCoreApplication` → `QApplication`
(뒤에 오는 창 테스트가 위젯을 만드는 순간 프로세스가 **에러 없이** 죽었다).

## 2026-09-05 (2) — 캡처 계층 2건 (FD-1 · CP-1)

실측으로 확정된 캡처 계층 결함 2건. 자세한 증상/원인은 `errors.md` FD-1 / CP-1.

### 1) FD-1 — 프레임 변화 감지가 본문 크기 글자를 전부 놓쳤다 (기능 결함)
1920x1080에서 자막 한 줄만 바뀌면 최대 셀 변화량이 28px 글자 기준 **3**(임계 8)이라 프레임이
통째로 스킵됐다. 60px 이상만 통과 = 본문·자막·채팅·툴팁은 **실시간 번역이 구조적으로 불가**.
원인은 `cheap_hash`의 **16x16 고정 격자** — 1080p에서 셀 하나가 120x67px라 한 줄이 평균에
희석되고, 화면이 클수록 더 희석된다(4K에선 20px 글자도 6).
격자를 **셀 픽셀 수 고정**(`FRAME_HASH_CELL_PX=16`, 변 = clamp(max(w,h)/16, 16, 256))으로 바꿔
해상도 불변으로 만들었다. F-1의 rect 헤더 계약(16바이트 int32, 불일치 시 `1<<30`)은 그대로.

| 한 줄 교체 최소 변화량 | 12px | 20px | 28px | 34px | 44px | 60px |
|---|---|---|---|---|---|---|
| before (16x16, 1080p) | 2 | 5 | **3** | **3** | **7** | 16 |
| after (g120, 1080p) | 27 | 79 | 64 | 80 | 109 | 111 |
| after (g240, 4K) | 27 | 75 | 63 | 80 | 109 | 118 |

셀이 작아지면서 새로 잡히던 텍스트 커서 깜빡임은 `FRAME_DIFF_MIN_ROW_CELLS=2`로 걸렀다
(캐럿은 어느 행에서도 셀 1개만 변한다 — 실측 항상 1, 12px 한 줄 교체는 8셀).
해시 비용은 5.7 ms → **1.7 ms**로 오히려 줄었다.

### 2) CP-1 — 캡처가 항상 가상 데스크톱 전체를 찍었다 (유휴 CPU 주범)
`ImageGrab.grab(all_screens=True)`는 bbox 크기와 무관하게 5760x2160 전체를 BitBlt 한 뒤 자른다
(1920x1080도 600x300도 똑같이 ~115 ms). 캡처 영역이 **주 모니터 안에 완전히 들어갈 때만**
`all_screens=False`로 내려간다(`needs_all_screens`). 주 모니터는 정의상 가상 좌표계 원점이라
bbox가 그대로 맞고, 크기는 PIL이 보는 것과 같은 `GetSystemMetrics(SM_CXSCREEN/SM_CYSCREEN)`로 읽는다.

| | 캡처 | 해시 | 유휴 폴 1회 | 유휴 CPU(1코어) |
|---|---|---|---|---|
| before | 116 ms | 5.7 ms | 372 ms | 24.2% |
| after | 32 ms | 1.7 ms | 284 ms | **8.0%** |

W-4/W-5 회귀 방지를 위해 ①주 모니터 안 ②보조 모니터 안 ③두 모니터에 걸침 ④음수 좌표
4케이스를 실기 캡처 이미지로 확인했다(검은 픽셀 없음, 분기/기준 차이 < 시간차 잡음).
레버 `CAPTURE_PRIMARY_FAST_PATH=False`로 즉시 구동작 복귀.

### 검증
`test_cocktail.py` 51 → **54건 통과**(신규 가드 3: 작은 글자 검출 / 정적 스킵 + 캐럿 / 캡처 분기),
`pre_release_check.py --allow-build-artifacts`, `cocktail.py --self-test` 전부 통과.

## 2026-09-05 — 강건성 1건 + 품질 2건 (TE-1/VR-1 · UB-1 · LC-1/LC-1b)

실측으로 확정된 결함 3건을 고쳤다. 자세한 증상/원인은 `errors.md` 참조.

### 1) TE-1 — 번역 그룹 하나가 실패하면 프레임 전체가 사라졌다
영어+중국어 혼합 화면에서 오버레이가 **한 줄도** 안 떴다. 중국어 1줄의 m2m100 폴백 로드가
CUDA OOM을 냈고, 그 예외가 `batch_translate`의 언어 그룹 루프 밖으로 올라가 이미 번역된
영어 50줄까지 같이 버려진 것. 그룹 격리(`try/except` + 실패한 그룹만 `None`), 실패는
`LAST_GROUP_ERRORS` → 상태줄에 표시(무음 실패 금지, CO-2). 덤으로 VR-1(로드 시점 fp16 —
`.to(cuda)` 후 `.half()` 순서가 VRAM 피크를 2배로 만들던 것)과 m2m100 비동기 로드
(첫 프레임 4000 ms 정지 → **95 ms**).

### 2) UB-1 — 전체화면에서 확대 재OCR이 한 번도 발동하지 않았다
구 예산 `확대 후 픽셀 ≤ 4M`은 1080p의 2배(8.3M)를 항상 거부한다. 문서 단어의 58%를 못 읽고
있었다. 예산을 **1차 패스 실측 시간**(`OCR_UPSCALE_TIME_BUDGET_MS = 900`)으로 바꾸고
배율을 `min(2, √(예산/1차시간))`으로 낮춰 잡는다. 픽셀 수는 비용의 대리지표가 아니다 —
같은 8.3M이라도 여백 많은 4K는 436 ms, 글자 빽빽한 4K는 4787 ms.

### 3) LC-1 / LC-1b — 목표 언어와 같은 언어팩을 상시 켜 OCR 2.5배
목표가 한국어일 때 인식된 한국어 줄은 SK-1이 어차피 전부 버린다. 그래서 목표 언어와 같은
언어팩만 뺀다(한→외 번역에서는 `kor` 유지). 단 그냥 빼면 한글이 라틴으로 오독돼(`"인벤토리"`
→ `"Be] Ast"`, conf 49) 엉뚱한 자막이 6줄 떴다 — 프레임 conf가 낮으면 `kor`을 붙여 재스캔하고
conf가 오를 때만 채택, 확인되면 12프레임 sticky(LC-1b).

### 실측 (같은 프로세스에서 구/신 파이프라인을 rep 단위로 번갈아 측정, 중앙값 5회)

| 시나리오 | OCR ms | 번역 ms | 합계 ms | 단어 인식률 / 유사도 |
|---|---|---|---|---|
| 영어 문서 1080p | 786 → **1105** | 348 → 1877 | 1134 → 2983 | 42% / 0.541 → **100% / 0.993** |
| 영어 UI 51줄 | 1088 → **687** | 1474 → 1461 | 2563 → **2148** | 84% / 0.617 (동일) |
| 중국어 sticky | 1342 → **861** | 1939 → 1758 | 3281 → **2619** | 108% / 0.869 (동일) |
| 1440p 문서(여백) | 1242 → 1138 | 1821 → 1809 | 3063 → 2947 | 83% / 0.840 → **100% / 0.995** |
| 4K 문서(여백) | 1294 → 1273 | 1772 → 1609 | 3066 → 2881 | 83% / 0.840 → **100% / 0.993** |
| 4K 문서(글자 빽빽) | 4831 → **1492** | 3616 → 3364 | 8448 → **4856** | — |
| 한국어 화면(sticky) | 573 → 552 | 0 → 0 | 573 → 552 | 오표시 6줄 → **0줄** |

영어 문서의 번역 시간이 늘어난 건 **읽은 글자가 120단어 → 287단어로 늘었기 때문**이다
(못 읽던 58%를 이제 번역한다). 측정 중 다른 앱이 GPU를 81% 쓰고 있어 번역 ms는 편차가 크다 —
그래서 구/신을 같은 프로세스에서 번갈아 쟀다.

### 검증
`py_compile` / `test_cocktail.py` **51건 통과**(신규 4건) / `pre_release_check.py --allow-build-artifacts` OK /
`cocktail.py --self-test` OK.

## 2026-08-11 — 중국어 화면 번역 전면 붕괴 (PY-1 / ZH-1 / M-9)

중국어 학습 앱(한 문장이 **병음 / 한자 / 앱 자체 한국어 번역** 3층으로 그려지는 화면)에서
오버레이가 뜻이 안 통하는 문장을 뱉었다.

| | |
|---|---|
| 우리 출력 | `작은 미는 강 옆에 서서 소에 붉은 공을 들고, 차가운 왈로 가득 찬 쓰었다.` |
| 우리 출력 | `4주간을 보며, 그녀는 "첫 번째 물 공은 누구에게 쏟아져야 하는가?"` |
| 정답 | `샤오메이는 강가에 서서 손에 빨간 국자를 들고 있었습니다…` |
| 정답 | `모두들 웃으며 뛰어다니며 서로에게 물을 뿌렸습니다…` |

### 진짜 원인 — 한자는 애초에 읽히지 않았다

fixture 를 파이프라인에 그대로 태워 단계별로 찍었다. 통념("`kor` traineddata 가 한자를
읽어 준다")이 **틀렸다**.

| 화면 | `eng+kor` (auto 기본) | `eng+kor+chi_sim` |
|---|---|---|
| 순수 중국어(합성) | **0줄**, CJK 0자 | 4줄, CJK 55자 |
| 학습앱 스샷 | 15줄인데 **CJK 0자** (전부 병음/한국어) | 16줄, 한자 정상 |

살아남은 건 한자 **위**의 병음 줄뿐이었고, SL-1이 라틴을 전부 `en`으로 확정하므로
병음이 영어로 취급돼 en→ko 모델에 들어갔다. `xiǎoměi`→`작은 미`, `sìzhōu`→`4주간`이 그 흔적.
번역 모델이 나빠서가 아니라 **OCR 언어 라우팅**이 1차 원인이었다.

뒤이어 두 개가 더 있었다.
- Tesseract 는 한자를 글자마다 별개 word 로 준다. `" ".join` → `大 家 笑 着` →
  `"큰 집은 웃고"` (공백 없으면 `"모두가 웃고"`).
- zh→ko **전용 opus-mt 는 존재하지 않는다**(Hub 확인). m2m100_418M 폴백은 주어를 흘렸다 —
  `小美站在河边…` → `**은** 강 옆에 서서…`.

### 수정

| 코드 | 파일 | 내용 |
|---|---|---|
| **PY-1** | `cocktail_ocr.py` | `is_pinyin_line()` / `drop_pinyin_lines()` — 병음 줄을 번역 입력에서 제외. Tesseract·Paddle 공통 길목 1곳 |
| **ZH-1** | `cocktail_ocr.py` `cocktail_engine.py` | 근거가 있을 때만 `chi_sim` 재OCR(`_zh_rescan_lang` + `_tess_scan`), 8프레임 sticky + 빈화면 20프레임 쿨다운. `join_ocr_words()` 로 CJK 경계 공백 제거 |
| **M-9** | `cocktail_translate.py` | `PIVOT_ROUTES` — zh→ko 를 zh→en→ko 피벗으로. 첫 로드는 백그라운드(워커 안 얼림) |

병음 판정은 두 갈래 OR — (a) 강성 성조부호(마크론/캐런/ǖǘǚǜ), (b) 병음 음절 분해율 +
최장 분해 토큰 길이. 어큐트/그레이브는 프랑스어·스페인어에 흔해 **일부러 뺐다**.
임계는 영어 6909줄 스윕으로 확정: `비율 0.8 + 최장 7` → 거짓양성 6줄(0.09%).

### zh→ko 후보 실측 (중국어 12문장 → 한국어, chrF)

| 안 | chrF | 배치 | 추가 용량 | 라이선스 |
|---|---|---|---|---|
| m2m100_418M 직행 (기존 폴백) | 24.5 | 582 ms | 0 | MIT |
| `shun89/opus-mt-zh-ko` (3rd party) | 28.3 | 294 ms | 310 MB | apache-2.0 (업로더 신고) |
| **zh→en → en→ko 피벗** | **32.0** | 497 ms | 312 MB | **CC-BY-4.0** ← 채택 |
| `opus-mt-tc-bible-big-mul-mul` | 4.5 | 434 ms | 1982 MB | apache-2.0 |

피벗을 고른 이유: 품질 1위 + 라이선스가 이미 배포 중인 en-ko 와 같은 CC-BY-4.0(Helsinki 공식).
shun89 는 베이스가 CC-BY-4.0 opus-mt 라 apache-2.0 신고를 믿을 수 없고 학습 데이터도
문서화돼 있지 않다. mul-mul 은 성경 코퍼스만 학습해 출력이 고문체 한국어(때때로 과라니어)다.
NLLB 는 CC-BY-NC(비상업)라 애초에 후보가 아니다.

### 속도 — 영어 회귀 없음

| fixture | before | after |
|---|---|---|
| 영어 문서 doc_55 (10문단) | 3183 ms | 3207 ms (+0.8%) |
| 영어 UI live_primary (52줄) | 3974 ms | 4027 ms (+1.3%) |
| 중국어 학습앱(합성) | 987 ms | 1659 ms |
| 중국어 학습앱(실스샷) | 2098 ms | 3357 ms |

늘어나는 건 중국어 화면뿐이다(chi_sim 패스 + 피벗 1회). 영어 화면은 진입 근거가 없어
2차 패스를 아예 안 탄다.

### 결과

`大家笑着，跑着，互相泼水。` → `모두 웃으며 뛰어가 서로에게 물을 던져`
`小美看看四周，她在想…` → `메이는 주위를 둘러보며 "누가 먼저야?"라고 생각했습니다.`
병음 줄에는 자막 박스가 더 이상 그려지지 않는다(오버레이 `grab()` 렌더로 육안 확인).

회귀 가드 5개 추가(`test_cocktail.py`): 병음 줄 제외 / 영어 본문 오판 금지 /
`drop_pinyin_lines` 병렬 conf 정합 / CJK 공백 없는 결합 / zh→ko 가 폴백이 아닌 피벗 라우팅.

---

## 2026-08-11 — 문단이 통째로 사라지던 버그 (CO-2)

영어 PDF를 번역하면 **특정 문단만** 자막이 안 떴다. 위아래 문단은 멀쩡한데 삽화 바로 아래
3줄짜리 문단 두 개가 원문 그대로 남았다. 숨김은 무음 실패라 로그에도 아무것도 안 남았다.

### 무엇이 원인이 아니었나 (재현 하네스로 배제)

그 PDF 4쪽을 화면과 같은 배율로 렌더해 **실제 파이프라인 그대로** 태우고 단계별 생존을 찍었다.

| 단계 | 대상 문단 | 수치 |
|---|---|---|
| OCR 라인 | 살아 있음 | conf 94.8, 라인 높이 17~24px |
| PM-1 문단 병합 | 3줄 → 1문단, 문장 온전 | 346자 / 204자 |
| SE-1 문장 분할 번역 | 문장 5개 전부 번역됨 | 번역/원문 길이비 0.49, 0.37 |
| 번역 길이 상한 | 잘린 적 없음 | `max_new_tokens` 실측 24~99 (상한 256) |
| **DG-1 표시 게이트** | **여기서 숨겨짐** | `carryover(...)` — 잔류 단어 **1개** |

번역이 상한에 잘려 길이가 붕괴한다는 가설(len-collapse)은 **틀렸다**. 길이비 0.37~0.49는
붕괴 임계 0.15의 3배다. LM-1 라인 분할, conf 필터, 상태 dedup도 전부 무관했다.

### 진짜 원인 — 한 줄용 임계를 문단에 쓴 것

CO-1(미번역 잔류 검사)의 허용치 0은 게임 툴팁 **한 줄**(30~60자) 기준으로 잡은 값이다.
그 뒤 PM-1(문단 병합)이 게이트의 판정 단위를 200~350자 문단으로 키웠다. 그래서:

> 40단어 문단에서 OCR이 **단어 하나**를 틀린다 → 번역기가 모르는 그 단어를 그대로 복사한다
> → 나머지 39단어가 완벽해도 **문단 전체가 사라진다**.

1글자 오독을 주입한 실측 8종 중 3종이 문단 전멸을 유발했다.

| 주입 | 번역 결과 | 판정 |
|---|---|---|
| `constrictor` → `consttictor` | 한국어 46자 중 영단어 1개 잔류 | 346자 문단 **전멸** |
| `elephant` → `elephanl` | 〃 | 346자 문단 **전멸** |
| `disheartened` → `disheattened` | 〃 | 204자 문단 **전멸** |
| `digesting`/`magnificent`/`painter`/`career`/`explained` 오독 | 잔류 0 | 정상 표시 |

삽화 옆 문단이 걸린 건 우연이 아니다. 그 문단들이 하필 `constrictor`, `elephant`,
`disheartened` 같은 **길고 희귀한 단어**를 담고 있어 OCR이 틀릴 확률도, 번역기가 모를 확률도 높다.

### 수정

허용치를 번역문 길이에 비례시켰다.

```
허용치 = max(DISPLAY_MAX_CARRYOVER_WORDS, 번역문 어절 수 × 0.12)
        → 9어절 미만 0 / 9~16 1 / 17~25 2 / …
```

짧은 줄은 **지금과 똑같이 무관용**이라 CO-1이 막던 "반쯤 번역된 헛소리"는 그대로 걸린다
(실측 fixture: 10어절 / 잔류 2 > 허용 1 → 숨김 유지). 긴 문단만 오독 한둘을 견딘다.

임계를 낮추는 게 아니라 **단위에 맞추는** 수정이다. 정상 문장을 숨기는 건 게이트의 실패다 —
깨진 자막 한 줄보다 사라진 문단 하나가 더 나쁘다.

### 관측성 — 다시는 같은 진단을 반복하지 않게

숨길 때마다 사유와 원문 머리를 찍는다.

```
[INFO] DG-1 숨김: carryover(constitctor) "My drawing was not a picture of a hat. It was a picture"
[INFO] DG-1 숨김: tiny-line(4px)/low-conf(40) "ae"
```

`display_gate_reject`(공개 진입점)가 `_display_gate_reason`(규칙 본체)을 감싸는 구조라
OCR 경로와 UIA 경로가 자동으로 같은 로그를 탄다. 정적 화면이 같은 줄을 초당 몇 번 재판정해도
(사유 종류, 원문 머리) 키로 최근 64건을 기억해 **라인당 1회**만 찍는다.

### 검증

- `test_cocktail.py` 41→43건 (빠른 테스트 1건 + `--full` 실동작 1건 추가) 전건 통과
- 비율을 0.0으로 되돌리는 뮤테이션에서 새 가드가 실패하는 것 확인
- `pre_release_check.py --allow-build-artifacts` OK / `cocktail.py --self-test` OK
- **속도 영향 0**: 40줄 게이트 1회 2.87ms → 2.87ms(노이즈 내). 허용치 계산은 잔류 단어가 있을 때만 돈다

## 2026-08-02 (8차 · 배포 준비 패스) — "받아서 더블클릭하면 되는 제품"으로 마감

7차까지 코드는 끝났다. 이번 패스는 **코드를 만지지 않고**(`Cocktail완성본.py` 무접촉)
그 코드가 사용자 손에서 실제로 켜지는지, 문서가 사실을 말하는지를 맞췄다.

### 1) 실행 — `run_cocktail.bat`이 앱을 못 띄우던 문제 (RB-1)

더블클릭하면 실패했다. 원인은 배치가 그냥 `python`을 부른 것. PATH에 먼저 잡히는 건
**시스템 Python 3.13이고 거기엔 PySide6/torch/transformers가 하나도 없다.** 앱은 멀쩡한데
첫 관문에서 막힌 상태였다.

하드코딩(`C:\Users\...\envs\cd\python.exe`)으로 때우면 이 PC에서만 돌아간다. 그래서 탐색으로 바꿨다:

| 순위 | 후보 | 판별 |
|---|---|---|
| 1 | `COCKTAIL_PYTHON` 환경변수 | 실행만 되면 **무조건 존중** (사용자가 지정한 것) |
| 2 | `%USERPROFILE%\anaconda3\|miniconda3\envs\cd\python.exe` | `import PySide6` |
| 3 | `py -3.11` (Windows 런처) | `import PySide6` |
| 4 | PATH의 `python` | `import PySide6` |

판별이 "python이 실행되는가"가 아니라 **`import PySide6`가 되는가**인 게 핵심이다.
전자로는 바로 이 버그(3.13이 실행은 됨)를 못 거른다. 전부 실패하면 조치 2개(의존성 설치 /
`COCKTAIL_PYTHON` 지정)를 찍고 `pause`.

**배치 파일은 ASCII 전용으로 썼다.** cmd는 `.bat`을 콘솔 OEM 코드페이지(한국어 Windows = CP949)로
읽는데 파일이 UTF-8이면 한글 주석·경로·메시지가 깨지면서 파싱까지 무너진다(오늘 `debug_run.bat`에서
실증). 한국어 안내는 `ENVIRONMENT.md`가 맡는다. `pause` 프롬프트는 OS가 한국어로 찍어준다.

곁다리로 `diagnose_environment.py`가 앱과 같은 규칙으로 `./tessdata`를 `TESSDATA_PREFIX`에 걸도록
고쳤다. 전에는 앱은 kor을 보는데 진단만 `eng, osd`라고 보고해서 사용자를 헷갈리게 했다.

**실측 검증 (3분기)**

| 조건 | 결과 |
|---|---|
| 정상 | `anaconda3\envs\cd\python.exe` 선택 → 진단 OK → 앱 기동 (프로세스 확인 후 정리) |
| `COCKTAIL_PYTHON` = 패키지 없는 3.13 | 지정 존중 → 진단이 결손 모듈 5개 보고 → 에러 + pause |
| 후보 전멸 (`PATH`/`USERPROFILE` 무력화) | 안내 메시지 + pause |

### 2) 사어(死語) 의존 정리 (DR-1)

SL-1에서 코드에서 걷어낸 `langdetect`가 4곳에 살아 있었다 — 안 쓰는 패키지를 설치 요구하고,
PyInstaller가 번들에 넣고, 진단은 없으면 "필수 모듈 누락"이라 겁을 준다.

| 파일 | 조치 |
|---|---|
| `requirements.txt` | 의존성 제거, 왜 없는지 묘비 주석만 남김 |
| `build.spec` | `collect_all` 패키지 목록에서 제거 |
| `diagnose_environment.py` | `REQUIRED_MODULES`에서 제거 |
| `pre_release_check.py` | 필수 의존성 목록에서 제거 |
| `LICENSE-3RDPARTY.md` §6 | "제3자 컴포넌트 없음 — `unicodedata`만" |

`smoke_tests.py` / `pre_release_check.py`의 **잔재 검사(import가 돌아오면 실패)는 그대로 뒀다.**
그건 되살아나는 것을 막는 가드다.

`osd.traineddata`도 같은 결이다. OSD script 판별이 SL-1에서 퇴역했으므로
`download_tessdata.py` 다운로드 목록과 `installer.iss` 컴포넌트/Source에서 뺐다.
**파일 자체는 지우지 않았다** (이미 설치된 환경에 무해).

### 3) "100% 로컬" 문구 사실화 (DR-1)

문서는 "100% 로컬 처리. 어떤 데이터도 외부로 나가지 않습니다"라고 했다. 사실은 다르다 —
최초 1회 번역 모델을 Hugging Face에서 받고, Tesseract는 별도 설치다.

| 전 | 후 |
|---|---|
| 100% 로컬 처리. 어떤 데이터도 외부로 나가지 않습니다. | 인식·번역 전 과정을 내 PC에서 처리합니다. 화면 내용이나 번역 결과는 외부로 전송되지 않습니다. 단, **최초 1회 번역 모델 다운로드(인터넷 필요)**와 **Tesseract OCR 엔진 별도 설치**가 필요합니다. |
| 1. 100% 로컬 처리 — 클라우드 전송 없음 | 1. 로컬 추론 — 화면 내용의 클라우드 전송 없음 (외부 통신은 최초 1회 모델 다운로드뿐) |

적용: `README.md`, `README.en.md`, `VISION_AND_ROADMAP.md`(§4/§5/랜딩 문구),
`HANDOFF.md`, `.github/workflows/release.yml` 릴리즈 노트.
README/README.en에는 "최초 1회 준비" 절(의존성 → Tesseract → 모델 다운로드)을 새로 넣었다.
지키는 약속은 그대로다 — **화면 내용은 어떤 경우에도 네트워크로 나가지 않는다.**

### 4) `ENVIRONMENT.md` 현행화

이 PC에 **존재하지 않는 `aigugbi` 환경**을 기준으로 쓰여 있었다(문서 표류). 실측으로 교체:

```
Python 3.11.15 (C:\Users\ui2030\anaconda3\envs\cd\python.exe), pip 26.0.1
PySide6 6.11.1 / transformers 5.8.1 / torch 2.11.0+cu126 (CUDA 사용 가능)
tokenizers 0.22.2 / sentencepiece 0.2.1 / sacremoses 0.1.1 / pytesseract 0.3.13
Pillow 12.1.1 / opencv-python 4.13.0.92 / numpy 1.26.0 / keyboard 0.13.5 / scipy 1.17.1
Tesseract 5.5.0  C:\tesseract\tesseract.exe
```

추가: 환경변수 표 6종(`COCKTAIL_PYTHON`, `TESSERACT_CMD`, `COCKTAIL_OCR_LANGS_AUTO`,
`COCKTAIL_PADDLE_*`, `COCKTAIL_SHOW_SETTINGS_ON_START`), `run_cocktail.bat` 탐색 순서,
`debug_run.bat` 진단 절차(타임스탬프 로그 → `debug_log.txt`), 시스템 Python 함정 경고.

### 5) `nllb_ct2/` (600MB) — 삭제하지 않음, 상태만 정확히 기록

사용자 승인 대기 중이라 **그대로 뒀다.** 대신 배포 오염 여부를 확인했다:

- `build.spec`은 `datas`에 명시한 것만 담는다 → `dist/`에 안 들어감
- `installer.iss`는 `dist\Cocktail\*`만 담는다 → 인스톨러에 안 들어감
- `.claudeignore`에 등록돼 있음

즉 **빌드 산출물은 깨끗하다.** 다만 소스 배포/저장소 공개 시엔 비상용(CC-BY-NC) 자산이 그대로
따라가므로, `README.md` 라이선스 절 / `BUILD.md` 배포 전 체크리스트 / `ENVIRONMENT.md` §8에
"배포 전 삭제 필요"를 명시했다.

### 6) 문서 정합 (틀린 것만)

| 파일 | 정정 |
|---|---|
| `HANDOFF.md` | 현재 상태 5/4 → 8/2, smoke 21건/pre-release 14건 → **35/17**, 코드 2,650줄 → **3,310줄**, 1~7차 패스 요약, `nllb_ct2` 대기 항목 |
| `ARCHITECTURE.md` | OCR 라우팅 표에 남아 있던 `_smart_lang_for_image`/`_detect_script_via_osd`/`_script_cache` → 실제 함수(`_src_hint`/`_ocr_langs`/`perform_ocr_with_boxes`) + SL-1 퇴역 표기 |
| `OCR_TEST_PLAN.md` | 기대 감지값이 SL-1 이전 그대로였다. `Gracias→es`/`Hvala→hr`/`Merci→fr`/`Danke→de` → auto에서는 **전부 `en`**(다른 언어는 source 콤보 명시). region 샘플의 M-7 "언어 island" 표현도 LM-1 라인 분리 기준으로 교체 |
| `README.md` | 현재 상태 `Phase 0` → `Phase 0~6 기능 완료`, DG-1/DP-1/멀티모니터/트레이·캐시·민감창 항목 추가, 설치·실행 절 재구성 |
| `README.en.md` | Display gate(DG-1) / physical-pixel(DP-1) / monitor hot-plug 추가, Install/Run 절에 최초 실행 안내 |
| `TECH_STACK.md` | `osd.traineddata` — 인스톨러 동봉 목록에서도 제외됐음을 명시 |

`업데이트.md`와 `errors.md` 과거 항목은 **날짜가 박힌 이력**이라 손대지 않았다(사후 수정 금지).

### 검증

| 항목 | 결과 |
|---|---|
| `smoke_tests.py` | **35/35 OK** |
| `pre_release_check.py --allow-build-artifacts` | **17/17 OK** |
| `Cocktail완성본.py --self-test` | OK (exit 0) |
| `run_cocktail.bat` 실행 | **앱 기동 확인** (3분기 실측, 잔류 프로세스 0) |

전부 `C:\Users\ui2030\anaconda3\envs\cd\python.exe`로 실행.

---

## 2026-08-02 (7차 · 코드 완성 패스) — 품질·안정·보안 결함 9건 일괄 마감

1~6차 수정은 전부 유지. **좌표 규약(DP-1 물리 픽셀 통일)**과 **언어 규약(SL-1 결정론 라우팅)**은
손대지 않았다. 번역 파이프라인(M-8 sepvoc 인코딩, `batch_translate` 라우팅/배치/캐시)도 무접촉.
개별 패치를 모은 게 아니라 "표시 → 안정 → 보안" 세 축을 한 번에 정리한 완성 패스다.

### 1) 표시 품질 — 안 보여줄 줄을 확실히 안 보여준다

| 결함 | 증상 | 수정 |
|---|---|---|
| **DG-1** | 초소형 캡션에서 "마우 냄비 thang에"류 그럴싸한 쓰레기가 뜸 | 표시 직전 최종 게이트 신설 |
| **LM-1** | 특정 라인이 화면 1/3을 덮는 거대 폰트로 뭉침 | word 높이 이상치 제거 + 두 단 분할 + 폰트 중앙값 |
| **FT-1** | 한국어 자막 자간/굵기가 환경마다 들쭉날쭉 | 한글 우선 폰트 스택 |
| **TM-1** | 다른 항상-위 창이 오버레이를 덮음 | `raise_()` + 2초 주기 topmost 재적용 |
| **CL-1** | 자막이 모니터 경계를 넘거나 빈 영역에 그려짐 | 클립을 "박스가 속한 모니터"로 축소 |

**DG-1 — "확신 없으면 숨긴다".** NZ-1은 번역 **이전**의 원문만 본다. OCR이 말이 되는 오독을 하면
그대로 통과한다. 그래서 번역이 끝난 뒤 마지막으로 한 번 더 묻는다:

```python
display_gate_reject(src, tgt, tgt_lang, line_px, conf)   # 사유 문자열이면 그 줄은 안 그림
#  1) line_px < 12  그리고  conf < 70          → 극소 라인 + 저 confidence
#  2) 번역문의 목표 언어 스크립트 비율 < 0.3    → 한국어 목표인데 한글 없음 = 번역 실패
#  3) 원문 20자 이상인데 번역문이 0.15배 미만   → 길이 붕괴
```

규칙 1은 **두 조건 동시 성립**일 때만이라 큰 글씨는 conf가 낮아도 살아남는다. 숨긴 줄 수는
상태줄에 `· 숨김 N줄(DG-1)`로 보인다. 라인 튜플 계약 `(text, bbox, font_size)`은 Paddle/UIA와
공용이라 바꾸지 않고, 라인별 conf만 병렬 리스트로 전달한다.

**LM-1 — 원인을 렌더 이미지로 재현해 확정했다.** 두 가지가 겹쳐 있었다.

| 재현 조건 | 수정 전 | 수정 후 |
|---|---|---|
| 세로 기둥(도형)이 `"|"` height 81px로 잡혀 본문 14px 라인에 합류 | `font=81`, 박스 높이 81 (오버레이 초기 폰트 **73px**) | `font=11`, 높이 14 |
| 60px 장식 "W"가 본문에 붙음 | `font=43` | `font=11` |
| 좌우 두 단이 같은 y (간격 255px) | 한 줄로 병합, bbox 폭 **645px** | 2줄로 분리 (167 / 144) |

`font_size = max(word height)`가 라인 안 이상치 하나에 그대로 끌려간 것이 첫째,
`--psm 6`이 좌우 단을 한 라인으로 묶은 것이 둘째다. `_line_segments()`가 높이 중앙값의 2.5배를
넘는 word를 빼고, 중앙값 높이의 3배를 넘는 가로 간격에서 라인을 쪼갠다. 폰트는 `max` → **중앙값**.

**TM-1 한계(명시)**: 전체화면 **독점** 앱 위에는 원리상 어떤 창도 못 올라간다. 게임/영상은
테두리 없는 창 모드로 바꿔야 자막이 보인다.

### 2) 안정성 — 워커 스레드가 지켜야 할 선

- **RC-1 (race)**: 워커의 "비울 게 있나" 판정 5곳이 **메인 스레드 큐드 슬롯**이 갱신하는
  `translated_text`를 읽고 있었다. 슬롯이 늦으면 "비울 게 없다"고 오판해 자막이 영구 잔상으로 남는다.
  워커가 마지막으로 emit한 상태를 자기 로컬(`_last_emitted`)로 추적하고, emit 경로를
  `_emit_state` 하나로 모았다(`_clear_overlay_if_any` / `_emit_state_if_changed`가 그 위에 얹힘).
- **GT-1 (Qt 규약)**: 워커가 `QCursor.pos()` / `screenAt` / `QApplication.screens()`를 직접 불렀다.
  화면 rect는 메인 스레드가 `region.screen_rects`에 캐시하고 워커는 읽기만, 커서는 win32
  `GetCursorPos`(물리 픽셀 → **DP-1 규약과 정합**)로 교체했다.
- **MS-1 (핫플러그)**: `screenAdded` / `screenRemoved` / `geometryChanged` → 오버레이 재적합 +
  화면 rect 캐시 갱신 + 열려 있는 RegionEditor 재적합. 연결은 `CocktailAppController`가 소유.

### 3) 보안 — SV-1: 민감 창이 OCR로 새던 구멍

`_uia_collect_and_translate`가 민감 창에서 **빈 리스트**를 돌려주면 호출부는 그것을 "UIA 실패"로
읽고 아래 OCR 경로로 계속 진행했다. 즉 민감 창(패스워드/결제/은행)이 캡처·OCR·번역되고
DPAPI 영구 캐시에까지 남았다. S-1 가드는 `MODE_BOX`에서 꺼지므로 **MODE_BOX + UIA 명시 선택**
조합에는 두 번째 방어선도 없었다.

```python
UIA_SKIP_FRAME   # 민감 창 = 이 프레임 전체 스킵 (OCR 폴백 금지)
None             # UIA 불가/추출 실패 → OCR 폴백 OK
[...]            # 정상
```

호출부는 sentinel이면 `ImageGrab.grab`에 **도달하기 전에** `continue` 한다. 가드 테스트가
"sentinel 처리가 캡처보다 앞선 줄인지"까지 소스 순서로 검사한다.

### 검증

| 항목 | 결과 |
|---|---|
| `py_compile` | OK |
| `smoke_tests.py` | **35/35 OK** (31 → 35, skip 0건) |
| `pre_release_check.py --allow-build-artifacts` | 17/17 OK |
| `Cocktail완성본.py --self-test` | OK |

신규 가드 4건: `test_dg1_display_gate`(정상 5종 통과 + 불량 6종 차단),
`test_lm1_line_merge_and_font_outlier`(합성 + **실제 렌더 이미지 E2E**),
`test_rc1_emit_state_uses_worker_local`(메인 슬롯 지연 race 재현),
`test_sv1_uia_sensitive_blocks_ocr_fallback`(sentinel 실동작 + 순서 검사).
기존 3건(`test_w4_w5` / `test_sc1` / `test_rt1`)은 **동작은 그대로 두고** 새 배선(캐시·로컬 상태)에
맞춰 단언만 갱신했다.

### 새 튜닝 레버

| 상수 | 기본값 | 올리면 / 내리면 |
|---|---|---|
| `DISPLAY_GATE_ENABLED` | `True` | False = 게이트 전체 무효(디버그) |
| `DISPLAY_TINY_LINE_PX` | `12` | ↑ 더 많이 숨김 / ↓ 관대 |
| `DISPLAY_TINY_LINE_MIN_CONF` | `70.0` | ↑ 더 많이 숨김 / ↓ 관대 |
| `DISPLAY_MIN_TGT_SCRIPT_RATIO` | `0.3` | ↑ 엄격(고유명사 많은 줄이 숨을 위험) |
| `DISPLAY_LEN_COLLAPSE_RATIO` | `0.15` | ↑ 엄격(짧은 정상 번역이 숨을 위험) |
| `OCR_WORD_HEIGHT_OUTLIER_RATIO` | `2.5` | ↓ 공격적(큰 제목 글자가 잘릴 위험) |
| `OCR_LINE_SPLIT_GAP_RATIO` | `3.0` | ↓ 더 잘게 분할(넓은 자간 문서에서 문장 끊김) |
| `OVERLAY_FONT_FAMILIES` | 맑은 고딕 우선 | 폰트 교체 시 여기만 |
| `OVERLAY_TOPMOST_INTERVAL_MS` | `2000` | 0 = 주기 재적용 OFF |

### 남은 것 (의도적 보류)

- **GUI 육안 검증 불가** — 이 환경은 캡처가 차단돼 있다. 자막 겹침/폰트 인상/topmost 복귀 같은
  "보기"의 최종 판정은 사용자 실기가 필요하다.
- **전체화면 독점 앱** 위 오버레이는 원리상 불가(TM-1 한계).
- **PaddleOCR 경로의 라인 conf** 는 전달하지 않는다 — DG-1 규칙 1이 그 경로에서만 무효.
  필요해지면 `rec_scores`를 같은 병렬 리스트에 채우면 된다(1줄).

---

## 2026-08-02 (6차) — SL-1 auto 모드 언어 라우팅 결정론화 (오판 장치 퇴역)

1~5차 수정(W/F/SC/LD/RT/NZ/DP/OU/RP 계열)은 전부 유지. 번역 파이프라인(M-8 sepvoc 인코딩,
`batch_translate` 라우팅/배치/캐시)은 무접촉. 이번 변경은 **기능 추가가 아니라 범위 축소**다.

**계기**: 5차의 OS-1 신뢰도 게이트(`OSD_SCRIPT_MIN_CONF = 1.0`)를 영어 PDF가 **conf 3.95의
"Arabic"** 판정으로 정면 통과했다. 같은 날 langdetect도 같은 화면에서 cy/hu/de/pt/so를 뱉었다.
게이트를 조여도 오판은 계속 나온다 — 문제는 임계값이 아니라 **추측한다는 설계 자체**였다.

### 무엇이 바뀌나 (auto 모드)

| 항목 | 이전 | 이후 (SL-1) |
|---|---|---|
| OCR 언어 | OSD로 script 분류 → 해당 언어팩 | **`eng+kor` 고정** (`OCR_LANGS_AUTO_DEFAULT`) |
| 라인 언어 | script fast-path + langdetect 추측 | **스크립트 확정만**, 라틴·불명은 전부 `en` |
| 사후 보정 | M-7 region 투표 + LG-1 지배 언어 스냅 | 없음 (보정할 추측이 사라짐) |
| m2m100 폴백 | 오판 라인마다 새어 들어감 | src 명시 또는 스크립트 확정 비라틴에서만 |

```python
# 라인 언어 판정 전부
def identify_language(text, fallback="en"):
    script, _ = detect_script(text)
    return _SCRIPT_TO_LANG.get(script, fallback)   # 라틴/불명 → en
```

**삭제**: `_detect_script_via_osd`, `_smart_lang_for_image`, `_script_cache`, `_SCRIPT_TO_TESS`,
`OSD_SCRIPT_MIN_CONF`, `langdetect` import, `_LANGDETECT_TO_ISO`, `_LATIN_SHORT_HINTS`,
`_latin_hint`, `_safe_detect`, `_language_confidence`, `_cluster_text_regions`,
`DOMINANT_LANG_*`. (3274 → 2949줄, -325)

### 트레이드오프 — 다른 언어는 콤보로 명시

자동 다국어 감지 야망을 버리고 실사용 시나리오(영어 화면 → 한국어)에 결정론으로 최적화했다.
프랑스어/독일어 화면은 auto가 더 이상 알아서 인식하지 않는다. **source 콤보에서 언어를 고르면**
OCR 언어와 번역 라우팅이 예전처럼 그 언어로 동작한다. 환경변수 탈출구도 유지:
`COCKTAIL_OCR_LANGS_AUTO=eng+jpn`.

### 속도

- OSD 호출 제거: 실측 **111ms/회** (캐시 miss마다, 창 전환·TTL 30초 만료 시)
- langdetect 제거: 15라인 프레임 실측 **~130ms** (라인당 ~9ms)
- 오판이 부르던 m2m100 폴백 배치(**~5초/건**)와 모델 로드가 통째로 소멸 — 실사용 체감은 여기서 나온다

### 가드

`smoke_tests.py` 32 → 31건. `test_m7_region_smoothing` / `test_lg1_dominant_language_snap` /
`test_os1_osd_confidence_gate_and_eng_fallback` 퇴역, `test_sl1_deterministic_line_languages`(영/한/일
혼합 → `[en, ko, ja]`만, 라틴은 무조건 en, src 명시는 그대로) + `test_sl1_no_guessing_engines_in_auto_path`
(AST로 langdetect import 잔재 + OSD 문자열 잔재 검사) 추가. pre-release는
`check_smartocr_wiring` → `check_sl1_auto_language_routing`으로 교체.

검증: `py_compile` OK / smoke 31/31 OK / `pre_release_check.py --allow-build-artifacts` OK /
`--self-test` OK.

## 2026-08-02 (5차) — OS-1 OSD 오판 격리 + RP-1 반복 붕괴 방지

1~4차 수정(W-4/W-5/W-6/UX-3/F-1/F-2/SC-1/LD-1/RT-1/RT-2/NZ-1/DP-1/OU-1/LG-1)은 전부 유지.
번역 파이프라인 구조(M-8 sepvoc 인코딩, `batch_translate` 라우팅/배치/캐시)는 무접촉 —
이번엔 `generate` 파라미터 1개 추가만 예외로 허용했다.

**증상**: 영어 PDF인데 상태줄이 `[ar]`, 오버레이에 "엘리자베스 엘리자베스 …"(×12),
"억원, 억원 …"(×16) 같은 반복 쓰레기. 창을 바꿔도 한동안 안 돌아옴.

**오염 사슬 (실기 재현 + 검증 실험으로 확정)**

| 단계 | 무슨 일 | 이번 차단 |
|---|---|---|
| 1 | OSD가 영어 문서를 `Arabic`으로 오판 (신뢰도 검사 없음) | **OS-1 게이트** |
| 2 | `_script_cache[hwnd]`(TTL 30s)에 고착 | 게이트 통과분만 캐시 |
| 3 | 전 프레임 `ara`로 OCR → `5 لاع هنالاعا- )5310` 쓰레기 | **OS-1 eng 상시 합류** |
| 4 | 전 줄 `ar` 판정 → m2m100 `ar→ko` → 같은 구 무한 반복 | **RP-1** |
| — | 성공 경로 무로그라 추적 불가 | **OS-1 로그 1줄** |

### OS-1 — OSD 신뢰도 게이트 + eng 상시 합류

```python
# Cocktail완성본.py — 상수
OSD_SCRIPT_MIN_CONF = 1.0   # ↑ 엄격(폴백 잦음) / ↓ 관대(오판 통과)

# _detect_script_via_osd — (script, conf) 반환으로 변경
conf = float(osd.get("script_conf"))        # 키 없음/파싱 실패 → 기존 동작(신뢰) 유지
if conf < OSD_SCRIPT_MIN_CONF:
    return None, conf                       # → 기존 eng+kor 폴백

# _smart_lang_for_image — 핵심 안전망
candidate_langs = list(self._SCRIPT_TO_TESS.get(script, ["eng", "kor"]))
if "eng" not in candidate_langs:
    candidate_langs.append("eng")           # Arabic → "ara+eng"
print(f"[INFO] OSD script={script} conf={conf} → langs={lang_str}")
```

- 게이트만으로는 "고확신 오판"을 못 막는다. **eng 상시 합류**가 진짜 안전망 —
  라틴 글리프는 eng 패턴이 이기므로 OSD가 틀려도 영어 단어는 영어로 읽힌다.
- OSD는 hwnd 캐시 miss에서만 도니 로그 1줄은 핫 패스 print가 아니다(P-6 무관).

### RP-1 — 반복 붕괴 방지 (`no_repeat_ngram_size`)

`OpusMtTranslator` / `M2M100Translator`의 `generate(...)`에 각각 한 줄:

```python
no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,  # 상수 = 4
```

**3이 아니라 4인 이유**: 주기 p의 반복은 길이 `n+p`만 넘으면 n-gram이 중복되므로 n=4도 모든 반복 루프를 잡는다.
반대로 n=3은 정상 한국어의 합법 반복을 깬다 — 실모델 실측
`"이 회사는 올해 새로운 정책을 발표했다. 이 회사는 모든 사무실에 적용했다."`에서 `"이 회사는"` 중복이 n=4에선 보존된다.
억제를 더 세게 하려면 3으로 내리면 된다.

### 검증 (개발 PC, `envs/cd`)

- `py_compile` OK (앱 + smoke)
- `smoke_tests.py` **32/32 OK** (30 → 32, 신규 `test_os1_osd_confidence_gate_and_eng_fallback` /
  `test_rp1_no_repeat_ngram_guard`, 둘 다 FAST_TESTS 등록). 기존 `test_m8` 실모델 통과 = 품질 회귀 1차 가드.
- `pre_release_check.py --allow-build-artifacts` OK
- `Cocktail완성본.py --self-test` OK (Tesseract + CUDA)
- 실사용 회귀(영어 PDF에서 `[en]` 유지 + 반복 쓰레기 소멸)는 사용자 PC 확인 필요.

---

## 2026-08-02 (4차) — OU-1 작은 글씨 적응 업스케일 + LG-1 지배 언어 스냅

1~3차 수정(W-4/W-5/W-6/UX-3/F-1/F-2/SC-1/LD-1/RT-1/RT-2/NZ-1/DP-1)은 전부 유지.
번역 파이프라인(M-8 sepvoc 인코딩, `batch_translate` 라우팅/배치) 무접촉 — LG-1은 입력 언어 리스트만 정돈한다.

**증상**: "작은 글씨는 번역이 아예 안 나온다" + 영어 한 화면인데 상태줄 언어가 `[de, en, pt, so]`로 파편화.

**실측 근거 (`doc_55.png` 724x954, 개발 PC)**

| | 라인 수 | 높이 중앙값 | 캡션 블록 | 감지 언어 |
|---|---|---|---|---|
| 수정 전 | 33 | 13px | **전멸** — `Sir mail pot fying aw` 류 잔재 1줄만 | `{en:32, af:1}` |
| OU-1만 | 36 | 12px | 복구 — `aur mail pilot flying`(conf 79.2), `the 1920s.`(92.7), 본문 캡션(95.7) | `{en:34, no:1, de:1}` |
| OU-1 + LG-1 | 36 | 12px | 복구 | **`{en:36}`** |

수정 전 캡션 라인은 conf 60대 + 철자 붕괴 → langdetect가 cy/hu/de/pt/so로 오판 → m2m100 폴백 경로로
새면서 배치가 쪼개지고(5초대) 번역이 쓰레기거나 NZ-1 필터에 잘려 화면에 안 떴다.

### OU-1 — 적응 업스케일 OCR

E-3의 업스케일은 **이미지 `min(h,w) < 400`** 이라는 크기 조건이라, 화면 전체를 캡처하는 실사용에서는
발동하지 않았다. "작은 이미지"는 커버했지만 "큰 이미지 속 작은 글자(7~9px 캡션)"는 사각지대.

```python
# Cocktail완성본.py — 상수
OCR_UPSCALE_TRIGGER_PX = 18       # 1차 OCR 라인 높이 중앙값 임계
OCR_UPSCALE_FACTOR = 2            # 확대 배율

# perform_ocr_with_boxes
lines = self._tess_lines(gray, lang_str, scale)
if scale == 1 and lines:
    median_px = float(np.median([font_size for _, _, font_size in lines]))
    if median_px < OCR_UPSCALE_TRIGGER_PX:
        self._ocr_scale = OCR_UPSCALE_FACTOR
        lines = self._tess_lines(gray, lang_str, OCR_UPSCALE_FACTOR)
```

- 전처리(grayscale→resize→OTSU→morph)+OCR을 `_tess_lines(gray, lang_str, scale)`로 묶어 1차/재OCR가 같은 경로.
- 좌표 환원은 기존 `_lines_from_tess(data, scale)`의 `// scale`을 그대로 재사용 — **새 좌표 경로 없음**.
- 확대 보간은 `cv2.INTER_LANCZOS4`로 통일(기존 소형 이미지 경로 포함). 실측상 텍스트 철자가 CUBIC보다 정확.
- **재OCR 비용은 트리거가 걸린 작은 글씨 화면에서만 발생**한다. 본문이 18px 이상인 일반 화면은 1차 OCR 한 번.
- 평균이 아니라 **중앙값**: 제목 한 줄이 커도 판정이 끌려가지 않게.
- PaddleOCR 경로 무접촉(자체 detector가 스케일을 다룸).

### LG-1 — 페이지 지배 언어 스냅

M-7 region 투표는 region **안**만 보므로, 같은 region의 이웃까지 같이 오판되면 못 잡는다.
region 투표 뒤에 페이지 전역 스냅을 얹는다.

```python
DOMINANT_LANG_MIN_SHARE = 0.6     # 지배 언어 최소 가중 비중
DOMINANT_LANG_SNAP_MAX_CONF = 2   # 이 이하 확신의 라틴 라인만 스냅
```

1. 확신(`_language_confidence` ≥ 2) **라틴 스크립트 라인만**의 가중 다수결로 지배 언어를 구한다.
2. 지배 비중 ≥ 0.6 **이고 지배 언어가 라틴 계열**일 때만,
3. 확신 ≤ 2인 **라틴 스크립트 라인**만 지배 언어로 스냅.

- 한/일/중/러/아랍 등 **비라틴 라인은 절대 스냅하지 않는다** — 진짜 다국어 화면 보존(UX-2).
  지배 언어가 비라틴이면 스냅 자체를 건너뛴다(영어 줄이 한국어로 끌려가지 않게).
- **투표 모집단 라틴 한정(자체 검수 보강)**: 초안은 비라틴까지 투표에 넣어, 영문 3줄 + 파편 2줄 +
  한국어 1줄(확신 4)이면 en 비중이 `6/12 = 0.5`로 희석 → 임계 미달로 **스냅이 통째로 꺼졌다**.
  한국어 UI가 섞이는 실사용 조건에서 상시 무력화되는 구조라 모집단을 라틴 라인으로 한정했다.
  비라틴은 스냅 대상이 아니므로 지배 판정에 거부권을 가져선 안 된다.
- 확신 3(라틴 짧은 힌트 적중: "Merci"/"Hvala")은 미접촉 → M-7 기존 동작 유지.
- 효과: 전 라인이 opus-mt 단일 경로로 묶여 배치 1회, 파편 줄의 번역 품질 회복.

### 검증

| 항목 | 결과 |
|---|---|
| `py_compile` (앱 + smoke_tests) | OK |
| `smoke_tests.py` | 30/30 OK (28건 + 신규 `test_ou1_adaptive_upscale`, `test_lg1_dominant_language_snap`) |
| `pre_release_check.py --allow-build-artifacts` | OK |
| `Cocktail완성본.py --self-test` | OK |

신규 가드는 외부 이미지에 의존하지 않는다 — PIL로 11px 텍스트를 합성 렌더해 트리거·좌표 환원을,
합성 OCR 라인 리스트로 언어 스냅을 검사한다. LG-1 가드에는 "고확신 한국어 라인이 섞여도
라틴 파편이 스냅된다"는 희석 회귀 케이스가 포함된다.

### 튜닝 레버

| 상수 | 초기값 | 올리면 | 내리면 |
|---|---|---|---|
| `OCR_UPSCALE_TRIGGER_PX` | 18 | 더 자주 확대(정확↑ 속도↓) | 작은 글씨를 놓침 |
| `OCR_UPSCALE_FACTOR` | 2 | 3 이상은 비용 대비 이득 미미 | 1이면 기능 무효 |
| `DOMINANT_LANG_MIN_SHARE` | 0.6 | 보수적(다국어 안전, 파편 잔존) | 공격적(수렴↑, 진짜 소수 언어 훼손 위험) |
| `DOMINANT_LANG_SNAP_MAX_CONF` | 2 | 3이면 라틴 힌트 적중까지 스냅(권장 안 함) | 1이면 긴 오판 라인을 못 잡음 |

---

## 2026-08-02 (3차) — DP-1: DPI 좌표계 물리 픽셀 통일 (주 모니터에 오버레이가 안 뜨던 근본 원인)

1·2차 수정(W-4/W-5/W-6/UX-3/F-1/F-2/SC-1/LD-1/RT-1/RT-2/NZ-1)은 전부 유지.
번역 파이프라인(M-8 sepvoc 인코딩, `batch_translate` 라우팅/배치) 무접촉.

**증상**: 100% 배율 주 모니터에는 오버레이 자막이 **전혀** 안 뜨고, 175% 배율 보조 4K에는
원문에서 어긋난 채 뜬다. 1·2차에서 캡처 영역을 아무리 정확히 계산해도 화면에 안 나오던 잔여 결함.

**실측 근거 (이 PC)**

| 모니터 | 물리 rect | Qt 논리 rect |
|---|---|---|
| 주 1920x1080 @100% | `(0, 0, 1920, 1080)` | 동일 |
| 보조 4K @175% | `(-3840, -535, 0, 1625)` | `(-3840, -535, -1646, 699)` |

Qt 논리 공간에 `-1646 ~ 0` 이라는, 어느 모니터에도 속하지 않는 **틈**이 생긴다.

**원인**: 좌표계 혼용. `GetWindowRect` / `ImageGrab` / OCR bbox는 물리 픽셀인데
Qt 지오메트리와 `QCursor.pos()`는 논리 좌표. 가상 데스크톱 전체를 덮는 단일 투명 창인
OverlayWindow가 보조 모니터의 1.75 배율로 스케일되면서, 주 모니터에 그려야 할 자막이
1.75배 우측으로 밀려 창 밖으로 나갔다.

**수정 (DP-1)**: Qt 논리 스케일링을 끄고 앱 전 좌표를 **물리 픽셀**로 통일.

```python
# Cocktail완성본.py 모듈 최상단 (import os 직후)
os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "0")
```

- 논리 == 물리(devicePixelRatio 1) → `GetWindowRect` · `ImageGrab` · OCR bbox · `paintEvent` ·
  `QCursor.pos()` · W-4 가상 데스크톱 클립 · SC-1 `screenAt` 판정이 전부 한 좌표계에서 만난다.
- `OverlayWindow._fit_to_screen` / `RegionEditor._fit_to_virtual_desktop`의 `screens()` 합집합은
  자동으로 물리 rect가 되어 **코드 변경 불필요**(읽고 확인, 주석만 추가).
- DPI awareness(퍼모니터 인식)는 Qt가 유지 → OS 블러 스트레칭 없음.
- `QApplication` 생성은 `CocktailAppController.__init__` 한 곳뿐(BG-2)이고 `main()`에서만 호출되므로,
  모듈 import 시점의 env 설정이 항상 앞선다.

**E-7과의 정합**: E-7(`_scale=1.0` 고정, devicePixelRatio 수동 곱셈 금지)은 그대로 유효.
새 규약이 그 위에 얹힌다 — "Qt 논리 스케일링 OFF, 전 좌표 = 물리 픽셀". 여전히 아무도 배율을 곱하지 않는다.

**의도된 트레이드오프**: 스케일링이 꺼져 설정 창(880x120)과 트레이 메뉴를 175% 모니터로 옮기면
물리 픽셀 그대로 작게 보인다. 100% 주 모니터에서는 기존과 동일. 오버레이 좌표 정합을
UI 크기보다 우선한 선택으로, errors.md DP-1에 한계로 기록.

- `smoke_tests.py`: `test_dp1_highdpi_scaling_disabled_before_qapp` 신규 + FAST_TESTS 등록 (27→28건).
  AST로 (a) 모듈 레벨 문장인지 (b) `QApplication(...)` 호출보다 앞선 줄인지 둘 다 검사.
- `errors.md`: DP-1 항목 신설 + E-7 후속 표기 + 5초 체크리스트 2줄 추가.
- `ARCHITECTURE.md` §1.5: 단위 규약 "물리 픽셀 통일 (`QT_ENABLE_HIGHDPI_SCALING=0`)" 명시.

검증 (`anaconda3/envs/cd` python):
- `python -m py_compile Cocktail완성본.py smoke_tests.py` ✅
- `python smoke_tests.py` → **28/28 OK** (기존 27건 무회귀)
- `python pre_release_check.py --allow-build-artifacts` → **OK** (17개 체크 전부 통과)
- `python "Cocktail완성본.py" --self-test` → **[SELFTEST] OK**
- 멀티 모니터 실기 육안 확인(주 모니터 자막 표시 / 원문 정렬 / 설정 창 크기)은 사용자 몫.

---

## 2026-08-02 (2차) — 실사용 성능/표시 결함 4건 (캡처 영역 / 모델 로드 / 재번역 폭주 / OCR 노이즈)

사용자 실기 로그(`debug_log.txt`) 기준 2차 수정. 1차(W-4/W-5/W-6/UX-3/F-1/F-2)는 유지하고,
번역 파이프라인 핵심(M-8 sepvoc 인코딩, `batch_translate` 라우팅/배치 구조)은 미변경.

- **SC-1 (오버레이가 모든 모니터에 꽉 참)**: `MODE_WINDOW`는 활성 창 rect를 그대로 캡처 영역으로 썼다.
  바탕화면(Progman/WorkerW)처럼 가상 데스크톱 전체를 덮는 창이 활성이면 전 모니터가 캡처 대상
  (W-5로 보조 모니터가 실제로 찍히면서 표면화). `update_for_mode()`에서 창 중심점이 속한
  **모니터 1개**(`screen_rect_at()` → `QGuiApplication.screenAt()`, 실패 시 `primaryScreen()`)로 교집합.
  `MODE_BOX`/`MODE_HOVER`, W-4 합집합 클립, W-5 `all_screens=True`는 그대로.
  레버: `CLAMP_WINDOW_TO_ACTIVE_SCREEN = True`.
- **LD-1 (모델 로드 69초)**: 로컬 HF 캐시가 있는데도 `from_pretrained`/`snapshot_download`가 매번
  Hub를 조회("unauthenticated requests" 경고 직후 정지). `_load_local_first()` 헬퍼로
  `local_files_only=True` 우선 → 실패 시에만 네트워크 재시도. 적용 4곳(opus-mt 토크나이저/모델,
  m2m100 토크나이저/모델, M-8 `snapshot_download`). 전역 `HF_HUB_OFFLINE` 강제는 첫 설치를
  망가뜨리므로 금지 — try/except 폴백으로만.
- **RT-1/RT-2 (1~3초마다 36~57줄 재번역)**: 정적 화면인데 캐시가 계속 miss 되던 원인은 OCR 공백 흔들림.
  캐시 키를 `_cache_key_text()`(연속 공백 collapse + strip)로 정규화(메모리 LRU + DPAPI 디스크 캐시 동일).
  표시용 원문/번역문과 모델 입력은 원본 유지. 추가로 `_emit_state_if_changed()` — 직전 상태와
  (원문, 번역문, `bbox // STATE_BBOX_ROUND_PX`) 기준 동일하면 `state_ready.emit` 생략(깜빡임/CPU 절감).
  OCR·UIA 두 경로 모두 이 헬퍼 경유.
- **NZ-1 (UI 부스러기까지 박스로 그림)**: 기존 필터는 word `conf < 30` 하나뿐. 라인 단위로
  평균 confidence(`OCR_LINE_MIN_CONF = 35.0`) + 문자 구성(`_ocr_line_is_noise()`: 최소 길이 2,
  영숫자/한글 비율 0.5, 글자 0개 라인 제외) 필터 추가. 렌더 이미지 E2E에서 정상 문장·"OK"·"Settings"는
  유지되고 `10:30`, `~ | * _`는 제거됨.

- `smoke_tests.py`: `test_sc1_active_screen_clamp`, `test_ld1_local_first_model_load`,
  `test_rt1_cache_key_and_state_dedup`, `test_nz1_ocr_noise_line_filter` 신규 + FAST_TESTS 등록 (23→27건).
- `errors.md`: SC-1 / LD-1 / RT-1 / RT-2 / NZ-1 항목 + 5초 체크리스트 5줄 추가.

검증 (`anaconda3/envs/cd` python):
- `python -m py_compile Cocktail완성본.py smoke_tests.py` ✅
- `python smoke_tests.py` → **27/27 OK** (기존 23건 무회귀, M-8 실모델 테스트 포함)
- `python pre_release_check.py --allow-build-artifacts` → **OK**
- `python "Cocktail완성본.py" --self-test` → **[SELFTEST] OK**
- HF 로드 실측(이 PC): snapshot local 0.00s / net 0.25s, tokenizer local 0.55s / net 0.99s —
  네트워크 왕복 자체를 제거하므로 사용자 환경의 수십 초 정지가 사라진다.
- 멀티 모니터 실기 육안 확인(오버레이 범위/박스 개수/깜빡임)은 사용자 몫.

## 2026-08-02 — 오버레이 표시 결함 6건 수정 (멀티 모니터 / 잔상 / frame-skip / 폰트 / 통과 토글)

감사에서 확정된 오버레이 표시 결함 6건을 `Cocktail완성본.py` 안에서 수정. 번역 파이프라인
(M-8 sepvoc 경로, `batch_translate`)은 미변경.

- **W-5 (보조 모니터 검은 캡처)**: `ImageGrab.grab(bbox=..., all_screens=True)`.
  PIL win32 경로가 가상 데스크톱 전체를 찍고 `offset`(가상 화면 원점) 기준으로 crop하므로
  절대 좌표 bbox가 그대로 맞음 — 설치된 `PIL/ImageGrab.py` 소스로 확인. 호출처는 워커 1곳뿐.
- **W-4 (보조 모니터 영구 정지)**: MODE_WINDOW 클립을 `QApplication.primaryScreen()` 단독에서
  `QApplication.screens()` 합집합(가상 데스크톱)으로 교체. 교집합 8px 미만이면 `continue` 전에
  `state_ready.emit([])`로 오버레이를 비움(B-15/S-1과 동일 패턴).
- **UX-3 (정지 후 자막 잔상)**: `BackgroundController.set_running(False)`에서 남은 상태가 있으면
  `state_ready.emit([])`. 버튼 / `Ctrl+Shift+T` / 트레이 모두 `toggle_running → set_running`
  단일 경로라 한 곳 수정으로 전 경로 커버.
- **F-1 (frame-skip 둔감)**: `cheap_hash(img, rect)`가 캡처 영역 좌표(int32×4)를 해시 헤더로 포함
  (영역이 바뀌면 무조건 재처리 → 창 이동 시 옛 좌표 자막 잔상 해소). 판정 기준을 평균 → 셀 단위
  **최대 변화량**으로 교체, 임계는 상수 `FRAME_DIFF_THRESHOLD = 8`(튜닝 레버).
  부수로 `cv2.COLOR_BGR2GRAY` → `COLOR_RGB2GRAY` 정정(입력은 PIL RGB, B-8 규약).
- **F-2 (폰트 1.2배 과대)**: `_fit_font`가 OCR **픽셀** 값을 `QFont(family, pointSize)`에 넘기던 문제 —
  `QFont("Arial")` + `setPixelSize(size)`로 교체. 초기 크기 계산·축소 루프 로직은 유지.
- **W-6 (통과 토글 첫 입력 무효)**: `_mouse_through` 초기값을 실제 상태(ON)와 일치시키고,
  `_sync_mouse_through(on)` 헬퍼로 플래그+트레이 체크를 한 곳에서 갱신.
  `set_capture_mode`가 `overlay.update_for_mode()`(통과 ON 강제) 직후 동기화하도록 연결.

- `smoke_tests.py`: `test_w4_w5_multi_monitor_capture`(AST 가드), `test_f1_frame_skip_local_change`
  (실동작 — 1셀 변화/영역 이동 감지) 신규 + FAST_TESTS 등록 (21→23건).
- `errors.md`: W-4 / W-5 / W-6 / UX-3 / F-1 / F-2 항목 + 5초 체크리스트 7줄 추가.

검증 (`anaconda3/envs/cd` python):
- `python -m py_compile Cocktail완성본.py smoke_tests.py` ✅
- `python smoke_tests.py` → **23/23 OK** (M-8 포함 기존 21건 무회귀)
- `python pre_release_check.py --allow-build-artifacts` → **OK**
- `python "Cocktail완성본.py" --self-test` → **[SELFTEST] OK**
- 멀티 모니터 실기 육안 확인(보조 모니터 캡처/자막 위치/통과 토글)은 사용자 몫.

## 2026-08-01 — M-8 en→ko 번역문 깨짐 수정 (opus-mt sepvoc 소스 인코딩)

en→ko 번역이 원문과 무관한 단어 나열로 나오던 문제를 근본 수정.
`Helsinki-NLP/opus-mt-tc-big-en-ko`는 소스/타깃 vocab이 분리된 sepvoc 모델인데 repo의
`tokenizer_config.json`이 `"separate_vocabs": false`로 잘못 적혀 있어, `AutoTokenizer`가
영어 입력을 타깃(한국어) vocab으로 인코딩 → 소스 조각 32,000개 중 25,330개가 `<unk>`.

- `OpusMtTranslator._load_source_spm()` 신설: 스냅샷의 `source.spm`을 sentencepiece로 직접 로드.
  sepvoc 판정은 설정 파일이 아니라 vocab 실측(샘플 20% 이상 미포함 → sepvoc). 공유 vocab 모델과
  로드 실패 시에는 기존 토크나이저 경로로 degrade.
- `OpusMtTranslator._encode_with_spm()` 신설: `sp.encode(text)[:255] + [eos]`, pad_token_id 패딩,
  attention_mask 직접 구성. 배치/`max_new_tokens`/LRU·디스크 캐시/m2m100 폴백은 변경 없음.
- `smoke_tests.py` `test_m8_opus_mt_source_encoding` 신규 + FAST_TESTS 등록 (20→21건).
  실제 모델로 한 문장 번역 후 한글 포함 + `<unk>` 부재 검사. 로컬 HF 캐시 없으면 자동 skip.
- `errors.md` M-8 항목 + 5초 체크리스트 1줄 추가.

검증 (`anaconda3/envs/cd` python):
- `python -m py_compile Cocktail완성본.py smoke_tests.py` ✅
- `python smoke_tests.py` → **21/21 OK** (신규 테스트 실제 실행, skip 아님)
- `python pre_release_check.py --allow-build-artifacts` → **OK**
- `python "Cocktail완성본.py" --self-test` → **[SELFTEST] OK**
- 이미지→OCR→번역 E2E 3문장 모두 정상 한국어
  (예: "Please save your work before closing the window." → "창을 닫기 전에 작업을 저장하십시오.")

## 2026-05-16 — BG-7 RegionEditor 추가 (MODE_BOX 영역 마우스 편집)

`Cocktail완성본.py`에 `RegionEditor(QWidget)` 신설. BG-5에서 `red_rect`가 settings window
위젯 좌표 → 화면 절대 좌표로 전환되며 사라졌던 "마우스로 박스를 편집" UX를 전용 위젯으로
복원. 분리된 5개 컴포넌트 책임 경계는 그대로 유지하고 SettingsWindow가 새 위젯을 lazy
소유하는 형태.

- 진입점: SettingsWindow 첫 행 "영역 편집..." 버튼 + 트레이 메뉴 동명 항목.
- 동작: 가상 데스크톱 전체를 덮는 frameless/translucent/always-on-top 위젯. 박스 윤곽 +
  8개 핸들. 본체 드래그=이동, 핸들 드래그=리사이즈, Enter=저장, Esc/우클릭=취소.
- 좌표계: 내부 박스 상태도 절대 좌표 `QRect`. paintEvent에서만 editor-내 좌표로 1회 변환
  (OverlayWindow와 동일 패턴, BG-5 결정과 일관).
- 안전:
  - B-10: `set_window_capture_affinity(exclude=True)`로 자기 캡처 피드백 차단.
  - SettingsWindow `_is_self_hwnd`가 RegionEditor winId도 자기 hwnd로 인식 → worker가
    편집 창을 캡처/번역하지 않음.
  - 민감 창 가드(S-1), capturable 가드(B-15)는 BackgroundController에 그대로. RegionEditor는
    geometry만 바꾸고 번역 파이프라인 미접근.
- UX: 저장 시 `region.red_rect` 절대 좌표 대입 + `set_capture_mode(MODE_BOX)` 자동 전환
  + `controller.reset_frame_hash()` 호출. 취소 시 변경 사항 폐기.
- 종료: SettingsWindow.closeEvent가 `region_editor.force_close()`로 정리.

문서/검증:
- `ARCHITECTURE.md` BG-7 섹션 추가, 좌표계/비결정 사항 갱신.
- `smoke_tests.py` `test_bg7_region_editor` 신규 + FAST_TESTS에 등록 (총 19→20건).
- `pre_release_check.py` `check_bg7_region_editor` 신규 + 게이트에 등록 (총 14→15건).
- `errors.md` 5초 체크리스트에 BG-7 항목 + 변경 이력 누적.

검증 결과 (사용자가 `pip install -r requirements.txt` 승인 후 실행):
- `python -m py_compile Cocktail완성본.py smoke_tests.py pre_release_check.py` ✅
- `python smoke_tests.py` → **20/20 OK** (`test_bg7_region_editor` 신규 포함, fast tests만)
- `python pre_release_check.py --allow-build-artifacts` → **15/15 OK** (`check_bg7_region_editor` 신규 포함)
- `python "Cocktail완성본.py" --self-test` → **[SELFTEST] OK** (env: Tesseract + CUDA, PaddleOCR 옵션 미설치)

실 GUI 검증 (드래그/리사이즈/저장/Esc 취소, MODE_BOX 자동 전환, RegionEditor B-10 가드)은
사용자 PC에서 직접 확인 필요. 코드 컴파일/구조/안전 토큰/AST 회귀는 모두 통과.

## 2026-05-04 — Phase 0 분리 사이클 종결 (BG-1 ~ BG-6)

5개 클래스 책임 정착 (자세한 내역은 `업데이트.md` / `errors.md` / `ARCHITECTURE.md`):

- **BG-1** Tray-first 백그라운드 — 설정창은 최초 1회만 표시.
- **BG-2** `CocktailAppController` — Qt 앱 생명주기 일원화.
- **BG-3** `CaptureRegionController` — 캡처 모드/영역 상태 분리.
- **BG-4** `BackgroundController` — 워커/엔진 시그널/OCR·UIA 라우팅/모델 lifecycle/캐시 타이머 분리.
  `RuntimeOptions(@dataclass(frozen=True))`로 콤보 immutable 스냅샷.
- **BG-5** `red_rect` 화면 절대 좌표 통일 — settings 창 이동/숨김 의존 제거.
- **BG-6** `TransparentWindow` → `SettingsWindow` 리네이밍 + region.owner 의존을 `is_self_hwnd` 콜백으로 통합.

산출물: `ARCHITECTURE.md` 신규, smoke 21건 / pre-release 14건. 빌드 786MB / 6372 files 유지.

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

## 2026-05-04 — SmartOCR (2026 트렌드 적용)

### UX-2. SmartOCR — M³ Multiplex 영감 5-Layer 아키텍처

사용자 통찰: "AI 시대인데 주먹구구로 합집합 던지면 안 되지. 자동으로 어떤 언어인지 파악하고 딱딱 꺼내 써야지."

박사 2026 트렌드 조사:
- M³ TextSpotter / Multiplex (script classifier + script-specific heads)
- Diacritic-driven lightweight 언어 식별
- End-to-end Vision-LLM (LightOnOCR-2-1B — 무거워 실시간 부적합)

박사 합성 — Cocktail SmartOCR 5 Layer:
1. UIA 자동 라우팅 (활성 창 + 기본 ocr_combo → UIA 우선)
2. 화면 가시 영역 클립 (primary screen intersect)
3. OSD script 분류 (`pytesseract.image_to_osd`, 30~50ms)
4. script → 단일/소수 언어 OCR (`_SCRIPT_TO_TESS` 매핑)
5. hwnd 캐시 (같은 활성 창은 OSD 건너뛰기)

예측 효과: OCR 27초 → ~2초.
추가 의존성: 0 (Tesseract OSD 활용).

박사 자체 리뷰 보강:
- `_script_cache` 인스턴스 변수로 (mutable 클래스 변수 안티 패턴 회피)
- UIA 자동 라우팅은 사용자 명시 선택 존중 (`Tesseract`/`PaddleOCRv5`면 UIA 우선 X)

### 권한 설정 강화 (사용자 요청)

`settings.local.json`에 자주 쓰는 명령 자동 허용 추가:
- `Bash(python *)`, `Bash(python -c *)`, `Bash(python -m *)`
- `Bash(pyinstaller *)`, `Bash(python -m PyInstaller *)`
- `Read`, `Glob`, `Grep`, `Skill(*)`, `WebSearch(*)`
- 기타 진단 도구

박사 작업 중 권한 프롬프트 최소화 → 사용자가 자리 비워도 진행 가능.

## 2026-05-04 — 사용자 GUI 검증 피드백 사이클

### B-15. B-10 자기 캡처 피드백 자동 가드

- 계기: 사용자가 "스샷OK"로 풀고 스크린샷 찍은 뒤 잊은 상태에서 번역 시작 → 화면 깨짐.
- 워커 진입부에 `self.capturable=True` 가드 추가 — OCR/번역 즉시 일시 중지.
- `_capturable_paused` 상태 플래그 + 사용자에게 빨간 경고 라벨로 안내.
- 사용자가 다시 "스샷차단" 토글하면 자동 재개.
- E-9 운용 규칙(사용자 책임)을 코드 자동화로 격상.

### UX-1. 비전 V-1 운영 모드로 기본값 정렬

사용자 피드백:
- "굳이 이런 창 없어도 돼" — 빨간 박스가 비전 진화 후 거의 안 쓰임.
- "속도가 느리네" — OCR 27초 (다국어 합집합 모드).
- "뒤죽박죽 나오기도" — 카탈로니아/웨일스/네덜란드까지 OCR 시도 → 노이즈.

수정:
- 기본 캡처 모드 `MODE_BOX` → `MODE_WINDOW` (활성 창).
- `_MODE_LABELS` 순서 재정렬 (활성 창 → hover → 영역 박스).
- `_ocr_langs()` auto: 합집합 → eng+kor 두 개. `COCKTAIL_OCR_LANGS_AUTO` 환경변수로 오버라이드 가능.
- OverlayWindow 빨간 박스 — `MODE_BOX`일 때만 그림.
- init에서 mode_combo ↔ capture_mode 동기화 (박사 자체 리뷰 보강).

예측 효과: OCR 27초 → ~3-4초 (~8배), 노이즈 제거.



### CR-1. 크리티컬 리뷰 수정

- `translation_thread` 생성/시작 코드가 `_on_cache_save_tick()` 안으로 들어간 치명적 들여쓰기 결함 수정.
- `_on_cache_save_tick()`은 이제 주기적 `PERSIST_CACHE.save()`만 담당.
- `스샷OK/스샷차단` 토글이 ControlWindow와 OverlayWindow 둘 다 같은 capture affinity로 변경하도록 수정.
- 활성 창/민감 창 자기 hwnd 가드에 OverlayWindow hwnd도 포함.
- UIA 경로도 `assign_region_languages()`를 적용해 OCR/Paddle 경로와 같은 M-7 언어 안정화 정책을 사용.

### ST-1. 다음 step 전 회귀 smoke test

- `smoke_tests.py` 추가.
- 빠른 기본 검사는 CR-1, OverlayWindow capture affinity, hwnd guard, UIA M-7, region smoothing, PersistentCache hash key를 확인.
- `--paddle` 옵션으로 PaddleOCR 실제 추론 smoke test를 별도 실행 가능.
- README.md에 `python smoke_tests.py`를 다음 개발 단계 전 검증 명령으로 추가.

### A-2. Phase 6 보강

- Windows 자동 실행 명령 생성 시 `.py` 실행 형태는 `pythonw.exe`를 우선 사용해 콘솔 창을 피하도록 수정.
- 주기적 `PERSIST_CACHE.save()`를 메인 스레드에서 백그라운드 daemon thread로 이동.
- `PersistentCache`에 `_mem_lock`을 추가해 save/get/put의 `_mem` 접근 경쟁을 완화.
- save 중 새 put이 들어오면 `_dirty`를 보존하도록 snapshot 비교를 추가.
- smoke test에 자동 실행/pythonw, 백그라운드 캐시 save 검사를 추가.

### R-1. 릴리즈 워크플로우 보강

- `build.spec`에 README, LICENSE, LICENSE-3RDPARTY, BUILD, CODE_SIGNING, UPDATE_LOG, TECH_STACK, VISION_AND_ROADMAP 문서 포함.
- `.github/workflows/release.yml`에서 Tesseract OCR을 설치한 뒤 진단/smoke/pre-release check 실행.
- 코드 서명 optional step을 zip 생성보다 앞으로 이동해 릴리즈 zip에 서명된 exe가 들어가도록 수정.
- GitHub Actions에서 Inno Setup을 설치하고 `installer.iss`로 `Cocktail-Setup-*.exe`를 자동 빌드.
- `pre_release_check.py` 추가. 필수 파일, 버전, workflow 순서, 라이선스/문서 링크, 생성 디렉터리 여부를 검사.
- `smoke_tests.py` 실행 후 생기는 `__pycache__`는 pre-release check에서 자동 정리하도록 보정.
- 로컬 `pyinstaller`가 다른 Python 3.12 환경을 사용해 의존성이 누락된 빌드를 만든 것을 확인.
  빌드 명령을 `python -m PyInstaller --noconfirm build.spec`로 고정.

### R-2. 배포 용량/패키지 자기검증 보강

- 기본 PyInstaller 빌드는 `COCKTAIL_BUNDLE_PADDLE=0`으로 PaddleOCR/PaddlePaddle을 제외.
- PaddleOCR 포함 full OCR 빌드는 `COCKTAIL_BUNDLE_PADDLE=1`일 때만 별도 생성.
- PyTorch `.lib` 및 `torch/include` 개발 산출물을 배포물에서 제외.
- `tessdata/*.traineddata`가 있으면 PyInstaller dist의 `tessdata/`에 포함해 zip 배포물도 OCR 언어 데이터를 갖도록 보강.
- `Cocktail완성본.py --self-test` 추가: GUI를 띄우지 않고 필수 모듈, Tesseract, `eng.traineddata`, 캐시 키 규칙 검사.
- GitHub Actions에 `dist\Cocktail\Cocktail.exe --self-test` 단계 추가.
- `smoke_tests.py` / `pre_release_check.py`에 R-2 회귀 검사를 추가.

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

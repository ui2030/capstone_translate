# Cocktail 업데이트 로그

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

# Cocktail

> **컴퓨터를 모국어로 보는 가장 짧은 길.**
> 인식·번역 전 과정을 내 PC에서 처리합니다. 화면 내용이나 번역 결과는 외부로 전송되지 않습니다.
> 단, **최초 1회 번역 모델 다운로드(인터넷 필요)**와 **Tesseract OCR 엔진 별도 설치**가 필요합니다.

---

## 비전

영역 박스로 한 곳을 번역하는 도구에서 시작했지만,
궁극 목표는 **OS 전반의 외국어를 사용자 모국어로 자연스럽게 보이게 하는 환경**입니다.
자세한 비전과 단계별 로드맵은 [VISION_AND_ROADMAP.md](VISION_AND_ROADMAP.md).

## 현재 상태 (Phase 0~6 기능 완료 — 베타 가능)

- ✅ 영역 박스 OCR + 한국어(또는 사용자 지정 언어) 오버레이
- ✅ Tesseract OCR (기본) + PaddleOCRv5 (선택) + UIA 자동 라우팅 (활성 창 모드)
- ✅ auto 모드 결정론 라우팅 (SL-1) — OCR은 `eng+kor` 고정, 라인 언어는 문자 스크립트로만 판정 (추측 없음)
- ✅ 번역: opus-mt(en→ko 전용) + m2m100(다국어 폴백) 하이브리드
- ✅ Unicode script 기반 다국어 자동 라우팅 (Hangul/Hiragana/CJK/Arabic/Cyrillic)
- ✅ YOLO식 bbox region smoothing — 좌우 다른 언어 동시 표시 시 안정
- ✅ 자기 캡처 피드백 차단 (`SetWindowDisplayAffinity`)
- ✅ 한국어 박스 자동 폰트 축소 + height 동적 확장
- ✅ 모델 lazy load (UI 즉시 표시, 모델은 백그라운드 준비)
- ✅ Tesseract 미설치/언어팩 누락도 시작 예외 X — 상태 라벨로 안내
- ✅ 표시 게이트 (DG-1) — 번역 신뢰도가 낮은 줄은 그리지 않고 상태줄에 숨긴 줄 수 표시
- ✅ 물리 픽셀 좌표 통일 (DP-1) — 고DPI/배율 환경에서 자막 위치 어긋남 제거
- ✅ 멀티 모니터 — 가상 데스크톱 전체 오버레이 + 모니터 연결/해제/해상도 변경 시 자동 재적합
- ✅ 트레이 상주 + 글로벌 단축키, DPAPI 암호화 영구 캐시, 민감 창 자동 일시정지, 시작 프로그램 등록

## 설치 / 실행

### 최초 1회 준비

1. **Python 3.10 / 3.11** + 의존성: `pip install -r requirements.txt`
2. **Tesseract OCR 엔진 설치** (앱에 포함돼 있지 않음):
   https://github.com/UB-Mannheim/tesseract/wiki
3. **첫 실행 시 번역 모델 다운로드** — opus-mt / m2m100을 Hugging Face에서 받습니다
   (약 1~2GB, 인터넷 필요). 받은 뒤에는 캐시에서 로드하므로 오프라인으로 동작합니다.

자세한 환경/버전/환경변수는 [ENVIRONMENT.md](ENVIRONMENT.md).

### 실행

[run_cocktail.bat](run_cocktail.bat)을 더블클릭하면 쓸 수 있는 Python을 자동으로 찾아
환경 진단 후 앱이 시작됩니다. (탐색 순서: `COCKTAIL_PYTHON` → conda `cd` env → `py -3.11` → `python`)

```bash
run_cocktail.bat
```

앱이 조용히 죽으면 `debug_run.bat`으로 타임스탬프 로그(`debug_log.txt`)를 남길 수 있습니다.

진단만 실행하려면:

```bash
python diagnose_environment.py
```

다음 개발 단계로 넘어가기 전 빠른 회귀 검사는:

```bash
python test_cocktail.py
python pre_release_check.py
```

## 빌드/배포

- 의존성 설치: `pip install -r requirements.txt`
- 개발/빌드 도구: `pip install -r requirements-dev.txt`
- 인스톨러용 traineddata 다운로드: `python download_tessdata.py`
- `.exe` 빌드: `pyinstaller build.spec`
- 인스톨러 빌드: Inno Setup으로 [installer.iss](installer.iss) 컴파일
  - traineddata는 사용자가 선택 (eng 필수, kor/jpn/chi_sim/fra/deu 선택)

자세한 4단계 빌드 절차는 [BUILD.md](BUILD.md).

## 코드 구조

2026-08-03에 단일 파일(`Cocktail완성본.py`, 3,310줄)에서 역할별 모듈로 분할했습니다.
의존 방향은 **아래에서 위로만** 흐릅니다 — 아랫층은 윗층을 import 하지 않습니다.

| 파일 | 역할 |
|---|---|
| `cocktail.py` | 엔트리포인트 — Qt 앱 생명주기, `--self-test` |
| `cocktail_ui.py` | 설정 창 / 투명 오버레이 / 영역 편집기 |
| `cocktail_engine.py` | 워커 루프 — 캡처→OCR→번역→표시 (`BackgroundController`) |
| `cocktail_translate.py` | 번역 모델·캐시·표시 게이트 |
| `cocktail_ocr.py` | Tesseract / PaddleOCR / UI Automation |
| `cocktail_capture.py` | 캡처 영역·프레임 변화 감지 |
| `cocktail_platform.py` | Windows API (창 속성 / DPAPI / 자동 실행) |

## 기술 스택

- UI: PySide6 (LGPL — closed-source 상용 OK)
- OCR: Tesseract (기본) / PaddleOCRv5 (선택)
- 번역: Helsinki-NLP/opus-mt-tc-big-en-ko (CC-BY-4.0, 출처 표시 필요) + facebook/m2m100_418M (MIT)
- 언어 감지: `unicodedata` script fast-path만 (라틴/불명 = en). 다른 언어는 source 콤보에서 명시
- 자세한 내용: [TECH_STACK.md](TECH_STACK.md), [COMMERCIAL_SOTA_REVIEW.md](COMMERCIAL_SOTA_REVIEW.md)

## 라이선스 정책

상용 호환만 사용:
- PySide6 (LGPL), opus-mt (CC-BY-4.0), m2m100 (MIT), Tesseract (Apache 2.0), tessdata (Apache 2.0)
- ❌ NLLB (CC-BY-NC) / PyQt5 (GPL) / 비상용 모델은 사용 안 함

> ✅ 폐기된 NLLB(CC-BY-NC) 변환 모델 잔재 `nllb_ct2/`(약 600MB)는 **2026-08-03 삭제 완료**.
> `.gitignore`·`pre_release_check.py`가 재추가를 차단합니다.

> ℹ️ **opus-mt는 CC-BY-4.0입니다** — 상용 사용은 가능하지만 **출처 표시 의무**가 있습니다.
> 배포물에 [LICENSE-3RDPARTY.md](LICENSE-3RDPARTY.md)를 동봉하세요(빌드에 이미 포함).
> 앱에서는 트레이 → "라이선스 정보"로 열람할 수 있습니다.

상세는 [BUILD.md](BUILD.md) 라이선스 체크리스트, [LICENSE-3RDPARTY.md](LICENSE-3RDPARTY.md),
[COMMERCIAL_SOTA_REVIEW.md](COMMERCIAL_SOTA_REVIEW.md), [VISION_AND_ROADMAP.md](VISION_AND_ROADMAP.md) 참고.

## 라이선스

본 프로젝트: **MIT** ([LICENSE](LICENSE)). 의존성 라이선스 정리: [LICENSE-3RDPARTY.md](LICENSE-3RDPARTY.md).

## 문서

- ⭐ [STORY.md](STORY.md) — **프로젝트 여정 전체 내러티브** (다른 세션/협업자 진입점)
- [VISION_AND_ROADMAP.md](VISION_AND_ROADMAP.md) — 장기 비전 + Phase 0~6 로드맵 + 보안 원칙
- [ARCHITECTURE.md](ARCHITECTURE.md) — 현재 클래스 구조 + 시그널/데이터 흐름 + 다음 분리 단계 설계
- [ENVIRONMENT.md](ENVIRONMENT.md) — 검증된 Python/패키지 버전, 환경 진단, 빠른 시작 가이드
- [TECH_STACK.md](TECH_STACK.md) — 사용 기술 한 페이지
- [COMMERCIAL_SOTA_REVIEW.md](COMMERCIAL_SOTA_REVIEW.md) — SOTA/상용 라이선스 기준선
- [UPDATE_LOG.md](UPDATE_LOG.md) — 변경 일지
- [test_cocktail.py](test_cocktail.py) — 실동작 회귀 테스트 (`--full`이면 번역 모델까지 로드)
- [pre_release_check.py](pre_release_check.py) — 배포 전 파일/워크플로우/라이선스 정책 검사
- [BUILD.md](BUILD.md) — 빌드/배포 절차
- [CODE_SIGNING.md](CODE_SIGNING.md) — Windows 코드 서명 (Authenticode) 가이드
- [ICONS.md](ICONS.md) — 아이콘 자산 가이드
- [OCR_TEST_PLAN.md](OCR_TEST_PLAN.md) — 베타 전 OCR/번역 테스트 시나리오
- [errors.md](errors.md) — 결함 누적 로그 + 5초 체크리스트 (코드 수정 전 필독)
- [README.en.md](README.en.md) — English entry point

## 보안/윤리 원칙

자세한 5개 원칙은 [VISION_AND_ROADMAP.md](VISION_AND_ROADMAP.md) §5 참고.
요약: **로컬 추론(화면·번역 데이터 외부 전송 없음) · 번역 데이터 비저장 · 민감 영역 자동 OFF ·
오픈 코어 · 즉시 토글**.

외부 통신은 **최초 1회 모델 다운로드(Hugging Face)** 뿐입니다. 그 이후 실행에서는
번역·OCR 모두 로컬에서만 수행되며 화면 내용은 네트워크로 나가지 않습니다.

---

## 프로젝트 학술 배경 (초기 작성)

> 현 프로젝트는 인공지능 모델을 기반으로 실시간 번역을 위해 만들어졌다.
> 실시간으로 인식하기 위해서 OCR이 필요한데, 초기에는 Tesseract만 사용했다.
> 원래는 Tesseract, Paddle, EasyOCR을 모두 사용하려 했지만, 활용을 제대로 하지 못해
> 1개 쓰는 것보다 못한 성능으로 결국 하나만 사용했다.
> (현재는 Tesseract 기본 + PaddleOCRv5 선택 백엔드로 두 OCR 모두 통합되었다.)
>
> 번역 모델로는 Hugging Face의 `facebook/m2m100_418M`을 사용했다.
> 다중 언어가 지원되는 모델이라 약 80여 개의 언어를 번역할 수 있다.
> 초기에는 영어 → 한국어만 다뤘으나, 현재는 다국어 자동 라우팅으로 확장되었다.
> (en→ko는 opus-mt 전용 모델로, 나머지는 m2m100 폴백으로 처리한다.)
>
> 발전 방향으로 "인식 언어 → 영어 → 원하는 언어" 피벗 형태를 고려했으나,
> opus-mt + m2m100 하이브리드로 직접 번역(any-to-any)을 사용하므로 영어 피벗은 불필요해졌다.
>
> 최적화 측면도 초기에는 느렸으나 현재는 batched generate, fp16(GPU),
> `torch.inference_mode`, frame-skip 해시, LRU 캐시, 모델 lazy load 등으로 개선되었다.

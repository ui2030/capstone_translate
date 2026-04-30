# Cocktail

> **컴퓨터를 모국어로 보는 가장 짧은 길.**
> 100% 로컬 처리. 어떤 데이터도 외부로 나가지 않습니다.

---

## 비전

영역 박스로 한 곳을 번역하는 도구에서 시작했지만,
궁극 목표는 **OS 전반의 외국어를 사용자 모국어로 자연스럽게 보이게 하는 환경**입니다.
자세한 비전과 단계별 로드맵은 [VISION_AND_ROADMAP.md](VISION_AND_ROADMAP.md).

## 현재 상태 (Phase 0 — 베타 가능)

- ✅ 영역 박스 OCR + 한국어(또는 사용자 지정 언어) 오버레이
- ✅ Tesseract OCR (기본) + PaddleOCRv5 (선택)
- ✅ 번역: opus-mt(en→ko 전용) + m2m100(다국어 폴백) 하이브리드
- ✅ Unicode script 기반 다국어 자동 라우팅 (Hangul/Hiragana/CJK/Arabic/Cyrillic)
- ✅ YOLO식 bbox region smoothing — 좌우 다른 언어 동시 표시 시 안정
- ✅ 자기 캡처 피드백 차단 (`SetWindowDisplayAffinity`)
- ✅ 한국어 박스 자동 폰트 축소 + height 동적 확장
- ✅ 모델 lazy load (UI 즉시 표시, 모델은 백그라운드 준비)
- ✅ Tesseract 미설치/언어팩 누락도 시작 예외 X — 상태 라벨로 안내

## 실행

개발 환경에서는 [run_cocktail.bat](run_cocktail.bat)을 더블클릭하면 환경 진단 후 앱이 시작됩니다.

```bash
run_cocktail.bat
```

진단만 실행하려면:

```bash
python diagnose_environment.py
```

## 빌드/배포

- 의존성 설치: `pip install -r requirements.txt`
- 개발/빌드 도구: `pip install -r requirements-dev.txt`
- 인스톨러용 traineddata 다운로드: `python download_tessdata.py`
- `.exe` 빌드: `pyinstaller build.spec`
- 인스톨러 빌드: Inno Setup으로 [installer.iss](installer.iss) 컴파일
  - traineddata는 사용자가 선택 (eng 필수, kor/jpn/chi_sim/fra/deu 선택)

자세한 4단계 빌드 절차는 [BUILD.md](BUILD.md).

## 기술 스택

- UI: PySide6 (LGPL — closed-source 상용 OK)
- OCR: Tesseract (기본) / PaddleOCRv5 (선택)
- 번역: Helsinki-NLP/opus-mt-tc-big-en-ko (Apache 2.0) + facebook/m2m100_418M (MIT)
- 언어 감지: `unicodedata` script fast-path + langdetect 폴백
- 자세한 내용: [TECH_STACK.md](TECH_STACK.md)

## 라이선스 정책

상용 호환만 사용:
- PySide6 (LGPL), opus-mt (Apache 2.0), m2m100 (MIT), Tesseract (Apache 2.0), tessdata (Apache 2.0)
- ❌ NLLB (CC-BY-NC) / PyQt5 (GPL) / 비상용 모델은 사용 안 함

상세는 [BUILD.md](BUILD.md) 라이선스 체크리스트와 [VISION_AND_ROADMAP.md](VISION_AND_ROADMAP.md) 참고.

## 라이선스

본 프로젝트: **MIT** ([LICENSE](LICENSE)). 의존성 라이선스 정리: [LICENSE-3RDPARTY.md](LICENSE-3RDPARTY.md).

## 문서

- ⭐ [STORY.md](STORY.md) — **프로젝트 여정 전체 내러티브** (다른 세션/협업자 진입점)
- [VISION_AND_ROADMAP.md](VISION_AND_ROADMAP.md) — 장기 비전 + Phase 0~6 로드맵 + 보안 원칙
- [ENVIRONMENT.md](ENVIRONMENT.md) — 검증된 Python/패키지 버전, 환경 진단, 빠른 시작 가이드
- [TECH_STACK.md](TECH_STACK.md) — 사용 기술 한 페이지
- [UPDATE_LOG.md](UPDATE_LOG.md) — 변경 일지
- [BUILD.md](BUILD.md) — 빌드/배포 절차
- [CODE_SIGNING.md](CODE_SIGNING.md) — Windows 코드 서명 (Authenticode) 가이드
- [ICONS.md](ICONS.md) — 아이콘 자산 가이드
- [OCR_TEST_PLAN.md](OCR_TEST_PLAN.md) — 베타 전 OCR/번역 테스트 시나리오
- [errors.md](errors.md) — 결함 누적 로그 + 5초 체크리스트 (코드 수정 전 필독)
- [README.en.md](README.en.md) — English entry point

## 보안/윤리 원칙

자세한 5개 원칙은 [VISION_AND_ROADMAP.md](VISION_AND_ROADMAP.md) §5 참고.
요약: **100% 로컬 처리 · 번역 데이터 비저장 · 민감 영역 자동 OFF · 오픈 코어 · 즉시 토글**.

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

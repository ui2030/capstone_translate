# Commercial SOTA Review

Last updated: 2026-05-04 (BG-6 완료 — Phase 0 분리 사이클 종결)

이 문서는 Cocktail Translator를 상용 배포 제품으로 만들기 위한 기술/라이선스 기준선이다.
새 OCR/번역 모델 또는 라이브러리를 추가하기 전 이 파일과 `LICENSE-3RDPARTY.md`를 함께 확인한다.

## 현재 판단

Cocktail은 논문 벤치마크 최고 점수만 좇는 구조가 아니라, 실시간 화면 번역 제품에 맞춘 실용형 SOTA 구조를 따른다.

- 표준 Windows UI는 UIA로 먼저 읽는다. OCR보다 빠르고 정확하며, 배경 번역기 UX에 맞다.
- OCR이 필요한 화면은 Tesseract OSD/script routing으로 언어 후보를 줄인 뒤 OCR한다.
- PaddleOCR/PP-OCRv5는 선택 백엔드로 둔다. 최신 OCR 성능 후보지만 기본 배포에는 무게/용량 리스크가 있다.
- 번역은 상용 호환 모델만 허용한다. 비상용 모델이 더 좋아도 기본 배포에는 넣지 않는다.

## 공식 확인 기준

| 영역 | 현재 선택 | 상용 판단 | 공식 근거 |
|---|---|---|---|
| OCR 기본 | Tesseract OCR | Apache 2.0, 상용 호환 | https://github.com/tesseract-ocr/tesseract |
| OCR 데이터 | tessdata_fast | Apache 2.0, 상용 호환 | https://github.com/tesseract-ocr/tessdata_fast |
| OCR 선택 | PaddleOCR / PP-OCRv5 | Apache 2.0 계열, 상용 호환 | https://github.com/PaddlePaddle/PaddleOCR |
| UI | PySide6 | LGPLv3/GPLv3/commercial. closed-source 배포 시 LGPL 의무 관리 필요 | https://doc.qt.io/qtforpython-6/index.html |
| 번역 폴백 | facebook/m2m100_418M | 모델 카드 기준 라이선스 확인 필요. 현재 문서 기준 MIT로 관리 | https://huggingface.co/facebook/m2m100_418M |
| ML 런타임 | transformers | Apache 2.0 | https://github.com/huggingface/transformers |
| ML 런타임 | PyTorch | BSD-style | https://github.com/pytorch/pytorch |

## 금지 정책

- `facebook/nllb-200-*` 같은 CC-BY-NC 모델은 기본 배포에 포함하지 않는다.
- GPL/AGPL 런타임 의존성은 closed-source 상용 배포 기본 경로에 넣지 않는다.
- Hugging Face 모델은 라이브러리 라이선스와 모델 라이선스가 다를 수 있으므로 모델별 카드 확인 없이는 추가하지 않는다.
- “성능이 더 좋다”는 이유만으로 비상용/불명확 라이선스를 채택하지 않는다.

## 구조 비판 리뷰

2026-05-04 BG-2에서 실행 생명주기를 `CocktailAppController`로 분리했다.
2026-05-04 BG-3에서 캡처 모드/캡처 사각형 상태를 `CaptureRegionController`로 분리했다.
2026-05-04 BG-4에서 워커/엔진 시그널/OCR·UIA 라우팅/모델 lifecycle/캐시 타이머를 `BackgroundController`로 분리했다.
2026-05-04 BG-5에서 `red_rect`를 화면 절대 좌표로 통일했다. settings window 이동/숨김 의존을 제거하고, 워커/오버레이의 좌표 변환 코드를 단순화했다.
2026-05-04 BG-6에서 `TransparentWindow` → `SettingsWindow` 리네이밍, `CaptureRegionController.owner` 의존을 `is_self_hwnd` 콜백으로 통합, dead code(빈 `paintEvent` override)를 정리했다. 이로써 Phase 0 분리 사이클이 종결된다.

이유:
- `__main__`에서 `QApplication`, 설정창, 오버레이를 직접 만드는 구조는 배포 후 종료/재시작/테스트 자동화가 약하다.
- 설정창과 오버레이 생성 순서, 연결 순서, `setQuitOnLastWindowClosed(False)` 정책을 한 곳에서 강제해야 한다.
- smoke/pre-release gate가 이 구조를 검사해 같은 결함이 다시 들어오지 않게 한다.
- 설정창 resize가 캡처 영역 편집으로 오해되면 백그라운드 제품 UX가 흔들린다.
  캡처 영역 상태는 settings panel 밖에서 관리해야 한다.

Phase 0 분리 사이클 종결:
- `SettingsWindow`(=구 `TransparentWindow`)는 UI(콤보/버튼/트레이/단축키)만 보유.
- `BackgroundController`가 워커/엔진/모델/캐시 lifecycle 소유.
- `CaptureRegionController`가 캡처 영역 상태 소유 (절대 좌표, callback 의존).
- `OverlayWindow`가 가상 데스크톱 렌더링 소유.
- `CocktailAppController`가 생명주기/연결 소유.

다음 사이클 후보 (Phase 1+):
- 영역 편집 UX: `RegionEditor` 위젯 — `CaptureRegionController.red_rect`(절대 좌표)를 마우스 드래그로 조작.
- VISION_AND_ROADMAP.md Phase 1+ 항목 (UIA 매트릭스 자동화, 글로벌 단축키 강화 등).

## 배포 전 체크

- `python test_cocktail.py`
- `python pre_release_check.py --allow-build-artifacts`
- `python cocktail.py --self-test`
- PyInstaller 후 `dist\Cocktail\Cocktail.exe --self-test`

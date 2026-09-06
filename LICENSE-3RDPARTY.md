# Third-Party Licenses

본 프로젝트(Cocktail Translator)는 다음 오픈소스 컴포넌트를 사용합니다.
각 컴포넌트는 자체 라이선스를 따르며, 본 저작물은 그 의무사항(저작권 고지 보존 등)을 준수합니다.

라이선스 정책 요약 — 모두 **상용 closed-source 호환**:
- ✅ Apache 2.0 / MIT / LGPL / BSD / Public Domain
- ❌ GPL / AGPL / CC-BY-NC / 비상용 한정 (`project_commercial_decisions.md` 참조)

---

## 1. UI 프레임워크

| 컴포넌트 | 라이선스 | 출처 | 비고 |
|---|---|---|---|
| **PySide6** | LGPL v3 | https://pypi.org/project/PySide6/ | LGPL — closed-source 상용 OK (동적 링크 유지) |

LGPL 의무: PySide6를 정적 링크하지 않고 동적 라이브러리로 함께 배포(또는 사용자 교체 가능 경로 제공).
PyInstaller `--onedir` 빌드는 동적 .pyd로 함께 묶이므로 의무사항 충족.

---

## 2. 번역 모델

| 모델 | 라이선스 | 출처 |
|---|---|---|
| `Helsinki-NLP/opus-mt-tc-big-en-ko` | **CC-BY-4.0** | Hugging Face |
| `Helsinki-NLP/opus-mt-zh-en` | **CC-BY-4.0** | Hugging Face |
| `facebook/m2m100_418M` | MIT | Hugging Face |

`opus-mt-zh-en`은 중국어→한국어 피벗 번역의 앞다리다(zh→en→ko, `errors.md` M-9).
중국어 화면이 실제로 나올 때만 지연 다운로드/로드된다.

### CC-BY-4.0 의무 이행 (필수 저작자 표시)

두 opus-mt 모델은 **CC-BY-4.0**이다 (모델 카드 YAML `license: cc-by-4.0` 실측 확인,
en-ko 2026-08-03 / zh-en 2026-08-11). 상용 사용은 허용되지만 **저작자 표시**와
**변경 사실 표시** 의무가 따른다.
아래 문구를 배포물(본 문서)에 동봉하며, 앱 트레이 메뉴 "라이선스 정보"에서 열람할 수 있다.

> Translation models: `Helsinki-NLP/opus-mt-tc-big-en-ko`, `Helsinki-NLP/opus-mt-zh-en`
> © Language Technology Research Group at the University of Helsinki (OPUS-MT / Tatoeba-Challenge)
> Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
> Cocktail은 모델 가중치를 **수정하지 않고** 원본 그대로 사용한다.
> (토크나이저 입력 인코딩만 `source.spm`으로 우회한다 — 모델 파일 자체는 무변경. errors.md M-8)

모델을 파인튜닝하거나 가중치를 변환(양자화/CT2 등)해 배포할 경우 **"변경했다"는 사실을 추가 명시**해야 한다.

**PK-2 (2026-09-05): 배포물에 en-ko 가중치가 실제로 들어간다.** `build.spec`이
`_internal/models/opus-mt-tc-big-en-ko/`(safetensors 399MB)를 동봉한다 — 그전까지는
사용자 PC가 Hugging Face에서 받아갔으므로 "배포"가 아니었지만, 이제는 **우리가 재배포**한다.
따라서 위 저작자 표시 문구를 인스톨러 산출물에 반드시 동봉해야 한다(`LICENSE-3RDPARTY.md`는
`build.spec`의 `project_docs` 목록에 있어 `_internal/`에 함께 설치된다 — 확인 완료).
`opus-mt-zh-en`과 `m2m100_418M`은 동봉하지 않고 필요할 때 사용자 PC가 내려받는다.

**❌ 사용하지 않음**: `facebook/nllb-200-*` (CC-BY-NC, 비상용) — `errors.md` M-5 참고.
2026-08-03 프로젝트 폴더의 `nllb_ct2/` 잔재(600MB)를 삭제 완료.

**❌ 사용하지 않음**: `shun89/opus-mt-zh-ko` — zh→ko 후보로 실측했으나 채택하지 않았다
(`errors.md` M-9). 라이선스를 apache-2.0으로 신고했지만 베이스가 CC-BY-4.0 opus-mt 라
신고를 신뢰할 수 없고, 학습 데이터가 문서화돼 있지 않다. 3rd party 파인튜닝 가중치는
상용 배포 기본 경로에서 제외한다.

---

## 3. OCR

| 컴포넌트 | 라이선스 | 비고 |
|---|---|---|
| **Tesseract OCR** (엔진) | Apache 2.0 | https://github.com/tesseract-ocr/tesseract |
| **traineddata** (eng/kor/jpn/...) | Apache 2.0 | https://github.com/tesseract-ocr/tessdata_fast |
| **Tesseract 동봉 DLL** (leptonica/libpng/libtiff/zlib/libcurl/libarchive 등) | Apache 2.0 / BSD / zlib / MIT 계열 | UB-Mannheim 배포판 그대로 |
| **pytesseract** (Python 바인딩) | Apache 2.0 | https://pypi.org/project/pytesseract/ |
| **PaddleOCR** (선택, optional) | Apache 2.0 | https://github.com/PaddlePaddle/PaddleOCR |
| **PaddlePaddle** (선택) | Apache 2.0 | https://github.com/PaddlePaddle/Paddle |

**PK-1 (2026-09-05): Tesseract 엔진(tesseract.exe + DLL 56개, v5.5.0)을 배포물에 동봉한다.**
사용자에게 별도 설치를 요구하면 "준비 5분"(합격기준 6)이 깨진다. Apache 2.0은 재배포 시
라이선스 원문과 NOTICE 동봉을 요구하므로 원본 배포판의 `doc/LICENSE`·`doc/AUTHORS`를
`_internal/tesseract/doc/`에 같이 넣는다(`build.spec`).

---

## 4. 머신러닝/추론

| 컴포넌트 | 라이선스 |
|---|---|
| **transformers** (Hugging Face) | Apache 2.0 |
| **tokenizers** (Hugging Face) | Apache 2.0 |
| **torch** (PyTorch) | BSD-3-Clause |
| **sentencepiece** (Google) | Apache 2.0 |
| **sacremoses** | LGPL v3 → MIT 재라이선스 (Marian 토크나이저용) |

---

## 5. 영상/이미지

| 컴포넌트 | 라이선스 |
|---|---|
| **opencv-python** | Apache 2.0 |
| **Pillow (PIL)** | HPND (BSD-style) |
| **numpy** | BSD-3-Clause |

---

## 6. 언어 감지

제3자 컴포넌트 없음. Python 표준 라이브러리 `unicodedata`의 스크립트 판정만 사용한다 (SL-1).
(과거에 쓰던 `langdetect`(Apache 2.0)는 퇴역했고 의존성에서 제거됐다.)

---

## 7. 시스템 통합 (선택)

| 컴포넌트 | 라이선스 | 비고 |
|---|---|---|
| **keyboard** | MIT | 글로벌 단축키. 미설치 시 graceful 비활성 |
| **uiautomation** | Apache 2.0 | UIA 어댑터 (Phase 2/3). 미설치 시 graceful 비활성 |

---

## 8. Windows API (Python 표준)

| 호출 | 라이선스 | 비고 |
|---|---|---|
| `ctypes.windll.user32` (`SetWindowDisplayAffinity`, `GetForegroundWindow` 등) | OS 시스템 API | 별도 라이선스 X |
| `ctypes.windll.crypt32` (`CryptProtectData`) | DPAPI | 별도 라이선스 X |
| `winreg` (자동 실행) | Python 표준 라이브러리 | PSF License |

---

## 9. 빌드 도구 (개발 시에만)

| 컴포넌트 | 라이선스 | 비고 |
|---|---|---|
| **PyInstaller** | GPL with exception (배포 산출물 라이선스 영향 X) | `.exe` 빌드 |
| **Inno Setup** (외부 도구) | MPL (Inno Setup 자체) | 인스톨러 빌드 |

PyInstaller의 GPL은 **빌드 도구**에만 적용되며, 산출물(.exe)은 본 프로젝트 라이선스(MIT)를 따릅니다.

---

## 라이선스 검증 (배포 전 체크리스트)

- [ ] 새 의존성 추가 시 라이선스 사전 확인 (CC-BY-NC, GPL, 상용 전용 금지)
- [ ] `requirements.txt`/`requirements-dev.txt`에 신규 항목 → 본 문서에 동시 등록
- [ ] `BUILD.md` 라이선스 체크리스트와 본 문서 동기화
- [ ] 모델 라이선스는 Hugging Face 페이지에서 직접 확인 (라이선스가 변경될 수 있음)

## 2026-05-04 공식 근거 재확인

- Tesseract OCR: Apache 2.0 — https://github.com/tesseract-ocr/tesseract
- tessdata_fast: Apache 2.0 — https://github.com/tesseract-ocr/tessdata_fast
- PaddleOCR / PP-OCRv5: Apache 2.0 계열 — https://github.com/PaddlePaddle/PaddleOCR
- PySide6 / Qt for Python: LGPLv3/GPLv3/commercial — https://doc.qt.io/qtforpython-6/index.html
- transformers: Apache 2.0 — https://github.com/huggingface/transformers
- PyTorch: BSD-style — https://github.com/pytorch/pytorch

주의: 라이브러리 라이선스와 모델 라이선스는 별개다. Hugging Face 모델은 모델 카드 기준으로 별도 확인한다.
성능상 좋아도 CC-BY-NC/비상용/불명확 모델은 상용 배포 기본 경로에서 제외한다.

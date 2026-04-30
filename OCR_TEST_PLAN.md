# OCR/번역 실사용 테스트 계획

## 목표

일반 사용자 베타 전에 화면 자막/이미지 텍스트에서 OCR, 언어 감지, 번역, 오버레이 렌더링이 각각 실패하는 지점을 분리해서 기록한다.

## 테스트 축

- OCR 백엔드: Tesseract, PaddleOCRv5
- source 언어: auto, English, Korean, Japanese, Chinese, French, German
- target 언어: Korean, English
- 화면 유형: 유튜브 자막, 강의 슬라이드, 게임 UI, PDF 이미지, 웹페이지 캡처
- 글자 조건: 흰 자막/검은 그림자, 작은 글자, 컬러 글자, 복잡한 배경, 2줄 자막
- 배치 조건: 좌우 2개 언어 column, 상하 분리 island, 같은 줄 혼합 script

## 최소 샘플 문장

| 입력 | 기대 감지 | 비고 |
|---|---|---|
| Thank you | en | 짧은 영어 |
| Gracias | es | UI 콤보에는 없지만 m2m100 라우팅 가능 |
| Hvala | hr | langdetect 오판 보정 케이스 |
| Merci | fr | 짧은 라틴 보정 |
| Danke | de | 짧은 라틴 보정 |
| 감사합니다 | ko | Unicode Hangul fast-path |
| ありがとう | ja | Unicode Hiragana fast-path |
| 謝謝 | zh | 순수 CJK는 zh 기본 |
| شكرا | ar | Unicode Arabic fast-path |
| Спасибо | ru | Unicode Cyrillic fast-path |

## 공간 region 샘플

| 배치 | 입력 | 기대 |
|---|---|---|
| 좌측 column | `Thank you` / `Hvala` | 좌측 라틴 region으로 분리, 각 라인 힌트 유지 |
| 우측 column | `감사합니다` / `안녕` | 우측 Korean region으로 분리 |
| 좌측 하단 island | `Merci` / `Bonjour` | 상단 좌측과 별도 French island |
| 우측 하단 island | `ありがとう` | Japanese island |

## 합격 기준

- 앱이 시작 중 예외로 종료되지 않는다.
- OCR 백엔드가 실패하면 UI 상태 라벨에 이유가 보인다.
- 같은 화면을 유지하면 불필요하게 계속 번역하지 않는다.
- 번역 박스가 원문을 충분히 가리고, 텍스트가 박스 밖으로 심하게 벗어나지 않는다.
- PaddleOCR 미설치 환경에서도 Tesseract 기본 백엔드로 계속 사용할 수 있다.
- `auto` source에서 좌우 다른 언어가 동시에 있어도 감지 언어가 한쪽으로 전역 오염되지 않는다.

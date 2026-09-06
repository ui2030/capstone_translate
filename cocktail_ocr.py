"""Cocktail — OCR 계층.

Tesseract 경로/언어 탐색, 노이즈 라인 필터, PaddleOCR(선택) 백엔드,
UI Automation 텍스트 추출, 그리고 OCR 튜닝 레버 전부.
번역·UI를 import 하지 않는다.
"""
import functools
import importlib.util
import os
import re
import sys
import threading
import unicodedata

import cv2
import numpy as np
import pytesseract

import cocktail_platform   # noqa: F401 — EN-1: import만으로 stdout이 cp949에서 안전해진다

# OCR 백엔드 콤보의 표시 문자열. 이 문자열이 곧 라우팅 키다.
OCR_BACKEND_TESSERACT = "Tesseract"
OCR_BACKEND_PADDLE = "PaddleOCRv5"
OCR_BACKEND_UIA = "UIA (활성 창)"


# --- Tesseract 경로 탐색 (E-1, E-2, E-6, PK-1) ------------------------------
# PK-1: 배포본은 Tesseract 엔진과 tessdata를 **동봉**한다. 사용자가 따로 설치하게 두면
# "준비 5분"이 깨진다(합격기준 6). onedir PyInstaller는 datas를 `_internal/`에 넣으므로
# (= sys._MEIPASS) 앱 폴더와 _MEIPASS 둘 다 본다. 동봉본이 시스템 설치본보다 우선이다 —
# 우리가 버전을 아는 쪽이 이긴다.
_APP_DIR = os.path.dirname(os.path.abspath(getattr(sys, "argv", [""])[0] or "."))
_SEARCH_DIRS = [d for d in (_APP_DIR, getattr(sys, "_MEIPASS", None)) if d]


def _resolve_tesseract():
    cand = [os.environ.get("TESSERACT_CMD")]
    cand += [os.path.join(d, "tesseract", "tesseract.exe") for d in _SEARCH_DIRS]
    cand += [
        r"C:\tesseract\tesseract.exe",
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for p in cand:
        if p and os.path.exists(p):
            return p
    return None

TESSERACT_ERROR = ""
_TESSERACT_CMD = _resolve_tesseract()
if _TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = _TESSERACT_CMD
    print(f"[INFO] Tesseract: {pytesseract.pytesseract.tesseract_cmd}")
else:
    TESSERACT_ERROR = (
        "Tesseract 실행 파일 없음. TESSERACT_CMD 환경변수 또는 "
        "C:\\Program Files\\Tesseract-OCR\\tesseract.exe 설치 필요."
    )
    print(f"[ERROR] {TESSERACT_ERROR}")

# 배포 시 앱 폴더(또는 _internal) 내 ./tessdata 가 있으면 그걸 사용 — 인스톨러 컴포넌트로
# 설치된 traineddata. 앱 폴더가 우선: 인스톨러가 사용자 선택 언어를 {app}\tessdata 에 깐다.
_LOCAL_TESSDATA = next(
    (os.path.join(d, "tessdata") for d in _SEARCH_DIRS
     if os.path.isdir(os.path.join(d, "tessdata"))), None)
if _LOCAL_TESSDATA:
    os.environ["TESSDATA_PREFIX"] = _LOCAL_TESSDATA
    print(f"[INFO] TESSDATA_PREFIX = {_LOCAL_TESSDATA}")

TESS_CONFIG = "--oem 1 --psm 6"
# ponytail: psm 은 6 고정. 콘텐츠별 최적값이 **상충**한다는 걸 실측했다(2026-08-05):
#     psm 6  게임 UI acc 0.869 | 문서 acc 0.940   ← 채택
#     psm 12 게임 UI acc 0.930 | 문서 acc 0.888
# 흩어진 UI 글자에는 sparse 모드(11/12)가 확실히 낫고 문단 문서에는 확실히 나쁘다.
# 자동 선택을 하려면 "이 화면이 sparse 인가"를 런타임에 판정해야 하는데, 후보로 삼았던
# 대리 지표(확신 글자 수)가 실측에서 acc 의 1등을 **틀리게** 골랐다. 틀리는 걸 아는
# 지표로 분기를 만들면 조용히 나빠진다. 신뢰할 신호를 찾기 전까지는 고정.
# 올리는 길: 두 psm 을 다 돌려 비교(비용 2배) 또는 레이아웃 밀도 기반 판정기 실측.
# oem 0/2(legacy)는 이 tessdata 에 legacy 데이터가 없어 아예 실행 불가.

# --- NZ-1 튜닝 레버: OCR 노이즈 라인 필터 -----------------------------------
# UI 부스러기(아이콘 글자, 시계, 테두리 잡음)까지 박스로 그려 "중구난방"으로 보이던 문제.
# 정상적인 짧은 문장이 죽지 않도록 임계는 보수적으로 낮게 시작한다.
OCR_LINE_MIN_CONF = 35.0          # 라인 평균 word confidence 최소치. 부스러기가 남으면 ↑(50~60), 진짜 문장이 사라지면 ↓
# --- WD-1: 라인 **안**의 단어 컷 -------------------------------------------
# 이 컷은 문장 한복판의 단어를 조용히 지운다. 실측(게임 UI 스샷, 2026-08-05):
#     "gives a speed boost and removes negative status" → "sives a boost and removes status"
#     "Right click/trigger to throw Pudding"            → "Risht to throw Pudding"
# "negative"가 빠지면 뜻이 뒤집히고, 번역기는 그 훼손된 문장을 성실히 번역한다.
# 라인 단위 판단(OCR_LINE_MIN_CONF)은 줄을 통째로 버리므로 문장을 훼손하지 않는다.
#
# 실측 A/B (acc = 정답 대비 문자 정확도, spur = 정답과 안 맞는 쓰레기 줄 수):
#                             게임 UI              문서(어린왕자)
#     컷 30 (구 동작)      acc 0.839  spur 9      acc 0.885  spur 13   (확대 x2: 0.940 / 18)
#     컷 없음 (현행)       acc 0.869  spur 2      acc 0.869  spur 13   (확대 x2: 0.938 / 18)
# 쓰레기 줄이 9 → 2로 준 게 핵심이다. 단어를 골라 버리면 깨진 라인이 "그럴싸한 조각"으로
# 살아남아 화면에 박스가 그려진다. 안 버리면 그 라인의 평균 conf가 떨어져
# OCR_LINE_MIN_CONF 에서 **통째로** 걸린다. 문서 쪽 손실은 실동작 지점(확대 x2)에서 0.002.
# 음수면 컷 없음(단어를 절대 안 버림).
OCR_WORD_MIN_CONF = -1.0
OCR_LINE_MIN_CHARS = 2            # 공백 제외 이보다 짧은 라인은 버림 (아이콘 글자 1자)
OCR_LINE_MIN_LETTER_RATIO = 0.5   # 공백 제외 문자 중 영숫자/한글/CJK 비율 최소치. 기호 잡음 컷. ↑면 더 엄격
OCR_LINE_REQUIRE_ALPHA = True     # 글자(isalpha) 없는 라인(시계 "10:30", 페이지 번호, "100%") 제외.
                                  # 숫자만 있는 원문도 번역하고 싶으면 False.
# 라인 폭이 글자 크기보다 좁으면 글자가 아니라 **아이콘 한 조각**이다.
# 실측: 아이콘이 'Is'(폭 8, 크기 26) / 'Hill'(폭 26, 크기 28)로 읽혀 화면에 박스가 그려지고,
# 문단 사이에 끼어 병합 사슬까지 끊었다. 진짜 두 글자 단어는 크기의 1.5배쯤 된다.
OCR_LINE_MIN_WIDTH_RATIO = 1.2    # 라인 폭 ≥ 글자 크기 × 이 값. ↑면 짧은 단어까지 버림
# ponytail: "단어 간격이 넓으면 문장이 아니다"는 규칙을 넣었다가 **되돌렸다**.
# 본문 간격이 글자 높이의 0.43배, 그림 조각이 0.95배라 갈릴 줄 알았는데, 임계 0.7에서
# 어린왕자 본문 문단이 14줄 → 8줄로 잘려나갔다(실제 문장의 간격 편차가 그만큼 크다).
# 흩어진 조각은 LM-1b(양끝 조각 제거) + 폭 규칙으로 잡는다.

# --- OU-1 튜닝 레버: 작은 글씨 적응 업스케일 ---------------------------------
# 화면상 글자 높이가 12px 아래로 내려가면 Tesseract가 통째로 무너진다(캡션/각주).
# 1차 OCR 결과의 라인 높이 "중앙값"이 임계 미만일 때만 2배 확대해 재OCR 한다.
OCR_UPSCALE_TRIGGER_PX = 18       # 라인 높이 중앙값(원본 px) 임계. ↑면 더 자주 확대(느려짐), ↓면 덜 확대
OCR_UPSCALE_FACTOR = 2            # 확대 배율. 3 이상은 비용 대비 이득이 거의 없다

# --- OU-2 튜닝 레버: 1차 오독에 안 흔들리는 확대 트리거 (실측 2026-09-07) ----
# 위 `OCR_UPSCALE_TRIGGER_PX` 는 1차 패스가 읽은 값이라 1차가 무너지면 같이 오염된다
# (10px 글자를 28px 로 보고 → 확대 꺼짐 → CER 10.7%, 강제로 켜면 0.00%).
# 그래서 높이 규칙이 "확대 불필요"라고 할 때만 `ink_glyph_px()`(이미지에서 직접 잰
# 글리프 높이)로 두 번째 의견을 묻는다. 실측 대응: 10px→4~8 · 14px→7 · 18px→9~10 · 24px→12~13.
OCR_UPSCALE_TRIGGER_INK_PX = 11   # 글리프 높이 중앙값이 이 미만이면 확대. ↑면 큰 글씨까지 확대(느려짐)
OCR_INK_GLYPH_MIN_PX = 3          # 이보다 낮은 성분은 점·획 조각 (글자로 세지 않음)
OCR_INK_GLYPH_MAX_PX = 200        # 이보다 높은 성분은 테두리·도형
OCR_INK_MIN_GLYPHS = 8            # 글자 성분이 이보다 적으면 판정하지 않는다(None)

# --- PF-1: OCR 입력 크기 안전밸브 (실측 기반, 2026-08-03) --------------------
# ⚠️ 이것은 **성능 최적화 레버가 아니다. 폭주 방지용 안전밸브다.**
#
# 축소 배율 실측(개발 PC, 5760x2160 가상 데스크톱 / 2560x2160 단일 모니터):
#     scale 1.00  8535 ms  113줄 (100%)   |  1684 ms  47줄 (100%)
#     scale 0.85  6351 ms   97줄 ( 86%)   |  1518 ms  42줄 ( 89%)
#     scale 0.70  5445 ms   73줄 ( 65%)   |  1585 ms  49줄 (104%)
#     scale 0.50  2471 ms   55줄 ( 49%)   |  1056 ms  29줄 ( 62%)
#     scale 0.30  1061 ms   22줄 ( 20%)   |   564 ms   9줄 ( 19%)
# → "Tesseract는 어느 해상도 위에서는 정확도가 안 오른다"는 통념은 **이 화면에서 틀렸다**.
#   축소는 속도를 글자 인식률과 거의 1:1로 맞바꾼다. 화면 글자를 읽는 게 이 앱의 존재
#   이유이므로, 축소를 일상 최적화로 쓰면 안 된다.
# → 실제 속도는 PF-2(변경 밴드)와 SC-1(활성 모니터 클램프)에서 가져온다. 그쪽은 같은
#   배율로 **더 적은 픽셀**을 보는 것이라 인식률을 깎지 않는다.
#
# 그래서 상한은 "한 프레임이 10초씩 걸리는 병리적 입력"만 막을 만큼 느슨하게 둔다.
# 6000px = 4K 두 대를 가로로 붙인 것보다 넓은 경우에만 발동한다. 흔한 듀얼 QHD/4K
# 구성(≤5760px)에서는 축소가 아예 일어나지 않는다 — 인식률을 한 줄도 잃지 않는다.
OCR_MAX_SIDE_PX = 6000            # 긴 변이 이보다 크면 INTER_AREA 축소 (좌표는 환원됨)
OCR_MIN_DOWNSCALE = 0.7           # 아무리 커도 이 배율 아래로는 절대 줄이지 않는다.
                                  # 0.7 미만은 위 실측에서 인식률이 무너지는 구간이다.
# --- UB-1 튜닝 레버: 확대 재OCR 예산 (실측 기반, 2026-09-05) -----------------
# 예전엔 "확대 후 픽셀 수 ≤ 4M"으로 막았다. 그 규칙은 1920x1080의 2배(8.3M)를 **항상**
# 거부해서, 전체화면에서 OU-1이 구조적으로 한 번도 발동하지 않았다(문서 단어의 58%를
# 못 읽음). 그런데 픽셀 수는 비용의 대리지표로 쓸 수 없다 — Tesseract 비용은 면적이
# 아니라 **글자 양**을 따른다. 실측(eng 단독, 어린왕자 p4):
#     3840x2160 여백 많음 (8.3M)  1차 436 ms → 2배(33M) 재OCR 포함 1456 ms
#     3840x2160 글자 빽빽 (8.3M)  1차 4787 ms (같은 픽셀 수인데 11배)
# → 예산은 픽셀이 아니라 **글자 양**으로 잡는다.
#
# RP-1(2026-09-05): 그 "글자 양"을 처음엔 1차 패스의 **실측 시간**(예산 900 ms)으로 쟀다.
# 실측 시간은 PC 부하에 따라 변하는 값이라 **같은 입력이 실행마다 다르게 읽혔다** —
# 같은 중국어 시료가 2줄/3줄로 갈리고 채점판 판정이 뒤집혔다. 재현 불가능한 채점판은
# 채점판이 아니다. 그래서 시간 대신 **1차 패스가 실제로 읽어낸 글자 수**를 쓴다.
# 이건 이미지에서 결정론적으로 나오는 값이고, 시간과 실제로 비례한다(시료 실측,
# 프로세스 기동 고정비 130 ms 제외한 순수 작업시간 / 글자 수):
#     fullhd_doc 1320자 0.236 ms/자 · doc_en_column 766자 0.210 · multi_context 654자 0.207
# 예산은 글자·배율² 단위로 잡는다(비용 ≈ 글자 수 × 배율²):
#     배율 = min(2, √(예산 / 1차 글자 수))
# 2700으로 잡으면 구 시간 규칙과 같은 지점에 선다: 1080p 문서 1.43배(구 1.43),
# 문서/다중문맥 1.9~2.0배(구 1.76~1.84), 4K 빽빽(≈2만 자)은 0.37로 떨어져 자동 생략.
# 배율별 실측(1920x1080, 확대 안 하면 42%/0.541):
#     x1.3  796 ms  99% 0.997   x1.5  892 ms  100% 0.995   x2.0 1097 ms  99% 0.995
# → 1.3배면 정확도는 이미 포화한다.
# ↑면 더 크게 확대(느려짐), ↓면 확대를 더 자주 포기한다.
OCR_UPSCALE_CHAR_BUDGET = 2700
OCR_UPSCALE_MIN_FACTOR = 1.3      # 이 배율 미만은 인식률 이득이 없어 확대를 아예 생략
# UA-2(2026-09-07): 채택 기준에 "글자를 버리지 않았을 것"을 더한다. 평균 conf 만 보면
# **줄을 통째로 흘린 패스가 이긴다** — 남은 줄만 깨끗하니 평균이 오른다. 실측: Impact 12px
# 실화면에서 2차가 가운데 줄을 잃고(248자→140자, 비 0.56) conf 81.8→89.5 라 채택되어
# CER 7.51%→45.85%. 같은 코퍼스에서 정상 조합의 글자수 비는 전부 1.00~2.07 이라 둘이 겹치지 않는다.
# ↑면 확대를 더 자주 거부(작은 글씨 이득을 잃음), ↓면 글자를 흘린 패스가 다시 새어든다.
OCR_UPSCALE_MIN_COVERAGE = 0.8    # 2차가 읽은 글자 수 ≥ 1차 글자 수 × 이 값일 때만 채택
# UA-3(2026-09-07): conf 는 절대 눈금이 아니다. 확대 결과가 더 정확한데 평균 conf 가
# 3~4점 낮아 거부된 사례가 있었다(Calibri italic 12px: 2차 CER 5.53% < 1차 11.46% 인데
# conf 85.5→81.8 이라 탈락). 반대로 UA-1 의 근거였던 "쓰레기 줄이 늘어난 확대"는
# conf 가 71.9→60.4 로 11.5점 떨어진다 — 둘 사이에 여유가 있다. 마진은 그 사이에 둔다.
# ↑면 확대를 더 자주 채택(쓰레기도 함께 들어옴), ↓면 좋은 확대를 거부한다.
OCR_UPSCALE_CONF_MARGIN = 4.0     # 2차 conf 가 1차보다 이만큼까지 낮아도 채택

# --- PF-2 튜닝 레버: 변경 밴드만 OCR ----------------------------------------
# 프레임 해시(16x16 셀)는 이미 "어느 셀이 변했는지"를 알고 있는데 버리고 있었다.
# 변한 셀이 걸친 **가로 전체 띠(세로 구간만 제한)**만 OCR 한다. 가로를 자르면 단어가
# 잘려 OCR이 망가지므로 세로만 자른다. 자막 한 줄 변화면 면적이 1/10 이하로 준다.
# 밴드 밖의 직전 자막은 그대로 유지되고, 밴드 안의 것만 새 결과로 교체된다.
OCR_DIRTY_BAND_ENABLED = True     # False면 항상 전체 프레임 OCR(구 동작)
OCR_DIRTY_FULL_RATIO = 0.6        # 변한 세로 비율이 이보다 크면 밴드 이득이 없어 전체 처리


# --- SL-1 설계 결정: auto 모드 언어 라우팅 단순화 (OS-1/LG-1/M-6/M-7 퇴역) ----
# 자동 "다국어 추측"(OSD script 판별 + langdetect)은 실측에서 오판 공장이었다.
# 영어 PDF를 OSD가 Arabic(conf 3.95)으로 읽어 신뢰도 게이트를 통과했고, langdetect는
# 같은 화면에서 cy/hu/de/pt/so를 뱉었다. 오판 1건의 대가는 쓰레기 OCR + m2m100 5초 배치.
# → auto는 이제 결정론만 쓴다: OCR 언어 eng+kor 고정, 라인 언어는 문자 스크립트로
#   확정되는 것만(한글/가나/CJK/키릴…) 인정하고 라틴·불명은 전부 en.
#   다른 언어를 읽어야 하면 사용자가 src 콤보에서 명시한다(그 경로는 그대로 동작).
OCR_LANGS_AUTO_DEFAULT = ("eng", "kor")   # 환경변수 COCKTAIL_OCR_LANGS_AUTO로 오버라이드 가능
# LC-1: 단, **목표 언어가 한국어면 kor 을 뺀다.** 인식된 한국어 줄은 SK-1(원문 언어 ==
# 목표 언어)이 어차피 전부 버리므로(실스크린샷 51줄 중 38줄) 값어치가 없는데,
# 언어팩 하나가 OCR 시간을 2.5배로 올린다. 실측(2026-09-05, 1920x1080 문서):
#     eng 313 ms   |   eng+kor 771 ms (+146%)   |   eng+kor+chi_sim 1201 ms
# 한→외 번역(목표가 한국어가 아님)에서는 kor 이 있어야 원문을 읽으므로 반드시 남긴다.
OCR_LANGS_TGT_REDUNDANT = {"ko": "kor"}   # 목표 언어 → auto 목록에서 뺄 traineddata
# LC-1b: 그런데 kor 을 빼면 **한글 화면에서 라틴 오독이 표시된다**. 실측(2026-09-05,
# 한국어 UI 스샷): eng 단독은 "인벤토리"를 "Be] Ast"로 읽고 conf 39~70으로 자신 있게
# 뱉는다 → 번역 게이트를 통과해 엉뚱한 자막 박스가 4개 떴다. kor 이 있으면 conf 89~97로
# 정확히 읽고 SK-1이 조용히 버린다.
# 그래서 conf 가 낮은 프레임에서만 kor 을 붙여 한 번 더 읽고, conf 가 실제로 오를 때만
# 그 결과를 채택한다(UA-1과 같은 규칙). 프레임 conf 실측(eng 단독 스캔, 확대 포함):
#     한국어 UI     평균 54.8, 저conf 라인 100%   ← 재스캔해야 하는 쪽
#     한국어 게임스샷 평균 57.1, 80%
#     4K 빽빽 문서   평균 75.6, 29%              ← 재스캔하면 헛수고
#     중국어 앱      평균 83.8, 20%
#     영어 UI/문서   평균 91.7~95.4, 0~6%
# 평균만 보면 화면 절반만 한글인 혼합 화면을 놓치고(평균이 가운데로 끌려감), 비율만 보면
# 흐릿한 영어 화면에서 헛도니 **둘 중 하나**로 건다.
OCR_KO_MEAN_CONF = 68.0           # 프레임 평균 conf 가 이 미만이면 재스캔 (한국어 57 vs 영어 76)
OCR_KO_JUNK_CONF = 72.0           # 이 미만 라인은 "오독 의심"
OCR_KO_JUNK_RATIO = 0.5           # 의심 라인이 이 비율 이상이어도 재스캔 (혼합 화면용)
OCR_KO_STICKY_FRAMES = 12         # 한글 화면으로 확인되면 이 프레임 수만큼 kor 을 계속 켠다
OCR_KO_FAIL_COOLDOWN = 20         # 재스캔이 헛수고였으면 이 프레임 수 동안 다시 시도 안 함


def has_hangul(text: str) -> bool:
    return any("가" <= ch <= "힣" or "ㄱ" <= ch <= "ㆎ" for ch in text or "")


# --- PY-1 튜닝 레버: 병음(로마자 발음표기) 줄 제외 ---------------------------
# 중국어 학습 앱은 한 문장을 3층으로 그린다: (위) 병음 + 성조부호 / (가운데) 한자 /
# (아래) 앱 자체 번역. 병음은 **글이 아니라 발음 힌트**라 번역 입력에 들어가면 안 된다.
# 실측(학습앱 스샷, 2026-08-11): auto(eng+kor)에서 한자는 한 글자도 안 읽히고 병음 줄만
# 살아남아 en으로 확정 → en-ko 모델에 그대로 들어갔다.
#     "xiǎoměi zhàn zài hébiān" → "작은 미는 강 옆에 서서"
#     "sìzhōu"                  → "4주간"
# 두 갈래 판정을 **OR**로 쓴다. 어느 쪽도 영어 본문을 잡아먹지 않게 임계를 실측으로 잡았다.
#
#   (a) 강성 성조부호: 마크론(āēīōū) / 캐런(ǎěǐǒǔ) / ǖǘǚǜ.
#       영어·한국어·독일어·프랑스어 화면에 사실상 안 나온다. 어큐트(áéíóú)와
#       그레이브(àèìòù)는 프랑스어/스페인어에 흔해서 **일부러 뺐다** — 넣으면
#       "café", "Saint-Exupéry" 가 든 영어 문서가 통째로 사라진다.
#   (b) 병음 음절 분해율: 토큰을 성모+운모 음절로 완전 분해할 수 있는 비율.
#       OCR이 병음의 음절 사이 공백을 자주 뭉개므로("oméizhanzaihébian") 최장
#       분해 토큰 길이도 같이 본다 — 영어에서 병음으로 분해되는 단어는 짧다(to/so/you/man).
#   (c) PY-2(2026-09-05): **띄어쓰기된 병음**. (b)는 "붙여쓴 병음"만 상정해서 최장 토큰
#       7자를 요구했는데, 실제 학습앱은 음절을 띄어 쓴다("ni hao huan ying shi yong…").
#       그러면 토큰이 전부 2~4자라 (b)가 통째로 False가 되고, 성조부호는 13px에서 OCR이
#       날려 먹어 (a)도 안 걸린다 → 병음 줄이 영어로 확정돼 번역기에 그대로 들어갔고
#       ("ni hao, huan ying…" → "니 하오, 후안 이 쯔 옌"), ZH-1 진입 근거도 사라져
#       한자 화면을 통째로 못 읽었다. 짧은 토큰이라도 **여러 개가 연달아 전부** 병음
#       음절로 분해되면 그건 영어가 아니다.
#
# 실측 임계 스윕 (영어 6909줄 = 어린왕자 PDF + 프로젝트 영문 문서, 10단어/4단어 분할):
#     비율 0.7 / 최장 5  → 거짓양성 121줄   ← 영어 본문을 먹는다
#     비율 0.8 / 최장 5  → 거짓양성  15줄
#     비율 0.8 / 최장 7  → 거짓양성   6줄 (0.09%)  ← 채택
#     비율 0.9 / 최장 7  → 거짓양성   0줄, 그러나 실제 병음 줄도 절반을 놓친다
# 거짓양성을 더 줄이려면 비율 ↑. 병음이 새어나오면 비율 ↓ 또는 최장 길이 ↓.
PINYIN_FILTER_ENABLED = True
PINYIN_MIN_WORDS = 3              # 라틴 토큰이 이보다 적은 줄은 (b) 판정 자체를 안 한다
PINYIN_MIN_SEG_RATIO = 0.8        # 병음 음절로 완전 분해되는 토큰 비율 하한
PINYIN_MIN_LONG_TOKEN = 7         # 분해되는 토큰 중 최소 하나는 이 길이 이상이어야 한다
PINYIN_TONE_MIN_WORDS = 2         # (a) 강성 성조부호 경로에 필요한 최소 라틴 토큰 수
# (c) 띄어쓰기 병음 경로. 최장 토큰 대신 **분해되는 토큰의 개수**로 건다.
# 실측 스윕 (2026-09-05, 부정 코퍼스 55,156조각 = 영어 54,993 + 불어 91 + 독어 72,
#  10단어/4단어 분할. 긍정 = 실제 병음 16줄(성조 있는 것/OCR이 성조를 날린 것/붙여쓴 것)):
#     현행(c 없음)        재현 5/16   거짓양성 92 (0.167%)
#     개수 4 / 비율 0.90  재현 14/16  거짓양성 135 (0.245%)   ← 영어를 먹기 시작
#     개수 5 / 비율 0.90  재현 14/16  거짓양성  96 (0.174%)
#     개수 6 / 비율 0.90  재현 14/16  거짓양성  93 (0.169%)  ← 채택 (신규 FP 단 1건)
#     개수 7 / 비율 0.90  재현 13/16  거짓양성  92 (0.167%)
# 채택값이 새로 만든 거짓양성은 전 코퍼스에서 1건("to change a[i:ai] into" — 파이썬
# 소스 주석 조각)이고 불어·독어 거짓양성은 늘지 않았다(1건 → 1건, café/Saint-Exupéry
# 계열 회귀 없음). 놓치는 건 3토막짜리 인사말("ni hao ma") 정도다.
# 병음이 새어나오면 개수 ↓, 영어 본문이 사라지면 개수 ↑(또는 비율 ↑).
PINYIN_MIN_SPACED_TOKENS = 6
PINYIN_SPACED_MIN_RATIO = 0.90
# ZH-1이 중국어 화면으로 확정한 뒤에는 개수 하한을 낮춘다. chi_sim 패스가 병음 한 줄을
# 두 토막으로 쪼개 놓는 일이 흔해서다(실측: "nǐ hǎo, huān yíng shǐ yòng zhè ge yìng yòng"
# → "mihao, huan ying shi" + "zhé ge ying yong", 각각 4토큰이라 6에서 새어나갔다).
# 대가는 같은 코퍼스에서 zh 모드 거짓양성 1.358% → 1.434% (+42줄). zh 모드는 애초에
# 분해율 하한이 0.6이라 원래 이 구간이 느슨하고, 그 값이 유효한 건 중국어 화면뿐이다.
PINYIN_ZH_MIN_SPACED_TOKENS = 4
# ZH-1이 "이 화면은 중국어"라고 확정한 뒤에는 **분해율만** 느슨하게 쓴다. 성조부호가 다
# 뭉개진 줄("dajia xiaozhe pa 02116 hixiangpdshui_", 분해율 0.75)이 여기서 잡힌다.
# 토큰 수(3)와 최장 길이(7)는 **일부러 안 낮췄다** — 낮추면 "data"(da+ta), "menu"(me+nu)
# 같은 한 토막 영어 UI 라벨이 중국어 화면에서 사라진다. 대가로 "divi"(dìyī) 같은 한 토막
# 병음은 남지만, 그건 4글자짜리 박스 하나다.
PINYIN_ZH_MIN_SEG_RATIO = 0.6
PINYIN_STRONG_TONE_CHARS = frozenset(
    "āēīōūǖĀĒĪŌŪǕ"      # 1성 (마크론)
    "ǎěǐǒǔǚǍĚǏǑǓǙ"      # 3성 (캐런)
    "ǘǜǗǛ"              # ü 위의 2/4성 — 병음 외에는 거의 안 쓰인다
)

# 성모 × 운모 조합. 실제 표준 음절(약 410개)의 상위집합이라 조금 관대하지만,
# 위 실측에서 거짓양성이 0.09%라 손으로 410개를 적어 넣을 이유가 없다.
_PINYIN_INITIALS = ("", "zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l",
                    "g", "k", "h", "j", "q", "x", "r", "z", "c", "s", "y", "w")
_PINYIN_FINALS = ("iang", "iong", "uang", "ueng", "ang", "eng", "ing", "ong", "iao",
                  "ian", "uai", "uan", "van", "ai", "ei", "ao", "ou", "an", "en",
                  "er", "ia", "ie", "in", "iu", "ua", "uo", "ui", "un", "ue", "ve",
                  "vn", "a", "e", "i", "o", "u", "v")
_PINYIN_SYLLABLES = frozenset(i + f for i in _PINYIN_INITIALS for f in _PINYIN_FINALS)
_LATIN_TOKEN = re.compile(r"[A-Za-zÀ-ɏ]+")


@functools.lru_cache(maxsize=4096)
def _pinyin_segmentable(token: str) -> bool:
    """토큰 전체를 병음 음절의 연속으로 쪼갤 수 있는가 (성조부호는 벗겨서 비교)."""
    t = "".join(c for c in unicodedata.normalize("NFD", token)
                if unicodedata.category(c) != "Mn").lower().replace("ü", "v")
    if not t or not t.isalpha():
        return False
    ok = [True] + [False] * len(t)
    for i in range(1, len(t) + 1):
        ok[i] = any(ok[j] and t[j:i] in _PINYIN_SYLLABLES
                    for j in range(max(0, i - 6), i))
    return ok[len(t)]


def is_pinyin_line(text: str, zh_screen: bool = False) -> bool:
    """PY-1: 이 줄은 한자 위에 붙은 병음(발음표기)인가?

    zh_screen=True 는 ZH-1이 중국어 화면으로 확정한 상태 — 임계를 느슨하게 쓴다.
    """
    if not PINYIN_FILTER_ENABLED:
        return False
    tokens = _LATIN_TOKEN.findall(str(text or ""))
    if len(tokens) >= PINYIN_TONE_MIN_WORDS and any(
            ch in PINYIN_STRONG_TONE_CHARS for ch in text):
        return True
    if len(tokens) < PINYIN_MIN_WORDS:
        return False
    min_ratio = PINYIN_ZH_MIN_SEG_RATIO if zh_screen else PINYIN_MIN_SEG_RATIO
    seg = [t for t in tokens if _pinyin_segmentable(t)]
    if (len(seg) >= len(tokens) * min_ratio
            and max((len(t) for t in seg), default=0) >= PINYIN_MIN_LONG_TOKEN):
        return True
    # (c) 띄어쓰기 병음: 짧은 토막이 여러 개 **연달아 전부** 병음 음절로 분해된다.
    min_spaced = (PINYIN_ZH_MIN_SPACED_TOKENS if zh_screen
                  else PINYIN_MIN_SPACED_TOKENS)
    return (len(seg) >= min_spaced
            and len(seg) >= len(tokens) * PINYIN_SPACED_MIN_RATIO)


def drop_pinyin_lines(lines, confs=None, zh_screen=False):
    """번역 입력에서 병음 줄을 뺀다. 반환: (lines, confs, 버린 줄 수).

    버린 수를 돌려주는 이유는 ZH-1이 그걸 "이 화면은 중국어다"의 근거로 쓰기 때문이다.
    OCR 백엔드(Tesseract/Paddle) 공통 길목에서 한 번만 호출한다.
    """
    if not PINYIN_FILTER_ENABLED:
        return lines, confs, 0
    keep = [i for i, (text, _, _) in enumerate(lines)
            if not is_pinyin_line(text, zh_screen)]
    dropped = len(lines) - len(keep)
    if not dropped:
        return lines, confs, 0
    return ([lines[i] for i in keep],
            [confs[i] for i in keep] if confs else confs,
            dropped)


# --- ZH-1 튜닝 레버: 한자가 보이면 chi_sim 을 실제로 태운다 -------------------
# 실측(2026-08-11)이 뒤집은 통념: `kor` traineddata 는 한자(한자어 병기용)를 읽어 주지 않는다.
#     순수 중국어 화면(합성)  eng+kor → **0줄** / eng+kor+chi_sim → 4줄, CJK 55자
#     학습앱 스샷            eng+kor → 15줄이지만 CJK **0자** (병음 줄만 읽힘)
# 즉 auto 모드에서 중국어는 "잘못 읽히는" 게 아니라 **아예 안 읽힌다**.
#
# 그렇다고 chi_sim 을 상시 추가할 수는 없다. 실측 OCR 시간(같은 이미지):
#     영어 문서 doc_55   eng+kor 2545 ms → eng+kor+chi_sim 3502 ms  (+38%)
#     학습앱 스샷        eng+kor 1819 ms → eng+kor+chi_sim 3047 ms  (+68%)
# → 근거가 있을 때만 켠다. 근거 세 가지(진입):
#     (a) 1차 패스에서 병음 줄(PY-1)이 잡혔다 — 병음 병기 화면
#     (b) 1차 패스가 한 줄도 못 읽었다 — 한자만 있는 화면은 eng+kor 에 통째로 안 보인다
#     (c) ZH-2(2026-09-05): 1차 패스가 **화면의 글자 대부분을 못 읽고 남겼다**.
#         (a)(b)만 두면 진입이 병음 판정 하나에 사실상 걸린다 — 병음이 없는 순수 한자
#         화면에 영어 UI 라벨이 한 줄만 있어도 (b)가 깨져서 chi_sim 이 영원히 안 돈다
#         (실측: zh_pinyin 시료 한자 5문단 중 0개 인식). 그래서 병음 판정과 **무관한**
#         신호를 하나 더 둔다: 이진화한 잉크 픽셀 중 인식된 라인 박스가 덮지 못한 비율.
#         시료 실측(진입 판정 지점 그대로 = 1차 패스 + LC-1b kor 재스캔 후):
#             영어 문서/UI/다중문맥/크기계단  0.000 ~ 0.197  ← 삽화·아이콘까지 포함한 값
#             한국어 화면 ko_screen           0.000          ← kor 재스캔이 먼저 읽어낸다
#             중국어 화면 zh_pinyin           1.000
#         → 임계 0.6 이 두 무리 한가운데다. 사진/게임 화면은 이 신호를 통과할 수 있지만,
#           그때는 chi_sim 패스가 한자를 못 찾고 EMPTY_COOLDOWN 으로 떨어진다.
# 한 번 켜지면 STICKY_FRAMES 동안 1차 패스를 건너뛰고 바로 chi_sim 을 태운다(프레임당
# 왕복 2회를 1회로). 그 사이 프레임에서 근거(CJK 글자 수 또는 병음 줄)가 다시 보이면
# 카운터가 갱신되고, 안 보이면 줄어들다 0에서 원래 언어로 복귀한다.
OCR_ZH_RESCAN_ENABLED = True
OCR_ZH_TESS_LANG = "chi_sim"      # 번체 화면을 주로 본다면 "chi_tra"
OCR_ZH_MIN_CJK_CHARS = 4          # 유지 판정용 한자 수 하한. 한국어 문서의 산발적 한자(1~2자)에
                                  # 끌려가지 않을 만큼만 높다. ↑면 중국어 화면을 더 빨리 놓는다
# 이 값은 **이탈 지연**만 정한다. 중국어를 보는 동안에는 매 프레임 근거가 갱신돼 최대치에
# 붙어 있기 때문이다. 중국어 → 영어로 옮기면 이만큼의 프레임이 +38% 느린 chi_sim 패스를 탄다.
# 크게 잡을 이유가 없다. 다만 1~2는 너무 짧다 — 중국어 페이지를 그림 영역까지 스크롤해
# 한 프레임 한자가 안 보이는 순간 바로 풀려서 다음 프레임에 진입 왕복을 또 한다.
OCR_ZH_STICKY_FRAMES = 8          # ↓면 언어 전환에 빨리 반응, ↑면 중국어 화면에서 재진입이 덜 잦다
OCR_ZH_EMPTY_COOLDOWN = 20        # 근거 (b)/(c)로 재OCR 했는데 한자가 안 나오면 이만큼 재탐색을 쉰다.
                                  # 영상/게임처럼 "글자 없이 계속 변하는" 화면에서 매 프레임
                                  # chi_sim 패스를 덧돌리는 걸 막는다 (실측 +140 ms/프레임).
OCR_ZH_UNREAD_INK_RATIO = 0.6     # (c) 진입 임계. ↓면 중국어를 더 잘 잡지만 그림 많은 화면에서
                                  # 헛 chi_sim 패스가 늘고, ↑면 한자 화면을 놓치기 쉬워진다
OCR_ZH_INK_MIN_PIXELS = 400       # 잉크가 이보다 적은(= 거의 빈) 화면은 비율이 노이즈다 — 판정 생략
# ponytail: 판정 전에 이미지를 축소해 값싸게 하려다 **되돌렸다**. 512px로 줄이면 다크 테마
# UI의 가는 획이 INTER_AREA에 뭉개져 Otsu가 패널 경계를 잉크로 잡는다 — ui_en_dense 시료가
# 0.041 → 0.659로 튀어 임계 0.6을 넘겼다(영어 화면에서 chi_sim 패스 +1.1초).
# 원본 해상도 그대로가 1080p에서 8 ms다. 축소는 비용이 아니라 정확도를 깎는 쪽이었다.

# 한자·가나·CJK 구두점·전각 기호. 한글(U+AC00~)은 **일부러 뺐다** — 한국어는 띄어쓰기를 쓴다.
_CJK_SPANS = ((0x3000, 0x303F), (0x3040, 0x30FF), (0x3400, 0x4DBF),
              (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0xFF01, 0xFF60))


def is_cjk_char(ch: str) -> bool:
    o = ord(ch)
    return any(lo <= o <= hi for lo, hi in _CJK_SPANS)


def cjk_char_count(text: str) -> int:
    return sum(1 for ch in str(text or "") if is_cjk_char(ch))


def ink_mask(gray):
    """잉크 픽셀 마스크(bool). Otsu 소수 클래스라 다크 테마도 그대로 통과한다.

    잉크가 거의 없는 화면(빈 화면)에서는 None — 그 위의 비율·통계는 전부 노이즈다.
    """
    if gray is None or not getattr(gray, "size", 0):
        return None
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = bw == 0
    if ink.mean() > 0.5:          # 다크 테마: 배경이 검은 쪽이므로 소수 클래스를 잉크로 본다
        ink = ~ink
    if int(ink.sum()) < OCR_ZH_INK_MIN_PIXELS:
        return None
    return ink


def ink_glyph_px(gray):
    """OU-2: 이미지의 **글리프 높이 중앙값**(px). OCR 결과를 하나도 쓰지 않는다.

    OU-1 확대 트리거는 원래 "1차 패스가 읽은 word 높이 중앙값"만 봤는데, 1차가 무너지면
    그 값이 같이 오염된다 — 실측(2026-09-07): 10px Georgia/Calibri/Times/Impact 를
    **28px 로 보고**해서 확대가 가장 필요한 조합에서만 확대가 꺼졌다.
    잉크 연결 성분의 높이는 오독과 무관하게 이미지에서만 나오는 값이라 2차 의견이 된다.
    실측 대응표(글꼴 px → 반환값): 10→4~8 · 12→6~9 · 14→7 · 18→9~10 · 24→12~13.
    글자가 거의 없으면 None(판정하지 않음). 비용 8~9 ms/1920x1080 — 그래서 호출부는
    **높이 규칙이 이미 확대를 켠 프레임에서는 부르지 않는다.**
    """
    ink = ink_mask(gray)
    if ink is None:
        return None
    n, _labels, stats, _cent = cv2.connectedComponentsWithStats(
        ink.astype(np.uint8), 8)
    if n <= 1:
        return None
    h = stats[1:, cv2.CC_STAT_HEIGHT]
    # 점·테두리선·큰 도형은 글자가 아니다. 남은 것의 중앙값만 본다.
    keep = h[(h >= OCR_INK_GLYPH_MIN_PX) & (h <= OCR_INK_GLYPH_MAX_PX)]
    if keep.size < OCR_INK_MIN_GLYPHS:
        return None
    return float(np.median(keep))


def unread_ink_ratio(gray, boxes) -> float:
    """ZH-2 근거 (c): 화면의 잉크 중 인식된 라인 박스가 덮지 못한 비율 (0~1).

    "글자가 있는데 안 읽혔다"를 언어와 무관하게, 결정론적으로, 값싸게 재는 신호다.
    잉크가 거의 없는 화면은 판정하지 않고 0.0 을 돌려준다 — 빈 화면에서 비율은 노이즈다.
    """
    ink = ink_mask(gray)
    if ink is None:
        return 0.0
    total = int(ink.sum())
    covered = np.zeros(ink.shape, dtype=bool)
    for (left, top, right, bottom) in boxes:
        covered[max(0, int(top)):int(bottom) + 1, max(0, int(left)):int(right) + 1] = True
    return 1.0 - float((ink & covered).sum()) / total


def join_ocr_words(words) -> str:
    """OCR word 들을 한 줄 문자열로 잇는다. CJK 경계에는 공백을 넣지 않는다.

    Tesseract 는 한자를 글자마다 별개 word 로 준다. 예전처럼 " ".join 하면
    "大 家 笑 着" 가 되고, 번역기는 이걸 다른 문장으로 읽는다. 실측(2026-08-11):
        "大 家 笑 着 ， 跑 着 ， 互 相 泼水 。" → "큰 집은 웃고, 달리고, 서로 물을 뿌린다"
        "大家笑着，跑着，互相泼水。"            → "모두가 웃고, 달리고, 서로 물을 뿌린다"
    """
    out = ""
    for w in words:
        w = str(w)
        if not w:
            continue
        if out and not (is_cjk_char(out[-1]) or is_cjk_char(w[0])):
            out += " "
        out += w
    return out


# --- LM-1 튜닝 레버: 라인 뭉침/거대 폰트 (7차) --------------------------------
# 실측 재현(개발 PC, 2026-08-02): Tesseract `--psm 6`은
#   (a) 라인 안에 장식 글자/도형(그래프 축을 "|"로 오인, 높이 81px)이 섞이면
#       `font_size = max(word height)`가 그 이상치를 그대로 물려받아 73px 폰트 + 81px 박스가 되고,
#   (b) 좌우 두 단(column)이 같은 y에 있으면 255px 간격을 건너뛰어 한 라인으로 병합한다
#       (bbox 폭 645px — "화면 1/3을 덮는 거대 폰트로 뭉침"의 정체).
# → 라인 내 word 높이 중앙값을 기준으로 이상치 word를 빼고, 큰 가로 간격에서 라인을 쪼갠다.
OCR_WORD_HEIGHT_OUTLIER_RATIO = 2.5   # word 높이가 라인 중앙값의 이 배를 넘으면 라인에서 제외. ↓면 더 공격적
OCR_LINE_SPLIT_GAP_RATIO = 3.0        # 단어 사이 가로 간격이 (중앙값 높이 × 이 값) 초과면 별개 라인. ↓면 더 잘게 쪼갬

# --- LM-1b 튜닝 레버: 라인 양끝 조각 떼어내기 --------------------------------
# 글자 왼쪽의 아이콘/장식이 'a', '®', '=', '**', 'ww' 같은 조각으로 읽혀 라인 앞에 붙는다.
# 피해가 셋이다: (1) 번역 입력 오염 (2) bbox 왼쪽이 아이콘까지 늘어나 문단 병합의 좌측
# 정렬 판정이 깨짐 (3) 오버레이 박스가 아이콘을 덮음.
#
# 실측(게임 툴팁, 2026-08-05) — 단어 사이 간격이 두 무리로 깨끗하게 갈린다:
#     본문 단어 사이      : 6 px (예외 없이 전부)
#     아이콘/조각 경계    : 10, 14, 30, 32, 40, 66, 174, 186, 210, 370, 376, 448 px
# 글자 높이 중앙값이 14 px 이므로 임계 0.6 (= 8.4 px) 이 두 무리 사이에 놓인다.
# LM-1(3.0 = 42 px)은 큰 간격만 쪼개므로 10~42 px 구간이 그대로 새어 들어왔다.
#
# 판정은 "큰 간격"과 "조각스러움"의 **논리곱**이다. 간격만 보면 들여쓰기가 잘리고,
# 생김새만 보면 문장 속 짧은 단어가 잘린다.
# 조각스러움을 "2글자 이하"로 잡았다가 문서에서 정상 단어 'as a', 'is' 가 잘리는 걸
# 실측으로 잡았다 → **글자 높이가 라인 중앙값에서 벗어남**으로 바꿨다.
# 아이콘 조각의 높이는 4, 8, 20, 24, 28, 32 px (본문 14 px) 로 확실히 벗어난다.
OCR_EDGE_TRIM_ENABLED = True
OCR_EDGE_TRIM_GAP_RATIO = 0.6     # 이 값(× 높이 중앙값) 이상 벌어진 지점이 조각 경계 후보
OCR_EDGE_TRIM_HEIGHT_LO = 0.55    # 글자 높이가 중앙값 대비 이 밖이면 "조각"으로 본다
OCR_EDGE_TRIM_HEIGHT_HI = 1.4
OCR_EDGE_TRIM_SHORT_CONF = 60     # 2글자 이하 단어는 confidence 가 이 아래일 때만 조각으로 본다.
                                  # 실측: 조각 'ra' 40 / 'os' 16 / 'ww' 35 / 'ee' 38
                                  #       본문 'as' 'is' 는 90 이상.
OCR_EDGE_TRIM_MAX = 4             # 한 라인에서 최대 몇 번 떼어낼지 (폭주 방지)


# --- PM-1 튜닝 레버: 문단 병합 -----------------------------------------------
# OCR은 "줄"을 준다. 번역기는 "문장"을 원한다. 줄 단위로 넘기면 한 문장이 조각나
# 조각마다 따로 번역된다. 실측(게임 툴팁, 2026-08-05):
#     "Enemies have a chance to drop a dish that grants"  → "적들은 접시를 떨어뜨릴 기회를 가지고 있어요"
#     "one stack (max 50). Each stack grants +0.1%"       → (문맥 없이 따로 번역)
# 줄바꿈으로 이어지는 줄들을 문단으로 합쳐 **한 번에** 번역한다.
#
# 문단이 끝나는 신호 두 가지를 쓴다:
#   (a) 앞 줄이 문장 종결부호로 끝남
#   (b) 앞 줄이 뒤 줄보다 확연히 짧음 — 줄바꿈된 줄은 오른쪽 여백까지 꽉 찬다.
#       제목("Cutlery")이 본문에 빨려들어가는 걸 막는 게 이 규칙이다.
PARA_MERGE_ENABLED = True
PARA_MERGE_GAP_RATIO = 0.9        # 세로 간격 ≤ (앞 줄 높이 × 이 값) 이어야 같은 문단. ↑면 더 잘 합침
PARA_MERGE_OVERLAP_RATIO = 0.6    # 세로로 이만큼(× 높이)까지 겹쳐도 다음 줄로 인정.
                                  # bbox는 글자의 위/아래 삐침에 따라 줄끼리 몇 px 겹친다
                                  # (실측: 어린왕자 본문에서 다음 줄 top 이 앞 줄 bottom 보다
                                  # 7px 위, 줄 높이 14px → 0.5). 겹침을 0으로 두면 거기서 끊긴다.
                                  # 넉넉해도 안전한 이유: 나란히 놓인 두 단은 좌측 정렬 규칙에서
                                  # 이미 걸러진다.
PARA_MERGE_LEFT_TOL_RATIO = 1.0   # 왼쪽 x 차이 ≤ (앞 줄 높이 × 이 값). 들여쓰기 허용치
PARA_MERGE_FONT_LO = 0.75         # 글자 크기 비율 하한 (뒤/앞)
PARA_MERGE_FONT_HI = 1.35         # 글자 크기 비율 상한
PARA_MERGE_MIN_WIDTH_RATIO = 0.6  # 앞 줄 폭 ≥ 뒤 줄 폭 × 이 값 이어야 "줄바꿈된 줄". ↓면 제목까지 합쳐짐
PARA_MERGE_MAX_LINES = 20         # 한 문단 최대 줄 수 (폭주 방지).
                                  # 12는 부족했다 — 어린왕자 본문 한 문단이 14줄이라
                                  # 여기서 잘려 마지막 두 줄이 따로 번역됐다.
                                  # 진짜 문단 끝 판정은 종결부호/폭 규칙이 한다.
PARA_MERGE_STOP_CHARS = (".", "!", "?", ":", ";", "。", "！", "？", "…")


def _ocr_line_is_noise(text: str) -> bool:
    """NZ-1: 문자 구성만으로 판정하는 노이즈 라인 필터 (confidence는 호출부에서 별도 검사)."""
    compact = "".join(str(text).split())
    if len(compact) < OCR_LINE_MIN_CHARS:
        return True
    real = sum(1 for ch in compact if ch.isalnum())
    if real < len(compact) * OCR_LINE_MIN_LETTER_RATIO:
        return True
    if OCR_LINE_REQUIRE_ALPHA and not any(ch.isalpha() for ch in compact):
        return True
    return False

# B-14: ISO 639-1 → Tesseract lang 코드 매핑. 사용자가 선택한 src 언어에 따라 동적 결정.
LANG_TO_TESS = {
    "en": "eng",
    "ko": "kor",
    "fr": "fra",
    "de": "deu",
    "zh": "chi_sim",
    "ja": "jpn",
}

def _check_tesseract_langs():
    """첫 실행 진단: 사용 가능한 traineddata 목록."""
    if not _TESSERACT_CMD:
        return set()
    try:
        return set(pytesseract.get_languages(config=""))
    except Exception as e:
        print(f"[WARN] Tesseract 언어 조회 실패: {e}")
        return set()

AVAILABLE_TESS_LANGS = _check_tesseract_langs()
print(f"[INFO] Tesseract 사용 가능 언어: {sorted(AVAILABLE_TESS_LANGS) or '(없음)'}")
if "eng" not in AVAILABLE_TESS_LANGS:
    print("[ERROR] 'eng' traineddata 누락 — 영어 OCR 불가. eng.traineddata 설치 필요.")


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def environment_summary(probe_torch: bool = False) -> tuple[str, list[str]]:
    """일반 사용자에게 보여줄 시작 진단 요약.

    LZ-1: `probe_torch=False`(기본)면 **torch를 import하지 않는다**. 이 함수는
    `BackgroundController.__init__` = 창이 뜨기 전 메인 스레드에서 불리는데, torch import
    한 번이 2.3초라 그대로 창 지연이 된다. 이미 import된 뒤라면 공짜로 실제 device를 읽고,
    아니면 "준비 중"으로 두고 모델 로드가 끝난 뒤 엔진이 다시 부른다(`probe_torch=True`).
    """
    problems = []
    tess_ok = bool(_TESSERACT_CMD and "eng" in AVAILABLE_TESS_LANGS)
    if not _TESSERACT_CMD:
        problems.append("Tesseract 미설치")
    elif "eng" not in AVAILABLE_TESS_LANGS:
        problems.append("eng.traineddata 누락")

    paddle_ok = _module_available("paddleocr") and _module_available("paddle")
    if not paddle_ok:
        problems.append("PaddleOCR 선택 패키지 미설치")

    # torch는 번역 계층 소유다. 진단 한 줄 때문에 OCR 모듈이 torch에 묶이지 않도록 지연 import.
    torch = sys.modules.get("torch")
    if torch is None and probe_torch:
        import torch
    model_device = ("CUDA" if torch.cuda.is_available() else "CPU") if torch else "준비 중"
    status = f"환경: Tesseract {'OK' if tess_ok else '점검'} | PaddleOCR {'OK' if paddle_ok else '옵션 미설치'} | 번역 {model_device}"
    return status, problems


# --- 선택 OCR 백엔드: PaddleOCR v3 / PP-OCRv5 (O-1) -----------------------
PADDLE_LANG_MAP = {
    "en": "en",
    "ko": "korean",
    "fr": "fr",
    "de": "de",
    "zh": "ch",
    "ja": "japan",
    "es": "es",
    "it": "it",
    "pt": "pt",
    "ru": "ru",
    "ar": "ar",
    "hi": "hi",
    "th": "th",
}


class PaddleOcrManager:
    """
    PP-OCRv5 선택 백엔드.
    - import/모델 로드는 사용자가 PaddleOCRv5를 선택했을 때만 수행한다.
    - 실패하면 호출자가 Tesseract로 폴백한다.
    """
    def __init__(self):
        self._models = {}
        self._lock = threading.Lock()
        self._warned_missing = False

    @staticmethod
    def lang_from_src(src: str) -> str:
        if src == "auto":
            return os.environ.get("COCKTAIL_PADDLE_LANG", "ch")
        return PADDLE_LANG_MAP.get(src, "en")

    def _get_model(self, lang: str):
        if lang in self._models:
            return self._models[lang]
        with self._lock:
            if lang in self._models:
                return self._models[lang]
            try:
                from paddleocr import PaddleOCR
            except Exception as e:
                if not self._warned_missing:
                    print(f"[WARN] PaddleOCR 사용 불가: {e}. Tesseract로 폴백합니다.")
                    self._warned_missing = True
                return None

            kwargs = {
                "lang": lang,
                "ocr_version": "PP-OCRv5",
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": False,
                "text_rec_score_thresh": 0.30,
                # Windows CPU에서 paddle_static+oneDNN 경로가 실패할 수 있어 동적 paddle 엔진을 기본으로 둔다.
                "engine": os.environ.get("COCKTAIL_PADDLE_ENGINE", "paddle"),
                "enable_mkldnn": False,
            }
            device_hint = os.environ.get("COCKTAIL_PADDLE_DEVICE")
            if device_hint:
                kwargs["device"] = device_hint

            print(f"[INFO] PaddleOCR PP-OCRv5 로드 중: lang={lang}")
            try:
                self._models[lang] = PaddleOCR(**kwargs)
            except Exception as e:
                print(f"[WARN] PaddleOCR 초기화 실패(lang={lang}): {e}. Tesseract로 폴백합니다.")
                self._models[lang] = None
            return self._models[lang]

    @staticmethod
    def _result_payload(result_obj):
        payload = getattr(result_obj, "json", None)
        if callable(payload):
            payload = payload()
        if not isinstance(payload, dict):
            return {}
        return payload.get("res", payload)

    @staticmethod
    def _as_list(value):
        if value is None:
            return []
        if hasattr(value, "tolist"):
            return value.tolist()
        return list(value)

    def recognize(self, image_rgb: np.ndarray, src_hint: str):
        lang = self.lang_from_src(src_hint)
        model = self._get_model(lang)
        if model is None:
            return None

        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        results = model.predict(image_bgr)
        out = []
        for result_obj in results:
            payload = self._result_payload(result_obj)
            texts = self._as_list(payload.get("rec_texts"))
            boxes = self._as_list(payload.get("rec_boxes"))
            scores = self._as_list(payload.get("rec_scores"))
            for i, text in enumerate(texts):
                if not text or not str(text).strip():
                    continue
                score = float(scores[i]) if i < len(scores) else 1.0
                if score < 0.30 or i >= len(boxes):
                    continue
                box = boxes[i]
                if len(box) != 4:
                    continue
                left, top, right, bottom = [int(v) for v in box]
                if right <= left or bottom <= top:
                    continue
                font_size = max(10, bottom - top)
                out.append((str(text).strip(), (left, top, right, bottom), font_size))
        out.sort(key=lambda item: (item[1][1], item[1][0]))
        return out


PADDLE_OCR = PaddleOcrManager()


# --- Phase 2/3: UIA 어댑터 — Windows UI Automation으로 텍스트 직접 추출 -------
# 비전 V-1: OCR 없이 표준 UI 텍스트를 정확히 읽기. 게임/Custom GDI는 폴백 OCR.
# uiautomation은 optional — 미설치 시 UIA 모드만 비활성, 본 앱은 정상 동작.
class UIATextProvider:
    """Windows UI Automation 트리에서 텍스트+bbox를 추출. 지연 import."""

    # Name이 있는 텍스트성 컨트롤만 수집 (모든 컨트롤 다 보면 노이즈)
    _TEXTUAL_CONTROL_TYPES = {
        "TextControl", "DocumentControl", "EditControl", "HyperlinkControl",
        "ButtonControl", "MenuItemControl", "TreeItemControl", "ListItemControl",
        "TabItemControl", "GroupControl", "HeaderItemControl", "DataItemControl",
        "StatusBarControl", "TitleBarControl",
    }

    def __init__(self):
        self._auto = None
        self._tried = False

    def _ensure(self):
        if self._auto is not None:
            return self._auto
        if self._tried:
            return None
        self._tried = True
        if sys.platform != "win32":
            print("[INFO] UIA는 Windows 전용 — 비활성")
            return None
        try:
            import uiautomation as auto  # type: ignore
            self._auto = auto
            print("[INFO] UIA 어댑터 활성화 — uiautomation 로드됨")
            return auto
        except ImportError:
            print("[INFO] uiautomation 미설치 — UIA 모드 비활성. (pip install uiautomation)")
            return None
        except Exception as e:
            print(f"[WARN] UIA 어댑터 초기화 실패: {e}")
            return None

    def is_available(self) -> bool:
        return self._ensure() is not None

    def extract(self, hwnd) -> list:
        """반환: [(text, (left, top, right, bottom), font_size), ...] — 화면 절대 좌표."""
        auto = self._ensure()
        if not auto or not hwnd:
            return []
        try:
            ctrl = auto.ControlFromHandle(int(hwnd))
            if not ctrl:
                return []
            results = []
            self._walk(ctrl, results, depth=0, max_depth=10, total_limit=300)
            return results
        except Exception as e:
            print(f"[WARN] UIA extract 실패: {e}")
            return []

    def _walk(self, elem, results, depth=0, max_depth=10, total_limit=300):
        if depth > max_depth or len(results) >= total_limit:
            return
        try:
            name = (getattr(elem, "Name", "") or "").strip()
            ctype = getattr(elem, "ControlTypeName", "")
        except Exception:
            return
        # 짧은 단어(1글자)는 제외 — 노이즈
        if name and len(name) >= 2 and ctype in self._TEXTUAL_CONTROL_TYPES:
            try:
                br = elem.BoundingRectangle
                if br and br.right > br.left and br.bottom > br.top:
                    bbox = (int(br.left), int(br.top), int(br.right), int(br.bottom))
                    # font_size 추정 — bbox 높이 기반
                    h = bbox[3] - bbox[1]
                    font_size = max(10, min(28, h // 2))
                    results.append((name, bbox, font_size))
            except Exception:
                pass
        try:
            children = elem.GetChildren()
        except Exception:
            children = []
        for c in children:
            self._walk(c, results, depth + 1, max_depth, total_limit)


UIA_PROVIDER = UIATextProvider()

"""
Cocktail Overlay Translator (commercial-ready)

라이선스 정책 (project_commercial_decisions.md 참고):
- UI: PySide6 (LGPL — closed-source 상용 OK)
- 번역 모델 1차: Helsinki-NLP/opus-mt-tc-big-en-ko (Apache 2.0, en→ko 최적)
- 번역 모델 2차 폴백: facebook/m2m100_418M (MIT, any-to-any 100언어)
- OCR: Tesseract (Apache 2.0). 한국어팩 별도 설치 필요 (errors.md E-6).

설계 원칙 (errors.md):
- 성능: batched generate, fp16(GPU), inference_mode, frame-skip 해시, LRU 캐시.
- UX: 모델 lazy load — UI는 즉시 뜨고 모델은 백그라운드 로드 (P-9).
- 자기 캡처 피드백 차단 (B-10 SetWindowDisplayAffinity), 사용자가 토글 가능 (E-9).
- 워커 ↔ paint race 차단 (B-11 Signal/Slot).
- 한국어 박스 자동 적응 (B-12 height 동적 확장 + B-13 폰트 축소 + alpha 235).

코드 수정 전 errors.md를 한 번 읽을 것 (재발 방지 체크리스트).
"""

import os

# DP-1: Qt 논리 스케일링 OFF → 앱 전 좌표를 물리 픽셀 하나로 통일한다.
# GetWindowRect / ImageGrab / OCR bbox는 전부 물리 픽셀인데, Qt 지오메트리와
# QCursor.pos()만 논리 좌표면 175% 보조 모니터가 섞이는 순간 가상 데스크톱에
# 어느 모니터에도 속하지 않는 좌표 구간이 생기고, 가상 데스크톱 전체를 덮는
# OverlayWindow가 보조 모니터 배율로 스케일돼 자막이 화면 밖으로 밀린다.
# QGuiApplication 인스턴스 생성 전에 설정해야 효력이 있으므로 모듈 최상단에 둔다
# (QApplication 생성은 CocktailAppController.__init__ 한 곳뿐 — errors.md BG-2).
# DPI awareness(퍼모니터 인식)는 Qt가 그대로 유지 → OS 블러 스트레칭 없음. E-7의 후속 결정.
os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "0")

import sys
import time
import ctypes
import ctypes.wintypes  # T-2: GetWindowRect의 RECT 구조체용
import threading
import asyncio
import unicodedata
import importlib.util
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import cv2
import torch
import pytesseract
from PIL import ImageGrab
# SL-1: langdetect 퇴역. 라틴 스크립트 추측이 오판(cy/hu/de/pt/so/ar)의 근원이라
# auto 경로에서 호출을 없앴다 → import 자체도 제거(사이클당 ~0.4s 절감).
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
)

from PySide6 import QtWidgets
from PySide6.QtCore import Qt, QRect, QPoint, Signal, Slot, QTimer, QSettings, QObject
from PySide6.QtGui import (
    QPainter, QColor, QPen, QFont, QKeySequence, QShortcut,
    QIcon, QPixmap, QCursor, QAction,
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QComboBox, QPushButton, QSizeGrip, QLabel,
    QSystemTrayIcon, QMenu, QWidget,
)


# --- Phase 1 (T-1, T-2): 캡처 모드 상수 + 글로벌 단축키 optional 의존성 -----
# 모드별 캡처 영역 결정: red_rect는 모든 모드의 공통 캡처 영역으로 동작 (워커 루프에서 자동 갱신).
MODE_WINDOW = "window"    # 활성 창의 영역 자동 캡처 (기본 — 비전 V-1)
MODE_BOX = "box"          # 사용자가 직접 빨간 박스로 영역 지정 (좁은 영역 정밀 번역용)
MODE_HOVER = "hover"      # 마우스 커서 주변 영역 캡처

# 콤보 첫 항목 = 기본 모드. 사용자 피드백(2026-05-04) "윈도우 자동 번역" 반영.
_MODE_LABELS = {
    "활성 창": MODE_WINDOW,
    "마우스 hover": MODE_HOVER,
    "영역 박스": MODE_BOX,
}
_MODE_LABELS_REVERSE = {v: k for k, v in _MODE_LABELS.items()}

# SC-1 튜닝 레버: 활성 창 모드의 캡처 영역을 "활성 창이 있는 모니터 1개"로 제한.
# 바탕화면(Progman/WorkerW)이나 가상 데스크톱 전체를 덮는 창이 활성이면 창 rect가
# 모든 모니터 합집합이 되어 전 화면을 OCR/렌더링하던 문제(오버레이가 모든 모니터에 꽉 참).
# False로 두면 구 동작(창 rect 전체). 창을 두 모니터에 걸쳐 쓰는 사용자는 False.
CLAMP_WINDOW_TO_ACTIVE_SCREEN = True

# 글로벌 단축키 — keyboard 패키지 (MIT, optional). 미설치 시 로컬 단축키만 동작.
try:
    import keyboard as _keyboard  # type: ignore
    _KEYBOARD_OK = True
except Exception as _e:
    _keyboard = None
    _KEYBOARD_OK = False
    print(f"[INFO] 글로벌 단축키 라이브러리 keyboard 미설치 — 로컬 단축키만 사용. ({_e})")


# --- Windows 캡처 토글 (B-10, E-9) -----------------------------------------
WDA_NONE = 0x00
WDA_EXCLUDEFROMCAPTURE = 0x11

def set_window_capture_affinity(hwnd: int, exclude: bool) -> bool:
    """exclude=True → 캡처에서 제외(B-10), False → 다시 캡처 가능(스크린샷용, E-9)."""
    if sys.platform != "win32":
        return False
    try:
        affinity = WDA_EXCLUDEFROMCAPTURE if exclude else WDA_NONE
        ok = ctypes.windll.user32.SetWindowDisplayAffinity(int(hwnd), affinity)
        return bool(ok)
    except Exception as e:
        print(f"[WARN] SetWindowDisplayAffinity 실패: {e}")
        return False


# --- TM-1 (A3, 7차): 오버레이를 Z-order 최상단으로 되돌리기 -------------------
HWND_TOPMOST = -1
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOACTIVATE = 0x0010   # 포커스를 뺏지 않는다 (사용자가 치던 창을 방해하면 안 됨)

def set_window_topmost(hwnd: int) -> bool:
    """항상-위를 재적용한다. 나중에 뜬 다른 topmost 창이 우리를 덮은 경우 복구용.

    전체화면 독점(exclusive fullscreen) 앱 위에는 원리상 어떤 창도 못 올라간다 — 한계.
    """
    if sys.platform != "win32":
        return False
    try:
        fn = ctypes.windll.user32.SetWindowPos
        # D-1 교훈: 64비트에서 HWND는 포인터 크기다. argtypes 없이 -1(HWND_TOPMOST)을 넘기면
        # 32비트 int로 전달돼 상위 비트가 쓰레기가 된다.
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                       ctypes.c_uint]
        fn.restype = ctypes.c_bool
        return bool(fn(int(hwnd), HWND_TOPMOST, 0, 0, 0, 0,
                       SWP_NOSIZE | SWP_NOMOVE | SWP_NOACTIVATE))
    except Exception as e:
        print(f"[WARN] SetWindowPos(HWND_TOPMOST) 실패: {e}")
        return False


# --- GT-1 (A6, 7차): 워커 스레드에서 쓸 수 있는 커서 좌표 --------------------
def cursor_pos_physical():
    """마우스 커서의 **물리 픽셀** 좌표 (x, y).

    `QCursor.pos()`는 GUI 전용 API라 워커 스레드에서 호출하면 Qt 규약 위반이다.
    win32 `GetCursorPos`는 스레드 무관하고, 프로세스가 per-monitor DPI aware라
    반환값이 물리 픽셀 — DP-1 좌표 규약과 그대로 정합한다.
    """
    if sys.platform == "win32":
        try:
            pt = ctypes.wintypes.POINT()
            if ctypes.windll.user32.GetCursorPos(ctypes.byref(pt)):
                return int(pt.x), int(pt.y)
        except Exception as e:
            print(f"[WARN] GetCursorPos 실패: {e}")
    p = QCursor.pos()   # 비-Windows 폴백 (이 경로는 메인 스레드에서만 쓰인다)
    return int(p.x()), int(p.y())


# --- W-1 (Phase 1B): 마우스 통과 토글 (활성 창/hover 모드 매끄러운 작동용) -----
GWL_EXSTYLE = -20
WS_EX_TRANSPARENT = 0x00000020
WS_EX_LAYERED = 0x00080000

def set_mouse_through(hwnd: int, on: bool) -> bool:
    """on=True 면 윈도우 영역 전체에서 마우스 이벤트가 통과 (밑 창으로 전달).
    윈도우 자신은 여전히 그려짐. 컨트롤 클릭은 불가능해지므로 단축키로 다시 OFF.
    """
    if sys.platform != "win32":
        return False
    try:
        user32 = ctypes.windll.user32
        try:
            _get = user32.GetWindowLongPtrW
            _set = user32.SetWindowLongPtrW
        except AttributeError:
            _get = user32.GetWindowLongW
            _set = user32.SetWindowLongW
        style = _get(int(hwnd), GWL_EXSTYLE)
        new_style = style | WS_EX_LAYERED  # 반투명 그리기 유지
        if on:
            new_style |= WS_EX_TRANSPARENT
        else:
            new_style &= ~WS_EX_TRANSPARENT
        _set(int(hwnd), GWL_EXSTYLE, new_style)
        return True
    except Exception as e:
        print(f"[WARN] 마우스 통과 설정 실패: {e}")
        return False


# --- S-1 (Phase 5, 보안 원칙 #3): 민감 영역 자동 OFF 휴리스틱 ---------------
# 비밀번호/로그인/결제/은행 키워드가 활성 창 제목/클래스 이름에 보이면 OCR/번역 일시 중지.
SENSITIVE_KEYWORDS = (
    # 영문
    "password", "passwd", "sign in", "sign-in", "signin",
    "log in", "log-in", "login", "logon",
    "checkout", "payment", "billing", "credit card", "card number",
    "cvv", "cvc", "ssn", "social security",
    "two-factor", "2fa", "verification code",
    # 한국어 보편
    "비밀번호", "암호", "로그인", "회원가입", "결제", "카드번호",
    "주민등록", "공인인증", "인증번호", "보안카드", "이체",
    # 한국 주요 은행/금융
    "kb국민", "kb스타뱅킹", "신한은행", "신한쏠", "우리은행", "우리원뱅킹",
    "하나은행", "하나원큐", "농협", "기업은행", "카카오뱅크", "토스",
)

def _get_window_text_class(hwnd: int):
    user32 = ctypes.windll.user32
    title = ""
    try:
        length = user32.GetWindowTextLengthW(int(hwnd))
        if length:
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(int(hwnd), buf, length + 1)
            title = buf.value or ""
    except Exception:
        pass
    cls_name = ""
    try:
        cls_buf = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(int(hwnd), cls_buf, 256)
        cls_name = cls_buf.value or ""
    except Exception:
        pass
    return title, cls_name

def is_sensitive_window(hwnd) -> bool:
    if sys.platform != "win32" or not hwnd:
        return False
    try:
        title, cls = _get_window_text_class(int(hwnd))
        haystack = f"{title}\n{cls}".lower()
        if not haystack.strip():
            return False
        return any(kw.lower() in haystack for kw in SENSITIVE_KEYWORDS)
    except Exception:
        return False


# --- Tesseract 경로 탐색 (E-1, E-2, E-6) -----------------------------------
def _resolve_tesseract():
    cand = [
        os.environ.get("TESSERACT_CMD"),
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

# 배포 시 앱 폴더 내 ./tessdata 가 있으면 그걸 사용 (인스톨러 컴포넌트로 설치된 traineddata)
_APP_DIR = os.path.dirname(os.path.abspath(getattr(sys, "argv", [""])[0] or "."))
_LOCAL_TESSDATA = os.path.join(_APP_DIR, "tessdata")
if os.path.isdir(_LOCAL_TESSDATA):
    os.environ["TESSDATA_PREFIX"] = _LOCAL_TESSDATA
    print(f"[INFO] TESSDATA_PREFIX = {_LOCAL_TESSDATA}")

TESS_CONFIG = "--oem 1 --psm 6"

# --- NZ-1 튜닝 레버: OCR 노이즈 라인 필터 -----------------------------------
# UI 부스러기(아이콘 글자, 시계, 테두리 잡음)까지 박스로 그려 "중구난방"으로 보이던 문제.
# 정상적인 짧은 문장이 죽지 않도록 임계는 보수적으로 낮게 시작한다.
OCR_LINE_MIN_CONF = 35.0          # 라인 평균 word confidence 최소치. 부스러기가 남으면 ↑(50~60), 진짜 문장이 사라지면 ↓
OCR_LINE_MIN_CHARS = 2            # 공백 제외 이보다 짧은 라인은 버림 (아이콘 글자 1자)
OCR_LINE_MIN_LETTER_RATIO = 0.5   # 공백 제외 문자 중 영숫자/한글/CJK 비율 최소치. 기호 잡음 컷. ↑면 더 엄격
OCR_LINE_REQUIRE_ALPHA = True     # 글자(isalpha) 없는 라인(시계 "10:30", 페이지 번호, "100%") 제외.
                                  # 숫자만 있는 원문도 번역하고 싶으면 False.

# --- OU-1 튜닝 레버: 작은 글씨 적응 업스케일 ---------------------------------
# 화면상 글자 높이가 12px 아래로 내려가면 Tesseract가 통째로 무너진다(캡션/각주).
# 1차 OCR 결과의 라인 높이 "중앙값"이 임계 미만일 때만 2배 확대해 재OCR 한다.
# → 큰 글씨 화면에서는 재OCR 비용이 아예 발생하지 않는다(트리거가 그것을 보장).
OCR_UPSCALE_TRIGGER_PX = 18       # 라인 높이 중앙값(원본 px) 임계. ↑면 더 자주 확대(느려짐), ↓면 덜 확대
OCR_UPSCALE_FACTOR = 2            # 확대 배율. 3 이상은 비용 대비 이득이 거의 없다

# --- SL-1 설계 결정: auto 모드 언어 라우팅 단순화 (OS-1/LG-1/M-6/M-7 퇴역) ----
# 자동 "다국어 추측"(OSD script 판별 + langdetect)은 실측에서 오판 공장이었다.
# 영어 PDF를 OSD가 Arabic(conf 3.95)으로 읽어 신뢰도 게이트를 통과했고, langdetect는
# 같은 화면에서 cy/hu/de/pt/so를 뱉었다. 오판 1건의 대가는 쓰레기 OCR + m2m100 5초 배치.
# → auto는 이제 결정론만 쓴다: OCR 언어 eng+kor 고정, 라인 언어는 문자 스크립트로
#   확정되는 것만(한글/가나/CJK/키릴…) 인정하고 라틴·불명은 전부 en.
#   다른 언어를 읽어야 하면 사용자가 src 콤보에서 명시한다(그 경로는 그대로 동작).
OCR_LANGS_AUTO_DEFAULT = ("eng", "kor")   # 환경변수 COCKTAIL_OCR_LANGS_AUTO로 오버라이드 가능

# --- RP-1 튜닝 레버: 신경망 번역 반복 붕괴 방지 ------------------------------
# 잡음 입력(OCR 쓰레기)에서 seq2seq가 같은 구를 무한 반복하는 고전적 붕괴를 막는다.
# 3은 한국어 정상 문장(어미·조사 반복)에도 걸릴 수 있어 4를 기본으로 둔다.
NO_REPEAT_NGRAM_SIZE = 4          # ↓3이면 더 강하게 억제(정상 문장 왜곡 위험), ↑5면 관대

# --- LM-1 튜닝 레버: 라인 뭉침/거대 폰트 (7차) --------------------------------
# 실측 재현(개발 PC, 2026-08-02): Tesseract `--psm 6`은
#   (a) 라인 안에 장식 글자/도형(그래프 축을 "|"로 오인, 높이 81px)이 섞이면
#       `font_size = max(word height)`가 그 이상치를 그대로 물려받아 73px 폰트 + 81px 박스가 되고,
#   (b) 좌우 두 단(column)이 같은 y에 있으면 255px 간격을 건너뛰어 한 라인으로 병합한다
#       (bbox 폭 645px — "화면 1/3을 덮는 거대 폰트로 뭉침"의 정체).
# → 라인 내 word 높이 중앙값을 기준으로 이상치 word를 빼고, 큰 가로 간격에서 라인을 쪼갠다.
OCR_WORD_HEIGHT_OUTLIER_RATIO = 2.5   # word 높이가 라인 중앙값의 이 배를 넘으면 라인에서 제외. ↓면 더 공격적
OCR_LINE_SPLIT_GAP_RATIO = 3.0        # 단어 사이 가로 간격이 (중앙값 높이 × 이 값) 초과면 별개 라인. ↓면 더 잘게 쪼갬

# --- DG-1 튜닝 레버: 표시 직전 최종 게이트 (7차) ------------------------------
# 원칙: "확신 없으면 숨긴다". 깨진 자막을 보여주는 것보다 안 보여주는 게 낫다.
# NZ-1(원문 문자 구성)은 OCR이 "말이 되는 오독"을 하면 뚫린다(초소형 캡션 →
# "마우 냄비 thang에"류 그럴싸한 쓰레기). 번역까지 끝난 뒤 마지막으로 한 번 더 거른다.
DISPLAY_GATE_ENABLED = True            # False면 게이트 전체 무효(디버그용)
DISPLAY_TINY_LINE_PX = 12              # 원문 라인 높이(물리 px)가 이 미만이면 "극소 라인"
DISPLAY_TINY_LINE_MIN_CONF = 70.0      # 극소 라인은 평균 conf가 이 이상일 때만 신뢰. ↑면 더 많이 숨김
DISPLAY_MIN_TGT_SCRIPT_RATIO = 0.3     # 번역문 글자 중 목표 언어 스크립트 비율 하한 (한국어 목표인데 한글 극소 = 실패)
DISPLAY_LEN_COLLAPSE_MIN_SRC = 20      # 원문이 이 길이 이상일 때만 길이 붕괴 검사
DISPLAY_LEN_COLLAPSE_RATIO = 0.15      # 번역문/원문 길이 하한. 한국어는 보통 0.5배 이상이라 0.15는 보수적

# 목표 언어별로 "번역이 실제로 일어났다면 나와야 하는" 스크립트.
_LANG_TO_SCRIPTS = {
    "ko": {"HANGUL"},
    "ja": {"HIRAGANA", "KATAKANA", "CJK"},
    "zh": {"CJK"},
    "ru": {"CYRILLIC"},
    "en": {"LATIN"},
    "fr": {"LATIN"},
    "de": {"LATIN"},
}

# --- FT-1 (A4): 오버레이 폰트 스택 ------------------------------------------
# "Arial" 하드코딩은 한글 글리프가 없어 Qt 폴백에 맡겨져 자간/굵기가 들쭉날쭉했다.
# 한글 우선 스택 → 첫 항목부터 설치된 것을 쓴다(Qt setFamilies가 폴백 체인을 처리).
OVERLAY_FONT_FAMILIES = ("Malgun Gothic", "Noto Sans KR", "Segoe UI", "Arial")

# --- TM-1 (A3): 오버레이 topmost 주기 재적용 --------------------------------
# 다른 항상-위 창(런처/미디어 플레이어 컨트롤 등)이 나중에 뜨면 Z-order에서 우리를 덮는다.
# 0으로 두면 주기 재적용을 끈다(수동 raise_만).
OVERLAY_TOPMOST_INTERVAL_MS = 2000


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


def environment_summary() -> tuple[str, list[str]]:
    """일반 사용자에게 보여줄 시작 진단 요약."""
    problems = []
    tess_ok = bool(_TESSERACT_CMD and "eng" in AVAILABLE_TESS_LANGS)
    if not _TESSERACT_CMD:
        problems.append("Tesseract 미설치")
    elif "eng" not in AVAILABLE_TESS_LANGS:
        problems.append("eng.traineddata 누락")

    paddle_ok = _module_available("paddleocr") and _module_available("paddle")
    if not paddle_ok:
        problems.append("PaddleOCR 선택 패키지 미설치")

    model_device = "CUDA" if torch.cuda.is_available() else "CPU"
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


# --- 번역 백엔드: HybridTranslator (M-5) -----------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_FP16 = device.type == "cuda"
print(f"[INFO] device={device}, fp16={USE_FP16}")


def _load_local_first(loader, model_name, **kwargs):
    """LD-1: 로컬 HF 캐시 우선 로드.

    `local_files_only=True`로 먼저 시도해 HF Hub 네트워크 왕복(캐시가 있어도 수십 초
    걸리던 rate-limit/파일 목록 확인)을 건너뛴다. 캐시가 없거나 불완전하면 예외가 나므로
    그때만 네트워크 경로로 재시도한다 — 첫 설치는 그대로 동작.
    `HF_HUB_OFFLINE` 같은 전역 오프라인 강제는 첫 설치를 망가뜨리므로 쓰지 않는다.
    """
    try:
        return loader(model_name, local_files_only=True, **kwargs)
    except Exception as e:
        print(f"[INFO] 로컬 캐시 미스 → 네트워크 로드 ({model_name}): {type(e).__name__}")
        return loader(model_name, **kwargs)


class _BaseTranslator:
    name = "base"
    def translate_batch(self, texts, src, tgt):
        raise NotImplementedError


class OpusMtTranslator(_BaseTranslator):
    """
    Helsinki-NLP/opus-mt-* — Apache 2.0. 언어쌍별 전용 모델.
    en→ko 등 자주 쓰는 쌍에서 m2m100/NLLB 대비 작고 빠르고 품질 우수.
    """
    name = "opus-mt"

    def __init__(self, model_name: str, progress_cb=None):
        print(f"[INFO] opus-mt 로드 중: {model_name}")
        if progress_cb: progress_cb("토크나이저 로드 중")
        # LD-1: 로컬 캐시 우선 (네트워크 체크로 로드가 수십 초 걸리던 문제)
        self.tok = _load_local_first(AutoTokenizer.from_pretrained, model_name)
        if progress_cb: progress_cb("모델 다운로드/로드 중")
        m = _load_local_first(AutoModelForSeq2SeqLM.from_pretrained, model_name).to(device)
        if USE_FP16:
            m = m.half()
        m.eval()
        self.model = m
        self.src_sp = self._load_source_spm(model_name)  # M-8
        if progress_cb: progress_cb("워밍업 중")
        try:
            with torch.inference_mode():
                enc = self.tok(["hello"], return_tensors="pt", padding=True).to(device)
                self.model.generate(**enc, max_new_tokens=8, num_beams=1)
        except Exception as e:
            print(f"[WARN] opus-mt 워밍업 실패: {e}")

    def _load_source_spm(self, model_name):
        """M-8: sepvoc(소스/타깃 vocab 분리) opus-mt 모델은 repo의 tokenizer_config.json이
        separate_vocabs=false로 잘못 적혀 있어 AutoTokenizer가 소스 문장을 타깃 vocab으로
        인코딩한다(→ 대부분 <unk>, 번역 결과가 쓰레기). source.spm으로 직접 인코딩한다.
        공유 vocab 모델이면 None(기존 경로 유지), 로드 실패해도 None으로 degrade."""
        try:
            import sentencepiece as spm
            from huggingface_hub import snapshot_download
            # LD-1: 로컬 스냅샷 우선 — 캐시가 있어도 매번 14개 파일 목록을 Hub에 묻던 경로.
            snapshot_dir = _load_local_first(snapshot_download, model_name)
            sp = spm.SentencePieceProcessor(
                model_file=os.path.join(snapshot_dir, "source.spm")
            )
            vocab = self.tok.get_vocab()
            size = sp.get_piece_size()
            missing = sum(1 for i in range(0, size, 16) if sp.id_to_piece(i) not in vocab)
            if missing * 16 < size * 0.2:
                return None  # 소스 조각이 토크나이저 vocab에 있음 = 공유 vocab, 기존 경로가 정상
            print(f"[INFO] sepvoc 모델 감지, source.spm 직접 인코딩 (M-8): {model_name}")
            return sp
        except Exception as e:
            print(f"[WARN] source.spm 로드 실패, 기존 토크나이저 경로 사용 (M-8): {e}")
            return None

    def _encode_with_spm(self, texts):
        pad = self.tok.pad_token_id
        seqs = [self.src_sp.encode(t)[:255] + [self.tok.eos_token_id] for t in texts]
        n = max(len(s) for s in seqs)
        return {
            "input_ids": torch.tensor(
                [s + [pad] * (n - len(s)) for s in seqs], dtype=torch.long).to(device),
            "attention_mask": torch.tensor(
                [[1] * len(s) + [0] * (n - len(s)) for s in seqs], dtype=torch.long).to(device),
        }

    def translate_batch(self, texts, src, tgt):
        # opus-mt는 언어쌍 전용이라 src/tgt는 라우팅 검증용으로만 사용 (모델은 이미 결정됨)
        if self.src_sp is not None:
            enc = self._encode_with_spm(texts)  # M-8
        else:
            enc = self.tok(
                texts, return_tensors="pt", padding=True, truncation=True, max_length=256
            ).to(device)
        in_len = enc["input_ids"].shape[1]
        max_new = min(256, int(in_len * 1.6) + 16)
        with torch.inference_mode():
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new,
                num_beams=1,
                do_sample=False,
                early_stopping=False,
                no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,  # RP-1
            )
        return self.tok.batch_decode(out, skip_special_tokens=True)


class M2M100Translator(_BaseTranslator):
    """
    facebook/m2m100_418M — MIT. any-to-any 100언어 폴백.
    opus-mt에 등록 안 된 쌍에서만 호출됨.
    """
    name = "m2m100"

    def __init__(self, model_name: str = "facebook/m2m100_418M"):
        print(f"[INFO] m2m100 로드 중: {model_name}")
        # LD-1: 로컬 캐시 우선
        self.tok = _load_local_first(M2M100Tokenizer.from_pretrained, model_name)
        m = _load_local_first(
            M2M100ForConditionalGeneration.from_pretrained, model_name).to(device)
        if USE_FP16:
            m = m.half()
        m.eval()
        self.model = m
        try:
            with torch.inference_mode():
                self.tok.src_lang = "en"
                enc = self.tok(["hello"], return_tensors="pt", padding=True).to(device)
                self.model.generate(
                    **enc,
                    forced_bos_token_id=self.tok.get_lang_id("ko"),
                    max_new_tokens=8,
                    num_beams=1,
                )
        except Exception as e:
            print(f"[WARN] m2m100 워밍업 실패: {e}")

    def translate_batch(self, texts, src, tgt):
        # src/tgt = ISO 639-1 ("en", "ko", "fr", ...)
        self.tok.src_lang = src
        enc = self.tok(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=256
        ).to(device)
        in_len = enc["input_ids"].shape[1]
        max_new = min(256, int(in_len * 1.6) + 16)
        with torch.inference_mode():
            out = self.model.generate(
                **enc,
                forced_bos_token_id=self.tok.get_lang_id(tgt),
                max_new_tokens=max_new,
                num_beams=1,
                do_sample=False,
                early_stopping=False,
                no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,  # RP-1
            )
        return self.tok.batch_decode(out, skip_special_tokens=True)


class HybridTranslator:
    """
    M-5: 라우팅
      - BILINGUAL_MODELS에 등록된 (src, tgt) → opus-mt (작고 빠름)
      - 그 외 → m2m100 (any-to-any 폴백)
    P-9: lazy load — __init__은 가볍고, 모델은 첫 사용/preload 시 로드.
    """
    name = "hybrid"

    # ISO 639-1 (src, tgt) → opus-mt 모델. 자주 쓰는 쌍을 늘리려면 여기 등록 (라이선스 확인 필수).
    BILINGUAL_MODELS = {
        ("en", "ko"): "Helsinki-NLP/opus-mt-tc-big-en-ko",
        # 예: ("ko", "en"): "Helsinki-NLP/opus-mt-tc-big-ko-en",
    }

    def __init__(self):
        self._bilingual = {}        # (src, tgt) -> OpusMtTranslator
        self._multilingual = None   # M2M100Translator (lazy)
        self._load_lock = threading.Lock()

    def _ensure_bilingual(self, src, tgt, progress_cb=None):
        key = (src, tgt)
        if key in self._bilingual:
            return self._bilingual[key]
        model_name = self.BILINGUAL_MODELS.get(key)
        if not model_name:
            return None
        with self._load_lock:
            if key not in self._bilingual:
                self._bilingual[key] = OpusMtTranslator(model_name, progress_cb=progress_cb)
            return self._bilingual[key]

    def _ensure_multilingual(self):
        if self._multilingual is not None:
            return self._multilingual
        with self._load_lock:
            if self._multilingual is None:
                self._multilingual = M2M100Translator("facebook/m2m100_418M")
            return self._multilingual

    def preload_default(self, progress_cb=None):
        """en→ko opus-mt 미리 로드 (백그라운드 호출용, P-9)."""
        self._ensure_bilingual("en", "ko", progress_cb=progress_cb)
        if progress_cb: progress_cb("준비 완료")

    def translate_batch(self, texts, src, tgt):
        b = self._ensure_bilingual(src, tgt)
        if b is not None:
            return b.translate_batch(texts, src, tgt)
        return self._ensure_multilingual().translate_batch(texts, src, tgt)


# 가벼운 인스턴스만 만들기 (모델은 아직 로드 안 됨, P-9)
TRANSLATOR = HybridTranslator()


# --- 번역 캐시 (P-5) --------------------------------------------------------
class LRU(OrderedDict):
    def __init__(self, capacity=512):
        super().__init__()
        self.capacity = capacity

    def get_or_none(self, key):
        if key in self:
            self.move_to_end(key)
            return self[key]
        return None

    def put(self, key, value):
        if key in self:
            self.move_to_end(key)
        self[key] = value
        while len(self) > self.capacity:
            self.popitem(last=False)


TRANS_CACHE = LRU(capacity=1024)
CACHE_LOCK = threading.Lock()


# --- 언어 코드 (ISO 639-1) — M-5: NLLB BCP47에서 변경 ---------------------
LANG_MAP = {
    "Korean":   "ko",
    "English":  "en",
    "French":   "fr",
    "German":   "de",
    "Chinese":  "zh",
    "Japanese": "ja",
}

# m2m100 tokenizer가 직접 처리할 수 있는 언어 코드.
# auto 감지에서 UI 콤보에 없는 언어가 나와도 multilingual fallback으로 라우팅하기 위함.
M2M100_LANGS = {
    "af", "am", "ar", "ast", "az", "ba", "be", "bg", "bn", "br", "bs", "ca",
    "ceb", "cs", "cy", "da", "de", "el", "en", "es", "et", "fa", "ff", "fi",
    "fr", "fy", "ga", "gd", "gl", "gu", "ha", "he", "hi", "hr", "ht", "hu",
    "hy", "id", "ig", "ilo", "is", "it", "ja", "jv", "ka", "kk", "km", "kn",
    "ko", "lb", "lg", "ln", "lo", "lt", "lv", "mg", "mk", "ml", "mn", "mr",
    "ms", "my", "ne", "nl", "no", "ns", "oc", "or", "pa", "pl", "ps", "pt",
    "ro", "ru", "sd", "si", "sk", "sl", "so", "sq", "sr", "ss", "su", "sv",
    "sw", "ta", "th", "tl", "tn", "tr", "uk", "ur", "uz", "vi", "wo", "xh",
    "yi", "yo", "zh", "zu",
}

_SCRIPT_TO_LANG = {
    "HANGUL": "ko",
    "HIRAGANA": "ja",
    "KATAKANA": "ja",
    "CJK": "zh",
    "ARABIC": "ar",
    "CYRILLIC": "ru",
    "HEBREW": "he",
    "DEVANAGARI": "hi",
    "THAI": "th",
    "GREEK": "el",
    "BENGALI": "bn",
    "GURMUKHI": "pa",
    "GUJARATI": "gu",
    "TAMIL": "ta",
    "MALAYALAM": "ml",
    "KHMER": "km",
    "LAO": "lo",
    "MYANMAR": "my",
}

def _unicode_script(ch: str) -> str:
    """OCR 문자열의 Unicode script를 대략 분류한다. 숫자/기호는 COMMON으로 무시."""
    if not ch or not ch.strip():
        return "COMMON"
    name = unicodedata.name(ch, "")
    cat = unicodedata.category(ch)
    if not cat.startswith("L"):
        return "COMMON"
    if "HANGUL" in name:
        return "HANGUL"
    if "HIRAGANA" in name:
        return "HIRAGANA"
    if "KATAKANA" in name:
        return "KATAKANA"
    if "CJK UNIFIED" in name or "CJK COMPATIBILITY" in name:
        return "CJK"
    for script in _SCRIPT_TO_LANG:
        if script in name:
            return script
    if "LATIN" in name:
        return "LATIN"
    if cat.startswith("L"):
        return "OTHER"
    return "COMMON"


def detect_script(text: str) -> tuple[str, dict[str, int]]:
    """문자열의 지배적인 Unicode script를 반환한다. 라틴은 SL-1에서 무조건 en."""
    counts = {}
    for ch in text or "":
        script = _unicode_script(ch)
        if script == "COMMON":
            continue
        counts[script] = counts.get(script, 0) + 1
    if not counts:
        return "UNKNOWN", counts

    # 일본어는 CJK 한자와 kana가 섞이는 경우가 많으므로 kana가 보이면 ja로 본다.
    if counts.get("HIRAGANA", 0) or counts.get("KATAKANA", 0):
        return "HIRAGANA" if counts.get("HIRAGANA", 0) >= counts.get("KATAKANA", 0) else "KATAKANA", counts

    dominant = max(counts.items(), key=lambda kv: kv[1])[0]
    if dominant == "LATIN":
        non_latin = {k: v for k, v in counts.items() if k != "LATIN"}
        if non_latin:
            script, n = max(non_latin.items(), key=lambda kv: kv[1])
            total = sum(counts.values())
            if script in _SCRIPT_TO_LANG and (n >= 2 or n / total >= 0.35):
                return script, counts
    return dominant, counts


def identify_language(text: str, fallback: str = "en") -> str:
    """
    SL-1: 문자 스크립트로 확정되는 언어만 인정한다(한글→ko, 가나→ja, CJK→zh, 키릴→ru…).
    라틴·불명 스크립트는 추측하지 않고 전부 fallback(기본 en) — 이 한 줄이
    "영어 화면을 cy/hu/de/pt/so로 오판 → m2m100 폴백 → 5초 배치" 사슬을 끊는다.
    다른 라틴 언어(프랑스어 등)를 읽어야 하면 사용자가 src 콤보에서 명시한다.
    """
    script, _ = detect_script(text)
    return _SCRIPT_TO_LANG.get(script, fallback)


def assign_region_languages(ocr_results, src_lang_hint):
    """
    src=auto면 라인마다 결정론적 스크립트 판정만 적용한다(SL-1).
    사용자가 source 언어를 명시하면 그대로 강제한다.
    (M-7 region 투표 / LG-1 지배 언어 스냅은 라틴 추측을 보정하려고 있던 장치라
     추측이 사라진 지금은 존재 이유가 없어 함께 퇴역.)
    """
    if src_lang_hint != "auto":
        return [src_lang_hint] * len(ocr_results)
    return [identify_language(text) for text, _, _ in ocr_results]


def display_gate_reject(src_text, tgt_text, tgt_lang, line_px=None, conf=None):
    """DG-1 (7차): 표시 직전 최종 게이트. 거부 사유 문자열을 반환하면 그 줄은 **그리지 않는다**.

    NZ-1은 OCR 원문의 문자 구성만 본다. 초소형 캡션(7px대)에서 OCR이 "말이 되는 오독"을
    하면(예: "마우 냄비 thang에") 원문도 번역문도 형태상 멀쩡해 보여 전부 통과한다.
    그래서 번역이 끝난 뒤 "이 줄을 믿을 근거가 있는가"를 마지막으로 한 번 더 묻는다.

    규칙은 셋 다 **거짓 양성(정상 줄을 숨김)이 나기 어려운 쪽**으로 잡았다:
      1. 극소 라인 + 낮은 원문 confidence — 둘 다 성립할 때만. 큰 글씨는 conf가 낮아도 통과.
      2. 목표 언어 스크립트 비율 — 한국어 목표인데 번역문에 한글이 거의 없다 = 번역이 실패했다.
      3. 길이 붕괴 — 긴 원문이 초단문으로 뭉개진 경우(모델이 입력을 버린 것).
    """
    if not DISPLAY_GATE_ENABLED:
        return None
    tgt = (tgt_text or "").strip()
    src = (src_text or "").strip()
    if not tgt:
        return "empty"

    # 1) 극소 라인 + 저 confidence. 확대 재OCR(OU-1)로도 못 살린 영역이 여기 걸린다.
    if (line_px is not None and 0 < line_px < DISPLAY_TINY_LINE_PX
            and conf is not None and 0 <= conf < DISPLAY_TINY_LINE_MIN_CONF):
        return f"tiny-line({int(line_px)}px)/low-conf({conf:.0f})"

    # 2) 번역문이 목표 언어 스크립트를 거의 안 담고 있으면 번역 자체가 안 일어난 것.
    wanted = _LANG_TO_SCRIPTS.get(tgt_lang)
    if wanted:
        scripts = [s for s in (_unicode_script(ch) for ch in tgt) if s != "COMMON"]
        if not scripts:
            return "no-letters"
        hit = sum(1 for s in scripts if s in wanted)
        if hit / len(scripts) < DISPLAY_MIN_TGT_SCRIPT_RATIO:
            return f"tgt-script({hit}/{len(scripts)})"

    # 3) 원문과 무관한 초단문(길이 붕괴).
    if (len(src) >= DISPLAY_LEN_COLLAPSE_MIN_SRC
            and len(tgt) < len(src) * DISPLAY_LEN_COLLAPSE_RATIO):
        return f"len-collapse({len(tgt)}/{len(src)})"
    return None


def _cache_key_text(text: str) -> str:
    """RT-1: 캐시 키용 보수적 정규화 — 연속 공백 collapse + 양끝 strip.

    OCR은 같은 정적 화면도 프레임마다 공백/줄바꿈이 미세하게 흔들려 원문 그대로를 키로 쓰면
    캐시가 계속 miss → 1~3초마다 수십 줄 재번역. 표시용 원문/번역문은 원본을 그대로 쓴다.
    (문자 자체를 건드리는 정규화는 다른 언어를 망칠 수 있어 하지 않는다.)
    """
    return " ".join(str(text).split())


def batch_translate(texts, src_lang_hint, tgt_lang):
    """
    여러 라인을 한 번의 generate 호출로 번역.
    src_lang_hint: 'auto', ISO 639-1, 또는 라인별 ISO list.
    tgt_lang: ISO 639-1.
    동일 src끼리 묶어 모델별로 1회 호출.
    """
    if not texts:
        return []

    results = [None] * len(texts)
    pending_idx = []
    pending_texts = []
    pending_keys = []   # RT-1: 캐시 저장용 정규화 텍스트 (모델 입력은 pending_texts 원본)
    pending_src = []

    src_list = src_lang_hint if isinstance(src_lang_hint, list) else None

    for i, t in enumerate(texts):
        if src_list is not None:
            src = src_list[i] if i < len(src_list) else "en"
        else:
            src = src_lang_hint if src_lang_hint != "auto" else identify_language(t)
        if src == tgt_lang:
            results[i] = t
            continue
        # RT-1: 캐시 키는 정규화 텍스트. OCR 공백 흔들림이 캐시 miss를 만들지 않게.
        norm = _cache_key_text(t)
        key = (src, tgt_lang, norm)
        with CACHE_LOCK:
            cached = TRANS_CACHE.get_or_none(key)
        if cached is None:
            # Phase 4: 영구 캐시(DPAPI 암호화)에서도 조회. 메모리 캐시로 승격.
            try:
                disk = PERSIST_CACHE.get(src, tgt_lang, norm)
            except Exception:
                disk = None
            if disk is not None:
                cached = disk
                with CACHE_LOCK:
                    TRANS_CACHE.put(key, disk)
        if cached is not None:
            results[i] = cached
            continue
        pending_idx.append(i)
        pending_texts.append(t)
        pending_keys.append(norm)
        pending_src.append(src)

    if not pending_idx:
        return results

    groups = {}
    for j, src in enumerate(pending_src):
        groups.setdefault(src, []).append(j)

    for src, js in groups.items():
        sub_texts = [pending_texts[j] for j in js]
        decoded = TRANSLATOR.translate_batch(sub_texts, src, tgt_lang)
        for local, j in enumerate(js):
            tgt_text = decoded[local]
            results[pending_idx[j]] = tgt_text
            with CACHE_LOCK:
                TRANS_CACHE.put((src, tgt_lang, pending_keys[j]), tgt_text)
            # Phase 4: 영구 캐시도 갱신 (저장은 closeEvent/주기적으로)
            try:
                PERSIST_CACHE.put(src, tgt_lang, pending_keys[j], tgt_text)
            except Exception:
                pass

    return results


# --- 화면 변화 감지 (P-4 / F-1) ---------------------------------------------
# F-1 튜닝 레버: 16x16 그레이 셀 중 "가장 크게 변한 셀"의 변화량이 이 값 미만이면
# 같은 프레임으로 보고 건너뛴다. 평균 기준(구 P-4)은 자막 한 줄 변화를 삼켰다.
FRAME_DIFF_THRESHOLD = 8
_HASH_HEADER_BYTES = 16  # int32 x 4 (캡처 영역 좌표)


def cheap_hash(img_np: np.ndarray, rect=None) -> bytes:
    small = cv2.resize(img_np, (16, 16), interpolation=cv2.INTER_AREA)
    if small.ndim == 3:
        # B-8: PIL ImageGrab 결과는 RGB (BGR 아님).
        small = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    # F-1(a): 캡처 영역이 바뀌면 내용이 같아도 다른 프레임이다 (창 이동 시 자막 잔상 방지).
    header = np.array(rect if rect else (0, 0, 0, 0), dtype=np.int32).tobytes()
    return header + small.tobytes()


# RT-2 튜닝 레버: 오버레이 상태 비교 시 bbox를 이 픽셀 단위로 반올림한다.
# OCR bbox가 프레임마다 1~2px 흔들려도 같은 상태로 보고 재그리기를 생략(깜빡임 감소).
# 올리면 더 둔감(자막이 조금 움직여도 갱신 안 함), 내리면 더 민감(0/1이면 사실상 무효).
STATE_BBOX_ROUND_PX = 4


def _state_signature(state, round_px=STATE_BBOX_ROUND_PX):
    """RT-2: (원문, 번역문, 반올림 bbox) 기준의 실질 동일성 지문."""
    r = max(1, int(round_px))
    sig = []
    for entry in state or []:
        try:
            src_text, tgt_text, bbox, _fsz = entry
            sig.append((src_text, tgt_text, tuple(int(v) // r for v in bbox)))
        except Exception:
            sig.append(repr(entry))
    return tuple(sig)


def hash_distance(a: bytes, b: bytes) -> int:
    if a is None or b is None or len(a) != len(b):
        return 1 << 30
    if a[:_HASH_HEADER_BYTES] != b[:_HASH_HEADER_BYTES]:
        return 1 << 30  # 캡처 영역 변경 = 무조건 재처리
    arr_a = np.frombuffer(a, dtype=np.uint8, offset=_HASH_HEADER_BYTES).astype(np.int16)
    arr_b = np.frombuffer(b, dtype=np.uint8, offset=_HASH_HEADER_BYTES).astype(np.int16)
    # F-1(b): 평균이 아니라 셀 단위 최대 변화량 — 국소 변화(자막 한 줄)에 민감.
    return int(np.abs(arr_a - arr_b).max())


# --- PySide6 UI (E-10) -----------------------------------------------------
# PySide6/Qt6은 high-DPI를 자동 활성화 (E-7 정책과 일치).


class CaptureRegionController:
    """Own capture mode and capture-region geometry.

    BG-3: settings window resize must not double as region editing. This controller
    is the bridge toward a dedicated RegionEditor while preserving the old callers.

    BG-5: ``red_rect`` is stored in **screen-absolute coordinates**.
    Worker / OverlayWindow read it directly without depending on the settings
    window's geometry. Settings-window movement no longer drags the capture
    region, which matches the "background translator" UX.

    BG-6: depends only on an ``is_self_hwnd`` callback for the self/overlay
    guard. The ``owner`` widget reference is gone, so the controller no longer
    knows about specific UI class internals.
    """

    def __init__(self, is_self_hwnd):
        self._is_self_hwnd = is_self_hwnd
        self.capture_mode = MODE_WINDOW
        self.hover_size = (400, 200)
        # BG-5: absolute screen coordinates. Default is a primary-screen-friendly
        # rectangle (settings panel previously sat at (140,140) with a widget rect
        # at (100,75) → (240,215)) so existing users see continuity.
        self.red_rect = QRect(240, 215, 600, 375)
        # GT-1/MS-1 (7차): 화면 rect 캐시. Qt 화면 API는 GUI 스레드 전용이라
        # 워커가 직접 부르면 안 된다(핫플러그 시 크래시 가능) → 메인 스레드가 갱신하고
        # 워커·OverlayWindow는 이 캐시만 읽는다. 좌표는 절대/물리 픽셀,
        # right/bottom은 **exclusive**(Qt의 마지막 픽셀 규약을 여기서 한 번만 흡수).
        self.screen_rects = []
        self.primary_rect = None
        self.refresh_screen_rects()

    def refresh_screen_rects(self):
        """MS-1: 모니터 구성이 바뀔 때 메인 스레드에서만 호출한다 (생성 시 + screenAdded/Removed/geometryChanged)."""
        try:
            rects = []
            for s in QApplication.screens():
                g = s.geometry()
                rects.append((g.left(), g.top(), g.right() + 1, g.bottom() + 1))
            primary = QApplication.primaryScreen()
            if primary is not None:
                pg = primary.geometry()
                self.primary_rect = (pg.left(), pg.top(), pg.right() + 1, pg.bottom() + 1)
            self.screen_rects = rects
        except Exception as e:
            print(f"[WARN] 화면 rect 캐시 갱신 실패: {e}")
        return self.screen_rects

    def virtual_desktop_rect(self):
        """W-4: 캐시된 화면들의 합집합 (l, t, r_excl, b_excl). 캐시가 비면 None."""
        rects = self.screen_rects
        if not rects:
            return None
        return (min(r[0] for r in rects), min(r[1] for r in rects),
                max(r[2] for r in rects), max(r[3] for r in rects))

    def foreground_window_rect(self):
        """Windows 활성 창의 절대 좌표. 우리 자신/오버레이면 None."""
        if sys.platform != "win32":
            return None
        try:
            user32 = ctypes.windll.user32
            hwnd = user32.GetForegroundWindow()
            if not hwnd:
                return None
            if self._is_self_hwnd(int(hwnd)):
                return None
            rect = ctypes.wintypes.RECT()
            if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                return None
            return (int(rect.left), int(rect.top), int(rect.right), int(rect.bottom))
        except Exception as e:
            print(f"[WARN] GetForegroundWindow 실패: {e}")
            return None

    def mouse_hover_rect(self):
        """마우스 커서 주변 절대 좌표 영역.

        GT-1: 이 메서드는 워커 스레드에서 호출된다 → `QCursor.pos()`(GUI API) 금지.
        `GetCursorPos`는 스레드 무관 + 물리 픽셀이라 DP-1 규약과도 맞는다.
        """
        cx, cy = cursor_pos_physical()
        w, h = self.hover_size
        return (int(cx - w // 2), int(cy - h // 2),
                int(cx + w // 2), int(cy + h // 2))

    def screen_rect_at(self, x, y):
        """SC-1: 주어진 절대 좌표가 속한 모니터 1개 (l, t, r_excl, b_excl). 없으면 primary.

        GT-1: 구현이 Qt의 화면 조회 API(GUI 전용) 직접 호출이었으나, 이 함수는 워커
        스레드에서 불린다 → 메인 스레드가 채워 둔 `screen_rects` 캐시만 읽는다.
        """
        x, y = int(x), int(y)
        for r in self.screen_rects:
            if r[0] <= x < r[2] and r[1] <= y < r[3]:
                return r
        return self.primary_rect

    def update_for_mode(self):
        """BG-5: red_rect를 화면 절대 좌표로 직접 갱신.

        MODE_BOX은 사용자가 정한 절대 좌표 영역을 그대로 둔다.
        MODE_WINDOW/MODE_HOVER는 활성 창/마우스 절대 좌표를 그대로 저장.
        owner.geometry() 의존이 제거되어 settings window 이동/숨김에 영향받지 않는다.

        SC-1: MODE_WINDOW는 활성 창이 걸친 **모니터 1개**로 잘라 낸다 (바탕화면처럼
        가상 데스크톱 전체를 덮는 창이 전 모니터 캡처를 유발하던 문제).
        """
        if self.capture_mode == MODE_BOX:
            return
        if self.capture_mode == MODE_WINDOW:
            rect = self.foreground_window_rect()
        elif self.capture_mode == MODE_HOVER:
            rect = self.mouse_hover_rect()
        else:
            return
        if not rect:
            return
        ax1, ay1, ax2, ay2 = rect
        if self.capture_mode == MODE_WINDOW and CLAMP_WINDOW_TO_ACTIVE_SCREEN:
            try:
                scr = self.screen_rect_at((ax1 + ax2) // 2, (ay1 + ay2) // 2)
            except Exception as e:
                print(f"[WARN] SC-1 활성 모니터 판정 실패: {e}")
                scr = None
            if scr is not None:
                # scr는 (l, t, r_excl, b_excl) — Qt의 "마지막 픽셀" 규약은 캐시에서 이미 흡수했다.
                ax1 = max(ax1, scr[0])
                ay1 = max(ay1, scr[1])
                ax2 = min(ax2, scr[2])
                ay2 = min(ay2, scr[3])
        if ax2 - ax1 < 8 or ay2 - ay1 < 8:
            return
        self.red_rect = QRect(ax1, ay1, ax2 - ax1, ay2 - ay1)


class SettingsWindow(QMainWindow):
    # BG-4: 엔진 시그널은 BackgroundController가 정의/방출. 이 창은 UI 슬롯만 가짐.

    # T-1: 글로벌 단축키 → 메인 스레드 진입용 시그널 (keyboard 콜백은 별도 스레드)
    global_toggle_signal = Signal()
    global_show_signal = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        # Phase 1C: OverlayWindow 참조 (main에서 setter로 주입). 없을 수도 있음.
        self.overlay = None

        # Tray-first settings persistence. The control window is a settings panel,
        # not the always-visible translation surface.
        self._settings = QSettings("Cocktail", "OverlayTranslator")
        self._first_run_settings = not self._settings.value(
            "ui/first_run_complete", False, type=bool
        )

        # T-1: 트레이 종료 vs hide 분기
        self._is_quitting = False

        # W-1 (Phase 1B): 마우스 통과 토글
        # W-6: 실제 시작 상태와 일치시킨다. OverlayWindow는 생성/모드전환 시
        # update_for_mode()로 항상 통과 ON — 초기값 False면 첫 토글이 무효였다.
        self._mouse_through = True

        # Settings window. It is shown only on first run or from the tray/hotkey.
        self.setWindowTitle("Cocktail Settings")
        # BG-7: 폭 740 → 880 — "영역 편집..." 버튼이 첫 행에 들어갈 공간 확보.
        self.setGeometry(140, 140, 880, 120)
        self.setWindowFlags(Qt.Window)
        self.setAttribute(Qt.WA_TranslucentBackground, False)
        self.setStyleSheet(
            "QMainWindow { background: #202124; color: white; } "
            "QComboBox, QPushButton { min-height: 24px; }"
        )

        # E-7 → DP-1: 이제 Qt 논리 스케일링 자체가 OFF(QT_ENABLE_HIGHDPI_SCALING=0)라
        # 논리 좌표 == 물리 픽셀. 수동 배율 곱셈은 여전히 금지 — _scale=1 고정 유지.
        self._scale = 1.0
        self.px = lambda v: int(v)

        # BG-3: capture mode/region state is owned outside the settings window.
        # BG-6: region depends on the same is_self_hwnd callback used by the controller.
        self.region = CaptureRegionController(is_self_hwnd=self._is_self_hwnd)

        # BG-7: RegionEditor는 lazy 생성. 첫 "영역 편집..." 클릭 시 인스턴스 생성.
        self.region_editor = None

        self.initUI()

        # BG-4: 엔진은 BackgroundController가 소유. UI는 시그널 연결만.
        self.controller = BackgroundController(
            region=self.region,
            options_provider=self._snapshot_options,
            is_self_hwnd=self._is_self_hwnd,
        )
        self.controller.state_ready.connect(self._on_state_ready)
        self.controller.failure_alert.connect(self._on_failure)
        self.controller.model_ready.connect(self._on_model_ready)
        self.controller.progress_msg.connect(self._on_progress)
        self.controller.status_msg.connect(self._set_status)
        self.global_toggle_signal.connect(self._on_global_toggle)
        self.global_show_signal.connect(self._on_global_show)

        self._settings_visible_once = False
        self._maybe_show_initial_settings()
        self._set_status(self.controller.env_status)

        # 자기 자신을 화면 캡처에서 제외 (B-10). E-9 단축키로 토글 가능.
        if set_window_capture_affinity(self.winId(), exclude=True):
            print("[INFO] 오버레이 윈도우를 화면 캡처에서 제외했습니다 (Ctrl+Shift+H로 토글).")
        else:
            print("[WARN] 캡처 제외 실패 — Windows 10 2004+ 가 아니거나 권한 문제일 수 있음.")

        QShortcut(QKeySequence("Ctrl+Shift+H"), self, activated=self.toggle_capturable)
        # W-1: 마우스 통과 토글 단축키
        QShortcut(QKeySequence("Ctrl+Shift+P"), self, activated=self.toggle_mouse_through)

        # T-1: 시스템 트레이 + 글로벌 단축키
        self._setup_tray()
        self._register_global_hotkeys()

        # P-9: 모델 lazy load — UI는 이미 떠 있고, 모델은 백그라운드로 준비
        self.run_button.setEnabled(False)
        self.run_button.setText("모델 준비 중…")

        # BG-4: 워커 + 모델 preload + 캐시 타이머 + 영구 캐시 lazy load는 controller가 소유.
        self.controller.start()

        self.old_pos = QPoint()

    # --- BG-4: controller 의존 helpers --------------------------------------
    def _snapshot_options(self) -> "RuntimeOptions":
        # 문자열 어노테이션: RuntimeOptions는 이 파일 후반부에 정의되므로 forward-ref.
        return RuntimeOptions(
            src_text=self.src_combo.currentText(),
            tgt_text=self.tgt_combo.currentText(),
            ocr_text=self.ocr_combo.currentText(),
        )

    def _is_self_hwnd(self, hwnd: int) -> bool:
        try:
            if hwnd == int(self.winId()):
                return True
        except Exception:
            pass
        try:
            if self.overlay is not None and hwnd == int(self.overlay.winId()):
                return True
        except Exception:
            pass
        # BG-7: RegionEditor도 자기 hwnd로 인식 (B-10 피드백 가드).
        try:
            if self.region_editor is not None and hwnd == int(self.region_editor.winId()):
                return True
        except Exception:
            pass
        return False

    @property
    def capture_mode(self):
        return self.region.capture_mode

    @capture_mode.setter
    def capture_mode(self, value):
        self.region.capture_mode = value

    @property
    def hover_size(self):
        return self.region.hover_size

    @hover_size.setter
    def hover_size(self, value):
        self.region.hover_size = value

    @property
    def red_rect(self):
        return self.region.red_rect

    @red_rect.setter
    def red_rect(self, value):
        self.region.red_rect = value

    # BG-4: engine state proxy properties for backward compat (OverlayWindow.paintEvent
    # reads ctl.translated_text; closeEvent paths still reference these).
    @property
    def translated_text(self):
        return self.controller.translated_text

    @property
    def model_loaded(self):
        return self.controller.model_loaded

    @property
    def capturable(self):
        return self.controller.capturable

    def _maybe_show_initial_settings(self):
        """Show settings only on first run or explicit env override."""
        force = os.environ.get("COCKTAIL_SHOW_SETTINGS_ON_START", "").strip() == "1"
        if self._first_run_settings or force:
            self._show_normal()
        else:
            self.hide()

    def _mark_settings_seen(self):
        if not self._settings.value("ui/first_run_complete", False, type=bool):
            self._settings.setValue("ui/first_run_complete", True)
            self._settings.sync()

    @Slot()
    def _on_model_ready(self):
        self.run_button.setEnabled(True)
        self.run_button.setText("번역 시작")
        self._set_status(f"{self.controller.env_status} | 모델 준비 완료")

    @Slot(str)
    def _on_progress(self, msg):
        self.run_button.setText(f"준비: {msg[:14]}")
        self._set_status(f"모델 준비 중: {msg}")

    @Slot(list)
    def _on_state_ready(self, _new_state):
        # BG-4: controller already updated its own translated_text on the main thread.
        # BG-6: settings window draws no translation overlays — only OverlayWindow needs redraw.
        if self.overlay is not None:
            self.overlay.update()

    @Slot(str)
    def _on_failure(self, msg):
        self.run_button.setText(f"⚠ {msg[:18]}…")
        self._set_status(f"주의: {msg}")

    @Slot(str)
    def _set_status(self, msg):
        self.status_label.setText(msg)
        self.status_label.adjustSize()
        max_w = max(240, self.width() - self.px(20))
        if self.status_label.width() > max_w:
            self.status_label.setFixedWidth(max_w)

    def toggle_capturable(self):
        new_state = not self.controller.capturable
        control_ok = set_window_capture_affinity(self.winId(), exclude=not new_state)
        overlay_ok = True
        if self.overlay is not None:
            overlay_ok = set_window_capture_affinity(self.overlay.winId(), exclude=not new_state)
        if not (control_ok and overlay_ok):
            print("[WARN] 캡처 affinity 변경 실패")
            return
        self.controller.set_capturable(new_state)
        self.capture_btn.setText("스샷OK" if new_state else "스샷차단")
        if new_state and self.controller.model_loaded:
            print("[WARN] 캡처 허용 모드: OCR 피드백 위험. 스크린샷 직후 다시 '스샷차단'으로.")

    # --- T-1: 시스템 트레이 ---------------------------------------------------
    def _setup_tray(self):
        # 임시 아이콘 — 빨간 작은 사각형. 추후 실제 아이콘 파일로 교체.
        pm = QPixmap(16, 16)
        pm.fill(QColor(220, 60, 60))
        icon = QIcon(pm)

        self.tray = QSystemTrayIcon(icon, self)
        self.tray.setToolTip("Cocktail Translator (Ctrl+Shift+T 토글, Ctrl+Shift+A 열기)")

        menu = QMenu()
        a_show = QAction("열기", self)
        a_show.triggered.connect(self._show_normal)
        menu.addAction(a_show)

        a_toggle = QAction("번역 시작/정지", self)
        a_toggle.triggered.connect(self.toggle_running)
        menu.addAction(a_toggle)

        menu.addSeparator()
        mode_menu = menu.addMenu("캡처 모드")
        for label, mode in _MODE_LABELS.items():
            act = QAction(label, self)
            act.triggered.connect(lambda checked=False, m=mode: self.set_capture_mode(m))
            mode_menu.addAction(act)

        # BG-7: 영역 편집 (MODE_BOX 박스 마우스 편집)
        a_edit_region = QAction("영역 편집...", self)
        a_edit_region.triggered.connect(self._open_region_editor)
        menu.addAction(a_edit_region)

        # W-1: 마우스 통과 토글 (checkable)
        self._mouse_through_action = QAction("마우스 통과 (Ctrl+Shift+P)", self)
        self._mouse_through_action.setCheckable(True)
        self._mouse_through_action.setChecked(self._mouse_through)  # W-6: 실제 상태 반영
        self._mouse_through_action.triggered.connect(lambda _: self.toggle_mouse_through())
        menu.addAction(self._mouse_through_action)

        # A-1 (Phase 6): Windows 시작 시 자동 실행 (checkable)
        self._autorun_action = QAction("Windows 시작 시 자동 실행", self)
        self._autorun_action.setCheckable(True)
        self._autorun_action.setChecked(is_autorun_enabled())
        self._autorun_action.triggered.connect(self._toggle_autorun)
        menu.addAction(self._autorun_action)

        menu.addSeparator()
        a_quit = QAction("종료", self)
        a_quit.triggered.connect(self._really_quit)
        menu.addAction(a_quit)

        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self._on_tray_activated)
        self.tray.show()

    def _on_tray_activated(self, reason):
        if reason == QSystemTrayIcon.DoubleClick:
            self._show_normal()

    @Slot()
    def _show_normal(self):
        self._settings_visible_once = True
        self.show()
        self.raise_()
        self.activateWindow()
        capturable = self.controller.capturable
        set_window_capture_affinity(self.winId(), exclude=not capturable)
        if self.overlay is not None:
            set_window_capture_affinity(self.overlay.winId(), exclude=not capturable)

    def _really_quit(self):
        self._is_quitting = True
        self.close()

    @Slot(bool)
    def _toggle_autorun(self, checked: bool):
        """A-1: 트레이 메뉴 → Windows 자동 실행 토글."""
        if not set_autorun(checked):
            # 실패 시 체크 상태 복원 (다른 OS 또는 권한 부족)
            actual = is_autorun_enabled()
            if hasattr(self, "_autorun_action"):
                self._autorun_action.blockSignals(True)
                self._autorun_action.setChecked(actual)
                self._autorun_action.blockSignals(False)
            self._set_status(f"{self.controller.env_status} | 자동 실행 변경 실패 (Windows 전용)")
            return
        self._set_status(f"{self.controller.env_status} | 자동 실행: {'ON' if checked else 'OFF'}")

    # --- T-1: 글로벌 단축키 ---------------------------------------------------
    def _register_global_hotkeys(self):
        if not _KEYBOARD_OK:
            print("[INFO] keyboard 미설치 — 글로벌 단축키 비활성. (pip install keyboard)")
            return
        try:
            # keyboard 콜백은 별도 스레드. 메인 스레드 진입은 시그널로.
            _keyboard.add_hotkey('ctrl+shift+t', lambda: self.global_toggle_signal.emit())
            _keyboard.add_hotkey('ctrl+shift+a', lambda: self.global_show_signal.emit())
            print("[INFO] 글로벌 단축키 등록: Ctrl+Shift+T (번역 토글), Ctrl+Shift+A (창 열기)")
        except Exception as e:
            print(f"[WARN] 글로벌 단축키 등록 실패 (관리자 권한 필요할 수 있음): {e}")

    @Slot()
    def _on_global_toggle(self):
        if self.controller.is_model_ok():
            self.toggle_running()
        else:
            print("[INFO] 모델 준비 중 — 잠시 후 다시 시도하세요.")

    @Slot()
    def _on_global_show(self):
        self._show_normal()

    # --- BG-7: RegionEditor 진입점 -------------------------------------------
    @Slot()
    def _open_region_editor(self):
        """영역 편집 위젯을 열고 현재 red_rect (절대 좌표)로 초기화."""
        if self.region_editor is None:
            self.region_editor = RegionEditor()
            self.region_editor.region_committed.connect(self._on_region_committed)
        self.region_editor.open_for(self.region.red_rect)
        self._set_status(
            f"{self.controller.env_status} | 영역 편집: 드래그=이동, 핸들=리사이즈, Enter=저장, Esc=취소"
        )

    @Slot(QRect)
    def _on_region_committed(self, rect: QRect):
        """RegionEditor 저장 콜백 — 절대 좌표 QRect를 region에 반영하고 MODE_BOX로 전환."""
        self.region.red_rect = QRect(rect)
        # 사용자가 영역을 편집했다는 의도 = 그 영역을 사용하고 싶다 → MODE_BOX 자동 전환.
        self.set_capture_mode(MODE_BOX)
        self.controller.reset_frame_hash()
        if self.overlay is not None:
            self.overlay.update()
        self._set_status(
            f"{self.controller.env_status} | 영역 저장: "
            f"({rect.x()},{rect.y()}) {rect.width()}x{rect.height()}"
        )

    # --- T-2: 캡처 모드 -------------------------------------------------------
    def _on_mode_combo_changed(self, label: str):
        mode = _MODE_LABELS.get(label, MODE_BOX)
        self.set_capture_mode(mode)

    def set_capture_mode(self, mode: str):
        if mode not in (MODE_BOX, MODE_WINDOW, MODE_HOVER):
            return
        self.capture_mode = mode
        # 콤보 상태 동기화 (트레이 메뉴에서 변경됐을 때)
        label = _MODE_LABELS_REVERSE.get(mode)
        if label and self.mode_combo.currentText() != label:
            self.mode_combo.blockSignals(True)
            self.mode_combo.setCurrentText(label)
            self.mode_combo.blockSignals(False)
        # 모드 전환 시 직전 캐시 무효화 (frame-skip 해시 리셋)
        self.controller.reset_frame_hash()
        # Phase 1C: OverlayWindow 마우스 통과 정책/그리기 갱신
        if hasattr(self, "overlay") and self.overlay is not None:
            self.overlay.update_for_mode()
            self.overlay.update()
            # W-6: update_for_mode()가 통과를 강제 ON으로 되돌리므로 토글 플래그도 동기화.
            self._sync_mouse_through(True)
        self._set_status(f"{self.controller.env_status} | 모드: {label or mode}")

    def _foreground_window_rect(self):
        return self.region.foreground_window_rect()

    def _mouse_hover_rect(self):
        return self.region.mouse_hover_rect()

    def _update_red_rect_for_mode(self):
        self.region.update_for_mode()

    # --- W-1 (Phase 1B → Phase 1C 보강): OverlayWindow 마우스 통과 토글 ------
    # 박사 자체 리뷰 결과: W-1 토글은 OverlayWindow에 적용되어야 함.
    # ControlWindow에 적용하면 사용자가 자기 컨트롤도 못 누름.
    def _sync_mouse_through(self, on: bool):
        """W-6: 통과 상태의 단일 출처. 실제 창 상태를 바꾼 쪽이 플래그/체크박스를 맞춘다."""
        self._mouse_through = on
        if hasattr(self, "_mouse_through_action"):
            self._mouse_through_action.setChecked(on)

    def toggle_mouse_through(self):
        target = self.overlay if self.overlay is not None else self
        new_state = not self._mouse_through
        if not set_mouse_through(target.winId(), new_state):
            print("[WARN] 마우스 통과 설정 실패 (Windows 전용)")
            return
        self._sync_mouse_through(new_state)
        target_name = "OverlayWindow" if self.overlay is not None else "ControlWindow"
        msg = f"ON ({target_name}) — 클릭이 밑 창으로 통과" if new_state else "OFF"
        self._set_status(f"{self.controller.env_status} | 마우스 통과: {msg}")

    # BG-4: UIA 라우팅과 민감 창 검사는 BackgroundController로 이동.

    def initUI(self):
        px = self.px

        # 소스 언어
        self.src_combo = QComboBox(self)
        self.src_combo.addItems(["auto", "English", "Korean", "French", "German", "Chinese", "Japanese"])
        self.src_combo.setGeometry(px(10), px(10), px(110), px(30))

        # 타겟 언어
        self.tgt_combo = QComboBox(self)
        self.tgt_combo.addItems(["Korean", "English", "French", "German", "Chinese", "Japanese"])
        self.tgt_combo.setGeometry(px(125), px(10), px(110), px(30))

        # OCR 백엔드 (Phase 2/3: UIA 옵션 추가)
        self.ocr_combo = QComboBox(self)
        items = ["Tesseract", "PaddleOCRv5"]
        if UIA_PROVIDER.is_available():
            items.append("UIA (활성 창)")
        self.ocr_combo.addItems(items)
        self.ocr_combo.setGeometry(px(240), px(10), px(110), px(30))
        self.ocr_combo.setToolTip(
            "Tesseract: 일반 OCR / PaddleOCRv5: 선택 시 lazy load / "
            "UIA: 활성 창 텍스트 직접 (Windows, OCR 불필요)"
        )

        # T-2: 캡처 모드 (활성 창 / 마우스 hover / 영역 박스). 첫 항목이 기본.
        self.mode_combo = QComboBox(self)
        for label in _MODE_LABELS.keys():
            self.mode_combo.addItem(label)
        self.mode_combo.setGeometry(px(355), px(10), px(110), px(30))
        self.mode_combo.setToolTip("활성 창: 현재 활성 창 자동 / 마우스 hover: 커서 주변 / 영역 박스: 빨간 박스 안만")
        self.mode_combo.currentTextChanged.connect(self._on_mode_combo_changed)
        # 박사 자체 리뷰: 콤보 표시와 실제 capture_mode 동기화 (기본 = 활성 창)
        default_label = _MODE_LABELS_REVERSE.get(self.capture_mode)
        if default_label:
            self.mode_combo.blockSignals(True)
            self.mode_combo.setCurrentText(default_label)
            self.mode_combo.blockSignals(False)

        # 번역 시작/정지
        self.run_button = QPushButton("번역 시작", self)
        self.run_button.setGeometry(px(470), px(10), px(100), px(30))
        self.run_button.clicked.connect(self.toggle_running)

        # 캡처 토글 (E-9)
        self.capture_btn = QPushButton("스샷차단", self)
        self.capture_btn.setGeometry(px(575), px(10), px(90), px(30))
        self.capture_btn.setToolTip("스크린샷에 번역창을 포함할지 토글 (Ctrl+Shift+H)")
        self.capture_btn.clicked.connect(self.toggle_capturable)

        # BG-7: 영역 편집 진입점 (MODE_BOX 박스 마우스 편집)
        self.edit_region_btn = QPushButton("영역 편집...", self)
        self.edit_region_btn.setGeometry(px(670), px(10), px(130), px(30))
        self.edit_region_btn.setToolTip(
            "영역 박스를 마우스로 직접 편집 (MODE_BOX 자동 전환)."
        )
        self.edit_region_btn.clicked.connect(self._open_region_editor)

        # 종료
        self.close_button = QPushButton("확인", self)
        self.close_button.setGeometry(px(810), px(10), px(50), px(30))
        self.close_button.clicked.connect(self.close)

        # 상태/진단 표시 (U-1)
        self.status_label = QLabel("환경 확인 중", self)
        self.status_label.setGeometry(px(10), px(45), px(840), px(24))
        self.status_label.setStyleSheet(
            "QLabel { color: white; background: rgba(20, 20, 20, 210); "
            "padding: 3px 6px; border-radius: 4px; }"
        )

        # BG-5: 캡처 영역은 CaptureRegionController가 절대 좌표로 소유.
        # BG-1: settings panel size must not act as the region editor.
        # A dedicated RegionEditor (BG-6+) will own resize handles. Hidden grip remains
        # so legacy code paths and smoke checks (`self.size_grip.hide()`) are stable.
        self.size_grip = QSizeGrip(self)
        self.size_grip.hide()

    def resizeEvent(self, event):
        px = self.px
        self.status_label.setFixedWidth(max(px(240), self.width() - px(20)))
        self.update()
        # Phase 1C 자체 리뷰 수정: OverlayWindow 갱신
        if getattr(self, "overlay", None) is not None:
            self.overlay.update()

    def moveEvent(self, event):
        # Phase 1C 자체 리뷰 수정: 윈도우 이동 시 OverlayWindow 갱신
        super().moveEvent(event)
        if getattr(self, "overlay", None) is not None:
            self.overlay.update()

    def toggle_running(self):
        if not self.controller.is_model_ok():
            print("[INFO] 모델 준비 중 — 잠시만 기다려주세요.")
            self._set_status("모델 준비 중입니다. 완료 후 번역을 시작할 수 있습니다.")
            return
        new_state = not self.controller.model_loaded
        self.controller.set_running(new_state)
        self.run_button.setText("번역 정지" if new_state else "번역 시작")
        self._set_status("번역 실행 중" if new_state else "번역 정지")

    # BG-4: 워커 / OCR / SmartOCR / 라인 그룹핑은 BackgroundController로 이동.
    # BG-4: 그리기 helper는 OverlayWindow에 있음.
    # BG-6: 빈 paintEvent override 제거 — Qt 기본이 위젯 트리를 알아서 그려줌.

    # --- 종료 / 마우스 -------------------------------------------------------
    def closeEvent(self, event):
        # T-1: 트레이 메뉴 "종료"를 거치지 않은 닫기는 hide → 트레이로 보냄
        if not self._is_quitting and hasattr(self, "tray") and self.tray.isVisible():
            event.ignore()
            self._mark_settings_seen()
            self.hide()
            try:
                self.tray.showMessage(
                    "Cocktail Translator",
                    "트레이에서 계속 동작 중. 우클릭 메뉴로 제어하세요. (Ctrl+Shift+A 다시 열기)",
                    QSystemTrayIcon.Information,
                    2000,
                )
            except Exception:
                pass
            return

        # 진짜 종료. BG-4: 워커 stop + 캐시 save를 controller가 수행.
        if _KEYBOARD_OK:
            try:
                _keyboard.unhook_all_hotkeys()
            except Exception:
                pass
        try:
            self.controller.stop()
        except Exception as e:
            print(f"[WARN] controller stop 실패: {e}")
        # Phase 1C: OverlayWindow 동시 종료
        if self.overlay is not None:
            try:
                self.overlay.close()
            except Exception:
                pass
        # BG-7: RegionEditor도 정리 (force_close로 hide-instead-of-close 가드 우회)
        if self.region_editor is not None:
            try:
                self.region_editor.force_close()
            except Exception:
                pass
        event.accept()

    def mousePressEvent(self, event):
        # E-10: PySide6/Qt6은 globalPosition() 사용 (globalPos는 deprecated)
        if event.button() == Qt.LeftButton:
            self.old_pos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            new_global = event.globalPosition().toPoint()
            delta = new_global - self.old_pos
            self.move(self.x() + delta.x(), self.y() + delta.y())
            self.old_pos = new_global
            # Phase 1C 자체 리뷰 수정: 드래그 중 OverlayWindow 즉시 갱신
            if getattr(self, "overlay", None) is not None:
                self.overlay.update()


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


# --- Phase 4: 디스크 캐시 — DPAPI 암호화 (보안 원칙 #2 강화) ------------------
# 메모리 LRU와 별개로 사용자 윈도우 계정으로만 복호화 가능한 영구 캐시.
# 키는 sha256(src|tgt|text)의 hex (평문 source 비저장).
# 값은 번역 결과(텍스트). 전체 dict를 DPAPI로 암호화해서 저장.
class _DataBlob(ctypes.Structure):
    _fields_ = [
        ("cbData", ctypes.c_ulong),
        ("pbData", ctypes.POINTER(ctypes.c_byte)),
    ]


def _setup_crypt32():
    """박사 자체 리뷰 보강: argtypes/restype 명시 — 64비트 Windows에서 정확한 호출 규약 보장."""
    if sys.platform != "win32":
        return None
    try:
        crypt32 = ctypes.windll.crypt32
        sig_args = [
            ctypes.POINTER(_DataBlob),  # pDataIn
            ctypes.c_wchar_p,           # szDataDescr
            ctypes.POINTER(_DataBlob),  # pOptionalEntropy
            ctypes.c_void_p,            # pvReserved
            ctypes.c_void_p,            # pPromptStruct
            ctypes.c_ulong,             # dwFlags
            ctypes.POINTER(_DataBlob),  # pDataOut
        ]
        crypt32.CryptProtectData.argtypes = sig_args
        crypt32.CryptProtectData.restype = ctypes.c_int  # BOOL
        crypt32.CryptUnprotectData.argtypes = sig_args
        crypt32.CryptUnprotectData.restype = ctypes.c_int
        return crypt32
    except Exception as e:
        print(f"[WARN] crypt32 시그니처 설정 실패: {e}")
        return None


_CRYPT32 = _setup_crypt32()


def dpapi_protect(data: bytes):
    if _CRYPT32 is None or not data:
        return None
    try:
        kernel32 = ctypes.windll.kernel32
        in_blob = _DataBlob()
        in_buf = ctypes.create_string_buffer(data, len(data))
        in_blob.cbData = len(data)
        in_blob.pbData = ctypes.cast(in_buf, ctypes.POINTER(ctypes.c_byte))
        out_blob = _DataBlob()
        if not _CRYPT32.CryptProtectData(
            ctypes.byref(in_blob), None, None, None, None, 0, ctypes.byref(out_blob)
        ):
            return None
        try:
            result = ctypes.string_at(out_blob.pbData, out_blob.cbData)
        finally:
            kernel32.LocalFree(out_blob.pbData)
        return result
    except Exception as e:
        print(f"[WARN] DPAPI protect 실패: {e}")
        return None


def dpapi_unprotect(data: bytes):
    if _CRYPT32 is None or not data:
        return None
    try:
        kernel32 = ctypes.windll.kernel32
        in_blob = _DataBlob()
        in_buf = ctypes.create_string_buffer(data, len(data))
        in_blob.cbData = len(data)
        in_blob.pbData = ctypes.cast(in_buf, ctypes.POINTER(ctypes.c_byte))
        out_blob = _DataBlob()
        if not _CRYPT32.CryptUnprotectData(
            ctypes.byref(in_blob), None, None, None, None, 0, ctypes.byref(out_blob)
        ):
            return None
        try:
            result = ctypes.string_at(out_blob.pbData, out_blob.cbData)
        finally:
            kernel32.LocalFree(out_blob.pbData)
        return result
    except Exception as e:
        print(f"[WARN] DPAPI unprotect 실패: {e}")
        return None


# --- Phase 6 (A-1): Windows 자동 실행 옵션 (HKCU\...\Run) ---------------------
WIN_RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
AUTORUN_NAME = "CocktailTranslator"


def _autorun_command() -> str:
    """현재 실행 형태(.py vs .exe)에 맞춰 자동 실행 명령 문자열 생성."""
    exe = sys.executable or ""
    script = os.path.abspath(sys.argv[0]) if sys.argv and sys.argv[0] else ""
    if script and exe.lower().endswith(("python.exe", "pythonw.exe")):
        # 개발 시 자동 실행은 콘솔 창이 뜨지 않도록 pythonw.exe 우선.
        pyw = os.path.join(os.path.dirname(exe), "pythonw.exe")
        runner = pyw if os.path.exists(pyw) else exe
        return f'"{runner}" "{script}"'
    # 배포 .exe: 인터프리터 자체가 앱
    return f'"{exe}"'


def is_autorun_enabled() -> bool:
    if sys.platform != "win32":
        return False
    try:
        import winreg  # type: ignore
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, WIN_RUN_KEY, 0, winreg.KEY_READ) as k:
            try:
                value, _ = winreg.QueryValueEx(k, AUTORUN_NAME)
                return bool(value)
            except FileNotFoundError:
                return False
    except Exception:
        return False


def set_autorun(enable: bool) -> bool:
    if sys.platform != "win32":
        return False
    try:
        import winreg  # type: ignore
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, WIN_RUN_KEY, 0, winreg.KEY_SET_VALUE) as k:
            if enable:
                winreg.SetValueEx(k, AUTORUN_NAME, 0, winreg.REG_SZ, _autorun_command())
            else:
                try:
                    winreg.DeleteValue(k, AUTORUN_NAME)
                except FileNotFoundError:
                    pass
        return True
    except Exception as e:
        print(f"[WARN] 자동 실행 설정 실패: {e}")
        return False


class PersistentCache:
    """DPAPI 암호화된 영구 캐시. 키는 hash, 값은 번역 결과.
    플랫폼 비-Windows 또는 DPAPI 실패 시 자동으로 무동작 (앱은 정상)."""

    def __init__(self, path: str, max_entries: int = 4096):
        self.path = path
        self.max_entries = max_entries
        self._mem = {}
        self._dirty = False
        self._enabled = sys.platform == "win32"
        self._loaded = False
        self._load_lock = threading.Lock()
        self._mem_lock = threading.RLock()
        # P-9 정신: 디스크 I/O는 import 시점에 X. 첫 get/put 또는 명시적 preload 시.

    def preload_async(self):
        """Phase 4: 백그라운드 스레드로 디스크 캐시 로드. UI 시작 안 막음."""
        if self._loaded or not self._enabled:
            return
        threading.Thread(target=self._load, daemon=True).start()

    @staticmethod
    def _key(src: str, tgt: str, text: str) -> str:
        import hashlib
        return hashlib.sha256(f"{src}|{tgt}|{text}".encode("utf-8")).hexdigest()

    def _load(self):
        with self._load_lock:
            if self._loaded:
                return
            self._loaded = True
            if not self._enabled or not os.path.exists(self.path):
                return
            try:
                with open(self.path, "rb") as f:
                    blob = f.read()
                if not blob:
                    return
                decrypted = dpapi_unprotect(blob)
                if decrypted is None:
                    print("[WARN] PersistentCache 복호화 실패 — 새 캐시로 시작")
                    return
                import json
                loaded = json.loads(decrypted.decode("utf-8"))
                with self._mem_lock:
                    self._mem = loaded
                print(f"[INFO] PersistentCache 로드: {len(loaded)} 항목")
            except Exception as e:
                print(f"[WARN] PersistentCache 로드 실패: {e}")
                with self._mem_lock:
                    self._mem = {}

    def save(self):
        try:
            import json
            with self._mem_lock:
                if not self._enabled or not self._dirty:
                    return
                snapshot = dict(self._mem)
            data = json.dumps(snapshot, ensure_ascii=False).encode("utf-8")
            blob = dpapi_protect(data)
            if not blob:
                return
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "wb") as f:
                f.write(blob)
            os.replace(tmp, self.path)
            with self._mem_lock:
                # 저장 중 새 put이 들어왔으면 dirty를 보존한다.
                if snapshot == self._mem:
                    self._dirty = False
        except Exception as e:
            print(f"[WARN] PersistentCache 저장 실패: {e}")

    def get(self, src: str, tgt: str, text: str):
        if not self._enabled:
            return None
        if not self._loaded:
            self._load()  # 동기 로드 (이미 preload_async가 마쳐있을 가능성 큼)
        with self._mem_lock:
            return self._mem.get(self._key(src, tgt, text))

    def put(self, src: str, tgt: str, text: str, translation: str):
        if not self._enabled:
            return
        if not self._loaded:
            self._load()
        with self._mem_lock:
            if len(self._mem) >= self.max_entries:
                # 단순 절반 비우기 (LRU는 메모리 LRU에 위임)
                keys = list(self._mem.keys())
                for k in keys[: len(keys) // 2]:
                    self._mem.pop(k, None)
            self._mem[self._key(src, tgt, text)] = translation
            self._dirty = True


# 사용자 폴더에 캐시 — 평문 출처 미저장
_PERSIST_CACHE_PATH = os.path.join(
    os.path.expanduser("~"), ".cocktail", "translation_cache.bin"
)
PERSIST_CACHE = PersistentCache(_PERSIST_CACHE_PATH)
# UI가 뜬 직후 백그라운드로 로드 (BackgroundController.start에서 호출)


# --- Phase 1C: OverlayWindow — 분리된 투명 오버레이 ---------------------------
# 비전 V-1의 매끄러운 작동: 설정 창(SettingsWindow)과 오버레이 창을 분리.
# OverlayWindow는 전체 화면 + 투명 + 마우스 통과(자동) + 캡처 차단(B-10) + 그리기만 담당.
# SettingsWindow는 UI 위젯/트레이/단축키만 보유 (BG-4: 엔진은 BackgroundController로 분리됨).
class OverlayWindow(QMainWindow):
    """전체 화면 투명 오버레이 — 빨간 박스/번역 텍스트만 그림. 마우스는 통과(모드별 자동)."""

    def __init__(self, control: "SettingsWindow"):
        super().__init__()
        self.control = control

        # 프레임/투명/항상 위. Qt.Tool 로 작업표시줄에 안 뜨게.
        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)

        # 전체 화면 (primary screen 기준)
        self._fit_to_screen()

        self.show()
        # TM-1: show() 직후 한 번 올려 둔다. Qt.WindowStaysOnTopHint만으로는
        # "나중에 뜬 다른 topmost 창"에 밀린다.
        self.raise_()
        set_window_topmost(self.winId())

        # B-10: 자기 캡처 피드백 차단
        if not set_window_capture_affinity(self.winId(), exclude=True):
            print("[WARN] OverlayWindow 캡처 제외 실패")

        # 모드에 따른 마우스 통과 자동 — 박스 모드도 통과 ON으로 (사용자는 ControlWindow에서 박스 위치 조정)
        # 활성 창/hover/박스 모두 통과 ON. 사용자가 OverlayWindow 위에서 다른 창 자유롭게 클릭.
        self.update_for_mode()

        # TM-1: 주기적 재적용. SWP_NOACTIVATE라 포커스를 뺏지 않고, 2초 간격이라 비용은 무시 가능.
        self._topmost_timer = None
        if OVERLAY_TOPMOST_INTERVAL_MS > 0:
            self._topmost_timer = QTimer(self)
            self._topmost_timer.timeout.connect(self._reapply_topmost)
            self._topmost_timer.start(OVERLAY_TOPMOST_INTERVAL_MS)

    @Slot()
    def _reapply_topmost(self):
        """TM-1: 다른 항상-위 창이 나중에 떠서 우리를 덮은 경우 Z-order를 되찾는다."""
        if not self.isVisible():
            return
        set_window_topmost(self.winId())

    def _fit_to_screen(self):
        # Phase 1D: 멀티 모니터 — 가상 데스크톱 전체 영역으로 확장.
        # 각 화면의 geometry는 절대 좌표 (보조 모니터는 음수 또는 +width 등).
        # DP-1: 스케일링 OFF라 이 합집합은 물리 픽셀 rect = ImageGrab/OCR bbox와 같은 좌표계.
        screens = QApplication.screens()
        if not screens:
            return
        rects = [s.geometry() for s in screens]
        x = min(r.left() for r in rects)
        y = min(r.top() for r in rects)
        right = max(r.right() for r in rects)
        bottom = max(r.bottom() for r in rects)
        self.setGeometry(x, y, right - x + 1, bottom - y + 1)

    def update_for_mode(self):
        """모드 변경 시 호출. 현재는 항상 마우스 통과 ON (오버레이 본연의 역할)."""
        set_mouse_through(self.winId(), True)

    # --- 그리기 (화면 절대 좌표 → overlay-내 좌표 변환 후 그림) -------
    @staticmethod
    def _overlay_font(px_size):
        """FT-1: 한글 우선 폰트 스택. `QFont("Arial")`은 한글 글리프가 없어 Qt 임의 폴백에 맡겨졌다.

        F-2: 크기는 OCR bbox에서 온 **픽셀** 값 — 생성자 2번째 인자(pointSize)가 아니라 setPixelSize.
        """
        font = QFont()
        font.setFamilies(list(OVERLAY_FONT_FAMILIES))
        font.setPixelSize(int(px_size))
        return font

    @staticmethod
    def _fit_font(painter, base_rect, text, initial_size, min_size=9):
        """B-13/C: 폰트를 점진적으로 축소해 needed.height ≤ base의 2배가 되는 가장 큰 폰트 반환."""
        target_max_h = max(base_rect.height() * 2, base_rect.height() + 4)
        size = initial_size
        font = OverlayWindow._overlay_font(size)
        needed = base_rect
        while size >= min_size:
            font = OverlayWindow._overlay_font(size)
            painter.setFont(font)
            needed = painter.fontMetrics().boundingRect(
                base_rect.x(), base_rect.y(),
                base_rect.width(), 10000,
                Qt.AlignLeft | Qt.TextWordWrap,
                text,
            )
            if needed.height() <= target_max_h:
                break
            size -= 1
        return font, needed

    def _clip_for_abs_bbox(self, abs_bbox, wx, wy):
        """CL-1: 박스 **중심이 속한 모니터** ∩ 창 rect 를 overlay-내 좌표로 돌려준다.

        기존 클립은 `self.rect()`(가상 데스크톱 전체)라, 모니터 경계에 걸친 자막이
        옆 모니터로 흘러넘치거나 모니터가 없는 빈 영역(멀티 모니터 배치의 틈)에 그려졌다.
        판정은 [A6]에서 만든 메인 스레드 화면 rect 캐시를 재사용한다.
        """
        full = self.rect()
        try:
            region = self.control.region
            cx = (int(abs_bbox[0]) + int(abs_bbox[2])) // 2
            cy = (int(abs_bbox[1]) + int(abs_bbox[3])) // 2
            scr = region.screen_rect_at(cx, cy)
        except Exception:
            scr = None
        if not scr:
            return full
        return QRect(scr[0] - wx, scr[1] - wy, scr[2] - scr[0], scr[3] - scr[1]).intersected(full)

    def _draw_one(self, painter, text, bbox, font_size, clip_rect=None):
        if not (isinstance(bbox, tuple) and len(bbox) == 4):
            return
        left, top, right, bottom = bbox  # 화면 절대 좌표

        # OverlayWindow는 (0,0)이 화면 (0,0). 그래서 좌표 그대로 사용.
        initial_size = max(10, int(round(font_size * 0.9)))
        base_rect = QRect(int(left), int(top), int(right - left), int(bottom - top))

        font, needed = self._fit_font(painter, base_rect, text, initial_size)
        painter.setFont(font)

        expanded_h = max(base_rect.height(), needed.height() + 4)
        expanded = QRect(base_rect.x(), base_rect.y(), base_rect.width(), expanded_h)

        # CL-1: 가상 데스크톱 전체가 아니라 "그 박스가 속한 모니터"로 클립.
        rect = expanded.intersected(clip_rect if clip_rect is not None else self.rect())
        if rect.isEmpty():
            return

        # B-12: alpha 235로 영문 차단
        painter.setBrush(QColor(20, 20, 20, 235))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(rect, 8, 8)

        painter.setPen(QPen(QColor(255, 255, 255), 1))
        painter.drawText(
            rect,
            Qt.AlignLeft | Qt.AlignVCenter | Qt.TextWordWrap,
            text,
        )

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Phase 1D: OverlayWindow가 가상 데스크톱이면 (0,0)이 화면 (0,0)이 아니다.
        # 화면 절대 좌표 → 윈도우 내 좌표로 변환 필요.
        wx, wy = self.geometry().x(), self.geometry().y()

        ctl = self.control
        if ctl is not None:
            # 사용자 피드백 (2026-05-04): 빨간 박스는 사용자가 영역 지정한 박스 모드에서만 보임.
            # 활성 창/마우스 hover에선 화면 가득한 빨간 박스가 시각적 노이즈라 안 그림.
            if ctl.capture_mode == MODE_BOX:
                # BG-5: red_rect는 절대 좌표. settings 창 위치 의존 제거.
                rect = ctl.red_rect
                screen_rect = QRect(
                    rect.x() - wx,
                    rect.y() - wy,
                    rect.width(),
                    rect.height(),
                )
                pen = QPen(QColor(255, 0, 0), 3)
                painter.setPen(pen)
                painter.drawRect(screen_rect)

            # 번역 오버레이 (bbox는 화면 절대 → 윈도우 내로 변환). 모든 모드에서 그림.
            for _src, tgt, bbox, fsz in ctl.translated_text:
                local_bbox = (
                    bbox[0] - wx, bbox[1] - wy,
                    bbox[2] - wx, bbox[3] - wy,
                )
                # CL-1: 클립은 이 박스가 속한 모니터 기준 (절대 bbox로 판정).
                self._draw_one(painter, tgt, local_bbox, fsz,
                               self._clip_for_abs_bbox(bbox, wx, wy))


class RegionEditor(QWidget):
    """BG-7: MODE_BOX 영역 박스를 마우스로 직접 편집하는 위젯.

    가상 데스크톱 전체를 덮는 frameless + translucent 윈도우. 내부 박스 상태는
    절대 좌표 QRect로 유지하며, paintEvent에서만 editor-window-내 좌표로 1회 변환한다
    (OverlayWindow와 동일 패턴).

    저장: ``region_committed(QRect)`` 시그널로 절대 좌표 emit → SettingsWindow가
    ``region.red_rect`` 대입 + ``set_capture_mode(MODE_BOX)`` 호출.
    취소: 변경 사항 폐기 후 hide.

    안전:
    - B-10: ``set_window_capture_affinity(exclude=True)``로 자기 캡처 피드백 차단.
    - SettingsWindow._is_self_hwnd가 이 위젯의 winId도 자기 hwnd로 인식 → worker가
      편집 창을 캡처/번역하지 않음.
    - 민감 창 가드(S-1)와 capturable guard(B-15)는 BackgroundController에 그대로 남아
      편집 중에도 정상 동작. RegionEditor는 geometry만 바꾸고 번역 파이프라인 미접근.
    """

    region_committed = Signal(QRect)

    _MIN_W = 60
    _MIN_H = 40
    _HANDLE_SIZE = 12
    _HANDLE_HIT_MARGIN = 4

    _HANDLES = ("nw", "n", "ne", "e", "se", "s", "sw", "w")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating, False)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)

        # 모든 좌표는 가상 데스크톱 절대 좌표.
        self._editing_rect = QRect()
        self._initial_rect = QRect()
        self._drag_mode = None      # "move" | handle id | None
        self._drag_start = QPoint()
        self._drag_origin_rect = QRect()
        self._force_closing = False
        self._capture_excluded = False

        self._fit_to_virtual_desktop()

        # 컨트롤 버튼: 박스 근처에 떠 있어 즉시 저장/취소 가능.
        btn_style = (
            "QPushButton { background: rgba(30,30,30,235); color: white; "
            "padding: 4px 12px; border: 1px solid white; border-radius: 4px; "
            "font-size: 12px; } "
            "QPushButton:hover { background: rgba(60,60,60,240); }"
        )
        self.save_btn = QPushButton("저장 (Enter)", self)
        self.cancel_btn = QPushButton("취소 (Esc)", self)
        self.save_btn.setStyleSheet(btn_style)
        self.cancel_btn.setStyleSheet(btn_style)
        self.save_btn.clicked.connect(self.commit)
        self.cancel_btn.clicked.connect(self.cancel)

    def _fit_to_virtual_desktop(self):
        # DP-1: 스케일링 OFF라 screens() 합집합 = 물리 픽셀 가상 데스크톱 (OverlayWindow와 동일).
        screens = QApplication.screens()
        if not screens:
            return
        rects = [s.geometry() for s in screens]
        x = min(r.left() for r in rects)
        y = min(r.top() for r in rects)
        right = max(r.right() for r in rects)
        bottom = max(r.bottom() for r in rects)
        self.setGeometry(x, y, right - x + 1, bottom - y + 1)

    def open_for(self, abs_rect: QRect):
        """현재 절대 좌표 박스로 편집을 시작한다."""
        self._fit_to_virtual_desktop()
        seed = QRect(abs_rect) if abs_rect and abs_rect.isValid() else QRect(240, 215, 600, 375)
        if seed.width() < self._MIN_W:
            seed.setWidth(self._MIN_W)
        if seed.height() < self._MIN_H:
            seed.setHeight(self._MIN_H)
        self._editing_rect = QRect(seed)
        self._initial_rect = QRect(seed)
        self._drag_mode = None
        self._reposition_buttons()
        self.show()
        self.raise_()
        self.activateWindow()
        self.setFocus()
        if not self._capture_excluded:
            if set_window_capture_affinity(self.winId(), exclude=True):
                self._capture_excluded = True
            else:
                print("[WARN] RegionEditor 캡처 제외 실패 — B-10 위험. Ctrl+Shift+H 보호 확인.")
        self.update()

    @Slot()
    def commit(self):
        rect = QRect(self._editing_rect)
        if rect.width() < self._MIN_W:
            rect.setWidth(self._MIN_W)
        if rect.height() < self._MIN_H:
            rect.setHeight(self._MIN_H)
        self.region_committed.emit(rect)
        self.hide()

    @Slot()
    def cancel(self):
        self._editing_rect = QRect(self._initial_rect)
        self.hide()

    def force_close(self):
        """앱 종료 경로 — closeEvent의 hide-instead-of-close 가드를 우회."""
        self._force_closing = True
        self.close()

    # --- 좌표 변환 ----------------------------------------------------------
    def _local_box(self) -> QRect:
        wx, wy = self.geometry().x(), self.geometry().y()
        r = self._editing_rect
        return QRect(r.x() - wx, r.y() - wy, r.width(), r.height())

    def _handle_rects_local(self):
        local = self._local_box()
        s = self._HANDLE_SIZE
        half = s // 2
        cx = local.x() + local.width() // 2
        cy = local.y() + local.height() // 2
        lx, ty = local.left(), local.top()
        rx, by = local.right(), local.bottom()
        return {
            "nw": QRect(lx - half, ty - half, s, s),
            "n":  QRect(cx - half, ty - half, s, s),
            "ne": QRect(rx - half, ty - half, s, s),
            "e":  QRect(rx - half, cy - half, s, s),
            "se": QRect(rx - half, by - half, s, s),
            "s":  QRect(cx - half, by - half, s, s),
            "sw": QRect(lx - half, by - half, s, s),
            "w":  QRect(lx - half, cy - half, s, s),
        }

    def _hit_handle(self, local_pt: QPoint):
        m = self._HANDLE_HIT_MARGIN
        for hid, rect in self._handle_rects_local().items():
            if rect.adjusted(-m, -m, m, m).contains(local_pt):
                return hid
        return None

    def _reposition_buttons(self):
        local = self._local_box()
        gap = 8
        h = 28
        target_y = local.top() - (h + gap)
        if target_y < 4:
            target_y = local.bottom() + gap
        self.save_btn.adjustSize()
        self.cancel_btn.adjustSize()
        x = max(4, local.left())
        self.save_btn.move(x, target_y)
        self.cancel_btn.move(x + self.save_btn.width() + gap, target_y)
        self.save_btn.raise_()
        self.cancel_btn.raise_()

    # --- 키/마우스 ----------------------------------------------------------
    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Escape:
            self.cancel()
            return
        if key in (Qt.Key_Return, Qt.Key_Enter):
            self.commit()
            return
        super().keyPressEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self.cancel()
            return
        if event.button() != Qt.LeftButton:
            return
        gpos = event.globalPosition().toPoint()
        wx, wy = self.geometry().x(), self.geometry().y()
        lpos = QPoint(gpos.x() - wx, gpos.y() - wy)
        hid = self._hit_handle(lpos)
        if hid:
            self._drag_mode = hid
        elif self._local_box().contains(lpos):
            self._drag_mode = "move"
        else:
            return
        self._drag_start = gpos
        self._drag_origin_rect = QRect(self._editing_rect)

    def mouseMoveEvent(self, event):
        if not self._drag_mode:
            return
        gpos = event.globalPosition().toPoint()
        dx = gpos.x() - self._drag_start.x()
        dy = gpos.y() - self._drag_start.y()
        orig = self._drag_origin_rect
        if self._drag_mode == "move":
            new_rect = QRect(orig)
            new_rect.moveTo(orig.x() + dx, orig.y() + dy)
        else:
            left, top = orig.left(), orig.top()
            right, bottom = orig.right(), orig.bottom()
            h = self._drag_mode
            if "n" in h:
                top = min(orig.bottom() - self._MIN_H + 1, top + dy)
            if "s" in h:
                bottom = max(orig.top() + self._MIN_H - 1, bottom + dy)
            if "w" in h:
                left = min(orig.right() - self._MIN_W + 1, left + dx)
            if "e" in h:
                right = max(orig.left() + self._MIN_W - 1, right + dx)
            new_rect = QRect(QPoint(left, top), QPoint(right, bottom))
        self._editing_rect = new_rect
        self._reposition_buttons()
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_mode = None

    # --- 그리기 -------------------------------------------------------------
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        local = self._local_box()
        # 박스 밖 4개 띠를 반투명 검정으로 어둡게 → 박스 영역 명확.
        dim = QColor(0, 0, 0, 80)
        ew, eh = self.width(), self.height()
        # top
        if local.top() > 0:
            painter.fillRect(0, 0, ew, local.top(), dim)
        # bottom
        if local.bottom() < eh - 1:
            painter.fillRect(0, local.bottom() + 1, ew, eh - local.bottom() - 1, dim)
        # left / right strips (only across the box height)
        if local.left() > 0:
            painter.fillRect(0, local.top(), local.left(), local.height(), dim)
        if local.right() < ew - 1:
            painter.fillRect(
                local.right() + 1, local.top(),
                ew - local.right() - 1, local.height(), dim,
            )

        # 박스 외곽선
        pen = QPen(QColor(255, 60, 60), 3)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(local)

        # 리사이즈 핸들 8개
        painter.setBrush(QColor(255, 255, 255))
        painter.setPen(QPen(QColor(20, 20, 20), 1))
        for rect in self._handle_rects_local().values():
            painter.drawRect(rect)

        # 상태 라벨
        r = self._editing_rect
        info = (
            f"({r.x()},{r.y()}) {r.width()}x{r.height()}  "
            "· 드래그=이동, 핸들=리사이즈, Enter=저장, Esc=취소"
        )
        painter.setFont(QFont("Arial", 10))
        fm = painter.fontMetrics()
        info_w = max(local.width(), fm.boundingRect(info).width() + 16)
        info_rect = QRect(local.x(), local.bottom() + 8, info_w, 22)
        painter.fillRect(info_rect, QColor(30, 30, 30, 220))
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(
            info_rect.adjusted(8, 0, 0, 0),
            Qt.AlignVCenter | Qt.AlignLeft,
            info,
        )

    def closeEvent(self, event):
        # X 키/Alt+F4 등으로 닫히면 cancel과 동일 처리 (인스턴스 보존).
        # 앱 종료 시점만 force_close()를 거쳐 실제 close.
        if not self._force_closing:
            self._editing_rect = QRect(self._initial_rect)
            event.ignore()
            self.hide()
            return
        super().closeEvent(event)


@dataclass(frozen=True)
class RuntimeOptions:
    """Per-iteration snapshot of UI combo state used by the worker thread.

    Immutable so the worker reads a consistent set of values per loop tick.
    """
    src_text: str   # combo display: "auto" | "English" | "Korean" | ...
    tgt_text: str   # combo display: "Korean" | "English" | ...
    ocr_text: str   # combo display: "Tesseract" | "PaddleOCRv5" | "UIA (활성 창)"


class _UiaSkipFrame:
    """SV-1 (A9): '이 프레임은 통째로 건너뛴다'는 sentinel.

    민감 창(S-1)에서 UIA가 빈 리스트를 돌려주면 호출부가 "UIA 실패"로 읽고 OCR 경로로
    계속 진행했다 → 민감 창이 캡처·OCR·번역되고 DPAPI 영구 캐시에까지 남는 보안 결함.
    `None`(=UIA 불가, OCR 폴백해도 됨) / `[]`(=텍스트 없음) 과 반드시 구분되어야 한다.
    """
    __slots__ = ()

    def __repr__(self):
        return "UIA_SKIP_FRAME"


UIA_SKIP_FRAME = _UiaSkipFrame()


class BackgroundController(QObject):
    """Engine-side controller (BG-4).

    Owns the worker thread, OCR/UIA routing, model lifecycle, auto-pause flags,
    persistent-cache timer, and engine→UI signals.

    BG-4: extracted from the settings window so the UI class only owns widgets.
    BG-5: `red_rect` is stored in absolute screen coordinates.
    BG-6: depends on `region`, `options_provider`, `is_self_hwnd` callbacks only —
    no widget references.
    """

    state_ready = Signal(list)
    failure_alert = Signal(str)
    model_ready = Signal()
    progress_msg = Signal(str)
    status_msg = Signal(str)

    def __init__(self, region, options_provider, is_self_hwnd):
        super().__init__()
        self.region = region
        self._get_options = options_provider
        self._is_self_hwnd = is_self_hwnd

        # Engine state
        self.translated_text = []
        # RC-1 (A5): 워커가 "마지막으로 emit한 상태"의 자기 로컬 사본.
        # `translated_text`는 메인 스레드 큐드 슬롯이 갱신하므로 워커가 그걸 읽고 판정하면
        # 슬롯이 아직 안 돈 사이에 "비울 게 없다"고 오판해 잔상이 영구화될 수 있다.
        self._last_emitted = []
        self.stop_thread = False
        self.model_loaded = False
        self.last_frame_hash = None
        self.fail_streak = 0
        self.no_ocr_streak = 0
        self.capturable = False
        self._model_ok = False
        self._sensitive_paused = False
        self._capturable_paused = False
        env_status, env_problems = environment_summary()
        self.env_status = env_status
        self.env_problems = env_problems

        self._cache_save_in_progress = False
        self._cache_save_timer = None
        self.translation_thread = None
        self._ocr_scale = 1
        self._ocr_line_confs = []   # DG-1: 직전 OCR의 라인별 평균 confidence (표시 게이트용)

        # Engine maintains its own translated_text on main thread.
        self.state_ready.connect(self._apply_state_to_self)

    # --- public API ----------------------------------------------------------
    def start(self):
        threading.Thread(target=self._preload_model, daemon=True).start()
        PERSIST_CACHE.preload_async()

        self._cache_save_timer = QTimer(self)
        self._cache_save_timer.timeout.connect(self._on_cache_save_tick)
        self._cache_save_timer.start(30 * 1000)

        self.translation_thread = threading.Thread(
            target=self.run_asyncio_loop, daemon=True
        )
        self.translation_thread.start()

    def stop(self):
        self.stop_thread = True
        if self.translation_thread is not None and self.translation_thread.is_alive():
            self.translation_thread.join(timeout=1.5)
        try:
            PERSIST_CACHE.save()
        except Exception:
            pass

    def set_running(self, value: bool):
        self.model_loaded = value
        # UX-3: 정지 시 오버레이 잔상 제거. 토글의 단일 진입점이라 버튼/단축키/트레이 전 경로 커버.
        if not value:
            self._clear_overlay_if_any()

    def is_model_ok(self) -> bool:
        return self._model_ok

    def set_capturable(self, value: bool):
        self.capturable = value

    def reset_frame_hash(self):
        self.last_frame_hash = None

    @Slot(list)
    def _apply_state_to_self(self, new_state):
        self.translated_text = new_state

    def _emit_state(self, new_state):
        """RC-1: 상태 emit의 **단일 경로**. 워커 로컬 사본을 같은 자리에서 갱신한다.

        (list 재바인딩은 CPython에서 원자적이라 별도 lock 없이 워커/메인이 공유해도 안전.
         ponytail: 여기에 lock을 걸 만한 경합 대상이 없다 — 필요해지면 그때 추가.)
        """
        self._last_emitted = list(new_state)
        self.state_ready.emit(new_state)

    def _clear_overlay_if_any(self):
        """일시정지/스킵 경로 공통 — 남은 자막이 있으면 오버레이를 비운다 (W-4/UX-3/B-15/S-1).

        RC-1: 판정 기준이 `translated_text`(메인 스레드가 갱신)가 아니라 워커 로컬이다.
        """
        if self._last_emitted:
            self._emit_state([])
            return True
        return False

    def _emit_state_if_changed(self, new_state):
        """RT-2: 직전 상태와 (원문, 번역문, 반올림 bbox) 기준 동일하면 emit 생략.
        OCR bbox 1~2px 흔들림으로 매 사이클 전체 재그리기 → 깜빡임/CPU 낭비를 막는다.

        RC-1: 비교 기준도 워커 로컬(`_last_emitted`) — 메인 슬롯 지연에 좌우되지 않는다."""
        if _state_signature(new_state) == _state_signature(self._last_emitted):
            return False
        self._emit_state(new_state)
        return True

    # --- model lifecycle -----------------------------------------------------
    def _preload_model(self):
        try:
            TRANSLATOR.preload_default(progress_cb=lambda msg: self.progress_msg.emit(msg))
            self._model_ok = True
            self.model_ready.emit()
        except Exception as e:
            print(f"[ERROR] 모델 로드 실패: {e}")
            self.failure_alert.emit("모델 로드 실패")

    @Slot()
    def _on_cache_save_tick(self):
        if self._cache_save_in_progress:
            return
        self._cache_save_in_progress = True

        def _save_bg():
            try:
                PERSIST_CACHE.save()
            except Exception as e:
                print(f"[WARN] 주기적 캐시 save 실패: {e}")
            finally:
                self._cache_save_in_progress = False

        threading.Thread(target=_save_bg, daemon=True).start()

    # --- worker --------------------------------------------------------------
    def run_asyncio_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.capture_and_translate_async())
        finally:
            loop.close()

    async def capture_and_translate_async(self):
        while not self.stop_thread:
            if not self.model_loaded or not self._model_ok:
                await asyncio.sleep(0.2)
                continue

            try:
                # B-15 (자동 가드): "스샷OK" + 번역 동작 = B-10 자기 캡처 피드백 위험.
                if self.capturable:
                    self._clear_overlay_if_any()   # RC-1
                    if not self._capturable_paused:
                        self._capturable_paused = True
                        self.status_msg.emit(
                            "⚠ 캡처 허용(스샷OK) 모드 — B-10 피드백 위험으로 번역 일시 중지. "
                            "다시 '스샷차단'을 누르거나 Ctrl+Shift+H로 보호를 켜세요."
                        )
                    await asyncio.sleep(1.0)
                    continue
                if self._capturable_paused:
                    self._capturable_paused = False
                    self.status_msg.emit(f"{self.env_status} | 캡처 차단 복구 — 번역 재개")

                # S-1 (Phase 5): 박스 모드 외에서 민감 창 활성 시 일시 중지.
                if self.region.capture_mode != MODE_BOX and self._check_sensitive_pause():
                    self._clear_overlay_if_any()   # RC-1
                    if not self._sensitive_paused:
                        self._sensitive_paused = True
                        self.status_msg.emit("민감 창 감지 — 번역 일시 중지 (보안 원칙 #3)")
                    await asyncio.sleep(1.0)
                    continue
                if self._sensitive_paused:
                    self._sensitive_paused = False
                    self.status_msg.emit(f"{self.env_status} | 민감 창 해제 — 번역 재개")

                # T-2: 모드에 따라 red_rect 자동 갱신.
                self.region.update_for_mode()

                # BG-5: red_rect는 절대 좌표. owner.geometry() 변환 불필요.
                rr = self.region.red_rect
                x1 = int(rr.x())
                y1 = int(rr.y())
                x2 = int(rr.x() + rr.width())
                y2 = int(rr.y() + rr.height())
                if x2 - x1 < 8 or y2 - y1 < 8:
                    await asyncio.sleep(0.2)
                    continue

                # UX-2 Layer 2: 활성 창 모드에서 화면 가시 영역으로 클립.
                # W-4: primaryScreen 단독이 아니라 모든 화면의 합집합(가상 데스크톱)으로 클립.
                # 보조 모니터의 창이 영구히 continue 되던 버그.
                # GT-1: Qt 화면 열거 API는 GUI 전용이라 워커 스레드에서 직접 부르지 않는다.
                # 메인 스레드가 갱신하는 region.screen_rects 캐시(가상 데스크톱 합집합)만 읽는다.
                if self.region.capture_mode == MODE_WINDOW:
                    try:
                        vd = self.region.virtual_desktop_rect()
                        if vd:
                            x1 = max(x1, vd[0])
                            y1 = max(y1, vd[1])
                            x2 = min(x2, vd[2])
                            y2 = min(y2, vd[3])
                            if x2 - x1 < 8 or y2 - y1 < 8:
                                # W-4: 화면 밖 창은 다른 일시정지 경로와 동일하게 오버레이를 비운다.
                                self._clear_overlay_if_any()   # RC-1
                                await asyncio.sleep(0.2)
                                continue
                    except Exception:
                        pass

                # UX-2 Layer 1: UIA 자동 라우팅.
                ocr_choice = self._get_options().ocr_text
                wants_uia = (
                    ocr_choice == "UIA (활성 창)" or
                    (self.region.capture_mode == MODE_WINDOW and UIA_PROVIDER.is_available()
                     and ocr_choice not in ("Tesseract", "PaddleOCRv5"))
                )
                if wants_uia:
                    uia_state = self._uia_collect_and_translate()
                    # SV-1: 민감 창은 이 프레임 전체를 건너뛴다. 여기서 OCR로 폴백하면
                    # 민감 창이 캡처·OCR·번역·영구 캐시된다 (S-1 가드를 우회하는 구멍이었음).
                    if uia_state is UIA_SKIP_FRAME:
                        await asyncio.sleep(1.0)
                        continue
                    if uia_state is not None and uia_state:
                        self._emit_state_if_changed(uia_state)  # RT-2
                        if self.fail_streak > 0:
                            self.failure_alert.emit("번역 정지" if self.model_loaded else "번역 시작")
                        self.fail_streak = 0
                        await asyncio.sleep(0.5)
                        continue

                # W-5: all_screens=True — win32 경로가 가상 데스크톱 전체를 찍고
                # offset(가상 화면 원점) 기준으로 crop하므로 절대 좌표 bbox가 그대로 맞는다.
                # (False면 primary만 찍어 보조 모니터 영역이 검은 이미지로 나옴)
                screenshot = ImageGrab.grab(bbox=(x1, y1, x2, y2), all_screens=True)
                screenshot_np = np.array(screenshot)

                # frame-skip (P-4 / F-1)
                h = cheap_hash(screenshot_np, (x1, y1, x2, y2))
                if hash_distance(self.last_frame_hash, h) < FRAME_DIFF_THRESHOLD:
                    await asyncio.sleep(0.25)
                    continue
                self.last_frame_hash = h

                ocr_results = self.perform_ocr_with_boxes(screenshot_np)

                if not ocr_results:
                    self._clear_overlay_if_any()   # RC-1
                    self.no_ocr_streak += 1
                    if self.no_ocr_streak in {1, 5, 20}:
                        self.status_msg.emit(
                            f"OCR 결과 없음: 영역/언어/OCR 백엔드 확인 필요 ({self._ocr_backend_name()})"
                        )
                    await asyncio.sleep(0.3)
                    continue
                self.no_ocr_streak = 0

                src_hint = self._src_hint()
                tgt_lang = LANG_MAP[self._get_options().tgt_text]
                src_texts = [t for t, _, _ in ocr_results]
                src_langs = assign_region_languages(ocr_results, src_hint)

                t0 = time.perf_counter()
                translated = batch_translate(src_texts, src_langs, tgt_lang)
                dt = (time.perf_counter() - t0) * 1000.0
                print(f"[PERF] translate {len(src_texts)} lines in {dt:.1f} ms")
                lang_summary = ",".join(sorted(set(src_langs)))

                # B-3: 매 프레임 화면 기준 재구성. bbox는 화면 절대 좌표.
                line_confs = self._ocr_line_confs
                hidden = 0
                new_state = []
                for i, ((src_text, bbox, fsz), tgt_text) in enumerate(zip(ocr_results, translated)):
                    if not (tgt_text and tgt_text.strip()):
                        continue
                    # DG-1: 표시 직전 최종 게이트 — 확신 없으면 아예 안 그린다.
                    conf = line_confs[i] if i < len(line_confs) else None
                    reason = display_gate_reject(
                        src_text, tgt_text, tgt_lang, bbox[3] - bbox[1], conf
                    )
                    if reason:
                        hidden += 1
                        continue
                    adjusted_bbox = (
                        bbox[0] + x1,
                        bbox[1] + y1,
                        bbox[2] + x1,
                        bbox[3] + y1,
                    )
                    new_state.append((src_text, tgt_text, adjusted_bbox, fsz))

                self.status_msg.emit(
                    f"{self._ocr_backend_name()} OCR {len(src_texts)}줄 [{lang_summary}] → 번역 완료 ({dt:.0f} ms)"
                    + (f" · 숨김 {hidden}줄(DG-1)" if hidden else "")
                )

                # RT-2: 실질 동일한 상태면 재그리기 생략.
                self._emit_state_if_changed(new_state)

                if self.fail_streak > 0:
                    self.failure_alert.emit("번역 정지" if self.model_loaded else "번역 시작")
                self.fail_streak = 0

            except Exception as e:
                # E-8: 일시적 실패는 스킵, 한도 초과 시 backoff.
                self.fail_streak += 1
                print(f"[WARN] capture/translate 실패 #{self.fail_streak}: {e}")
                if self.fail_streak >= 20:
                    print("[WARN] 연속 실패 한도 — 5초 backoff 후 재시도")
                    self.failure_alert.emit("일시 정지(5s)")
                    self.fail_streak = 0
                    await asyncio.sleep(5.0)
                    continue

            await asyncio.sleep(0.25)

    # --- routing helpers -----------------------------------------------------
    def _src_hint(self):
        v = self._get_options().src_text
        return "auto" if v == "auto" else LANG_MAP[v]

    def _ocr_backend_name(self):
        return self._get_options().ocr_text

    # --- OCR 언어 라우팅 (SL-1) ----------------------------------------------
    def _ocr_langs(self):
        """auto = eng+kor 고정. OSD script 판별(UX-2)은 오판이 잦아 SL-1에서 퇴역."""
        src = self._src_hint()
        if src != "auto":
            tess = LANG_TO_TESS.get(src)
            if tess and tess in AVAILABLE_TESS_LANGS:
                return tess
            return "eng" if "eng" in AVAILABLE_TESS_LANGS else (
                "+".join(sorted(AVAILABLE_TESS_LANGS - {"osd"})[:2]) or "eng"
            )

        override = os.environ.get("COCKTAIL_OCR_LANGS_AUTO", "").strip()
        if override:
            wanted = [t.strip() for t in override.split("+") if t.strip()]
        else:
            wanted = list(OCR_LANGS_AUTO_DEFAULT)
        avail = [t for t in wanted if t in AVAILABLE_TESS_LANGS]
        return "+".join(avail) if avail else "eng"

    # --- OCR -----------------------------------------------------------------
    def perform_ocr_with_boxes(self, image):
        # DG-1: 라인별 평균 confidence를 표시 게이트에서 쓰기 위해 매 프레임 초기화한다.
        # (라인 튜플 자체는 (text, bbox, font_size) 3원소 계약 유지 — Paddle/UIA 경로와 공용.
        #  ponytail: 병렬 리스트 한 개면 충분하다. 계약을 바꾸면 5곳이 같이 흔들린다.)
        self._ocr_line_confs = []
        src_hint = self._src_hint()
        if self._get_options().ocr_text == "PaddleOCRv5":
            try:
                paddle_results = PADDLE_OCR.recognize(image, src_hint)
                if paddle_results is not None:
                    return paddle_results
                self.status_msg.emit("PaddleOCR 사용 불가 또는 초기화 실패. Tesseract로 폴백합니다.")
            except Exception as e:
                print(f"[WARN] PaddleOCR 실패: {e}. Tesseract로 폴백합니다.")
                self.status_msg.emit("PaddleOCR 추론 실패. Tesseract로 폴백합니다.")

        lang_str = self._ocr_langs()
        if not _TESSERACT_CMD:
            self.status_msg.emit(TESSERACT_ERROR)
            return []
        if not (AVAILABLE_TESS_LANGS - {"osd"}):
            self.status_msg.emit("Tesseract traineddata가 없습니다. tessdata 설치가 필요합니다.")
            return []
        try:
            # B-8: PIL ImageGrab은 RGB 순서.
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            h, w = gray.shape[:2]
            scale = OCR_UPSCALE_FACTOR if min(h, w) < 400 else 1
            self._ocr_scale = scale
            confs = []
            lines = self._tess_lines(gray, lang_str, scale, confs)

            # OU-1: 작은 글씨(캡션/각주)는 1차 OCR이 통째로 무너진다. 통과한 라인들의
            # 높이 중앙값이 임계 미만일 때만 확대 재OCR — 큰 글씨 화면은 여기 안 들어온다.
            if scale == 1 and lines:
                median_px = float(np.median([font_size for _, _, font_size in lines]))
                if median_px < OCR_UPSCALE_TRIGGER_PX:
                    self._ocr_scale = OCR_UPSCALE_FACTOR
                    confs = []
                    lines = self._tess_lines(gray, lang_str, OCR_UPSCALE_FACTOR, confs)
            self._ocr_line_confs = confs
            return lines
        except Exception as e:
            print(f"[WARN] OCR 실패: {e}")
            return []

    def _tess_lines(self, gray, lang_str, scale, conf_out=None):
        """grayscale 이미지를 scale배 확대 → 이진화 → Tesseract → 원본 좌표 라인.
        좌표 환원은 _lines_from_tess의 기존 scale 나눗셈이 담당한다.
        conf_out: 주면 라인별 평균 confidence를 같은 순서로 채운다 (DG-1)."""
        if scale != 1:
            h, w = gray.shape[:2]
            gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((1, 1), np.uint8))
        data = pytesseract.image_to_data(
            binary,
            lang=lang_str,
            output_type=pytesseract.Output.DICT,
            config=TESS_CONFIG,
        )
        return self._lines_from_tess(data, scale, conf_out)

    @staticmethod
    def _line_segments(words):
        """LM-1: 한 Tesseract 라인의 word들을 "실제로 한 줄인 것"으로 정리한다.

        1) 높이 이상치 word 제거 — 장식 글자/도형(그래프 축이 "|"로 잡히는 등)이 섞이면
           그 하나가 라인의 font_size와 bbox를 통째로 끌고 간다(실측: 본문 14px 라인이 81px).
        2) 큰 가로 간격에서 분할 — `--psm 6`은 좌우 두 단(column)을 한 라인으로 묶는다
           (실측: 255px 간격을 건너뛰어 bbox 폭 645px 한 덩어리).

        입력/출력 word 튜플: (text, left, top, width, height, conf)
        """
        if not words:
            return []
        words = sorted(words, key=lambda w: w[1])
        med_h = float(np.median([w[4] for w in words])) or 1.0
        kept = [w for w in words if w[4] <= med_h * OCR_WORD_HEIGHT_OUTLIER_RATIO]
        if not kept:
            kept = words
        gap_limit = max(8.0, med_h * OCR_LINE_SPLIT_GAP_RATIO)
        segments = [[kept[0]]]
        for prev, cur in zip(kept, kept[1:]):
            if cur[1] - (prev[1] + prev[3]) > gap_limit:
                segments.append([cur])
            else:
                segments[-1].append(cur)
        return segments

    @staticmethod
    def _lines_from_tess(data, scale, conf_out=None):
        lines = {}
        n = len(data["text"])
        for i in range(n):
            txt = data["text"][i]
            if not txt or not txt.strip():
                continue
            try:
                conf = float(data.get("conf", ["-1"] * n)[i])
            except (TypeError, ValueError):
                conf = -1.0
            if conf >= 0 and conf < 30:
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            entry = lines.setdefault(key, [])
            entry.append((
                txt,
                data["left"][i] // scale,
                data["top"][i] // scale,
                data["width"][i] // scale,
                data["height"][i] // scale,
                conf,  # NZ-1: 라인 평균 confidence 계산용 / DG-1: 표시 게이트용
            ))

        out = []
        for key in sorted(lines.keys()):
            # LM-1: 이상치 제거 + 큰 간격 분할 후 세그먼트별로 기존 필터를 적용한다.
            for words in BackgroundController._line_segments(lines[key]):
                full_text = " ".join(w[0] for w in words).strip()
                if not full_text:
                    continue
                # NZ-1: UI 부스러기(아이콘 글자, 시계, 테두리 잡음) 제외.
                confs = [w[5] for w in words if w[5] >= 0]
                avg_conf = (sum(confs) / len(confs)) if confs else -1.0
                if confs and avg_conf < OCR_LINE_MIN_CONF:
                    continue
                if _ocr_line_is_noise(full_text):
                    continue
                left = min(w[1] for w in words)
                top = min(w[2] for w in words)
                right = max(w[1] + w[3] for w in words)
                bottom = max(w[2] + w[4] for w in words)
                # LM-1: max → **중앙값**. max는 라인 안 이상치 하나에 그대로 끌려간다.
                font_size = max(1, int(round(float(np.median([w[4] for w in words])))))
                out.append((full_text, (left, top, right, bottom), font_size))
                if conf_out is not None:
                    conf_out.append(avg_conf)
        return out

    # --- sensitive window check ---------------------------------------------
    def _check_sensitive_pause(self):
        if sys.platform != "win32":
            return False
        try:
            fg = ctypes.windll.user32.GetForegroundWindow()
        except Exception:
            return False
        if not fg:
            return False
        if self._is_self_hwnd(int(fg)):
            return False
        return is_sensitive_window(fg)

    # --- UIA route -----------------------------------------------------------
    def _uia_collect_and_translate(self):
        """반환 의미 3종 (SV-1):
        - `UIA_SKIP_FRAME`: 민감 창 — **이 프레임 전체 스킵**. OCR 폴백 금지.
        - `None`: UIA 사용 불가/추출 실패 → OCR 폴백 OK.
        - list: 번역된 상태 (빈 리스트면 텍스트 없음).
        """
        if sys.platform != "win32":
            return None
        if not UIA_PROVIDER.is_available():
            return None
        try:
            fg = ctypes.windll.user32.GetForegroundWindow()
        except Exception:
            return None
        if not fg or self._is_self_hwnd(int(fg)):
            return []

        if is_sensitive_window(fg):
            # SV-1: 예전에는 []를 돌려줘 호출부가 "UIA 실패"로 읽고 OCR 경로를 계속 탔다.
            # MODE_BOX + UIA 명시 선택 조합에서는 S-1(박스 모드 제외)도 안 걸리므로
            # 민감 창이 그대로 캡처·OCR·번역·영구 캐시됐다. sentinel로 프레임을 끊는다.
            self._clear_overlay_if_any()   # RC-1
            if not self._sensitive_paused:
                self._sensitive_paused = True
                self.status_msg.emit("UIA: 민감 창 감지 — 일시 중지 (보안 원칙 #3)")
            return UIA_SKIP_FRAME
        if self._sensitive_paused:
            self._sensitive_paused = False
            self.status_msg.emit(f"{self.env_status} | 민감 창 해제 — UIA 재개")

        uia_results = UIA_PROVIDER.extract(fg)
        if not uia_results:
            return None
        if len(uia_results) > 100:
            uia_results = uia_results[:100]

        src_hint = self._src_hint()
        try:
            tgt_lang = LANG_MAP[self._get_options().tgt_text]
        except KeyError:
            tgt_lang = "ko"
        texts = [t for t, _, _ in uia_results]
        src_langs = assign_region_languages(uia_results, src_hint)
        try:
            translated = batch_translate(texts, src_langs, tgt_lang)
        except Exception as e:
            print(f"[WARN] UIA 번역 실패: {e}")
            return None

        new_state = []
        for (src_text, bbox, fsz), tgt_text in zip(uia_results, translated):
            if not (tgt_text and tgt_text.strip()):
                continue
            # DG-1: UIA 원문은 OCR 오독이 없으므로 conf=None (극소 라인 규칙은 자동 무효),
            # 번역 실패(목표 스크립트 없음/길이 붕괴)만 걸러진다.
            if display_gate_reject(src_text, tgt_text, tgt_lang, None, None):
                continue
            new_state.append((src_text, tgt_text, bbox, fsz))
        return new_state


class CocktailAppController:
    """Own the Qt app lifecycle and top-level windows.

    배포형 백그라운드 앱에서는 실행 생명주기를 한 곳에 모아야 한다.
    SettingsWindow는 설정/상태, OverlayWindow는 렌더링, 이 클래스는 생성/연결/실행만 담당한다.
    """

    def __init__(self, argv=None):
        self.argv = list(sys.argv if argv is None else argv)
        self.app = QApplication.instance() or QApplication(self.argv)
        self.app.setQuitOnLastWindowClosed(False)
        self.settings_window = None
        self.overlay_window = None
        self._geom_connected = set()   # MS-1: geometryChanged를 이미 연결한 QScreen

    def start(self) -> int:
        self.settings_window = SettingsWindow()
        self.overlay_window = OverlayWindow(self.settings_window)
        self.settings_window.overlay = self.overlay_window
        self.overlay_window.update_for_mode()
        self._wire_screen_changes()
        return int(self.app.exec())

    # --- MS-1 (A7): 모니터 구성 변경 대응 ------------------------------------
    def _wire_screen_changes(self):
        """모니터를 꽂거나 빼거나 해상도를 바꿔도 오버레이/캡처 좌표가 따라오게 한다.

        모든 갱신은 **메인 스레드**에서만 일어난다 — 워커는 region.screen_rects 캐시만 읽는다(GT-1).
        """
        self.app.screenAdded.connect(self._on_screens_changed)
        self.app.screenRemoved.connect(self._on_screens_changed)
        self._connect_screen_geometry()
        self._on_screens_changed()

    def _connect_screen_geometry(self):
        """해상도 변경(geometryChanged)까지 잡는다. 이미 연결한 화면은 건너뛴다.

        (Qt.UniqueConnection은 수신자가 QObject일 때만 확실해서, 연결한 화면을 직접 기억한다.)
        """
        current = self.app.screens()
        for screen in current:
            if screen in self._geom_connected:
                continue
            try:
                screen.geometryChanged.connect(self._on_screens_changed)
                self._geom_connected.add(screen)
            except Exception as e:
                print(f"[WARN] geometryChanged 연결 실패: {e}")
        # 빠진 모니터의 QScreen 참조는 들고 있지 않는다.
        self._geom_connected.intersection_update(current)

    def _on_screens_changed(self, *_args):
        self._connect_screen_geometry()   # 새로 붙은 화면의 geometryChanged도 잡는다
        if self.settings_window is not None:
            self.settings_window.region.refresh_screen_rects()
            editor = self.settings_window.region_editor
            if editor is not None and editor.isVisible():
                editor._fit_to_virtual_desktop()
        if self.overlay_window is not None:
            self.overlay_window._fit_to_screen()
            self.overlay_window.update()


def _run_self_test() -> int:
    """Packaged-app smoke test. It must not create QApplication or load models."""
    status, problems = environment_summary()
    print(f"[SELFTEST] {status}")

    required_modules = [
        "PySide6",
        "cv2",
        "PIL",
        "pytesseract",
        "torch",
        "transformers",
        "numpy",
    ]
    missing = [name for name in required_modules if not _module_available(name)]
    if missing:
        print("[SELFTEST] missing modules: " + ", ".join(missing))
        return 1

    if not _TESSERACT_CMD:
        print("[SELFTEST] missing Tesseract executable")
        return 1
    if "eng" not in AVAILABLE_TESS_LANGS:
        print("[SELFTEST] missing eng.traineddata")
        return 1

    key = PersistentCache._key("en", "ko", "self-test")
    if len(key) != 64:
        print("[SELFTEST] invalid persistent cache key")
        return 1

    if problems:
        print("[SELFTEST] non-blocking warnings: " + "; ".join(problems))
    print("[SELFTEST] OK")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv if argv is None else argv)
    if "--self-test" in argv:
        return _run_self_test()

    return CocktailAppController(argv).start()


if __name__ == "__main__":
    sys.exit(main())

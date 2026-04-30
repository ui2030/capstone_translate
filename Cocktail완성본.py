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
import sys
import time
import ctypes
import ctypes.wintypes  # T-2: GetWindowRect의 RECT 구조체용
import threading
import asyncio
import unicodedata
import importlib.util
from collections import OrderedDict

import numpy as np
import cv2
import torch
import pytesseract
from PIL import ImageGrab
from langdetect import detect, DetectorFactory, LangDetectException
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
)

from PySide6 import QtWidgets
from PySide6.QtCore import Qt, QRect, QPoint, Signal, Slot, QTimer
from PySide6.QtGui import (
    QPainter, QColor, QPen, QFont, QKeySequence, QShortcut,
    QIcon, QPixmap, QCursor, QAction,
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QComboBox, QPushButton, QSizeGrip, QLabel,
    QSystemTrayIcon, QMenu,
)

DetectorFactory.seed = 0  # langdetect 재현성


# --- Phase 1 (T-1, T-2): 캡처 모드 상수 + 글로벌 단축키 optional 의존성 -----
# 모드별 캡처 영역 결정: red_rect는 모든 모드의 공통 캡처 영역으로 동작 (워커 루프에서 자동 갱신).
MODE_BOX = "box"          # 사용자가 직접 빨간 박스로 영역 지정 (기본)
MODE_WINDOW = "window"    # 활성 창의 영역 자동 캡처
MODE_HOVER = "hover"      # 마우스 커서 주변 영역 캡처

_MODE_LABELS = {
    "영역 박스": MODE_BOX,
    "활성 창": MODE_WINDOW,
    "마우스 hover": MODE_HOVER,
}
_MODE_LABELS_REVERSE = {v: k for k, v in _MODE_LABELS.items()}

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
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if progress_cb: progress_cb("모델 다운로드/로드 중")
        m = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        if USE_FP16:
            m = m.half()
        m.eval()
        self.model = m
        if progress_cb: progress_cb("워밍업 중")
        try:
            with torch.inference_mode():
                enc = self.tok(["hello"], return_tensors="pt", padding=True).to(device)
                self.model.generate(**enc, max_new_tokens=8, num_beams=1)
        except Exception as e:
            print(f"[WARN] opus-mt 워밍업 실패: {e}")

    def translate_batch(self, texts, src, tgt):
        # opus-mt는 언어쌍 전용이라 src/tgt는 라우팅 검증용으로만 사용 (모델은 이미 결정됨)
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
        self.tok = M2M100Tokenizer.from_pretrained(model_name)
        m = M2M100ForConditionalGeneration.from_pretrained(model_name).to(device)
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

# langdetect 출력 → ISO 639-1 / m2m100 코드
_LANGDETECT_TO_ISO = {
    "af": "af",
    "ar": "ar",
    "bg": "bg",
    "bn": "bn",
    "ca": "ca",
    "cs": "cs",
    "cy": "cy",
    "da": "da",
    "de": "de",
    "el": "el",
    "ko": "ko",
    "en": "en",
    "es": "es",
    "et": "et",
    "fa": "fa",
    "fi": "fi",
    "fr": "fr",
    "gu": "gu",
    "he": "he",
    "hi": "hi",
    "hr": "hr",
    "hu": "hu",
    "id": "id",
    "it": "it",
    "ja": "ja",
    "lt": "lt",
    "lv": "lv",
    "mk": "mk",
    "ml": "ml",
    "mr": "mr",
    "ne": "ne",
    "nl": "nl",
    "no": "no",
    "pa": "pa",
    "pl": "pl",
    "pt": "pt",
    "ro": "ro",
    "ru": "ru",
    "sk": "sk",
    "sl": "sl",
    "so": "so",
    "sq": "sq",
    "sv": "sv",
    "sw": "sw",
    "ta": "ta",
    "te": "te",
    "th": "th",
    "tl": "tl",
    "tr": "tr",
    "uk": "uk",
    "ur": "ur",
    "vi": "vi",
    "zh-cn": "zh",
    "zh-tw": "zh",
    "zh": "zh",
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

_LATIN_SHORT_HINTS = {
    # langdetect가 짧은 라틴 인사/감사 표현에서 자주 틀리는 케이스만 좁게 보정한다.
    "thank you": "en",
    "thanks": "en",
    "gracias": "es",
    "muchas gracias": "es",
    "hola": "es",
    "merci": "fr",
    "bonjour": "fr",
    "danke": "de",
    "danke schon": "de",
    "grazie": "it",
    "ciao": "it",
    "obrigado": "pt",
    "obrigada": "pt",
    "hvala": "hr",
    "dziekuje": "pl",
    "dziekuje bardzo": "pl",
    "tack": "sv",
    "salamat": "tl",
    "terima kasih": "id",
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
    """문자열의 지배적인 Unicode script를 반환한다. 라틴은 langdetect 폴백 대상."""
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


def _latin_hint(text: str) -> str | None:
    normalized = unicodedata.normalize("NFKD", text or "")
    key_chars = []
    for ch in normalized:
        if unicodedata.category(ch).startswith("M"):
            continue
        if ch.isalpha() or ch.isspace():
            key_chars.append(ch.lower())
        else:
            key_chars.append(" ")
    key = " ".join("".join(key_chars).split())
    return _LATIN_SHORT_HINTS.get(key)


def identify_language(text: str, fallback: str = "en") -> str:
    """
    M-6: Unicode script로 확정 가능한 언어는 결정론적으로 처리하고,
    라틴 계열 짧은 텍스트만 langdetect에 맡긴다.
    """
    script, _ = detect_script(text)
    if script in _SCRIPT_TO_LANG:
        return _SCRIPT_TO_LANG[script]
    if script not in {"LATIN", "UNKNOWN", "OTHER"}:
        return fallback
    if script == "LATIN":
        hinted = _latin_hint(text)
        if hinted in M2M100_LANGS:
            return hinted
    try:
        code = detect(text)
    except LangDetectException:
        return fallback
    mapped = _LANGDETECT_TO_ISO.get(code, fallback)
    return mapped if mapped in M2M100_LANGS else fallback


def _safe_detect(text: str, fallback: str = "en") -> str:
    return identify_language(text, fallback=fallback)


def _language_confidence(text: str) -> int:
    """Region 언어 smoothing용 신뢰도. 비라틴 script > 라틴 힌트 > 긴 라틴 > 짧은 라틴."""
    script, counts = detect_script(text)
    letters = sum(counts.values())
    if script in _SCRIPT_TO_LANG and script != "CJK":
        return 4
    if script == "CJK":
        return 3
    if script == "LATIN":
        if _latin_hint(text):
            return 3
        return 2 if letters >= 12 else 1
    return 0


def _cluster_text_regions(ocr_results):
    """
    YOLO식 후처리: OCR 라인 객체를 화면상 가까운 region/column으로 묶는다.
    좌우에 다른 언어가 동시에 있을 때 짧은 단어가 반대쪽 문맥을 오염시키지 않도록 한다.
    """
    if not ocr_results:
        return []
    if len(ocr_results) == 1:
        return [[0]]

    items = []
    widths = []
    heights = []
    for i, (_, bbox, _) in enumerate(ocr_results):
        left, top, right, bottom = bbox
        width = max(1, right - left)
        height = max(1, bottom - top)
        items.append((i, left, top, right, bottom, (left + right) / 2.0))
        widths.append(width)
        heights.append(height)

    median_w = float(np.median(widths)) if widths else 80.0
    median_h = float(np.median(heights)) if heights else 20.0
    x_gap_threshold = max(90.0, median_w * 0.85)
    y_gap_threshold = max(45.0, median_h * 2.8)

    # 먼저 column 후보를 만든다. 좌우로 충분히 떨어진 자막/문단은 별도 region.
    columns = []
    for item in sorted(items, key=lambda it: it[5]):
        placed = False
        for col in columns:
            centers = [it[5] for it in col]
            if abs(item[5] - (sum(centers) / len(centers))) <= x_gap_threshold:
                col.append(item)
                placed = True
                break
        if not placed:
            columns.append([item])

    regions = []
    for col in columns:
        current = []
        last_bottom = None
        for item in sorted(col, key=lambda it: (it[2], it[1])):
            if last_bottom is not None and item[2] - last_bottom > y_gap_threshold:
                if current:
                    regions.append([it[0] for it in current])
                current = []
            current.append(item)
            last_bottom = max(last_bottom if last_bottom is not None else item[4], item[4])
        if current:
            regions.append([it[0] for it in current])
    return regions


def assign_region_languages(ocr_results, src_lang_hint):
    """
    src=auto일 때 라인별 감지 결과를 공간 region 단위로 안정화한다.
    사용자가 source 언어를 명시하면 그대로 강제한다.
    """
    if src_lang_hint != "auto":
        return [src_lang_hint] * len(ocr_results)

    detected = []
    confidence = []
    for text, _, _ in ocr_results:
        detected.append(identify_language(text))
        confidence.append(_language_confidence(text))

    final = list(detected)
    for region in _cluster_text_regions(ocr_results):
        votes = {}
        for idx in region:
            lang = detected[idx]
            weight = confidence[idx]
            if weight <= 0:
                continue
            votes[lang] = votes.get(lang, 0) + weight
        if not votes:
            continue
        region_lang, region_weight = max(votes.items(), key=lambda kv: kv[1])
        for idx in region:
            # 짧고 불안정한 라틴/불명 라인만 region 문맥으로 보정한다.
            if confidence[idx] <= 1 and region_weight >= 3:
                final[idx] = region_lang
    return final


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
    pending_src = []

    src_list = src_lang_hint if isinstance(src_lang_hint, list) else None

    for i, t in enumerate(texts):
        if src_list is not None:
            src = src_list[i] if i < len(src_list) else "en"
        else:
            src = src_lang_hint if src_lang_hint != "auto" else _safe_detect(t)
        if src == tgt_lang:
            results[i] = t
            continue
        key = (src, tgt_lang, t)
        with CACHE_LOCK:
            cached = TRANS_CACHE.get_or_none(key)
        if cached is None:
            # Phase 4: 영구 캐시(DPAPI 암호화)에서도 조회. 메모리 캐시로 승격.
            try:
                disk = PERSIST_CACHE.get(src, tgt_lang, t)
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
                TRANS_CACHE.put((src, tgt_lang, pending_texts[j]), tgt_text)
            # Phase 4: 영구 캐시도 갱신 (저장은 closeEvent/주기적으로)
            try:
                PERSIST_CACHE.put(src, tgt_lang, pending_texts[j], tgt_text)
            except Exception:
                pass

    return results


# --- 화면 변화 감지 (P-4) ---------------------------------------------------
def cheap_hash(img_np: np.ndarray) -> bytes:
    small = cv2.resize(img_np, (16, 16), interpolation=cv2.INTER_AREA)
    if small.ndim == 3:
        small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return small.tobytes()


def hash_distance(a: bytes, b: bytes) -> int:
    if a is None or b is None or len(a) != len(b):
        return 1 << 30
    arr_a = np.frombuffer(a, dtype=np.uint8)
    arr_b = np.frombuffer(b, dtype=np.uint8)
    return int(np.abs(arr_a.astype(np.int16) - arr_b.astype(np.int16)).mean())


# --- PySide6 UI (E-10) -----------------------------------------------------
# PySide6/Qt6은 high-DPI를 자동 활성화 (E-7 정책과 일치).


class TransparentWindow(QMainWindow):
    # 워커 → 메인 시그널 (B-11 race 차단)
    state_ready = Signal(list)
    failure_alert = Signal(str)
    model_ready = Signal()
    progress_msg = Signal(str)  # P-9: 모델 로드 진행 상황
    status_msg = Signal(str)    # U-1: 일반 사용자용 상태/진단 표시

    # T-1: 글로벌 단축키 → 메인 스레드 진입용 시그널 (keyboard 콜백은 별도 스레드)
    global_toggle_signal = Signal()
    global_show_signal = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        # Phase 1C: OverlayWindow 참조 (main에서 setter로 주입). 없을 수도 있음.
        self.overlay = None

        # 상태
        self.translated_text = []   # [(src_text, tgt_text, bbox, font_size), ...]
        self.stop_thread = False
        self.model_loaded = False
        self.last_frame_hash = None
        self.fail_streak = 0
        self.no_ocr_streak = 0
        self.capturable = False
        self._model_ok = False  # P-9: 모델 lazy load 완료 플래그
        self._env_status, self._env_problems = environment_summary()

        # T-2: 캡처 모드 상태
        self.capture_mode = MODE_BOX
        self.hover_size = (400, 200)  # 마우스 hover 영역 (width, height)

        # T-1: 트레이 종료 vs hide 분기
        self._is_quitting = False

        # W-1 (Phase 1B): 마우스 통과 토글
        self._mouse_through = False

        # S-1 (Phase 5): 민감 창 일시 중지 상태
        self._sensitive_paused = False

        # 윈도우
        self.setWindowTitle("Cocktail Translator")
        self.setGeometry(100, 100, 880, 600)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Window)
        self.setAttribute(Qt.WA_TranslucentBackground)

        # E-7: PySide6 자동 high-dpi → _scale=1로 고정
        self._scale = 1.0
        self.px = lambda v: int(v)

        self.initUI()

        # 시그널/슬롯 연결 (메인 스레드에서 상태 갱신)
        self.state_ready.connect(self._apply_state)
        self.failure_alert.connect(self._on_failure)
        self.model_ready.connect(self._on_model_ready)
        self.progress_msg.connect(self._on_progress)
        self.status_msg.connect(self._set_status)
        self.global_toggle_signal.connect(self._on_global_toggle)
        self.global_show_signal.connect(self._on_global_show)

        self.show()
        self._set_status(self._env_status)

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
        threading.Thread(target=self._preload_model, daemon=True).start()

        # Phase 4: 영구 캐시도 백그라운드 로드
        PERSIST_CACHE.preload_async()

        # Phase 4 보강 (A-1): 비정상 종료 손실 방지 — 30초마다 자동 save (변경 없으면 no-op)
        self._cache_save_timer = QTimer(self)
        self._cache_save_timer.timeout.connect(self._on_cache_save_tick)
        self._cache_save_timer.start(30 * 1000)

    @Slot()
    def _on_cache_save_tick(self):
        try:
            PERSIST_CACHE.save()
        except Exception as e:
            print(f"[WARN] 주기적 캐시 save 실패: {e}")

        # 워커 스레드 (모델 준비 + model_loaded=True 둘 다 만족할 때만 실제 캡처)
        self.translation_thread = threading.Thread(
            target=self.run_asyncio_loop, daemon=True
        )
        self.translation_thread.start()

        self.old_pos = QPoint()

    def _preload_model(self):
        try:
            TRANSLATOR.preload_default(progress_cb=lambda msg: self.progress_msg.emit(msg))
            self.model_ready.emit()
        except Exception as e:
            print(f"[ERROR] 모델 로드 실패: {e}")
            self.failure_alert.emit("모델 로드 실패")

    @Slot()
    def _on_model_ready(self):
        self._model_ok = True
        self.run_button.setEnabled(True)
        self.run_button.setText("번역 시작")
        self._set_status(f"{self._env_status} | 모델 준비 완료")

    @Slot(str)
    def _on_progress(self, msg):
        self.run_button.setText(f"준비: {msg[:14]}")
        self._set_status(f"모델 준비 중: {msg}")

    @Slot(list)
    def _apply_state(self, new_state):
        self.translated_text = new_state
        self.update()
        # Phase 1C: OverlayWindow도 함께 갱신
        if hasattr(self, "overlay") and self.overlay is not None:
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
        new_state = not self.capturable
        if not set_window_capture_affinity(self.winId(), exclude=not new_state):
            print("[WARN] 캡처 affinity 변경 실패")
            return
        self.capturable = new_state
        self.capture_btn.setText("스샷OK" if self.capturable else "스샷차단")
        if self.capturable and self.model_loaded:
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

        # W-1: 마우스 통과 토글 (checkable)
        self._mouse_through_action = QAction("마우스 통과 (Ctrl+Shift+P)", self)
        self._mouse_through_action.setCheckable(True)
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
        self.show()
        self.raise_()
        self.activateWindow()

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
            self._set_status(f"{self._env_status} | 자동 실행 변경 실패 (Windows 전용)")
            return
        self._set_status(f"{self._env_status} | 자동 실행: {'ON' if checked else 'OFF'}")

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
        if self._model_ok:
            self.toggle_running()
        else:
            print("[INFO] 모델 준비 중 — 잠시 후 다시 시도하세요.")

    @Slot()
    def _on_global_show(self):
        self._show_normal()

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
        self.last_frame_hash = None
        # Phase 1C: OverlayWindow 마우스 통과 정책/그리기 갱신
        if hasattr(self, "overlay") and self.overlay is not None:
            self.overlay.update_for_mode()
            self.overlay.update()
        self._set_status(f"{self._env_status} | 모드: {label or mode}")

    def _foreground_window_rect(self):
        """T-2: Windows 활성 창의 절대 좌표. 우리 자신이면 None."""
        if sys.platform != "win32":
            return None
        try:
            user32 = ctypes.windll.user32
            hwnd = user32.GetForegroundWindow()
            if not hwnd:
                return None
            # B-10 피드백 방지: 우리 자신이 활성이면 캡처 X
            if hwnd == int(self.winId()):
                return None
            rect = ctypes.wintypes.RECT()
            if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                return None
            return (int(rect.left), int(rect.top), int(rect.right), int(rect.bottom))
        except Exception as e:
            print(f"[WARN] GetForegroundWindow 실패: {e}")
            return None

    def _mouse_hover_rect(self):
        """T-2: 마우스 커서 주변 절대 좌표 영역."""
        pos = QCursor.pos()
        w, h = self.hover_size
        return (int(pos.x() - w // 2), int(pos.y() - h // 2),
                int(pos.x() + w // 2), int(pos.y() + h // 2))

    def _update_red_rect_for_mode(self):
        """T-2: 현재 모드에 따라 red_rect를 자동 갱신.
        red_rect는 위젯 내 좌표. 절대 좌표 → 위젯 좌표 변환."""
        if self.capture_mode == MODE_BOX:
            return  # 사용자 수동 (red_rect 그대로)
        if self.capture_mode == MODE_WINDOW:
            rect = self._foreground_window_rect()
        elif self.capture_mode == MODE_HOVER:
            rect = self._mouse_hover_rect()
        else:
            return
        if not rect:
            return
        ax1, ay1, ax2, ay2 = rect
        if ax2 - ax1 < 8 or ay2 - ay1 < 8:
            return
        wx, wy = self.geometry().x(), self.geometry().y()
        self.red_rect = QRect(ax1 - wx, ay1 - wy, ax2 - ax1, ay2 - ay1)

    # --- W-1 (Phase 1B → Phase 1C 보강): OverlayWindow 마우스 통과 토글 ------
    # 박사 자체 리뷰 결과: W-1 토글은 OverlayWindow에 적용되어야 함.
    # ControlWindow에 적용하면 사용자가 자기 컨트롤도 못 누름.
    def toggle_mouse_through(self):
        target = self.overlay if self.overlay is not None else self
        new_state = not self._mouse_through
        if not set_mouse_through(target.winId(), new_state):
            print("[WARN] 마우스 통과 설정 실패 (Windows 전용)")
            return
        self._mouse_through = new_state
        if hasattr(self, "_mouse_through_action"):
            self._mouse_through_action.setChecked(new_state)
        target_name = "OverlayWindow" if self.overlay is not None else "ControlWindow"
        msg = f"ON ({target_name}) — 클릭이 밑 창으로 통과" if new_state else "OFF"
        self._set_status(f"{self._env_status} | 마우스 통과: {msg}")

    # --- Phase 2/3: UIA 모드 — 활성 창 텍스트 직접 → 번역 -------------------
    def _uia_collect_and_translate(self):
        """반환: new_state list (성공) 또는 None (UIA 사용 불가/실패 → OCR 폴백)."""
        if sys.platform != "win32":
            return None
        if not UIA_PROVIDER.is_available():
            return None
        try:
            fg = ctypes.windll.user32.GetForegroundWindow()
        except Exception:
            return None
        if not fg or fg == int(self.winId()):
            # B-10 피드백 가드: 우리 자신이면 무시
            return []
        # 자식 OverlayWindow의 hwnd도 무시
        try:
            if self.overlay is not None and fg == int(self.overlay.winId()):
                return []
        except Exception:
            pass

        # 박사 자체 리뷰 결함 6: UIA 모드는 mode_combo와 무관하게 활성 창 자체로 작동.
        # 민감 창 검사는 항상 적용 (보안 원칙 #3).
        if is_sensitive_window(fg):
            if self.translated_text:
                self.state_ready.emit([])
            if not self._sensitive_paused:
                self._sensitive_paused = True
                self.status_msg.emit("UIA: 민감 창 감지 — 일시 중지 (보안 원칙 #3)")
            return []
        if self._sensitive_paused:
            self._sensitive_paused = False
            self.status_msg.emit(f"{self._env_status} | 민감 창 해제 — UIA 재개")

        uia_results = UIA_PROVIDER.extract(fg)
        if not uia_results:
            return None  # 데이터 없음 → 폴백 신호
        # 박사 자체 리뷰 결함 5: paintEvent 부담 회피 — 상위 100개만
        if len(uia_results) > 100:
            uia_results = uia_results[:100]

        # 번역 (M-6/M-7 자동 라우팅 그대로 사용)
        src_hint = self._src_hint()
        try:
            tgt_lang = LANG_MAP[self.tgt_combo.currentText()]
        except KeyError:
            tgt_lang = "ko"
        texts = [t for t, _, _ in uia_results]
        try:
            translated = batch_translate(texts, src_hint, tgt_lang)
        except Exception as e:
            print(f"[WARN] UIA 번역 실패: {e}")
            return None

        new_state = []
        for (src_text, bbox, fsz), tgt_text in zip(uia_results, translated):
            if not (tgt_text and tgt_text.strip()):
                continue
            # bbox는 이미 화면 절대 좌표 (UIA BoundingRectangle)
            new_state.append((src_text, tgt_text, bbox, fsz))
        return new_state

    # --- S-1 (Phase 5): 민감 창 검사 + 일시 중지 -----------------------------
    def _check_sensitive_pause(self):
        """활성 창이 민감 키워드를 포함하면 True (워커는 OCR 건너뜀).
        false positive를 줄이기 위해 박스 모드에선 적용 X — 사용자가 명시적으로 박스 지정한 경우엔
        그 박스 안만 보므로 보안 위험이 모드 비교 시 낮음."""
        if sys.platform != "win32":
            return False
        try:
            fg = ctypes.windll.user32.GetForegroundWindow()
        except Exception:
            return False
        if not fg:
            return False
        # 우리 자신은 무시 (B-10 자기 캡처 가드와 동일)
        try:
            if fg == int(self.winId()):
                return False
        except Exception:
            pass
        return is_sensitive_window(fg)

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

        # T-2: 캡처 모드 (영역 박스 / 활성 창 / 마우스 hover)
        self.mode_combo = QComboBox(self)
        for label in _MODE_LABELS.keys():
            self.mode_combo.addItem(label)
        self.mode_combo.setGeometry(px(355), px(10), px(110), px(30))
        self.mode_combo.setToolTip("영역 박스: 빨간 박스 안만 / 활성 창: 현재 활성 창 자동 / 마우스 hover: 커서 주변")
        self.mode_combo.currentTextChanged.connect(self._on_mode_combo_changed)

        # 번역 시작/정지
        self.run_button = QPushButton("번역 시작", self)
        self.run_button.setGeometry(px(470), px(10), px(100), px(30))
        self.run_button.clicked.connect(self.toggle_running)

        # 캡처 토글 (E-9)
        self.capture_btn = QPushButton("스샷차단", self)
        self.capture_btn.setGeometry(px(575), px(10), px(90), px(30))
        self.capture_btn.setToolTip("스크린샷에 번역창을 포함할지 토글 (Ctrl+Shift+H)")
        self.capture_btn.clicked.connect(self.toggle_capturable)

        # 종료
        self.close_button = QPushButton("X", self)
        self.close_button.setGeometry(px(680), px(10), px(30), px(30))
        self.close_button.clicked.connect(self.close)

        # 상태/진단 표시 (U-1)
        self.status_label = QLabel("환경 확인 중", self)
        self.status_label.setGeometry(px(10), px(45), px(700), px(24))
        self.status_label.setStyleSheet(
            "QLabel { color: white; background: rgba(20, 20, 20, 210); "
            "padding: 3px 6px; border-radius: 4px; }"
        )

        # 캡처 영역
        self.red_rect = QRect(px(100), px(75), px(600), px(375))

        # 리사이즈 핸들
        self.size_grip = QSizeGrip(self)
        self.size_grip.setGeometry(
            self.red_rect.right() - px(10),
            self.red_rect.bottom() - px(10),
            px(20), px(20),
        )

    def resizeEvent(self, event):
        px = self.px
        self.red_rect.setWidth(int(self.width() - px(200)))
        self.red_rect.setHeight(int(self.height() - px(175)))
        self.status_label.setFixedWidth(max(px(240), self.width() - px(20)))
        self.size_grip.setGeometry(
            self.red_rect.right() - px(18),
            self.red_rect.bottom() - px(15),
            px(24), px(20),
        )
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
        if not self._model_ok:
            print("[INFO] 모델 준비 중 — 잠시만 기다려주세요.")
            self._set_status("모델 준비 중입니다. 완료 후 번역을 시작할 수 있습니다.")
            return
        self.model_loaded = not self.model_loaded
        self.run_button.setText("번역 정지" if self.model_loaded else "번역 시작")
        self._set_status("번역 실행 중" if self.model_loaded else "번역 정지")

    # --- 비동기 워커 ---------------------------------------------------------
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
                # S-1 (Phase 5): 박스 모드 외에서 민감 창이 활성이면 OCR/번역 일시 중지 (보안 원칙)
                if self.capture_mode != MODE_BOX and self._check_sensitive_pause():
                    if self.translated_text:
                        self.state_ready.emit([])
                    if not self._sensitive_paused:
                        self._sensitive_paused = True
                        self.status_msg.emit("민감 창 감지 — 번역 일시 중지 (보안 원칙 #3)")
                    await asyncio.sleep(1.0)
                    continue
                if self._sensitive_paused:
                    self._sensitive_paused = False
                    self.status_msg.emit(f"{self._env_status} | 민감 창 해제 — 번역 재개")

                # T-2: 모드에 따라 red_rect 자동 갱신 (활성 창/hover). 박스 모드는 그대로.
                self._update_red_rect_for_mode()

                x1 = int(self.geometry().x() + self.red_rect.x())
                y1 = int(self.geometry().y() + self.red_rect.y())
                x2 = int(x1 + self.red_rect.width())
                y2 = int(y1 + self.red_rect.height())
                if x2 - x1 < 8 or y2 - y1 < 8:
                    await asyncio.sleep(0.2)
                    continue

                # Phase 2/3: UIA 모드면 OCR 건너뛰고 활성 창 텍스트 직접 추출
                if self.ocr_combo.currentText() == "UIA (활성 창)":
                    uia_state = self._uia_collect_and_translate()
                    if uia_state is not None:
                        self.state_ready.emit(uia_state)
                        if self.fail_streak > 0:
                            self.failure_alert.emit("번역 정지" if self.model_loaded else "번역 시작")
                        self.fail_streak = 0
                        await asyncio.sleep(0.5)  # UIA는 OCR보다 빠르지만 활성 창 변화 빈도 낮음
                        continue
                    # UIA 결과 없거나 실패 → OCR 폴백 (기존 경로 진행)

                screenshot = ImageGrab.grab(bbox=(x1, y1, x2, y2))
                screenshot_np = np.array(screenshot)

                # frame-skip (P-4)
                h = cheap_hash(screenshot_np)
                if hash_distance(self.last_frame_hash, h) < 2:
                    await asyncio.sleep(0.25)
                    continue
                self.last_frame_hash = h

                # OCR — Tesseract 기본, PaddleOCRv5 선택 백엔드(O-1)
                ocr_results = self.perform_ocr_with_boxes(screenshot_np)

                if not ocr_results:
                    if self.translated_text:
                        self.state_ready.emit([])
                    self.no_ocr_streak += 1
                    if self.no_ocr_streak in {1, 5, 20}:
                        self.status_msg.emit(f"OCR 결과 없음: 영역/언어/OCR 백엔드 확인 필요 ({self._ocr_backend_name()})")
                    await asyncio.sleep(0.3)
                    continue
                self.no_ocr_streak = 0

                src_hint = self._src_hint()
                tgt_lang = LANG_MAP[self.tgt_combo.currentText()]
                src_texts = [t for t, _, _ in ocr_results]
                src_langs = assign_region_languages(ocr_results, src_hint)

                t0 = time.perf_counter()
                translated = batch_translate(src_texts, src_langs, tgt_lang)
                dt = (time.perf_counter() - t0) * 1000.0
                print(f"[PERF] translate {len(src_texts)} lines in {dt:.1f} ms")
                lang_summary = ",".join(sorted(set(src_langs)))
                self.status_msg.emit(
                    f"{self._ocr_backend_name()} OCR {len(src_texts)}줄 [{lang_summary}] → 번역 완료 ({dt:.0f} ms)"
                )

                # B-3: 매 프레임 화면 기준으로 재구성 (누적 X)
                # Phase 1C: bbox는 화면 절대 좌표로 저장 — OverlayWindow가 그대로 그림.
                new_state = []
                for (src_text, bbox, fsz), tgt_text in zip(ocr_results, translated):
                    if not (tgt_text and tgt_text.strip()):
                        continue
                    adjusted_bbox = (
                        bbox[0] + x1,
                        bbox[1] + y1,
                        bbox[2] + x1,
                        bbox[3] + y1,
                    )
                    new_state.append((src_text, tgt_text, adjusted_bbox, fsz))

                # B-11: 메인 스레드로 상태 전달
                self.state_ready.emit(new_state)

                if self.fail_streak > 0:
                    self.failure_alert.emit("번역 정지" if self.model_loaded else "번역 시작")
                self.fail_streak = 0

            except Exception as e:
                # E-8: 일시적 실패는 스킵, 한도 초과 시 backoff (스레드 죽이지 않음)
                self.fail_streak += 1
                print(f"[WARN] capture/translate 실패 #{self.fail_streak}: {e}")
                if self.fail_streak >= 20:
                    print("[WARN] 연속 실패 한도 — 5초 backoff 후 재시도")
                    self.failure_alert.emit("일시 정지(5s)")
                    self.fail_streak = 0
                    await asyncio.sleep(5.0)
                    continue

            await asyncio.sleep(0.25)

    def _src_hint(self):
        v = self.src_combo.currentText()
        return "auto" if v == "auto" else LANG_MAP[v]

    def _ocr_backend_name(self):
        return self.ocr_combo.currentText()

    def _ocr_langs(self):
        """B-14: 선택된 src 언어에 따라 Tesseract lang 문자열 동적 생성."""
        src = self._src_hint()
        if src == "auto":
            # auto: 사용 가능한 모든 traineddata (osd 제외) — 다국어 자막에 강건
            usable = sorted(AVAILABLE_TESS_LANGS - {"osd"})
            return "+".join(usable) if usable else "eng"
        tess = LANG_TO_TESS.get(src)
        if tess and tess in AVAILABLE_TESS_LANGS:
            return tess
        # 사용자가 선택한 언어 traineddata가 없으면 eng로 폴백
        return "eng" if "eng" in AVAILABLE_TESS_LANGS else "+".join(sorted(AVAILABLE_TESS_LANGS - {"osd"})) or "eng"

    # --- OCR ----------------------------------------------------------------
    def perform_ocr_with_boxes(self, image):
        src_hint = self._src_hint()
        if self.ocr_combo.currentText() == "PaddleOCRv5":
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
            # B-8: PIL ImageGrab은 RGB 순서
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            h, w = gray.shape[:2]
            if min(h, w) < 400:
                scale = 2
                gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
                self._ocr_scale = scale
            else:
                self._ocr_scale = 1

            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((1, 1), np.uint8))

            data = pytesseract.image_to_data(
                binary,
                lang=lang_str,
                output_type=pytesseract.Output.DICT,
                config=TESS_CONFIG,
            )
            return self._lines_from_tess(data, self._ocr_scale)
        except Exception as e:
            print(f"[WARN] OCR 실패: {e}")
            return []

    @staticmethod
    def _lines_from_tess(data, scale):
        """Tesseract block/par/line 인덱스로 라인 그룹핑 (B-4: line_height 휴리스틱보다 정확)."""
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
            ))

        out = []
        for key in sorted(lines.keys()):
            words = lines[key]
            full_text = " ".join(w[0] for w in words).strip()
            if not full_text:
                continue
            left = min(w[1] for w in words)
            top = min(w[2] for w in words)
            right = max(w[1] + w[3] for w in words)
            bottom = max(w[2] + w[4] for w in words)
            font_size = max(w[4] for w in words)
            out.append((full_text, (left, top, right, bottom), font_size))
        return out

    # --- 그리기 --------------------------------------------------------------
    @staticmethod
    def _fit_font(painter, base_rect, text, initial_size, min_size=9):
        """B-13/C: 폰트를 점진적으로 축소해 needed.height ≤ base의 2배가 되는 가장 큰 폰트 반환."""
        target_max_h = max(base_rect.height() * 2, base_rect.height() + 4)
        size = initial_size
        font = QFont("Arial", size)
        needed = base_rect
        while size >= min_size:
            font = QFont("Arial", size)
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

    def draw_translated_text(self, painter, text, bbox, font_size):
        if not (isinstance(bbox, tuple) and len(bbox) == 4):
            return
        left, top, right, bottom = bbox

        initial_size = max(10, int(round(font_size * self._scale * 0.9)))
        base_rect = QRect(int(left), int(top), int(right - left), int(bottom - top))

        # B-12 + B-13: 폰트 축소 + height 동적 확장
        font, needed = self._fit_font(painter, base_rect, text, initial_size)
        painter.setFont(font)

        expanded_h = max(base_rect.height(), needed.height() + 4)
        expanded = QRect(base_rect.x(), base_rect.y(), base_rect.width(), expanded_h)

        rect = expanded.intersected(self.rect())
        if rect.isEmpty():
            return

        # B-12: alpha 235로 영문 차단
        painter.setBrush(QColor(20, 20, 20, 235))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(rect, int(8 * self._scale), int(8 * self._scale))

        painter.setPen(QPen(QColor(255, 255, 255), int(1 * self._scale)))
        painter.drawText(
            rect,
            Qt.AlignLeft | Qt.AlignVCenter | Qt.TextWordWrap,
            text,
        )

    def paintEvent(self, event):
        # Phase 1C: 그리기는 OverlayWindow가 화면 절대 좌표로 담당.
        # ControlWindow(=TransparentWindow)는 컨트롤 위젯만 보여줌.
        # 빈 paintEvent — 부모 기본 처리.
        super().paintEvent(event)

    # --- 종료 / 마우스 -------------------------------------------------------
    def closeEvent(self, event):
        # T-1: 트레이 메뉴 "종료"를 거치지 않은 닫기는 hide → 트레이로 보냄
        if not self._is_quitting and hasattr(self, "tray") and self.tray.isVisible():
            event.ignore()
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

        # 진짜 종료
        self.stop_thread = True
        if _KEYBOARD_OK:
            try:
                _keyboard.unhook_all_hotkeys()
            except Exception:
                pass
        if self.translation_thread.is_alive():
            self.translation_thread.join(timeout=1.5)
        # Phase 4: 영구 캐시를 디스크로 저장 (DPAPI 암호화)
        try:
            PERSIST_CACHE.save()
        except Exception:
            pass
        # Phase 1C: OverlayWindow 동시 종료
        if self.overlay is not None:
            try:
                self.overlay.close()
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
        # 개발 시: python.exe Cocktail완성본.py
        return f'"{exe}" "{script}"'
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
                self._mem = json.loads(decrypted.decode("utf-8"))
                print(f"[INFO] PersistentCache 로드: {len(self._mem)} 항목")
            except Exception as e:
                print(f"[WARN] PersistentCache 로드 실패: {e}")
                self._mem = {}

    def save(self):
        if not self._enabled or not self._dirty:
            return
        try:
            import json
            data = json.dumps(self._mem, ensure_ascii=False).encode("utf-8")
            blob = dpapi_protect(data)
            if not blob:
                return
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "wb") as f:
                f.write(blob)
            os.replace(tmp, self.path)
            self._dirty = False
        except Exception as e:
            print(f"[WARN] PersistentCache 저장 실패: {e}")

    def get(self, src: str, tgt: str, text: str):
        if not self._enabled:
            return None
        if not self._loaded:
            self._load()  # 동기 로드 (이미 preload_async가 마쳐있을 가능성 큼)
        return self._mem.get(self._key(src, tgt, text))

    def put(self, src: str, tgt: str, text: str, translation: str):
        if not self._enabled:
            return
        if not self._loaded:
            self._load()
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
# UI가 뜬 직후 백그라운드로 로드 (TransparentWindow.__init__에서 호출)


# --- Phase 1C: OverlayWindow — 분리된 투명 오버레이 ---------------------------
# 비전 V-1의 매끄러운 작동: 컨트롤 윈도우(TransparentWindow)와 오버레이 윈도우를 분리.
# OverlayWindow는 전체 화면 + 투명 + 마우스 통과(자동) + 캡처 차단(B-10) + 그리기만 담당.
# TransparentWindow는 콤보/버튼/상태/워커/시그널을 모두 보유 (사용자가 컨트롤 클릭 가능).
class OverlayWindow(QMainWindow):
    """전체 화면 투명 오버레이 — 빨간 박스/번역 텍스트만 그림. 마우스는 통과(모드별 자동)."""

    def __init__(self, control: "TransparentWindow"):
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

        # B-10: 자기 캡처 피드백 차단
        if not set_window_capture_affinity(self.winId(), exclude=True):
            print("[WARN] OverlayWindow 캡처 제외 실패")

        # 모드에 따른 마우스 통과 자동 — 박스 모드도 통과 ON으로 (사용자는 ControlWindow에서 박스 위치 조정)
        # 활성 창/hover/박스 모두 통과 ON. 사용자가 OverlayWindow 위에서 다른 창 자유롭게 클릭.
        self.update_for_mode()

    def _fit_to_screen(self):
        # Phase 1D: 멀티 모니터 — 가상 데스크톱 전체 영역으로 확장.
        # 각 화면의 geometry는 절대 좌표 (보조 모니터는 음수 또는 +width 등).
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

    # --- 그리기 (TransparentWindow에서 옮겨온 로직, 화면 절대 좌표 기반) -------
    @staticmethod
    def _fit_font(painter, base_rect, text, initial_size, min_size=9):
        """B-13/C: 폰트를 점진적으로 축소해 needed.height ≤ base의 2배가 되는 가장 큰 폰트 반환."""
        target_max_h = max(base_rect.height() * 2, base_rect.height() + 4)
        size = initial_size
        font = QFont("Arial", size)
        needed = base_rect
        while size >= min_size:
            font = QFont("Arial", size)
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

    def _draw_one(self, painter, text, bbox, font_size):
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

        rect = expanded.intersected(self.rect())
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
            cx = ctl.geometry().x()
            cy = ctl.geometry().y()
            rect = ctl.red_rect
            # 화면 절대 → 윈도우 내
            screen_rect = QRect(
                cx + rect.x() - wx,
                cy + rect.y() - wy,
                rect.width(),
                rect.height(),
            )
            pen = QPen(QColor(255, 0, 0), 3)
            painter.setPen(pen)
            painter.drawRect(screen_rect)

            # 번역 오버레이 (bbox는 화면 절대 → 윈도우 내로 변환)
            for _src, tgt, bbox, fsz in ctl.translated_text:
                local_bbox = (
                    bbox[0] - wx, bbox[1] - wy,
                    bbox[2] - wx, bbox[3] - wy,
                )
                self._draw_one(painter, tgt, local_bbox, fsz)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TransparentWindow()
    overlay = OverlayWindow(window)
    # Phase 1C: ControlWindow ↔ OverlayWindow 연결
    window.overlay = overlay
    overlay.update_for_mode()  # 초기 모드(박스) 반영
    sys.exit(app.exec())

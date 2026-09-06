"""Cocktail — Windows 시스템 통합 계층.

창 속성(캡처 제외/항상 위/마우스 통과), 커서 좌표, 민감 창 판정, DPAPI 암호화,
자동 실행 레지스트리. **의존 방향의 바닥**이라 Qt/OCR/번역을 import 하지 않는다.
(cursor_pos_physical의 비-Windows 폴백만 Qt를 지연 import 한다.)
"""
import ctypes
import ctypes.wintypes   # T-2: GetWindowRect의 RECT 구조체용
import os
import sys


# --- EN-1: 콘솔 인코딩 안전화 -----------------------------------------------
# 한국어 Windows 콘솔은 기본이 cp949라 `—`, `→`, `…` 를 못 찍는다. 그런 print 한 줄이
# UnicodeEncodeError를 던지고, import 도중이면 프로세스가 통째로 죽는다
# (`[INFO] uiautomation 미설치 — ...` 한 줄에서 실제로 재현). 진단 출력이 앱을 죽이면 안 된다.
#
# **의존 방향의 바닥인 여기에 둔다** — 엔트리(cocktail.py)에만 두면 모듈을 직접 import 하는
# 경로(테스트 프로브, bench/score.py, 외부 스크립트)가 전부 그대로 죽는다. 위 계층은
# 이 모듈을 import 하는 것만으로 안전해진다.
def _make_console_utf8_safe():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass   # 콘솔 없이 뜨면(pythonw/패키지 exe) buffer가 없다 — 그때는 할 일도 없다


_make_console_utf8_safe()


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
SWP_NOZORDER = 0x0004
SWP_NOACTIVATE = 0x0010   # 포커스를 뺏지 않는다 (사용자가 치던 창을 방해하면 안 됨)


def get_window_rect(hwnd):
    """창의 **물리 픽셀** rect (x, y, w, h). 실패하면 None."""
    if sys.platform != "win32":
        return None
    try:
        r = ctypes.wintypes.RECT()
        if not ctypes.windll.user32.GetWindowRect(int(hwnd), ctypes.byref(r)):
            return None
        return (int(r.left), int(r.top), int(r.right - r.left), int(r.bottom - r.top))
    except Exception as e:
        print(f"[WARN] GetWindowRect 실패: {e}")
        return None


def set_window_rect(hwnd, x, y, w, h) -> bool:
    """창을 **물리 픽셀** 좌표로 직접 배치한다 (Qt의 논리 좌표 변환을 우회).

    OG-1: Qt의 `setGeometry`는 모니터 배율이 섞인 환경에서 창 크기를 그 창이 놓인
    화면의 배율로 나눠 적용하는 경우가 있다. 가상 데스크톱 전체를 덮어야 하는
    오버레이에서는 그 순간 화면 일부가 통째로 비어 자막이 안 보인다.
    """
    if sys.platform != "win32":
        return False
    try:
        fn = ctypes.windll.user32.SetWindowPos
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                       ctypes.c_uint]
        fn.restype = ctypes.c_bool
        return bool(fn(int(hwnd), None, int(x), int(y), int(w), int(h),
                       SWP_NOZORDER | SWP_NOACTIVATE))
    except Exception as e:
        print(f"[WARN] SetWindowPos(rect) 실패: {e}")
        return False

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
    # 비-Windows 폴백에서만 Qt를 끌어온다 — 이 모듈은 의존 방향의 바닥이라 Qt를 상단 import 하지 않는다.
    from PySide6.QtGui import QCursor
    p = QCursor.pos()   # (이 경로는 메인 스레드에서만 쓰인다)
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

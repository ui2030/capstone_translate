"""Cocktail — 캡처 영역/프레임 변화 감지 계층.

캡처 모드 상수, 화면 rect 캐시, 프레임 해시(F-1)와 변경 밴드(PF-2),
워커에 넘기는 옵션 스냅샷(RuntimeOptions).
좌표는 전부 **화면 절대·물리 픽셀**이다 (DP-1).
"""
import ctypes
import ctypes.wintypes
import math
import sys
from dataclasses import dataclass

import cv2
import numpy as np
from PySide6.QtCore import QRect
from PySide6.QtWidgets import QApplication

from cocktail_platform import cursor_pos_physical


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

# OCR 백엔드 콤보의 표시 문자열. 이 문자열이 곧 라우팅 키라 여기저기 하드코딩돼 있었다.
OCR_BACKEND_TESSERACT = "Tesseract"
OCR_BACKEND_PADDLE = "PaddleOCRv5"
OCR_BACKEND_UIA = "UIA (활성 창)"

# SC-1 튜닝 레버: 활성 창 모드의 캡처 영역을 "활성 창이 있는 모니터 1개"로 제한.
# 바탕화면(Progman/WorkerW)이나 가상 데스크톱 전체를 덮는 창이 활성이면 창 rect가
# 모든 모니터 합집합이 되어 전 화면을 OCR/렌더링하던 문제(오버레이가 모든 모니터에 꽉 참).
# False로 두면 구 동작(창 rect 전체). 창을 두 모니터에 걸쳐 쓰는 사용자는 False.
CLAMP_WINDOW_TO_ACTIVE_SCREEN = True


# --- 화면 변화 감지 (P-4 / F-1 / FD-1) ---------------------------------------
# F-1 튜닝 레버: 그레이 셀 중 "가장 크게 변한 셀"의 변화량이 이 값 미만이면
# 같은 프레임으로 보고 건너뛴다. 평균 기준(구 P-4)은 자막 한 줄 변화를 삼켰다.
FRAME_DIFF_THRESHOLD = 8

# FD-1 튜닝 레버: 셀 하나가 담을 화면 픽셀 목표치. 격자를 16x16으로 고정했더니
# 1920x1080에서 셀 하나가 120x67px라 자막 한 줄(28px)이 셀 평균에 희석돼
# 변화량이 3(임계 8 미만)으로 나왔다 = 실시간 자막이 통째로 스킵됐다.
# 화면이 클수록 격자를 키워 셀 픽셀 수를 일정하게 유지한다(해상도 불변).
# 내리면 더 민감(비용은 거의 안 변함), 올리면 작은 글자를 다시 놓친다.
# 실측(한 줄 교체 최소 변화량, 1920x1080 / 3840x2160):
#   셀 32px → 12px 글자 6/6 (임계 미달) | 셀 24px → 15/17 | **셀 16px → 27/27** | 셀 12px → 47/31
FRAME_HASH_CELL_PX = 16
FRAME_HASH_GRID_MIN = 16    # 아주 작은 캡처 영역에서 격자가 무의미해지지 않게
FRAME_HASH_GRID_MAX = 256   # 5760px 가상 데스크톱에서도 해시 64KB 이하

# FD-1 튜닝 레버: "한 행에서 이만큼의 셀이 같이 변해야" 진짜 변화로 본다.
# 셀이 작아지니 깜빡이는 텍스트 커서(캐럿)까지 잡혔다 — 캐럿은 폭이 2~3px라
# 어느 행에서도 셀 1개만 변한다(실측: 항상 1). 자막 한 줄 교체는 12px 글자에서도
# 한 행에 8셀이 변한다. 1로 내리면 캐럿 깜빡임마다 OCR이 돌고, 3 이상으로 올리면
# 짧은 단어 하나가 바뀌는 변화(실측 2~3셀)를 놓친다.
FRAME_DIFF_MIN_ROW_CELLS = 2

_HASH_HEADER_BYTES = 16  # int32 x 4 (캡처 영역 좌표)

# PL-1 튜닝 레버: 워커의 폴 간격. 예전에는 `time.sleep(0.25)` 고정이라 캡처가
# 116→32 ms로 빨라진 뒤에도 "화면 변화 → 자막" 지연에 매 사이클 250 ms가 그대로 붙었다.
# 실측(1920x1080 실기 하네스): poll_wait 평균 250 ms / 전체 지연의 10~25%.
#
#   최근에 변화가 있었다  → POLL_ACTIVE_S (자막·채팅은 또 바뀐다)
#   POLL_ACTIVE_WINDOW_S 동안 아무 변화 없다 → POLL_IDLE_S (= 예전 값 그대로)
#
# **POLL_IDLE_S를 내리지 말 것** — B0에서 유휴 CPU를 24%→8%로 내린 것이 이 값이다.
# 올리면 유휴 CPU는 더 내려가지만 정적 화면에서 첫 자막이 그만큼 늦게 뜬다.
# POLL_ACTIVE_S를 더 내리면 캡처(32 ms)가 폴 비용을 지배해 '활동 중' CPU만 오른다.
POLL_ACTIVE_S = 0.06
POLL_IDLE_S = 0.25
POLL_ACTIVE_WINDOW_S = 1.2


def _grid_for(shape) -> int:
    """FD-1: 캡처 크기에 맞춘 정사각 격자 변 길이. 정사각이라 해시 길이만으로 복원된다."""
    h, w = int(shape[0]), int(shape[1])
    g = round(max(w, h) / FRAME_HASH_CELL_PX)
    return int(min(FRAME_HASH_GRID_MAX, max(FRAME_HASH_GRID_MIN, g)))


def _cells(buf: bytes) -> np.ndarray:
    """해시 바이트 → (g, g) 셀 격자. g는 페이로드 길이에서 역산한다."""
    n = len(buf) - _HASH_HEADER_BYTES
    g = math.isqrt(n)
    return np.frombuffer(buf, dtype=np.uint8,
                         offset=_HASH_HEADER_BYTES).astype(np.int16).reshape(g, g)


def cheap_hash(img_np: np.ndarray, rect=None) -> bytes:
    g = _grid_for(img_np.shape)
    small = cv2.resize(img_np, (g, g), interpolation=cv2.INTER_AREA)
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


def _changed_rows(a: bytes, b: bytes, threshold):
    """(셀 절대차, 행별 '변한 셀 수') — hash_distance / dirty_row_band 공용."""
    diff = np.abs(_cells(a) - _cells(b))
    return diff, (diff >= threshold).sum(axis=1)


def hash_distance(a: bytes, b: bytes) -> int:
    if a is None or b is None or len(a) != len(b):
        return 1 << 30
    if a[:_HASH_HEADER_BYTES] != b[:_HASH_HEADER_BYTES]:
        return 1 << 30  # 캡처 영역 변경 = 무조건 재처리
    # F-1(b): 평균이 아니라 셀 단위 최대 변화량 — 국소 변화(자막 한 줄)에 민감.
    # FD-1: 단, 한 행에 셀 하나만 변한 것은 캐럿 깜빡임 쪽이라 변화로 치지 않는다.
    diff, per_row = _changed_rows(a, b, FRAME_DIFF_THRESHOLD)
    if int(per_row.max()) < FRAME_DIFF_MIN_ROW_CELLS:
        return 0
    return int(diff.max())


def dirty_row_band(prev: bytes, cur: bytes, threshold=FRAME_DIFF_THRESHOLD):
    """PF-2: 두 프레임 해시를 비교해 **변한 셀의 세로 구간** (row0, row1)을 돌려준다.

    반환값은 16등분 격자의 행 인덱스 [row0, row1) — 호출부가 실제 픽셀로 환산한다.
    `None`이면 "밴드를 못 정한다 = 전체 프레임을 처리하라"는 뜻이다:
      - 이전 해시가 없다(첫 프레임)
      - 캡처 영역이 바뀌었다(헤더 불일치) → 좌표계가 통째로 달라 밴드 비교가 무의미
      - 변한 셀이 하나도 없다(호출부가 이미 스킵했어야 하는 경우)
    """
    if prev is None or cur is None or len(prev) != len(cur):
        return None
    if prev[:_HASH_HEADER_BYTES] != cur[:_HASH_HEADER_BYTES]:
        return None
    _diff, per_row = _changed_rows(prev, cur, threshold)
    rows = np.where(per_row >= FRAME_DIFF_MIN_ROW_CELLS)[0]
    if rows.size == 0:
        return None
    # FD-1: 내부 격자는 화면 크기에 따라 달라지지만, 반환값은 예전대로 **16등분 기준**이다
    # (호출부가 row * 높이 // 16 로 환산한다). 밴드 여유(pad)가 어차피 이보다 굵다.
    g = per_row.size
    row0 = int(rows[0]) * 16 // g
    row1 = min(16, ((int(rows[-1]) + 1) * 16 + g - 1) // g)
    return row0, max(row0 + 1, row1)


# --- 캡처 경로 선택 (CP-1) ---------------------------------------------------
# CP-1 튜닝 레버: `ImageGrab.grab(all_screens=True)`는 bbox 크기와 무관하게 **가상
# 데스크톱 전체**(여기선 5760x2160)를 찍고 잘라낸다. 실측 1920x1080 캡처가 131 ms,
# `all_screens=False`면 33 ms. 그래서 캡처 영역이 주 모니터 안에 완전히 들어갈 때만
# False로 내려간다. 주 모니터는 정의상 가상 좌표계의 원점(0,0)이라 bbox가 그대로 맞는다.
# False로 두면 항상 예전 동작(가상 데스크톱 전체 캡처) — 보조 모니터가 검게 나오는
# W-4/W-5 류 회귀가 의심되면 여기부터 끈다.
CAPTURE_PRIMARY_FAST_PATH = True


def needs_all_screens(x1, y1, x2, y2) -> bool:
    """CP-1: 이 bbox를 찍는 데 가상 데스크톱 전체 캡처가 필요한가.

    주 모니터 밖으로 1px이라도 나가면 True — 아니면 그 부분이 **검은 이미지**가 된다.
    크기는 `GetSystemMetrics(SM_CXSCREEN/SM_CYSCREEN)`로 읽는다. PIL의 win32 캡처가
    보는 것과 같은 값이라, DPI unaware 프로세스에서 Qt 지오메트리를 쓰는 것보다 안전하다.
    """
    if not CAPTURE_PRIMARY_FAST_PATH or sys.platform != "win32":
        return True
    try:
        user32 = ctypes.windll.user32
        pw, ph = int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
    except Exception as e:
        print(f"[WARN] CP-1 주 모니터 크기 조회 실패: {e}")
        return True
    if pw <= 0 or ph <= 0:
        return True
    return not (0 <= x1 < x2 <= pw and 0 <= y1 < y2 <= ph)


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


@dataclass(frozen=True)
class RuntimeOptions:
    """Per-iteration snapshot of UI combo state used by the worker thread.

    Immutable so the worker reads a consistent set of values per loop tick.
    """
    src_text: str   # combo display: "auto" | "English" | "Korean" | ...
    tgt_text: str   # combo display: "Korean" | "English" | ...
    ocr_text: str   # combo display: "Tesseract" | "PaddleOCRv5" | "UIA (활성 창)"

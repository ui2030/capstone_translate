"""Cocktail — UI 계층.

설정 창(트레이/단축키/콤보), 투명 오버레이(자막 렌더링), 영역 편집기.
엔진을 소유하지 않고 시그널로만 대화한다 (BG-4).
"""
import os

from PySide6.QtCore import (
    QEvent, QPoint, QRect, QRectF, QSettings, Qt, QTimer, Signal, Slot,
)
from PySide6.QtGui import (
    QAction, QColor, QFont, QIcon, QKeySequence, QPainter, QPen, QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication, QComboBox, QHBoxLayout, QLabel, QMainWindow, QMenu,
    QPushButton, QSizePolicy, QSystemTrayIcon, QVBoxLayout, QWidget,
)

from cocktail_capture import (
    CaptureRegionController, MODE_BOX, MODE_HOVER, MODE_WINDOW, RuntimeOptions,
    _MODE_LABELS, _MODE_LABELS_REVERSE,
)
from cocktail_engine import BackgroundController
from cocktail_ocr import (
    OCR_BACKEND_PADDLE, OCR_BACKEND_TESSERACT, OCR_BACKEND_UIA, UIA_PROVIDER,
    _APP_DIR,
)
from cocktail_platform import (
    get_window_rect, is_autorun_enabled, set_autorun, set_mouse_through,
    set_window_capture_affinity, set_window_rect, set_window_topmost,
)
from cocktail_translate import PERSIST_CACHE_DEFAULT_ON, PERSIST_CACHE_TTL_DAYS


# --- PV-3 튜닝 레버: 전역 단축키(키 후킹)는 opt-in --------------------------
# `keyboard` 패키지(MIT, optional)는 저수준 키보드 후킹으로 **모든 키 입력을 가로챈다** —
# 키로거와 같은 API다. "전부 로컬"을 내세우는 앱에서 기본 활성은 모순이고, 백신 오탐의
# 단골이라 합격기준 6("백신 경고 없이 실행되는가")과도 직결된다.
# 끈 상태에서도 기능은 하나도 줄지 않는다: 트레이 메뉴(열기 / 번역 시작·정지 / 캡처 모드 /
# 영역 편집 / 마우스 통과 / 자동 실행 / 기록 / 종료) + 창 단축키(Ctrl+Shift+H, Ctrl+Shift+P)로
# 전부 닿는다. 잃는 것은 "창이 안 보일 때 키만으로 토글"뿐이다.
# ↑True 로 되돌리면 예전 동작. 사용자는 트레이 메뉴에서 언제든 켤 수 있다.
GLOBAL_HOTKEYS_DEFAULT = False

# 후킹 라이브러리는 **켤 때만** import 한다. import 자체가 후킹을 걸지는 않지만,
# 안 쓰는 후킹 라이브러리를 프로세스에 올려 둘 이유도 없다(시작도 그만큼 가벼워진다).
_keyboard = None


def _load_keyboard():
    """전역 단축키를 켤 때만 호출. 실패하면 None (앱은 정상 동작)."""
    global _keyboard
    if _keyboard is None:
        try:
            import keyboard as _kb  # type: ignore
            _keyboard = _kb
        except Exception as e:
            print(f"[INFO] 글로벌 단축키 라이브러리 keyboard 미설치 — "
                  f"트레이·창 단축키만 사용. ({e})")
            _keyboard = False
    return _keyboard or None


# --- FT-1 (A4): 오버레이 폰트 스택 ------------------------------------------
# "Arial" 하드코딩은 한글 글리프가 없어 Qt 폴백에 맡겨져 자간/굵기가 들쭉날쭉했다.
# 한글 우선 스택 → 첫 항목부터 설치된 것을 쓴다(Qt setFamilies가 폴백 체인을 처리).
OVERLAY_FONT_FAMILIES = ("Malgun Gothic", "Noto Sans KR", "Segoe UI", "Arial")

# --- TM-1 (A3): 오버레이 topmost 주기 재적용 --------------------------------
# 다른 항상-위 창(런처/미디어 플레이어 컨트롤 등)이 나중에 뜨면 Z-order에서 우리를 덮는다.
# 0으로 두면 주기 재적용을 끈다(수동 raise_만).
OVERLAY_TOPMOST_INTERVAL_MS = 2000
# B-12: 자막 박스 배경 불투명도. 원문을 가리는 게 목적이다.
# 실측(2026-08-05): 235(92%)는 **어두운 배경 위 밝은 글자**를 못 가린다 — 남은 8%로
# 픽셀 폰트 원문이 번역문 뒤에 비쳐 "반쯤 번역된 것처럼" 보였다. 게임 UI가 대부분 이 조합.
# ↓면 뒤가 비쳐 배경 맥락이 보이고, ↑면 확실히 가린다.
OVERLAY_BOX_ALPHA = 250

# --- UI-1: 글자가 잘리지 않게 텍스트 위젯이 확보할 여유 폭 -------------------
# 위젯 폭 < 글자 폭이면 문구가 "번역 시…"로 잘려 보인다. 고DPI(폰트 DPI 168)에서
# 폰트만 커지고 폭은 그대로였던 것이 원인이었다. ↓면 잘림 위험, ↑면 창이 넓어진다.
TEXT_PAD_PX = 20


def make_app_icon() -> QIcon:
    """IC-1: 트레이/창 아이콘을 코드로 그린다.

    외부 이미지를 받아오지 않는다(라이선스 오염 금지 — ICONS.md §6).
    구성은 ICONS.md §2 그대로: 따뜻한 색 둥근 사각형 + 원문/번역 글자 쌍(A → 가).
    16px에서도 읽히도록 요소는 글자 2개뿐이고, 크기별로 따로 그려 스케일 뭉개짐을 피한다.
    """
    icon = QIcon()
    for size in (16, 32, 64, 256):
        # 크게 그려서 줄인다. 16px에 직접 그리면 글자에 서브픽셀 색 번짐(빨강·파랑 테두리)이
        # 남아 아이콘이 지저분해진다 — 실측으로 확인하고 supersampling으로 바꿨다.
        ss = 8 if size <= 64 else 2
        s = size * ss
        pm = QPixmap(s, s)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setRenderHint(QPainter.TextAntialiasing, True)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(224, 108, 43))          # 따뜻한 오렌지 (모국어 = 따뜻함)
        inset = s * 0.02
        p.drawRoundedRect(QRectF(inset, inset, s - 2 * inset, s - 2 * inset),
                          s * 0.22, s * 0.22)
        p.setPen(QColor(255, 255, 255))
        for family, text, x, y in (
            ("Segoe UI", "A", -0.02, -0.04),      # 원문
            ("Malgun Gothic", "가", 0.40, 0.42),   # 번역
        ):
            f = QFont(family)
            f.setBold(True)
            f.setPixelSize(int(s * 0.56))
            f.setStyleStrategy(QFont.PreferAntialias | QFont.NoSubpixelAntialias)
            p.setFont(f)
            p.drawText(QRectF(x * s, y * s, s * 0.56, s * 0.56),
                       Qt.AlignCenter | Qt.TextDontClip, text)
        p.end()
        icon.addPixmap(pm.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation))
    return icon


class _FitButton(QPushButton):
    """UI-1: 문구가 바뀌어도 글자가 잘리지 않는 버튼.

    버튼 문구는 실행 중에 바뀐다("번역 시작" → "모델 준비 중…" → "⚠ …").
    호출부마다 폭을 다시 재는 대신 문구가 바뀌는 한 곳(setText)에서 최소 폭을 갱신한다.
    """

    def setText(self, text):
        super().setText(text)
        self._refit()

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.FontChange:
            self._refit()   # DPI/글꼴이 바뀌면 글자 폭도 바뀐다

    def _refit(self):
        self.setMinimumWidth(
            self.fontMetrics().horizontalAdvance(self.text() or "") + TEXT_PAD_PX)


class SettingsWindow(QMainWindow):
    # BG-4: 엔진 시그널은 BackgroundController가 정의/방출. 이 창은 UI 슬롯만 가짐.

    # T-1: 글로벌 단축키 → 메인 스레드 진입용 시그널 (keyboard 콜백은 별도 스레드)
    global_toggle_signal = Signal()
    global_show_signal = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        # Phase 1C: OverlayWindow 참조 (main에서 setter로 주입). 없을 수도 있음.
        self.overlay = None
        # UI-2: 현재 적용된 글꼴 픽셀 크기. setGeometry가 moveEvent를 먼저 쏘므로 여기서 먼저 잡는다.
        self._font_px = 0

        # Tray-first settings persistence. The control window is a settings panel,
        # not the always-visible translation surface.
        self._settings = QSettings("Cocktail", "OverlayTranslator")
        self._first_run_settings = not self._settings.value(
            "ui/first_run_complete", False, type=bool
        )

        # T-1: 트레이 종료 vs hide 분기
        self._is_quitting = False

        # PV-1/PV-3: 프라이버시 스위치 두 개. 둘 다 저장된 설정(PS-1)에서 복원하고
        # 기본은 "남기지 않음 / 후킹하지 않음"이다. 실제 적용은 트레이·컨트롤러 준비 뒤.
        self._hotkeys_on = False
        self._persist_cache_on = self._settings.value(
            "ui/persist_cache", PERSIST_CACHE_DEFAULT_ON, type=bool)

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
            "QMainWindow, #central { background: #202124; color: white; } "
            "QComboBox, QPushButton { min-height: 24px; }"
        )

        # E-7 → DP-1: 이제 Qt 논리 스케일링 자체가 OFF(QT_ENABLE_HIGHDPI_SCALING=0)라
        # 논리 좌표 == 물리 픽셀. 캡처/오버레이 좌표에는 절대 배율을 곱하지 않는다.
        # DC-1: 쓰이지 않던 `self._scale` 필드는 삭제.
        # UI-2: 항등 함수였던 `self.px`도 삭제 — 좌표를 곱해 봐야 글자는 안 커진다.
        #       설정창의 DPI 보정은 `_apply_screen_font_scale()`(글꼴만)이 담당한다.

        # BG-3: capture mode/region state is owned outside the settings window.
        # BG-6: region depends on the same is_self_hwnd callback used by the controller.
        self.region = CaptureRegionController(is_self_hwnd=self._is_self_hwnd)

        # BG-7: RegionEditor는 lazy 생성. 첫 "영역 편집..." 클릭 시 인스턴스 생성.
        self.region_editor = None

        self.initUI()
        self._apply_screen_font_scale()
        # PS-1: 저장된 사용자 선택(언어/OCR/모드/영역) 복원 — initUI 직후, controller 생성 전.
        self._load_persisted()

        # BG-4: 엔진은 BackgroundController가 소유. UI는 시그널 연결만.
        # TS-1: 콤보 스냅샷은 지금(메인 스레드) 한 번 떠서 넘기고, 이후 변경은 push한다.
        self.controller = BackgroundController(
            region=self.region,
            options=self._snapshot_options(),
            is_self_hwnd=self._is_self_hwnd,
        )
        self.controller.state_ready.connect(self._on_state_ready)
        self.controller.failure_alert.connect(self._on_failure)
        self.controller.model_ready.connect(self._on_model_ready)
        self.controller.model_failed.connect(self._on_model_failed)
        self.controller.recovered.connect(self._on_recovered)
        self.controller.progress_msg.connect(self._on_progress)
        self.controller.status_msg.connect(self._set_status)
        self.global_toggle_signal.connect(self._on_global_toggle)
        self.global_show_signal.connect(self._on_global_show)

        # TS-1 + PS-1: 콤보가 바뀌면 (a) 워커에 스냅샷 push, (b) 디스크에 저장.
        for combo in (self.src_combo, self.tgt_combo, self.ocr_combo):
            combo.currentTextChanged.connect(self._on_options_changed)

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
        # PV-3: 전역 단축키는 저장된 설정에 따라. 기본 OFF라 첫 실행에서는 후킹이 없다.
        if self._settings.value("ui/global_hotkeys", GLOBAL_HOTKEYS_DEFAULT, type=bool):
            self._register_global_hotkeys()
        else:
            self._sync_hotkey_ui()
        # PV-1: 영구 캐시 on/off 적용. 꺼져 있으면 지난 기록도 여기서 사라진다.
        self.controller.set_persist_cache(self._persist_cache_on)

        # P-9: 모델 lazy load — UI는 이미 떠 있고, 모델은 백그라운드로 준비
        self.run_button.setEnabled(False)
        self.run_button.setText("모델 준비 중…")

        # BG-4: 워커 + 모델 preload + 캐시 타이머 + 영구 캐시 lazy load는 controller가 소유.
        self.controller.start()

        self.old_pos = QPoint()

    # --- BG-4: controller 의존 helpers --------------------------------------
    def _snapshot_options(self) -> "RuntimeOptions":
        """TS-1: **메인 스레드 전용.** 위젯을 읽는 유일한 지점이다."""
        # 문자열 어노테이션: RuntimeOptions는 이 파일 후반부에 정의되므로 forward-ref.
        return RuntimeOptions(
            src_text=self.src_combo.currentText(),
            tgt_text=self.tgt_combo.currentText(),
            ocr_text=self.ocr_combo.currentText(),
        )

    @Slot()
    def _on_options_changed(self, *_args):
        """TS-1: 콤보 변경 → 워커에 스냅샷 push. PS-1: 같은 자리에서 디스크 저장."""
        self.controller.set_options(self._snapshot_options())
        self.controller.reset_frame_hash()   # 언어/백엔드가 바뀌면 직전 프레임 결과는 무효
        self._save_persisted()

    # --- PS-1: 사용자 선택 영구 저장 ------------------------------------------
    # 트레이에 상주하며 매일 쓰는 앱인데 매 실행마다 auto/Korean/Tesseract/활성창으로
    # 초기화되던 문제. 저장 대상은 "사용자가 고른 것"만 — 엔진 상태는 저장하지 않는다.
    _PERSIST_COMBOS = (
        ("ui/src", "src_combo"),
        ("ui/tgt", "tgt_combo"),
        ("ui/ocr", "ocr_combo"),
        ("ui/mode", "mode_combo"),
    )

    def _load_persisted(self):
        for key, attr in self._PERSIST_COMBOS:
            combo = getattr(self, attr, None)
            value = self._settings.value(key, "", type=str)
            # findText: 저장된 값이 현재 콤보에 없으면(예: uiautomation 제거 후 "UIA") 무시한다.
            if combo is None or not value or combo.findText(value) < 0:
                continue
            combo.blockSignals(True)
            combo.setCurrentText(value)
            combo.blockSignals(False)
        # 콤보를 blockSignals로 채웠으니 capture_mode는 여기서 직접 맞춘다.
        mode = _MODE_LABELS.get(self.mode_combo.currentText())
        if mode:
            self.region.capture_mode = mode
        raw = self._settings.value("ui/red_rect", "", type=str)
        try:
            x, y, w, h = (int(v) for v in raw.split(","))
            if w >= 8 and h >= 8:
                self.region.red_rect = QRect(x, y, w, h)
        except Exception:
            pass   # 값이 없거나 깨졌으면 기본 rect 유지

    def _save_persisted(self):
        try:
            for key, attr in self._PERSIST_COMBOS:
                combo = getattr(self, attr, None)
                if combo is not None:
                    self._settings.setValue(key, combo.currentText())
            r = self.region.red_rect
            self._settings.setValue("ui/red_rect", f"{r.x()},{r.y()},{r.width()},{r.height()}")
            self._settings.sync()
        except Exception as e:
            print(f"[WARN] 설정 저장 실패: {e}")

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

    # DC-1: `hover_size` 프로퍼티 쌍은 호출부가 하나도 없어 삭제 (hover 크기를 바꾸는 UI 자체가
    # 없다). 필요해지면 region.hover_size를 직접 쓰면 된다.

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
        if getattr(self, "hint_label", None) is not None:
            self.hint_label.hide()   # UI-3: 첫 실행 안내는 한 번 보고 나면 자리를 비운다
        if not self._settings.value("ui/first_run_complete", False, type=bool):
            self._settings.setValue("ui/first_run_complete", True)
            self._settings.sync()

    @Slot()
    def _on_model_ready(self):
        self.run_button.setEnabled(True)
        self._sync_run_button()
        self._set_status(f"{self.controller.env_status} | 모델 준비 완료")

    @Slot(str)
    def _on_model_failed(self, msg):
        """MR-1: 로드 실패해도 앱을 죽이지 않는다 — 버튼을 살려 재시도 진입점으로 쓴다."""
        self.run_button.setEnabled(True)
        self.run_button.setText("모델 다시 시도")
        self._set_status(
            f"⚠ {msg} — 인터넷 연결을 확인한 뒤 '모델 다시 시도'를 누르세요."
        )

    def _sync_run_button(self):
        """RB-1: 버튼 문구의 단일 출처. 실행 상태에서 파생시켜 경고 문구가 눌어붙지 않게 한다."""
        if not self.controller.is_model_ok():
            return
        self.run_button.setText("번역 정지" if self.controller.model_loaded else "번역 시작")

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

    @Slot()
    def _on_recovered(self):
        """RB-1: 연속 실패에서 빠져나왔을 때 경고 문구를 걷어낸다.

        예전엔 복구도 `failure_alert`로 보내서 버튼이 "⚠ 번역 정지…"로 남았다.
        """
        self._sync_run_button()

    @Slot(str)
    def _set_status(self, msg):
        self._status_full = msg
        self._elide(self.status_label, msg)

    def _elide(self, label, full):
        """UI-1: 창보다 긴 문구는 창 폭에 맞춰 줄이고 전체는 툴팁으로 남긴다.

        폭은 라벨이 아니라 **창**에서 뽑는다 — 라벨의 최종 폭은 레이아웃이 정하므로
        resizeEvent 시점엔 아직 예전 값일 수 있다. (라벨은 창 폭 - 좌우 여백 20을 차지한다.)
        """
        label.setToolTip(full)
        label.setText(label.fontMetrics().elidedText(
            full, Qt.ElideRight, max(60, self.width() - 44)))

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
        # IC-1: 빨간 사각형 placeholder → 코드로 그린 앱 아이콘 (트레이 + 창 아이콘 공용).
        icon = make_app_icon()
        self.setWindowIcon(icon)

        self.tray = QSystemTrayIcon(icon, self)
        # 툴팁의 단축키 안내는 실제 등록 상태를 따라간다(PV-3) — 안 되는 키를 광고하지 않는다.
        self.tray.setToolTip("Cocktail Translator")

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

        # PV-3: 전역 단축키 = 키 후킹. 기본 꺼짐, 여기서 켠다.
        self._hotkeys_action = QAction("전역 단축키 (Ctrl+Shift+Q/A)", self)
        self._hotkeys_action.setCheckable(True)
        self._hotkeys_action.setChecked(self._hotkeys_on)
        self._hotkeys_action.setToolTip(
            "켜면 창이 안 보여도 키로 토글할 수 있지만, 모든 키 입력을 가로채는 후킹을 겁니다"
            " (백신이 경고할 수 있음). 꺼도 이 트레이 메뉴로 전부 조작됩니다."
        )
        self._hotkeys_action.triggered.connect(self._toggle_global_hotkeys)
        menu.addAction(self._hotkeys_action)

        # PV-1: 번역 기록(영구 캐시) — 남길지 여부 + 즉시 삭제.
        self._persist_action = QAction("번역 기록을 디스크에 저장", self)
        self._persist_action.setCheckable(True)
        self._persist_action.setChecked(self._persist_cache_on)
        self._persist_action.setToolTip(
            f"켜면 번역 결과가 이 PC에 암호화돼 저장돼 다음 실행이 빨라집니다"
            f" ({int(PERSIST_CACHE_TTL_DAYS)}일 뒤 자동 삭제). "
            f"끄면 메모리에만 두고 디스크 기록을 지웁니다."
        )
        self._persist_action.triggered.connect(self._toggle_persist_cache)
        menu.addAction(self._persist_action)

        a_clear = QAction("번역 기록 삭제", self)
        a_clear.setToolTip("지금까지의 번역 기록을 메모리와 디스크에서 즉시 지웁니다.")
        a_clear.triggered.connect(self._clear_translation_history)
        menu.addAction(a_clear)

        # CC-1: opus-mt 모델이 CC-BY-4.0(출처 표시 의무)이라 앱에서 표기에 닿을 수 있어야 한다.
        a_license = QAction("라이선스 정보", self)
        a_license.triggered.connect(self._open_license_notice)
        menu.addAction(a_license)

        menu.addSeparator()
        a_quit = QAction("종료", self)
        a_quit.triggered.connect(self._really_quit)
        menu.addAction(a_quit)

        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self._on_tray_activated)
        self.tray.show()

    @Slot()
    def _open_license_notice(self):
        """CC-1: 동봉된 LICENSE-3RDPARTY.md를 연다 (CC-BY-4.0 저작자 표시 의무 이행 경로).

        배포판에서는 exe 옆(_internal 포함)에, 개발 중에는 소스 폴더에 있다.
        """
        candidates = [
            os.path.join(_APP_DIR, "LICENSE-3RDPARTY.md"),
            os.path.join(_APP_DIR, "_internal", "LICENSE-3RDPARTY.md"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "LICENSE-3RDPARTY.md"),
        ]
        for path in candidates:
            if os.path.exists(path):
                try:
                    os.startfile(path)   # Windows 전용 — 기본 편집기로 열기
                    return
                except Exception as e:
                    print(f"[WARN] 라이선스 문서 열기 실패: {e}")
                    break
        self._set_status(
            "번역 모델 Helsinki-NLP/opus-mt-tc-big-en-ko, opus-mt-zh-en "
            "— © University of Helsinki, CC BY 4.0. "
            "전체 고지는 LICENSE-3RDPARTY.md 참고."
        )

    def _on_tray_activated(self, reason):
        if reason == QSystemTrayIcon.DoubleClick:
            self._show_normal()

    @Slot()
    def _show_normal(self):
        self.show()
        self.raise_()
        self.activateWindow()
        capturable = self.controller.capturable
        set_window_capture_affinity(self.winId(), exclude=not capturable)
        if self.overlay is not None:
            set_window_capture_affinity(self.overlay.winId(), exclude=not capturable)

    def _really_quit(self):
        self._is_quitting = True
        # PS-1: 첫 실행에서 트레이 "종료"로 끄면 이 플래그가 영영 안 찍혀
        # 매 실행마다 설정 창이 다시 떴다 (closeEvent의 hide 분기에서만 찍고 있었다).
        self._mark_settings_seen()
        self._save_persisted()
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

    # --- T-1: 글로벌 단축키 (PV-3: 기본 OFF, 트레이에서 토글) -----------------
    def _register_global_hotkeys(self):
        kb = _load_keyboard()
        if kb is None:
            print("[INFO] keyboard 미설치 — 글로벌 단축키 비활성. (pip install keyboard)")
            self._hotkeys_on = False
            self._sync_hotkey_ui()
            return False
        try:
            # keyboard 콜백은 별도 스레드. 메인 스레드 진입은 시그널로.
            kb.add_hotkey('ctrl+shift+q', lambda: self.global_toggle_signal.emit())
            kb.add_hotkey('ctrl+shift+a', lambda: self.global_show_signal.emit())
            print("[INFO] 글로벌 단축키 등록: Ctrl+Shift+Q (번역 토글), Ctrl+Shift+A (창 열기)")
            self._hotkeys_on = True
        except Exception as e:
            print(f"[WARN] 글로벌 단축키 등록 실패 (관리자 권한 필요할 수 있음): {e}")
            self._hotkeys_on = False
        self._sync_hotkey_ui()
        return self._hotkeys_on

    def _unregister_global_hotkeys(self):
        """후킹을 실제로 걷어낸다. 라이브러리를 안 물었으면 걷을 것도 없다."""
        if _keyboard:
            try:
                _keyboard.unhook_all_hotkeys()
            except Exception as e:
                print(f"[WARN] 글로벌 단축키 해제 실패: {e}")
        self._hotkeys_on = False
        self._sync_hotkey_ui()

    def _sync_hotkey_ui(self):
        """PV-3: 후킹 상태의 단일 출처 — 체크박스와 트레이 툴팁을 실제 상태에 맞춘다."""
        act = getattr(self, "_hotkeys_action", None)
        if act is not None:
            act.blockSignals(True)
            act.setChecked(self._hotkeys_on)
            act.blockSignals(False)
        if getattr(self, "tray", None) is not None:
            self.tray.setToolTip(
                "Cocktail Translator (Ctrl+Shift+Q 토글, Ctrl+Shift+A 열기)"
                if self._hotkeys_on else "Cocktail Translator — 트레이 메뉴로 제어")

    @Slot(bool)
    def _toggle_global_hotkeys(self, checked: bool):
        if checked:
            self._register_global_hotkeys()
        else:
            self._unregister_global_hotkeys()
        self._settings.setValue("ui/global_hotkeys", self._hotkeys_on)
        self._settings.sync()
        if checked and not self._hotkeys_on:
            # 켜려 했는데 못 켠 경우(라이브러리 없음/권한). 체크는 이미 풀렸다 — 이유를 말한다.
            msg = "켜지 못했습니다 (keyboard 미설치 또는 권한). 트레이 메뉴로 조작하세요"
        else:
            msg = ("ON — Ctrl+Shift+Q 토글 / Ctrl+Shift+A 열기" if self._hotkeys_on
                   else "OFF — 트레이 메뉴와 창 단축키로 모두 조작할 수 있습니다")
        self._set_status(f"{self.controller.env_status} | 전역 단축키: {msg}")

    # --- PV-1: 번역 기록 ------------------------------------------------------
    @Slot(bool)
    def _toggle_persist_cache(self, checked: bool):
        self._persist_cache_on = bool(checked)
        self.controller.set_persist_cache(self._persist_cache_on)
        self._settings.setValue("ui/persist_cache", self._persist_cache_on)
        self._settings.sync()
        self._set_status(
            f"{self.controller.env_status} | 번역 기록 저장: "
            + (f"ON — 이 PC에 남습니다({int(PERSIST_CACHE_TTL_DAYS)}일 뒤 자동 삭제)"
               if self._persist_cache_on else "OFF — 디스크 기록을 지웠습니다"))

    @Slot()
    def _clear_translation_history(self):
        """트레이 '번역 기록 삭제'. 즉시 지우고 **결과를 숫자로** 보여준다."""
        mem, disk, gone = self.controller.clear_translation_history()
        msg = (f"번역 기록 삭제: 메모리 {mem}건 · 디스크 {disk}건"
               + (" · 파일 삭제 확인" if gone else " · ⚠ 파일이 남아 있습니다"))
        # 확인 피드백이 본론이다 — 긴 환경 문구를 앞에 붙이면 상태줄에서 잘려 안 보인다.
        self._set_status(msg)
        try:
            self.tray.showMessage("Cocktail Translator", msg,
                                  QSystemTrayIcon.Information, 3000)
        except Exception:
            pass

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
        self._save_persisted()   # PS-1: 공들여 잡은 영역이 재시작으로 날아가지 않게
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
        self._save_persisted()   # PS-1
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
        # UI-1: 고정 픽셀 배치(setGeometry) → 레이아웃. 창을 줄여도 위젯이 밖으로 나가지 않고,
        # 폰트가 커지면(고DPI) 위젯이 글자에 맞춰 자란다. 배치·기능은 그대로다.
        central = QWidget(self)
        central.setObjectName("central")   # 스타일시트의 어두운 배경이 여기에도 걸리게
        self.setCentralWidget(central)

        # 소스 언어
        self.src_combo = QComboBox(central)
        self.src_combo.addItems(["auto", "English", "Korean", "French", "German", "Chinese", "Japanese"])

        # 타겟 언어
        self.tgt_combo = QComboBox(central)
        self.tgt_combo.addItems(["Korean", "English", "French", "German", "Chinese", "Japanese"])

        # OCR 백엔드 (Phase 2/3: UIA 옵션 추가)
        self.ocr_combo = QComboBox(central)
        items = [OCR_BACKEND_TESSERACT, OCR_BACKEND_PADDLE]
        if UIA_PROVIDER.is_available():
            items.append(OCR_BACKEND_UIA)
        self.ocr_combo.addItems(items)
        self.ocr_combo.setToolTip(
            "Tesseract: 일반 OCR / PaddleOCRv5: 선택 시 lazy load / "
            "UIA: 활성 창 텍스트 직접 (Windows, OCR 불필요)"
        )

        # T-2: 캡처 모드 (활성 창 / 마우스 hover / 영역 박스). 첫 항목이 기본.
        self.mode_combo = QComboBox(central)
        for label in _MODE_LABELS.keys():
            self.mode_combo.addItem(label)
        self.mode_combo.setToolTip("활성 창: 현재 활성 창 자동 / 마우스 hover: 커서 주변 / 영역 박스: 빨간 박스 안만")
        self.mode_combo.currentTextChanged.connect(self._on_mode_combo_changed)
        # 박사 자체 리뷰: 콤보 표시와 실제 capture_mode 동기화 (기본 = 활성 창)
        default_label = _MODE_LABELS_REVERSE.get(self.capture_mode)
        if default_label:
            self.mode_combo.blockSignals(True)
            self.mode_combo.setCurrentText(default_label)
            self.mode_combo.blockSignals(False)

        # 번역 시작/정지
        self.run_button = _FitButton(central)
        self.run_button.setText("번역 시작")
        self.run_button.clicked.connect(self.toggle_running)

        # 캡처 토글 (E-9)
        self.capture_btn = _FitButton(central)
        self.capture_btn.setText("스샷차단")
        self.capture_btn.setToolTip("스크린샷에 번역창을 포함할지 토글 (Ctrl+Shift+H)")
        self.capture_btn.clicked.connect(self.toggle_capturable)

        # BG-7: 영역 편집 진입점 (MODE_BOX 박스 마우스 편집)
        self.edit_region_btn = _FitButton(central)
        self.edit_region_btn.setText("영역 편집...")
        self.edit_region_btn.setToolTip(
            "영역 박스를 마우스로 직접 편집 (MODE_BOX 자동 전환)."
        )
        self.edit_region_btn.clicked.connect(self._open_region_editor)

        # 종료
        self.close_button = _FitButton(central)
        self.close_button.setText("확인")
        self.close_button.clicked.connect(self.close)

        # 상태/진단 표시 (U-1)
        self._status_full = "환경 확인 중"
        self.status_label = QLabel(self._status_full, central)
        # 창보다 긴 상태 문구가 창 최소 폭을 밀어 올리면 안 된다 → 폭은 레이아웃에 맡기고
        # 문구는 `_elide()`가 줄인다.
        self.status_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.status_label.setStyleSheet(
            "QLabel { color: white; background: rgba(20, 20, 20, 210); "
            "padding: 3px 6px; border-radius: 4px; }"
        )

        # UI-3: 첫 실행 안내 — "무엇을 눌러야 자막이 나오는가". 첫 실행에만 보인다.
        self.hint_label = QLabel(central)
        self.hint_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.hint_label.setStyleSheet(
            "QLabel { color: #ffd9a0; background: rgba(60, 40, 20, 220); "
            "padding: 3px 6px; border-radius: 4px; }"
        )
        # 기본 창 폭(880px)에서 잘리지 않고 다 읽히는 길이로 맞춰 두었다 — 자세한 것은 툴팁.
        self._hint_full = (
            "처음이신가요? 번역할 창을 띄운 뒤 [번역 시작]을 누르세요. "
            "전역 단축키·기록 저장은 기본 꺼짐(트레이에서 켜기)."
        )
        self._elide(self.hint_label, self._hint_full)
        self.hint_label.setToolTip(
            "① 번역할 창을 화면에 띄운다  ② 캡처 모드를 고른다(기본: 활성 창)\n"
            "③ [번역 시작]을 누른다 → 그 창의 글자 위에 번역 자막이 겹쳐 뜬다\n"
            "이 창을 닫아도 트레이에서 계속 동작한다 (트레이 아이콘 더블클릭으로 다시 열기).\n"
            "\n"
            "프라이버시 기본값 — 둘 다 트레이 메뉴에서 켤 수 있습니다:\n"
            "· 전역 단축키(Ctrl+Shift+Q/A) OFF — 켜면 모든 키 입력을 가로채는 후킹을 겁니다\n"
            "· 번역 기록을 디스크에 저장 OFF — 켜면 번역 결과가 이 PC에 남습니다"
        )
        self.hint_label.setVisible(self._first_run_settings)

        row = QHBoxLayout()
        row.setSpacing(6)
        for w in (self.src_combo, self.tgt_combo, self.ocr_combo, self.mode_combo,
                  self.run_button, self.capture_btn, self.edit_region_btn,
                  self.close_button):
            row.addWidget(w)
        row.addStretch(1)   # 창이 넓어지면 남는 폭은 오른쪽에 — 기존 좌측 정렬 배치 유지

        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(6)
        root.addLayout(row)
        root.addWidget(self.status_label)
        root.addWidget(self.hint_label)
        root.addStretch(1)

        # BG-5: 캡처 영역은 CaptureRegionController가 절대 좌표로 소유.
        # BG-1: settings panel size must not act as the region editor — RegionEditor가 담당.
        # DC-1: 만들자마자 숨기던 `QSizeGrip`은 삭제. 남겨 뒀던 이유가 "smoke 체크가 그 문자열을
        # 찾기 때문"이었는데, 그 테스트 자체를 실동작 테스트로 교체했으므로 붙잡을 이유가 없다.

    def _apply_screen_font_scale(self):
        """UI-2 (DP-1 보정): 설정창 글꼴을 **포인트 대신 픽셀 크기**로, 그 화면 DPI에 맞춰 잡는다.

        DP-1으로 Qt 스케일링을 껐어도 **글자만은** 창이 있는 모니터 DPI로 그려진다(실측:
        175% 모니터로 옮기면 글자가 1.75배로 커진다). 반면 레이아웃이 쓰는 `fontMetrics()`는
        주 모니터 DPI(96) 그대로다 → 상자는 그대로인데 글자만 커져 잘린다. 이게 이 창에서
        DP-1이 실제로 만들어 내던 증상이다.
        포인트 크기는 화면마다 다른 픽셀로 환산되지만 **픽셀 크기는 어디서나 그 픽셀**이라
        그리기와 측정이 다시 일치한다. 좌표에는 손대지 않으므로 캡처·오버레이는 무영향.
        `QT_FONT_DPI`를 강제한 환경에서도 같은 픽셀 값이 나와 이중 적용이 없다.
        """
        scr = self.screen()
        if scr is None or self.centralWidget() is None:
            return   # setGeometry가 initUI 전에 moveEvent를 쏜다 — 그때는 잴 자식이 없다
        base = QApplication.font()
        pt = base.pointSizeF() if base.pointSizeF() > 0 else 9.0
        px = max(8, min(64, round(pt * (scr.logicalDotsPerInch() or 96.0) / 72.0)))
        if px == self._font_px:
            return
        self._font_px = px
        f = QFont(base)
        f.setPixelSize(px)
        # 부모에 setFont 해도 자식의 `font()`는 안 따라온다(스타일시트가 걸린 창에서 실측) →
        # 레이아웃이 옛 크기로 계산한다. 자식에 직접 준다.
        for w in self.findChildren(QWidget):
            w.setFont(f)
        layout = self.centralWidget().layout() if self.centralWidget() else None
        if layout is not None:
            layout.invalidate()   # 없으면 **옛 글꼴로 캐시한** 크기를 그대로 돌려준다
            layout.activate()
        if self.isVisible():
            self.adjustSize()     # 글자가 커진 만큼 창도 키운다 (뜨기 전이면 첫 크기를 존중)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not hasattr(self, "status_label"):
            return
        self._elide(self.status_label, getattr(self, "_status_full", ""))
        self._elide(self.hint_label, self._hint_full)
        self.update()
        # Phase 1C 자체 리뷰 수정: OverlayWindow 갱신
        if getattr(self, "overlay", None) is not None:
            self.overlay.update()

    def moveEvent(self, event):
        # Phase 1C 자체 리뷰 수정: 윈도우 이동 시 OverlayWindow 갱신
        super().moveEvent(event)
        self._apply_screen_font_scale()   # UI-2: 다른 배율의 모니터로 끌고 가면 글꼴도 따라간다
        if getattr(self, "overlay", None) is not None:
            self.overlay.update()

    def toggle_running(self):
        self.hint_label.hide()   # UI-3: 안내대로 눌렀으면 안내는 역할을 다했다
        if not self.controller.is_model_ok():
            # MR-1: 로드에 실패한 상태면 이 클릭이 곧 재시도다.
            if self.controller.retry_model_load():
                self.run_button.setEnabled(False)
                self.run_button.setText("모델 준비 중…")
                self._set_status("모델을 다시 불러오는 중입니다…")
            else:
                self._set_status("모델 준비 중입니다. 완료 후 번역을 시작할 수 있습니다.")
            return
        new_state = not self.controller.model_loaded
        self.controller.set_running(new_state)
        self._sync_run_button()
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
        if _keyboard:
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
        """TM-1: 다른 항상-위 창이 나중에 떠서 우리를 덮은 경우 Z-order를 되찾는다.

        OG-1: 같은 타이머에서 지오메트리도 재확인한다. 별도 타이머를 만들 이유가 없고,
        모니터를 꽂았다 빼는 순간의 잘못된 크기가 영구히 남는 것을 이 주기가 막는다.
        """
        if not self.isVisible():
            return
        self._enforce_geometry()
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
        self._want_rect = (x, y, right - x + 1, bottom - y + 1)
        self.setGeometry(*self._want_rect)
        self._enforce_geometry()

    def _enforce_geometry(self):
        """OG-1: Qt가 만든 **실제** 창 rect가 요청과 다르면 win32 물리 좌표로 바로잡는다.

        실측(2026-08-03, 1920x1080@100% + 3840x2160@175% 듀얼): 모니터 구성이 바뀌는
        순간 Qt가 창 **크기만** 그 창이 놓인 화면의 배율로 나눠 적용해, 오버레이가
        5760x2160 → 3291x1234(정확히 1/1.75)로 쪼그라들었다. 위치는 그대로라 주 모니터
        (x=0 이후)를 통째로 못 덮고, 그 화면에서는 자막이 **한 줄도 안 그려진다**.
        더 나쁜 건 이 상태가 영구히 latch된다는 것 — `_fit_to_screen`은 화면 변경
        시그널에만 반응하므로, 한 번 잘못 잡히면 스스로 회복하지 못한다.

        그래서 (a) 설정 직후 즉시 검증하고 (b) topmost 타이머가 주기적으로 재확인한다.
        `_want_rect`가 곧 진실이고 Qt 지오메트리는 참고값이다.
        """
        want = getattr(self, "_want_rect", None)
        if not want:
            return
        got = get_window_rect(self.winId())
        if got == want:
            return
        if set_window_rect(self.winId(), *want) and got is not None:
            print(f"[INFO] OG-1 오버레이 지오메트리 교정: {got} → {want}")

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

        # B-12: 배경을 덮어 원문 차단
        painter.setBrush(QColor(20, 20, 20, OVERLAY_BOX_ALPHA))
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

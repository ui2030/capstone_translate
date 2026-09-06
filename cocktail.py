"""Cocktail Overlay Translator — 엔트리포인트.

화면의 글자를 읽어 그 자리에 번역을 덮어 그리는 상주형 번역기.
인식·번역 전 과정이 로컬에서 돈다.

구조 (역할별 모듈):
    cocktail_platform.py  Windows API (창 속성/DPAPI/자동 실행)
    cocktail_capture.py   캡처 영역·프레임 변화 감지
    cocktail_ocr.py       Tesseract / PaddleOCR / UI Automation
    cocktail_translate.py 번역 모델·캐시·표시 게이트
    cocktail_engine.py    워커 루프 (BackgroundController)
    cocktail_ui.py        설정 창 / 오버레이 / 영역 편집기

라이선스 (LICENSE-3RDPARTY.md 참조):
- UI: PySide6 (LGPL — closed-source 상용 OK, 동적 링크 유지)
- 번역 1차: Helsinki-NLP/opus-mt-tc-big-en-ko (**CC-BY-4.0 — 출처 표시 의무**)
- 번역 2차: facebook/m2m100_418M (MIT)
- OCR: Tesseract (Apache 2.0). 한국어팩 별도 설치 필요 (errors.md E-6).

코드 수정 전 errors.md를 한 번 읽을 것 (재발 방지 체크리스트).
테스트: `python test_cocktail.py` (빠름) / `--full` (모델 로드까지).
"""
import os

# DP-1: Qt 논리 스케일링 OFF → 앱 전 좌표를 물리 픽셀 하나로 통일한다.
# GetWindowRect / ImageGrab / OCR bbox는 전부 물리 픽셀인데, Qt 지오메트리와
# QCursor.pos()만 논리 좌표면 175% 보조 모니터가 섞이는 순간 가상 데스크톱에
# 어느 모니터에도 속하지 않는 좌표 구간이 생기고, 가상 데스크톱 전체를 덮는
# OverlayWindow가 보조 모니터 배율로 스케일돼 자막이 화면 밖으로 밀린다.
# QGuiApplication 인스턴스 생성 전 = **다른 모듈을 import 하기 전**에 설정해야 한다.
os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "0")

import sys   # noqa: E402 — 위 환경변수 설정이 반드시 먼저다

# EN-1: 콘솔 인코딩 안전화. 본체는 cocktail_platform(의존 방향의 바닥)에 있고,
# 여기서는 **다른 모듈보다 먼저** 불러 첫 print 전에 걸리게만 한다.
from cocktail_platform import _make_console_utf8_safe   # noqa: E402

_make_console_utf8_safe()

from PySide6.QtWidgets import QApplication   # noqa: E402

from cocktail_ocr import (   # noqa: E402
    AVAILABLE_TESS_LANGS, _TESSERACT_CMD, _module_available, environment_summary,
)
from cocktail_translate import PersistentCache   # noqa: E402
from cocktail_ui import OverlayWindow, SettingsWindow   # noqa: E402


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
    # 자가진단은 무거워도 된다 — 여기선 torch를 실제로 import해서 확인한다 (LZ-1).
    status, problems = environment_summary(probe_torch=True)
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

    # LZ-1: torch/transformers는 이제 앱 시작 경로에서 import되지 않는다(창을 먼저 띄우려고).
    # 그러면 "spec은 있는데 DLL이 깨져 실제 import는 실패"하는 배포 사고를 자가진단이 놓친다.
    # 여기서만 진짜로 import해서 확인한다 — 자가진단은 느려도 되는 명령이다.
    try:
        import transformers   # noqa: F401
    except Exception as e:
        print(f"[SELFTEST] transformers import failed: {type(e).__name__}: {e}")
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


def _make_icon_file(path=os.path.join("assets", "icon.ico")) -> int:
    """IC-2 (빌드용): `make_app_icon()`이 코드로 그린 아이콘을 멀티사이즈 .ico로 굽는다.

    아이콘의 유일한 원본은 코드다 — 외부에서 이미지를 받지 않는다(ICONS.md §6, 라이선스 오염 금지).
    `build.spec`(exe 아이콘)과 `installer.iss`(SetupIconFile)가 이 파일을 참조한다.
    """
    import io

    from PIL import Image
    from PySide6.QtCore import QBuffer
    from PySide6.QtWidgets import QApplication

    _ = QApplication.instance() or QApplication([])   # QPixmap은 QGuiApplication이 필요
    from cocktail_ui import make_app_icon
    icon = make_app_icon()
    frames = []
    for size in (256, 128, 64, 48, 32, 16):
        buf = QBuffer()
        buf.open(QBuffer.ReadWrite)
        icon.pixmap(size, size).save(buf, "PNG")
        frames.append(Image.open(io.BytesIO(bytes(buf.data()))).convert("RGBA"))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    frames[0].save(path, format="ICO",
                   sizes=[(f.width, f.height) for f in frames],
                   append_images=frames[1:])
    print(f"[ICON] {path} ({os.path.getsize(path)} bytes)")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv if argv is None else argv)
    if "--self-test" in argv:
        return _run_self_test()
    if "--make-icon" in argv:
        return _make_icon_file()

    return CocktailAppController(argv).start()


if __name__ == "__main__":
    sys.exit(main())

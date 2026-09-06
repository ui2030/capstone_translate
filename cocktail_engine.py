"""Cocktail — 엔진 계층 (BG-4).

워커 스레드, OCR/UIA 라우팅, 모델 생명주기, 일시정지 가드, 영구 캐시 타이머,
엔진→UI 시그널. **위젯을 절대 참조하지 않는다** (BG-6/TS-1).
"""
import ctypes
import math
import os
import sys
import threading
import time

import cv2
import numpy as np
import pytesseract
from PIL import ImageGrab
from PySide6.QtCore import QObject, QTimer, Signal, Slot

from cocktail_capture import (
    FRAME_DIFF_THRESHOLD, MODE_WINDOW, POLL_ACTIVE_S, POLL_ACTIVE_WINDOW_S,
    POLL_IDLE_S, RuntimeOptions,
    _state_signature, cheap_hash, dirty_row_band, hash_distance, needs_all_screens,
)
from cocktail_ocr import (
    AVAILABLE_TESS_LANGS, LANG_TO_TESS, OCR_BACKEND_PADDLE, OCR_BACKEND_UIA,
    OCR_DIRTY_BAND_ENABLED, OCR_DIRTY_FULL_RATIO, OCR_KO_FAIL_COOLDOWN,
    OCR_KO_JUNK_CONF, OCR_KO_JUNK_RATIO, OCR_KO_MEAN_CONF, OCR_KO_STICKY_FRAMES,
    OCR_LANGS_AUTO_DEFAULT, OCR_LANGS_TGT_REDUNDANT,
    OCR_EDGE_TRIM_ENABLED, OCR_EDGE_TRIM_GAP_RATIO, OCR_EDGE_TRIM_HEIGHT_HI,
    OCR_EDGE_TRIM_HEIGHT_LO, OCR_EDGE_TRIM_MAX, OCR_EDGE_TRIM_SHORT_CONF,
    OCR_LINE_MIN_CONF, OCR_LINE_MIN_WIDTH_RATIO, OCR_LINE_SPLIT_GAP_RATIO,
    OCR_MAX_SIDE_PX,
    OCR_MIN_DOWNSCALE, OCR_UPSCALE_CHAR_BUDGET, OCR_UPSCALE_FACTOR,
    OCR_UPSCALE_CONF_MARGIN, OCR_UPSCALE_MIN_COVERAGE, OCR_UPSCALE_MIN_FACTOR,
    OCR_UPSCALE_TRIGGER_INK_PX, OCR_UPSCALE_TRIGGER_PX,
    OCR_WORD_HEIGHT_OUTLIER_RATIO, OCR_WORD_MIN_CONF,
    OCR_ZH_EMPTY_COOLDOWN, OCR_ZH_MIN_CJK_CHARS, OCR_ZH_RESCAN_ENABLED,
    OCR_ZH_STICKY_FRAMES, OCR_ZH_TESS_LANG, OCR_ZH_UNREAD_INK_RATIO,
    PADDLE_OCR, PARA_MERGE_ENABLED, PARA_MERGE_FONT_HI, PARA_MERGE_FONT_LO,
    PARA_MERGE_GAP_RATIO, PARA_MERGE_LEFT_TOL_RATIO, PARA_MERGE_MAX_LINES,
    PARA_MERGE_MIN_WIDTH_RATIO, PARA_MERGE_OVERLAP_RATIO, PARA_MERGE_STOP_CHARS,
    TESSERACT_ERROR, TESS_CONFIG,
    UIA_PROVIDER, _TESSERACT_CMD, _ocr_line_is_noise, cjk_char_count,
    drop_pinyin_lines, environment_summary, has_hangul, ink_glyph_px,
    join_ocr_words, unread_ink_ratio,
)
from cocktail_platform import is_sensitive_window
from cocktail_translate import (
    CACHE_LOCK, LANG_MAP, LAST_GROUP_ERRORS, PERSIST_CACHE, TRANSLATOR,
    TRANS_CACHE,
    assign_region_languages, batch_translate, display_gate_reject,
)


def _mean_conf(confs):
    """UA-1: 유효한 라인 conf 의 평균. 하나도 없으면 -1(= 항상 진다)."""
    ok = [c for c in (confs or []) if c >= 0]
    return sum(ok) / len(ok) if ok else -1.0


def merge_paragraph_lines(lines, confs=None):
    """PM-1: 줄바꿈으로 이어진 OCR 라인들을 한 문단으로 합친다.

    입력/출력 라인 튜플은 `(text, (l, t, r, b), font_size)` 3원소 계약 그대로다 —
    Paddle/UIA 경로와 공용이라 계약을 바꾸면 다섯 군데가 같이 흔들린다.
    `confs`를 주면 같은 순서로 합쳐 돌려준다(길이 가중 평균).

    합치는 조건은 `cocktail_ocr.PARA_MERGE_*` 참고. 핵심은 "앞 줄이 뒤 줄보다 확연히
    짧으면 합치지 않는다" — 제목/마지막 줄이 본문에 빨려들어가는 걸 막는 규칙이다.
    """
    if not PARA_MERGE_ENABLED or len(lines) < 2:
        return lines, confs

    order = sorted(range(len(lines)), key=lambda i: (lines[i][1][1], lines[i][1][0]))
    pos = {idx: k for k, idx in enumerate(order)}
    groups = []
    for k, idx in enumerate(order):
        # 바로 앞 줄이 아니라 **열려 있는 모든 문단**과 대본다. 위→아래 정렬만 쓰면
        # 옆 단(column)이나 화면 구석의 아이콘 한 조각이 사이에 끼는 순간 문단이 끊긴다.
        for g in reversed(groups):
            if len(g) >= PARA_MERGE_MAX_LINES or not _same_paragraph(lines[g[-1]], lines[idx]):
                continue
            # PM-2: 그런데 건너뛴 줄이 **합쳐질 문단의 가로 구간 안**에 있으면 읽기 순서가
            # 뒤집힌다. 실측(2026-09-07): 1·2·4번 줄이 한 문단이 되고 3번이 따로 남아
            # 문장이 "1 2 4 3" 순서로 나갔다 — CER 8.70→45.45 / 10.67→45.06 / 4.35→33.99.
            # 옆 단(column)·구석 아이콘은 이 구간 밖이라 원래 의도(칼럼 건너뛰기)는 산다.
            left = min(lines[i][1][0] for i in g + [idx])
            right = max(lines[i][1][2] for i in g + [idx])
            if any(min(right, lines[order[j]][1][2]) > max(left, lines[order[j]][1][0])
                   for j in range(pos[g[-1]] + 1, k)):
                continue
            g.append(idx)
            break
        else:
            groups.append([idx])
    groups.sort(key=lambda g: (lines[g[0]][1][1], lines[g[0]][1][0]))

    out_lines, out_confs = [], ([] if confs is not None else None)
    for g in groups:
        members = [lines[i] for i in g]
        # ZH-1: 줄 사이도 CJK 경계면 공백을 넣지 않는다 (중국어/일본어는 띄어쓰기가 없다).
        text = join_ocr_words(m[0].strip() for m in members if m[0].strip())
        left = min(m[1][0] for m in members)
        top = min(m[1][1] for m in members)
        right = max(m[1][2] for m in members)
        bottom = max(m[1][3] for m in members)
        font_size = int(round(float(np.median([m[2] for m in members]))))
        out_lines.append((text, (left, top, right, bottom), max(1, font_size)))
        if out_confs is not None:
            # 길이 가중 평균 — 긴 줄이 문단 신뢰도를 더 많이 결정한다.
            pairs = [(len(lines[i][0]), confs[i]) for i in g
                     if i < len(confs) and confs[i] >= 0]
            total = sum(n for n, _ in pairs)
            out_confs.append(sum(n * c for n, c in pairs) / total if total else -1.0)
    return out_lines, out_confs


def _same_paragraph(prev, cur):
    _ptext, (pl, ptop, pr, pb), pfs = prev
    _ctext, (cl, ctop, cr, _cb), cfs = cur
    height = max(1, pb - ptop)
    if not (-height * PARA_MERGE_OVERLAP_RATIO
            <= ctop - pb <= height * PARA_MERGE_GAP_RATIO):
        return False
    if abs(cl - pl) > height * PARA_MERGE_LEFT_TOL_RATIO:
        return False
    ratio = cfs / float(max(1, pfs))
    if not (PARA_MERGE_FONT_LO <= ratio <= PARA_MERGE_FONT_HI):
        return False
    if _ptext.rstrip().endswith(PARA_MERGE_STOP_CHARS):
        return False
    # 앞 줄이 뒤 줄보다 확연히 짧다 = 줄바꿈된 줄이 아니라 제목이거나 문단 마지막 줄.
    if (pr - pl) < (cr - cl) * PARA_MERGE_MIN_WIDTH_RATIO:
        return False
    return True


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
    BG-6: depends on `region`, `options` snapshot, `is_self_hwnd` callback only —
    no widget references (TS-1: 워커가 위젯을 읽던 마지막 경로도 제거됨).
    """

    state_ready = Signal(list)
    failure_alert = Signal(str)
    model_ready = Signal()
    model_failed = Signal(str)   # MR-1: 모델 로드 실패 — UI가 재시도 버튼을 살린다
    recovered = Signal()         # RB-1: 연속 실패에서 복구됨 (경고 문구 해제용)
    progress_msg = Signal(str)
    status_msg = Signal(str)

    def __init__(self, region, options, is_self_hwnd):
        super().__init__()
        self.region = region
        # TS-1: 워커는 위젯을 만지지 않는다. 콤보 상태는 **메인 스레드가 스냅샷을 떠서
        # 여기에 밀어 넣고**(set_options), 워커는 이 frozen dataclass를 읽기만 한다.
        # 예전에는 워커가 매 사이클 `src_combo.currentText()`를 직접 불렀다 — GT-1(QCursor)
        # /MS-1(QApplication.screens)에서 그렇게 조심한 것과 정확히 같은 종류의 규약 위반.
        # frozen dataclass 재바인딩은 CPython에서 원자적이라 별도 lock이 필요 없다.
        self.options = options
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
        self._model_loading = False   # MR-1: 재시도 중복 실행 가드
        self._sensitive_paused = False
        self._capturable_paused = False
        self._refresh_env_status()

        self._cache_save_in_progress = False
        self._cache_save_timer = None
        self.translation_thread = None
        self._ocr_line_confs = []   # DG-1: 직전 OCR의 라인별 평균 confidence (표시 게이트용)
        self._last_change_at = 0.0   # PL-1: 마지막으로 "일거리가 있었던" 시각(monotonic)
        self._zh_frames = 0         # ZH-1: chi_sim을 계속 태울 남은 프레임 수 (0이면 진입 판정부터)
        self._ko_frames = 0         # LC-1b: kor을 도로 켜 둘 남은 프레임 수 (음수면 재스캔 쿨다운)

        # Engine maintains its own translated_text on main thread.
        self.state_ready.connect(self._apply_state_to_self)

    # --- public API ----------------------------------------------------------
    def start(self):
        self.retry_model_load()
        PERSIST_CACHE.preload_async()

        self._cache_save_timer = QTimer(self)
        self._cache_save_timer.timeout.connect(self._on_cache_save_tick)
        self._cache_save_timer.start(30 * 1000)

        self.translation_thread = threading.Thread(
            target=self.run_worker_loop, daemon=True
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

    # --- PV-1: 번역 기록(영구 캐시) — UI가 부르는 두 진입점 -------------------
    def set_persist_cache(self, on: bool):
        """설정 토글. 끄면 디스크 기록이 같이 지워진다(PersistentCache.set_enabled)."""
        PERSIST_CACHE.set_enabled(on)
        if on:
            PERSIST_CACHE.preload_async()

    def clear_translation_history(self):
        """'번역 기록 삭제' — 메모리 캐시 + 디스크 기록을 즉시 지운다.

        `(메모리 건수, 디스크 건수, 파일이 실제로 사라졌는가)`. 마지막 항목은 추측이
        아니라 삭제 뒤에 다시 확인한 값이다 — 사용자에게 그대로 보여준다.
        """
        with CACHE_LOCK:
            mem = len(TRANS_CACHE)
            TRANS_CACHE.clear()
        disk = PERSIST_CACHE.clear()
        return mem, disk, not os.path.exists(PERSIST_CACHE.path)

    def set_running(self, value: bool):
        self.model_loaded = value
        # UX-3: 정지 시 오버레이 잔상 제거. 토글의 단일 진입점이라 버튼/단축키/트레이 전 경로 커버.
        if not value:
            self._clear_overlay_if_any()

    def is_model_ok(self) -> bool:
        return self._model_ok

    def set_capturable(self, value: bool):
        self.capturable = value

    def set_options(self, options: "RuntimeOptions"):
        """TS-1: 메인 스레드 전용. 콤보가 바뀔 때마다 스냅샷을 밀어 넣는다."""
        self.options = options

    def reset_frame_hash(self):
        self.last_frame_hash = None

    def poll_interval(self):
        """PL-1: 다음 캡처까지 쉴 시간. 최근에 일거리가 있었으면 짧게, 조용하면 예전 값 그대로.

        자막·채팅은 한 번 바뀌면 곧 또 바뀐다 — 그 구간에서만 촘촘히 본다.
        아무 변화 없이 POLL_ACTIVE_WINDOW_S 가 지나면 POLL_IDLE_S 로 돌아가므로
        **유휴 CPU는 예전과 같다**(B0의 24%→8% 성과 보존).

        기준 시각은 "변화를 잡은 순간"이 아니라 **"직전 사이클이 끝난 순간"**이다.
        빽빽한 화면은 한 사이클이 2초씩 걸려서, 시작 시각으로 재면 창이 사이클 도중에
        만료되고 곧바로 다음 자막을 250 ms 늦게 잡는다(실측: 채팅 화면 poll 252 ms 그대로).
        """
        if time.monotonic() - self._last_change_at < POLL_ACTIVE_WINDOW_S:
            return POLL_ACTIVE_S
        return POLL_IDLE_S

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
    def retry_model_load(self) -> bool:
        """MR-1: 모델 로드를 (다시) 시작한다. 이미 로드됐거나 로드 중이면 False.

        구버전은 `_preload_model`이 실패하면 `run_button`이 비활성 상태로 굳어 **재시도
        경로가 아예 없었다**. 첫 실행 시 모델 1~2GB를 받아야 하는 앱에서 인터넷이 잠깐
        끊기면 앱이 영구히 죽은 것과 같았다(재시작해도 같은 실패 반복).
        """
        if self._model_ok or self._model_loading:
            return False
        self._model_loading = True
        threading.Thread(target=self._preload_model, daemon=True).start()
        return True

    def _refresh_env_status(self):
        """LZ-1: 진단 한 줄. torch 미로드 상태에선 '번역 준비 중'으로 나오고,
        모델 로드가 끝난 뒤 한 번 더 불러 CUDA/CPU를 확정한다."""
        status, problems = environment_summary()
        # 진단 문제를 저장만 하고 아무데도 안 쓰던 필드였다 → 상태줄에 실제로 노출한다.
        self.env_status = (f"{status} | 점검: {', '.join(problems)}"
                           if problems else status)

    def _preload_model(self):
        try:
            TRANSLATOR.preload_default(progress_cb=lambda msg: self.progress_msg.emit(msg))
            self._model_ok = True
            self._refresh_env_status()   # LZ-1: 이제 torch가 떠 있으니 CUDA/CPU 확정
            self.model_ready.emit()
        except Exception as e:
            self._model_ok = False
            print(f"[ERROR] 모델 로드 실패: {e}")
            self.model_failed.emit(f"모델 로드 실패 ({type(e).__name__})")
        finally:
            self._model_loading = False

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
    def _dirty_band_view(self, frame, prev_hash, cur_hash, y1):
        """PF-2: (OCR 대상 이미지, 세로 오프셋, 유지할 직전 자막) 3-튜플을 돌려준다.

        오프셋이 `None`이면 "밴드를 못 정했으니 전체 프레임을 처리하라"는 뜻이다.
        가로는 절대 자르지 않는다 — 단어가 반으로 잘리면 OCR이 통째로 망가진다.
        """
        if not OCR_DIRTY_BAND_ENABLED:
            return frame, None, []
        band = dirty_row_band(prev_hash, cur_hash)
        if band is None:
            return frame, None, []
        row0, row1 = band
        full_h = frame.shape[0]
        pad = max(8, full_h // 32)          # 셀 경계에 걸친 글자가 잘리지 않게 여유
        top = max(0, row0 * full_h // 16 - pad)
        bottom = min(full_h, row1 * full_h // 16 + pad)

        # 밴드에 조금이라도 걸친 직전 자막은 **통째로** 밴드에 포함시킨다.
        # 반쪽만 재OCR 되면 잘린 글자가 새 자막으로 올라온다.
        for _s, _t, bb, _f in self._last_emitted:
            b_top, b_bottom = bb[1] - y1, bb[3] - y1
            if b_bottom > top and b_top < bottom:
                top = max(0, min(top, b_top - 2))
                bottom = min(full_h, max(bottom, b_bottom + 2))

        height = bottom - top
        if height < 16 or height >= full_h * OCR_DIRTY_FULL_RATIO:
            return frame, None, []          # 밴드가 너무 크면 이득이 없다
        keep_top, keep_bottom = y1 + top, y1 + bottom
        kept = [e for e in self._last_emitted
                if e[2][3] <= keep_top or e[2][1] >= keep_bottom]
        return frame[top:bottom], top, kept

    def run_worker_loop(self):
        """AS-1: 예전 이름 `run_asyncio_loop`. 코루틴이 하나뿐이고 그 안의 OCR/번역은
        전부 동기 블로킹이라 asyncio가 하는 일이 `sleep` 하나뿐이었다 —
        이벤트 루프 생성/종료 비용과 디버깅 난이도만 얹은 순수 장식이었다."""
        self.capture_and_translate()

    def capture_and_translate(self):
        while not self.stop_thread:
            if not self.model_loaded or not self._model_ok:
                time.sleep(0.2)
                continue

            try:
                # SB-1: 일시정지 판정은 pause_reason() 하나로 모았다 (캡처 모드 무관).
                reason = self.pause_reason()
                if reason == "capturable":
                    self._clear_overlay_if_any()   # RC-1
                    if not self._capturable_paused:
                        self._capturable_paused = True
                        self.status_msg.emit(
                            "⚠ 캡처 허용(스샷OK) 모드 — B-10 피드백 위험으로 번역 일시 중지. "
                            "다시 '스샷차단'을 누르거나 Ctrl+Shift+H로 보호를 켜세요."
                        )
                    time.sleep(1.0)
                    continue
                if reason == "sensitive":
                    self._clear_overlay_if_any()   # RC-1
                    if not self._sensitive_paused:
                        self._sensitive_paused = True
                        self.status_msg.emit("민감 창 감지 — 번역 일시 중지 (보안 원칙 #3)")
                    time.sleep(1.0)
                    continue

                if self._capturable_paused:
                    self._capturable_paused = False
                    self.status_msg.emit(f"{self.env_status} | 캡처 차단 복구 — 번역 재개")
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
                    time.sleep(0.2)
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
                                time.sleep(0.2)
                                continue
                    except Exception:
                        pass

                # UX-2 Layer 1: UIA 자동 라우팅.
                # DC-1: 예전의 두 번째 조건절(`MODE_WINDOW and UIA 가능 and 백엔드가
                # Tesseract/Paddle이 아님`)은 도달 불가능한 죽은 분기였다 — 콤보 항목이
                # 그 셋뿐이라 "Tesseract도 Paddle도 아니다" == "UIA다" 였다.
                if self.options.ocr_text == OCR_BACKEND_UIA:
                    uia_state = self._uia_collect_and_translate()
                    # SV-1: 민감 창은 이 프레임 전체를 건너뛴다. 여기서 OCR로 폴백하면
                    # 민감 창이 캡처·OCR·번역·영구 캐시된다 (S-1 가드를 우회하는 구멍이었음).
                    if uia_state is UIA_SKIP_FRAME:
                        time.sleep(1.0)
                        continue
                    if uia_state is not None and uia_state:
                        self._emit_state_if_changed(uia_state)  # RT-2
                        if self.fail_streak > 0:
                            self.recovered.emit()   # RB-1: 복구는 실패 채널이 아니라 복구 채널로
                        self.fail_streak = 0
                        time.sleep(0.5)
                        continue

                # W-5: all_screens=True — win32 경로가 가상 데스크톱 전체를 찍고
                # offset(가상 화면 원점) 기준으로 crop하므로 절대 좌표 bbox가 그대로 맞는다.
                # (False면 primary만 찍어 보조 모니터 영역이 검은 이미지로 나옴)
                # CP-1: 그래서 **주 모니터 안에 완전히 들어갈 때만** False로 내려간다 (131→33 ms).
                screenshot = ImageGrab.grab(bbox=(x1, y1, x2, y2),
                                            all_screens=needs_all_screens(x1, y1, x2, y2))
                screenshot_np = np.array(screenshot)

                # frame-skip (P-4 / F-1)
                prev_hash = self.last_frame_hash
                h = cheap_hash(screenshot_np, (x1, y1, x2, y2))
                if hash_distance(prev_hash, h) < FRAME_DIFF_THRESHOLD:
                    time.sleep(self.poll_interval())
                    continue
                self.last_frame_hash = h
                self._last_change_at = time.monotonic()   # PL-1

                # PF-2: 변한 세로 밴드만 OCR. `band_dy is None`이면 전체 프레임 처리다.
                ocr_image, band_dy, kept = self._dirty_band_view(
                    screenshot_np, prev_hash, h, y1)
                dy = band_dy or 0

                t_ocr = time.perf_counter()
                ocr_results = self.perform_ocr_with_boxes(ocr_image)
                ocr_ms = (time.perf_counter() - t_ocr) * 1000.0

                if not ocr_results:
                    # PF-2: 밴드 모드면 "밴드 안에서만" 글자가 사라진 것이다.
                    # 화면 전체를 비우면 밴드 밖의 멀쩡한 자막까지 날아간다.
                    if band_dy is None:
                        self._clear_overlay_if_any()   # RC-1
                    else:
                        self._emit_state_if_changed(kept)
                    self.no_ocr_streak += 1
                    if self.no_ocr_streak in {1, 5, 20}:
                        self.status_msg.emit(
                            f"OCR 결과 없음: 영역/언어/OCR 백엔드 확인 필요 ({self._ocr_backend_name()})"
                        )
                    # PL-1: 변화는 잡혔는데 글자가 없다 = 자막이 방금 사라진 순간이다.
                    # 다음 자막이 곧 뜨므로 고정 0.3s로 물러나지 않는다.
                    self._last_change_at = time.monotonic()
                    time.sleep(self.poll_interval())
                    continue
                self.no_ocr_streak = 0

                src_hint = self._src_hint()
                tgt_lang = LANG_MAP[self.options.tgt_text]
                src_texts = [t for t, _, _ in ocr_results]
                src_langs = assign_region_languages(ocr_results, src_hint)

                t0 = time.perf_counter()
                translated = batch_translate(src_texts, src_langs, tgt_lang)
                dt = (time.perf_counter() - t0) * 1000.0
                print(f"[PERF] ocr {ocr_image.shape[1]}x{ocr_image.shape[0]} "
                      f"{ocr_ms:.0f} ms | translate {len(src_texts)} lines {dt:.1f} ms"
                      f"{' | band' if band_dy is not None else ''}")
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
                    # PF-2: 밴드를 잘라 OCR 했으면 세로 오프셋(dy)을 더해 원래 프레임 좌표로.
                    adjusted_bbox = (
                        bbox[0] + x1,
                        bbox[1] + y1 + dy,
                        bbox[2] + x1,
                        bbox[3] + y1 + dy,
                    )
                    new_state.append((src_text, tgt_text, adjusted_bbox, fsz))

                # PF-2: 밴드 밖에서 살아남은 직전 자막과 합친다.
                # 정렬은 RT-2 지문(_state_signature)이 순서에 좌우되지 않게 하려는 것 —
                # 전체 프레임 패스와 밴드 패스의 결과 순서가 달라지면 매번 헛 재그리기가 난다.
                new_state = sorted(kept + new_state, key=lambda e: (e[2][1], e[2][0]))

                self.status_msg.emit(
                    f"{self._ocr_backend_name()} OCR {len(src_texts)}줄 [{lang_summary}] → 번역 완료 "
                    f"(OCR {ocr_ms:.0f} + 번역 {dt:.0f} ms)"
                    + (f" · 숨김 {hidden}줄(DG-1)" if hidden else "")
                    # TE-1: 실패한 언어 그룹은 조용히 사라지지 않는다.
                    + (f" · {' / '.join(LAST_GROUP_ERRORS)}" if LAST_GROUP_ERRORS else "")
                )

                # RT-2: 실질 동일한 상태면 재그리기 생략.
                self._emit_state_if_changed(new_state)

                if self.fail_streak > 0:
                    self.recovered.emit()   # RB-1: 복구는 실패 채널이 아니라 복구 채널로
                self.fail_streak = 0

            except Exception as e:
                # E-8: 일시적 실패는 스킵, 한도 초과 시 backoff.
                self.fail_streak += 1
                print(f"[WARN] capture/translate 실패 #{self.fail_streak}: {e}")
                if self.fail_streak >= 20:
                    print("[WARN] 연속 실패 한도 — 5초 backoff 후 재시도")
                    self.failure_alert.emit("일시 정지(5s)")
                    self.fail_streak = 0
                    time.sleep(5.0)
                    continue

            # PL-1: 한 사이클을 실제로 처리했다 = 방금까지 일거리가 있었다.
            self._last_change_at = time.monotonic()
            time.sleep(self.poll_interval())

    # --- routing helpers -----------------------------------------------------
    def _src_hint(self):
        v = self.options.src_text
        return "auto" if v == "auto" else LANG_MAP[v]

    def _ocr_backend_name(self):
        return self.options.ocr_text

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
            # LC-1: 목표 언어와 같은 언어팩은 읽어 봐야 SK-1이 버린다. 시간만 2.5배.
            # LC-1b: 단 한글 화면으로 확인된 동안(_ko_frames > 0)에는 도로 켠다 —
            # 빼 두면 한글을 라틴으로 오독해 엉뚱한 자막이 뜬다.
            redundant = OCR_LANGS_TGT_REDUNDANT.get(LANG_MAP.get(self.options.tgt_text))
            if redundant and len(wanted) > 1 and self._ko_frames <= 0:
                wanted = [t for t in wanted if t != redundant]
        avail = [t for t in wanted if t in AVAILABLE_TESS_LANGS]
        return "+".join(avail) if avail else "eng"

    def _zh_rescan_lang(self, base_lang):
        """ZH-1: auto 모드에서 chi_sim 을 덧붙일 수 있으면 붙인 언어 문자열, 아니면 None."""
        if not (OCR_ZH_RESCAN_ENABLED and self._src_hint() == "auto"):
            return None
        if OCR_ZH_TESS_LANG not in AVAILABLE_TESS_LANGS:
            return None
        if OCR_ZH_TESS_LANG in base_lang.split("+"):
            return None
        return base_lang + "+" + OCR_ZH_TESS_LANG

    def _ko_rescan(self, gray, lang_str, lines, confs):
        """LC-1b: kor 을 뺀 auto 스캔이 한글 화면을 만났으면 kor 을 붙여 다시 읽는다.

        LC-1로 kor 을 빼면 한글이 라틴 오독(conf 39~70)으로 나와 자막 박스가 뜬다.
        "낮은 conf 라인 비율"이 근거이고, 채택 기준은 UA-1과 같은 평균 conf 개선이다.
        한글이 확인되면 sticky 로 kor 을 계속 켜 두므로(_ocr_langs) 왕복은 진입 프레임 1회뿐.
        """
        if self._src_hint() != "auto" or "kor" not in AVAILABLE_TESS_LANGS:
            return lines, confs
        if "kor" in lang_str.split("+"):
            # 이미 kor 로 읽는 중(sticky). 한글이 계속 보이면 유지, 아니면 서서히 만료.
            if any(has_hangul(text) for text, _, _ in lines):
                self._ko_frames = OCR_KO_STICKY_FRAMES
            elif self._ko_frames > 0:
                self._ko_frames -= 1
            return lines, confs
        if self._ko_frames < 0:          # 직전 재스캔이 헛수고였다 — 쿨다운 소진
            self._ko_frames += 1
            return lines, confs
        if not confs:
            return lines, confs
        junk = sum(1 for c in confs if 0 <= c < OCR_KO_JUNK_CONF)
        if (_mean_conf(confs) >= OCR_KO_MEAN_CONF
                and junk / len(confs) < OCR_KO_JUNK_RATIO):
            return lines, confs
        lines2, confs2 = self._tess_scan(gray, lang_str + "+kor")
        if _mean_conf(confs2) > _mean_conf(confs):
            self._ko_frames = OCR_KO_STICKY_FRAMES
            return lines2, confs2
        self._ko_frames = -OCR_KO_FAIL_COOLDOWN   # 그냥 흐릿한 영어 화면이었다
        return lines, confs

    # --- OCR -----------------------------------------------------------------
    def perform_ocr_with_boxes(self, image):
        # DG-1: 라인별 평균 confidence를 표시 게이트에서 쓰기 위해 매 프레임 초기화한다.
        # (라인 튜플 자체는 (text, bbox, font_size) 3원소 계약 유지 — Paddle/UIA 경로와 공용.
        #  ponytail: 병렬 리스트 한 개면 충분하다. 계약을 바꾸면 5곳이 같이 흔들린다.)
        self._ocr_line_confs = []
        src_hint = self._src_hint()
        if self.options.ocr_text == OCR_BACKEND_PADDLE:
            try:
                paddle_results = PADDLE_OCR.recognize(image, src_hint)
                if paddle_results is not None:
                    # PY-1: 병음 제외는 백엔드 공통이다 (Paddle 도 병음 줄을 그대로 읽는다).
                    return drop_pinyin_lines(paddle_results)[0]
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
            zh_lang = self._zh_rescan_lang(lang_str)

            # ZH-1: 이미 중국어 화면으로 확인된 동안에는 1차 패스를 건너뛰고 바로 chi_sim.
            # (매 프레임 왕복 2회 → 1회. 근거가 끊기면 카운터가 0이 되며 원래대로 돌아간다.)
            zh_screen = bool(zh_lang) and self._zh_frames > 0
            entered_zh = False
            if zh_screen:
                lines, confs = self._tess_scan(gray, zh_lang)
            else:
                lines, confs = self._tess_scan(gray, lang_str)
                lines, confs = self._ko_rescan(gray, lang_str, lines, confs)   # LC-1b
                # 진입 근거: (a) 병음 줄이 있다(병기 화면) / (b) 한 줄도 못 읽었다 /
                # (c) ZH-2: 읽은 라인이 화면의 잉크를 거의 못 덮는다(한자를 통째로 흘림).
                probe, _, pinyin_hits = drop_pinyin_lines(lines, confs)
                # `_zh_frames < 0` 은 "직전에 헛수고했다" 쿨다운이다. 없으면 영상/게임처럼
                # 글자 없이 계속 변하는 화면에서 매 프레임 chi_sim 패스를 한 번 더 돌린다
                # (실측 +140 ms/프레임). 병음이 보이면 쿨다운을 무시한다.
                cold = self._zh_frames != 0
                if zh_lang and (pinyin_hits or (not cold and (
                        not probe
                        or unread_ink_ratio(gray, [b for _, b, _ in probe])
                        >= OCR_ZH_UNREAD_INK_RATIO))):
                    lines, confs = self._tess_scan(gray, zh_lang)
                    zh_screen = True
                    entered_zh = True

            # PY-1: 병음은 발음 힌트지 글이 아니다. 번역 입력에서 뺀다.
            lines, confs, pinyin_dropped = drop_pinyin_lines(lines, confs, zh_screen)

            if zh_lang:
                cjk = sum(cjk_char_count(text) for text, _, _ in lines)
                if cjk >= OCR_ZH_MIN_CJK_CHARS or pinyin_dropped:
                    self._zh_frames = OCR_ZH_STICKY_FRAMES
                elif entered_zh:
                    # 진입 근거를 믿고 chi_sim 을 태웠는데 한자가 없었다 — 그림 많은 화면이거나
                    # 글자 없는 화면이다. 매 프레임 헛 왕복하지 않게 쿨다운으로 떨어뜨린다.
                    self._zh_frames = -OCR_ZH_EMPTY_COOLDOWN
                elif self._zh_frames:
                    # 양수는 줄이고(중국어에서 이탈), 음수는 늘린다(쿨다운 소진). 둘 다 0으로 수렴.
                    self._zh_frames += 1 if self._zh_frames < 0 else -1

            # PM-1: 번역기에 넘기기 전에 줄바꿈으로 이어진 줄들을 문단으로 합친다.
            lines, confs = merge_paragraph_lines(lines, confs)
            self._ocr_line_confs = confs
            return lines
        except Exception as e:
            print(f"[WARN] OCR 실패: {e}")
            return []

    def _tess_scan(self, gray, lang_str):
        """한 언어 문자열로 OCR 한 번(+OU-1 확대 재시도). 반환: (lines, confs)."""
        h, w = gray.shape[:2]
        # PF-1: 크기 안전밸브. 축소는 인식률을 그대로 깎으므로(cocktail_ocr.py 실측표)
        # 병리적으로 큰 입력만 막고, OCR_MIN_DOWNSCALE 아래로는 절대 줄이지 않는다.
        scale = min(1.0, OCR_MAX_SIDE_PX / float(max(h, w, 1)))
        scale = max(scale, OCR_MIN_DOWNSCALE)
        confs = []
        lines = self._tess_lines(gray, lang_str, scale, confs)

        # OU-1: 작은 글씨(캡션/각주)는 1차 OCR이 통째로 무너진다 → 확대 재OCR.
        # UB-1/RP-1: 배율은 **1차 패스가 읽어낸 글자 수**로 정한다(예산 안에 들어오게 낮춘다).
        #   비용 ≈ 글자 수 × 배율² 이므로 factor = √(예산 / 글자 수), 상한 2배.
        #   실측 시간을 쓰면 PC 부하에 따라 같은 입력이 다르게 읽힌다(RP-1) — 글자 수는
        #   같은 이미지에서 항상 같은 값이라 결과가 재현된다.
        # 싼 조건(배율 예산)을 먼저 본다 — `_wants_upscale`은 8~9 ms 짜리 이미지 측정을
        # 할 수도 있어서, 어차피 확대 못 할 프레임에서 그 값을 치르면 안 된다.
        first_chars = sum(len(text) for text, _, _ in lines)
        factor = min(OCR_UPSCALE_FACTOR,
                     math.sqrt(OCR_UPSCALE_CHAR_BUDGET / max(first_chars, 1)))
        if factor >= OCR_UPSCALE_MIN_FACTOR and self._wants_upscale(gray, lines):
            confs2 = []
            lines2 = self._tess_lines(gray, lang_str, scale * factor, confs2,
                                      blind=not lines)
            # UA-1: 채택 기준은 **라인 수가 아니라 평균 confidence**다.
            # 라인 수로 고르면 "쓰레기 줄이 더 많이 생긴 쪽"이 이긴다.
            # 실측(2026-08-05): 픽셀 폰트 UI는 확대하면 라인이 20→22로 늘지만
            # 정확도는 0.869→0.850으로 떨어진다(구 규칙은 이걸 채택했다).
            # 평균 conf는 71.9→60.4로 정직하게 떨어져 확대를 거부한다.
            # 문서에서는 반대로 93.9→94.7이라 확대를 채택하고 0.869→0.938을 얻는다.
            # UA-2: conf 만 보면 **줄을 통째로 흘린 패스가 이긴다**(남은 줄만 깨끗하니
            # 평균이 오른다). 그래서 "글자를 버리지 않았을 것"을 함께 요구한다.
            # 1차가 0줄이면 비교할 것이 없다 — 확대 결과를 그대로 쓴다(OU-2).
            if not lines or (_mean_conf(confs2) > _mean_conf(confs) - OCR_UPSCALE_CONF_MARGIN
                             and sum(len(t) for t, _, _ in lines2)
                             >= first_chars * OCR_UPSCALE_MIN_COVERAGE):
                lines, confs = lines2, confs2
        return lines, confs

    @staticmethod
    def _wants_upscale(gray, lines):
        """OU-2: 확대 재OCR을 켤 것인가. **1차 오독에 흔들리지 않는 판정.**

        예전 규칙은 `if lines:` 안에서 1차가 읽은 word 높이 중앙값 하나만 봤다. 그런데
        1차가 무너지면 그 값도 같이 오염된다 — 실측(2026-09-07):
          - 10px Georgia/Calibri/Times/Impact 를 **28px 로 보고** → 확대 꺼짐, CER 10.7%.
            강제로 켜면 0.00%. 즉 확대가 가장 필요한 조합에서만 확대가 꺼졌다.
          - 1차가 **0줄**이면 `if lines:` 에 걸려 확대를 아예 건너뛰었다. 0줄이야말로
            확대가 필요한 상황이다(Courier New 10px 100% → 2배 재시도로 0.00%, OU-3 포함).
        → 높이 규칙이 "확대 불필요"라고 할 때만 이미지에서 직접 글리프 높이를 재어
          두 번째 의견을 묻는다. 빈 화면은 None 이라 확대가 켜지지 않는다(비용 방지).
        결정론은 유지된다(RP-1): 두 신호 다 같은 이미지에서 항상 같은 값이 나온다.
        """
        if lines:
            median_px = float(np.median([font_size for _, _, font_size in lines]))
            if median_px < OCR_UPSCALE_TRIGGER_PX:
                return True
        glyph_px = ink_glyph_px(gray)
        return glyph_px is not None and glyph_px < OCR_UPSCALE_TRIGGER_INK_PX

    def _tess_lines(self, gray, lang_str, scale, conf_out=None, blind=False):
        """grayscale 이미지를 scale배(1보다 작으면 축소) 리샘플 → Tesseract → 원본 좌표 라인.

        BZ-1: 예전에는 Otsu 전역 이진화 + `MORPH_OPEN(1x1)`을 먼저 걸었다. 실측 결과
          - `MORPH_OPEN`의 1x1 커널은 **항등 연산**이다(입력==출력). 매 프레임 전체 순회만 낭비.
          - Otsu는 정확도에 영향이 없었다. Tesseract 5가 자체 적응형 이진화를 이미 하기 때문에
            (단일 밝기 24/24 vs 24/24, 다크+라이트 혼합 20/24 vs 20/24, mean conf 동일)
            앞단 이진화는 순수 비용이었다.
        → grayscale을 그대로 넘긴다.

        conf_out: 주면 라인별 평균 confidence를 같은 순서로 채운다 (DG-1).
        blind: 1차 패스가 **한 줄도 못 읽은** 뒤의 확대 재시도인가 (OU-3).
        """
        if abs(scale - 1.0) > 1e-6:
            h, w = gray.shape[:2]
            # 축소는 INTER_AREA(에일리어싱 없이 글자 획을 보존).
            # OU-3(2026-09-07): 확대 보간은 **1차 패스가 뭐라도 읽었는지**에 따라 갈린다.
            #   Lanczos 는 음수 로브(링잉)로 획을 또렷하게 만든다 — 글자가 이미 분해되는
            #   해상도에서는 이득이지만(Calibri Light 10px 실화면 0.40% vs Cubic 8.30%),
            #   해상도 한계의 좁은 글꼴에서는 링잉 halo 가 글자를 붙여 버린다:
            #   Impact 11px 은 **정확히 2.0배에서만** 0줄(CER 100%), 1.5/2.5/3.0배는 멀쩡하다.
            #   그런 글자는 1차가 애초에 0줄이므로 그때만 링잉 없는 INTER_LINEAR 로 바꾼다.
            #   실측(bench size_·screen_ 23종, 이 규칙 기준 CER 최악):
            #     전부 LANCZOS4  100.00%(Impact 11px 0줄)   전부 CUBIC 8.30%   전부 LINEAR 5.53%
            #     1차 0줄일 때만 LINEAR → **2.77%**  ← 채택
            #   0줄 재시도 CER: Courier New 10px 4.35→0.00 · 같은 이탤릭 3.56→0.00 · Impact 11px 100→1.19
            interp = ((cv2.INTER_LINEAR if blind else cv2.INTER_LANCZOS4)
                      if scale > 1.0 else cv2.INTER_AREA)
            gray = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale))),
                              interpolation=interp)
        data = pytesseract.image_to_data(
            gray,
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
    def _trim_edge_noise(words):
        """LM-1b: 라인 양끝에 붙은 아이콘/장식 조각을 떼어낸다.

        판정은 "큰 간격 + 조각스러움"의 **논리곱**이다. 간격만 보면 들여쓰기가 잘리고,
        생김새만 보면 문장 속 짧은 단어("a", "of")가 잘린다. 둘 다일 때만 뗀다.

        입력/출력 word 튜플: (text, left, top, width, height, conf)
        """
        if not OCR_EDGE_TRIM_ENABLED or len(words) < 2:
            return words
        med_h = float(np.median([w[4] for w in words])) or 1.0
        gap_limit = max(4.0, med_h * OCR_EDGE_TRIM_GAP_RATIO)

        def junky(w):
            text = w[0].strip()
            if not any(ch.isalnum() for ch in text):
                return True    # '®', '+', '=', '**' — 글자가 아니다
            if not (med_h * OCR_EDGE_TRIM_HEIGHT_LO
                    <= w[4] <= med_h * OCR_EDGE_TRIM_HEIGHT_HI):
                return True    # 글자 높이가 라인 중앙값에서 벗어남
            # "짧으면 조각"만으로 판정하면 문서의 'as a', 'is' 같은 정상 단어가 잘린다.
            # 실측: 그림 조각 'ra'(conf 40) 'os'(16) 'ww'(35) vs 본문 'as'/'is'(conf 90+).
            return len(text) <= 2 and 0 <= w[5] < OCR_EDGE_TRIM_SHORT_CONF

        def gap_at(k):   # words[k-1] 과 words[k] 사이
            return words[k][1] - (words[k - 1][1] + words[k - 1][3])

        for _ in range(OCR_EDGE_TRIM_MAX):
            changed = False
            for k in range(1, len(words)):        # 앞쪽: 첫 큰 간격까지가 전부 조각이면 뗀다
                if gap_at(k) >= gap_limit:
                    if all(junky(w) for w in words[:k]):
                        words, changed = words[k:], True
                    break
            if len(words) < 2:
                break
            for k in range(len(words) - 1, 0, -1):  # 뒤쪽: 마지막 큰 간격부터가 전부 조각이면 뗀다
                if gap_at(k) >= gap_limit:
                    if all(junky(w) for w in words[k:]):
                        words, changed = words[:k], True
                    break
            if not changed or len(words) < 2:
                break
        return words

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
            # WD-1: 라인 안 단어 컷은 문장을 훼손한다. 레버로 노출(음수면 컷 없음).
            if OCR_WORD_MIN_CONF >= 0 and 0 <= conf < OCR_WORD_MIN_CONF:
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            entry = lines.setdefault(key, [])
            entry.append((
                txt,
                # PF-1: scale은 이제 실수(축소 시 1 미만)라 정수 나눗셈을 쓸 수 없다.
                int(data["left"][i] / scale),
                int(data["top"][i] / scale),
                int(data["width"][i] / scale),
                int(data["height"][i] / scale),
                conf,  # NZ-1: 라인 평균 confidence 계산용 / DG-1: 표시 게이트용
            ))

        out = []
        for key in sorted(lines.keys()):
            # LM-1: 이상치 제거 + 큰 간격 분할 후 세그먼트별로 기존 필터를 적용한다.
            for words in BackgroundController._line_segments(lines[key]):
                # LM-1b: 세그먼트 양끝의 아이콘/장식 조각 제거 (bbox·font_size 계산 전에).
                words = BackgroundController._trim_edge_noise(words)
                # ZH-1: 한자는 글자마다 별개 word 로 오므로 CJK 경계에는 공백을 넣지 않는다.
                full_text = join_ocr_words(w[0] for w in words).strip()
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
                # NZ-1: 글자 크기보다 좁은 라인 = 아이콘 한 조각.
                if (right - left) < font_size * OCR_LINE_MIN_WIDTH_RATIO:
                    continue
                out.append((full_text, (left, top, right, bottom), font_size))
                if conf_out is not None:
                    conf_out.append(avg_conf)
        return out

    # --- pause guards --------------------------------------------------------
    def pause_reason(self):
        """워커 루프의 일시정지 판정 **전부**. `None`이면 계속 진행.

        SB-1: 루프 본문에서 분리한 이유는 두 가지다.
          (a) 보안 결정이라 테스트가 루프를 돌리지 않고 직접 검증할 수 있어야 한다.
          (b) 여기가 바로 `capture_mode != MODE_BOX` 같은 모드별 예외가 슬며시 끼어들던
              자리다. 그 예외 하나 때문에 영역 박스 모드에서 은행/로그인 창이 그대로
              OCR·번역되고 DPAPI 영구 캐시에까지 남았다.
        **이 함수는 캡처 모드를 보지 않는다. 그것이 계약이다.**
        """
        if self.capturable:
            return "capturable"   # B-15: 스샷OK + 번역 = B-10 자기 캡처 피드백 위험
        if self._check_sensitive_pause():
            return "sensitive"    # S-1: 민감 창 (보안 원칙 #3)
        return None

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
            tgt_lang = LANG_MAP[self.options.tgt_text]
        except KeyError:
            tgt_lang = "ko"
        texts = [t for t, _, _ in uia_results]
        src_langs = assign_region_languages(uia_results, src_hint)
        try:
            translated = batch_translate(texts, src_langs, tgt_lang)
        except Exception as e:
            print(f"[WARN] UIA 번역 실패: {e}")
            self.status_msg.emit(f"UIA 번역 실패: {type(e).__name__}: {e}")   # CO-2
            return None
        if LAST_GROUP_ERRORS:
            self.status_msg.emit("UIA " + " / ".join(LAST_GROUP_ERRORS))      # TE-1

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

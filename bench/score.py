"""bench/score.py — Cocktail "합격선 6개" 자동 채점판.

이 파일 하나가 프로젝트의 완성 여부를 숫자로 답한다. 사람 판단이 들어가는 자리는
**측정하지 않고 체크리스트로 출력**한다 — 못 잰 것을 잰 척하지 않는 것이 이 도구의 계약이다.

    python bench/score.py                 전 항목
    python bench/score.py --only 1,4      일부만
    python bench/score.py --reps 20       기준 4: 자막/채팅이 각각 바뀌는 횟수

산출물:
    bench/last_score.json    원자료 (다음 실행과 비교용)
    bench/last_samples.md    기준 2의 사람 확인용 표본 (원문/번역문/판정)
    bench/last_ui/*.png      기준 5의 설정창·오버레이 렌더

설계 규칙:
  - 앱의 **진짜 함수**만 호출한다. OCR/문단병합/번역/표시게이트를 재구현하지 않는다.
    (`BackgroundController.perform_ocr_with_boxes` → `assign_region_languages` →
     `batch_translate` → `display_gate_reject` = 워커 루프와 같은 순서)
  - 엔진은 `from cocktail_ocr import ...` 로 상수를 복사해 간다. 레버를 바꿔 재보려면
    `cocktail_engine.<상수>` 를 갈아끼워야 한다 (`cocktail_ocr.<상수>` 는 효과 없음).
  - 기준 4는 **실제 앱을 띄워** 화면 변화 → 자막 지연을 잰다(별도 프로세스, `_latency_probe`).
    환경에 오염되므로 측정 전 GPU/CPU 여유를 재고, 오염이면 경고 + 결과에 기록한다.
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "bench"

# cocktail_ocr 는 `sys.argv[0]` 폴더의 tessdata 를 찾는다. bench/ 에서 실행하면
# 프로젝트 tessdata(chi_sim/kor 등 10종)를 못 보고 시스템 tessdata(eng/osd)로 떨어진다.
os.environ.setdefault("TESSDATA_PREFIX", str(ROOT / "tessdata"))
os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "0")
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

TGT_COMBO = "Korean"
TGT_LANG = "ko"

# 기준 4 오염 판정 임계 (넘으면 경고 + 결과에 오염 기록).
GPU_BUSY_PCT = 25
CPU_BUSY_PCT = 30
VRAM_FREE_MIN_MB = 3000

# "이 OCR 문단은 정답 문장을 제대로 읽은 것"으로 인정할 유사도. 기준 2의 오탐률 분모.
NORMAL_SIM = 0.80


# --- 문자열 유틸 -------------------------------------------------------------
def norm(s):
    return " ".join(str(s or "").split())


def lev(a, b):
    """편집 거리. numpy 행 DP + prefix-min 스캔이라 수천 자도 즉시 끝난다."""
    import numpy as np
    if a == b:
        return 0
    if not a or not b:
        return len(a) + len(b)
    barr = np.array([ord(c) for c in b], dtype=np.int64)
    idx = np.arange(len(b) + 1, dtype=np.int64)
    prev = idx.copy()
    for ch in a:
        cost = (barr != ord(ch)).astype(np.int64)
        m = np.empty(len(b) + 1, dtype=np.int64)
        m[0] = prev[0] + 1
        m[1:] = np.minimum(prev[1:] + 1, prev[:-1] + cost)
        # 가로(삽입) 항: cur[j] = min(m[j], cur[j-1]+1) = j + min_{k<=j}(m[k]-k)
        prev = np.minimum.accumulate(m - idx) + idx
    return int(prev[-1])


def sim(a, b):
    import difflib
    return difflib.SequenceMatcher(None, a, b).ratio()


_TOKEN = re.compile(r"[0-9A-Za-zÀ-ɏ]+|[㐀-鿿가-힯]")


def tokens(s):
    return _TOKEN.findall(str(s or "").lower())


# --- 앱 모듈 (lazy import: 기준 6은 import 비용 자체를 재므로 미리 물면 안 된다) ---
_APP = {}


def app():
    if not _APP:
        t0 = time.perf_counter()
        import cocktail_capture as cap
        import cocktail_engine as eng
        import cocktail_ocr as ocr
        import cocktail_translate as tr
        _APP.update(cap=cap, eng=eng, ocr=ocr, tr=tr,
                    import_s=time.perf_counter() - t0)
    return _APP


class _StubRegion:
    """`perform_ocr_with_boxes` 는 region 을 읽지 않는다. 생성자 계약만 채우는 스텁."""
    capture_mode = None

    def update_for_mode(self):
        pass

    def virtual_desktop_rect(self):
        return None


def make_ctrl(src="auto", tgt=TGT_COMBO, backend="Tesseract"):
    a = app()
    opts = a["cap"].RuntimeOptions(src_text=src, tgt_text=tgt, ocr_text=backend)
    ctrl = a["eng"].BackgroundController(
        region=_StubRegion(), options=opts, is_self_hwnd=lambda _h: False)
    # 벤치가 사용자의 DPAPI 영구 캐시를 읽거나 더럽히지 않게 한다(속도 측정도 오염된다).
    a["tr"].PERSIST_CACHE._enabled = False
    return ctrl


def load_image(path):
    import numpy as np
    from PIL import Image
    return np.array(Image.open(path).convert("RGB"))


def fixtures(names=None):
    out = []
    for p in sorted((BENCH / "truth").glob("*.json")):
        doc = json.loads(p.read_text(encoding="utf-8"))
        img = BENCH / "fixtures" / f"{doc['name']}.png"
        if not img.exists():
            print(f"[WARN] 이미지 없음: {img}")
            continue
        if names and doc["name"] not in names:
            continue
        doc["image"] = img
        out.append(doc)
    return out


# --- 앱 파이프라인 한 프레임 (워커 루프와 동일한 순서) -------------------------
def analyze(ctrl, image, tgt_lang=TGT_LANG):
    a = app()
    tr = a["tr"]
    # 화면이 바뀌면 sticky 카운터는 새 화면 기준으로 시작한다(프레임 간 오염 제거).
    ctrl._zh_frames = 0
    ctrl._ko_frames = 0

    t0 = time.perf_counter()
    lines = ctrl.perform_ocr_with_boxes(image)
    ocr_ms = (time.perf_counter() - t0) * 1000
    confs = list(ctrl._ocr_line_confs)

    src_langs = tr.assign_region_languages(lines, ctrl._src_hint())
    t1 = time.perf_counter()
    outs = tr.batch_translate([t for t, _, _ in lines], src_langs, tgt_lang)
    tr_ms = (time.perf_counter() - t1) * 1000

    t2 = time.perf_counter()
    rows = []
    for i, ((src_text, bbox, fsz), tgt_text) in enumerate(zip(lines, outs)):
        conf = confs[i] if i < len(confs) else None
        if not (tgt_text and tgt_text.strip()):
            status, reason = "no-output", ("SK-1(원문=목표언어)"
                                           if src_langs[i] == tgt_lang else "번역 실패/빈 출력")
        else:
            reason = tr.display_gate_reject(src_text, tgt_text, tgt_lang,
                                            bbox[3] - bbox[1], conf)
            status = "hidden" if reason else "shown"
        rows.append({"src": src_text, "tgt": tgt_text, "bbox": list(bbox),
                     "font_px": fsz, "conf": conf, "src_lang": src_langs[i],
                     "status": status, "reason": reason})
    gate_ms = (time.perf_counter() - t2) * 1000

    rows.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))
    return {"rows": rows, "ocr_ms": ocr_ms, "translate_ms": tr_ms, "gate_ms": gate_ms,
            "group_errors": list(tr.LAST_GROUP_ERRORS)}


def prewarm(ctrl):
    """모델 로드 + 첫 추론(커널 컴파일)을 측정 밖으로 뺀다."""
    a = app()
    t0 = time.perf_counter()
    a["tr"].TRANSLATOR.preload_default()
    # zh→ko 는 zh→en→ko 피벗(M-9). 두 다리를 모두 동기 로드해 둔다.
    a["tr"].TRANSLATOR._ensure_bilingual("zh", "en")
    a["tr"].batch_translate(["warm up the engine please"], ["en"], TGT_LANG)
    a["tr"].batch_translate(["今天天气很好"], ["zh"], TGT_LANG)
    a["tr"].TRANS_CACHE.clear()
    return time.perf_counter() - t0



def run_py(cmd, env=None, timeout=300):
    """자식 프로세스는 UTF-8로 말하게 하고, 콘솔 코드페이지(cp949) 디코드 사고를 막는다."""
    e = dict(os.environ)
    e["PYTHONIOENCODING"] = "utf-8"
    if env:
        e.update(env)
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", env=e, cwd=str(ROOT), timeout=timeout)


# --- 환경 오염 점검 (기준 4) --------------------------------------------------
def env_probe():
    info = {"gpu_util_pct": None, "vram_used_mb": None, "vram_total_mb": None,
            "cpu_pct": None, "contaminated": False, "notes": []}
    try:
        samples = []
        for _ in range(5):
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10)
            util, used, total = [int(x.strip()) for x in r.stdout.strip().split(",")]
            samples.append((util, used, total))
            time.sleep(0.2)
        info["gpu_util_pct"] = int(statistics.median(s[0] for s in samples))
        info["vram_used_mb"] = samples[-1][1]
        info["vram_total_mb"] = samples[-1][2]
    except Exception as e:
        info["notes"].append(f"nvidia-smi 조회 실패: {type(e).__name__}")
    try:
        import psutil
        info["cpu_pct"] = psutil.cpu_percent(interval=1.0)
        info["ram_free_mb"] = int(psutil.virtual_memory().available / 1e6)
    except Exception as e:
        info["notes"].append(f"psutil 조회 실패: {type(e).__name__}")

    if info["gpu_util_pct"] is not None and info["gpu_util_pct"] > GPU_BUSY_PCT:
        info["contaminated"] = True
        info["notes"].append(f"다른 프로세스가 GPU {info['gpu_util_pct']}% 사용 중")
    if info["vram_total_mb"]:
        free = info["vram_total_mb"] - info["vram_used_mb"]
        info["vram_free_mb"] = free
        if free < VRAM_FREE_MIN_MB:
            info["contaminated"] = True
            info["notes"].append(f"VRAM 여유 {free}MB (< {VRAM_FREE_MIN_MB}MB)")
    if info["cpu_pct"] is not None and info["cpu_pct"] > CPU_BUSY_PCT:
        info["contaminated"] = True
        info["notes"].append(f"CPU {info['cpu_pct']}% 사용 중")
    return info


# --- 기준 1: 작은 글씨 --------------------------------------------------------
def page_cer(truth_paras, rows):
    """페이지 단위 문자 오류율. 양쪽 다 (top,left) 읽기 순서로 이어 붙여 비교한다."""
    ref = norm(" ".join(truth_paras))
    hyp = norm(" ".join(r["src"] for r in rows))
    if not ref:
        return None, 0, 0
    return lev(ref, hyp) / len(ref), len(ref), len(hyp)


def criterion1(ctrl, results):
    rows = []
    for fx in fixtures():
        if not fx["name"].startswith(("size_", "screen_")):
            continue
        res = results[fx["name"]]
        cer, ref_len, hyp_len = page_cer(fx["paragraphs"], res["rows"])
        rows.append({"fixture": fx["name"], "font_px": fx["font_px"],
                     "cer": cer, "ref_chars": ref_len, "hyp_chars": hyp_len,
                     "ocr_lines": len(res["rows"])})
    graded = [r for r in rows if r["font_px"] in (10, 12, 14)]
    worst = max((r["cer"] for r in graded), default=1.0)
    return {"pass": bool(graded) and worst <= 0.05,
            "measured": f"10/12/14px CER 최대 {worst*100:.1f}%",
            "target": "≤ 5%", "detail": rows}


# --- 기준 2: 오번역 없음 ------------------------------------------------------
def best_truth(fx, text):
    best, score = None, 0.0
    for p in fx["paragraphs"]:
        s = sim(norm(p), norm(text))
        if s > score:
            best, score = p, s
    return best, score


def criterion2(ctrl, results, samples_path):
    tr = app()["tr"]
    normal_total = normal_hidden = normal_lost = 0
    hide_reasons = Counter()
    ko_misdisplay = []
    pinyin_leak = []
    pinyin_gated = []
    proxy_hits = []
    unread = []
    per_fixture = []
    sample_lines = ["# 기준 2 표본 (사람 확인용)", "",
                    "`정상판정` = OCR 원문이 정답 문장과 유사도 ≥ 0.80 (= 이 줄은 제대로 읽혔다).",
                    "`proxy` 열은 **대리 지표**다 — 뜻이 뒤집혔는지는 기계가 판정하지 못한다.", ""]

    for fx in fixtures():
        res = results[fx["name"]]
        shown = [r for r in res["rows"] if r["status"] == "shown"]
        f_norm = f_hidden = f_lost = 0

        if not fx.get("expect_display", True):
            # 목표 언어와 같은 화면 — 한 줄이라도 뜨면 오표시.
            ko_misdisplay += [{"fixture": fx["name"], "src": r["src"], "tgt": r["tgt"]}
                              for r in shown]
        else:
            for r in res["rows"]:
                _, score = best_truth(fx, r["src"])
                r["truth_sim"] = round(score, 3)
                if score < NORMAL_SIM:
                    continue
                f_norm += 1
                if r["status"] == "hidden":
                    f_hidden += 1
                    hide_reasons[str(r["reason"]).split("(")[0]] += 1
                elif r["status"] == "no-output":
                    f_lost += 1

        # 병음/금칙 문자열이 문장으로 번역됐는가. 화면에 떴으면 확정 사고,
        # 게이트에 걸려 숨었으면 "운 좋게 막힌 것"이라 따로 센다.
        for r in res["rows"]:
            for bad in fx.get("forbidden", []):
                if sim(norm(bad), norm(r["src"])) >= 0.6:
                    (pinyin_leak if r["status"] == "shown" else pinyin_gated).append(
                        {"fixture": fx["name"], "src": r["src"], "tgt": r["tgt"],
                         "status": r["status"], "reason": r["reason"]})
                    break

        for r in shown:
            flags = []
            left = tr.carryover_words(r["tgt"])
            if left:
                flags.append(f"미번역잔류:{','.join(left[:3])}")
            han = sum(1 for ch in r["tgt"] if "가" <= ch <= "힯")
            letters = sum(1 for ch in r["tgt"] if ch.isalpha())
            if letters and han / letters < 0.5:
                flags.append(f"한글비율 {han}/{letters}")
            if len(norm(r["src"])) >= 12:
                ratio = len(norm(r["tgt"])) / max(1, len(norm(r["src"])))
                if ratio < 0.25 or ratio > 3.0:
                    flags.append(f"길이비 {ratio:.2f}")
            for bad in fx.get("forbidden", []):
                if sim(norm(bad), norm(r["src"])) >= 0.6:
                    flags.append("병음을 문장으로 번역(PY-1 누수)")
                    break
            if flags:
                proxy_hits.append({"fixture": fx["name"], "src": r["src"],
                                   "tgt": r["tgt"], "flags": flags})
            sample_lines.append(
                f"- [{fx['name']}] `{norm(r['src'])[:90]}`\n"
                f"  → `{norm(r['tgt'])[:90]}`"
                + (f"  · proxy: {'; '.join(flags)}" if flags else ""))

        normal_total += f_norm
        normal_hidden += f_hidden
        normal_lost += f_lost
        recall = (f_norm / len(fx["paragraphs"])) if fx.get("expect_display", True) else None
        if recall is not None and recall < 0.5:
            unread.append({"fixture": fx["name"], "recall": round(recall, 2),
                           "truth_paragraphs": len(fx["paragraphs"])})
        per_fixture.append({"fixture": fx["name"], "normal_lines": f_norm,
                            "truth_paragraphs": len(fx["paragraphs"]),
                            "truth_recall": None if recall is None else round(recall, 2),
                            "hidden_by_gate": f_hidden, "no_output": f_lost,
                            "shown": len(shown), "ocr_lines": len(res["rows"]),
                            "group_errors": res["group_errors"]})

    # 숨겨진 정상 문장도 표본에 남긴다 — "왜 자막이 없지"의 유일한 단서.
    sample_lines += ["", "## 게이트가 숨긴 정상 문장", ""]
    for fx in fixtures():
        for r in results[fx["name"]]["rows"]:
            if r["status"] == "hidden" and r.get("truth_sim", 0) >= NORMAL_SIM:
                sample_lines.append(
                    f"- [{fx['name']}] ({r['reason']}) `{norm(r['src'])[:90]}`"
                    f"\n  → `{norm(r['tgt'])[:90]}`")
    samples_path.write_text("\n".join(sample_lines), encoding="utf-8")

    fp_rate = (normal_hidden / normal_total) if normal_total else None
    exact_critical = len(ko_misdisplay) + len(pinyin_leak)
    ok = (fp_rate is not None and fp_rate <= 0.01) and exact_critical == 0
    return {
        "pass": bool(ok),
        "measured": (f"오탐(게이트가 숨긴 정상문장) {normal_hidden}/{normal_total}"
                     f" = {fp_rate*100:.1f}%" if normal_total else "정상문장 0개(측정 불가)")
                    + f" · 확정 오역 {exact_critical}건"
                    + f" · 대리지표 의심 {len(proxy_hits)}건(사람 확인)"
                    + (f" · 병음 오번역이 게이트에 걸려 가려짐 {len(pinyin_gated)}건"
                       if pinyin_gated else "")
                    + (f" · 인식 실패 화면 {len(unread)}개" if unread else ""),
        "target": "오탐 ≤1% + 치명적 오역 0건",
        "detail": {"false_positive_rate": fp_rate,
                   "normal_lines": normal_total, "hidden_by_gate": normal_hidden,
                   "hidden_reasons": dict(hide_reasons),
                   "normal_lines_lost_without_gate": normal_lost,
                   "ko_screen_misdisplay": ko_misdisplay,
                   "pinyin_leak": pinyin_leak,
                   "pinyin_translated_but_gated": pinyin_gated,
                   "unreadable_fixtures": unread,
                   "proxy_suspects": proxy_hits,
                   "per_fixture": per_fixture,
                   "proxy_note": "치명적 오역은 자동 판정 불가 — 위 proxy 는 대리 지표이고 "
                                 "표본은 bench/last_samples.md 에 있다."}}


# --- 기준 3: 문단 경계 --------------------------------------------------------
def boundary_errors(truth_paras, ocr_texts):
    """잘못 합침/쪼갬 건수. 정답 문단에만 있는 '판별 토큰'의 분포로 센다."""
    owner = defaultdict(set)
    for i, p in enumerate(truth_paras):
        for t in set(tokens(p)):
            owner[t].add(i)
    disc = {t: next(iter(s)) for t, s in owner.items() if len(s) == 1}
    disc_count = Counter(disc.values())

    merges, hit = 0, defaultdict(set)
    for k, text in enumerate(ocr_texts):
        c = Counter(disc[t] for t in tokens(text) if t in disc)
        present = [i for i, n in c.items()
                   if n >= 2 or (disc_count[i] <= 3 and n >= 1)]
        if len(present) > 1:
            merges += len(present) - 1
        for i in present:
            hit[i].add(k)
    splits = sum(max(0, len(v) - 1) for v in hit.values())
    matched = len(hit)
    # 정답 문단의 절반도 못 읽었으면 "경계 오류 0건"은 거짓말이다 — 잴 게 없었을 뿐.
    unreadable = matched < len(truth_paras) * 0.5
    return {"merge_errors": merges, "split_errors": splits,
            "total": merges + splits, "truth_paragraphs": len(truth_paras),
            "matched_truth_paragraphs": matched, "ocr_paragraphs": len(ocr_texts),
            "unreadable": unreadable}


def criterion3(ctrl, results):
    detail = []
    for fx in fixtures():
        if fx["name"].startswith(("size_", "screen_")) or not fx.get("expect_display", True):
            continue
        texts = [r["src"] for r in results[fx["name"]]["rows"]]
        d = boundary_errors(fx["paragraphs"], texts)
        d["fixture"] = fx["name"]
        d["graded"] = fx["name"] in ("multi_context", "doc_en_column")
        detail.append(d)
    graded = [d for d in detail if d["graded"]]
    worst = max((d["total"] for d in graded), default=99)
    ok = bool(graded) and worst <= 1 and not any(d["unreadable"] for d in graded)
    return {"pass": ok,
            "measured": " · ".join(
                f"{d['fixture']} " + ("인식실패로 측정불가" if d["unreadable"] else
                                      f"{d['total']}건(합침{d['merge_errors']}/쪼갬{d['split_errors']})")
                for d in detail),
            "target": "화면당 ≤ 1건", "detail": detail}


# --- 기준 4: 속도 = "화면 변화 -> 자막" --------------------------------------
# 사용자가 정한 문장이 원래 이것이다. 예전 채점은 **한 사이클 시간**(캡처+OCR+번역)만
# 재서, 워커 폴 간격과 "앞 사이클이 끝나기를 기다린 시간"이 통째로 빠져 있었다.
# 실측(2026-09-07): 그 둘이 실제 지연의 25~45%였고, 그래서 채점은 통과인데 사용자는
# 2~3초를 느꼈다. 이제 **실제 앱을 띄워** 화면의 글자를 바꾼 시각 t0 과 그 번역이
# 오버레이에 그려진 시각 t1 을 잰다 (오버레이는 WDA_EXCLUDEFROMCAPTURE 라 스크린샷으로는
# 안 잡힌다 -> paintEvent 훅이 유일한 방법).
LAT_PROBE_MARK = "===LATENCY-PROBE-JSON==="
LAT_STATIC_PAGES = 3        # 문서 한 장을 한 번 띄우는 경우
LAT_INTERVAL_S = 2.0        # 자막이 바뀌는 주기 (사용자 제보: 자막/채팅)
LAT_MATCH_RATIO = 0.55      # OCR 오독을 감안한 "이 자막이 그 문장인가" 판정

# 캐시가 절대 안 듣게(계속 바뀌는 글의 실제 조건) 전부 다른 문장이다.
LAT_LINES = [
    "The harbor lights flickered as the ferry pulled away from the pier.",
    "She had never seen the mountain look so quiet in the morning.",
    "He counted the coins twice before handing them to the driver.",
    "Nobody expected the letter to arrive on a Sunday afternoon.",
    "The old radio hummed a tune that everyone in the room knew.",
    "They walked along the river until the streetlamps came on.",
    "A small dog followed them for three blocks without barking.",
    "The bakery on the corner closes earlier than it used to.",
    "Rain started falling right after the last train had left.",
    "His notebook was full of numbers that meant nothing to her.",
    "The garden behind the school had grown wild over the summer.",
    "She promised to call as soon as the meeting was finished.",
    "Every window on the second floor was painted a pale green.",
    "The map showed a road that no longer existed on the ground.",
    "He forgot the name of the town but remembered the bridge.",
    "Two children argued about who had found the shell first.",
    "The clock in the hallway was always eleven minutes fast.",
    "A stranger asked for directions to a street she invented.",
    "Snow covered the field before anyone could finish the work.",
    "The engine coughed once and then ran perfectly for hours.",
    "They served the soup in bowls that did not match at all.",
    "Her brother had painted the fence a color nobody liked.",
    "The library kept one copy of the book behind the counter.",
    "He left the keys on the table and closed the door softly.",
]


def _latency_probe(changes):
    """별도 프로세스 전용. 실제 앱 + 실제 화면으로 "화면 변화 -> 자막"을 잰다."""
    import difflib

    from PySide6.QtCore import QRect, Qt
    from PySide6.QtGui import QColor, QFont, QPainter
    from PySide6.QtWidgets import QApplication, QWidget

    import cocktail_capture as capmod
    import cocktail_engine as engmod
    import cocktail_translate as trmod
    import cocktail_ui as uimod

    trmod.PERSIST_CACHE._enabled = False   # 사용자 캐시를 읽지도 더럽히지도 않는다

    ev = {"stimuli": [], "hits": {}, "phase": []}

    _ocr = engmod.BackgroundController.perform_ocr_with_boxes

    def ocr_hook(self, image):
        t = time.perf_counter()
        try:
            return _ocr(self, image)
        finally:
            ev["phase"].append(["ocr", round((time.perf_counter() - t) * 1000, 1),
                                int(image.shape[0])])
    engmod.BackgroundController.perform_ocr_with_boxes = ocr_hook

    _bt = engmod.batch_translate

    def bt_hook(*a, **kw):
        t = time.perf_counter()
        try:
            return _bt(*a, **kw)
        finally:
            ev["phase"].append(["translate",
                                round((time.perf_counter() - t) * 1000, 1), 0])
    engmod.batch_translate = bt_hook

    def _n(s):
        return " ".join(str(s or "").lower().split())

    _paint = uimod.OverlayWindow.paintEvent

    def paint_hook(self, event):
        _paint(self, event)
        t = time.perf_counter()
        srcs = [s for s, _t, _b, _f in self.control.controller.translated_text]
        for st in ev["stimuli"]:
            if str(st["i"]) in ev["hits"] or t < st["t0"]:
                continue
            if any(difflib.SequenceMatcher(None, _n(s), _n(st["text"])).ratio()
                   >= LAT_MATCH_RATIO for s in srcs):
                ev["hits"][str(st["i"])] = t
    uimod.OverlayWindow.paintEvent = paint_hook

    class Stim(QWidget):
        """캡처 대상이 될 실제 창. 캡처 제외를 걸지 않는다(걸면 OCR이 못 본다)."""

        def __init__(self, rect):
            super().__init__()
            self.setWindowTitle("Bench Text Panel")
            self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint
                                | Qt.FramelessWindowHint)
            self.setGeometry(*rect)
            self.body, self.subtitle = [], ""
            self.show()

        def paintEvent(self, _e):
            p = QPainter(self)
            p.fillRect(self.rect(), QColor(250, 250, 248))
            p.setPen(QColor(20, 20, 20))
            f = QFont("Segoe UI")
            f.setPixelSize(18)
            p.setFont(f)
            y = 60
            for line in self.body:
                p.drawText(40, y, line)
                y += 32
            if self.subtitle:
                fs = QFont("Segoe UI")
                fs.setPixelSize(24)
                p.setFont(fs)
                p.drawText(QRect(40, self.height() - 120, self.width() - 80, 96),
                           Qt.AlignLeft | Qt.TextWordWrap, self.subtitle)

    qapp = QApplication.instance() or QApplication([sys.argv[0]])
    win = uimod.SettingsWindow()
    overlay = uimod.OverlayWindow(win)
    win.overlay = overlay
    overlay.update_for_mode()
    win.hide()

    g = qapp.primaryScreen().geometry()
    rect = (0, 0, min(1920, g.width()), min(1080, g.height()))
    stim = Stim(rect)
    win.region.capture_mode = capmod.MODE_BOX
    win.region.red_rect = QRect(*rect)
    win.controller.reset_frame_hash()

    def pump(sec):
        end = time.perf_counter() + sec
        while time.perf_counter() < end:
            qapp.processEvents()
            time.sleep(0.004)

    t_wait = time.perf_counter()
    while not win.controller.is_model_ok() and time.perf_counter() - t_wait < 240:
        qapp.processEvents()
        time.sleep(0.05)
    if not win.controller.is_model_ok():
        print(LAT_PROBE_MARK + json.dumps({"error": "모델 준비 실패"}))
        return 1
    win.controller.set_running(True)

    def shot(i, text, scenario):
        stim.repaint()               # 동기 페인트 = 글자가 화면에 올라간 시각
        qapp.processEvents()
        ev["stimuli"].append({"i": i, "text": text, "t0": time.perf_counter(),
                              "scenario": scenario})

    # (1) 계속 바뀌는 글 - 자막/채팅. 배경은 그대로 두고 아래 한 줄만 새 문장으로 교체.
    stim.body = ["Transcript view", "Speaker: narrator", "Session 4 of 9"]
    stim.subtitle = ""
    stim.repaint()
    pump(2.0)
    win.controller.reset_frame_hash()
    pump(1.0)
    for i in range(changes):
        stim.subtitle = LAT_LINES[i % len(LAT_LINES)]
        shot(i, stim.subtitle, "changing")
        pump(LAT_INTERVAL_S)
    pump(3.0)

    # (2) 채팅 - 새 줄이 아래에 붙고 목록 전체가 위로 밀린다.
    # 자막(배경 고정, 한 줄만 교체)과 달리 **화면 전체가 변해** 변경 밴드(PF-2)가 무효가
    # 되고 매 사이클 전체 OCR이 돈다. 사용자가 지목한 두 용도 중 무거운 쪽이라 반드시 잰다.
    seed = [f"user{j % 3 + 1}: An earlier note number {j} kept in this conversation log."
            for j in range(14)]
    stim.subtitle = ""
    stim.body = list(seed)
    stim.repaint()
    pump(2.0)
    win.controller.reset_frame_hash()
    pump(1.0)
    for i in range(changes):
        line = f"user{i % 3 + 1}: " + LAT_LINES[(i + 7) % len(LAT_LINES)]
        stim.body = (stim.body + [line])[-14:]
        shot(500 + i, line, "chat")
        pump(LAT_INTERVAL_S)
    pump(4.0)

    # (3) 문서 한 장을 한 번 띄우는 경우.
    for k in range(LAT_STATIC_PAGES):
        stim.body, stim.subtitle = [], ""
        stim.repaint()
        pump(1.2)
        win.controller.reset_frame_hash()
        pump(0.8)
        page = [f"{k+1}.{j+1} " + LAT_LINES[(k * 6 + j) % len(LAT_LINES)]
                for j in range(6)]
        stim.body = page
        shot(1000 + k, page[0], "static")
        pump(5.0)

    win.controller.stop()
    win._is_quitting = True
    for w in (overlay, stim, win):
        try:
            w.close()
        except Exception:
            pass
    t_zero = min(s["t0"] for s in ev["stimuli"])
    out = {"rows": [{"i": s["i"], "scenario": s["scenario"],
                     "latency_ms": (None if str(s["i"]) not in ev["hits"] else
                                    round((ev["hits"][str(s["i"])] - s["t0"]) * 1000, 1))}
                    for s in ev["stimuli"]],
           "phase": ev["phase"], "wall_s": round(time.perf_counter() - t_zero, 1)}
    print(LAT_PROBE_MARK + json.dumps(out, ensure_ascii=False))
    return 0


def criterion4(changes, env):
    p = run_py([sys.executable, str(Path(__file__)), "--_latency_probe",
                "--reps", str(changes)], timeout=900)
    line = next((l for l in p.stdout.splitlines() if l.startswith(LAT_PROBE_MARK)), None)
    if line is None:
        return {"pass": False, "measured": "측정 실패(프로브가 결과를 못 냄)",
                "target": "p50 <1.5s, p95 <3s",
                "detail": {"stdout_tail": (p.stdout[-1200:] + p.stderr[-800:]).strip()}}
    data = json.loads(line[len(LAT_PROBE_MARK):])
    if "error" in data:
        return {"pass": False, "measured": data["error"],
                "target": "p50 <1.5s, p95 <3s", "detail": data}

    rows = data["rows"]

    def stats(sel):
        got = sorted(r["latency_ms"] for r in rows
                     if r["latency_ms"] is not None and sel(r))
        n_all = sum(1 for r in rows if sel(r))
        if not got:
            return {"n": 0, "missed": n_all, "p50_ms": None, "p95_ms": None}
        return {"n": len(got), "missed": n_all - len(got),
                "p50_ms": round(statistics.median(got), 1),
                "p95_ms": round(got[min(len(got) - 1, int(len(got) * 0.95))], 1),
                "min_ms": got[0], "max_ms": got[-1]}

    changing = stats(lambda r: r["scenario"] == "changing")
    chat = stats(lambda r: r["scenario"] == "chat")
    static = stats(lambda r: r["scenario"] == "static")
    pooled = stats(lambda _r: True)
    ocr = [x[1] for x in data["phase"] if x[0] == "ocr"]
    trs = [x[1] for x in data["phase"] if x[0] == "translate"]
    # 자막이 아예 안 뜬 것(missed)은 통과를 막는다 - 덜 띄워서 빨라지는 건 개선이 아니다.
    ok = (pooled["n"] > 0 and pooled["missed"] == 0
          and pooled["p50_ms"] < 1500 and pooled["p95_ms"] < 3000)
    parts = []
    if pooled["n"]:
        parts.append(f"p50 {pooled['p50_ms']/1000:.2f}s / p95 {pooled['p95_ms']/1000:.2f}s")
    else:
        parts.append("샘플 없음")
    if changing["n"]:
        parts.append(f"자막 p50 {changing['p50_ms']/1000:.2f}/"
                     f"p95 {changing['p95_ms']/1000:.2f}")
    if chat["n"]:
        parts.append(f"채팅 p50 {chat['p50_ms']/1000:.2f}/"
                     f"p95 {chat['p95_ms']/1000:.2f}")
    if static["n"]:
        parts.append(f"정적 p50 {static['p50_ms']/1000:.2f}/"
                     f"p95 {static['p95_ms']/1000:.2f}")
    parts.append(f"누락 {pooled['missed']}")
    parts.append("오염=" + ("예" if env["contaminated"] else "아니오"))
    return {
        "pass": bool(ok),
        "measured": " · ".join(parts),
        "target": "p50 <1.5s, p95 <3s",
        "detail": {"changing": changing, "chat": chat, "static": static,
                   "pooled": pooled,
                   "rows": rows, "env": env, "probe_wall_s": data["wall_s"],
                   "ocr_ms_p50": round(statistics.median(ocr), 1) if ocr else None,
                   "translate_ms_p50": round(statistics.median(trs), 1) if trs else None,
                   "note": "실제 앱을 띄워 t0(화면 글자 교체)~t1(오버레이 paintEvent에 그 "
                           "번역이 그려짐)을 잰다. 폴 간격 · 앞 사이클 대기 · 캡처 · OCR · "
                           "번역 · 그리기가 전부 들어간다. 자막이 안 뜬 건 missed 로 세고 "
                           "통과를 막는다."}}


# --- 기준 5: UI (부분 자동) ---------------------------------------------------
UI_PROBE_MARK = "===UI-PROBE-JSON==="


def _ui_probe(font_dpi=None):
    """별도 프로세스에서만 돈다. 설정창을 실제로 띄워 위젯이 창 밖으로 나가는지 본다."""
    from PySide6.QtCore import QRect
    from PySide6.QtWidgets import QApplication, QWidget

    import cocktail_ui
    # 모델 로드/워커 스레드/화면 캡처 금지 — UI 지오메트리만 본다.
    cocktail_ui.BackgroundController.start = lambda self: None

    qapp = QApplication.instance() or QApplication([sys.argv[0]])
    out = {"font_dpi": font_dpi, "sizes": [], "shots": []}
    win = cocktail_ui.SettingsWindow()
    # 첫 실행 안내: 저장된 설정을 건드리지 않고 그 분기만 다시 태운다
    # (QSettings 는 Windows 레지스트리라 격리가 안 된다 — 사용자 설정을 쓰지 않는 쪽을 택했다).
    win._first_run_settings = True
    win._maybe_show_initial_settings()
    qapp.processEvents()
    out["first_run_shows_settings"] = bool(win.isVisible())
    win.show()          # 지오메트리 검사는 창이 실제로 떠 있어야 의미가 있다
    qapp.processEvents()
    out["tray_actions"] = len(win.tray.contextMenu().actions()) if getattr(win, "tray", None) else 0
    from PySide6.QtGui import QShortcut
    out["shortcuts"] = sorted(s.key().toString() for s in win.findChildren(QShortcut))
    out["global_hotkeys"] = bool(getattr(cocktail_ui, "HAS_KEYBOARD", False))

    shots = BENCH / "last_ui"
    shots.mkdir(parents=True, exist_ok=True)
    for w, h in [(880, 120), (640, 400), (1400, 900), (480, 220)]:
        win.resize(w, h)
        qapp.processEvents()
        actual = win.size()
        rect = QRect(0, 0, actual.width(), actual.height())
        over, clipped = [], []
        for child in win.findChildren(QWidget):
            if not child.isVisible() or child.findChildren(QWidget):
                continue   # 컨테이너는 건너뛰고 말단 위젯만 본다
            top_left = child.mapTo(win, child.rect().topLeft())
            r = QRect(top_left, child.size())
            text = (child.text() if hasattr(child, "text") else
                    child.currentText() if hasattr(child, "currentText") else "")
            label = str(text or child.objectName() or child.metaObject().className())
            if not rect.contains(r):
                over.append({"widget": child.metaObject().className(),
                             "label": label[:40],
                             "rect": [r.x(), r.y(), r.width(), r.height()]})
            # 창 안에 있어도 글자가 위젯보다 넓으면 잘려 보인다(고DPI/큰 폰트에서 발생).
            if text:
                need = child.fontMetrics().horizontalAdvance(str(text)) + 12
                if need > child.width():
                    clipped.append({"widget": child.metaObject().className(),
                                    "label": label[:40], "need_px": need,
                                    "have_px": child.width()})
        tag = f"settings_{w}x{h}" + (f"_dpi{font_dpi}" if font_dpi else "")
        png = shots / f"{tag}.png"
        win.grab().save(str(png))
        out["sizes"].append({"requested": [w, h],
                             "actual": [actual.width(), actual.height()],
                             "overflow": over, "clipped_text": clipped,
                             "shot": png.name})
        out["shots"].append(png.name)

    # 오버레이 자막 렌더 (WDA_EXCLUDEFROMCAPTURE 때문에 스크린샷엔 안 잡힌다 — grab 필수)
    try:
        overlay = cocktail_ui.OverlayWindow(win)
        g = overlay.geometry()
        win.controller.translated_text = [
            ("The kettle on the far shelf had been whistling.",
             "저 멀리 선반 위의 주전자가 한참 전부터 끓고 있었다.",
             (g.x() + 60, g.y() + 60, g.x() + 520, g.y() + 84), 16),
            ("Storage almost full", "저장 공간이 거의 찼습니다",
             (g.x() + 60, g.y() + 140, g.x() + 320, g.y() + 164), 18),
        ]
        overlay.update()
        qapp.processEvents()
        png = shots / "overlay_subtitles.png"
        overlay.grab(QRect(0, 0, 700, 260)).save(str(png))
        out["shots"].append(png.name)
        overlay.close()
    except Exception as e:
        out["overlay_error"] = f"{type(e).__name__}: {e}"

    win._is_quitting = True
    win.close()
    print(UI_PROBE_MARK + json.dumps(out, ensure_ascii=False))
    return 0


def criterion5():
    runs = {}
    for tag, extra_env in (("100%", {}), ("고DPI(font dpi 168)", {"QT_FONT_DPI": "168"})):
        cmd = [sys.executable, str(Path(__file__)), "--_ui_probe"]
        if extra_env:
            cmd += ["--font-dpi", "168"]
        p = run_py(cmd, env=extra_env, timeout=180)
        line = next((l for l in p.stdout.splitlines() if l.startswith(UI_PROBE_MARK)), None)
        if line is None:
            runs[tag] = {"error": (p.stdout[-800:] + p.stderr[-800:]).strip()}
        else:
            runs[tag] = json.loads(line[len(UI_PROBE_MARK):])
    overflow = sum(len(s["overflow"]) for r in runs.values()
                   for s in r.get("sizes", []))
    clipped = sum(len(s.get("clipped_text", [])) for r in runs.values()
                  for s in r.get("sizes", []))
    ok_runs = [r for r in runs.values() if "error" not in r]
    tray = max((r.get("tray_actions", 0) for r in ok_runs), default=0)
    shortcuts = max((len(r.get("shortcuts", [])) for r in ok_runs), default=0)
    first_run = any(r.get("first_run_shows_settings") for r in ok_runs)
    auto_pass = (bool(ok_runs) and overflow == 0 and clipped == 0
                 and tray >= 2 and shortcuts >= 2 and first_run)
    return {"pass": auto_pass,
            "measured": f"창 밖 위젯 {overflow}개 · 글자 잘림 {clipped}개 · 트레이 메뉴 {tray}개 · "
                        f"단축키 {shortcuts}개 · 첫 실행 설정창 {'표시' if first_run else '미표시'}",
            "target": "이탈 0 + 트레이/단축키/첫 실행 안내 존재",
            "detail": runs,
            "manual": [
                "설정창을 마우스로 실제 드래그 리사이즈했을 때 글자 겹침이 없는가",
                "175% 보조 모니터로 창을 끌고 갔을 때 자막 위치가 맞는가 (앱은 DPI unaware)",
                "첫 실행 안내가 '무엇을 눌러야 하는지'까지 알려주는가 (창만 뜨는 것과 다름)",
                "트레이 아이콘이 실제로 읽히는가 (IC-1: 코드로 그린 'A→가', 16px에서 확인)",
                "단축키가 다른 앱과 충돌하지 않는가 (Ctrl+Shift+H / P)",
            ]}


# --- 기준 6: 접근성 (부분 자동) -----------------------------------------------
# LZ-1: 사람이 기다리는 건 **창이 뜨는 시간**이지 import 시간이 아니다. import 시간은
# 그 대리 지표일 뿐이고, 무거운 것을 백그라운드 스레드로 옮기면 둘이 갈라진다
# (실측 2026-09-05: import 1.05s인데 창은 1.25s — Qt 창 생성/트레이 등록이 그 차이).
# 그래서 채점은 프로세스 밖에서 실제 창을 기다려 잰다. import 시간은 참고로만 남긴다.
STARTUP_WINDOW_S = 2.0   # 합격선: 프로세스 시작 → 설정 창 보임
WINDOW_TITLE = "Cocktail Settings"


def window_visible_seconds(cmd, timeout=120):
    """프로세스 시작 → 설정 창이 화면에 보일 때까지(초). 못 보면 None.

    앱 안에서 재면 '언제부터'를 못 잡는다 — Popen 직전을 t0으로 삼는 외부 측정이다.
    `COCKTAIL_SHOW_SETTINGS_ON_START=1` 은 첫 실행 여부와 무관하게 창을 띄우는 기존 오버라이드.
    """
    import ctypes
    user32 = ctypes.windll.user32
    env = dict(os.environ, COCKTAIL_SHOW_SETTINGS_ON_START="1", PYTHONUNBUFFERED="1")
    if user32.FindWindowW(None, WINDOW_TITLE):
        return None   # 이미 앱이 떠 있다 — 남의 창을 우리 것으로 오인하면 0초가 나온다
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, cwd=str(ROOT), env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        while time.perf_counter() - t0 < timeout:
            h = user32.FindWindowW(None, WINDOW_TITLE)
            if h and user32.IsWindowVisible(h):
                return round(time.perf_counter() - t0, 2)
            if p.poll() is not None:
                return None
            time.sleep(0.02)
        return None
    finally:
        p.terminate()
        try:
            p.wait(timeout=10)
        except Exception:
            p.kill()


def criterion6():
    detail = {}
    t0 = time.perf_counter()
    p = run_py([sys.executable, "-c",
                "import sys,time;t=time.perf_counter();import cocktail_ui;"
                "print(time.perf_counter()-t, 'torch' in sys.modules)"])
    if p.returncode == 0:
        raw = p.stdout.strip().splitlines()[-1].split()
        detail["import_s"] = round(float(raw[0]), 2)
        # LZ-1 회귀 가드: 창을 띄우는 경로가 torch를 끌고 오면 시작이 다시 10초가 된다.
        detail["torch_imported_at_ui_import"] = raw[1] == "True"
    else:
        detail["import_s"] = None
        detail["torch_imported_at_ui_import"] = None
    detail["import_process_s"] = round(time.perf_counter() - t0, 2)

    dist = ROOT / "dist"
    exe = next(dist.rglob("Cocktail.exe"), None) if dist.exists() else None
    detail["dist_exists"] = dist.exists()

    # 자가진단은 배포물이 있으면 **배포물로** 돌린다 — "파이썬 불필요"를 실제로 재는 유일한 방법.
    st_cmd = [str(exe), "--self-test"] if exe else [sys.executable, "cocktail.py", "--self-test"]
    detail["self_test_target"] = "packaged exe" if exe else "source"
    t0 = time.perf_counter()
    p = subprocess.run(st_cmd, cwd=str(ROOT), capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=600)
    detail["self_test_s"] = round(time.perf_counter() - t0, 2)
    detail["self_test_ok"] = p.returncode == 0
    if not detail["self_test_ok"]:
        detail["self_test_tail"] = (p.stdout + p.stderr)[-600:]

    if exe:
        newest_src = max(f.stat().st_mtime for f in ROOT.glob("cocktail*.py"))
        detail["exe"] = str(exe.relative_to(ROOT))
        detail["exe_mtime"] = time.strftime("%Y-%m-%d", time.localtime(exe.stat().st_mtime))
        detail["exe_stale_vs_source"] = exe.stat().st_mtime < newest_src
        detail["dist_size_mb"] = round(
            sum(f.stat().st_size for f in dist.rglob("*") if f.is_file()) / 1e6)
        # PK-1/PK-2: 별도 설치를 요구하는 것이 하나라도 있으면 "준비 5분"은 성립하지 않는다.
        base = exe.parent
        detail["bundled_tesseract"] = any(base.rglob("tesseract/tesseract.exe"))
        detail["bundled_tessdata"] = any(base.rglob("tessdata/eng.traineddata"))
        detail["bundled_model"] = any(base.rglob("models/*/config.json"))
        detail["exe_has_icon"] = _exe_has_icon(exe)

    # 창 보임 시간 — 배포물이 있으면 배포물로(사용자가 실제로 실행할 물건).
    start_cmd = [str(exe)] if exe else [sys.executable, "cocktail.py"]
    detail["startup_target"] = "packaged exe" if exe else "source"
    detail["window_visible_s"] = window_visible_seconds(start_cmd)

    detail["installer_script"] = (ROOT / "installer.iss").exists()
    installer = next((ROOT / "Output").glob("*.exe"), None) if (ROOT / "Output").exists() else None
    detail["installer_exe"] = str(installer.relative_to(ROOT)) if installer else None
    if installer:
        detail["installer_size_mb"] = round(installer.stat().st_size / 1e6)

    win_s = detail.get("window_visible_s")
    ok = (bool(exe)
          and not detail.get("exe_stale_vs_source", True)
          and detail["self_test_ok"]
          and detail.get("bundled_tesseract", False)
          and detail.get("bundled_tessdata", False)
          and detail.get("bundled_model", False)
          and detail.get("exe_has_icon", False)
          and detail.get("torch_imported_at_ui_import") is False
          and win_s is not None and win_s <= STARTUP_WINDOW_S)
    fresh = ("없음" if not exe
             else f"구버전({detail['exe_mtime']})" if detail["exe_stale_vs_source"]
             else f"최신({detail['exe_mtime']}, {detail.get('dist_size_mb')}MB)")
    return {"pass": ok,
            "measured": f"창 보임 {win_s if win_s is not None else '측정실패'}s"
                        f"(≤{STARTUP_WINDOW_S}) · self-test {detail['self_test_s']}s"
                        f"[{detail['self_test_target']}] · 동봉 "
                        f"tesseract {'O' if detail.get('bundled_tesseract') else 'X'}/"
                        f"tessdata {'O' if detail.get('bundled_tessdata') else 'X'}/"
                        f"모델 {'O' if detail.get('bundled_model') else 'X'} · 배포물 {fresh}",
            "target": "설치파일 하나 · 파이썬 불필요 · 준비 5분",
            "detail": detail,
            "manual": [
                f"인스톨러 컴파일 — Inno Setup(ISCC.exe) 설치 후 `ISCC installer.iss` "
                f"(현재 Output 산출물: {detail['installer_exe'] or '없음'}). 설치는 사용자 승인 사항",
                "깨끗한 PC(파이썬·아나콘다 없음)에서 설치 파일 하나로 설치되는가",
                "설치 후 첫 실행에서 5분 안에 자막이 뜨는가 "
                "(en→ko 모델은 동봉이라 오프라인, 그 외 언어쌍은 첫 사용 시 다운로드)",
                "백신/SmartScreen 경고 없이 실행되는가 (코드 서명 인증서 필요 — 미보유)",
                "175% 보조 모니터·다중 모니터에서 설치본이 정상인가",
            ]}


def _exe_has_icon(exe: Path) -> bool:
    """IC-2: exe에 아이콘 리소스가 박혔는지. Win32 리소스를 직접 센다(외부 도구 없이).

    D-1과 같은 함정: **argtypes/restype를 반드시 명시**한다. HMODULE을 기본 c_int로
    받으면 64비트 핸들이 잘려 항상 실패한다(그래서 아이콘을 넣고도 false가 나왔다).
    """
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.LoadLibraryExW.argtypes = [wintypes.LPCWSTR, wintypes.HANDLE, wintypes.DWORD]
    k32.LoadLibraryExW.restype = wintypes.HMODULE
    # 리소스 타입/이름은 **정수 ID일 수 있다**(MAKEINTRESOURCE). LPCWSTR로 선언하면
    # ctypes가 그 정수를 포인터로 알고 역참조해서 프로세스가 죽는다 — 전부 c_void_p로 받는다.
    ENUMRESNAMEPROC = ctypes.WINFUNCTYPE(
        wintypes.BOOL, wintypes.HMODULE, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p)
    k32.EnumResourceNamesW.argtypes = [wintypes.HMODULE, ctypes.c_void_p,
                                       ENUMRESNAMEPROC, ctypes.c_void_p]
    k32.EnumResourceNamesW.restype = wintypes.BOOL
    k32.FreeLibrary.argtypes = [wintypes.HMODULE]

    h = k32.LoadLibraryExW(str(exe), None, 0x00000002)   # LOAD_LIBRARY_AS_DATAFILE
    if not h:
        return False
    found = []
    cb = ENUMRESNAMEPROC(lambda hm, t, name, param: (found.append(name), True)[1])
    try:
        k32.EnumResourceNamesW(h, 14, cb, None)   # RT_GROUP_ICON
    finally:
        k32.FreeLibrary(h)
    return bool(found)


# --- 실행/보고 ----------------------------------------------------------------
CRITERIA = {
    1: "작은 글씨 (문자 오류율)",
    2: "오번역 없음 (오탐/치명 오역)",
    3: "다중 문맥 (문단 경계)",
    4: "속도 (화면변화→자막)",
    5: "UI (리사이즈·고DPI·트레이)",
    6: "접근성 (설치·준비 시간)",
}


def main():
    ap = argparse.ArgumentParser(description="Cocktail 합격선 자동 채점판")
    ap.add_argument("--only", default="", help="예: --only 1,4")
    ap.add_argument("--reps", type=int, default=10,
                    help="기준 4: 자막/채팅이 각각 바뀌는 횟수 (기본 10)")
    ap.add_argument("--_ui_probe", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--_latency_probe", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--font-dpi", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if getattr(args, "_ui_probe"):
        return _ui_probe(args.font_dpi)
    if getattr(args, "_latency_probe"):
        return _latency_probe(args.reps)

    want = ({int(x) for x in args.only.replace(" ", "").split(",") if x}
            if args.only else set(CRITERIA))
    started = time.perf_counter()
    report = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "criteria": {}}

    # 기준 4는 이제 **별도 프로세스에서 실제 앱을 띄워** 재므로 in-process 파이프라인이
    # 필요 없다 (`--only 4` 가 모델을 두 번 올리던 낭비도 사라진다).
    report["env"] = (env_probe() if 4 in want
                     else {"contaminated": False, "notes": []})
    if report["env"].get("contaminated"):
        print("[경고] 측정 환경 오염: " + "; ".join(report["env"]["notes"]))
        print("       속도 수치는 이 상태에서 낸 것이다. 판정에 그대로 쓰지 마라.")

    needs_pipeline = bool(want & {1, 2, 3})
    results = {}
    ctrl = None
    if needs_pipeline:
        ctrl = make_ctrl()
        print(f"[bench] 모델 준비...")
        report["model_warmup_s"] = round(prewarm(ctrl), 1)
        print(f"[bench] 모델 준비 {report['model_warmup_s']}s")
        for fx in fixtures():
            results[fx["name"]] = analyze(ctrl, load_image(fx["image"]))

    samples = BENCH / "last_samples.md"
    if 1 in want:
        report["criteria"]["1"] = criterion1(ctrl, results)
    if 2 in want:
        report["criteria"]["2"] = criterion2(ctrl, results, samples)
    if 3 in want:
        report["criteria"]["3"] = criterion3(ctrl, results)
    if 4 in want:
        report["criteria"]["4"] = criterion4(args.reps, report["env"])
    if 5 in want:
        report["criteria"]["5"] = criterion5()
    if 6 in want:
        report["criteria"]["6"] = criterion6()

    report["elapsed_s"] = round(time.perf_counter() - started, 1)
    (BENCH / "last_score.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 100)
    print(f"{'#':<3}{'기준':<26}{'판정':<8}{'측정값':<44}목표")
    print("-" * 100)
    for n in sorted(int(k) for k in report["criteria"]):
        c = report["criteria"][str(n)]
        mark = "통과" if c["pass"] else "실패"
        print(f"{n:<3}{CRITERIA[n]:<26}{mark:<8}{c['measured'][:42]:<44}{c['target']}")
        if len(c["measured"]) > 42:
            print(f"{'':<37}{c['measured'][42:]}")
    print("=" * 100)
    for n in sorted(int(k) for k in report["criteria"]):
        manual = report["criteria"][str(n)].get("manual")
        if manual:
            print(f"\n[기준 {n}] 자동으로 못 재는 것 — 사람이 확인:")
            for item in manual:
                print(f"  [ ] {item}")
    if report.get("env", {}).get("contaminated"):
        print("\n[경고] 이번 측정은 환경이 오염된 상태였다: "
              + "; ".join(report["env"]["notes"]))
    print(f"\n소요 {report['elapsed_s']}s · 원자료 {BENCH/'last_score.json'}")
    if 2 in want:
        print(f"기준 2 표본 {samples}")
    return 0 if all(c["pass"] for c in report["criteria"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())

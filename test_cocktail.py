"""Cocktail 실동작 테스트.

구 `smoke_tests.py`는 앱 소스를 AST로 파싱해 "특정 문자열이 들어 있는가"를 검사했다.
243개 assert가 전부 코드 지문 검사라, M-8(번역이 통째로 깨진 버그)을 하나도 못 잡으면서
변수 이름만 바꿔도 실패했다. 여기서는 **실제로 함수를 호출해 결과를 본다.**

실행:
    python test_cocktail.py            # 빠른 테스트만 (모델 로드 없음, 수 초)
    python test_cocktail.py --full     # 번역 모델까지 실제 로드해 검증 (수십 초, 캐시 필요)

프레임워크 없음 — assert와 함수 이름 규약(test_*)만 쓴다.
"""
import argparse
import os
import sys
import unicodedata

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# EN-1: 한국어 콘솔(cp949)에서 `—` 하나 때문에 테스트가 UnicodeEncodeError로 죽었다.
# 앱은 cocktail.py 에서 같은 처리를 한다. 테스트는 그 엔트리를 안 거치므로 여기서도 건다.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass

# 계층별로 import 한다 — 어떤 모듈이 무엇을 소유하는지 테스트가 그대로 문서 역할을 한다.
import cocktail_capture as cap        # noqa: E402  캡처 영역 / 프레임 변화
import cocktail_engine as eng         # noqa: E402  워커 루프
import cocktail_ocr as ocr            # noqa: E402  OCR 백엔드
import cocktail_platform as plat      # noqa: E402  Windows API
import cocktail_translate as tr       # noqa: E402  번역 / 캐시 / 표시 게이트


# --- OCR 노이즈 필터 (NZ-1) -------------------------------------------------
def test_noise_filter_drops_ui_crumbs():
    assert ocr._ocr_line_is_noise("|||")           # 기호 잡음
    assert ocr._ocr_line_is_noise("x")             # 1글자 아이콘
    assert ocr._ocr_line_is_noise("10:30")         # 시계 (글자 없음)
    assert ocr._ocr_line_is_noise("100%")          # 퍼센트 표시
    assert not ocr._ocr_line_is_noise("Hello")
    assert not ocr._ocr_line_is_noise("안녕하세요")
    assert not ocr._ocr_line_is_noise("Save file")


# --- 언어 판정 (SL-1) --------------------------------------------------------
def test_script_detection_is_deterministic():
    assert tr.identify_language("안녕하세요") == "ko"
    assert tr.identify_language("こんにちは") == "ja"
    assert tr.identify_language("Привет") == "ru"
    # SL-1의 핵심: 라틴 문자는 추측하지 않고 전부 fallback(en)으로 보낸다.
    assert tr.identify_language("Bonjour tout le monde") == "en"
    assert tr.identify_language("Guten Morgen") == "en"
    assert tr.identify_language("12345") == "en"   # 글자 없음 → fallback


def test_assign_region_languages_respects_explicit_src():
    results = [("Hello", (0, 0, 10, 10), 12), ("안녕", (0, 20, 10, 30), 12)]
    assert tr.assign_region_languages(results, "fr") == ["fr", "fr"]
    assert tr.assign_region_languages(results, "auto") == ["en", "ko"]


# --- 병음 줄 제외 (PY-1) -----------------------------------------------------
def test_pinyin_lines_are_excluded_from_translation_input():
    """중국어 학습앱 화면의 병음(발음표기) 줄은 번역 입력에 들어가면 안 된다.

    실측 증상(2026-08-11): auto(eng+kor)가 한자를 한 글자도 못 읽고 병음 줄만 살려서
    en 으로 확정 → "xiǎoměi zhàn zài hébiān" 이 "작은 미는 강 옆에 서서"가 됐다.
    """
    for line in (
        "xiǎoměi zhàn zài hébiān, shǒu li ná zhe yīgè hóngsè de piáo.",   # 성조부호 그대로
        "dàjiā xiàozhe, pǎozhe, hùxiāng pōshuǐ.",
        "xidoméi zhan zai hébian, li na zhe yige hongse de piao.",        # OCR이 성조를 뭉갠 실측 출력
        "xia oméikankan sizhan ta zAixia ne",
        # PY-2: **띄어쓰기된** 병음. 구 규칙은 "7자 이상 토큰 하나"를 요구해서 여기서
        # 전부 False가 났고, 그 결과 병음 줄이 영어로 확정돼 번역됐다
        # ("ni hao, huan ying shi yong zhe ge ying yong" → "니 하오, 후안 이 쯔 옌").
        "nǐ hǎo, huān yíng shǐ yòng zhè ge yìng yòng",
        "ni hao, huan ying shi yong zhe ge ying yong",          # OCR이 성조를 날린 실측 출력
        "jin tian tian qi hen hao, wo men qu gong yuan ba",
        "zhe ben shu shi wo peng you song gei wo de li wu",
    ):
        assert ocr.is_pinyin_line(line), f"병음 줄을 못 걸렀다: {line!r}"


def test_pinyin_filter_does_not_eat_english():
    """거짓 양성 금지 — 영어/독일어/프랑스어 본문이 병음으로 오판되면 안 된다.

    실측 스윕(영어 6909줄): 비율 0.8 + 최장분해 7 에서 거짓양성 6줄(0.09%).
    아래 문장들은 그 스윕에서 병음 음절 분해율이 가장 높았던 부류다.
    """
    for line in (
        "My drawing was not a picture of a hat.",
        "be taken seriously. Then he added: \"So you, too, come",
        "give you a string, too, so that you can tie",
        "Please enter your password to continue.",
        "Über die Straße gehen wir zusammen.",        # 독일어 움라우트
        "Saint-Exupéry, his love of aviation inspired stories",   # 프랑스어 어큐트
        "Right click/trigger to throw Pudding",
        # PY-2 회귀: 띄어쓰기 병음 규칙(짧은 토큰 여러 개)이 본문을 먹으면 안 된다.
        # 아래는 코퍼스 스윕에서 병음 음절 분해율이 높게 나온 영/불/독 문장들이다.
        "Le café était fermé, alors nous sommes allés à la boulangerie voisine.",
        "Nous avons pris un cafe sur la terrasse avant de rentrer a la maison.",
        "Die Sitzung des Gemeinderats findet am naechsten Dienstag im Rathaus statt.",
        "Wir haben lange ueber die Sicherheitsprobleme des Netzwerks gesprochen.",
        "The council approved the plan on Tuesday after a short debate.",
        "Open the storage page to choose which folders keep a local copy.",
    ):
        assert not ocr.is_pinyin_line(line), f"영어 본문을 병음으로 오판: {line!r}"
    # 중국어 화면으로 확정된 뒤(완화 임계)에도 짧은 영어 UI 라벨은 살아야 한다.
    for word in ("data", "menu", "china", "panda", "media", "Settings"):
        assert not ocr.is_pinyin_line(word, zh_screen=True), f"영어 UI 라벨이 사라짐: {word!r}"
    # 완화 임계에서만 잡히는 실측 병음 줄 (성조부호가 OCR에서 다 뭉개진 경우)
    mangled = "dajia xiaozhe pa 02116 hixiangpdshui_"
    assert not ocr.is_pinyin_line(mangled), "엄격 모드가 너무 공격적이다"
    assert ocr.is_pinyin_line(mangled, zh_screen=True), "중국어 확정 화면에서 병음을 놓쳤다"


def test_drop_pinyin_lines_reports_hits_for_zh_detection():
    lines = [("xiǎoměi zhàn zài hébiān", (0, 0, 200, 20), 12),
             ("大家笑着，跑着，互相泼水。", (0, 30, 200, 70), 40),
             ("Hello there, my friend.", (0, 80, 200, 100), 14)]
    kept, confs, dropped = ocr.drop_pinyin_lines(lines, [80.0, 85.0, 90.0])
    assert dropped == 1, f"병음 1줄을 버려야 한다 (버린 수={dropped})"
    assert [t for t, _, _ in kept] == ["大家笑着，跑着，互相泼水。", "Hello there, my friend."]
    assert confs == [85.0, 90.0], "conf 병렬 리스트가 라인과 어긋났다"


def test_chinese_screen_is_read_and_result_is_reproducible():
    """ZH-2 + RP-1: 중국어 화면에서 한자가 실제로 읽히고, 같은 입력 5회가 완전히 같아야 한다.

    실측 결함(2026-09-05): `bench/fixtures/zh_pinyin.png` 의 한자 5문단 중 **0개**를
    읽었다. Tesseract 는 같은 이미지를 chi_sim 으로 완벽히 읽는데도 ZH-1 재OCR 진입
    조건(병음 판정 / 1차 패스 0줄)이 둘 다 성립하지 않아 chi_sim 패스가 영원히 안 돌았다.
    그리고 같은 입력이 실행마다 다르게 읽혔다(구 규칙 5회에 서로 다른 결과 5종).
    """
    fixture = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "bench", "fixtures", "zh_pinyin.png")
    if not os.path.exists(fixture) or ocr.OCR_ZH_TESS_LANG not in ocr.AVAILABLE_TESS_LANGS:
        return   # 시료나 chi_sim traineddata 가 없는 환경에서는 검사 의미 없음
    from PIL import Image
    image = np.array(Image.open(fixture).convert("RGB"))
    controller = _make_controller()
    runs = []
    for _ in range(5):
        controller._zh_frames = 0     # 화면이 새로 바뀐 상태 (프레임 간 sticky 오염 제거)
        controller._ko_frames = 0
        runs.append([(t, tuple(b), f)
                     for t, b, f in controller.perform_ocr_with_boxes(image)])
    texts = [t for t, _, _ in runs[0]]
    cjk = sum(ocr.cjk_char_count(t) for t in texts)
    assert cjk >= 40, f"중국어 화면인데 한자를 못 읽었다 (CJK {cjk}자): {texts}"
    assert not any(ocr.is_pinyin_line(t, zh_screen=True) for t in texts), \
        f"병음 줄이 번역 입력에 그대로 남았다: {texts}"
    assert all(r == runs[0] for r in runs[1:]), \
        f"같은 입력 5회가 다른 결과를 냈다 (RP-1 회귀): {[len(r) for r in runs]}"


def test_upscale_factor_ignores_wall_clock():
    """RP-1: 확대 배율이 1차 패스의 **실측 시간**에 좌우되면 안 된다.

    구 규칙은 `배율 = √(900ms / 1차시간)` 이라 PC 부하가 그대로 인식 결과로 샜다.
    실측(전 시료 5회, 2~5회차는 CPU 부하): 구 규칙은 서로 다른 결과 5종(ui 화면 줄
    수 20~22), 새 규칙(1차 패스 글자 수 예산)은 5회 전부 동일.
    """
    import time as _time
    controller = _make_controller()
    controller.options = cap.RuntimeOptions("auto", "English", ocr.OCR_BACKEND_TESSERACT)
    got = []

    def make_fake(delay):
        def fake(_gray, _lang, scale, conf_out=None, **_kw):
            _time.sleep(delay)
            got[-1].append(round(scale, 4))
            if conf_out is not None:
                conf_out.append(80.0)
            return [("a small caption the first pass already read", (0, 0, 300, 11), 11)]
        return fake

    for delay in (0.0, 0.4):
        got.append([])
        controller._tess_lines = make_fake(delay)
        controller._zh_frames = 0
        controller._ko_frames = 0
        controller.perform_ocr_with_boxes(np.zeros((1080, 1920, 3), dtype=np.uint8))
    assert len(got[0]) == 2, f"확대 재OCR이 발동하지 않았다: {got}"
    assert got[0] == got[1], f"1차 패스가 느려지자 확대 배율이 달라졌다: {got}"


# --- 한자 공백 (ZH-1) --------------------------------------------------------
def test_cjk_words_join_without_spaces():
    """Tesseract 는 한자를 글자마다 별개 word 로 준다. 공백으로 이으면 다른 문장이 된다.

    실측: "大 家 笑 着" → "큰 집은 웃고" / "大家笑着" → "모두가 웃고"
    """
    assert ocr.join_ocr_words(["大", "家", "笑", "着", "，", "跑", "着"]) == "大家笑着，跑着"
    assert ocr.join_ocr_words(["Hello", "world"]) == "Hello world"        # 라틴은 그대로
    assert ocr.join_ocr_words(["안녕", "하세요"]) == "안녕 하세요"          # 한글은 띄어쓰기 유지
    assert ocr.join_ocr_words(["読", "み", "ます"]) == "読みます"           # 일본어도 붙인다


# --- zh→ko 라우팅 (M-9) ------------------------------------------------------
def test_zh_ko_routes_through_pivot_not_m2m100_fallback():
    """중국어→한국어는 m2m100 폴백이 아니라 zh→en→ko 피벗을 타야 한다.

    실측(12문장 chrF): m2m100 24.5 < 3rd-party zh-ko 28.3 < **피벗 32.0**.
    전용 opus-mt zh→ko 는 존재하지 않는다(2026-08-11 Hub 확인).
    """
    hyb = tr.HybridTranslator
    assert ("zh", "ko") not in hyb.BILINGUAL_MODELS, "전용 zh-ko 모델은 없다 — 있으면 피벗을 지워라"
    assert hyb.PIVOT_ROUTES.get(("zh", "ko")) == "en"
    # 피벗 두 다리가 실제로 등록돼 있어야 폴백으로 안 샌다.
    assert ("zh", "en") in hyb.BILINGUAL_MODELS
    assert ("en", "ko") in hyb.BILINGUAL_MODELS

    calls = []

    class FakeLeg:
        def __init__(self, tag): self.tag = tag
        def translate_batch(self, texts, src, tgt, num_beams=1):
            calls.append((src, tgt))
            return [f"{self.tag}:{t}" for t in texts]

    h = hyb()
    h._bilingual = {("zh", "en"): FakeLeg("en"), ("en", "ko"): FakeLeg("ko")}
    h._ensure_multilingual = lambda **kw: (_ for _ in ()).throw(
        AssertionError("m2m100 폴백으로 샜다 — 피벗 라우팅이 안 걸렸다"))
    out = h.translate_batch(["大家笑着"], "zh", "ko")
    assert calls == [("zh", "en"), ("en", "ko")], f"피벗 순서가 틀렸다: {calls}"
    assert out == ["ko:en:大家笑着"]


def test_unregistered_pair_still_falls_back_to_m2m100():
    """피벗을 넣었다고 폴백 경로가 죽으면 안 된다 (아랍어 등)."""
    used = []

    class FakeM2M:
        def translate_batch(self, texts, src, tgt, num_beams=1):
            used.append((src, tgt))
            return list(texts)

    h = tr.HybridTranslator()
    h._ensure_multilingual = lambda **kw: FakeM2M()
    h.translate_batch(["مرحبا"], "ar", "ko")
    assert used == [("ar", "ko")], "등록 안 된 쌍이 m2m100 폴백으로 안 갔다"


def test_translation_group_failure_does_not_kill_other_groups():
    """TE-1: 한 언어 그룹의 실패가 프레임 전체를 날리면 안 된다.

    실측(2026-09-05): en-ko + zh-en 이 떠 있는 상태에서 m2m100 폴백 로드가 CUDA OOM
    → 예외가 batch_translate 밖으로 나가 **정상 영어 줄까지 전부** 사라졌다(오버레이 공백).
    실패한 그룹의 줄만 None 이 되고, 실패 사실은 LAST_GROUP_ERRORS 에 남아야 한다.
    """
    class Boom:
        def translate_batch(self, texts, src, tgt, num_beams=1):
            if src == "ar":
                raise RuntimeError("CUDA out of memory")
            return [f"번역:{t}" for t in texts]

    class NoDisk:            # 사용자의 영구 캐시를 테스트 문장으로 더럽히지 않는다
        def get(self, *a): return None
        def put(self, *a): pass

    real, real_disk = tr.TRANSLATOR, tr.PERSIST_CACHE
    tr.TRANSLATOR, tr.PERSIST_CACHE = Boom(), NoDisk()
    try:
        with tr.CACHE_LOCK:
            tr.TRANS_CACHE.clear()
        out = tr.batch_translate(["hello", "مرحبا", "world"], ["en", "ar", "en"], "ko")
    finally:
        tr.TRANSLATOR, tr.PERSIST_CACHE = real, real_disk
    assert out[0] and out[2], f"멀쩡한 영어 줄이 같이 날아갔다: {out}"
    assert out[1] is None, f"실패한 줄에 값이 들어갔다: {out}"
    assert tr.LAST_GROUP_ERRORS and "ar" in tr.LAST_GROUP_ERRORS[0], \
        f"실패가 조용히 삼켜졌다(CO-2): {tr.LAST_GROUP_ERRORS}"


# --- 반복 붕괴 재번역 (RP-2) --------------------------------------------------
# 실측된 사고: OCR CER 0.00% 인 문장이
#   "The kettle on the far shelf had been whistling for a while before anyone noticed it."
#   → "그 때, 그 때, 한 때의 그 시절, 그 시절, 그때의 그 시절을 떠올리는 사람이 있었다."
# 로 나왔다. greedy 디코딩의 탐색 실패이고, 같은 문장을 빔 서치로 뽑으면 정상이다.
COLLAPSE_SAMPLE = ("그 때, 그 때, 한 때의 그 시절, 그 시절, "
                   "그때의 그 시절을 떠올리는 사람이 있었다.")
COLLAPSE_SRC = ("The kettle on the far shelf had been whistling for a while "
                "before anyone noticed it.")
GOOD_SAMPLE = "이윽고 찻잔은 누군가가 눈치 채기 전에 잠시 동안 휘파람을 불고 있었다."


def test_repetition_collapse_is_detected():
    """붕괴한 번역문은 초과 반복률로 검출되고, 정상 번역문은 검출되지 않는다."""
    bad = tr.repetition_excess(COLLAPSE_SRC, COLLAPSE_SAMPLE)
    good = tr.repetition_excess(COLLAPSE_SRC, GOOD_SAMPLE)
    assert bad >= tr.RETRY_REPETITION_EXCESS, f"붕괴를 못 잡는다: {bad:+.3f}"
    assert good < tr.RETRY_REPETITION_EXCESS, f"정상 번역을 붕괴로 오검출: {good:+.3f}"


def test_normal_translations_are_not_flagged_as_collapse():
    """오검출 가드. 원문 자신이 반복이면 번역문도 반복이라 그건 붕괴가 아니다."""
    pairs = [
        ("Your changes have been saved to the local drive.",
         "변경 사항이 로컬 드라이브에 저장되었습니다."),
        ("An enhancement factor of 0.0 gives a blurred image, a factor of 1.0 gives "
         "the original image, and a factor of 2.0 gives a sharpened image.",
         "0.0의 향상 계수는 흐린 이미지를 제공하고, 1.0의 계수는 원본 이미지를 제공하며, "
         "2.0의 계수는 선명하게 된 이미지를 제공합니다."),
        ("Storage almost full", "저장 공간이 거의 찼습니다"),
        ("Measurements", "측정"),
    ]
    for src, tgt in pairs:
        ex = tr.repetition_excess(src, tgt)
        assert ex < tr.RETRY_REPETITION_EXCESS, f"정상 문장 오검출({ex:+.3f}): {tgt}"


def test_collapsed_piece_is_retranslated_with_beams():
    """붕괴한 조각만 빔 서치로 다시 뽑고, 더 나아졌을 때만 바꿔치기한다."""
    seen = []

    class FakeCollapse:
        def translate_batch(self, texts, src, tgt, num_beams=1):
            seen.append((tuple(texts), num_beams))
            if num_beams == 1:
                return [COLLAPSE_SAMPLE if t == COLLAPSE_SRC else "정상 번역입니다."
                        for t in texts]
            return [GOOD_SAMPLE for _ in texts]

    class NoDisk:
        def get(self, *a): return None
        def put(self, *a): pass

    real, real_disk = tr.TRANSLATOR, tr.PERSIST_CACHE
    tr.TRANSLATOR, tr.PERSIST_CACHE = FakeCollapse(), NoDisk()
    try:
        with tr.CACHE_LOCK:
            tr.TRANS_CACHE.clear()
        out = tr.batch_translate([COLLAPSE_SRC, "A quiet street."], ["en", "en"], "ko")
    finally:
        tr.TRANSLATOR, tr.PERSIST_CACHE = real, real_disk
    assert out[0] == GOOD_SAMPLE, f"붕괴가 그대로 표시된다: {out[0]}"
    assert len(seen) == 2 and seen[1][1] == tr.RETRY_NUM_BEAMS, \
        f"재시도가 빔 서치로 안 돌았다: {seen}"
    assert seen[1][0] == (COLLAPSE_SRC,), f"붕괴 안 한 줄까지 재시도했다: {seen[1][0]}"


def test_normal_batch_never_triggers_retry():
    """정상 번역만 있는 프레임에서는 재시도가 아예 안 돈다(속도 회귀 가드)."""
    beams = []

    class FakeOk:
        def translate_batch(self, texts, src, tgt, num_beams=1):
            beams.append(num_beams)
            return [f"{t} 의 정상적인 한국어 번역문입니다." for t in texts]

    class NoDisk:
        def get(self, *a): return None
        def put(self, *a): pass

    real, real_disk = tr.TRANSLATOR, tr.PERSIST_CACHE
    tr.TRANSLATOR, tr.PERSIST_CACHE = FakeOk(), NoDisk()
    try:
        with tr.CACHE_LOCK:
            tr.TRANS_CACHE.clear()
        tr.batch_translate(["The ferry left the harbor at dawn.",
                            "Nobody on board spoke for the first hour."],
                           ["en", "en"], "ko")
    finally:
        tr.TRANSLATOR, tr.PERSIST_CACHE = real, real_disk
    assert beams == [1], f"재시도가 헛돌았다: {beams}"


# --- 표시 게이트 (DG-1) ------------------------------------------------------
def test_carryover_ignores_identifiers_glued_to_digits():
    """CO-3: `user1:` 같은 식별자는 미번역 잔류가 아니다 (채팅 11/12줄이 숨었던 사고).

    반대로 글자 **사이에** 숫자가 낀 것(`cre2ted`)은 OCR 오독이라 계속 잡아야 한다.
    """
    assert tr.carryover_words("user1: 어제 로그 파일을 봤나요") == []
    assert tr.carryover_words("mp3 파일을 여세요") == []
    assert tr.carryover_words("PyQt5, PySide2 순서") == []
    assert tr.carryover_words("파일은 cre2ted 될 수 있습니다"), "OCR 오독을 놓쳤다"
    assert "dealing" in tr.carryover_words("4회씩 각 dealing 156 피해")
    assert tr.display_gate_reject("user1: did anyone look at the log file",
                                  "user1: 어제 로그 파일을 봤나요", "ko") is None


def test_display_gate_hides_untranslated_output():
    # 목표가 한국어인데 번역문에 한글이 없다 = 번역이 안 일어났다.
    assert tr.display_gate_reject("Hello world", "Hello world", "ko") is not None
    # 정상 번역은 통과.
    assert tr.display_gate_reject("Hello world", "안녕 세상", "ko") is None


def test_display_gate_hides_length_collapse():
    src = "This is a reasonably long sentence that should not collapse."
    assert tr.display_gate_reject(src, "응", "ko") is not None


def test_display_gate_hides_low_conf_source_lines():
    """DG-1c: 게이트의 유일한 '원문을 보는' 규칙. 라인 높이에 기대면 안 된다.

    OU-2와 같은 뿌리: 1차 OCR이 무너지면 라인 높이가 부풀어(10px 글자 → 28px 보고)
    예전의 '극소 라인 AND 저 conf' 조건이 영원히 안 걸렸고, conf 52짜리 **틀린 자막**이
    그대로 떴다. 높이가 아무리 커도 conf 가 낮으면 숨겨야 한다.
    """
    assert tr.display_gate_reject("hello", "안녕", "ko", 7, 40.0) is not None
    # 부풀려진 높이(28px)여도 저 conf면 숨긴다 — 이게 회귀 가드의 핵심이다.
    assert tr.display_gate_reject("hello", "안녕", "ko", 28, 52.0) is not None
    # conf 가 높으면 라인 높이와 무관하게 통과 (거짓 양성 방지).
    assert tr.display_gate_reject("hello", "안녕", "ko", 7, 95.0) is None
    assert tr.display_gate_reject("hello", "안녕", "ko", 30, 95.0) is None
    # conf 를 모르면(UIA 경로 등) 이 규칙은 판정하지 않는다.
    assert tr.display_gate_reject("hello", "안녕", "ko", None, None) is None


# --- 캐시 키 정규화 (RT-1) ---------------------------------------------------
def test_cache_key_absorbs_ocr_whitespace_jitter():
    assert tr._cache_key_text("hello   world") == tr._cache_key_text(" hello world\n")


def test_lru_evicts_oldest():
    lru = tr.LRU(capacity=2)
    lru.put("a", 1)
    lru.put("b", 2)
    lru.get_or_none("a")      # a를 최신으로 올린다
    lru.put("c", 3)           # 가장 오래된 b가 밀려나야 한다
    assert lru.get_or_none("b") is None
    assert lru.get_or_none("a") == 1
    assert lru.get_or_none("c") == 3


# --- 프레임 해시 / 변경 밴드 (F-1, PF-2) --------------------------------------
def _frame(value, band=None):
    img = np.full((64, 64, 3), value, dtype=np.uint8)
    if band:
        img[band[0]:band[1], :, :] = 255 - value
    return img


def test_hash_distance_detects_region_change():
    a = cap.cheap_hash(_frame(10), (0, 0, 100, 100))
    b = cap.cheap_hash(_frame(10), (5, 0, 105, 100))   # 같은 그림, 다른 캡처 영역
    assert cap.hash_distance(a, b) > cap.FRAME_DIFF_THRESHOLD
    assert cap.hash_distance(a, a) == 0


def test_dirty_row_band_locates_changed_rows():
    rect = (0, 0, 64, 64)
    before = cap.cheap_hash(_frame(10), rect)
    after = cap.cheap_hash(_frame(10, band=(32, 40)), rect)
    band = cap.dirty_row_band(before, after)
    assert band is not None, "변경 밴드를 찾지 못했다"
    row0, row1 = band
    # 64px 이미지의 32~40행 = 16등분 격자의 8~10행 근처
    assert 6 <= row0 <= 8 and 9 <= row1 <= 11, f"밴드가 엉뚱하다: {band}"


def test_dirty_row_band_returns_none_when_region_moved():
    """캡처 영역이 바뀌면 좌표계가 달라 밴드 비교가 무의미 → 전체 프레임 처리."""
    a = cap.cheap_hash(_frame(10), (0, 0, 64, 64))
    b = cap.cheap_hash(_frame(200), (9, 9, 73, 73))
    assert cap.dirty_row_band(a, b) is None
    assert cap.dirty_row_band(None, b) is None


# --- 작은 글자 검출 / 정적 스킵 (FD-1) ----------------------------------------
def _text_screen(w=1920, h=1080, line_px=28, line_top=700, phase=0, caret=False):
    """자막 한 줄을 '세로 획 패턴'으로 흉내낸 화면 (폰트 설치에 의존하지 않는다).

    phase를 옮기면 = 같은 자리에 다른 문장이 뜬 것과 같은 픽셀 변화가 된다.
    """
    img = np.full((h, w, 3), 30, dtype=np.uint8)
    img[:h // 18] = 48                                   # 정적 배경(툴바)
    stroke_w = max(1, line_px // 8)
    for x in range(300 + phase, min(w - 10, 1400), max(3, line_px // 2)):
        img[line_top:line_top + line_px, x:x + stroke_w] = 235
    if caret:
        img[line_top:line_top + line_px, 290:293] = 235   # 폭 3px 텍스트 커서
    return img


def test_small_text_line_change_is_detected():
    """FD-1: 본문/자막 크기(12~60px) 한 줄 교체가 프레임 스킵에 먹히면 안 된다.

    16x16 고정 격자 시절엔 1920x1080에서 28px 한 줄 변화가 3(임계 8 미만)으로 나와
    실시간 자막 번역이 구조적으로 동작하지 않았다.
    """
    for w, h in ((1920, 1080), (3840, 2160)):
        rect = (0, 0, w, h)
        for px in (12, 20, 28, 34, 44, 60):
            top = h * 2 // 3
            a = cap.cheap_hash(_text_screen(w, h, px, top), rect)
            b = cap.cheap_hash(_text_screen(w, h, px, top, phase=max(1, px // 4)), rect)
            dist = cap.hash_distance(a, b)
            assert dist >= cap.FRAME_DIFF_THRESHOLD, \
                f"{w}x{h} {px}px 한 줄 교체를 놓쳤다: dist={dist}"
            band = cap.dirty_row_band(a, b)
            assert band is not None and band[0] <= top * 16 // h < band[1], \
                f"{w}x{h} {px}px 변경 밴드가 엉뚱하다: {band}"


def test_static_screen_is_skipped():
    """FD-1: 정적 화면에서 오검출 0. 깨지면 유휴 CPU와 OCR 비용이 폭발한다."""
    frame = _text_screen()
    rect = (0, 0, 1920, 1080)
    base = cap.cheap_hash(frame, rect)
    for _ in range(50):
        assert cap.hash_distance(base, cap.cheap_hash(frame.copy(), rect)) == 0
    # 깜빡이는 텍스트 커서는 어느 행에서도 셀 1개만 바꾼다 → 변화로 치지 않는다.
    caret = cap.cheap_hash(_text_screen(caret=True), rect)
    assert cap.hash_distance(base, caret) < cap.FRAME_DIFF_THRESHOLD, \
        "캐럿 깜빡임이 프레임 변화로 잡혔다 (FRAME_DIFF_MIN_ROW_CELLS 확인)"


def _idle_controller():
    """워커 스레드/모델 없이 poll_interval 만 물어볼 수 있는 최소 컨트롤러."""
    opts = cap.RuntimeOptions(src_text="auto", tgt_text="Korean", ocr_text="Tesseract")

    class _Region:
        capture_mode = None

        def update_for_mode(self):
            pass

    return eng.BackgroundController(region=_Region(), options=opts,
                                    is_self_hwnd=lambda _h: False)


def test_poll_backs_off_to_idle_when_screen_is_quiet():
    """PL-1 유휴 CPU 가드. 조용할 때 폴 간격이 예전(0.25s)보다 짧아지면 안 된다.

    B0에서 유휴 CPU를 24%→8%로 내린 값이 바로 이 0.25s다. 여기가 깨지면
    가만히 있는 PC의 CPU/발열이 그대로 되돌아간다.
    """
    import time as _t
    assert cap.POLL_IDLE_S >= 0.25, "유휴 폴 간격을 0.25s 아래로 내리면 유휴 CPU가 오른다"
    ctrl = _idle_controller()
    ctrl._last_change_at = _t.monotonic() - (cap.POLL_ACTIVE_WINDOW_S + 1.0)
    assert ctrl.poll_interval() == cap.POLL_IDLE_S


def test_poll_speeds_up_right_after_a_change():
    """PL-1: 자막이 방금 바뀌었으면 다음 프레임을 촘촘히 본다(체감 지연의 250ms 세금 제거)."""
    import time as _t
    assert cap.POLL_ACTIVE_S < cap.POLL_IDLE_S
    ctrl = _idle_controller()
    ctrl._last_change_at = _t.monotonic()
    assert ctrl.poll_interval() == cap.POLL_ACTIVE_S


def test_worker_loop_has_no_hardcoded_poll_sleep():
    """PL-1 회귀 가드: 워커 루프의 대기는 전부 poll_interval() 을 거쳐야 한다.

    예전에는 `time.sleep(0.25)` / `time.sleep(0.3)` 이 루프 본문에 박혀 있어서
    캡처가 116→32ms로 빨라진 뒤에도 체감 지연이 그대로였다. 상수를 다시 박으면 여기서 잡는다.
    """
    import inspect
    src = inspect.getsource(eng.BackgroundController.capture_and_translate)
    bad = [ln.strip() for ln in src.splitlines()
           if "time.sleep(0.25)" in ln or "time.sleep(0.3)" in ln]
    assert not bad, f"폴 간격이 하드코딩됐다: {bad}"
    assert src.count("self.poll_interval()") >= 3


def test_capture_uses_primary_fast_path_only_inside_primary():
    """CP-1: 주 모니터를 벗어나면 all_screens=True — 아니면 보조 모니터가 검게 나온다."""
    if sys.platform != "win32":
        return
    import ctypes
    pw = int(ctypes.windll.user32.GetSystemMetrics(0))
    ph = int(ctypes.windll.user32.GetSystemMetrics(1))
    assert not cap.needs_all_screens(0, 0, pw, ph), "주 모니터 전체는 빠른 경로여야 한다"
    assert not cap.needs_all_screens(100, 100, 700, 500)
    assert cap.needs_all_screens(-3000, -400, -2400, 0), "음수 좌표(왼쪽 보조 모니터)"
    assert cap.needs_all_screens(pw - 50, 0, pw + 50, 100), "오른쪽으로 걸침"
    assert cap.needs_all_screens(0, 0, pw, ph + 1), "아래로 걸침"


# --- Tesseract 라인 재구성 (LM-1) --------------------------------------------
def _tess_data(words):
    """(text, left, top, width, height, conf, line) → pytesseract DICT 형태."""
    keys = ("text", "left", "top", "width", "height", "conf",
            "block_num", "par_num", "line_num")
    data = {k: [] for k in keys}
    for text, left, top, w, h, conf, line in words:
        data["text"].append(text)
        data["left"].append(left)
        data["top"].append(top)
        data["width"].append(w)
        data["height"].append(h)
        data["conf"].append(conf)
        data["block_num"].append(1)
        data["par_num"].append(1)
        data["line_num"].append(line)
    return data


def test_line_font_size_ignores_outlier_glyph():
    """LM-1: 장식 글자 하나(81px)가 라인 전체 폰트를 끌고 가면 안 된다.
    이 케이스는 `_line_segments`의 높이 이상치 제거가 잡는다."""
    data = _tess_data([
        ("Hello", 10, 100, 50, 14, 90, 1),
        ("world", 70, 100, 50, 14, 90, 1),
        ("|", 130, 60, 4, 81, 90, 1),      # 그래프 축이 세로 막대로 잡힌 경우
    ])
    lines = eng.BackgroundController._lines_from_tess(data, 1.0)
    assert lines, "라인이 사라지면 안 된다"
    _text, _bbox, font_size = lines[0]
    assert font_size < 30, f"이상치 글자에 폰트가 끌려갔다: {font_size}px"


def test_line_font_size_uses_median_not_max():
    """LM-1: 이상치 필터(중앙값 x2.5)를 **통과하는** 큰 글자가 섞여도 폰트는 중앙값이어야 한다.

    필터 임계 아래(14 x 2.5 = 35)인 30px 글자는 살아남는다. `font_size = max(...)`면
    본문 14px 라인이 30px 폰트로 그려져 자막 박스가 부풀어 오른다.
    """
    data = _tess_data([
        ("Hello", 10, 100, 40, 14, 90, 1),
        ("world", 60, 100, 40, 14, 90, 1),
        ("Q", 110, 96, 20, 30, 90, 1),      # 대문자/장식 — 필터는 통과(30 < 35)
    ])
    lines = eng.BackgroundController._lines_from_tess(data, 1.0)
    _text, _bbox, font_size = lines[0]
    assert font_size == 14, f"중앙값이 아니라 최대값을 썼다: {font_size}px"


def test_edge_trim_removes_icon_fragment_before_text():
    """LM-1b: 글자 왼쪽 아이콘이 조각으로 읽혀 라인 앞에 붙는 문제.

    실측 수치를 그대로 쓴다 — 본문 단어 사이 6px, 아이콘 경계 14px, 글자 높이 14px.
    조각이 남으면 bbox 왼쪽이 끌려가 문단 병합의 좌측 정렬 판정이 깨진다.
    """
    data = _tess_data([
        ("a", 609, 100, 12, 20, 66, 1),        # 아이콘 조각 (1글자 + 높이 이상)
        ("®", 645, 96, 20, 28, 88, 1),         # 아이콘 조각 (글자 아님)
        ("Enemies", 687, 100, 66, 14, 72, 1),  # 여기부터 진짜 본문
        ("have", 759, 100, 38, 14, 93, 1),
        ("a", 803, 100, 8, 10, 94, 1),         # 문장 속 짧은 단어 — 살아남아야 한다
        ("chance", 817, 100, 56, 14, 90, 1),
    ])
    lines = eng.BackgroundController._lines_from_tess(data, 1.0)
    text, bbox, _fs = lines[0]
    assert text == "Enemies have a chance", f"조각이 남았거나 본문이 잘렸다: {text!r}"
    assert bbox[0] == 687, f"bbox 왼쪽이 아이콘까지 끌려갔다: {bbox}"


def test_edge_trim_keeps_normally_spaced_short_word():
    """LM-1b 반대 방향: 본문 단어는 짧아도, 간격이 조금 벌어져도 떼면 안 된다.

    "짧으면 조각"으로 판정했다가 문서에서 'as a', 'is' 가 잘리는 걸 실측으로 잡았다.
    마지막 'is' 는 간격이 임계를 넘지만(20px > 14×0.6) 높이가 본문과 같으므로 살아야 한다.
    """
    data = _tess_data([
        ("of", 100, 100, 16, 14, 90, 1),
        ("mice", 122, 100, 36, 14, 90, 1),
        ("and", 164, 100, 30, 14, 90, 1),
        ("men", 200, 100, 30, 14, 90, 1),
        ("is", 250, 100, 14, 14, 90, 1),     # 간격 20px — 임계는 넘지만 본문 높이
    ])
    lines = eng.BackgroundController._lines_from_tess(data, 1.0)
    assert lines[0][0] == "of mice and men is", f"정상 단어가 잘렸다: {lines[0][0]!r}"


def test_edge_trim_keeps_oversized_initial_glued_to_text():
    """LM-1b: 간격 임계를 낮추면 본문에 **붙어 있는** 큰 첫 글자(장식 대문자)까지 떼어낸다.

    아이콘과 장식 대문자는 둘 다 높이가 이상치다. 갈라주는 건 간격뿐이다 —
    아이콘은 멀리(10px+) 떨어져 있고 장식 대문자는 다음 글자에 붙어 있다(5px).
    """
    data = _tess_data([
        ("T", 100, 92, 20, 26, 88, 1),      # 장식 대문자 (높이 이상치, 간격 5px)
        ("he", 125, 100, 18, 14, 90, 1),
        ("quick", 148, 100, 44, 14, 90, 1),
        ("brown", 196, 100, 48, 14, 90, 1),
        ("fox", 248, 100, 28, 14, 90, 1),
    ])
    text = eng.BackgroundController._lines_from_tess(data, 1.0)[0][0]
    assert text.startswith("T he"), f"본문에 붙은 첫 글자가 잘렸다: {text!r}"


def test_glyph_sized_line_is_dropped():
    """NZ-1: 폭이 글자 크기보다 좁은 라인 = 아이콘 한 조각. 실측 'Hill'(폭 26, 크기 28)."""
    data = _tess_data([("Hill", 213, 450, 26, 28, 76, 1)])
    assert eng.BackgroundController._lines_from_tess(data, 1.0) == []
    # 같은 글자라도 폭이 정상이면 살아야 한다.
    ok = _tess_data([("Hill", 213, 450, 60, 14, 76, 1)])
    assert eng.BackgroundController._lines_from_tess(ok, 1.0), "정상 라인까지 버렸다"


def test_sprite_read_as_text_is_dropped():
    """LM-1b + 폭 규칙: 캐릭터 그림을 글자로 읽은 'ra 7 ff'(실측)가 화면에 박스를 그렸다.

    조각은 짧고 confidence 가 낮다. 두 글자짜리 본문 단어(conf 90+)와 갈리는 지점이 거기다.
    """
    junk = _tess_data([("ra", 223, 372, 26, 28, 40, 1),      # 간격 30, 40
                       ("7", 279, 358, 16, 50, 21, 1),
                       ("ff", 371, 371, 22, 37, 80, 1)])
    assert eng.BackgroundController._lines_from_tess(junk, 1.0) == []
    real = _tess_data([("On", 687, 497, 26, 14, 95, 1),      # 간격 6 (실측값)
                       ("Break", 719, 497, 44, 14, 95, 1)])
    assert eng.BackgroundController._lines_from_tess(real, 1.0), "정상 문장까지 버렸다"


# --- 문단 병합 (PM-1) --------------------------------------------------------
def _para_line(text, left, top, width, height=14):
    return (text, (left, top, left + width, top + height), height)


def test_paragraph_merge_joins_wrapped_sentence():
    """PM-1: 줄바꿈으로 이어진 줄은 한 문단으로 합쳐 **한 번에** 번역해야 한다."""
    lines = [
        _para_line("Enemies have a chance to drop a dish that grants", 687, 253, 408),
        _para_line("one stack (max 50). Each stack grants +0.1%", 687, 269, 368),
        _para_line("haste. Lose 50% stacks when hit.", 687, 285, 274),
    ]
    merged, confs = eng.merge_paragraph_lines(lines, [90.0, 80.0, 70.0])
    assert len(merged) == 1, f"합쳐지지 않았다: {[m[0] for m in merged]}"
    assert merged[0][0].startswith("Enemies have")
    assert merged[0][0].endswith("when hit.")
    assert merged[0][1] == (687, 253, 1095, 299), f"문단 bbox가 틀렸다: {merged[0][1]}"
    assert 70.0 < confs[0] < 90.0, f"길이 가중 평균이 아니다: {confs[0]}"


def test_paragraph_merge_keeps_header_separate():
    """PM-1: 제목은 본문보다 짧다. 폭 규칙이 없으면 제목이 본문에 빨려들어간다."""
    lines = [
        _para_line("Busboy", 687, 233, 60),
        _para_line("Enemies have a chance to drop a dish that grants", 687, 253, 408),
    ]
    merged, _ = eng.merge_paragraph_lines(lines, None)
    assert len(merged) == 2, f"제목이 본문에 합쳐졌다: {merged[0][0]!r}"


def test_paragraph_merge_tolerates_overlapping_line_boxes():
    """PM-1: 줄 bbox 는 글자의 위/아래 삐침 때문에 서로 몇 px 겹친다(실측: -3px).

    겹침 허용치를 0으로 두면 그 줄에서 문단이 끊긴다.
    """
    lines = [
        _para_line("At the end of his conversation with the Little Prince,", 457, 856, 421),
        _para_line("manages to fix his plane and both continue on their way.", 457, 863, 561),
    ]
    merged, _ = eng.merge_paragraph_lines(lines, None)
    assert len(merged) == 1, f"겹친 줄에서 문단이 끊겼다: {[m[0][:20] for m in merged]}"


def test_paragraph_merge_stops_at_sentence_end():
    """PM-1: 종결부호로 끝난 줄 뒤는 다른 문단이다. 폭·간격이 맞아도 합치면 안 된다."""
    lines = [
        _para_line("and both he and the little prince continue on their journeys.", 457, 856, 561),
        _para_line("Shortly after completing the book, he finally got his wish.", 457, 872, 557),
    ]
    merged, _ = eng.merge_paragraph_lines(lines, None)
    assert len(merged) == 2, f"문단 경계를 넘어 합쳐졌다: {merged[0][0][:60]!r}"


def test_paragraph_merge_survives_junk_between_lines():
    """PM-1: 화면 구석의 조각 하나가 문단 사슬을 끊으면 안 된다.

    정렬 순서상 문단 두 줄 **사이**에 끼는 아이콘이 실제로 있었다(실측: 'Hill' T=457).
    바로 앞 줄하고만 비교하면 여기서 문단이 두 동강 난다.
    """
    lines = [
        _para_line("Right click/trigger to throw Pudding that heals", 687, 429, 388),
        _para_line("gives a speed boost and removes negative status", 687, 445, 408),
        _para_line("Hill", 213, 457, 26, height=28),          # 옆 동네 조각
        _para_line("effects (12s cooldown).", 687, 461, 200),
    ]
    merged, _ = eng.merge_paragraph_lines(lines, None)
    body = [m for m in merged if m[0].startswith("Right click")]
    assert len(body) == 1 and body[0][0].endswith("(12s cooldown)."), \
        f"조각이 문단을 끊었다: {[m[0] for m in merged]}"


# --- 미번역 잔류 게이트 (CO-1) ------------------------------------------------
def test_carryover_gate_rejects_half_translated_output():
    """CO-1: 번역기는 모르는 토큰을 그대로 복사한다. 반쯤 번역된 헛소리는 숨긴다."""
    src = "Stab 4 times each dealins 156 damase (4.25 attach speed)"
    tgt = "4회씩 각 dealing 156 피해 (4.25 첨부 speed)"
    assert tr.display_gate_reject(src, tgt, "ko", 14, 80) is not None


def test_carryover_gate_sees_words_glued_to_korean_particles():
    """CO-1: 번역문에서는 'sive를'처럼 조사가 바로 붙는다.

    한글도 `isalpha()`가 True라, 문자 단위로 "영문만 남기기"를 하면 통째로 새어나간다.
    """
    assert tr.carryover_words("sive를 치유하고 부정적인 상태를 제거") == ["sive"]


def test_carryover_gate_allows_proper_nouns_and_acronyms():
    """CO-1: 대문자로 시작하거나 전부 대문자인 토큰은 번역문에 남는 게 정상이다."""
    assert tr.carryover_words("잃어버린 HP를 회복합니다") == []
    assert tr.carryover_words("Pudding을 던진다 (Ss 쿨다운)") == []
    ok = "5초 동안 피해를 받은 후 천천히 일부 잃어버린 HP를 회복합니다."
    assert tr.display_gate_reject("Avoid damage after getting hit", ok, "ko", 14, 80) is None


def test_carryover_gate_allowance_scales_with_paragraph_length():
    """CO-2: 절대 허용치 0은 **한 줄** 기준이었다. PM-1 문단 병합이 판정 단위를 키운 뒤로는
    40단어 문단에서 OCR 오독 한 단어가 문단 전체를 숨겼다 (실측 어린왕자 4쪽, 2026-08-11).

    아래 두 문자열은 실측 그대로다 — 짧은 툴팁은 여전히 숨고, 긴 문단은 살아남아야 한다.
    """
    tooltip = "4회씩 각 dealing 156 피해 (4.25 첨부 speed) (3.55 쿨다운,"
    assert tr.carryover_allowance(tooltip) == 1
    assert tr.display_gate_reject("Stab 4 times each dealins 156 damase", tooltip, "ko") is not None

    para = ("내 그림은 모자가 아니었다. 그것은 코끼리를 소화하는 보아의 constitctor의 그림이었다. "
            "하지만 어른들이 이해할 수 없었기 때문에, 저는 또 다른 그림을 그렸습니다. "
            "저는 보아뱀의 내부를 그렸죠. 어른들이 그것을 분명히 볼 수 있도록 말이죠. "
            "항상 설명해야 할 것들이 있습니다. 제 그림 2는 이렇게 생겼습니다:")
    assert tr.carryover_words(para) == ["constitctor"]
    assert tr.display_gate_reject("My drawing was not a picture of a hat." + "x" * 300,
                                  para, "ko", 79, 94.8) is None, \
        "오독 단어 하나로 문단 전체가 숨는다 (CO-2 재발)"


def test_carryover_gate_skips_latin_targets():
    """CO-1: 목표 언어가 라틴이면 영단어가 남는 게 당연하다 — 규칙을 적용하면 안 된다."""
    assert tr.display_gate_reject("안녕하세요 여러분", "hello everyone", "en", 14, 80) is None


# --- 확대 재OCR 채택 기준 (UA-1) ---------------------------------------------
def test_upscale_accepted_only_when_confidence_improves():
    """UA-1: 확대 재OCR 채택 기준. **호출부를 실제로 태운다.**

    헬퍼(`_mean_conf`)만 검사하면 호출부의 `if` 를 라인 수 비교로 되돌려도 안 잡힌다 —
    실제로 뮤테이션 테스트에서 그 구멍이 드러났다.
    실측 수치를 그대로 쓴다: 픽셀 폰트 UI는 확대하면 라인이 늘고(20→22) 정확도는
    떨어진다(0.869→0.850). conf 는 71.9→60.4 로 정직하게 떨어진다.
    """
    controller = _make_controller()
    # LC-1b(kor 재스캔)가 끼어들면 패스 수가 늘어난다. 여기서는 UA-1만 본다 —
    # 목표를 영어로 두면 kor 이 처음부터 언어팩에 있어 재스캔 조건이 성립하지 않는다.
    controller.options = cap.RuntimeOptions("auto", "English", ocr.OCR_BACKEND_TESSERACT)
    passes = []

    def fake_tess_lines(_gray, _lang, scale, conf_out=None, **_kw):
        passes.append(scale)
        if len(passes) == 1:                    # 1차: 작은 글씨 → OU-1 확대 발동
            lines, confs = [("real sentence here", (0, 0, 120, 10), 10)], [71.9]
        else:                                   # 2차(확대): 줄은 늘고 confidence 는 떨어짐
            lines = [("reol senlence here", (0, 0, 120, 10), 10),
                     ("junk", (0, 20, 60, 30), 10)]
            confs = [60.4, 60.4]
        if conf_out is not None:
            conf_out.extend(confs)
        return lines

    controller._tess_lines = fake_tess_lines
    out = controller.perform_ocr_with_boxes(np.zeros((100, 200, 3), dtype=np.uint8))
    assert len(passes) == 2, f"확대 재OCR이 발동하지 않았다: {passes}"
    assert [t for t, _b, _f in out] == ["real sentence here"], \
        f"라인 수만 보고 나쁜 확대 결과를 채택했다: {out}"


def test_upscale_rejected_when_it_drops_text():
    """UA-2: conf 만 보면 **줄을 통째로 흘린 패스가 이긴다**(남은 줄만 깨끗하니 평균이 오른다).

    실측(2026-09-07): Impact 12px 실화면에서 2차가 가운데 줄을 잃고(248자→140자)
    conf 는 81.8→89.5 라 채택되어 CER 7.51%→45.85%.
    """
    controller = _make_controller()
    controller.options = cap.RuntimeOptions("auto", "English", ocr.OCR_BACKEND_TESSERACT)
    passes = []

    def fake_tess_lines(_gray, _lang, scale, conf_out=None, **_kw):
        passes.append(scale)
        if len(passes) == 1:
            lines = [("first line of the paragraph here", (0, 0, 200, 10), 10),
                     ("second line that must not vanish", (0, 12, 200, 22), 10),
                     ("third line of the paragraph here", (0, 24, 200, 34), 10)]
            confs = [81.8] * 3
        else:                       # 2차: 가운데 줄을 통째로 흘렸는데 conf 는 더 높다
            lines = [("first line of the paragraph here", (0, 0, 200, 10), 10),
                     ("third line of the paragraph here", (0, 24, 200, 34), 10)]
            confs = [89.5] * 2
        if conf_out is not None:
            conf_out.extend(confs)
        return lines

    controller._tess_lines = fake_tess_lines
    out = controller.perform_ocr_with_boxes(np.zeros((100, 200, 3), dtype=np.uint8))
    assert len(passes) == 2, f"확대 재OCR이 발동하지 않았다: {passes}"
    assert "second line that must not vanish" in " ".join(t for t, _b, _f in out), \
        f"글자를 버린 확대 결과를 conf 만 보고 채택했다: {out}"


def test_upscale_fires_on_fullscreen_sized_input():
    """UB-1: 전체화면(1920x1080)에서도 확대 재OCR이 발동해야 한다.

    구 규칙(확대 후 픽셀 ≤ 4M)은 1080p의 2배(8.3M)를 **항상** 거부해서
    전체화면 문서의 단어 58%를 못 읽었다(실측 42% 인식). 이제 예산은 픽셀이 아니라
    1차 패스 실측 시간이므로, 화면 크기만으로 확대가 막히면 안 된다.
    """
    controller = _make_controller()
    scales = []

    def fake_tess_lines(_gray, _lang, scale, conf_out=None, **_kw):
        scales.append(scale)
        if conf_out is not None:
            conf_out.append(80.0 if len(scales) == 1 else 90.0)
        return [("small caption text", (0, 0, 300, 12), 12)]

    controller._tess_lines = fake_tess_lines
    controller.perform_ocr_with_boxes(np.zeros((1080, 1920, 3), dtype=np.uint8))
    assert len(scales) == 2, f"전체화면에서 확대 재OCR이 생략됐다: {scales}"
    assert scales[1] >= ocr.OCR_UPSCALE_MIN_FACTOR, f"확대 배율이 너무 낮다: {scales}"


def _text_image(px, height_px=140):
    """진짜 글자가 그려진 회색조 이미지. `ink_glyph_px` 는 이미지만 보므로 렌더가 필요하다."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (900, height_px), (255, 255, 255))
    d = ImageDraw.Draw(img)
    for row in range(4):
        d.text((20, 10 + row * (px + 6)),
               "the kettle on the far shelf had been whistling for a while",
               fill=(0, 0, 0), font_size=px)
    return np.array(img)


def test_upscale_fires_even_when_first_pass_inflates_line_height():
    """OU-2: 1차 오독은 라인 높이를 부풀린다 — 그 값으로 확대를 끄면 안 된다.

    실측(2026-09-07): 10px Georgia/Calibri/Times/Impact 를 1차가 **28px 로 보고**해서
    `median_px < 18` 이 성립하지 않았고, 확대가 가장 필요한 조합에서만 확대가 꺼졌다
    (CER 10.67%; 강제로 켜면 0.00%). 트리거는 1차 결과와 무관한 신호도 봐야 한다.
    """
    controller = _make_controller()
    controller.options = cap.RuntimeOptions("auto", "English", ocr.OCR_BACKEND_TESSERACT)
    scales = []

    def fake_tess_lines(_gray, _lang, scale, conf_out=None, **_kw):
        scales.append(scale)
        if conf_out is not None:
            conf_out.append(62.0 if len(scales) == 1 else 94.0)
        # 1차가 세 줄을 한 덩어리로 잘못 묶어 높이를 28px 로 보고한 상황.
        return [("garbled blob of three rows", (20, 10, 800, 38), 28)]

    controller._tess_lines = fake_tess_lines
    controller.perform_ocr_with_boxes(_text_image(10))
    assert len(scales) == 2, f"부풀려진 높이(28px) 때문에 확대가 꺼졌다: {scales}"


def test_upscale_retries_when_first_pass_reads_nothing():
    """OU-2: 1차가 0줄이면 확대를 **더** 해야 한다. 예전엔 `if lines:` 로 건너뛰었다.

    실측: Courier New 10px 100% → 2배 재OCR 하면 4.35%.
    단 빈 화면(잉크 없음)에서는 헛돈을 쓰면 안 된다 — 그건 켜지지 않아야 한다.
    """
    controller = _make_controller()
    controller.options = cap.RuntimeOptions("auto", "English", ocr.OCR_BACKEND_TESSERACT)
    scales = []

    def fake_tess_lines(_gray, _lang, scale, conf_out=None, **_kw):
        scales.append(scale)
        if len(scales) == 1:
            return []                              # 1차: 한 줄도 못 읽음
        if conf_out is not None:
            conf_out.append(91.0)
        return [("the kettle on the far shelf", (20, 10, 800, 22), 9)]

    controller._tess_lines = fake_tess_lines
    out = controller.perform_ocr_with_boxes(_text_image(10))
    assert len(scales) >= 2 and scales[1] > scales[0], f"0줄인데 확대를 건너뛰었다: {scales}"
    assert out, "확대가 읽어낸 줄이 버려졌다(1차 0줄과 비교하면 안 된다)"

    # 잉크가 없는 빈 화면은 확대하지 않는다 (비용 방지).
    scales.clear()
    controller.perform_ocr_with_boxes(np.full((1080, 1920, 3), 250, dtype=np.uint8))
    assert len(scales) == 1, f"빈 화면에서 확대 재OCR이 돌았다: {scales}"


def test_paragraph_merge_preserves_reading_order():
    """PM-2: 문단 병합이 읽기 순서를 뒤섞으면 안 된다.

    실측(2026-09-07): 같은 행이 좌/우로 쪼개진 화면에서 1·2·4번 줄이 한 문단이 되고
    3번이 따로 남아 "1 2 4 3" 순서로 나갔다 — CER 8.70→45.45 / 4.35→33.99.
    합쳐질 문단의 가로 구간 안에 다른 줄이 끼어 있으면 그 줄을 건너뛰어 합치지 않는다.
    """
    lines = [("first row of the paragraph", (40, 20, 578, 31), 8),
             ("second row left half", (40, 38, 322, 48), 8),
             ("second row right half", (363, 38, 585, 48), 8),
             ("third row of the paragraph", (40, 55, 189, 65), 8)]
    merged, _ = eng.merge_paragraph_lines(lines, [90.0] * 4)
    joined = " ".join(t for t, _b, _f in merged)
    for a, b in zip(lines, lines[1:]):
        assert joined.index(a[0]) < joined.index(b[0]), \
            f"읽기 순서가 뒤집혔다: {joined}"

    # 진짜 2단(column) 문서는 여전히 건너뛰어 합쳐야 한다 (원래 의도 유지).
    two_col = [("left column line one", (40, 20, 300, 32), 10),
               ("right column line one", (600, 20, 860, 32), 10),
               ("left column line two", (40, 34, 300, 46), 10),
               ("right column line two", (600, 34, 860, 46), 10)]
    merged2, _ = eng.merge_paragraph_lines(two_col, [90.0] * 4)
    assert len(merged2) == 2, f"2단 문서의 칼럼 병합이 깨졌다: {merged2}"


def test_ocr_langs_keeps_kor_when_target_is_not_korean():
    """LC-1: 한→외 번역에서는 kor 를 빼면 원문을 아예 못 읽는다.

    목표가 한국어일 때만 뺀다(그 줄들은 SK-1이 어차피 버리므로 시간만 2.5배 든다).
    """
    if "kor" not in ocr.AVAILABLE_TESS_LANGS:
        return   # 언어팩이 없는 환경에서는 검사 의미 없음
    controller = _make_controller()
    os.environ.pop("COCKTAIL_OCR_LANGS_AUTO", None)
    controller.options = cap.RuntimeOptions("auto", "English", ocr.OCR_BACKEND_TESSERACT)
    assert "kor" in controller._ocr_langs().split("+"), \
        f"목표가 한국어가 아닌데 kor 이 빠졌다: {controller._ocr_langs()}"
    controller.options = cap.RuntimeOptions("auto", "Korean", ocr.OCR_BACKEND_TESSERACT)
    assert "kor" not in controller._ocr_langs().split("+"), \
        f"목표가 한국어인데 kor 이 남아 시간을 2.5배 쓴다: {controller._ocr_langs()}"


def test_low_conf_latin_screen_rescans_with_kor():
    """LC-1b: kor 을 뺀 스캔이 한글을 라틴으로 오독하면(저 conf) kor 로 다시 읽는다.

    실측: 한국어 UI를 eng 단독으로 읽으면 "인벤토리" → "Be] Ast" (conf 39~70),
    kor 을 붙이면 conf 89~97. 재스캔이 없으면 그 오독이 번역돼 자막 박스로 뜬다.
    """
    if "kor" not in ocr.AVAILABLE_TESS_LANGS:
        return
    controller = _make_controller()
    os.environ.pop("COCKTAIL_OCR_LANGS_AUTO", None)
    langs_used = []

    def fake_tess_lines(_gray, lang, _scale, conf_out=None, **_kw):
        langs_used.append(lang)
        if "kor" in lang:                       # 제대로 읽음
            lines, confs = [("인벤토리", (0, 0, 60, 20), 20)], [95.0]
        else:                                   # 라틴 오독
            lines, confs = [("Be] Ast", (0, 0, 60, 20), 20)], [49.0]
        if conf_out is not None:
            conf_out.extend(confs)
        return lines

    controller._tess_lines = fake_tess_lines
    out = controller.perform_ocr_with_boxes(np.zeros((300, 600, 3), dtype=np.uint8))
    assert any("kor" in l for l in langs_used), f"kor 재스캔이 안 걸렸다: {langs_used}"
    assert [t for t, _b, _f in out] == ["인벤토리"], f"오독 결과를 채택했다: {out}"
    assert controller._ko_frames > 0, "한글 화면 sticky 가 안 켜졌다 — 매 프레임 2패스가 된다"
    assert "kor" in controller._ocr_langs().split("+"), \
        "sticky 중에는 kor 을 처음부터 켜야 왕복이 1회로 준다"


def test_low_confidence_word_is_not_dropped_mid_sentence():
    """WD-1: 라인 안 단어 컷은 문장을 훼손한다.

    실측: "removes negative status" 에서 'negative'(conf 12)가 사라져
    "removes status" 가 되고, 번역기는 뜻이 뒤집힌 그 문장을 성실히 번역했다.
    라인 평균 conf 는 (90+12+90)/3 = 64 로 라인 게이트(35)를 통과하므로,
    이 줄은 **통째로 살아남되 단어를 잃지 않아야** 한다.
    """
    data = _tess_data([
        ("removes", 100, 100, 66, 14, 90, 1),
        ("negative", 172, 100, 70, 14, 12, 1),
        ("status", 248, 100, 52, 14, 90, 1),
    ])
    text = eng.BackgroundController._lines_from_tess(data, 1.0)[0][0]
    assert text == "removes negative status", f"문장 한복판 단어가 사라졌다: {text!r}"


def test_line_splits_on_large_horizontal_gap():
    """LM-1: --psm 6이 좌우 두 단을 한 줄로 묶는 문제."""
    data = _tess_data([
        ("Left", 10, 100, 40, 14, 90, 1),
        ("Right", 500, 100, 40, 14, 90, 1),   # 450px 건너뜀
    ])
    lines = eng.BackgroundController._lines_from_tess(data, 1.0)
    assert len(lines) == 2, f"단이 분리되지 않았다: {lines}"


def test_lines_scale_back_to_original_coords():
    """PF-1: scale이 실수(축소)여도 좌표가 원본으로 환원돼야 한다."""
    data = _tess_data([("Hello", 100, 200, 60, 20, 90, 1),
                       ("world", 170, 200, 60, 20, 90, 1)])
    lines = eng.BackgroundController._lines_from_tess(data, 0.5)   # 절반으로 줄여 OCR했다
    _text, bbox, _fs = lines[0]
    assert bbox[0] == 200 and bbox[1] == 400, f"좌표 환원 실패: {bbox}"
    # `//` 정수나눗셈을 쓰면 실수 scale에서 float가 나온다 — 좌표는 반드시 int.
    assert all(isinstance(v, int) for v in bbox), f"bbox에 float가 섞였다: {bbox}"


def test_conf_out_stays_aligned_with_lines():
    """DG-1: 라인 리스트와 confidence 리스트의 인덱스가 어긋나면 게이트가 엉뚱한 줄을 본다."""
    data = _tess_data([
        ("Hello", 10, 100, 40, 14, 95, 1),
        ("world", 10, 130, 40, 14, 55, 2),
        ("again", 10, 160, 40, 14, 80, 3),
    ])
    confs = []
    lines = eng.BackgroundController._lines_from_tess(data, 1.0, confs)
    assert len(lines) == len(confs), f"{len(lines)} lines vs {len(confs)} confs"
    assert confs == [95.0, 55.0, 80.0]


# --- 번역 라우팅 (SK-1) ------------------------------------------------------
def test_same_language_line_is_not_drawn():
    """SK-1: 원문 언어 == 목표 언어면 그리지 않는다(None). 예전엔 원문을 그대로 돌려줘서
    한국어 화면에 '같은 글자가 적힌 검은 박스'가 덮였다."""
    out = tr.batch_translate(["안녕하세요"], ["ko"], "ko")
    assert out == [None], f"같은 언어 줄은 None이어야 한다: {out}"


def test_batch_translate_empty_input():
    assert tr.batch_translate([], "auto", "ko") == []


# --- 영구 캐시 ---------------------------------------------------------------
def test_persistent_cache_key_is_sha256_hex():
    key = tr.PersistentCache._key("en", "ko", "self-test")
    assert len(key) == 64 and all(c in "0123456789abcdef" for c in key)
    # 평문이 키에 남지 않아야 한다.
    assert "self-test" not in key


def test_persistent_cache_evicts_at_capacity():
    cache = tr.PersistentCache(path=os.devnull, max_entries=4)
    cache._enabled = True
    cache._loaded = True
    for i in range(6):
        cache.put("en", "ko", f"text{i}", f"번역{i}")
    assert len(cache._mem) <= 4, f"용량 초과: {len(cache._mem)}"


# --- 민감 창 (S-1) -----------------------------------------------------------
def test_dirty_band_view_crops_and_keeps_outside_subtitles():
    """PF-2: 변한 밴드만 잘라 내고, 밴드 밖 직전 자막은 살려 둔다.

    이게 깨지면 (a) 매 프레임 전체화면 OCR로 되돌아가거나(3.3초/프레임)
    (b) 밴드 밖 자막이 프레임마다 사라졌다 나타났다 한다.
    """
    controller = _make_controller()
    rect = (0, 0, 64, 64)
    before = cap.cheap_hash(_frame(10), rect)
    after = cap.cheap_hash(_frame(10, band=(32, 40)), rect)
    # 밴드(y=32~40) 밖의 자막 하나 + 밴드 안의 자막 하나. y1=0 기준 절대 좌표.
    controller._last_emitted = [
        ("outside", "밖", (0, 2, 30, 10), 12),
        ("inside", "안", (0, 33, 30, 39), 12),
    ]
    frame = _frame(10, band=(32, 40))
    ocr_image, band_dy, kept = controller._dirty_band_view(frame, before, after, 0)

    assert band_dy is not None, "밴드를 잡지 못하고 전체 프레임으로 떨어졌다"
    assert ocr_image.shape[0] < frame.shape[0], "세로가 잘리지 않았다"
    assert ocr_image.shape[1] == frame.shape[1], "가로를 자르면 단어가 잘려 OCR이 망가진다"
    kept_texts = [e[0] for e in kept]
    assert "outside" in kept_texts, "밴드 밖 자막이 사라졌다"
    assert "inside" not in kept_texts, "밴드 안 자막은 새 OCR 결과로 교체돼야 한다"


def _qt_app():
    """Qt 애플리케이션 싱글턴.

    **QApplication(GUI)으로 만든다.** 예전엔 QCoreApplication이었는데, 뒤에 오는
    설정창 테스트가 QWidget을 만드는 순간 프로세스가 그대로 죽었다 — 이미 만들어진
    애플리케이션 객체는 GUI용으로 바꿔 끼울 수 없다.
    """
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "0")   # DP-1: 앱과 같은 조건
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([sys.argv[0]])
        _qt_app.app = app   # GC 방지
    return app


def _make_controller():
    """위젯 없이 BackgroundController만 만든다 (BG-6: 콜백 3개에만 의존)."""
    _qt_app()
    region = cap.CaptureRegionController(is_self_hwnd=lambda _h: False)
    return eng.BackgroundController(
        region=region,
        options=cap.RuntimeOptions("auto", "Korean", ocr.OCR_BACKEND_TESSERACT),
        is_self_hwnd=lambda _h: False,
    )


def test_sensitive_pause_applies_in_every_capture_mode():
    """SB-1 회귀 방지 — 이 구멍으로 은행 로그인 화면이 번역·영구 캐시됐다.

    예전 워커 루프는 `capture_mode != MODE_BOX and _check_sensitive_pause()` 였다.
    판정이 캡처 모드를 보는 순간 다시 뚫린다.
    """
    controller = _make_controller()
    original = eng.is_sensitive_window
    try:
        eng.is_sensitive_window = lambda _hwnd: True
        for mode in (cap.MODE_WINDOW, cap.MODE_HOVER, cap.MODE_BOX):
            controller.region.capture_mode = mode
            assert controller.pause_reason() == "sensitive", \
                f"{mode} 모드에서 민감 창 가드가 뚫렸다"
    finally:
        eng.is_sensitive_window = original


def test_capturable_guard_pauses_before_sensitive_check():
    """B-15: 스샷OK 상태는 민감 창 여부와 무관하게 먼저 멈춘다."""
    controller = _make_controller()
    assert controller.pause_reason() != "capturable"
    controller.set_capturable(True)
    assert controller.pause_reason() == "capturable"


def test_sensitive_keywords_cover_login_and_banking():
    assert plat.SENSITIVE_KEYWORDS, "민감 키워드 목록이 비었다"
    lowered = [k.lower() for k in plat.SENSITIVE_KEYWORDS]
    for must in ("password", "login", "비밀번호", "결제", "카드번호"):
        assert must in lowered, f"민감 키워드 누락: {must}"


# --- 통합: 실제 번역 (M-8 회귀 방지) -----------------------------------------
def test_opus_mt_actually_translates_english_to_korean():
    """M-8 회귀 방지 — 이 테스트 하나가 구 smoke_tests 243개보다 가치 있다.

    `opus-mt-tc-big-en-ko`는 소스/타깃 vocab이 분리된(sepvoc) 모델인데 repo의
    tokenizer_config.json이 `separate_vocabs: false`로 잘못 적혀 있다. 그대로 두면
    AutoTokenizer가 **소스 문장을 타깃 vocab으로 인코딩**해 대부분 <unk>가 되고
    'Please enter your password' → '항우울제 방콕 heart.' 같은 쓰레기가 나온다.
    형태상 멀쩡한 한국어라 DG-1 표시 게이트도 통과한다 — 사람이 보기 전엔 안 걸린다.
    """
    translator = tr.OpusMtTranslator(
        tr.HybridTranslator.BILINGUAL_MODELS[("en", "ko")])
    cases = [
        "Please enter your password to continue.",
        "This application translates the screen in real time.",
    ]
    outputs = translator.translate_batch(cases, "en", "ko")
    for src, out in zip(cases, outputs):
        compact = out.replace(" ", "")
        hangul = sum(1 for ch in compact if "HANGUL" in unicodedata.name(ch, ""))
        ratio = hangul / max(1, len(compact))
        assert ratio > 0.5, f"번역 실패 — 한글 비율 {ratio:.0%}\n  {src!r}\n  -> {out!r}"


def test_long_paragraph_survives_to_display():
    """CO-2 회귀 방지 — **실동작**으로 문단 하나가 번역되어 표시까지 살아남는지 본다.

    증상은 "삽화 밑 3줄짜리 문단만 자막이 안 뜬다"였다. 재현(어린왕자 4쪽, 2026-08-11):
    OCR이 `constrictor`를 `consttictor`로 한 글자 틀리면 번역기가 그 단어를 그대로 복사하고,
    CO-1의 절대 허용치 0이 **346자 문단 전체**를 숨겼다. 나머지 39단어는 멀쩡했다.
    게이트 단독 테스트로는 못 잡는다 — 번역기가 실제로 무엇을 내놓는지가 조건이라
    batch_translate 를 그대로 태운다.
    """
    para = ("My drawing was not a picture of a hat. It was a picture of a boa consttictor "
            "digesting an elephant. But since the grown-ups were not able to understand it, "
            "I made another drawing: I drew the inside of a boa constrictor, so that the "
            "grown-ups could see it clearly. They always need to have things explained. "
            "My Drawing Number Two looked like this:")
    tgt = tr.batch_translate([para], ["en"], "ko")[0]
    assert tgt and tgt.strip(), "번역 결과가 비었다"
    reason = tr.display_gate_reject(para, tgt, "ko", 79, 94.8)
    assert reason is None, f"긴 문단이 표시 게이트에서 사라졌다: {reason}\n  -> {tgt!r}"


# --- UI 레이아웃 (UI-1) ------------------------------------------------------
def test_settings_window_keeps_widgets_inside_at_any_size():
    """UI-1 회귀 방지 — 창을 어떤 크기로 줄여도 위젯이 창 밖으로 나가지 않는다.

    고정 픽셀 배치(setGeometry)였을 때 640x400에서 버튼 3개, 480x220에서 4개가
    창 밖으로 나갔고 고DPI(폰트 DPI 168)에서는 버튼 글자가 잘렸다. 지오메트리는
    **창을 실제로 띄워야** 보이므로 여기서만 Qt 창을 하나 만든다.
    """
    from PySide6.QtCore import QRect
    from PySide6.QtWidgets import QWidget

    import cocktail_ui

    started = eng.BackgroundController.start
    eng.BackgroundController.start = lambda self: None   # 모델 로드/워커/화면 캡처 금지
    app = _qt_app()
    win = cocktail_ui.SettingsWindow()
    try:
        win.show()
        app.processEvents()
        for w, h in [(880, 120), (1400, 900), (640, 400), (480, 220)]:
            win.resize(w, h)
            app.processEvents()
            window = QRect(0, 0, win.width(), win.height())
            for child in win.findChildren(QWidget):
                if not child.isVisible() or child.findChildren(QWidget):
                    continue   # 컨테이너는 건너뛰고 말단 위젯만 본다
                rect = QRect(child.mapTo(win, child.rect().topLeft()), child.size())
                name = child.metaObject().className()
                assert window.contains(rect), (
                    f"{w}x{h}: {name} 이 창 밖으로 나갔다 {rect} (창 {window})")
                text = child.text() if hasattr(child, "text") else ""
                if text:
                    need = child.fontMetrics().horizontalAdvance(text) + 12
                    assert need <= child.width(), (
                        f"{w}x{h}: {name} 글자가 잘린다 {text!r} "
                        f"({need}px 필요 / {child.width()}px 확보)")
    finally:
        win._is_quitting = True
        win.close()
        eng.BackgroundController.start = started


def test_window_opens_without_importing_torch():
    """LZ-1 회귀 방지 — 창을 띄우는 경로가 torch/transformers를 끌고 오면 안 된다.

    실측(2026-09-05): 최상단 `import torch` 하나로 프로세스 시작→창이 12~14초였다.
    torch 2.3s + transformers 2.0s(그 안에서 torch._dynamo·sklearn.metrics를 또 끈다).
    모델은 `BackgroundController._preload_model` 백그라운드 스레드가 로드하므로 창은
    기다릴 이유가 없다.

    **별도 프로세스로 도는 이유**: 이 파일의 다른 테스트가 이미 torch를 로드해 두면
    같은 프로세스에서는 아무것도 증명하지 못한다. `start`는 no-op으로 막는다 —
    모델 로드 스레드가 뜨면 그게 비동기로 torch를 올려 판정이 흔들린다.
    """
    import subprocess

    code = (
        "import sys;"
        "import cocktail_engine, cocktail_ui;"
        "from PySide6.QtWidgets import QApplication;"
        "app = QApplication.instance() or QApplication([]);"
        "cocktail_engine.BackgroundController.start = lambda self: None;"
        "w = cocktail_ui.SettingsWindow();"
        "w._is_quitting = True; w.close();"
        "print('HEAVY', [m for m in ('torch', 'transformers') if m in sys.modules])"
    )
    p = subprocess.run([sys.executable, "-c", code],
                       cwd=os.path.dirname(os.path.abspath(__file__)),
                       capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=300)
    line = next((l for l in p.stdout.splitlines() if l.startswith("HEAVY")), None)
    assert line is not None, f"프로브 실패:\n{p.stdout[-800:]}\n{p.stderr[-800:]}"
    assert line == "HEAVY []", f"창이 뜨기 전에 무거운 모듈이 import됐다: {line}"


FULL_ONLY = {
    "test_opus_mt_actually_translates_english_to_korean",
    "test_long_paragraph_survives_to_display",   # 모델 로드 필요 (--full)
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="번역 모델을 실제로 로드하는 통합 테스트까지 실행")
    args = parser.parse_args()

    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failures = []
    skipped = 0
    for name, fn in tests:
        if name in FULL_ONLY and not args.full:
            skipped += 1
            continue
        try:
            fn()
            print(f"[OK]   {name}")
        except AssertionError as e:
            failures.append((name, str(e)))
            print(f"[FAIL] {name}: {e}")
        except Exception as e:
            failures.append((name, f"{type(e).__name__}: {e}"))
            print(f"[ERR]  {name}: {type(e).__name__}: {e}")

    total = len(tests) - skipped
    print(f"\n== {total - len(failures)}/{total} passed"
          + (f", {skipped} skipped (--full로 실행)" if skipped else "") + " ==")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

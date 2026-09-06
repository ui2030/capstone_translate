"""bench/make_fixtures.py — 채점용 시료(이미지 + 정답) 생성기.

`python bench/make_fixtures.py` 를 돌리면 `bench/fixtures/*.png` 와 `bench/truth/*.json` 이
**결정론적으로** 다시 만들어진다. 정답 텍스트는 "내가 그린 문자열 그대로"라서 사람이
받아쓸 필요가 없다 — 시료와 정답이 어긋날 수 없는 것이 이 방식의 유일한 목적이다.

원칙:
  - 실제 폰트(Segoe UI / Malgun Gothic / Microsoft YaHei / Segoe Fluent Icons)와
    실제 UI 크기(10~24px)만 쓴다. "합성이지만 렌더링 조건은 실물"이 요구사항이었다.
  - 문단 정답은 **렌더링 좌표 순서(top, left)** 로 저장한다. 채점기가 OCR 결과를 같은
    기준으로 정렬해 비교하므로 순서 불일치가 점수를 오염시키지 않는다.
  - 이미지에 그렸지만 정답에 넣지 않는 글자를 만들지 마라. 채점기는 "정답에 없는 텍스트"를
    전부 오인식(삽입 오류)으로 센다.
"""
import json
import os
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
FIX = ROOT / "fixtures"
TRUTH = ROOT / "truth"

FONTS = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
SEGOE = str(FONTS / "segoeui.ttf")
SEGOE_BD = str(FONTS / "segoeuib.ttf")
MALGUN = str(FONTS / "malgun.ttf")
YAHEI = str(FONTS / "msyh.ttc")
ICONS = str(FONTS / "SegoeIcons.ttf")


def font(path, size):
    return ImageFont.truetype(path, size)


class Sheet:
    """캔버스 + 문단 정답 수집기."""

    def __init__(self, w, h, bg=(255, 255, 255)):
        self.img = Image.new("RGB", (w, h), bg)
        self.d = ImageDraw.Draw(self.img)
        self.w, self.h = w, h
        self.blocks = []      # (top, left, text)
        self.forbidden = []   # 출력에 절대 나오면 안 되는 문자열(병음 등)

    def wrap(self, text, f, max_w):
        out, cur = [], ""
        for word in text.split():
            cand = (cur + " " + word).strip()
            if not cur or self.d.textlength(cand, font=f) <= max_w:
                cur = cand
            else:
                out.append(cur)
                cur = word
        if cur:
            out.append(cur)
        return out

    def para(self, text, x, y, f, max_w, fill=(17, 17, 17), leading=1.38, truth=True):
        """문단 하나를 그린다. 반환값은 다음 y."""
        lh = max(f.size + 2, int(round(f.size * leading)))
        for line in self.wrap(text, f, max_w):
            self.d.text((x, y), line, font=f, fill=fill)
            y += lh
        if truth:
            self.blocks.append((y, x, text))
        return y

    def line(self, text, x, y, f, fill=(17, 17, 17), truth=True):
        """줄바꿈 없는 한 줄(= 그 자체가 한 문단)."""
        self.d.text((x, y), text, font=f, fill=fill)
        if truth:
            self.blocks.append((y, x, text))
        return y + max(f.size + 2, int(round(f.size * 1.38)))

    def save(self, name, **meta):
        FIX.mkdir(parents=True, exist_ok=True)
        TRUTH.mkdir(parents=True, exist_ok=True)
        self.img.save(FIX / f"{name}.png")
        paragraphs = [t for _, _, t in sorted(self.blocks, key=lambda b: (b[0], b[1]))]
        doc = {"name": name, "paragraphs": paragraphs,
               "forbidden": self.forbidden, **meta}
        (TRUTH / f"{name}.json").write_text(
            json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  {name}.png  {self.w}x{self.h}  문단 {len(paragraphs)}개")


# --- 1. 글자 크기 계단 --------------------------------------------------------
LADDER_TEXT = (
    "The kettle on the far shelf had been whistling for a while before anyone "
    "noticed it. Marta lifted it off the heat and poured two cups without asking. "
    "Outside the rain had finally stopped and the street lamps were coming on one "
    "by one along the wet road."
)


def make_ladder():
    for px in (10, 12, 14, 18, 24):
        f = font(SEGOE, px)
        max_w = min(1160, px * 46)
        s = Sheet(max_w + 80, 40, (255, 255, 255))
        lines = s.wrap(LADDER_TEXT, f, max_w)
        height = 40 + len(lines) * max(px + 2, int(px * 1.38))
        s = Sheet(max_w + 80, height, (255, 255, 255))
        s.para(LADDER_TEXT, 40, 20, f, max_w)
        s.save(f"size_{px:02d}", lang="en", font_px=px, expect_display=True,
               role="criterion1",
               notes=f"같은 문장을 {px}px Segoe UI로 렌더. 문단 1개.")


# --- 2. 영어 문서 (삽화 옆 좁은 단) -------------------------------------------
def make_doc():
    s = Sheet(1120, 780, (252, 251, 248))
    title = font(SEGOE_BD, 26)
    body = font(SEGOE, 16)
    small = font(SEGOE, 12)

    y = s.line("Chapter Four: The Lamplighter's Hour", 60, 40, title)
    y = s.line("collected notes, page 41", 60, y + 4, small, fill=(110, 110, 110))
    y = s.para(
        "The road out of the village climbed slowly for two miles and then gave up "
        "climbing altogether. Travellers who had expected a view were disappointed, "
        "and travellers who had expected nothing were pleased, which is the usual "
        "arrangement in that part of the country.",
        60, y + 18, body, 1000)

    # 삽화 + 그 오른쪽의 좁은 단 — 문단 병합이 깨진 전력이 있는 배치.
    ill_top = y + 24
    s.d.rectangle([60, ill_top, 470, ill_top + 260], outline=(150, 150, 150), width=2,
                  fill=(238, 236, 230))
    s.d.ellipse([150, ill_top + 40, 380, ill_top + 200], outline=(120, 120, 120), width=3)
    s.d.line([265, ill_top + 200, 265, ill_top + 240], fill=(120, 120, 120), width=3)
    s.d.line([200, ill_top + 240, 330, ill_top + 240], fill=(120, 120, 120), width=3)
    s.line("Figure 4. the lamp on the hill road", 60, ill_top + 268, small,
           fill=(110, 110, 110))

    col_x, col_w = 510, 300
    cy = s.para(
        "He carried a short ladder and a can of oil, and he never spoke to anyone "
        "before the lamp was lit.",
        col_x, ill_top, body, col_w)
    cy = s.para(
        "Afterwards he would sit on the top step and watch the road, waiting for the "
        "hour when the light was no longer needed and he could put it out again.",
        col_x, cy + 20, body, col_w)

    y = max(ill_top + 300, cy + 20)
    s.para(
        "Nobody in the village could remember when the lamp had first been lit, and "
        "nobody thought to ask. It was simply part of the hill, like the wall and the "
        "two crooked pines beside it.",
        60, y + 20, body, 1000)
    s.save("doc_en_column", lang="en", font_px=16, expect_display=True,
           role="criterion1,2,3",
           notes="삽화(도형) 왼쪽 + 좁은 단 오른쪽. 문단 병합이 깨졌던 배치 재현.")


# --- 2b. 전체 화면 크기의 빽빽한 문서 (기준 4의 현실성) -----------------------
FULLHD_PARAS = [
    "The survey team spent three weeks measuring the old quay before anyone agreed "
    "on what the numbers meant. Half the stones had settled unevenly, and the rest "
    "had been replaced at some point without a record being kept.",
    "A second reading was taken in the spring, when the water was low enough to see "
    "the base of the wall. The difference between the two readings was smaller than "
    "expected, which pleased the engineers and disappointed the local paper.",
    "Funding was approved on the condition that the work would not interrupt the "
    "ferry service. That condition turned out to be the hardest part of the plan, "
    "because the only crane large enough had to stand where the ferry docks.",
    "The contractor proposed working at night. The council rejected the idea after "
    "three residents pointed out that the noise would carry across the whole bay, "
    "and the matter was referred back to the harbour committee.",
    "By August a compromise had been found. The crane would be mounted on a barge "
    "and moved twice a day, which cost more but kept the timetable intact for the "
    "whole of the summer season.",
    "Work is expected to finish before the first winter storms. The engineers are "
    "confident, the harbour master is not, and the fishing fleet has already been "
    "told to plan for two weeks of restricted access in November.",
]


def make_fullhd_doc():
    """1920x1080 빽빽 화면. 시료가 작으면 속도 점수가 실사용보다 낙관적으로 나온다."""
    s = Sheet(1920, 1080, (253, 253, 251))
    title = font(SEGOE_BD, 28)
    head = font(SEGOE_BD, 18)
    body = font(SEGOE, 16)
    small = font(SEGOE, 12)

    s.line("Harbour survey — interim report", 60, 40, title)
    s.line("prepared for the works committee, second draft", 60, 82, small,
           fill=(110, 110, 110))
    for col, x in enumerate((60, 700, 1340)):
        y = 130
        y = s.line(["Measurements", "Funding", "Schedule"][col], x, y, head)
        y += 10
        for text in FULLHD_PARAS[col * 2:col * 2 + 2]:
            y = s.para(text, x, y, body, 520) + 22
    s.line("All figures are provisional until the November reading is complete.",
           60, 1010, small, fill=(110, 110, 110))
    s.save("fullhd_doc", lang="en", font_px=16, expect_display=True,
           role="criterion4",
           notes="1920x1080 3단 빽빽 문서. 속도(기준 4)를 실화면 크기로 재기 위한 시료.")


# --- 3. 영어 UI / 툴팁 (조밀 + 아이콘) ----------------------------------------
def make_ui():
    s = Sheet(1000, 640, (32, 33, 36))
    menu = font(SEGOE, 12)
    label = font(SEGOE, 12)
    small = font(SEGOE, 11)
    icons = font(ICONS, 16)
    white = (232, 232, 232)
    dim = (170, 172, 176)

    s.d.rectangle([0, 0, 1000, 26], fill=(43, 44, 48))
    x = 12
    for item in ["File", "Edit", "Selection", "View", "Terminal", "Help"]:
        s.line(item, x, 6, menu, fill=white)
        x += int(s.d.textlength(item, font=menu)) + 22

    s.d.rectangle([0, 26, 1000, 60], fill=(38, 39, 43))
    x = 14
    for glyph, text in [("\ue74e", "Save"), ("\ue721", "Search"),
                        ("\ue713", "Settings"), ("\ue72c", "Refresh"),
                        ("\ue7ba", "Warnings")]:
        s.d.text((x, 34), glyph, font=icons, fill=(150, 190, 230))
        s.line(text, x + 22, 36, small, fill=white)
        x += 22 + int(s.d.textlength(text, font=small)) + 26

    s.d.rectangle([0, 60, 250, 640], fill=(37, 38, 42))
    y = 74
    for item in ["Explorer", "Open Editors", "Timeline", "Outline",
                 "Problems panel", "Source control", "Extensions"]:
        y = s.line(item, 16, y, label, fill=dim) + 8

    y = 74
    for key, value in [("Font size", "12 px"), ("Tab width", "4 spaces"),
                       ("Word wrap", "bounded"), ("Auto save", "after delay"),
                       ("Line numbers", "relative")]:
        s.line(key, 700, y, small, fill=dim)
        s.line(value, 860, y, small, fill=white)
        y += 26

    s.d.rectangle([300, 220, 650, 300], fill=(58, 60, 66), outline=(90, 92, 98))
    s.para("This action cannot be undone. The selected files will be moved to the "
           "recycle bin.", 314, 234, small, 320, fill=white)

    s.d.rectangle([0, 614, 1000, 640], fill=(43, 44, 48))
    s.line("Ready", 12, 620, small, fill=dim)
    s.line("Line 42, Column 8", 120, 620, small, fill=dim)
    s.line("Spaces: 4", 280, 620, small, fill=dim)
    s.save("ui_en_dense", lang="en", font_px=12, expect_display=True,
           role="criterion1,2,3",
           notes="다크 테마 조밀 UI. 아이콘 글리프(Segoe Fluent Icons)는 정답에 없음 "
                 "— 아이콘을 글자로 읽으면 삽입 오류로 잡힌다.")


# --- 4. 중국어 + 병음 ---------------------------------------------------------
ZH_ROWS = [
    ("nǐ hǎo, huān yíng shǐ yòng zhè ge yìng yòng", "你好，欢迎使用这个应用"),
    ("jīn tiān tiān qì hěn hǎo, wǒ men qù gōng yuán ba", "今天天气很好，我们去公园吧"),
    ("zhè běn shū shì wǒ péng you sòng gěi wǒ de lǐ wù", "这本书是我朋友送给我的礼物"),
]


def make_zh():
    s = Sheet(940, 620, (255, 255, 255))
    han = font(YAHEI, 30)
    pin = font(SEGOE, 13)
    body = font(YAHEI, 20)
    title = font(YAHEI, 22)

    s.line("第三课 · 日常用语", 60, 40, title)
    y = 100
    for pinyin, hanzi in ZH_ROWS:
        s.d.text((60, y), pinyin, font=pin, fill=(120, 120, 130))
        s.blocks.append((y + 18, 60, hanzi))
        s.d.text((60, y + 20), hanzi, font=han, fill=(17, 17, 17))
        s.forbidden.append(pinyin)
        y += 92

    s.para("学习中文需要每天练习。你可以先记住常用的词语，然后再练习写句子。",
           60, y + 10, body, 820)
    s.save("zh_pinyin", lang="zh", font_px=30, expect_display=True,
           role="criterion2,3",
           notes="한자 위 로마자 병음. 병음 줄은 forbidden — 자막으로 뜨면 오역 사고(PY-1).")


# --- 5. 한국어 화면 (아무것도 안 떠야 정상) -----------------------------------
def make_ko():
    s = Sheet(900, 560, (250, 250, 252))
    title = font(MALGUN, 20)
    body = font(MALGUN, 15)
    small = font(MALGUN, 13)

    s.line("환경 설정", 50, 40, title)
    y = s.para("화면에 보이는 글자를 자동으로 인식해서 번역합니다. 번역할 영역은 아래에서 "
               "직접 지정할 수 있습니다.", 50, 90, body, 780)
    y = s.line("번역 시작", 50, y + 24, body)
    y = s.line("영역 편집", 50, y + 8, body)
    y = s.line("최근에 사용한 언어 목록을 초기화합니다", 50, y + 24, small,
               fill=(90, 90, 95))
    y = s.line("저장하지 않고 닫기", 50, y + 20, small, fill=(90, 90, 95))
    s.line("준비 완료 · 모델이 메모리에 올라와 있습니다", 50, 500, small,
           fill=(90, 90, 95))
    s.save("ko_screen", lang="ko", font_px=15, expect_display=False,
           role="criterion2",
           notes="목표 언어(한국어)와 같은 화면 — 자막이 한 줄이라도 뜨면 오표시(SK-1 위반).")


# --- 6. 다중 문맥 화면 --------------------------------------------------------
def make_multi():
    s = Sheet(1280, 800, (245, 246, 248))
    head = font(SEGOE_BD, 18)
    body = font(SEGOE, 15)
    small = font(SEGOE, 13)

    # A: 기사 본문 (좌상)
    s.d.rectangle([40, 40, 620, 330], fill=(255, 255, 255), outline=(220, 222, 226))
    y = s.line("Harbour repairs begin next month", 60, 60, head)
    s.para("The council approved the plan on Tuesday after a short debate. Work on the "
           "outer wall will start once the fishing season ends, and the ferry timetable "
           "will change for six weeks.", 60, y + 10, body, 540)

    # B: 사이드바 목록 (우상) — 서로 무관한 짧은 항목들
    s.d.rectangle([660, 40, 1240, 330], fill=(255, 255, 255), outline=(220, 222, 226))
    y = s.line("Saved searches", 680, 60, head)
    y += 14
    for item in ["Weather station logs", "Ferry timetable archive",
                 "Harbour wall photographs", "Council meeting minutes",
                 "Tide tables for August"]:
        y = s.line(item, 680, y, small, fill=(60, 62, 68)) + 14

    # C: 채팅 (좌하) — 서로 다른 발화 3개. 세로 간격이 좁아 병합 함정.
    s.d.rectangle([40, 360, 620, 760], fill=(255, 255, 255), outline=(220, 222, 226))
    y = s.line("Team chat", 60, 380, head)
    y += 12
    for msg in ["Did the parts arrive this morning?",
                "Only the small ones, the rest ship on Friday.",
                "Fine, I will move the inspection to next week."]:
        y = s.para(msg, 60, y, body, 520) + 18

    # D: 알림 카드 (우하) — 제목 + 본문 2문단
    s.d.rectangle([660, 360, 1240, 760], fill=(255, 255, 255), outline=(220, 222, 226))
    y = s.line("Storage almost full", 680, 380, head)
    y = s.para("You have used 94 percent of the space on this device. Older backups will "
               "be removed automatically unless you change the setting.",
               680, y + 12, body, 540)
    s.para("Open the storage page to choose which folders keep their history.",
           680, y + 20, body, 540)
    s.save("multi_context", lang="en", font_px=15, expect_display=True,
           role="criterion3",
           notes="무관한 덩어리 4구획(기사/저장검색/채팅/알림) 12문단. 채팅 3발화와 "
                 "사이드바 5항목이 병합 함정.")


# --- 1b. 글자 크기 계단 (실화면 캡처) ----------------------------------------
def make_ladder_onscreen():
    """같은 문장을 **실제 화면에 띄워** ImageGrab으로 캡처한다.

    PIL 렌더는 회색조 안티에일리어싱이라 실물보다 깨끗하다 — 10px에서 오류율 0%가
    나오면 시료가 앱을 봐준 것이다. Windows 화면 글자는 ClearType(서브픽셀, 색 번짐)이라
    작은 글씨에서 훨씬 어렵다. 그 조건을 그대로 담으려면 화면을 찍는 수밖에 없다.
    """
    from PIL import ImageGrab
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QLabel

    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "0")
    qapp = QApplication.instance() or QApplication([])
    made = []
    for px in (10, 12, 14, 18, 24):
        max_w = min(1160, px * 46)
        probe = Sheet(10, 10)
        lines = probe.wrap(LADDER_TEXT, font(SEGOE, px), max_w)
        label = QLabel("\n".join(lines))
        label.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        label.setStyleSheet(
            f"QLabel {{ background: #ffffff; color: #111111; padding: 20px;"
            f" font-family: 'Segoe UI'; font-size: {px}px; }}")
        label.move(120, 120)
        label.show()
        label.raise_()
        # 창이 실제로 그려지기 전에 찍으면 검은 이미지가 나온다(실측). 흰 배경이
        # 보일 때까지 이벤트 루프를 돌리며 재시도한다.
        img = None
        for _ in range(30):
            for _ in range(10):
                qapp.processEvents()
            time.sleep(0.1)
            g = label.frameGeometry()
            shot = ImageGrab.grab(
                bbox=(g.x(), g.y(), g.x() + g.width(), g.y() + g.height()),
                all_screens=False)
            if min(shot.convert("L").getextrema()) < 100 < max(shot.convert("L").getextrema()):
                img = shot
                break
        if img is None:
            raise RuntimeError(f"{px}px 실화면 캡처 실패 (창이 그려지지 않음)")
        label.close()
        qapp.processEvents()
        name = f"screen_{px:02d}"
        FIX.mkdir(parents=True, exist_ok=True)
        img.save(FIX / f"{name}.png")
        (TRUTH / f"{name}.json").write_text(json.dumps(
            {"name": name, "paragraphs": [LADDER_TEXT], "forbidden": [],
             "lang": "en", "font_px": px, "expect_display": True,
             "role": "criterion1",
             "notes": f"같은 문장을 실제 화면(ClearType)에 {px}px로 띄워 ImageGrab 캡처."},
            ensure_ascii=False, indent=2), encoding="utf-8")
        made.append(name)
        print(f"  {name}.png  {img.width}x{img.height}  실화면 캡처")
    return made


def main():
    print("시료 생성:")
    make_ladder()
    make_doc()
    make_fullhd_doc()
    make_ui()
    make_zh()
    make_ko()
    make_multi()
    try:
        make_ladder_onscreen()
    except Exception as e:
        print(f"  [건너뜀] 실화면 계단 시료 생성 실패: {type(e).__name__}: {e}")
    print(f"완료 → {FIX}")


if __name__ == "__main__":
    main()

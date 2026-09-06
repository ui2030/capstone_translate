# 아이콘 가이드 (Phase 6)

> Cocktail의 시각 정체성. 트레이 아이콘 / .exe 아이콘 / 인스톨러 아이콘.

---

## 1. 필요한 형식

| 위치 | 파일 | 크기 |
|---|---|---|
| `.exe` 아이콘 | `assets/icon.ico` | 256×256 (멀티 사이즈 ico: 16/32/48/64/128/256) |
| 트레이 아이콘 | `assets/tray.png` (또는 .ico의 16×16 슬라이스) | 16×16, 32×32 |
| 인스톨러 헤더 | `assets/installer.bmp` | 164×314 (Inno Setup) |
| 인스톨러 작은 아이콘 | `assets/installer-small.bmp` | 55×58 |
| Tesseract/PaddleOCR 미설치 fallback | (코드 내 임시 — 빨간 사각형) | 코드 내장 |

---

## 2. 디자인 원칙 (비전 V-1 반영)

- **모국어 = 따뜻함**: 차가운 파란색보다 따뜻한 색조 (오렌지/레드/금색)
- **번역 = 순환**: 화살표 또는 양방향 모티브
- **오버레이 = 투명도**: 반투명 레이어 또는 그림자

추천 스케치:
- 둥근 사각형 위에 한글 자모(예: ㄱ) + 라틴 글자(A) 페어
- 또는 칵테일 잔(이름과 일치) + 글자

---

## 3. 변환 워크플로우

### PNG → ICO
```bash
# Pillow 사용 (이미 의존성에 있음)
python -c "from PIL import Image; Image.open('assets/icon-256.png').save('assets/icon.ico', sizes=[(16,16),(32,32),(48,48),(64,64),(128,128),(256,256)])"
```

### Inno Setup BMP
```bash
# 24-bit BMP로 저장
python -c "from PIL import Image; Image.open('assets/installer-164x314.png').convert('RGB').save('assets/installer.bmp')"
```

---

## 4. 코드 통합

### 메인 윈도우 아이콘 (코드)
```python
# cocktail_ui.py — SettingsWindow.__init__
icon_path = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "assets", "icon.ico")
if os.path.exists(icon_path):
    self.setWindowIcon(QIcon(icon_path))
```

### 트레이 아이콘 (코드)
현재 임시: `QPixmap(16, 16)` + `fill(QColor(220,60,60))`.
교체:
```python
icon_path = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "assets", "tray.png")
if os.path.exists(icon_path):
    icon = QIcon(icon_path)
else:
    pm = QPixmap(16, 16); pm.fill(QColor(220, 60, 60))
    icon = QIcon(pm)
```

### PyInstaller .exe 아이콘
`build.spec`:
```python
exe = EXE(
    ...,
    icon="assets/icon.ico",
)
```

### Inno Setup 인스톨러
`installer.iss`:
```pascal
[Setup]
SetupIconFile=assets\icon.ico
WizardImageFile=assets\installer.bmp
WizardSmallImageFile=assets\installer-small.bmp
```

---

## 5. 임시 placeholder (디자이너 확보 전)

배포 전엔 디자이너에 의뢰 권장. 단, 임시로 빠르게 만들고 싶으면:

- **emoji-based**: 🍸 같은 이모지 → PNG 변환 (저작권 무관 emoji 폰트 사용: Noto Color Emoji, Apache 2.0)
- **자체 그리기**: matplotlib/PIL로 단순 도형
- **무료 아이콘 사이트**: Flaticon, Icons8 (라이선스 확인 필수 — CC-BY 출처 표기 의무 등)

---

## 6. 라이선스 주의

- 외부 아이콘 사용 시 라이선스 확인 (CC-BY/CC0/Apache/MIT 등)
- 상용 배포 시 **귀속 표기**(attribution) 필요한 라이선스는 README에 별도 섹션
- 본 프로젝트 자체 아이콘 디자인 시 작업물의 저작권/라이선스 결정 필요 (MIT/CC0 권장)

"""
인스톨러 빌드 전에 traineddata 파일을 ./tessdata/ 로 다운로드.
사용자 인스톨러에서 컴포넌트로 선택 설치 (eng 필수, 나머지 선택).

사용법:
    python download_tessdata.py

기본 소스: tessdata_fast (속도/정확도 균형). _best는 더 정확하지만 큼.
"""
import os
import sys
import urllib.request

# osd는 받지 않는다 — OSD script 판별(UX-2)이 SL-1에서 퇴역해 앱이 쓰지 않는다.
# 추가 언어 후보:
# - chi_tra: 중국어 번체 (대만/홍콩)
# - rus: 러시아어 (Cyrillic)
# - ara: 아랍어
# - heb: 히브리어
# - hin: 힌디어 (Devanagari)
# - tha: 태국어
# - ell: 그리스어
LANGS = [
    "eng", "kor", "jpn", "chi_sim", "fra", "deu",
    "chi_tra", "rus", "ara",            # 추가 script 커버 (선택)
    # "heb", "hin", "tha", "ell",       # 매우 자주 쓰진 않음 — 필요 시 주석 해제
]
BASE = "https://github.com/tesseract-ocr/tessdata_fast/raw/main"
OUT_DIR = "tessdata"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for lang in LANGS:
        fname = f"{lang}.traineddata"
        dest = os.path.join(OUT_DIR, fname)
        if os.path.exists(dest):
            print(f"  - {fname} (이미 존재, 건너뜀)")
            continue
        url = f"{BASE}/{fname}"
        print(f"  - 다운로드 중: {url}")
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as e:
            print(f"    실패: {e}")
            sys.exit(1)
    print(f"\n완료. {OUT_DIR}/ 에 저장됨.")

if __name__ == "__main__":
    main()

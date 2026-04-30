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

LANGS = ["eng", "kor", "jpn", "chi_sim", "fra", "deu"]
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

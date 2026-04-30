"""
Phase 2 PoC — Windows UI Automation 탐색 스크립트.

목적:
    UIA로 어디까지 OCR 없이 텍스트를 직접 읽을 수 있는지 검증.
    이 결과가 좋으면 본 앱 코드에 UIA + OCR 하이브리드를 통합 (Phase 3).

설치:
    pip install -r requirements-dev.txt
    (또는 직접: pip install uiautomation)

사용법:
    python uia_explore.py            # 기본: 5초 후 활성 창 트리 출력
    python uia_explore.py --depth 6  # 트리 깊이 제한
    python uia_explore.py --watch    # 1초마다 활성 창 새로 출력 (Ctrl+C로 종료)

비고:
    - uiautomation 라이선스: Apache 2.0 (상용 OK).
    - 본 스크립트는 본 앱과 분리됨. 본 앱은 uiautomation 의존성 없이 동작.
    - VISION_AND_ROADMAP.md Phase 2 참고.
"""
import sys
import time
import argparse

try:
    import uiautomation as auto  # type: ignore
except ImportError:
    print("[ERROR] uiautomation 패키지가 필요합니다.")
    print("        pip install -r requirements-dev.txt")
    print("        또는 pip install uiautomation")
    sys.exit(1)


def walk(elem, depth: int = 0, max_depth: int = 8, total_limit: int = 500):
    """UIA 트리를 순회하면서 이름이 있는 요소를 출력. 너무 깊거나 많으면 중단."""
    counter = {"n": 0}

    def _walk(e, d):
        if d > max_depth or counter["n"] >= total_limit:
            return
        try:
            name = (e.Name or "").strip()
            ctype = e.ControlTypeName
        except Exception:
            return
        if name:
            print(f"{'  ' * d}[{ctype:20s}] {name[:120]}")
            counter["n"] += 1
        try:
            children = e.GetChildren()
        except Exception:
            children = []
        for c in children:
            _walk(c, d + 1)

    _walk(elem, depth)
    if counter["n"] >= total_limit:
        print(f"... (출력 한도 {total_limit}개에 도달, 일부 생략)")


def snapshot(max_depth: int):
    win = auto.GetForegroundControl()
    if not win:
        print("[WARN] 활성 창을 찾을 수 없습니다.")
        return
    try:
        title = win.Name or "(제목 없음)"
        cls = win.ClassName or "(?)"
    except Exception:
        title, cls = "(접근 실패)", "(?)"
    print(f"활성 창: '{title}'  (ClassName={cls})")
    print("-" * 80)
    walk(win, max_depth=max_depth)
    print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description="Phase 2 UIA PoC")
    parser.add_argument("--depth", type=int, default=8, help="트리 최대 깊이 (default 8)")
    parser.add_argument("--delay", type=float, default=5.0, help="시작 전 지연 초 (default 5)")
    parser.add_argument("--watch", action="store_true", help="1초마다 활성 창 트리 재출력")
    args = parser.parse_args()

    if not args.watch:
        print(f"{args.delay:.0f}초 후 활성 창의 UIA 트리를 출력합니다. 탐색할 창에 포커스를 주세요.")
        for i in range(int(args.delay), 0, -1):
            print(f"  {i}...", end="\r", flush=True)
            time.sleep(1)
        print()
        snapshot(max_depth=args.depth)
        print("끝.")
    else:
        print("watch 모드 — Ctrl+C 로 종료. 1초마다 활성 창 트리 재출력.")
        try:
            while True:
                print("\n" + "=" * 80)
                snapshot(max_depth=args.depth)
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n종료.")


if __name__ == "__main__":
    main()

"""
swpc_alert_run.py
================
fsm_output/ 아래 모든 fsm_event_*.csv 를 스캔하여
SWPC(전자 경보 카탈로그)에 일괄 매칭한다.
swpc_alert_espe_match.py 가 있는 폴더에서 실행.

파일명에서 k / onset / peak 자동 추출 후 매칭 스크립트에 전달.
  fsm_event_quietoff_mad_w30_k10_on0.5_pk2.csv -> --k 10 --onset 0.5 --peak 2

사용:
  python swpc_alert_run.py          # 실행
  python swpc_alert_run.py --dry    # 명령만 출력 (실행 안 함)
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

# ── 경로 설정 (이 파일 기준; 본인 구조에 맞게 수정) ───────────────
FSM_OUTPUT_DIR = Path("../count_FSM/fsm_output")
COUNT_DIR      = Path("../KSEM_count/ksem_cache_parquet")
MATCH_SCRIPT   = Path("./swpc_alert_espe_match.py")
CATALOG        = Path("./espe_cache_parquet")
LOG_TAG        = "swpc"
# ─────────────────────────────────────────────────────────────────


def extract_param(stem: str, pat: str):
    """파일명 stem에서 _<pat><숫자> 추출. pat = 'k' | 'on' | 'pk'."""
    m = re.search(rf"_{pat}([0-9]+(?:\.[0-9]+)?)", stem)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser(description=f"{LOG_TAG} 일괄 매칭")
    ap.add_argument("--dry", action="store_true", help="명령만 출력, 실행 안 함")
    args = ap.parse_args()

    events = sorted(FSM_OUTPUT_DIR.rglob("fsm_event_*.csv"))
    if not events:
        print(f"[{LOG_TAG}] fsm_event_*.csv 를 찾지 못함: {FSM_OUTPUT_DIR}")
        sys.exit(1)

    print(f"[{LOG_TAG}] {len(events)} 개 event CSV -> {LOG_TAG.upper()} 매칭")
    print("-" * 60)

    for ev in events:
        stem = ev.stem
        k  = extract_param(stem, "k")
        on = extract_param(stem, "on")
        pk = extract_param(stem, "pk")

        if not (k and on and pk):
            print(f"[{LOG_TAG}] SKIP (파라미터 추출 실패): {ev.name}")
            continue

        print(f"\n### {ev.name}  (k={k} onset={on} peak={pk})")
        cmd = [
            sys.executable, str(MATCH_SCRIPT),
            "--events", str(ev),
            "--catalog", str(CATALOG),
            "--count-dir", str(COUNT_DIR),
            "--k", k, "--onset", on, "--peak", pk,
        ]

        if args.dry:
            print(" ".join(cmd))
        else:
            print(">>> " + " ".join(cmd))
            subprocess.run(cmd, check=False)

    print(f"\n[{LOG_TAG}] 완료.")


if __name__ == "__main__":
    main()

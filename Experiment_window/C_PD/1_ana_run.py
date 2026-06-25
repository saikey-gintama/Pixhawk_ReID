"""
1_ana_run.py
============
통합 ana 실행 러너 (의존성 최상류).

detector(gk2a / metop03 / noaa19)를 받아 ana1~3 + ana_event 4개를 순서대로 실행.
ana_event가 생성한 noaa_spe_event_count_stats.csv 를 2_fsm_run의 blc1_fixed가 읽는다.

  --stage all    (기본) ana1 → ana2 → ana3 → ana_event 전부 실행
  --stage plots  ana1 + ana2 + ana3 만 (그림·CSV 분석 산출물)
  --stage event  ana_event 만 (2_fsm const-csv 생성, 빠른 재실행용)

GK2A   ana1/2/3 : --cache --io(기본 ksem_io) --out
       ana_event: --cache --out
POES   ana1/2/3 : --io --cache --out
       ana_event: --count --out   ← GK2A의 --cache 와 인자명이 다름, 주의.

ana2는 카탈로그 오버레이(--spe/--espe)를 포함해서 호출한다(선택 인자지만 표준 포함).
--dry : 명령만 출력, 실행 안 함.

사용:
  python 1_ana_run.py --detector gk2a   [--stage all] [--dry]
  python 1_ana_run.py --detector metop03 --stage event --dry
  python 1_ana_run.py --detector noaa19
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent   # C_PD/

# ── ana 스크립트 경로 ────────────────────────────────────────────────
_GK2A_DIR = HERE / "GK2A"
_POES_DIR = HERE / "POES"

_SCRIPTS = {
    "gk2a": {
        "ana1":  _GK2A_DIR / "ana1_channel_stats.py",
        "ana2":  _GK2A_DIR / "ana2_condition_profile.py",
        "ana3":  _GK2A_DIR / "ana3_coord_dependence.py",
        "event": _GK2A_DIR / "ana_event_count_profile.py",
    },
    "poes": {
        "ana1":  _POES_DIR / "ana1_channel_stats_poes.py",
        "ana2":  _POES_DIR / "ana2_condition_profile_poes.py",
        "ana3":  _POES_DIR / "ana3_coord_dependence_poes.py",
        "event": _POES_DIR / "ana_event_count_profile_poes.py",
    },
}

# ── 출력 경로 매핑 ────────────────────────────────────────────────────
_OUT_DIR = {
    "gk2a":    HERE / "GK2A"  / "KSEM_count"   / "gk2a_output"    / "1_ana",
    "metop03": HERE / "POES"  / "MetOp03_count" / "metop03_output" / "1_ana",
    "noaa19":  HERE / "POES"  / "NOAA19_count"  / "noaa19_output"  / "1_ana",
}

# ── 입력 (cache / io) ────────────────────────────────────────────────
_CACHE = {
    "gk2a":    HERE / "GK2A"  / "KSEM_count"   / "ksem_cache_parquet",
    "metop03": HERE / "POES"  / "MetOp03_count" / "poes_metop03_cache_parquet",
    "noaa19":  HERE / "POES"  / "NOAA19_count"  / "poes_noaa19_cache_parquet",
}
_IO = {
    "gk2a":    "ksem_io",
    "metop03": "poes_metop03_io",
    "noaa19":  "poes_noaa19_io",
}

# ── 카탈로그 경로 (ana2 오버레이용) ──────────────────────────────────
_SPE_CAT  = HERE / "NOAA_GOES"  / "noaa_goes_spe_cache_parquet"
_ESPE_CAT = HERE / "SWPC_Alert" / "swpc_espe_cache_parquet"


def _sgroup(detector: str) -> str:
    return "gk2a" if detector == "gk2a" else "poes"


def _build_plots_cmds(detector: str, out_dir: Path) -> list[tuple[str, list[str]]]:
    """ana1 / ana2 / ana3 명령 목록 반환."""
    sg    = _sgroup(detector)
    cache = str(_CACHE[detector])
    out   = str(out_dir)
    cmds: list[tuple[str, list[str]]] = []

    # ana1
    cmd = [sys.executable, str(_SCRIPTS[sg]["ana1"]),
           "--cache", cache, "--out", out]
    if detector != "gk2a":
        cmd[2:2] = ["--io", _IO[detector]]   # --io 를 --cache 앞에 삽입
    cmds.append(("ana1", cmd))

    # ana2 (카탈로그 오버레이 포함)
    cmd = [sys.executable, str(_SCRIPTS[sg]["ana2"]),
           "--cache", cache, "--out", out,
           "--spe",  str(_SPE_CAT),
           "--espe", str(_ESPE_CAT)]
    if detector != "gk2a":
        cmd[2:2] = ["--io", _IO[detector]]
    cmds.append(("ana2", cmd))

    # ana3
    cmd = [sys.executable, str(_SCRIPTS[sg]["ana3"]),
           "--cache", cache, "--out", out]
    if detector != "gk2a":
        cmd[2:2] = ["--io", _IO[detector]]
    cmds.append(("ana3", cmd))

    return cmds


def _build_event_cmd(detector: str, out_dir: Path) -> tuple[str, list[str]]:
    """ana_event_count_profile 명령 반환.
    GK2A: --cache / --out
    POES: --count / --out  (인자명 다름)
    """
    sg    = _sgroup(detector)
    cache = str(_CACHE[detector])
    out   = str(out_dir)

    if detector == "gk2a":
        cmd = [sys.executable, str(_SCRIPTS["gk2a"]["event"]),
               "--cache", cache, "--out", out]
    else:
        # POES ana_event는 --count (캐시 parquet 디렉토리)
        cmd = [sys.executable, str(_SCRIPTS["poes"]["event"]),
               "--count", cache, "--out", out]

    return ("ana_event", cmd)


def main():
    ap = argparse.ArgumentParser(description="통합 ana 실행 러너")
    ap.add_argument("--detector", required=True, choices=["gk2a", "metop03", "noaa19"])
    ap.add_argument("--stage",    default="all",
                    choices=["all", "plots", "event"],
                    help="all=ana1~3+event(기본), plots=ana1~3만, event=ana_event만")
    ap.add_argument("--dry", action="store_true", help="명령만 출력, 실행 안 함")
    args = ap.parse_args()

    det     = args.detector
    out_dir = _OUT_DIR[det]

    # const-csv 정렬 확인용 경로 (ana_event 출력 기대치 = 2_fsm_run 참조 경로)
    const_csv_expected = out_dir / "noaa_spe_event_count_stats.csv"

    print(f"[1_ana] detector={det}  stage={args.stage}")
    print(f"[1_ana] out_dir : {out_dir}")
    print(f"[1_ana] const-csv 기대 경로 (2_fsm 참조): {const_csv_expected}")

    # 실행할 명령 목록 구성
    cmds: list[tuple[str, list[str]]] = []
    if args.stage in ("all", "plots"):
        cmds.extend(_build_plots_cmds(det, out_dir))
    if args.stage in ("all", "event"):
        cmds.append(_build_event_cmd(det, out_dir))

    print(f"\n[1_ana] 명령 ({len(cmds)}개):")
    for label, cmd in cmds:
        print(f"  [{label}]")
        print("    " + " ".join(str(t) for t in cmd))

    if args.dry:
        print("\n[1_ana] --dry: 실행 생략.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    for label, cmd in cmds:
        print(f"\n>>> {label}")
        subprocess.run(cmd, check=False)

    print("\n[1_ana] 완료.")


if __name__ == "__main__":
    main()

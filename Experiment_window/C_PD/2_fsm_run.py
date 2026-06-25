"""
2_fsm_run.py
============
통합 FSM 실행 러너.

detector(gk2a / metop03 / noaa19)를 받아 해당 count_FSM 디렉터리의
fsm_count_spe_*.py 를 glob해서 일괄 실행한다.

  GK2A  : --out 만 전달 (--io/--cache 는 스크립트 내부 하드코딩).
  POES  : --io --cache --out 모두 전달. --no-geo 미전달(geo 태깅 on).
  blc1_fixed(_poes): --mode const --const-csv <1_ana 출력 CSV> 추가.
                     CSV 미존재 시 WARNING 출력 후 명령 포함 유지.

각 FSM은 내부에서 --out/<runtag>/ 하위폴더를 생성한다.

사용:
  python 2_fsm_run.py --detector gk2a   [--dry]
  python 2_fsm_run.py --detector metop03 [--dry]
  python 2_fsm_run.py --detector noaa19  [--dry]
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent   # C_PD/

# ── FSM 스크립트 디렉터리 ────────────────────────────────────────────
_GK2A_FSM_DIR = HERE / "GK2A" / "count_FSM"
_POES_FSM_DIR = HERE / "POES" / "count_FSM"

# ── 출력 경로 매핑 (각 FSM에 넘길 --out 부모; FSM이 내부에서 runtag 하위폴더 생성)
_OUT_DIR = {
    "gk2a":    HERE / "GK2A"  / "KSEM_count"   / "gk2a_output"    / "2_fsm",
    "metop03": HERE / "POES"  / "MetOp03_count" / "metop03_output" / "2_fsm",
    "noaa19":  HERE / "POES"  / "NOAA19_count"  / "noaa19_output"  / "2_fsm",
}

# ── POES 입력 (io 모듈명, cache 경로) ──────────────────────────────
_POES_IO = {
    "metop03": (
        "poes_metop03_io",
        HERE / "POES" / "MetOp03_count" / "poes_metop03_cache_parquet",
    ),
    "noaa19": (
        "poes_noaa19_io",
        HERE / "POES" / "NOAA19_count" / "poes_noaa19_cache_parquet",
    ),
}

# ── blc1_fixed const-csv 경로 (1_ana 결과 기준; 1_ana 미실행 시 미존재)
_CONST_CSV = {
    "gk2a":    HERE / "GK2A"  / "KSEM_count"   / "gk2a_output"    / "1_ana" / "noaa_spe_event_count_stats.csv",
    "metop03": HERE / "POES"  / "MetOp03_count" / "metop03_output" / "1_ana" / "noaa_spe_event_count_stats.csv",
    "noaa19":  HERE / "POES"  / "NOAA19_count"  / "noaa19_output"  / "1_ana" / "noaa_spe_event_count_stats.csv",
}


def _fsm_dir(detector: str) -> Path:
    return _GK2A_FSM_DIR if detector == "gk2a" else _POES_FSM_DIR


def _is_blc1_fixed(script: Path) -> bool:
    return "blc1_fixed" in script.name


def build_cmd(script: Path, detector: str, out_dir: Path) -> list[str]:
    cmd = [sys.executable, str(script)]

    if detector != "gk2a":
        io_mod, cache = _POES_IO[detector]
        cmd += ["--io", io_mod, "--cache", str(cache)]

    cmd += ["--out", str(out_dir)]

    if _is_blc1_fixed(script):
        csv_path = _CONST_CSV[detector]
        if not csv_path.exists():
            print(f"  WARNING: --const-csv 미존재 (1_ana 먼저 실행 필요) → {csv_path}")
        cmd += ["--mode", "const", "--const-csv", str(csv_path)]

    return cmd


def main():
    ap = argparse.ArgumentParser(description="통합 FSM 실행 러너")
    ap.add_argument("--detector", required=True, choices=["gk2a", "metop03", "noaa19"])
    ap.add_argument("--dry", action="store_true", help="명령만 출력, 실행 안 함")
    args = ap.parse_args()

    det = args.detector
    fsm_scripts = sorted(_fsm_dir(det).glob("fsm_count_spe_*.py"))
    out_parent  = _OUT_DIR[det]

    if not fsm_scripts:
        raise SystemExit(f"[2_fsm] ERROR: FSM 스크립트 없음 → {_fsm_dir(det)}")

    print(f"[2_fsm] detector={det}  FSM {len(fsm_scripts)}개")
    print(f"[2_fsm] out_parent : {out_parent}")

    cmds: list[tuple[str, list[str]]] = []
    for script in fsm_scripts:
        cmd = build_cmd(script, det, out_parent)
        cmds.append((script.name, cmd))

    print(f"\n[2_fsm] 명령 ({len(cmds)}개):")
    for name, cmd in cmds:
        print(f"  [{name}]")
        print("    " + " ".join(str(t) for t in cmd))

    if args.dry:
        print("\n[2_fsm] --dry: 실행 생략.")
        return

    for name, cmd in cmds:
        print(f"\n>>> {name}")
        subprocess.run(cmd, check=False)

    print("\n[2_fsm] 완료.")


if __name__ == "__main__":
    main()

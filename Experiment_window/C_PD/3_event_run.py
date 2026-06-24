"""
3_event_run.py
==============
통합 이벤트 매칭 러너 (run_matcher.py 승격판).

detector × catalog 조합으로 FSM 출력 CSV ↔ 카탈로그를 매칭한다.

run_matcher.py 대비 변경점:
  --fsm-dir  : FSM 출력 폴더를 인자로. 기본값은 detector별 표준 fsm2_output.
               STEP 1에서 FSM --out이 생겨 임의 경로의 FSM 결과도 매칭 가능.
  출력 경로  : {detector}_output/3_event/<catalog>_<YYYYMMDD>/ (날짜 스탬프 기본값).
               기존 *_match_output, *_match_output_* 디렉터리를 절대 덮지 않는다.
  --out      : 명시적으로 출력 경로를 지정할 때만 사용.

동작 (run_matcher 현행 유지):
  GK2A  : 매처를 --events-dir <fsm_dir>로 1회 호출 (매처 내부에서 CSV 처리).
  POES  : fsm_dir/**fsm_<kind>_*.csv 를 glob 후 CSV마다 개별 호출.
  --dry : 실행 없이 명령만 출력. POES에 CSV가 없으면 경고 후 경로 정보만 출력.

사용:
  python 3_event_run.py --detector gk2a   --catalog noaa [--kind onset] [--dry]
  python 3_event_run.py --detector metop03 --catalog swpc [--fsm-dir custom/path]
  python 3_event_run.py --detector noaa19  --catalog noaa --out custom_out/

  # 전체 6조합 dry
  for det in gk2a metop03 noaa19:
    for cat in noaa swpc:
      python 3_event_run.py --detector $det --catalog $cat --dry
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent   # C_PD/

# ── 공통 경로 테이블 ─────────────────────────────────────────────────
_GK2A_COUNT   = HERE / "GK2A"  / "KSEM_count"  / "ksem_cache_parquet"
_NOAA_CAT     = HERE / "NOAA_GOES" / "noaa_goes_spe_cache_parquet"
_SWPC_CAT     = HERE / "SWPC_Alert" / "swpc_espe_cache_parquet"
_NOAA_MATCH   = HERE / "GK2A"  / "event_MATCHER" / "noaa_goes_spe_match.py"
_SWPC_MATCH   = HERE / "GK2A"  / "event_MATCHER" / "swpc_alert_espe_match.py"
_NOAA_MATCH_P = HERE / "POES"  / "event_MATCHER" / "noaa_goes_spe_match_poes.py"
_SWPC_MATCH_P = HERE / "POES"  / "event_MATCHER" / "swpc_alert_espe_match_poes.py"
_NOAA_IO      = HERE / "NOAA_GOES"  / "noaa_goes_spe_io"
_SWPC_IO      = HERE / "SWPC_Alert" / "swpc_alert_espe_io"


def _poes_sat_dir(detector: str) -> str:
    """metop03 → 'MetOp03_count', noaa19 → 'NOAA19_count'."""
    return {"metop03": "MetOp03_count", "noaa19": "NOAA19_count"}[detector]


def _default_fsm_dir(detector: str) -> Path:
    """detector별 기본 FSM 출력 폴더."""
    if detector == "gk2a":
        return HERE / "GK2A" / "count_FSM" / "fsm2_output"
    return HERE / "POES" / _poes_sat_dir(detector) / "fsm2_output"


def _default_out_dir(detector: str, catalog: str) -> Path:
    """detector별 기본 매칭 결과 폴더.
    {detector}_output/3_event/<catalog>_<YYYYMMDD>/
      GK2A    : GK2A/KSEM_count/gk2a_output/3_event/<catalog>_<stamp>/
      MetOp03 : POES/MetOp03_count/metop03_output/3_event/<catalog>_<stamp>/
      NOAA19  : POES/NOAA19_count/noaa19_output/3_event/<catalog>_<stamp>/
    """
    stamp = date.today().strftime("%Y%m%d")
    sub   = f"{catalog}_{stamp}"
    if detector == "gk2a":
        return HERE / "GK2A" / "KSEM_count" / "gk2a_output" / "3_event" / sub
    sat  = _poes_sat_dir(detector)                    # "MetOp03_count" | "NOAA19_count"
    name = detector                                    # "metop03" | "noaa19"
    return HERE / "POES" / sat / f"{name}_output" / "3_event" / sub


def build_gk2a_cmd(catalog: str, kind: str,
                   fsm_dir: Path, out_dir: Path) -> list[str]:
    script   = _NOAA_MATCH if catalog == "noaa" else _SWPC_MATCH
    cat_path = _NOAA_CAT   if catalog == "noaa" else _SWPC_CAT
    return [
        sys.executable, str(script),
        "--events-dir", str(fsm_dir),
        "--kind",       kind,
        "--catalog",    str(cat_path),
        "--out",        str(out_dir),
        "--count-dir",  str(_GK2A_COUNT),
    ]


def build_poes_cmds(detector: str, catalog: str, kind: str,
                    fsm_dir: Path, out_dir: Path) -> list[list[str]]:
    script   = _NOAA_MATCH_P if catalog == "noaa" else _SWPC_MATCH_P
    cat_path = _NOAA_CAT     if catalog == "noaa" else _SWPC_CAT
    io_path  = _NOAA_IO      if catalog == "noaa" else _SWPC_IO
    csvs     = sorted(fsm_dir.rglob(f"fsm_{kind}_*.csv")) if fsm_dir.exists() else []
    return [
        [
            sys.executable, str(script),
            "--events",  str(csv),
            "--catalog", str(cat_path),
            "--spe-io",  str(io_path),
            "--out",     str(out_dir),
        ]
        for csv in csvs
    ]


def main():
    ap = argparse.ArgumentParser(description="detector × catalog 통합 이벤트 매칭 러너")
    ap.add_argument("--detector", required=True, choices=["gk2a", "metop03", "noaa19"])
    ap.add_argument("--catalog",  required=True, choices=["noaa", "swpc"])
    ap.add_argument("--kind",    default="onset", choices=["onset", "event"],
                    help="FSM 출력 종류 (default onset)")
    ap.add_argument("--fsm-dir", default=None,
                    help="FSM 출력 폴더 (기본: detector별 표준 fsm2_output)")
    ap.add_argument("--out",     default=None,
                    help="매칭 결과 출력 폴더 "
                         "(기본: {detector}_output/3_event/<catalog>_<YYYYMMDD>)")
    ap.add_argument("--dry",     action="store_true", help="명령만 출력, 실행 안 함")
    args = ap.parse_args()

    fsm_dir = Path(args.fsm_dir) if args.fsm_dir else _default_fsm_dir(args.detector)
    out_dir = Path(args.out)     if args.out     else _default_out_dir(args.detector, args.catalog)

    print(f"[3_event] detector={args.detector}  catalog={args.catalog}  kind={args.kind}")
    print(f"[3_event] fsm_dir : {fsm_dir}")
    print(f"[3_event] out_dir : {out_dir}")

    if args.detector == "gk2a":
        cmd = build_gk2a_cmd(args.catalog, args.kind, fsm_dir, out_dir)
        print(f"[3_event] 명령 (1개):")
        print("  " + " ".join(cmd))
        if args.dry:
            print("[3_event] --dry: 실행 생략.")
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        print("\n>>> " + " ".join(cmd))
        subprocess.run(cmd, check=False)

    else:
        cmds = build_poes_cmds(args.detector, args.catalog, args.kind, fsm_dir, out_dir)
        if not cmds:
            print(f"[3_event] WARNING: fsm_{args.kind}_*.csv 없음 → {fsm_dir}")
            print(f"[3_event] 매처 스크립트 : "
                  f"{_NOAA_MATCH_P if args.catalog == 'noaa' else _SWPC_MATCH_P}")
            if args.dry:
                print("[3_event] --dry: CSV 없어 명령 생략 (FSM 먼저 실행 필요).")
                return
            raise SystemExit(f"[3_event] ERROR: CSV 없음 — FSM 먼저 실행하세요 → {fsm_dir}")
        print(f"[3_event] {len(cmds)}개 CSV → 개별 호출:")
        for cmd in cmds:
            print("  " + " ".join(cmd))
        if args.dry:
            print("[3_event] --dry: 실행 생략.")
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        for cmd in cmds:
            print("\n>>> " + " ".join(cmd))
            subprocess.run(cmd, check=False)

    print("\n[3_event] 완료.")


if __name__ == "__main__":
    main()

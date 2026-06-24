"""
run_matcher.py
==============
detector × catalog 조합으로 FSM 출력 CSV 를 일괄 매칭하는 통합 런너.

detector : gk2a | metop03 | noaa19
catalog  : noaa | swpc
kind     : onset(기본) | event

경로 규칙 (이 파일이 C_PD/ 에 위치):
  GK2A  fsm2_output : GK2A/count_FSM/fsm2_output/
  POES  fsm2_output : POES/{detector}_count/fsm2_output/  (MetOp03 → MetOp03_count)

사용:
  # GK2A + NOAA (onset)
  python run_matcher.py --detector gk2a --catalog noaa

  # POES MetOp03 + SWPC
  python run_matcher.py --detector metop03 --catalog swpc

  # dry-run (명령만 출력)
  python run_matcher.py --detector noaa19 --catalog noaa --dry

  # 전체 6조합 일괄
  for det in gk2a metop03 noaa19; do
    for cat in noaa swpc; do
      python run_matcher.py --detector $det --catalog $cat
    done
  done
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent   # C_PD/

# ── 경로 테이블 (HERE 기준 상대경로) ───────────────────────────────
_GK2A_COUNT   = HERE / "GK2A" / "KSEM_count"  / "ksem_cache_parquet"
_NOAA_CAT     = HERE / "NOAA_GOES" / "noaa_goes_spe_cache_parquet"
_SWPC_CAT     = HERE / "SWPC_Alert" / "swpc_espe_cache_parquet"
_NOAA_MATCH   = HERE / "GK2A"       / "event_MATCHER" / "noaa_goes_spe_match.py"
_SWPC_MATCH   = HERE / "GK2A"       / "event_MATCHER" / "swpc_alert_espe_match.py"
_NOAA_MATCH_P = HERE / "POES"       / "event_MATCHER" / "noaa_goes_spe_match_poes.py"
_SWPC_MATCH_P = HERE / "POES"       / "event_MATCHER" / "swpc_alert_espe_match_poes.py"
_NOAA_IO      = HERE / "NOAA_GOES"  / "noaa_goes_spe_io"
_SWPC_IO      = HERE / "SWPC_Alert" / "swpc_alert_espe_io"


def _poes_sat_dir(detector: str) -> str:
    """metop03 → 'MetOp03_count', noaa19 → 'NOAA19_count'."""
    return {"metop03": "MetOp03_count", "noaa19": "NOAA19_count"}[detector]


def _poes_count_cache(detector: str) -> Path:
    sat = _poes_sat_dir(detector)
    return HERE / "POES" / sat / f"poes_{detector}_cache_parquet"


def _poes_count_io(detector: str) -> Path:
    return HERE / "POES" / f"{_poes_sat_dir(detector)}" / f"poes_{detector}_io"


def build_cmd(detector: str, catalog: str, kind: str,
              out_suffix: str) -> list[str]:
    """매칭 명령 리스트를 반환."""
    if detector == "gk2a":
        script   = _NOAA_MATCH   if catalog == "noaa" else _SWPC_MATCH
        cat_path = _NOAA_CAT     if catalog == "noaa" else _SWPC_CAT
        fsm_dir  = HERE / "GK2A" / "count_FSM" / "fsm2_output"
        out_dir  = HERE / "GK2A" / f"{catalog}_match{out_suffix}"
        return [
            sys.executable, str(script),
            "--events-dir", str(fsm_dir),
            "--kind",       kind,
            "--catalog",    str(cat_path),
            "--out",        str(out_dir),
            "--count-dir",  str(_GK2A_COUNT),
        ]
    else:
        script   = _NOAA_MATCH_P if catalog == "noaa" else _SWPC_MATCH_P
        cat_path = _NOAA_CAT     if catalog == "noaa" else _SWPC_CAT
        io_path  = _NOAA_IO      if catalog == "noaa" else _SWPC_IO
        sat      = _poes_sat_dir(detector)
        fsm_dir  = HERE / "POES" / sat / "fsm2_output"
        out_dir  = HERE / "POES" / sat / f"{catalog}_match{out_suffix}"
        return [
            sys.executable, str(script),
            "--events",  str(next(fsm_dir.rglob(f"fsm_{kind}_*.csv"), "")),
            "--catalog", str(cat_path),
            "--spe-io",  str(io_path),
            "--out",     str(out_dir),
        ]


def main():
    ap = argparse.ArgumentParser(
        description="detector × catalog 조합 매칭 런너")
    ap.add_argument("--detector", required=True,
                    choices=["gk2a", "metop03", "noaa19"])
    ap.add_argument("--catalog",  required=True,
                    choices=["noaa", "swpc"])
    ap.add_argument("--kind",   default="onset", choices=["onset", "event"],
                    help="FSM 출력 종류 (default onset)")
    ap.add_argument("--out-suffix", default="_output",
                    help="출력 디렉터리 접미 (default _output → noaa_match_output)")
    ap.add_argument("--dry", action="store_true", help="명령만 출력, 실행 안 함")
    args = ap.parse_args()

    if args.detector != "gk2a":
        # POES: fsm2_output 아래 CSV 개별 처리 (glob 후 루프)
        sat     = _poes_sat_dir(args.detector)
        fsm_dir = HERE / "POES" / sat / "fsm2_output"
        cat_path = _NOAA_CAT if args.catalog == "noaa" else _SWPC_CAT
        io_path  = _NOAA_IO  if args.catalog == "noaa" else _SWPC_IO
        script   = _NOAA_MATCH_P if args.catalog == "noaa" else _SWPC_MATCH_P
        out_dir  = HERE / "POES" / sat / f"{args.catalog}_match{args.out_suffix}"
        csvs     = sorted(fsm_dir.rglob(f"fsm_{args.kind}_*.csv"))
        if not csvs:
            raise SystemExit(f"[run] {fsm_dir} 아래 fsm_{args.kind}_*.csv 없음")
        print(f"[run] {args.detector}/{args.catalog}/{args.kind} — {len(csvs)}개 CSV")
        for csv in csvs:
            cmd = [
                sys.executable, str(script),
                "--events",  str(csv),
                "--catalog", str(cat_path),
                "--spe-io",  str(io_path),
                "--out",     str(out_dir),
            ]
            if args.dry:
                print(" ".join(cmd))
            else:
                print("\n>>> " + " ".join(cmd))
                subprocess.run(cmd, check=False)
    else:
        # GK2A: --events-dir で一括
        script   = _NOAA_MATCH   if args.catalog == "noaa" else _SWPC_MATCH
        cat_path = _NOAA_CAT     if args.catalog == "noaa" else _SWPC_CAT
        fsm_dir  = HERE / "GK2A" / "count_FSM" / "fsm2_output"
        out_dir  = HERE / "GK2A" / f"{args.catalog}_match{args.out_suffix}"
        cmd = [
            sys.executable, str(script),
            "--events-dir", str(fsm_dir),
            "--kind",       args.kind,
            "--catalog",    str(cat_path),
            "--out",        str(out_dir),
            "--count-dir",  str(_GK2A_COUNT),
        ]
        if args.dry:
            print(" ".join(cmd))
        else:
            print("\n>>> " + " ".join(cmd))
            subprocess.run(cmd, check=False)

    print("\n[run] 완료.")


if __name__ == "__main__":
    main()

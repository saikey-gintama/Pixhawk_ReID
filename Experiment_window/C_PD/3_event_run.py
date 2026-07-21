"""
3_event_run.py
==============
통합 이벤트 매칭 러너 (run_matcher.py 승격판).

detector × catalog 조합으로 FSM 출력 CSV ↔ 카탈로그를 매칭한다.

run_matcher.py 대비 변경점:
  --fsm-dir  : FSM 출력 폴더를 인자로. 기본값은 2_fsm_run.py 표준 출력
               ({detector}_output/2_fsm — 구 fsm2_output 아님).
               STEP 1에서 FSM --out이 생겨 임의 경로의 FSM 결과도 매칭 가능.
  peak-dedup : kind=onset에서 peak만 다른 runtag의 onset CSV는
               byte-identical(4_summarize에서 검증)이라 기본 1개만 매칭.
               끄려면 --no-dedup.
  resume     : 매처 성공(종료코드 0)한 CSV를 out_dir/_done_events.txt에
               기록. 같은 out_dir로 재실행하면 기록분 skip (끄려면 --no-resume).
  출력 경로  : {detector}_output/3_event/<catalog>_<kind>/ (고정 경로, 날짜 스탬프
               없음 — resume이 날짜에 안 묶이도록. 언제 돌렸는지는 폴더명이 아니라
               _done_events.txt/실행 로그가 담당).
               기존 *_match_output, *_match_output_* 디렉터리를 절대 덮지 않는다.
               고정 폴더가 이미 있는데 _done_events.txt가 없으면(구버전/외부 결과일
               수 있음) 덮어쓰지 않고 중단 — --out으로 다른 경로를 지정하거나
               기존 폴더를 정리할 것.
  --out      : 명시적으로 출력 경로를 지정할 때만 사용(이 경우 위 보호 검사 생략).
  --max-onsets : 과검출 가드. CSV의 채널별 onset 건수 min이 이 값을
               초과하면(=모든 채널이 구조적으로 과검출) 매칭을 생략하고
               out_dir/_skipped_events.txt에 기록한다. 한 채널이라도 기준
               이하면 통과(그 CSV의 과검출 채널은 sweep_table에 FAR≈1로
               정직하게 남는다). 기본 1000, 0=가드 해제. dry에서는 행을
               세지 않고 가드 활성 여부만 표시.

동작 (GK2A·POES 완전 동일 — detector 무관 공통 빌더 하나가 처리):
  fsm_dir/**fsm_<kind>_*.csv 를 glob 후 CSV마다 개별 호출 (peak-dedup →
  resume → 과검출 가드 순으로 필터). GK2A 매처(noaa_goes_spe_match.py /
  swpc_alert_espe_match.py)는 원래 --events-dir(폴더 1회 호출)와 --events
  (단일 CSV) 둘 다 지원하던 스크립트라 매처 쪽 변경 없이 --events 단일
  호출만 골라 쓰면 된다 (--events-dir 경로는 러너가 더 이상 안 쓸 뿐 남아있음).
  --dry : 실행 없이 명령만 출력. CSV가 없으면 경고 후 경로 정보만 출력.

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
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

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
    """detector별 기본 FSM 출력 폴더.
    2_fsm_run.py의 _OUT_DIR과 반드시 동일해야 함:
      {detector}_output/2_fsm/<runtag>/fsm_<kind>_<runtag>.csv
    (구 run_matcher 시절의 fsm2_output 경로는 더 이상 기본값이 아님 —
     옛 결과를 매칭하려면 --fsm-dir로 명시.)
    """
    if detector == "gk2a":
        return HERE / "GK2A" / "KSEM_count" / "gk2a_output" / "2_fsm"
    sat = _poes_sat_dir(detector)                     # "MetOp03_count" | "NOAA19_count"
    return HERE / "POES" / sat / f"{detector}_output" / "2_fsm"


def _default_out_dir(detector: str, catalog: str, kind: str) -> Path:
    """detector별 고정 매칭 결과 폴더 (날짜 스탬프 없음).
    {detector}_output/3_event/<catalog>_<kind>/
      GK2A    : GK2A/KSEM_count/gk2a_output/3_event/<catalog>_<kind>/
      MetOp03 : POES/MetOp03_count/metop03_output/3_event/<catalog>_<kind>/
      NOAA19  : POES/NOAA19_count/noaa19_output/3_event/<catalog>_<kind>/
    kind(onset/event)을 경로에 넣는 이유: 같은 catalog라도 onset/event 매칭은
    별개 결과라 섞이면 안 됨. 날짜 스탬프를 뺀 이유: resume(_done_events.txt)이
    날짜 폴더에 묶여 자정을 넘겨 재실행하면 새 폴더가 생겨 resume이 끊기던 문제.
    """
    sub = f"{catalog}_{kind}"
    if detector == "gk2a":
        return HERE / "GK2A" / "KSEM_count" / "gk2a_output" / "3_event" / sub
    sat  = _poes_sat_dir(detector)                    # "MetOp03_count" | "NOAA19_count"
    name = detector                                    # "metop03" | "noaa19"
    return HERE / "POES" / sat / f"{name}_output" / "3_event" / sub


# onset CSV는 peak 값만 다른 runtag끼리 byte-identical (4_summarize.py에서 검증된 사실
# — peak_floor는 onset 판정에 영향 없음). kind=onset일 때 pk 토큰만 다른 CSV는 1개만 남긴다.
# 토큰은 runtag "끝"의 _pk<숫자(.숫자)?> 만 매칭(anchor) -- baseline 이름 파싱에 의존하지
# 않고 위치만으로 동작.
_PK_TOKEN_RE = re.compile(r"_pk[\d.]+$")


def _dedup_peak_csvs(csvs: list[Path], kind: str, dedup: bool,
                     fsm_dir: Path) -> tuple[list[Path], int]:
    """kind=onset & dedup=True일 때 peak만 다른 중복 CSV 제거. (kept, n_skipped) 반환.
    키는 파일명이 아니라 fsm_dir 기준 상대경로 -- CSV의 부모 폴더(runtag)에서 끝의
    _pk<val> 토큰만 제거하고, 그 위 폴더 계층(예: 2-2_fsm_binned의 bin_60_70/)은
    그대로 키에 남긴다. 파일명(stem)만 보면 다른 bin의 동일 runtag 파일을 같은
    "peak-중복"으로 오인해 하나만 남기고 나머지 bin을 조용히 버리는 버그가 있었음
    -- 표준 2_fsm(폴더 계층 1단, runtag별 유일)에서는 stem 키와 결과가 동일하므로
    회귀 없음."""
    if kind != "onset" or not dedup:
        return csvs, 0
    seen: set[str] = set()
    kept: list[Path] = []
    n_dup = 0
    for csv in csvs:
        parts = csv.parent.relative_to(fsm_dir).parts
        if parts:
            key = "/".join(parts[:-1] + (_PK_TOKEN_RE.sub("", parts[-1]),))
        else:
            key = _PK_TOKEN_RE.sub("", csv.stem)   # fsm_dir 바로 아래 CSV(폴더 계층 없음) 대비
        if key in seen:
            n_dup += 1
            continue
        seen.add(key)
        kept.append(csv)
    return kept, n_dup


# ── resume (GK2A·POES 공통): out_dir/_done_events.txt 매니페스트 기반 ────
# 기본 out_dir은 고정 경로({detector}_output/3_event/<catalog>_<kind>)이므로 같은
# 명령을 재실행하면 자동으로 이어진다. 매처 종료코드 0인 CSV만 기록, --no-resume로 해제.
_DONE_NAME = "_done_events.txt"


def _load_done(out_dir: Path) -> set[str]:
    f = out_dir / _DONE_NAME
    if not f.exists():
        return set()
    return {ln.strip() for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()}


def _mark_done(out_dir: Path, csv: Path) -> None:
    with (out_dir / _DONE_NAME).open("a", encoding="utf-8") as f:
        f.write(str(csv.resolve()) + "\n")


# ── 과검출 가드 (GK2A·POES 공통): out_dir/_skipped_events.txt 기록 ─────────
# 채널별 min 건수가 --max-onsets 초과인 "전채널 구조적 과검출" CSV만 거른다.
# 한 채널이라도 기준 이하면 통과시킨다 -- 그 CSV는 sweep_table에서 과검출
# 채널만 FAR≈1로 정직하게 남으므로 데이터로서 가치가 있다. resume(_done_events.txt)
# 과는 별개 매니페스트: skip된 CSV는 성공이 아니므로 _done에 남기지 않는다
# (--max-onsets 0으로 재실행하면 다시 시도되어야 함).
_SKIPPED_NAME = "_skipped_events.txt"


def _over_detect_min_count(csv: Path, max_onsets: int) -> int | None:
    """전채널 min 건수가 max_onsets 초과면 그 min 건수를, 아니면 None을 반환.
    max_onsets<=0(가드 해제)이거나 빈 CSV(채널 행 없음)면 항상 None
    -- 가드 미적용, 기존 동작(빈 CSV는 매처가 그대로 처리) 유지.
    """
    if max_onsets <= 0:
        return None
    try:
        counts = pd.read_csv(csv, usecols=["channel"])["channel"].value_counts()
    except pd.errors.EmptyDataError:
        return None
    if counts.empty:
        return None
    min_count = int(counts.min())
    return min_count if min_count > max_onsets else None


def _mark_skipped(out_dir: Path, csv: Path, min_count: int, limit: int) -> None:
    with (out_dir / _SKIPPED_NAME).open("a", encoding="utf-8") as f:
        f.write(f"{csv.resolve()}\tmin={min_count}\tlimit={limit}\n")


def _match_script_and_extra_args(detector: str, catalog: str) -> tuple[Path, list[str]]:
    """detector별 매처 스크립트 경로와 고정 추가 인자.
    GK2A 매처(--count-dir로 전 채널 overlay)와 POES 매처(--spe-io로 카탈로그
    io 모듈 지정)는 부가 인자만 다르고 --events/--catalog/--out 인터페이스는
    동일 -- GK2A 매처는 원래 --events(단일 CSV)도 지원해서 스크립트 변경 없이
    바로 개별 호출로 전환 가능.
    """
    if detector == "gk2a":
        script = _NOAA_MATCH if catalog == "noaa" else _SWPC_MATCH
        return script, ["--count-dir", str(_GK2A_COUNT)]
    io_path = _NOAA_IO if catalog == "noaa" else _SWPC_IO
    script  = _NOAA_MATCH_P if catalog == "noaa" else _SWPC_MATCH_P
    return script, ["--spe-io", str(io_path)]


def build_match_cmds(detector: str, catalog: str, kind: str,
                     fsm_dir: Path, out_dir: Path,
                     dedup: bool = True) -> tuple[list[tuple[Path, list[str]]], int]:
    """(csv경로, 명령) 쌍 리스트와 peak-중복 skip 수 반환. detector 무관 공통 빌더
    -- GK2A/POES 모두 fsm_dir/**fsm_<kind>_*.csv 를 glob 후 CSV마다 개별 호출."""
    script, extra_args = _match_script_and_extra_args(detector, catalog)
    cat_path = _NOAA_CAT if catalog == "noaa" else _SWPC_CAT
    csvs     = sorted(fsm_dir.rglob(f"fsm_{kind}_*.csv")) if fsm_dir.exists() else []
    csvs, n_dup = _dedup_peak_csvs(csvs, kind, dedup, fsm_dir)
    cmds = [
        (csv, [
            sys.executable, str(script),
            "--events",  str(csv),
            "--catalog", str(cat_path),
            "--out",     str(out_dir),
            *extra_args,
        ])
        for csv in csvs
    ]
    return cmds, n_dup


def main():
    ap = argparse.ArgumentParser(description="detector × catalog 통합 이벤트 매칭 러너")
    ap.add_argument("--detector", required=True, choices=["gk2a", "metop03", "noaa19"])
    ap.add_argument("--catalog",  required=True, choices=["noaa", "swpc"])
    ap.add_argument("--kind",    default="onset", choices=["onset", "event"],
                    help="FSM 출력 종류 (default onset)")
    ap.add_argument("--fsm-dir", default=None,
                    help="FSM 출력 폴더 (기본: 2_fsm_run.py 표준 "
                         "{detector}_output/2_fsm)")
    ap.add_argument("--out",     default=None,
                    help="매칭 결과 출력 폴더 "
                         "(기본: {detector}_output/3_event/<catalog>_<kind>, 고정 경로. "
                         "명시 시 기존 결과 보호 검사 생략)")
    ap.add_argument("--dry",     action="store_true", help="명령만 출력, 실행 안 함")
    ap.add_argument("--no-dedup", action="store_true",
                    help="kind=onset에서 peak만 다른 중복 CSV skip을 끈다 "
                         "(기본: peak-중복 skip -- onset CSV는 peak과 무관하게 동일)")
    ap.add_argument("--no-resume", action="store_true",
                    help="resume(out_dir/_done_events.txt 기반 skip)을 끄고 전부 재실행")
    ap.add_argument("--max-onsets", type=int, default=1000,
                    help="과검출 가드: CSV의 채널별 onset 건수 min이 이 값을 "
                         "초과하면(=전채널 구조적 과검출) 매칭 생략 후 "
                         "out_dir/_skipped_events.txt에 기록 (min 기준 -- 한 채널이라도 "
                         "기준 이하면 통과). 기본 1000, 0=가드 해제")
    args = ap.parse_args()

    fsm_dir = Path(args.fsm_dir) if args.fsm_dir else _default_fsm_dir(args.detector)
    out_dir = (Path(args.out) if args.out
              else _default_out_dir(args.detector, args.catalog, args.kind))

    print(f"[3_event] detector={args.detector}  catalog={args.catalog}  kind={args.kind}")
    print(f"[3_event] fsm_dir : {fsm_dir}")
    print(f"[3_event] out_dir : {out_dir}")

    # ── 기존 결과 보호: 고정 폴더가 있는데 resume 매니페스트가 없으면 이 러너가
    # 만든 결과가 아닐 수 있음(구버전 날짜 폴더 등) — 덮어쓰지 않고 중단.
    # --out 명시 시엔 사용자 책임이라 검사 생략.
    if (not args.out and out_dir.exists() and any(out_dir.iterdir())
            and not (out_dir / _DONE_NAME).exists()):
        msg = (f"[3_event] 출력 폴더가 이미 있는데 {_DONE_NAME}이 없습니다 -> {out_dir}\n"
              f"  이 러너가 만든 결과가 아닐 수 있어 덮어쓰지 않습니다. "
              f"--out으로 다른 경로를 지정하거나 기존 폴더를 정리하세요.")
        if args.dry:
            print(f"[3_event] WARNING: {msg}")
        else:
            raise SystemExit(f"[3_event] ERROR: {msg}")

    cmds, n_dup = build_match_cmds(args.detector, args.catalog, args.kind,
                                   fsm_dir, out_dir, dedup=not args.no_dedup)
    if not cmds:
        script, _ = _match_script_and_extra_args(args.detector, args.catalog)
        print(f"[3_event] WARNING: fsm_{args.kind}_*.csv 없음 -> {fsm_dir}")
        print(f"[3_event] (2_fsm_run.py 출력 경로 {{detector}}_output/2_fsm 기준. "
              f"다른 곳이면 --fsm-dir로 지정)")
        print(f"[3_event] 매처 스크립트 : {script}")
        if args.dry:
            print("[3_event] --dry: CSV 없어 명령 생략 (FSM 먼저 실행 필요).")
            return
        raise SystemExit(f"[3_event] ERROR: CSV 없음 -- FSM 먼저 실행하세요 -> {fsm_dir}")
    if n_dup:
        print(f"[3_event] peak-중복 skip: {n_dup}개 (onset CSV는 peak 무관 동일 "
              f"-- 전부 돌리려면 --no-dedup)")

    # -- resume: 이 out_dir에서 이미 성공한 CSV는 skip --
    done = set() if args.no_resume else _load_done(out_dir)
    if done:
        before = len(cmds)
        cmds = [(csv, cmd) for csv, cmd in cmds if str(csv.resolve()) not in done]
        n_resume = before - len(cmds)
        if n_resume:
            print(f"[3_event] resume skip: {n_resume}개 "
                  f"(기록: {out_dir / _DONE_NAME}, 전부 재실행은 --no-resume)")
        if not cmds:
            print("[3_event] 실행할 CSV 없음 (전부 resume skip). 완료 상태.")
            return

    # -- 과검출 가드: 전채널 min > --max-onsets 이면 매칭 생략 (dedup/resume 통과분만 검사) --
    if args.max_onsets <= 0:
        pass
    elif args.dry:
        print(f"[3_event] 과검출 가드 활성 (min>{args.max_onsets}건 -> 매칭 생략 예정, "
              f"실제 계수는 실행 시에만 -- --max-onsets 0 이면 해제)")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        kept = []
        n_over = 0
        for csv, cmd in cmds:
            min_count = _over_detect_min_count(csv, args.max_onsets)
            if min_count is None:
                kept.append((csv, cmd))
                continue
            n_over += 1
            print(f"[3_event] WARN: 전채널 과검출(min={min_count}건>{args.max_onsets}) "
                  f"-- 매칭 생략: {csv.name}")
            _mark_skipped(out_dir, csv, min_count, args.max_onsets)
        cmds = kept
        if n_over:
            print(f"[3_event] 과검출 skip: {n_over}개 "
                  f"(기록: {out_dir / _SKIPPED_NAME}, 전부 돌리려면 --max-onsets 0)")
        if not cmds:
            print("[3_event] 실행할 CSV 없음 (전부 과검출 skip).")
            return

    print(f"[3_event] {len(cmds)}개 CSV -> 개별 호출:")
    _show = cmds if len(cmds) <= 5 else cmds[:3]
    for _, cmd in _show:
        print("  " + " ".join(cmd))
    if len(cmds) > 5:
        print(f"  ... 외 {len(cmds) - 3}개 (동일 형식, --events 경로만 다름)")
    if args.dry:
        print("[3_event] --dry: 실행 생략.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    n_fail = 0
    for i, (csv, cmd) in enumerate(cmds, 1):
        print(f"\n>>> [{i}/{len(cmds)}] " + " ".join(cmd))
        rc = subprocess.run(cmd, check=False).returncode
        if rc == 0:
            _mark_done(out_dir, csv)          # 성공한 것만 기록 -> 실패분은 재실행 대상
        else:
            n_fail += 1
            print(f"[3_event] WARNING: 매처 종료코드 {rc} -> done 미기록 "
                  f"(재실행 시 다시 시도): {csv.name}")
    if n_fail:
        print(f"\n[3_event] 실패 {n_fail}개 -- 같은 명령 재실행하면 실패분만 다시 돕니다.")

    print("\n[3_event] 완료.")


if __name__ == "__main__":
    main()
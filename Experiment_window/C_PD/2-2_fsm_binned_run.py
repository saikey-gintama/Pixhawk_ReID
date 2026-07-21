"""
2-2_fsm_binned_run.py
======================
POES 조건부배경(condbg) FSM 실행 러너 — 2_fsm_run.py와 동일 구조.

설계 전환(2026-07-21): 이전 버전(캐시를 bin별로 물리 분할해 bin 폴더마다
별도 서브프로세스로 돌리고, 러너 레벨에서 gap guard/geo enrich 후처리로
"가짜 duration" 버그를 사후 보정하던 래퍼)은 전량 폐기됐다. 원인 자체를
없앤 재설계로 교체: count_FSM/에 조건부배경(condbg) 엔진을 갖춘 새 스크립트
2개(fsm_count_spe_cusum_condbg_poes.py, fsm_count_spe_quietoff_condbg_poes.py,
"binned"는 구현방법이라 파일명에서 뺐다 — 정체는 배경을 무엇에 조건화하는가)를
추가하고, 이 러너는 그 2개만을 대상으로 2_fsm_run.py의 sweep/resume/--jobs
로직을 그대로 재사용하는 얇은 래퍼가 됐다. bin 배정·독립배경·(cusum의)
세그먼트 병합은 이제 FSM 스크립트 자체 안에서 처리되므로, 이 러너는 더 이상
캐시를 쪼개거나 bin별 결과를 병합할 필요가 없다 — 캐시 분할기(build)/병합기/
bALL 집계는 전부 제거됨. 이 러너 자신의 파일명(2-2_fsm_binned_run.py)은
번호/큰 목적(POES 위도 bin FSM 실행)이 여전히 유효해 유지 — "_binned"를
뺀 건 FSM 스크립트 2개뿐(정체 표현이 필요한 건 배경 산출 로직 쪽).

--maglat-bins 은 sweep 축이 아니라 매 실행에 그대로 전달되는 고정 파라미터
(runtag의 mlb 토큰에 반영 — FSM 스크립트의 build_runtag과 동일 규칙).

대상은 POES 2종(metop03, noaa19)만 — gk2a(정지궤도)는 maglat 비닝이 무의미해
전과 동일하게 제외.

출력 경로는 2_fsm_run.py와 **동일한** {detector}_output/2_fsm/ (별도 트리
아님) — runtag에 mlb 토큰이 있으면 condbg, 없으면 시간순. 4_summarize가
이 토큰으로 구분해 maglat_bins 컬럼을 채운다(별도 매치 루트 불필요).

blc1_lowe/blc1_fixed의 condbg 판은 아직 없음(후순위) — --baselines로 이
둘을 지정하면 "스크립트 없음" 에러가 난다(quietoff_condbg/cusum_condbg만
등록 대상).

사용:
  python 2-2_fsm_binned_run.py --detector metop03 --dry
  python 2-2_fsm_binned_run.py --detector metop03 --maglat-bins 15,30,45,60,75 \\
      --win 10 --k 10 --onset 0.5 --peak 1 --baselines cusum_condbg
  python 2-2_fsm_binned_run.py --detector metop03 --win 10,30 --k 5,10 \\
      --jobs 4 --stagger-sec 0.5
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from pathlib import Path

HERE = Path(__file__).resolve().parent  # C_PD/

_POES_FSM_DIR = HERE / "POES" / "count_FSM"

_OUT_DIR = {
    "metop03": HERE / "POES" / "MetOp03_count" / "metop03_output" / "2_fsm",
    "noaa19":  HERE / "POES" / "NOAA19_count"  / "noaa19_output"  / "2_fsm",
}
_POES_IO = {
    "metop03": ("poes_metop03_io", HERE / "POES" / "MetOp03_count" / "poes_metop03_cache_parquet"),
    "noaa19":  ("poes_noaa19_io",  HERE / "POES" / "NOAA19_count"  / "poes_noaa19_cache_parquet"),
}

_DEFAULT_MAGLAT_BINS = "15,30,45,60,75"
_CUSUM_H_FIXED  = 5.0
_GAP_H_LAT      = 2.0
_GAP_H_SAA      = 12.0
_MERGE_GAP_H    = 3.0

_AXIS_CLI = {"win": "--window", "k": "--k", "onset": "--onset", "peak": "--peak"}

_FSM_AXES = {
    "quietoff_condbg": ("win", "k", "onset", "peak"),
    "cusum_condbg":     ("win", "k", "onset", "peak"),
}
_TAG_MAP = {
    "fsm_count_spe_quietoff_condbg_poes": "quietoff_mad",
    "fsm_count_spe_cusum_condbg_poes":    "cusum",
}
_FSM_DEFAULTS = {
    "quietoff_condbg": {"win": 30, "k": 10.0, "onset": 0.5, "peak": 2.0},
    "cusum_condbg":     {"win": 30, "k": 3.0,  "onset": 0.5, "peak": 2.0},
}


def _fsm_type(script: Path) -> str:
    name = script.stem
    if "quietoff_condbg" in name: return "quietoff_condbg"
    if "cusum_condbg"    in name: return "cusum_condbg"
    raise ValueError(f"알 수 없는 condbg FSM: {name}")


def _numstr(v) -> str:
    f = float(v)
    return str(int(f)) if f.is_integer() else str(f)


def _parse_list(s: str | None, typ) -> list | None:
    if s is None:
        return None
    return [typ(v.strip()) for v in s.split(",")]


def _mlb_token(spec: str) -> str:
    """--maglat-bins 값 -> runtag mlb 토큰. FSM 스크립트 build_runtag과 동일 규칙
    (정렬+중복제거 후 _numstr join)이어야 resume이 정확히 맞아떨어진다."""
    vals = sorted(set(float(v.strip()) for v in spec.split(",") if v.strip() != ""))
    return "-".join(_numstr(v) for v in vals)


def _predict_runtag(script: Path, fsm_type: str, combo: dict, mlb: str) -> str:
    tag = _TAG_MAP[script.stem]
    d   = _FSM_DEFAULTS[fsm_type]
    w   = combo.get("win",   d.get("win"))
    k   = combo.get("k",     d.get("k"))
    on  = combo.get("onset", d["onset"])
    pk  = combo.get("peak",  d["peak"])
    if fsm_type == "cusum_condbg":
        return f"{tag}_mlb{mlb}_w{w}_k{_numstr(k)}_h{_numstr(_CUSUM_H_FIXED)}_on{_numstr(on)}_pk{_numstr(pk)}"
    return f"{tag}_mlb{mlb}_w{w}_k{_numstr(k)}_on{_numstr(on)}_pk{_numstr(pk)}"


def _build_combos(fsm_type: str, win_vals, k_vals, onset_vals, peak_vals) -> tuple[list[dict], int]:
    supported = _FSM_AXES[fsm_type]
    sweep_map = {"win": win_vals, "k": k_vals, "onset": onset_vals, "peak": peak_vals}
    axes, vals = [], []
    for ax in supported:
        v = sweep_map[ax]
        if v is not None:
            axes.append(ax); vals.append(v)
    if not axes:
        return [{}], 0
    combos, n_invalid = [], 0
    for combo_vals in product(*vals):
        combo = dict(zip(axes, combo_vals))
        if fsm_type == "cusum_condbg" and combo.get("k", 9999) <= 1:
            n_invalid += 1  # ln(k)<=0 -> 검출 불가 (FSM 자체 방어와 동일 기준)
            continue
        combos.append(combo)
    return combos, n_invalid


def _build_cmd(script: Path, detector: str, out_parent: Path, combo: dict,
               maglat_bins: str, gap_h: float, gap_h_saa: float, merge_gap_h: float) -> list[str]:
    io_mod, cache = _POES_IO[detector]
    cmd = [sys.executable, str(script), "--io", io_mod, "--cache", str(cache),
           "--out", str(out_parent), "--maglat-bins", maglat_bins]
    if _fsm_type(script) == "cusum_condbg":
        cmd += ["--h", str(_CUSUM_H_FIXED), "--gap-h", str(gap_h),
                "--gap-h-saa", str(gap_h_saa), "--merge-gap-h", str(merge_gap_h)]
    for ax, val in combo.items():
        cmd += [_AXIS_CLI[ax], str(val)]
    return cmd


# ── --jobs>1 병렬 실행 (기본 --jobs 1 경로는 절대 안 거침, 2_fsm_run.py와 동일) ──
_launch_lock = threading.Lock()
_last_launch = [0.0]


def _stagger_wait(stagger_sec: float) -> None:
    if stagger_sec <= 0:
        return
    with _launch_lock:
        now = time.monotonic()
        wait = _last_launch[0] + stagger_sec - now
        if wait > 0:
            time.sleep(wait)
        _last_launch[0] = time.monotonic()


def _run_one_captured(cmd: list[str], stagger_sec: float) -> bytes:
    _stagger_wait(stagger_sec)
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return res.stdout


def main():
    ap = argparse.ArgumentParser(
        description="POES |maglat| bin별 독립배경 FSM 실행 러너 (2_fsm_run.py와 동일 구조)")
    ap.add_argument("--detector", required=True, choices=["metop03", "noaa19"])
    ap.add_argument("--maglat-bins", default=_DEFAULT_MAGLAT_BINS, metavar="LIST",
                    help=f'|maglat| bin 내부 경계 콤마 리스트 (기본 "{_DEFAULT_MAGLAT_BINS}"). '
                         f'sweep 축 아님 — 매 실행에 그대로 전달, runtag mlb 토큰에 반영.')
    ap.add_argument("--gap-h",       type=float, default=_GAP_H_LAT,
                    help="cusum_condbg: 위도 bin gap guard[h] (기본 2)")
    ap.add_argument("--gap-h-saa",   type=float, default=_GAP_H_SAA,
                    help="cusum_condbg: SAA bin gap guard[h] (기본 12)")
    ap.add_argument("--merge-gap-h", type=float, default=_MERGE_GAP_H,
                    help="cusum_condbg: 세그먼트 병합 gap[h] (기본 3)")
    ap.add_argument("--dry",   action="store_true", help="조합 수/예시 명령만 출력, 실행 안 함")
    ap.add_argument("--limit", type=int, default=0, metavar="N",
                    help="앞에서 N개만 실행 (0=무제한, 메커니즘 검증용)")
    ap.add_argument("--win",   default=None, metavar="LIST")
    ap.add_argument("--onset", default=None, metavar="LIST")
    ap.add_argument("--peak",  default=None, metavar="LIST")
    ap.add_argument("--k",     default=None, metavar="LIST")
    ap.add_argument("--baselines", default=None, metavar="LIST",
                    help="quietoff_condbg,cusum_condbg 중 필터 (미지정=둘 다)")
    ap.add_argument("--jobs", type=int, default=1, metavar="N",
                    help="병렬 실행 워커 수 (기본 1 = 기존과 동일한 순차 실행)")
    ap.add_argument("--stagger-sec", type=float, default=0.0, metavar="S",
                    help="--jobs>1일 때 서브프로세스 시작 최소 간격[sec] (기본 0)")
    args = ap.parse_args()

    det = args.detector
    out_parent = _OUT_DIR[det]
    mlb = _mlb_token(args.maglat_bins)
    scripts = sorted(_POES_FSM_DIR.glob("fsm_count_spe_*_condbg_poes.py"))

    if args.baselines:
        wanted = {b.strip() for b in args.baselines.split(",") if b.strip()}
        scripts = [s for s in scripts if _fsm_type(s) in wanted]
    if not scripts:
        raise SystemExit(f"[2-2] ERROR: condbg FSM 스크립트 없음 -> {_POES_FSM_DIR}"
                         + (f" (--baselines {args.baselines} 필터 결과 0개)" if args.baselines else ""))

    win_vals   = _parse_list(args.win,   int)
    k_vals     = _parse_list(args.k,     float)
    onset_vals = _parse_list(args.onset, float)
    peak_vals  = _parse_list(args.peak,  float)
    is_sweep = any(v is not None for v in (win_vals, k_vals, onset_vals, peak_vals))

    script_data = []
    total_invalid = 0
    for script in scripts:
        fsm_t = _fsm_type(script)
        combos, n_invalid = _build_combos(fsm_t, win_vals, k_vals, onset_vals, peak_vals)
        script_data.append((script, fsm_t, combos, n_invalid))
        total_invalid += n_invalid

    if args.dry:
        print(f"[2-2] --dry  detector={det}  FSM {len(scripts)}개  mlb={mlb}")
        parts = []
        total_valid = 0
        for script, fsm_t, combos, n_invalid in script_data:
            short = script.stem.replace("fsm_count_spe_", "").replace("_poes", "")
            n = len(combos); total_valid += n
            entry = f"{short} {n}"
            if n_invalid:
                entry += f"(+{n_invalid}skip)"
            parts.append(entry)
        print(f"[2-2] {det}: " + ", ".join(parts))
        print(f"[2-2] 총 조합: {total_valid}개"
              + (f"  (무효 skip: {total_invalid}개 -- cusum_condbg k<=1)" if total_invalid else ""))
        if script_data:
            script, fsm_t, combos, _ = script_data[0]
            example = combos[0] if combos else {}
            cmd = _build_cmd(script, det, out_parent, example, args.maglat_bins,
                             args.gap_h, args.gap_h_saa, args.merge_gap_h)
            print("[2-2] 예시 명령:")
            print("    " + " ".join(str(t) for t in cmd))
        print("[2-2] --dry: 실행 생략.")
        return

    run_items: list[tuple[str, dict, list[str]]] = []
    total_resume = 0
    for script, fsm_t, combos, _ in script_data:
        for combo in combos:
            if is_sweep:
                runtag  = _predict_runtag(script, fsm_t, combo, mlb)
                out_dir = out_parent / runtag
                if out_dir.exists():
                    total_resume += 1
                    continue
            cmd = _build_cmd(script, det, out_parent, combo, args.maglat_bins,
                             args.gap_h, args.gap_h_saa, args.merge_gap_h)
            run_items.append((script.name, combo, cmd))

    if not run_items:
        print("[2-2] 실행할 조합 없음 (전부 resume skip).")
        return
    if args.limit > 0:
        run_items = run_items[:args.limit]
        print(f"[2-2] --limit {args.limit}: {len(run_items)}개만 실행.")

    total_valid = sum(len(c) for _, _, c, _ in script_data)
    print(f"[2-2] detector={det}  조합 {total_valid}개  실행 {len(run_items)}개  mlb={mlb}")
    print(f"[2-2] out_parent : {out_parent}")
    if total_invalid:
        print(f"[2-2] 무효 skip: {total_invalid}개 (cusum_condbg k<=1)")
    if total_resume:
        print(f"[2-2] resume skip: {total_resume}개")

    if args.jobs <= 1:
        for name, combo, cmd in run_items:
            combo_str = "  ".join(f"{k}={v}" for k, v in combo.items()) if combo else "(default)"
            print(f"\n>>> {name}  {combo_str}")
            subprocess.run(cmd, check=False)
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futures = {}
            for name, combo, cmd in run_items:
                combo_str = "  ".join(f"{k}={v}" for k, v in combo.items()) if combo else "(default)"
                fut = ex.submit(_run_one_captured, cmd, args.stagger_sec)
                futures[fut] = (name, combo_str)
            for fut in as_completed(futures):
                name, combo_str = futures[fut]
                output = fut.result()
                print(f"\n>>> {name}  {combo_str}")
                sys.stdout.flush()
                sys.stdout.buffer.write(output)
                sys.stdout.buffer.flush()

    print("\n[2-2] 완료.")


if __name__ == "__main__":
    main()

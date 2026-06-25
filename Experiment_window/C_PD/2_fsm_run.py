"""
2_fsm_run.py
============
통합 FSM 실행 러너 (파라미터 sweep 지원).

detector(gk2a / metop03 / noaa19)를 받아 해당 count_FSM 디렉터리의
fsm_count_spe_*.py를 glob해서 일괄 실행한다.

  GK2A  : --out 만 전달 (--io/--cache 는 스크립트 내부 하드코딩).
  POES  : --io --cache --out 모두 전달. --no-geo 미전달(geo 태깅 on).
  blc1_fixed(_poes): --mode const --const-csv <1_ana 출력 CSV> 추가.

Sweep:
  --win/--onset/--peak/--k 를 콤마 리스트로 주면 데카르트 곱만큼 실행.
  미지정 시 FSM default 1회 (하위호환).

  FSM별 지원 축:
    quietoff_mad/std, quiet7_mad/std : win, k, onset, peak
    blc1_lowe                         : win, onset, peak  (k 미사용)
    blc1_fixed                        : onset, peak       (win/k 미사용)
  quiet7: win <= 7 조합은 자동 skip (배경 추정 불가 — quiet_days=7 고정).

안전장치:
  --dry   : 조합 수만 출력, 실행 안 함.
  --limit : 앞에서 N개만 실행 (0=무제한, 메커니즘 검증용).
  resume  : sweep 모드에서 runtag 출력폴더 존재 시 skip.

사용:
  python 2_fsm_run.py --detector gk2a [--dry]
  python 2_fsm_run.py --detector gk2a \\
      --win 1,3,5,7,10,15,30 --onset 0,0.1,0.3,0.5,0.7,1.0 \\
      --peak 0,0.5,1.0,2.0,3.0 --k 0,1,3,5,7,10,15,20 [--dry] [--limit 10]
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from itertools import product
from pathlib import Path

HERE = Path(__file__).resolve().parent   # C_PD/

# ── FSM 스크립트 디렉터리 ────────────────────────────────────────────────
_GK2A_FSM_DIR = HERE / "GK2A" / "count_FSM"
_POES_FSM_DIR = HERE / "POES" / "count_FSM"

# ── 출력 경로 매핑 (각 FSM에 넘길 --out 부모; FSM이 내부에서 runtag 하위폴더 생성)
_OUT_DIR = {
    "gk2a":    HERE / "GK2A"  / "KSEM_count"   / "gk2a_output"    / "2_fsm",
    "metop03": HERE / "POES"  / "MetOp03_count" / "metop03_output" / "2_fsm",
    "noaa19":  HERE / "POES"  / "NOAA19_count"  / "noaa19_output"  / "2_fsm",
}

# ── POES 입력 (io 모듈명, cache 경로) ──────────────────────────────────
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

# ── blc1_fixed const-csv 경로 (1_ana 결과 기준; 1_ana 미실행 시 미존재) ──
_CONST_CSV = {
    "gk2a":    HERE / "GK2A"  / "KSEM_count"   / "gk2a_output"    / "1_ana" / "noaa_spe_event_count_stats.csv",
    "metop03": HERE / "POES"  / "MetOp03_count" / "metop03_output" / "1_ana" / "noaa_spe_event_count_stats.csv",
    "noaa19":  HERE / "POES"  / "NOAA19_count"  / "noaa19_output"  / "1_ana" / "noaa_spe_event_count_stats.csv",
}

# ── FSM 타입 분류 (파일명 패턴 기반) ────────────────────────────────────
def _fsm_type(script: Path) -> str:
    name = script.stem
    if "quiet7"     in name: return "quiet7"
    if "quietoff"   in name: return "quietoff"
    if "blc1_fixed" in name: return "blc1_fixed"
    if "blc1_lowe"  in name: return "blc1_lowe"
    raise ValueError(f"알 수 없는 FSM: {name}")

# ── FSM 타입별 지원 sweep 축 ─────────────────────────────────────────────
_FSM_AXES = {
    "quietoff":   ("win", "k", "onset", "peak"),
    "quiet7":     ("win", "k", "onset", "peak"),
    "blc1_lowe":  ("win", "onset", "peak"),
    "blc1_fixed": ("onset", "peak"),
}

# ── internal key → FSM CLI 인자명 ────────────────────────────────────────
_AXIS_CLI = {"win": "--window", "k": "--k", "onset": "--onset", "peak": "--peak"}

# ── TAG 매핑 (runtag 예측용 — FSM 스크립트 내 TAG 상수와 동일해야 함) ──────
_TAG_MAP = {
    "fsm_count_spe_quietoff_mad":      "quietoff_mad",
    "fsm_count_spe_quiet7_mad":        "quiet7_mad",
    "fsm_count_spe_quietoff_std":      "quietoff_std",
    "fsm_count_spe_quiet7_std":        "quiet7_std",
    "fsm_count_spe_blc1_lowe":         "blc1_lowe",
    "fsm_count_spe_blc1_fixed":        "blc1_fixed",
    "fsm_count_spe_quietoff_mad_poes": "quietoff_mad",
    "fsm_count_spe_blc1_lowe_poes":    "blc1_lowe",
    "fsm_count_spe_blc1_fixed_poes":   "blc1_fixed",
}

# ── FSM default 파라미터 (resume runtag 예측에서 미지정 축 보완용) ────────────
# FSM 스크립트 상수와 동기화 필요: BG_WINDOW_DAYS, K, ONSET_FLOOR, PEAK_FLOOR
_FSM_DEFAULTS = {
    "quietoff":   {"win": 30, "k": 10.0, "onset": 0.5, "peak": 2.0},
    "quiet7":     {"win": 30, "k": 10.0, "onset": 0.5, "peak": 2.0},
    "blc1_lowe":  {"win": 5,  "onset": 0.5, "peak": 2.0},
    "blc1_fixed": {"onset": 0.5, "peak": 2.0},
}

# blc1_lowe --mult default (runtag에 m{mult} 포함 — sweep 축이 아님)
_LOWE_MULT_DEFAULT = 1.25


# ── 유틸 ────────────────────────────────────────────────────────────────
def _numstr(v) -> str:
    """10.0 → '10', 0.5 → '0.5'  (FSM 스크립트 내 _numstr과 동일)."""
    f = float(v)
    return str(int(f)) if f.is_integer() else str(f)


def _parse_list(s: str | None, typ) -> list | None:
    if s is None:
        return None
    return [typ(v.strip()) for v in s.split(",")]


def _predict_runtag(script: Path, fsm_type: str, combo: dict) -> str:
    """combo의 각 축값(미지정 시 FSM default)으로 runtag 폴더명 예측."""
    tag = _TAG_MAP[script.stem]
    d   = _FSM_DEFAULTS[fsm_type]
    w   = combo.get("win",   d.get("win"))
    k   = combo.get("k",     d.get("k"))
    on  = combo.get("onset", d["onset"])
    pk  = combo.get("peak",  d["peak"])

    if fsm_type in ("quietoff", "quiet7"):
        return f"{tag}_w{w}_k{_numstr(k)}_on{_numstr(on)}_pk{_numstr(pk)}"
    if fsm_type == "blc1_lowe":
        # mult는 sweep 축 아님 — default 고정
        return f"{tag}_w{w}_m{_numstr(_LOWE_MULT_DEFAULT)}_on{_numstr(on)}_pk{_numstr(pk)}"
    # blc1_fixed → 이 러너는 항상 const 모드
    return f"{tag}_const_on{_numstr(on)}_pk{_numstr(pk)}"


def _build_combos(
    fsm_type: str,
    win_vals:   list | None,
    k_vals:     list | None,
    onset_vals: list | None,
    peak_vals:  list | None,
) -> tuple[list[dict], int]:
    """
    FSM 타입별 지원 축만 사용해 데카르트 곱 생성.
    Returns (combos, n_invalid_skipped).
    n_invalid: quiet7에서 win <= 7로 제외된 조합 수.
    """
    supported = _FSM_AXES[fsm_type]
    sweep_map = {"win": win_vals, "k": k_vals, "onset": onset_vals, "peak": peak_vals}

    axes, vals = [], []
    for ax in supported:
        v = sweep_map[ax]
        if v is not None:
            axes.append(ax)
            vals.append(v)

    if not axes:
        return [{}], 0   # no sweep → FSM default 1회

    combos, n_invalid = [], 0
    for combo_vals in product(*vals):
        combo = dict(zip(axes, combo_vals))
        if fsm_type == "quiet7" and combo.get("win", 9999) <= 7:
            n_invalid += 1
            continue
        combos.append(combo)
    return combos, n_invalid


def _build_cmd(script: Path, detector: str, out_parent: Path, combo: dict) -> list[str]:
    cmd = [sys.executable, str(script)]

    if detector != "gk2a":
        io_mod, cache = _POES_IO[detector]
        cmd += ["--io", io_mod, "--cache", str(cache)]

    cmd += ["--out", str(out_parent)]

    if "blc1_fixed" in script.name:
        csv_path = _CONST_CSV[detector]
        if not csv_path.exists():
            print(f"  WARNING: --const-csv 미존재 (1_ana 먼저 실행 필요) → {csv_path}")
        cmd += ["--mode", "const", "--const-csv", str(csv_path)]

    for ax, val in combo.items():
        cmd += [_AXIS_CLI[ax], str(val)]

    return cmd


def _fsm_dir(detector: str) -> Path:
    return _GK2A_FSM_DIR if detector == "gk2a" else _POES_FSM_DIR


def main():
    ap = argparse.ArgumentParser(description="통합 FSM 실행 러너 (sweep 지원)")
    ap.add_argument("--detector", required=True, choices=["gk2a", "metop03", "noaa19"])
    ap.add_argument("--dry",   action="store_true", help="명령/조합 수만 출력, 실행 안 함")
    ap.add_argument("--limit", type=int, default=0, metavar="N",
                    help="앞에서 N개만 실행 (0=무제한, 메커니즘 검증용)")
    ap.add_argument("--win",   default=None, metavar="LIST",
                    help="window 값 콤마 리스트. 예) 1,3,5,7,10,15,30")
    ap.add_argument("--onset", default=None, metavar="LIST",
                    help="onset 값 콤마 리스트.  예) 0,0.1,0.3,0.5,0.7,1.0")
    ap.add_argument("--peak",  default=None, metavar="LIST",
                    help="peak 값 콤마 리스트.   예) 0,0.5,1.0,2.0,3.0")
    ap.add_argument("--k",     default=None, metavar="LIST",
                    help="k 값 콤마 리스트.      예) 0,1,3,5,7,10,15,20")
    args = ap.parse_args()

    det        = args.detector
    out_parent = _OUT_DIR[det]
    scripts    = sorted(_fsm_dir(det).glob("fsm_count_spe_*.py"))

    if not scripts:
        raise SystemExit(f"[2_fsm] ERROR: FSM 스크립트 없음 → {_fsm_dir(det)}")

    win_vals   = _parse_list(args.win,   int)
    k_vals     = _parse_list(args.k,     float)
    onset_vals = _parse_list(args.onset, float)
    peak_vals  = _parse_list(args.peak,  float)

    is_sweep = any(v is not None for v in (win_vals, k_vals, onset_vals, peak_vals))

    # ── Step 1: 조합 생성 (FSM별) ─────────────────────────────────────────
    script_data: list[tuple[Path, str, list[dict], int]] = []
    total_invalid = 0

    for script in scripts:
        fsm_t = _fsm_type(script)
        combos, n_invalid = _build_combos(fsm_t, win_vals, k_vals, onset_vals, peak_vals)
        script_data.append((script, fsm_t, combos, n_invalid))
        total_invalid += n_invalid

    # ── Step 2: --dry 출력 ───────────────────────────────────────────────
    if args.dry:
        print(f"[2_fsm] --dry  detector={det}  FSM {len(scripts)}개")
        if is_sweep:
            parts = []
            total_valid = 0
            for script, fsm_t, combos, n_invalid in script_data:
                short = script.stem.replace("fsm_count_spe_", "").replace("_poes", "")
                n = len(combos)
                total_valid += n
                entry = f"{short} {n}"
                if n_invalid:
                    entry += f"(+{n_invalid}skip)"
                parts.append(entry)
            print(f"[2_fsm] {det}: " + ", ".join(parts))
            print(f"[2_fsm] 총 조합: {total_valid}개", end="")
            if total_invalid:
                print(f"  (무효 skip: {total_invalid}개 -- quiet7 win<=7)", end="")
            print()
        else:
            print(f"[2_fsm] out_parent : {out_parent}")
            print(f"\n[2_fsm] 명령 ({len(scripts)}개):")
            for script, fsm_t, combos, _ in script_data:
                cmd = _build_cmd(script, det, out_parent, {})
                print(f"  [{script.name}]")
                print("    " + " ".join(str(t) for t in cmd))
        print("\n[2_fsm] --dry: 실행 생략.")
        return

    # ── Step 3: resume 필터 + 실행 목록 구성 ──────────────────────────────
    run_items: list[tuple[str, dict, list[str]]] = []
    total_resume = 0

    for script, fsm_t, combos, _ in script_data:
        for combo in combos:
            if is_sweep:
                runtag  = _predict_runtag(script, fsm_t, combo)
                out_dir = out_parent / runtag
                if out_dir.exists():
                    total_resume += 1
                    continue
            cmd = _build_cmd(script, det, out_parent, combo)
            run_items.append((script.name, combo, cmd))

    # ── Step 4: 실행 ────────────────────────────────────────────────────
    if not run_items:
        print(f"[2_fsm] 실행할 조합 없음 (전부 resume skip).")
        return

    if args.limit > 0:
        run_items = run_items[:args.limit]
        print(f"[2_fsm] --limit {args.limit}: {len(run_items)}개만 실행.")

    total_valid = sum(len(c) for _, _, c, _ in script_data)
    print(f"[2_fsm] detector={det}  조합 {total_valid}개  실행 {len(run_items)}개")
    print(f"[2_fsm] out_parent : {out_parent}")
    if total_invalid:
        print(f"[2_fsm] 무효 skip: {total_invalid}개 (quiet7 win<=7)")
    if total_resume:
        print(f"[2_fsm] resume skip: {total_resume}개")

    for name, combo, cmd in run_items:
        combo_str = "  ".join(f"{k}={v}" for k, v in combo.items()) if combo else "(default)"
        print(f"\n>>> {name}  {combo_str}")
        subprocess.run(cmd, check=False)

    print("\n[2_fsm] 완료.")


if __name__ == "__main__":
    main()

"""
2-2_fsm_binned_run.py
======================
POES count를 |자기위도(maglat)| bin별 미니 캐시로 쪼개고, bin마다 기존 FSM
스윕(2_fsm_run.py 동일 로직)을 실행하는 러너. 2_fsm_run.py를 복사해 bin
루프를 내장한 것 — 래퍼 아님.

절대 원칙:
  - count_FSM/의 FSM 스크립트 4종(quietoff_mad/blc1_lowe/blc1_fixed/cusum,
    각 _poes 접미사): 한 줄도 수정하지 않는다. subprocess로만 호출.
  - 2_fsm_run.py 원본: 수정하지 않는다 (이 파일은 그 복사+개조본).
  - 원본 캐시 디렉터리(poes_*_cache_parquet/): 읽기 전용, 절대 쓰지 않는다.
  - 기존 {detector}_output/2_fsm/ 아래에는 절대 쓰지 않는다. 이 러너의 출력은
    전부 {detector}_output/2-2_fsm_binned/ 아래.
  - 대상은 POES 2종(metop03, noaa19)만. gk2a(정지궤도)는 maglat 비닝이
    무의미하므로 --detector choices에서 제외.

geo(lat/lon) -> coords_igrf.dipole_maglat() -> |maglat| bin 판정
(Bmag/IGRF 추가 계산 없음 -- diag_rolling_threshold_poes.py와 동일 방식 재사용).

온보드 등가성: bin별 rolling 배경(quietoff/quiet7/cusum의 win 윈도)은 해당 bin의
과거 샘플에만 의존한다 -- 다른 bin 샘플을 참조하지 않는다. 따라서 여기서 하는
"위성 전체 시계열을 bin별로 미리 쪼갠 뒤 bin마다 배치로 FSM을 돌리는" 방식은,
실제 온보드에서 샘플이 들어올 때마다 (1) IGRF로 maglat 계산 -> (2) 해당 bin으로
라우팅 -> (3) 그 bin의 FSM 상태만 갱신 하는 스트리밍 처리와 수학적으로 동일한
결과를 낸다 (bin 경계·윈도 정의가 같다면 두 처리 순서가 각 bin의 rolling 통계에
주는 표본 집합이 동일하기 때문). 즉 이 스크립트는 온보드 구현의 지상 검증용
재정식화(reformulation)이지, 근사가 아니다.

서브커맨드:
  build : bin 미니 캐시 생성 (원본 캐시와 동일 파일 포맷 -- io 모듈 무수정으로 열림)
  run   : 빌드된 bin마다 2_fsm_run.py와 동일한 sweep 로직 실행

사용:
  # bin 캐시 빌드 (기본 10도 간격 9개 bin: 0-10,...,80-90)
  python 2-2_fsm_binned_run.py build --detector metop03
  python 2-2_fsm_binned_run.py build --detector metop03 --bins "0,10,20,30,40,50,60,70,80,90"
  python 2-2_fsm_binned_run.py build --detector metop03 --channels pro_tel0_p5,omni_p7
  python 2-2_fsm_binned_run.py build --detector metop03 --dry
  python 2-2_fsm_binned_run.py build --detector metop03 --force

  # bin별 sweep 실행 (2_fsm_run.py의 sweep 인자 전부 동일 지원)
  python 2-2_fsm_binned_run.py run --detector metop03 --dry
  python 2-2_fsm_binned_run.py run --detector metop03 \\
      --bins-select "60_70,70_80,80_90" --baselines quietoff \\
      --win 10,30 --k 5,10 --onset 0.5 --peak 2.0
  python 2-2_fsm_binned_run.py run --detector metop03 --limit 2

병렬 실행 (2_fsm_run.py와 동일):
  --jobs N (기본 1) : bin별 조합 실행을 N개 워커로 동시 실행. 기본값(1)에서는
      실행 경로가 기존과 완전히 동일. N>1일 때만 워커 풀 사용 -- 서브프로세스
      stdout/stderr를 캡처했다가 해당 조합이 끝나는 시점에 한 덩어리로 그대로
      출력(내용/포맷 무변경, resume/성공-실패 카운트 로직도 무변경).
  --stagger-sec S (기본 0) : --jobs>1일 때 서브프로세스 launch 최소 간격[sec].
  python 2-2_fsm_binned_run.py run --detector metop03 --jobs 4 --stagger-sec 0.5

출력 트리 (runtag는 bin 정보를 포함하지 않음 -- 폴더 계층이 bin을 담당,
기존 3_event/4_summarize의 runtag 파서와 호환 유지 목적):
  {detector}_output/2-2_fsm_binned/bin_60_70/<runtag>/fsm_onset_<runtag>.csv

resume: 원본 2_fsm_run.py는 출력 폴더 존재만으로 완료를 판정한다(빈 폴더를
완료로 오인하는 약점). 이 복사본은 fsm_onset_<runtag>.csv 파일 존재로 판정한다.
"""
from __future__ import annotations
import argparse
import importlib
import json
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent  # C_PD/

sys.path.insert(0, str(HERE / "POES"))
sys.path.insert(0, str(HERE / "POES" / "count_FSM"))
import coords_igrf  # dipole_maglat (bin 판정용, 좌표만 -- Bmag/IGRF 계산 없음)

_POES_FSM_DIR = HERE / "POES" / "count_FSM"

_POES_IO = {
    "metop03": ("poes_metop03_io", HERE / "POES" / "MetOp03_count" / "poes_metop03_cache_parquet"),
    "noaa19":  ("poes_noaa19_io",  HERE / "POES" / "NOAA19_count"  / "poes_noaa19_cache_parquet"),
}
_OUT_ROOT = {
    "metop03": HERE / "POES" / "MetOp03_count" / "metop03_output" / "2-2_fsm_binned",
    "noaa19":  HERE / "POES" / "NOAA19_count"  / "noaa19_output"  / "2-2_fsm_binned",
}
# blc1_fixed의 --const-csv -- 1_ana 출력(bin과 무관한 전역 통계표), 원본 2_fsm_run.py와 동일 경로.
_CONST_CSV = {
    "metop03": HERE / "POES" / "MetOp03_count" / "metop03_output" / "1_ana" / "noaa_spe_event_count_stats.csv",
    "noaa19":  HERE / "POES" / "NOAA19_count"  / "noaa19_output"  / "1_ana" / "noaa_spe_event_count_stats.csv",
}

_DEFAULT_BINS = "0,10,20,30,40,50,60,70,80,90"
_BIN_TOLERANCE = pd.Timedelta("2min")  # 캐시 1분 케이던스 기준

# ── FSM 타입 분류 (POES 4종만 -- count_FSM/에 실제 존재하는 파일명 기준) ─────
def _fsm_type(script: Path) -> str:
    name = script.stem
    if "quietoff"   in name: return "quietoff"
    if "blc1_fixed" in name: return "blc1_fixed"
    if "blc1_lowe"  in name: return "blc1_lowe"
    if "cusum"      in name: return "cusum"
    raise ValueError(f"알 수 없는 FSM: {name}")

_FSM_AXES = {
    "quietoff":   ("win", "k", "onset", "peak"),
    "blc1_lowe":  ("win", "onset", "peak"),
    "blc1_fixed": ("onset", "peak"),
    "cusum":      ("win", "k", "onset", "peak"),
}

_CUSUM_H_FIXED = 5.0
_LOWE_MULT_DEFAULT = 1.25
_AXIS_CLI = {"win": "--window", "k": "--k", "onset": "--onset", "peak": "--peak"}

_TAG_MAP = {
    "fsm_count_spe_quietoff_mad_poes": "quietoff_mad",
    "fsm_count_spe_blc1_lowe_poes":    "blc1_lowe",
    "fsm_count_spe_blc1_fixed_poes":   "blc1_fixed",
    "fsm_count_spe_cusum_poes":        "cusum",
}

_FSM_DEFAULTS = {
    "quietoff":   {"win": 30, "k": 10.0, "onset": 0.5, "peak": 2.0},
    "blc1_lowe":  {"win": 5,  "onset": 0.5, "peak": 2.0},
    "blc1_fixed": {"onset": 0.5, "peak": 2.0},
    "cusum":      {"win": 30, "k": 3.0, "onset": 0.5, "peak": 2.0},
}


# ── sweep 유틸 (2_fsm_run.py와 동일 로직 -- POES 전용으로 축약 복제) ────────
def _numstr(v) -> str:
    f = float(v)
    return str(int(f)) if f.is_integer() else str(f)


def _parse_list(s: str | None, typ) -> list | None:
    if s is None:
        return None
    return [typ(v.strip()) for v in s.split(",")]


def _predict_runtag(script: Path, fsm_type: str, combo: dict) -> str:
    tag = _TAG_MAP[script.stem]
    d   = _FSM_DEFAULTS[fsm_type]
    w   = combo.get("win",   d.get("win"))
    k   = combo.get("k",     d.get("k"))
    on  = combo.get("onset", d["onset"])
    pk  = combo.get("peak",  d["peak"])

    if fsm_type == "quietoff":
        return f"{tag}_w{w}_k{_numstr(k)}_on{_numstr(on)}_pk{_numstr(pk)}"
    if fsm_type == "cusum":
        return f"{tag}_w{w}_k{_numstr(k)}_h{_numstr(_CUSUM_H_FIXED)}_on{_numstr(on)}_pk{_numstr(pk)}"
    if fsm_type == "blc1_lowe":
        return f"{tag}_w{w}_m{_numstr(_LOWE_MULT_DEFAULT)}_on{_numstr(on)}_pk{_numstr(pk)}"
    return f"{tag}_const_on{_numstr(on)}_pk{_numstr(pk)}"


def _build_combos(fsm_type: str, win_vals, k_vals, onset_vals, peak_vals) -> tuple[list[dict], int]:
    supported = _FSM_AXES[fsm_type]
    sweep_map = {"win": win_vals, "k": k_vals, "onset": onset_vals, "peak": peak_vals}

    axes, vals = [], []
    for ax in supported:
        v = sweep_map[ax]
        if v is not None:
            axes.append(ax)
            vals.append(v)

    if not axes:
        return [{}], 0

    combos, n_invalid = [], 0
    for combo_vals in product(*vals):
        combo = dict(zip(axes, combo_vals))
        if fsm_type == "cusum" and combo.get("k", 9999) <= 1:
            n_invalid += 1  # ln(k)<=0 -> 검출 불가 (FSM 자체 방어와 동일 기준)
            continue
        combos.append(combo)
    return combos, n_invalid


def _build_cmd(script: Path, detector: str, cache_dir: Path, out_parent: Path, combo: dict) -> list[str]:
    io_mod, _ = _POES_IO[detector]
    cmd = [sys.executable, str(script), "--io", io_mod, "--cache", str(cache_dir), "--out", str(out_parent)]

    if "blc1_fixed" in script.name:
        csv_path = _CONST_CSV[detector]
        if not csv_path.exists():
            print(f"  WARNING: --const-csv 미존재 (1_ana 먼저 실행 필요) -> {csv_path}")
        cmd += ["--mode", "const", "--const-csv", str(csv_path)]

    if _fsm_type(script) == "cusum":
        cmd += ["--h", str(_CUSUM_H_FIXED)]

    for ax, val in combo.items():
        cmd += [_AXIS_CLI[ax], str(val)]

    return cmd


# ── io 모듈 임포트 ──────────────────────────────────────────────────────
def _import_poes_io(detector: str):
    io_name, cache = _POES_IO[detector]
    for p in (str(cache.parent), str(HERE / "POES")):
        if p not in sys.path:
            sys.path.insert(0, p)
    return importlib.import_module(io_name)


# ── bin 파싱/명명 유틸 ───────────────────────────────────────────────────
def _parse_bins(spec: str) -> list[tuple[float, float]]:
    """--bins "0,10,...,90" -> [(0,10),(10,20),...,(80,90)] (edge 오름차순, 0~90)."""
    try:
        edges = [float(v.strip()) for v in spec.split(",") if v.strip() != ""]
    except ValueError:
        raise SystemExit(f"[binned] --bins 형식 오류: '{spec}' (콤마로 구분된 숫자 리스트여야 함)")
    if len(edges) < 2:
        raise SystemExit(f"[binned] --bins 값 부족: '{spec}' (경계값 2개 이상 필요)")
    if any(e < 0.0 or e > 90.0 for e in edges):
        raise SystemExit(f"[binned] --bins 범위 오류: '{spec}' (0~90 사이여야 함)")
    for i in range(len(edges) - 1):
        if edges[i] >= edges[i + 1]:
            raise SystemExit(f"[binned] --bins 오름차순 아님: '{spec}'")
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def _fmt_edge(v: float) -> str:
    return f"{int(round(v)):02d}" if float(v).is_integer() else str(v)


def _bin_name(lo: float, hi: float) -> str:
    return f"bin_{_fmt_edge(lo)}_{_fmt_edge(hi)}"


def _bins_root(detector: str) -> Path:
    """bin 캐시 루트 폴더명은 반드시 _cache_parquet로 끝나야 .gitignore(*_cache_parquet/)에
    잡힌다 -- 원본 cache.name(예: poes_metop03_cache_parquet)의 _cache_parquet 접미사를
    떼고 _mlatbins_cache_parquet를 붙인다 (예: poes_metop03_mlatbins_cache_parquet)."""
    _, cache = _POES_IO[detector]
    base = cache.name.removesuffix("_cache_parquet")
    return cache.parent / f"{base}_mlatbins_cache_parquet"


def _parse_channels_arg(io, spec: str | None):
    """--channels "pro_tel0_p5,omni_p7" -> [('pro','tel0','p5'),...]. None이면 전 채널(None 반환)."""
    if not spec:
        return None
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        tpl = io.fname_to_tuple(part)
        if tpl is None:
            raise SystemExit(f"[binned:build] --channels 형식 오류: '{part}'")
        out.append(tpl)
    if not out:
        raise SystemExit(f"[binned:build] --channels 값 없음: '{spec}'")
    return out


def _discover_bins(root: Path) -> list[tuple[float, float, str, Path]]:
    """root 아래 bin_* 폴더 -> [(lo,hi,name,path),...] (lo 오름차순).
    각 폴더의 _poes_meta.json 의 mlat_bin 을 우선 신뢰, 없으면 폴더명에서 파싱."""
    if not root.exists():
        return []
    out = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not d.name.startswith("bin_"):
            continue
        lo = hi = None
        meta_path = d / "_poes_meta.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    m = json.load(f)
                mb = m.get("mlat_bin")
                if mb and len(mb) == 2:
                    lo, hi = float(mb[0]), float(mb[1])
            except Exception:
                lo = hi = None
        if lo is None:
            m2 = re.match(r"^bin_(.+)_(.+)$", d.name)
            if not m2:
                continue
            try:
                lo, hi = float(m2.group(1)), float(m2.group(2))
            except ValueError:
                continue
        out.append((lo, hi, d.name, d))
    return sorted(out, key=lambda t: t[0])


def _parse_bins_select(spec: str) -> list[tuple[float, float]]:
    """--bins-select "60_70,70_80" -> [(60.0,70.0),(70.0,80.0)]."""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^([\d.]+)_([\d.]+)$", part)
        if not m:
            raise SystemExit(f"[binned:run] --bins-select 형식 오류: '{part}' (기대: lo_hi, 예: 60_70)")
        out.append((float(m.group(1)), float(m.group(2))))
    if not out:
        raise SystemExit(f"[binned:run] --bins-select 값 없음: '{spec}'")
    return out


# ── --jobs>1 병렬 실행 (기본 --jobs 1 경로는 절대 안 거침, 2_fsm_run.py와 동일) ──
_launch_lock = threading.Lock()
_last_launch = [0.0]


def _stagger_wait(stagger_sec: float) -> None:
    """연속 서브프로세스 launch 사이 최소 간격 보장 (--stagger-sec, HDD 동시 로드 경합 완화)."""
    if stagger_sec <= 0:
        return
    with _launch_lock:
        now = time.monotonic()
        wait = _last_launch[0] + stagger_sec - now
        if wait > 0:
            time.sleep(wait)
        _last_launch[0] = time.monotonic()


def _run_one_captured(cmd: list[str], stagger_sec: float) -> tuple[int, bytes]:
    """subprocess stdout+stderr를 합쳐 raw bytes로 캡처(인코딩 왕복 없이 그대로 보존)."""
    _stagger_wait(stagger_sec)
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return res.returncode, res.stdout


# ══════════════════════════════════════════════════════════════════════
# build
# ══════════════════════════════════════════════════════════════════════
def cmd_build(args):
    det = args.detector
    io = _import_poes_io(det)
    _, cache = _POES_IO[det]
    bins = _parse_bins(args.bins)
    channels = _parse_channels_arg(io, args.channels)

    print(f"[binned:build] detector={det}  cache={cache}")
    band_desc = ", ".join(
        f"[{lo:g},{hi:g}{']' if i == len(bins) - 1 else ')'}"
        for i, (lo, hi) in enumerate(bins))
    print(f"[binned:build] bins({len(bins)}): {band_desc}")

    geo = io.get_geo(str(cache), with_bmag=False)
    if geo is None or geo.empty or "lat" not in geo.columns or "lon" not in geo.columns:
        raise SystemExit("[binned:build] ERROR geo(lat/lon) 로드 실패 -- bin 캐시 생성 불가")
    if geo.index.tz is None:
        geo.index = geo.index.tz_localize("UTC")
    amag_native = pd.Series(
        np.abs(coords_igrf.dipole_maglat(geo["lat"].to_numpy(), geo["lon"].to_numpy())),
        index=geo.index)

    df, meta = io.load(str(cache), channels=channels)
    if df.empty:
        raise SystemExit("[binned:build] ERROR count 캐시 로드 실패 (빈 데이터)")
    n_total = len(df)
    print(f"[binned:build] count 원본 행수 {n_total:,}  채널 {df.shape[1]}개"
          + (f" (--channels 지정: {args.channels})" if channels is not None else " (전 채널)"))

    amag_on_count = amag_native.reindex(df.index, method="nearest", tolerance=_BIN_TOLERANCE)
    n_reindex_nan = int(amag_on_count.isna().sum())

    root = _bins_root(det)

    if args.dry:
        print(f"[binned:build] --dry: bin {len(bins)}개 -> {root}")
        for i, (lo, hi) in enumerate(bins):
            is_last = i == len(bins) - 1
            m = (amag_on_count >= lo) & ((amag_on_count <= hi) if is_last else (amag_on_count < hi))
            print(f"  {_bin_name(lo, hi)}: count행 {int(m.sum()):,}개 -> {root / _bin_name(lo, hi)}")
        print("[binned:build] --dry: 생성 생략.")
        return

    root.mkdir(parents=True, exist_ok=True)
    summary = []
    for i, (lo, hi) in enumerate(bins):
        is_last = i == len(bins) - 1
        name = _bin_name(lo, hi)
        bin_dir = root / name
        mask_count = (amag_on_count >= lo) & ((amag_on_count <= hi) if is_last else (amag_on_count < hi))
        n_rows = int(mask_count.sum())

        if bin_dir.exists() and not args.force:
            print(f"[binned:build] {name}: 이미 존재 -> skip (--force로 재생성)  (count행 {n_rows:,})")
            summary.append((name, n_rows))
            continue
        if bin_dir.exists() and args.force:
            shutil.rmtree(bin_dir)
        bin_dir.mkdir(parents=True)

        mask_geo = (amag_native >= lo) & ((amag_native <= hi) if is_last else (amag_native < hi))
        geo_bin = geo.loc[mask_geo]
        for g in ("lat", "lon", "alt"):
            if g in geo_bin.columns:
                s = geo_bin[g].dropna()
                if not s.empty:
                    s.to_frame(name=g).to_parquet(bin_dir / f"{g}.parquet", compression="snappy")

        for col in df.columns:
            s = df[col].dropna()
            m = mask_count.reindex(s.index, fill_value=False)
            s_bin = s[m]
            if s_bin.empty:
                continue
            stem = io.tuple_to_fname(tuple(col))
            s_bin.to_frame(name="count").to_parquet(bin_dir / f"{stem}.parquet", compression="snappy")

        bin_meta = dict(meta)
        bin_meta["mlat_bin"]     = [lo, hi]
        bin_meta["mlat_method"]  = "dipole"
        bin_meta["parent_cache"] = str(cache)
        bin_meta["built"]        = pd.Timestamp.now(tz="UTC").isoformat()
        bin_meta["bins_spec"]    = args.bins
        with open(bin_dir / "_poes_meta.json", "w", encoding="utf-8") as f:
            json.dump(bin_meta, f, indent=2, ensure_ascii=False)

        print(f"[binned:build] {name}: count행 {n_rows:,} 저장 -> {bin_dir}")
        summary.append((name, n_rows))

    total_binned = sum(n for _, n in summary)
    diff = n_total - total_binned
    pct = (diff / n_total * 100) if n_total else 0.0
    print("\n[binned:build] 무결성 요약:")
    for name, n in summary:
        print(f"  {name:14s} {n:>12,}")
    print(f"  {'합계':14s} {total_binned:>12,}")
    print(f"  {'원본 행수':14s} {n_total:>12,}")
    print(f"  미부여(bin 미소속): {diff:,}개 ({pct:.4f}%)  "
          f"(reindex tolerance({_BIN_TOLERANCE}) 초과로 maglat 미부여: {n_reindex_nan:,}개)")
    if diff != n_reindex_nan:
        print(f"  주의: 미부여 수({diff:,})가 tolerance 초과 수({n_reindex_nan:,})와 다름 -- "
              f"--bins 범위가 [0,90] 전체를 덮지 않음(설계된 동작일 수 있음)")


# ══════════════════════════════════════════════════════════════════════
# run
# ══════════════════════════════════════════════════════════════════════
def cmd_run(args):
    det = args.detector
    root = _bins_root(det)
    available = _discover_bins(root)
    if not available:
        raise SystemExit(f"[binned:run] ERROR bin 캐시 없음 -> 먼저 build 실행: {root}")

    if args.bins_select:
        wanted = _parse_bins_select(args.bins_select)
        name_map = {name: (lo, hi, name, path) for lo, hi, name, path in available}
        selected = []
        for lo, hi in wanted:
            name = _bin_name(lo, hi)
            if name not in name_map:
                raise SystemExit(f"[binned:run] ERROR 존재하지 않는 bin: {name}  "
                                 f"(가능: {', '.join(sorted(name_map))})")
            selected.append(name_map[name])
    else:
        selected = available

    scripts = sorted(_POES_FSM_DIR.glob("fsm_count_spe_*.py"))
    if args.baselines:
        wanted_b = {b.strip() for b in args.baselines.split(",") if b.strip()}
        scripts = [s for s in scripts if _fsm_type(s) in wanted_b]
    if not scripts:
        raise SystemExit(f"[binned:run] ERROR FSM 스크립트 없음 -> {_POES_FSM_DIR}"
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
    n_combo_per_bin = sum(len(c) for _, _, c, _ in script_data)

    out_root = _OUT_ROOT[det]

    if args.dry:
        print(f"[binned:run] --dry  detector={det}  bin {len(selected)}개  FSM {len(scripts)}개")
        parts = []
        for script, fsm_t, combos, n_invalid in script_data:
            short = script.stem.replace("fsm_count_spe_", "").replace("_poes", "")
            entry = f"{short} {len(combos)}"
            if n_invalid:
                entry += f"(+{n_invalid}skip)"
            parts.append(entry)
        print(f"[binned:run] bin당 조합: " + ", ".join(parts) + f"  (합계 {n_combo_per_bin}개)")
        print(f"[binned:run] bin 목록({len(selected)}): " + ", ".join(n for _, _, n, _ in selected))
        print(f"[binned:run] 총 호출 수: bin {len(selected)} x 조합 {n_combo_per_bin} "
              f"= {len(selected) * n_combo_per_bin}")
        if selected and script_data:
            script, fsm_t, combos, _ = script_data[0]
            example_combo = combos[0] if combos else {}
            lo, hi, name, path = selected[0]
            cmd = _build_cmd(script, det, path, out_root / name, example_combo)
            print("[binned:run] 예시 명령:")
            print("    " + " ".join(str(t) for t in cmd))
        print("[binned:run] --dry: 실행 생략.")
        return

    bin_summary = []
    for lo, hi, name, cache_path in selected:
        out_parent = out_root / name
        print(f"\n[binned:run] === {name}  cache={cache_path} ===")

        run_items = []
        n_resume = 0
        for script, fsm_t, combos, _ in script_data:
            for combo in combos:
                runtag  = _predict_runtag(script, fsm_t, combo)
                out_dir = out_parent / runtag
                onset_csv = out_dir / f"fsm_onset_{runtag}.csv"
                if is_sweep and onset_csv.exists():
                    n_resume += 1
                    continue
                run_items.append((script, combo, runtag, onset_csv))

        if args.limit > 0 and len(run_items) > args.limit:
            print(f"[binned:run] {name}: --limit {args.limit}: {len(run_items)}개 중 {args.limit}개만 실행.")
            run_items = run_items[:args.limit]

        n_ok = n_fail = 0
        try:
            if args.jobs <= 1:
                for script, combo, runtag, onset_csv in run_items:
                    combo_str = "  ".join(f"{k}={v}" for k, v in combo.items()) if combo else "(default)"
                    print(f"  >>> {script.name}  {combo_str}")
                    cmd = _build_cmd(script, det, cache_path, out_parent, combo)
                    res = subprocess.run(cmd, check=False)
                    if res.returncode == 0:
                        n_ok += 1
                    else:
                        n_fail += 1
                        print(f"  !! 실패 (returncode={res.returncode}): {script.name} {combo_str}")
            else:
                with ThreadPoolExecutor(max_workers=args.jobs) as ex:
                    futures = {}
                    for script, combo, runtag, onset_csv in run_items:
                        combo_str = ("  ".join(f"{k}={v}" for k, v in combo.items())
                                    if combo else "(default)")
                        cmd = _build_cmd(script, det, cache_path, out_parent, combo)
                        fut = ex.submit(_run_one_captured, cmd, args.stagger_sec)
                        futures[fut] = (script.name, combo_str)
                    for fut in as_completed(futures):
                        sname, combo_str = futures[fut]
                        returncode, output = fut.result()
                        print(f"  >>> {sname}  {combo_str}")
                        sys.stdout.flush()
                        sys.stdout.buffer.write(output)
                        sys.stdout.buffer.flush()
                        if returncode == 0:
                            n_ok += 1
                        else:
                            n_fail += 1
                            print(f"  !! 실패 (returncode={returncode}): {sname} {combo_str}")
        except Exception as e:
            print(f"  !! {name} 처리 중 예외 발생, 다음 bin으로 진행: {e}")

        print(f"[binned:run] {name} 완료: 실행 {n_ok + n_fail}개 (성공 {n_ok} 실패 {n_fail})  "
              f"resume skip {n_resume}개 (기준: fsm_onset_<runtag>.csv 존재)")
        bin_summary.append((name, n_ok, n_fail, n_resume))

    print("\n[binned:run] 전체 요약:")
    tot_ok = tot_fail = tot_resume = 0
    for name, n_ok, n_fail, n_resume in bin_summary:
        print(f"  {name:14s} 성공 {n_ok:>4}  실패 {n_fail:>4}  resume {n_resume:>4}")
        tot_ok += n_ok; tot_fail += n_fail; tot_resume += n_resume
    print(f"  {'합계':14s} 성공 {tot_ok:>4}  실패 {tot_fail:>4}  resume {tot_resume:>4}")


def main():
    ap = argparse.ArgumentParser(
        description="POES maglat bin 캐시 빌더 + bin별 FSM sweep 러너 (2_fsm_run.py 복사 개조)")
    sub = ap.add_subparsers(dest="command", required=True)

    ap_b = sub.add_parser("build", help="|maglat| bin 미니 캐시 생성")
    ap_b.add_argument("--detector", required=True, choices=["metop03", "noaa19"])
    ap_b.add_argument("--bins", default=_DEFAULT_BINS,
                      help=f'bin 경계 콤마 리스트 (기본 "{_DEFAULT_BINS}", 마지막 bin만 상한 포함)')
    ap_b.add_argument("--channels", default=None,
                      help="콤마 리스트 (예: pro_tel0_p5,omni_p7). 미지정 시 전 채널")
    ap_b.add_argument("--force", action="store_true", help="이미 존재하는 bin도 지우고 재생성")
    ap_b.add_argument("--dry", action="store_true", help="bin별 예상 행수만 출력, 생성 안 함")
    ap_b.set_defaults(func=cmd_build)

    ap_r = sub.add_parser("run", help="빌드된 bin마다 FSM sweep 실행")
    ap_r.add_argument("--detector", required=True, choices=["metop03", "noaa19"])
    ap_r.add_argument("--bins-select", default=None, metavar="LIST",
                      help='실행할 bin "lo_hi" 콤마 리스트 (예: "60_70,70_80,80_90"). 미지정=빌드된 전 bin')
    ap_r.add_argument("--baselines", default=None, metavar="LIST",
                      help="baseline 필터 콤마 리스트 (예: quietoff,cusum). 미지정=전체")
    ap_r.add_argument("--win",   default=None, metavar="LIST", help="window 값 콤마 리스트")
    ap_r.add_argument("--onset", default=None, metavar="LIST", help="onset 값 콤마 리스트")
    ap_r.add_argument("--peak",  default=None, metavar="LIST", help="peak 값 콤마 리스트")
    ap_r.add_argument("--k",     default=None, metavar="LIST", help="k 값 콤마 리스트")
    ap_r.add_argument("--dry",   action="store_true", help="bin/조합/호출 수와 예시 명령만 출력")
    ap_r.add_argument("--limit", type=int, default=0, metavar="N",
                      help="bin당 앞에서 N개만 실행 (0=무제한, 메커니즘 검증용)")
    ap_r.add_argument("--jobs", type=int, default=1, metavar="N",
                      help="bin당 병렬 실행 워커 수 (기본 1 = 기존과 동일한 순차 실행). "
                           "N>1일 때만 워커 풀 사용")
    ap_r.add_argument("--stagger-sec", type=float, default=0.0, metavar="S",
                      help="--jobs>1일 때 서브프로세스 시작 최소 간격[sec] "
                           "(HDD 동시 parquet 로드 경합 완화, 기본 0)")
    ap_r.set_defaults(func=cmd_run)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

"""
ksem_io.py
==========
KSEM 캐시 파일(JSON / Parquet)을 읽고 쓰는 범용 I/O 모듈.

플롯, ML, 통계, 카탈로그 비교 등 어디서든 동일한 인터페이스로
깔끔한 pandas DataFrame / Series 를 반환합니다.

─────────────────────────────────────────────────────────────────
데이터 모델
─────────────────────────────────────────────────────────────────
  load_* 함수는 모두 (df, meta) 튜플을 반환합니다.

  df   : pd.DataFrame
    index   = DatetimeIndex (UTC, 1분 간격)
    columns = MultiIndex  (pd_key, side, logic)
              예: ("PD1", "A", "O"), ("PD2", "B", "TRASH"), ...

    이 형태로 쓰면:
      df["PD1"]           → PD1 전체 (side × logic 컬럼)
      df["PD1"]["A"]      → PD1-A 전체
      df["PD1", "A", "O"] → PD1-A의 O 채널 Series

  meta : dict
    {
      "created"     : str   ISO8601 생성일시
      "start"       : str   "YYYYMM"
      "end"         : str   "YYYYMM"
      "energy_meta" : dict  { logic: (lo_kev, hi_kev) }
    }

─────────────────────────────────────────────────────────────────
공개 API
─────────────────────────────────────────────────────────────────
  load_json(path_or_paths)     → (df, meta)
  load_parquet(directory)      → (df, meta)
  load(path_or_paths)          → (df, meta)   ← 포맷 자동 감지

  save_json(df, meta, path, indent=None)
  save_parquet(df, meta, directory)

  to_plot_format(df)           → pd_data dict  (플롯 코드 호환)
  from_plot_format(pd_data, energy_meta) → (df, meta)

─────────────────────────────────────────────────────────────────
사용 예
─────────────────────────────────────────────────────────────────
  import ksem_io

  # 로드 (포맷 자동 감지)
  df, meta = ksem_io.load("ksem_cache.json")
  df, meta = ksem_io.load("ksem_cache_parquet/")     # 디렉터리
  df, meta = ksem_io.load(["chunk1.json", "chunk2.json"])

  # 조회
  s = df["PD1", "A", "O"]                       # Series
  sub = df.loc["2022-01":"2022-03", "PD1"]       # 기간 슬라이싱
  proton = df.loc[:, (slice(None), slice(None),
                      ["O","OU","CR","OUT"])]    # 양성자만

  # 저장 (포맷 변환)
  ksem_io.save_parquet(df, meta, "ksem_cache_parquet/")
  ksem_io.save_json(df, meta, "ksem_cache.json")

  # 플롯 코드와 함께 쓸 때
  pd_data = ksem_io.to_plot_format(df)
  # → pd_data["PD1"][("A","O")] : pd.Series  (기존 plot 함수 그대로 사용)
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

# ─────────────────────────────────────────────────────────────────
# 도메인 상수
# ─────────────────────────────────────────────────────────────────
PD_KEYS    = ["PD1", "PD2", "PD3"]
SIDES      = ["A", "B"]
LOGICS     = ["O", "OU", "CR", "OUT", "F", "FT", "FTU", "FTUO", "TRASH"]
PROTON_LOGICS   = ["O", "OU", "CR", "OUT"]
ELECTRON_LOGICS = ["F", "FT", "FTU", "FTUO"]

_META_FILENAME = "_ksem_meta.json"   # Parquet 디렉터리 안에 저장될 메타 파일


# ─────────────────────────────────────────────────────────────────
# 내부 헬퍼
# ─────────────────────────────────────────────────────────────────
def _empty_df() -> pd.DataFrame:
    cols = pd.MultiIndex.from_tuples(
        [(p, s, l) for p in PD_KEYS for s in SIDES for l in LOGICS],
        names=["pd_key", "side", "logic"],
    )
    return pd.DataFrame(columns=cols, dtype=float)


def _merge_dfs(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(dfs).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    return combined


def _norm_meta(raw_meta: dict) -> dict:
    """energy_meta 값을 항상 (lo, hi) 튜플로 정규화."""
    em = raw_meta.get("energy_meta", {})
    normalized = {}
    for k, v in em.items():
        if v is None or (isinstance(v, (list, tuple)) and len(v) == 2):
            normalized[k] = (v[0], v[1]) if v else (None, None)
        else:
            normalized[k] = (None, None)
    return {**raw_meta, "energy_meta": normalized}


# ─────────────────────────────────────────────────────────────────
# JSON 로드
# ─────────────────────────────────────────────────────────────────
def _load_single_json(path: Path) -> Tuple[pd.DataFrame, dict]:
    size_mb = path.stat().st_size / 1024 / 1024
    print(f"[ksem_io] JSON 로드: {path.name}  ({size_mb:.1f} MB)", flush=True)

    with open(path, "r", encoding="utf-8") as f:
        cache = json.load(f)

    meta = _norm_meta(cache.get("meta", {}))
    data_section = cache.get("data", {})

    series_dict: Dict[Tuple[str, str, str], pd.Series] = {}
    for pd_key in PD_KEYS:
        for side in SIDES:
            for logic in LOGICS:
                raw = data_section.get(pd_key, {}).get(side, {}).get(logic, {})
                if raw:
                    idx  = pd.to_datetime(list(raw.keys()))
                    vals = list(raw.values())
                    s = pd.Series(vals, index=idx, dtype=float, name=(pd_key, side, logic))
                    s.index.name = "Time"
                    series_dict[(pd_key, side, logic)] = s
                else:
                    series_dict[(pd_key, side, logic)] = pd.Series(
                        dtype=float, name=(pd_key, side, logic))

    df = pd.DataFrame(series_dict)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["pd_key", "side", "logic"])
    df = df.sort_index()
    return df, meta


def load_json(
    path_or_paths: Union[str, Path, List[Union[str, Path]]],
) -> Tuple[pd.DataFrame, dict]:
    """JSON 캐시 파일(단일 또는 청크 리스트)을 로드합니다."""
    if isinstance(path_or_paths, (str, Path)):
        path_or_paths = [path_or_paths]

    dfs, meta = [], {}
    for p in path_or_paths:
        df_i, m_i = _load_single_json(Path(p))
        dfs.append(df_i)
        if not meta:
            meta = m_i

    result = _merge_dfs(dfs) if len(dfs) > 1 else dfs[0]
    return result, meta


# ─────────────────────────────────────────────────────────────────
# Parquet 로드 / 저장
# ─────────────────────────────────────────────────────────────────
def _ensure_pyarrow():
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        raise ImportError(
            "Parquet 지원을 위해 pyarrow 가 필요합니다.\n"
            "  pip install pyarrow"
        )


def load_parquet(directory: Union[str, Path]) -> Tuple[pd.DataFrame, dict]:
    """
    Parquet 디렉터리에서 KSEM 데이터를 로드합니다.
    디렉터리 구조: <directory>/<pd_key>_<side>_<logic>.parquet
    """
    _ensure_pyarrow()
    directory = Path(directory)
    print(f"[ksem_io] Parquet 로드: {directory}", flush=True)

    # 메타 로드
    meta_path = directory / _META_FILENAME
    meta: dict = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = _norm_meta(json.load(f))

    series_dict: Dict[Tuple[str, str, str], pd.Series] = {}
    for pd_key in PD_KEYS:
        for side in SIDES:
            for logic in LOGICS:
                fpath = directory / f"{pd_key}_{side}_{logic}.parquet"
                if fpath.exists():
                    s = pd.read_parquet(fpath).squeeze()
                    s.index.name = "Time"
                    s.name = (pd_key, side, logic)
                    series_dict[(pd_key, side, logic)] = s.astype(float)
                else:
                    series_dict[(pd_key, side, logic)] = pd.Series(
                        dtype=float, name=(pd_key, side, logic))

    df = pd.DataFrame(series_dict)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["pd_key", "side", "logic"])
    df = df.sort_index()
    return df, meta


def save_parquet(
    df: pd.DataFrame,
    meta: dict,
    directory: Union[str, Path],
) -> None:
    """
    DataFrame을 Parquet 디렉터리에 저장합니다.
    컬럼별로 개별 .parquet 파일로 나눠 저장 (빠른 부분 로드 가능).
    """
    _ensure_pyarrow()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    print(f"[ksem_io] Parquet 저장: {directory}", flush=True)

    for col in df.columns:
        pd_key, side, logic = col
        fpath = directory / f"{pd_key}_{side}_{logic}.parquet"
        s = df[col].dropna()
        s.to_frame(name="count").to_parquet(fpath, compression="snappy")

    # 메타 직렬화 (튜플 → 리스트 변환)
    meta_out = dict(meta)
    if "energy_meta" in meta_out:
        meta_out["energy_meta"] = {
            k: list(v) if v else [None, None]
            for k, v in meta_out["energy_meta"].items()
        }
    with open(directory / _META_FILENAME, "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2, ensure_ascii=False)

    size_total = sum(
        (directory / f"{p}_{s}_{l}.parquet").stat().st_size
        for p in PD_KEYS for s in SIDES for l in LOGICS
        if (directory / f"{p}_{s}_{l}.parquet").exists()
    )
    print(f"  저장 완료: {len(list(directory.glob('*.parquet')))}개 파일, "
          f"총 {size_total/1024/1024:.1f} MB")


# ─────────────────────────────────────────────────────────────────
# JSON 저장
# ─────────────────────────────────────────────────────────────────
def save_json(
    df: pd.DataFrame,
    meta: dict,
    path: Union[str, Path],
    indent: Optional[int] = None,
) -> None:
    """DataFrame을 JSON 캐시 파일로 저장합니다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[ksem_io] JSON 저장: {path}", flush=True)

    data_section: dict = {}
    for pd_key in PD_KEYS:
        data_section[pd_key] = {}
        for side in SIDES:
            data_section[pd_key][side] = {}
            for logic in LOGICS:
                s = df.get((pd_key, side, logic), pd.Series(dtype=float))
                s = s.dropna()
                data_section[pd_key][side][logic] = {
                    ts.isoformat(): float(v)
                    for ts, v in s.items()
                }

    meta_out = dict(meta)
    if "energy_meta" in meta_out:
        meta_out["energy_meta"] = {
            k: list(v) if v else [None, None]
            for k, v in meta_out["energy_meta"].items()
        }

    cache = {"meta": meta_out, "data": data_section}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=indent, ensure_ascii=False)

    size_mb = path.stat().st_size / 1024 / 1024
    print(f"  저장 완료: {size_mb:.1f} MB")


# ─────────────────────────────────────────────────────────────────
# 포맷 자동 감지 로드
# ─────────────────────────────────────────────────────────────────
def load(
    path_or_paths: Union[str, Path, List[Union[str, Path]]],
) -> Tuple[pd.DataFrame, dict]:
    """
    파일 경로를 보고 JSON / Parquet을 자동으로 감지해 로드합니다.

    - .json 파일 (단일 또는 리스트) → load_json
    - 디렉터리 (parquet 포함)       → load_parquet
    - 확장자 없는 단일 경로         → 디렉터리로 시도, 실패 시 JSON
    """
    if isinstance(path_or_paths, list):
        # 리스트는 JSON 청크로 간주
        return load_json(path_or_paths)

    p = Path(path_or_paths)
    if p.is_dir():
        return load_parquet(p)
    if p.suffix.lower() == ".json":
        return load_json(p)
    if p.suffix.lower() in (".parquet",):
        raise ValueError(
            "개별 .parquet 파일이 아니라 parquet 디렉터리 경로를 입력하세요.\n"
            f"  예: load('{p.parent}')"
        )
    # 확장자 불명확 → JSON 시도
    return load_json(p)


# ─────────────────────────────────────────────────────────────────
# 플롯 코드 호환 변환
# ─────────────────────────────────────────────────────────────────
def to_plot_format(
    df: pd.DataFrame,
) -> Dict[str, Dict[Tuple[str, str], pd.Series]]:
    """
    ksem_io DataFrame → ksem_vs_goes_plot.py 의 pd_data 형식으로 변환.

    반환:
      pd_data[pd_key][(side, logic)] = pd.Series
      예: pd_data["PD1"][("A", "O")]
    """
    pd_data: Dict[str, Dict[Tuple[str, str], pd.Series]] = {}
    for pd_key in PD_KEYS:
        pd_data[pd_key] = {}
        for side in SIDES:
            for logic in LOGICS:
                try:
                    pd_data[pd_key][(side, logic)] = df[pd_key, side, logic]
                except KeyError:
                    pd_data[pd_key][(side, logic)] = pd.Series(dtype=float)
    return pd_data


def from_plot_format(
    pd_data: Dict[str, Dict[Tuple[str, str], pd.Series]],
    energy_meta: Optional[dict] = None,
    start: str = "",
    end: str = "",
) -> Tuple[pd.DataFrame, dict]:
    """
    ksem_vs_goes_plot.py 의 pd_data → ksem_io DataFrame 으로 변환.
    기존 코드 결과를 저장하거나 다른 용도로 활용할 때 사용.
    """
    series_dict: Dict[Tuple[str, str, str], pd.Series] = {}
    for pd_key, side_logics in pd_data.items():
        for (side, logic), s in side_logics.items():
            series_dict[(pd_key, side, logic)] = s

    df = pd.DataFrame(series_dict)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["pd_key", "side", "logic"])
    df = df.sort_index()

    em = {}
    if energy_meta:
        for k, v in energy_meta.items():
            em[k] = tuple(v) if v else (None, None)

    meta = {
        "created":     datetime.now(timezone.utc).isoformat(),
        "start":       start,
        "end":         end,
        "energy_meta": em,
    }
    return df, meta


# ─────────────────────────────────────────────────────────────────
# 편의 조회 함수
# ─────────────────────────────────────────────────────────────────
def get_series(
    df: pd.DataFrame,
    pd_key: str,
    side: str,
    logic: str,
    t_start: Optional[str] = None,
    t_end:   Optional[str] = None,
) -> pd.Series:
    """단일 채널 Series 반환. 시간 범위 필터링 포함."""
    s = df[pd_key, side, logic]
    if t_start or t_end:
        s = s.loc[t_start:t_end]
    return s


def get_proton(
    df: pd.DataFrame,
    pd_key: str = "PD1",
    side: str = "A",
    t_start: Optional[str] = None,
    t_end:   Optional[str] = None,
) -> pd.DataFrame:
    """양성자 채널(O, OU, CR, OUT) DataFrame 반환."""
    sub = df.loc[:, (pd_key, side, PROTON_LOGICS)]
    if t_start or t_end:
        sub = sub.loc[t_start:t_end]
    return sub


def get_electron(
    df: pd.DataFrame,
    pd_key: str = "PD1",
    side: str = "A",
    t_start: Optional[str] = None,
    t_end:   Optional[str] = None,
) -> pd.DataFrame:
    """전자 채널(F, FT, FTU, FTUO) DataFrame 반환."""
    sub = df.loc[:, (pd_key, side, ELECTRON_LOGICS)]
    if t_start or t_end:
        sub = sub.loc[t_start:t_end]
    return sub


def summary(df: pd.DataFrame, meta: dict) -> None:
    """캐시 요약 정보를 출력합니다."""
    print("=" * 55)
    print(f"  생성일시 : {meta.get('created', '?')[:19]}")
    print(f"  범위     : {meta.get('start', '?')} ~ {meta.get('end', '?')}")
    print(f"  시간 인덱스: {df.index[0]} ~ {df.index[-1]}" if len(df) else "  (데이터 없음)")
    print(f"  총 row 수: {len(df):,}")
    print()
    print("  에너지 밴드:")
    for logic, (lo, hi) in meta.get("energy_meta", {}).items():
        print(f"    {logic:6s}: {lo} - {hi} keV")
    print()
    print("  채널별 유효 row 수 (결측 제외):")
    for pd_key in PD_KEYS:
        row_parts = []
        for side in SIDES:
            for logic in LOGICS:
                try:
                    n = df[pd_key, side, logic].notna().sum()
                    row_parts.append(f"{side}/{logic}={n:,}")
                except KeyError:
                    pass
        print(f"    {pd_key}: " + "  ".join(row_parts[:4]) + " ...")
    print("=" * 55)

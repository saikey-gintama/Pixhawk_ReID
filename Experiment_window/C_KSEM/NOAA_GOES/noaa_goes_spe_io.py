"""
noaa_goes_spe_io.py
==============
NOAA SPE (Solar Proton Events) 캐시 파일(JSON / Parquet)을 읽고 쓰는 범용 I/O 모듈.

noaa_goes_spe_build_cache.py 가 만든 캐시를 어디서든 import 해서 사용합니다.
파싱·다운로드 로직은 noaa_goes_spe_build_cache.py 에만 있습니다.

─────────────────────────────────────────────────────────────────
데이터 모델
─────────────────────────────────────────────────────────────────
  load_* 함수는 모두 (df, meta) 튜플을 반환합니다.

  df   : pd.DataFrame
    index   = DatetimeIndex (UTC, begin_time 기준)
    columns = [
        "max_time",          # datetime
        "max_pfu",           # float  (>10 MeV 최대 플럭스, pfu)
        "region",            # str    (AR 번호, 없으면 "")
        "location",          # str    (예: N14W25, 없으면 "")
        "flare_class",       # str    (예: "X2", 없으면 "")
        "flare_importance",  # str    (예: "2B", 없으면 "")
        "flare_time",        # datetime (없으면 NaT)
        "type2",             # bool | None
        "type4",             # bool | None
        "cme_speed_kms",     # float | NaN
    ]

  meta : dict
    {
      "created"  : str   ISO8601 캐시 생성일시
      "source"   : str   원본 URL
      "fetched"  : str   ISO8601 HTML 파싱 일시
      "n_events" : int   총 이벤트 수
    }

─────────────────────────────────────────────────────────────────
공개 API
─────────────────────────────────────────────────────────────────
  load_json(path)          → (df, meta)
  load_parquet(directory)  → (df, meta)
  load(path_or_dir)        → (df, meta)   ← 포맷 자동 감지

  save_json(df, meta, path, indent=None)
  save_parquet(df, meta, directory)

  filter_by_pfu(df, min_pfu)              → df
  filter_by_date(df, start, end)          → df
  summary(df, meta)

─────────────────────────────────────────────────────────────────
사용 예
─────────────────────────────────────────────────────────────────
  import noaa_goes_spe_io

  df, meta = noaa_goes_spe_io.load("spe_cache.json")
  df, meta = noaa_goes_spe_io.load("spe_cache_parquet/")   # 포맷 자동 감지

  # 필터링
  big   = noaa_goes_spe_io.filter_by_pfu(df, 1000)
  year  = noaa_goes_spe_io.filter_by_date(df, "2003-01-01", "2003-12-31")
  xray  = df[df["flare_class"].str.startswith("X")]

  noaa_goes_spe_io.summary(df, meta)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd

# ─────────────────────────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────────────────────────
SOURCE_URL = (
    "https://www.ngdc.noaa.gov/stp/space-weather/interplanetary-data/"
    "solar-proton-events/SEP%20page%20code.html"
)
_META_FILENAME = "_spe_meta.json"

# index 제외 컬럼 순서 (저장 / 로드 일관성)
COLUMNS = [
    "max_time",
    "max_pfu",
    "region",
    "location",
    "flare_class",
    "flare_importance",
    "flare_time",
    "type2",
    "type4",
    "cme_speed_kms",
]

_DT_COLS = {"max_time", "flare_time"}   # JSON 역직렬화 시 datetime 복원 대상


# ─────────────────────────────────────────────────────────────────
# 내부 직렬화 헬퍼
# ─────────────────────────────────────────────────────────────────
def _df_to_records(df: pd.DataFrame) -> list:
    """DataFrame → JSON 직렬화 가능한 records 리스트 (index 포함)."""
    out = []
    for idx, row in df.iterrows():
        rec: dict = {"begin_time": idx.isoformat() if pd.notna(idx) else None}
        for col in df.columns:
            v = row[col]
            if isinstance(v, (pd.Timestamp, datetime)):
                rec[col] = v.isoformat() if pd.notna(v) else None
            elif isinstance(v, float) and pd.isna(v):
                rec[col] = None
            else:
                rec[col] = v
        out.append(rec)
    return out


def _records_to_df(records: list) -> pd.DataFrame:
    """JSON records 리스트 → DataFrame (index = begin_time)."""
    rows = []
    for rec in records:
        r: dict = {}
        for k, v in rec.items():
            if k in _DT_COLS | {"begin_time"} and v is not None:
                r[k] = pd.to_datetime(v, utc=True)
            else:
                r[k] = v
        rows.append(r)

    df = pd.DataFrame(rows, columns=["begin_time"] + COLUMNS)
    df = df.set_index("begin_time").sort_index()
    df.index.name = "begin_time"

    df["max_pfu"]       = pd.to_numeric(df["max_pfu"],       errors="coerce")
    df["cme_speed_kms"] = pd.to_numeric(df["cme_speed_kms"], errors="coerce")
    return df


# ─────────────────────────────────────────────────────────────────
# JSON 저장 / 로드
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
    print(f"[noaa_goes_spe_io] JSON 저장: {path}", flush=True)

    cache = {"meta": meta, "data": _df_to_records(df)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=indent, ensure_ascii=False, default=str)

    size_mb = path.stat().st_size / 1024 / 1024
    print(f"  저장 완료: {size_mb:.2f} MB  ({meta.get('n_events', len(df))}개 이벤트)")


def load_json(path: Union[str, Path]) -> Tuple[pd.DataFrame, dict]:
    """JSON 캐시 파일에서 SPE 데이터를 로드합니다."""
    path = Path(path)
    size_mb = path.stat().st_size / 1024 / 1024
    print(f"[noaa_goes_spe_io] JSON 로드: {path.name}  ({size_mb:.2f} MB)", flush=True)

    with open(path, "r", encoding="utf-8") as f:
        cache = json.load(f)

    df   = _records_to_df(cache.get("data", []))
    meta = cache.get("meta", {})
    print(f"  로드 완료: {len(df)}개 이벤트")
    return df, meta


# ─────────────────────────────────────────────────────────────────
# Parquet 저장 / 로드
# ─────────────────────────────────────────────────────────────────
def _ensure_pyarrow() -> None:
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        raise ImportError(
            "Parquet 지원을 위해 pyarrow 가 필요합니다.\n"
            "  pip install pyarrow"
        )


def save_parquet(
    df: pd.DataFrame,
    meta: dict,
    directory: Union[str, Path],
) -> None:
    """DataFrame을 Parquet 디렉터리에 저장합니다."""
    _ensure_pyarrow()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    print(f"[noaa_goes_spe_io] Parquet 저장: {directory}", flush=True)

    # bool 컬럼은 object 로 변환해 None 허용
    df_save = df.copy()
    for col in ["type2", "type4"]:
        df_save[col] = df_save[col].astype(object)

    fpath = directory / "spe_events.parquet"
    df_save.to_parquet(fpath, compression="snappy")

    with open(directory / _META_FILENAME, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    size_mb = fpath.stat().st_size / 1024 / 1024
    print(f"  저장 완료: {size_mb:.2f} MB  ({meta.get('n_events', len(df))}개 이벤트)")


def load_parquet(directory: Union[str, Path]) -> Tuple[pd.DataFrame, dict]:
    """Parquet 디렉터리에서 SPE 데이터를 로드합니다."""
    _ensure_pyarrow()
    directory = Path(directory)
    print(f"[noaa_goes_spe_io] Parquet 로드: {directory}", flush=True)

    meta_path = directory / _META_FILENAME
    meta: dict = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    fpath = directory / "spe_events.parquet"
    if not fpath.exists():
        raise FileNotFoundError(f"Parquet 파일 없음: {fpath}")

    df = pd.read_parquet(fpath)
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "begin_time"
    print(f"  로드 완료: {len(df)}개 이벤트")
    return df, meta


# ─────────────────────────────────────────────────────────────────
# 포맷 자동 감지 로드
# ─────────────────────────────────────────────────────────────────
def load(path_or_dir: Union[str, Path]) -> Tuple[pd.DataFrame, dict]:
    """
    경로를 보고 JSON / Parquet 을 자동으로 감지해 로드합니다.
      .json 파일   → load_json
      디렉터리     → load_parquet
    """
    p = Path(path_or_dir)
    if p.is_dir():
        return load_parquet(p)
    if p.suffix.lower() == ".json":
        return load_json(p)
    return load_json(p)   # 확장자 불명확 → JSON 시도


# ─────────────────────────────────────────────────────────────────
# 편의 조회 함수
# ─────────────────────────────────────────────────────────────────
def filter_by_pfu(df: pd.DataFrame, min_pfu: float) -> pd.DataFrame:
    """최소 플럭스(pfu) 이상인 이벤트만 반환."""
    return df[df["max_pfu"] >= min_pfu]


def filter_by_date(
    df: pd.DataFrame,
    start: Optional[str] = None,
    end:   Optional[str] = None,
) -> pd.DataFrame:
    """날짜 범위 필터링. start/end 는 pandas 가 해석하는 형식 ('YYYY-MM-DD' 등)."""
    return df.loc[start:end]


def summary(df: pd.DataFrame, meta: dict) -> None:
    """캐시 요약 정보를 출력합니다."""
    print("=" * 55)
    print(f"  출처     : {meta.get('source', '?')}")
    print(f"  생성일시 : {meta.get('created', '?')[:19]}")
    print(f"  총 이벤트: {meta.get('n_events', len(df))}개")
    if len(df):
        print(f"  기간     : {df.index[0].date()} ~ {df.index[-1].date()}")
        print(f"  최대 pfu : {df['max_pfu'].max():,.0f}")
        print(f"  X-class  : {df['flare_class'].str.startswith('X').sum()}개")
    print("=" * 55)

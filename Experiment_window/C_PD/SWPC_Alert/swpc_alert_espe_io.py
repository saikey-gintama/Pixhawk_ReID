"""
swpc_alert_espe_io.py
=====================
SWPC >2 MeV 전자 경보(ALTEF3) 캐시 파일(JSON / Parquet) I/O 모듈.

noaa_goes_spe_io.py 와 동일한 API·데이터모델. ground truth가 proton SPE 대신
GEO 방사선대 전자 경보(>2 MeV, 1000 pfu 초과)라는 점만 다르다. 스키마를 맞춰
noaa_goes_spe_match 계열 매칭기가 그대로 동작한다.

데이터 모델:
  df : index = DatetimeIndex(UTC, begin_time)
       columns = ["max_time", "max_pfu", "station", "serial"]
         max_pfu = Yesterday Maximum 2MeV Flux 최댓값(pfu),
                   CONTINUED 없으면 임계 1000.
  meta : {created, source, fetched, n_events}

공개 API: load_json/load_parquet/load, save_json/save_parquet,
          filter_by_pfu, filter_by_date, summary
"""

from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd

SOURCE_URL = "SWPC alerts archive (ALTEF3: >2 MeV Electron 2MeV Integral Flux > 1000 pfu)"
_META_FILENAME = "_espe_meta.json"

COLUMNS = ["max_time", "max_pfu", "station", "serial"]
_DT_COLS = {"max_time"}


def _df_to_records(df: pd.DataFrame) -> list:
    out = []
    for idx, row in df.iterrows():
        rec = {"begin_time": idx.isoformat() if pd.notna(idx) else None}
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
    rows = []
    for rec in records:
        r = {}
        for k, v in rec.items():
            if k in _DT_COLS | {"begin_time"} and v is not None:
                r[k] = pd.to_datetime(v, utc=True)
            else:
                r[k] = v
        rows.append(r)
    df = pd.DataFrame(rows, columns=["begin_time"] + COLUMNS)
    df = df.set_index("begin_time").sort_index()
    df.index.name = "begin_time"
    df["max_pfu"] = pd.to_numeric(df["max_pfu"], errors="coerce")
    return df


def save_json(df, meta, path, indent=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[espe_io] JSON 저장: {path}", flush=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "data": _df_to_records(df)},
                  f, indent=indent, ensure_ascii=False, default=str)
    print(f"  저장 완료: {meta.get('n_events', len(df))}개 이벤트")


def load_json(path) -> Tuple[pd.DataFrame, dict]:
    path = Path(path)
    print(f"[espe_io] JSON 로드: {path.name}", flush=True)
    with open(path, "r", encoding="utf-8") as f:
        cache = json.load(f)
    df = _records_to_df(cache.get("data", []))
    meta = cache.get("meta", {})
    print(f"  로드 완료: {len(df)}개 이벤트")
    return df, meta


def _ensure_pyarrow():
    try:
        import pyarrow  # noqa
    except ImportError:
        raise ImportError("Parquet 지원에 pyarrow 필요: pip install pyarrow")


def save_parquet(df, meta, directory):
    _ensure_pyarrow()
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    print(f"[espe_io] Parquet 저장: {directory}", flush=True)
    fpath = directory / "espe_events.parquet"
    df.to_parquet(fpath, compression="snappy")
    with open(directory / _META_FILENAME, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  저장 완료: {meta.get('n_events', len(df))}개 이벤트")


def load_parquet(directory) -> Tuple[pd.DataFrame, dict]:
    _ensure_pyarrow()
    directory = Path(directory)
    print(f"[espe_io] Parquet 로드: {directory}", flush=True)
    meta = {}
    mp = directory / _META_FILENAME
    if mp.exists():
        with open(mp, "r", encoding="utf-8") as f:
            meta = json.load(f)
    fpath = directory / "espe_events.parquet"
    if not fpath.exists():
        raise FileNotFoundError(f"Parquet 파일 없음: {fpath}")
    df = pd.read_parquet(fpath)
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "begin_time"
    print(f"  로드 완료: {len(df)}개 이벤트")
    return df, meta


def load(path_or_dir) -> Tuple[pd.DataFrame, dict]:
    p = Path(path_or_dir)
    if p.is_dir():
        return load_parquet(p)
    if p.suffix.lower() == ".json":
        return load_json(p)
    return load_json(p)


def filter_by_pfu(df, min_pfu):
    """최소 flux(pfu) 이상 이벤트만 반환 (noaa_goes_spe_io와 동일)."""
    return df[df["max_pfu"] >= min_pfu]


def filter_by_date(df, start=None, end=None):
    return df.loc[start:end]


def summary(df, meta):
    print("=" * 55)
    print(f"  출처     : {meta.get('source','?')}")
    print(f"  생성일시 : {meta.get('created','?')[:19]}")
    print(f"  총 이벤트: {meta.get('n_events', len(df))}개 (>2MeV 전자 경보)")
    if len(df):
        print(f"  기간     : {df.index[0].date()} ~ {df.index[-1].date()}")
        print(f"  최대 pfu : {df['max_pfu'].max():,.0f}")
    print("=" * 55)

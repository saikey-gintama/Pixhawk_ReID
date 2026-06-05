"""
kma_ksem_flux_io.py
===============
KSEM flux 캐시 파일(JSON / Parquet)을 읽고 쓰는 범용 I/O 모듈.
kma_ksem_io.py 와 동일한 논리 구성.

─────────────────────────────────────────────────────────────────
데이터 모델
─────────────────────────────────────────────────────────────────
  load() 반환값: (sensor_data, meta)

  sensor_data : dict
    sensor_data['electron'] = pd.DataFrame
    sensor_data['proton']   = pd.DataFrame

    DataFrame:
      index   = DatetimeIndex (UTC, 1분 간격)
      E1..EN  : 전자 flux (cm-2 sr-1 s-1 keV-1), NaN=결측
      E1..EN  : 양성자 flux
      Att_Flag, Det_Flag1..N : 상태 플래그
      E1_QEF..EN_QEF         : 품질 평가 플래그
      IntegNum               : 평균 사용 데이터 수

  meta : dict
    created, start, end
    sensors:
      electron:
        version, data_description, units
        energy_ranges : { 'E1': '50-100 keV', ... }
        flux_channels : ['E1', 'E2', ...]
        flag_channels : ['Att_Flag', ...]
      proton: (동일 구조)

─────────────────────────────────────────────────────────────────
사용 예
─────────────────────────────────────────────────────────────────
  import kma_ksem_flux_io

  sensor_data, meta = kma_ksem_flux_io.load('kma_ksem_flux_cache.json')
  sensor_data, meta = kma_ksem_flux_io.load('kma_ksem_flux_cache_parquet/')

  df_e = sensor_data['electron']          # 전자 DataFrame
  df_p = sensor_data['proton']            # 양성자 DataFrame

  # 특정 기간 슬라이싱
  event = df_e.loc['2022-09-05':'2022-09-08', ['E1','E2','E3']]

  # 품질 필터 (QEF==0 인 것만)
  good = df_e[df_e['E1_QEF'] == 0]['E1']

  kma_ksem_flux_io.summary(sensor_data, meta)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd


_META_FILENAME  = '_kma_ksem_flux_meta.json'
_SENSORS        = ['electron', 'proton']


# ─────────────────────────────────────────────────────────────────
# 내부 헬퍼
# ─────────────────────────────────────────────────────────────────
def _ensure_pyarrow():
    try:
        import pyarrow  # noqa
    except ImportError:
        raise ImportError('pip install pyarrow')


def _df_to_records(df: pd.DataFrame) -> dict:
    """DataFrame → {col: {iso_ts: value}} JSON 직렬화용"""
    out = {}
    for col in df.columns:
        s = df[col]
        col_dict = {}
        for ts, v in s.items():
            key = ts.isoformat()
            if isinstance(v, float) and np.isnan(v):
                col_dict[key] = None
            elif hasattr(v, 'item'):
                col_dict[key] = v.item()
            else:
                col_dict[key] = v
        out[col] = col_dict
    return out


def _records_to_df(records: dict) -> pd.DataFrame:
    """JSON records → DataFrame"""
    if not records:
        return pd.DataFrame()
    cols = {}
    idx  = None
    for col, col_dict in records.items():
        if idx is None:
            idx = pd.DatetimeIndex(col_dict.keys()).tz_convert('UTC')
        cols[col] = list(col_dict.values())
    df = pd.DataFrame(cols, index=idx)
    df.index.name = 'Time'
    return df


# ─────────────────────────────────────────────────────────────────
# JSON
# ─────────────────────────────────────────────────────────────────
def load_json(path: Union[str, Path]) -> tuple[dict, dict]:
    path = Path(path)
    size_mb = path.stat().st_size / 1024 / 1024
    print(f'[kma_ksem_flux_io] JSON 로드: {path.name}  ({size_mb:.1f} MB)', flush=True)

    with open(path, 'r', encoding='utf-8') as f:
        cache = json.load(f)

    meta = cache.get('meta', {})
    sensor_data = {}
    for sensor in _SENSORS:
        records = cache.get('data', {}).get(sensor, {})
        sensor_data[sensor] = _records_to_df(records)

    return sensor_data, meta


def save_json(
    sensor_data: dict,
    meta: dict,
    path: Union[str, Path],
    indent: Optional[int] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f'[kma_ksem_flux_io] JSON 저장: {path}', flush=True)

    data_section = {}
    for sensor, df in sensor_data.items():
        if df.empty:
            data_section[sensor] = {}
            continue
        print(f'  {sensor}: {len(df):,} rows 직렬화 중...', flush=True)
        data_section[sensor] = _df_to_records(df)

    cache = {'meta': meta, 'data': data_section}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=indent, ensure_ascii=False)

    size_mb = path.stat().st_size / 1024 / 1024
    print(f'  저장 완료: {size_mb:.1f} MB')


# ─────────────────────────────────────────────────────────────────
# Parquet
# ─────────────────────────────────────────────────────────────────
def load_parquet(directory: Union[str, Path]) -> tuple[dict, dict]:
    _ensure_pyarrow()
    directory = Path(directory)
    print(f'[kma_ksem_flux_io] Parquet 로드: {directory}', flush=True)

    meta: dict = {}
    meta_path = directory / _META_FILENAME
    if meta_path.exists():
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)

    sensor_data = {}
    for sensor in _SENSORS:
        fpath = directory / f'{sensor}.parquet'
        if fpath.exists():
            df = pd.read_parquet(fpath)
            df.index = pd.DatetimeIndex(df.index).tz_localize('UTC') \
                if df.index.tz is None else df.index
            df.index.name = 'Time'
            sensor_data[sensor] = df
        else:
            sensor_data[sensor] = pd.DataFrame()

    return sensor_data, meta


def save_parquet(
    sensor_data: dict,
    meta: dict,
    directory: Union[str, Path],
) -> None:
    _ensure_pyarrow()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    print(f'[kma_ksem_flux_io] Parquet 저장: {directory}', flush=True)

    for sensor, df in sensor_data.items():
        if df.empty:
            continue
        fpath = directory / f'{sensor}.parquet'
        df.to_parquet(fpath, compression='snappy')
        size_mb = fpath.stat().st_size / 1024 / 1024
        print(f'  {sensor}: {len(df):,} rows → {size_mb:.1f} MB')

    with open(directory / _META_FILENAME, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print('  저장 완료')


# ─────────────────────────────────────────────────────────────────
# 포맷 자동 감지 (kma_ksem_io.load 와 동일한 인터페이스)
# ─────────────────────────────────────────────────────────────────
def load(path: Union[str, Path]) -> tuple[dict, dict]:
    p = Path(path)
    if p.is_dir():
        return load_parquet(p)
    return load_json(p)


# ─────────────────────────────────────────────────────────────────
# 편의 조회 함수
# ─────────────────────────────────────────────────────────────────
def get_flux(
    sensor_data: dict,
    sensor: str,
    channels: Optional[list[str]] = None,
    t_start: Optional[str] = None,
    t_end:   Optional[str] = None,
    quality_filter: bool = False,
) -> pd.DataFrame:
    """
    flux 채널만 추출. quality_filter=True 이면 QEF==0 인 값만 반환.

    Parameters
    ----------
    sensor         : 'electron' 또는 'proton'
    channels       : ['E1','E2',...] 또는 None (전체)
    t_start/t_end  : 시간 범위 필터
    quality_filter : True 이면 QEF!=0 인 값을 NaN으로 마스킹
    """
    df = sensor_data.get(sensor, pd.DataFrame())
    if df.empty:
        return df

    # flux 컬럼만
    flux_cols = [c for c in df.columns
                 if re.match(r'^[EP]\d+$', c)] if channels is None else channels
    sub = df[flux_cols].copy()

    if quality_filter:
        for ch in flux_cols:
            qef_col = f'{ch}_QEF'
            if qef_col in df.columns:
                sub.loc[df[qef_col] != 0, ch] = float('nan')

    if t_start or t_end:
        sub = sub.loc[t_start:t_end]

    return sub


def summary(sensor_data: dict, meta: dict) -> None:
    print('=' * 55)
    print(f"  생성일시 : {meta.get('created', '?')[:19]}")
    print(f"  범위     : {meta.get('start', '?')} ~ {meta.get('end', '?')}")
    for sensor in _SENSORS:
        df = sensor_data.get(sensor, pd.DataFrame())
        sm = meta.get('sensors', {}).get(sensor, {})
        print()
        print(f"  [{sensor}]")
        print(f"    버전    : {sm.get('version', '?')}")
        print(f"    row 수  : {len(df):,}")
        if len(df):
            print(f"    기간    : {df.index[0]} ~ {df.index[-1]}")
        flux_chs = sm.get('flux_channels', [])
        print(f"    채널    : {flux_chs}")
        er = sm.get('energy_ranges', {})
        for ch in flux_chs:
            print(f"      {ch}: {er.get(ch, '?')}")
    print('=' * 55)


# re 모듈 필요
import re

# ─────────────────────────────────────────────────────────────────
# 단독 실행 시 요약 출력
# ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('사용법: python kma_ksem_flux_io.py <cache.json|parquet_dir>')
        sys.exit(0)
    sd, m = load(sys.argv[1])
    summary(sd, m)

"""
poes_noaa19_io.py  (v2 — 채널 1개 = 파일 1개, MultiIndex 반환)
==========
POES/MetOp SEM-2 count 캐시 I/O 모듈. ksem_io.py 와 동일한 형태:
  - 반환 타입: (df, meta).  df 는 MultiIndex 컬럼 DataFrame.
  - 채널 인덱싱: 튜플 (species, direction, energy).
      ('pro','tel0','p5'), ('pro','tel90','p1'), ('omni','-','p6'), ('ele','tel90','e3')
  - Parquet: 채널 1개 = 파일 1개 (pro_tel0_p5.parquet, omni_p6.parquet, ele_tel90_e3.parquet ...).
             빠른 부분 로드 — 특정 채널만 쓸 때 그 파일 하나만 읽으면 됨.
  - geo(lat/lon/alt): df 에 섞지 않고 개별 파일(lat.parquet ...)로 저장.
                      get_geo() 로만 접근. get_geo(..., with_bmag=True) 면
                      coords_igrf 로 Bmag/maglat 파생.

데이터 모델
  load() 반환값: (df, meta)
  df : index=DatetimeIndex(UTC), columns=MultiIndex[species,direction,energy]
       species 'pro'|'omni'|'ele' / direction 'tel0'|'tel90'|'-' / energy 'p1'..'p9'|'e1'..'e3'
       값 = count rate(#/s), NaN=결측/valid_range 밖
  채널 인덱싱: df[('pro','tel90','p5')] / df['pro'] / df.loc[t0:t1, ('omni','-','p6')]
  geo: get_geo(dir, with_bmag=True) -> DataFrame[lat,lon,alt(,Bmag,maglat)]
  meta: created,start,end,satellite,global_attrs,*_variants,meta_periods,
        channels(저장 채널 목록), energy_ranges(파일명 stem 키)

사용 예
  import poes_noaa19_io as io
  df, meta = io.load('poes_metop03_cache_parquet')
  s = df[('pro','tel90','p5')].dropna()          # 채널 1개만 로드하려면:
  df1, _ = io.load('poes_metop03_cache_parquet', channels=[('pro','tel90','p5')])
  geo = io.get_geo('poes_metop03_cache_parquet', with_bmag=True)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional, Union, List, Tuple

import numpy as np
import pandas as pd


_META_FILENAME = '_poes_meta.json'
_GEO_CHANNELS  = ['alt', 'lat', 'lon']

_CH_DIR_RE  = re.compile(r'^mep_(pro|ele)_(tel0|tel90)_cps_(p\d+|e\d+)$')
_CH_OMNI_RE = re.compile(r'^mep_(omni)_cps_(p\d+)$')


def flat_to_tuple(ch: str) -> Optional[Tuple[str, str, str]]:
    """'mep_pro_tel0_cps_p5' -> ('pro','tel0','p5'); 'mep_omni_cps_p6' -> ('omni','-','p6')."""
    m = _CH_DIR_RE.match(ch)
    if m:
        return (m.group(1), m.group(2), m.group(3))
    m = _CH_OMNI_RE.match(ch)
    if m:
        return (m.group(1), '-', m.group(2))
    return None


def tuple_to_flat(tpl: Tuple[str, str, str]) -> str:
    """('pro','tel0','p5') -> 'mep_pro_tel0_cps_p5'; ('omni','-','p6') -> 'mep_omni_cps_p6'."""
    species, direction, energy = tpl
    if species == 'omni':
        return f'mep_omni_cps_{energy}'
    return f'mep_{species}_{direction}_cps_{energy}'


def tuple_to_fname(tpl: Tuple[str, str, str]) -> str:
    """('pro','tel0','p5') -> 'pro_tel0_p5'; ('omni','-','p6') -> 'omni_p6'."""
    species, direction, energy = tpl
    if direction == '-':
        return f'{species}_{energy}'
    return f'{species}_{direction}_{energy}'


def fname_to_tuple(stem: str) -> Optional[Tuple[str, str, str]]:
    """'pro_tel0_p5' -> ('pro','tel0','p5'); 'omni_p6' -> ('omni','-','p6')."""
    parts = stem.split('_')
    if len(parts) == 3:
        return (parts[0], parts[1], parts[2])
    if len(parts) == 2:
        return (parts[0], '-', parts[1])
    return None


def _ensure_pyarrow():
    try:
        import pyarrow  # noqa
    except ImportError:
        raise ImportError('pip install pyarrow')


def _empty_df() -> pd.DataFrame:
    cols = pd.MultiIndex.from_tuples([], names=['species', 'direction', 'energy'])
    return pd.DataFrame(columns=cols, dtype=float)


def _localize(idx) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    return idx.tz_localize('UTC') if idx.tz is None else idx


# ── Parquet 저장: 채널 1개 = 파일 1개, geo 개별 파일 ──
def save_parquet(df: pd.DataFrame, meta: dict,
                 directory: Union[str, Path],
                 geo: Optional[pd.DataFrame] = None) -> None:
    _ensure_pyarrow()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    print(f'[poes_noaa19_io] Parquet 저장: {directory}', flush=True)

    n_ch = 0
    for col in df.columns:
        stem = tuple_to_fname(tuple(col))
        s = df[col].dropna()
        if s.empty:
            continue
        s.to_frame(name='count').to_parquet(directory / f'{stem}.parquet',
                                            compression='snappy')
        n_ch += 1

    n_geo = 0
    if geo is not None and not geo.empty:
        for g in _GEO_CHANNELS:
            if g in geo.columns:
                gs = geo[g].dropna()
                if gs.empty:
                    continue
                gs.to_frame(name=g).to_parquet(directory / f'{g}.parquet',
                                               compression='snappy')
                n_geo += 1

    with open(directory / _META_FILENAME, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    total = sum(p.stat().st_size for p in directory.glob('*.parquet'))
    print(f'  저장 완료: count {n_ch}채널 + geo {n_geo} 파일, 총 {total/1024/1024:.1f} MB')


# ── Parquet 로드: 채널 파일 합쳐 MultiIndex DataFrame ──
def load_parquet(directory: Union[str, Path],
                 channels: Optional[List[Tuple[str, str, str]]] = None
                 ) -> Tuple[pd.DataFrame, dict]:
    _ensure_pyarrow()
    directory = Path(directory)
    print(f'[poes_noaa19_io] Parquet 로드: {directory}', flush=True)

    meta: dict = {}
    meta_path = directory / _META_FILENAME
    if meta_path.exists():
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)

    if channels is not None:
        stems = [tuple_to_fname(tuple(c)) for c in channels]
    else:
        geo_set = set(_GEO_CHANNELS)
        stems = sorted(p.stem for p in directory.glob('*.parquet')
                       if p.stem not in geo_set)

    series_dict = {}
    for stem in stems:
        fpath = directory / f'{stem}.parquet'
        tpl = fname_to_tuple(stem)
        if tpl is None or not fpath.exists():
            continue
        s = pd.read_parquet(fpath).squeeze('columns')
        s.index = _localize(s.index)
        s.index.name = 'Time'
        s.name = tpl
        series_dict[tpl] = s.astype(float)

    if not series_dict:
        return _empty_df(), meta

    df = pd.DataFrame(series_dict)
    df.columns = pd.MultiIndex.from_tuples(df.columns,
                                           names=['species', 'direction', 'energy'])
    df = df.sort_index()
    df.index.name = 'Time'
    return df, meta


def load(path: Union[str, Path],
         channels: Optional[List[Tuple[str, str, str]]] = None
         ) -> Tuple[pd.DataFrame, dict]:
    p = Path(path)
    if not p.is_dir():
        raise ValueError(f'parquet 디렉터리 경로가 아님: {p}')
    return load_parquet(p, channels=channels)


def get_counts(df: pd.DataFrame,
               species: Optional[str] = None,
               channels: Optional[List[Tuple[str, str, str]]] = None,
               t_start: Optional[str] = None,
               t_end: Optional[str] = None) -> pd.DataFrame:
    """species('pro'/'omni'/'ele') 그룹 또는 channels 개별 선택."""
    if df.empty:
        return df
    sub = df
    if channels is not None:
        sub = df.loc[:, [tuple(c) for c in channels]]
    elif species is not None:
        sub = df.loc[:, df.columns.get_level_values('species') == species]
    sub = sub.copy()
    if t_start or t_end:
        sub = sub.loc[t_start:t_end]
    return sub


def get_geo(source: Union[str, Path],
            t_start: Optional[str] = None,
            t_end: Optional[str] = None,
            with_bmag: bool = False) -> pd.DataFrame:
    """geo(lat/lon/alt) 개별 파일 → 한 DataFrame. with_bmag=True 면 Bmag/maglat 파생."""
    directory = Path(source)
    cols = {}
    for g in _GEO_CHANNELS:
        fp = directory / f'{g}.parquet'
        if fp.exists():
            s = pd.read_parquet(fp).squeeze('columns')
            s.index = _localize(s.index)
            cols[g] = s
    if not cols:
        return pd.DataFrame()
    geo = pd.DataFrame(cols).sort_index()
    geo.index.name = 'Time'
    if t_start or t_end:
        geo = geo.loc[t_start:t_end]
    if with_bmag:
        try:
            import coords_igrf
            geo = coords_igrf.add_bmag_maglat(geo)
        except Exception as e:
            print(f'[poes_noaa19_io] WARN Bmag/maglat 파생 실패: {e}')
    return geo


def summary(df: pd.DataFrame, meta: dict) -> None:
    print('=' * 55)
    print(f"  생성일시 : {str(meta.get('created', '?'))[:19]}")
    print(f"  위성     : {meta.get('satellite', '?')}")
    print(f"  범위     : {meta.get('start', '?')} ~ {meta.get('end', '?')}")
    print(f"  row 수   : {len(df):,}")
    if len(df):
        print(f"  기간     : {df.index[0]} ~ {df.index[-1]}")
    if not df.empty:
        species = df.columns.get_level_values('species').unique().tolist()
        print(f"  채널 수  : {df.shape[1]}개  (species: {', '.join(species)})")
        er = meta.get('energy_ranges', {})
        for col in df.columns:
            stem = tuple_to_fname(tuple(col))
            print(f"    {str(tuple(col)):28s} {er.get(stem, '')}")
    ga = meta.get('global_attrs', {})
    if ga:
        print(f"  [global_attrs] {len(ga)}개 보존됨")
        warn_keys = [k for k in ga if 'comment' in k.lower() or 'contam' in k.lower()
                     or 'warn' in k.lower()]
        if warn_keys:
            print(f"    (주의성 속성: {', '.join(warn_keys)})")
    mp = meta.get('meta_periods', [])
    if len(mp) > 1:
        print(f"  [메타 구간] {len(mp)}개: " +
              ', '.join(f"{p['start']}~{p['end']}" for p in mp))
    print('=' * 55)


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('사용법: python poes_noaa19_io.py <parquet_dir>')
        sys.exit(0)
    d, m = load(sys.argv[1])
    summary(d, m)
"""
kma_ksem_flux_build_cache.py
========================
KSEM flux .nc 파일들을 읽어 JSON / Parquet 캐시로 저장합니다.
kma_ksem_build_cache.py (Raw Count CSV 버전)과 논리 흐름이 동일합니다.

파일 구조:
  - 전자: gk2a_ksem_pd_e_1m_le1_YYYYMMDD.nc  채널: E1~E6(V1.0) 또는 E1~E10(V1.2)
  - 양성자: gk2a_ksem_pd_p_1m_le1_YYYYMMDD.nc  채널: P1~PN

사용법:
  # JSON
  python kma_ksem_flux_build_cache.py --root ./kma_flux_nc --start 201905 --end 202412 --out kma_ksem_flux_cache.json

  # Parquet
  python kma_ksem_flux_build_cache.py --root ./kma_flux_nc --start 201905 --end 202412 --out kma_ksem_flux_cache_parquet

  # 둘 다
  python kma_ksem_flux_build_cache.py --root ./kma_flux_nc --start 201905 --end 202412 \\
      --out kma_ksem_flux_cache.json --also-parquet kma_ksem_flux_cache_parquet

옵션:
  --root          nc 파일 루트 폴더 (kma_ksem_download.py 의 --out 경로)
  --start         시작 월 YYYYMM
  --end           종료 월 YYYYMM
  --out           출력 경로 (.json → JSON, 확장자없음 → Parquet)
  --also-parquet  JSON 외에 Parquet 도 저장할 디렉터리
  --chunk-months  N개월 단위 JSON 분할 저장 (기본 0 = 단일 파일)
  --indent        JSON 들여쓰기 (기본: compact)
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import netCDF4 as nc
import numpy as np
import pandas as pd

import kma_ksem_flux_io

# ─────────────────────────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────────────────────────
MISSING_VALUE = -99999
EPOCH = np.datetime64('2000-01-01T12:00:00', 'ns')

# 자료종류별 설정
SENSOR_TYPES = {
    'electron': {
        'subdir':    'electron_1m',
        'glob':      'gk2a_ksem_pd_e_1m_le1_*.nc',
        'flux_prefix': 'E',
    },
    'proton': {
        'subdir':    'proton_1m',
        'glob':      'gk2a_ksem_pd_p_1m_le1_*.nc',
        'flux_prefix': 'P',
    },
}


# ─────────────────────────────────────────────────────────────────
# nc 파일 1개 파싱
# ─────────────────────────────────────────────────────────────────
def parse_nc(filepath: Path) -> tuple[pd.DataFrame, dict]:
    """
    nc 파일 1개 → (df, file_meta)

    df 컬럼:
      Time (DatetimeIndex, UTC)
      E1..EN 또는 P1..PN  : flux (cm-2 sr-1 s-1 keV-1), NaN으로 결측 처리
      Att_Flag             : 감쇠기 상태 (0=open, 1=closed)
      Det_Flag1..N         : 검출기 상태 (0=정상, 1=비정상)
      E1_QEF..EN_QEF       : 품질 평가 플래그 (0=정상, 1=상한초과, 2=하한미달)
      IntegNum             : 평균에 사용된 데이터 수

    file_meta:
      version, data_description, energy_ranges, flux_channels, flag_channels
    """
    ds = nc.Dataset(str(filepath))

    # ── 글로벌 속성 ──
    version  = ds.getncattr('Data Version') if 'Data Version' in ds.ncattrs() else '?'
    desc     = ds.getncattr('Data description') if 'Data description' in ds.ncattrs() else '?'
    units    = ds.getncattr('Units') if 'Units' in ds.ncattrs() else '?'
    missing  = float(ds.getncattr('Definition of data missing')) if 'Definition of data missing' in ds.ncattrs() else MISSING_VALUE

    # 에너지 범위 파싱 (Variables 속성 텍스트에서)
    energy_ranges = {}
    var_desc = ds.getncattr('Variables') if 'Variables' in ds.ncattrs() else ''
    for line in var_desc.splitlines():
        m = re.match(r'#\s+([EP]\d+)\s*=\s*(.+)', line.strip())
        if m:
            energy_ranges[m.group(1)] = m.group(2).strip()

    # ── Time_Tag → DatetimeIndex ──
    tt = ds.variables['Time_Tag'][:, 0].astype('float64')
    times = pd.DatetimeIndex(
        EPOCH + (tt * 1e9).astype('timedelta64[ns]')
    ).tz_localize('UTC')

    # ── 모든 변수 읽기 ──
    data = {'Time': times}
    flux_channels = []
    flag_channels = []
    qef_channels  = []

    for name, var in ds.variables.items():
        if name == 'Time_Tag':
            continue
        arr = var[:, 0].astype('float64') if var.dtype.kind == 'f' else var[:, 0].astype('int32')

        if name.startswith(('E', 'P')) and 'QEF' not in name and 'Flag' not in name:
            # flux 채널: 결측값 → NaN
            arr = arr.astype('float64')
            arr[arr == missing] = np.nan
            flux_channels.append(name)
        elif 'QEF' in name:
            qef_channels.append(name)
            flag_channels.append(name)
        elif 'Flag' in name or name == 'IntegNum':
            flag_channels.append(name)

        data[name] = arr

    ds.close()

    df = pd.DataFrame(data).set_index('Time')

    file_meta = {
        'version':          version,
        'data_description': desc,
        'units':            units,
        'energy_ranges':    energy_ranges,
        'flux_channels':    flux_channels,
        'flag_channels':    flag_channels,
    }
    return df, file_meta


# ─────────────────────────────────────────────────────────────────
# 월별 폴더 탐색 (kma_ksem_build_cache.py 와 동일한 구조)
# ─────────────────────────────────────────────────────────────────
def ym_folders(root: Path, subdir: str, start_ym: str, end_ym: str) -> list[Path]:
    base = root / subdir
    if not base.exists():
        return []
    folders = sorted([f for f in base.iterdir()
                      if f.is_dir() and re.fullmatch(r'\d{6}', f.name)])
    return [f for f in folders if start_ym <= f.name <= end_ym]


def find_nc_files(folder: Path, glob: str) -> list[Path]:
    return sorted(folder.glob(glob))


# ─────────────────────────────────────────────────────────────────
# 센서 전체 로드 → DataFrame
# ─────────────────────────────────────────────────────────────────
def load_sensor(root: Path, folders: list[Path],
                sensor: str) -> tuple[pd.DataFrame, dict]:
    cfg   = SENSOR_TYPES[sensor]
    parts = []
    meta  = None
    total = len(folders)

    for i, folder in enumerate(folders, 1):
        files = find_nc_files(folder, cfg['glob'])
        if not files:
            print(f'  [{i:3d}/{total}] {folder.name}: {sensor} 파일 없음')
            continue
        print(f'  [{i:3d}/{total}] {folder.name}  ({len(files)}파일)', end='', flush=True)
        for fp in files:
            try:
                df_day, fm = parse_nc(fp)
                parts.append(df_day)
                if meta is None:
                    meta = fm
                else:
                    # 채널 수 다를 경우 최신(많은 쪽) 기준 유지
                    if len(fm['flux_channels']) > len(meta['flux_channels']):
                        meta = fm
            except Exception as e:
                print(f'\n  [WARN] {fp.name}: {e}', end='')
        print()

    if not parts:
        return pd.DataFrame(), (meta or {})

    combined = pd.concat(parts).sort_index()
    combined = combined[~combined.index.duplicated(keep='first')]
    return combined, (meta or {})


# ─────────────────────────────────────────────────────────────────
# 빌드 → kma_ksem_flux_io 로 저장
# ─────────────────────────────────────────────────────────────────
def build_cache(root: Path, start_ym: str, end_ym: str,
                out_path: Path, indent: Optional[int],
                chunk_months: int, also_parquet: Optional[Path]) -> None:

    sensor_data = {}
    sensor_meta = {}

    for sensor in SENSOR_TYPES:
        folders = ym_folders(root, SENSOR_TYPES[sensor]['subdir'], start_ym, end_ym)
        if not folders:
            print(f'[WARN] {sensor}: {start_ym}~{end_ym} 폴더 없음 ({root / SENSOR_TYPES[sensor]["subdir"]})')
            continue
        print(f'\n=== {sensor} 로딩 ({len(folders)}개월) ===')
        df, meta = load_sensor(root, folders, sensor)
        sensor_data[sensor] = df
        sensor_meta[sensor] = meta
        print(f'  → {len(df):,} rows')

    if not sensor_data:
        print('[ERROR] 로딩된 데이터 없음')
        sys.exit(1)

    global_meta = {
        'created':  datetime.now(timezone.utc).isoformat(),
        'start':    start_ym,
        'end':      end_ym,
        'sensors':  sensor_meta,
    }

    is_parquet = not out_path.suffix

    if chunk_months > 0 and not is_parquet:
        _save_chunks(sensor_data, global_meta, out_path, indent, chunk_months)
    else:
        if is_parquet:
            kma_ksem_flux_io.save_parquet(sensor_data, global_meta, out_path)
        else:
            kma_ksem_flux_io.save_json(sensor_data, global_meta, out_path, indent)

    if also_parquet:
        kma_ksem_flux_io.save_parquet(sensor_data, global_meta, also_parquet)


def _save_chunks(sensor_data, global_meta, base_path, indent, chunk_months):
    """월 단위 청크 분할 저장"""
    # 전체 월 목록 추출
    all_months = sorted(set(
        df.index.strftime('%Y%m').unique().tolist()
        for df in sensor_data.values()
        if len(df)
    ).pop() if sensor_data else [])

    chunks = [all_months[i:i+chunk_months]
              for i in range(0, len(all_months), chunk_months)]

    print(f'\n{len(chunks)}개 청크로 분할 저장')
    for chunk in chunks:
        c_start, c_end = chunk[0], chunk[-1]
        chunk_path = base_path.with_name(
            f'{base_path.stem}_{c_start}_{c_end}{base_path.suffix}')
        print(f'\n청크: {c_start}~{c_end} → {chunk_path.name}')

        chunk_data = {}
        for sensor, df in sensor_data.items():
            mask = (df.index.strftime('%Y%m') >= c_start) & \
                   (df.index.strftime('%Y%m') <= c_end)
            chunk_data[sensor] = df[mask]

        chunk_meta = {**global_meta, 'start': c_start, 'end': c_end}
        kma_ksem_flux_io.save_json(chunk_data, chunk_meta, chunk_path, indent)


# ─────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='KSEM flux nc → JSON/Parquet 캐시 변환기')
    parser.add_argument('--root',  required=True, help='nc 파일 루트 폴더')
    parser.add_argument('--start', required=True, help='시작 월 YYYYMM')
    parser.add_argument('--end',   required=True, help='종료 월 YYYYMM')
    parser.add_argument('--out',   default='kma_ksem_flux_cache.json',
                        help='출력 경로 (.json → JSON, 확장자없음 → Parquet)')
    parser.add_argument('--also-parquet', metavar='DIR', default=None)
    parser.add_argument('--chunk-months', type=int, default=0, metavar='N')
    parser.add_argument('--indent', type=int, default=None)
    args = parser.parse_args()

    root = Path(args.root)
    out  = Path(args.out)
    also = Path(args.also_parquet) if args.also_parquet else None

    print(f'루트: {root}')
    print(f'범위: {args.start} ~ {args.end}')
    print(f'출력: {out}')

    t0 = datetime.now()
    build_cache(root, args.start, args.end, out, args.indent, args.chunk_months, also)
    print(f'\n총 소요: {(datetime.now()-t0).total_seconds():.1f}초')


if __name__ == '__main__':
    main()

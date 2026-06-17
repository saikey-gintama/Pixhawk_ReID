"""
poes_metop03_build_cache.py
====================
POES/MetOp SEM-2 L1a netCDF 파일들을 읽어 JSON / Parquet 캐시로 저장합니다.
kma_ksem_flux_build_cache.py 와 논리 흐름이 동일하되, POES 고유 구조에 맞춰
다음을 추가로 처리합니다.

  1. 시간 복원: year + day(of year) + msec  ->  UTC DatetimeIndex
  2. 리샘플링 : 2초 raw  ->  1분 (KSEM 1분 데이터와 동일 분해능으로 비교 위함)
                proton/electron count 는 평균(mean), 위치(lat/lon/alt)도 평균.
  3. 채널 추출: MEPED proton (tel0/tel90 p1~p6) + omni (p6~p9)
                MEPED electron (tel0/tel90 e1~e3)  ← 대조군
                위치/지자기 정보 (lat, lon, alt)   ← SAA/위도 마스킹용 보존
  4. valid_range 밖 값 -> NaN

주의:
  - 파일 단위는 'count rate (#/s)'. calibration-free onset 감지에 그대로 사용.
  - global attr 에 "known contamination problems" 경고 있음 (특히 MEPED 저에너지
    proton 채널의 상대론적 전자 오염, SAA/고위도 변동). 이는 LEO 고유 이슈이며
    배경 처리 단계에서 다룰 분석 대상.

사용법:
  # JSON
  python poes_metop03_build_cache.py --root ./poes_nc --start 201901 --end 202512 --out poes_m03_cache.json

  # Parquet
  python poes_metop03_build_cache.py --root ./poes_nc --start 201901 --end 202512 --out poes_m03_cache_parquet

  # 둘 다 + 1분 리샘플
  python poes_metop03_build_cache.py --root ./poes_nc --start 201901 --end 202512 \\
      --out poes_m03_cache.json --also-parquet poes_m03_cache_parquet --resample 1min

옵션:
  --root          nc 파일 루트 폴더 (poes_metop03_download.py 의 --out 경로; 연도 하위폴더 구조)
  --start         시작 월 YYYYMM
  --end           종료 월 YYYYMM
  --out           출력 경로 (.json -> JSON, 확장자없음 -> Parquet)
  --also-parquet  JSON 외에 Parquet 도 저장할 디렉터리
  --resample      리샘플 주기 (기본 '1min'; 'none' 이면 원본 2초 유지)
  --sat-code      파일명 코드 (기본 m03)
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

import poes_metop03_io

# ─────────────────────────────────────────────────────────────────
# 상수: 추출할 변수 그룹
# ─────────────────────────────────────────────────────────────────
# proton count rate 채널 (방향성 telescope + omnidirectional)
PROTON_CHANNELS = [
    'mep_pro_tel0_cps_p1', 'mep_pro_tel0_cps_p2', 'mep_pro_tel0_cps_p3',
    'mep_pro_tel0_cps_p4', 'mep_pro_tel0_cps_p5', 'mep_pro_tel0_cps_p6',
    'mep_pro_tel90_cps_p1', 'mep_pro_tel90_cps_p2', 'mep_pro_tel90_cps_p3',
    'mep_pro_tel90_cps_p4', 'mep_pro_tel90_cps_p5', 'mep_pro_tel90_cps_p6',
    'mep_omni_cps_p6', 'mep_omni_cps_p7', 'mep_omni_cps_p8', 'mep_omni_cps_p9',
]
# electron count rate 채널 (대조군)
ELECTRON_CHANNELS = [
    'mep_ele_tel0_cps_e1', 'mep_ele_tel0_cps_e2', 'mep_ele_tel0_cps_e3',
    'mep_ele_tel90_cps_e1', 'mep_ele_tel90_cps_e2', 'mep_ele_tel90_cps_e3',
]
# 위치/지자기 (SAA·위도 마스킹용)
GEO_CHANNELS = ['alt', 'lat', 'lon']

SENSOR_GROUPS = {
    'proton':   PROTON_CHANNELS,
    'electron': ELECTRON_CHANNELS,
}


def _attr_to_py(v):
    """netCDF attribute(numpy scalar/array/bytes) -> JSON 직렬화 가능한 파이썬 값."""
    if isinstance(v, (bytes, bytearray)):
        return v.decode('utf-8', 'replace')
    if isinstance(v, np.ndarray):
        return [_attr_to_py(x) for x in v.tolist()]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


# ─────────────────────────────────────────────────────────────────
# 메타 병합: 위성 수명 전체의 속성을 빠짐없이 보존
# ─────────────────────────────────────────────────────────────────
# 'if meta is None: meta = fm' 방식은 첫 파일 메타만 남겨, 수명 말기에 추가되는
# contamination 경고/채널 상태 변화를 잃는다. 아래는 전 파일 메타를 union 하고
# (값 충돌은 variants 로 날짜와 함께 기록), 메타가 실제로 바뀐 구간을 분할한다.
def _attr_signature(global_attrs: dict, channel_attrs: dict) -> str:
    """메타 '내용 지문'. 같은 지문 = 같은 메타 구간."""
    import hashlib
    blob = json.dumps({'g': global_attrs, 'c': channel_attrs},
                      sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.md5(blob.encode('utf-8')).hexdigest()[:12]


def merge_file_metas(file_metas: list) -> tuple[dict, list]:
    """file_metas: [(date_str, fm), ...] 시간순.
    반환 (merged, periods):
      merged.global_attrs / channel_attrs : 전 파일 union (첫 값 유지)
      merged.*_variants : 값이 달라진 속성 {key: [(date, value), ...]}
      periods : [{start,end,sig,global_attrs,channel_attrs}] 메타 변화 구간별
    energy_ranges/title/processing_level/native_resolution 은 첫 유효본 사용."""
    g_union, c_union = {}, {}
    g_variants, c_variants = {}, {}
    periods = []
    prev_sig = None
    first = file_metas[0][1] if file_metas else {}

    for date_str, fm in file_metas:
        ga = fm.get('global_attrs', {})
        ca = fm.get('channel_attrs', {})
        sig = _attr_signature(ga, ca)
        if sig != prev_sig:
            if periods:
                periods[-1]['end'] = date_str
            periods.append({'start': date_str, 'end': date_str, 'sig': sig,
                            'global_attrs': ga, 'channel_attrs': ca})
            prev_sig = sig
        else:
            periods[-1]['end'] = date_str
        for k, v in ga.items():
            if k not in g_union:
                g_union[k] = v
            elif g_union[k] != v:
                g_variants.setdefault(k, [(periods[0]['start'], g_union[k])])
                if (date_str, v) not in g_variants[k]:
                    g_variants[k].append((date_str, v))
        for ch, attrs in ca.items():
            if ch not in c_union:
                c_union[ch] = dict(attrs)
            else:
                for ak, av in attrs.items():
                    if ak not in c_union[ch]:
                        c_union[ch][ak] = av
                    elif c_union[ch][ak] != av:
                        key = f'{ch}.{ak}'
                        c_variants.setdefault(key, [(periods[0]['start'], c_union[ch][ak])])
                        if (date_str, av) not in c_variants[key]:
                            c_variants[key].append((date_str, av))

    merged = {
        'title':             first.get('title', '?'),
        'processing_level':  first.get('processing_level', '?'),
        'native_resolution': first.get('native_resolution', '?'),
        'energy_ranges':     first.get('energy_ranges', {}),
        'global_attrs':      g_union,
        'channel_attrs':     c_union,
    }
    if g_variants:
        merged['global_attrs_variants'] = g_variants
    if c_variants:
        merged['channel_attrs_variants'] = c_variants
    return merged, periods


def _write_period_metas(periods: list, sat_code: str,
                        out_path: Path, also_parquet: Optional[Path]) -> None:
    """메타 변화 구간마다 _{sat}_meta_{start}-{end}.json 떨굼.
    저장 위치: parquet 디렉터리(있으면) 와 out_path 부모. 정보 손실 0 증거용."""
    if not periods:
        return
    targets = []
    # parquet 출력이면 그 디렉터리 안에, 아니면 out_path 부모에
    if not out_path.suffix:          # parquet dir
        targets.append(out_path)
    else:
        targets.append(out_path.parent)
    if also_parquet:
        targets.append(also_parquet)
    for tgt in targets:
        Path(tgt).mkdir(parents=True, exist_ok=True)
        for p in periods:
            fname = f'_{sat_code}_meta_{p["start"]}-{p["end"]}.json'
            with open(Path(tgt) / fname, 'w', encoding='utf-8') as f:
                json.dump({'satellite': f'poes_{sat_code}',
                           'period_start': p['start'], 'period_end': p['end'],
                           'signature': p['sig'],
                           'global_attrs': p['global_attrs'],
                           'channel_attrs': p['channel_attrs']},
                          f, indent=2, ensure_ascii=False)
        print(f'  기간별 메타 {len(periods)}개 → {tgt}')


# ─────────────────────────────────────────────────────────────────
# nc 파일 1개 파싱
# ─────────────────────────────────────────────────────────────────
def parse_nc(filepath: Path) -> tuple[dict, dict]:
    """
    nc 1개 -> ({sensor: df}, file_meta)
    각 df 의 index 는 복원된 UTC DatetimeIndex. 위치(GEO) 컬럼은 양쪽 df 에 공통 포함.
    """
    ds = nc.Dataset(str(filepath))

    # ── 시간 복원: year + day(of year) + msec ──
    year = np.array(ds.variables['year'][:], dtype='int64')
    doy  = np.array(ds.variables['day'][:],  dtype='int64')
    msec = np.array(ds.variables['msec'][:], dtype='int64')

    # 연도 시작 (1월 1일 00:00 UTC) + (doy-1)일 + msec
    base = pd.to_datetime(year.astype(str) + '-01-01', utc=True)
    times = base + pd.to_timedelta(doy - 1, unit='D') + pd.to_timedelta(msec, unit='ms')
    times = pd.DatetimeIndex(times)

    # ── 위치 변수 ──
    geo = {}
    for g in GEO_CHANNELS:
        if g in ds.variables:
            geo[g] = np.array(ds.variables[g][:], dtype='float64')

    # ── 채널별 count 추출 + valid_range 마스킹 ──
    energy_ranges = {}
    channel_attrs = {}   # 채널별 nc 변수 속성 전체 보존 (units/valid_range/long_name 등)
    sensor_data = {}
    for sensor, channels in SENSOR_GROUPS.items():
        cols = {}
        for ch in channels:
            if ch not in ds.variables:
                continue
            var = ds.variables[ch]
            arr = np.array(var[:], dtype='float64')

            # valid_range 밖 -> NaN
            if 'valid_range' in var.ncattrs():
                lo, hi = var.getncattr('valid_range')
                arr[(arr < lo) | (arr > hi)] = np.nan
            # _FillValue / missing_value
            for fv_attr in ('_FillValue', 'missing_value'):
                if fv_attr in var.ncattrs():
                    fv = float(var.getncattr(fv_attr))
                    arr[arr == fv] = np.nan

            cols[ch] = arr
            if 'long_name' in var.ncattrs():
                energy_ranges[ch] = var.getncattr('long_name')
            # 변수 속성 전체를 JSON 직렬화 가능한 형태로 보존
            channel_attrs[ch] = {a: _attr_to_py(var.getncattr(a)) for a in var.ncattrs()}

        df = pd.DataFrame(cols, index=times)
        # 위치 컬럼 부착
        for g, v in geo.items():
            df[g] = v
        df.index.name = 'Time'
        sensor_data[sensor] = df

    # ── global attributes 전체 보존 (L0 정신: 데이터는 count+geo만, 메타는 빠짐없이) ──
    global_attrs = {a: _attr_to_py(ds.getncattr(a)) for a in ds.ncattrs()}
    title = global_attrs.get('title', '?')
    proc  = global_attrs.get('processing_level', '?')
    res   = global_attrs.get('time_coverage_resolution', '?')
    ds.close()

    file_meta = {
        'title':             str(title)[:120],
        'processing_level':  str(proc),
        'native_resolution': str(res),
        'energy_ranges':     energy_ranges,
        'channel_attrs':     channel_attrs,   # 채널별 변수 속성 전체
        'global_attrs':      global_attrs,     # nc 전역 속성 전체 (contamination 경고 등)
    }
    return sensor_data, file_meta


# ─────────────────────────────────────────────────────────────────
# 1분 리샘플
# ─────────────────────────────────────────────────────────────────
def resample_df(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """count rate / 위치 모두 평균으로 리샘플. 빈 구간은 NaN 유지."""
    if df.empty or rule.lower() == 'none':
        return df
    # 평균 집계 (count rate 는 #/s 이므로 분당 평균이 물리적으로 타당)
    return df.resample(rule).mean()


# ─────────────────────────────────────────────────────────────────
# 월별 폴더 탐색 (POES 는 연도 하위폴더; 파일명에서 YYYYMM 추출)
# ─────────────────────────────────────────────────────────────────
def find_files(root: Path, sat_code: str, start_ym: str, end_ym: str) -> list[Path]:
    """root/{YYYY}/poes_{sat}_{YYYYMMDD}_raw.nc 들을 YYYYMM 범위로 필터."""
    pat = re.compile(rf'poes_{sat_code}_(\d{{8}})_raw\.nc$')
    out = []
    for year_dir in sorted(root.iterdir()):
        if not (year_dir.is_dir() and re.fullmatch(r'\d{4}', year_dir.name)):
            continue
        for fp in sorted(year_dir.glob(f'poes_{sat_code}_*_raw.nc')):
            m = pat.search(fp.name)
            if not m:
                continue
            ym = m.group(1)[:6]
            if start_ym <= ym <= end_ym:
                out.append(fp)
    return out


# ─────────────────────────────────────────────────────────────────
# 전체 로드
# ─────────────────────────────────────────────────────────────────
def build_cache(root: Path, sat_code: str, start_ym: str, end_ym: str,
                resample: str, out_path: Path, indent: Optional[int],
                also_parquet: Optional[Path]) -> None:

    files = find_files(root, sat_code, start_ym, end_ym)
    if not files:
        print(f'[ERROR] {root} 에서 poes_{sat_code}_*_raw.nc 파일을 찾지 못함 '
              f'({start_ym}~{end_ym})')
        sys.exit(1)

    print(f'대상 파일: {len(files)}개')

    parts = {'proton': [], 'electron': []}
    file_metas = []        # [(date_str, fm), ...] 전 파일 메타 수집 → 병합용
    _date_pat = re.compile(rf'poes_{sat_code}_(\d{{8}})_raw\.nc$')
    total = len(files)

    for i, fp in enumerate(files, 1):
        try:
            sd, fm = parse_nc(fp)
            for sensor in parts:
                df = resample_df(sd[sensor], resample)
                parts[sensor].append(df)
            dm = _date_pat.search(fp.name)
            date_str = dm.group(1) if dm else f'{i:08d}'
            file_metas.append((date_str, fm))
        except Exception as e:
            print(f'\n  [WARN] {fp.name}: {e}', flush=True)
        if i % 50 == 0 or i == total:
            print(f'  [{i:4d}/{total}] {fp.name}', flush=True)

    # 전 파일 메타 병합 (union + 변화 구간 분할)
    merged_meta, meta_periods = merge_file_metas(file_metas)
    meta = merged_meta   # 이하 기존 코드가 meta 를 그대로 참조

    # ── proton/electron 평면 count 컬럼 → MultiIndex(species,direction,energy) df ──
    #    geo(alt/lat/lon)는 분리해 별도 저장. 채널 1개 = 파일 1개 구조로 io 가 저장.
    series_dict = {}
    geo_df = None
    energy_ranges_stem = {}   # 파일명 stem 키의 energy_ranges (io.summary 용)
    for sensor in parts:
        if not parts[sensor]:
            continue
        combined = pd.concat(parts[sensor]).sort_index()
        combined = combined[~combined.index.duplicated(keep='first')]
        # geo 추출(첫 등장 1회)
        if geo_df is None:
            geo_cols = [c for c in combined.columns if c in GEO_CHANNELS]
            if geo_cols:
                geo_df = combined[geo_cols].copy()
        # count 채널 → 튜플 컬럼
        for c in combined.columns:
            if c in GEO_CHANNELS:
                continue
            tpl = poes_metop03_io.flat_to_tuple(c)
            if tpl is None:
                print(f'  [WARN] 채널명 파싱 실패, 건너뜀: {c}')
                continue
            series_dict[tpl] = combined[c]
            stem = poes_metop03_io.tuple_to_fname(tpl)
            er = (meta.get('energy_ranges', {}) if meta else {}).get(c)
            if er is not None:
                energy_ranges_stem[stem] = er
        print(f'  {sensor}: {len(combined):,} rows, '
              f'{len([c for c in combined.columns if c not in GEO_CHANNELS])} channels')

    if series_dict:
        count_df = pd.DataFrame(series_dict).sort_index()
        count_df.columns = pd.MultiIndex.from_tuples(
            count_df.columns, names=['species', 'direction', 'energy'])
    else:
        count_df = pd.DataFrame()

    global_meta = {
        'created':  datetime.now(timezone.utc).isoformat(),
        'start':    start_ym,
        'end':      end_ym,
        'satellite': f'poes_{sat_code}',
        'global_attrs': meta.get('global_attrs', {}) if meta else {},  # 전 파일 union
        'global_attrs_variants': meta.get('global_attrs_variants', {}),   # 값 변화 이력
        'channel_attrs_variants': meta.get('channel_attrs_variants', {}),
        'channel_attrs': meta.get('channel_attrs', {}) if meta else {},
        'energy_ranges': energy_ranges_stem,   # 파일명 stem 키
        'channels': [list(t) for t in count_df.columns] if not count_df.empty else [],
        'meta_periods': [{'start': p['start'], 'end': p['end'], 'sig': p['sig']}
                         for p in meta_periods],   # 메타 변화 구간 요약
    }

    # 기간별 메타 파일도 떨군다 (속성이 바뀐 구간마다 1개) — 사람이 보는 증거용
    _write_period_metas(meta_periods, sat_code, out_path, also_parquet)

    # 채널 1개 = 파일 1개 + geo 개별 파일로 저장 (parquet 전용)
    if out_path.suffix:
        print('[WARN] 이 빌더는 parquet 디렉터리 출력만 지원합니다(.json 미지원). '
              '확장자 없는 경로를 주세요.')
    poes_metop03_io.save_parquet(count_df, global_meta, out_path, geo=geo_df)
    if also_parquet:
        poes_metop03_io.save_parquet(count_df, global_meta, also_parquet, geo=geo_df)


# ─────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='POES/MetOp SEM-2 nc -> JSON/Parquet 캐시 변환기')
    parser.add_argument('--root',  required=True, help='nc 파일 루트 폴더 (연도 하위폴더 구조)')
    parser.add_argument('--start', required=True, help='시작 월 YYYYMM')
    parser.add_argument('--end',   required=True, help='종료 월 YYYYMM')
    parser.add_argument('--out',   default='poes_cache.json',
                        help='출력 경로 (.json -> JSON, 확장자없음 -> Parquet)')
    parser.add_argument('--also-parquet', metavar='DIR', default=None)
    parser.add_argument('--resample', default='1min',
                        help="리샘플 주기 (기본 '1min', 'none' 이면 2초 원본 유지)")
    parser.add_argument('--sat-code', default='m03', help='파일명 위성 코드 (기본 m03)')
    parser.add_argument('--indent', type=int, default=None)
    args = parser.parse_args()

    root = Path(args.root)
    out  = Path(args.out)
    also = Path(args.also_parquet) if args.also_parquet else None

    print(f'루트    : {root}')
    print(f'위성코드: {args.sat_code}')
    print(f'범위    : {args.start} ~ {args.end}')
    print(f'리샘플  : {args.resample}')
    print(f'출력    : {out}')
    print()

    t0 = datetime.now()
    build_cache(root, args.sat_code, args.start, args.end,
                args.resample, out, args.indent, also)
    print(f'\n총 소요: {(datetime.now()-t0).total_seconds():.1f}초')


if __name__ == '__main__':
    main()
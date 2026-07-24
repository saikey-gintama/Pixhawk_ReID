"""
_coords_igrf.py
===============
POES geo(lat/lon/alt) → IGRF |B| (nT) 와 dipole 자기위도(maglat, deg) 산출.

poes_metop03_io 의 geo 채널은 alt/lat/lon 뿐이라 Bmag/maglat 이 없다.
ana 단계에서 SAA(|B|<25000 nT) 판정과 maglat-bin 배경을 쓰려면 이 둘을
좌표로부터 만들어야 한다. 이 모듈은 그 단일 책임만 갖는다.

  - Bmag  : ppigrf(IGRF-14) 로 시각·위치별 전자기장 세기. SAA 판정용.
  - maglat: 중심 쌍극자(centered dipole) 근사 자기위도. IGRF 불필요(좌표만).
            배경 bin 라벨용 — 정밀 invariant latitude 가 아니라 bin 경계용
            단조 좌표면 충분하다는 사용자 설계에 맞춘 경량 근사.

설계 메모
---------
- Bmag 는 시간에 의존(영년변화)하므로 date 가 필요하다. 15분 캐시 전 구간을
  매 점 계산하면 느리므로, 월 1회 epoch 로 묶어 계산 후 그 달 전체에 적용한다
  (영년변화는 월 단위로 무시 가능). 위치는 점별로 그대로 쓴다.
- 입력 geo 에 NaN 이 있으면 그 점의 Bmag/maglat 도 NaN.
"""
from __future__ import annotations
import datetime as _dt
import numpy as np
import pandas as pd

# 2020 IGRF geomagnetic north pole (geographic) — dipole maglat 근사용 고정 상수.
# 영년변화로 매년 ~수십 km 이동하나 bin 경계용으로는 무시 가능.
_POLE_LAT = np.deg2rad(80.65)
_POLE_LON = np.deg2rad(-72.68)

SAA_BMAG_NT = 25000.0   # |B| < 이 값 → SAA 상태 비트


def dipole_maglat(lat_deg, lon_deg):
    """중심 쌍극자 근사 자기위도[deg]. IGRF 불필요, 좌표만.
    geomagnetic latitude = arcsin( sinφ sinφp + cosφ cosφp cos(λ-λp) ).
    """
    lat = np.deg2rad(np.asarray(lat_deg, dtype=float))
    lon = np.deg2rad(np.asarray(lon_deg, dtype=float))
    sin_mlat = (np.sin(lat) * np.sin(_POLE_LAT)
                + np.cos(lat) * np.cos(_POLE_LAT) * np.cos(lon - _POLE_LON))
    sin_mlat = np.clip(sin_mlat, -1.0, 1.0)
    return np.rad2deg(np.arcsin(sin_mlat))


def _bmag_for_month(lat, lon, alt_km, date):
    """한 시점(date)의 IGRF 계수로 (lat,lon,alt) 배열의 |B|[nT] 계산."""
    import ppigrf
    # ppigrf.igrf(lon, lat, h_km, date) → (Be, Bn, Bu), 각 shape=(1, N)
    Be, Bn, Bu = ppigrf.igrf(np.asarray(lon, dtype=float),
                             np.asarray(lat, dtype=float),
                             np.asarray(alt_km, dtype=float), date)
    B = np.sqrt(np.asarray(Be)**2 + np.asarray(Bn)**2 + np.asarray(Bu)**2)
    return np.asarray(B).reshape(-1)


def add_bmag_maglat(geo_rs: pd.DataFrame,
                    default_alt_km: float = 820.0) -> pd.DataFrame:
    """geo_rs[lat,lon,(alt)] → 같은 인덱스에 Bmag(nT), maglat(deg) 컬럼 추가본 반환.

    Bmag 는 월 1회 epoch 로 묶어 계산(영년변화 월 단위 무시). alt 결측은
    default_alt_km(POES ~820km) 로 대체. lat/lon NaN 점은 Bmag/maglat=NaN.
    """
    if geo_rs is None or geo_rs.empty \
       or "lat" not in geo_rs.columns or "lon" not in geo_rs.columns:
        return geo_rs
    out = geo_rs.copy()
    lat = out["lat"].to_numpy(dtype=float)
    lon = out["lon"].to_numpy(dtype=float)
    if "alt" in out.columns:
        alt = out["alt"].to_numpy(dtype=float)
        alt = np.where(np.isfinite(alt), alt, default_alt_km)
    else:
        alt = np.full(len(out), default_alt_km)

    # maglat: 좌표만 (벡터 일괄)
    out["maglat"] = dipole_maglat(lat, lon)

    # Bmag: 월별 epoch 로 묶어 계산
    bmag = np.full(len(out), np.nan)
    finite = np.isfinite(lat) & np.isfinite(lon)
    idx = out.index
    # 월 키 (YYYY-MM) 별 그룹
    ym = pd.PeriodIndex(idx, freq="M")
    for period in pd.unique(ym):
        sel = (ym == period) & finite
        if not sel.any():
            continue
        # 그 달 15일을 대표 epoch 로
        date = _dt.datetime(period.year, period.month, 15)
        try:
            b = _bmag_for_month(lat[sel], lon[sel], alt[sel], date)
            bmag[np.where(sel)[0]] = b
        except Exception as e:  # IGRF 실패 시 그 달만 NaN
            print(f"[_coords_igrf] WARN {period}: {e}")
    out["Bmag"] = bmag
    return out


def saa_mask_from_bmag(geo_with_b: pd.DataFrame,
                       thresh_nt: float = SAA_BMAG_NT) -> pd.Series:
    """|B| < thresh_nt → True(=SAA). Bmag 없으면 전부 False."""
    if "Bmag" not in geo_with_b.columns:
        return pd.Series(False, index=geo_with_b.index)
    return (geo_with_b["Bmag"] < thresh_nt).fillna(False)

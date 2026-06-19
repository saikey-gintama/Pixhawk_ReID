"""
noaa_goes_spe_build_cache.py
=======================
NOAA SPE HTML 카탈로그 → JSON / Parquet 캐시 변환기.

noaa_goes_spe_io.py 와 함께 사용합니다.
이 스크립트는 일회성으로 실행해 캐시를 만드는 용도이며,
분석·플롯 코드에서는 noaa_goes_spe_io.py 만 import 하면 됩니다.

사용법:
  # 웹에서 직접 내려받아 JSON 저장
  python noaa_goes_spe_build_cache.py --out spe_cache.json

  # 로컬에 저장한 HTML 파일에서 파싱 (NOAA 서버 403 우회)
  python noaa_goes_spe_build_cache.py --from-file sep_page.html --out spe_cache.json

  # Parquet 디렉터리로 저장 (확장자 없으면 Parquet)
  python noaa_goes_spe_build_cache.py --from-file sep_page.html --out spe_cache_parquet

  # JSON + Parquet 동시 저장
  python noaa_goes_spe_build_cache.py --from-file sep_page.html \\
      --out spe_cache.json --also-parquet spe_cache_parquet

  # JSON 들여쓰기
  python noaa_goes_spe_build_cache.py --from-file sep_page.html \\
      --out spe_cache.json --indent 2
"""

import argparse
import re
import sys
import urllib.request
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

import noaa_goes_spe_io
from noaa_goes_spe_io import SOURCE_URL, COLUMNS


# ─────────────────────────────────────────────────────────────────
# 파싱 내부 헬퍼
# ─────────────────────────────────────────────────────────────────
_NA_VALS = {"n/a", "na", "", "-", "–", "—", "none"}


def _is_na(s: str) -> bool:
    return s.strip().lower() in _NA_VALS


def _parse_bool(s: str) -> Optional[bool]:
    """'Yes'/'No'/'N/A' → True / False / None"""
    v = s.strip().lower()
    if v == "yes":
        return True
    if v == "no":
        return False
    return None


def _parse_datetime(year_str: str, md_time_str: str) -> Optional[datetime]:
    """
    NOAA 테이블 날짜 표현 → datetime (UTC).
      year_str    : "1976"
      md_time_str : "04/30 2120"  /  "07/07/1010" (슬래시 3개 이상) 등
    """
    parts = re.split(r"[/\s]+", md_time_str.strip())
    try:
        if len(parts) == 3:
            month, day, hhmm = parts
        elif len(parts) == 2:
            md, hhmm = parts
            month, day = (md[:2], md[2:]) if "/" not in md else md.split("/")
        else:
            return None
        return datetime(int(year_str), int(month), int(day),
                        int(hhmm[:2]), int(hhmm[2:]),
                        tzinfo=timezone.utc)
    except Exception:
        return None


def _parse_flare(year_str: str, field: str) -> Tuple[str, str, Optional[datetime]]:
    """
    flare 컬럼 파싱.
      "X2/2B 04/30 2114"  → ("X2", "2B", datetime(...))
      "M3/N/A 03/30 0049" → ("M3", "N/A", datetime(...))
      "N/A"  또는  "N/A 06/14" → ("", "", None)
    반환: (flare_class, flare_importance, flare_datetime)
    """
    s = re.sub(r"<[^>]+>", "", field).strip()   # 잔류 HTML 태그 제거
    if _is_na(s) or re.match(r"^N/A", s, re.IGNORECASE):
        return "", "", None

    m = re.match(
        r"([A-Z]\d*(?:\.\d+)?)"        # 플레어 등급  예: X2, M7, C4, X12
        r"(?:/([^\s]+))?"               # /중요도      예: /2B, /N/A
        r"(?:\s+(\S+)\s+(\d{4}))?",     # 날짜 시각    예: 04/30 2114
        s,
    )
    if not m:
        return "", "", None

    fc = m.group(1) or ""
    fi = m.group(2) or ""
    dt = None
    if m.group(3) and m.group(4):
        dt = _parse_datetime(year_str, f"{m.group(3)} {m.group(4)}")
    return fc, fi, dt


def _parse_speed(field: str) -> Optional[float]:
    """"785 km/s" → 785.0   /  "N/A" → None"""
    s = re.sub(r"<[^>]+>", "", field).strip()
    m = re.search(r"([\d,]+)\s*km/s", s, re.IGNORECASE)
    return float(m.group(1).replace(",", "")) if m else None


def _parse_pfu(s: str) -> Optional[float]:
    """"24,000" → 24000.0"""
    try:
        return float(s.strip().replace(",", ""))
    except ValueError:
        return None


# ─────────────────────────────────────────────────────────────────
# HTML → DataFrame
# ─────────────────────────────────────────────────────────────────
def parse_html(html: str) -> Tuple[pd.DataFrame, dict]:
    """
    NOAA SPE 페이지 HTML 문자열을 파싱해 (df, meta) 반환.
    pandas.read_html 로 테이블을 추출하고, 날짜·플레어·속도는 직접 처리합니다.
    """
    try:
        tables = pd.read_html(StringIO(html), header=0)
    except Exception as e:
        raise ValueError(f"HTML 테이블 파싱 실패: {e}") from e

    # 컬럼 수 ≥ 9 인 첫 번째 테이블이 SPE 목록
    raw = next((t for t in tables if t.shape[1] >= 9), None)
    if raw is None:
        raise ValueError("SPE 테이블을 찾을 수 없습니다 (컬럼 수 ≥ 9 없음).")

    # MultiIndex 컬럼 평탄화
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [" ".join(str(v) for v in col if v).strip()
                       for col in raw.columns]
    else:
        raw.columns = [str(c).strip() for c in raw.columns]

    records = []
    for _, row in raw.iterrows():
        vals = [str(v).strip() for v in row.values]
        if len(vals) < 9:
            continue

        # 컬럼 위치: 0=Begin, 1=Max, 2=pfu, 3=Region, 4=Location,
        #            5=Flare, 6=TypeII, 7=TypeIV, 8=Speed, (9=Imagery)
        begin_raw, max_raw = vals[0], vals[1]
        pfu_raw    = vals[2]
        region_raw = vals[3]
        loc_raw    = vals[4]
        flare_raw  = vals[5]
        type2_raw  = vals[6]
        type4_raw  = vals[7]
        speed_raw  = vals[8]

        # 연도 + 월일시각 분리  ("1976 04/30 2120")
        ym = re.match(r"(\d{4})\s+(.+)", begin_raw)
        if not ym:
            continue
        year, begin_body = ym.group(1), ym.group(2)

        mm = re.match(r"(\d{4})\s+(.+)", max_raw)
        max_year  = mm.group(1) if mm else year
        max_body  = mm.group(2) if mm else max_raw

        begin_dt = _parse_datetime(year, begin_body)
        if begin_dt is None:
            continue   # 파싱 불가 행 스킵

        fc, fi, flare_dt = _parse_flare(year, flare_raw)

        records.append({
            "begin_time":       begin_dt,
            "max_time":         _parse_datetime(max_year, max_body),
            "max_pfu":          _parse_pfu(pfu_raw),
            "region":           "" if _is_na(region_raw) else region_raw,
            "location":         "" if _is_na(loc_raw)    else loc_raw,
            "flare_class":      fc,
            "flare_importance": fi,
            "flare_time":       flare_dt,
            "type2":            _parse_bool(type2_raw),
            "type4":            _parse_bool(type4_raw),
            "cme_speed_kms":    _parse_speed(speed_raw),
        })

    df = (pd.DataFrame(records, columns=["begin_time"] + COLUMNS)
            .set_index("begin_time")
            .sort_index())
    df.index.name = "begin_time"

    now_iso = datetime.now(timezone.utc).isoformat()
    meta = {
        "created":  now_iso,
        "source":   SOURCE_URL,
        "fetched":  now_iso,
        "n_events": len(df),
    }
    print(f"[spe_build] 파싱 완료: {len(df)}개 이벤트")
    return df, meta


# ─────────────────────────────────────────────────────────────────
# 다운로드
# ─────────────────────────────────────────────────────────────────
def fetch_html(url: str = SOURCE_URL) -> str:
    """
    NOAA SPE 페이지 HTML을 내려받아 문자열로 반환합니다.
    서버가 403을 반환하는 경우 브라우저로 저장 후 --from-file 옵션을 사용하세요.
    """
    print(f"[spe_build] 다운로드: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        html = resp.read().decode("utf-8", errors="replace")
    print(f"[spe_build] 다운로드 완료 ({len(html) // 1024} KB)")
    return html


# ─────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="NOAA SPE HTML 카탈로그 → 캐시 변환기"
    )
    parser.add_argument("--url", default=SOURCE_URL,
                        help="파싱 대상 URL (기본: NOAA SPE 페이지)")
    parser.add_argument("--from-file", metavar="FILE", default=None,
                        help="로컬 HTML 파일에서 파싱 (웹 요청 없이)")
    parser.add_argument("--out", default="spe_cache.json",
                        help="출력 경로 (.json → JSON,  확장자없음 → Parquet 디렉터리)")
    parser.add_argument("--also-parquet", metavar="DIR", default=None,
                        help="JSON 저장과 동시에 Parquet 도 저장할 디렉터리")
    parser.add_argument("--indent", type=int, default=None,
                        help="JSON 들여쓰기 (기본: compact)")
    args = parser.parse_args()

    # ── HTML 확보 ──
    if args.from_file:
        print(f"[spe_build] 로컬 파일 파싱: {args.from_file}")
        html = Path(args.from_file).read_text(encoding="utf-8", errors="replace")
    else:
        html = fetch_html(args.url)

    # ── 파싱 ──
    df, meta = parse_html(html)
    noaa_goes_spe_io.summary(df, meta)

    # ── 저장 ──
    out = Path(args.out)
    is_parquet = not out.suffix

    t0 = datetime.now()

    if is_parquet:
        noaa_goes_spe_io.save_parquet(df, meta, out)
    else:
        noaa_goes_spe_io.save_json(df, meta, out, indent=args.indent)

    if args.also_parquet:
        noaa_goes_spe_io.save_parquet(df, meta, Path(args.also_parquet))

    elapsed = (datetime.now() - t0).total_seconds()
    print(f"\n완료! 총 소요: {elapsed:.1f}초")


if __name__ == "__main__":
    main()

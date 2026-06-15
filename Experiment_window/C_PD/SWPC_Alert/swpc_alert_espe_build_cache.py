"""
swpc_alert_espe_build_cache.py
==============================
SWPC 전자 경보(ALTEF3) HTML → JSON / Parquet 캐시 변환기.

noaa_goes_spe_build_cache.py 와 동일한 구조. HTML 파싱부만 SPE 테이블(read_html)
대신 ALTEF3 경보 블록 파싱으로 교체했다. 출력 스키마는 noaa_goes 와 호환되도록
begin_time(index) + max_time/max_pfu 를 유지하여 매칭기가 동일 동작한다.

ALTEF3 메시지 두 형태:
  INITIAL  : "Threshold Reached: 2024 May 15 1525 UTC"  (begin, pfu 없음=임계 1000)
  CONTINUED: "Begin Time: 2019 May 30 1535 UTC"
             "Continuation of Serial Number: NNNN"        (원 이벤트로 연결)
             "Yesterday Maximum 2MeV Flux: 3771 pfu"      (그날 최대 flux)

이벤트 묶기:
  serial 체인(Continuation of Serial Number)으로 같은 이벤트를 추적하고,
  동일 begin_time 으로 그룹화한다. 그룹의 Yesterday Maximum 최댓값을 max_pfu,
  그 보고 시점을 max_time 으로 채택. 초기 ALERT만 있고 CONTINUED가 없으면
  max_pfu=1000(임계값), max_time=begin.

사용법:
  python swpc_alert_espe_build_cache.py --from-dir ./ftp_html \\
      --out espe_cache.json --also-parquet espe_cache_parquet
"""

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

import swpc_alert_espe_io
from swpc_alert_espe_io import SOURCE_URL, COLUMNS

THRESHOLD_PFU = 1000.0   # INITIAL ALERT만 있을 때 max_pfu 기본값(>2MeV 경보 임계)

_MONTHS = {m: i for i, m in enumerate(
    ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], 1)}


# ─────────────────────────────────────────────────────────────────
# 파싱 헬퍼
# ─────────────────────────────────────────────────────────────────
def _parse_swpc_time(s: str) -> Optional[datetime]:
    """'2024 May 15 1525' 또는 '... UTC' → datetime(UTC)."""
    m = re.search(r"(\d{4})\s+([A-Za-z]{3})\s+(\d{1,2})\s+(\d{2})(\d{2})", s.strip())
    if not m:
        return None
    y, mon, d, hh, mm = m.groups()
    mon_n = _MONTHS.get(mon.title())
    if not mon_n:
        return None
    try:
        return datetime(int(y), mon_n, int(d), int(hh), int(mm), tzinfo=timezone.utc)
    except ValueError:
        return None


def _strip(block: str) -> str:
    t = re.sub(r"<[^>]+>", " ", block)
    return re.sub(r"\s+", " ", t).strip()


# ─────────────────────────────────────────────────────────────────
# HTML → 레코드
# ─────────────────────────────────────────────────────────────────
def parse_altef3(html: str) -> list:
    """HTML 문자열에서 ALTEF3 경보 블록을 파싱해 dict 리스트 반환."""
    out = []
    for raw in re.split(r"Space Weather Message Code:", html):
        b = _strip(raw)
        if not b.startswith("ALTEF3"):
            continue

        serial = (re.search(r"Serial Number:\s*(\d+)", b) or [None, None])[1] \
            if re.search(r"Serial Number:\s*(\d+)", b) else None
        m = re.search(r"Serial Number:\s*(\d+)", b)
        serial = m.group(1) if m else None
        m = re.search(r"Continuation of Serial Number:\s*(\d+)", b)
        cont_of = m.group(1) if m else None

        # begin: INITIAL은 Threshold Reached, CONTINUED는 Begin Time
        begin_dt = None
        m = re.search(r"Threshold Reached:\s*(\d{4} [A-Za-z]{3} \d{1,2} \d{4})", b)
        if not m:
            m = re.search(r"Begin Time:\s*(\d{4} [A-Za-z]{3} \d{1,2} \d{4})", b)
        if m:
            begin_dt = _parse_swpc_time(m.group(1))
        if begin_dt is None:
            continue

        # pfu: CONTINUED의 Yesterday Maximum 2MeV Flux
        max_pfu = None
        m = re.search(r"Yesterday Maximum 2MeV Flux:\s*([\d,]+)\s*pfu", b)
        if m:
            try:
                max_pfu = float(m.group(1).replace(",", ""))
            except ValueError:
                max_pfu = None

        # 보고 시각(Issue Time) — max_pfu가 보고된 시점 추정용
        issue_dt = None
        m = re.search(r"Issue Time:\s*(\d{4} [A-Za-z]{3} \d{1,2} \d{4})", b)
        if m:
            issue_dt = _parse_swpc_time(m.group(1))

        station = ""
        m = re.search(r"Station:\s*([A-Za-z0-9\-]+)", b)
        if m:
            station = m.group(1)

        out.append({
            "serial": serial, "cont_of": cont_of, "begin_time": begin_dt,
            "max_pfu": max_pfu, "issue_time": issue_dt, "station": station,
        })
    return out


def build_dataframe(records: list) -> pd.DataFrame:
    """
    serial 체인 + 동일 begin_time 으로 이벤트 그룹화.
    그룹별 최대 Yesterday Maximum → max_pfu, 그 보고시점 → max_time.
    CONTINUED 없이 INITIAL만 있으면 max_pfu=THRESHOLD_PFU, max_time=begin.
    """
    if not records:
        empty = pd.DataFrame(columns=["begin_time"] + COLUMNS).set_index("begin_time")
        empty.index = pd.to_datetime(empty.index, utc=True)
        return empty

    # begin_time 기준 그룹화 (같은 이벤트의 INITIAL/CONTINUED가 동일 begin 공유)
    groups: dict = {}
    for r in records:
        key = r["begin_time"]
        g = groups.setdefault(key, {
            "begin_time": key, "max_time": pd.NaT, "max_pfu": None,
            "station": r["station"], "serial": r["serial"],
        })
        if r["station"] and not g["station"]:
            g["station"] = r["station"]
        # 최대 pfu 및 그 보고시점 갱신
        if r["max_pfu"] is not None and (g["max_pfu"] is None or r["max_pfu"] > g["max_pfu"]):
            g["max_pfu"] = r["max_pfu"]
            g["max_time"] = r["issue_time"] if r["issue_time"] else key

    rows = []
    for g in groups.values():
        if g["max_pfu"] is None:          # INITIAL만 있던 이벤트
            g["max_pfu"] = THRESHOLD_PFU
            g["max_time"] = g["begin_time"]
        rows.append(g)

    df = (pd.DataFrame(rows, columns=["begin_time"] + COLUMNS)
            .set_index("begin_time").sort_index())
    df.index.name = "begin_time"
    df.index = pd.to_datetime(df.index, utc=True)
    df["max_pfu"] = pd.to_numeric(df["max_pfu"], errors="coerce")
    return df


# ─────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="SWPC 전자경보(ALTEF3) HTML → 캐시 변환기")
    parser.add_argument("--from-dir", metavar="DIR", default=None,
                        help="alerts_*.html 들이 든 폴더")
    parser.add_argument("--from-file", metavar="FILE", default=None,
                        help="단일 HTML 파일")
    parser.add_argument("--out", default="espe_cache.json",
                        help="출력 경로 (.json → JSON, 확장자없음 → Parquet 디렉터리)")
    parser.add_argument("--also-parquet", metavar="DIR", default=None,
                        help="JSON 저장과 동시에 Parquet 도 저장할 디렉터리")
    parser.add_argument("--indent", type=int, default=None)
    args = parser.parse_args()

    # ── HTML 확보 ──
    if args.from_dir:
        files = sorted(Path(args.from_dir).glob("*.html"))
        print(f"[espe_build] 폴더 파싱: {len(files)}개 HTML")
    elif args.from_file:
        files = [Path(args.from_file)]
    else:
        parser.error("--from-dir 또는 --from-file 필요")

    # ── 파싱 ──
    all_records = []
    for fp in files:
        html = fp.read_text(encoding="utf-8", errors="replace")
        recs = parse_altef3(html)
        all_records.extend(recs)
        if recs:
            print(f"  {fp.name}: ALTEF3 {len(recs)}블록")

    df = build_dataframe(all_records)
    now_iso = datetime.now(timezone.utc).isoformat()
    meta = {"created": now_iso, "source": SOURCE_URL,
            "fetched": now_iso, "n_events": len(df)}
    print(f"[espe_build] 파싱 완료: {len(df)}개 이벤트 (원시 경보 {len(all_records)}건)")
    swpc_alert_espe_io.summary(df, meta)

    # ── 저장 ──
    out = Path(args.out)
    if not out.suffix:
        swpc_alert_espe_io.save_parquet(df, meta, out)
    else:
        swpc_alert_espe_io.save_json(df, meta, out, indent=args.indent)
    if args.also_parquet:
        swpc_alert_espe_io.save_parquet(df, meta, Path(args.also_parquet))
    print("\n완료!")


if __name__ == "__main__":
    main()

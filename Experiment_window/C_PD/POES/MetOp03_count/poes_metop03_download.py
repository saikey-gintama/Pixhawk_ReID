"""
poes_metop03_download.py
=========================
NOAA NCEI POES/MetOp Space Environment Monitor (SEM-2) L1a netCDF 파일을
날짜 범위로 자동 다운로드합니다.

kma_ksem_flux_download.py 와 동일한 구조/스타일입니다.

데이터 소스:
  https://www.ncei.noaa.gov/data/poes-metop-space-environment-monitor/access/l1a/v01r00/{YYYY}/{SAT}/poes_{SAT_CODE}_{YYYYMMDD}_raw.nc

사용법:
  # 기본 (metop03, 2019~2025)
  python poes_metop03_download.py

  # 기간/위성 지정
  python poes_metop03_download.py --sat metop03 --start 20190101 --end 20251231

  # 저장 경로 지정
  python poes_metop03_download.py --out D:/VS_code/Pixhawk_ReID/Experiment_window/C_METOP03/raw_nc

옵션:
  --sat       위성 코드 (기본: metop03). 폴더명/파일명에 모두 사용.
              metop03 -> 파일명 prefix 'm03', noaa19 -> 'n19' 등 SAT_CODE_MAP 참고
  --start     시작일 YYYYMMDD (기본: 20190101)
  --end       종료일 YYYYMMDD (기본: 20251231)
  --out       저장 루트 폴더 (기본: ./poes_nc)
  --no-skip   이미 받은 파일도 다시 다운로드
  --workers   병렬 다운로드 수 (기본: 4)
  --delay     요청 간 대기 시간(초) (기본: 0.5)
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ─────────────────────────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────────────────────────
BASE_URL = "https://www.ncei.noaa.gov/data/poes-metop-space-environment-monitor/access/l1a/v01r00"

# 위성 폴더명(NCEI) -> 파일명에 들어가는 코드
SAT_CODE_MAP = {
    "metop01": "m01",
    "metop02": "m02",
    "metop03": "m03",
    "noaa15":  "n15",
    "noaa18":  "n18",
    "noaa19":  "n19",
}

USER_AGENT = "Mozilla/5.0 (research data download script; contact: jeongin@khu.ac.kr)"


# ─────────────────────────────────────────────────────────────────
# HTTP 세션 (자동 재시도)
# ─────────────────────────────────────────────────────────────────
def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({"User-Agent": USER_AGENT})
    return session


# ─────────────────────────────────────────────────────────────────
# 날짜 리스트 생성
# ─────────────────────────────────────────────────────────────────
def date_range(start: str, end: str):
    """YYYYMMDD 문자열 -> datetime.date 리스트"""
    s = datetime.strptime(start, "%Y%m%d").date()
    e = datetime.strptime(end, "%Y%m%d").date()
    d = s
    while d <= e:
        yield d
        d += timedelta(days=1)


# ─────────────────────────────────────────────────────────────────
# 단일 파일 다운로드
# ─────────────────────────────────────────────────────────────────
def download_one(
    session: requests.Session,
    sat: str,
    sat_code: str,
    date,
    out_root: Path,
    skip_existing: bool,
    delay: float,
) -> tuple[bool, str]:
    """
    Returns (success, message)
    """
    date_str = date.strftime("%Y%m%d")
    year = date.strftime("%Y")

    out_dir = out_root / year
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"poes_{sat_code}_{date_str}_raw.nc"
    out_file = out_dir / fname

    if skip_existing and out_file.exists() and out_file.stat().st_size > 1000:
        return True, f"SKIP  {fname}"

    url = f"{BASE_URL}/{year}/{sat}/{fname}"

    try:
        time.sleep(delay)
        resp = session.get(url, timeout=60, stream=True)

        if resp.status_code == 404:
            return False, f"MISS  {date_str}  (해당 날짜 파일 없음, 404)"
        if resp.status_code != 200:
            return False, f"FAIL  {date_str}  HTTP {resp.status_code}"

        ct = resp.headers.get("Content-Type", "")
        if "text/html" in ct:
            return False, f"FAIL  {date_str}  응답이 HTML (파일 아님)"

        tmp_file = out_file.with_suffix(".nc.part")
        with open(tmp_file, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
        tmp_file.rename(out_file)

        size_kb = out_file.stat().st_size / 1024
        return True, f"OK    {fname}  ({size_kb:.0f} KB)"

    except Exception as e:
        return False, f"ERROR {date_str}  {e}"


# ─────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="POES/MetOp SEM-2 L1a nc 자동 다운로드")
    parser.add_argument("--sat", default="metop03",
                        choices=list(SAT_CODE_MAP),
                        help="위성 (기본: metop03)")
    parser.add_argument("--start", default="20190101", help="시작일 YYYYMMDD")
    parser.add_argument("--end", default="20251231", help="종료일 YYYYMMDD")
    parser.add_argument("--out", default="./poes_nc", help="저장 루트 폴더")
    parser.add_argument("--no-skip", action="store_true",
                        help="이미 받은 파일도 다시 다운로드")
    parser.add_argument("--workers", type=int, default=4, help="병렬 다운로드 수")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="요청 간 대기 시간(초)")
    args = parser.parse_args()

    out_root = Path(args.out)
    skip_existing = not args.no_skip
    sat_code = SAT_CODE_MAP[args.sat]

    dates = list(date_range(args.start, args.end))

    print("다운로드 계획")
    print(f"  위성    : {args.sat}  (파일명 코드: {sat_code})")
    print(f"  기간    : {args.start} ~ {args.end}  ({len(dates)}일)")
    print(f"  저장위치: {out_root.resolve()}")
    print(f"  병렬수  : {args.workers}")
    print()

    session = make_session()
    ok_count = fail_count = skip_count = miss_count = 0
    failures = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_one, session, args.sat, sat_code,
                date, out_root, skip_existing, args.delay
            ): date
            for date in dates
        }

        for i, future in enumerate(as_completed(futures), 1):
            date = futures[future]
            success, msg = future.result()

            if "SKIP" in msg:
                skip_count += 1
            elif "MISS" in msg:
                miss_count += 1
                failures.append(msg)
            elif success:
                ok_count += 1
            else:
                fail_count += 1
                failures.append(msg)

            if i % 100 == 0 or i == len(dates):
                print(f"[{i:5d}/{len(dates)}] OK={ok_count} SKIP={skip_count} "
                      f"MISS={miss_count} FAIL={fail_count}  {msg}")
            elif not success:
                print(f"  {msg}")

    print()
    print("=== 완료 ===")
    print(f"  성공  : {ok_count}개")
    print(f"  건너뜀: {skip_count}개")
    print(f"  결측  : {miss_count}개  (서버에 해당 날짜 파일 없음)")
    print(f"  실패  : {fail_count}개")

    if failures:
        fail_log = out_root / f"download_log_{args.sat}.txt"
        fail_log.write_text("\n".join(failures), encoding="utf-8")
        print(f"\n결측/실패 목록 저장: {fail_log}")


if __name__ == "__main__":
    main()

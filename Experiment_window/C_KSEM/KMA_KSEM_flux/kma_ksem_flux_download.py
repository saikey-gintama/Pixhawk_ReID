"""
ksem_download.py
================
기상청 API Hub에서 KSEM 입자 flux .nc 파일을 날짜 범위로 자동 다운로드합니다.

사용법:
  # 양성자 + 전자 1분 데이터, 2019~2024년 전체
  python ksem_download.py --key YOUR_AUTH_KEY --start 20190101 --end 20241231

  # 특정 자료종류만
  python ksem_download.py --key YOUR_AUTH_KEY --start 20190101 --end 20241231 --types PD-P-1M PD-E-1M

  # 저장 경로 지정
  python ksem_download.py --key YOUR_AUTH_KEY --start 20190101 --end 20241231 --out D:/KSEM_nc

옵션:
  --key       API 인증키 (필수)
  --start     시작일 YYYYMMDD (기본: 20190501)
  --end       종료일 YYYYMMDD (기본: 20241231)
  --types     자료종류 (기본: PD-P-1M PD-E-1M)
              선택 가능: PD-P-1M PD-P-5M PD-E-1M PD-E-5M MG-1M MG-5M CM-1M CM-5M
  --out       저장 루트 폴더 (기본: ./ksem_nc)
  --skip      이미 받은 파일 건너뜀 (기본: True)
  --workers   병렬 다운로드 수 (기본: 4)
  --delay     요청 간 대기 시간(초) (기본: 0.5)
"""

import argparse
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ─────────────────────────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────────────────────────
BASE_URL  = "https://apihub.kma.go.kr/api/typ05/api/GK2A/LV1"
# 자료종류 → (API 경로, 저장 서브폴더명, 파일명 패턴)
DATA_TYPES = {
    "PD-P-1M": ("PD-P-1M/NA", "proton_1m",   "gk2a_ksem_pd_p_1m_le1_{date}.nc"),
    "PD-P-5M": ("PD-P-5M/NA", "proton_5m",   "gk2a_ksem_pd_p_5m_le1_{date}.nc"),
    "PD-E-1M": ("PD-E-1M/NA", "electron_1m", "gk2a_ksem_pd_e_1m_le1_{date}.nc"),
    "PD-E-5M": ("PD-E-5M/NA", "electron_5m", "gk2a_ksem_pd_e_5m_le1_{date}.nc"),
    "MG-1M":   ("MG-1M/NA",   "mag_1m",      "gk2a_ksem_mg_auto_1m_le1_{date}.nc"),
    "MG-5M":   ("MG-5M/NA",   "mag_5m",      "gk2a_ksem_mg_auto_5m_le1_{date}.nc"),
    "CM-1M":   ("CM-1M/NA",   "cm_1m",       "gk2a_ksem_cm_1m_le1_{date}.nc"),
    "CM-5M":   ("CM-5M/NA",   "cm_5m",       "gk2a_ksem_cm_5m_le1_{date}.nc"),
}


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
    return session


# ─────────────────────────────────────────────────────────────────
# 날짜 리스트 생성
# ─────────────────────────────────────────────────────────────────
def date_range(start: str, end: str):
    """YYYYMMDD 문자열 → datetime.date 리스트"""
    s = datetime.strptime(start, "%Y%m%d").date()
    e = datetime.strptime(end,   "%Y%m%d").date()
    d = s
    while d <= e:
        yield d
        d += timedelta(days=1)


# ─────────────────────────────────────────────────────────────────
# 단일 파일 다운로드
# ─────────────────────────────────────────────────────────────────
def download_one(
    session:   requests.Session,
    auth_key:  str,
    dtype:     str,
    date:      "datetime.date",
    out_root:  Path,
    skip_existing: bool,
    delay:     float,
) -> tuple[bool, str]:
    """
    Returns (success, message)
    """
    api_path, subdir, fname_tpl = DATA_TYPES[dtype]
    date_str  = date.strftime("%Y%m%d")
    # API는 HHMM 필요 → 하루치 파일은 0000 고정
    date_hhmm = date_str + "0000"

    out_dir  = out_root / subdir / date.strftime("%Y%m")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / fname_tpl.format(date=date_str)

    if skip_existing and out_file.exists() and out_file.stat().st_size > 1000:
        return True, f"SKIP  {out_file.name}"

    url = f"{BASE_URL}/{api_path}/data?date={date_hhmm}&authKey={auth_key}"

    try:
        time.sleep(delay)
        resp = session.get(url, timeout=60, stream=True)

        # 응답이 nc 파일인지 확인 (오류 시 JSON/HTML 반환)
        ct = resp.headers.get("Content-Type", "")
        if resp.status_code != 200:
            return False, f"FAIL  {date_str} {dtype}  HTTP {resp.status_code}"
        if "application" not in ct and "octet" not in ct:
            # 오류 메시지일 가능성 — 내용 확인
            text = resp.text[:200]
            return False, f"FAIL  {date_str} {dtype}  응답이 파일 아님: {text}"

        with open(out_file, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)

        size_kb = out_file.stat().st_size / 1024
        return True, f"OK    {out_file.name}  ({size_kb:.0f} KB)"

    except Exception as e:
        return False, f"ERROR {date_str} {dtype}  {e}"


# ─────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="KSEM flux .nc 자동 다운로드")
    parser.add_argument("--key",     required=True, help="API 인증키")
    parser.add_argument("--start",   default="20190501", help="시작일 YYYYMMDD")
    parser.add_argument("--end",     default="20241231", help="종료일 YYYYMMDD")
    parser.add_argument("--types",   nargs="+",
                        default=["PD-P-1M", "PD-E-1M"],
                        choices=list(DATA_TYPES),
                        help="자료종류 (기본: PD-P-1M PD-E-1M)")
    parser.add_argument("--out",     default="./ksem_nc", help="저장 루트 폴더")
    parser.add_argument("--no-skip", action="store_true",
                        help="이미 받은 파일도 다시 다운로드")
    parser.add_argument("--workers", type=int, default=4, help="병렬 다운로드 수")
    parser.add_argument("--delay",   type=float, default=0.5,
                        help="요청 간 대기 시간(초)")
    args = parser.parse_args()

    out_root      = Path(args.out)
    skip_existing = not args.no_skip

    dates = list(date_range(args.start, args.end))
    tasks = [(dtype, d) for d in dates for dtype in args.types]

    print(f"다운로드 계획")
    print(f"  기간    : {args.start} ~ {args.end}  ({len(dates)}일)")
    print(f"  자료종류: {args.types}")
    print(f"  총 파일 : {len(tasks)}개")
    print(f"  저장위치: {out_root.resolve()}")
    print(f"  병렬수  : {args.workers}")
    print()

    session = make_session()
    ok_count = fail_count = skip_count = 0
    failures = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_one, session, args.key,
                dtype, date, out_root, skip_existing, args.delay
            ): (dtype, date)
            for dtype, date in tasks
        }

        for i, future in enumerate(as_completed(futures), 1):
            dtype, date = futures[future]
            success, msg = future.result()

            if "SKIP" in msg:
                skip_count += 1
            elif success:
                ok_count += 1
            else:
                fail_count += 1
                failures.append(msg)

            # 진행 상황 출력 (100개마다 + 마지막)
            if i % 100 == 0 or i == len(tasks):
                print(f"[{i:5d}/{len(tasks)}] OK={ok_count} SKIP={skip_count} FAIL={fail_count}  {msg}")
            elif not success or "SKIP" not in msg:
                print(f"  {msg}")

    print()
    print(f"=== 완료 ===")
    print(f"  성공: {ok_count}개")
    print(f"  건너뜀: {skip_count}개")
    print(f"  실패: {fail_count}개")

    if failures:
        print("\n실패 목록:")
        for f in failures:
            print(f"  {f}")
        # 실패 목록 파일로도 저장
        fail_log = out_root / "download_failures.txt"
        fail_log.write_text("\n".join(failures), encoding="utf-8")
        print(f"\n실패 목록 저장: {fail_log}")


if __name__ == "__main__":
    main()

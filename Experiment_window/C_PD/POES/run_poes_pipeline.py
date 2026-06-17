"""
run_poes_pipeline.py
====================
POES 한 위성의 데이터 파이프라인을 순서대로 자동 실행 (다운로드 -> 캐시 빌드).
PowerShell 에서 .sh 제약 없이 바로 실행하기 위한 Python 런너.

폴더 구조 (C_PD/POES 에서 실행):
  C_PD/POES/
    run_poes_pipeline.py            <- 이 파일
    MetOp03_count/
      poes_metop03_download.py
      poes_metop03_build_cache.py
      poes_metop03_io.py
    NOAA19_count/
      poes_noaa19_download.py
      poes_noaa19_build_cache.py
      poes_noaa19_io.py

각 위성 모듈은 같은 폴더의 io 를 import 하므로, 실행 시 그 위성 폴더를
작업 디렉터리(cwd)로 삼아 subprocess 를 돌린다. 따라서 nc/캐시 출력도
그 위성 폴더 안에 생성된다.

사용법 (PowerShell):
  python run_poes_pipeline.py metop03
  python run_poes_pipeline.py noaa19
  python run_poes_pipeline.py noaa19 --start 20190101 --end 20250616
  python run_poes_pipeline.py metop03 --skip-download      # 이미 받았으면 빌드만

기본 종료일: metop03=20251231, noaa19=20250616 (NOAA19는 2025-06-16까지만 관측).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# ── 위성별 설정 (폴더명 / 파일 prefix / sat-code / 기본 종료일) ──
SAT_CONFIG = {
    "metop03": {
        "folder":   "MetOp03_count",
        "prefix":   "poes_metop03",
        "sat_code": "m03",
        "end_default": "20251231",
    },
    "noaa19": {
        "folder":   "NOAA19_count",
        "prefix":   "poes_noaa19",
        "sat_code": "n19",
        "end_default": "20250616",
    },
}

ROOT = Path(__file__).parent.resolve()   # C_PD/POES (이 파일이 있는 폴더)


def run_step(title: str, cmd: list[str], cwd: Path) -> None:
    """한 단계 실행. 실패하면 즉시 중단(이후 단계 진행 안 함)."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print(f" cwd: {cwd}")
    print(f" cmd: {' '.join(cmd)}")
    print("=" * 60, flush=True)
    result = subprocess.run(cmd, cwd=str(cwd))
    if result.returncode != 0:
        print(f"\n[중단] '{title}' 단계가 실패했습니다 "
              f"(returncode={result.returncode}). 이후 단계를 진행하지 않습니다.")
        sys.exit(result.returncode)


def main():
    p = argparse.ArgumentParser(
        description="POES 데이터 파이프라인 (다운로드 -> 캐시 빌드)")
    p.add_argument("sat", choices=list(SAT_CONFIG),
                   help="위성 (metop03 | noaa19)")
    p.add_argument("--start", default="20190101", help="시작일 YYYYMMDD")
    p.add_argument("--end", default=None,
                   help="종료일 YYYYMMDD (미지정 시 위성별 기본값)")
    p.add_argument("--resample", default="1min",
                   help="캐시 리샘플 주기 (기본 1min, 'none'=2초 원본)")
    p.add_argument("--skip-download", action="store_true",
                   help="다운로드 건너뛰고 캐시 빌드만")
    p.add_argument("--nc-subdir", default="raw_nc",
                   help="위성 폴더 안 nc 저장 하위폴더명 (기본 raw_nc)")
    args = p.parse_args()

    cfg = SAT_CONFIG[args.sat]
    sat_dir = ROOT / cfg["folder"]
    if not sat_dir.is_dir():
        print(f"[ERROR] 위성 폴더가 없습니다: {sat_dir}")
        sys.exit(1)

    end = args.end or cfg["end_default"]
    start_ym = args.start[:6]
    end_ym = end[:6]

    nc_root = args.nc_subdir                       # sat_dir 기준 상대경로
    cache_dir = f"{cfg['prefix']}_cache_parquet"   # sat_dir 안에 생성

    download_py = f"{cfg['prefix']}_download.py"
    build_py = f"{cfg['prefix']}_build_cache.py"

    # 모듈 존재 확인
    for fn in (download_py, build_py, f"{cfg['prefix']}_io.py"):
        if not (sat_dir / fn).exists():
            print(f"[ERROR] {sat_dir} 에 {fn} 이 없습니다.")
            sys.exit(1)

    print("#" * 60)
    print(f"# POES 파이프라인: {args.sat} ({cfg['sat_code']})")
    print(f"#   기간   : {args.start} ~ {end}")
    print(f"#   위성폴더: {sat_dir}")
    print(f"#   nc 루트 : {sat_dir / nc_root}")
    print(f"#   캐시    : {sat_dir / cache_dir}")
    print(f"#   다운로드: {'건너뜀' if args.skip_download else '실행'}")
    print("#" * 60)

    # ── 1. 다운로드 ──
    if not args.skip_download:
        run_step(
            "[1/2] 다운로드",
            [sys.executable, download_py,
             "--sat", args.sat,
             "--start", args.start, "--end", end,
             "--out", nc_root],
            cwd=sat_dir,
        )
    else:
        print("\n[1/2] 다운로드 건너뜀 (--skip-download)")

    # ── 2. 캐시 빌드 (parquet) ──
    run_step(
        "[2/2] 캐시 빌드 (parquet)",
        [sys.executable, build_py,
         "--root", nc_root,
         "--start", start_ym, "--end", end_ym,
         "--out", cache_dir,
         "--sat-code", cfg["sat_code"],
         "--resample", args.resample],
        cwd=sat_dir,
    )

    print("\n" + "#" * 60)
    print(f"# 완료: {args.sat}")
    print(f"#   캐시: {sat_dir / cache_dir}")
    print(f"#   메타: {sat_dir / cache_dir / '_poes_meta.json'}")
    print(f"#         (+ 기간별 _{cfg['sat_code']}_meta_*.json)")
    print("#" * 60)


if __name__ == "__main__":
    main()
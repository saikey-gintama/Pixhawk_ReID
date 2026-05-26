"""
ksem_build_cache.py  (v2 — ksem_io 기반)
=========================================
KSEM Raw Count CSV → JSON / Parquet 캐시 변환기.

ksem_io.py 와 함께 사용합니다.
이 스크립트는 일회성으로 실행해 캐시를 만드는 용도이며,
분석·플롯 코드에서는 ksem_io.py 만 import 하면 됩니다.

사용법:
  # JSON 단일 파일
  python ksem_build_cache.py \\
      --root  "D:\\Raw Count" \\
      --start 201905 --end 202412 \\
      --out   ksem_cache.json

  # Parquet 디렉터리 (크기 작고 로딩 빠름)
  python ksem_build_cache.py \\
      --root  "D:\\Raw Count" \\
      --start 201905 --end 202412 \\
      --out   ksem_cache_parquet    # 확장자 없으면 Parquet으로 저장

  # 둘 다 저장
  python ksem_build_cache.py \\
      --root  "D:\\Raw Count" \\
      --start 201905 --end 202412 \\
      --out   ksem_cache.json \\
      --also-parquet ksem_cache_parquet

  # 연도별 JSON 청크 분할
  python ksem_build_cache.py \\
      --root  "D:\\Raw Count" \\
      --start 201905 --end 202412 \\
      --out   ksem_cache.json \\
      --chunk-months 12
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

import ksem_io
from ksem_io import PD_KEYS, SIDES, LOGICS

# ─────────────────────────────────────────────────────────────────
# bin 정의
# ─────────────────────────────────────────────────────────────────
O_BINS    = list(range(1,   24))
OU_BINS   = list(range(25,  42))
CR_BINS   = list(range(42,  45))
OUT_BINS  = list(range(46,  57))
F_BINS    = list(range(58,  81))
FT_BINS   = list(range(82, 105))
FTU_BINS  = list(range(106, 117))
FTUO_BINS = list(range(118, 127))
EXCLUDE   = {0, 24, 45, 57, 81, 105, 117, 127}
TRASH_BINS = sorted(EXCLUDE)

PROTON_LOGICS   = [("O", O_BINS), ("OU", OU_BINS),
                   ("CR", CR_BINS), ("OUT", OUT_BINS)]
ELECTRON_LOGICS = [("F", F_BINS), ("FT", FT_BINS),
                   ("FTU", FTU_BINS), ("FTUO", FTUO_BINS)]
ALL_LOGICS      = PROTON_LOGICS + ELECTRON_LOGICS + [("TRASH", TRASH_BINS)]

_COL_RE = re.compile(r"^([AB])(\d+)\((\d+)\)$")


def parse_columns(columns):
    result = {}
    for c in columns:
        m = _COL_RE.match(str(c).strip())
        if m:
            result[c] = (int(m.group(2)), int(m.group(3)))
    return result


def energy_range(col_meta, prefix, bin_list):
    kevs = [kev for c, (idx, kev) in col_meta.items()
            if c.startswith(prefix) and idx in bin_list and kev > 0]
    return (min(kevs), max(kevs)) if kevs else (None, None)


def load_one_csv(filepath: Path):
    raw = pd.read_csv(str(filepath))
    raw["Time"] = pd.to_datetime(raw["Time"])
    raw = raw.set_index("Time").sort_index()
    col_meta = parse_columns(raw.columns)

    def extract(prefix):
        sub = {c: idx for c, (idx, kev) in col_meta.items()
               if c.startswith(prefix)}
        if not sub:
            return pd.DataFrame(index=raw.index)
        df = raw[list(sub)].copy()
        df.columns = [sub[c] for c in df.columns]
        return df[sorted(df.columns)]

    df_A, df_B = extract("A"), extract("B")
    energy_meta = {name: list(energy_range(col_meta, "A", bins))
                   for name, bins in ALL_LOGICS}
    return df_A, df_B, energy_meta


def logic_sum(df, bin_list):
    cols = [b for b in bin_list if b in df.columns and b not in EXCLUDE]
    return df[cols].sum(axis=1) if cols else pd.Series(0, index=df.index, dtype=float)


def trash_sum(df):
    cols = [b for b in TRASH_BINS if b in df.columns]
    return df[cols].sum(axis=1) if cols else pd.Series(0, index=df.index, dtype=float)


def ym_folders(root: Path, start_ym: str, end_ym: str) -> List[Path]:
    folders = sorted([f for f in root.iterdir()
                      if f.is_dir() and f.name.isdigit() and len(f.name) == 6])
    return [f for f in folders if start_ym <= f.name <= end_ym]


def find_pd_files(folder: Path, pd_key: str) -> List[Path]:
    return sorted([f for f in folder.glob("*.csv")
                   if pd_key.lower() in f.name.lower()])


def load_pd_series(root: Path, folders: List[Path],
                   pd_key: str) -> Tuple[dict, dict]:
    logic_names = [n for n, _ in ALL_LOGICS]
    parts = {(s, l): [] for s in SIDES for l in logic_names}
    energy_meta = None
    total = len(folders)

    for i, folder in enumerate(folders, 1):
        files = find_pd_files(folder, pd_key)
        if not files:
            print(f"  [WARN] {folder.name}: {pd_key} 없음")
            continue
        print(f"  [{i:3d}/{total}] {folder.name}  ({len(files)}파일)", end="", flush=True)
        for fp in files:
            try:
                df_A, df_B, meta = load_one_csv(fp)
                if energy_meta is None:
                    energy_meta = meta
                for side, df in [("A", df_A), ("B", df_B)]:
                    for name, bins in PROTON_LOGICS + ELECTRON_LOGICS:
                        parts[(side, name)].append(logic_sum(df, bins))
                    parts[(side, "TRASH")].append(trash_sum(df))
            except Exception as e:
                print(f"\n  [WARN] {fp.name}: {e}", end="")
        print()

    result = {}
    for key, slist in parts.items():
        if not slist:
            result[key] = pd.Series(dtype=float)
            continue
        combined = pd.concat(slist).sort_index()
        result[key] = combined[~combined.index.duplicated(keep="first")]

    return result, (energy_meta or {})


# ─────────────────────────────────────────────────────────────────
# 폴더 목록 → ksem_io DataFrame
# ─────────────────────────────────────────────────────────────────
def build_dataframe(root: Path, folders: List[Path],
                    start_ym: str, end_ym: str):
    """CSV 폴더 목록을 읽어 ksem_io 형식의 (df, meta) 반환."""
    series_dict = {}
    energy_meta_all = None

    for pd_key in PD_KEYS:
        print(f"\n=== {pd_key} 로딩 ===")
        series, em = load_pd_series(root, folders, pd_key)
        if energy_meta_all is None and em:
            energy_meta_all = em
        for side in SIDES:
            for logic in LOGICS:
                series_dict[(pd_key, side, logic)] = series.get(
                    (side, logic), pd.Series(dtype=float))

    df = pd.DataFrame(series_dict)
    df.columns = pd.MultiIndex.from_tuples(
        df.columns, names=["pd_key", "side", "logic"])
    df = df.sort_index()

    em_tuples = {}
    for k, v in (energy_meta_all or {}).items():
        em_tuples[k] = (v[0], v[1]) if v else (None, None)

    meta = {
        "created":     datetime.now(timezone.utc).isoformat(),
        "start":       start_ym,
        "end":         end_ym,
        "energy_meta": em_tuples,
    }
    return df, meta


# ─────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="KSEM Raw Count CSV → 캐시 변환기")
    parser.add_argument("--root",  required=True, help='Raw Count 루트 폴더')
    parser.add_argument("--start", required=True, help="시작 월 YYYYMM")
    parser.add_argument("--end",   required=True, help="종료 월 YYYYMM")
    parser.add_argument("--out",   default="ksem_cache.json",
                        help="출력 경로 (.json → JSON, 확장자없음 → Parquet)")
    parser.add_argument("--also-parquet", metavar="DIR", default=None,
                        help="JSON 저장과 동시에 Parquet 도 저장할 디렉터리")
    parser.add_argument("--indent", type=int, default=None,
                        help="JSON 들여쓰기 (기본: compact)")
    parser.add_argument("--chunk-months", type=int, default=0, metavar="N",
                        help="N개월 단위로 JSON 분할 저장 (Parquet엔 적용 안 됨)")
    args = parser.parse_args()

    root = Path(args.root)
    out  = Path(args.out)
    is_parquet = not out.suffix  # 확장자 없으면 Parquet 디렉터리로 간주

    all_folders = ym_folders(root, args.start, args.end)
    if not all_folders:
        print(f"[ERROR] {root} 에 {args.start}~{args.end} 폴더 없음")
        sys.exit(1)
    print(f"대상 폴더 {len(all_folders)}개: "
          f"{all_folders[0].name} ~ {all_folders[-1].name}")

    t0 = datetime.now()

    if args.chunk_months > 0 and not is_parquet:
        # ── 청크 분할 JSON ──
        chunks = [all_folders[i:i+args.chunk_months]
                  for i in range(0, len(all_folders), args.chunk_months)]
        print(f"\n{len(chunks)}개 청크로 분할 저장")
        for chunk in chunks:
            c_start, c_end = chunk[0].name, chunk[-1].name
            chunk_path = out.with_name(
                f"{out.stem}_{c_start}_{c_end}{out.suffix}")
            print(f"\n{'='*50}\n청크: {c_start}~{c_end} → {chunk_path.name}")
            df, meta = build_dataframe(root, chunk, c_start, c_end)
            ksem_io.save_json(df, meta, chunk_path, indent=args.indent)
    else:
        # ── 단일 저장 ──
        df, meta = build_dataframe(root, all_folders, args.start, args.end)

        if is_parquet:
            ksem_io.save_parquet(df, meta, out)
        else:
            ksem_io.save_json(df, meta, out, indent=args.indent)

        if args.also_parquet:
            ksem_io.save_parquet(df, meta, Path(args.also_parquet))

    elapsed = (datetime.now() - t0).total_seconds()
    print(f"\n완료! 총 소요: {elapsed:.1f}초")


if __name__ == "__main__":
    main()

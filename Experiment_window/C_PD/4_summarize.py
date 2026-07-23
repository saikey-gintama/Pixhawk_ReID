"""
4_summarize.py
==============
detector(gk2a/metop03/noaa19) x catalog(noaa/swpc) x baseline 전체 조합의 2_fsm
sweep runtag를 순회 매칭해서 POD/FAR을 하나의 master_table.csv로 모으는 러너.
_match_core_poes.sweep_table()/match_events()를 그대로 재사용한다 (onset CSV는
peak_floor/in_saa 컬럼이 없어도 그대로 동작 -- GK2A는 애초에 이 두 컬럼이 없음).

평가 대상: fsm_onset_<runtag>.csv (peak_floor 미사용 = 정답 기준).

runtag 폴더명 (2_fsm_run.py 산출, baseline 컬럼 값은 폴더명 접두사 그대로):
  blc1_fixed_const_on{onset}_pk{peak}
  blc1_lowe_w{win}_m{mult}_on{onset}_pk{peak}     (mult는 k_or_mult 컬럼에 기록)
  quietoff_mad_w{win}_k{k}_on{onset}_pk{peak}
  cusum_w{win}_k{k}_h{h}_on{onset}_pk{peak}       (h는 신규 컬럼, 다른 baseline은 NaN)
  quiet7_mad_w{win}_k{k}_on{onset}_pk{peak}       (GK2A 전용)
  quiet7_std_w{win}_k{k}_on{onset}_pk{peak}       (GK2A 전용)
  quietoff_std_w{win}_k{k}_on{onset}_pk{peak}     (GK2A 전용 -- POES는 quietoff_mad만 있음)

  quietoff/cusum은 2-2_fsm_binned_run.py(POES 조건부배경/condbg 판인
  fsm_count_spe_{cusum,quietoff}_condbg_poes.py 등록 러너)의 mlb 토큰도
  선택적으로 붙는다: quietoff_mad_mlb15-30-45-60-75_w{win}_k{k}_on..,
  cusum_mlb15-30-45-60-75_w{win}_k{k}_h{h}_on.. -- mlb 토큰이 있으면 그 값
  그대로(예: "15-30-45-60-75") maglat_bins 컬럼에 기록, 없으면 NaN(=시간순
  배경). 나머지 4계열(blc1_*, quiet7_*, quietoff_std)은 condbg 판이 아직
  없어 파싱 무변경.

detector별 catalog 매칭 기간(era)이 다름 -- GK2A(KSEM_ERA 2019~2024)는 POES(ERA
2019~2025)보다 1년 짧다. 같은 catalog라도 detector별로 자기 era로 필터링된 카탈로그와
매칭한다(공유 카탈로그를 그대로 쓰면 GK2A가 존재할 수 없는 2025년 이벤트까지
"놓친 이벤트"로 잡혀 POD가 부당하게 낮아짐).

사용:
  python 4_summarize.py --detectors metop03,noaa19 --catalogs noaa,swpc
  python 4_summarize.py --detectors metop03 --baselines quietoff --out custom/master_table.csv
  python 4_summarize.py --detectors gk2a --baselines quietoff,quiet7_mad,quiet7_std,quietoff_std

NOAA19은 2_fsm sweep이 아직 진행 중일 수 있음 — 존재하는 runtag 폴더만 처리하고
없는 조합은 그냥 skip한다 (에러 아님). sweep 완료 후 재실행하면 됨.

--out 미지정 시 기본 경로는 4_summary/<catalogs 태그>/master_table.csv (예:
--catalogs noaa -> 4_summary/noaa/, --catalogs noaa,swpc -> 4_summary/noaa_swpc/)
-- 다른 --catalogs 조합끼리 같은 파일을 덮어쓰지 않도록 자동 분리된다. 기본
경로에 이미 결과가 있으면 중단(경고) -- 덮어쓰려면 --force. --out을 명시하면
그 경로를 그대로 쓰고 이 보호검사는 생략(사용자 책임).
"""
from __future__ import annotations
import argparse
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent  # C_PD/
sys.path.insert(0, str(HERE / "POES" / "event_MATCHER"))
import _match_core_poes as core  # match_events, sweep_table, MATCH_TOL_H, ERA, _import_event_io
sys.path.insert(0, str(HERE / "GK2A" / "event_MATCHER"))
import _match_core as gk2a_core  # KSEM_ERA (GK2A는 POES보다 매칭 기간이 1년 짧음)

# ── match-root: detector별 2_fsm 표준 출력 폴더 (2_fsm_run.py의 _OUT_DIR과 동일) ──
_MATCH_ROOT = {
    "gk2a":    HERE / "GK2A" / "KSEM_count"    / "gk2a_output"    / "2_fsm",
    "metop03": HERE / "POES" / "MetOp03_count" / "metop03_output" / "2_fsm",
    "noaa19":  HERE / "POES" / "NOAA19_count"  / "noaa19_output"  / "2_fsm",
}

_CATALOG = {
    "noaa": (HERE / "NOAA_GOES"  / "noaa_goes_spe_cache_parquet", "noaa_goes_spe_io"),
    "swpc": (HERE / "SWPC_Alert" / "swpc_espe_cache_parquet",     "swpc_alert_espe_io"),
}

# detector별 catalog 매칭 기간 -- GK2A는 KSEM_ERA(1년 짧음), POES 2종은 기존 ERA 공유.
_ERA_BY_DETECTOR = {
    "gk2a":    gk2a_core.KSEM_ERA,
    "metop03": core.ERA,
    "noaa19":  core.ERA,
}

# detector count 캐시 (io 모듈명, 캐시 parquet 경로) — 채널 유니버스 확정용 (POES 2종만;
# GK2A는 컬럼이 tuple이어도 tuple_to_fname이 없어 별도 분기 -- _channel_universe 참고)
_POES_IO = {
    "metop03": ("poes_metop03_io", HERE / "POES" / "MetOp03_count" / "poes_metop03_cache_parquet"),
    "noaa19":  ("poes_noaa19_io",  HERE / "POES" / "NOAA19_count"  / "poes_noaa19_cache_parquet"),
}
_GK2A_COUNT_CACHE = HERE / "GK2A" / "KSEM_count" / "ksem_cache_parquet"

_DEFAULT_OUT_ROOT = HERE / "4_summary"


def _default_out(catalogs: list[str]) -> Path:
    """--out 미지정 시 기본 master_table.csv 경로. catalogs 조합을 폴더명에
    태그로 넣어(정렬 후 "_" join, 예: noaa / swpc / noaa_swpc) 서로 다른
    --catalogs 실행이 같은 파일을 덮어쓰지 않게 한다 -- 예전엔 --catalogs
    noaa 뒤에 --catalogs swpc를 돌리면 같은 4_summary/master_table.csv를
    그대로 덮어써서 noaa 결과가 통째로 사라지는 사고가 있었음."""
    tag = "_".join(sorted(catalogs))
    return _DEFAULT_OUT_ROOT / tag / "master_table.csv"

_RUNTAG_RE = {
    "blc1_fixed":   re.compile(r"^blc1_fixed_const_on(?P<onset>[\d.]+)_pk(?P<peak>[\d.]+)$"),
    "blc1_lowe":    re.compile(r"^blc1_lowe_w(?P<win>[\d.]+)_m(?P<mult>[\d.]+)"
                               r"_on(?P<onset>[\d.]+)_pk(?P<peak>[\d.]+)$"),
    "quietoff":     re.compile(r"^quietoff_mad(?:_mlb(?P<mlb>[\d.\-]+))?_w(?P<win>[\d.]+)_k(?P<k>[\d.]+)"
                               r"_on(?P<onset>[\d.]+)_pk(?P<peak>[\d.]+)$"),
    "cusum":        re.compile(r"^cusum(?:_mlb(?P<mlb>[\d.\-]+))?_w(?P<win>\d+)_k(?P<k>[\d.]+)_h(?P<h>[\d.]+)"
                               r"_on(?P<onset>[\d.]+)_pk(?P<peak>[\d.]+)$"),
    # GK2A 전용 (POES에는 이 3종 FSM 스크립트 자체가 없음) -- 실물 폴더명 확인 후 추가:
    # quiet7_mad_w10_k0_on0.1_pk0 / quiet7_std_w10_k0_on0.1_pk0 / quietoff_std_w10_k0_on0.1_pk0
    "quiet7_mad":   re.compile(r"^quiet7_mad_w(?P<win>[\d.]+)_k(?P<k>[\d.]+)"
                               r"_on(?P<onset>[\d.]+)_pk(?P<peak>[\d.]+)$"),
    "quiet7_std":   re.compile(r"^quiet7_std_w(?P<win>[\d.]+)_k(?P<k>[\d.]+)"
                               r"_on(?P<onset>[\d.]+)_pk(?P<peak>[\d.]+)$"),
    "quietoff_std": re.compile(r"^quietoff_std_w(?P<win>[\d.]+)_k(?P<k>[\d.]+)"
                               r"_on(?P<onset>[\d.]+)_pk(?P<peak>[\d.]+)$"),
}


def _parse_runtag(runtag: str) -> dict | None:
    """runtag 폴더명 -> {baseline, win, k_or_mult, h, onset, peak, maglat_bins}. 매칭
    실패 시 None. h는 cusum 전용 신규 필드, maglat_bins는 quietoff/cusum의 mlb
    토큰 전용 신규 필드 — 나머지는 항상 NaN(기존 파싱 로직 불변)."""
    for baseline, pat in _RUNTAG_RE.items():
        m = pat.match(runtag)
        if not m:
            continue
        g = m.groupdict()
        mlb = g.get("mlb")
        return {
            "baseline":     baseline,
            "win":          float(g["win"]) if "win" in g else float("nan"),
            "k_or_mult":    float(g["k"] if "k" in g else g.get("mult", "nan")),
            "h":            float(g["h"]) if "h" in g else float("nan"),
            "onset":        float(g["onset"]),
            "peak":         float(g["peak"]),
            "maglat_bins":  mlb if mlb else float("nan"),
        }
    return None


def _nan_key(v: float):
    """float('nan')은 매번 다른 객체라 nan != nan -> set 중복판정 실패. dedup 키용 sentinel 치환."""
    return None if isinstance(v, float) and math.isnan(v) else v


def _load_catalog(catalog: str, era: tuple[str, str]) -> pd.DataFrame:
    cache_dir, io_name = _CATALOG[catalog]
    io = core._import_event_io(io_name, str(cache_dir))
    cat_all, _ = io.load(str(cache_dir))
    return io.filter_by_date(cat_all, *era)


def _channel_universe(detector: str) -> set[str]:
    """detector의 count 캐시(전체 telemetry 채널)에서 채널 유니버스를 읽는다.
    baseline/sweep 결과와 무관한 고정값 — 어떤 --baselines로 돌려도 동일해야 함.
    GK2A는 컬럼이 (pd_key,side,logic) 튜플이지만 ksem_io엔 POES식 tuple_to_fname이
    없어 "PD1A-OU" 형식(=FSM 출력 channel 컬럼과 동일, 실측 확인됨)으로 직접 조합한다.
    logic="TRASH"는 품질 플래그일 뿐 FSM이 검출 채널로 쓰지 않아 제외(실측: 54개 중
    6개 제외 -> 48개, 실제 fsm_onset CSV의 고유 channel 수 48개와 일치 확인됨)."""
    if detector == "gk2a":
        io = core._import_event_io("ksem_io", str(_GK2A_COUNT_CACHE))
        df, _ = io.load(str(_GK2A_COUNT_CACHE))
        return {f"{pd_key}{side}-{logic}" for pd_key, side, logic in df.columns
               if logic != "TRASH"}
    io_name, cache_dir = _POES_IO[detector]
    io = core._import_event_io(io_name, str(cache_dir))
    df, _ = io.load(str(cache_dir))
    return {io.tuple_to_fname(tuple(c)) for c in df.columns}


def build_master_table(detectors: list[str], catalogs: list[str],
                       baselines: list[str], tol: float) -> pd.DataFrame:
    """fsm_onset_*.csv 전용. peak-dedup(아래 seen 키)이 걸려 있어 event CSV(peak_floor
    유효)를 처리하는 함수로 확장/재사용하려면 이 dedup부터 제거해야 함 — 그대로 쓰면
    peak별로 달라야 할 event 행이 하나로 뭉개짐."""
    # catalog는 detector별 era(_ERA_BY_DETECTOR)로 다르게 필터링해야 하므로
    # (catalog, era) 키로 캐시 -- GK2A/POES가 같은 catalog를 요청해도 각자의
    # era로 별도 로드됨.
    cat_cache: dict[tuple[str, tuple], pd.DataFrame] = {}

    def _get_cat(catalog: str, era: tuple[str, str]) -> pd.DataFrame:
        key = (catalog, era)
        if key not in cat_cache:
            cat_cache[key] = _load_catalog(catalog, era)
            print(f"[4_summarize] catalog={catalog} era={era[0]}~{era[1]}  "
                  f"{len(cat_cache[key])}개 이벤트 로드")
        return cat_cache[key]

    rows = []
    for detector in detectors:
        era = _ERA_BY_DETECTOR[detector]
        root = _MATCH_ROOT[detector]
        runtag_dirs = sorted(d for d in root.iterdir() if d.is_dir()) if root.exists() else []
        n_total = len(runtag_dirs)
        n_done = 0
        n_dup = 0
        n_missing_csv = 0
        n_unparsed = 0
        unparsed_examples: list[str] = []
        n_baseline_filtered = 0
        seen: set[tuple] = set()
        # detector 고정 채널 유니버스 (count 캐시 기준) — --baselines를 뭘로 제한해도 불변.
        channel_universe = _channel_universe(detector)
        # 1차 패스: (parsed, catalog, tbl) 버퍼링.
        buffered: list[tuple[dict, str, pd.DataFrame]] = []
        for d in runtag_dirs:
            parsed = _parse_runtag(d.name)
            if parsed is None:
                # 조용한 skip 금지 -- 알려진 4+3종 baseline 어디에도 안 맞는 runtag는
                # 미지의 형식(오타/신규 FSM/수동 폴더)일 수 있어 항상 카운트+예시 보존.
                n_unparsed += 1
                if len(unparsed_examples) < 5:
                    unparsed_examples.append(d.name)
                continue
            if parsed["baseline"] not in baselines:
                n_baseline_filtered += 1  # 정상 파싱됐지만 --baselines 필터로 제외 (의도된 skip)
                continue
            # onset 전용 dedup (검증: peak만 다른 fsm_onset_*.csv는 byte-identical,
            # cusum도 GK2A로 md5 확인함). event CSV는 peak_floor가 실제 축이라 이
            # 스킵을 적용하면 안 됨. h/maglat_bins는 cusum/quietoff-condbg 전용 축 —
            # 다른 baseline은 항상 NaN이라 키에 포함해도 기존 dedup 동작엔 영향 없음.
            # maglat_bins를 반드시 포함해야 함 -- 없으면 서로 다른 --maglat-bins로 돌린
            # 두 condbg 실행이 (win,k,onset)만 같으면 "peak만 다른 중복"으로 오인되어
            # 하나가 유실된다(실제로는 bin 경계가 달라 onset CSV 내용 자체가 다름).
            key = (parsed["baseline"], _nan_key(parsed["win"]),
                  _nan_key(parsed["k_or_mult"]), _nan_key(parsed["h"]), parsed["onset"],
                  _nan_key(parsed["maglat_bins"]))
            if key in seen:
                n_dup += 1
                continue
            csv = d / f"fsm_onset_{d.name}.csv"
            if not csv.exists():
                # key를 seen에 등록하지 않는다 -- 폴더만 있고 CSV가 없는 FSM 크래시
                # 잔재가 dedup 키를 먼저 선점하면, peak만 다른 정상 runtag가 같은
                # 키로 오인되어 skip되고 그 조합 전체가 master_table에서 조용히
                # 유실된다.
                n_missing_csv += 1
                continue
            seen.add(key)
            for catalog in catalogs:
                tbl = core.sweep_table(csv, _get_cat(catalog, era), tol)
                buffered.append((parsed, catalog, tbl))
            n_done += 1
            if n_done % 100 == 0:
                print(f"[4_summarize] {detector}: {n_done}개 조합 처리 중 "
                      f"({n_total}개 폴더 중 {n_dup}개 peak-중복 skip)")
        print(f"[4_summarize] {detector}: 총 {n_total}개 폴더 -> "
              f"{n_done}개 고유 조합 처리, {n_dup}개 peak-중복 skip, "
              f"{n_missing_csv}개 CSV 없음, {n_baseline_filtered}개 --baselines 필터 제외, "
              f"{n_unparsed}개 파싱 실패  (채널 유니버스 {len(channel_universe)}개)")
        if n_missing_csv:
            print(f"[4_summarize] WARNING: fsm_onset CSV 없는 runtag 폴더 {n_missing_csv}개 "
                  f"-> FSM 크래시 잔재 의심, 해당 폴더 확인 필요")
        if n_unparsed:
            print(f"[4_summarize] WARNING: {detector} 파싱 실패 runtag {n_unparsed}개 "
                  f"-> 알려진 baseline 패턴 미매칭(예시: {unparsed_examples}) -- "
                  f"_RUNTAG_RE에 패턴 추가 필요할 수 있음")

        # 2차 패스: 실제 행 + 검출 0인 채널의 명시적 행(POD=0.0, FAR=NaN) 방출.
        for parsed, catalog, tbl in buffered:
            present = set(tbl["channel"])
            for _, row in tbl.iterrows():
                rows.append({
                    "detector": detector, "catalog": catalog,
                    "baseline": parsed["baseline"],
                    "win": parsed["win"], "k_or_mult": parsed["k_or_mult"],
                    "h": parsed["h"], "onset": parsed["onset"],
                    "maglat_bins": parsed["maglat_bins"],
                    "channel": row["channel"],
                    "POD": row["POD"], "FAR": row["FAR"],
                    "n_det": row["n_det"], "n_hit": row["n_hit"],
                    "n_fa": row["n_fa"], "n_fa_saa": row["n_fa_saa"],
                    "fa_maglat_median": row["fa_maglat_median"],
                    "fa_maglat_p10": row["fa_maglat_p10"],
                    "fa_maglat_p90": row["fa_maglat_p90"],
                    "tp_maglat_median": row["tp_maglat_median"],
                    "fa_bmag_median": row["fa_bmag_median"],
                    "detected": True,
                })
            for ch in sorted(channel_universe - present):
                rows.append({
                    "detector": detector, "catalog": catalog,
                    "baseline": parsed["baseline"],
                    "win": parsed["win"], "k_or_mult": parsed["k_or_mult"],
                    "h": parsed["h"], "onset": parsed["onset"],
                    "maglat_bins": parsed["maglat_bins"],
                    "channel": ch,
                    "POD": 0.0, "FAR": float("nan"),
                    "n_det": 0, "n_hit": 0,
                    "n_fa": 0, "n_fa_saa": float("nan"),
                    "fa_maglat_median": float("nan"),
                    "fa_maglat_p10": float("nan"),
                    "fa_maglat_p90": float("nan"),
                    "tp_maglat_median": float("nan"),
                    "fa_bmag_median": float("nan"),
                    "detected": False,
                })

    return pd.DataFrame(rows)


# ── STEP C: best_table ──────────────────────────────────────────────────

CRITERIA = ("min_far", "max_pod", "youden", "f1")

# 기준별: (score 컬럼, score 오름차순 정렬 여부, tie-break 컬럼, tie-break 오름차순 여부)
_CRITERION_KEY = {
    "min_far": ("FAR",    True,  "POD", False),
    "max_pod": ("POD",    False, "FAR", True),
    "youden":  ("youden", False, "POD", False),
    "f1":      ("f1",     False, "POD", False),
}

# 예측 모델 라벨 앵커 후보 채널 (pro proton telescope p4/p5, tel0/tel90)
ANCHOR_CHANNELS = ["pro_tel0_p4", "pro_tel0_p5", "pro_tel90_p4", "pro_tel90_p5"]


def _add_scores(df: pd.DataFrame) -> pd.DataFrame:
    """youden/f1 score 컬럼을 부착 (min_far/max_pod는 기존 POD/FAR 그대로 사용)."""
    df = df.copy()
    df["youden"] = df["POD"] - df["FAR"]
    denom = df["n_hit"] + df["n_fa"]
    prec = (df["n_hit"] / denom.replace(0, np.nan)).fillna(0.0)
    rec = df["POD"]
    f1_denom = prec + rec
    df["f1"] = (2 * prec * rec / f1_denom.replace(0, np.nan)).fillna(0.0)
    return df


def select_best(df: pd.DataFrame, criterion: str, min_n_hit: int) -> pd.DataFrame:
    """(detector,catalog,baseline) 그룹별 criterion 1등 + 커버리지 컬럼.
    FAR=NaN(detected=False) 행은 notna() 필터로 자동 제외, n_hit<min_n_hit도 제외."""
    scored = _add_scores(df)
    score_col, ascending, tie_col, tie_ascending = _CRITERION_KEY[criterion]

    out_rows = []
    for keys, grp in scored.groupby(["detector", "catalog", "baseline"], sort=False):
        detector, catalog, baseline = keys
        n_nan = int(grp["FAR"].isna().sum())
        n_low_nhit = int((grp["FAR"].notna() & (grp["n_hit"] < min_n_hit)).sum())
        cand = grp[grp["FAR"].notna() & (grp["n_hit"] >= min_n_hit)]
        if cand.empty:
            print(f"[best] WARNING {detector}/{catalog}/{baseline} [{criterion}]: "
                  f"후보 없음 (FAR-NaN {n_nan}, n_hit<{min_n_hit} {n_low_nhit}) -> skip")
            continue
        rank = cand[score_col].rank(method="min", ascending=ascending)
        ranked = cand.assign(_rank=rank).sort_values(
            [score_col, tie_col], ascending=[ascending, tie_ascending])
        best = ranked.iloc[0]
        row = best.to_dict()
        row["criterion"]            = criterion
        row["score"]                = best[score_col]
        row["n_channels_in_group"]  = int(cand["channel"].nunique())
        row["rank_of_best"]         = int(best["_rank"])
        row["n_excluded_far_nan"]   = n_nan
        row["n_excluded_low_nhit"]  = n_low_nhit
        row.pop("_rank", None)
        out_rows.append(row)
        print(f"[best] {detector}/{catalog}/{baseline} [{criterion}]: "
              f"후보 {len(cand)}개(FAR-NaN {n_nan}, n_hit<{min_n_hit} {n_low_nhit} 제외) "
              f"-> channel={best['channel']} POD={best['POD']} FAR={best['FAR']}")
    return pd.DataFrame(out_rows)


def build_best_table_all_criteria(df: pd.DataFrame, min_n_hit: int) -> pd.DataFrame:
    frames = [select_best(df, c, min_n_hit) for c in CRITERIA]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def anchor_channel_report(df: pd.DataFrame, min_n_hit: int) -> pd.DataFrame:
    """ANCHOR_CHANNELS(pro p4/p5 tel0/tel90)의 (detector,catalog,baseline)별
    최고 POD / 최저 FAR(둘 다 n_hit>=min_n_hit)과 그 조합을 뽑는다."""
    sub = df[df["channel"].isin(ANCHOR_CHANNELS)
            & df["FAR"].notna() & (df["n_hit"] >= min_n_hit)]
    rows = []
    for keys, grp in sub.groupby(["detector", "catalog", "baseline", "channel"], sort=False):
        detector, catalog, baseline, channel = keys
        bp = grp.loc[grp["POD"].idxmax()]
        bf = grp.loc[grp["FAR"].idxmin()]
        rows.append({
            "detector": detector, "catalog": catalog,
            "baseline": baseline, "channel": channel,
            "best_POD": bp["POD"], "best_POD_FAR": bp["FAR"], "best_POD_n_hit": bp["n_hit"],
            "best_POD_win": bp["win"], "best_POD_k_or_mult": bp["k_or_mult"],
            "best_POD_onset": bp["onset"],
            "best_FAR": bf["FAR"], "best_FAR_POD": bf["POD"], "best_FAR_n_hit": bf["n_hit"],
            "best_FAR_win": bf["win"], "best_FAR_k_or_mult": bf["k_or_mult"],
            "best_FAR_onset": bf["onset"],
        })
    return pd.DataFrame(rows)


def _print_anchor_highlight(best_df: pd.DataFrame):
    hi = best_df[best_df["channel"].isin(["pro_tel0_p4", "pro_tel0_p5"])]
    if hi.empty:
        print("[best] pro_tel0_p4/p5는 이번 criterion에서 best로 뽑히지 않음.")
        return
    print("[best] ▶ pro_tel0_p4/p5 강조:")
    cols = ["detector", "catalog", "baseline", "channel", "win", "k_or_mult",
            "onset", "POD", "FAR", "n_hit"]
    cols = [c for c in cols if c in hi.columns]
    print(hi[cols].to_string(index=False))


def main():
    ap = argparse.ArgumentParser(
        description="detector(gk2a/POES) sweep runtag 순회 매칭 -> master_table.csv")
    ap.add_argument("--detectors", default="metop03,noaa19",
                    help="콤마 리스트. 선택: gk2a, metop03, noaa19 "
                         "(gk2a는 기본값에 없음 — 명시적으로 추가해야 포함됨)")
    ap.add_argument("--catalogs",  default="noaa,swpc",
                    help="콤마 리스트. 선택: noaa, swpc")
    ap.add_argument("--baselines", default="blc1_fixed,blc1_lowe,quietoff",
                    help="콤마 리스트. 선택: blc1_fixed, blc1_lowe, quietoff, cusum, "
                         "quiet7_mad, quiet7_std, quietoff_std (cusum과 GK2A 전용 3종은 "
                         "기본값에 없음 — 명시적으로 추가해야 포함됨. quiet7_mad/quiet7_std/"
                         "quietoff_std는 POES엔 해당 FSM이 없어 실질적으로 GK2A 전용)")
    ap.add_argument("--tol", type=float, default=core.MATCH_TOL_H)
    ap.add_argument("--out", default=None,
                    help="master_table.csv 경로 (기본: 4_summary/<catalogs 태그>/"
                         "master_table.csv -- --catalogs 조합별로 자동 분리돼 다른 "
                         "조합의 결과를 덮어쓰지 않음. 명시 시 그 경로 그대로 사용하고 "
                         "아래 덮어쓰기 보호검사는 생략 -- 사용자 책임)")
    ap.add_argument("--force", action="store_true",
                    help="기본 경로(--out 미지정)에 이미 master_table.csv가 있어도 "
                         "덮어쓰기 허용 (--out을 명시하면 항상 통과이므로 무관)")
    ap.add_argument("--master-in", default=None,
                    help="master_table.csv를 새로 만들지 않고 기존 파일을 재사용")
    ap.add_argument("--criterion", default="min_far", choices=CRITERIA,
                    help="best_table.csv 선정 기준 (기본 min_far)")
    ap.add_argument("--min-n-hit", type=int, default=3,
                    help="best 후보에서 제외할 n_hit 최소치 (기본 3)")
    ap.add_argument("--best-out", default=None,
                    help="best_table.csv 경로 (기본: --out과 같은 폴더)")
    ap.add_argument("--best-by-criterion-out", default=None,
                    help="best_table_by_criterion.csv 경로 (기본: --out과 같은 폴더)")
    ap.add_argument("--anchor-out", default=None,
                    help="anchor 채널(pro p4/p5) 리포트 경로 (기본: --out과 같은 폴더)")
    ap.add_argument("--xlsx", action="store_true",
                    help="master/best/best_by_criterion/anchor 4개를 시트로 묶은 "
                         "summary.xlsx도 같이 저장")
    ap.add_argument("--xlsx-out", default=None,
                    help="summary.xlsx 경로 (기본: --out과 같은 폴더의 summary.xlsx)")
    args = ap.parse_args()

    detectors = [s.strip() for s in args.detectors.split(",") if s.strip()]
    catalogs  = [s.strip() for s in args.catalogs.split(",") if s.strip()]
    baselines = [s.strip() for s in args.baselines.split(",") if s.strip()]

    for d in detectors:
        if d not in _MATCH_ROOT:
            raise SystemExit(f"[4_summarize] ERROR: unknown detector '{d}' "
                             f"(choices: {list(_MATCH_ROOT)})")
    for c in catalogs:
        if c not in _CATALOG:
            raise SystemExit(f"[4_summarize] ERROR: unknown catalog '{c}' "
                             f"(choices: {list(_CATALOG)})")
    for b in baselines:
        if b not in _RUNTAG_RE:
            raise SystemExit(f"[4_summarize] ERROR: unknown baseline '{b}' "
                             f"(choices: {list(_RUNTAG_RE)})")

    out_explicit = args.out is not None
    out = Path(args.out) if out_explicit else _default_out(catalogs)
    t0 = time.time()
    if args.master_in:
        df = pd.read_csv(args.master_in)
        print(f"[4_summarize] master_table 로드 <- {args.master_in} ({len(df)} rows)")
    else:
        # 기본 경로(--out 미지정)에서만 덮어쓰기 보호 -- --catalogs 조합별로
        # 폴더가 이미 분리돼 있어(_default_out) 정상적으로는 안 겹치지만, 같은
        # 조합을 그대로 재실행하는 경우까지 실수로 덮어쓰지 않도록 막는다.
        # --out 명시 시엔 3_event_run.py와 동일 원칙으로 사용자 책임 -- 검사 생략.
        if not out_explicit and out.exists() and not args.force:
            raise SystemExit(
                f"[4_summarize] ERROR: 출력 파일이 이미 있습니다 -> {out}\n"
                f"  다른 --catalogs 조합 결과를 덮어쓸 위험이 있어 중단합니다. "
                f"덮어쓰려면 --force, 다른 경로로 저장하려면 --out을 지정하세요.")
        df = build_master_table(detectors, catalogs, baselines, args.tol)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"[4_summarize] master_table 저장 -> {out} "
              f"({len(df)} rows, {time.time() - t0:.1f}s)")

    if df.empty:
        print("[4_summarize] master_table이 비어 있음 (조건에 맞는 runtag/CSV 없음) "
              "-> best_table 단계 생략.")
        return

    # ── STEP C: best_table ──────────────────────────────────────────────
    best_out       = Path(args.best_out) if args.best_out else out.parent / "best_table.csv"
    best_all_out   = Path(args.best_by_criterion_out) if args.best_by_criterion_out \
                     else out.parent / "best_table_by_criterion.csv"
    anchor_out     = Path(args.anchor_out) if args.anchor_out \
                     else out.parent / "anchor_channels.csv"

    best = select_best(df, args.criterion, args.min_n_hit)
    best_out.parent.mkdir(parents=True, exist_ok=True)
    best.to_csv(best_out, index=False)
    print(f"[4_summarize] best_table[{args.criterion}] 저장 -> {best_out} ({len(best)} rows)")
    _print_anchor_highlight(best)

    best_all = build_best_table_all_criteria(df, args.min_n_hit)
    best_all.to_csv(best_all_out, index=False)
    print(f"[4_summarize] best_table_by_criterion 저장 -> {best_all_out} ({len(best_all)} rows)")

    anchor = anchor_channel_report(df, args.min_n_hit)
    anchor.to_csv(anchor_out, index=False)
    print(f"[4_summarize] anchor 채널 리포트 저장 -> {anchor_out} ({len(anchor)} rows)")
    if not anchor.empty:
        print(anchor.to_string(index=False))

    if args.xlsx:
        xlsx_out = Path(args.xlsx_out) if args.xlsx_out else out.parent / "summary.xlsx"
        xlsx_out.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(xlsx_out, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="master_table", index=False)
            best.to_excel(writer, sheet_name="best_table", index=False)
            best_all.to_excel(writer, sheet_name="best_table_by_criterion", index=False)
            anchor.to_excel(writer, sheet_name="anchor_channels", index=False)
        print(f"[4_summarize] summary.xlsx 저장 -> {xlsx_out} "
              f"(4시트: master_table/best_table/best_table_by_criterion/anchor_channels)")


if __name__ == "__main__":
    main()

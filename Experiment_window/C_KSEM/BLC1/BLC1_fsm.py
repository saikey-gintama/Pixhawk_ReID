"""
ksem_fsm.py
===========
KSEM Particle Detector 데이터로부터 입자 환경 상태를 분류하는 FSM.

사용법:
    python ksem_fsm.py --data_dir /path/to/csv/folder
    python ksem_fsm.py --data_dir /path/to/csv/folder --output_dir ./output

파일명 패턴: *PD1*, *PD2*, *PD3* (대소문자 무관)
"""

import argparse
import glob
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# ============================================================
# 0. 설정 (CONFIG)
# ============================================================

# --- Bin 구조 정의 (논문 Table 3.3 / 3.4 기반) ---
# 형식: (bin_index_start, bin_index_end_inclusive, logic_name, particle, energy_min_keV, energy_max_keV)
BIN_GROUPS = [
    # Proton channels
    ( 0,  0, "O_UF",    "proton",   None,   None),   # O underflow (구분자, 제외)
    ( 1, 23, "O",       "proton",     50,   6000),
    (24, 24, "OU_UF",   "proton",   None,   None),   # OU underflow (구분자, 제외)
    (25, 41, "OU",      "proton",   4000,  12000),
    (42, 44, "UT",      "cosmic",  10000,  14000),   # cosmic ray (UT event logic)
    (45, 45, "OUT_UF",  "proton",   None,   None),   # OUT underflow (구분자, 제외)
    (46, 56, "OUT",     "proton",   8000,  18000),
    # Electron channels
    (57, 57, "F_UF",    "electron", None,   None),   # F underflow (구분자, 제외)
    (58, 80, "F",       "electron",   50,    600),
    (81, 81, "FT_UF",   "electron", None,   None),
    (82, 104,"FT",      "electron",  400,   1500),
    (105,105,"FTU_UF",  "electron", None,   None),
    (106,116,"FTU",     "electron", 1400,   2400),
    (117,117,"FTUO_UF", "electron", None,   None),
    (118,126,"FTUO",    "electron", 2200,   3800),
    # Special
    (127,127,"Trash",   "invalid",  None,   None),
]

# bin_index → group 이름 룩업 생성
BIN_TO_GROUP: Dict[int, str] = {}
for (s, e, name, *_) in BIN_GROUPS:
    for i in range(s, e + 1):
        BIN_TO_GROUP[i] = name

# 분석에 사용할 proton 그룹 (underflow / cosmic / trash 제외)
PROTON_GROUPS   = ["O", "OU", "OUT"]
ELECTRON_GROUPS = ["F", "FT", "FTU", "FTUO"]
EXCLUDE_GROUPS  = {"O_UF", "OU_UF", "OUT_UF", "F_UF", "FT_UF", "FTU_UF", "FTUO_UF",
                   "Trash", "UT"}

# --- FSM 임계값 (쉽게 바꿀 수 있도록 한 곳에) ---
# 기준선(baseline) 추정: 하루 데이터의 하위 percentile
BASELINE_PERCENTILE = 10       # %

# 상태 전이 배수 (O event logic 합 기준, 기준선 대비)
THRESH_ELEVATED =  2.0         # baseline × 2 → ELEVATED
THRESH_ACTIVE   =  5.0         # baseline × 5 → ACTIVE
THRESH_STORM    = 10.0         # baseline × 10 → STORM

# 변화율 임계값 (전 분 대비 상대 변화율)
RATE_ACTIVE     =  1.0         # 100% 상승/분 → 즉시 ACTIVE 고려
RATE_STORM      =  3.0         # 300% 상승/분 → 즉시 STORM 고려

# 디바운스: 상위 전이는 N분 연속, 하위 복귀는 M분 연속
DEBOUNCE_UP   = 3              # 분 (QUIET→ELEVATED→ACTIVE→STORM)
DEBOUNCE_DOWN = 10             # 분 (STORM→ACTIVE→ELEVATED→QUIET)

# Trash 비율 임계: 이 이상이면 low_confidence 플래그
TRASH_RATIO_WARN = 0.30        # 30%

# 롤링 평균 윈도우
ROLLING_WINDOW_SHORT = 5       # 분
ROLLING_WINDOW_LONG  = 15      # 분

# PD 투표: 최소 동의 PD 수 (1 or 2 or 3)
MIN_PD_VOTE = 2


# ============================================================
# 1. STATE 정의
# ============================================================

class State(Enum):
    QUIET    = 0
    ELEVATED = 1
    ACTIVE   = 2
    STORM    = 3

STATE_COLORS = {
    State.QUIET:    "#4a9ede",
    State.ELEVATED: "#f5a623",
    State.ACTIVE:   "#e85d04",
    State.STORM:    "#9b2226",
}

STATE_LABELS = {
    State.QUIET:    "QUIET",
    State.ELEVATED: "ELEVATED",
    State.ACTIVE:   "ACTIVE",
    State.STORM:    "STORM",
}


# ============================================================
# 2. LOADER
# ============================================================

def find_csv_files(data_dir: str) -> Dict[str, str]:
    """폴더에서 PD1/PD2/PD3 CSV 파일을 자동 탐색."""
    data_dir = Path(data_dir)
    result = {}
    for pd_key in ["PD1", "PD2", "PD3"]:
        pattern_upper = str(data_dir / f"*{pd_key}*.csv")
        pattern_lower = str(data_dir / f"*{pd_key.lower()}*.csv")
        matches = glob.glob(pattern_upper) + glob.glob(pattern_lower)
        # 대소문자 혼합 대비: 직접 검색
        if not matches:
            for f in data_dir.glob("*.csv"):
                if pd_key.lower() in f.name.lower():
                    matches.append(str(f))
        if matches:
            result[pd_key] = matches[0]
    return result


def parse_bin_index(col: str, prefix: str) -> Optional[int]:
    """'A12(548)' → 12, 'B0(0)' → 0."""
    try:
        rest = col[len(prefix):]          # '12(548)'
        idx = int(rest.split("(")[0])
        return idx
    except Exception:
        return None


def load_pd_csv(filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    CSV를 읽어 (df_A, df_B) 반환.
    각 DataFrame: index=Time, columns=bin_index(int), 값=count/min.
    """
    raw = pd.read_csv(filepath)
    raw["Time"] = pd.to_datetime(raw["Time"])
    raw = raw.set_index("Time").sort_index()

    def extract_detector(prefix: str) -> pd.DataFrame:
        cols = {c: parse_bin_index(c, prefix)
                for c in raw.columns if c.startswith(prefix)}
        cols = {c: idx for c, idx in cols.items() if idx is not None}
        df = raw[list(cols.keys())].copy()
        df.columns = [cols[c] for c in df.columns]   # int bin index로 rename
        df = df[sorted(df.columns)]
        return df

    return extract_detector("A"), extract_detector("B")


# ============================================================
# 3. PREPROCESSOR
# ============================================================

def get_group_cols(df: pd.DataFrame, group_names: List[str]) -> List[int]:
    """지정한 그룹에 속하는 bin index 목록 반환."""
    return [idx for idx in df.columns
            if BIN_TO_GROUP.get(idx) in group_names]


def compute_features(df_A: pd.DataFrame,
                     df_B: pd.DataFrame,
                     pd_label: str) -> pd.DataFrame:
    """
    특징 벡터 생성.
    Returns DataFrame with columns:
        o_total, ou_total, out_total, proton_total,
        trash_count, trash_ratio,
        o_rate (변화율), low_confidence, pd_label
    """
    feat = pd.DataFrame(index=df_A.index)

    # 각 event logic 합산 (A + B 검출기 평균)
    for grp in PROTON_GROUPS + ["Trash"]:
        cols_A = get_group_cols(df_A, [grp])
        cols_B = get_group_cols(df_B, [grp])
        sum_A = df_A[cols_A].sum(axis=1) if cols_A else 0
        sum_B = df_B[cols_B].sum(axis=1) if cols_B else 0
        feat[f"{grp.lower()}_total"] = (sum_A + sum_B) / 2.0

    # proton_total = O + OU + OUT
    feat["proton_total"] = (feat["o_total"]
                            + feat["ou_total"]
                            + feat["out_total"])

    # trash_ratio = Trash / (proton_total + Trash), NaN → 0
    denom = feat["proton_total"] + feat["trash_total"]
    feat["trash_ratio"] = np.where(denom > 0,
                                   feat["trash_total"] / denom, 0.0)

    # 롤링 평균 (노이즈 완화)
    feat["o_smooth_short"] = (feat["o_total"]
                              .rolling(ROLLING_WINDOW_SHORT, min_periods=1)
                              .mean())
    feat["o_smooth_long"]  = (feat["o_total"]
                              .rolling(ROLLING_WINDOW_LONG,  min_periods=1)
                              .mean())

    # 변화율 (전 분 대비, 상대값)
    prev = feat["o_smooth_short"].shift(1)
    feat["o_rate"] = np.where(prev > 0,
                              (feat["o_smooth_short"] - prev) / prev,
                              0.0)
    feat["o_rate"] = feat["o_rate"].fillna(0.0)

    # 기준선(baseline): 하루 데이터 하위 percentile
    baseline = float(np.percentile(feat["o_total"].dropna(),
                                   BASELINE_PERCENTILE))
    baseline = max(baseline, 1.0)   # 0 나누기 방지
    feat["baseline"] = baseline
    feat["o_ratio"]  = feat["o_smooth_short"] / baseline

    # low_confidence 플래그
    feat["low_confidence"] = feat["trash_ratio"] > TRASH_RATIO_WARN

    feat["pd_label"] = pd_label
    return feat


# ============================================================
# 4. FSM
# ============================================================

@dataclass
class FSMContext:
    """FSM 내부 상태 (디바운스 카운터 포함)."""
    current_state: State = State.QUIET
    candidate_state: State = State.QUIET   # 전이 후보
    counter: int = 0                       # 연속 유지 분 수

    def reset_counter(self):
        self.counter = 0

    def tick(self):
        self.counter += 1


def _raw_target_state(o_ratio: float, o_rate: float) -> State:
    """임계값만 보고 이 순간의 '원하는 상태'를 반환 (디바운스 미적용)."""
    # 변화율 기반 즉시 격상 판단
    if o_rate >= RATE_STORM:
        return State.STORM
    if o_rate >= RATE_ACTIVE:
        return State.ACTIVE

    # 절대 수준 기반
    if o_ratio >= THRESH_STORM:
        return State.STORM
    if o_ratio >= THRESH_ACTIVE:
        return State.ACTIVE
    if o_ratio >= THRESH_ELEVATED:
        return State.ELEVATED
    return State.QUIET


def fsm_step(ctx: FSMContext, o_ratio: float, o_rate: float) -> State:
    """
    1분 타임스텝 FSM 전이 함수.
    디바운스 적용: 상위 전이 DEBOUNCE_UP분, 하위 복귀 DEBOUNCE_DOWN분.
    """
    target = _raw_target_state(o_ratio, o_rate)

    going_up   = target.value > ctx.current_state.value
    going_down = target.value < ctx.current_state.value
    staying    = target == ctx.current_state

    if staying:
        # 현재 상태와 동일 → 후보 초기화
        ctx.candidate_state = ctx.current_state
        ctx.reset_counter()
        return ctx.current_state

    # 후보가 바뀌었으면 카운터 리셋
    if target != ctx.candidate_state:
        ctx.candidate_state = target
        ctx.reset_counter()

    ctx.tick()

    debounce = DEBOUNCE_UP if going_up else DEBOUNCE_DOWN

    if ctx.counter >= debounce:
        ctx.current_state = ctx.candidate_state
        ctx.reset_counter()

    return ctx.current_state


def run_fsm_on_features(feat: pd.DataFrame) -> pd.DataFrame:
    """특징 DataFrame에 FSM을 적용해 state 컬럼 추가."""
    ctx = FSMContext()
    states = []
    raw_targets = []

    for _, row in feat.iterrows():
        raw = _raw_target_state(row["o_ratio"], row["o_rate"])
        state = fsm_step(ctx, row["o_ratio"], row["o_rate"])
        states.append(state)
        raw_targets.append(raw)

    feat = feat.copy()
    feat["state"]      = states
    feat["raw_target"] = raw_targets
    feat["state_val"]  = [s.value for s in states]
    return feat


# ============================================================
# 5. VOTING (PD 간 통합)
# ============================================================

def vote_states(results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    3개 PD 결과를 투표로 통합.
    MIN_PD_VOTE개 이상 동의한 상태를 채택. 미달 시 가장 높은 상태.
    """
    pd_keys = list(results.keys())
    ref_idx = results[pd_keys[0]].index

    vote_df = pd.DataFrame(index=ref_idx)
    for key, df in results.items():
        vote_df[key] = df["state_val"].values

    def _vote_row(row):
        counts = {}
        for v in row:
            counts[v] = counts.get(v, 0) + 1
        # MIN_PD_VOTE 이상 동의
        for state_val, cnt in sorted(counts.items(), reverse=True):
            if cnt >= MIN_PD_VOTE:
                return state_val
        # 동의 없으면 최대값 (보수적 선택)
        return int(max(row))

    vote_df["voted_state_val"] = vote_df[pd_keys].apply(_vote_row, axis=1)
    vote_df["voted_state"] = vote_df["voted_state_val"].map(
        lambda v: State(v))

    # low_confidence: 어느 한 PD라도 플래그 있으면
    for key, df in results.items():
        vote_df[f"lc_{key}"] = df["low_confidence"].values
    lc_cols = [c for c in vote_df.columns if c.startswith("lc_")]
    vote_df["low_confidence"] = vote_df[lc_cols].any(axis=1)

    return vote_df


# ============================================================
# 6. VISUALIZER
# ============================================================

def plot_results(results: Dict[str, pd.DataFrame],
                 vote_df: pd.DataFrame,
                 output_path: str):
    """
    4-panel 플롯:
      1) O total 시계열 (3 PD)
      2) FSM 상태 (3 PD + 투표)
      3) Trash ratio (3 PD)
      4) O 변화율 (PD1)
    """
    pd_colors = {"PD1": "#3266ad", "PD2": "#d4653b", "PD3": "#2a9d6a"}
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle("KSEM PD Particle Environment FSM", fontsize=13, y=0.98)

    # --- Panel 1: O total ---
    ax = axes[0]
    for pd_key, df in results.items():
        ax.plot(df.index, df["o_total"],
                color=pd_colors[pd_key], alpha=0.4, linewidth=0.8,
                label=f"{pd_key} raw")
        ax.plot(df.index, df["o_smooth_short"],
                color=pd_colors[pd_key], linewidth=1.5,
                label=f"{pd_key} {ROLLING_WINDOW_SHORT}min avg")
    ax.set_ylabel("O event\n(count/min)", fontsize=9)
    ax.legend(fontsize=7, ncol=3, loc="upper left")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # --- Panel 2: FSM 상태 ---
    ax = axes[1]
    state_map = {State.QUIET: 0, State.ELEVATED: 1,
                 State.ACTIVE: 2, State.STORM: 3}
    cmap = ListedColormap([STATE_COLORS[s] for s in
                           [State.QUIET, State.ELEVATED,
                            State.ACTIVE, State.STORM]])

    # 투표 결과를 배경 색상으로
    times = vote_df.index
    svals = vote_df["voted_state_val"].values
    for i in range(len(times) - 1):
        ax.axvspan(times[i], times[i + 1],
                   alpha=0.25,
                   color=STATE_COLORS[State(svals[i])])

    # 각 PD 상태 선
    offset = {0: 0.0, 1: 0.1, 2: -0.1}
    for j, (pd_key, df) in enumerate(results.items()):
        y = df["state_val"].values + offset[j]
        ax.plot(df.index, y, color=pd_colors[pd_key],
                linewidth=1.2, alpha=0.8, label=pd_key)

    # 투표 결과 굵은 선
    ax.plot(vote_df.index, vote_df["voted_state_val"],
            color="black", linewidth=2.0, linestyle="--",
            label="Voted", zorder=5)

    # low_confidence 음영
    lc_mask = vote_df["low_confidence"].values
    for i in range(len(times) - 1):
        if lc_mask[i]:
            ax.axvspan(times[i], times[i + 1],
                       alpha=0.15, color="gray", hatch="//")

    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["QUIET", "ELEVATED", "ACTIVE", "STORM"], fontsize=8)
    ax.set_ylabel("FSM State", fontsize=9)
    ax.legend(fontsize=7, loc="upper left", ncol=4)
    ax.grid(True, alpha=0.3, axis="x")

    # 패치 범례
    patches = [mpatches.Patch(color=STATE_COLORS[s], label=STATE_LABELS[s],
                               alpha=0.5)
               for s in State]
    ax.legend(handles=patches + [
        mpatches.Patch(color="gray", alpha=0.3, hatch="//",
                       label="Low confidence")],
              fontsize=7, loc="upper right", ncol=5)

    # --- Panel 3: Trash ratio ---
    ax = axes[2]
    for pd_key, df in results.items():
        ax.plot(df.index, df["trash_ratio"] * 100,
                color=pd_colors[pd_key], linewidth=1.2, label=pd_key)
    ax.axhline(TRASH_RATIO_WARN * 100, color="red",
               linestyle="--", linewidth=1, alpha=0.7,
               label=f"Low-conf threshold ({TRASH_RATIO_WARN*100:.0f}%)")
    ax.set_ylabel("Trash ratio (%)", fontsize=9)
    ax.legend(fontsize=7, loc="upper left", ncol=4)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # --- Panel 4: O 변화율 (PD1) ---
    ax = axes[3]
    df1 = list(results.values())[0]
    ax.plot(df1.index, df1["o_rate"] * 100,
            color=pd_colors["PD1"], linewidth=1.0)
    ax.axhline(RATE_ACTIVE * 100, color="#e85d04",
               linestyle="--", linewidth=1, alpha=0.8,
               label=f"ACTIVE rate threshold ({RATE_ACTIVE*100:.0f}%/min)")
    ax.axhline(RATE_STORM * 100, color="#9b2226",
               linestyle="--", linewidth=1, alpha=0.8,
               label=f"STORM rate threshold ({RATE_STORM*100:.0f}%/min)")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_ylabel("O rate (%/min)\nPD1", fontsize=9)
    ax.set_xlabel("Time (UTC)", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-200, 600)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] 저장 완료: {output_path}")


# ============================================================
# 7. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="KSEM PD FSM")
    parser.add_argument("--data_dir",   default=".",
                        help="CSV 파일이 있는 폴더 경로")
    parser.add_argument("--output_dir", default="./output",
                        help="결과 저장 폴더")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- 파일 탐색 ---
    csv_files = find_csv_files(args.data_dir)
    if not csv_files:
        raise FileNotFoundError(
            f"'{args.data_dir}' 폴더에서 PD1/PD2/PD3 CSV 파일을 찾을 수 없습니다.")
    print(f"[Load] 발견된 파일:")
    for k, v in csv_files.items():
        print(f"  {k}: {v}")

    # --- 로드 & 특징 추출 & FSM ---
    results: Dict[str, pd.DataFrame] = {}
    for pd_key, filepath in sorted(csv_files.items()):
        print(f"[Process] {pd_key} ...")
        df_A, df_B = load_pd_csv(filepath)
        feat = compute_features(df_A, df_B, pd_label=pd_key)
        feat = run_fsm_on_features(feat)
        results[pd_key] = feat

        # 기준선 출력
        baseline = float(feat["baseline"].iloc[0])
        print(f"  baseline (O logic, {BASELINE_PERCENTILE}th pct) = "
              f"{baseline:.2f} count/min")

    # --- 투표 ---
    vote_df = vote_states(results)

    # --- 상태 전이 요약 출력 ---
    print("\n[FSM] 상태 전이 요약 (투표 결과):")
    prev_state = None
    transitions = []
    for t, row in vote_df.iterrows():
        s = row["voted_state"]
        if s != prev_state:
            transitions.append((t, s, row["low_confidence"]))
            prev_state = s
    for t, s, lc in transitions:
        lc_mark = " [low confidence]" if lc else ""
        print(f"  {t.strftime('%H:%M')}  →  {STATE_LABELS[s]}{lc_mark}")

    # 각 상태 체류 시간
    print("\n[FSM] 상태별 체류 시간:")
    state_counts = vote_df["voted_state"].value_counts()
    total = len(vote_df)
    for s in State:
        cnt = state_counts.get(s, 0)
        pct = cnt / total * 100
        print(f"  {STATE_LABELS[s]:10s}: {cnt:5d} 분 ({pct:5.1f}%)")

    # --- CSV 저장 ---
    out_csv = os.path.join(args.output_dir, "fsm_states.csv")
    # 투표 결과 + PD별 상태 합본
    save_df = vote_df[["voted_state_val", "low_confidence"]].copy()
    save_df["voted_state"] = save_df["voted_state_val"].map(
        lambda v: STATE_LABELS[State(v)])
    for pd_key, df in results.items():
        save_df[f"{pd_key}_state"] = df["state"].map(
            lambda s: STATE_LABELS[s]).values
        save_df[f"{pd_key}_o_total"]    = df["o_total"].values
        save_df[f"{pd_key}_o_ratio"]    = df["o_ratio"].round(3).values
        save_df[f"{pd_key}_trash_ratio"] = df["trash_ratio"].round(4).values
    save_df.to_csv(out_csv)
    print(f"\n[Output] CSV 저장: {out_csv}")

    # --- 플롯 저장 ---
    out_plot = os.path.join(args.output_dir, "fsm_plot.png")
    plot_results(results, vote_df, out_plot)

    print("\n완료.")


if __name__ == "__main__":
    main()

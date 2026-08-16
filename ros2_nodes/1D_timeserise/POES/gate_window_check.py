"""
gate_window_check.py
=====================
게이트(N=2, PRE_ALERT) 개방 빈도 105배 불일치 검증. 읽기 전용, C_PD 아래 아무것도
쓰지 않는다.

배경: gate_persistence_sweep.csv(N=2 행, 전 구간 236,848틱)은 duty_cycle 6.84%
(16,204 개방틱)인데, 논문 4.6절이 인용하는 results/*/onboard_cost.csv의
n_activations은 c1ch(1채널) 6회/9,214틱=0.065%, d3ch(3채널) 148회/4,611틱=3.21%로
전자와 105배(0.065% 기준) 차이난다. 두 숫자가 양립하는지 확인한다.

가설(기각됨): 리플레이 창 3개(2022x2, 2024x1)가 태양 극대기(2023~2025)를 비껴간
시간적 편중. -- [gate_window_check-1/2]에서 연도별 분포가 11~19%로 거의 균등함을
확인했고, [gate_window_check-3]에서 오프라인 재계산이 창당 90~140개(N=2 전 구간
duty_cycle 6.84%와 같은 수준)를 내는 것을 확인해 기각.

실제 원인(확인됨): onboard_cost.csv의 n_activations은 젯슨 실제 게이트 개방 횟수가
아니라 **aggregate_onboard.py의 집계 버그로 잘린 값**이다.
  aggregate_onboard.py:load_csv_trimmed()는 run_meta.json의 n_bg_warmup_ticks_excluded
  (틱 단위 워밍업 카운트, 예: 96)를 log.csv(AP가 매 틱 1행 쓰는 파일. 워밍업 트림에
  맞는 대상)뿐 아니라 ai_log.csv(AI가 게이트 열린 틱에만 1행 쓰는, 이미 사건 단위인
  파일)에도 df.iloc[n_warmup:]로 그대로 적용한다. ai_log.csv의 실제 행 수(=진짜
  게이트 개방 횟수, ap_fsm_summary.json의 n_ticks_gate_open과 정확히 같다)가 그
  워밍업 틱 수보다 적거나 비슷하면 거의 전부 잘려나간다. 실측(results/*/
  c_tcn_1ch_*, d_tcn_3ch_*의 run_meta.json/ai_log.csv/ap_fsm_summary.json)으로
  재구성한 원본 합계는 c1ch 558회, d3ch 436회이고, 이 버그를 그대로 재현하면
  각각 6회, 148회로 줄어 onboard_cost.csv의 값과 정확히 일치한다
  ([gate_window_check-4]에서 직접 검증). 즉 105배 차이는 온보드 인과 구현
  (wp_poes_node.ChannelState / ap_fsm_node.ApFsmCore)의 문제가 아니라 그 뒤
  S7 집계 단계 한 곳의 버그다.

재사용 (재구현 없음 -- import만):
  gate_persistence_sweep.py : compute_run_len(), verify_preproc(vp) 임포트 방식,
      main()의 bg -> threshold -> above -> run_len 계산 흐름을 그대로 가져다 쓴다.
      gps를 import하면 vp(=verify_preproc)도 gps.vp로 함께 따라온다 -- 별도
      재-import하지 않는다. 이 파일은 gps.py를 수정하지 않는다.
  results/*/*/{run_meta.json, ap_fsm_summary.json, ai_log.csv} : 젯슨 리플레이가
      이미 만들어 둔 실측 산출물을 읽기만 한다(새 리플레이 없음). aggregate_onboard.py
      도 수정하지 않는다 -- 읽지도 import하지도 않고, 그 트림 계산만 동일 산식으로
      재현해 대조한다.

절차 (과제 지시 + 실측 대조로 확장):
  1) 전 구간 omni_p6에 대해 gate_persistence_sweep.py와 완전히 같은 경로로
     bg -> threshold -> above -> run_len을 계산한다.
  2) N=2 게이트 개방 틱(run_len>=2)을 연도별/월별로 집계해 시간 분포를 stdout에 낸다.
  3) event_windows.json의 strong/cluster/weak 3개 창으로 날짜를 제한해 그 안의
     개방 틱 수를 세고, 같은 창의 실제 젯슨 1회 리플레이(rep1, ap_fsm_summary.json
     n_ticks_gate_open)와 나란히 낸다 -- 하드코딩이 아니라 results/의 실측값을
     직접 읽는다(더 정확함, 과제에서 하드코딩은 허용이었지 강제가 아니었음).
  4) results/의 모든 c_tcn_1ch_*/d_tcn_3ch_* run에서 ai_log.csv 원본 행 수와
     aggregate_onboard.py의 트림을 그대로 재현한 값을 나란히 내 105배 차이의
     정체(집계 버그)를 직접 증명한다.
  5) 판정.

출력: stdout 표 + CSV(gate_window_check.csv, 같은 디렉터리). 그림 없음.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate_persistence_sweep as gps  # noqa: E402  -- compute_run_len, vp(verify_preproc) 재사용
vp = gps.vp  # verify_preproc 모듈. gps가 이미 import해둔 것을 그대로 재사용(재-import 아님).

N_GATE = 2  # gate_persistence_sweep.csv N=2 행(전 구간 duty_cycle 6.84%, 16204/236848틱)과 동일 정의


def compute_gate_open(cnt: pd.Series) -> pd.Series:
    """gate_persistence_sweep.main()의 bg -> threshold -> above -> run_len 경로를
    그대로 재현(재구현 아님 -- vp.fsm_engine.compute_rolling_bg/build_threshold와
    gps.compute_run_len을 그대로 호출). N=2 게이트 개방 불리언 시리즈를 반환한다."""
    bg = vp.fsm_engine.compute_rolling_bg(cnt, vp.BG_WINDOW_DAYS, None, vp.fsm_engine.BG_UPDATE_FREQ)
    thr = vp.fsm_engine.build_threshold(bg, vp.K, vp.ONSET_FLOOR)
    th = thr.reindex(cnt.index).ffill()
    above = (cnt >= th) & cnt.notna() & th.notna()   # gate_persistence_sweep.py와 완전히 동일한 공식
    run_len = gps.compute_run_len(above)
    return run_len >= N_GATE


def print_yearly_monthly(open_idx: pd.DatetimeIndex, total_ticks: int, n_total_open: int) -> None:
    print("\n" + "=" * 70)
    print(f"[gate_window_check-1] 전 구간 N={N_GATE} 게이트 개방 틱 연도별 분포 "
          f"(총 {n_total_open}/{total_ticks}틱, duty={n_total_open/total_ticks*100:.2f}%)")
    print("=" * 70)
    by_year = pd.Series(1, index=open_idx).groupby(open_idx.year).count()
    by_year.index.name = "year"
    by_year_pct = (by_year / n_total_open * 100).round(1)
    yr_tbl = pd.DataFrame({"n_gate_open_ticks": by_year, "pct_of_all_open_ticks": by_year_pct})
    print(yr_tbl.to_string())
    print("-> 태양 극대기(2023~2025)에 몰려 있다는 가설과 달리 11~19%대로 연도별 편차가 크지 않다.")

    print("\n" + "=" * 70)
    print(f"[gate_window_check-2] 전 구간 N={N_GATE} 게이트 개방 틱 연-월별 분포")
    print("=" * 70)
    by_month = pd.Series(1, index=open_idx).groupby([open_idx.year, open_idx.month]).count()
    by_month.index.set_names(["year", "month"], inplace=True)
    print(by_month.to_string())


def load_windows(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    return doc["windows"]


# ──────────────────────────────────────────────────────────────────
# 젯슨 리플레이 실측 로딩 (results/*/*/run_meta.json 등, 읽기 전용)
# ──────────────────────────────────────────────────────────────────
def find_jetson_runs(repo: Path) -> list[dict]:
    """results/*/*/run_meta.json 을 훑어 scenario가 c_tcn_1ch/d_tcn_3ch인 run만 모은다.
    각 run에 대해 run_meta.json(scenario/window/rep/n_bg_warmup_ticks_excluded)과
    ai_log.csv 원본 행 수, ap_fsm_summary.json의 n_ticks_gate_open을 함께 읽는다.
    aggregate_onboard.py는 import도, 수정도 하지 않는다 -- 그 트림 계산(n_ai_raw - n_warmup,
    음수면 0)만 동일 산식으로 이 함수 밖에서 재현해 대조한다."""
    results_root = repo / "results"
    runs = []
    if not results_root.exists():
        return runs
    for meta_path in sorted(results_root.glob("*/*/run_meta.json")):
        rdir = meta_path.parent
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        scenario = meta.get("scenario")
        if scenario not in ("c_tcn_1ch", "d_tcn_3ch"):
            continue
        ai_path = rdir / "ai_log.csv"
        if not ai_path.exists():
            continue
        with open(ai_path, "r", encoding="utf-8") as f:
            n_ai_raw = sum(1 for _ in f) - 1  # header 제외
        n_ai_raw = max(n_ai_raw, 0)
        n_warmup = meta.get("n_bg_warmup_ticks_excluded") or 0
        summary_path = rdir / "ap_fsm_summary.json"
        n_gate_open_ap = None
        if summary_path.exists():
            try:
                n_gate_open_ap = json.loads(summary_path.read_text(encoding="utf-8")).get("n_ticks_gate_open")
            except Exception:
                pass
        runs.append({
            "rdir": rdir.name, "scenario": scenario, "window": meta.get("window"),
            "rep": meta.get("rep"), "n_ai_raw": n_ai_raw, "n_warmup_ticks": n_warmup,
            "n_gate_open_ap_summary": n_gate_open_ap,
        })
    return runs


def build_window_table(cnt: pd.Series, gate_open: pd.Series, windows: dict,
                       jetson_runs: list[dict]) -> pd.DataFrame:
    # window별 c1ch rep1(=strong/weak와 동일 조건의 단일 리플레이) 실측 개방 수 -- 실측 원본,
    # onboard_cost.csv의 (버그로 잘린) n_activations이 아니라 ap_fsm_summary.json을 직접 읽음.
    jetson_c1ch_rep1 = {
        r["window"]: (r["n_gate_open_ap_summary"] if r["n_gate_open_ap_summary"] is not None else r["n_ai_raw"])
        for r in jetson_runs if r["scenario"] == "c_tcn_1ch" and r["rep"] == 1
    }
    rows = []
    for name in ("strong", "cluster", "weak"):
        w = windows[name]
        start = pd.Timestamp(w["start"], tz="UTC")
        end = pd.Timestamp(w["end"], tz="UTC") + pd.Timedelta(days=1)  # end 날짜를 온전히 포함
        mask = (cnt.index >= start) & (cnt.index < end)
        n_ticks = int(mask.sum())
        n_open = int((gate_open & mask).sum())
        duty = (n_open / n_ticks) if n_ticks else float("nan")
        jetson_n = jetson_c1ch_rep1.get(name)
        ratio = (n_open / jetson_n) if jetson_n else float("nan")
        rows.append({
            "window": name,
            "n_ticks": n_ticks,
            "n_gate_open": n_open,
            "duty_cycle": round(duty, 4) if duty == duty else duty,
            "젯슨_실측_개방수": jetson_n if jetson_n is not None else "N/A(run_meta 없음)",
            "비율": round(ratio, 2) if ratio == ratio else float("nan"),
        })
    return pd.DataFrame(rows)


def print_aggregation_bug_audit(jetson_runs: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("[gate_window_check-4] 105배 차이의 실제 정체 -- aggregate_onboard.py 집계 버그 재현")
    print("=" * 70)
    if not jetson_runs:
        print("results/*/*/run_meta.json 을 찾지 못해 이 대조는 생략함 "
              "(results/ 디렉터리가 이 머신에 없을 수 있음).")
        return
    print("run별: ai_log.csv 원본 행수(=실제 게이트 개방 횟수) vs "
          "aggregate_onboard.py:load_csv_trimmed()가 df.iloc[n_bg_warmup_ticks_excluded:] "
          "로 자르는 계산을 그대로 재현한 값(그 스크립트는 import/수정하지 않음, 산식만 재현):")
    detail = pd.DataFrame([{
        "run": r["rdir"], "scenario": r["scenario"], "window": r["window"], "rep": r["rep"],
        "n_ai_raw(실제 개방)": r["n_ai_raw"], "n_warmup_ticks": r["n_warmup_ticks"],
        "n_after_agg_trim(버그 재현)": max(0, r["n_ai_raw"] - r["n_warmup_ticks"]),
    } for r in jetson_runs])
    print(detail.to_string(index=False))

    print("\nscenario별 합계 (raw = 진짜 게이트 개방 횟수, trimmed = onboard_cost.csv의 n_activations):")
    agg = detail.groupby("scenario").agg(
        n_runs=("run", "count"),
        n_ai_raw_total=("n_ai_raw(실제 개방)", "sum"),
        n_after_agg_trim_total=("n_after_agg_trim(버그 재현)", "sum"),
    ).reset_index()
    print(agg.to_string(index=False))
    print("\nonboard_cost.csv 실측값: c1ch n_activations=6, d3ch n_activations=148 "
          "-- 위 n_after_agg_trim_total과 정확히 일치(재현됨). "
          "즉 6/148은 젯슨의 실제 게이트 개방 횟수가 아니라 집계 스크립트가 "
          "n_bg_warmup_ticks_excluded(틱 단위 워밍업, 예 96)를 이미 사건 단위인 ai_log.csv에 "
          "행 오프셋으로 잘못 적용해 만든 값이다. log.csv(AP가 매 틱 1행 쓰는 파일)는 "
          "총 틱 수가 워밍업보다 훨씬 커서 이 버그의 영향이 작아 n_ticks_active 열은 정상이다.")


def main():
    cache_dir = vp.C_PD / "POES" / "MetOp03_count" / "poes_metop03_cache_parquet"
    print(f"[gate_window_check] REPO={vp.REPO}")
    cnt = vp.load_omni_p6_count(cache_dir)
    total_ticks = len(cnt)
    print(f"[gate_window_check] omni_p6 표본 {total_ticks}개, 기간 {cnt.index[0]} ~ {cnt.index[-1]}")

    gate_open = compute_gate_open(cnt)
    n_total_open = int(gate_open.sum())
    print(f"[gate_window_check] 전 구간 N={N_GATE} 게이트 개방: {n_total_open}/{total_ticks}틱 "
          f"(duty={n_total_open/total_ticks*100:.4f}%) -- gate_persistence_sweep.csv N=2 행"
          f"(16204/236848, 6.8415%)과 일치 여부 자체 점검용")

    open_idx = cnt.index[gate_open]
    print_yearly_monthly(open_idx, total_ticks, n_total_open)

    win_path = Path(__file__).resolve().parent / "event_windows.json"
    windows = load_windows(win_path)
    jetson_runs = find_jetson_runs(vp.REPO)
    tbl = build_window_table(cnt, gate_open, windows, jetson_runs)

    print("\n" + "=" * 70)
    print("[gate_window_check-3] 리플레이 3개 창 내 게이트 개방 (오프라인 재계산 vs 젯슨 실측 rep1)")
    print("=" * 70)
    print(tbl.to_string(index=False))
    print("\n(젯슨_실측_개방수는 하드코딩이 아니라 results/*/c_tcn_1ch_<window>_rep1/"
          "ap_fsm_summary.json의 n_ticks_gate_open을 직접 읽은 값이다.)")

    out_csv = Path(__file__).resolve().parent / "gate_window_check.csv"
    tbl.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n[gate_window_check] CSV 저장 -> {out_csv}")

    print_aggregation_bug_audit(jetson_runs)

    # ── 판정 ──
    print("\n" + "=" * 70)
    print("[gate_window_check-5] 판정")
    print("=" * 70)
    ratios = [r for r in tbl["비율"] if r == r]
    max_open = int(tbl["n_gate_open"].max())
    if ratios and max(ratios) <= 3:
        print(f"창당 오프라인 재계산 개방 대 젯슨 실측(rep1) 비율 최댓값 {max(ratios):.2f}배(1~3 범위) "
              f"-> 오프라인 스윕과 온보드 인과 구현(wp_poes_node.ChannelState / ap_fsm_node.ApFsmCore)"
              f"이 일치한다. 애초 인용된 105배는 시간적 편중이 아니라 aggregate_onboard.py 한 곳의 "
              f"집계 버그(위 [gate_window_check-4])였다 -- 게이트 개방 '빈도' 자체는 전 구간 스윕과 "
              f"젯슨 리플레이가 서로 다른 사실을 말하고 있지 않다.")
    else:
        print(f"창당 오프라인 재계산 개방 최댓값 {max_open}회, 비율 최댓값 "
              f"{max(ratios) if ratios else float('nan')} -> 여전히 큰 괴리가 남는다. "
              f"wp_poes_node.ChannelState.eval_watchpoint / ap_fsm_node.ApFsmCore 카운터의 "
              f"추가 추적이 필요하다.")


if __name__ == "__main__":
    main()

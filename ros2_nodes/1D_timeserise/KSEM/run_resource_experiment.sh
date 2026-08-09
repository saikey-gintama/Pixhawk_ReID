#!/usr/bin/env bash
# ============================================================================
# run_resource_experiment.sh  —  PADO-SAT Experiment C 온보드 리소스/지연 측정
# ----------------------------------------------------------------------------
# 옵션 E: 5 윈도우 × 2 설정(W30,W7) × 1 rep  +  W3 × 2 설정 × 2 추가 rep = 14 run
#   - latency(wp_eval/dds_transport/fsm_infer): per-tick 측정이라 1 rep 로 충분
#   - power/GPU(tegrastats): W3 에 3 rep 두어 rep-to-rep 분산 추정
#
# 각 run: wp_ksem_node(WP) → ap_fsm_node(AP, dry-run) → sc_offboard_node(SC, passive)
#   분리 배속: warm-up x7200 (배경 버퍼 채우기) → 첫 이벤트 후 active x1800 (측정)
#
# 측정 불변식: pub_ts/wp_eval_ms/action_ms 등 측정값은 판정 로직에 미사용.
#              배속 전환도 타이머 주기만 바꿈(판정·측정 입력 아님).
# ============================================================================
set -euo pipefail

# ── 경로 (git pull 후 동일 가정) ──
REPO="${REPO:-$HOME/jeongin/Pixhawk_ReID}"
NODE_DIR="${NODE_DIR:-$REPO/ros2_nodes/1D_timeserise}"
DATA="${DATA:-$REPO/Experiment_window/C_KSEM/KSEM_count/ksem_cache_parquet}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO/results}"

# ── 배속 ──
WARMUP_SPEED=7200      # warm-up 구간 (측정 불필요)
ACTIVE_SPEED=1800      # active 구간 (측정 대상). 더 빨리: 3600 / 더 정밀: 900
K=10
PERSIST=3

# ── 이벤트(시간구간) 정의: 라벨  START(=첫이벤트-30d)  EVENT(첫이벤트)  END ──
#   (begin_time 기준. EVENT 이후 active 배속으로 전환)
#   각 원소 = replay 할 SEP 이벤트 구간. (오프라인 fsm 의 "event" 와 용어 일치)
EVENTS=(
  "E1_2021-05  2021-04-29  2021-05-29  2021-06-14"   # 15 pfu, 저배경(상승기)
  "E2_2022-03  2022-02-26  2022-03-28  2022-04-19"   # 19,11,32 pfu 클러스터
  "E3_2023-07  2023-06-16  2023-07-16  2023-08-09"   # 18,620,154.. 강+혼합 (rep 이벤트)
  "E4_2024-03  2024-02-22  2024-03-23  2024-04-09"   # 956 pfu 단일 강
  "E5_2024-10  2024-09-09  2024-10-09  2024-10-31"   # 1810,364 pfu, 태양극대 고배경
)
WINDOWS=(30 7)          # BG_WINDOW_DAYS (배경 슬라이딩 윈도우 = 오프라인 fsm 의 --window)
REP_EVENT="E3_2023-07"  # 추가 rep 을 둘 이벤트
REP_EXTRA=2             # 추가 rep 수 (기본 1 + 2 = 총 3 rep)

# ── 스모크/부분 실행 필터 (환경변수로 제어) ──
#   ONLY_EVENT : 라벨 접두어로 한 이벤트만. 예) ONLY_EVENT=E3  → E3_2023-07 만
#   ONLY_WINDOW: 한 배경윈도우만. 예) ONLY_WINDOW=7  → W7(7일) 만
#   둘 다 비우면 전체 14 run.
#   필터가 걸리면 추가 rep 도 끄고 각 (이벤트×윈도우) 1 run 만 (스모크용).
#   사용 예 (E3/W7 스모크):  ONLY_EVENT=E3 ONLY_WINDOW=7 bash run_resource_experiment.sh
ONLY_EVENT="${ONLY_EVENT:-}"
ONLY_WINDOW="${ONLY_WINDOW:-}"

# ── 한 run 실행 함수 ──
# args: label start event end window_days rep
run_one() {
  local label="$1" start="$2" event="$3" end="$4" win="$5" rep="$6"
  local tag="${label}_W${win}_rep${rep}"
  local rdir="$RESULTS_ROOT/$(date +%Y%m%d_%H%M%S)_${tag}"
  mkdir -p "$rdir"
  export RESULT_DIR="$rdir"

  echo "════════════════════════════════════════════════════════════"
  echo "  RUN: $tag"
  echo "  span=$start..$end  event=$event  BG_WINDOW=${win}d"
  echo "  RESULT_DIR=$rdir"
  echo "════════════════════════════════════════════════════════════"

  # offboard(SC, passive) — alert 수신 대기. 먼저 띄움.
  python3 "$NODE_DIR/sc_offboard_node.py" --passive &
  local pid_off=$!
  sleep 2

  # fsm(AP, dry-run) — provenance 로 window/k 전달
  python3 "$NODE_DIR/ap_fsm_node.py" --window "$win" --k "$K" --persistence "$PERSIST" --dry-run &
  local pid_fsm=$!
  sleep 2

  # ksem(WP) — 마지막. 완료 시 RESULT_DIR/REPLAY_DONE 마커 쓰고 자체 shutdown.
  python3 "$NODE_DIR/wp_ksem_node.py" \
      --data "$DATA" \
      --window "$win" --k "$K" \
      --start "$start" --end "$end" --event-time "$event" \
      --warmup-speed "$WARMUP_SPEED" --replay-speed "$ACTIVE_SPEED" \
      --onset-floor 0.5 \
      &
  local pid_ksem=$!

  # 완료 마커 폴링 (ksem 이 replay 끝나면 REPLAY_DONE 생성 후 종료)
  local waited=0
  while [[ ! -f "$rdir/REPLAY_DONE" ]]; do
    if ! kill -0 "$pid_ksem" 2>/dev/null; then
      echo "  !! ksem exited without marker — check logs in $rdir" >&2
      break
    fi
    sleep 5
    waited=$((waited+5))
  done
  # active 구간 잔여 메시지가 offboard/fsm 로그에 flush 될 시간
  sleep 3

  echo "  -- run done, shutting down nodes --"
  kill -INT "$pid_ksem" "$pid_fsm" "$pid_off" 2>/dev/null || true
  sleep 3
  kill -KILL "$pid_ksem" "$pid_fsm" "$pid_off" 2>/dev/null || true
  wait 2>/dev/null || true
  sleep 2
  echo "  -- $tag complete: $rdir --"
  echo
}

# ── 메인 루프 ──
echo "node_dir=$NODE_DIR"
echo "data=$DATA"
echo "results=$RESULTS_ROOT"
echo "windows(BG_days)=W${WINDOWS[*]}  warmup=x$WARMUP_SPEED active=x$ACTIVE_SPEED"
echo

# 필터 상태 표시
if [[ -n "$ONLY_EVENT" || -n "$ONLY_WINDOW" ]]; then
  echo "FILTER: ONLY_EVENT='${ONLY_EVENT:-(all)}'  ONLY_WINDOW='${ONLY_WINDOW:-(all)}'  (추가 rep 비활성 — 스모크 모드)"
  echo
fi

# 기본 14 run: 각 이벤트 × 2 윈도우 × rep1
for win in "${WINDOWS[@]}"; do
  # 배경윈도우 필터
  if [[ -n "$ONLY_WINDOW" && "$win" != "$ONLY_WINDOW" ]]; then
    continue
  fi
  for ev in "${EVENTS[@]}"; do
    read -r label start event end <<< "$ev"
    # 이벤트 필터 (접두어 매칭: E3 → E3_2023-07)
    if [[ -n "$ONLY_EVENT" && "$label" != "$ONLY_EVENT"* ]]; then
      continue
    fi
    run_one "$label" "$start" "$event" "$end" "$win" 1
    # E3 추가 rep — 단, 필터(스모크) 중엔 건너뜀
    if [[ -z "$ONLY_EVENT" && -z "$ONLY_WINDOW" && "$label" == "$REP_EVENT" ]]; then
      for ((r=2; r<=REP_EXTRA+1; r++)); do
        run_one "$label" "$start" "$event" "$end" "$win" "$r"
      done
    fi
  done
done

echo "════════════════════════════════════════════════════════════"
echo "  ALL RUNS COMPLETE. 집계: python3 $NODE_DIR/aggregate_resource.py $RESULTS_ROOT"
echo "════════════════════════════════════════════════════════════"
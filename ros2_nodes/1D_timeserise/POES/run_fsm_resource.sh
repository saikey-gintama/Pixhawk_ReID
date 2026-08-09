#!/usr/bin/env bash
# ============================================================================
# run_fsm_resource.sh -- S5 시나리오 (b) FSM: wp + ap
#
# 3구간(strong/weak/cluster) x 1rep + REP_WINDOW 에서 3rep 추가(rep-to-rep 분산
# 추정) = 6 run. 추가로 계측 오버헤드 측정을 위해 REP_WINDOW 를 --no-log 로 1회
# 더 돈다(그 차이가 계측 비용) -- 표에 한 줄로 낼 것.
#
# S5.1 패치: active 배속 1800 -> 7200(_run_common.sh 기본값). 이 스크립트가
# 그 타당성을 직접 대조한다 -- seg_strong 을 1800x/7200x 로 각 1회씩 별도로
# 더 돌려(위 6 run 과 무관한 전용 run) M1/M2/M3/M5 median 을 비교, speed_sanity.csv
# 로 낸다(논문 4.3절 인용 대상). 이 2 run 은 smoke 에서는 생략(빠른 확인용이 아님).
#
# 기동 순서: 하류 먼저(ap -> wp). wp 가 REPLAY_DONE 쓰면 폴링 종료 -> SIGINT ->
# 3초 -> SIGKILL.
#
# 사용:
#   bash run_fsm_resource.sh
#   bash run_fsm_resource.sh --smoke        # strong 구간만 1회, speed-sanity 생략
#   bash run_fsm_resource.sh --dry-run
#   ONLY_WINDOW=weak bash run_fsm_resource.sh   # 부분 실행
# ============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_run_common.sh"

WINDOWS=(strong weak cluster)
REP_WINDOW="${REP_WINDOW:-cluster}"   # 추가 rep 을 둘 구간(여러 이벤트가 몰려 가장 까다로움)
REP_EXTRA=3                            # md 지시: "3회 더 반복" = 추가 3회(기본 1 + 추가 3 = 총 4)
ONLY_WINDOW="${ONLY_WINDOW:-}"
SPEED_SANITY_WINDOW="${SPEED_SANITY_WINDOW:-strong}"
DO_SPEED_SANITY=1

for arg in "$@"; do
  case "$arg" in
    --smoke)   SMOKE=1; ONLY_WINDOW="strong"; REP_EXTRA=0; DO_SPEED_SANITY=0 ;;
    --dry-run) DRY_RUN=1 ;;
    *) echo "unknown arg: $arg" >&2; exit 1 ;;
  esac
done

ensure_event_windows

# run_one <window> <rep> [no_log_flag] [active_speed_override]
run_one() {
  local window="$1" rep="$2" no_log_flag="${3:-}" speed="${4:-$ACTIVE_SPEED}"
  local tag="b_fsm_${window}_rep${rep}"
  [[ -n "$no_log_flag" ]] && tag="${tag}_nolog"
  local rdir
  rdir="$(new_result_dir "$tag")"
  export RESULT_DIR="$rdir"
  LAST_RDIR="$rdir"   # 호출부(speed-sanity)가 읽어갈 수 있게 전역에 남김

  local start end event
  if [[ "$DRY_RUN" == "1" && ! -f "$EVENT_WINDOWS_JSON" ]]; then
    start="<dry-run:no-event-windows-json-yet>"; end="$start"; event="$start"
  else
    start="$(get_window_field "$window" start)"
    end="$(get_window_field "$window" end)"
    event="$(get_window_field "$window" onset_time)"
  fi

  echo "────────────────────────────────────────────────────────────"
  echo "  RUN: $tag  span=$start..$end  event=$event  active_speed=${speed}x  RESULT_DIR=$rdir"
  echo "────────────────────────────────────────────────────────────"

  start_tegrastats "$rdir"

  run_cmd "AP" python3 "$NODE_DIR/ap_fsm_node.py" --dry-run $no_log_flag
  local pid_ap="$LAST_PID"
  sleep 2

  run_cmd "WP" python3 "$NODE_DIR/wp_poes_node.py" \
    --data "$DATA" --start "$start" --end "$end" --event-time "$event" \
    --warmup-speed "$WARMUP_SPEED" --replay-speed "$speed"
  local pid_wp="$LAST_PID"

  poll_replay_done "$rdir" "$pid_wp"
  shutdown_nodes "$pid_wp" "$pid_ap"
  stop_tegrastats "$rdir"

  local instrumented="true"
  [[ -n "$no_log_flag" ]] && instrumented="false"
  write_run_meta "$rdir" "b_fsm" --window "$window" --rep "$rep" \
    --warmup-speed "$WARMUP_SPEED" --active-speed "$speed" \
    --instrumented "$instrumented" \
    --tegra-interval-ms "$TEGRA_INTERVAL_MS" --tegra-prestart-sec "$TEGRA_PRESTART_SEC"

  echo "  -- $tag complete: $rdir --"
  echo
}

for w in "${WINDOWS[@]}"; do
  if [[ -n "$ONLY_WINDOW" && "$w" != "$ONLY_WINDOW" ]]; then
    continue
  fi
  run_one "$w" 1
  if [[ -z "$ONLY_WINDOW" && "$w" == "$REP_WINDOW" ]]; then
    for ((r = 2; r <= REP_EXTRA + 1; r++)); do
      run_one "$w" "$r"
    done
    # 계측 오버헤드 A/B: 같은 구간을 --no-log 로 한 번 더 -- 그 차이가 계측 비용.
    run_one "$w" "nolog1" "--no-log"
  fi
done

# ── 배속 타당성 대조 (필수, 위 6 run 과 별개) ──
if [[ "$DO_SPEED_SANITY" == "1" ]]; then
  echo "════════════════════════════════════════════════════════════"
  echo "  SPEED SANITY: b_fsm / $SPEED_SANITY_WINDOW  1800x vs 7200x"
  echo "════════════════════════════════════════════════════════════"
  run_one "$SPEED_SANITY_WINDOW" "speedsanity1800x" "" 1800
  RDIR_1800="$LAST_RDIR"
  run_one "$SPEED_SANITY_WINDOW" "speedsanity7200x" "" 7200
  RDIR_7200="$LAST_RDIR"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  [dry-run] python3 $NODE_DIR/_speed_sanity_compare.py $RDIR_1800 $RDIR_7200"
  else
    python3 "$NODE_DIR/_speed_sanity_compare.py" "$RDIR_1800" "$RDIR_7200"
  fi
fi

echo "════════════════════════════════════════════════════════════"
echo "  b_fsm COMPLETE."
echo "════════════════════════════════════════════════════════════"

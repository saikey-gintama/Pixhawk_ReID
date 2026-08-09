#!/usr/bin/env bash
# ============================================================================
# run_idle_resource.sh -- S5 시나리오 (a) idle / (a') idle+ROS
#
# (a)  idle       : 노드 0개. tegrastats 만 지정 시간(기본 10분) 기동.
# (a') idle+ROS   : sc/ai/ap/wp(--idle) 전부 기동하되 리플레이는 시작하지 않음.
#                    (b)-(a') 가 "알고리즘 자체의 증분"이고, (b)-(a) 는 거기에
#                    파이썬/rclpy/DDS 기동 비용까지 얹은 값이다. 두 차분을 다 봐야
#                    4.3절 해석이 된다.
#
# --with-ros 로 (a')를 선택. 기본은 (a). (a)/(a') 는 "구간" 개념이 없어 각 3회
# 반복(스모크면 1회). REPLAY_DONE 마커가 없으므로(리플레이가 없음) 고정 시간
# sleep 후 러너가 직접 종료시킨다.
#
# 사용:
#   bash run_idle_resource.sh                 # (a) 3회, 10분씩
#   bash run_idle_resource.sh --with-ros       # (a') 3회
#   bash run_idle_resource.sh --smoke          # (a) 1회, 짧게
#   bash run_idle_resource.sh --dry-run        # 노드 기동 없이 경로/인자만 출력
# ============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_run_common.sh"

WITH_ROS=0
DURATION="${DURATION:-600}"     # (a)/(a') 기동 시간(초). 기본 10분.
REPS=3

for arg in "$@"; do
  case "$arg" in
    --with-ros) WITH_ROS=1 ;;
    --smoke)    SMOKE=1; REPS=1; DURATION="${DURATION_SMOKE:-20}" ;;
    --dry-run)  DRY_RUN=1 ;;
    *) echo "unknown arg: $arg" >&2; exit 1 ;;
  esac
done

if [[ "$WITH_ROS" == "1" ]]; then
  SCENARIO="a_prime_idle_ros"
else
  SCENARIO="a_idle"
fi

echo "════════════════════════════════════════════════════════════"
echo "  SCENARIO: $SCENARIO  (with_ros=$WITH_ROS duration=${DURATION}s reps=$REPS smoke=$SMOKE dry_run=$DRY_RUN)"
echo "════════════════════════════════════════════════════════════"
print_power_warning

run_one() {
  local rep="$1"
  local tag="${SCENARIO}_rep${rep}"
  local rdir
  rdir="$(new_result_dir "$tag")"
  export RESULT_DIR="$rdir"

  echo "────────────────────────────────────────────────────────────"
  echo "  RUN: $tag  RESULT_DIR=$rdir"
  echo "────────────────────────────────────────────────────────────"

  start_tegrastats "$rdir"

  local pids=()
  if [[ "$WITH_ROS" == "1" ]]; then
    # 기동 순서: 하류 먼저(sc -> ai -> ap -> wp), wp 는 --idle(리플레이 안 함).
    run_cmd "SC" python3 "$NODE_DIR/sc_offboard_node.py" --passive
    pids+=("$LAST_PID"); sleep 2
    run_cmd "AI" python3 "$NODE_DIR/ai_tcn_node.py" --folds-mode single --fold 0
    pids+=("$LAST_PID"); sleep 2
    run_cmd "AP" python3 "$NODE_DIR/ap_fsm_node.py" --dry-run
    pids+=("$LAST_PID"); sleep 2
    run_cmd "WP" python3 "$NODE_DIR/wp_poes_node.py" --data "$DATA" --idle
    pids+=("$LAST_PID")
  fi

  echo "  대기 ${DURATION}s (idle 측정 구간) ..."
  if [[ "$DRY_RUN" != "1" ]]; then
    sleep "$DURATION"
  fi

  if [[ "$WITH_ROS" == "1" ]]; then
    shutdown_nodes "${pids[@]}"
  fi
  stop_tegrastats "$rdir"

  write_run_meta "$rdir" "$SCENARIO" --window none --rep "$rep" \
    --warmup-speed 0 --active-speed 0 --instrumented false \
    --tegra-interval-ms "$TEGRA_INTERVAL_MS" --tegra-prestart-sec "$TEGRA_PRESTART_SEC" \
    --extra-json "{\"with_ros\": $( [[ $WITH_ROS == 1 ]] && echo true || echo false ), \"duration_sec\": $DURATION}"

  echo "  -- $tag complete: $rdir --"
  echo
}

for ((r = 1; r <= REPS; r++)); do
  run_one "$r"
done

echo "════════════════════════════════════════════════════════════"
echo "  $SCENARIO COMPLETE ($REPS run(s))."
echo "════════════════════════════════════════════════════════════"

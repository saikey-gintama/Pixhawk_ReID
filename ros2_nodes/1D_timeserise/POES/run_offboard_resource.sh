#!/usr/bin/env bash
# ============================================================================
# run_offboard_resource.sh -- S5 시나리오 (e) end-to-end: wp + ap + ai + sc
#   --ai-gate shadow 로 돈다(md 명시). shadow 는 제어 경로가 off 와 바이트 단위로
#   동일하므로(S4 검증 ③) DeltaPower 가 TCN 비용만 분리한다. gate 모드는 비용
#   측정에 쓰지 않는다.
#
# 3구간 x 1rep = 3 run(추가 rep 없음 -- md 는 (b)(c)에만 추가 rep 지시).
#
# 기동 순서: 하류 먼저(sc -> ai -> ap -> wp).
#
# 사용:
#   bash run_offboard_resource.sh
#   bash run_offboard_resource.sh --smoke
#   bash run_offboard_resource.sh --dry-run
# ============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_run_common.sh"

WINDOWS=(strong weak cluster)
ONLY_WINDOW="${ONLY_WINDOW:-}"
CHANNELS="omni_p6"
RUN_NAME="omni_p6_binary"

for arg in "$@"; do
  case "$arg" in
    --smoke)   SMOKE=1; ONLY_WINDOW="strong" ;;
    --dry-run) DRY_RUN=1 ;;
    *) echo "unknown arg: $arg" >&2; exit 1 ;;
  esac
done

ensure_event_windows
print_power_warning

run_one() {
  local window="$1" rep="$2"
  local tag="e_offboard_${window}_rep${rep}"
  local rdir
  rdir="$(new_result_dir "$tag")"
  export RESULT_DIR="$rdir"

  local start end event
  if [[ "$DRY_RUN" == "1" && ! -f "$EVENT_WINDOWS_JSON" ]]; then
    start="<dry-run:no-event-windows-json-yet>"; end="$start"; event="$start"
  else
    start="$(get_window_field "$window" start)"
    end="$(get_window_field "$window" end)"
    event="$(get_window_field "$window" onset_time)"
  fi

  echo "────────────────────────────────────────────────────────────"
  echo "  RUN: $tag  span=$start..$end  event=$event  RESULT_DIR=$rdir"
  echo "────────────────────────────────────────────────────────────"

  start_tegrastats "$rdir"

  run_cmd "SC" python3 "$NODE_DIR/sc_offboard_node.py" --passive --ai-gate shadow
  local pid_sc="$LAST_PID"
  sleep 2

  run_cmd "AI" python3 "$NODE_DIR/ai_tcn_node.py" \
    --channels "$CHANNELS" --ckpt-root "$CKPT_ROOT" --run-name "$RUN_NAME" \
    --folds-mode single --fold 0
  local pid_ai="$LAST_PID"
  sleep 2

  run_cmd "AP" python3 "$NODE_DIR/ap_fsm_node.py" --channels "$CHANNELS" --dry-run
  local pid_ap="$LAST_PID"
  sleep 2

  run_cmd "WP" python3 "$NODE_DIR/wp_poes_node.py" \
    --channels "$CHANNELS" --data "$DATA" --start "$start" --end "$end" --event-time "$event" \
    --warmup-speed "$WARMUP_SPEED" --replay-speed "$ACTIVE_SPEED"
  local pid_wp="$LAST_PID"

  poll_replay_done "$rdir" "$pid_wp"
  shutdown_nodes "$pid_wp" "$pid_ap" "$pid_ai" "$pid_sc"
  stop_tegrastats "$rdir"

  write_run_meta "$rdir" "e_offboard" --window "$window" --rep "$rep" \
    --warmup-speed "$WARMUP_SPEED" --active-speed "$ACTIVE_SPEED" --instrumented true \
    --ai-gate shadow --passive true \
    --tegra-interval-ms "$TEGRA_INTERVAL_MS" --tegra-prestart-sec "$TEGRA_PRESTART_SEC"

  echo "  -- $tag complete: $rdir --"
  echo
}

for w in "${WINDOWS[@]}"; do
  if [[ -n "$ONLY_WINDOW" && "$w" != "$ONLY_WINDOW" ]]; then
    continue
  fi
  run_one "$w" 1
done

echo "════════════════════════════════════════════════════════════"
echo "  e_offboard COMPLETE."
echo "════════════════════════════════════════════════════════════"

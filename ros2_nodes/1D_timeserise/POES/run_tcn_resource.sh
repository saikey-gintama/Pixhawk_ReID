#!/usr/bin/env bash
# ============================================================================
# run_tcn_resource.sh -- S5 시나리오 (c) FSM+TCN 1채널 / (d) FSM+TCN 3채널
#   wp + ap + ai.  --ai-gate 없음(SC 미기동, sc_offboard_node 는 이 시나리오에
#   참여하지 않는다 -- md 명시).
#
# 채널 수를 첫 인자로 받는다: 1 -> omni_p6_binary,  3 -> omni_p6_pro_tel0_p5_pro_tel90_p5_binary
# (c)(1채널)만 REP_WINDOW 에서 3rep 추가(rep-to-rep 분산). (d)는 추가 rep 없음(md 명시).
#
# 기동 순서: 하류 먼저(ai -> ap -> wp).
#
# 사용:
#   bash run_tcn_resource.sh 1
#   bash run_tcn_resource.sh 3
#   bash run_tcn_resource.sh 1 --smoke
#   bash run_tcn_resource.sh 1 --dry-run
# ============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_run_common.sh"

N_CH="${1:-}"
shift || true
if [[ "$N_CH" != "1" && "$N_CH" != "3" ]]; then
  echo "usage: $0 {1|3} [--smoke] [--dry-run]" >&2
  exit 1
fi

WINDOWS=(strong weak cluster)
REP_WINDOW="${REP_WINDOW:-cluster}"
REP_EXTRA=3   # md 지시: "3회 더 반복" = 추가 3회(기본 1 + 추가 3 = 총 4), (c)만 해당
ONLY_WINDOW="${ONLY_WINDOW:-}"
GATE_N="${GATE_N:-2}"   # 게이트 지속성. 기본 2 = 기존 실행과 동일.

for arg in "$@"; do
  case "$arg" in
    --smoke)   SMOKE=1; ONLY_WINDOW="strong"; REP_EXTRA=0 ;;
    --dry-run) DRY_RUN=1 ;;
    *) echo "unknown arg: $arg" >&2; exit 1 ;;
  esac
done

if [[ "$N_CH" == "1" ]]; then
  SCENARIO="c_tcn_1ch"
  CHANNELS="omni_p6"
  RUN_NAME="omni_p6_binary"
  ALLOW_EXTRA_REP=1
else
  SCENARIO="d_tcn_3ch"
  CHANNELS="omni_p6,pro_tel0_p5,pro_tel90_p5"
  RUN_NAME="omni_p6_pro_tel0_p5_pro_tel90_p5_binary"
  ALLOW_EXTRA_REP=0
fi

ensure_event_windows
print_power_warning

run_one() {
  local window="$1" rep="$2"
  local tag="${SCENARIO}_${window}_rep${rep}"
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
  echo "  RUN: $tag  channels=$CHANNELS  span=$start..$end  event=$event  RESULT_DIR=$rdir"
  echo "────────────────────────────────────────────────────────────"

  start_tegrastats "$rdir"

  run_cmd "AI" python3 "$NODE_DIR/ai_tcn_node.py" \
    --channels "$CHANNELS" --ckpt-root "$CKPT_ROOT" --run-name "$RUN_NAME" \
    --folds-mode single --fold 0
  local pid_ai="$LAST_PID"
  sleep 2

  run_cmd "AP" python3 "$NODE_DIR/ap_fsm_node.py" --channels "$CHANNELS" --gate-n "$GATE_N" --dry-run
  local pid_ap="$LAST_PID"
  sleep 2

  run_cmd "WP" python3 "$NODE_DIR/wp_poes_node.py" \
    --channels "$CHANNELS" --data "$DATA" --start "$start" --end "$end" --event-time "$event" \
    --warmup-speed "$WARMUP_SPEED" --replay-speed "$ACTIVE_SPEED" --gate-n "$GATE_N"
  local pid_wp="$LAST_PID"

  poll_replay_done "$rdir" "$pid_wp"
  shutdown_nodes "$pid_wp" "$pid_ap" "$pid_ai"
  stop_tegrastats "$rdir"

  # _assemble_run_meta.py 는 parse_args()(known_args 아님)라 모르는 플래그에 죽는다 --
  # --gate-n 은 네이티브 지원이 아니므로 --extra-json 으로만 넣는다(run_offboard_resource.sh
  # 의 active_run 과 같은 관례).
  write_run_meta "$rdir" "$SCENARIO" --window "$window" --rep "$rep" \
    --warmup-speed "$WARMUP_SPEED" --active-speed "$ACTIVE_SPEED" --instrumented true \
    --n-channels-arg "$N_CH" \
    --tegra-interval-ms "$TEGRA_INTERVAL_MS" --tegra-prestart-sec "$TEGRA_PRESTART_SEC" \
    --extra-json "{\"gate_n\": $GATE_N}"

  echo "  -- $tag complete: $rdir --"
  echo
}

for w in "${WINDOWS[@]}"; do
  if [[ -n "$ONLY_WINDOW" && "$w" != "$ONLY_WINDOW" ]]; then
    continue
  fi
  run_one "$w" 1
  if [[ "$ALLOW_EXTRA_REP" == "1" && -z "$ONLY_WINDOW" && "$w" == "$REP_WINDOW" ]]; then
    for ((r = 2; r <= REP_EXTRA + 1; r++)); do
      run_one "$w" "$r"
    done
  fi
done

echo "════════════════════════════════════════════════════════════"
echo "  $SCENARIO COMPLETE."
echo "════════════════════════════════════════════════════════════"

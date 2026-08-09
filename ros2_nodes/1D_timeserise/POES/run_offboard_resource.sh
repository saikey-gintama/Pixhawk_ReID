#!/usr/bin/env bash
# ============================================================================
# run_offboard_resource.sh -- S5 시나리오 (e) end-to-end: wp + ap + ai + sc
#   기본(--active 없음): --ai-gate shadow 로 돈다(md 명시). shadow 는 제어 경로가
#   off 와 바이트 단위로 동일하므로(S4 검증 ③) DeltaPower 가 TCN 비용만 분리한다.
#   gate 모드는 이 기본 경로의 비용 측정에 쓰지 않는다.
#
# 3구간 x 1rep = 3 run(추가 rep 없음 -- md 는 (b)(c)에만 추가 rep 지시).
#
# 기동 순서: 하류 먼저(sc -> ai -> ap -> wp).
#
# --active: 실기 오프보드 실증(ARM/Offboard 전환 + DISARM, SC 재발사 가드 작업).
#   --active 없이 실행한 동작(노드 인자/배속/구간/run_meta)은 기존과 바이트 단위로
#   동일하다(비용 측정 (e) 행과의 비교 가능성 보존). --active 는 별도 태그
#   (e_offboard_active_*)로 결과 폴더를 분리하고 별도 --extra-json 필드로 표시돼
#   집계에서 섞이지 않는다.
#   기본 창은 strong 으로 좁힌다(데이터 기반 확인: weak 창은 전 구간에서 ALERT 가
#   한 번도 안 떠 ARM 자체가 안 나온다 -- ONLY_WINDOW 로 다른 창을 명시하면 존중).
#   기본 배속(ACTIVE_SPEED=60)/구간(ACTIVE_SPAN_HOURS=10)도 데이터 기반 권고값 --
#   환경변수로 직접 준 값은 존중한다.
#
# 사용:
#   bash run_offboard_resource.sh
#   bash run_offboard_resource.sh --smoke
#   bash run_offboard_resource.sh --dry-run
#   bash run_offboard_resource.sh --active            # 실기 ARM/DISARM 데모(안전 배너 확인 필요)
#   bash run_offboard_resource.sh --active --dry-run
# ============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ACTIVE_SPEED 를 사용자가 이미 환경변수로 줬는지 _run_common.sh 소싱(기본값 주입)
# 전에 기록해둔다 -- --active 일 때 "환경변수로 준 값은 존중" 하려면 이 구분이 필요.
_USER_SET_ACTIVE_SPEED="${ACTIVE_SPEED+1}"

source "$HERE/_run_common.sh"

WINDOWS=(strong weak cluster)
ONLY_WINDOW="${ONLY_WINDOW:-}"
CHANNELS="omni_p6"
RUN_NAME="omni_p6_binary"
ACTIVE_RUN=0
ACTIVE_SPAN_HOURS="${ACTIVE_SPAN_HOURS:-10}"   # strong 창 실측 기준(2사이클, onset+8.5h 필요)+여유

for arg in "$@"; do
  case "$arg" in
    --smoke)   SMOKE=1; ONLY_WINDOW="strong" ;;
    --dry-run) DRY_RUN=1 ;;
    --active)  ACTIVE_RUN=1 ;;
    *) echo "unknown arg: $arg" >&2; exit 1 ;;
  esac
done

if [[ "$ACTIVE_RUN" == "1" && "$SMOKE" == "1" ]]; then
  echo "ERROR: --active 와 --smoke 는 동시에 쓸 수 없습니다." >&2
  exit 1
fi

if [[ "$ACTIVE_RUN" == "1" ]]; then
  # 데이터 기반(causal FSM 전 구간 리플레이 실측): strong 은 onset+8.5h 안에 ARM->DISARM
  # 2사이클이 닫혀 가장 빠르다. weak 는 창 전체에서 ALERT 가 한 번도 안 뜬다(ARM 자체가
  # 안 남) -- 명시적으로 다른 창을 지정하지 않는 한 strong 으로 좁힌다.
  ONLY_WINDOW="${ONLY_WINDOW:-strong}"
  if [[ -z "$_USER_SET_ACTIVE_SPEED" ]]; then
    ACTIVE_SPEED=60   # 권고값(4단계 재계산, V1/V2 반영 후 확정) -- 사용자가 준 값은 위에서 이미 존중됨
  fi
fi

ensure_event_windows

print_active_safety_banner() {
  echo "############################################################################"
  echo "# ACTIVE 모드 -- 실기 오프보드 명령이 발행됩니다.                          #"
  echo "#                                                                          #"
  echo "# - Offboard 전환 + ARM 명령이 실제로 FMU 로 나갑니다.                     #"
  echo "# - 호버 셋포인트 고도: 1.5m.                                              #"
  echo "# - !! 프로펠러 장착 상태에서 실행 금지 !!                                 #"
  echo "#   착륙 시퀀스(LAND)가 없어 고도 1.5m 에서 그대로 DISARM 합니다 -- 자유낙하#"
  echo "#   합니다(D3=(a), VEHICLE_CMD_NAV_LAND 미구현. sc_offboard_node.py 의      #"
  echo "#   _publish_hover_setpoint()/_disarm() TODO 주석 참고).                   #"
  echo "# - 이벤트가 HOLD(6초) 중 재개되면 재ARM 됩니다(D2=(b), 방어적 코드경로 --  #"
  echo "#   기본 설정(S=60, hold-scale=1.0)에서는 실데이터로 도달 안 함을 확인했지만#"
  echo "#   배속/hold-scale 을 바꾸면 ARM/DISARM 채터링이 날 수 있습니다(최소 재ARM #"
  echo "#   간격 1.0초 가드 있음, D4).                                            #"
  echo "# - QGC 가 실행 중이어야 합니다.                                          #"
  echo "############################################################################"

  if compgen -G "/dev/ttyACM*" > /dev/null 2>&1; then
    echo "  /dev/ttyACM* 발견."
  else
    echo "  경고: /dev/ttyACM* 없음(이더넷 연결이면 정상일 수 있음 -- 중단하지 않음)."
  fi

  if [[ "$DRY_RUN" != "1" ]]; then
    read -r -p "type ACTIVE to continue: " _confirm
    if [[ "$_confirm" != "ACTIVE" ]]; then
      echo "확인 문자열 불일치 -- 중단." >&2
      exit 1
    fi
  fi
}

if [[ "$ACTIVE_RUN" == "1" ]]; then
  print_active_safety_banner
else
  print_power_warning
fi

# estimate_wallclock_sec <start> <event> <end> <warmup_speed> <active_speed>
# ACTIVE 데모용 예상 벽시계 소요시간(초). warmup 구간(start~event)/active 구간
# (event~end) 각각의 물리시간을 해당 배속으로 나눠 더한다.
estimate_wallclock_sec() {
  local start="$1" event="$2" end="$3" warmup_speed="$4" active_speed="$5"
  python3 -c "
import pandas as pd
def _p(s):
    t = pd.Timestamp(s)
    return t.tz_localize('UTC') if t.tzinfo is None else t
start, event, end = _p('$start'), _p('$event'), _p('$end')
warmup_span = max(0.0, (event - start).total_seconds())
active_span = max(0.0, (end - event).total_seconds())
wall = warmup_span / $warmup_speed + active_span / $active_speed
print(f'{wall:.0f}')
" 2>/dev/null
}

run_one() {
  local window="$1" rep="$2"
  local tag="e_offboard_${window}_rep${rep}"
  [[ "$ACTIVE_RUN" == "1" ]] && tag="e_offboard_active_${window}_rep${rep}"
  local rdir
  rdir="$(new_result_dir "$tag")"
  export RESULT_DIR="$rdir"

  local start end event
  if [[ "$DRY_RUN" == "1" && ! -f "$EVENT_WINDOWS_JSON" ]]; then
    start="<dry-run:no-event-windows-json-yet>"; end="$start"; event="$start"
  else
    start="$(get_window_field "$window" start)"
    event="$(get_window_field "$window" onset_time)"
    if [[ "$ACTIVE_RUN" == "1" ]]; then
      # --end 만 onset+ACTIVE_SPAN_HOURS 로 좁힌다. --start 는 그대로 둬 배경 워밍업
      # (96틱, --start 기준 상대적)을 보존한다(1단계 Q6 확인).
      end="$(python3 -c "import pandas as pd; print((pd.Timestamp('$event') + pd.Timedelta(hours=$ACTIVE_SPAN_HOURS)).isoformat())")"
    else
      end="$(get_window_field "$window" end)"
    fi
  fi

  echo "────────────────────────────────────────────────────────────"
  echo "  RUN: $tag  span=$start..$end  event=$event  RESULT_DIR=$rdir"
  if [[ "$ACTIVE_RUN" == "1" ]]; then
    local wall_sec=""
    [[ "$DRY_RUN" != "1" || -f "$EVENT_WINDOWS_JSON" ]] && \
      wall_sec="$(estimate_wallclock_sec "$start" "$event" "$end" "$WARMUP_SPEED" "$ACTIVE_SPEED")"
    echo "  warmup_speed=${WARMUP_SPEED}x  active_speed=${ACTIVE_SPEED}x  "\
"active_span_hours=${ACTIVE_SPAN_HOURS}  예상 벽시계 소요=${wall_sec:-?}초"
  fi
  echo "────────────────────────────────────────────────────────────"

  start_tegrastats "$rdir"

  local sc_args
  if [[ "$ACTIVE_RUN" == "1" ]]; then
    sc_args=(--active --ai-gate gate --ai-threshold 0.85 --hold-scale 1.0)
  else
    sc_args=(--passive --ai-gate shadow)
  fi
  run_cmd "SC" python3 "$NODE_DIR/sc_offboard_node.py" "${sc_args[@]}"
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

  if [[ "$ACTIVE_RUN" == "1" ]]; then
    # _assemble_run_meta.py 는 parse_args()(known_args 아님)라 모르는 플래그에 죽는다
    # (--active-run/--ai-threshold/--active-span-hours/--sc-guard-rev 는 지원 안 함) --
    # --extra-json 으로만 넣는다(7단계 사전확인). --ai-gate/--passive/--hold-scale 은
    # 이미 네이티브 지원이라 그대로 전달.
    write_run_meta "$rdir" "e_offboard" --window "$window" --rep "$rep" \
      --warmup-speed "$WARMUP_SPEED" --active-speed "$ACTIVE_SPEED" --instrumented true \
      --ai-gate gate --passive false --hold-scale 1.0 \
      --tegra-interval-ms "$TEGRA_INTERVAL_MS" --tegra-prestart-sec "$TEGRA_PRESTART_SEC" \
      --extra-json "{\"active_run\": true, \"ai_threshold\": 0.85, \"active_span_hours\": $ACTIVE_SPAN_HOURS, \"sc_guard_rev\": \"v1_rearm_disarm_guard\"}"
  else
    write_run_meta "$rdir" "e_offboard" --window "$window" --rep "$rep" \
      --warmup-speed "$WARMUP_SPEED" --active-speed "$ACTIVE_SPEED" --instrumented true \
      --ai-gate shadow --passive true \
      --tegra-interval-ms "$TEGRA_INTERVAL_MS" --tegra-prestart-sec "$TEGRA_PRESTART_SEC"
  fi

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

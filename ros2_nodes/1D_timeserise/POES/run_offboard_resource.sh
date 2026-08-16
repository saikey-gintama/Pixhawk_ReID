#!/usr/bin/env bash
# ============================================================================
# run_offboard_resource.sh -- S5 시나리오 (e) end-to-end: wp + ap + ai + sc
#   기본(--active/--bench/--rehearse 없음): --ai-gate shadow 로 돈다(md 명시).
#   shadow 는 제어 경로가 off 와 바이트 단위로 동일하므로(S4 검증 ③) DeltaPower 가
#   TCN 비용만 분리한다. gate 모드는 이 기본 경로의 비용 측정에 쓰지 않는다.
#
# 3구간 x 1rep = 3 run(추가 rep 없음 -- md 는 (b)(c)에만 추가 rep 지시).
#
# 기동 순서: 하류 먼저(sc -> ai -> ap -> wp).
#
# --active/--bench/--rehearse 는 서로 배타적인 별도 모드다(sc_offboard_node.py 의
# --land-before-disarm 기본 on(D3)/--setpoint-mode/--bench-thrust/--no-px4 도입 반영).
# 셋 다 없는 기본 경로(비용 측정용, 시나리오 e, 논문 4.6절 Table 10)는 노드 인자·
# 배속·구간·태그·run_meta 가 지금과 바이트 단위로 동일하게 유지된다.
#   --active   : 실기 오프보드 실증(ARM/Offboard 전환 + DISARM, position 셋포인트).
#                기본은 NAV_LAND -> 착륙확인(vehicle_land_detected)/타임아웃 뒤
#                DISARM 이다(D3 -- 더 이상 착륙 없이 자유낙하하지 않는다).
#   --bench    : 실기 벤치 시연(!! 프로펠러 반드시 탈거 !!). attitude 셋포인트를
#                써서 로컬 위치 추정 없이도(GPS/광류/VIO 없는 실내) Offboard 진입/
#                ARM 이 거부되지 않는다(실측: pre_flight_checks_pass=false 확인됨).
#                BENCH_THRUST 환경변수로 thrust 크기 조절(기본 0.1).
#   --rehearse : 드론/PX4 미연결 리허설(--no-px4). 명령은 event_log.csv 에 기록만
#                되고 실제로는 안 나간다(안전 확인 불필요). 착륙확인 콜백이 없어
#                항상 --land-timeout-sec(여기선 2초로 단축)로 DISARM 한다.
#                tegrastats 는 켜지 않는다(전력 측정 대상 아님).
#   각 모드는 별도 태그(e_offboard_active_*/e_offboard_bench_*/e_offboard_rehearse_*)
#   로 결과 폴더를 분리하고 run_meta 의 --extra-json 필드로도 구분돼 기본 경로의
#   집계와 섞이지 않는다.
#   기본 창은 strong 으로 좁힌다(데이터 기반 확인: weak 창은 전 구간에서 ALERT 가
#   한 번도 안 떠 ARM 자체가 안 나온다 -- ONLY_WINDOW 로 다른 창을 명시하면 존중).
#   기본 배속(ACTIVE_SPEED=60)/구간(ACTIVE_SPAN_HOURS=10)도 데이터 기반 권고값 --
#   환경변수로 직접 준 값은 존중한다.
#
# 사용:
#   bash run_offboard_resource.sh
#   bash run_offboard_resource.sh --smoke
#   bash run_offboard_resource.sh --dry-run
#   bash run_offboard_resource.sh --active              # 실기 ARM/DISARM 데모(안전 배너)
#   bash run_offboard_resource.sh --active --dry-run
#   bash run_offboard_resource.sh --bench               # 실기 벤치(프로펠러 탈거, 안전 배너)
#   BENCH_THRUST=0.15 bash run_offboard_resource.sh --bench
#   bash run_offboard_resource.sh --rehearse            # PX4 미연결 리허설
#   bash run_offboard_resource.sh --rehearse --dry-run
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
BENCH_RUN=0
REHEARSE_RUN=0
ACTIVE_SPAN_HOURS="${ACTIVE_SPAN_HOURS:-10}"   # strong 창 실측 기준(2사이클, onset+8.5h 필요)+여유
BENCH_THRUST="${BENCH_THRUST:-0.1}"            # --bench 전용(attitude thrust_body 크기)

for arg in "$@"; do
  case "$arg" in
    --smoke)    SMOKE=1; ONLY_WINDOW="strong" ;;
    --dry-run)  DRY_RUN=1 ;;
    --active)   ACTIVE_RUN=1 ;;
    --bench)    BENCH_RUN=1 ;;
    --rehearse) REHEARSE_RUN=1 ;;
    *) echo "unknown arg: $arg" >&2; exit 1 ;;
  esac
done

_N_MODES=$((ACTIVE_RUN + BENCH_RUN + REHEARSE_RUN))
if [[ "$_N_MODES" -gt 1 ]]; then
  echo "ERROR: --active/--bench/--rehearse 는 동시에 쓸 수 없습니다(모드 1개만 선택)." >&2
  exit 1
fi
if [[ "$_N_MODES" -ge 1 && "$SMOKE" == "1" ]]; then
  echo "ERROR: --active/--bench/--rehearse 와 --smoke 는 동시에 쓸 수 없습니다." >&2
  exit 1
fi

# 세 모드 모두 ai-gate gate 로 실제 ARM 을 트리거하므로 창/배속 데이터 기반 좁히기가
# 똑같이 적용된다(SPECIAL_MODE 로 한 번만 판정).
SPECIAL_MODE=0
[[ "$_N_MODES" -ge 1 ]] && SPECIAL_MODE=1

if [[ "$SPECIAL_MODE" == "1" ]]; then
  # 데이터 기반(causal FSM 전 구간 리플레이 실측): strong 은 onset+8.5h 안에 ARM->DISARM
  # 2사이클이 닫혀 가장 빠르다. weak 는 창 전체에서 ALERT 가 한 번도 안 뜬다(ARM 자체가
  # 안 남) -- 명시적으로 다른 창을 지정하지 않는 한 strong 으로 좁힌다.
  ONLY_WINDOW="${ONLY_WINDOW:-strong}"
  if [[ -z "$_USER_SET_ACTIVE_SPEED" ]]; then
    ACTIVE_SPEED=60   # 권고값(4단계 재계산, V1/V2 반영 후 확정) -- 사용자가 준 값은 위에서 이미 존중됨
  fi
fi

ensure_event_windows

# print_active_safety_banner <ACTIVE|BENCH> -- --active/--bench 공용. 착륙 시퀀스가
# 기본 on(D3)임을 전제로 다시 썼다 -- 더 이상 "착륙 없이 자유낙하" 가 기본이 아니다.
print_active_safety_banner() {
  local mode_label="$1"
  echo "############################################################################"
  if [[ "$mode_label" == "BENCH" ]]; then
    echo "# BENCH 모드 -- 실기 오프보드 명령이 발행됩니다(attitude 셋포인트).       #"
  else
    echo "# ACTIVE 모드 -- 실기 오프보드 명령이 발행됩니다(position 셋포인트).      #"
  fi
  echo "#                                                                          #"
  echo "# - Offboard 전환 + ARM 명령이 실제로 FMU 로 나갑니다.                     #"
  if [[ "$mode_label" == "BENCH" ]]; then
    echo "# - 셋포인트: attitude(수평 자세 + thrust_body z=-${BENCH_THRUST}).       #"
    echo "# - !! 프로펠러 반드시 탈거 -- 모터가 실제로 회전합니다 !!                #"
  else
    echo "# - 셋포인트: position(호버 고도 1.5m). 유효한 로컬 위치 추정 필요.       #"
    echo "# - !! 프로펠러 장착 상태에서 실외/구속 없이 실행 시 실제 비행함 !!       #"
  fi
  echo "# - 기본은 NAV_LAND -> 착륙확인(vehicle_land_detected) 또는                #"
  echo "#   --land-timeout-sec(기본 20s) 타임아웃 뒤 DISARM 입니다(D3).            #"
  echo "#   자유낙하 경고는 --no-land-before-disarm 을 쓸 때만 해당합니다 --       #"
  echo "#   이 스크립트는 기본값(land-before-disarm on)을 그대로 씁니다.           #"
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
    read -r -p "type $mode_label to continue: " _confirm
    if [[ "$_confirm" != "$mode_label" ]]; then
      echo "확인 문자열 불일치 -- 중단." >&2
      exit 1
    fi
  fi
}

# print_rehearse_banner -- PX4 미연결이라 하드웨어 위험이 없으므로 확인 프롬프트 없음.
print_rehearse_banner() {
  echo "############################################################################"
  echo "# REHEARSE 모드 -- PX4 미연결(--no-px4). 드론/FMU 로 명령이 나가지 않고    #"
  echo "# event_log.csv 에 기록만 됩니다(ScFsmCore 판정은 실기와 동일하게 돕니다). #"
  echo "# 착륙확인 콜백이 없어 항상 --land-timeout-sec 2s 로 강제 DISARM 합니다.   #"
  echo "# tegrastats 는 켜지 않습니다(전력 측정 대상 아님).                        #"
  echo "############################################################################"
}

if [[ "$ACTIVE_RUN" == "1" ]]; then
  print_active_safety_banner "ACTIVE"
elif [[ "$BENCH_RUN" == "1" ]]; then
  print_active_safety_banner "BENCH"
elif [[ "$REHEARSE_RUN" == "1" ]]; then
  print_rehearse_banner
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
  if [[ "$ACTIVE_RUN" == "1" ]]; then
    tag="e_offboard_active_${window}_rep${rep}"
  elif [[ "$BENCH_RUN" == "1" ]]; then
    tag="e_offboard_bench_${window}_rep${rep}"
  elif [[ "$REHEARSE_RUN" == "1" ]]; then
    tag="e_offboard_rehearse_${window}_rep${rep}"
  fi
  local rdir
  rdir="$(new_result_dir "$tag")"
  export RESULT_DIR="$rdir"

  local start end event
  if [[ "$DRY_RUN" == "1" && ! -f "$EVENT_WINDOWS_JSON" ]]; then
    start="<dry-run:no-event-windows-json-yet>"; end="$start"; event="$start"
  else
    start="$(get_window_field "$window" start)"
    event="$(get_window_field "$window" onset_time)"
    if [[ "$SPECIAL_MODE" == "1" ]]; then
      # --end 만 onset+ACTIVE_SPAN_HOURS 로 좁힌다. --start 는 그대로 둬 배경 워밍업
      # (96틱, --start 기준 상대적)을 보존한다(1단계 Q6 확인).
      end="$(python3 -c "import pandas as pd; print((pd.Timestamp('$event') + pd.Timedelta(hours=$ACTIVE_SPAN_HOURS)).isoformat())")"
    else
      end="$(get_window_field "$window" end)"
    fi
  fi

  echo "────────────────────────────────────────────────────────────"
  echo "  RUN: $tag  span=$start..$end  event=$event  RESULT_DIR=$rdir"
  if [[ "$SPECIAL_MODE" == "1" ]]; then
    local wall_sec=""
    [[ "$DRY_RUN" != "1" || -f "$EVENT_WINDOWS_JSON" ]] && \
      wall_sec="$(estimate_wallclock_sec "$start" "$event" "$end" "$WARMUP_SPEED" "$ACTIVE_SPEED")"
    echo "  warmup_speed=${WARMUP_SPEED}x  active_speed=${ACTIVE_SPEED}x  "\
"active_span_hours=${ACTIVE_SPAN_HOURS}  예상 벽시계 소요=${wall_sec:-?}초"
  fi
  echo "────────────────────────────────────────────────────────────"

  if [[ "$REHEARSE_RUN" == "1" ]]; then
    echo "  (REHEARSE) tegrastats 생략(전력 측정 대상 아님)."
  else
    start_tegrastats "$rdir"
  fi

  local sc_args
  if [[ "$BENCH_RUN" == "1" ]]; then
    sc_args=(--active --ai-gate gate --ai-threshold 0.85 --hold-scale 1.0 \
             --setpoint-mode attitude --bench-thrust "$BENCH_THRUST")
  elif [[ "$REHEARSE_RUN" == "1" ]]; then
    sc_args=(--active --ai-gate gate --ai-threshold 0.85 --hold-scale 1.0 \
             --no-px4 --land-timeout-sec 2)
  elif [[ "$ACTIVE_RUN" == "1" ]]; then
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
  if [[ "$REHEARSE_RUN" != "1" ]]; then
    stop_tegrastats "$rdir"
  fi

  # _assemble_run_meta.py 는 parse_args()(known_args 아님)라 모르는 플래그에 죽는다
  # (--active-run/--bench-run/--rehearse-run/--ai-threshold/--active-span-hours/
  # --sc-guard-rev/--setpoint-mode/--bench-thrust/--no-px4/--land-timeout-sec 는
  # 지원 안 함) -- 새 필드는 전부 --extra-json 으로만 넣는다(7단계 사전확인).
  # --ai-gate/--passive/--hold-scale 은 이미 네이티브 지원이라 그대로 전달.
  if [[ "$BENCH_RUN" == "1" ]]; then
    write_run_meta "$rdir" "e_offboard" --window "$window" --rep "$rep" \
      --warmup-speed "$WARMUP_SPEED" --active-speed "$ACTIVE_SPEED" --instrumented true \
      --ai-gate gate --passive false --hold-scale 1.0 \
      --tegra-interval-ms "$TEGRA_INTERVAL_MS" --tegra-prestart-sec "$TEGRA_PRESTART_SEC" \
      --extra-json "{\"active_run\": true, \"bench_run\": true, \"setpoint_mode\": \"attitude\", \"bench_thrust\": $BENCH_THRUST, \"no_px4\": false, \"ai_threshold\": 0.85, \"active_span_hours\": $ACTIVE_SPAN_HOURS, \"sc_guard_rev\": \"v1_rearm_disarm_guard\"}"
  elif [[ "$REHEARSE_RUN" == "1" ]]; then
    # active_run:false(의도적) -- sc_offboard_node.py 는 --active 로 돌지만(내부
    # passive=0), --no-px4 라 실제로는 아무 것도 안 나간다. 실기 결과와 절대 섞이면
    # 안 되므로 "진짜 실기" 를 뜻하는 active_run 은 여기서 false 로 못박는다.
    write_run_meta "$rdir" "e_offboard" --window "$window" --rep "$rep" \
      --warmup-speed "$WARMUP_SPEED" --active-speed "$ACTIVE_SPEED" --instrumented true \
      --ai-gate gate --passive false --hold-scale 1.0 \
      --tegra-interval-ms "$TEGRA_INTERVAL_MS" --tegra-prestart-sec "$TEGRA_PRESTART_SEC" \
      --extra-json "{\"active_run\": false, \"rehearse_run\": true, \"no_px4\": true, \"land_timeout_sec\": 2, \"ai_threshold\": 0.85, \"active_span_hours\": $ACTIVE_SPAN_HOURS, \"sc_guard_rev\": \"v1_rearm_disarm_guard\", \"tegrastats_skipped\": true}"
  elif [[ "$ACTIVE_RUN" == "1" ]]; then
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

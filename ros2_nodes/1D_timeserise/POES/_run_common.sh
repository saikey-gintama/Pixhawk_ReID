#!/usr/bin/env bash
# ============================================================================
# _run_common.sh -- S5 실행 스크립트 4개(run_idle/fsm/tcn/offboard_resource.sh)가
# source 하는 공통 라이브러리. 직접 실행하지 않는다(실행 스크립트가 아님).
#
# KSEM/run_resource_experiment.sh 의 실행·종료·마커 폴링 패턴 승계.
# 노드 기동 순서는 역방향(하류 먼저): sc -> ai -> ap -> wp. wp 가 REPLAY_DONE
# 마커를 쓰면 폴링 종료 -> SIGINT -> 3초 -> SIGKILL.
# ============================================================================
set -uo pipefail   # 개별 run_one() 안 실패로 전체 스위트가 죽지 않게 -e 는 안 씀(호출부가 판단)

# ── 경로 (git pull 후 동일 가정, KSEM 패턴 그대로) ──
REPO="${REPO:-$HOME/jeongin/Pixhawk_ReID}"
NODE_DIR="${NODE_DIR:-$REPO/ros2_nodes/1D_timeserise/POES}"
DATA="${DATA:-$REPO/Experiment_window/C_PD/POES/MetOp03_count/poes_metop03_cache_parquet}"
CKPT_ROOT="${CKPT_ROOT:-$REPO/Experiment_window/C_PD/predict_v0/runs}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO/results}"
EVENT_WINDOWS_JSON="${EVENT_WINDOWS_JSON:-$NODE_DIR/event_windows.json}"

# ── 배속 (S5.1 패치: active 1800 -> 7200, warm-up 은 7200 유지) ──
# 근거: 틱 주기 900/7200=125ms, 틱당 작업 약 5ms -> 여유 25배. 17일 구간(1,632틱)이
# 13.6분 -> 3.4분/run, 24 run 이 5.4시간 -> 1.4시간. run_fsm_resource.sh 의
# speed-sanity 대조(1800x vs 7200x, M1/M2/M3/M5 median 비교)로 지연 왜곡 없음을
# 확인한 뒤 채택 -- 불일치하면 7200 을 버리고 1800 으로 되돌릴 것(그 스크립트 참고).
WARMUP_SPEED="${WARMUP_SPEED:-7200}"
ACTIVE_SPEED="${ACTIVE_SPEED:-7200}"

# ── tegrastats (모든 시나리오 동일 기동, (a) 에서도 돔 -- 차분에서 상쇄) ──
TEGRA_INTERVAL_MS="${TEGRA_INTERVAL_MS:-1000}"
TEGRA_PRESTART_SEC="${TEGRA_PRESTART_SEC:-30}"   # 노드 기동 전 대기(집계 제외 구간), T 기록

# ── 스모크 / dry-run (환경변수 또는 --smoke --dry-run 인자로 각 스크립트가 설정) ──
SMOKE="${SMOKE:-0}"
DRY_RUN="${DRY_RUN:-0}"

TEGRA_PID=""

print_power_warning() {
  echo "############################################################################"
  echo "# POWER/CPU AT ACCELERATED REPLAY -- NOT FLIGHT CONDITIONS.                #"
  echo "# For per-operation energy see bench_micro burst mode (S6).                #"
  echo "# (지연 측정 M1~M10 은 배속과 무관하게 유효 -- 이 경고 대상 아님)          #"
  echo "############################################################################"
}

# event_windows.json 없으면 1회 생성(멱등 -- 있으면 재사용)
ensure_event_windows() {
  if [[ ! -f "$EVENT_WINDOWS_JSON" ]]; then
    echo "[common] event_windows.json 없음 -> select_event_windows.py 실행"
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "  [dry-run] REPO=$REPO python3 $NODE_DIR/select_event_windows.py"
    else
      REPO="$REPO" python3 "$NODE_DIR/select_event_windows.py"
    fi
  fi
}

# get_window_field <strong|weak|cluster> <start|end|onset_time|...>
get_window_field() {
  local name="$1" field="$2"
  python3 -c "
import json
d = json.load(open('$EVENT_WINDOWS_JSON', encoding='utf-8'))
print(d['windows']['$name']['$field'])
"
}

# run_cmd <설명> <cmd...>  -- DRY_RUN=1 이면 echo 만, 아니면 백그라운드 실행 후 PID 를
# 전역변수 LAST_PID 에 남긴다(bash 함수는 PID 를 직접 return 할 수 없어서).
run_cmd() {
  local desc="$1"; shift
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  [dry-run] ($desc) $*"
    LAST_PID=0
    return 0
  fi
  echo "  ($desc) $*"
  "$@" &
  LAST_PID=$!
}

start_tegrastats() {
  local rdir="$1"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  [dry-run] tegrastats --interval $TEGRA_INTERVAL_MS --logfile $rdir/tegrastats.log"
    TEGRA_PID=""
    echo "0" > "$rdir/tegrastats_start_ts.txt"
    return 0
  fi
  tegrastats --interval "$TEGRA_INTERVAL_MS" --logfile "$rdir/tegrastats.log" &
  TEGRA_PID=$!
  date +%s.%N > "$rdir/tegrastats_start_ts.txt"
  echo "  tegrastats 시작 (pid=$TEGRA_PID interval=${TEGRA_INTERVAL_MS}ms) -> $rdir/tegrastats.log"
  echo "  노드 기동 전 대기 T=${TEGRA_PRESTART_SEC}s (집계 제외 구간, run_meta 에 기록)"
  sleep "$TEGRA_PRESTART_SEC"
}

stop_tegrastats() {
  local rdir="$1"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  [dry-run] tegrastats 종료"
    echo "0" > "$rdir/tegrastats_end_ts.txt"
    return 0
  fi
  if [[ -n "$TEGRA_PID" ]]; then
    kill -INT "$TEGRA_PID" 2>/dev/null || true
    sleep 1
    kill -KILL "$TEGRA_PID" 2>/dev/null || true
    wait "$TEGRA_PID" 2>/dev/null || true
  fi
  date +%s.%N > "$rdir/tegrastats_end_ts.txt"
  echo "  tegrastats 종료"
}

# poll_replay_done <rdir> <wp_pid>
poll_replay_done() {
  local rdir="$1" wp_pid="$2"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  [dry-run] REPLAY_DONE 폴링 생략"
    return 0
  fi
  local waited=0
  while [[ ! -f "$rdir/REPLAY_DONE" ]]; do
    if ! kill -0 "$wp_pid" 2>/dev/null; then
      echo "  !! wp 프로세스가 마커 없이 종료됨 -- $rdir 로그 확인" >&2
      break
    fi
    sleep 5
    waited=$((waited + 5))
  done
  sleep 3   # active 구간 잔여 메시지가 하류 노드 로그에 flush 될 시간
}

# shutdown_nodes <pid...>  -- 역순 상관없이 한꺼번에 SIGINT -> 3초 -> SIGKILL(md 지시 그대로)
shutdown_nodes() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  [dry-run] 노드 종료(SIGINT -> 3s -> SIGKILL) 생략"
    return 0
  fi
  local pids=("$@")
  echo "  -- shutting down nodes: ${pids[*]} --"
  kill -INT "${pids[@]}" 2>/dev/null || true
  sleep 3
  kill -KILL "${pids[@]}" 2>/dev/null || true
  wait 2>/dev/null || true
  sleep 2
}

# write_run_meta <rdir> <scenario> [--window ... --rep ... ...나머지는 _assemble_run_meta.py 로 그대로 전달]
write_run_meta() {
  local rdir="$1" scenario="$2"; shift 2
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  [dry-run] python3 $NODE_DIR/_assemble_run_meta.py $rdir --scenario $scenario $* --event-windows-json $EVENT_WINDOWS_JSON"
    return 0
  fi
  python3 "$NODE_DIR/_assemble_run_meta.py" "$rdir" --scenario "$scenario" \
    --event-windows-json "$EVENT_WINDOWS_JSON" "$@"
}

# new_result_dir <tag> -- results/<타임스탬프>_<tag>/ 생성 후 경로 echo
new_result_dir() {
  local tag="$1"
  local rdir="$RESULTS_ROOT/$(date +%Y%m%d_%H%M%S)_${tag}"
  mkdir -p "$rdir"
  echo "$rdir"
}

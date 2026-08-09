"""
_assemble_run_meta.py
======================
S5 run_meta.json 조립기. run_*.sh 4개가 매번 같은 파이썬 로직(JSON 병합)을 반복하지
않도록 분리. rdir 안에 각 노드가 이미 써 둔 산출물(있는 것만) 을 읽어 스크립트가
넘긴 시나리오 메타(scenario/window/rep/배속 등)와 합쳐 run_meta.json 을 만든다.

읽는 파일 (전부 optional -- 없으면 그 절만 비움, 시나리오에 없는 노드는 당연히 없음):
  wp_stats.json        (wp_poes_node.py 가 종료 시 씀)  -> n_ticks_total, warm-up 제외 틱 수
  ap_fsm_summary.json  (ap_fsm_node.py 가 종료 시 씀)   -> n_ticks_gate_open 등
  tcn_startup.json     (ai_tcn_node.py 가 기동 시 씀)   -> torch_threads/device/folds_mode

사용:
  python3 _assemble_run_meta.py <rdir> --scenario b --window strong --rep 1 \
      --warmup-speed 7200 --active-speed 1800 --instrumented true \
      --event-windows-json event_windows.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_if_exists(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def main():
    ap = argparse.ArgumentParser(description="S5 run_meta.json 조립기")
    ap.add_argument("rdir")
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--window", default=None)
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--warmup-speed", type=float, default=None)
    ap.add_argument("--active-speed", type=float, default=None)
    ap.add_argument("--instrumented", default="true", choices=["true", "false"])
    ap.add_argument("--ai-gate", default=None)
    ap.add_argument("--passive", default=None)
    ap.add_argument("--hold-scale", type=float, default=None)
    ap.add_argument("--n-channels-arg", type=int, default=None,
                    help="run_tcn_resource.sh 가 넘기는 요청 채널 수(1/3) -- 실제 로드된 채널 수는 tcn_startup.json 것")
    ap.add_argument("--tegra-interval-ms", type=float, default=None)
    ap.add_argument("--tegra-prestart-sec", type=float, default=None)
    ap.add_argument("--event-windows-json", default=None)
    ap.add_argument("--extra-json", default=None, help="추가 필드(JSON 객체 문자열) 그대로 병합")
    args = ap.parse_args()

    rdir = Path(args.rdir)
    meta = {
        "scenario": args.scenario, "window": args.window, "rep": args.rep,
        "warmup_speed": args.warmup_speed, "active_speed": args.active_speed,
        "instrumented": args.instrumented == "true",
        "ai_gate_mode": args.ai_gate, "passive": args.passive, "hold_scale": args.hold_scale,
        "n_channels_requested": args.n_channels_arg,
        "tegra_interval_ms": args.tegra_interval_ms,
        "tegra_prestart_sec_excluded": args.tegra_prestart_sec,
        "power_cpu_caveat": ("가속 리플레이 tegrastats 전력/CPU% 는 비행 전력 아님. "
                             "1x 배속 SITL 실증 별도(4.4절). per-op 에너지는 bench_micro 버스트(S6)."),
    }

    if args.window and args.event_windows_json:
        ew = _load_if_exists(Path(args.event_windows_json))
        if ew and args.window in ew.get("windows", {}):
            meta["event_window_detail"] = ew["windows"][args.window]

    wp_stats = _load_if_exists(rdir / "wp_stats.json")
    if wp_stats:
        meta["n_ticks_total"] = wp_stats.get("n_ticks_total")
        meta["n_bg_warmup_ticks_excluded"] = wp_stats.get("n_bg_warmup_ticks")
        meta["bg_warmup_end_ts"] = wp_stats.get("bg_warmup_end_ts")
        meta["channels"] = wp_stats.get("channels")

    ap_summary = _load_if_exists(rdir / "ap_fsm_summary.json")
    if ap_summary:
        meta["n_ticks_total_ap"] = ap_summary.get("n_ticks_total")
        meta["n_ticks_gate_open"] = ap_summary.get("n_ticks_gate_open")
        meta["n_ticks_alert_open"] = ap_summary.get("n_ticks_alert_open")
        meta["n_transitions_by_type"] = ap_summary.get("n_transitions_by_type")
        meta["n_downlinked_alerts"] = ap_summary.get("n_downlinked_alerts")
        meta["evs_suppressed_total"] = ap_summary.get("evs_suppressed_total")

    tcn_startup = _load_if_exists(rdir / "tcn_startup.json")
    if tcn_startup:
        meta["folds_mode"] = tcn_startup.get("folds_mode")
        meta["fold"] = tcn_startup.get("fold")
        meta["torch_threads"] = tcn_startup.get("torch_threads")
        meta["device"] = tcn_startup.get("device")
        meta["model_load_ms"] = tcn_startup.get("model_load_ms")
        meta["n_models"] = tcn_startup.get("n_models")
        meta["ai_channels"] = tcn_startup.get("channels")

    if args.extra_json:
        try:
            meta.update(json.loads(args.extra_json))
        except Exception as e:
            meta["_extra_json_parse_error"] = str(e)

    # 표본 부족 확인용 -- M6~M9 p95 를 여기서 계산하진 않지만(집계는 S7 aggregate_onboard.py
    # 몫), 최소한 게이트 개방 틱 수를 run_meta 에서 바로 볼 수 있게 해 표본 부족을 즉시 알아채게 한다.
    if meta.get("n_ticks_gate_open") is not None:
        meta["_note_sample_size"] = (
            f"n_ticks_gate_open={meta['n_ticks_gate_open']} -- M6~M9 p95 표본수와 같다. "
            f"너무 작으면(수십 미만) 구간/반복을 늘릴 것.")

    out = rdir / "run_meta.json"
    out.write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"[assemble_run_meta] 저장 -> {out}")


if __name__ == "__main__":
    main()

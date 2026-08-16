"""
fig5_pipeline.py
=================
IAA 논문용 파이프라인 블록도(그림 5) -- POES count -> WP -> AP -> AI(게이트) -> SC(RTS)
4단 데이터 흐름 + WP 배경 통계(median/MAD) 공유를 강조. matplotlib만 사용(외부 도구 없음).
읽기 전용 그림 스크립트 -- 파이프라인 노드 코드는 참조만 하고 import/수정하지 않는다.

수치/사실 출처 (이 세션에서 실측 확인됨):
  WP  0.536 ms/sample     : results/260809_2225/bench_summary.csv M1(resample) bench_pure_ms
  AP  0.0057 ms/tick      : results/260809_2225/bench_summary.csv M3(fsm_eval) bench_pure_ms
  AI  2.101 ms/activation : results/260809_2225/derived_cost.csv input_size=3,single,
                            M9_median_ms(S6 벤치 구성값, "3채널 bare")
  gate duty 6.84%         : gate_persistence_sweep.csv N=2 행 duty_cycle(16204/236848)
  gate 조건 "2 consecutive exceedances" : wp_poes_node.py GATE_N=2(연속 2샘플=30min)
  토픽명 /wp_results / /sep_alert / /ai_verdict / /fmu/in/vehicle_command
      : wp_poes_node.py / ap_fsm_node.py / ai_tcn_node.py / sc_offboard_node.py 의
        create_publisher 실제 토픽 문자열 그대로.
  채널 수: WP/AP는 CHANNELS 기본값 omni_p6 단일채널(1ch), AI는 d_tcn_3ch 시나리오
        (omni_p6 + pro_tel0_p5 + pro_tel90_p5, 3ch).
  WP -> AP/AI 이중 화살표 근거: wp_poes_node.py는 /wp_results 1개만 발행하고
        ap_fsm_node.py(watch 불리언으로 판정)와 ai_tcn_node.py(z를 링버퍼에 적재)
        둘 다 그 토픽을 직접 구독한다(AP를 거쳐 중계되는 게 아님) -- 논문 3.2절
        "두 계층이 하나의 배경 추정을 공유한다"가 이 그림의 핵심 메시지.

출력: fig5_pipeline.png, fig5_pipeline.pdf (같은 디렉터리, dpi=300)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle

OUT_DIR = Path(__file__).resolve().parent

# ══════════════════════════════════════════════════════
# 캔버스 (논문 인쇄용 -- 6.2 x 3.2 in, dpi=300, 여백 0, 1 data unit ~= 1 in)
# ══════════════════════════════════════════════════════
FIG_W, FIG_H = 6.2, 3.2
# 해칭은 AI 박스 전용 신호(상시 구동 아님)라 옅은 회색 + 더 얇은 선 + 넓은 간격으로
# 글자와 부딪히지 않게 한다(테두리 edgecolor는 박스별로 검정 그대로, 해칭 색만
# 별도 rcParam으로 분리). AP/SC는 아예 해칭을 빼서(box(...hatch=None)) 이 신호를
# AI가 독점하게 한다.
plt.rcParams["hatch.color"] = "#c4c4c4"
plt.rcParams["hatch.linewidth"] = 0.4
fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=300)
ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
ax.set_xlim(-0.05, FIG_W + 0.05)
ax.set_ylim(-0.06, FIG_H + 0.08)
ax.axis("off")

# ── 팔레트: 색 + 해칭/선종류를 같이 써서 흑백 인쇄에서도 구분되게 ──
C_POES = "#eeeeee"
C_WP   = "#dce9f7"
C_AP   = "#e3efd8"
C_AI   = "#f3ecd9"
C_SC   = "#f4e0e6"
C_EDGE = "#161616"
C_ALWAYS = "#2b567e"
C_GATED  = "#8a2f2f"
C_HILITE = "#ffe98a"
C_SHARE  = "#555555"

FS_TITLE, FS_SUB, FS_BULLET, FS_BADGE, FS_CH, FS_COST, FS_TOPIC = 9, 7.0, 7.0, 7.6, 7.6, 7.2, 6.2


def box(x, y, w, h, fc, hatch=None, ls="-", lw=1.15, zorder=2):
    p = FancyBboxPatch((x, y), w, h,
                        boxstyle="round,pad=0.0,rounding_size=0.045",
                        linewidth=lw, edgecolor=C_EDGE, facecolor=fc,
                        linestyle=ls, hatch=hatch, zorder=zorder)
    ax.add_patch(p)
    return p


def txt(x, y, s, size=7, weight="normal", style="normal", ha="center", va="center",
        color="black", zorder=3, linespacing=1.15, family="DejaVu Sans"):
    # halo(흰 테두리)는 해칭이 AI 박스 하나로 줄고 그마저 옅어져서 더 필요 없다 --
    # AP/SC는 이제 해칭 자체가 없고, AI도 해칭이 성겨서 맨 텍스트로 충분히 읽힌다.
    return ax.text(x, y, s, fontsize=size, fontweight=weight, fontstyle=style, ha=ha, va=va,
                   color=color, zorder=zorder, linespacing=linespacing, family=family)


def pipe_arrow(x0, y0, x1, y1, lw=1.3, color=C_EDGE, ls="-"):
    p = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8.5,
                         linewidth=lw, linestyle=ls, color=color, zorder=2.6,
                         shrinkA=0, shrinkB=0)
    ax.add_patch(p)


def share_arrow(x0, y0, x1, y1):
    p = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6.5,
                         linewidth=0.9, linestyle=(0, (3, 1.6)), color=C_SHARE, zorder=2.4,
                         shrinkA=0, shrinkB=1.5)
    ax.add_patch(p)


# ══════════════════════════════════════════════════════
# 레이아웃 상수
# ══════════════════════════════════════════════════════
Y_ARROW = 1.85                # 메인 파이프라인 화살표 높이
BOX_Y, BOX_H = 0.85, 2.00     # WP/AP/AI/SC 처리 박스
TOP = BOX_Y + BOX_H            # 2.85 -- 박스 상단
POES_Y, POES_H = 1.45, 0.80

POES_X, POES_W = 0.05, 0.68
WP_X,   PROC_W = 1.05, 0.95
GAP = 0.25
AP_X = WP_X + PROC_W + GAP
AI_X = AP_X + PROC_W + GAP
SC_X = AI_X + PROC_W + GAP

cx = lambda x: x + PROC_W / 2  # noqa: E731

# 박스 내부 공통 수직 리듬 (4개 박스 모두 불릿 3줄로 통일 -- 어긋남 방지)
TITLE_Y = TOP - 0.18
SUB1_Y  = TOP - 0.38
SUB2_Y  = TOP - 0.56
BUL_Y   = [TOP - 0.76, TOP - 0.97, TOP - 1.18]
BADGE_Y = BOX_Y + 0.58
PILL_Y  = BOX_Y + 0.28

# 박스 위 라벨 두 개 층(레인) -- 좁은 박스 간격 안에 글자를 넣으면 옆 박스와 겹치므로
# 박스 "밖"(TOP보다 위, 박스 콘텐츠가 없는 영역)에 눕혀서 겹침을 원천 차단한다.
LANE_TOPIC = TOP + 0.09     # ROS2 토픽명 (모든 화살표 공통)
LANE_GATE  = TOP + 0.27     # 게이트 조건 콜아웃 (AP->AI 화살표 전용, item 1)

# ── POES 소스 박스 ──
box(POES_X, POES_Y, POES_W, POES_H, C_POES, lw=1.0)
txt(POES_X + POES_W / 2, POES_Y + POES_H / 2, "POES\n1-min\ncounts", size=6.8, weight="bold")

# ── 처리 4단: WP / AP / AI / SC -- 해칭은 AI 박스가 독점(상시 구동 아님 신호).
# AP/SC는 옅은 단색 채움 + 실선 테두리로 통일(해칭 없음).
box(WP_X, BOX_Y, PROC_W, BOX_H, C_WP)
box(AP_X, BOX_Y, PROC_W, BOX_H, C_AP)
box(AI_X, BOX_Y, PROC_W, BOX_H, C_AI, hatch="/", ls=(0, (4, 2)))   # AI만 점선+해칭 = 상시 구동 아님
box(SC_X, BOX_Y, PROC_W, BOX_H, C_SC)


def node_header(x, code, node_file, quoted):
    cxp = cx(x)
    txt(cxp, TITLE_Y, code, size=FS_TITLE, weight="bold")
    txt(cxp, SUB1_Y, f"({node_file})", size=FS_SUB, style="italic", color="#333333")
    txt(cxp, SUB2_Y, f"“{quoted}”", size=FS_SUB, style="italic", color="#333333")


def bullets(x, lines):
    cxp = cx(x)
    for y, ln in zip(BUL_Y, lines):
        txt(cxp, y, ln, size=FS_BULLET, ha="center")


def badge(x, lines, color):
    txt(cx(x), BADGE_Y, "\n".join(lines), size=FS_BADGE, weight="bold", color=color, linespacing=1.25)


def channel_pill(x, label, bold=True):
    cxp = cx(x)
    w, h = 0.86, 0.24
    box(cxp - w / 2, PILL_Y - h / 2, w, h, C_HILITE, lw=0.8)
    txt(cxp, PILL_Y, label, size=FS_CH, weight="bold" if bold else "normal")


# ── WP ──
node_header(WP_X, "WP", "wp_poes_node", "cFS LC_watch")
bullets(WP_X, ["• 15-min resample", "• 7-day MAD bg", "• threshold, z"])
badge(WP_X, ["ALWAYS ON", "every tick"], C_ALWAYS)
channel_pill(WP_X, "omni_p6 (1 ch)")

# ── AP ──
node_header(AP_X, "AP", "ap_fsm_node", "cFS LC_action")
bullets(AP_X, ["• RPN combine", "• persistence cnt.", "• → N/PRE/ALERT"])
badge(AP_X, ["ALWAYS ON", "every tick"], C_ALWAYS)
channel_pill(AP_X, "omni_p6 (1 ch)")

# ── AI ──
node_header(AI_X, "AI", "ai_tcn_node", "gated TCN")
bullets(AI_X, ["• 14-smpl z window", "• TCN forward", "• → p_event"])
badge(AI_X, ["GATED", "6.84% of ticks"], C_GATED)
channel_pill(AI_X, "3 ch", bold=True)

# ── SC ──
node_header(SC_X, "SC", "sc_offboard_node", "cFS SC / RTS")
bullets(SC_X, ["• ALERT →", "• PX4 Offboard", "• + ARM"])
badge(SC_X, ["LATCHED ALERT", "ONLY"], C_GATED)
# SC엔 채널 표기 없음(항목4 요구 없음) -- 자리 비워 4단 badge 높이만 정렬 유지

# ══════════════════════════════════════════════════════
# 메인 파이프라인 화살표 + ROS2 토픽 라벨(박스 밖 레인, item 5)
# ══════════════════════════════════════════════════════
pipe_arrow(POES_X + POES_W, Y_ARROW, WP_X, Y_ARROW)

pipe_arrow(WP_X + PROC_W, Y_ARROW, AP_X, Y_ARROW)
txt(WP_X + PROC_W + GAP / 2, LANE_TOPIC, "/wp_results", size=FS_TOPIC, family="monospace")

pipe_arrow(AP_X + PROC_W, Y_ARROW, AI_X, Y_ARROW)
txt(AP_X + PROC_W + GAP / 2, LANE_TOPIC, "/sep_alert", size=FS_TOPIC, family="monospace")

pipe_arrow(AI_X + PROC_W, Y_ARROW, SC_X, Y_ARROW)
txt(AI_X + PROC_W + GAP / 2, LANE_TOPIC, "/ai_verdict", size=FS_TOPIC, family="monospace")

# SC -> PX4 (외부 FCU, 박스 없이 화살표만 우측 여백으로)
PX4_TIP = SC_X + PROC_W + 0.42
pipe_arrow(SC_X + PROC_W, Y_ARROW, PX4_TIP, Y_ARROW)
txt(SC_X + PROC_W + 0.21, LANE_TOPIC, "/fmu/in/\nvehicle_command",
    size=FS_TOPIC, family="monospace", linespacing=1.05)
txt(SC_X + PROC_W + 0.21, Y_ARROW - 0.27, "PX4 FCU", size=6.6, weight="bold")

# 게이트 조건 콜아웃 (AP -> AI 화살표 전용 강조, item 1) -- 토픽 레인보다 한 층 위
txt(AP_X + PROC_W + GAP / 2, LANE_GATE,
    "gate: 2 consecutive\nexceedances (6.84% duty)",
    size=6.5, weight="bold", color=C_GATED, linespacing=1.1)

# ══════════════════════════════════════════════════════
# WP 배경 통계(median, MAD) 공유 -- WP 내부(임계 판정에 쓰는 그 추정치) -> AI 한 갈래만.
# AP는 안 받는다: ap_fsm_node.py docstring이 명시하듯 AP는 /wp_results의 watch
# 불리언(TRUE/FALSE/STALE)만 판정 입력으로 쓰고, WP의 run_len/gate_open/alert_open은
# HK 참고값일 뿐 판정에 쓰지 않는다 -- 배경 통계 자체를 받는 게 아니다.
# ══════════════════════════════════════════════════════
JX, JY = cx(WP_X), BOX_Y - 0.28
ax.plot([cx(WP_X), JX], [BOX_Y, JY], color=C_SHARE, lw=0.9,
        linestyle=(0, (3, 1.6)), zorder=2.3, solid_capstyle="round")
ax.add_patch(Circle((JX, JY), 0.022, facecolor=C_SHARE, edgecolor="none", zorder=2.5))
share_arrow(JX, JY, cx(AI_X), BOX_Y)
txt(JX + 0.34, JY - 0.17,
    "shared background (median, MAD):\nthreshold and z from one estimate",
    size=6.3, style="italic", color=C_SHARE, ha="left", linespacing=1.15)

# ══════════════════════════════════════════════════════
# 실측 비용 (item 3, 박스 아래)
# ══════════════════════════════════════════════════════
COST_Y = 0.13
txt(cx(WP_X), COST_Y, "0.536 ms/sample", size=FS_COST, style="italic", color="#333333")
txt(cx(AP_X), COST_Y, "0.0057 ms/tick", size=FS_COST, style="italic", color="#333333")
txt(cx(AI_X), COST_Y, "2.101 ms/activation", size=FS_COST, style="italic", color="#333333")
txt(cx(SC_X), COST_Y, "(latched only)", size=FS_COST, style="italic", color="#333333")

# ══════════════════════════════════════════════════════
# 저장
# ══════════════════════════════════════════════════════
png_path = OUT_DIR / "fig5_pipeline.png"
pdf_path = OUT_DIR / "fig5_pipeline.pdf"
fig.savefig(png_path, dpi=300)
fig.savefig(pdf_path)
plt.close(fig)
print(f"[fig5] saved -> {png_path}")
print(f"[fig5] saved -> {pdf_path}")

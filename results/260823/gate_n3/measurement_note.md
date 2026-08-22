# 측정 조건 노트 (논문 4.3절 인용용)

**경고: 전력·CPU 는 가속 리플레이 값이며 비행 조건이 아니다. 연산 1회당 에너지는 버스트 모드(S6)에서만 유효하다.**

**하루 비용(daily_ms/daily_energy)은 해석적으로 계산된 값이다 -- 젯슨은 1회 비용만 실측했고(S5/S6), 하루 몇 번 발생하는지는 S0.5 전 구간(236,848틱) duty_cycle(6.84%) x TICKS_PER_DAY(96) = 6.568회/일을 그대로 썼다.**

## 플랫폼/환경
- torch_threads: (bench 없음)
- core_pin: (bench 없음)
- clocks(nvpmodel/jetson_clocks): (bench 없음)
- device: (bench 없음)
- os: (bench 없음)

## 리플레이 구간 3개 (event_windows.json)
- event_id=4, 2022-03-18~2022-04-04, peak_count=3, 기준: 고정 [onset-10d, onset+7d] 창에 완결 이벤트가 3개(자기 포함) 들어오는 최다밀집 기준 이벤트
- event_id=2, 2022-01-10~2022-01-27, peak_count=1152.03, 기준: peak_count 최댓값(완결 이벤트 중 가장 강한 SPE)
- event_id=55, 2024-01-12~2024-01-29, peak_count=0.73, 기준: peak_count 최솟값(완결 이벤트 중 가장 약한 SPE, 경계 검출 사례)

## 배속
- warm-up/active 배속 조합: ['(7200.0, 7200.0)']
- speed_sanity(1800x vs 7200x): speed_sanity.csv 없음(아직 실행 안 됨)

## warm-up 제외
- 배경 워밍업 제외 틱 수(run_meta 실측값, run별): [96]
- 기준: bg_median/bg_std 가 전 채널에서 처음 유효해지는 시각 이전 전부 (wp_poes_node.py 실측, 휴리스틱 아님)

## 반복 수
- 시나리오 c1ch: n_runs=6, n_ticks_active=9204, n_activations=30

## 버스트 모드
- 버스트 결과 없음(--burst 없이 실행됐거나 tegrastats 미가용)

## 계측 오버헤드 ((b) 로깅 on/off 차이)
- --no-log 대조 run 없음(run_fsm_resource.sh 아직 안 돌림)

## rep 분산 (cv > 10% 항목만 표시)
- 모든 항목 cv <= 10%

## 파생 상수
- TICKS_PER_DAY=96, SAMPLES_PER_TICK=15, GATE_OPENINGS_PER_DAY=6.568(duty_cycle x TICKS_PER_DAY 기준), DUTY_CYCLE=0.0684(S0.5 확정)

# E2E 게이팅 파이프라인 검출 성능 요약

공통 시간 인덱스(A/B, timestamp inner join): **236739틱** (2019-01-03 02:45:00+00:00 ~ 2025-12-31 23:45:00+00:00).

## 구성 정의

- **A** TCN 단독: `p_event >= thr` (연속 확률 -> 세그먼트, 1시간 필터)

- **B** 게이트 제한 TCN: `p_event >= thr AND gate_open(N=2)` (연속 확률 -> 세그먼트, 1시간 필터, break/skip 두 변형)

- **C** ALERT ∧ TCN veto: ALERT latch 순간에 `p_event(그 틱) >= thr` (이산 순간조건, 지속필터 없음)

- **D** FSM ALERT 단독: ALERT latch 그 자체 (이산 전이, 지속필터 없음)


## C/D 재정의 (범주 오류 수정)

ALERT(alert_n=4)는 latched transition(이산 상태 전이)이지 연속 신호가 아니다. 이전 버전은 여기에 probs_to_onsets 의 1시간 연속 세그먼트 필터를 한 번 더 적용해 항상 0건이 나왔다(ALERT 지속시간 중앙값 30분·최댓값 45분, 1시간을 구조적으로 못 채움). C/D는 이제 ALERT latch 이벤트(216건)를 원재료로 쓴다: D는 그대로, C는 latch 순간의 p_event(그 틱)가 thr 이상인지만 순간적으로 확인한다.


## 게이트 닫힌 틱의 세그먼트 연속성 처리 (B만 해당)

게이트가 닫힌 틱에서 p_event는 '임계 미만'이 아니라 '추론 없음'이다. 두 해석을 모두 계산했다:

- **break**: 조건 계산에서 False로 취급 -- 진행 중이던 세그먼트를 끊는다.

- **skip**: 그 틱을 시계열에서 제거 -- 연속성 판정에 나타나지 않으므로 앞뒤의 True 상태가 이어진다(S1/S2 STALE 동결과 동일한 태도).

C는 순간조건이라 연속성 개념 자체가 없어 이 구분이 필요 없다.


## 필수 관문 ①: A 대 기존 comparison_table.csv 대조

  [PASS] noaa@th=0.99: 전부 일치

  [PASS] manual@th=0.85: 전부 일치


## 필수 관문 ②: D 대 논문 4.6절 causal ALERT 재현

  [PASS] D vs 논문 4.6절(causal ALERT, noaa): n_det=216(기대 216, OK)  POD=0.7381->round3=0.738(기대 0.738, OK)  event_FAR=0.0606->round3=0.061(기대 0.061, OK)


## 게이팅이 제거한 검출 (A 최적임계, F1 최대 기준)


### 기준: noaa (A 최적임계, F1=0.8696, th=0.99)


- n_det_A=217, n_det_B_gated_tcn(break)=148, n_removed=80 (true=33, false=47)

- n_det_A=217, n_det_B_gated_tcn(skip)=136, n_removed=12 (true=2, false=10)

- n_det_A=217, n_det_C_alert_and_tcn(na)=187, n_removed=63 (true=25, false=38)


### 기준: manual (A 최적임계, F1=0.6822, th=0.96)


- n_det_A=189, n_det_B_gated_tcn(break)=178, n_removed=114 (true=29, false=85)

- n_det_A=189, n_det_B_gated_tcn(skip)=87, n_removed=47 (true=8, false=39)

- n_det_A=189, n_det_C_alert_and_tcn(na)=206, n_removed=106 (true=27, false=79)

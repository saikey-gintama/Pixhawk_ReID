#!/bin/bash
# ============================================================
# run_experiment.sh
# 실행: bash run_experiment.sh
# 종료: Ctrl+C → 모든 노드 자동 종료
# 결과:
#   - log.csv            : 매 프레임 성능 로그
#   - detection_log.csv  : 이벤트 기반 탐지 로그
#   - event_log.csv      : offboard FSM 이벤트 로그
#   - bw_camera.txt      : /camera 대역폭
#   - bw_roi.txt         : /target_roi 대역폭
# ============================================================

source /opt/ros/humble/setup.bash
source /home/moon/jeongin/ROS2/ws_sensor_combined/install/setup.bash
source /home/moon/jeongin/ROS2/px4_ros_uxrce_dds_ws/install/setup.bash
export PYTHONPATH=/usr/lib/python3/dist-packages:/opt/ros/humble/lib/python3.10/site-packages:$PYTHONPATH

# ── 결과 저장 폴더 ──
export RESULT_DIR=~/jeongin/Pixhawk_ReID/results/$(date +"%Y%m%d_%H%M%S")
mkdir -p "$RESULT_DIR"
echo "[INFO] 결과 저장 경로: $RESULT_DIR"

NODE_DIR=~/jeongin/Pixhawk_ReID/ros2_nodes

# ── 종료 시 정리 ──
cleanup() {
    echo ""
    echo "[INFO] 종료 중..."

    kill $PID_TAIL $PID_BW_CAM $PID_BW_ROI 2>/dev/null
    kill $PID_AGENT $PID_CAM $PID_YOLO $PID_OFFBOARD 2>/dev/null

    wait $PID_AGENT $PID_CAM $PID_YOLO $PID_OFFBOARD \
         $PID_BW_CAM $PID_BW_ROI $PID_TAIL 2>/dev/null

    echo "[INFO] 완료. 결과: $RESULT_DIR"
    ls "$RESULT_DIR"
    exit 0
}
trap cleanup SIGINT SIGTERM

# ── T1: MicroXRCE-DDS Agent ──
echo "[T1] MicroXRCEAgent 시작..."
MicroXRCEAgent udp4 -p 8888 > "$RESULT_DIR/agent.log" 2>&1 &
PID_AGENT=$!
sleep 3

# ── T2: 카메라 노드 ──
echo "[T2] camera_node 시작..."
cd "$NODE_DIR"
python3 camera_node.py > "$RESULT_DIR/camera_node.log" 2>&1 &
PID_CAM=$!
sleep 2

# ── T3: YOLO 노드 ──
echo "[T3] yolo_node 시작..."
python3 yolo_node.py > "$RESULT_DIR/yolo_node.log" 2>&1 &
PID_YOLO=$!
sleep 2

# ── T4: Offboard 노드 ──
echo "[T4] offboard_node 시작..."
python3 offboard_node.py > "$RESULT_DIR/offboard_node.log" 2>&1 &
PID_OFFBOARD=$!
sleep 1

# ── 대역폭 측정 ──
echo "[실험] 대역폭 측정 시작..."
ros2 topic bw /camera     > "$RESULT_DIR/bw_camera.txt" 2>&1 &
PID_BW_CAM=$!
ros2 topic bw /target_roi > "$RESULT_DIR/bw_roi.txt"    2>&1 &
PID_BW_ROI=$!

echo ""
echo "================================================"
echo " 모든 노드 실행 중"
echo " 카메라 앞에 타겟을 갖다 대세요"
echo " 종료: Ctrl+C"
echo "================================================"
echo ""
echo "[INFO] 주요 결과 파일"
echo "  - $RESULT_DIR/log.csv"
echo "  - $RESULT_DIR/detection_log.csv"
echo "  - $RESULT_DIR/event_log.csv"
echo "  - $RESULT_DIR/bw_camera.txt"
echo "  - $RESULT_DIR/bw_roi.txt"
echo ""

# ── 실시간 yolo 로그 출력 ──
tail -f "$RESULT_DIR/yolo_node.log" &
PID_TAIL=$!

wait $PID_YOLO
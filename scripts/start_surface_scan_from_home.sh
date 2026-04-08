#!/usr/bin/env bash

set -euo pipefail

ARM_ID="fer"
WORKSPACE_ROOT="/home/hchen82/ros2_ws"
PKG_ROOT="/home/hchen82/ros2_ws/src/franka_surface_impedance_controller"
CFG_FILE="$PKG_ROOT/config/surface_impedance_controller.yaml"
URDF_PATH="/tmp/${ARM_ID}.urdf"
SCAN_TCP_XYZ="0 0 0.142"
ROS_LOG_DIR="/tmp/ros_logs"

START_RVIZ=1
SPAWN_HEMISPHERE=1
RECORD_DEBUG=0
FRESH_START=1
AUTO_SWITCH=1

LAUNCH_PID=""
SPAWNER_PID=""
BAG_PID=""
ECHO_PID=""

usage() {
  cat <<'EOF'
Usage:
  ros2 run franka_surface_impedance_controller start_surface_scan_from_home.sh [options]

Options:
  --record-debug      Start rosbag + debug_state capture before switching controllers.
  --no-rviz           Do not start RViz.
  --no-hemisphere     Do not spawn the hemisphere model.
  --keep-existing     Do not kill existing Gazebo / RViz processes before launch.
  --no-switch         Leave surface_impedance_controller inactive after spawning it.
  -h, --help          Show this help.
EOF
}

log() {
  printf '[surface-start] %s\n' "$*"
}

strip_ansi() {
  sed -r 's/\x1B\[[0-9;]*[mK]//g'
}

wait_for_controllers() {
  local timeout_s="$1"
  local needle_name="$2"
  local needle_state="$3"
  local deadline=$((SECONDS + timeout_s))
  while (( SECONDS < deadline )); do
    local status
    status="$(ros2 control list_controllers 2>/dev/null | strip_ansi || true)"
    if grep -Eq "^${needle_name}[[:space:]].*[[:space:]]${needle_state}([[:space:]]|\$)" <<<"$status"; then
      return 0
    fi
    sleep 1
  done
  return 1
}

wait_for_controller_manager() {
  local deadline=$((SECONDS + 60))
  while (( SECONDS < deadline )); do
    if ros2 control list_controllers >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM
  if [[ -n "$ECHO_PID" ]] && kill -0 "$ECHO_PID" 2>/dev/null; then
    kill "$ECHO_PID" 2>/dev/null || true
  fi
  if [[ -n "$BAG_PID" ]] && kill -0 "$BAG_PID" 2>/dev/null; then
    kill -INT "$BAG_PID" 2>/dev/null || true
    wait "$BAG_PID" 2>/dev/null || true
  fi
  if [[ -n "$SPAWNER_PID" ]] && kill -0 "$SPAWNER_PID" 2>/dev/null; then
    kill "$SPAWNER_PID" 2>/dev/null || true
  fi
  if [[ -n "$LAUNCH_PID" ]] && kill -0 "$LAUNCH_PID" 2>/dev/null; then
    kill -INT "$LAUNCH_PID" 2>/dev/null || true
    wait "$LAUNCH_PID" 2>/dev/null || true
  fi
  exit "$exit_code"
}

while (($# > 0)); do
  case "$1" in
    --record-debug)
      RECORD_DEBUG=1
      ;;
    --no-rviz)
      START_RVIZ=0
      ;;
    --no-hemisphere)
      SPAWN_HEMISPHERE=0
      ;;
    --keep-existing)
      FRESH_START=0
      ;;
    --no-switch)
      AUTO_SWITCH=0
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
  shift
done

trap cleanup EXIT INT TERM

source /opt/ros/humble/setup.bash
source "$WORKSPACE_ROOT/install/setup.bash"

export ROS_LOG_DIR
mkdir -p "$ROS_LOG_DIR"

if (( FRESH_START )); then
  log "Stopping existing Gazebo / RViz processes for a clean home-pose start"
  pkill -9 -f ros_gz_sim || true
  pkill -9 -f "ign gazebo gui" || true
  pkill -9 -f "ign gazebo server" || true
  pkill -9 -f "gz sim" || true
  pkill -9 -f robot_state_publisher || true
  pkill -9 -f joint_state_publisher || true
  pkill -9 -f rviz2 || true
  sleep 2
fi

log "Generating URDF at $URDF_PATH"
FER_XACRO="$(ros2 pkg prefix --share franka_description)/robots/${ARM_ID}/${ARM_ID}.urdf.xacro"
ros2 run xacro xacro "$FER_XACRO" \
  arm_id:="$ARM_ID" hand:=true ee_id:=franka_hand ros2_control:=true gazebo:=true gazebo_effort:=true lock_gripper_closed:=true tcp_xyz:="$SCAN_TCP_XYZ" \
  > "$URDF_PATH"

log "Launching Gazebo bringup"
ros2 launch franka_gazebo_bringup gazebo_joint_impedance_controller_example.launch.py \
  arm_id:="$ARM_ID" load_gripper:=true franka_hand:=franka_hand lock_gripper_closed:=true tcp_xyz:="$SCAN_TCP_XYZ" \
  > /tmp/franka_surface_scan_gazebo.log 2>&1 &
LAUNCH_PID=$!

log "Waiting for controller_manager"
wait_for_controller_manager || {
  echo "Timed out waiting for controller_manager" >&2
  exit 1
}

log "Waiting for joint_impedance_example_controller to become active"
wait_for_controllers 60 "joint_impedance_example_controller" "active" || {
  echo "joint_impedance_example_controller did not become active" >&2
  exit 1
}

log "Configuring surface_impedance_controller runtime params"
ros2 param set /controller_manager surface_impedance_controller.use_franka_semantic false >/dev/null
ros2 param set /controller_manager surface_impedance_controller.urdf_path "$URDF_PATH" >/dev/null
ros2 param set /controller_manager surface_impedance_controller.base_link fer_link0 >/dev/null
ros2 param set /controller_manager surface_impedance_controller.ee_link fer_hand_tcp >/dev/null
ros2 param set /controller_manager surface_impedance_controller.debug.publish_state true >/dev/null
ros2 param set /controller_manager surface_impedance_controller.debug.publish_markers true >/dev/null
ros2 param set /controller_manager surface_impedance_controller.debug.publish_rate_hz 100.0 >/dev/null

if (( RECORD_DEBUG )); then
  DBG_TS="$(date +%Y%m%d_%H%M%S)"
  DBG_ROOT="$PKG_ROOT/debug"
  DBG_DIR="$DBG_ROOT/surface_scan_debug_$DBG_TS"
  mkdir -p "$DBG_DIR"
  log "Recording debug data to $DBG_DIR"
  ros2 control list_controllers > "$DBG_DIR/controllers_before_switch.txt"
  ros2 param dump /controller_manager > "$DBG_DIR/controller_manager_params.yaml"
  ros2 param dump /surface_impedance_controller > "$DBG_DIR/surface_impedance_controller_params.yaml"
  ros2 topic echo /surface_impedance_controller/debug_state > "$DBG_DIR/debug_state.txt" &
  ECHO_PID=$!
fi

log "Spawning surface_impedance_controller launch helpers"
ros2 launch franka_surface_impedance_controller spawn_surface_impedance_controller.launch.py \
  controller_manager:=/controller_manager \
  params_file:="$CFG_FILE" \
  spawn_hemisphere:="$SPAWN_HEMISPHERE" \
  start_rviz:="$START_RVIZ" \
  spawn_joint_state_broadcaster:=false \
  spawn_franka_robot_state_broadcaster:=false \
  spawn_surface_inactive:=true \
  > /tmp/franka_surface_scan_spawner.log 2>&1 &
SPAWNER_PID=$!

log "Waiting for surface_impedance_controller to load inactive"
wait_for_controllers 60 "surface_impedance_controller" "inactive" || {
  echo "surface_impedance_controller did not load inactive" >&2
  exit 1
}

if (( RECORD_DEBUG )); then
  ros2 bag record \
    -o "$DBG_DIR/bag" \
    /clock \
    /joint_states \
    /dynamic_joint_states \
    /tf \
    /tf_static \
    /parameter_events \
    /rosout \
    /surface_impedance_controller/debug_state \
    /surface_impedance_controller/debug_markers \
    /surface_impedance_controller/transition_event \
    /joint_impedance_example_controller/transition_event \
    > /tmp/franka_surface_scan_bag.log 2>&1 &
  BAG_PID=$!
  sleep 3
fi

if (( AUTO_SWITCH )); then
  log "Switching from joint_impedance_example_controller to surface_impedance_controller"
  ros2 control switch_controllers \
    --deactivate joint_impedance_example_controller \
    --activate surface_impedance_controller \
    --strict \
    --activate-asap
  log "Surface scan is active. Press Ctrl-C to stop everything."
else
  log "Surface controller is loaded inactive. Activate it manually when ready."
fi

wait "$LAUNCH_PID"

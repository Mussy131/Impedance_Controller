# franka_surface_impedance_controller

`franka_surface_impedance_controller` is a ROS 2 `ros2_control` controller plugin for Franka robots that focuses on surface-following and hemisphere scanning with Cartesian impedance. The package is currently set up first for Gazebo simulation, with a paper-style surface task (`paper_surface_scan`) running on top of a simplified hemisphere model.

The repository contains:

- A controller plugin implemented in C++.
- Launch files for spawning the controller and a matching hemisphere model in Gazebo.
- A helper script that brings the robot up from a clean home-pose simulation and switches controllers automatically.

## Highlights

- `ros2_control` controller plugin: `franka_surface_impedance_controller/SurfaceImpedanceController`
- Default control mode: `paper_surface_scan`
- Default track generator: `paper_surface_lissajous`
- Default surface model: hemisphere with center `[0.45, 0.0, 0.02]`, radius `0.10 m`
- Simulation backend: URDF + KDL
- Optional hardware-oriented backend in code: Franka semantic interfaces

## Repository Layout

| Path | Purpose |
| --- | --- |
| `src/`, `include/` | Controller implementation and public headers |
| `config/surface_impedance_controller.yaml` | Main controller and controller_manager parameters |
| `config/surface_debug_markers.rviz` | RViz setup for debug markers |
| `launch/spawn_surface_impedance_controller.launch.py` | Spawn the controller and optionally spawn the hemisphere model |
| `launch/gazebo_surface_scan_from_home.launch.py` | Launch Gazebo, robot description, broadcasters, and the surface scan stack |
| `scripts/start_surface_scan_from_home.sh` | End-to-end helper for clean startup, controller handoff, and optional debug recording |
| `scripts/render_surface_scan_figure.py` | Render a publication-style scan trajectory figure from a recorded rosbag |
| `scripts/render_surface_scan_metrics_figure.py` | Render signed-distance, orientation-error, and torque metrics for one scan cycle |
| `models/hemi_surface/` | Gazebo model for the hemisphere target |

## Requirements

This package is intended to live inside a ROS 2 Humble workspace that already contains the Franka and Gazebo dependencies used by the launch files.

At minimum, the runtime workflow expects:

- ROS 2 Humble
- `controller_manager` / `ros2_control`
- `franka_ros2`
- `franka_description`
- `franka_gazebo_bringup`
- `ros_gz_sim`
- `xacro`
- `rviz2`

The helper script and default YAML are currently tuned for the `fer` arm variant in simulation.

## Build

From the workspace root:

```bash
cd /home/hchen82/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select franka_surface_impedance_controller
source install/setup.bash
```

## Quick Start

### 1. Launch the Gazebo scan stack directly

This path is useful when you want a pure launch-file workflow.

```bash
source /opt/ros/humble/setup.bash
source /home/hchen82/ros2_ws/install/setup.bash
ros2 launch franka_surface_impedance_controller gazebo_surface_scan_from_home.launch.py
```

Useful launch arguments:

- `arm_id:=fer`
- `tcp_xyz:="0 0 0.115"`
- `start_rviz:=true|false`
- `spawn_hemisphere:=true|false`
- `params_file:=/absolute/path/to/surface_impedance_controller.yaml`

### 2. Spawn the controller into an existing controller manager

If Gazebo and the robot are already running, use the spawner launch directly:

```bash
ros2 launch franka_surface_impedance_controller spawn_surface_impedance_controller.launch.py \
  controller_manager:=/controller_manager \
  params_file:=/home/hchen82/ros2_ws/src/franka_surface_impedance_controller/config/surface_impedance_controller.yaml \
  start_rviz:=true \
  spawn_hemisphere:=true \
  spawn_surface_inactive:=true
```

Useful arguments for this launch file:

- `spawn_surface_inactive:=true|false`
- `spawn_joint_state_broadcaster:=true|false`
- `spawn_franka_robot_state_broadcaster:=true|false`
- `hemisphere_x:=from_params|<x>`
- `hemisphere_y:=from_params|<y>`
- `hemisphere_z:=from_params|<z>`

When `hemisphere_x/y/z` are left as `from_params`, the launch file reads `surface.hemisphere.center` from the YAML config so the visual model and controller use the same target geometry.

## Configuration

The main configuration file is [`config/surface_impedance_controller.yaml`](config/surface_impedance_controller.yaml).

Important parameter groups:

| Group | Purpose |
| --- | --- |
| `control` | High-level mode selection |
| `impedance` | Cartesian translational and rotational stiffness/damping |
| `scan` | Scan trajectory generator, reachable sector, speed limits, acquire/reacquire logic |
| `outer` | Surface-following outer-loop parameters |
| `paper` | Paper-style task gains in surface coordinates |
| `nullspace` | Redundancy stabilization |
| `safety` | Torque clamp, torque-rate clamp, and controller-switch protection |
| `surface` | Surface type and hemisphere geometry |
| `debug` | Marker and numeric debug stream publishing |

Default modes currently recognized by the controller:

- `paper_p`
- `cartesian_surface`
- `hemi_scan_cartesian`
- `paper_surface_scan`

The current default is:

```yaml
surface_impedance_controller:
  ros__parameters:
    control:
      mode: paper_surface_scan
```

Key scan defaults:

- `scan.track_generator: paper_surface_lissajous`
- `scan.theta_min/max: [-1.50, 1.50]`
- `scan.phi_min/max: [0.25, 1.00]`
- `scan.max_linear_speed: 0.030`
- `scan.max_angular_speed: 0.35`

Key surface defaults:

```yaml
surface:
  type: hemisphere
  hemisphere:
    center: [0.45, 0.0, 0.02]
    radius: 0.10
    axis: [0.0, 0.0, 1.0]
```

## Interfaces and Debugging

The controller publishes the following custom debug topics when enabled:

- `/surface_impedance_controller/debug_markers`
- `/surface_impedance_controller/debug_state`

The default config enables both:

```yaml
debug:
  publish_state: true
  publish_markers: true
  publish_rate_hz: 100.0
```

`debug_state` is a numeric `std_msgs/msg/Float64MultiArray` stream that includes scan time, control mode code, contact/surface state flags, signed distance to the hemisphere, current and desired TCP pose terms, tool-axis alignment error, torque norm, and several reacquire/progress diagnostics.

Useful inspection commands:

```bash
ros2 control list_controllers
ros2 topic echo /surface_impedance_controller/debug_state
ros2 topic hz /surface_impedance_controller/debug_markers
```

For RViz, the package includes:

- `config/surface_debug_markers.rviz`

## Recording and Figure Generation

To record a debug session:

```bash
ros2 run franka_surface_impedance_controller start_surface_scan_from_home.sh --record-debug
```

Recorded sessions are stored under:

```text
debug/surface_scan_debug_<timestamp>/
```

The helper script records, among others:

- `/clock`
- `/joint_states`
- `/dynamic_joint_states`
- `/tf`
- `/tf_static`
- `/surface_impedance_controller/debug_state`
- `/surface_impedance_controller/debug_markers`
- `/surface_impedance_controller/transition_event`
- `/joint_impedance_example_controller/transition_event`

To render a scan figure from the newest bag:

```bash
python3 scripts/render_surface_scan_figure.py
python3 scripts/render_surface_scan_metrics_figure.py
```

To render from a specific bag:

```bash
python3 scripts/render_surface_scan_figure.py \
  --bag debug/surface_scan_debug_<timestamp>/bag \
  --output docs/figures/hemisphere_scan_debug.png

python3 scripts/render_surface_scan_metrics_figure.py \
  --bag debug/surface_scan_debug_<timestamp>/bag \
  --output docs/figures/surface_scan_metrics.png
```

## Current Assumptions and Limitations

- The provided launch and helper workflows are simulation-first.
- `scripts/start_surface_scan_from_home.sh` currently hardcodes the workspace root as `/home/hchen82/ros2_ws`, the package root as `/home/hchen82/ros2_ws/src/franka_surface_impedance_controller`, and the arm id as `fer`.
- The default simulation backend uses `use_franka_semantic: false` and relies on an expanded URDF plus KDL.
- `surface.type` is effectively implemented for `hemisphere`.
- The default launch flow assumes the controller manager is available at `/controller_manager`.

The controller code also contains a Franka semantic backend for real hardware integration, but the supplied launch files and helper script are currently tuned around Gazebo and the KDL simulation path.

## License

Apache-2.0

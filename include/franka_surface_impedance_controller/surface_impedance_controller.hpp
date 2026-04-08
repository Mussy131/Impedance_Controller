#pragma once

#include <memory>
#include <string>
#include <vector>

#include <controller_interface/controller_interface.hpp>
#include <hardware_interface/loaned_command_interface.hpp>
#include <hardware_interface/loaned_state_interface.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/state.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <franka/model.h>
#include <franka_semantic_components/franka_robot_model.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry>

// KDL（仿真后端）
#include <kdl/tree.hpp>
#include <kdl/chain.hpp>
#include <kdl/jntarray.hpp>
#include <kdl/jacobian.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainjnttojacsolver.hpp>


namespace franka_surface_impedance_controller {

class SurfaceImpedanceController : public controller_interface::ControllerInterface {
public:
  SurfaceImpedanceController() = default;

  controller_interface::CallbackReturn on_init() override;

  controller_interface::InterfaceConfiguration command_interface_configuration() const override;
  controller_interface::InterfaceConfiguration state_interface_configuration() const override;

  controller_interface::CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;
  controller_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;
  controller_interface::CallbackReturn on_deactivate(const rclcpp_lifecycle::State& previous_state) override;

  controller_interface::return_type update(const rclcpp::Time& time,
                                           const rclcpp::Duration& period) override;

private:
  // ----- Parameters -----
  std::string arm_id_{"panda"};
  std::vector<std::string> joint_names_;

  // KDL caches for the simulation backend

  // Backend selection:
  // - true  : real robot (Franka semantic interfaces: arm_id/robot_model, arm_id/robot_state)
  // - false : simulation (Ignition Gazebo): use URDF+KDL for FK/Jacobian
  bool use_franka_semantic_{true};

  // Simulation backend (KDL) parameters
  std::string urdf_path_{""};  // path to expanded .urdf file (NOT xacro)
  std::string base_link_{""};  // KDL chain base link name
  std::string ee_link_{""};    // KDL chain end-effector link name

  // Cartesian impedance gains
  double k_trans_{300.0};
  double d_trans_{40.0};
  double k_rot_{30.0};
  double d_rot_{5.0};

  // Joint damping (nullspace / regularization)
  double joint_damping_{0.2};

  // Outer loop (surface following)
  double d_des_{0.02};
  double k_outer_pos_{0.15};

   // ===== Paper-like surface-task impedance (p = [d, eps]) =====
  std::string control_mode_{"paper_p"};   // "paper_p", "cartesian_surface", "hemi_scan_cartesian", "paper_surface_scan"

  // p-task gains
  double k_d_{200.0};    // N/m
  double d_d_{40.0};     // N*s/m
  Eigen::Vector2d k_s_{150.0, 150.0};  // N/m in surface coordinates
  Eigen::Vector2d d_s_{25.0, 25.0};    // N*s/m in surface coordinates
  double k_eps_{8.0};    // (unitless->torque) gain for epsilon
  double d_eps_{1.0};    // damping for epsilon

  double d_max_{0.05};   // [m] enable paper mode only when |d| <= d_max
  bool enable_curvature_{true}; // approximate J_{eps v} for sphere

  // Hemisphere scan trajectory (theta/phi over time)
  // theta: azimuth around hemisphere axis [rad]
  // phi  : polar angle from hemisphere axis [rad], hemisphere side uses phi in [0, pi/2]
  double scan_theta_center_{0.0};
  double scan_theta_amp_{0.8};
  double scan_theta_freq_{0.03};   // [Hz]
  double scan_theta_phase_{0.0};   // [rad]
  bool scan_theta_orbit_{false};   // true: theta advances continuously around hemisphere
  bool scan_theta_smooth_turnaround_{true}; // smooth sector reversal instead of hard reflection
  double scan_theta_min_{-3.14159265358979323846}; // [rad] optional reachable azimuth sector lower bound
  double scan_theta_max_{3.14159265358979323846};  // [rad] optional reachable azimuth sector upper bound
  std::string scan_track_generator_{"legacy_angles"}; // "legacy_angles", "paper_surface_lissajous"
  Eigen::Vector2d scan_surface_lissajous_amplitude_{0.045, 0.045}; // [m] Eq. (23)-style path amplitude in surface coordinates
  Eigen::Vector2d scan_surface_lissajous_omega_{0.36, 0.48};       // [rad/s] Eq. (23): 3*omega0, 4*omega0 for omega0=0.12
  Eigen::Vector2d scan_surface_lissajous_phase_{0.0, 0.0};         // [rad]
  Eigen::Vector2d scan_surface_lissajous_offset_{0.0, 0.0};        // [m] optional bias in the surface parameter domain
  double scan_phi_center_{0.6};
  double scan_phi_amp_{0.2};
  double scan_phi_freq_{0.04};     // [Hz]
  double scan_phi_phase_{1.57};    // [rad]
  double scan_phi_min_{0.2};       // [rad]
  double scan_phi_max_{1.2};       // [rad]
  bool scan_reset_time_on_activate_{true};
  double scan_time_{0.0};          // [s]
  double scan_ramp_elapsed_{0.0};  // [s] accumulated scan progress used only for ramp-in
  double scan_ramp_time_{3.0};     // [s]
  double scan_phase_speed_scale_{1.0}; // [-] extra multiplier on scan phase advance after gating
  double scan_max_linear_speed_{0.03};   // [m/s] desired target slew limit
  double scan_max_angular_speed_{1.0};   // [rad/s] desired target slew limit
  double scan_acquire_max_linear_speed_{0.05};   // [m/s] faster pull-in speed before scan progress is enabled
  double scan_acquire_max_angular_speed_{0.35};  // [rad/s] faster orientation slew during scan acquisition
  double scan_acquire_release_surface_tol_{0.005};   // [m] require near-surface convergence before switching into paper scan
  double scan_acquire_reacquire_surface_tol_{0.010}; // [m] fall back to acquisition if the TCP drifts away from the surface
  double scan_acquire_reacquire_surface_time_{0.12}; // [s] require surface-distance violation to persist before dropping back to acquisition
  double scan_acquire_release_band_tol_{0.08};   // [rad] allow scan start once the contact patch is close enough to the valid band
  double scan_acquire_reacquire_band_tol_{0.14}; // [rad] fall back to acquisition if the contact patch drifts too far outside the band
  double scan_acquire_hover_surface_offset_{0.008}; // [m] stay slightly above the surface while recovering back into the valid scan band
  double scan_acquire_recovery_height_margin_{0.015}; // [m] reacquire toward a band interior above the minimum scan height
  double scan_acquire_recovery_max_angular_speed_{0.18}; // [rad/s] slow orientation slews while recovering from low-height/off-band states
  double scan_max_position_error_{0.10}; // [m] cap |p_d - p| to avoid unsafe pull-in
  double scan_max_radial_error_{0.10};   // [m] cap radial pull-in relative to local surface normal
  double scan_max_tangential_error_{0.10}; // [m] cap tangential scan lead along the surface
  double scan_max_orientation_error_{0.7}; // [rad] cap orientation tracking error
  bool scan_progress_gate_{true};        // pause scan-time when tracking falls behind
  double scan_progress_pos_tol_{0.02};   // [m] allow scan-time advance only if |p_d-p| <= tol
  double scan_progress_ori_tol_{0.20};   // [rad] allow scan-time advance only if axis error <= tol
  double scan_progress_hard_stop_pos_tol_{0.05}; // [m] hard-freeze scan-time if the pose gap grows too large
  double scan_progress_hard_stop_ori_tol_{0.35}; // [rad] hard-freeze scan-time if axis tracking falls too far behind
  double scan_progress_hard_stop_surface_dist_tol_{0.03}; // [m] only hard-stop once the TCP is near the surface
  double scan_progress_band_tol_{0.05}; // [rad] slow scan-time when contact strays outside the preferred band
  double scan_progress_hard_stop_band_tol_{0.12}; // [rad] hard-freeze scan-time once contact drifts too far outside the band
  double scan_progress_cruise_min_scale_{0.0}; // [-] minimum scan-time advance scale when tracking quality is comfortably inside the gate
  double scan_progress_cruise_quality_threshold_{0.85}; // [-] enable cruise floor only once non-phase tracking ratios exceed this quality
  double scan_progress_hard_stop_surface_reacquire_time_{0.12}; // [s] if surface/band hard-stop persists this long, fall back to acquisition
  double scan_progress_hard_stop_reacquire_time_{0.20}; // [s] if orientation hard-stop persists this long, fall back to acquisition
  Eigen::Vector3d scan_tool_axis_local_{0.0, 0.0, -1.0}; // ee local axis to align with surface normal
  Eigen::Vector3d scan_tool_axis_runtime_{0.0, 0.0, -1.0}; // resolved axis used in update/debug
  bool scan_align_inward_normal_{true};   // true: align tool axis with -n; false: +n
  bool scan_hold_current_roll_{false};    // when false, use minimal-twist axis alignment to avoid unnecessary wrist roll
  bool scan_auto_pick_tool_axis_on_activate_{true}; // pick +/-X/Y/Z local axis nearest to target on activate
  bool scan_seed_from_current_on_activate_{true};   // seed (theta,phi) from current pose on activate
  bool scan_lock_lissajous_mapping_{true}; // keep the paper-scan chart/orbit fixed across activate/acquire/recovery
  double scan_seed_blend_time_{1.5};     // [s] blend from current surface normal into the reachable scan band
  double scan_min_height_above_base_{0.0}; // [m] keep scan points at least this high above the hemisphere base plane
  Eigen::Vector3d scan_roll_hint_base_{0.0, 1.0, 0.0}; // latched base-frame secondary direction used to disambiguate free roll
  bool scan_roll_hint_valid_{false};
  double scan_theta_seed_{0.0};
  double scan_phi_seed_{0.6};
  double scan_theta_start_{0.0};         // [rad] actual start angle before band/sector clamping
  double scan_phi_start_{0.6};           // [rad] actual start polar angle before band clamping
  Eigen::Vector3d scan_surface_origin_normal_{0.0, 0.0, 1.0}; // unit normal defining the local APRP chart origin
  bool scan_surface_origin_normal_valid_{false};
  Eigen::Vector3d scan_normal_cmd_{0.0, 0.0, 1.0}; // filtered commanded surface normal shared by acquisition and scan
  bool scan_normal_cmd_valid_{false};
  Eigen::Vector2d scan_surface_seed_{0.0, 0.0};    // [m] center of the paper-style path in the local chart
  Eigen::Vector2d scan_surface_start_{0.0, 0.0};   // [m] local-chart start point used for blend-in
  Eigen::Vector2d scan_surface_cmd_{0.0, 0.0};     // [m] commanded paper-scan point after rate limiting
  bool scan_surface_cmd_valid_{false};
  Eigen::Vector2d scan_surface_raw_prev_{0.0, 0.0}; // [m] previous raw APRP reference before slew limiting
  bool scan_surface_raw_prev_valid_{false};
  Eigen::Vector2d scan_surface_actual_prev_{0.0, 0.0}; // [m] previous realized surface point from the TCP
  bool scan_surface_actual_prev_valid_{false};
  double scan_surface_phase_progress_ratio_{1.0};   // [-] achieved surface motion relative to raw APRP progress
  Eigen::Vector2d scan_surface_des_prev_{0.0, 0.0};
  bool scan_surface_des_prev_valid_{false};
  double scan_seed_blend_elapsed_{0.0};  // [s] wall-time elapsed since the current scan seed was latched
  double scan_acquire_surface_reacquire_elapsed_{0.0};
  double scan_acquire_band_reacquire_elapsed_{0.0};
  bool scan_progress_hard_stop_active_{false};
  double scan_progress_hard_stop_surface_elapsed_{0.0};
  double scan_progress_hard_stop_ori_elapsed_{0.0};
  bool scan_recovery_hold_prev_{false};
  bool paper_surface_acquired_{false};
  bool debug_scan_seed_blend_active_{false};
  bool debug_paper_acquire_active_{false};
  double debug_scan_surface_err_{0.0};
  double debug_scan_surface_dist_err_{0.0};
  double debug_scan_surface_ori_err_{0.0};
  double debug_scan_cmd_ori_err_{0.0};
  double debug_scan_band_err_{0.0};
  double debug_scan_reacquire_reason_{0.0};
  double debug_scan_roll_err_{0.0};
  double debug_scan_roll_rate_{0.0};

  // Nullspace (stabilize redundancy like paper mentions)
  bool nullspace_enable_{true};
  double null_kp_{10.0};
  double null_kd_{2.0};
  double nullspace_damping_{0.02};
  bool null_q_des_initialized_{false};
  Eigen::Matrix<double, 7, 1> null_q_des_ = Eigen::Matrix<double, 7, 1>::Zero();

  // Safety
  double max_tau_{40.0}; // [Nm], <=0 means no clamp
  double max_tau_rate_{80.0}; // [Nm/s], <=0 means no slew-rate clamp
  bool switch_protection_enable_{true};
  double switch_hold_time_{0.4};   // [s] hold cartesian target after activate
  double switch_blend_time_{1.0};  // [s] blend commanded torque from previous controller
  double switch_elapsed_time_{0.0};
  bool tau_prev_valid_{false};
  Eigen::Matrix<double, 7, 1> tau_prev_cmd_ = Eigen::Matrix<double, 7, 1>::Zero();
  Eigen::Matrix<double, 7, 1> tau_start_cmd_ = Eigen::Matrix<double, 7, 1>::Zero();

  // Debug: joint PD mode (no model-based cartesian impedance)
  bool debug_joint_pd_mode_{false};

  // Debug PD parameters/target
  double debug_pd_kp_{200.0};
  double debug_pd_kd_{20.0};
  bool debug_pd_hold_current_on_activate_{true};
  bool debug_pd_target_initialized_{false};
  Eigen::Matrix<double, 7, 1> debug_pd_q_target_ = Eigen::Matrix<double, 7, 1>::Zero();

  // Debug telemetry (RViz markers + numeric stream)
  bool debug_publish_markers_{true};
  bool debug_publish_state_{true};
  double debug_publish_rate_hz_{20.0};
  double debug_publish_accum_{0.0};
  std::shared_ptr<rclcpp::Publisher<visualization_msgs::msg::MarkerArray>> debug_marker_pub_;
  std::shared_ptr<rclcpp::Publisher<std_msgs::msg::Float64MultiArray>> debug_state_pub_;

  // Surface model: hemisphere
  double hemisphere_radius_{0.10};
  Eigen::Vector3d hemisphere_center_{0.65, 0.0, 0.02};
  Eigen::Vector3d hemisphere_axis_{0.0, 0.0, 1.0}; // cut plane normal (+axis side is valid)

  // ----- Precomputed gains -----
  Eigen::Matrix<double, 6, 6> K_cart_ = Eigen::Matrix<double, 6, 6>::Zero();
  Eigen::Matrix<double, 6, 6> D_cart_ = Eigen::Matrix<double, 6, 6>::Zero();
  Eigen::Matrix<double, 7, 1> D_joint_ = Eigen::Matrix<double, 7, 1>::Zero();

  // ----- Desired pose -----
  bool desired_initialized_{false};
  Eigen::Vector3d p_d_ = Eigen::Vector3d::Zero();
  Eigen::Quaterniond q_d_ = Eigen::Quaterniond::Identity();

  // ----- Franka semantic component -----
  std::unique_ptr<franka_semantic_components::FrankaRobotModel> franka_robot_model_;

  // ----- KDL model (simulation backend) -----
  bool kdl_initialized_{false};
  KDL::Tree kdl_tree_;
  KDL::Chain kdl_chain_;
  std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk_solver_;
  std::unique_ptr<KDL::ChainJntToJacSolver> jac_solver_;
  KDL::JntArray kdl_q_;
  KDL::Jacobian kdl_J_;

  // Cached interface pointers (filled in on_activate)
  std::vector<hardware_interface::LoanedCommandInterface*> cmd_effort_;
  std::vector<hardware_interface::LoanedStateInterface*> state_pos_;
  std::vector<hardware_interface::LoanedStateInterface*> state_vel_;

  // ----- Helpers -----
  hardware_interface::LoanedCommandInterface* get_command_handle_(const std::string& joint,
                                                                 const std::string& interface);
  hardware_interface::LoanedStateInterface* get_state_handle_(const std::string& joint,
                                                             const std::string& interface);

  Eigen::Matrix<double, 7, 1> get_q_() const;
  Eigen::Matrix<double, 7, 1> get_dq_() const;

  bool query_hemisphere_true_(const Eigen::Vector3d& p_base, double& signed_distance,
                              Eigen::Vector3d& normal_base) const;
  bool query_hemisphere_proxy_(const Eigen::Vector3d& p_base, Eigen::Vector3d& proxy_base,
                               double& signed_distance, Eigen::Vector3d& normal_base) const;
  bool query_hemisphere_(const Eigen::Vector3d& p_base, double& signed_distance,
                         Eigen::Vector3d& normal_base) const;
  bool build_hemisphere_basis_(Eigen::Vector3d& axis, Eigen::Vector3d& e1, Eigen::Vector3d& e2) const;
  double effective_scan_phi_max_() const;
  bool normal_to_hemi_angles_(const Eigen::Vector3d& normal_base, double& theta, double& phi) const;
  bool project_normal_to_scan_band_(
    const Eigen::Vector3d& normal_in, Eigen::Vector3d& normal_out,
    double* theta_out = nullptr, double* phi_out = nullptr) const;
  bool normal_to_hemi_surface_coords_(const Eigen::Vector3d& normal_base, Eigen::Vector2d& s) const;
  bool hemi_surface_coords_to_normal_(const Eigen::Vector2d& s, Eigen::Vector3d& normal_base) const;

  // KDL helpers (used when use_franka_semantic_ == false)
  bool init_kdl_from_urdf_file_();
  bool compute_fk_jacobian_kdl_(const Eigen::Matrix<double, 7, 1>& q,
                                Eigen::Vector3d& p_base,
                                Eigen::Quaterniond& q_base,
                                Eigen::Matrix<double, 6, 7>& J_out);
};

}  // namespace franka_surface_impedance_controller

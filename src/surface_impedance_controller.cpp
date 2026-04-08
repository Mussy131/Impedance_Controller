#include "franka_surface_impedance_controller/surface_impedance_controller.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <fstream>
#include <sstream>

#include <kdl_parser/kdl_parser.hpp>
#include <pluginlib/class_list_macros.hpp>

namespace franka_surface_impedance_controller {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kTwoPi = 2.0 * kPi;

std::vector<std::string> default_joint_names(const std::string & arm_id) {
  return {
    arm_id + "_joint1",
    arm_id + "_joint2",
    arm_id + "_joint3",
    arm_id + "_joint4",
    arm_id + "_joint5",
    arm_id + "_joint6",
    arm_id + "_joint7",
  };
}

Eigen::Vector3d vec3_from_param(const std::vector<double> & v, const Eigen::Vector3d & fallback) {
  if (v.size() != 3) {
    return fallback;
  }
  return Eigen::Vector3d(v[0], v[1], v[2]);
}

Eigen::Vector2d vec2_from_param(const std::vector<double> & v, const Eigen::Vector2d & fallback) {
  if (v.size() != 2) {
    return fallback;
  }
  return Eigen::Vector2d(v[0], v[1]);
}

Eigen::Vector3d preferred_secondary_hint_axis(const Eigen::Vector3d & primary) {
  return (std::abs(primary.dot(Eigen::Vector3d::UnitX())) < 0.8) ?
    Eigen::Vector3d::UnitX() : Eigen::Vector3d::UnitY();
}

bool resolve_secondary_axis_local(
  const Eigen::Vector3d & primary,
  Eigen::Vector3d & secondary_out) {
  if (primary.norm() < 1e-9) {
    return false;
  }

  const Eigen::Vector3d u = primary.normalized();
  Eigen::Vector3d v = preferred_secondary_hint_axis(u);
  v -= v.dot(u) * u;
  if (v.norm() < 1e-9) {
    const Eigen::Vector3d seed =
      (std::abs(u.z()) < 0.9) ? Eigen::Vector3d::UnitZ() : Eigen::Vector3d::UnitX();
    v = seed - seed.dot(u) * u;
    if (v.norm() < 1e-9) {
      return false;
    }
  }
  secondary_out = v.normalized();
  return true;
}

bool compute_roll_hint_in_base(
  const Eigen::Quaterniond & q_reference,
  const Eigen::Vector3d & local_primary,
  Eigen::Vector3d & base_hint_out) {
  Eigen::Vector3d local_secondary = Eigen::Vector3d::Zero();
  if (!resolve_secondary_axis_local(local_primary, local_secondary)) {
    return false;
  }
  base_hint_out = q_reference.toRotationMatrix() * local_secondary;
  if (base_hint_out.norm() < 1e-9) {
    return false;
  }
  base_hint_out.normalize();
  return true;
}

double wrap_angle_pi(double angle) {
  double wrapped = std::fmod(angle + kPi, kTwoPi);
  if (wrapped < 0.0) {
    wrapped += kTwoPi;
  }
  return wrapped - kPi;
}

double unwrap_angle_near(double angle, double reference) {
  return reference + wrap_angle_pi(angle - reference);
}

bool theta_sector_active(double theta_min, double theta_max) {
  return std::isfinite(theta_min) && std::isfinite(theta_max) &&
         (theta_max > theta_min + 1e-6) && ((theta_max - theta_min) < (kTwoPi - 1e-6));
}

double clamp_theta_to_sector(double theta, double theta_min, double theta_max) {
  if (!theta_sector_active(theta_min, theta_max)) {
    return theta;
  }
  const double center = 0.5 * (theta_min + theta_max);
  return std::clamp(unwrap_angle_near(theta, center), theta_min, theta_max);
}

double reflect_theta_in_sector(double theta, double theta_min, double theta_max) {
  if (!theta_sector_active(theta_min, theta_max)) {
    return theta;
  }
  const double center = 0.5 * (theta_min + theta_max);
  const double span = theta_max - theta_min;
  const double period = 2.0 * span;
  double phase = std::fmod(unwrap_angle_near(theta, center) - theta_min, period);
  if (phase < 0.0) {
    phase += period;
  }
  return (phase <= span) ? (theta_min + phase) : (theta_max - (phase - span));
}

double smooth_theta_in_sector(
  double theta_seed, double phase, double theta_min, double theta_max) {
  if (!theta_sector_active(theta_min, theta_max)) {
    return theta_seed + phase;
  }
  const double center = 0.5 * (theta_min + theta_max);
  const double half_span = 0.5 * (theta_max - theta_min);
  if (half_span <= 1e-9) {
    return center;
  }
  const double theta_seed_clamped = clamp_theta_to_sector(theta_seed, theta_min, theta_max);
  const double seed_ratio = std::clamp((theta_seed_clamped - center) / half_span, -1.0, 1.0);
  const double seed_phase = std::asin(seed_ratio);
  return center + half_span * std::sin(seed_phase + phase);
}

template <int TaskDim>
Eigen::Matrix<double, 7, 7> damped_nullspace_projector(
  const Eigen::Matrix<double, TaskDim, 7> & J_task, double damping) {
  Eigen::Matrix<double, TaskDim, TaskDim> JJt = J_task * J_task.transpose();
  const double lambda = std::max(0.0, damping);
  JJt.diagonal().array() += std::max(1e-9, lambda * lambda);

  Eigen::LDLT<Eigen::Matrix<double, TaskDim, TaskDim>> ldlt(JJt);
  if (ldlt.info() != Eigen::Success) {
    return Eigen::Matrix<double, 7, 7>::Zero();
  }

  const Eigen::Matrix<double, TaskDim, TaskDim> JJt_inv =
    ldlt.solve(Eigen::Matrix<double, TaskDim, TaskDim>::Identity());
  if (ldlt.info() != Eigen::Success || !JJt_inv.allFinite()) {
    return Eigen::Matrix<double, 7, 7>::Zero();
  }

  const Eigen::Matrix<double, 7, 7> N =
    Eigen::Matrix<double, 7, 7>::Identity() - J_task.transpose() * JJt_inv * J_task;
  return N.allFinite() ? N : Eigen::Matrix<double, 7, 7>::Zero();
}

bool build_surface_chart_basis(
  const Eigen::Vector3d & origin_normal,
  const Eigen::Vector3d & preferred_axis,
  Eigen::Vector3d & t1,
  Eigen::Vector3d & t2) {
  if (origin_normal.norm() < 1e-9) {
    return false;
  }

  const Eigen::Vector3d n = origin_normal.normalized();
  Eigen::Vector3d hint = preferred_axis;
  if (hint.norm() < 1e-9 || std::abs(hint.normalized().dot(n)) > 0.999) {
    hint = (std::abs(n.z()) < 0.9) ? Eigen::Vector3d::UnitZ() : Eigen::Vector3d::UnitX();
  }

  t1 = hint - hint.dot(n) * n;
  if (t1.norm() < 1e-9) {
    const Eigen::Vector3d fallback =
      (std::abs(n.x()) < 0.9) ? Eigen::Vector3d::UnitX() : Eigen::Vector3d::UnitY();
    t1 = fallback - fallback.dot(n) * n;
    if (t1.norm() < 1e-9) {
      return false;
    }
  }
  t1.normalize();

  t2 = n.cross(t1);
  if (t2.norm() < 1e-9) {
    return false;
  }
  t2.normalize();

  t1 = t2.cross(n);
  if (t1.norm() < 1e-9) {
    return false;
  }
  t1.normalize();
  return true;
}

bool local_surface_coords_to_normal(
  const Eigen::Vector3d & origin_normal,
  const Eigen::Vector3d & preferred_axis,
  double radius,
  const Eigen::Vector2d & s_local,
  Eigen::Vector3d & normal_out) {
  if (!(std::isfinite(radius) && radius > 1e-6) || !s_local.allFinite()) {
    return false;
  }

  Eigen::Vector3d t1 = Eigen::Vector3d::Zero();
  Eigen::Vector3d t2 = Eigen::Vector3d::Zero();
  if (!build_surface_chart_basis(origin_normal, preferred_axis, t1, t2)) {
    return false;
  }

  const Eigen::Vector3d n0 = origin_normal.normalized();
  const Eigen::Vector3d tangent = s_local.x() * t1 + s_local.y() * t2;
  const double arc = tangent.norm();
  if (arc < 1e-9) {
    normal_out = n0;
    return true;
  }

  const double alpha = arc / radius;
  const Eigen::Vector3d tangent_dir = tangent / arc;
  normal_out = std::cos(alpha) * n0 + std::sin(alpha) * tangent_dir;
  if (normal_out.norm() < 1e-9) {
    return false;
  }
  normal_out.normalize();
  return normal_out.allFinite();
}

bool normal_to_local_surface_coords(
  const Eigen::Vector3d & origin_normal,
  const Eigen::Vector3d & preferred_axis,
  double radius,
  const Eigen::Vector3d & normal,
  Eigen::Vector2d & s_local) {
  if (!(std::isfinite(radius) && radius > 1e-6) || normal.norm() < 1e-9) {
    return false;
  }

  Eigen::Vector3d t1 = Eigen::Vector3d::Zero();
  Eigen::Vector3d t2 = Eigen::Vector3d::Zero();
  if (!build_surface_chart_basis(origin_normal, preferred_axis, t1, t2)) {
    return false;
  }

  const Eigen::Vector3d n0 = origin_normal.normalized();
  const Eigen::Vector3d n = normal.normalized();
  const double dot_val = std::clamp(n0.dot(n), -1.0, 1.0);
  const double alpha = std::acos(dot_val);
  if (!std::isfinite(alpha)) {
    return false;
  }
  if (alpha < 1e-9) {
    s_local.setZero();
    return true;
  }

  const Eigen::Vector3d tangent = n - dot_val * n0;
  const double tangent_norm = tangent.norm();
  if (tangent_norm < 1e-9) {
    return false;
  }

  const Eigen::Vector3d tangent_dir = tangent / tangent_norm;
  s_local.x() = radius * alpha * tangent_dir.dot(t1);
  s_local.y() = radius * alpha * tangent_dir.dot(t2);
  return s_local.allFinite();
}

}  // namespace

controller_interface::CallbackReturn SurfaceImpedanceController::on_init() {
  try {
    auto_declare<std::string>("arm_id", arm_id_);
    auto_declare<std::vector<std::string>>("joint_names", {});

    auto_declare<double>("impedance.k_trans", k_trans_);
    auto_declare<double>("impedance.d_trans", d_trans_);
    auto_declare<double>("impedance.k_rot", k_rot_);
    auto_declare<double>("impedance.d_rot", d_rot_);

    auto_declare<double>("joint_damping", joint_damping_);

    auto_declare<double>("outer.d_des", d_des_);
    auto_declare<double>("outer.k_outer_pos", k_outer_pos_);

    auto_declare<bool>("debug_joint_pd_mode", debug_joint_pd_mode_);
    auto_declare<double>("debug_pd.kp", debug_pd_kp_);
    auto_declare<double>("debug_pd.kd", debug_pd_kd_);
    auto_declare<bool>("debug_pd.hold_current_on_activate", debug_pd_hold_current_on_activate_);
    auto_declare<bool>("debug.publish_markers", debug_publish_markers_);
    auto_declare<bool>("debug.publish_state", debug_publish_state_);
    auto_declare<double>("debug.publish_rate_hz", debug_publish_rate_hz_);

    // Backend selection parameters
    auto_declare<bool>("use_franka_semantic", true);
    auto_declare<std::string>("urdf_path", "");
    auto_declare<std::string>("base_link", "");
    auto_declare<std::string>("ee_link", "");

    auto_declare<std::string>("surface.type", std::string("hemisphere"));
    auto_declare<double>("surface.hemisphere.radius", hemisphere_radius_);
    auto_declare<std::vector<double>>(
      "surface.hemisphere.center", {hemisphere_center_.x(), hemisphere_center_.y(), hemisphere_center_.z()});
    auto_declare<std::vector<double>>(
      "surface.hemisphere.axis", {hemisphere_axis_.x(), hemisphere_axis_.y(), hemisphere_axis_.z()});

    // Outer loop (surface following) parameters
    auto_declare<std::string>("control.mode", control_mode_);

    // Hemisphere scan trajectory
    auto_declare<double>("scan.theta_center", scan_theta_center_);
    auto_declare<double>("scan.theta_amp", scan_theta_amp_);
    auto_declare<double>("scan.theta_freq", scan_theta_freq_);
    auto_declare<double>("scan.theta_phase", scan_theta_phase_);
    auto_declare<bool>("scan.theta_orbit", scan_theta_orbit_);
    auto_declare<bool>("scan.theta_smooth_turnaround", scan_theta_smooth_turnaround_);
    auto_declare<double>("scan.theta_min", scan_theta_min_);
    auto_declare<double>("scan.theta_max", scan_theta_max_);
    auto_declare<std::string>("scan.track_generator", scan_track_generator_);
    auto_declare<std::vector<double>>(
      "scan.surface_lissajous.amplitude",
      {scan_surface_lissajous_amplitude_.x(), scan_surface_lissajous_amplitude_.y()});
    auto_declare<std::vector<double>>(
      "scan.surface_lissajous.omega",
      {scan_surface_lissajous_omega_.x(), scan_surface_lissajous_omega_.y()});
    auto_declare<std::vector<double>>(
      "scan.surface_lissajous.phase",
      {scan_surface_lissajous_phase_.x(), scan_surface_lissajous_phase_.y()});
    auto_declare<std::vector<double>>(
      "scan.surface_lissajous.offset",
      {scan_surface_lissajous_offset_.x(), scan_surface_lissajous_offset_.y()});
    auto_declare<double>("scan.phi_center", scan_phi_center_);
    auto_declare<double>("scan.phi_amp", scan_phi_amp_);
    auto_declare<double>("scan.phi_freq", scan_phi_freq_);
    auto_declare<double>("scan.phi_phase", scan_phi_phase_);
    auto_declare<double>("scan.phi_min", scan_phi_min_);
    auto_declare<double>("scan.phi_max", scan_phi_max_);
    auto_declare<bool>("scan.reset_time_on_activate", scan_reset_time_on_activate_);
    auto_declare<double>("scan.ramp_time", scan_ramp_time_);
    auto_declare<double>("scan.phase_speed_scale", scan_phase_speed_scale_);
    auto_declare<double>("scan.max_linear_speed", scan_max_linear_speed_);
    auto_declare<double>("scan.max_angular_speed", scan_max_angular_speed_);
    auto_declare<double>("scan.acquire_max_linear_speed", scan_acquire_max_linear_speed_);
    auto_declare<double>("scan.acquire_max_angular_speed", scan_acquire_max_angular_speed_);
    auto_declare<double>("scan.acquire_release_surface_tol", scan_acquire_release_surface_tol_);
    auto_declare<double>("scan.acquire_reacquire_surface_tol", scan_acquire_reacquire_surface_tol_);
    auto_declare<double>("scan.acquire_reacquire_surface_time", scan_acquire_reacquire_surface_time_);
    auto_declare<double>("scan.acquire_release_band_tol", scan_acquire_release_band_tol_);
    auto_declare<double>("scan.acquire_reacquire_band_tol", scan_acquire_reacquire_band_tol_);
    auto_declare<double>("scan.acquire_hover_surface_offset", scan_acquire_hover_surface_offset_);
    auto_declare<double>("scan.acquire_recovery_height_margin", scan_acquire_recovery_height_margin_);
    auto_declare<double>(
      "scan.acquire_recovery_max_angular_speed", scan_acquire_recovery_max_angular_speed_);
    auto_declare<double>("scan.max_position_error", scan_max_position_error_);
    auto_declare<double>("scan.max_radial_error", scan_max_radial_error_);
    auto_declare<double>("scan.max_tangential_error", scan_max_tangential_error_);
    auto_declare<double>("scan.max_orientation_error", scan_max_orientation_error_);
    auto_declare<bool>("scan.progress_gate", scan_progress_gate_);
    auto_declare<double>("scan.progress_pos_tol", scan_progress_pos_tol_);
    auto_declare<double>("scan.progress_ori_tol", scan_progress_ori_tol_);
    auto_declare<double>("scan.progress_hard_stop_pos_tol", scan_progress_hard_stop_pos_tol_);
    auto_declare<double>("scan.progress_hard_stop_ori_tol", scan_progress_hard_stop_ori_tol_);
    auto_declare<double>(
      "scan.progress_hard_stop_surface_dist_tol", scan_progress_hard_stop_surface_dist_tol_);
    auto_declare<double>("scan.progress_band_tol", scan_progress_band_tol_);
    auto_declare<double>("scan.progress_hard_stop_band_tol", scan_progress_hard_stop_band_tol_);
    auto_declare<double>("scan.progress_cruise_min_scale", scan_progress_cruise_min_scale_);
    auto_declare<double>(
      "scan.progress_cruise_quality_threshold", scan_progress_cruise_quality_threshold_);
    auto_declare<double>(
      "scan.progress_hard_stop_surface_reacquire_time",
      scan_progress_hard_stop_surface_reacquire_time_);
    auto_declare<double>(
      "scan.progress_hard_stop_reacquire_time", scan_progress_hard_stop_reacquire_time_);
    auto_declare<std::vector<double>>(
      "scan.tool_axis_local", {scan_tool_axis_local_.x(), scan_tool_axis_local_.y(), scan_tool_axis_local_.z()});
    auto_declare<bool>("scan.align_inward_normal", scan_align_inward_normal_);
    auto_declare<bool>("scan.hold_current_roll", scan_hold_current_roll_);
    auto_declare<bool>("scan.auto_pick_tool_axis_on_activate", scan_auto_pick_tool_axis_on_activate_);
    auto_declare<bool>("scan.seed_from_current_on_activate", scan_seed_from_current_on_activate_);
    auto_declare<bool>("scan.lock_lissajous_mapping", scan_lock_lissajous_mapping_);
    auto_declare<double>("scan.seed_blend_time", scan_seed_blend_time_);
    auto_declare<double>("scan.min_height_above_base", scan_min_height_above_base_);

    auto_declare<double>("paper.k_d", k_d_);
    auto_declare<double>("paper.d_d", d_d_);
    auto_declare<std::vector<double>>("paper.k_s", {k_s_.x(), k_s_.y()});
    auto_declare<std::vector<double>>("paper.d_s", {d_s_.x(), d_s_.y()});
    auto_declare<double>("paper.k_eps", k_eps_);
    auto_declare<double>("paper.d_eps", d_eps_);
    auto_declare<double>("paper.d_max", d_max_);
    auto_declare<bool>("paper.enable_curvature", enable_curvature_);

    auto_declare<bool>("nullspace.enable", nullspace_enable_);
    auto_declare<double>("nullspace.kp", null_kp_);
    auto_declare<double>("nullspace.kd", null_kd_);
    auto_declare<double>("nullspace.damping", nullspace_damping_);
    auto_declare<std::vector<double>>("nullspace.q_des", {}); // optional 7d target

    auto_declare<double>("safety.max_tau", max_tau_);
    auto_declare<double>("safety.max_tau_rate", max_tau_rate_);
    auto_declare<bool>("safety.switch_protection.enable", switch_protection_enable_);
    auto_declare<double>("safety.switch_protection.hold_time", switch_hold_time_);
    auto_declare<double>("safety.switch_protection.blend_time", switch_blend_time_);

  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_node()->get_logger(), "SurfaceImpedanceController on_init failed: %s", e.what());
    return controller_interface::CallbackReturn::ERROR;
  }
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration SurfaceImpedanceController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  const auto arm_id = get_node()->get_parameter("arm_id").as_string();
  auto joints = get_node()->get_parameter("joint_names").as_string_array();
  if (joints.empty()) {
    joints = default_joint_names(arm_id);
  }

  config.names.reserve(joints.size());
  for (const auto & j : joints) {
    config.names.push_back(j + "/effort");
  }
  return config;
}

controller_interface::InterfaceConfiguration SurfaceImpedanceController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  const auto arm_id = get_node()->get_parameter("arm_id").as_string();
  auto joints = get_node()->get_parameter("joint_names").as_string_array();
  if (joints.empty()) {
    joints = default_joint_names(arm_id);
  }

  config.names.reserve(joints.size() * 2 + 2);
  for (const auto & j : joints) {
    config.names.push_back(j + "/position");
    config.names.push_back(j + "/velocity");
  }

  // Franka semantic model expects these two special state interfaces.
  const bool use_semantic = get_node()->get_parameter("use_franka_semantic").as_bool();
  if (use_semantic) {
    config.names.push_back(arm_id + "/robot_model");
    config.names.push_back(arm_id + "/robot_state");
  }
  return config;
}

controller_interface::CallbackReturn SurfaceImpedanceController::on_configure(
  const rclcpp_lifecycle::State & /*previous_state*/) {
  try {
    arm_id_ = get_node()->get_parameter("arm_id").as_string();
    joint_names_ = get_node()->get_parameter("joint_names").as_string_array();
    if (joint_names_.empty()) {
      joint_names_ = default_joint_names(arm_id_);
    }

    // Backend selection 
    use_franka_semantic_ = get_node()->get_parameter("use_franka_semantic").as_bool();
    urdf_path_ = get_node()->get_parameter("urdf_path").as_string();
    base_link_ = get_node()->get_parameter("base_link").as_string();
    ee_link_ = get_node()->get_parameter("ee_link").as_string();

    k_trans_ = get_node()->get_parameter("impedance.k_trans").as_double();
    d_trans_ = get_node()->get_parameter("impedance.d_trans").as_double();
    k_rot_ = get_node()->get_parameter("impedance.k_rot").as_double();
    d_rot_ = get_node()->get_parameter("impedance.d_rot").as_double();

    joint_damping_ = get_node()->get_parameter("joint_damping").as_double();

    d_des_ = get_node()->get_parameter("outer.d_des").as_double();
    k_outer_pos_ = get_node()->get_parameter("outer.k_outer_pos").as_double();

    control_mode_ = get_node()->get_parameter("control.mode").as_string();

    scan_theta_center_ = get_node()->get_parameter("scan.theta_center").as_double();
    scan_theta_amp_ = get_node()->get_parameter("scan.theta_amp").as_double();
    scan_theta_freq_ = get_node()->get_parameter("scan.theta_freq").as_double();
    scan_theta_phase_ = get_node()->get_parameter("scan.theta_phase").as_double();
    scan_theta_orbit_ = get_node()->get_parameter("scan.theta_orbit").as_bool();
    scan_theta_smooth_turnaround_ =
      get_node()->get_parameter("scan.theta_smooth_turnaround").as_bool();
    scan_theta_min_ = get_node()->get_parameter("scan.theta_min").as_double();
    scan_theta_max_ = get_node()->get_parameter("scan.theta_max").as_double();
    scan_track_generator_ = get_node()->get_parameter("scan.track_generator").as_string();
    scan_surface_lissajous_amplitude_ = vec2_from_param(
      get_node()->get_parameter("scan.surface_lissajous.amplitude").as_double_array(),
      scan_surface_lissajous_amplitude_);
    scan_surface_lissajous_omega_ = vec2_from_param(
      get_node()->get_parameter("scan.surface_lissajous.omega").as_double_array(),
      scan_surface_lissajous_omega_);
    scan_surface_lissajous_phase_ = vec2_from_param(
      get_node()->get_parameter("scan.surface_lissajous.phase").as_double_array(),
      scan_surface_lissajous_phase_);
    scan_surface_lissajous_offset_ = vec2_from_param(
      get_node()->get_parameter("scan.surface_lissajous.offset").as_double_array(),
      scan_surface_lissajous_offset_);
    scan_phi_center_ = get_node()->get_parameter("scan.phi_center").as_double();
    scan_phi_amp_ = get_node()->get_parameter("scan.phi_amp").as_double();
    scan_phi_freq_ = get_node()->get_parameter("scan.phi_freq").as_double();
    scan_phi_phase_ = get_node()->get_parameter("scan.phi_phase").as_double();
    scan_phi_min_ = get_node()->get_parameter("scan.phi_min").as_double();
    scan_phi_max_ = get_node()->get_parameter("scan.phi_max").as_double();
    scan_reset_time_on_activate_ = get_node()->get_parameter("scan.reset_time_on_activate").as_bool();
    scan_ramp_time_ = get_node()->get_parameter("scan.ramp_time").as_double();
    scan_phase_speed_scale_ = get_node()->get_parameter("scan.phase_speed_scale").as_double();
    scan_max_linear_speed_ = get_node()->get_parameter("scan.max_linear_speed").as_double();
    scan_max_angular_speed_ = get_node()->get_parameter("scan.max_angular_speed").as_double();
    scan_acquire_max_linear_speed_ = get_node()->get_parameter("scan.acquire_max_linear_speed").as_double();
    scan_acquire_max_angular_speed_ = get_node()->get_parameter("scan.acquire_max_angular_speed").as_double();
    scan_acquire_release_surface_tol_ =
      get_node()->get_parameter("scan.acquire_release_surface_tol").as_double();
    scan_acquire_reacquire_surface_tol_ =
      get_node()->get_parameter("scan.acquire_reacquire_surface_tol").as_double();
    scan_acquire_reacquire_surface_time_ =
      get_node()->get_parameter("scan.acquire_reacquire_surface_time").as_double();
    scan_acquire_release_band_tol_ =
      get_node()->get_parameter("scan.acquire_release_band_tol").as_double();
    scan_acquire_reacquire_band_tol_ =
      get_node()->get_parameter("scan.acquire_reacquire_band_tol").as_double();
    scan_acquire_hover_surface_offset_ =
      get_node()->get_parameter("scan.acquire_hover_surface_offset").as_double();
    scan_acquire_recovery_height_margin_ =
      get_node()->get_parameter("scan.acquire_recovery_height_margin").as_double();
    scan_acquire_recovery_max_angular_speed_ =
      get_node()->get_parameter("scan.acquire_recovery_max_angular_speed").as_double();
    scan_max_position_error_ = get_node()->get_parameter("scan.max_position_error").as_double();
    scan_max_radial_error_ = get_node()->get_parameter("scan.max_radial_error").as_double();
    scan_max_tangential_error_ = get_node()->get_parameter("scan.max_tangential_error").as_double();
    scan_max_orientation_error_ = get_node()->get_parameter("scan.max_orientation_error").as_double();
    scan_progress_gate_ = get_node()->get_parameter("scan.progress_gate").as_bool();
    scan_progress_pos_tol_ = get_node()->get_parameter("scan.progress_pos_tol").as_double();
    scan_progress_ori_tol_ = get_node()->get_parameter("scan.progress_ori_tol").as_double();
    scan_progress_hard_stop_pos_tol_ =
      get_node()->get_parameter("scan.progress_hard_stop_pos_tol").as_double();
    scan_progress_hard_stop_ori_tol_ =
      get_node()->get_parameter("scan.progress_hard_stop_ori_tol").as_double();
    scan_progress_hard_stop_surface_dist_tol_ =
      get_node()->get_parameter("scan.progress_hard_stop_surface_dist_tol").as_double();
    scan_progress_band_tol_ = get_node()->get_parameter("scan.progress_band_tol").as_double();
    scan_progress_hard_stop_band_tol_ =
      get_node()->get_parameter("scan.progress_hard_stop_band_tol").as_double();
    scan_progress_cruise_min_scale_ =
      get_node()->get_parameter("scan.progress_cruise_min_scale").as_double();
    scan_progress_cruise_quality_threshold_ =
      get_node()->get_parameter("scan.progress_cruise_quality_threshold").as_double();
    scan_progress_hard_stop_surface_reacquire_time_ =
      get_node()->get_parameter("scan.progress_hard_stop_surface_reacquire_time").as_double();
    scan_progress_hard_stop_reacquire_time_ =
      get_node()->get_parameter("scan.progress_hard_stop_reacquire_time").as_double();
    scan_tool_axis_local_ = vec3_from_param(
      get_node()->get_parameter("scan.tool_axis_local").as_double_array(), scan_tool_axis_local_);
    scan_align_inward_normal_ = get_node()->get_parameter("scan.align_inward_normal").as_bool();
    scan_hold_current_roll_ = get_node()->get_parameter("scan.hold_current_roll").as_bool();
    scan_auto_pick_tool_axis_on_activate_ =
      get_node()->get_parameter("scan.auto_pick_tool_axis_on_activate").as_bool();
    scan_seed_from_current_on_activate_ =
      get_node()->get_parameter("scan.seed_from_current_on_activate").as_bool();
    scan_lock_lissajous_mapping_ =
      get_node()->get_parameter("scan.lock_lissajous_mapping").as_bool();
    scan_seed_blend_time_ = get_node()->get_parameter("scan.seed_blend_time").as_double();
    scan_min_height_above_base_ = get_node()->get_parameter("scan.min_height_above_base").as_double();
    if (scan_tool_axis_local_.norm() < 1e-9) {
      scan_tool_axis_local_ = Eigen::Vector3d(0.0, 0.0, -1.0);
    }
    scan_tool_axis_local_.normalize();
    scan_tool_axis_runtime_ = scan_tool_axis_local_;
    scan_theta_seed_ = scan_theta_center_;
    scan_phi_seed_ = scan_phi_center_;
    scan_theta_start_ = scan_theta_seed_;
    scan_phi_start_ = scan_phi_seed_;
    scan_surface_origin_normal_ = Eigen::Vector3d(0.0, 0.0, 1.0);
    scan_surface_origin_normal_valid_ = false;
    scan_surface_seed_ = scan_surface_lissajous_offset_;
    scan_surface_start_.setZero();

    k_d_  = get_node()->get_parameter("paper.k_d").as_double();
    d_d_  = get_node()->get_parameter("paper.d_d").as_double();
    k_s_ = vec2_from_param(get_node()->get_parameter("paper.k_s").as_double_array(), k_s_);
    d_s_ = vec2_from_param(get_node()->get_parameter("paper.d_s").as_double_array(), d_s_);
    k_eps_= get_node()->get_parameter("paper.k_eps").as_double();
    d_eps_= get_node()->get_parameter("paper.d_eps").as_double();
    d_max_= get_node()->get_parameter("paper.d_max").as_double();
    enable_curvature_ = get_node()->get_parameter("paper.enable_curvature").as_bool();

    nullspace_enable_ = get_node()->get_parameter("nullspace.enable").as_bool();
    null_kp_ = get_node()->get_parameter("nullspace.kp").as_double();
    null_kd_ = get_node()->get_parameter("nullspace.kd").as_double();
    nullspace_damping_ = get_node()->get_parameter("nullspace.damping").as_double();
    max_tau_ = get_node()->get_parameter("safety.max_tau").as_double();
    max_tau_rate_ = get_node()->get_parameter("safety.max_tau_rate").as_double();
    switch_protection_enable_ =
      get_node()->get_parameter("safety.switch_protection.enable").as_bool();
    switch_hold_time_ =
      get_node()->get_parameter("safety.switch_protection.hold_time").as_double();
    switch_blend_time_ =
      get_node()->get_parameter("safety.switch_protection.blend_time").as_double();

    // optional nullspace q_des
    const auto qdes = get_node()->get_parameter("nullspace.q_des").as_double_array();
    if (qdes.size() == 7) {
      for (int i = 0; i < 7; ++i) null_q_des_(i) = qdes[i];
      null_q_des_initialized_ = true;
    } else {
      null_q_des_initialized_ = false; // on_activate 里再用当前 q 初始化
    }

    debug_joint_pd_mode_ = get_node()->get_parameter("debug_joint_pd_mode").as_bool();
    debug_pd_kp_ = get_node()->get_parameter("debug_pd.kp").as_double();
    debug_pd_kd_ = get_node()->get_parameter("debug_pd.kd").as_double();
    debug_pd_hold_current_on_activate_ =
      get_node()->get_parameter("debug_pd.hold_current_on_activate").as_bool();
    debug_publish_markers_ = get_node()->get_parameter("debug.publish_markers").as_bool();
    debug_publish_state_ = get_node()->get_parameter("debug.publish_state").as_bool();
    debug_publish_rate_hz_ = get_node()->get_parameter("debug.publish_rate_hz").as_double();

    // Surface model (hemisphere replacement)
    const auto surface_type = get_node()->get_parameter("surface.type").as_string();
    if (surface_type != "hemisphere") {
      RCLCPP_WARN(
        get_node()->get_logger(),
        "surface.type is '%s' (supported: 'hemisphere'). Controller will still run Cartesian impedance.",
        surface_type.c_str());
    }
    hemisphere_radius_ = get_node()->get_parameter("surface.hemisphere.radius").as_double();
    hemisphere_center_ = vec3_from_param(
      get_node()->get_parameter("surface.hemisphere.center").as_double_array(), hemisphere_center_);
    hemisphere_axis_ = vec3_from_param(
      get_node()->get_parameter("surface.hemisphere.axis").as_double_array(), hemisphere_axis_);
    if (hemisphere_axis_.norm() < 1e-9) {
      hemisphere_axis_ = Eigen::Vector3d(0.0, 0.0, 1.0);
    }
    hemisphere_axis_.normalize();
    scan_surface_origin_normal_ = hemisphere_axis_;
    scan_surface_origin_normal_valid_ = true;

    // Build gains
    K_cart_.setZero();
    D_cart_.setZero();
    K_cart_.topLeftCorner<3, 3>() = k_trans_ * Eigen::Matrix3d::Identity();
    K_cart_.bottomRightCorner<3, 3>() = k_rot_ * Eigen::Matrix3d::Identity();
    D_cart_.topLeftCorner<3, 3>() = d_trans_ * Eigen::Matrix3d::Identity();
    D_cart_.bottomRightCorner<3, 3>() = d_rot_ * Eigen::Matrix3d::Identity();

    D_joint_.setConstant(joint_damping_);

    debug_publish_accum_ = 0.0;
    if (debug_publish_markers_) {
      debug_marker_pub_ = get_node()->create_publisher<visualization_msgs::msg::MarkerArray>(
        "~/debug_markers", rclcpp::SystemDefaultsQoS());
    } else {
      debug_marker_pub_.reset();
    }
    if (debug_publish_state_) {
      debug_state_pub_ = get_node()->create_publisher<std_msgs::msg::Float64MultiArray>(
        "~/debug_state", rclcpp::SystemDefaultsQoS());
    } else {
      debug_state_pub_.reset();
    }

    // Franka semantic component: pass full state interface names (name/interface).

    // Init backend 
    if (use_franka_semantic_) {
      // Real robot backend
      franka_robot_model_ = std::make_unique<franka_semantic_components::FrankaRobotModel>(
        arm_id_ + "/robot_model", arm_id_ + "/robot_state");
      kdl_initialized_ = false;
      fk_solver_.reset();
      jac_solver_.reset();
      RCLCPP_INFO(get_node()->get_logger(), "Configured with Franka semantic backend.");
    } else {
      // Simulation backend
      franka_robot_model_.reset();
      if (!init_kdl_from_urdf_file_()) {
        RCLCPP_ERROR(get_node()->get_logger(), "Failed to init KDL from URDF. Check urdf_path/base_link/ee_link.");
        return controller_interface::CallbackReturn::ERROR;
      }
      RCLCPP_INFO(get_node()->get_logger(), "Configured with KDL simulation backend.");
    }

    desired_initialized_ = false;

    RCLCPP_INFO(get_node()->get_logger(), "SurfaceImpedanceController configured.");
    RCLCPP_INFO(
      get_node()->get_logger(),
      "arm_id=%s, joints=%zu, k_trans=%.3f, d_trans=%.3f, k_rot=%.3f, d_rot=%.3f, joint_damping=%.3f",
      arm_id_.c_str(), joint_names_.size(), k_trans_, d_trans_, k_rot_, d_rot_, joint_damping_);
    RCLCPP_INFO(
      get_node()->get_logger(),
      "outer: d_des=%.4f, k_outer_pos=%.4f, hemisphere: center=[%.3f %.3f %.3f], r=%.3f, axis=[%.3f %.3f %.3f]",
      d_des_, k_outer_pos_,
      hemisphere_center_.x(), hemisphere_center_.y(), hemisphere_center_.z(),
      hemisphere_radius_,
      hemisphere_axis_.x(), hemisphere_axis_.y(), hemisphere_axis_.z());

  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_node()->get_logger(), "on_configure failed: %s", e.what());
    return controller_interface::CallbackReturn::ERROR;
  }

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn SurfaceImpedanceController::on_activate(
  const rclcpp_lifecycle::State & /*previous_state*/) {
  
  RCLCPP_DEBUG(get_node()->get_logger(),
  "DEBUG: command_interfaces_.size=%zu state_interfaces_.size=%zu",
  command_interfaces_.size(), state_interfaces_.size());

  for (auto & ci : command_interfaces_) {
    RCLCPP_DEBUG(get_node()->get_logger(),
      "DEBUG CI: name='%s' iface='%s'",
      ci.get_name().c_str(), ci.get_interface_name().c_str());
  }

  for (auto & si : state_interfaces_) {
    RCLCPP_DEBUG(get_node()->get_logger(),
      "DEBUG SI: name='%s' iface='%s'",
      si.get_name().c_str(), si.get_interface_name().c_str());
  }

    
  try {
    // Bind joint interfaces
    cmd_effort_.clear();
    state_pos_.clear();
    state_vel_.clear();
    cmd_effort_.reserve(joint_names_.size());
    state_pos_.reserve(joint_names_.size());
    state_vel_.reserve(joint_names_.size());

    for (const auto & j : joint_names_) {
      auto * cmd = get_command_handle_(j, "effort");
      auto * pos = get_state_handle_(j, "position");
      auto * vel = get_state_handle_(j, "velocity");
      if (!cmd || !pos || !vel) {
        RCLCPP_ERROR(
          get_node()->get_logger(),
          "Missing required interfaces for joint '%s' (need effort cmd, position+velocity state).", j.c_str());
        return controller_interface::CallbackReturn::ERROR;
      }
      cmd_effort_.push_back(cmd);
      state_pos_.push_back(pos);
      state_vel_.push_back(vel);
    }

    const Eigen::Matrix<double, 7, 1> q_current = get_q_();
    switch_elapsed_time_ = 0.0;
    tau_prev_cmd_.setZero();
    for (size_t i = 0; i < cmd_effort_.size(); ++i) {
      const double tau_prev = cmd_effort_[i]->get_value();
      tau_prev_cmd_(static_cast<int>(i)) = std::isfinite(tau_prev) ? tau_prev : 0.0;
    }
    tau_start_cmd_ = tau_prev_cmd_;
    tau_prev_valid_ = true;

   // Initialize desired pose from current pose (two backends)
    if (use_franka_semantic_) {
      if (!franka_robot_model_) {
        RCLCPP_ERROR(get_node()->get_logger(), "Franka semantic backend selected but franka_robot_model_ is null.");
        return controller_interface::CallbackReturn::ERROR;
      }
      // Bind Franka semantic component to state interfaces
      franka_robot_model_->assign_loaned_state_interfaces(state_interfaces_);

      const auto pose_array = franka_robot_model_->getPoseMatrix(franka::Frame::kEndEffector);
      Eigen::Map<const Eigen::Matrix<double, 4, 4>> T(pose_array.data());
      const Eigen::Affine3d Tf(T);
      p_d_ = Tf.translation();
      q_d_ = Eigen::Quaterniond(Tf.linear());
      q_d_.normalize();
      desired_initialized_ = true;
    } else {
      // Simulation backend: KDL FK
      if (!kdl_initialized_ || !fk_solver_) {
        RCLCPP_ERROR(get_node()->get_logger(), "KDL backend not initialized. Check urdf_path/base_link/ee_link.");
        return controller_interface::CallbackReturn::ERROR;
      }
      // Fill KDL joint array from current joint states
      if (kdl_q_.rows() != 7) {
        RCLCPP_ERROR(get_node()->get_logger(), "KDL joint array size = %d (expected 7).", (int)kdl_q_.rows());
        return controller_interface::CallbackReturn::ERROR;
      }
      for (int i = 0; i < 7; ++i) {
        kdl_q_(i) = q_current(i);
      }

      KDL::Frame ee_frame;
      const int ret = fk_solver_->JntToCart(kdl_q_, ee_frame);
      if (ret < 0) {
        RCLCPP_ERROR(get_node()->get_logger(), "KDL FK failed, ret=%d", ret);
        return controller_interface::CallbackReturn::ERROR;
      }

      p_d_ = Eigen::Vector3d(ee_frame.p.x(), ee_frame.p.y(), ee_frame.p.z());
      double qx, qy, qz, qw;
      ee_frame.M.GetQuaternion(qx, qy, qz, qw); // KDL returns (x,y,z,w)
      q_d_ = Eigen::Quaterniond(qw, qx, qy, qz);
      q_d_.normalize();
      desired_initialized_ = true;
    }

    if (!null_q_des_initialized_) {
      null_q_des_ = q_current;
      null_q_des_initialized_ = true;
    }

    // Reset debug PD target on each activation
    debug_pd_target_initialized_ = false;
    if (debug_joint_pd_mode_ && debug_pd_hold_current_on_activate_) {
      debug_pd_q_target_ = q_current;
      debug_pd_target_initialized_ = true;
    }
    if (scan_reset_time_on_activate_) {
      scan_time_ = 0.0;
      scan_ramp_elapsed_ = 0.0;
    }
    scan_seed_blend_elapsed_ = 0.0;
    const bool lock_lissajous_mapping =
      scan_lock_lissajous_mapping_ && scan_track_generator_ == "paper_surface_lissajous";
    scan_acquire_surface_reacquire_elapsed_ = 0.0;
    scan_acquire_band_reacquire_elapsed_ = 0.0;
    scan_progress_hard_stop_active_ = false;
    scan_progress_hard_stop_surface_elapsed_ = 0.0;
    scan_progress_hard_stop_ori_elapsed_ = 0.0;
    scan_recovery_hold_prev_ = false;
    debug_scan_reacquire_reason_ = 0.0;
    scan_theta_seed_ = clamp_theta_to_sector(scan_theta_center_, scan_theta_min_, scan_theta_max_);
    scan_phi_seed_ = scan_phi_center_;
    scan_theta_start_ = scan_theta_seed_;
    scan_phi_start_ = scan_phi_seed_;
    scan_surface_origin_normal_ = hemisphere_axis_;
    scan_surface_origin_normal_valid_ = true;
    scan_normal_cmd_ = scan_surface_origin_normal_;
    scan_normal_cmd_valid_ = scan_surface_origin_normal_valid_;
    if (scan_track_generator_ == "paper_surface_lissajous") {
      Eigen::Vector3d axis = Eigen::Vector3d::Zero();
      Eigen::Vector3d e1 = Eigen::Vector3d::Zero();
      Eigen::Vector3d e2 = Eigen::Vector3d::Zero();
      if (build_hemisphere_basis_(axis, e1, e2)) {
        const double phi_min = std::max(0.0, scan_phi_min_);
        const double phi_max = effective_scan_phi_max_();
        const double phi_seed =
          (phi_max <= phi_min + 1e-6) ? phi_min : std::clamp(scan_phi_seed_, phi_min, phi_max);
        const Eigen::Vector3d t_circle =
          std::cos(scan_theta_seed_) * e1 + std::sin(scan_theta_seed_) * e2;
        scan_surface_origin_normal_ = std::cos(phi_seed) * axis + std::sin(phi_seed) * t_circle;
        scan_surface_origin_normal_valid_ = scan_surface_origin_normal_.norm() > 1e-9;
        if (scan_surface_origin_normal_valid_) {
          scan_surface_origin_normal_.normalize();
        }
        scan_normal_cmd_ = scan_surface_origin_normal_;
        scan_normal_cmd_valid_ = scan_surface_origin_normal_valid_;
      }
    }
    scan_surface_seed_ = scan_surface_lissajous_offset_;
    scan_surface_start_ = lock_lissajous_mapping ? scan_surface_seed_ : Eigen::Vector2d::Zero();
    scan_surface_cmd_.setZero();
    scan_surface_cmd_valid_ = false;
    scan_surface_raw_prev_.setZero();
    scan_surface_raw_prev_valid_ = false;
    scan_surface_actual_prev_.setZero();
    scan_surface_actual_prev_valid_ = false;
    scan_surface_phase_progress_ratio_ = 1.0;
    scan_surface_des_prev_valid_ = false;
    paper_surface_acquired_ = false;
    scan_tool_axis_runtime_ = scan_tool_axis_local_;
    scan_roll_hint_valid_ = false;
    if (lock_lissajous_mapping) {
      scan_seed_blend_elapsed_ = std::max(0.0, scan_seed_blend_time_);
    }

    if (control_mode_ == "hemi_scan_cartesian" || control_mode_ == "paper_surface_scan") {
      if (scan_seed_from_current_on_activate_ && !lock_lissajous_mapping) {
        Eigen::Vector3d n_seed = Eigen::Vector3d::Zero();
        Eigen::Vector3d proxy_seed = Eigen::Vector3d::Zero();
        double d_seed = 0.0;
        bool have_seed_surface = false;

        if (scan_track_generator_ == "paper_surface_lissajous" && hemisphere_axis_.norm() > 1e-9) {
          const Eigen::Vector3d v_seed = p_d_ - hemisphere_center_;
          if (v_seed.norm() > 1e-9) {
            const Eigen::Vector3d n_candidate = v_seed.normalized();
            constexpr double kSeedAxisSlack = 1e-3;
            const double phi_max = effective_scan_phi_max_();
            const double min_axis_dot = std::cos(phi_max);
            if (n_candidate.dot(hemisphere_axis_.normalized()) >= (min_axis_dot - kSeedAxisSlack)) {
              n_seed = n_candidate;
              have_seed_surface = true;
            }
          }
        }

        if (!have_seed_surface &&
            query_hemisphere_proxy_(p_d_, proxy_seed, d_seed, n_seed) && n_seed.norm() > 1e-9) {
          n_seed.normalize();
          have_seed_surface = true;
        } else {
          const Eigen::Vector3d v = p_d_ - hemisphere_center_;
          if (v.norm() > 1e-9) {
            n_seed = v.normalized();
            have_seed_surface = true;
          }
        }

        if (have_seed_surface) {
          double theta_now = scan_theta_center_;
          double phi_now = scan_phi_center_;
          Eigen::Vector3d n_seed_projected = n_seed;
          if (project_normal_to_scan_band_(n_seed, n_seed_projected, &theta_now, &phi_now)) {
            n_seed = n_seed_projected;
            scan_theta_start_ = theta_now;
            scan_phi_start_ = phi_now;
            scan_theta_seed_ = theta_now;
            scan_phi_seed_ = phi_now;
          }
          if (scan_track_generator_ == "paper_surface_lissajous") {
            scan_surface_origin_normal_ = n_seed;
            scan_surface_origin_normal_valid_ = true;
            scan_normal_cmd_ = n_seed;
            scan_normal_cmd_valid_ = true;
            scan_surface_start_.setZero();
            scan_surface_seed_ = scan_surface_lissajous_offset_;
            scan_surface_cmd_.setZero();
            scan_surface_cmd_valid_ = false;
            scan_surface_raw_prev_.setZero();
            scan_surface_raw_prev_valid_ = false;
            scan_surface_actual_prev_.setZero();
            scan_surface_actual_prev_valid_ = false;
            scan_surface_phase_progress_ratio_ = 1.0;
          } else {
            Eigen::Vector2d s_now = Eigen::Vector2d::Zero();
            if (normal_to_hemi_surface_coords_(n_seed, s_now)) {
              scan_surface_start_ = s_now;
              scan_surface_seed_ = s_now + scan_surface_lissajous_offset_;
              scan_surface_cmd_ = scan_surface_start_;
              scan_surface_cmd_valid_ = true;
              scan_surface_raw_prev_ = scan_surface_start_;
              scan_surface_raw_prev_valid_ = true;
              scan_surface_actual_prev_ = scan_surface_start_;
              scan_surface_actual_prev_valid_ = true;
              scan_surface_phase_progress_ratio_ = 1.0;
            }
          }
        }
      }

      if (scan_auto_pick_tool_axis_on_activate_) {
        Eigen::Vector3d n_seed = Eigen::Vector3d::Zero();
        if (scan_track_generator_ == "paper_surface_lissajous") {
          if (scan_surface_origin_normal_valid_) {
            local_surface_coords_to_normal(
              scan_surface_origin_normal_, hemisphere_axis_, hemisphere_radius_, scan_surface_seed_, n_seed);
          }
        } else {
          Eigen::Vector3d axis = Eigen::Vector3d::Zero();
          Eigen::Vector3d e1 = Eigen::Vector3d::Zero();
          Eigen::Vector3d e2 = Eigen::Vector3d::Zero();
          if (build_hemisphere_basis_(axis, e1, e2)) {
            const Eigen::Vector3d t_circle =
              std::cos(scan_theta_seed_) * e1 + std::sin(scan_theta_seed_) * e2;
            n_seed = std::cos(scan_phi_seed_) * axis + std::sin(scan_phi_seed_) * t_circle;
          }
        }
        if (n_seed.norm() > 1e-9) {
          n_seed.normalize();
          const Eigen::Vector3d target_axis = scan_align_inward_normal_ ? -n_seed : n_seed;
          const Eigen::Matrix3d R = q_d_.toRotationMatrix();
          const Eigen::Vector3d candidates[6] = {
            Eigen::Vector3d::UnitX(), -Eigen::Vector3d::UnitX(),
            Eigen::Vector3d::UnitY(), -Eigen::Vector3d::UnitY(),
            Eigen::Vector3d::UnitZ(), -Eigen::Vector3d::UnitZ()
          };
          double best_score = -std::numeric_limits<double>::infinity();
          Eigen::Vector3d best_axis = scan_tool_axis_local_;
          for (const auto & c : candidates) {
            const Eigen::Vector3d c_base = R * c;
            const double score = c_base.dot(target_axis);
            if (score > best_score) {
              best_score = score;
              best_axis = c;
            }
          }
          scan_tool_axis_runtime_ = best_axis;
          RCLCPP_INFO(
            get_node()->get_logger(),
            "scan tool axis auto-picked: local=[%.0f %.0f %.0f], score=%.3f (1.0 is perfect)",
            scan_tool_axis_runtime_.x(), scan_tool_axis_runtime_.y(), scan_tool_axis_runtime_.z(), best_score);
        }
      }

      if (compute_roll_hint_in_base(q_d_, scan_tool_axis_runtime_, scan_roll_hint_base_)) {
        scan_roll_hint_valid_ = true;
      }

    }

    RCLCPP_INFO(
      get_node()->get_logger(),
      "switch protection: enable=%d hold=%.3fs blend=%.3fs max_tau=%.1fNm max_tau_rate=%.1fNm/s",
      switch_protection_enable_ ? 1 : 0,
      std::max(0.0, switch_hold_time_),
      std::max(0.0, switch_blend_time_),
      max_tau_,
      max_tau_rate_);

    RCLCPP_INFO(get_node()->get_logger(), "SurfaceImpedanceController activated.");
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_node()->get_logger(), "on_activate failed: %s", e.what());
    return controller_interface::CallbackReturn::ERROR;
  }

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn SurfaceImpedanceController::on_deactivate(
  const rclcpp_lifecycle::State & /*previous_state*/) {
  debug_pd_target_initialized_ = false;
  for (auto * ci : cmd_effort_) {
    if (ci) {
      ci->set_value(0.0);
    }
  }
  cmd_effort_.clear();
  state_pos_.clear();
  state_vel_.clear();
  desired_initialized_ = false;
  debug_publish_accum_ = 0.0;
  switch_elapsed_time_ = 0.0;
  scan_surface_origin_normal_valid_ = false;
  scan_surface_cmd_valid_ = false;
  scan_surface_raw_prev_valid_ = false;
  scan_surface_actual_prev_valid_ = false;
  scan_surface_phase_progress_ratio_ = 1.0;
  scan_surface_des_prev_valid_ = false;
  scan_acquire_surface_reacquire_elapsed_ = 0.0;
  scan_acquire_band_reacquire_elapsed_ = 0.0;
  scan_progress_hard_stop_active_ = false;
  scan_progress_hard_stop_surface_elapsed_ = 0.0;
  scan_progress_hard_stop_ori_elapsed_ = 0.0;
  scan_recovery_hold_prev_ = false;
  paper_surface_acquired_ = false;
  tau_prev_valid_ = false;
  tau_prev_cmd_.setZero();
  tau_start_cmd_.setZero();

  RCLCPP_INFO(get_node()->get_logger(), "SurfaceImpedanceController deactivated.");
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::return_type SurfaceImpedanceController::update(
  const rclcpp::Time & /*time*/, const rclcpp::Duration & period) {

  if (!desired_initialized_) {
    return controller_interface::return_type::OK;
  }

  const double dt = period.seconds();
  const double dt_nonneg = (std::isfinite(dt) && dt > 0.0) ? dt : 0.0;
  const double dt_slew = std::min(dt_nonneg, 0.01); // cap slew/integrator dt to avoid switch-time jumps
  if (dt_slew > 0.0) {
    switch_elapsed_time_ += dt_slew;
    if (control_mode_ == "hemi_scan_cartesian" || control_mode_ == "paper_surface_scan") {
      scan_seed_blend_elapsed_ += dt_slew;
    }
  }
  const double switch_hold = std::max(0.0, switch_hold_time_);
  const double switch_blend = std::max(0.0, switch_blend_time_);
  const bool switch_hold_active =
    switch_protection_enable_ && (switch_elapsed_time_ < switch_hold);
  const bool switch_transient_active =
    switch_protection_enable_ && (switch_elapsed_time_ < (switch_hold + switch_blend));
  RCLCPP_INFO_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 1000, "[DEBUG] dt = %.6f s", dt);

  const Eigen::Matrix<double, 7, 1> q = get_q_();
  const Eigen::Matrix<double, 7, 1> dq = get_dq_();

  // Debug: conservative joint PD (no model terms).
  if (debug_joint_pd_mode_) {
    if (!debug_pd_target_initialized_) {
      debug_pd_q_target_ = q;                 // first entry -> hold current
      debug_pd_target_initialized_ = true;
    }

    const double Kp = debug_pd_kp_;
    const double Kd = debug_pd_kd_;
    Eigen::Matrix<double, 7, 1> tau = -Kp * (q - debug_pd_q_target_) - Kd * dq;

    RCLCPP_INFO_THROTTLE(
      get_node()->get_logger(), *get_node()->get_clock(), 500,
      "[DEBUG PD] ||tau|| = %.3f", tau.norm());

    for (size_t i = 0; i < cmd_effort_.size(); ++i) {
      cmd_effort_[i]->set_value(tau(static_cast<int>(i)));
    }
    return controller_interface::return_type::OK;
  }

  // ===============================
  // Current EE pose + Jacobian (two backends)
  // ===============================
  Eigen::Vector3d p = Eigen::Vector3d::Zero();
  Eigen::Quaterniond q_ee = Eigen::Quaterniond::Identity();
  Eigen::Matrix<double, 6, 7> J = Eigen::Matrix<double, 6, 7>::Zero();
  Eigen::Matrix<double, 7, 1> model_feedforward = Eigen::Matrix<double, 7, 1>::Zero();

  if (use_franka_semantic_) {
    // ---- Real robot backend (Franka semantic) ----
    const auto pose_array = franka_robot_model_->getPoseMatrix(franka::Frame::kEndEffector);
    Eigen::Map<const Eigen::Matrix<double, 4, 4>> T(pose_array.data());  // column-major
    const Eigen::Affine3d Tf_sem(T);
    p = Tf_sem.translation();
    q_ee = Eigen::Quaterniond(Tf_sem.linear());
    q_ee.normalize();

    const auto jacobian_array = franka_robot_model_->getZeroJacobian(franka::Frame::kEndEffector);
    Eigen::Map<const Eigen::Matrix<double, 6, 7>> J_map(jacobian_array.data());
    J = J_map;

    const auto coriolis_array = franka_robot_model_->getCoriolisForceVector();
    Eigen::Map<const Eigen::Matrix<double, 7, 1>> coriolis_map(coriolis_array.data());
    model_feedforward = coriolis_map;
  } else {
    // ---- Simulation backend (KDL) ----
    if (!kdl_initialized_) {
      RCLCPP_ERROR_THROTTLE(
        get_node()->get_logger(), *get_node()->get_clock(), 1000,
        "KDL backend not initialized; cannot run update().");
      return controller_interface::return_type::ERROR;
    }
    if (!compute_fk_jacobian_kdl_(q, p, q_ee, J)) {
      RCLCPP_ERROR_THROTTLE(
        get_node()->get_logger(), *get_node()->get_clock(), 1000,
        "compute_fk_jacobian_kdl_ failed.");
      return controller_interface::return_type::ERROR;
    }
    // Gazebo's Franka effort hardware plugin already adds gravity torques for
    // every effort controller command. Keep the controller-side feedforward
    // zero in simulation to avoid double compensation.
    model_feedforward.setZero();
  }

  // Build Tf for axis computations
  Eigen::Affine3d Tf = Eigen::Affine3d::Identity();
  Tf.linear() = q_ee.toRotationMatrix();
  Tf.translation() = p;

  const Eigen::Vector3d p_d_prev = p_d_;
  Eigen::Quaterniond q_d_prev = q_d_;
  if (q_d_prev.norm() < 1e-9) {
    q_d_prev = Eigen::Quaterniond::Identity();
  } else {
    q_d_prev.normalize();
  }
  Eigen::Matrix<double, 6, 1> v_des_cart = Eigen::Matrix<double, 6, 1>::Zero();

  // Cartesian twist v = J * dq
  const Eigen::Matrix<double, 6, 1> v = J * dq;
  const Eigen::Vector3d v_lin = v.head<3>();
  const Eigen::Vector3d v_ang = v.tail<3>();

  auto skew = [](const Eigen::Vector3d& a) {
    Eigen::Matrix3d S;
    S <<     0.0, -a.z(),  a.y(),
          a.z(),    0.0, -a.x(),
         -a.y(),  a.x(),   0.0;
    return S;
  };

  auto clamp_tau = [&](Eigen::Matrix<double,7,1>& tau) {
    if (max_tau_ > 0.0) {
      tau = tau.cwiseMax(-max_tau_).cwiseMin(max_tau_);
    }
  };

  auto add_nullspace_posture = [&](const auto & J_task, Eigen::Matrix<double, 7, 1>& tau) {
    if (!nullspace_enable_ || !null_q_des_initialized_) {
      return;
    }

    const Eigen::Matrix<double, 7, 7> N = damped_nullspace_projector(J_task, nullspace_damping_);
    tau += N * (null_kp_ * (null_q_des_ - q) - null_kd_ * dq);
  };

  auto apply_switch_safety = [&](Eigen::Matrix<double,7,1>& tau) {
    if (switch_protection_enable_) {
      const double blend_alpha =
        (switch_blend > 1e-6) ? std::clamp(switch_elapsed_time_ / switch_blend, 0.0, 1.0) : 1.0;
      tau = (1.0 - blend_alpha) * tau_start_cmd_ + blend_alpha * tau;
    }

    if (!tau_prev_valid_) {
      tau_prev_cmd_ = tau;
      tau_prev_valid_ = true;
    }

    // Keep torque-rate clamp strict during controller switch transient.
    // After transient, let scan dynamics use full controller authority.
    const bool apply_rate_limit = (!switch_protection_enable_) || switch_transient_active;
    if (apply_rate_limit && max_tau_rate_ > 0.0 && dt_slew > 0.0) {
      const double max_step = max_tau_rate_ * dt_slew;
      for (int i = 0; i < 7; ++i) {
        const double delta = tau(i) - tau_prev_cmd_(i);
        const double delta_limited = std::clamp(delta, -max_step, max_step);
        tau(i) = tau_prev_cmd_(i) + delta_limited;
      }
    }

    clamp_tau(tau);
    tau_prev_cmd_ = tau;
  };

  const bool lock_lissajous_mapping =
    scan_lock_lissajous_mapping_ && scan_track_generator_ == "paper_surface_lissajous";
  const bool scan_seed_blend_active =
    (control_mode_ == "hemi_scan_cartesian" || control_mode_ == "paper_surface_scan") &&
    !lock_lissajous_mapping &&
    (scan_seed_blend_time_ > 1e-6) &&
    (scan_seed_blend_elapsed_ + 1e-6 < scan_seed_blend_time_);
  debug_scan_seed_blend_active_ = scan_seed_blend_active;
  debug_paper_acquire_active_ = false;
  debug_scan_surface_err_ = std::numeric_limits<double>::quiet_NaN();
  debug_scan_surface_dist_err_ = std::numeric_limits<double>::quiet_NaN();
  debug_scan_surface_ori_err_ = std::numeric_limits<double>::quiet_NaN();
  debug_scan_cmd_ori_err_ = std::numeric_limits<double>::quiet_NaN();
  debug_scan_band_err_ = std::numeric_limits<double>::quiet_NaN();
  debug_scan_roll_err_ = std::numeric_limits<double>::quiet_NaN();
  debug_scan_roll_rate_ = std::numeric_limits<double>::quiet_NaN();

  auto compute_hemi_scan_pose = [&](double t, Eigen::Vector3d& p_des, Eigen::Vector3d& n_des) -> bool {
    if (!(std::isfinite(hemisphere_radius_) && hemisphere_radius_ > 1e-6)) {
      return false;
    }

    const double ramp_alpha =
      (scan_ramp_time_ > 1e-6) ? std::clamp(scan_ramp_elapsed_ / scan_ramp_time_, 0.0, 1.0) : 1.0;
    const double seed_blend_alpha =
      (scan_seed_blend_time_ > 1e-6) ?
      std::clamp(scan_seed_blend_elapsed_ / scan_seed_blend_time_, 0.0, 1.0) : 1.0;
    const double scan_phase_time = scan_seed_blend_active ? 0.0 : t;

    if (scan_track_generator_ == "paper_surface_lissajous") {
      if (!scan_surface_origin_normal_valid_) {
        return false;
      }
      Eigen::Vector2d s_des = scan_surface_seed_;
      s_des.x() +=
        ramp_alpha * scan_surface_lissajous_amplitude_.x() *
        std::sin(scan_surface_lissajous_omega_.x() * scan_phase_time + scan_surface_lissajous_phase_.x());
      s_des.y() +=
        ramp_alpha * scan_surface_lissajous_amplitude_.y() *
        std::sin(scan_surface_lissajous_omega_.y() * scan_phase_time + scan_surface_lissajous_phase_.y());
      s_des = scan_surface_start_ + seed_blend_alpha * (s_des - scan_surface_start_);
      if (!local_surface_coords_to_normal(
            scan_surface_origin_normal_, hemisphere_axis_, hemisphere_radius_, s_des, n_des)) {
        return false;
      }
    } else {
      Eigen::Vector3d axis = Eigen::Vector3d::Zero();
      Eigen::Vector3d e1 = Eigen::Vector3d::Zero();
      Eigen::Vector3d e2 = Eigen::Vector3d::Zero();
      if (!build_hemisphere_basis_(axis, e1, e2)) {
        return false;
      }

      double theta = scan_theta_seed_;
      if (scan_theta_orbit_) {
        const double theta_phase =
          scan_theta_phase_ + ramp_alpha * (kTwoPi * scan_theta_freq_ * scan_phase_time);
        if (scan_theta_smooth_turnaround_) {
          theta = smooth_theta_in_sector(
            scan_theta_seed_, theta_phase, scan_theta_min_, scan_theta_max_);
        } else {
          const double theta_unwrapped = scan_theta_seed_ + theta_phase;
          theta = reflect_theta_in_sector(theta_unwrapped, scan_theta_min_, scan_theta_max_);
        }
      } else {
        theta +=
          (ramp_alpha * scan_theta_amp_) * std::sin(kTwoPi * scan_theta_freq_ * t + scan_theta_phase_);
        theta = clamp_theta_to_sector(theta, scan_theta_min_, scan_theta_max_);
      }

      const double theta_start = scan_theta_start_;
      theta = theta_start +
        seed_blend_alpha * (unwrap_angle_near(theta, theta_start) - theta_start);

      double phi = scan_phi_seed_ +
        (ramp_alpha * scan_phi_amp_) * std::sin(kTwoPi * scan_phi_freq_ * scan_phase_time + scan_phi_phase_);
      const double phi_min = std::max(0.0, scan_phi_min_);
      const double phi_max = effective_scan_phi_max_();
      if (phi_max <= phi_min + 1e-6) {
        phi = phi_min;
      } else {
        phi = std::clamp(phi, phi_min, phi_max);
      }
      phi = scan_phi_start_ + seed_blend_alpha * (phi - scan_phi_start_);
      phi = std::clamp(phi, 0.0, 0.5 * kPi);

      const Eigen::Vector3d t_circle = std::cos(theta) * e1 + std::sin(theta) * e2;
      n_des = std::cos(phi) * axis + std::sin(phi) * t_circle; // outward normal
      if (n_des.norm() < 1e-9) {
        return false;
      }
      n_des.normalize();
    }

    const double r_des = hemisphere_radius_ + d_des_;
    if (!(std::isfinite(r_des) && r_des > 1e-4)) {
      return false;
    }
    p_des = hemisphere_center_ + r_des * n_des;
    return true;
  };

  auto axis_angle_error = [&](const Eigen::Vector3d& a, const Eigen::Vector3d& b) -> double {
    if (a.norm() < 1e-9 || b.norm() < 1e-9) {
      return std::numeric_limits<double>::infinity();
    }
    const double dot_val = std::clamp(a.normalized().dot(b.normalized()), -1.0, 1.0);
    return std::acos(dot_val);
  };

  auto update_scan_normal_command =
    [&](const Eigen::Vector3d& target_normal,
        double max_angular_speed,
        bool reset) -> Eigen::Vector3d {
      if (target_normal.norm() < 1e-9) {
        return scan_normal_cmd_valid_ ? scan_normal_cmd_ : Eigen::Vector3d::UnitZ();
      }
      const Eigen::Vector3d target = target_normal.normalized();
      if (reset || !scan_normal_cmd_valid_) {
        scan_normal_cmd_ = target;
        scan_normal_cmd_valid_ = true;
        return scan_normal_cmd_;
      }

      const double max_da = std::max(0.0, max_angular_speed) * dt_slew;
      if (!(dt_slew > 1e-9) || max_da <= 1e-9) {
        scan_normal_cmd_ = target;
        return scan_normal_cmd_;
      }

      Eigen::Quaterniond q_delta = Eigen::Quaterniond::FromTwoVectors(scan_normal_cmd_, target);
      q_delta.normalize();
      Eigen::AngleAxisd aa_delta(q_delta);
      const double ang = std::abs(aa_delta.angle());
      if (std::isfinite(ang) && aa_delta.axis().allFinite() && ang > max_da) {
        scan_normal_cmd_ =
          (Eigen::AngleAxisd(max_da, aa_delta.axis()) * scan_normal_cmd_).normalized();
      } else {
        scan_normal_cmd_ = target;
      }
      scan_normal_cmd_valid_ = true;
      return scan_normal_cmd_;
    };

  auto slerp_unit_vectors =
    [&](const Eigen::Vector3d& from_raw,
        const Eigen::Vector3d& to_raw,
        double alpha_raw) -> Eigen::Vector3d {
      Eigen::Vector3d from = from_raw;
      Eigen::Vector3d to = to_raw;
      if (from.norm() < 1e-9 || to.norm() < 1e-9) {
        return (to.norm() >= from.norm() ? to : from);
      }
      from.normalize();
      to.normalize();
      const double alpha = std::clamp(alpha_raw, 0.0, 1.0);
      if (alpha <= 1e-9) {
        return from;
      }
      if (alpha >= 1.0 - 1e-9) {
        return to;
      }
      Eigen::Quaterniond q_delta = Eigen::Quaterniond::FromTwoVectors(from, to);
      q_delta.normalize();
      Eigen::AngleAxisd aa_delta(q_delta);
      if (!std::isfinite(aa_delta.angle()) || !aa_delta.axis().allFinite()) {
        return to;
      }
      return (Eigen::AngleAxisd(alpha * aa_delta.angle(), aa_delta.axis()) * from).normalized();
    };

  auto normal_from_theta_height =
    [&](double theta, double height, Eigen::Vector3d& normal_out) -> bool {
      if (!(std::isfinite(hemisphere_radius_) && hemisphere_radius_ > 1e-9) ||
          hemisphere_axis_.norm() < 1e-9 ||
          !std::isfinite(theta)) {
        return false;
      }
      Eigen::Vector3d axis = Eigen::Vector3d::Zero();
      Eigen::Vector3d e1 = Eigen::Vector3d::Zero();
      Eigen::Vector3d e2 = Eigen::Vector3d::Zero();
      if (!build_hemisphere_basis_(axis, e1, e2)) {
        return false;
      }

      const double h = std::clamp(height, 0.0, hemisphere_radius_);
      const double cos_phi = std::clamp(h / hemisphere_radius_, 0.0, 1.0);
      const double sin_phi = std::sqrt(std::max(0.0, 1.0 - cos_phi * cos_phi));
      const Eigen::Vector3d t_circle = std::cos(theta) * e1 + std::sin(theta) * e2;
      normal_out = cos_phi * axis + sin_phi * t_circle;
      if (!normal_out.allFinite() || normal_out.norm() < 1e-9) {
        return false;
      }
      normal_out.normalize();
      return true;
    };

  // Build a right-handed orthonormal frame from a constrained primary axis and a secondary hint.
  // This is used to remove free-roll ambiguity when aligning only one tool axis to the surface normal.
  auto build_frame_from_primary = [&](const Eigen::Vector3d& primary,
                                      const Eigen::Vector3d& secondary_hint,
                                      Eigen::Matrix3d& frame_out) -> bool {
    if (primary.norm() < 1e-9) {
      return false;
    }
    const Eigen::Vector3d u = primary.normalized();

    Eigen::Vector3d v = secondary_hint - secondary_hint.dot(u) * u;
    if (v.norm() < 1e-9) {
      const Eigen::Vector3d seed =
        (std::abs(u.z()) < 0.9) ? Eigen::Vector3d::UnitZ() : Eigen::Vector3d::UnitX();
      v = seed - seed.dot(u) * u;
      if (v.norm() < 1e-9) {
        return false;
      }
    }
    v.normalize();

    Eigen::Vector3d w = u.cross(v);
    if (w.norm() < 1e-9) {
      return false;
    }
    w.normalize();
    v = w.cross(u);
    v.normalize();

    frame_out.col(0) = u;
    frame_out.col(1) = v;
    frame_out.col(2) = w;
    return true;
  };

  // ===============================
  // Fallback: Cartesian impedance HOLD (do NOT touch p_d_/q_d_ here)
  // ===============================
  auto compute_tau_cartesian_hold = [&]() -> Eigen::Matrix<double, 7, 1> {
    const Eigen::Vector3d e_p = p_d_ - p;

    Eigen::Quaterniond q_ee_fix = q_ee;
    if (q_d_.dot(q_ee_fix) < 0.0) {
      q_ee_fix.coeffs() *= -1.0;
    }
    const Eigen::Quaterniond q_err = q_d_ * q_ee_fix.conjugate();
    Eigen::AngleAxisd aa(q_err);
    if (!std::isfinite(aa.angle())) {
      aa = Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitX());
    }
    const Eigen::Vector3d e_o = aa.angle() * aa.axis();

    Eigen::Matrix<double, 6, 1> e;
    e.head<3>() = e_p;
    e.tail<3>() = e_o;

    Eigen::Matrix<double, 6, 1> de;
    de.head<3>() = v_des_cart.head<3>() - v_lin;
    de.tail<3>() = v_des_cart.tail<3>() - v_ang;

    const Eigen::Matrix<double, 6, 1> F_cart = K_cart_ * e + D_cart_ * de;

    Eigen::Matrix<double, 7, 1> tau = J.transpose() * F_cart;
    add_nullspace_posture(J, tau);
    tau += model_feedforward;
    tau -= D_joint_.cwiseProduct(dq);
    clamp_tau(tau);
    return tau;
  };

  // ===============================
  // Try "paper-like" p=[d,eps] impedance; fail -> fallback
  // ===============================
  auto try_compute_tau_paper = [&](Eigen::Matrix<double,7,1>& tau_out) -> bool {
    Eigen::Vector3d proxy_base = Eigen::Vector3d::Zero();
    double d = 0.0;
    Eigen::Vector3d n_base = Eigen::Vector3d::UnitZ();
    if (!query_hemisphere_proxy_(p, proxy_base, d, n_base)) {
      return false;
    }
    if (std::abs(d) > d_max_) {
      return false;
    }
    if (n_base.norm() < 1e-9) {
      return false;
    }
    n_base.normalize();

    // Convention consistent with your old outer loop:
    // align tool local -Z axis with -n_base
    const Eigen::Vector3d z_tool_local(0.0, 0.0, -1.0);
    Eigen::Vector3d z_tool_base = Tf.linear() * z_tool_local;
    if (z_tool_base.norm() < 1e-9) {
      return false;
    }
    z_tool_base.normalize();
    const Eigen::Vector3d z_target = (-n_base).normalized();

    Eigen::Quaterniond h = Eigen::Quaterniond::FromTwoVectors(z_tool_base, z_target);
    h.normalize();
    if (h.w() < 0.0) { h.coeffs() *= -1.0; }

    const double lambda = h.w();
    const Eigen::Vector3d eps = h.vec();

    // J_{eps,omega} (paper eq. style)
    Eigen::Matrix3d J_eps_omega;
    J_eps_omega <<  lambda,    eps.z(),  -eps.y(),
                   -eps.z(),   lambda,    eps.x(),
                    eps.y(),  -eps.x(),   lambda;
    J_eps_omega *= 0.5;

    // J_{eps,v}: curvature effect; sphere offset curvature approx: kappa_tilde = kappa/(1 - d*kappa)
    Eigen::Matrix3d J_eps_v = Eigen::Matrix3d::Zero();
    if (enable_curvature_ && hemisphere_radius_ > 1e-6) {
      const double kappa = 1.0 / hemisphere_radius_;
      const double denom = std::max(1e-6, 1.0 - d * kappa);
      const double kappa_tilde = kappa / denom;

      const Eigen::Matrix3d P_tan = Eigen::Matrix3d::Identity() - z_target * z_target.transpose();
      J_eps_v = J_eps_omega * skew(z_target) * (kappa_tilde * P_tan);
    }

    const Eigen::Matrix<double,3,7> J_v = J.topRows<3>();
    const Eigen::Matrix<double,3,7> J_w = J.bottomRows<3>();

    const Eigen::Vector3d c_dir =
      ((p - proxy_base).norm() > 1e-9) ? (p - proxy_base).normalized() : n_base;

    Eigen::Matrix<double,4,7> J_p;
    J_p.row(0) = c_dir.transpose() * J_v;                          // d_dot = c_hat^T v
    J_p.block<3,7>(1,0) = J_eps_v * J_v + J_eps_omega * J_w;       // eps_dot

    // task state
    Eigen::Matrix<double,4,1> p_task;
    p_task << d, eps.x(), eps.y(), eps.z();

    // desired: d_des_ and eps_des=0
    Eigen::Matrix<double,4,1> p_des;
    p_des << d_des_, 0.0, 0.0, 0.0;

    const Eigen::Matrix<double,4,1> p_dot = J_p * dq;

    // gains in p-space
    Eigen::Matrix<double,4,1> F_p;
    F_p.setZero();
    F_p(0) = k_d_ * (p_des(0) - p_task(0)) - d_d_ * p_dot(0);
    F_p.segment<3>(1) = k_eps_ * (p_des.segment<3>(1) - p_task.segment<3>(1)) - d_eps_ * p_dot.segment<3>(1);

    Eigen::Matrix<double,7,1> tau = J_p.transpose() * F_p;

    // nullspace spring (geometric projection)
    add_nullspace_posture(J_p, tau);

    tau += model_feedforward;
    tau -= D_joint_.cwiseProduct(dq);

    clamp_tau(tau);
    tau_out = tau;
    return true;
  };

  // ===============================
  // Mode switch + fallback
  // ===============================
  Eigen::Matrix<double,7,1> tau_cmd = Eigen::Matrix<double,7,1>::Zero();
  bool used_paper = false;
  bool scan_advance = !switch_hold_active;
  double scan_advance_scale = scan_advance ? 1.0 : 0.0;
  bool have_scan_ref = false;
  Eigen::Vector3d p_scan_ref = p_d_;
  Eigen::Vector3d n_scan_ref = Eigen::Vector3d::Zero();
  bool have_cmd_surface_ref = false;
  Eigen::Vector3d n_cmd_ref = Eigen::Vector3d::Zero();
  bool scan_surface_cmd_saturated = false;
  double scan_surface_cmd_ratio = 1.0;
  double scan_surface_ref_projection_err = 0.0;
  double scan_surface_safe_hold_lead_metric = 0.0;
  Eigen::Vector2d raw_scan_surface_ref = Eigen::Vector2d::Zero();
  bool have_raw_scan_surface_ref = false;

  auto local_surface_arc_metric =
    [&](const Eigen::Vector2d& s_a, const Eigen::Vector2d& s_b) {
      const double local_ds = (s_b - s_a).norm();
      if (!(std::isfinite(hemisphere_radius_) && hemisphere_radius_ > 1e-6)) {
        return local_ds;
      }
      Eigen::Vector3d n_a = Eigen::Vector3d::Zero();
      Eigen::Vector3d n_b = Eigen::Vector3d::Zero();
      if (!local_surface_coords_to_normal(
            scan_surface_origin_normal_, hemisphere_axis_, hemisphere_radius_, s_a, n_a) ||
          !local_surface_coords_to_normal(
            scan_surface_origin_normal_, hemisphere_axis_, hemisphere_radius_, s_b, n_b) ||
          n_a.norm() < 1e-9 || n_b.norm() < 1e-9) {
        return local_ds;
      }
      const double dot_val =
        std::clamp(n_a.normalized().dot(n_b.normalized()), -1.0, 1.0);
      const double ds_arc = hemisphere_radius_ * std::acos(dot_val);
      return std::isfinite(ds_arc) ? ds_arc : local_ds;
    };

  auto compute_scan_surface_reference =
    [&](double t,
        Eigen::Vector2d& s_ref,
        Eigen::Vector2d& s_dot_ref,
        Eigen::Vector3d& p_ref,
        Eigen::Vector3d& n_ref) -> bool {
      if (scan_track_generator_ != "paper_surface_lissajous") {
        return false;
      }
      if (!(std::isfinite(hemisphere_radius_) && hemisphere_radius_ > 1e-6) ||
          !scan_surface_origin_normal_valid_) {
        return false;
      }

      const double ramp_alpha = lock_lissajous_mapping ?
        1.0 :
        ((scan_ramp_time_ > 1e-6) ? std::clamp(scan_ramp_elapsed_ / scan_ramp_time_, 0.0, 1.0) : 1.0);
      const double seed_blend_alpha = lock_lissajous_mapping ?
        1.0 :
        ((scan_seed_blend_time_ > 1e-6) ?
        std::clamp(scan_seed_blend_elapsed_ / scan_seed_blend_time_, 0.0, 1.0) : 1.0);
      const double scan_phase_time = (lock_lissajous_mapping || !scan_seed_blend_active) ? t : 0.0;
      scan_surface_cmd_saturated = false;
      scan_surface_cmd_ratio = 1.0;
      scan_surface_ref_projection_err = 0.0;
      have_raw_scan_surface_ref = false;

      s_ref = scan_surface_seed_;
      s_ref.x() +=
        ramp_alpha * scan_surface_lissajous_amplitude_.x() *
        std::sin(scan_surface_lissajous_omega_.x() * scan_phase_time + scan_surface_lissajous_phase_.x());
      s_ref.y() +=
        ramp_alpha * scan_surface_lissajous_amplitude_.y() *
        std::sin(scan_surface_lissajous_omega_.y() * scan_phase_time + scan_surface_lissajous_phase_.y());
      s_ref = scan_surface_start_ + seed_blend_alpha * (s_ref - scan_surface_start_);

      Eigen::Vector2d s_raw = s_ref;
      if (!local_surface_coords_to_normal(
            scan_surface_origin_normal_, hemisphere_axis_, hemisphere_radius_, s_raw, n_ref)) {
        return false;
      }
      Eigen::Vector3d n_ref_projected = n_ref;
      if (!project_normal_to_scan_band_(n_ref, n_ref_projected)) {
        return false;
      }
      scan_surface_ref_projection_err = axis_angle_error(n_ref, n_ref_projected);
      n_ref = n_ref_projected;
      if (!normal_to_local_surface_coords(
            scan_surface_origin_normal_, hemisphere_axis_, hemisphere_radius_, n_ref, s_ref)) {
        return false;
      }
      raw_scan_surface_ref = s_ref;
      have_raw_scan_surface_ref = true;

      const double r_ref = hemisphere_radius_ + d_des_;
      if (!(std::isfinite(r_ref) && r_ref > 1e-4)) {
        return false;
      }
      p_ref = hemisphere_center_ + r_ref * n_ref;

      if (dt_slew > 1e-9 && scan_surface_des_prev_valid_) {
        s_dot_ref = (s_ref - scan_surface_des_prev_) / dt_slew;
      } else {
        s_dot_ref.setZero();
      }
      return true;
    };

  auto resolve_local_scan_surface_reference =
    [&](const Eigen::Vector2d& s_local_in,
        Eigen::Vector2d& s_local_out,
        Eigen::Vector3d& p_out,
        Eigen::Vector3d& n_out) -> bool {
      if (!(std::isfinite(hemisphere_radius_) && hemisphere_radius_ > 1e-6) ||
          !scan_surface_origin_normal_valid_ ||
          !s_local_in.allFinite()) {
        return false;
      }

      Eigen::Vector3d n_local = Eigen::Vector3d::Zero();
      if (!local_surface_coords_to_normal(
            scan_surface_origin_normal_, hemisphere_axis_, hemisphere_radius_, s_local_in, n_local)) {
        return false;
      }

      Eigen::Vector3d n_projected = n_local;
      if (!project_normal_to_scan_band_(n_local, n_projected)) {
        return false;
      }

      if (!normal_to_local_surface_coords(
            scan_surface_origin_normal_, hemisphere_axis_, hemisphere_radius_, n_projected, s_local_out)) {
        return false;
      }

      const double r_ref = hemisphere_radius_ + d_des_;
      if (!(std::isfinite(r_ref) && r_ref > 1e-4)) {
        return false;
      }

      n_out = n_projected;
      p_out = hemisphere_center_ + r_ref * n_out;
      return p_out.allFinite() && n_out.allFinite();
    };

  auto build_scan_display_quaternion =
    [&](const Eigen::Vector3d& surface_normal,
        Eigen::Quaterniond& q_scan_out,
        Eigen::Vector3d& target_axis_out) -> bool {
      target_axis_out = scan_align_inward_normal_ ? -surface_normal : surface_normal;
      if (target_axis_out.norm() < 1e-9) {
        return false;
      }
      target_axis_out.normalize();

      Eigen::Vector3d tool_axis_base = Tf.linear() * scan_tool_axis_runtime_;
      if (tool_axis_base.norm() < 1e-9) {
        return false;
      }
      tool_axis_base.normalize();

      if (!scan_hold_current_roll_) {
        // For surface scanning we only need the tool axis to follow the surface normal.
        // Pick the minimal-twist rotation from the current EE pose so the wrist does
        // not chase an arbitrary absolute roll reference that does not help the scan.
        q_scan_out = Eigen::Quaterniond::FromTwoVectors(tool_axis_base, target_axis_out) * q_ee;
        q_scan_out.normalize();
        return true;
      }

      const Eigen::Vector3d local_primary = scan_tool_axis_runtime_;
      Eigen::Vector3d local_secondary = Eigen::Vector3d::Zero();
      Eigen::Matrix3d R_local = Eigen::Matrix3d::Identity();
      Eigen::Matrix3d R_base_des = Eigen::Matrix3d::Identity();
      bool have_q_scan = false;
      q_scan_out = q_d_;

      if (resolve_secondary_axis_local(local_primary, local_secondary) &&
          build_frame_from_primary(local_primary, local_secondary, R_local)) {
        Eigen::Vector3d base_hint_prev_des = q_d_.toRotationMatrix() * R_local.col(1);
        if (base_hint_prev_des.norm() > 1e-9) {
          base_hint_prev_des.normalize();
        }
        const bool have_locked_roll_hint =
          scan_hold_current_roll_ && scan_roll_hint_valid_ &&
          build_frame_from_primary(target_axis_out, scan_roll_hint_base_, R_base_des);
        if (have_locked_roll_hint ||
            build_frame_from_primary(target_axis_out, base_hint_prev_des, R_base_des) ||
            build_frame_from_primary(target_axis_out, hemisphere_axis_, R_base_des)) {
          const Eigen::Matrix3d R_scan = R_base_des * R_local.transpose();
          q_scan_out = Eigen::Quaterniond(R_scan);
          q_scan_out.normalize();
          have_q_scan = true;
        }
      }
      if (!have_q_scan) {
        q_scan_out = Eigen::Quaterniond::FromTwoVectors(tool_axis_base, target_axis_out) * q_ee;
        q_scan_out.normalize();
      }
      return true;
    };

  [[maybe_unused]] auto command_local_scan_pose =
    [&](const Eigen::Vector3d& p_target,
        const Eigen::Vector3d& n_target,
        bool align_orientation,
        bool cap_tracking_error) -> bool {
      if (p_target.norm() < 1e-9 || n_target.norm() < 1e-9) {
        return false;
      }

      have_scan_ref = true;
      p_scan_ref = p_target;
      n_scan_ref = n_target.normalized();

      const double linear_speed_limit =
        std::max(0.0, align_orientation ? scan_max_linear_speed_ : scan_acquire_max_linear_speed_);
      const double max_dp = linear_speed_limit * dt_slew;
      Eigen::Vector3d dp = p_target - p_d_;
      if (max_dp > 1e-9 && dp.norm() > max_dp) {
        dp = (max_dp / dp.norm()) * dp;
      }
      p_d_ += dp;

      if (cap_tracking_error) {
        Eigen::Vector3d e_track = p_d_ - p;
        Eigen::Vector3d surface_normal = n_target;
        if (surface_normal.norm() < 1e-9) {
          surface_normal = p - hemisphere_center_;
        }
        if (surface_normal.norm() < 1e-9) {
          surface_normal = Eigen::Vector3d::UnitZ();
        }
        surface_normal.normalize();

        const double max_pos_err = std::max(0.0, scan_max_position_error_);
        const double max_rad_err = std::max(
          0.0, (scan_max_radial_error_ > 0.0) ? scan_max_radial_error_ : max_pos_err);
        const double max_tan_err = std::max(
          0.0, (scan_max_tangential_error_ > 0.0) ? scan_max_tangential_error_ : max_pos_err);
        const bool use_total_pos_cap =
          (scan_max_radial_error_ <= 0.0) && (scan_max_tangential_error_ <= 0.0);

        double e_rad = e_track.dot(surface_normal);
        Eigen::Vector3d e_tan = e_track - e_rad * surface_normal;
        if (max_rad_err > 1e-9) {
          e_rad = std::clamp(e_rad, -max_rad_err, max_rad_err);
        }
        if (max_tan_err > 1e-9 && e_tan.norm() > max_tan_err) {
          e_tan *= max_tan_err / e_tan.norm();
        }
        if (use_total_pos_cap && max_pos_err > 1e-9) {
          Eigen::Vector3d e_safe = e_rad * surface_normal + e_tan;
          if (e_safe.norm() > max_pos_err) {
            e_safe *= max_pos_err / e_safe.norm();
            e_rad = e_safe.dot(surface_normal);
            e_tan = e_safe - e_rad * surface_normal;
          }
        }
        p_d_ = p + e_rad * surface_normal + e_tan;
      }

      have_cmd_surface_ref = false;

      if (align_orientation) {
        Eigen::Vector3d n_cmd = n_target.normalized();
        have_cmd_surface_ref = true;
        n_cmd_ref = n_cmd;

        Eigen::Quaterniond q_scan = q_d_;
        Eigen::Vector3d target_axis_cmd = Eigen::Vector3d::Zero();
        if (build_scan_display_quaternion(n_cmd, q_scan, target_axis_cmd)) {
          double max_da = std::max(0.0, scan_max_angular_speed_) * dt_slew;
          if (q_d_.dot(q_scan) < 0.0) {
            q_scan.coeffs() *= -1.0;
          }
          if (max_da > 1e-9) {
            Eigen::Quaterniond q_err = q_d_.conjugate() * q_scan;
            q_err.normalize();
            Eigen::AngleAxisd aa(q_err);
            const double ang = std::abs(aa.angle());
            if (std::isfinite(ang) && ang > max_da) {
              q_d_ = q_d_.slerp(max_da / ang, q_scan);
              q_d_.normalize();
            } else {
              q_d_ = q_scan;
            }
          } else {
            q_d_ = q_scan;
          }

          const double max_ori_err = std::max(0.0, scan_max_orientation_error_);
          if (max_ori_err > 1e-9) {
            Eigen::Quaterniond q_d_fix = q_d_;
            if (q_ee.dot(q_d_fix) < 0.0) {
              q_d_fix.coeffs() *= -1.0;
            }
            Eigen::Quaterniond q_err_track = q_ee.conjugate() * q_d_fix;
            q_err_track.normalize();
            Eigen::AngleAxisd aa_track(q_err_track);
            const double ang_track = std::abs(aa_track.angle());
            if (std::isfinite(ang_track) && ang_track > max_ori_err) {
              q_d_ = q_ee.slerp(max_ori_err / ang_track, q_d_fix);
              q_d_.normalize();
            }
          }
        }
      } else {
        v_des_cart.tail<3>().setZero();
      }

      if (dt_slew > 1e-9) {
        v_des_cart.head<3>() = (p_d_ - p_d_prev) / dt_slew;
        if (align_orientation) {
          Eigen::Quaterniond q_d_prev_fix = q_d_prev;
          if (q_d_prev_fix.dot(q_d_) < 0.0) {
            q_d_prev_fix.coeffs() *= -1.0;
          }
          Eigen::Quaterniond q_delta = q_d_ * q_d_prev_fix.conjugate();
          q_delta.normalize();
          Eigen::AngleAxisd aa_delta(q_delta);
          if (std::isfinite(aa_delta.angle()) && aa_delta.axis().allFinite()) {
            v_des_cart.tail<3>() = (aa_delta.angle() / dt_slew) * aa_delta.axis();
          }
        }
      }

      return true;
    };

  auto try_compute_tau_paper_surface = [&](Eigen::Matrix<double,7,1>& tau_out) -> bool {
    auto fail_paper_surface = [&](const char* why) -> bool {
      RCLCPP_WARN_THROTTLE(
        get_node()->get_logger(), *get_node()->get_clock(), 1000,
        "paper_surface_scan unavailable: %s", why);
      scan_surface_raw_prev_valid_ = false;
      scan_surface_actual_prev_valid_ = false;
      scan_surface_phase_progress_ratio_ = 1.0;
      return false;
    };

    double d = 0.0;
    Eigen::Vector3d n_base_raw = Eigen::Vector3d::Zero();
    if (!query_hemisphere_true_(p, d, n_base_raw)) {
      return fail_paper_surface("query_hemisphere_true");
    }
    if (!std::isfinite(d) ||
        (std::isfinite(hemisphere_radius_) && (d <= (-0.95 * hemisphere_radius_)))) {
      return fail_paper_surface("invalid_signed_distance");
    }
    if (n_base_raw.norm() < 1e-9) {
      return fail_paper_surface("degenerate_surface_normal");
    }
    n_base_raw.normalize();

    Eigen::Vector3d proxy_base_raw = hemisphere_center_ + hemisphere_radius_ * n_base_raw;
    Eigen::Vector3d proxy_base = proxy_base_raw;
    Eigen::Vector3d n_base = n_base_raw;
    double theta_band = scan_theta_seed_;
    double phi_band = scan_phi_seed_;
    if (project_normal_to_scan_band_(n_base_raw, n_base, &theta_band, &phi_band)) {
      proxy_base = hemisphere_center_ + hemisphere_radius_ * n_base;
    } else {
      n_base = n_base_raw;
      proxy_base = proxy_base_raw;
    }
    // Keep APRP / band projection for the tangential chart only. The radial
    // surface-distance task must stay tied to the true surface normal; using
    // the projected in-band normal here can push the TCP away from the actual
    // surface once the contact patch starts drifting toward the band edge.
    const Eigen::Vector3d d_dir = n_base_raw;
    const double d_surface = d;
    if (!std::isfinite(d_surface)) {
      return fail_paper_surface("nonfinite_projected_surface_distance");
    }
    const double acquire_band_err = axis_angle_error(n_base_raw, n_base);
    debug_scan_surface_dist_err_ = std::abs(d_surface - d_des_);
    debug_scan_band_err_ = acquire_band_err;
    double actual_surface_height_above_base = std::numeric_limits<double>::infinity();
    double actual_tcp_height_above_base = std::numeric_limits<double>::infinity();
    bool actual_surface_above_scan_height_release = true;
    bool actual_tcp_above_scan_height_release = true;
    bool actual_surface_above_scan_height_reacquire = true;
    bool actual_tcp_above_scan_height_reacquire = true;
    constexpr double kAcquireHeightReacquireSlack = 0.003;  // [m]
    constexpr double kAcquireHeightReleaseMargin = 0.005;   // [m]
    constexpr double kAcquireMinHeightScale = 0.015;      // [m]
    const double min_scan_height = std::max(0.0, scan_min_height_above_base_);
    const double min_tcp_height =
      std::max(0.0, min_scan_height + std::min(0.0, d_des_));
    if (hemisphere_axis_.norm() > 1e-9 && hemisphere_radius_ > 1e-9) {
      const Eigen::Vector3d hemi_axis = hemisphere_axis_.normalized();
      // The hemisphere "base" is the flat cut plane through the sphere center,
      // not the south pole of the full sphere.
      actual_surface_height_above_base = hemi_axis.dot(proxy_base_raw - hemisphere_center_);
      actual_tcp_height_above_base = hemi_axis.dot(p - hemisphere_center_);
      if (scan_min_height_above_base_ > 1e-9) {
        actual_surface_above_scan_height_reacquire =
          actual_surface_height_above_base >=
          (min_scan_height - kAcquireHeightReacquireSlack);
        actual_tcp_above_scan_height_reacquire =
          actual_tcp_height_above_base >=
          (min_tcp_height - kAcquireHeightReacquireSlack);
        actual_surface_above_scan_height_release =
          actual_surface_height_above_base >=
          (min_scan_height + kAcquireHeightReleaseMargin);
        actual_tcp_above_scan_height_release =
          actual_tcp_height_above_base >=
          (min_tcp_height + kAcquireHeightReleaseMargin);
      }
    }
    const double acquire_band_release_tol = std::max(0.0, scan_acquire_release_band_tol_);
    const double progress_band_tol = std::max(0.0, scan_progress_band_tol_);
    const double hard_stop_band_tol = std::max(
      progress_band_tol + 1e-6,
      std::max(0.0, scan_progress_hard_stop_band_tol_));
    const double acquire_band_reacquire_tol = std::max(
      std::max(acquire_band_release_tol + 1e-6, hard_stop_band_tol + 1e-6),
      std::max(0.0, scan_acquire_reacquire_band_tol_));
    const double acquire_band_release_check_tol = std::min(
      acquire_band_reacquire_tol - 1e-6,
      std::max(acquire_band_release_tol, progress_band_tol));
    const bool actual_in_scan_band = acquire_band_err <= acquire_band_release_check_tol;
    Eigen::Vector2d s_ref = Eigen::Vector2d::Zero();
    Eigen::Vector2d s_dot_ref = Eigen::Vector2d::Zero();
    Eigen::Vector3d p_ref = p_d_;
    Eigen::Vector3d n_ref = Eigen::Vector3d::UnitZ();
    if (!compute_scan_surface_reference(scan_time_, s_ref, s_dot_ref, p_ref, n_ref)) {
      return fail_paper_surface("compute_scan_surface_reference");
    }
    const Eigen::Vector3d p_ref_raw = p_ref;
    const Eigen::Vector3d n_ref_raw = n_ref;

    const double acquire_tol_limit = std::max(0.002, d_max_);
    const double default_paper_acquire_release_tol = std::min(
      std::max(
        0.005,
        (scan_max_radial_error_ > 0.0) ? scan_max_radial_error_ : std::max(0.0, scan_progress_pos_tol_)),
      acquire_tol_limit);
    const double paper_acquire_release_tol = std::clamp(
      (scan_acquire_release_surface_tol_ > 0.0) ?
        scan_acquire_release_surface_tol_ : default_paper_acquire_release_tol,
      0.002, acquire_tol_limit);
    const double default_paper_reacquire_tol = std::max(0.008, 1.5 * paper_acquire_release_tol);
    const double paper_reacquire_tol = std::clamp(
      (scan_acquire_reacquire_surface_tol_ > 0.0) ?
        scan_acquire_reacquire_surface_tol_ : default_paper_reacquire_tol,
      paper_acquire_release_tol + 1e-6,
      std::max(paper_acquire_release_tol + 1e-6, acquire_tol_limit));
    const double paper_acquire_ori_tol = std::max(0.45, std::max(0.0, scan_progress_ori_tol_));
    const double paper_reacquire_release_ori_tol =
      (debug_scan_reacquire_reason_ >= 4.0 && scan_progress_ori_tol_ > 1e-9) ?
      std::min(paper_acquire_ori_tol, std::max(0.0, scan_progress_ori_tol_)) :
      paper_acquire_ori_tol;
    const double paper_reacquire_release_surface_tol =
      (debug_scan_reacquire_reason_ >= 4.0) ?
      std::min(
        paper_reacquire_tol,
        std::max(
          paper_acquire_release_tol,
          0.75 * std::max(
            paper_acquire_release_tol,
            std::max(0.0, scan_progress_hard_stop_surface_dist_tol_)))) :
      paper_acquire_release_tol;
    Eigen::Vector3d n_acquire_target = lock_lissajous_mapping ? n_ref : n_base;
    double theta_acquire_target = scan_theta_seed_;
    double phi_acquire_target = scan_phi_seed_;
    if (!project_normal_to_scan_band_(
          lock_lissajous_mapping ? n_ref : n_base,
          n_acquire_target,
          &theta_acquire_target,
          &phi_acquire_target)) {
      return fail_paper_surface("project_normal_to_scan_band_acquire");
    }
    const double acquire_surface_err = std::abs(d_surface - d_des_);
    const double acquire_blend_far = std::max(0.06, paper_reacquire_tol);
    const double acquire_surface_alpha =
      (acquire_blend_far > paper_acquire_release_tol + 1e-6) ?
      std::clamp(
        1.0 - ((acquire_surface_err - paper_acquire_release_tol) /
               (acquire_blend_far - paper_acquire_release_tol)),
        0.0, 1.0) : 1.0;
    const double band_contact_alpha =
      (acquire_band_reacquire_tol > acquire_band_release_tol + 1e-6) ?
      std::clamp(
        1.0 - ((acquire_band_err - acquire_band_release_tol) /
               (acquire_band_reacquire_tol - acquire_band_release_tol)),
        0.0, 1.0) :
      (actual_in_scan_band ? 1.0 : 0.0);
    const double band_recovery_alpha = std::clamp(1.0 - band_contact_alpha, 0.0, 1.0);
    double recovery_height_alpha = 0.0;
    double recovery_tcp_alpha = 0.0;
    Eigen::Vector3d n_acquire_goal = n_acquire_target;
    if (scan_min_height_above_base_ > 1e-9 &&
        hemisphere_axis_.norm() > 1e-9 &&
        std::isfinite(hemisphere_radius_) &&
        hemisphere_radius_ > 1e-9) {
      const double recovery_surface_height = std::clamp(
        min_scan_height + std::max(0.0, scan_acquire_recovery_height_margin_),
        min_scan_height, hemisphere_radius_);
      const double recovery_tcp_height =
        std::max(0.0, recovery_surface_height + std::min(0.0, d_des_));
      const double surface_deficit = std::max(
        0.0, recovery_surface_height - actual_surface_height_above_base);
      const double tcp_deficit = std::max(
        0.0, recovery_tcp_height - actual_tcp_height_above_base);
      recovery_height_alpha = std::clamp(
        surface_deficit / std::max(kAcquireMinHeightScale, recovery_surface_height),
        0.0, 1.0);
      recovery_tcp_alpha = std::clamp(
        tcp_deficit / std::max(kAcquireMinHeightScale, recovery_surface_height),
        0.0, 1.0);

      Eigen::Vector3d n_height_recovery = Eigen::Vector3d::Zero();
      if (normal_from_theta_height(
            theta_acquire_target, recovery_surface_height, n_height_recovery)) {
        const double recovery_alpha = std::max(recovery_height_alpha, recovery_tcp_alpha);
        n_acquire_goal =
          slerp_unit_vectors(n_acquire_target, n_height_recovery, recovery_alpha);
      }
    }
    if (scan_surface_origin_normal_valid_ &&
        scan_surface_origin_normal_.norm() > 1e-9 &&
        band_recovery_alpha > 1e-6) {
      Eigen::Vector3d n_chart_recovery = scan_surface_origin_normal_;
      if (project_normal_to_scan_band_(scan_surface_origin_normal_, n_chart_recovery)) {
        n_acquire_goal =
          slerp_unit_vectors(n_acquire_goal, n_chart_recovery, band_recovery_alpha);
      }
    }
    const double height_contact_alpha =
      std::clamp(1.0 - std::max(recovery_height_alpha, recovery_tcp_alpha), 0.0, 1.0);
    const double acquire_contact_alpha =
      std::min(acquire_surface_alpha, std::min(band_contact_alpha, height_contact_alpha));
    const double acquire_hover_offset = std::max(0.0, scan_acquire_hover_surface_offset_);
    const double d_acquire_target =
      acquire_hover_offset + acquire_contact_alpha * (d_des_ - acquire_hover_offset);
    const double r_acquire = hemisphere_radius_ + d_acquire_target;
    if (!(std::isfinite(r_acquire) && r_acquire > 1e-4)) {
      return fail_paper_surface("invalid_acquire_radius");
    }
    const double acquire_recovery_ang_speed = std::clamp(
      scan_acquire_recovery_max_angular_speed_, 0.0, std::max(0.0, scan_acquire_max_angular_speed_));
    const double acquire_ang_speed_nominal =
      acquire_recovery_ang_speed +
      acquire_contact_alpha *
      std::max(0.0, scan_acquire_max_angular_speed_ - acquire_recovery_ang_speed);
    const double acquire_ang_speed =
      std::max(
        acquire_ang_speed_nominal,
        band_recovery_alpha * std::max(0.0, scan_acquire_max_angular_speed_));
    const Eigen::Vector3d n_acquire_cmd =
      update_scan_normal_command(n_acquire_goal, acquire_ang_speed, false);
    Eigen::Vector3d n_acquire_source = n_base_raw;
    if ((scan_min_height_above_base_ > 1e-9 &&
         (!actual_surface_above_scan_height_release || !actual_tcp_above_scan_height_release)) ||
        (acquire_band_err > acquire_band_release_check_tol + 1e-6) ||
        (hemisphere_axis_.norm() > 1e-9 &&
         n_acquire_source.dot(hemisphere_axis_.normalized()) < 0.0)) {
      n_acquire_source = n_acquire_goal;
    }
    const double acquire_position_alpha =
      std::clamp(
        std::max(
          std::max(acquire_surface_alpha, band_recovery_alpha),
          std::max(recovery_height_alpha, recovery_tcp_alpha)),
        0.0, 1.0);
    const Eigen::Vector3d n_acquire_pos =
      slerp_unit_vectors(n_acquire_source, n_acquire_cmd, acquire_position_alpha);
    const Eigen::Vector3d p_acquire = hemisphere_center_ + r_acquire * n_acquire_pos;
    bool paper_just_acquired = false;
    double acq_cmd_ori_err = std::numeric_limits<double>::infinity();
    Eigen::Vector3d tool_axis_cur = Tf.linear() * scan_tool_axis_runtime_;
    if (tool_axis_cur.norm() > 1e-9) {
      tool_axis_cur.normalize();
    }
    {
      Eigen::Quaterniond q_acq_gate = q_d_;
      Eigen::Vector3d target_axis_gate = Eigen::Vector3d::Zero();
      if (build_scan_display_quaternion(n_acquire_cmd, q_acq_gate, target_axis_gate) &&
          target_axis_gate.norm() > 1e-9) {
        target_axis_gate.normalize();
        acq_cmd_ori_err = axis_angle_error(tool_axis_cur, target_axis_gate);
      }
    }

    const bool reacquire_for_surface = std::abs(d_surface - d_des_) >= paper_reacquire_tol;
    const bool suppress_acquire_reacquire =
      paper_surface_acquired_ &&
      scan_progress_hard_stop_active_ &&
      (scan_progress_hard_stop_ori_elapsed_ > 1e-9) &&
      (scan_progress_hard_stop_surface_elapsed_ <= 1e-9);
    if (paper_surface_acquired_ &&
        reacquire_for_surface &&
        dt_slew > 1e-9 &&
        !suppress_acquire_reacquire) {
      scan_acquire_surface_reacquire_elapsed_ += dt_slew;
    } else {
      scan_acquire_surface_reacquire_elapsed_ = 0.0;
    }
    const bool reacquire_for_band =
      (scan_min_height_above_base_ > 1e-9 &&
       (!actual_surface_above_scan_height_reacquire || !actual_tcp_above_scan_height_reacquire)) ||
      (acquire_band_err > acquire_band_reacquire_tol);
    if (paper_surface_acquired_ &&
        reacquire_for_band &&
        dt_slew > 1e-9 &&
        !suppress_acquire_reacquire) {
      scan_acquire_band_reacquire_elapsed_ += dt_slew;
    } else {
      scan_acquire_band_reacquire_elapsed_ = 0.0;
    }
    const bool reacquire_for_surface_persisted =
      reacquire_for_surface &&
      (scan_acquire_reacquire_surface_time_ <= 1e-9 ||
       scan_acquire_surface_reacquire_elapsed_ >= scan_acquire_reacquire_surface_time_);
    const bool reacquire_for_band_persisted =
      reacquire_for_band &&
      (scan_acquire_reacquire_surface_time_ <= 1e-9 ||
       scan_acquire_band_reacquire_elapsed_ >= scan_acquire_reacquire_surface_time_);
    if (paper_surface_acquired_ &&
        (reacquire_for_surface_persisted || reacquire_for_band_persisted)) {
      RCLCPP_WARN(
        get_node()->get_logger(),
        "paper_scan reacquire: surface=%d band=%d dist_err=%.4f/%.4f band_err=%.4f/%.4f "
        "height(surface=%.4f,tcp=%.4f,min=%.4f,ok=%d/%d) acq_cmd_ori=%.4f/%.4f",
        reacquire_for_surface_persisted ? 1 : 0,
        reacquire_for_band_persisted ? 1 : 0,
        acquire_surface_err,
        paper_reacquire_tol,
        acquire_band_err,
        acquire_band_reacquire_tol,
        actual_surface_height_above_base,
        actual_tcp_height_above_base,
        scan_min_height_above_base_,
        actual_surface_above_scan_height_reacquire ? 1 : 0,
        actual_tcp_above_scan_height_reacquire ? 1 : 0,
        acq_cmd_ori_err,
        paper_reacquire_release_ori_tol);
      paper_surface_acquired_ = false;
      scan_surface_cmd_valid_ = false;
      scan_surface_raw_prev_valid_ = false;
      scan_surface_actual_prev_valid_ = false;
      scan_surface_phase_progress_ratio_ = 1.0;
      scan_surface_des_prev_valid_ = false;
      scan_acquire_surface_reacquire_elapsed_ = 0.0;
      scan_acquire_band_reacquire_elapsed_ = 0.0;
      scan_progress_hard_stop_active_ = false;
      scan_progress_hard_stop_surface_elapsed_ = 0.0;
      scan_progress_hard_stop_ori_elapsed_ = 0.0;
      double reacquire_reason = 0.0;
      if (reacquire_for_surface_persisted) {
        reacquire_reason += 1.0;
      }
      if (reacquire_for_band_persisted) {
        reacquire_reason += 2.0;
      }
      debug_scan_reacquire_reason_ = reacquire_reason;
    }
    if (!paper_surface_acquired_ &&
        std::abs(d_surface - d_des_) <= paper_reacquire_release_surface_tol &&
        actual_in_scan_band &&
        actual_surface_above_scan_height_release &&
        actual_tcp_above_scan_height_release &&
        std::isfinite(acq_cmd_ori_err) &&
        acq_cmd_ori_err <= paper_reacquire_release_ori_tol) {
      const Eigen::Vector3d n_chart_origin = n_acquire_goal;
      const double theta_acquired = theta_acquire_target;
      const double phi_acquired = phi_acquire_target;
      if (!lock_lissajous_mapping) {
        scan_theta_start_ = theta_acquired;
        scan_phi_start_ = phi_acquired;
        scan_theta_seed_ = theta_acquired;
        scan_phi_seed_ = phi_acquired;
      }
      paper_surface_acquired_ = true;
      paper_just_acquired = true;
      if (scan_hold_current_roll_ &&
          compute_roll_hint_in_base(q_ee, scan_tool_axis_runtime_, scan_roll_hint_base_)) {
        scan_roll_hint_valid_ = true;
      }
      if (!lock_lissajous_mapping) {
        scan_time_ = 0.0;
        scan_ramp_elapsed_ = 0.0;
      }
      scan_acquire_surface_reacquire_elapsed_ = 0.0;
      scan_acquire_band_reacquire_elapsed_ = 0.0;
      scan_progress_hard_stop_active_ = false;
      scan_progress_hard_stop_surface_elapsed_ = 0.0;
      scan_progress_hard_stop_ori_elapsed_ = 0.0;
      debug_scan_reacquire_reason_ = 0.0;
      if (!lock_lissajous_mapping) {
        scan_surface_origin_normal_ = n_chart_origin;
        scan_surface_origin_normal_valid_ = true;
      }
      Eigen::Vector2d s_acquire_cur = Eigen::Vector2d::Zero();
      const bool have_s_acquire_cur =
        normal_to_local_surface_coords(
          scan_surface_origin_normal_, hemisphere_axis_, hemisphere_radius_, n_base, s_acquire_cur);
      if (lock_lissajous_mapping) {
        scan_surface_start_ = scan_surface_seed_;
        if (have_s_acquire_cur) {
          scan_surface_cmd_ = s_acquire_cur;
          scan_surface_cmd_valid_ = true;
          scan_surface_actual_prev_ = s_acquire_cur;
          scan_surface_actual_prev_valid_ = true;
        } else {
          scan_surface_cmd_.setZero();
          scan_surface_cmd_valid_ = false;
          scan_surface_actual_prev_.setZero();
          scan_surface_actual_prev_valid_ = false;
        }
        scan_surface_raw_prev_.setZero();
        scan_surface_raw_prev_valid_ = false;
        scan_surface_phase_progress_ratio_ = 1.0;
      } else if (have_s_acquire_cur) {
        // Re-express the current contact point in the newly latched APRP chart.
        // Keep the scan center tied to the safer chart origin rather than the
        // current contact patch, then blend from the current local point into
        // that chart so we do not permanently center the orbit on a low edge.
        scan_surface_start_ = s_acquire_cur;
        scan_surface_seed_ = scan_surface_lissajous_offset_;
        scan_surface_cmd_ = scan_surface_start_;
        scan_surface_cmd_valid_ = true;
        scan_surface_raw_prev_.setZero();
        scan_surface_raw_prev_valid_ = false;
        scan_surface_actual_prev_ = scan_surface_start_;
        scan_surface_actual_prev_valid_ = true;
        scan_surface_phase_progress_ratio_ = 1.0;
      } else {
        scan_surface_start_.setZero();
        scan_surface_seed_ = scan_surface_lissajous_offset_;
        scan_surface_cmd_.setZero();
        scan_surface_cmd_valid_ = false;
        scan_surface_raw_prev_.setZero();
        scan_surface_raw_prev_valid_ = false;
        scan_surface_actual_prev_.setZero();
        scan_surface_actual_prev_valid_ = false;
        scan_surface_phase_progress_ratio_ = 1.0;
      }
      if (lock_lissajous_mapping) {
        scan_seed_blend_elapsed_ = std::max(0.0, scan_seed_blend_time_);
      } else {
        const bool need_post_acquire_seed_blend =
          (scan_seed_blend_time_ > 1e-6) &&
          ((scan_surface_start_ - scan_surface_seed_).norm() > 1e-6);
        scan_seed_blend_elapsed_ =
          need_post_acquire_seed_blend ? 0.0 : std::max(0.0, scan_seed_blend_time_);
      }
      scan_surface_des_prev_valid_ = false;
    }
    const bool paper_acquire_active =
      scan_seed_blend_active || !paper_surface_acquired_ || paper_just_acquired;
    debug_paper_acquire_active_ = paper_acquire_active;
    debug_scan_cmd_ori_err_ = acq_cmd_ori_err;
    if (paper_acquire_active) {
      scan_progress_hard_stop_surface_elapsed_ = 0.0;
      scan_progress_hard_stop_ori_elapsed_ = 0.0;
      if (!command_local_scan_pose(p_acquire, n_acquire_cmd, false, false)) {
        return fail_paper_surface("command_local_scan_pose");
      }
      have_cmd_surface_ref = true;
      n_cmd_ref = n_acquire_cmd;

      Eigen::Quaterniond q_acq = q_d_;
      Eigen::Vector3d target_axis_dbg = Eigen::Vector3d::Zero();
      if (build_scan_display_quaternion(n_acquire_cmd, q_acq, target_axis_dbg)) {
        const double max_da = std::max(0.0, scan_acquire_max_angular_speed_) * dt_slew;
        if (q_d_.dot(q_acq) < 0.0) {
          q_acq.coeffs() *= -1.0;
        }
        if (max_da > 1e-9) {
          Eigen::Quaterniond q_err = q_d_.conjugate() * q_acq;
          q_err.normalize();
          Eigen::AngleAxisd aa(q_err);
          const double ang = std::abs(aa.angle());
          if (std::isfinite(ang) && ang > max_da) {
            q_d_ = q_d_.slerp(max_da / ang, q_acq);
            q_d_.normalize();
          } else {
            q_d_ = q_acq;
          }
        } else {
          q_d_ = q_acq;
        }
      }

      if (dt_slew > 1e-9) {
        Eigen::Quaterniond q_d_prev_fix = q_d_prev;
        if (q_d_prev_fix.dot(q_d_) < 0.0) {
          q_d_prev_fix.coeffs() *= -1.0;
        }
        Eigen::Quaterniond q_delta = q_d_ * q_d_prev_fix.conjugate();
        q_delta.normalize();
        Eigen::AngleAxisd aa_delta(q_delta);
        if (std::isfinite(aa_delta.angle()) && aa_delta.axis().allFinite()) {
          v_des_cart.tail<3>() = (aa_delta.angle() / dt_slew) * aa_delta.axis();
        } else {
          v_des_cart.tail<3>().setZero();
        }
      } else {
        v_des_cart.tail<3>().setZero();
      }

      Eigen::Quaterniond q_ee_fix = q_ee;
      if (q_d_.dot(q_ee_fix) < 0.0) {
        q_ee_fix.coeffs() *= -1.0;
      }
      const Eigen::Quaterniond q_err = q_d_ * q_ee_fix.conjugate();
      Eigen::AngleAxisd aa(q_err);
      if (!std::isfinite(aa.angle())) {
        aa = Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitX());
      }

      Eigen::Matrix<double, 6, 1> e = Eigen::Matrix<double, 6, 1>::Zero();
      e.head<3>() = p_d_ - p;
      e.tail<3>() = aa.angle() * aa.axis();

      Eigen::Matrix<double, 6, 1> de = Eigen::Matrix<double, 6, 1>::Zero();
      de.head<3>() = v_des_cart.head<3>() - v_lin;
      de.tail<3>() = v_des_cart.tail<3>() - v_ang;

      const Eigen::Matrix<double, 6, 1> F_acq = K_cart_ * e + D_cart_ * de;
      Eigen::Matrix<double,7,1> tau = J.transpose() * F_acq;
      // Acquisition commands a full Cartesian pose, so there is only one
      // redundant joint DOF left. Use only a weak posture bias here: enough
      // to discourage the arm from flattening into an awkward shape, but not
      // so much that it noticeably fights the pull-in motion toward the scan
      // patch.
      if (nullspace_enable_ && null_q_des_initialized_) {
        constexpr double kAcquireNullspaceScale = 0.26;
        double surface_height_alpha = 0.0;
        double tcp_height_alpha = 0.0;
        if (hemisphere_axis_.norm() > 1e-9 &&
            std::isfinite(hemisphere_radius_) &&
            hemisphere_radius_ > 1e-9 &&
            scan_min_height_above_base_ > 1e-9) {
          const double desired_surface_height = std::clamp(
            scan_min_height_above_base_ + std::max(0.0, scan_acquire_recovery_height_margin_),
            0.0, hemisphere_radius_);
          const double desired_tcp_height =
            std::max(0.0, desired_surface_height + std::min(0.0, d_des_));
          const double actual_surface_height =
            std::isfinite(actual_surface_height_above_base) ?
            actual_surface_height_above_base : desired_surface_height;
          const double actual_tcp_height =
            std::isfinite(actual_tcp_height_above_base) ?
            actual_tcp_height_above_base : desired_tcp_height;
          const double surface_height_deficit =
            std::max(0.0, desired_surface_height - actual_surface_height);
          const double tcp_height_deficit =
            std::max(0.0, desired_tcp_height - actual_tcp_height);
          surface_height_alpha = std::clamp(
            surface_height_deficit / std::max(0.02, desired_surface_height),
            0.0, 1.0);
          tcp_height_alpha = std::clamp(
            tcp_height_deficit / std::max(0.02, desired_surface_height),
            0.0, 1.0);
        }
        // Keep only a height-protection posture bias here. Using distance-to-surface
        // as a nullspace trigger makes the whole arm hunt left-right while the TCP
        // is still far from the patch, which hurts acquisition much more than it helps.
        const double posture_alpha = std::max(surface_height_alpha, tcp_height_alpha);
        if (posture_alpha > 1e-6) {
          const Eigen::Matrix<double, 7, 7> N =
            damped_nullspace_projector(J, nullspace_damping_);
          tau += (kAcquireNullspaceScale * posture_alpha) *
            N * (null_kp_ * (null_q_des_ - q) - null_kd_ * dq);
        }
      }
      tau += model_feedforward;
      tau -= D_joint_.cwiseProduct(dq);
      clamp_tau(tau);
      tau_out = tau;
      scan_progress_hard_stop_active_ = false;
      scan_surface_cmd_valid_ = false;
      scan_surface_des_prev_valid_ = false;
      scan_advance = false;
      scan_advance_scale = 0.0;
      return true;
    }

    Eigen::Vector2d s_cur = Eigen::Vector2d::Zero();
    if (!scan_surface_origin_normal_valid_ ||
        !normal_to_local_surface_coords(
          scan_surface_origin_normal_, hemisphere_axis_, hemisphere_radius_, n_base, s_cur)) {
      return fail_paper_surface("normal_to_local_surface_coords_current");
    }

    Eigen::Vector2d scan_recovery_surface_ref = Eigen::Vector2d::Zero();
    const bool have_scan_recovery_surface_ref =
      scan_surface_origin_normal_valid_ &&
        n_acquire_cmd.norm() > 1e-9 &&
        normal_to_local_surface_coords(
          scan_surface_origin_normal_, hemisphere_axis_, hemisphere_radius_,
          n_acquire_cmd, scan_recovery_surface_ref);
    double cmd_ori_soft_err = 0.0;
    bool have_cmd_ori_soft_err = false;
    if (have_raw_scan_surface_ref) {
      auto soft_gate_ratio = [&](double err, double release_tol, double hard_tol) {
        if (!std::isfinite(err)) {
          return 0.0;
        }
        const double soft_tol = std::max(0.0, release_tol);
        const double hard_stop_tol = std::max(
          soft_tol + 1e-6,
          std::max(0.0, hard_tol));
        if (err <= soft_tol) {
          return 1.0;
        }
        if (err >= hard_stop_tol) {
          return 0.0;
        }
        return std::clamp(
          (hard_stop_tol - err) / std::max(1e-6, hard_stop_tol - soft_tol),
          0.0, 1.0);
      };

      double height_recovery_ratio = 1.0;
      if (scan_min_height_above_base_ > 1e-9 &&
          std::isfinite(actual_surface_height_above_base) &&
          std::isfinite(actual_tcp_height_above_base)) {
        const double surface_release_height = min_scan_height + kAcquireHeightReleaseMargin;
        const double tcp_release_height = min_tcp_height + kAcquireHeightReleaseMargin;
        const double surface_reacquire_height = min_scan_height - kAcquireHeightReacquireSlack;
        const double tcp_reacquire_height = min_tcp_height - kAcquireHeightReacquireSlack;
        const double surface_range =
          std::max(1e-6, surface_release_height - surface_reacquire_height);
        const double tcp_range =
          std::max(1e-6, tcp_release_height - tcp_reacquire_height);
        const double surface_ratio = std::clamp(
          (actual_surface_height_above_base - surface_reacquire_height) / surface_range,
          0.0, 1.0);
        const double tcp_ratio = std::clamp(
          (actual_tcp_height_above_base - tcp_reacquire_height) / tcp_range,
          0.0, 1.0);
        height_recovery_ratio = std::min(surface_ratio, tcp_ratio);
      }

      double cmd_ori_recovery_ratio = 1.0;
      Eigen::Vector3d target_axis_soft = Eigen::Vector3d::Zero();
      if (scan_normal_cmd_valid_) {
        target_axis_soft = scan_align_inward_normal_ ? -scan_normal_cmd_ : scan_normal_cmd_;
      } else {
        target_axis_soft = scan_align_inward_normal_ ? -n_ref : n_ref;
      }
      if (tool_axis_cur.norm() > 1e-9 && target_axis_soft.norm() > 1e-9) {
        target_axis_soft.normalize();
        cmd_ori_soft_err = axis_angle_error(tool_axis_cur, target_axis_soft);
        have_cmd_ori_soft_err = std::isfinite(cmd_ori_soft_err);
        cmd_ori_recovery_ratio = soft_gate_ratio(
          cmd_ori_soft_err,
          std::max(0.0, scan_progress_ori_tol_),
          std::max(0.0, scan_progress_hard_stop_ori_tol_));
      }

      const double band_recovery_ratio = soft_gate_ratio(
        acquire_band_err,
        acquire_band_release_check_tol,
        std::max(hard_stop_band_tol, acquire_band_reacquire_tol));
      const double ref_projection_ratio = soft_gate_ratio(
        scan_surface_ref_projection_err,
        0.01,
        std::max(0.04, acquire_band_release_tol));
      const double surface_recovery_ratio = soft_gate_ratio(
        acquire_surface_err,
        paper_acquire_release_tol,
        std::max(
          paper_reacquire_tol,
          std::max(0.0, scan_progress_hard_stop_surface_dist_tol_)));
      const double scan_soft_quality = std::clamp(
        std::min(
          std::min(
            std::min(band_recovery_ratio, ref_projection_ratio),
            surface_recovery_ratio),
          std::min(cmd_ori_recovery_ratio, height_recovery_ratio)),
        0.0, 1.0);
      constexpr double kMinSoftPathScale = 0.18;
      double soft_path_scale =
        kMinSoftPathScale + (1.0 - kMinSoftPathScale) * scan_soft_quality;
      if (scan_progress_hard_stop_active_) {
        soft_path_scale = std::min(soft_path_scale, 0.35);
      }

      // Shrink the paper-scan orbit before the controller reaches hard-stop.
      // When the band / orientation quality softens, pull the reference toward
      // the safer chart center (and, if needed, toward the acquire-band interior)
      // so the TCP stays moving without dragging the contact patch into reacquire.
      Eigen::Vector2d soft_path_center = scan_surface_seed_;
      if (have_scan_recovery_surface_ref) {
        const double recovery_center_alpha = std::clamp(
          std::max(
            std::max(1.0 - band_recovery_ratio, 1.0 - ref_projection_ratio),
            1.0 - height_recovery_ratio),
          0.0, 1.0);
        soft_path_center =
          scan_surface_seed_ +
          recovery_center_alpha * (scan_recovery_surface_ref - scan_surface_seed_);
      }
      raw_scan_surface_ref =
        soft_path_center + soft_path_scale * (raw_scan_surface_ref - soft_path_center);
    }

    const double soft_ori_hold_tol = std::clamp(
      std::max(
        std::max(0.0, scan_progress_ori_tol_) + 0.02,
        0.6 * std::max(0.0, scan_progress_hard_stop_ori_tol_)),
      0.0,
      std::max(0.0, scan_progress_hard_stop_ori_tol_ - 0.05));
    const bool scan_preemptive_ori_hold_active =
      have_cmd_ori_soft_err &&
      (cmd_ori_soft_err > soft_ori_hold_tol);
    const bool scan_ori_hold_recovery_active =
      scan_progress_hard_stop_active_ &&
      (scan_progress_hard_stop_ori_elapsed_ > 1e-9) &&
      (scan_progress_hard_stop_surface_elapsed_ <= 1e-9);
    const bool scan_recovery_hold_active =
      scan_preemptive_ori_hold_active ||
      scan_ori_hold_recovery_active;
    const bool entering_scan_recovery_hold =
      scan_recovery_hold_active && !scan_recovery_hold_prev_;
    if (entering_scan_recovery_hold) {
      // Orientation recovery works best if we stop where contact is actually
      // realized instead of marching the chart reference toward an interior
      // recovery point. That migration was creating the repeated flip-away /
      // flip-back cycles seen in simulation.
      scan_surface_cmd_ = s_cur;
      scan_surface_cmd_valid_ = true;
      scan_surface_raw_prev_valid_ = false;
      scan_surface_actual_prev_ = s_cur;
      scan_surface_actual_prev_valid_ = true;
      scan_surface_phase_progress_ratio_ = 1.0;
      scan_surface_des_prev_valid_ = false;
    }
    if (scan_recovery_hold_active) {
      raw_scan_surface_ref = s_cur;
      have_raw_scan_surface_ref = true;
      scan_surface_cmd_ratio = 1.0;
    }
    const double scan_ref_linear_speed_limit =
      scan_recovery_hold_active ?
      std::min(
        std::max(0.0, scan_max_linear_speed_),
        std::max(0.0, scan_acquire_max_linear_speed_)) :
      std::max(0.0, scan_max_linear_speed_);
    const double scan_ref_angular_speed_limit =
      scan_recovery_hold_active ?
      std::max(
        0.0,
        (scan_acquire_recovery_max_angular_speed_ > 0.0) ?
          scan_acquire_recovery_max_angular_speed_ :
          std::min(
            std::max(0.0, scan_max_angular_speed_),
            std::max(0.0, scan_acquire_max_angular_speed_))) :
      std::max(0.0, scan_max_angular_speed_);

    Eigen::Vector3d n_ref_cmd_target = n_ref;
    if (paper_surface_acquired_ &&
        !scan_recovery_hold_active &&
        n_base_raw.norm() > 1e-9) {
      const double d_err_for_ori = std::abs(d_surface - d_des_);
      const double d_ori_release_tol =
        std::max(0.02, std::max(0.0, scan_progress_pos_tol_));
      const double d_ori_hard_tol =
        std::max(d_ori_release_tol + 1e-6,
                 std::max(0.0, scan_progress_hard_stop_surface_dist_tol_));
      const double off_surface_ori_alpha = std::clamp(
        (d_err_for_ori - d_ori_release_tol) /
        std::max(1e-6, d_ori_hard_tol - d_ori_release_tol),
        0.0, 1.0);
      if (off_surface_ori_alpha > 1e-6) {
        // When the TCP has drifted well off the shell, following the future scan
        // normal can pry the wrist farther away from the current contact patch.
        // Blend the orientation target back toward the current true surface
        // normal so the radial recovery task can re-seat contact first.
        n_ref_cmd_target =
          slerp_unit_vectors(n_ref, n_base_raw.normalized(), off_surface_ori_alpha);
      }
    }
    Eigen::Vector3d n_ref_cmd =
      update_scan_normal_command(
        n_ref_cmd_target, scan_ref_angular_speed_limit, entering_scan_recovery_hold);
    scan_recovery_hold_prev_ = scan_recovery_hold_active;
    Eigen::Quaterniond q_scan_task = q_d_;
    Eigen::Vector3d target_axis_task = Eigen::Vector3d::Zero();
    const bool have_q_scan_task =
      build_scan_display_quaternion(n_ref_cmd, q_scan_task, target_axis_task);
    Eigen::Vector3d v_scan_task_ang = Eigen::Vector3d::Zero();
    if (have_q_scan_task && dt_slew > 1e-9) {
      Eigen::Quaterniond q_task_prev = q_d_prev;
      if (q_task_prev.dot(q_scan_task) < 0.0) {
        q_task_prev.coeffs() *= -1.0;
      }
      Eigen::Quaterniond q_delta = q_scan_task * q_task_prev.conjugate();
      q_delta.normalize();
      Eigen::AngleAxisd aa_delta(q_delta);
      if (std::isfinite(aa_delta.angle()) && aa_delta.axis().allFinite()) {
        v_scan_task_ang = (aa_delta.angle() / dt_slew) * aa_delta.axis();
      }
    }

    if (have_raw_scan_surface_ref) {
      const double max_ds = scan_ref_linear_speed_limit * dt_slew;
      double nominal_cmd_lead =
        (scan_max_tangential_error_ > 0.0) ?
        scan_max_tangential_error_ : std::max(0.0, scan_progress_pos_tol_);
      if (scan_progress_hard_stop_pos_tol_ > 1e-9) {
        // Let the rate-limited APRP command point stay slightly ahead of the measured
        // contact so the TCP can keep moving, but keep a bounded margin to the hard-stop.
        nominal_cmd_lead = std::max(
          nominal_cmd_lead,
          0.8 * std::max(0.0, scan_progress_hard_stop_pos_tol_));
      }
      nominal_cmd_lead = std::max(0.0, nominal_cmd_lead);

      Eigen::Vector2d s_cmd_base = scan_surface_cmd_valid_ ? scan_surface_cmd_ : s_cur;
      const double prev_cmd_lead_norm =
        scan_surface_cmd_valid_ ? (scan_surface_cmd_ - s_cur).norm() : 0.0;

      Eigen::Vector2d ds_cmd = raw_scan_surface_ref - s_cmd_base;
      const double ds_norm = ds_cmd.norm();
      double ds_metric = ds_norm;
      if (std::isfinite(hemisphere_radius_) && hemisphere_radius_ > 1e-6 && ds_norm > 1e-9) {
        Eigen::Vector2d s_cmd_base_resolved = Eigen::Vector2d::Zero();
        Eigen::Vector3d p_cmd_base = Eigen::Vector3d::Zero();
        Eigen::Vector3d n_cmd_base = Eigen::Vector3d::Zero();
        if (resolve_local_scan_surface_reference(
              s_cmd_base, s_cmd_base_resolved, p_cmd_base, n_cmd_base) &&
            n_cmd_base.norm() > 1e-9 &&
            n_ref_raw.norm() > 1e-9) {
          const double ds_arc =
            hemisphere_radius_ * axis_angle_error(n_cmd_base.normalized(), n_ref_raw.normalized());
          if (std::isfinite(ds_arc)) {
            ds_metric = ds_arc;
          }
        }
      }
      if (max_ds > 1e-9 && ds_metric > max_ds) {
        scan_surface_cmd_saturated = true;
        scan_surface_cmd_ratio = std::clamp(max_ds / ds_metric, 0.0, 1.0);
        ds_cmd *= scan_surface_cmd_ratio;
      } else if (max_ds <= 1e-9 && ds_metric > 1e-9) {
        scan_surface_cmd_saturated = true;
        scan_surface_cmd_ratio = 0.0;
      }
      s_ref = s_cmd_base + ds_cmd;
      double phase_lead_budget = nominal_cmd_lead;
      if ((!paper_surface_acquired_ || scan_recovery_hold_active || scan_progress_hard_stop_active_) &&
          nominal_cmd_lead > 1e-9) {
        // During acquisition / recovery keep the commanded chart point close to the
        // realized contact. In nominal acquired scanning, rely on the existing
        // raw-reference rate limit plus progress/hard-stop gates instead; otherwise
        // this extra shell can pin s_ref above the downhill raw sphere trajectory.
        Eigen::Vector2d cmd_lead = s_ref - s_cur;
        const double cmd_lead_norm = cmd_lead.norm();
        if (cmd_lead_norm > 1e-9) {
          const double hard_cmd_lead =
            std::min(1.5 * nominal_cmd_lead, std::max(nominal_cmd_lead, prev_cmd_lead_norm + max_ds));
          phase_lead_budget = std::max(phase_lead_budget, hard_cmd_lead);
          if (cmd_lead_norm > hard_cmd_lead) {
            cmd_lead *= hard_cmd_lead / cmd_lead_norm;
            s_ref = s_cur + cmd_lead;
          }
        }
      }
      Eigen::Vector2d s_ref_resolved = Eigen::Vector2d::Zero();
      if (!resolve_local_scan_surface_reference(s_ref, s_ref_resolved, p_ref, n_ref)) {
        return fail_paper_surface("resolve_local_scan_surface_reference_cmd");
      }
      s_ref = s_ref_resolved;
      double cmd_step_metric = 0.0;
      double command_progress_ratio = 1.0;
      if (scan_surface_cmd_valid_) {
        cmd_step_metric = local_surface_arc_metric(scan_surface_cmd_, s_ref);
        if (scan_surface_raw_prev_valid_) {
          const double raw_step_metric =
            local_surface_arc_metric(scan_surface_raw_prev_, raw_scan_surface_ref);
          if (raw_step_metric > 1e-9) {
            command_progress_ratio = std::clamp(cmd_step_metric / raw_step_metric, 0.0, 1.0);
          } else {
            // When scan_time_ is currently frozen, the raw APRP reference can be
            // identical to the previous cycle. Do not let that zero raw-step create
            // a self-locking zero progress ratio; let the tracking gates decide
            // whether the phase can move again.
            command_progress_ratio = 1.0;
          }
        } else {
          command_progress_ratio = 1.0;
        }
      }
      double raw_backlog_progress_ratio = 1.0;
      // Keep the debug ratio meaningful for both per-cycle command progress and
      // how far the rate-limited command has fallen behind the raw paper orbit.
      // The previous "horizontal scan stalls" bug came from the actual-progress
      // term self-locking; after fixing that below, we do want raw backlog to
      // slow the phase again once the raw orbit materially outruns s_ref.
      scan_surface_cmd_ratio = command_progress_ratio;
      double actual_progress_ratio = 1.0;
      if (scan_surface_actual_prev_valid_) {
        const double actual_step_metric = local_surface_arc_metric(scan_surface_actual_prev_, s_cur);
        if (cmd_step_metric > 1e-6) {
          // Compare the realized contact motion against this cycle's commanded
          // surface-step, not against the full max-speed step. Once scan_time has
          // already slowed down, dividing by max_ds can drive the ratio toward 0
          // even when the TCP is keeping up with the tiny commanded tangential move,
          // which creates the self-locking "horizontal scan stalls forever" failure.
          actual_progress_ratio = std::clamp(actual_step_metric / cmd_step_metric, 0.0, 1.0);
        } else {
          actual_progress_ratio = 1.0;
        }
      }
      const double command_progress_floor = std::clamp(
        scan_progress_cruise_min_scale_, 0.0, 0.5);
      const bool nominal_scan_phase_mode =
        paper_surface_acquired_ &&
        !scan_recovery_hold_active &&
        !scan_progress_hard_stop_active_;
      double phase_progress_ratio = 1.0;
      if (nominal_scan_phase_mode) {
        // In steady acquired scanning, the current-cycle tangential/surface error is
        // already folded into catchup_quality below. Feeding the realized TCP step
        // back into scan_time_ a second time makes the downhill lissajous branches
        // self-lock: once the TCP falls behind, actual_step_metric becomes tiny,
        // phase_progress_ratio collapses, and the commanded scan point stops moving
        // down the sphere even though the command layer itself is still progressing.
        phase_progress_ratio = command_progress_ratio;
      } else {
        phase_progress_ratio = std::max(
          actual_progress_ratio,
          std::min(command_progress_ratio, command_progress_floor));
      }
      const double cmd_lead_metric = local_surface_arc_metric(s_cur, s_ref);
      phase_lead_budget = std::max(
        phase_lead_budget,
        std::max(2.0 * max_ds, std::max(0.0, scan_progress_pos_tol_)));
      const double hard_stop_hold_budget =
        (scan_progress_hard_stop_pos_tol_ > 1e-9) ?
        0.8 * std::max(0.0, scan_progress_hard_stop_pos_tol_) : phase_lead_budget;
      scan_surface_safe_hold_lead_metric =
        std::max(0.0, std::min(phase_lead_budget, hard_stop_hold_budget));
      if (phase_lead_budget > 1e-9) {
        // Slow the phase only in the slack between the nominal lead budget and the
        // hard-stop envelope. The old full-budget normalization could collapse the
        // phase progress to ~0 while the command point was still inside the
        // configured hard-stop tolerance, which left the scan effectively frozen.
        const double hard_phase_lead_budget = std::max(
          phase_lead_budget, std::max(0.0, scan_progress_hard_stop_pos_tol_));
        double lead_progress_ratio = 1.0;
        if (cmd_lead_metric > phase_lead_budget) {
          if (hard_phase_lead_budget > phase_lead_budget + 1e-6) {
            lead_progress_ratio = std::clamp(
              (hard_phase_lead_budget - cmd_lead_metric) /
              (hard_phase_lead_budget - phase_lead_budget),
              0.0, 1.0);
          } else {
            lead_progress_ratio = std::clamp(
              (phase_lead_budget - cmd_lead_metric) / phase_lead_budget,
              0.0, 1.0);
          }
        }
        phase_progress_ratio = std::max(phase_progress_ratio, lead_progress_ratio);
      }
      const double raw_backlog_metric = local_surface_arc_metric(s_ref, raw_scan_surface_ref);
      if (phase_lead_budget > 1e-9) {
        const double hard_raw_backlog_budget = std::max(
          phase_lead_budget,
          std::max(0.0, scan_progress_hard_stop_pos_tol_));
        if (raw_backlog_metric > phase_lead_budget) {
          if (hard_raw_backlog_budget > phase_lead_budget + 1e-6) {
            raw_backlog_progress_ratio = std::clamp(
              (hard_raw_backlog_budget - raw_backlog_metric) /
              (hard_raw_backlog_budget - phase_lead_budget),
              0.0, 1.0);
          } else {
            raw_backlog_progress_ratio = std::clamp(
              (phase_lead_budget - raw_backlog_metric) / phase_lead_budget,
              0.0, 1.0);
          }
        }
        phase_progress_ratio = std::min(phase_progress_ratio, raw_backlog_progress_ratio);
      }
      scan_surface_cmd_ratio = std::min(scan_surface_cmd_ratio, raw_backlog_progress_ratio);
      // Use a modest commanded-progress floor for the phase gate. With a 1 kHz
      // loop, the realized contact motion can become numerically tiny once the
      // TCP settles into steady scanning, which collapses the raw actual-progress
      // ratio and effectively freezes scan_time_. Keep only a limited floor here
      // so the reference does not jump from "almost frozen" to "full speed" and
      // immediately drag the contact patch back out of band.
      scan_surface_phase_progress_ratio_ = phase_progress_ratio;
      scan_surface_raw_prev_ = raw_scan_surface_ref;
      scan_surface_raw_prev_valid_ = true;
      scan_surface_actual_prev_ = s_cur;
      scan_surface_actual_prev_valid_ = true;
      scan_surface_cmd_ = s_ref;
      scan_surface_cmd_valid_ = true;
      if (dt_slew > 1e-9 && scan_surface_des_prev_valid_) {
        s_dot_ref = (s_ref - scan_surface_des_prev_) / dt_slew;
      } else {
        s_dot_ref.setZero();
      }
    } else {
      scan_surface_cmd_valid_ = false;
    }

    Eigen::Matrix<double, 2, 3> J_sv = Eigen::Matrix<double, 2, 3>::Zero();
    const double fd_step = 1e-4;
    for (int axis_idx = 0; axis_idx < 3; ++axis_idx) {
      Eigen::Vector3d delta = Eigen::Vector3d::Zero();
      delta(axis_idx) = fd_step;
      double d_plus = 0.0;
      double d_minus = 0.0;
      Eigen::Vector3d n_plus = Eigen::Vector3d::Zero();
      Eigen::Vector3d n_minus = Eigen::Vector3d::Zero();
      Eigen::Vector2d s_plus = Eigen::Vector2d::Zero();
      Eigen::Vector2d s_minus = Eigen::Vector2d::Zero();
      if (!query_hemisphere_true_(p + delta, d_plus, n_plus) ||
          !query_hemisphere_true_(p - delta, d_minus, n_minus)) {
        return false;
      }
      Eigen::Vector3d n_plus_projected = n_plus;
      Eigen::Vector3d n_minus_projected = n_minus;
      if (!project_normal_to_scan_band_(n_plus, n_plus_projected) ||
          !project_normal_to_scan_band_(n_minus, n_minus_projected) ||
          !normal_to_local_surface_coords(
            scan_surface_origin_normal_, hemisphere_axis_, hemisphere_radius_, n_plus_projected, s_plus) ||
          !normal_to_local_surface_coords(
            scan_surface_origin_normal_, hemisphere_axis_, hemisphere_radius_, n_minus_projected, s_minus)) {
        return fail_paper_surface("normal_to_local_surface_coords_fd");
      }
      J_sv.col(axis_idx) = (s_plus - s_minus) / (2.0 * fd_step);
    }

    Eigen::Vector3d tool_axis_base = Tf.linear() * scan_tool_axis_runtime_;
    if (tool_axis_base.norm() < 1e-9) {
      return fail_paper_surface("degenerate_tool_axis");
    }
    tool_axis_base.normalize();
    Eigen::Vector3d z_target = scan_align_inward_normal_ ? -n_ref_cmd : n_ref_cmd;
    if (z_target.norm() < 1e-9) {
      return fail_paper_surface("degenerate_desired_scan_normal");
    }
    z_target.normalize();

    Eigen::Quaterniond h = Eigen::Quaterniond::FromTwoVectors(tool_axis_base, z_target);
    h.normalize();
    if (h.w() < 0.0) {
      h.coeffs() *= -1.0;
    }
    const double lambda = h.w();
    const Eigen::Vector3d eps = h.vec();

    Eigen::Matrix3d J_eps_omega;
    J_eps_omega <<  lambda,    eps.z(),  -eps.y(),
                   -eps.z(),   lambda,    eps.x(),
                    eps.y(),  -eps.x(),   lambda;
    J_eps_omega *= 0.5;

    Eigen::Matrix3d J_eps_v = Eigen::Matrix3d::Zero();
    if (enable_curvature_ && hemisphere_radius_ > 1e-6) {
      const double kappa = 1.0 / hemisphere_radius_;
      const double denom = std::max(1e-6, 1.0 - d_surface * kappa);
      const double kappa_tilde = kappa / denom;
      const Eigen::Matrix3d P_tan = Eigen::Matrix3d::Identity() - z_target * z_target.transpose();
      J_eps_v = J_eps_omega * skew(z_target) * (kappa_tilde * P_tan);
    }

    const Eigen::Matrix<double,3,7> J_v = J.topRows<3>();
    const Eigen::Matrix<double,3,7> J_w = J.bottomRows<3>();

    Eigen::Matrix<double,6,7> J_rho = Eigen::Matrix<double,6,7>::Zero();
    J_rho.block<2,7>(0,0) = J_sv * J_v;
    J_rho.row(2) = d_dir.transpose() * J_v;
    J_rho.block<3,7>(3,0) = J_eps_v * J_v + J_eps_omega * J_w;

    Eigen::Matrix<double,6,1> rho = Eigen::Matrix<double,6,1>::Zero();
    rho << s_cur.x(), s_cur.y(), d_surface, eps.x(), eps.y(), eps.z();

    Eigen::Matrix<double,6,1> rho_des = Eigen::Matrix<double,6,1>::Zero();
    rho_des << s_ref.x(), s_ref.y(), d_des_, 0.0, 0.0, 0.0;

    const Eigen::Matrix<double,6,1> rho_dot = J_rho * dq;
    Eigen::Matrix<double,6,1> rho_dot_des = Eigen::Matrix<double,6,1>::Zero();
    rho_dot_des(0) = s_dot_ref.x();
    rho_dot_des(1) = s_dot_ref.y();

    Eigen::Matrix<double,6,1> F_rho = Eigen::Matrix<double,6,1>::Zero();
    F_rho(0) = k_s_.x() * (rho_des(0) - rho(0)) + d_s_.x() * (rho_dot_des(0) - rho_dot(0));
    F_rho(1) = k_s_.y() * (rho_des(1) - rho(1)) + d_s_.y() * (rho_dot_des(1) - rho_dot(1));
    F_rho(2) = k_d_ * (rho_des(2) - rho(2)) + d_d_ * (rho_dot_des(2) - rho_dot(2));
    // The chart task handles surface coordinates and normal-distance. Use the
    // same full scan quaternion as the display/reference path for orientation
    // control below; the old axis-only epsilon task was the main source of the
    // repeated flip-away / flip-back instability.
    F_rho.segment<3>(3).setZero();

    // In paper_surface_scan the p=[s,d,eps] task regulates tangential motion only
    // indirectly through the local chart Jacobian. Add a Cartesian tangential assist
    // toward the resolved surface point so the TCP visibly tracks the scan on the sphere
    // instead of letting the displayed p_d_ move while the real TCP barely does.
    //
    // Important: use the *current* contact tangent plane for Cartesian tangential
    // tracking. When the TCP lags behind on a downhill branch, projecting the
    // desired point into the desired tangent plane injects an artificial inward /
    // downward component; the radial distance task then fights that component and
    // the end-effector appears to "keep the normal but stop descending". Driving
    // the catch-up in the current-contact tangent plane (or, in nominal mode,
    // directly along the sphere geodesic) keeps the assist aligned with the
    // motion that actually advances the TCP around the hemisphere.
    const Eigen::Vector3d n_tan_ref =
      (n_ref.norm() > 1e-9) ? n_ref.normalized() : d_dir.normalized();
    const Eigen::Vector3d n_tan_cur =
      (d_dir.norm() > 1e-9) ? d_dir.normalized() : n_tan_ref;
    const Eigen::Matrix3d P_tan_cur =
      Eigen::Matrix3d::Identity() - n_tan_cur * n_tan_cur.transpose();
    const double tangential_track_radius = std::max(
      1e-6,
      (std::isfinite(hemisphere_radius_ + d_des_) && ((hemisphere_radius_ + d_des_) > 1e-6)) ?
      (hemisphere_radius_ + d_des_) :
      std::max(1e-6, hemisphere_radius_));
    auto geodesic_tangent_error = [&](
      const Eigen::Vector3d & from_normal,
      const Eigen::Vector3d & to_normal,
      double radius) {
      Eigen::Vector3d e_geo = Eigen::Vector3d::Zero();
      if (!(std::isfinite(radius) && radius > 1e-6) ||
          from_normal.norm() < 1e-9 ||
          to_normal.norm() < 1e-9) {
        return e_geo;
      }

      const Eigen::Vector3d n_from = from_normal.normalized();
      const Eigen::Vector3d n_to = to_normal.normalized();
      const double dot_val = std::clamp(n_from.dot(n_to), -1.0, 1.0);
      const double ang = std::acos(dot_val);
      if (!std::isfinite(ang) || ang < 1e-9) {
        return e_geo;
      }

      Eigen::Vector3d tan_dir = n_to - dot_val * n_from;
      const double tan_norm = tan_dir.norm();
      if (tan_norm < 1e-9) {
        return e_geo;
      }
      tan_dir /= tan_norm;
      e_geo = radius * ang * tan_dir;
      return e_geo.allFinite() ? e_geo : Eigen::Vector3d::Zero();
    };
    Eigen::Vector3d p_track_cmd = p_ref;
    const bool limit_cartesian_pull_in =
      !paper_surface_acquired_ ||
      scan_recovery_hold_active ||
      scan_progress_hard_stop_active_;
    if (limit_cartesian_pull_in) {
      Eigen::Vector3d e_track = p_track_cmd - p;
      double e_rad = e_track.dot(n_ref);
      Eigen::Vector3d e_tan = e_track - e_rad * n_ref;

      const double max_pos_err = std::max(0.0, scan_max_position_error_);
      const double max_rad_err = std::max(
        0.0, (scan_max_radial_error_ > 0.0) ? scan_max_radial_error_ : max_pos_err);
      const double max_tan_err = std::max(
        0.0, (scan_max_tangential_error_ > 0.0) ? scan_max_tangential_error_ : max_pos_err);
      const bool use_total_pos_cap =
        (scan_max_radial_error_ <= 0.0) && (scan_max_tangential_error_ <= 0.0);

      if (max_rad_err > 1e-9) {
        e_rad = std::clamp(e_rad, -max_rad_err, max_rad_err);
      }
      if (max_tan_err > 1e-9 && e_tan.norm() > max_tan_err) {
        e_tan *= max_tan_err / e_tan.norm();
      }
      if (use_total_pos_cap && max_pos_err > 1e-9) {
        Eigen::Vector3d e_safe = e_rad * n_ref + e_tan;
        if (e_safe.norm() > max_pos_err) {
          e_safe *= max_pos_err / e_safe.norm();
          e_rad = e_safe.dot(n_ref);
          e_tan = e_safe - e_rad * n_ref;
        }
      }

      p_track_cmd = p + e_rad * n_ref + e_tan;
    }
    Eigen::Vector3d v_track_cmd = Eigen::Vector3d::Zero();
    if (dt_slew > 1e-9) {
      v_track_cmd = (p_track_cmd - p_d_prev) / dt_slew;
    }

    auto pin_scan_reference_to_current_contact = [&](bool preserve_safe_tangential_lead) {
      // When phase progress is paused, re-anchor the local chart reference at the
      // realized contact point. Keeping the old tangentially-lagging s_ref alive
      // during a down-stroke hard-stop was creating the repeated freeze / snap-back
      // limit cycle seen in the debug bags. For pure tangential hard-stop, keep a
      // bounded lead toward the already-resolved command so the TCP can keep
      // walking downhill on the sphere while scan_time stays frozen.
      Eigen::Vector2d s_hold_target = s_cur;
      if (preserve_safe_tangential_lead && scan_surface_safe_hold_lead_metric > 1e-9) {
        const Eigen::Vector2d hold_lead = s_ref - s_cur;
        const double hold_lead_metric = local_surface_arc_metric(s_cur, s_ref);
        if (hold_lead.norm() > 1e-9 && hold_lead_metric > 1e-9) {
          const double hold_alpha =
            std::clamp(scan_surface_safe_hold_lead_metric / hold_lead_metric, 0.0, 1.0);
          s_hold_target =
            s_cur + hold_alpha * hold_lead;
        }
      }
      scan_surface_cmd_ = s_hold_target;
      scan_surface_cmd_valid_ = true;
      scan_surface_raw_prev_valid_ = false;
      scan_surface_actual_prev_ = s_cur;
      scan_surface_actual_prev_valid_ = true;
      scan_surface_phase_progress_ratio_ = 1.0;
      scan_surface_des_prev_valid_ = false;
      raw_scan_surface_ref = s_hold_target;
      have_raw_scan_surface_ref = true;
      scan_surface_cmd_ratio = 1.0;
      s_ref = s_hold_target;
      s_dot_ref.setZero();

      Eigen::Vector2d s_hold = Eigen::Vector2d::Zero();
      Eigen::Vector3d p_hold = Eigen::Vector3d::Zero();
      Eigen::Vector3d n_hold = Eigen::Vector3d::Zero();
      if (resolve_local_scan_surface_reference(s_hold_target, s_hold, p_hold, n_hold)) {
        s_ref = s_hold;
        p_ref = p_hold;
        n_ref = n_hold;
        p_track_cmd = p_hold;
      } else if (n_base.norm() > 1e-9) {
        n_ref = n_base.normalized();
        const double r_hold = hemisphere_radius_ + d_des_;
        if (std::isfinite(r_hold) && r_hold > 1e-4) {
          p_ref = hemisphere_center_ + r_hold * n_ref;
          p_track_cmd = p_ref;
        } else {
          p_track_cmd = p;
        }
      } else {
        p_track_cmd = p;
      }

      if (dt_slew > 1e-9) {
        v_track_cmd = (p_track_cmd - p_d_prev) / dt_slew;
      } else {
        v_track_cmd.setZero();
      }
    };

    double scan_tangential_tracking_scale = 1.0;
    double nominal_scan_cartesian_assist_scale = 0.5;
    double nominal_scan_tangential_catchup_gain = 1.0;
    double nominal_scan_chart_tangential_scale = 1.0;
    bool scan_tangential_hold_recovery = false;
    bool nominal_cartesian_tangential_mode = false;
    if (scan_progress_gate_) {
      constexpr double kGatePosEps = 1e-4;  // [m]
      constexpr double kGateOriEps = 1e-4;  // [rad]
      double s_err = (s_ref - s_cur).norm();
      if (std::isfinite(hemisphere_radius_) &&
          hemisphere_radius_ > 1e-6 &&
          n_base.norm() > 1e-9 &&
          n_ref.norm() > 1e-9) {
        // Gate scan-time using the physical geodesic separation on the hemisphere
        // instead of the raw APRP chart coordinates. A fixed local chart can drift
        // away from the current contact patch over time and artificially inflate the
        // tangential error, which then hard-stops scan_time_ even though the robot is
        // still close to the desired surface track.
        const double s_arc_err =
          hemisphere_radius_ * axis_angle_error(n_base.normalized(), n_ref.normalized());
        if (std::isfinite(s_arc_err)) {
          s_err = s_arc_err;
        }
      }
      const double d_err = std::abs(d_surface - d_des_);
      const double surf_ori_err = 2.0 * std::asin(std::clamp(eps.norm(), 0.0, 1.0));
      const double band_err = axis_angle_error(n_base_raw, n_base);
      const Eigen::Vector3d tangential_surface_err_vec =
        geodesic_tangent_error(n_tan_cur, n_tan_ref, tangential_track_radius);
      const double tangential_surface_err =
        (tangential_surface_err_vec.norm() > 1e-9) ?
        tangential_surface_err_vec.norm() :
        (P_tan_cur * (p_ref - p)).norm();
      const double s_gate_err =
        std::min(s_err, std::isfinite(tangential_surface_err) ? tangential_surface_err : s_err);
      double height_ratio = 1.0;
      if (scan_min_height_above_base_ > 1e-9 &&
          std::isfinite(actual_surface_height_above_base) &&
          std::isfinite(actual_tcp_height_above_base)) {
        const double surface_release_height = min_scan_height + kAcquireHeightReleaseMargin;
        const double tcp_release_height = min_tcp_height + kAcquireHeightReleaseMargin;
        const double surface_reacquire_height = min_scan_height - kAcquireHeightReacquireSlack;
        const double tcp_reacquire_height = min_tcp_height - kAcquireHeightReacquireSlack;
        const double surface_range =
          std::max(1e-6, surface_release_height - surface_reacquire_height);
        const double tcp_range =
          std::max(1e-6, tcp_release_height - tcp_reacquire_height);
        const double surface_height_ratio = std::clamp(
          (actual_surface_height_above_base - surface_reacquire_height) / surface_range,
          0.0, 1.0);
        const double tcp_height_ratio = std::clamp(
          (actual_tcp_height_above_base - tcp_reacquire_height) / tcp_range,
          0.0, 1.0);
        height_ratio = std::min(surface_height_ratio, tcp_height_ratio);
      }
      double cmd_ori_err = std::numeric_limits<double>::infinity();
      const Eigen::Vector3d target_axis_cmd =
        (scan_align_inward_normal_ ? -n_ref_cmd : n_ref_cmd).normalized();
      if (tool_axis_base.norm() > 1e-9 && target_axis_cmd.norm() > 1e-9) {
        cmd_ori_err = axis_angle_error(tool_axis_base, target_axis_cmd);
      }
      debug_scan_surface_err_ = s_err;
      debug_scan_surface_dist_err_ = d_err;
      debug_scan_surface_ori_err_ = surf_ori_err;
      debug_scan_cmd_ori_err_ = cmd_ori_err;
      debug_scan_band_err_ = band_err;
      const double s_tol = std::max(0.01, std::max(0.0, scan_progress_pos_tol_));
      const double d_tol = std::max(0.02, std::max(0.0, scan_progress_pos_tol_));
      const double ori_tol = std::max(0.0, scan_progress_ori_tol_);
      const double hard_s_tol = std::max(0.0, scan_progress_hard_stop_pos_tol_);
      const double hard_d_tol = std::max(0.0, scan_progress_hard_stop_surface_dist_tol_);
      const double hard_ori_tol = std::max(0.0, scan_progress_hard_stop_ori_tol_);
      const double band_tol = std::max(0.0, scan_progress_band_tol_);
      const double hard_band_tol = std::max(
        band_tol + 1e-6,
        std::max(0.0, scan_progress_hard_stop_band_tol_));
      // When the TCP starts lagging tangentially, especially around lissajous
      // turnarounds or downhill branches, nominal tracking needs more than the
      // old "0.5x Cartesian assist and no chart spring" recipe. The recent bags
      // show the command point staying close to the raw scan while the real TCP
      // still sits ~20-25 mm behind it. Keep the assist light near perfect
      // tracking, but ramp it up smoothly as the geodesic surface error
      // approaches the hard-stop envelope. Use the smaller of the geodesic
      // surface error and the commanded Cartesian tangential gap for gating so
      // we do not over-slow scan_time purely because the local chart metric
      // stretches away from the physically commanded TCP point.
      const double assist_soft_tol = std::max(0.0, 0.5 * s_tol);
      const double assist_full_tol =
        std::max(assist_soft_tol + 1e-6, std::max(s_tol, hard_s_tol));
      const double assist_alpha = std::clamp(
        (s_gate_err - assist_soft_tol) / (assist_full_tol - assist_soft_tol),
        0.0, 1.0);
      nominal_scan_cartesian_assist_scale = 0.5 + 1.5 * assist_alpha;
      nominal_scan_tangential_catchup_gain = 1.0 + 1.0 * assist_alpha;
      nominal_scan_chart_tangential_scale = 0.25 + 0.25 * assist_alpha;

      const double d_ratio =
        (d_tol > 1e-9) ? std::clamp((d_tol + kGatePosEps) / std::max(d_err, 1e-9), 0.0, 1.0) : 1.0;
      // Surface-distance loss needs a much stronger recovery response than the
      // old linear d_ratio. Around the lower Lissajous turnaround the TCP can be
      // 25-28 mm off the target shell while tangential scan tracking is still
      // trying to run at ~70% speed, which keeps dragging the robot along the
      // path instead of letting the radial task pull it back onto the surface.
      double surface_dist_recovery_ratio = d_ratio;
      if (hard_d_tol > d_tol + 1e-6) {
        surface_dist_recovery_ratio = std::clamp(
          (hard_d_tol - d_err) / (hard_d_tol - d_tol),
          0.0, 1.0);
      }
      const double band_ratio =
        (band_tol > 1e-9) ? std::clamp((band_tol + kGateOriEps) / std::max(band_err, 1e-9), 0.0, 1.0) : 1.0;
      const double hard_surf_ori_ratio =
        (hard_ori_tol > 1e-9) ?
        std::clamp((hard_ori_tol + kGateOriEps) / std::max(surf_ori_err, 1e-9), 0.0, 1.0) : 1.0;
      const double hard_cmd_ori_ratio =
        (hard_ori_tol > 1e-9) ?
        std::clamp((hard_ori_tol + kGateOriEps) / std::max(cmd_ori_err, 1e-9), 0.0, 1.0) : 1.0;
      const double hard_ori_hold_ratio =
        std::clamp(std::min(hard_surf_ori_ratio, hard_cmd_ori_ratio), 0.0, 1.0);
      const double ori_hold_tracking_scale = std::clamp(
        0.25 * hard_ori_hold_ratio *
        std::min(std::min(surface_dist_recovery_ratio, band_ratio), height_ratio),
        0.0, 1.0);
      const bool hard_band_bad = band_err > (hard_band_tol + kGateOriEps);
      const bool hard_ori_bad_base =
        ((hard_ori_tol > 1e-9) && (surf_ori_err > (hard_ori_tol + kGateOriEps))) ||
        ((hard_ori_tol > 1e-9) && (cmd_ori_err > (hard_ori_tol + kGateOriEps)));
      const bool hard_band_recoverable_in_scan =
        hard_band_bad && (hard_ori_bad_base || scan_ori_hold_recovery_active);
      const bool hard_surface_pos_bad =
        (hard_s_tol > 1e-9) && (s_gate_err > (hard_s_tol + kGatePosEps));
      const double hard_d_release_tol =
        (hard_d_tol > 1e-9) ?
        std::clamp(
          std::min(d_tol, 0.75 * hard_d_tol),
          0.0,
          std::max(0.0, hard_d_tol - 2.0 * kGatePosEps)) :
        0.0;
      const bool hard_surface_dist_bad =
        (hard_d_tol > 1e-9) &&
        (d_err >
         (((scan_progress_hard_stop_active_ && (d_err > hard_d_release_tol)) ?
           hard_d_release_tol : hard_d_tol) + kGatePosEps));
      const bool hard_surface_band_bad =
        hard_band_bad && !hard_band_recoverable_in_scan;
      const bool hard_surface_bad =
        hard_surface_pos_bad || hard_surface_dist_bad || hard_surface_band_bad;
      const bool tangential_surface_hard_stop =
        hard_surface_pos_bad && !hard_surface_dist_bad && !hard_surface_band_bad;
      const bool hard_ori_bad =
        hard_ori_bad_base || hard_band_recoverable_in_scan;
      const bool soft_ori_hold_bad =
        !hard_surface_bad &&
        !hard_ori_bad &&
        ((surf_ori_err > (soft_ori_hold_tol + kGateOriEps)) ||
         (cmd_ori_err > (soft_ori_hold_tol + kGateOriEps)));
      const bool entering_tangential_hard_stop =
        tangential_surface_hard_stop && !scan_progress_hard_stop_active_;
      const bool entering_ori_hard_stop =
        hard_ori_bad && !hard_surface_bad && !scan_progress_hard_stop_active_;
      if (hard_surface_bad && dt_slew > 1e-9) {
        scan_progress_hard_stop_surface_elapsed_ += dt_slew;
      } else {
        scan_progress_hard_stop_surface_elapsed_ = 0.0;
      }
      if (hard_ori_bad && dt_slew > 1e-9) {
        scan_progress_hard_stop_ori_elapsed_ += dt_slew;
      } else {
        scan_progress_hard_stop_ori_elapsed_ = 0.0;
      }
      const bool hard_surface_reacquire =
        hard_surface_bad &&
        (scan_progress_hard_stop_surface_reacquire_time_ <= 1e-9 ||
         scan_progress_hard_stop_surface_elapsed_ >= scan_progress_hard_stop_surface_reacquire_time_);
      if (hard_surface_reacquire) {
        // Do not keep the paper scan frozen at a stale reference while the real TCP
        // continues to slide away on the surface. If tangential, near-surface, or
        // band tracking stays in hard-stop long enough, drop back to the pull-in
        // phase and let acquisition recover the contact patch.
        RCLCPP_WARN(
          get_node()->get_logger(),
          "paper_scan hard-stop reacquire: surf_err=%.4f/%.4f dist_err=%.4f/%.4f "
          "band_err=%.4f/%.4f surf_ori=%.4f cmd_ori=%.4f hard_ori_tol=%.4f",
          s_err,
          hard_s_tol,
          d_err,
          hard_d_tol,
          band_err,
          hard_band_tol,
          surf_ori_err,
          cmd_ori_err,
          hard_ori_tol);
        paper_surface_acquired_ = false;
        scan_surface_cmd_valid_ = false;
        scan_surface_raw_prev_valid_ = false;
        scan_surface_actual_prev_valid_ = false;
        scan_surface_phase_progress_ratio_ = 1.0;
        scan_surface_des_prev_valid_ = false;
        scan_progress_hard_stop_active_ = false;
        scan_progress_hard_stop_surface_elapsed_ = 0.0;
        scan_progress_hard_stop_ori_elapsed_ = 0.0;
        scan_advance_scale = 0.0;
        scan_tangential_tracking_scale = 0.0;
        scan_tangential_hold_recovery = true;
        debug_scan_reacquire_reason_ = 4.0;
      } else if (hard_surface_bad) {
        scan_progress_hard_stop_active_ = true;
        scan_advance_scale = 0.0;
        if (tangential_surface_hard_stop) {
          if (entering_tangential_hard_stop) {
            pin_scan_reference_to_current_contact(true);
          }
          scan_tangential_tracking_scale = 1.0;
          scan_tangential_hold_recovery = false;
        } else {
          // For distance/band hard-stops, do not leave the hidden paper-scan
          // command marching ahead behind the Cartesian pull-in clamp. If we do,
          // clearing the hard-stop lets p_d_ snap back to that stale scan point,
          // which is exactly the "bottom-turn upward" jump seen in the debug bags.
          pin_scan_reference_to_current_contact(false);
          scan_tangential_tracking_scale = 0.0;
          scan_tangential_hold_recovery = true;
        }
      } else if (hard_ori_bad) {
        if (entering_ori_hard_stop) {
          pin_scan_reference_to_current_contact(false);
        }
        scan_progress_hard_stop_active_ = true;
        scan_advance_scale = 0.0;
        // Freeze phase progress while the tool axis catches up. Keep only a weak
        // Cartesian tangential pin so the contact patch stays nearby without
        // continuing to drag the wrist deeper into the unstable roll/axis mode.
        scan_tangential_hold_recovery = true;
        scan_tangential_tracking_scale = ori_hold_tracking_scale;
      } else if (soft_ori_hold_bad) {
        scan_progress_hard_stop_surface_elapsed_ = 0.0;
        scan_progress_hard_stop_ori_elapsed_ = 0.0;
        scan_progress_hard_stop_active_ = false;
        scan_advance_scale = 0.0;
        scan_tangential_tracking_scale = ori_hold_tracking_scale;
        pin_scan_reference_to_current_contact(false);
      } else {
        scan_progress_hard_stop_surface_elapsed_ = 0.0;
        scan_progress_hard_stop_ori_elapsed_ = 0.0;
        scan_progress_hard_stop_active_ = false;
        const double s_ratio =
          (s_tol > 1e-9) ?
          std::clamp((s_tol + kGatePosEps) / std::max(s_gate_err, 1e-9), 0.0, 1.0) : 1.0;
        const double surf_ori_ratio =
          (ori_tol > 1e-9) ?
          std::clamp((ori_tol + kGateOriEps) / std::max(surf_ori_err, 1e-9), 0.0, 1.0) : 1.0;
        const double cmd_ori_ratio =
          (ori_tol > 1e-9) ?
          std::clamp((ori_tol + kGateOriEps) / std::max(cmd_ori_err, 1e-9), 0.0, 1.0) : 1.0;
        const double phase_progress_ratio =
          std::clamp(scan_surface_phase_progress_ratio_, 0.0, 1.0);
        const double tracking_quality = std::clamp(
          std::min(
            std::min(s_ratio, surface_dist_recovery_ratio),
            std::min(std::min(surf_ori_ratio, cmd_ori_ratio), std::min(band_ratio, height_ratio))),
          0.0, 1.0);
        // Tangential lag dominates the down-stroke failures in the recent bags,
        // but the previous quartic law (roughly s_ratio^4 when tangential error
        // dominates) was overreacting: with s_err only about 2x the soft limit,
        // scan_time nearly froze and the TCP lost the downhill scan motion. Keep
        // the phase sensitive to tangential lag, but back the penalty off to an
        // approximate s_ratio^2 profile so the command can still make progress.
        const double catchup_quality = std::clamp(
          tracking_quality * s_ratio,
          0.0, 1.0);
        // When the normal, band, or surface-distance tracking starts to soften,
        // back off tangential motion before the controller hits hard-stop. This
        // gives the radial/axis-alignment parts of the task room to recover
        // instead of continuing to drag the TCP along the surface until it
        // slides into reacquire.
        scan_tangential_tracking_scale = std::clamp(
          std::min(
            std::min(surf_ori_ratio, cmd_ori_ratio),
            std::min(surface_dist_recovery_ratio, std::min(band_ratio, height_ratio))),
          0.0, 1.0);
        nominal_cartesian_tangential_mode =
          paper_surface_acquired_ &&
          !scan_recovery_hold_active &&
          !scan_progress_hard_stop_active_;
        scan_advance_scale = std::clamp(
          std::min(catchup_quality, phase_progress_ratio),
          0.0, 1.0);
        if (scan_progress_cruise_min_scale_ > 1e-9) {
          const double cruise_quality_threshold = std::clamp(
            scan_progress_cruise_quality_threshold_, 0.0, 0.999999);
          const double cruise_quality_alpha = std::clamp(
            (catchup_quality - cruise_quality_threshold) /
            std::max(1e-6, 1.0 - cruise_quality_threshold),
            0.0, 1.0);
          const double cruise_floor = std::clamp(
            scan_progress_cruise_min_scale_ * cruise_quality_alpha,
            0.0, catchup_quality);
          scan_advance_scale = std::max(scan_advance_scale, cruise_floor);
        }
      }
      scan_advance = (scan_advance_scale > 1e-6);
    } else if (scan_surface_cmd_saturated) {
      scan_progress_hard_stop_active_ = false;
      scan_advance_scale = std::clamp(scan_surface_phase_progress_ratio_, 0.0, 1.0);
      scan_advance = (scan_advance_scale > 1e-6);
    }

    if (scan_tangential_hold_recovery) {
      F_rho(0) = 0.0;
      F_rho(1) = 0.0;
      rho_dot_des(0) = 0.0;
      rho_dot_des(1) = 0.0;
    } else {
      const double tangential_task_scale =
        nominal_cartesian_tangential_mode ?
        (nominal_scan_chart_tangential_scale *
         scan_tangential_tracking_scale *
         nominal_scan_tangential_catchup_gain) :
        (scan_tangential_tracking_scale * nominal_scan_tangential_catchup_gain);
      F_rho(0) *= tangential_task_scale;
      F_rho(1) *= tangential_task_scale;
    }

    Eigen::Matrix<double,7,1> tau = J_rho.transpose() * F_rho;
    Eigen::Vector3d e_tan_cart = P_tan_cur * (p_track_cmd - p);
    if (nominal_cartesian_tangential_mode) {
      const Eigen::Vector3d e_tan_geo =
        geodesic_tangent_error(n_tan_cur, n_tan_ref, tangential_track_radius);
      if (e_tan_geo.norm() > 1e-9) {
        e_tan_cart = e_tan_geo;
      }
    }
    const Eigen::Vector3d de_tan_cart = P_tan_cur * (v_track_cmd - v_lin);
    Eigen::Vector3d F_tan_cart =
      K_cart_.topLeftCorner<3,3>() * e_tan_cart +
      D_cart_.topLeftCorner<3,3>() * de_tan_cart;
    // In nominal acquired scanning, the chart-space reference is already rate-limited.
    // Re-centering a second Cartesian pull-in shell around the live TCP caused the
    // command point to stick on that shell, then snap around it near Lissajous
    // turnarounds. Keep the protective shell only for recovery / hard-stop states.
    // Also keep direct Cartesian tangential tracking as the main nominal driver,
    // but do not completely disable the chart spring anymore: the latest downhill
    // bags show that Cartesian-only tangential control can still leave the TCP
    // materially behind the commanded sphere track.
    const double cartesian_tangential_assist_scale =
      limit_cartesian_pull_in ? scan_tangential_tracking_scale :
      ((nominal_cartesian_tangential_mode ?
        (0.5 + nominal_scan_cartesian_assist_scale) :
        nominal_scan_cartesian_assist_scale) *
       nominal_scan_tangential_catchup_gain *
       scan_tangential_tracking_scale);
    F_tan_cart *= cartesian_tangential_assist_scale;
    tau += J_v.transpose() * F_tan_cart;
    if (have_q_scan_task) {
      Eigen::Quaterniond q_ee_fix = q_ee;
      if (q_scan_task.dot(q_ee_fix) < 0.0) {
        q_ee_fix.coeffs() *= -1.0;
      }
      Eigen::Quaterniond q_err = q_scan_task * q_ee_fix.conjugate();
      q_err.normalize();
      Eigen::AngleAxisd aa_err(q_err);
      if (std::isfinite(aa_err.angle()) && aa_err.axis().allFinite()) {
        const Eigen::Vector3d e_ori = aa_err.angle() * aa_err.axis();
        const Eigen::Vector3d de_ori = v_scan_task_ang - v_ang;
        const Eigen::Vector3d F_ori_cart =
          K_cart_.bottomRightCorner<3,3>() * e_ori +
          D_cart_.bottomRightCorner<3,3>() * de_ori;
        tau += J_w.transpose() * F_ori_cart;
      }
    }
    // Keep publishing roll diagnostics, but let the full scan quaternion above
    // handle the missing roll DOF directly. The previous dedicated roll torque
    // is no longer needed once the real control law uses the same unique
    // orientation reference as q_d_.
    if (paper_surface_acquired_ && scan_hold_current_roll_ && scan_roll_hint_valid_) {
      Eigen::Vector3d local_roll_secondary = Eigen::Vector3d::Zero();
      Eigen::Matrix3d R_roll_des = Eigen::Matrix3d::Identity();
      if (resolve_secondary_axis_local(scan_tool_axis_runtime_, local_roll_secondary) &&
          build_frame_from_primary(z_target, scan_roll_hint_base_, R_roll_des)) {
        Eigen::Vector3d roll_axis_cur = Tf.linear() * local_roll_secondary;
        Eigen::Vector3d roll_axis_des = R_roll_des.col(1);
        roll_axis_cur -= roll_axis_cur.dot(z_target) * z_target;
        roll_axis_des -= roll_axis_des.dot(z_target) * z_target;
        if (roll_axis_cur.norm() > 1e-9 && roll_axis_des.norm() > 1e-9) {
          roll_axis_cur.normalize();
          roll_axis_des.normalize();
          const double roll_err = std::atan2(
            z_target.dot(roll_axis_cur.cross(roll_axis_des)),
            std::clamp(roll_axis_cur.dot(roll_axis_des), -1.0, 1.0));
          const double roll_rate = z_target.dot(v_ang);
          debug_scan_roll_err_ = roll_err;
          debug_scan_roll_rate_ = roll_rate;
        }
      }
    }

    // A posture target in the same nullspace would still pump this free roll DOF,
    // so keep the old joint-space nullspace spring disabled for paper_surface_scan.
    tau += model_feedforward;
    tau -= D_joint_.cwiseProduct(dq);
    clamp_tau(tau);

    have_scan_ref = true;
    p_scan_ref = p_ref_raw;
    n_scan_ref = n_ref_raw;
    have_cmd_surface_ref = true;
    n_cmd_ref = n_ref_cmd;
    p_d_ = p_track_cmd;

    Eigen::Quaterniond q_ref = q_d_;
    if (have_q_scan_task) {
      q_d_ = q_scan_task;
      q_ref = q_scan_task;
    }

    if (dt_slew > 1e-9) {
      v_des_cart.head<3>() = (p_d_ - p_d_prev) / dt_slew;
      Eigen::Quaterniond q_d_prev_fix = q_d_prev;
      if (q_d_prev_fix.dot(q_d_) < 0.0) {
        q_d_prev_fix.coeffs() *= -1.0;
      }
      Eigen::Quaterniond q_delta = q_d_ * q_d_prev_fix.conjugate();
      q_delta.normalize();
      Eigen::AngleAxisd aa_delta(q_delta);
      if (std::isfinite(aa_delta.angle()) && aa_delta.axis().allFinite()) {
        v_des_cart.tail<3>() = (aa_delta.angle() / dt_slew) * aa_delta.axis();
      }
    }

    scan_surface_des_prev_ = s_ref;
    scan_surface_des_prev_valid_ = true;
    tau_out = tau;
    return true;
  };

  if (switch_hold_active) {
    tau_cmd = compute_tau_cartesian_hold();
  } else if (control_mode_ == "paper_p") {
    used_paper = try_compute_tau_paper(tau_cmd);
    if (!used_paper) {
      tau_cmd = compute_tau_cartesian_hold();
    }
  } else if (control_mode_ == "paper_surface_scan") {
    used_paper = try_compute_tau_paper_surface(tau_cmd);
    if (!used_paper) {
      scan_surface_cmd_valid_ = false;
      scan_surface_raw_prev_valid_ = false;
      scan_surface_actual_prev_valid_ = false;
      scan_surface_phase_progress_ratio_ = 1.0;
      scan_surface_des_prev_valid_ = false;
      tau_cmd = compute_tau_cartesian_hold();
      scan_advance = false;
      scan_advance_scale = 0.0;
    }
  } else if (control_mode_ == "cartesian_surface") {
    // Old-style surface following, but IMPORTANT: gate updates to avoid ruining HOLD when k_outer_pos_=0
    double d = 0.0;
    Eigen::Vector3d n_base = Eigen::Vector3d::UnitZ();
    if (query_hemisphere_(p, d, n_base) && (std::abs(d) <= d_max_)) {
      if (k_outer_pos_ > 1e-9) {
        const double e_d = d - d_des_;
        p_d_ = p - k_outer_pos_ * e_d * n_base;

        const Eigen::Vector3d z_tool_local(0.0, 0.0, -1.0);
        Eigen::Vector3d z_tool_base = Tf.linear() * z_tool_local;
        if (z_tool_base.norm() > 1e-9) {
          z_tool_base.normalize();
          const Eigen::Vector3d z_target = -n_base;
          const Eigen::Quaterniond q_align = Eigen::Quaterniond::FromTwoVectors(z_tool_base, z_target);
          q_d_ = q_align * q_ee;
          q_d_.normalize();
        }
      }
    }
    tau_cmd = compute_tau_cartesian_hold();
  } else if (control_mode_ == "hemi_scan_cartesian") {
    Eigen::Vector3d p_scan = p_d_;
    Eigen::Vector3d n_scan = Eigen::Vector3d::UnitZ();
    if (compute_hemi_scan_pose(scan_time_, p_scan, n_scan)) {
      have_scan_ref = true;
      p_scan_ref = p_scan;
      n_scan_ref = n_scan;
      // Align configurable tool axis to the scan normal, but keep the commanded
      // target pose close to what the robot can actually track.
      Eigen::Vector3d tool_axis_base = Tf.linear() * scan_tool_axis_runtime_;
      Eigen::Vector3d target_axis_raw = scan_align_inward_normal_ ? -n_scan : n_scan;
      if (tool_axis_base.norm() > 1e-9 && target_axis_raw.norm() > 1e-9) {
        tool_axis_base.normalize();
        target_axis_raw.normalize();

        // Slew-limit desired position to avoid hard jumps while still letting the
        // scan target pull the robot in from a farther initial offset.
        const double max_dp = std::max(0.0, scan_max_linear_speed_) * dt_slew;
        Eigen::Vector3d dp = p_scan - p_d_;
        if (max_dp > 1e-9 && dp.norm() > max_dp) {
          dp = (max_dp / dp.norm()) * dp;
        }
        p_d_ += dp;

        // Limit pull-in anisotropically: keep radial error conservative while allowing
        // a larger tangential lead so scan motion can move around the surface.
        Eigen::Vector3d e_track = p_d_ - p;
        Eigen::Vector3d surface_normal = p - hemisphere_center_;
        if (surface_normal.norm() < 1e-9) {
          surface_normal = n_scan;
        }
        if (surface_normal.norm() < 1e-9) {
          surface_normal = Eigen::Vector3d::UnitZ();
        }
        surface_normal.normalize();

        const double max_pos_err = std::max(0.0, scan_max_position_error_);
        const double max_rad_err = std::max(
          0.0, (scan_max_radial_error_ > 0.0) ? scan_max_radial_error_ : max_pos_err);
        const double max_tan_err = std::max(
          0.0, (scan_max_tangential_error_ > 0.0) ? scan_max_tangential_error_ : max_pos_err);
        const bool use_total_pos_cap =
          (scan_max_radial_error_ <= 0.0) && (scan_max_tangential_error_ <= 0.0);
        const double effective_pos_cap =
          use_total_pos_cap ? max_pos_err : std::hypot(max_rad_err, max_tan_err);

        double e_rad = e_track.dot(surface_normal);
        Eigen::Vector3d e_tan = e_track - e_rad * surface_normal;
        if (max_rad_err > 1e-9) {
          e_rad = std::clamp(e_rad, -max_rad_err, max_rad_err);
        }
        if (max_tan_err > 1e-9 && e_tan.norm() > max_tan_err) {
          e_tan *= max_tan_err / e_tan.norm();
        }

        if (use_total_pos_cap && max_pos_err > 1e-9) {
          Eigen::Vector3d e_safe = e_rad * surface_normal + e_tan;
          if (e_safe.norm() > max_pos_err) {
            e_safe *= max_pos_err / e_safe.norm();
            e_rad = e_safe.dot(surface_normal);
            e_tan = e_safe - e_rad * surface_normal;
          }
        }

        p_d_ = p + e_rad * surface_normal + e_tan;
        e_track = p_d_ - p;

        // Build the commanded normal from the point we actually command, not the
        // raw scan pose, so the desired axis does not get ahead of the desired point.
        Eigen::Vector3d n_cmd = p_d_ - hemisphere_center_;
        if (n_cmd.norm() > 1e-9) {
          n_cmd.normalize();
        } else {
          n_cmd = n_scan;
        }
        have_cmd_surface_ref = true;
        n_cmd_ref = n_cmd;
        Eigen::Vector3d target_axis_cmd = scan_align_inward_normal_ ? -n_cmd : n_cmd;
        if (target_axis_cmd.norm() < 1e-9) {
          target_axis_cmd = target_axis_raw;
        }
        target_axis_cmd.normalize();

        // Resolve the remaining roll DOF with a secondary hint. Using the current
        // tool-frame roll hint frozen at activate/acquire avoids chasing incidental
        // wrist twists around an axis the scan task does not otherwise constrain.
        const Eigen::Vector3d local_primary = scan_tool_axis_runtime_;
        Eigen::Vector3d local_secondary = Eigen::Vector3d::Zero();
        Eigen::Matrix3d R_local = Eigen::Matrix3d::Identity();
        Eigen::Matrix3d R_base_des = Eigen::Matrix3d::Identity();
        bool have_q_scan = false;
        Eigen::Quaterniond q_scan = q_d_;

        if (resolve_secondary_axis_local(local_primary, local_secondary) &&
            build_frame_from_primary(local_primary, local_secondary, R_local)) {
          Eigen::Vector3d base_hint_prev_des = q_d_.toRotationMatrix() * R_local.col(1);
          if (base_hint_prev_des.norm() > 1e-9) {
            base_hint_prev_des.normalize();
          }
          const bool have_locked_roll_hint =
            scan_hold_current_roll_ && scan_roll_hint_valid_ &&
            build_frame_from_primary(target_axis_cmd, scan_roll_hint_base_, R_base_des);
          if (have_locked_roll_hint ||
              build_frame_from_primary(target_axis_cmd, base_hint_prev_des, R_base_des) ||
              build_frame_from_primary(target_axis_cmd, hemisphere_axis_, R_base_des)) {
            const Eigen::Matrix3d R_scan = R_base_des * R_local.transpose();
            q_scan = Eigen::Quaterniond(R_scan);
            q_scan.normalize();
            have_q_scan = true;
          }
        }
        if (!have_q_scan) {
          // Last-resort fallback for robustness.
          q_scan = Eigen::Quaterniond::FromTwoVectors(tool_axis_base, target_axis_cmd) * q_ee;
          q_scan.normalize();
        }

        // Slew-limit desired orientation with slerp.
        double max_da = std::max(0.0, scan_max_angular_speed_) * dt_slew;
        if (q_d_.dot(q_scan) < 0.0) {
          q_scan.coeffs() *= -1.0;
        }
        if (max_da > 1e-9) {
          Eigen::Quaterniond q_err = q_d_.conjugate() * q_scan;
          q_err.normalize();
          Eigen::AngleAxisd aa(q_err);
          const double ang = std::abs(aa.angle());
          if (std::isfinite(ang) && ang > max_da) {
            q_d_ = q_d_.slerp(max_da / ang, q_scan);
            q_d_.normalize();
          } else {
            q_d_ = q_scan;
          }
        } else {
          q_d_ = q_scan;
        }

        const double max_ori_err = std::max(0.0, scan_max_orientation_error_);
        if (max_ori_err > 1e-9) {
          Eigen::Quaterniond q_d_fix = q_d_;
          if (q_ee.dot(q_d_fix) < 0.0) {
            q_d_fix.coeffs() *= -1.0;
          }
          Eigen::Quaterniond q_err_track = q_ee.conjugate() * q_d_fix;
          q_err_track.normalize();
          Eigen::AngleAxisd aa_track(q_err_track);
          const double ang_track = std::abs(aa_track.angle());
          if (std::isfinite(ang_track) && ang_track > max_ori_err) {
            q_d_ = q_ee.slerp(max_ori_err / ang_track, q_d_fix);
            q_d_.normalize();
          }
        }

        Eigen::Vector3d tool_axis_des_cmd = q_d_.toRotationMatrix() * scan_tool_axis_runtime_;
        if (tool_axis_des_cmd.norm() > 1e-9) {
          tool_axis_des_cmd.normalize();
        }

        if (dt_slew > 1e-9) {
          v_des_cart.head<3>() = (p_d_ - p_d_prev) / dt_slew;

          Eigen::Quaterniond q_d_prev_fix = q_d_prev;
          if (q_d_prev_fix.dot(q_d_) < 0.0) {
            q_d_prev_fix.coeffs() *= -1.0;
          }
          Eigen::Quaterniond q_delta = q_d_ * q_d_prev_fix.conjugate();
          q_delta.normalize();
          Eigen::AngleAxisd aa_delta(q_delta);
          if (std::isfinite(aa_delta.angle()) && aa_delta.axis().allFinite()) {
            v_des_cart.tail<3>() = (aa_delta.angle() / dt_slew) * aa_delta.axis();
          }
        }

        if (scan_progress_gate_) {
          double pos_tol = std::max(0.0, scan_progress_pos_tol_);
          const double ori_tol = std::max(0.0, scan_progress_ori_tol_);
          const double hard_stop_pos_tol = std::max(0.0, scan_progress_hard_stop_pos_tol_);
          const double hard_stop_ori_tol = std::max(0.0, scan_progress_hard_stop_ori_tol_);
          const double hard_stop_surface_dist_tol =
            std::max(0.0, scan_progress_hard_stop_surface_dist_tol_);
          if (effective_pos_cap > 1e-9) {
            // Start slowing the scan before the commanded point hits its own tracking cap.
            const double kProgressCapMargin = 0.75;
            const double pos_tol_from_cap = kProgressCapMargin * effective_pos_cap;
            pos_tol = (pos_tol > 1e-9) ? std::min(pos_tol, pos_tol_from_cap) : pos_tol_from_cap;
          }
          // Prevent scan-time deadlock caused by tiny numeric jitter exactly at threshold.
          constexpr double kGatePosEps = 1e-4;  // [m]
          constexpr double kGateOriEps = 1e-4;  // [rad]
          const double actual_pos_err = e_track.norm();
          const double actual_ori_err = axis_angle_error(tool_axis_base, tool_axis_des_cmd);
          const double command_pos_gap = (p_scan - p_d_).norm();
          double hard_stop_surface_dist = std::numeric_limits<double>::infinity();
          Eigen::Vector3d hard_stop_surface_normal = Eigen::Vector3d::Zero();
          const bool have_hard_stop_surface = query_hemisphere_true_(
            p, hard_stop_surface_dist, hard_stop_surface_normal);
          const bool hard_stop_allowed =
            (hard_stop_surface_dist_tol <= 1e-9) ||
            (have_hard_stop_surface &&
             std::abs(hard_stop_surface_dist) <= (hard_stop_surface_dist_tol + kGatePosEps));
          const bool hard_pos_bad =
            hard_stop_allowed &&
            (hard_stop_pos_tol > 1e-9) &&
            ((actual_pos_err > (hard_stop_pos_tol + kGatePosEps)) ||
             (command_pos_gap > (hard_stop_pos_tol + kGatePosEps)));
          const bool hard_ori_bad =
            hard_stop_allowed &&
            (hard_stop_ori_tol > 1e-9) &&
            (actual_ori_err > (hard_stop_ori_tol + kGateOriEps));
          const bool pos_bad =
            (pos_tol > 1e-9) &&
            ((actual_pos_err > (pos_tol + kGatePosEps)) || (command_pos_gap > (pos_tol + kGatePosEps)));
          const bool ori_bad =
            (ori_tol > 1e-9) &&
            (actual_ori_err > (ori_tol + kGateOriEps));
          if (hard_pos_bad || hard_ori_bad) {
            if (!scan_progress_hard_stop_active_) {
              double theta_hold = scan_theta_seed_;
              double phi_hold = scan_phi_seed_;
              Eigen::Vector3d hold_normal =
                (n_cmd.norm() > 1e-9) ? n_cmd.normalized() : n_scan.normalized();
              Eigen::Vector3d hold_normal_projected = hold_normal;
              if (hold_normal.norm() > 1e-9 &&
                  project_normal_to_scan_band_(
                    hold_normal, hold_normal_projected, &theta_hold, &phi_hold)) {
                hold_normal = hold_normal_projected;
                if (lock_lissajous_mapping) {
                  Eigen::Vector2d s_hold = Eigen::Vector2d::Zero();
                  if (scan_surface_origin_normal_valid_ &&
                      normal_to_local_surface_coords(
                        scan_surface_origin_normal_, hemisphere_axis_, hemisphere_radius_, hold_normal, s_hold)) {
                    scan_surface_cmd_ = s_hold;
                    scan_surface_cmd_valid_ = true;
                    scan_surface_actual_prev_ = s_hold;
                    scan_surface_actual_prev_valid_ = true;
                  } else {
                    scan_surface_cmd_.setZero();
                    scan_surface_cmd_valid_ = false;
                    scan_surface_actual_prev_.setZero();
                    scan_surface_actual_prev_valid_ = false;
                  }
                  scan_surface_raw_prev_.setZero();
                  scan_surface_raw_prev_valid_ = false;
                  scan_surface_des_prev_valid_ = false;
                } else {
                  scan_theta_start_ = theta_hold;
                  scan_phi_start_ = phi_hold;
                  scan_theta_seed_ = theta_hold;
                  scan_phi_seed_ = phi_hold;
                  scan_seed_blend_elapsed_ = 0.0;
                  if (scan_track_generator_ == "paper_surface_lissajous") {
                    scan_surface_origin_normal_ = hold_normal;
                    scan_surface_origin_normal_valid_ = true;
                    scan_surface_start_.setZero();
                    scan_surface_seed_ = scan_surface_lissajous_offset_;
                    scan_surface_cmd_.setZero();
                    scan_surface_cmd_valid_ = false;
                    scan_surface_des_prev_valid_ = false;
                  } else {
                    Eigen::Vector2d s_hold = Eigen::Vector2d::Zero();
                    if (normal_to_hemi_surface_coords_(hold_normal, s_hold)) {
                      scan_surface_start_ = s_hold;
                      scan_surface_seed_ = s_hold + scan_surface_lissajous_offset_;
                      scan_surface_cmd_ = scan_surface_start_;
                      scan_surface_cmd_valid_ = true;
                      scan_surface_des_prev_valid_ = false;
                    }
                  }
                  if (scan_reset_time_on_activate_) {
                    scan_time_ = 0.0;
                  }
                }
                p_scan = hemisphere_center_ + hemisphere_radius_ * hold_normal;
                n_scan = hold_normal;
                p_scan_ref = p_scan;
                n_scan_ref = n_scan;
              }
            }
            scan_progress_hard_stop_active_ = true;
            // When the scan reference gets too far ahead, fully freeze scan-time until the
            // commanded pose catches back up instead of letting the raw orbit drift away.
            scan_advance_scale = 0.0;
          } else if (pos_bad || ori_bad) {
            scan_progress_hard_stop_active_ = false;
            // Pause scan-time whenever the robot or the commanded pose falls behind the raw scan.
            const double actual_pos_ratio =
              (pos_tol > 1e-9) ?
              std::clamp((pos_tol + kGatePosEps) / std::max(actual_pos_err, 1e-9), 0.0, 1.0) : 1.0;
            const double command_pos_ratio =
              (pos_tol > 1e-9) ?
              std::clamp((pos_tol + kGatePosEps) / std::max(command_pos_gap, 1e-9), 0.0, 1.0) : 1.0;
            const double actual_ori_ratio =
              (ori_tol > 1e-9) ?
              std::clamp((ori_tol + kGateOriEps) / std::max(actual_ori_err, 1e-9), 0.0, 1.0) : 1.0;
            scan_advance_scale = std::clamp(
              std::min(std::min(actual_pos_ratio, command_pos_ratio), actual_ori_ratio),
              0.0, 1.0);
          } else {
            scan_progress_hard_stop_active_ = false;
            scan_advance_scale = 1.0;
          }
          scan_advance = (scan_advance_scale > 1e-6);
        }
      } else {
        RCLCPP_WARN_THROTTLE(
          get_node()->get_logger(), *get_node()->get_clock(), 1000,
          "hemi_scan_cartesian: invalid tool/target axis, using hold target.");
        scan_advance = false;
      }
    } else {
      RCLCPP_WARN_THROTTLE(
        get_node()->get_logger(), *get_node()->get_clock(), 1000,
        "hemi_scan_cartesian: failed to compute scan pose, using hold target.");
      scan_advance = false;
    }
    tau_cmd = compute_tau_cartesian_hold();
  } else {
    // default: pure cartesian hold
    tau_cmd = compute_tau_cartesian_hold();
  }

  if (scan_seed_blend_active) {
    // Keep the scan phase pinned during the activation catch-up blend so the raw
    // orbit reference cannot run away from the commanded point before tracking settles.
    scan_advance = false;
    scan_advance_scale = 0.0;
  }

  if ((control_mode_ == "hemi_scan_cartesian" || control_mode_ == "paper_surface_scan") &&
      dt_slew > 0.0 && scan_advance) {
    const double scan_progress_dt = dt_slew * scan_advance_scale;
    scan_time_ += scan_progress_dt * std::max(0.0, scan_phase_speed_scale_);
    // Keep the amplitude ramp tied to the gated physical progress instead of the
    // accelerated phase advance so cold starts still ease into the scan orbit.
    scan_ramp_elapsed_ += scan_progress_dt;
  }

  apply_switch_safety(tau_cmd);

  RCLCPP_INFO_THROTTLE(
    get_node()->get_logger(), *get_node()->get_clock(), 500,
    "[DEBUG] mode=%s used_paper=%d hold=%d switch_transient=%d scan_advance=%d scale=%.3f ||tau_cmd||=%.3f",
    control_mode_.c_str(), used_paper ? 1 : 0, switch_hold_active ? 1 : 0,
    switch_transient_active ? 1 : 0, scan_advance ? 1 : 0, scan_advance_scale, tau_cmd.norm());

  // Optional debug telemetry for RViz/plotting.
  debug_publish_accum_ += dt_nonneg;
  const bool publish_debug_now =
    (debug_publish_rate_hz_ <= 0.0) || (debug_publish_accum_ >= (1.0 / debug_publish_rate_hz_));
  if (publish_debug_now) {
    debug_publish_accum_ = 0.0;

    double d_dbg = std::numeric_limits<double>::quiet_NaN();
    Eigen::Vector3d n_dbg = Eigen::Vector3d::Zero();
    const bool have_surface = query_hemisphere_true_(p, d_dbg, n_dbg);
    if (have_surface && n_dbg.norm() > 1e-9) {
      n_dbg.normalize();
    }

    Eigen::Vector3d tool_axis_cur = Tf.linear() * scan_tool_axis_runtime_;
    if (tool_axis_cur.norm() > 1e-9) {
      tool_axis_cur.normalize();
    }
    Eigen::Vector3d tool_axis_des = q_d_.toRotationMatrix() * scan_tool_axis_runtime_;
    if (tool_axis_des.norm() > 1e-9) {
      tool_axis_des.normalize();
    }
    double normal_align_err_rad = std::numeric_limits<double>::quiet_NaN();
    if (have_surface && tool_axis_cur.norm() > 1e-9 && n_dbg.norm() > 1e-9) {
      const Eigen::Vector3d target_axis_dbg = scan_align_inward_normal_ ? -n_dbg : n_dbg;
      const double dot_val = std::clamp(tool_axis_cur.dot(target_axis_dbg), -1.0, 1.0);
      normal_align_err_rad = std::acos(dot_val);
    }

    auto to_point = [](const Eigen::Vector3d& v) {
      geometry_msgs::msg::Point p_msg;
      p_msg.x = v.x();
      p_msg.y = v.y();
      p_msg.z = v.z();
      return p_msg;
    };

    if (debug_marker_pub_) {
      visualization_msgs::msg::MarkerArray ma;
      const rclcpp::Time now = get_node()->now();
      auto make_marker = [&](int id, int type) {
        visualization_msgs::msg::Marker m;
        m.header.stamp = now;
        m.header.frame_id = base_link_;
        m.ns = "surface_impedance_debug";
        m.id = id;
        m.type = type;
        m.action = visualization_msgs::msg::Marker::ADD;
        m.pose.orientation.w = 1.0;
        return m;
      };

      auto m_curr = make_marker(0, visualization_msgs::msg::Marker::SPHERE);
      m_curr.pose.position = to_point(p);
      m_curr.scale.x = 0.015; m_curr.scale.y = 0.015; m_curr.scale.z = 0.015;
      m_curr.color.r = 1.0f; m_curr.color.g = 0.55f; m_curr.color.b = 0.15f; m_curr.color.a = 0.95f;
      ma.markers.push_back(m_curr);

      auto m_des = make_marker(1, visualization_msgs::msg::Marker::SPHERE);
      m_des.pose.position = to_point(p_d_);
      m_des.scale.x = 0.015; m_des.scale.y = 0.015; m_des.scale.z = 0.015;
      m_des.color.r = 0.0f; m_des.color.g = 0.85f; m_des.color.b = 0.90f; m_des.color.a = 0.95f;
      ma.markers.push_back(m_des);

      auto m_line = make_marker(2, visualization_msgs::msg::Marker::LINE_LIST);
      m_line.scale.x = 0.003;
      m_line.color.r = 1.0f; m_line.color.g = 0.95f; m_line.color.b = 0.60f; m_line.color.a = 0.8f;
      m_line.points.push_back(to_point(p));
      m_line.points.push_back(to_point(p_d_));
      ma.markers.push_back(m_line);

      if (have_surface) {
        auto m_n = make_marker(3, visualization_msgs::msg::Marker::ARROW);
        m_n.scale.x = 0.004; m_n.scale.y = 0.008; m_n.scale.z = 0.01;
        m_n.color.r = 1.0f; m_n.color.g = 0.25f; m_n.color.b = 0.25f; m_n.color.a = 0.9f;
        m_n.points.push_back(to_point(p));
        m_n.points.push_back(to_point(p + 0.08 * n_dbg));
        ma.markers.push_back(m_n);
      }

      auto m_tool_cur = make_marker(4, visualization_msgs::msg::Marker::ARROW);
      m_tool_cur.scale.x = 0.004; m_tool_cur.scale.y = 0.008; m_tool_cur.scale.z = 0.01;
      m_tool_cur.color.r = 0.35f; m_tool_cur.color.g = 1.0f; m_tool_cur.color.b = 0.25f; m_tool_cur.color.a = 0.9f;
      m_tool_cur.points.push_back(to_point(p));
      m_tool_cur.points.push_back(to_point(p + 0.08 * tool_axis_cur));
      ma.markers.push_back(m_tool_cur);

      auto m_tool_des = make_marker(5, visualization_msgs::msg::Marker::ARROW);
      m_tool_des.scale.x = 0.004; m_tool_des.scale.y = 0.008; m_tool_des.scale.z = 0.01;
      m_tool_des.color.r = 0.15f; m_tool_des.color.g = 0.45f; m_tool_des.color.b = 1.0f; m_tool_des.color.a = 0.9f;
      m_tool_des.points.push_back(to_point(p_d_));
      m_tool_des.points.push_back(to_point(p_d_ + 0.08 * tool_axis_des));
      ma.markers.push_back(m_tool_des);

      if (have_scan_ref) {
        auto m_scan = make_marker(8, visualization_msgs::msg::Marker::SPHERE);
        m_scan.pose.position = to_point(p_scan_ref);
        m_scan.scale.x = 0.012; m_scan.scale.y = 0.012; m_scan.scale.z = 0.012;
        m_scan.color.r = 1.0f; m_scan.color.g = 0.25f; m_scan.color.b = 0.85f; m_scan.color.a = 0.9f;
        ma.markers.push_back(m_scan);
      }

      if (have_cmd_surface_ref && n_cmd_ref.norm() > 1e-9) {
        Eigen::Vector3d cmd_surface_axis =
          scan_align_inward_normal_ ? -n_cmd_ref.normalized() : n_cmd_ref.normalized();

        auto m_scan_axis = make_marker(9, visualization_msgs::msg::Marker::ARROW);
        m_scan_axis.scale.x = 0.003; m_scan_axis.scale.y = 0.007; m_scan_axis.scale.z = 0.009;
        m_scan_axis.color.r = 1.0f; m_scan_axis.color.g = 0.25f; m_scan_axis.color.b = 0.85f; m_scan_axis.color.a = 0.9f;
        m_scan_axis.points.push_back(to_point(p_d_));
        m_scan_axis.points.push_back(to_point(p_d_ + 0.08 * cmd_surface_axis));
        ma.markers.push_back(m_scan_axis);
      }

      if (have_scan_ref && n_scan_ref.norm() > 1e-9) {
        const Eigen::Vector3d raw_axis =
          scan_align_inward_normal_ ? -n_scan_ref.normalized() : n_scan_ref.normalized();

        auto m_raw_axis = make_marker(10, visualization_msgs::msg::Marker::ARROW);
        m_raw_axis.scale.x = 0.0025; m_raw_axis.scale.y = 0.0055; m_raw_axis.scale.z = 0.0075;
        m_raw_axis.color.r = 1.0f; m_raw_axis.color.g = 0.60f; m_raw_axis.color.b = 0.90f; m_raw_axis.color.a = 0.45f;
        m_raw_axis.points.push_back(to_point(p_scan_ref));
        m_raw_axis.points.push_back(to_point(p_scan_ref + 0.08 * raw_axis));
        ma.markers.push_back(m_raw_axis);

        auto m_scan_gap = make_marker(11, visualization_msgs::msg::Marker::LINE_LIST);
        m_scan_gap.scale.x = 0.002;
        m_scan_gap.color.r = 1.0f; m_scan_gap.color.g = 0.60f; m_scan_gap.color.b = 0.90f; m_scan_gap.color.a = 0.45f;
        m_scan_gap.points.push_back(to_point(p_d_));
        m_scan_gap.points.push_back(to_point(p_scan_ref));
        ma.markers.push_back(m_scan_gap);
      }

      auto m_center = make_marker(6, visualization_msgs::msg::Marker::SPHERE);
      m_center.pose.position = to_point(hemisphere_center_);
      m_center.scale.x = 0.012; m_center.scale.y = 0.012; m_center.scale.z = 0.012;
      m_center.color.r = 0.75f; m_center.color.g = 0.35f; m_center.color.b = 1.0f; m_center.color.a = 0.9f;
      ma.markers.push_back(m_center);

      // RViz-only surface visualization: draw the hemisphere mesh at the configured center/axis.
      auto m_surface = make_marker(7, visualization_msgs::msg::Marker::MESH_RESOURCE);
      m_surface.mesh_resource =
        "package://franka_surface_impedance_controller/models/hemi_surface/meshes/hemisphere_r100_centered_meters.stl";
      m_surface.mesh_use_embedded_materials = false;
      m_surface.pose.position = to_point(hemisphere_center_);
      if (hemisphere_axis_.norm() > 1e-9) {
        const Eigen::Quaterniond q_mesh =
          Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitZ(), hemisphere_axis_.normalized());
        m_surface.pose.orientation.x = q_mesh.x();
        m_surface.pose.orientation.y = q_mesh.y();
        m_surface.pose.orientation.z = q_mesh.z();
        m_surface.pose.orientation.w = q_mesh.w();
      }
      const double mesh_scale = (hemisphere_radius_ > 1e-9) ? (hemisphere_radius_ / 0.10) : 1.0;
      m_surface.scale.x = mesh_scale;
      m_surface.scale.y = mesh_scale;
      m_surface.scale.z = mesh_scale;
      m_surface.color.r = 0.70f; m_surface.color.g = 0.90f; m_surface.color.b = 1.0f; m_surface.color.a = 0.22f;
      ma.markers.push_back(m_surface);

      const bool scan_mode_active =
        (control_mode_ == "hemi_scan_cartesian" || control_mode_ == "paper_surface_scan");
      if (scan_mode_active) {
        auto m_expected_traj = make_marker(12, visualization_msgs::msg::Marker::LINE_STRIP);
        m_expected_traj.scale.x = 0.0025;
        m_expected_traj.color.r = 0.15f;
        m_expected_traj.color.g = 1.0f;
        m_expected_traj.color.b = 0.95f;
        m_expected_traj.color.a = 0.90f;

        double horizon_s = 30.0;
        if (scan_track_generator_ == "paper_surface_lissajous") {
          const double wx = std::max(1e-3, std::abs(scan_surface_lissajous_omega_.x()));
          const double wy = std::max(1e-3, std::abs(scan_surface_lissajous_omega_.y()));
          const double min_w = std::max(1e-3, std::min(wx, wy));
          horizon_s = std::clamp(12.566370614359172 / min_w, 20.0, 90.0);  // 4*pi/min_w
        } else {
          horizon_s = std::clamp(1.0 / std::max(1e-4, scan_theta_freq_), 8.0, 45.0);
        }

        constexpr int kTrajectorySamples = 220;
        for (int i = 0; i < kTrajectorySamples; ++i) {
          const double alpha =
            (kTrajectorySamples > 1) ? static_cast<double>(i) / (kTrajectorySamples - 1) : 0.0;
          Eigen::Vector3d p_traj = Eigen::Vector3d::Zero();
          Eigen::Vector3d n_traj = Eigen::Vector3d::UnitZ();
          if (compute_hemi_scan_pose(scan_time_ + alpha * horizon_s, p_traj, n_traj)) {
            m_expected_traj.points.push_back(to_point(p_traj));
          }
        }

        if (m_expected_traj.points.size() >= 2U) {
          ma.markers.push_back(m_expected_traj);
        }
      } else {
        auto m_expected_traj_del = make_marker(12, visualization_msgs::msg::Marker::LINE_STRIP);
        m_expected_traj_del.action = visualization_msgs::msg::Marker::DELETE;
        ma.markers.push_back(m_expected_traj_del);
      }

      debug_marker_pub_->publish(ma);
    }

    if (debug_state_pub_) {
      std_msgs::msg::Float64MultiArray msg;
      auto mode_to_code = [&](const std::string& m) {
        if (m == "paper_p") return 1.0;
        if (m == "cartesian_surface") return 2.0;
        if (m == "hemi_scan_cartesian") return 3.0;
        if (m == "paper_surface_scan") return 4.0;
        return 0.0;
      };
      msg.data = {
        scan_time_,
        mode_to_code(control_mode_),
        used_paper ? 1.0 : 0.0,
        have_surface ? 1.0 : 0.0,
        d_dbg,
        p.x(), p.y(), p.z(),
        p_d_.x(), p_d_.y(), p_d_.z(),
        n_dbg.x(), n_dbg.y(), n_dbg.z(),
        tool_axis_cur.x(), tool_axis_cur.y(), tool_axis_cur.z(),
        tool_axis_des.x(), tool_axis_des.y(), tool_axis_des.z(),
        tau_cmd.norm(),
        normal_align_err_rad,
        scan_advance ? 1.0 : 0.0,
        scan_advance_scale,
        scan_surface_cmd_ratio,
        scan_progress_hard_stop_active_ ? 1.0 : 0.0,
        paper_surface_acquired_ ? 1.0 : 0.0,
        debug_scan_seed_blend_active_ ? 1.0 : 0.0,
        debug_paper_acquire_active_ ? 1.0 : 0.0,
        debug_scan_surface_err_,
        debug_scan_surface_dist_err_,
        debug_scan_surface_ori_err_,
        debug_scan_cmd_ori_err_,
        debug_scan_band_err_,
        debug_scan_reacquire_reason_,
        debug_scan_roll_err_,
        debug_scan_roll_rate_
      };
      debug_state_pub_->publish(msg);
    }
  }

  for (size_t i = 0; i < cmd_effort_.size(); ++i) {
    cmd_effort_[i]->set_value(tau_cmd(static_cast<int>(i)));
  }

  return controller_interface::return_type::OK;
}

hardware_interface::LoanedCommandInterface * SurfaceImpedanceController::get_command_handle_(
  const std::string & joint_name, const std::string & interface_name) {

  const std::string full = joint_name + "/" + interface_name;

  auto it = std::find_if(
    command_interfaces_.begin(), command_interfaces_.end(),
    [&](hardware_interface::LoanedCommandInterface & ci) {
      const auto & n = ci.get_name();
      const auto & in = ci.get_interface_name();

      // Case 1: name="fer_joint1", interface="effort"
      if (n == joint_name && in == interface_name) return true;
      // Case 2: name="fer_joint1/effort"
      if (n == full) return true;
      // Case 3: interface_name() 里带 full（有些后端会这样拼）
      if (in == full) return true;

      return false;
    });

  if (it == command_interfaces_.end()) return nullptr;
  return &(*it);
}



hardware_interface::LoanedStateInterface * SurfaceImpedanceController::get_state_handle_(
  const std::string & joint_name, const std::string & interface_name) {

  const std::string full = joint_name + "/" + interface_name;

  auto it = std::find_if(
    state_interfaces_.begin(), state_interfaces_.end(),
    [&](hardware_interface::LoanedStateInterface & si) {
      if (si.get_name() == joint_name && si.get_interface_name() == interface_name) {
        return true;
      }
      if (si.get_name() == full) {
        return true;
      }
      return false;
    });

  if (it == state_interfaces_.end()) {
    return nullptr;
  }
  return &(*it);
}


Eigen::Matrix<double, 7, 1> SurfaceImpedanceController::get_q_() const {
  Eigen::Matrix<double, 7, 1> q;
  q.setZero();
  for (size_t i = 0; i < state_pos_.size(); ++i) {
    q(static_cast<int>(i)) = state_pos_[i]->get_value();
  }
  return q;
}

Eigen::Matrix<double, 7, 1> SurfaceImpedanceController::get_dq_() const {
  Eigen::Matrix<double, 7, 1> dq;
  dq.setZero();
  for (size_t i = 0; i < state_vel_.size(); ++i) {
    dq(static_cast<int>(i)) = state_vel_[i]->get_value();
  }
  return dq;
}

bool SurfaceImpedanceController::compute_fk_jacobian_kdl_(
  const Eigen::Matrix<double, 7, 1>& q,
  Eigen::Vector3d& p_base,
  Eigen::Quaterniond& q_base,
  Eigen::Matrix<double, 6, 7>& J_out) {

  if (!kdl_initialized_ || !fk_solver_ || !jac_solver_) {
    return false;
  }
  if (kdl_q_.rows() != 7) {
    RCLCPP_ERROR(get_node()->get_logger(), "KDL joint array size is %d (expected 7).", (int)kdl_q_.rows());
    return false;
  }

  // IMPORTANT: assumes KDL chain joint order matches fer_joint1..7 order
  for (int i = 0; i < 7; ++i) {
    kdl_q_(i) = q(i);
  }

  // FK
  KDL::Frame ee_frame;
  const int fk_ret = fk_solver_->JntToCart(kdl_q_, ee_frame);
  if (fk_ret < 0) {
    RCLCPP_ERROR(get_node()->get_logger(), "KDL FK failed, ret=%d", fk_ret);
    return false;
  }

  p_base = Eigen::Vector3d(ee_frame.p.x(), ee_frame.p.y(), ee_frame.p.z());
  double qx, qy, qz, qw;
  ee_frame.M.GetQuaternion(qx, qy, qz, qw);  // (x,y,z,w)
  q_base = Eigen::Quaterniond(qw, qx, qy, qz);
  q_base.normalize();

  // Jacobian
  const int jac_ret = jac_solver_->JntToJac(kdl_q_, kdl_J_);
  if (jac_ret < 0) {
    RCLCPP_ERROR(get_node()->get_logger(), "KDL Jacobian failed, ret=%d", jac_ret);
    return false;
  }
  if (kdl_J_.data.cols() != 7) {
    RCLCPP_ERROR(get_node()->get_logger(), "KDL Jacobian cols=%d (expected 7).", (int)kdl_J_.data.cols());
    return false;
  }
  J_out = kdl_J_.data;

  return true;
}


bool SurfaceImpedanceController::build_hemisphere_basis_(
  Eigen::Vector3d& axis, Eigen::Vector3d& e1, Eigen::Vector3d& e2) const {

  if (hemisphere_axis_.norm() < 1e-9) {
    return false;
  }

  axis = hemisphere_axis_.normalized();
  const Eigen::Vector3d seed = (std::abs(axis.z()) < 0.9) ? Eigen::Vector3d::UnitZ() : Eigen::Vector3d::UnitX();
  e1 = seed - seed.dot(axis) * axis;
  if (e1.norm() < 1e-9) {
    return false;
  }
  e1.normalize();
  e2 = axis.cross(e1);
  if (e2.norm() < 1e-9) {
    return false;
  }
  e2.normalize();
  return true;
}

double SurfaceImpedanceController::effective_scan_phi_max_() const {
  double phi_max = std::min(0.5 * kPi, scan_phi_max_);
  if (!(std::isfinite(phi_max))) {
    return 0.5 * kPi;
  }
  if (scan_min_height_above_base_ <= 1e-9 ||
      !(std::isfinite(hemisphere_radius_) && hemisphere_radius_ > 1e-9)) {
    return std::clamp(phi_max, 0.0, 0.5 * kPi);
  }

  const double min_height = std::clamp(scan_min_height_above_base_, 0.0, hemisphere_radius_);
  const double min_axis_dot = std::clamp(min_height / hemisphere_radius_, 0.0, 1.0);
  const double phi_max_from_height = std::acos(min_axis_dot);
  phi_max = std::min(phi_max, phi_max_from_height);
  return std::clamp(phi_max, 0.0, 0.5 * kPi);
}

bool SurfaceImpedanceController::normal_to_hemi_angles_(
  const Eigen::Vector3d& normal_base, double& theta, double& phi) const {

  if (normal_base.norm() < 1e-9) {
    return false;
  }

  Eigen::Vector3d axis = Eigen::Vector3d::Zero();
  Eigen::Vector3d e1 = Eigen::Vector3d::Zero();
  Eigen::Vector3d e2 = Eigen::Vector3d::Zero();
  if (!build_hemisphere_basis_(axis, e1, e2)) {
    return false;
  }

  constexpr double kPi = 3.14159265358979323846;
  const Eigen::Vector3d n = normal_base.normalized();
  phi = std::acos(std::clamp(n.dot(axis), -1.0, 1.0));
  phi = std::clamp(phi, 0.0, 0.5 * kPi);

  Eigen::Vector3d proj = n - n.dot(axis) * axis;
  if (proj.norm() < 1e-9) {
    theta = 0.0;
    return true;
  }
  proj.normalize();
  theta = std::atan2(proj.dot(e2), proj.dot(e1));
  return std::isfinite(theta) && std::isfinite(phi);
}

bool SurfaceImpedanceController::project_normal_to_scan_band_(
  const Eigen::Vector3d& normal_in, Eigen::Vector3d& normal_out,
  double* theta_out, double* phi_out) const {

  if (normal_in.norm() < 1e-9) {
    return false;
  }

  double theta = 0.0;
  double phi = 0.0;
  if (!normal_to_hemi_angles_(normal_in, theta, phi)) {
    return false;
  }

  theta = clamp_theta_to_sector(theta, scan_theta_min_, scan_theta_max_);
  const double phi_min = std::max(0.0, scan_phi_min_);
  const double phi_max = effective_scan_phi_max_();
  phi = (phi_max <= phi_min + 1e-6) ? phi_min : std::clamp(phi, phi_min, phi_max);

  Eigen::Vector3d axis = Eigen::Vector3d::Zero();
  Eigen::Vector3d e1 = Eigen::Vector3d::Zero();
  Eigen::Vector3d e2 = Eigen::Vector3d::Zero();
  if (!build_hemisphere_basis_(axis, e1, e2)) {
    return false;
  }

  const Eigen::Vector3d t_circle = std::cos(theta) * e1 + std::sin(theta) * e2;
  normal_out = std::cos(phi) * axis + std::sin(phi) * t_circle;
  if (normal_out.norm() < 1e-9) {
    return false;
  }
  normal_out.normalize();

  if (theta_out != nullptr) {
    *theta_out = theta;
  }
  if (phi_out != nullptr) {
    *phi_out = phi;
  }
  return true;
}

bool SurfaceImpedanceController::normal_to_hemi_surface_coords_(
  const Eigen::Vector3d& normal_base, Eigen::Vector2d& s) const {

  if (!(std::isfinite(hemisphere_radius_) && hemisphere_radius_ > 1e-6)) {
    return false;
  }

  double theta = 0.0;
  double phi = 0.0;
  if (!normal_to_hemi_angles_(normal_base, theta, phi)) {
    return false;
  }

  const double rho = hemisphere_radius_ * phi;
  s.x() = rho * std::cos(theta);
  s.y() = rho * std::sin(theta);
  return s.allFinite();
}

bool SurfaceImpedanceController::hemi_surface_coords_to_normal_(
  const Eigen::Vector2d& s, Eigen::Vector3d& normal_base) const {

  if (!(std::isfinite(hemisphere_radius_) && hemisphere_radius_ > 1e-6) || !s.allFinite()) {
    return false;
  }

  Eigen::Vector3d axis = Eigen::Vector3d::Zero();
  Eigen::Vector3d e1 = Eigen::Vector3d::Zero();
  Eigen::Vector3d e2 = Eigen::Vector3d::Zero();
  if (!build_hemisphere_basis_(axis, e1, e2)) {
    return false;
  }

  const double rho = s.norm();
  double theta = (rho > 1e-9) ? std::atan2(s.y(), s.x()) : 0.0;
  double phi = rho / hemisphere_radius_;
  theta = clamp_theta_to_sector(theta, scan_theta_min_, scan_theta_max_);

  const double phi_min = std::max(0.0, scan_phi_min_);
  const double phi_max = effective_scan_phi_max_();
  if (phi_max <= phi_min + 1e-6) {
    phi = phi_min;
  } else {
    phi = std::clamp(phi, phi_min, phi_max);
  }

  const Eigen::Vector3d t_circle = std::cos(theta) * e1 + std::sin(theta) * e2;
  normal_base = std::cos(phi) * axis + std::sin(phi) * t_circle;
  if (normal_base.norm() < 1e-9) {
    return false;
  }
  normal_base.normalize();
  return true;
}

bool SurfaceImpedanceController::query_hemisphere_proxy_(
  const Eigen::Vector3d & p_base, Eigen::Vector3d & proxy_base, double & signed_distance,
  Eigen::Vector3d & normal_base) const {

  if (!(std::isfinite(hemisphere_radius_) && hemisphere_radius_ > 0.0) ||
      hemisphere_axis_.norm() < 1e-9) {
    return false;
  }

  const Eigen::Vector3d axis = hemisphere_axis_.normalized();
  const Eigen::Vector3d v = p_base - hemisphere_center_;
  const double r = v.norm();

  if (r > 1e-9) {
    const Eigen::Vector3d sphere_normal = v / r;
    if (sphere_normal.dot(axis) >= 0.0) {
      normal_base = sphere_normal;
      proxy_base = hemisphere_center_ + hemisphere_radius_ * normal_base;
    } else {
      const double axial = v.dot(axis);
      const Eigen::Vector3d v_tan = v - axial * axis;
      Eigen::Vector3d rim_dir = v_tan;
      if (rim_dir.norm() < 1e-9) {
        Eigen::Vector3d axis_tmp = Eigen::Vector3d::Zero();
        Eigen::Vector3d e1 = Eigen::Vector3d::Zero();
        Eigen::Vector3d e2 = Eigen::Vector3d::Zero();
        if (!build_hemisphere_basis_(axis_tmp, e1, e2)) {
          return false;
        }
        rim_dir = e1;
      }
      rim_dir.normalize();
      proxy_base = hemisphere_center_ + hemisphere_radius_ * rim_dir;
      normal_base = rim_dir;
    }
  } else {
    Eigen::Vector3d axis_tmp = Eigen::Vector3d::Zero();
    Eigen::Vector3d e1 = Eigen::Vector3d::Zero();
    Eigen::Vector3d e2 = Eigen::Vector3d::Zero();
    if (!build_hemisphere_basis_(axis_tmp, e1, e2)) {
      return false;
    }
    proxy_base = hemisphere_center_ + hemisphere_radius_ * e1;
    normal_base = e1;
  }

  if (normal_base.norm() < 1e-9) {
    return false;
  }
  normal_base.normalize();

  const Eigen::Vector3d c = p_base - proxy_base;
  const double dist = c.norm();
  if (!std::isfinite(dist)) {
    return false;
  }
  const double signed_proj = normal_base.dot(c);
  signed_distance = (signed_proj >= 0.0 ? 1.0 : -1.0) * dist;
  return proxy_base.allFinite() && normal_base.allFinite() && std::isfinite(signed_distance);
}

bool SurfaceImpedanceController::query_hemisphere_true_(
  const Eigen::Vector3d & p_base, double & signed_distance, Eigen::Vector3d & normal_base) const {

  if (!(std::isfinite(hemisphere_radius_) && hemisphere_radius_ > 0.0) ||
      hemisphere_axis_.norm() < 1e-9) {
    return false;
  }

  const Eigen::Vector3d axis = hemisphere_axis_.normalized();
  const Eigen::Vector3d v = p_base - hemisphere_center_;
  const double r = v.norm();

  if (r > 1e-9) {
    normal_base = v / r;
    signed_distance = r - hemisphere_radius_;
  } else {
    normal_base = axis;
    signed_distance = -hemisphere_radius_;
  }

  if (normal_base.norm() < 1e-9) {
    return false;
  }
  normal_base.normalize();
  return normal_base.allFinite() && std::isfinite(signed_distance);
}

bool SurfaceImpedanceController::query_hemisphere_(
  const Eigen::Vector3d & p_base, double & signed_distance, Eigen::Vector3d & normal_base) const {

  return query_hemisphere_true_(p_base, signed_distance, normal_base);
}

bool SurfaceImpedanceController::init_kdl_from_urdf_file_() {
  if (urdf_path_.empty() || base_link_.empty() || ee_link_.empty()) {
    RCLCPP_ERROR(
      get_node()->get_logger(),
      "KDL backend requires non-empty urdf_path, base_link, ee_link. Got urdf_path='%s', base_link='%s', ee_link='%s'",
      urdf_path_.c_str(), base_link_.c_str(), ee_link_.c_str());
    return false;
  }

  std::ifstream ifs(urdf_path_);
  if (!ifs.good()) {
    RCLCPP_ERROR(get_node()->get_logger(), "Cannot open URDF file: %s", urdf_path_.c_str());
    return false;
  }
  std::stringstream buffer;
  buffer << ifs.rdbuf();
  const std::string urdf_xml = buffer.str();
  if (urdf_xml.empty()) {
    RCLCPP_ERROR(get_node()->get_logger(), "URDF file is empty: %s", urdf_path_.c_str());
    return false;
  }

  KDL::Tree tree;
  if (!kdl_parser::treeFromString(urdf_xml, tree)) {
    RCLCPP_ERROR(get_node()->get_logger(), "kdl_parser::treeFromString failed.");
    return false;
  }

  KDL::Chain chain;
  if (!tree.getChain(base_link_, ee_link_, chain)) {
    RCLCPP_ERROR(
      get_node()->get_logger(),
      "KDL getChain failed: base_link='%s' -> ee_link='%s'. Check link names in URDF.",
      base_link_.c_str(), ee_link_.c_str());
    return false;
  }

  kdl_tree_ = tree;
  kdl_chain_ = chain;
  fk_solver_ = std::make_unique<KDL::ChainFkSolverPos_recursive>(kdl_chain_);
  jac_solver_ = std::make_unique<KDL::ChainJntToJacSolver>(kdl_chain_);
  kdl_q_ = KDL::JntArray(kdl_chain_.getNrOfJoints());
  kdl_J_ = KDL::Jacobian(kdl_chain_.getNrOfJoints());
  kdl_initialized_ = true;

  RCLCPP_INFO(
    get_node()->get_logger(),
    "KDL initialized: joints=%u, segments=%u, base='%s', ee='%s'",
    kdl_chain_.getNrOfJoints(), kdl_chain_.getNrOfSegments(),
    base_link_.c_str(), ee_link_.c_str());

  // 可选：提醒一下不是 7 DOF 就要小心映射
  if (kdl_chain_.getNrOfJoints() != 7) {
    RCLCPP_WARN(get_node()->get_logger(), "KDL chain joint count is %u (expected 7). Mapping must be checked.",
                kdl_chain_.getNrOfJoints());
  }
  return true;
}

}  // namespace franka_surface_impedance_controller

PLUGINLIB_EXPORT_CLASS(
  franka_surface_impedance_controller::SurfaceImpedanceController,
  controller_interface::ControllerInterface)

#ifdef LOG
#undef LOG
#endif
#include "gazebo_plugins/quad_sim_effort_controller_plugin.hpp"

#include <angles/angles.h>

#include <algorithm>

namespace gz_plugins {
namespace {

template <typename ParamType>
void declare_if_missing(const rclcpp::Node::SharedPtr& node,
                        const std::string& param_name,
                        const ParamType& default_value) {
  if (!node->has_parameter(param_name)) {
    node->declare_parameter<ParamType>(param_name, default_value);
  }
}

bool load_leg_joint_names(const rclcpp::Node::SharedPtr& node, int leg_idx,
                          std::array<std::string, 3>& joint_names) {
  const std::string leg_ns = "leg_" + std::to_string(leg_idx);
  declare_if_missing<std::vector<std::string>>(
      node, leg_ns + ".joint_names", std::vector<std::string>({"", "", ""}));
  declare_if_missing<std::string>(node, leg_ns + ".joints.abad.name", "");
  declare_if_missing<std::string>(node, leg_ns + ".joints.hip.name", "");
  declare_if_missing<std::string>(node, leg_ns + ".joints.knee.name", "");

  std::string abad_name;
  std::string hip_name;
  std::string knee_name;
  if (node->get_parameter(leg_ns + ".joints.abad.name", abad_name) &&
      node->get_parameter(leg_ns + ".joints.hip.name", hip_name) &&
      node->get_parameter(leg_ns + ".joints.knee.name", knee_name) &&
      !abad_name.empty() && !hip_name.empty() && !knee_name.empty()) {
    joint_names = {abad_name, hip_name, knee_name};
    return true;
  }

  std::vector<std::string> legacy_joint_names;
  if (node->get_parameter(leg_ns + ".joint_names", legacy_joint_names) &&
      legacy_joint_names.size() == 3) {
    joint_names = {legacy_joint_names[0], legacy_joint_names[1],
                   legacy_joint_names[2]};
    return true;
  }

  return false;
}

bool load_motor_limits(const rclcpp::Node::SharedPtr& node,
                       std::vector<double>& torque_lims,
                       std::vector<double>& speed_lims) {
  declare_if_missing<std::vector<double>>(node, "motor_limits.torque", {});
  declare_if_missing<std::vector<double>>(node, "motor_limits.speed", {});
  if (!node->get_parameter("motor_limits.torque", torque_lims) ||
      !node->get_parameter("motor_limits.speed", speed_lims)) {
    return false;
  }
  return torque_lims.size() == 3 && speed_lims.size() == 3;
}

}  // namespace

void QuadSimEffortController::Configure(
    const gz::sim::Entity& entity, const std::shared_ptr<const sdf::Element>& sdf,
    gz::sim::EntityComponentManager& ecm, gz::sim::EventManager& /*eventMgr*/) {
  this->model_ = gz::sim::Model(entity);

  if (!rclcpp::ok()) {
    rclcpp::init(0, nullptr);
  }

  std::string robot_ns = this->model_.Name(ecm);
  if (sdf->HasElement("namespace")) {
    robot_ns = sdf->Get<std::string>("namespace");
  }

  std::string params_file;
  if (sdf->HasElement("parameters")) {
    params_file = sdf->Get<std::string>("parameters");
  }

  rclcpp::NodeOptions options;
  options.automatically_declare_parameters_from_overrides(true);
  options.allow_undeclared_parameters(true);
  options.parameter_overrides({rclcpp::Parameter("use_sim_time", true)});
  if (!params_file.empty()) {
    options.arguments({"--ros-args", "--params-file", params_file});
  }

  this->node_ =
      std::make_shared<rclcpp::Node>("quad_sim_effort_controller", robot_ns,
                                     options);

  if (!this->LoadConfig()) {
    RCLCPP_FATAL(this->node_->get_logger(),
                 "Failed to load QuadSimEffortController parameters");
    return;
  }

  this->joints_.clear();
  for (const auto& joint_name : this->joint_names_) {
    auto joint_entity = this->model_.JointByName(ecm, joint_name);
    if (joint_entity == gz::sim::kNullEntity) {
      RCLCPP_FATAL(this->node_->get_logger(), "Could not find joint '%s'",
                   joint_name.c_str());
      this->joints_.clear();
      return;
    }
    gz::sim::Joint joint(joint_entity);
    joint.EnablePositionCheck(ecm, true);
    joint.EnableVelocityCheck(ecm, true);
    this->joints_.push_back(joint);
  }

  std::string joint_command_topic = "control/joint_command";
  declare_if_missing<std::string>(this->node_, "topics.control.joint_command",
                                  joint_command_topic);
  this->node_->get_parameter("topics.control.joint_command",
                             joint_command_topic);

  this->sub_command_ =
      this->node_->create_subscription<quad_msgs::msg::LegCommandArray>(
          joint_command_topic, 1,
          std::bind(&QuadSimEffortController::CommandCallback, this,
                    std::placeholders::_1));

  RCLCPP_INFO(this->node_->get_logger(),
              "QuadSimEffortController listening on '%s'",
              this->sub_command_->get_topic_name());
}

bool QuadSimEffortController::LoadConfig() {
  this->leg_map_ = {std::make_pair(0, 1),  std::make_pair(0, 2),
                    std::make_pair(1, 1),  std::make_pair(1, 2),
                    std::make_pair(2, 1),  std::make_pair(2, 2),
                    std::make_pair(3, 1),  std::make_pair(3, 2),
                    std::make_pair(0, 0),  std::make_pair(1, 0),
                    std::make_pair(2, 0),  std::make_pair(3, 0)};

  std::array<std::array<std::string, 3>, 4> leg_joint_names;
  for (int leg_idx = 0; leg_idx < 4; ++leg_idx) {
    if (!load_leg_joint_names(this->node_, leg_idx, leg_joint_names[leg_idx])) {
      return false;
    }
  }

  this->joint_names_ = {
      leg_joint_names[0][1], leg_joint_names[0][2], leg_joint_names[1][1],
      leg_joint_names[1][2], leg_joint_names[2][1], leg_joint_names[2][2],
      leg_joint_names[3][1], leg_joint_names[3][2], leg_joint_names[0][0],
      leg_joint_names[1][0], leg_joint_names[2][0], leg_joint_names[3][0]};

  return load_motor_limits(this->node_, this->torque_lims_, this->speed_lims_);
}

void QuadSimEffortController::PreUpdate(const gz::sim::UpdateInfo& /*info*/,
                                        gz::sim::EntityComponentManager& ecm) {
  if (!this->node_ || this->joints_.size() != this->joint_names_.size()) {
    return;
  }

  rclcpp::spin_some(this->node_);

  std::vector<quad_msgs::msg::LegCommand> commands;
  {
    std::lock_guard<std::mutex> lock(this->command_mutex_);
    commands = this->commands_;
  }

  const bool has_command =
      commands.size() >= 4 && !commands.front().motor_commands.empty();
  static const double sit_angles[3] = {0.0, 1.36, -2.65};
  static const double hold_kp = 40.0;
  static const double hold_kd = 2.0;

  for (size_t i = 0; i < this->joints_.size(); ++i) {
    auto position = this->joints_[i].Position(ecm);
    auto velocity = this->joints_[i].Velocity(ecm);
    if (!position || position->empty() || !velocity || velocity->empty()) {
      continue;
    }

    const auto [leg_idx, motor_idx] = this->leg_map_[i];
    double target_pos = sit_angles[motor_idx];
    double target_vel = 0.0;
    double kp = hold_kp;
    double kd = hold_kd;
    double torque_ff = 0.0;

    if (has_command &&
        commands[leg_idx].motor_commands.size() > static_cast<size_t>(motor_idx)) {
      const auto& cmd = commands[leg_idx].motor_commands[motor_idx];
      target_pos = cmd.pos_setpoint;
      target_vel = cmd.vel_setpoint;
      kp = cmd.kp;
      kd = cmd.kd;
      torque_ff = cmd.torque_ff;
      this->first_command_received_ = true;
    }

    const double pos_error =
        angles::shortest_angular_distance(position->front(), target_pos);
    const double vel_error = target_vel - velocity->front();
    double torque = kp * pos_error + kd * vel_error + torque_ff;
    const double torque_lim = this->torque_lims_[motor_idx];
    torque = std::clamp(torque, -torque_lim, torque_lim);
    this->joints_[i].SetForce(ecm, {torque});
  }
}

void QuadSimEffortController::CommandCallback(
    const quad_msgs::msg::LegCommandArray::SharedPtr msg) {
  std::lock_guard<std::mutex> lock(this->command_mutex_);
  this->commands_ = msg->leg_commands;
}

}  // namespace gz_plugins

GZ_ADD_PLUGIN(gz_plugins::QuadSimEffortController, gz::sim::System,
              gz::sim::ISystemConfigure, gz::sim::ISystemPreUpdate)

GZ_ADD_PLUGIN_ALIAS(gz_plugins::QuadSimEffortController,
                    "quad_sim_effort_controller")

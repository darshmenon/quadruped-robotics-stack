#ifndef GAZEBO_PLUGINS_QUAD_SIM_EFFORT_CONTROLLER_PLUGIN
#define GAZEBO_PLUGINS_QUAD_SIM_EFFORT_CONTROLLER_PLUGIN

#ifdef LOG
#undef LOG
#endif

#include <gz/plugin/Register.hh>
#include <gz/sim/Entity.hh>
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/EventManager.hh>
#include <gz/sim/Joint.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/components.hh>

#include <quad_msgs/msg/leg_command.hpp>
#include <quad_msgs/msg/leg_command_array.hpp>
#include <rclcpp/rclcpp.hpp>

#include <array>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace gz_plugins {

class QuadSimEffortController : public gz::sim::System,
                                public gz::sim::ISystemConfigure,
                                public gz::sim::ISystemPreUpdate {
 public:
  QuadSimEffortController() = default;

  void Configure(const gz::sim::Entity& entity,
                 const std::shared_ptr<const sdf::Element>& sdf,
                 gz::sim::EntityComponentManager& ecm,
                 gz::sim::EventManager& eventMgr) override;

  void PreUpdate(const gz::sim::UpdateInfo& info,
                 gz::sim::EntityComponentManager& ecm) override;

 private:
  bool LoadConfig();
  void CommandCallback(const quad_msgs::msg::LegCommandArray::SharedPtr msg);

  gz::sim::Model model_;
  rclcpp::Node::SharedPtr node_;
  rclcpp::Subscription<quad_msgs::msg::LegCommandArray>::SharedPtr sub_command_;

  std::vector<std::string> joint_names_;
  std::vector<gz::sim::Joint> joints_;
  std::array<std::pair<int, int>, 12> leg_map_;
  std::vector<double> torque_lims_;
  std::vector<double> speed_lims_;

  std::mutex command_mutex_;
  std::vector<quad_msgs::msg::LegCommand> commands_;
  bool first_command_received_{false};
};

}  // namespace gz_plugins

#endif  // GAZEBO_PLUGINS_QUAD_SIM_EFFORT_CONTROLLER_PLUGIN

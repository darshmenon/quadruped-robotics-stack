cmake_minimum_required(VERSION 3.8)
project(gazebo_plugins)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)

# gz-sim8 must be found before quad_utils: quad_utils transitively pulls in
# jsoncpp via CONFIG mode (through its Pinocchio/PCL/VTK chain), and
# gz-fuel_tools9 (a gz-sim8 dependency) does the same. jsoncpp's own
# jsoncpp-namespaced-targets.cmake unconditionally add_library()s
# JsonCpp::JsonCpp with no `if(NOT TARGET ...)` guard, so whichever chain
# runs second hard-errors on redefining it. Finding gz-sim8 first avoids
# that -- confirmed empirically, not just theorized.
find_package(gz-sim8 REQUIRED)
find_package(gz-plugin2 REQUIRED)

find_package(controller_interface REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(angles REQUIRED)
find_package(realtime_tools REQUIRED)
find_package(urdf REQUIRED)
find_package(quad_utils REQUIRED)
find_package(std_msgs REQUIRED)
find_package(quad_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(pinocchio REQUIRED)

if(NOT DEFINED QUADSDK_DEPS_PREFIX)
  set(_quadsdk_default_prefix "${CMAKE_CURRENT_SOURCE_DIR}/../../../quad_sdk_external/local")
  if(EXISTS "${_quadsdk_default_prefix}")
    get_filename_component(QUADSDK_DEPS_PREFIX "${_quadsdk_default_prefix}" ABSOLUTE)
  endif()
endif()
if(DEFINED QUADSDK_DEPS_PREFIX)
  list(APPEND CMAKE_PREFIX_PATH "${QUADSDK_DEPS_PREFIX}")
endif()

find_package(RBDL REQUIRED COMPONENTS URDFREADER)
# list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../quad_utils/cmake")
# find_package(RBDL REQUIRED COMPONENTS URDFREADER)

# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

# Ground Truth Estimator

add_library(ground_truth_estimator SHARED
  src/estimator_plugin.cpp
)

add_library(quad_wheel_ground_truth_estimator SHARED
  src/quad_wheel_estimator_plugin.cpp
)

add_library(quad_controller SHARED
 src/controller_plugin.cpp
)

add_library(quad_wheel_controller SHARED
  src/quad_wheel_controller_plugin.cpp
)

add_library(quad_sim_effort_controller SHARED
  src/quad_sim_effort_controller_plugin.cpp
)

add_library(spawn_lock SHARED
  src/spawn_lock_plugin.cpp
)

# These four (not spawn_lock) transitively include quad_utils's
# ros_utils.hpp/quad_kd2.hpp. ament_target_dependencies(quad_utils) alone
# doesn't propagate pinocchio's INTERFACE_COMPILE_DEFINITIONS (e.g.
# BOOST_MPL_LIMIT_LIST_SIZE=30, needed since Pinocchio's joint-type variant
# now exceeds Boost's default 20-type mpl::list limit), so link the modern
# imported target directly.
target_link_libraries(ground_truth_estimator pinocchio::pinocchio)
target_link_libraries(quad_wheel_ground_truth_estimator pinocchio::pinocchio)
target_link_libraries(quad_controller pinocchio::pinocchio)
target_link_libraries(quad_wheel_controller pinocchio::pinocchio)


target_include_directories(ground_truth_estimator PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_include_directories(quad_wheel_ground_truth_estimator PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_include_directories(quad_controller PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_include_directories(quad_wheel_controller PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_include_directories(quad_sim_effort_controller PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_include_directories(spawn_lock PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(ground_truth_estimator
  rclcpp
  std_msgs
  quad_utils
  quad_msgs
  gz-sim8
  gz-plugin2
)

ament_target_dependencies(quad_wheel_ground_truth_estimator
  rclcpp
  std_msgs
  quad_utils
  quad_msgs
  gz-sim8
  gz-plugin2
)

ament_target_dependencies(spawn_lock
  gz-sim8
  gz-plugin2
)

ament_target_dependencies(quad_controller
  rclcpp
  controller_interface
  hardware_interface
  pluginlib
  angles
  realtime_tools
  urdf
  quad_msgs
  quad_utils
)

ament_target_dependencies(quad_wheel_controller
  rclcpp
  controller_interface
  hardware_interface
  pluginlib
  angles
  realtime_tools
  urdf
  quad_msgs
  quad_utils
)

ament_target_dependencies(quad_sim_effort_controller
  rclcpp
  gz-sim8
  gz-plugin2
  angles
  quad_msgs
)

pluginlib_export_plugin_description_file(controller_interface controller_plugin.xml)

install(TARGETS
  ground_truth_estimator
  quad_wheel_ground_truth_estimator
  quad_controller
  quad_wheel_controller
  quad_sim_effort_controller
  spawn_lock
  DESTINATION lib
)

install(
  DIRECTORY include/
  DESTINATION include
)



ament_export_include_directories(include)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  # set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  set(ament_cmake_uncrustify_FOUND TRUE)
  # ament_lint_auto_find_test_dependencies()
endif()

ament_package()

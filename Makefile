PY ?= /usr/bin/python3
ROS ?= /opt/ros/humble/setup.bash

.PHONY: build test train-curriculum train-mujoco indoor check record

build:
	cd ros2 && . $(ROS) && colcon build --symlink-install --cmake-args -DBUILD_TESTING=OFF

test:
	$(PY) -m pytest -q scripts/test_frontier_geometry.py tests

train-curriculum:
	$(PY) -u training/train_curriculum.py --asymmetric --obs-history 5

train-mujoco:
	$(PY) training/train_mujoco.py --timesteps 2000000

indoor:
	. $(ROS) && . ros2/install/setup.bash && ros2 launch launch/indoor_autonomy_go2.launch.py headless:=true rviz:=false

check:
	$(PY) scripts/check_stack_go2.py

record:
	./scripts/record_go2_session.sh

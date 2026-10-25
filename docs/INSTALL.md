# Install

- Ubuntu 22.04, ROS2 Humble, Gazebo Harmonic (`ros_gz_sim`), Python 3.8+
- NVIDIA GPU with 10GB+ VRAM for RL training

```bash
pip install -r requirements.txt
make build          # colcon build of ros2/
make test           # pure-Python tests
make train-curriculum
```

Use the system `python3` (`/usr/bin/python3`) for training; a venv that lacks
`stable_baselines3` will fail at import. See [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

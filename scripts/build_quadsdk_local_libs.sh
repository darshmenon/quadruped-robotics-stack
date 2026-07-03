#!/usr/bin/env bash
# Builds RBDL and IPOPT (Quad-SDK's NMPC solver deps) into a repo-local
# prefix, so no sudo/root install to /usr/local is needed. Run AFTER
# ./scripts/setup_quadsdk_apt_deps.sh has installed the system apt packages
# (gcc/g++/gfortran/liblapack-dev/libmetis-dev/urdf etc).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXT_DIR="$REPO_ROOT/ros2/quad_sdk_external"
LOCAL_DEPS="$EXT_DIR/local"
mkdir -p "$LOCAL_DEPS"

echo "### Building RBDL -> $LOCAL_DEPS"
cd "$EXT_DIR/rbdl-orb"
mkdir -p build && cd build
cmake -D CMAKE_POLICY_VERSION_MINIMUM=3.5 \
      -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX="$LOCAL_DEPS" \
      -D RBDL_BUILD_ADDON_URDFREADER=ON \
      ..
make -j"$(nproc)"
make install

echo "### Building IPOPT (via coinbrew) -> $LOCAL_DEPS"
mkdir -p "$EXT_DIR/ipopt_build" && cd "$EXT_DIR/ipopt_build"
if [ ! -f coinbrew ]; then
  wget -q https://raw.githubusercontent.com/coin-or/coinbrew/v2.0/coinbrew
  chmod u+x coinbrew
fi
./coinbrew fetch Ipopt --no-prompt
./coinbrew build Ipopt --latest-release --tests none --prefix="$LOCAL_DEPS" --no-prompt

echo "### Done. Point colcon at these libs with:"
echo "  cd ros2 && colcon build --symlink-install --cmake-args -DQUADSDK_DEPS_PREFIX=$LOCAL_DEPS"

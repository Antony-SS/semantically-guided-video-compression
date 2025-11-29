#!/bin/bash
set -e

echo "Setting up semantically-guided-video-compression..."

# Detect conda environment
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "Detected conda environment: $CONDA_DEFAULT_ENV"
fi

# Auto-detect and source ROS2
ROS_DISTRO=""
ROS_SETUP_PATH=""

# Check common ROS2 installation paths
for distro in humble iron jazzy rolling; do
    if [ -f "/opt/ros/${distro}/setup.bash" ]; then
        ROS_DISTRO="$distro"
        ROS_SETUP_PATH="/opt/ros/${distro}/setup.bash"
        break
    fi
done

# Source ROS2 if found
if [ -n "$ROS_SETUP_PATH" ] && [ -f "$ROS_SETUP_PATH" ]; then
    echo "Sourcing ROS2 setup: $ROS_SETUP_PATH"
    source "$ROS_SETUP_PATH"
    echo "ROS2 distribution: $ROS_DISTRO"
elif [ -z "$ROS_DISTRO" ]; then
    echo "Error: ROS2 not found. Please install ROS2 first."
    echo "Common distributions: humble, iron, jazzy, rolling"
    exit 1
fi

# Install ROS2 Python packages (if not already installed)
echo "Checking ROS2 Python packages..."
if ! python3 -c "import rosidl_runtime_py" 2>/dev/null; then
    echo "Installing ROS2 Python packages..."
    apt-get update && apt-get install -y \
        python3-rosidl-runtime-py \
        ros-${ROS_DISTRO}-cv-bridge \
        ros-${ROS_DISTRO}-sensor-msgs \
        ros-${ROS_DISTRO}-builtin-interfaces || true
fi

# Install perception-stack packages as editable dependencies
echo "Installing perception-stack packages..."
if [ ! -d "perception-stack" ]; then
    git clone https://github.com/Antony-SS/perception-stack.git
fi
python3 -m pip install -r requirements.txt

# Install main project and all dependencies
echo "Installing main project..."
python3 -m pip install -e .

echo "Installation complete!"
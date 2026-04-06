
if [ $# -eq 0 ]; then
    echo "Usage: source settings.sh <Version>"
    return
fi

version=$1
echo "Setting up Xilinx tools for version ${version}"
source /tools/xilinx/Vitis/${version}/.settings64-Vitis.sh
source /tools/xilinx/Vitis_HLS/${version}/.settings64-Vitis_HLS.sh
source /tools/xilinx/Vivado/${version}/.settings64-Vivado.sh
source /opt/xilinx/xrt/setup.sh

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libc.so.6:/usr/lib/x86_64-linux-gnu/libudev.so.1
export XILINXD_LICENSE_FILE=/home-ssd/roberto

# Must be run on gandalf
# sudo sshfs -o allow_other roberto@smaug:/home-ssd/datasets/ /tools/datasets/

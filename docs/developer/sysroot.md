# Generating A Board Sysroot

nn2FPGA uses a target sysroot when cross-compiling artifacts that must run on the FPGA board. The sysroot should match the board operating system, glibc version, XRT installation, and runtime library versions.

Using a sysroot generated from a different board image can produce build or runtime errors such as `GLIBC_2.xx not found`, missing XRT dependencies, or unresolved Boost/uuid symbols.

## Prerequisites

On the host machine:

```bash
sudo apt-get install rsync zstd
```

On the board:

```bash
sudo apt-get install rsync
```

You also need SSH access to the board and permissions to read `/lib`, `/usr/lib`, `/usr/include`, and the XRT installation paths.

## Create A Staging Directory

Run these commands from the nn2FPGA repository root on the host machine:

```bash
mkdir -p /tmp/nn2fpga-sysroot/sysroot-aarch64
```

The top-level directory must be named `sysroot-aarch64`, because the Dockerfile expects that name when extracting `docker/sysroot-aarch64.tar.zst`.

## Copy Runtime Libraries

Replace `<BOARD_IP>` with the board hostname or IP address.

```bash
rsync -a --copy-unsafe-links root@<BOARD_IP>:/lib/aarch64-linux-gnu/ \
  /tmp/nn2fpga-sysroot/sysroot-aarch64/lib/aarch64-linux-gnu/

rsync -a --copy-unsafe-links root@<BOARD_IP>:/usr/lib/aarch64-linux-gnu/ \
  /tmp/nn2fpga-sysroot/sysroot-aarch64/usr/lib/aarch64-linux-gnu/
```

Use `rsync -a` to preserve symlinks such as `libboost_filesystem.so -> libboost_filesystem.so.1.74.0`. These symlinks are needed by the linker.

## Copy Headers

```bash
rsync -a root@<BOARD_IP>:/usr/include/ \
  /tmp/nn2fpga-sysroot/sysroot-aarch64/usr/include/
```

## Copy XRT Files

XRT installation paths can vary between board images. First check where the headers and libraries are installed:

```bash
ssh root@<BOARD_IP> \
  "find /usr/include /usr/lib /opt -name 'xrt_bo.h' -o -name 'libxrt_coreutil.so*'"
```

For the common layout under `/usr/include/xrt` and `/usr/lib/aarch64-linux-gnu`, copy:

```bash
rsync -a root@<BOARD_IP>:/usr/include/xrt/ \
  /tmp/nn2fpga-sysroot/sysroot-aarch64/usr/include/xrt/

rsync -a root@<BOARD_IP>:/usr/lib/aarch64-linux-gnu/libxrt* \
  /tmp/nn2fpga-sysroot/sysroot-aarch64/usr/lib/aarch64-linux-gnu/
```

If your board installs XRT under a different path, copy the corresponding headers and libraries into the same sysroot locations used above.

## Verify Required Files

Check that the staged sysroot contains the libraries commonly needed by XRT and nn2FPGA builds:

```bash
ls -l /tmp/nn2fpga-sysroot/sysroot-aarch64/usr/lib/aarch64-linux-gnu/libxrt*
ls -l /tmp/nn2fpga-sysroot/sysroot-aarch64/usr/lib/aarch64-linux-gnu/libboost_filesystem.so*
ls -l /tmp/nn2fpga-sysroot/sysroot-aarch64/usr/lib/aarch64-linux-gnu/libboost_system.so*
ls -l /tmp/nn2fpga-sysroot/sysroot-aarch64/usr/lib/aarch64-linux-gnu/libuuid.so*
```

If any of these files are missing, verify that the corresponding packages are installed on the board and repeat the relevant `rsync` command.

## Create The Archive

From the nn2FPGA repository root:

```bash
tar --zstd -cf docker/sysroot-aarch64.tar.zst \
  -C /tmp/nn2fpga-sysroot sysroot-aarch64
```

## Validate The Archive

```bash
tar --zstd -tf docker/sysroot-aarch64.tar.zst | head
tar --zstd -tf docker/sysroot-aarch64.tar.zst | grep 'libxrt_coreutil'
```

The first command should show entries starting with:

```text
sysroot-aarch64/
```

## Use The New Sysroot

Rebuild the Docker image after replacing `docker/sysroot-aarch64.tar.zst`. During the image build, the archive is extracted to:

```text
/opt/sysroots/board
```

The container entrypoint sets:

```text
SYSROOT=/opt/sysroots/board
```

## Troubleshooting

`GLIBC_2.xx not found`

The binary was built against a newer sysroot than the board runtime. Regenerate the sysroot from the same board image where the binary will run.

`cannot find -lboost_filesystem` or `cannot find -lboost_system`

The Boost libraries or linker symlinks are missing from the sysroot. Ensure both the versioned libraries and unversioned symlinks were copied from `/usr/lib/aarch64-linux-gnu`.

`cannot find -luuid`

The uuid library or linker symlink is missing from the sysroot. Copy `libuuid.so*` from the board.

`libxrt_coreutil.so: undefined reference`

One or more transitive XRT dependencies are missing from the sysroot. Check `ldd` on the board for `libxrt_coreutil.so` and copy the missing libraries.

`tar: zstd: Cannot exec`

Install `zstd` on the host machine.

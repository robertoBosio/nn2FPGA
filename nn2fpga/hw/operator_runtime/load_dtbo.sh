#!/usr/bin/env bash
set -euo pipefail

# Defaults AXI-lite window for Vivado base accelerators:
BASE_HEX="0xA0000000"
SIZE_HEX="0x00010000"   # 64 KiB

DTBO_PATH=""
OVERLAY_NAME=""

usage() {
  cat <<EOF
Usage:
  sudo $0 [--dtbo /path/to/acc.dtbo]
          [--base 0xA0000000] [--size 0x10000]
          [--overlay-name nn2fpga_overlay]
          [--detach]

- Checks if the requested [base,size] region is already covered by any /sys/class/uio map.
- If not covered, loads the given DTBO via configfs overlays.
- Defaults: base=0xA0000000, size=0x10000 (typical Vivado window for AXI-lite).
EOF
}

# --- helpers ---
hex_to_dec() {
  local v="$1"
  printf "%d" "$v"
}

find_uio_maps() {
  # prints: "<addr_hex> <size_hex>" per map
  [[ -d /sys/class/uio ]] || return 0
  for a in /sys/class/uio/uio*/maps/map*/addr; do
    [[ -e "$a" ]] || continue
    local s="${a%/addr}/size"
    echo "$(cat "$a") $(cat "$s")"
  done
}

covered_by_any_uio() {
  local base_hex="$1" size_hex="$2"
  local base=$(hex_to_dec "$base_hex")
  local size=$(hex_to_dec "$size_hex")
  local end=$((base + size))
  while read -r a_hex s_hex; do
    [[ -n "$a_hex" && -n "$s_hex" ]] || continue
    local a=$(hex_to_dec "$a_hex")
    local s=$(hex_to_dec "$s_hex")
    local ae=$((a + s))
    if (( a <= base && ae >= end )); then
      return 0
    fi
  done < <(find_uio_maps)
  return 1
}

detach_overlay() {
  local dir="/sys/kernel/config/device-tree/overlays/${OVERLAY_NAME}"
  if [[ -d "$dir" ]]; then
    sudo rmdir "$dir" || { echo "ERROR: failed to remove overlay (busy?)"; exit 1; }
    echo "Overlay ${OVERLAY_NAME} removed."
  else
    echo "Overlay ${OVERLAY_NAME} not present."
  fi
  exit 0
}

apply_dtbo() {
  local dtbo="$1"
  [[ -f "$dtbo" ]] || { echo "ERROR: DTBO not found: $dtbo"; exit 1; }
  mount | grep -q "configfs on /sys/kernel/config" || sudo mount -t configfs none /sys/kernel/config
  local cfg="/sys/kernel/config/device-tree/overlays/${OVERLAY_NAME}"
  sudo mkdir -p "$cfg"
  sudo sh -c "cat '$dtbo' > '$cfg/dtbo'"
  # brief settle
  sleep 0.2
}

# --- args ---
while (( $# )); do
  case "$1" in
    --dtbo) DTBO_PATH="$2"; shift;;
    --base) BASE_HEX="$2"; shift;;
    --size) SIZE_HEX="$2"; shift;;
    --overlay-name) OVERLAY_NAME="$2"; shift;;
    --detach) detach_overlay;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac; shift || true
done

echo "Checking UIO coverage for base=$BASE_HEX size=$SIZE_HEX …"
if covered_by_any_uio "$BASE_HEX" "$SIZE_HEX"; then
  echo "OK: requested region is already covered by UIO. Nothing to do."
  exit 0
fi

[[ -n "$DTBO_PATH" ]] || { echo "Region not covered and no --dtbo provided. Pass the .dtbo path."; exit 2; }

echo "Region not covered. Loading DTBO: $DTBO_PATH (overlay name: $OVERLAY_NAME)…"
apply_dtbo "$DTBO_PATH"

echo "Verifying after load…"
if covered_by_any_uio "$BASE_HEX" "$SIZE_HEX"; then
  echo "Success: UIO now covers base=$BASE_HEX size=$SIZE_HEX."
  # Optional: list maps
  for f in /sys/class/uio/uio*/maps/map*/addr; do
    s="${f%/addr}/size"; echo -n "$f -> "; cat "$f"; echo -n "           size="; cat "$s"
  done
  exit 0
else
  echo "ERROR: After applying DTBO, region is still not covered."
  exit 3
fi

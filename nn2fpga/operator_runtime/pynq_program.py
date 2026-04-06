#!/usr/bin/env python3
import sys
from pynq import Overlay, PL

if len(sys.argv) != 2:
    print("usage: pynq_program.py <bit_or_xclbin>", file=sys.stderr)
    sys.exit(2)

path = sys.argv[1]
PL.reset()
# Overlay() programs via FPGA Manager on ZynqMP and sets AXI widths/metadata
Overlay(path)
print(f"Overlay loaded: {path}")
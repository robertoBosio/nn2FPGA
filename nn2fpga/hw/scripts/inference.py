import pynq
from pynq import Overlay
from pynq import allocate
from pynq import Clocks
from pynq import PL

import sys
import os
import numpy as np

from boards import hw_inference, ZCU102PowerSensor
from datasets import cifar10_dataloader, coco_dataloader, vw_dataloader, imagenet_dataloader, cifar10_4bit_dataloader
from coco import postprocess as coco_postprocess
from cifar10 import postprocess as cifar10_postprocess
from vw import postprocess as vw_postprocess
from imagenet import postprocess as imagenet_postprocess

def print_sorted_nested_dict(dictionary, indent=0):
    for key in sorted(dictionary.keys()):
        value = dictionary[key]
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            print_sorted_nested_dict(value, indent + 1)
        else:
            print("  " * indent + f"{key}: {value}")


if __name__ == "__main__":
    # Uploading all data

    PL.reset()
    supported_boards = ["KRIA", "ULTRA96v2", "ZCU102"]
    supported_datasets = ["cifar10", "vw", "coco", "imagenet"]

    if (len(sys.argv) < 4):
        print(
            "Wrong number of arguments: inference.py board dataset input_scale_factor frequency\n"
        )
        sys.exit(0)

    sel_board = sys.argv[1]
    if (sel_board not in supported_boards):
        print(f"Error: selected {sel_board}, allowed board selection: {supported_boards}")
        sys.exit(0)

    sel_dataset = sys.argv[2]
    if (sel_dataset not in supported_datasets):
        print(f"Error: selected {sel_dataset}, allowed dataset selection: {supported_datasets}")
        sys.exit(0)

    # Check that the input scale factor must be a positive integer number
    try:
        scale_factor = int(sys.argv[3])
        if scale_factor <= 0:
            raise ValueError
    except ValueError:
        print("Error: input scale factor must be a positive integer number")
        sys.exit(0)
    
    # Check that the frequency must be a positive integer number
    try:
        frequency = int(sys.argv[4])
        if frequency <= 0:
            raise ValueError
    except ValueError:
        print("Error: frequency (MHz) must be a positive integer number")
        sys.exit(0)

    batch_size = 400

    if (sel_dataset == "cifar10"):
        dataloader = cifar10_dataloader
        postprocess = cifar10_postprocess
    elif (sel_dataset == "cifar10_4bit"):
        dataloader = cifar10_4bit_dataloader
        postprocess = cifar10_postprocess
    elif (sel_dataset == "vw"):
        dataloader = vw_dataloader
        postprocess = vw_postprocess
    elif (sel_dataset == "coco"):
        dataloader = coco_dataloader
        postprocess = coco_postprocess
    elif (sel_dataset == "imagenet"):
        dataloader = imagenet_dataloader
        postprocess = imagenet_postprocess

    test_loader, buffer_dim = dataloader(batch_size)
    
    # Board aware section
    board = {}
    pl_divisor = 10
    if (sel_board == "ULTRA96v2"):
        rails = pynq.get_rails()
        board["sensor"] = rails["INT"].power
        board["sensor_name"] = "INT_power"
        pl_divisor = 1500 // frequency

    if (sel_board == "KRIA"):
        rails = pynq.get_rails()
        board["sensor"] = rails["power1"].power
        board["sensor_name"] = "power1_power"
        pl_divisor = 1000 // frequency
    
    if (sel_board == "ZCU102"):
        board["sensor"] = ZCU102PowerSensor("INT_power", "W")
        board["sensor_name"] = "INT_power"
        pl_divisor = 1500 // frequency

    print("Loading overlay", flush=True)
    overlay = Overlay('./overlay/design_1.bit')
    
    print("Loaded overlay", flush=True)
    print(f"Setting PL clock with divisor: {pl_divisor}", flush=True)
    Clocks._instance.PL_CLK_CTRLS[0].DIVISOR0 = pl_divisor
    
    print("Loading params", flush=True)
    dma_params = overlay.axi_dma_1
    params_vector = np.load("overlay/uram.npy")
    params_buffer = allocate(shape=(params_vector.shape[0], ), dtype=np.int8)
    params_buffer[:] = params_vector[:]
    dma_params.sendchannel.transfer(params_buffer)
    dma_params.sendchannel.wait()
    print("Loaded params", flush=True)

    dma = overlay.axi_dma_0
    in_buffer = allocate(shape=(batch_size*buffer_dim[0], ), dtype=np.int8)
    out_buffer = allocate(shape=(batch_size, buffer_dim[1], ), dtype=np.int8)

    hw_inference(
        test_loader,
        in_buffer,
        out_buffer,
        dma,
        batch_size,
        scale_factor,
        postprocess,
        board,
    )

    del in_buffer, out_buffer, params_buffer

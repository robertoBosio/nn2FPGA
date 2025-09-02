import os
import subprocess
import numpy as np
import onnxruntime as ort
import csnake
from onnx import TensorProto, helper

PROJECT_NAME = "proj_unit_test"
FILE_DIR = f"/workspace/NN2FPGA/nn2fpga/library"
FILENAME = "StreamingConv"

def generate_config_file(config_dict):

    IN_HEIGHT = (
        (config_dict["OUT_HEIGHT"] - 1) * config_dict["STRIDE_H"]
        + config_dict["DIL_H"] * (config_dict["FH"] - 1)
        + 1
        - config_dict["PAD_T"]
        - config_dict["PAD_B"]
    )
    IN_WIDTH = (
        (config_dict["OUT_WIDTH"] - 1) * config_dict["STRIDE_W"]
        + config_dict["DIL_W"] * (config_dict["FW"] - 1)
        + 1
        - config_dict["PAD_L"]
        - config_dict["PAD_R"]
    )

    input_tensor = np.random.randint(
        -128, 127, size=(1, config_dict["IN_CH"], IN_HEIGHT, IN_WIDTH), dtype=np.int8
    )
    weight_tensor = np.random.randint(
        -128,
        127,
        size=(
            config_dict["OUT_CH"],
            config_dict["IN_CH"],
            config_dict["FH"],
            config_dict["FW"],
        ),
        dtype=np.int8,
    )
    bias_tensor = np.random.randint(
        -1000, 1000, size=(config_dict["OUT_CH"],), dtype=np.int32
    )

    X = helper.make_tensor_value_info(
        "X", TensorProto.INT8, [1, config_dict["IN_CH"], IN_HEIGHT, IN_WIDTH]
    )
    Y = helper.make_tensor_value_info(
        "Y",
        TensorProto.INT8,
        [1, config_dict["OUT_CH"], config_dict["OUT_HEIGHT"], config_dict["OUT_WIDTH"]],
    )

    W = helper.make_tensor(
        "W",
        TensorProto.INT8,
        [config_dict["OUT_CH"], config_dict["IN_CH"], config_dict["FH"], config_dict["FW"]],
        weight_tensor.flatten().tolist(),
    )
    B = helper.make_tensor(
        "B",
        TensorProto.INT32,
        [config_dict["OUT_CH"]],
        bias_tensor.flatten().tolist(),
    )

    # Scales and zero points
    X_scale  = helper.make_tensor("X_scale",  TensorProto.FLOAT, [], [config_dict["X_SCALE"]])
    X_zp     = helper.make_tensor("X_zp",     TensorProto.INT8, [], [config_dict["X_ZP"]])
    W_scale  = helper.make_tensor("W_scale",  TensorProto.FLOAT, [], [config_dict["W_SCALE"]])
    W_zp     = helper.make_tensor("W_zp",     TensorProto.INT8, [], [config_dict["W_ZP"]])
    Y_scale  = helper.make_tensor("Y_scale",  TensorProto.FLOAT, [], [config_dict["Y_SCALE"]])
    Y_zp     = helper.make_tensor("Y_zp",     TensorProto.INT8, [], [config_dict["Y_ZP"]])

    # QLinearConv node
    qconv = helper.make_node(
        "QLinearConv",
        inputs=["X", "X_scale", "X_zp", "W", "W_scale", "W_zp", "Y_scale", "Y_zp", "B"],
        outputs=["Y"],
        strides=[config_dict["STRIDE_H"], config_dict["STRIDE_W"]],
        pads=[config_dict["PAD_T"], config_dict["PAD_B"], config_dict["PAD_L"], config_dict["PAD_R"]],
        dilations=[config_dict["DIL_H"], config_dict["DIL_W"]],
        group=config_dict["GROUP"],
        kernel_shape=[config_dict["FH"], config_dict["FW"]],
    )

    graph = helper.make_graph(
        [qconv],
        "qconv_test",
        [X],
        [Y],
        initializer=[W, B, X_scale, X_zp, W_scale, W_zp, Y_scale, Y_zp],
    )

    model = helper.make_model(graph, producer_name="qonnx")
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])

    # Execute model
    outputs = sess.run(None, {"X": input_tensor})
    y = outputs[0]

    # Dump the tensors in a hpp file
    cwr = csnake.CodeWriter()
    cwr.include("<cstdint>")
    cwr.include("<array>")
    cwr.include("<ap_int.h>")
    cwr.add_line("namespace test_config {")
    cwr.indent()
    for key, value in config_dict.items():
        if key in ["X_SCALE", "W_SCALE", "Y_SCALE"]:
            cwr.add_line(f"const float {key} = {value}f;")
        else:   
            cwr.add_line(f"const int {key} = {value};")
    cwr.add_line(f"typedef DequantQuantPo2<5, {config_dict['ACC_DATAWIDTH']}, {config_dict['OUTPUT_DATAWIDTH']}> Quantizer;")
    cwr.add_line(f"typedef ap_int<{config_dict['INPUT_DATAWIDTH']}> TInput;")
    cwr.add_line(f"typedef ap_int<{config_dict['WEIGHT_DATAWIDTH']}> TWeight;")
    cwr.add_line(f"typedef ap_int<{config_dict['BIAS_DATAWIDTH']}> TBias;")
    cwr.add_line(f"typedef ap_int<{config_dict['OUTPUT_DATAWIDTH']}> TOutput;")
    cwr.add_line(f"typedef ap_int<{config_dict['ACC_DATAWIDTH']}> TAcc;")
    input_tensor_variable = csnake.Variable("input_tensor", primitive=f"ap_int<{config_dict['INPUT_DATAWIDTH']}>", value=input_tensor)
    weight_tensor_variable = csnake.Variable("weight_tensor", primitive=f"ap_int<{config_dict['WEIGHT_DATAWIDTH']}>", value=weight_tensor)
    bias_tensor_variable = csnake.Variable("bias_tensor", primitive=f"ap_int<{config_dict['BIAS_DATAWIDTH']}>", value=bias_tensor)
    output_tensor_variable = csnake.Variable("output_tensor", primitive=f"ap_int<{config_dict['OUTPUT_DATAWIDTH']}>", value=y)
    cwr.add_lines(input_tensor_variable.generate_initialization())
    cwr.add_lines(weight_tensor_variable.generate_initialization())
    cwr.add_lines(bias_tensor_variable.generate_initialization())
    cwr.add_lines(output_tensor_variable.generate_initialization())
    cwr.dedent()
    cwr.add_line("}")
    return cwr.code

def generate_hls_script(steps: str) -> str:
    steps = [step.strip() for step in steps.split(",")]
    do_csim   = "csim"   in steps
    do_csynth = "csynth" in steps
    do_cosim  = "cosim"  in steps

    if not (do_csim or do_csynth or do_cosim):
        do_csim = True  # default to csim if no valid step is provided

    if do_cosim and not do_csynth:
        do_csynth = True  # csynth is required for cosim

    print(f"Running HLS steps: {', '.join(steps)}")
    lines = [
        f'open_project -reset "{PROJECT_NAME}"',
        'open_solution -reset solution0',
        f'add_files {FILE_DIR}/include/{FILENAME}.hpp -cflags "-I/workspace/NN2FPGA/nn2fpga/library/include"',
        f'add_files {FILE_DIR}/test/Unit{FILENAME}.cpp -cflags "-I/workspace/NN2FPGA/nn2fpga/library/include -I/workspace/NN2FPGA/nn2fpga/"',
        f'add_files -tb {FILE_DIR}/test/Unit{FILENAME}.cpp -cflags "-I/workspace/NN2FPGA/nn2fpga/library/include -I/workspace/NN2FPGA/nn2fpga/"',
        "set_top wrap_run",
        "set_part xck26-sfvc784-2LV-c",
        "create_clock -period 3.33ns",
        "config_compile -pipeline_style frp",
        "",
    ]
    if do_csim:
        lines.append("csim_design -argv csim")
    if do_csynth:
        lines.append("csynth_design")
    if do_cosim:
        lines.append("cosim_design -argv cosim")
    lines.append("exit")
    return "\n".join(lines)

def runhls(tcl_file):
    # Write the Tcl script to a temporary file

    return subprocess.run(
        ["vitis_hls", "-f", tcl_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

def run(config_dict, steps: str):
    testconfig_file = "test_config.hpp"
    with open(testconfig_file, "w") as f:
        f.write(generate_config_file(config_dict))

    tcl_file = "script.tcl"
    with open(tcl_file, 'w') as f:
        f.write(generate_hls_script(steps))

    result = runhls(tcl_file)
    assert result.returncode == 0, f"Simulation failed: {result.stderr}"
    if "csim" in steps or "cosim" in steps:
        assert "passed" in result.stdout.lower(), f"Test did not pass: {result.stdout}"
    if "csynth" in steps:
        assert "All loop constraints were satisfied" in result.stdout, f"Loop constraints not satisfied: {result.stdout}"

    # Clean up the temporary Tcl file
    os.remove(tcl_file)
    os.remove(testconfig_file)

    # Clean up the project directory
    if os.path.exists(PROJECT_NAME):
        os.system(f"rm -rf {PROJECT_NAME}")

    # Clean up vitis_hls log files
    log_files = [f for f in os.listdir('.') if f.startswith('vitis_hls') and f.endswith('.log')]
    for log_file in log_files:
        os.remove(log_file)

def test_pointwise_pertensor_po2(request):
    steps = request.config.getoption("--hls-steps")

    # set a fixed random seed for reproducibility
    np.random.seed(42)

    config_dict = {
        "ACC_DATAWIDTH": 32,
        "INPUT_DATAWIDTH": 8,
        "WEIGHT_DATAWIDTH": 8,
        "BIAS_DATAWIDTH": 32,
        "OUTPUT_DATAWIDTH": 8,
        "OUT_HEIGHT": 4,
        "OUT_WIDTH": 4,
        "IN_CH": 4,
        "OUT_CH": 4,
        "FH": 1,
        "FW": 1,
        "STRIDE_H": 1,
        "STRIDE_W": 1,
        "PAD_T": 0,
        "PAD_B": 0,
        "PAD_L": 0,
        "PAD_R": 0,
        "DIL_H": 1,
        "DIL_W": 1,
        "IN_CH_PAR": 2,
        "OUT_CH_PAR": 2,
        "W_PAR": 1,
        "GROUP": 1,
        "X_SCALE": 2**-5,
        "W_SCALE": 2**-5,
        "Y_SCALE": 2**-5,
        "X_ZP": 0,
        "W_ZP": 0,
        "Y_ZP": 0,
        "PIPELINE_DEPTH": 5
    }

    run(config_dict, steps)

def test_pointwise_pertensor_po2_stride(request):
    steps = request.config.getoption("--hls-steps")

    # set a fixed random seed for reproducibility
    np.random.seed(42)

    config_dict = {
        "ACC_DATAWIDTH": 32,
        "INPUT_DATAWIDTH": 8,
        "WEIGHT_DATAWIDTH": 8,
        "BIAS_DATAWIDTH": 32,
        "OUTPUT_DATAWIDTH": 8,
        "OUT_HEIGHT": 2,
        "OUT_WIDTH": 2,
        "IN_CH": 4,
        "OUT_CH": 4,
        "FH": 1,
        "FW": 1,
        "STRIDE_H": 2,
        "STRIDE_W": 2,
        "PAD_T": 0,
        "PAD_B": 0,
        "PAD_L": 0,
        "PAD_R": 0,
        "DIL_H": 1,
        "DIL_W": 1,
        "IN_CH_PAR": 2,
        "OUT_CH_PAR": 2,
        "W_PAR": 2,
        "GROUP": 1,
        "X_SCALE": 2**-5,
        "W_SCALE": 2**-5,
        "Y_SCALE": 2**-5,
        "X_ZP": 0,
        "W_ZP": 0,
        "Y_ZP": 0,
        "PIPELINE_DEPTH": 3
    }

    run(config_dict, steps)

def test_3x3_pertensor_po2(request):
    steps = request.config.getoption("--hls-steps")

    # set a fixed random seed for reproducibility
    np.random.seed(42)

    config_dict = {
        "ACC_DATAWIDTH": 32,
        "INPUT_DATAWIDTH": 8,
        "WEIGHT_DATAWIDTH": 8,
        "BIAS_DATAWIDTH": 32,
        "OUTPUT_DATAWIDTH": 8,
        "OUT_HEIGHT": 4,
        "OUT_WIDTH": 4,
        "IN_CH": 4,
        "OUT_CH": 4,
        "FH": 3,
        "FW": 3,
        "STRIDE_H": 1,
        "STRIDE_W": 1,
        "PAD_T": 1,
        "PAD_B": 1,
        "PAD_L": 1,
        "PAD_R": 1,
        "DIL_H": 1,
        "DIL_W": 1,
        "IN_CH_PAR": 2,
        "OUT_CH_PAR": 2,
        "W_PAR": 1,
        "GROUP": 1,
        "X_SCALE": 2**-5,
        "W_SCALE": 2**-5,
        "Y_SCALE": 2**-5,
        "X_ZP": 0,
        "W_ZP": 0,
        "Y_ZP": 0,
        "PIPELINE_DEPTH": 1
    }

    run(config_dict, steps)

def test_1x5_pertensor_po2(request):
    steps = request.config.getoption("--hls-steps")

    # set a fixed random seed for reproducibility
    np.random.seed(42)

    config_dict = {
        "ACC_DATAWIDTH": 32,
        "INPUT_DATAWIDTH": 8,
        "WEIGHT_DATAWIDTH": 8,
        "BIAS_DATAWIDTH": 32,
        "OUTPUT_DATAWIDTH": 8,
        "OUT_HEIGHT": 2,
        "OUT_WIDTH": 2,
        "IN_CH": 4,
        "OUT_CH": 4,
        "FH": 1,
        "FW": 5,
        "STRIDE_H": 1,
        "STRIDE_W": 1,
        "PAD_T": 0,
        "PAD_B": 0,
        "PAD_L": 0,
        "PAD_R": 0,
        "DIL_H": 1,
        "DIL_W": 1,
        "IN_CH_PAR": 2,
        "OUT_CH_PAR": 2,
        "W_PAR": 2,
        "GROUP": 1,
        "X_SCALE": 2**-5,
        "W_SCALE": 2**-5,
        "Y_SCALE": 2**-5,
        "X_ZP": 0,
        "W_ZP": 0,
        "Y_ZP": 0,
        "PIPELINE_DEPTH": 2
    }

    run(config_dict, steps)

def test_fullyconnected_pertensor_po2(request):
    steps = request.config.getoption("--hls-steps")

    # set a fixed random seed for reproducibility
    np.random.seed(42)

    config_dict = {
        "ACC_DATAWIDTH": 32,
        "INPUT_DATAWIDTH": 8,
        "WEIGHT_DATAWIDTH": 8,
        "BIAS_DATAWIDTH": 32,
        "OUTPUT_DATAWIDTH": 8,
        "OUT_HEIGHT": 1,
        "OUT_WIDTH": 1,
        "IN_CH": 4,
        "OUT_CH": 4,
        "FH": 1,
        "FW": 1,
        "STRIDE_H": 1,
        "STRIDE_W": 1,
        "PAD_T": 0,
        "PAD_B": 0,
        "PAD_L": 0,
        "PAD_R": 0,
        "DIL_H": 1,
        "DIL_W": 1,
        "IN_CH_PAR": 4,
        "OUT_CH_PAR": 2,
        "W_PAR": 1,
        "GROUP": 1,
        "X_SCALE": 2**-5,
        "W_SCALE": 2**-5,
        "Y_SCALE": 2**-5,
        "X_ZP": 0,
        "W_ZP": 0,
        "Y_ZP": 0,
        "PIPELINE_DEPTH": 3
    }

    run(config_dict, steps)
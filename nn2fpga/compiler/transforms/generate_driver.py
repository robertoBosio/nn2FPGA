import os
import shutil
from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from nn2fpga.compiler.core.acceleratorpackage import AcceleratorPackage
from nn2fpga.compiler.transforms.convert_to_QCDQ import ConvertToQCDQ
from nn2fpga.compiler.transforms.set_dynamic_batchsize import SetDynamicBatchSize
from nn2fpga.compiler.utils.codegen_utils import NewCodeWriter
from nn2fpga.compiler.utils.board_util import read_board_info
from nn2fpga.compiler.core.tensor_quant import TensorQuant
from onnx import NodeProto
import numpy as np

def get_onnxruntime_dtype(tensor_quant: TensorQuant) -> str:
    """ Get the ONNX Runtime data type for a given tensor quantization. """
    if tensor_quant.signed:
        if tensor_quant.bitwidth <= 8:
            return "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8"
        elif tensor_quant.bitwidth <= 16:
            return "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16"
        elif tensor_quant.bitwidth <= 32:
            return "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32"
    else:
        if tensor_quant.bitwidth <= 8:
            return "ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8"
        elif tensor_quant.bitwidth <= 16:
            return "ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16"
        elif tensor_quant.bitwidth <= 32:
            return "ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32"
    raise ValueError(f"Unsupported bitwidth: {tensor_quant.bitwidth}")

def get_spec_dtype(tensor_quant: TensorQuant) -> str:
    """ Get the data type string for a given tensor quantization, suitable for the spec file. """
    if tensor_quant.signed:
        if tensor_quant.bitwidth <= 8:
            return "i8"
        elif tensor_quant.bitwidth <= 16:
            return "i16"
        elif tensor_quant.bitwidth <= 32:
            return "i32"
    else:
        if tensor_quant.bitwidth <= 8:
            return "u8"
        elif tensor_quant.bitwidth <= 16:
            return "u16"
        elif tensor_quant.bitwidth <= 32:
            return "u32"
    raise ValueError(f"Unsupported bitwidth: {tensor_quant.bitwidth}")

def generate_spec(
    model: ModelWrapper,
    nn2FPGA_node: NodeProto,
    deploy_dir: str,
    Nmax: int,
    Pll_index: int,
    Pll_frequency: int,
    frequency: int,
    axilite_base_addr: int,
    axilite_size: int,
    design_id: str,
) -> None:

    ap = AcceleratorPackage.from_json(
        getCustomOp(nn2FPGA_node).get_nodeattr("accelerator_package")
    )

    cwr = NewCodeWriter()
    cwr.add_autogen_comment()

    cwr.add_line("#pragma once")
    cwr.include("nn2FPGA_spec.hpp")
    cwr.include("<onnxruntime_cxx_api.h>")

    cwr.add_line("struct OpSpec {")
    cwr.indent()
    cwr.add_line('static constexpr const char *kOpName = "nn2fpgaPartition";')
    cwr.add_line('static constexpr const char *kDomain = "ai.nn2FPGA";')
    cwr.add_line("static constexpr int kOpVersion = 1;")
    cwr.add_line(f"static constexpr int N_MAX = {Nmax};")
    cwr.add_line(f"static constexpr int PllIndex = {Pll_index};")
    cwr.add_line(f"static constexpr int Freq_MHz = {frequency};")
    cwr.add_line(f"static constexpr int PLLFreq_MHz = {Pll_frequency};")
    cwr.add_line(f'static constexpr uint32_t DesignID = {int(design_id)};')
    cwr.add_line(f"static constexpr uint64_t AXIL_BASE = 0x{axilite_base_addr:X};")
    cwr.add_line(f"static constexpr size_t AXIL_SIZE = 0x{axilite_size:X};")

    cwr.add_line(f"static inline const std::array<PortDesc, {len(ap.input_map)}> Inputs{{{{")
    cwr.indent()
    for name, value in sorted(ap.input_map.items(), key=lambda x: x[1]['index']):
        tensor_shape = value["shape"]
        tensor_shape_nobatch = tensor_shape[1:]  # Exclude batch size
        str_tensor_shape = ', '.join(map(str, tensor_shape_nobatch))
        quant = TensorQuant.from_canonical_name(value["quant"])
        mode = "PortMode::StaticInit" if value['value'] is not None else "PortMode::Dynamic"
        buffer_size = np.dtype(quant.get_numpy_dtype()).itemsize * np.prod(tensor_shape_nobatch)
        if value['value'] is None:
            buffer_size *= Nmax
        cwr.add_line(
            f"PortDesc{{DType::{get_spec_dtype(quant)}, {{{str_tensor_shape}}}, 0x{value['axi_offset']:X}, {mode}, {buffer_size}}}, // {name}"
        )
    cwr.dedent()
    cwr.add_line("}};")

    cwr.add_line(f"static inline const std::array<PortDesc, {len(ap.output_map)}> Outputs{{{{")
    cwr.indent()
    for name, value in sorted(ap.output_map.items(), key=lambda x: x[1]['index']):
        tensor_shape = value["shape"]
        tensor_shape_nobatch = tensor_shape[1:]  # Exclude batch size
        str_tensor_shape = ', '.join(map(str, tensor_shape_nobatch))
        quant = TensorQuant.from_canonical_name(value["quant"])
        mode = "PortMode::StaticInit" if value['value'] is not None else "PortMode::Dynamic"
        buffer_size = np.dtype(quant.get_numpy_dtype()).itemsize * np.prod(tensor_shape_nobatch)
        if value['value'] is None:
            buffer_size *= Nmax
        cwr.add_line(
            f"PortDesc{{DType::{get_spec_dtype(quant)}, {{{str_tensor_shape}}}, 0x{value['axi_offset']:X}, {mode}, {buffer_size}}}, // {name}"
        )
    cwr.dedent()
    cwr.add_line("}};")

    cwr.add_line("static inline const std::array<ONNXTensorElementDataType, "
                 f"{len(ap.input_map)}> OrtInputTypes{{{{")
    cwr.indent()
    for name in ap.input_map:
        quant = TensorQuant.from_canonical_name(ap.input_map[name]["quant"])
        cwr.add_line(f"{get_onnxruntime_dtype(quant)}, // {name}")
    cwr.dedent()
    cwr.add_line("}};")

    cwr.add_line("static inline const std::array<ONNXTensorElementDataType, "
                 f"{len(ap.output_map)}> OrtOutputTypes{{{{")
    cwr.indent()
    for name in ap.output_map:
        quant = TensorQuant.from_canonical_name(ap.output_map[name]["quant"])
        cwr.add_line(f"{get_onnxruntime_dtype(quant)}, // {name}")
    cwr.dedent()
    cwr.add_line("}};")

    cwr.dedent()
    cwr.add_line("};")

    return cwr.code 


def generate_pynq_test(nn2FPGA_node: NodeProto) -> str:
    # Generate a test script to only test the nn2FPGA custom operator on PYNQ.
    # This can be used to validate the operator and the bitstream independently of the full ONNX Runtime integration.
    # The test generates random input data, streams the data and checks the throughput.
    ap = AcceleratorPackage.from_json(
        getCustomOp(nn2FPGA_node).get_nodeattr("accelerator_package")
    )

    test_code = """
import numpy as np
from pynq import Overlay
from pynq import allocate
from pynq import PL
import time

PL.reset()
ol = Overlay("Overlay/design.bit")
"""
    # Xilinx DMA has a maximum transfer size of 64MB, so we need to choose a batch size that fits 
    # within this limit based on the input and output buffer sizes defined in the accelerator package.
    max_batch_size = 1024
    for name, value in sorted(ap.input_map.items(), key=lambda x: x[1]['index']):
        if value['value'] is not None:
            continue
        tensor_shape = value["shape"]
        tensor_shape_nobatch = tensor_shape[1:]  # Exclude batch size
        quant = TensorQuant.from_canonical_name(value["quant"])
        np_dtype = quant.get_numpy_dtype()
        buffer_size = np.dtype(np_dtype).itemsize * np.prod(tensor_shape_nobatch)
        max_batch_size = min(max_batch_size, 64 * 1024 * 1024 // buffer_size)

    for name, value in sorted(ap.output_map.items(), key=lambda x: x[1]['index']):
        if value['value'] is not None:
            continue
        tensor_shape = value["shape"]
        tensor_shape_nobatch = tensor_shape[1:]  # Exclude batch size
        quant = TensorQuant.from_canonical_name(value["quant"])
        np_dtype = quant.get_numpy_dtype()
        buffer_size = np.dtype(np_dtype).itemsize * np.prod(tensor_shape_nobatch)
        max_batch_size = min(max_batch_size, 64 * 1024 * 1024 // buffer_size)

    test_code += f"BATCH = {max_batch_size}\n"

    # Generate code to allocate input and output buffers based on the accelerator package specification.
    for name, value in sorted(ap.input_map.items(), key=lambda x: x[1]['index']):
        tensor_shape = value["shape"]
        dma_name = value['new_name'] 
        tensor_shape_nobatch = tensor_shape[1:]  # Exclude batch size
        str_tensor_shape = ', '.join(map(str, tensor_shape_nobatch))
        quant = TensorQuant.from_canonical_name(value["quant"])
        np_dtype = quant.get_numpy_dtype()
        np_dtype_info = np.iinfo(np_dtype)
        buffer_size = np.dtype(np_dtype).itemsize * np.prod(tensor_shape_nobatch)
        if value['value'] is None:
            str_tensor_shape = f"BATCH, {str_tensor_shape}"
        else:
            str_tensor_shape = f"{str_tensor_shape},"
        test_code += f"{dma_name}_buffer = allocate(shape=({str_tensor_shape}), dtype=\"{np_dtype.__name__}\")\n"
        test_code += f"{dma_name}_data = np.random.randint({np_dtype_info.min}, {np_dtype_info.max}, size=({str_tensor_shape}), dtype=\"{np_dtype.__name__}\")\n"

    for name, value in sorted(ap.output_map.items(), key=lambda x: x[1]['index']):
        tensor_shape = value["shape"]
        dma_name = value['new_name']
        tensor_shape_nobatch = tensor_shape[1:]  # Exclude batch size
        str_tensor_shape = ', '.join(map(str, tensor_shape_nobatch))
        quant = TensorQuant.from_canonical_name(value["quant"])
        np_dtype = quant.get_numpy_dtype()
        buffer_size = np.dtype(np_dtype).itemsize * np.prod(tensor_shape_nobatch)
        str_tensor_shape = f"BATCH, {str_tensor_shape}"
        test_code += f"{dma_name}_buffer = allocate(shape=({str_tensor_shape}), dtype=\"{np_dtype.__name__}\")\n"
        test_code += f"{dma_name}_data = np.zeros(({str_tensor_shape}), dtype=\"{np_dtype.__name__}\")\n"
        test_code += f"ol.{dma_name}_dma.recvchannel._max_size = {buffer_size}\n"
        test_code += f"ol.{dma_name}_dma.recvchannel._align = 1\n"

    # Load the static inputs
    for name, value in sorted(ap.input_map.items(), key=lambda x: x[1]['index']):
        if value['value'] is not None:
            dma_name = value['new_name']
            test_code += f"{dma_name}_buffer[:] = {dma_name}_data[:]\n"
            test_code += f"ol.{dma_name}_dma.sendchannel.transfer({dma_name}_buffer)\n"
            test_code += f"ol.{dma_name}_dma.sendchannel.wait()\n"
            test_code += f"print('Static input {dma_name} loaded')\n"

    test_code += """
batch_lat_s = []
img_lat_s = []
total_images = 0
total_time_s = 0.0
for batch_idx in range(10):  # Run 10 batches for testing\n"""

    for name, value in sorted(ap.input_map.items(), key=lambda x: x[1]['index']):
        if value['value'] is not None:
            continue
        dma_name = value['new_name']
        test_code += f"""    {dma_name}_buffer[:] = {dma_name}_data[:]\n"""
    test_code += """    start_time = time.perf_counter()\n"""
    for name, value in sorted(ap.output_map.items(), key=lambda x: x[1]['index']):
        dma_name = value['new_name']
        test_code += f"    ol.{dma_name}_dma.recvchannel.transfer({dma_name}_buffer)\n"
    for name, value in sorted(ap.input_map.items(), key=lambda x: x[1]['index']):
        if value['value'] is not None:
            continue
        dma_name = value['new_name']
        test_code += f"    ol.{dma_name}_dma.sendchannel.transfer({dma_name}_buffer)\n"
    for name, value in sorted(ap.output_map.items(), key=lambda x: x[1]['index']):
        dma_name = value['new_name']
        test_code += f"    ol.{dma_name}_dma.recvchannel.wait()\n"
    test_code += """    end_time = time.perf_counter()\n"""
    test_code += """    batch_time = end_time - start_time\n"""
    test_code += """    batch_lat_s.append(batch_time)\n"""
    test_code += """    img_lat_s.append(batch_time / BATCH)\n"""
    test_code += """    total_images += BATCH\n"""
    test_code += """    total_time_s += batch_time\n"""
    test_code += """
throughput = total_images / total_time_s
avg_batch_ms = (sum(batch_lat_s) / len(batch_lat_s)) * 1e3
avg_img_ms = (sum(img_lat_s) / len(img_lat_s)) * 1e3

print("===== Benchmark results =====")
print(f"Measured batches:        {len(batch_lat_s)}")
print(f"Measured images:         {total_images}")
print(f"Total measured time (s): {total_time_s:.6f}")
print(f"Throughput (img/s):      {throughput:.2f}")
print(f"Avg batch latency (ms):  {avg_batch_ms:.3f}")
print(f"Avg img latency (ms):    {avg_img_ms:.3f}")
"""

    for name, value in sorted(ap.output_map.items(), key=lambda x: x[1]['index']):
        dma_name = value['new_name']
        test_code += f"del {dma_name}_buffer\n"
    for name, value in sorted(ap.input_map.items(), key=lambda x: x[1]['index']):
        dma_name = value['new_name']
        test_code += f"del {dma_name}_buffer\n"
    return test_code

def make_deploy_directory(work_dir: str, top_name: str) -> str:
    """Create a deployment directory for the FPGA project."""
    deploy_dir = f"{work_dir}/build"
    if not os.path.exists(deploy_dir):
        os.makedirs(deploy_dir)
    return deploy_dir

class GenerateDriver(Transformation):

    def __init__(self, work_dir: str, original_model: ModelWrapper = None):
        super().__init__()
        self.work_dir = work_dir
        self.original_model = original_model

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        top_name = model.get_metadata_prop("top_name")
        axilite_address = int(model.get_metadata_prop("axilite_address"))
        axilite_size = int(model.get_metadata_prop("axilite_size"))
        board = model.get_metadata_prop("board_name")
        frequency = model.get_metadata_prop("frequency")
        design_id = model.get_metadata_prop("design_id")
        Pll_frequency = read_board_info(board)["PLL_frequency"]
        nn2FPGA_node = model.get_nodes_by_op_type("nn2fpgaPartition")[0]

        deploy_dir = make_deploy_directory(self.work_dir, top_name)
        model = model.transform(SetDynamicBatchSize())

        # Save the model to the work directory.
        model.save(f"{deploy_dir}/nn2FPGA_{top_name}.onnx")

        # Write the SpecOP.
        spec_file_path = os.path.join(deploy_dir, "generated_spec.hpp")
        with open(spec_file_path, "w") as f:
            f.write(
                generate_spec(
                    model,
                    nn2FPGA_node,
                    deploy_dir,
                    Nmax=10,
                    Pll_index=0,
                    Pll_frequency=Pll_frequency,
                    frequency=frequency,
                    axilite_base_addr=axilite_address,
                    axilite_size=axilite_size,
                    design_id=design_id,
                )
            )

        # Move generated_spec.hpp files to the deployment directory.
        shutil.move(spec_file_path, "/workspace/NN2FPGA/nn2fpga/operator_runtime/generated_spec.hpp")

        # Compile the custom operator.
        os.system(
            f"/workspace/NN2FPGA/tools/build_customop.sh /workspace/NN2FPGA/nn2fpga/operator_runtime/register_op.cpp {deploy_dir}"
        )

        # Check if the custom operator was built successfully.
        custom_op_path = os.path.join(deploy_dir, "libnn2fpga_customop.so")
        if not os.path.exists(custom_op_path):
            raise RuntimeError(f"Custom operator not built: {custom_op_path}")

        # Remove all the copies of the spec file.
        os.remove("/workspace/NN2FPGA/nn2fpga/operator_runtime/generated_spec.hpp")

        # Temporarily copy the pynq utility needed to upload the bitstream.
        shutil.copy(
            "/workspace/NN2FPGA/nn2fpga/operator_runtime/pynq_program.py",
            f"{deploy_dir}"
        )

        if self.original_model is not None:
            # Save the original model with QCDQ quantization for deployment.
            original_model = self.original_model.transform(ConvertToQCDQ())
            original_model = original_model.transform(SetDynamicBatchSize())
            original_model.save(f"{deploy_dir}/original_model_qcdq.onnx")
        
        with open(f"{deploy_dir}/throughput_test.py", "w") as f:
            f.write(generate_pynq_test(nn2FPGA_node))

        return model, False

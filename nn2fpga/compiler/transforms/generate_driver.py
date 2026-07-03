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
from nn2fpga.compiler.core.tensor_type import TensorType
from onnx import NodeProto
import numpy as np

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
    control_axi_offset: int,
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
    cwr.add_line(f"static constexpr off_t ControlAxiOffset = 0x{control_axi_offset:X};")

    cwr.add_line(f"static inline const std::array<PortDesc, {len(ap.input_map)}> Inputs{{{{")
    cwr.indent()
    for name, value in sorted(ap.input_map.items(), key=lambda x: x[1]['index']):
        tensor_shape = value["shape"]
        tensor_shape_nobatch = tensor_shape[1:]  # Exclude batch size
        str_tensor_shape = ', '.join(map(str, tensor_shape_nobatch))
        tensor_type = TensorType.from_canonical_name(value["quant"])
        mode = "PortMode::StaticInit" if value['value'] is not None else "PortMode::Dynamic"
        buffer_size = np.dtype(tensor_type.get_numpy_dtype()).itemsize * np.prod(tensor_shape_nobatch)
        if value['value'] is None:
            buffer_size *= Nmax
        cwr.add_line(
            f"PortDesc{{DType::{tensor_type.get_spec_type()}, {{{str_tensor_shape}}}, 0x{value['axi_offset']:X}, {mode}, {buffer_size}}}, // {name}"
        )
    cwr.dedent()
    cwr.add_line("}};")

    cwr.add_line(f"static inline const std::array<PortDesc, {len(ap.output_map)}> Outputs{{{{")
    cwr.indent()
    for name, value in sorted(ap.output_map.items(), key=lambda x: x[1]['index']):
        tensor_shape = value["shape"]
        tensor_shape_nobatch = tensor_shape[1:]  # Exclude batch size
        str_tensor_shape = ', '.join(map(str, tensor_shape_nobatch))
        tensor_type = TensorType.from_canonical_name(value["quant"])
        mode = "PortMode::StaticInit" if value['value'] is not None else "PortMode::Dynamic"
        buffer_size = np.dtype(tensor_type.get_numpy_dtype()).itemsize * np.prod(tensor_shape_nobatch)
        if value['value'] is None:
            buffer_size *= Nmax
        cwr.add_line(
            f"PortDesc{{DType::{tensor_type.get_spec_type()}, {{{str_tensor_shape}}}, 0x{value['axi_offset']:X}, {mode}, {buffer_size}}}, // {name}"
        )
    cwr.dedent()
    cwr.add_line("}};")

    cwr.add_line(f"static inline const std::array<BufferDesc, {len(ap.buffer_map)}> Buffers{{{{")
    cwr.indent()
    for buffer_name, buffer in ap.buffer_map.items():
        cwr.add_line(
            f"BufferDesc{{{buffer['size_bytes']}, 0x{buffer['read_axi_offset']:X}, 0x{buffer['write_axi_offset']:X}}}, // {buffer_name}"
        )
    cwr.dedent()
    cwr.add_line("}};")

    cwr.add_line("static inline const std::array<ONNXTensorElementDataType, "
                 f"{len(ap.input_map)}> OrtInputTypes{{{{")
    cwr.indent()
    for name in ap.input_map:
        tensor_type = TensorType.from_canonical_name(ap.input_map[name]["quant"])
        cwr.add_line(f"{tensor_type.get_onnxruntime_type()}, // {name}")
    cwr.dedent()
    cwr.add_line("}};")

    cwr.add_line("static inline const std::array<ONNXTensorElementDataType, "
                 f"{len(ap.output_map)}> OrtOutputTypes{{{{")
    cwr.indent()
    for name in ap.output_map:
        tensor_type = TensorType.from_canonical_name(ap.output_map[name]["quant"])
        cwr.add_line(f"{tensor_type.get_onnxruntime_type()}, // {name}")
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

    test_code = """# Auto-generated test script for nn2FPGA custom operator on PYNQ

import numpy as np
from pynq import Overlay
from pynq import allocate
from pynq import PL
import time

PL.reset()
ol = Overlay("Overlay/design.bit")
"""
    if ap.buffer_map:
        test_code += f"kernel = ol.{ap.top_name}_0\n"
        test_code += """
def write_u64(ip, offset, value):
    ip.write(offset, value & 0xFFFFFFFF)
    ip.write(offset + 4, (value >> 32) & 0xFFFFFFFF)

"""
    test_code += """
FRAMES = 300
RING_DEPTH = 3
INPUT_BUFFERS = RING_DEPTH
OUTPUT_BUFFERS = RING_DEPTH
POLL_SLEEP_S = 0.0005

def release_buffer(buf):
    freebuffer = getattr(buf, "freebuffer", None)
    if freebuffer is not None:
        freebuffer()

def release_buffer_list(buffers):
    for buf in buffers:
        release_buffer(buf)
    buffers.clear()

DMA_MM2S = 1
DMA_S2MM = 0
DMACR_RS = 0x00000001
DMACR_RESET = 0x00000004
DMASR_HALTED = 0x00000001
DMASR_IDLE = 0x00000002
DMASR_ERROR_MASK = 0x00000770
BD_STS_COMPLETE = 1 << 31
BD_STS_ERROR_MASK = 0x30000000
BD_CTRL_SOF = 1 << 27
BD_CTRL_EOF = 1 << 26
BD_CTRL_LEN_MASK = 0x03FFFFFF

class SgDmaRing:
    def __init__(self, dma_ip, direction, depth, name):
        self.mmio = dma_ip.mmio
        self.direction = direction
        self.depth = depth
        self.name = name
        self.offset = 0x00 if direction == DMA_MM2S else 0x30
        self.flush_before = direction == DMA_MM2S
        self.desc = allocate(shape=(depth, 16), dtype=np.uint32)
        self.free = list(range(depth))
        self.queued = []
        self.started = False
        self.tail = None
        self._init_descriptors()
        self.reset()

    def _reg(self, off):
        return self.offset + off

    def _desc_addr(self, idx):
        return self.desc.physical_address + idx * 16 * 4

    def _init_descriptors(self):
        self.desc[:] = 0
        for idx in range(self.depth):
            next_addr = self._desc_addr((idx + 1) % self.depth)
            self.desc[idx, 0] = next_addr & 0xFFFFFFFF
            self.desc[idx, 1] = (next_addr >> 32) & 0xFFFFFFFF
        self.desc.flush()

    def reset(self):
        self.mmio.write(self._reg(0x00), DMACR_RESET)
        while self.mmio.read(self._reg(0x00)) & DMACR_RESET:
            pass
        self.mmio.write(self._reg(0x04), 0xFFFFFFFF)
        self.mmio.write(self._reg(0x00), 0x00000000)
        self.started = False
        self.tail = None

    def has_free(self):
        return bool(self.free)

    def status(self):
        return self.mmio.read(self._reg(0x04))

    def _check_error(self):
        sr = self.status()
        if sr & DMASR_ERROR_MASK:
            raise RuntimeError(f"{self.name} SG DMA error: status=0x{sr:08x}")

    def _start_at(self, idx):
        addr = self._desc_addr(idx)
        self.mmio.write(self._reg(0x08), addr & 0xFFFFFFFF)
        self.mmio.write(self._reg(0x0C), (addr >> 32) & 0xFFFFFFFF)
        self.mmio.write(self._reg(0x00), DMACR_RS)
        while self.mmio.read(self._reg(0x04)) & DMASR_HALTED:
            self._check_error()
        self.started = True

    def enqueue(self, seq, buf, nbytes=None):
        if not self.free:
            raise RuntimeError(f"{self.name}: no free SG descriptors")
        if nbytes is None:
            nbytes = buf.nbytes
        if nbytes <= 0 or nbytes > BD_CTRL_LEN_MASK:
            raise ValueError(f"{self.name}: invalid transfer length {nbytes}")

        idx = self.free.pop(0)
        addr = buf.physical_address
        self.desc[idx, 2] = addr & 0xFFFFFFFF
        self.desc[idx, 3] = (addr >> 32) & 0xFFFFFFFF
        self.desc[idx, 4] = 0
        self.desc[idx, 5] = 0
        self.desc[idx, 6] = (nbytes & BD_CTRL_LEN_MASK) | BD_CTRL_SOF | BD_CTRL_EOF
        self.desc[idx, 7] = 0
        self.desc[idx, 8:16] = 0
        self.desc.flush()

        if self.flush_before:
            buf.flush()

        if not self.started:
            self._start_at(idx)

        tail_addr = self._desc_addr(idx)
        self.mmio.write(self._reg(0x10), tail_addr & 0xFFFFFFFF)
        self.mmio.write(self._reg(0x14), (tail_addr >> 32) & 0xFFFFFFFF)
        self.tail = idx
        self.queued.append({"seq": seq, "idx": idx, "buf": buf, "nbytes": nbytes})
        return self.queued[-1]

    def poll(self):
        self._check_error()
        completed = []
        while self.queued:
            head = self.queued[0]
            idx = head["idx"]
            self.desc.invalidate()
            status = int(self.desc[idx, 7])
            if not (status & BD_STS_COMPLETE):
                break
            if status & BD_STS_ERROR_MASK:
                raise RuntimeError(f"{self.name}: descriptor {idx} error status=0x{status:08x}")
            if not self.flush_before:
                head["buf"].invalidate()
            completed.append(self.queued.pop(0))
        return completed

    def reclaim(self, desc_info):
        idx = desc_info["idx"]
        self.desc[idx, 7] = 0
        self.desc.flush()
        self.free.append(idx)

    def close(self):
        self.reset()
        release_buffer(self.desc)

"""

    # Generate code to allocate input and output buffers based on the accelerator package specification.
    for name, value in sorted(ap.input_map.items(), key=lambda x: x[1]['index']):
        tensor_shape = value["shape"]
        dma_name = value['new_name'] 
        tensor_shape_nobatch = tensor_shape[1:]  # Exclude batch size
        str_tensor_shape = ', '.join(map(str, tensor_shape_nobatch))
        tensor_type = TensorType.from_canonical_name(value["quant"])
        np_dtype = tensor_type.get_numpy_dtype()
        np_dtype_info = np.iinfo(np_dtype)
        if value['value'] is None:
            test_code += f"{dma_name}_buffers = [allocate(shape=({str_tensor_shape}), dtype=\"{np_dtype.__name__}\") for _ in range(INPUT_BUFFERS)]\n"
            test_code += f"for _buf in {dma_name}_buffers:\n"
            test_code += f"    _buf[:] = np.random.randint({np_dtype_info.min}, {np_dtype_info.max}, size=({str_tensor_shape}), dtype=\"{np_dtype.__name__}\")\n"
        else:
            str_tensor_shape = f"{str_tensor_shape},"
            test_code += f"{dma_name}_buffer = allocate(shape=({str_tensor_shape}), dtype=\"{np_dtype.__name__}\")\n"
            test_code += f"{dma_name}_data = np.random.randint({np_dtype_info.min}, {np_dtype_info.max}, size=({str_tensor_shape}), dtype=\"{np_dtype.__name__}\")\n"

    for name, value in sorted(ap.output_map.items(), key=lambda x: x[1]['index']):
        tensor_shape = value["shape"]
        dma_name = value['new_name']
        tensor_shape_nobatch = tensor_shape[1:]  # Exclude batch size
        str_tensor_shape = ', '.join(map(str, tensor_shape_nobatch))
        tensor_type = TensorType.from_canonical_name(value["quant"])
        np_dtype = tensor_type.get_numpy_dtype()
        buffer_size = np.dtype(np_dtype).itemsize * np.prod(tensor_shape_nobatch)
        test_code += f"{dma_name}_buffers = [allocate(shape=({str_tensor_shape}), dtype=\"{np_dtype.__name__}\") for _ in range(OUTPUT_BUFFERS)]\n"
        #test_code += f"ol.{dma_name}_dma.recvchannel._max_size = {buffer_size}\n"
        #test_code += f"ol.{dma_name}_dma.recvchannel._align = 1\n"

    for buffer_name, buffer in ap.buffer_map.items():
        test_code += f"{buffer_name}_buffer = allocate(shape=({buffer['size_bytes']},), dtype=np.int8)\n"
        test_code += f"{buffer_name}_buffer[:] = 0\n"
        test_code += f"{buffer_name}_addr = {buffer_name}_buffer.device_address\n"
        test_code += f"write_u64(kernel, 0x{buffer['read_axi_offset']:X}, {buffer_name}_addr)\n"
        test_code += f"write_u64(kernel, 0x{buffer['write_axi_offset']:X}, {buffer_name}_addr)\n"

    # Load the static inputs
    for name, value in sorted(ap.input_map.items(), key=lambda x: x[1]['index']):
        if value['value'] is not None:
            dma_name = value['new_name']
            test_code += f"{dma_name}_buffer[:] = {dma_name}_data[:]\n"
            test_code += f"ol.{dma_name}_dma.sendchannel.transfer({dma_name}_buffer)\n"
            test_code += f"ol.{dma_name}_dma.sendchannel.wait()\n"
            test_code += f"print('Static input {dma_name} loaded')\n"

    dynamic_inputs = [
        value['new_name']
        for _, value in sorted(ap.input_map.items(), key=lambda x: x[1]['index'])
        if value['value'] is None
    ]
    output_names = [
        value['new_name']
        for _, value in sorted(ap.output_map.items(), key=lambda x: x[1]['index'])
    ]

    test_code += """
input_rings = {}
output_rings = {}
"""
    for dma_name in dynamic_inputs:
        test_code += f"input_rings['{dma_name}'] = SgDmaRing(ol.{dma_name}_dma, DMA_MM2S, RING_DEPTH, '{dma_name}_mm2s')\n"
    for dma_name in output_names:
        test_code += f"output_rings['{dma_name}'] = SgDmaRing(ol.{dma_name}_dma, DMA_S2MM, RING_DEPTH, '{dma_name}_s2mm')\n"

    test_code += """
free_slots = list(range(RING_DEPTH))
active = {}
sent_frames = 0
completed_frames = 0
blocked_no_slot = 0
blocked_no_desc = 0
submit_times = {}
latencies_s = []

def rings_have_capacity():
    return all(ring.has_free() for ring in input_rings.values()) and all(
        ring.has_free() for ring in output_rings.values()
    )

start_s = time.perf_counter()

while completed_frames < FRAMES:
    now = time.perf_counter()

    for name, ring in input_rings.items():
        for desc in ring.poll():
            req = active.get(desc["seq"])
            if req is not None:
                req["input_done"][name] = True

    for name, ring in output_rings.items():
        for desc in ring.poll():
            req = active.get(desc["seq"])
            if req is not None:
                req["output_done"][name] = True

    for seq in sorted(list(active.keys())):
        req = active[seq]
        if all(req["output_done"].values()):
            completed_frames += 1
            submit_s = submit_times.pop(seq, None)
            if submit_s is not None:
                latencies_s.append(now - submit_s)
            for desc in req["input_descs"]:
                desc["ring"].reclaim(desc)
            for desc in req["output_descs"]:
                desc["ring"].reclaim(desc)
            free_slots.append(req["slot"])
            del active[seq]

    while sent_frames < FRAMES:
        if not free_slots:
            blocked_no_slot += 1
            break
        if not rings_have_capacity():
            blocked_no_desc += 1
            break

        slot = free_slots.pop(0)
        seq = sent_frames
        req = {
            "slot": slot,
            "input_done": {name: False for name in input_rings},
            "output_done": {name: False for name in output_rings},
            "input_descs": [],
            "output_descs": [],
        }

"""
    for dma_name in output_names:
        test_code += f"        desc = output_rings['{dma_name}'].enqueue(seq, {dma_name}_buffers[slot])\n"
        test_code += f"        desc['ring'] = output_rings['{dma_name}']\n"
        test_code += "        req['output_descs'].append(desc)\n"
    for dma_name in dynamic_inputs:
        test_code += f"        desc = input_rings['{dma_name}'].enqueue(seq, {dma_name}_buffers[slot])\n"
        test_code += f"        desc['ring'] = input_rings['{dma_name}']\n"
        test_code += "        req['input_descs'].append(desc)\n"
    test_code += """
        active[seq] = req
        submit_times[seq] = now
        sent_frames += 1

    if POLL_SLEEP_S > 0:
        time.sleep(POLL_SLEEP_S)

total_s = time.perf_counter() - start_s
sorted_latencies = sorted(latencies_s)
avg_latency_ms = (sum(latencies_s) / len(latencies_s)) * 1e3 if latencies_s else float("nan")
p50_latency_ms = sorted_latencies[len(sorted_latencies) // 2] * 1e3 if sorted_latencies else float("nan")
max_latency_ms = max(latencies_s) * 1e3 if latencies_s else float("nan")

print("===== SG streaming benchmark results =====")
print(f"Frames submitted:          {sent_frames}")
print(f"Frames completed:          {completed_frames}")
print(f"Total measured time (s):   {total_s:.6f}")
print(f"Completed throughput img/s:{completed_frames / total_s:.2f}")
print(f"Avg submit-to-output ms:   {avg_latency_ms:.3f}")
print(f"P50 submit-to-output ms:   {p50_latency_ms:.3f}")
print(f"Max submit-to-output ms:   {max_latency_ms:.3f}")
print(f"No-free-slot polls:        {blocked_no_slot}")
print(f"No-free-desc polls:        {blocked_no_desc}")

for ring in input_rings.values():
    ring.close()
for ring in output_rings.values():
    ring.close()
"""

    for name, value in sorted(ap.output_map.items(), key=lambda x: x[1]['index']):
        dma_name = value['new_name']
        test_code += f"release_buffer_list({dma_name}_buffers)\n"
    for name, value in sorted(ap.input_map.items(), key=lambda x: x[1]['index']):
        dma_name = value['new_name']
        if value['value'] is not None:
            test_code += f"release_buffer({dma_name}_buffer)\n"
        else:
            test_code += f"release_buffer_list({dma_name}_buffers)\n"
    for buffer_name in ap.buffer_map:
        test_code += f"release_buffer({buffer_name}_buffer)\n"
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
        control_axi_offset = int(model.get_metadata_prop("control_axi_offset") or 0)
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
                    control_axi_offset=control_axi_offset,
                    design_id=design_id,
                )
            )

        # Move generated_spec.hpp files to the deployment directory.
        shutil.move(spec_file_path, "/workspace/NN2FPGA/nn2fpga/hw/operator_runtime/generated_spec.hpp")

        # Compile the custom operator.
        os.system(
            f"/workspace/NN2FPGA/tools/build_customop.sh /workspace/NN2FPGA/nn2fpga/hw/operator_runtime/register_op.cpp {deploy_dir}"
        )

        # Check if the custom operator was built successfully.
        custom_op_path = os.path.join(deploy_dir, "libnn2fpga_customop.so")
        if not os.path.exists(custom_op_path):
            raise RuntimeError(f"Custom operator not built: {custom_op_path}")

        # Remove all the copies of the spec file.
        os.remove("/workspace/NN2FPGA/nn2fpga/hw/operator_runtime/generated_spec.hpp")

        # Temporarily copy the pynq utility needed to upload the bitstream.
        shutil.copy(
            "/workspace/NN2FPGA/nn2fpga/hw/operator_runtime/pynq_program.py",
            f"{deploy_dir}"
        )

        if self.original_model is not None:
            # Save the original model with QCDQ quantization for deployment.
            original_model = self.original_model.transform(ConvertToQCDQ())
            original_model = original_model.transform(SetDynamicBatchSize())
            original_model.save(f"{deploy_dir}/original_model_qcdq.onnx")
        
        with open(f"{deploy_dir}/throughput_test.py", "w") as f:
            f.write(generate_pynq_test(nn2FPGA_node))

        bitstream_path = f"{self.work_dir}/vivado/vivadoproj/vivadoproj.runs/impl_1/{top_name}_bd.bit"
        hwh_path = f"{self.work_dir}/vivado/vivadoproj/vivadoproj.gen/sources_1/bd/{top_name}_bd/hw_handoff/{top_name}_bd.hwh"
        shutil.copy(
            bitstream_path,
            f"{deploy_dir}/design.bit"
        )
        shutil.copy(
            hwh_path,
            f"{deploy_dir}/design.hwh"
        )

        return model, False

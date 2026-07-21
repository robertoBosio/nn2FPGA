import os
import re
from collections import OrderedDict

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation

from nn2fpga.compiler.core.acceleratorpackage import AcceleratorPackage
from nn2fpga.compiler.transforms.embed_hls_code import generate_hls_code
from nn2fpga.compiler.utils.board_util import read_board_info


INTERFACE_PRAGMA_RE = re.compile(r"^\s*#\s*pragma\s+HLS\s+INTERFACE\b")
ARRAY_RESHAPE_RE = re.compile(r"^\s*#\s*pragma\s+HLS\s+ARRAY_RESHAPE\b", re.IGNORECASE)
ARRAY_PARTITION_RE = re.compile(r"^\s*#\s*pragma\s+HLS\s+array_partition\b", re.IGNORECASE)
ARRAY_DECL_RE = re.compile(
    r"""^\s*
        (?:[A-Za-z_].*?)
        \b([A-Za-z_]\w*)\b
        \s*(\[[^\]]*\])\s*
        (?:\s*(\[[^\]]*\]\s*))*
        \s*;
        \s*$
        """,
    re.VERBOSE,
)


def split_top_level_commas(text: str) -> list[str]:
    parts = []
    current = []
    angle = paren = bracket = 0
    for ch in text:
        if ch == "<":
            angle += 1
        elif ch == ">":
            angle -= 1
        elif ch == "(":
            paren += 1
        elif ch == ")":
            paren -= 1
        elif ch == "[":
            bracket += 1
        elif ch == "]":
            bracket -= 1
        if ch == "," and angle == paren == bracket == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current).strip())
    return [part for part in parts if part]


def strip_cpp_comments(text: str) -> str:
    return "\n".join(line.split("//", 1)[0] for line in text.splitlines())


def find_matching(text: str, start: int, opening: str, closing: str) -> int:
    depth = 0
    for pos in range(start, len(text)):
        if text[pos] == opening:
            depth += 1
        elif text[pos] == closing:
            depth -= 1
            if depth == 0:
                return pos
    raise RuntimeError(f"Unmatched {opening}{closing} in generated HLS code")


def find_top_signature(text: str, top_name: str) -> tuple[int, int, str]:
    match = re.search(rf"\bvoid\s+{re.escape(top_name)}\s*\(", text)
    if not match:
        raise ValueError(f"Top function '{top_name}' not found in generated HLS code.")
    open_paren = text.find("(", match.start())
    close_paren = find_matching(text, open_paren, "(", ")")
    return match.start(), close_paren + 1, text[open_paren + 1 : close_paren]


def build_signature(top_name: str, params: list[str]) -> str:
    if not params:
        return f"void {top_name}()"
    return "void {}(\n{}\n)".format(
        top_name,
        ",\n".join(f"    {param}" for param in params),
    )


def find_object_block(text: str, class_name: str, start: int = 0):
    match = re.search(rf"\b{re.escape(class_name)}\s*<", text[start:])
    if not match:
        return None
    obj_start = start + match.start()
    open_angle = text.find("<", obj_start)
    close_angle = find_matching(text, open_angle, "<", ">")
    after = text[close_angle + 1 :]
    obj_match = re.match(r"\s*([A-Za-z_]\w*)\s*;", after)
    if not obj_match:
        raise RuntimeError(f"Could not parse {class_name} object declaration.")
    obj_name = obj_match.group(1)
    obj_end = close_angle + 1 + obj_match.end()
    template_args = split_top_level_commas(strip_cpp_comments(text[open_angle + 1 : close_angle]))
    return obj_start, obj_end, obj_name, template_args


def find_run_call(text: str, obj_name: str, start: int):
    match = re.search(rf"\b{re.escape(obj_name)}\s*\.\s*run\s*<[^>]*>\s*\(", text[start:])
    if not match:
        raise RuntimeError(f"Could not find run call for object '{obj_name}'.")
    call_start = start + match.start()
    open_paren = text.find("(", call_start)
    close_paren = find_matching(text, open_paren, "(", ")")
    semicolon = text.find(";", close_paren)
    args = split_top_level_commas(text[open_paren + 1 : close_paren])
    return call_start, semicolon + 1, args


def remove_boundary_adapters(text: str) -> tuple[str, list[str]]:
    params = []
    offset = 0
    while True:
        block = find_object_block(text, "AXIToStream", offset)
        if block is None:
            break
        obj_start, obj_end, obj_name, template_args = block
        call_start, call_end, call_args = find_run_call(text, obj_name, obj_end)
        t_output_word = template_args[1]
        dim0 = int(template_args[5])
        dim1 = int(template_args[6])
        dim2 = int(template_args[7])
        dim1_unroll = int(template_args[8])
        dim2_unroll = int(template_args[9])
        depth = (dim0 * dim1 * dim2) // (dim1_unroll * dim2_unroll)
        input_name, output_stream = call_args[0], call_args[1]
        params.append(f"std::array<{t_output_word}, {dim1_unroll}> {input_name}[{depth}]")
        replacement = f"mm2s_word<{t_output_word}, {dim1_unroll}, {depth}>({input_name}, {output_stream});"
        text = text[:call_start] + replacement + text[call_end:]
        text = text[:obj_start] + text[obj_end:]
        offset = obj_start + len(replacement)

    offset = 0
    while True:
        block = find_object_block(text, "StreamToAXI", offset)
        if block is None:
            break
        obj_start, obj_end, obj_name, template_args = block
        call_start, call_end, call_args = find_run_call(text, obj_name, obj_end)
        t_input_word = template_args[0]
        iter_count = int(template_args[4])
        dim1_unroll = int(template_args[9])
        input_stream, output_name = call_args[0], call_args[1]
        params.append(f"std::array<{t_input_word}, {dim1_unroll}> {output_name}[{iter_count}]")
        replacement = f"s2mm_word<{t_input_word}, {dim1_unroll}, {iter_count}>({input_stream}, {output_name});"
        text = text[:call_start] + replacement + text[call_end:]
        text = text[:obj_start] + text[obj_end:]
        offset = obj_start + len(replacement)

    return text, params


def collect_and_hoist_reshaped_memories(lines: list[str]) -> tuple[list[str], list[str]]:
    declarations = OrderedDict()
    remove_indexes = set()
    for index, line in enumerate(lines):
        if not (ARRAY_RESHAPE_RE.match(line) or ARRAY_PARTITION_RE.match(line)):
            continue
        decl_index = index - 1
        while decl_index >= 0 and (
            ARRAY_RESHAPE_RE.match(lines[decl_index])
            or ARRAY_PARTITION_RE.match(lines[decl_index])
            or not lines[decl_index].strip()
        ):
            decl_index -= 1
        if decl_index < 0:
            continue
        candidate = lines[decl_index].rstrip("\n")
        if ARRAY_DECL_RE.match(candidate):
            declarations.setdefault(candidate, None)
            remove_indexes.add(decl_index)
    return [line for i, line in enumerate(lines) if i not in remove_indexes], [d.rstrip(";").strip() for d in declarations]


def generate_stream_utils() -> str:
    return """#pragma once
#include "hls_stream.h"
#include "ap_int.h"
#include "ap_float.h"
#include <array>
#include <cstddef>

template <typename TWord, size_t W_PAR, size_t DEPTH>
void mm2s_word(std::array<TWord, W_PAR> in_data[DEPTH],
               hls::stream<TWord> out_stream[W_PAR]) {
  for (size_t d = 0; d < DEPTH; d++) {
    for (size_t w = 0; w < W_PAR; w++) {
      out_stream[w].write(in_data[d][w]);
    }
  }
}

template <typename TWord, size_t W_PAR, size_t DEPTH>
void s2mm_word(hls::stream<TWord> in_stream[W_PAR],
               std::array<TWord, W_PAR> out_data[DEPTH]) {
  for (size_t d = 0; d < DEPTH; d++) {
    for (size_t w = 0; w < W_PAR; w++) {
      out_data[d][w] = in_stream[w].read();
    }
  }
}
"""


def generate_testbench(top_name: str, signature: str) -> str:
    params_text = signature[signature.find("(") + 1 : signature.rfind(")")].strip()
    params = split_top_level_commas(params_text) if params_text else []
    declarations = []
    args = []
    for param in params:
        match = re.search(r"\b([A-Za-z_]\w*)\b\s*(\[[^\]]*\].*)$", param)
        if not match:
            continue
        name = match.group(1)
        declarations.append(f"    {param};")
        args.append(name)
    call = f"    {top_name}({', '.join(args)});"
    return f"""#include "ap_int.h"
#include "hls_stream.h"
#include "ap_int.h"
#include "ap_float.h"
#include <array>

extern {signature};

int main(int argc, char** argv)
{{
    (void)argc;
    (void)argv;
{chr(10).join(declarations)}

{call}
{call}
{call}
    return 0;
}}
"""


def generate_tcl(top_name: str, part_name: str, frequency: str, hls_version: str) -> str:
    t_clk = f"{1e3 / int(frequency):.2f}ns"
    lines = [
        "# Auto-generated TCL script for LightningSim-compatible HLS project",
        "# Generated by nn2FPGA.",
        "",
    ]
    if float(hls_version) > 2025:
        lines.append('open_component -reset "lightningsim_hls" -flow_target vivado')
    else:
        lines.extend([
            'open_project -reset "lightningsim_hls"',
            'open_solution -reset solution0',
        ])
    lines.extend([
        'add_files kernel.cpp -cflags "-I/workspace/NN2FPGA/nn2fpga/hw/library/include -Iinclude"',
        'add_files -tb testbench.cpp -cflags "-I/workspace/NN2FPGA/nn2fpga/hw/library/include -Iinclude"',
        f'set_top "{top_name}"',
        f'set_part {part_name}',
        f'create_clock -period {t_clk}',
        'config_compile -pipeline_style flp',
        'csim_design',
        'csynth_design',
        'exit',
    ])
    return "\n".join(lines)


def make_lightningsim_kernel(code: str, top_name: str) -> tuple[str, str]:
    lines = [line for line in code.splitlines(True) if not INTERFACE_PRAGMA_RE.match(line)]
    lines, memory_params = collect_and_hoist_reshaped_memories(lines)
    text = "".join(lines)
    text, boundary_params = remove_boundary_adapters(text)
    sig_start, sig_end, _ = find_top_signature(text, top_name)
    params = boundary_params + memory_params
    signature = build_signature(top_name, params)
    text = text[:sig_start] + signature + text[sig_end:]
    if 'utils/stream_utils.hpp' not in text:
        text = '#include "utils/stream_utils.hpp"\n' + text
    return text, signature


class GenerateLightningSimCode(Transformation):
    """Generate LightningSim-compatible HLS code from a pre-AddStreamingParams HLS model."""

    def __init__(self, work_root: str = "/tmp"):
        super().__init__()
        self.work_root = work_root

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        ap = AcceleratorPackage.from_json(model.get_metadata_prop("accelerator_package"))
        top_name = model.get_metadata_prop("top_name")
        board_name = model.get_metadata_prop("board_name")
        frequency = model.get_metadata_prop("frequency")
        hls_version = model.get_metadata_prop("hls_version")
        part_name = read_board_info(board_name)["part"]

        out_dir = os.path.join(self.work_root, "lightningsim")
        include_dir = os.path.join(out_dir, "include", "utils")
        os.makedirs(include_dir, exist_ok=True)

        kernel_code, signature = make_lightningsim_kernel(generate_hls_code(model, ap), top_name)
        with open(os.path.join(out_dir, "kernel.cpp"), "w") as f:
            f.write(kernel_code)
        with open(os.path.join(out_dir, "testbench.cpp"), "w") as f:
            f.write(generate_testbench(top_name, signature))
        with open(os.path.join(out_dir, "hls_lightningsim.tcl"), "w") as f:
            f.write(generate_tcl(top_name, part_name, frequency, hls_version))
        with open(os.path.join(include_dir, "stream_utils.hpp"), "w") as f:
            f.write(generate_stream_utils())

        return model, False

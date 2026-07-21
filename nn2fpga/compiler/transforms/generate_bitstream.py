from pathlib import Path
from typing import Optional
import base64
import os
import subprocess
import re
import logging
from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from nn2fpga.compiler.core.acceleratorpackage import AcceleratorPackage
from nn2fpga.compiler.utils.board_util import read_board_info

SOLUTION_NAME = "solution0"
PROJECT_NAME = "hlsproj"
SMARTCONNECT_MAX_SI = 16
HP_PORTS = [0, 1, 2, 3]
logger = logging.getLogger(__name__)

def split_ddr_masters(ddr_masters: list[dict]) -> list[list[dict]]:
    """Balance DDR AXI masters across PS HP ports without exceeding SmartConnect SI limits."""
    if len(ddr_masters) > len(HP_PORTS) * SMARTCONNECT_MAX_SI:
        raise ValueError(
            f"Cannot connect {len(ddr_masters)} DDR AXI masters: "
            f"capacity is {len(HP_PORTS) * SMARTCONNECT_MAX_SI} masters "
            f"({len(HP_PORTS)} HP ports x {SMARTCONNECT_MAX_SI} SmartConnect SI)."
        )

    groups = [[] for _ in HP_PORTS]
    for master in ddr_masters:
        group = min(groups, key=len)
        group.append(master)

    return groups

def optimize_ram_decomp(v_file: Path):
    # Match any Verilog attribute block: (* ... *)
    attr_pattern = re.compile(r'\(\*(.*?)\*\)', re.DOTALL)

    text = v_file.read_text()

    replaced = 0  # how many attribute blocks we actually modified

    def add_ram_decomp(match: re.Match) -> str:
        nonlocal replaced
        attrs = match.group(1)

        # Only touch attribute lists that mention ram_style
        if "ram_style" in attrs and "ram_decomp" not in attrs:
            # Preserve original trailing whitespace inside the attribute block
            attrs_strip_right = attrs.rstrip()
            trailing_ws = attrs[len(attrs_strip_right):]

            # Decide whether to add a comma or not
            if attrs_strip_right.strip().endswith(","):
                new_attrs = attrs_strip_right + ' ram_decomp="area"'
            else:
                new_attrs = attrs_strip_right + ', ram_decomp="area"'

            replaced += 1
            return f'(*{new_attrs}{trailing_ws}*)'
        else:
            # No change to this attribute block
            return match.group(0)

    new_text = attr_pattern.sub(add_ram_decomp, text)

    if replaced > 0:
        v_file.write_text(new_text)
        print(f"Updated {v_file} (modified {replaced} attribute block(s)) to add ram_decomp.")
    else:
        print(f"No ram_style attributes patched in {v_file}, skipped.")

def process_verilog_files(hls_dir: Path):
    dat_files = list(hls_dir.rglob("*.dat"))
    for dat in dat_files:
        v_file = dat.with_suffix(".v")
        if v_file.exists():
            optimize_ram_decomp(v_file)

def gate_entry_proc_on_configured_buffers(hls_dir: Path, top_name: str, buffer_names: list[str]) -> None:
    """Hold HLS entry_proc until DDRStream AXI-Lite pointer registers are programmed."""
    if not buffer_names:
        logger.info("Skipping entry_proc gate patch because there are no DDR stream buffers.")
        return

    top_v = hls_dir / f"{top_name}.v"
    if not top_v.exists():
        logger.info("Skipping entry_proc gate patch because %s does not exist.", top_v)
        return

    text = top_v.read_text()
    patch_marker = "// nn2fpga patch: gate entry_proc until AXI-Lite DDR buffer addresses are configured"
    if patch_marker in text:
        logger.info("entry_proc gate patch already present in %s, skipped.", top_v)
        return

    start_assignment = "assign entry_proc_U0_ap_start = 1'b1;"
    occurrences = text.count(start_assignment)
    if occurrences != 1:
        raise RuntimeError(
            f"Unable to patch {top_v}: expected exactly one '{start_assignment}', found {occurrences}."
        )

    checks = []
    for buffer_name in buffer_names:
        for suffix in ("read", "write"):
            signal = f"{buffer_name}_{suffix}"
            if not re.search(rf"\b{re.escape(signal)}\b", text):
                raise RuntimeError(
                    f"Unable to patch {top_v}: expected buffer address signal '{signal}' was not found."
                )
            checks.append(f"    ({signal} != 64'd0)")

    valid_expr = " &&\n".join(checks) if checks else "    1'b1"
    patch = (
        f"{patch_marker}\n"
        "wire entry_proc_cfg_addrs_valid;\n\n"
        "assign entry_proc_cfg_addrs_valid =\n"
        f"{valid_expr};\n"
        "// nn2fpga patch end\n\n"
        "assign entry_proc_U0_ap_start = entry_proc_cfg_addrs_valid;"
    )

    top_v.write_text(text.replace(start_assignment, patch, 1))
    logger.info("Patched %s to gate entry_proc on %d DDR buffer address(es).", top_v, len(buffer_names))

def recover_buffer_axilite_offsets(hls_output_dir: Path, top_name: str, buffer_map: dict) -> None:
    """Recover HLS AXI-Lite pointer register offsets from the generated driver header."""
    if not buffer_map:
        return

    header_path = (
        hls_output_dir
        / "impl"
        / "ip"
        / "drivers"
        / f"{top_name}_v1_0"
        / "src"
        / f"x{top_name}_hw.h"
    )
    if not header_path.exists():
        matches = list(hls_output_dir.rglob(f"x{top_name}_hw.h"))
        if not matches:
            raise RuntimeError(
                f"Unable to recover buffer AXI-Lite offsets: x{top_name}_hw.h was not found under {hls_output_dir}."
            )
        header_path = matches[0]

    define_pattern = re.compile(
        rf"^#define\s+X{re.escape(top_name).upper()}_CONTROL_ADDR_(?P<arg>[A-Z0-9_]+)_DATA\s+(?P<offset>0x[0-9A-Fa-f]+|\d+)\s*$"
    )
    offsets = {}
    for line in header_path.read_text().splitlines():
        match = define_pattern.match(line.strip())
        if match:
            offsets[match.group("arg")] = int(match.group("offset"), 0)

    missing = []
    for buffer_name, buffer in buffer_map.items():
        read_arg = f"{buffer_name}_read".upper()
        write_arg = f"{buffer_name}_write".upper()
        if read_arg not in offsets or write_arg not in offsets:
            missing.append(buffer_name)
            continue
        buffer["read_axi_offset"] = offsets[read_arg]
        buffer["write_axi_offset"] = offsets[write_arg]
        buffer["axi_offset"] = offsets[read_arg]

    if missing:
        raise RuntimeError(
            "Unable to recover AXI-Lite offsets for buffers "
            f"{missing} from {header_path}."
        )

def dump_tcl_script(
    top_name,
    part_name,
    frequency,
    hls_version,
    reset=True,
    silvia_packing=False,
    synthesize=True,
    export=True,
) -> str:
    """Dump a TCL script to set up the HLS project and run the simulation."""

    t_clk = f"{1e3 / int(frequency):.2f}ns" # Convert frequency in MHz to clock period in ns
    csynt_command = "csynth_design" 
    lines = list()
    lines.append("# Auto-generated TCL script for HLS project setup")
    lines.append("# Generated by nn2FPGA simulation flow")
    lines.append("")

    if silvia_packing:
        lines.append('source "/workspace/NN2FPGA/deps/SILVIA/scripts/SILVIA.tcl"')
        lines.append('set SILVIA::ROOT "/workspace/NN2FPGA/deps/SILVIA"')
        lines.append('set SILVIA::LLVM_ROOT "${SILVIA::ROOT}/llvm-project/install"')
        lines.append('set SILVIA::PASSES [list [dict create OP "muladd" MAX_CHAIN_LEN 3 OP_SIZE 4] [dict create OP "muladd" INLINE 1 MAX_CHAIN_LEN 3] [dict create OP "muladd" MUL_ONLY 1 INLINE 1]]')
        lines.append('set SILVIA::DEBUG 1')
        csynt_command = "SILVIA::csynth_design"

    # Check the HLS version to determine the correct syntax
    if float(hls_version) > 2025:
        lines.append(
            f'open_component {"-reset" if reset else ""} "{PROJECT_NAME}" -flow_target vivado',
        )
    else:
        lines.extend(
            [
                f'open_project {"-reset" if reset else ""} "{PROJECT_NAME}"',
                f'open_solution {"-reset" if reset else ""} {SOLUTION_NAME}',
            ]
        )

    lines.extend(
        [
            f'add_files kernel.cpp -cflags " -I/workspace/NN2FPGA/nn2fpga/hw/library/include"',
            f'set_top {top_name}',
            f'set_part {part_name}',
            f'create_clock -period {t_clk}',
            # 'config_compile -pipeline_style stp -enable_auto_rewind=false',
            'config_compile -pipeline_style flp',
            # 'config_compile -pipeline_style stp',
            # 'config_storage fifo -impl lutram',
            f'{csynt_command} ' if synthesize else '',
            # 'export_design -flow syn' if export else '',
            f'export_design -format ip_catalog -ipname {top_name} -library "ml" -vendor "polito.nn2FPGA" -version "1.0" -description "Generated by nn2FPGA"' if export else '', 
            'exit',
        ]
    )

    return "\n".join(lines)

def vivado_tcl_script(
    work_dir: str,
    top_name: str,
    part_name: str,
    board_part_name: str,
    frequency: int,
    hls_version: str,
    axilite_base_addr: int,
    axilite_dma_window: int,
    interface_width: int,
    design_id: int,
    inputs: list,
    outputs: list,
    buffers: list,
    control_axi_offset: Optional[int],
) -> str:
    """Generate a Vivado TCL script for the HLS project setup."""

    lines = list()
    lines.append("# Auto-generated Vivado TCL script for bitstream generation")
    lines.append("# Generated by nn2FPGA simulation flow")
    lines.append("")

    # Project creation and setup
    lines.append(f'create_project vivadoproj {work_dir}/vivadoproj -part {part_name} -force') 
    lines.append(f'set_property board_part {board_part_name} [current_project]')

    # Add HLS IP
    lines.append(f'set_property ip_repo_paths {work_dir}/hlsproj [current_project]')
    lines.append(f'update_ip_catalog')
    lines.append(f'set ip_found [llength [get_ipdefs -filter \"NAME == {top_name}\"]]')
    lines.append(f'if {{$ip_found == 0}} {{')
    lines.append(f'  puts \"Error: IP {top_name} not found in catalog\"')
    lines.append(f'  exit 1')
    lines.append(f'}}')

    # Create the block design
    lines.append(f'create_bd_design "{top_name}_bd"')

    # Add the PS block
    lines.append(
        f"create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.5 zynq_ultra_ps_e_0"
    )
    lines.append(
        f'apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {{apply_board_preset "1" }}  [get_bd_cells zynq_ultra_ps_e_0]'
    )
    lines.extend(
        [
            "set_property -dict [list \\",
            f"  CONFIG.PSU__FPGA_PL1_ENABLE {{{0}}} \\",
            f"  CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {{{frequency}}} \\",
            f"  CONFIG.PSU__USE__M_AXI_GP1 {{{0}}} \\",
            f"  CONFIG.PSU__USE__S_AXI_GP2 {{{1}}} \\",
            f"  CONFIG.PSU__USE__S_AXI_GP3 {{{1}}} \\",
            f"  CONFIG.PSU__USE__S_AXI_GP4 {{{1}}} \\",
            f"  CONFIG.PSU__USE__S_AXI_GP5 {{{1}}} \\",
            f"  CONFIG.PSU__USE__IRQ0 {{{0}}} \\",
            "] [get_bd_cells zynq_ultra_ps_e_0]",
        ]
    )

    # Add the HLS IP block
    lines.append(
        f'create_bd_cell -type ip -vlnv polito.nn2FPGA:ml:{top_name}:1.0 {top_name}_0'
    )

    buffer_master_interfaces = []
    for buffer in buffers:
        buffer_master_interfaces.append((f"{buffer}_read", f"{top_name}_0/m_axi_{buffer}_read_bundle"))
        buffer_master_interfaces.append((f"{buffer}_write", f"{top_name}_0/m_axi_{buffer}_write_bundle"))

    ddr_masters = []
    for input, _, is_static in inputs:
        ddr_masters.append({"net_name": f"{input}_maxi", "interface": f"{input}_dma/M_AXI_MM2S", "kind": "data"})
        if not is_static:
            ddr_masters.append({"net_name": f"{input}_sg_maxi", "interface": f"{input}_dma/M_AXI_SG", "kind": "sg"})
    for output, _ in outputs:
        ddr_masters.append({"net_name": f"{output}_maxi", "interface": f"{output}_dma/M_AXI_S2MM", "kind": "data"})
        ddr_masters.append({"net_name": f"{output}_sg_maxi", "interface": f"{output}_dma/M_AXI_SG", "kind": "sg"})
    ddr_masters.extend(
        {"net_name": f"{buffer}_maxi", "interface": interface, "kind": "data"}
        for buffer, interface in buffer_master_interfaces
    )
    hp_groups = split_ddr_masters(ddr_masters)

    # Add the Process System Reset
    lines.append(
        f'create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 proc_sys_reset_0'
    )

    # Add DMAs
    for input, _, is_static in inputs:
        lines.append(
            f'create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 {input}_dma'
        )
        sg_options = " CONFIG.C_SG_INCLUDE_STSCNTRL_STRM {0}" if not is_static else ""
        lines.append(
            f'set_property -dict [list CONFIG.C_INCLUDE_MM2S {{{1}}} CONFIG.C_INCLUDE_S2MM {{{0}}} CONFIG.C_INCLUDE_SG {{{0 if is_static else 1}}} CONFIG.C_SG_LENGTH_WIDTH {{{26}}}{sg_options}] [get_bd_cells {input}_dma]'
        )
        lines.append(f'set_property -dict [list CONFIG.C_M_AXI_MM2S_DATA_WIDTH {{{interface_width}}} CONFIG.C_M_AXIS_MM2S_TDATA_WIDTH {{{interface_width}}}] [get_bd_cells {input}_dma]')

    for output, _ in outputs:
        lines.append(
            f'create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 {output}_dma'
        )
        lines.append(
            f"set_property -dict [list CONFIG.C_INCLUDE_MM2S {{{0}}} CONFIG.C_INCLUDE_S2MM {{{1}}} CONFIG.C_INCLUDE_SG {{{1}}} CONFIG.C_SG_LENGTH_WIDTH {{{26}}} CONFIG.C_SG_INCLUDE_STSCNTRL_STRM {{{0}}}] [get_bd_cells {output}_dma]"
        )
        lines.append(f'set_property -dict [list CONFIG.C_M_AXI_S2MM_DATA_WIDTH {{{interface_width}}} CONFIG.C_S_AXIS_S2MM_TDATA_WIDTH {{{interface_width}}} CONFIG.c_include_s2mm_dre {{1}}] [get_bd_cells {output}_dma]')

    # Add design ID register (AXI GPIO + xlconstant)
    lines.append(f'create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio:2.0 axi_gpio_id')
    lines.append(f'set_property -dict [list CONFIG.C_GPIO_WIDTH {{32}} CONFIG.C_ALL_INPUTS {{1}} CONFIG.C_IS_DUAL {{0}}] [get_bd_cells axi_gpio_id]')
    lines.append(f'create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 xlconst_id')
    lines.append(f'set_property -dict [list CONFIG.CONST_WIDTH {{32}} CONFIG.CONST_VAL {{{design_id}}}] [get_bd_cells xlconst_id]')
    lines.append(f'connect_bd_net [get_bd_pins xlconst_id/dout] [get_bd_pins axi_gpio_id/gpio_io_i]')

    # Add smartconnect for AXI lite interfaces
    has_control_axilite = len(buffers) > 0
    axilite_mi_count = len(inputs) + len(outputs) + 1 + int(has_control_axilite)
    lines.append(f'create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smartconnect_axilite_0')
    lines.append(f'set_property -dict [list CONFIG.NUM_SI {{{1}}} CONFIG.NUM_MI {{{axilite_mi_count}}}] [get_bd_cells smartconnect_axilite_0]')

    # Add SmartConnects for DDR AXI masters. Each SmartConnect has at most 16 SI.
    for hp_idx, group in enumerate(hp_groups):
        if not group:
            continue
        lines.append(f'create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smartconnect_hp_{hp_idx}')
        lines.append(f'set_property -dict [list CONFIG.NUM_SI {{{len(group)}}} CONFIG.NUM_MI {{{1}}}] [get_bd_cells smartconnect_hp_{hp_idx}]')

    # Connect clock to every block
    lines.append(f'connect_bd_net -net ps_clk [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins {top_name}_0/ap_clk]')
    lines.append(f'connect_bd_net -net ps_clk [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins proc_sys_reset_0/slowest_sync_clk]')
    lines.append(f'connect_bd_net -net ps_clk [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins smartconnect_axilite_0/aclk]')
    lines.append(f'connect_bd_net -net ps_clk [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins zynq_ultra_ps_e_0/maxihpm0_fpd_aclk]')
    lines.append(f'connect_bd_net -net ps_clk [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins zynq_ultra_ps_e_0/saxihp0_fpd_aclk]')
    lines.append(f'connect_bd_net -net ps_clk [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins zynq_ultra_ps_e_0/saxihp1_fpd_aclk]')
    lines.append(f'connect_bd_net -net ps_clk [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins zynq_ultra_ps_e_0/saxihp2_fpd_aclk]')
    lines.append(f'connect_bd_net -net ps_clk [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins zynq_ultra_ps_e_0/saxihp3_fpd_aclk]')
    lines.append(f'connect_bd_net -net ps_clk [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins axi_gpio_id/s_axi_aclk]')
    for hp_idx, group in enumerate(hp_groups):
        if group:
            lines.append(f'connect_bd_net -net ps_clk [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins smartconnect_hp_{hp_idx}/aclk]')
    for input, _, is_static in inputs:
        lines.append(f'connect_bd_net -net ps_clk [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins {input}_dma/s_axi_lite_aclk]')
        lines.append(f'connect_bd_net -net ps_clk [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins {input}_dma/m_axi_mm2s_aclk]')
        if not is_static:
            lines.append(f'connect_bd_net -net ps_clk [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins {input}_dma/m_axi_sg_aclk]')
    for output, _ in outputs:
        lines.append(f'connect_bd_net -net ps_clk [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins {output}_dma/s_axi_lite_aclk]')
        lines.append(f'connect_bd_net -net ps_clk [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins {output}_dma/m_axi_s2mm_aclk]')
        lines.append(f'connect_bd_net -net ps_clk [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins {output}_dma/m_axi_sg_aclk]')
    # Connect reset to every block
    lines.append(f'connect_bd_net -net ps_rst [get_bd_pins proc_sys_reset_0/ext_reset_in] [get_bd_pins zynq_ultra_ps_e_0/pl_resetn0]')
    lines.append(f'connect_bd_net -net a_rst [get_bd_pins proc_sys_reset_0/peripheral_aresetn] [get_bd_pins {top_name}_0/ap_rst_n]')
    lines.append(f'connect_bd_net -net a_rst [get_bd_pins proc_sys_reset_0/peripheral_aresetn] [get_bd_pins smartconnect_axilite_0/aresetn]')
    lines.append(f'connect_bd_net -net a_rst [get_bd_pins proc_sys_reset_0/peripheral_aresetn] [get_bd_pins axi_gpio_id/s_axi_aresetn]')
    for hp_idx, group in enumerate(hp_groups):
        if group:
            lines.append(f'connect_bd_net -net a_rst [get_bd_pins proc_sys_reset_0/peripheral_aresetn] [get_bd_pins smartconnect_hp_{hp_idx}/aresetn]')
    for input, _, _ in inputs:
        lines.append(f'connect_bd_net -net a_rst [get_bd_pins proc_sys_reset_0/peripheral_aresetn] [get_bd_pins {input}_dma/axi_resetn]')
    for output, _ in outputs:
        lines.append(f'connect_bd_net -net a_rst [get_bd_pins proc_sys_reset_0/peripheral_aresetn] [get_bd_pins {output}_dma/axi_resetn]')

    # Connect AXI lite interfaces to the smartconnect
    for i, (input, _, _) in enumerate(inputs):
        lines.append(f'connect_bd_intf_net -intf_net {input}_axi_lite [get_bd_intf_pins {input}_dma/S_AXI_LITE] [get_bd_intf_pins smartconnect_axilite_0/M0{i}_AXI]')
    for i, (output, _) in enumerate(outputs):
        lines.append(f'connect_bd_intf_net -intf_net {output}_axi_lite [get_bd_intf_pins {output}_dma/S_AXI_LITE] [get_bd_intf_pins smartconnect_axilite_0/M0{i + len(inputs)}_AXI]')
    lines.append(f'connect_bd_intf_net -intf_net axi_gpio_id_axilite [get_bd_intf_pins axi_gpio_id/S_AXI] [get_bd_intf_pins smartconnect_axilite_0/M0{len(inputs)+len(outputs)}_AXI]')
    if has_control_axilite:
        lines.append(f'connect_bd_intf_net -intf_net {top_name}_control_axilite [get_bd_intf_pins {top_name}_0/s_axi_control] [get_bd_intf_pins smartconnect_axilite_0/M0{len(inputs)+len(outputs)+1}_AXI]')

    # Connect SmartConnect AXI interfaces to the PS
    lines.append(f'connect_bd_intf_net -intf_net ps_axilite [get_bd_intf_pins zynq_ultra_ps_e_0/M_AXI_HPM0_FPD] [get_bd_intf_pins smartconnect_axilite_0/S00_AXI]')

    # Connect HLS IP streams to the DMAs
    for input, _, _ in inputs:
        lines.append(f'connect_bd_intf_net -intf_net {input}_axis [get_bd_intf_pins {top_name}_0/{input}] [get_bd_intf_pins {input}_dma/M_AXIS_MM2S]')
    for output, _ in outputs:
        lines.append(f'connect_bd_intf_net -intf_net {output}_axis [get_bd_intf_pins {top_name}_0/{output}] [get_bd_intf_pins {output}_dma/S_AXIS_S2MM]')

    # Connect DMA and HLS master AXI interfaces to PS DDR through balanced HP groups.
    for hp_idx, group in enumerate(hp_groups):
        if not group:
            continue
        for si_idx, master in enumerate(group):
            lines.append(
                f'connect_bd_intf_net -intf_net {master["net_name"]} '
                f'[get_bd_intf_pins smartconnect_hp_{hp_idx}/S{si_idx:02d}_AXI] '
                f'[get_bd_intf_pins {master["interface"]}]'
            )
        lines.append(
            f'connect_bd_intf_net -intf_net ps_hp_{hp_idx} '
            f'[get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HP{hp_idx}_FPD] '
            f'[get_bd_intf_pins smartconnect_hp_{hp_idx}/M00_AXI]'
        )

    # Assign addresses to the PS interfaces
    lines.append(f'assign_bd_address')

    # Delete existing address segments for DMAs
    for input, _, _ in inputs:
        lines.append(f'delete_bd_objs [get_bd_addr_segs {{zynq_ultra_ps_e_0/Data/SEG_{input}_dma_Reg}}]')
    for output, _ in outputs:
        lines.append(f'delete_bd_objs [get_bd_addr_segs {{zynq_ultra_ps_e_0/Data/SEG_{output}_dma_Reg}}]')

    # Reduce the axi_lite range of DMAs to 4 KiB
    axilite_window_str = f"{int(axilite_dma_window // 1024)}K"
    lines.append(
        f"assign_bd_address -offset 0x{axilite_base_addr:X} -range {axilite_window_str} "
        f"-target_address_space /zynq_ultra_ps_e_0/Data [get_bd_addr_segs axi_gpio_id/S_AXI/Reg] -force"
    )
    for input, offset, _ in inputs:
        lines.append(
            f"assign_bd_address -offset 0x{(axilite_base_addr + offset):X} -range {axilite_window_str} -target_address_space /zynq_ultra_ps_e_0/Data [get_bd_addr_segs {input}_dma/S_AXI_LITE/Reg] -force"
        )
    for output, offset in outputs:
        lines.append(
            f"assign_bd_address -offset 0x{(axilite_base_addr + offset):X} -range {axilite_window_str} -target_address_space /zynq_ultra_ps_e_0/Data [get_bd_addr_segs {output}_dma/S_AXI_LITE/Reg] -force"
        )
    if has_control_axilite:
        lines.append(
            f"assign_bd_address -offset 0x{(axilite_base_addr + control_axi_offset):X} -range {axilite_window_str} -target_address_space /zynq_ultra_ps_e_0/Data [get_bd_addr_segs {top_name}_0/s_axi_control/Reg] -force"
        )

    # Validate the block design
    lines.append(f'validate_bd_design')

    # Save the block design
    lines.append(f'save_bd_design')

    # Disable OOC synthesis for the BD
    lines.append(
        f'set_property synth_checkpoint_mode None '
        f'[get_files {work_dir}/vivadoproj/vivadoproj.srcs/sources_1/bd/{top_name}_bd/{top_name}_bd.bd]'
    )

    # Generate GLOBAL (in-context) targets for the BD
    lines.append(
        f'generate_target all '
        f'[get_files {work_dir}/vivadoproj/vivadoproj.srcs/sources_1/bd/{top_name}_bd/{top_name}_bd.bd]'
    )

    # Make wrapper
    lines.append(f'make_wrapper -files [get_files {work_dir}/vivadoproj/vivadoproj.srcs/sources_1/bd/{top_name}_bd/{top_name}_bd.bd] -top')
    lines.append(f'add_files -norecurse {work_dir}/vivadoproj/vivadoproj.gen/sources_1/bd/{top_name}_bd/hdl/{top_name}_bd_wrapper.v')
    lines.append(f'set_property top {top_name}_bd [current_fileset]')
    lines.append(f'update_compile_order -fileset sources_1')

    # Launch synthesis
    lines.append(f'set_property strategy Flow_AreaOptimized_high [get_runs synth_1]')
    lines.append(f'launch_runs synth_1 -jobs 8')
    lines.append(f'wait_on_run synth_1')

    # Launch implementation
    lines.append(f'set_property strategy Congestion_SpreadLogic_high [get_runs impl_1]')
    lines.append(f'launch_runs impl_1 -to_step write_bitstream -jobs 8')
    lines.append(f'wait_on_run impl_1')

    return "\n".join(lines)

def make_build_dir(work_dir: str) -> None:
    """Create the working directory for the simulation."""
    os.makedirs(work_dir, exist_ok=True)

class GenerateBitstream(Transformation):

    def __init__(
        self,
        work_dir: str,
        erase: bool = True,
        axilite_dma_window: int = 4096,
        only_synthesize: bool = False,
        already_exported: bool = False,
        vivado_already_done: bool = False,
    ):
        super().__init__()
        self.work_dir = work_dir
        self.erase = erase
        self.only_synthesize = only_synthesize
        self.already_exported = already_exported
        self.vivado_already_done = vivado_already_done

        # Check axilite_dma_window is a power of two
        if axilite_dma_window & (axilite_dma_window - 1) != 0:
            raise ValueError("axilite_dma_window must be a power of two.")

        # Check axilite has not an unreasonable size.
        # This is not a strict requirement, probably DMA can even fit in less space.
        if axilite_dma_window < 1024 or axilite_dma_window > 65536:
            raise ValueError("axilite_dma_window must be between 1K and 64K bytes.")

        # Set the axilite parameters
        self.axilite_dma_window = axilite_dma_window

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:

        partition_node = model.get_nodes_by_op_type("nn2fpgaPartition")[0]
        ap = AcceleratorPackage.from_json(getCustomOp(partition_node).get_nodeattr("accelerator_package"))
        work_dir = ap.work_dir
        work_dir = f"{os.path.abspath(work_dir)}/vivado"
        make_build_dir(work_dir)

        top_name = model.get_metadata_prop("top_name")
        board = model.get_metadata_prop("board_name")
        frequency = model.get_metadata_prop("frequency")
        hls_version = model.get_metadata_prop("hls_version")
        axilite_size = int(model.get_metadata_prop("axilite_size"))
        axilite_address = int(model.get_metadata_prop("axilite_address"))
        silvia_packing = str(model.get_metadata_prop("silvia_packing")).lower() == "true"
        design_id = model.get_metadata_prop("design_id")
        interface_width = read_board_info(board)["axi_bitwidth"]
        part_name = read_board_info(board)["part"]
        board_part_name = read_board_info(board)["board_part"]
        hls_output_dir = Path(work_dir) / PROJECT_NAME / SOLUTION_NAME
        if float(model.get_metadata_prop("hls_version")) > 2025:
            hls_output_dir = Path(work_dir) / PROJECT_NAME / "hls"

        axilite_windows = 1 + len(ap.input_map) + len(ap.output_map)
        if ap.buffer_map:
            axilite_windows += 1
        total_axilite_size = axilite_windows * self.axilite_dma_window
        if total_axilite_size > axilite_size:
            raise ValueError(
                f"Total AXI lite size ({total_axilite_size}) exceeds the maximum allowed size ({axilite_size})."
            )

        if not self.already_exported:

            # Write the HLS code to a file
            with open(f"{work_dir}/kernel.cpp", "w") as f:
                f.write(base64.b64decode(ap.hls_code_b64).decode())

            # Generate the TCL script
            tcl_script = dump_tcl_script(
                top_name=top_name,
                part_name=part_name,
                frequency=frequency,
                hls_version=hls_version,
                reset=True,
                synthesize=True,
                silvia_packing=silvia_packing,
                export=not self.only_synthesize,
            )

            # Write the TCL script to a file
            with open(f"{work_dir}/setup.tcl", "w") as f:
                f.write(tcl_script)

            # Synthesize the design.
            vitis_command = ["vitis_hls", "-f", f"{work_dir}/setup.tcl"]
            if float(model.get_metadata_prop("hls_version")) > 2025:
                vitis_command = [
                    "vitis-run",
                    "--mode",
                    "hls",
                    "--tcl",
                    f"{work_dir}/setup.tcl",
                ]

            subprocess.run(vitis_command, cwd=work_dir, check=True)

            if self.only_synthesize:
                return model, False

        recover_buffer_axilite_offsets(hls_output_dir, top_name, ap.buffer_map)

        if not self.already_exported:
            # Patch the Verilog files to optimize RAM usage
            process_verilog_files(hls_output_dir / "impl/verilog")
            process_verilog_files(hls_output_dir / "impl/ip/hdl/verilog")
            buffer_names = list(ap.buffer_map.keys())
            gate_entry_proc_on_configured_buffers(hls_output_dir / "impl/verilog", top_name, buffer_names)
            gate_entry_proc_on_configured_buffers(hls_output_dir / "impl/ip/hdl/verilog", top_name, buffer_names)

        # Save the design ID in a constant register of the BD,
        # and reserve the axilite space.
        axi_offset = 0x0
        axi_offset += self.axilite_dma_window  # Reserve 4K for design ID

        # Retrieve input list and assign AXI offsets
        input_list = []
        inputs = ap.input_map
        for value in inputs.values():
            value["axi_offset"] = axi_offset
            input_list.append((value["new_name"], axi_offset, value["value"] is not None))
            axi_offset += self.axilite_dma_window

        # Retrieve output list
        output_list = []
        outputs = ap.output_map
        for value in outputs.values():
            value["axi_offset"] = axi_offset
            output_list.append((value["new_name"], axi_offset))
            axi_offset += self.axilite_dma_window

        buffer_list = list(ap.buffer_map.keys())
        control_axi_offset = None
        if buffer_list:
            control_axi_offset = axi_offset
            axi_offset += self.axilite_dma_window
        model.set_metadata_prop("control_axi_offset", str(control_axi_offset or 0))

        if self.vivado_already_done:
            logger.info("Skipping Vivado synthesis/implementation because vivado_already_done=True.")
        else:
            # Write the Vivado block design.
            with open(f"{work_dir}/vivado.tcl", "w") as f:
                f.write(
                    vivado_tcl_script(
                        work_dir=work_dir,
                        top_name=top_name,
                        part_name=part_name,
                        board_part_name=board_part_name,
                        frequency=frequency,
                        hls_version=hls_version,
                        axilite_base_addr=axilite_address,
                        axilite_dma_window=self.axilite_dma_window,
                        interface_width=interface_width,
                        design_id=design_id,
                        inputs=input_list,
                        outputs=output_list,
                        buffers=buffer_list,
                        control_axi_offset=control_axi_offset,
                    )
                )

            # Run Vivado to generate the bitstream.
            subprocess.run(
                ["vivado", "-mode", "batch", "-source", f"{work_dir}/vivado.tcl"],
                cwd=work_dir,
                check=True
            )

        # Check if the bitstream was generated successfully.
        bitstream_path = f"{work_dir}/vivadoproj/vivadoproj.runs/impl_1/{top_name}_bd.bit"
        if not os.path.exists(bitstream_path):
            raise RuntimeError(f"Bitstream generation failed: {bitstream_path} does not exist.")

        # Set the bitstream in the accelerator package.
        ap.set_bitstream(bitstream_path)

        # Check the HWH file.
        hwh_path = f"{work_dir}/vivadoproj/vivadoproj.gen/sources_1/bd/{top_name}_bd/hw_handoff/{top_name}_bd.hwh"
        if not os.path.exists(hwh_path):
            raise RuntimeError(f"HWH file generation failed: {hwh_path} does not exist.")

        # Set the HWH file in the accelerator package.
        ap.set_hwh(hwh_path)

        # Update the accelerator package in the partition node.
        getCustomOp(partition_node).set_nodeattr("accelerator_package", ap.to_json())

        return model, False

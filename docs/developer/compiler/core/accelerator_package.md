# AcceleratorPackage

## Overview

`AcceleratorPackage` is a dataclass that encapsulates all artifacts and metadata required to represent an FPGA accelerator in a portable and serializable form.

It serves as a unified container for:

* Generated HLS source code
* Compiled FPGA bitstream
* Hardware handoff description (`.hwh`)
* Input/output tensor name mappings
* DDR-backed buffer metadata
* Build and platform configuration

This enables the accelerator to be serialized (e.g., to JSON), stored, transferred, and reconstructed across different stages of the compilation and deployment pipeline.

---

## Source location

```python
nn2fpga/compiler/core/acceleratorpackage.py
```

---

## Purpose

The class is designed to:

* Aggregate all accelerator-related artifacts into a single object
* Provide serialization/deserialization support
* Encode binary and text assets safely using base64
* Preserve metadata required for simulation and deployment

---

## Data Fields

### Encoded Hardware Artifacts

#### `hls_code_b64`

Base64-encoded generated HLS source code. The value is text encoded as base64 so it can be stored safely in JSON.

* Use `set_hls_code()` to store plain text
* Use `get_hls_code()` to retrieve decoded text
* Populated when the compiler emits the HLS top-level source

#### `bitstream_b64`

Base64-encoded FPGA `.bit` file. The value is binary data encoded as base64.

* Use `set_bitstream()` to load from a file
* Use `get_bitstream()` to retrieve raw binary data
* Populated after Vivado bitstream generation succeeds

#### `hwh_b64`

Base64-encoded Vivado hardware handoff (`.hwh`) content. The value is text encoded as base64.

* Use `set_hwh()` to load from a file
* Use `get_hwh()` to retrieve decoded text
* Populated after Vivado generates the block design handoff file

---

### Tensor Interface Metadata

#### Input Map Schema

Input tensors are described with the following structure:

```python
input_map = {
    "<onnx_input_name>": {
        "name": "<original_or_internal_name>",
        "new_name": "<dma_and_stream_name>",
        "index": 0,
        "shape": [1, 3, 224, 224],
        "layout": None,
        "quant": "Q[8,0,1.0,0,0,ROUND]",
        "value": None,
        "axi_offset": 0x1000,
    }
}
```

#### Output Map Schema

Output tensors follow the same convention:

```python
output_map = {
    "<onnx_output_name>": {
        "name": "<original_or_internal_name>",
        "new_name": "<dma_and_stream_name>",
        "index": 0,
        "shape": [1, 1000],
        "layout": None,
        "quant": "Q[8,0,1.0,0,0,ROUND]",
        "value": None,
        "axi_offset": 0x2000,
    }
}
```

#### Buffer Map Schema

DDR-backed buffers are described with the following structure:

```python
buffer_map = {
    "<buffer_name>": {
        "hls_type": "ap_uint<128>",
        "depth": "1024",
        "size_bytes": 16384,
        "read_axi_offset": 0x10,
        "write_axi_offset": 0x1C,
        "axi_offset": 0x10,
    }
}
```

#### `input_map`

Dictionary describing input tensors. Since names of the inputs may change during the flow, it is necessary to maintain a mapping between the wrapper model and the nn2FPGA one.

Common fields are:

* `name`: Original or intermediate tensor name retained for traceability.
* `new_name`: Internal accelerator name used for generated streams, DMA instances, and driver references.
* `index`: Input order in the original ONNX interface.
* `shape`: Tensor shape, including batch dimension when present.
* `layout`: Tensor layout metadata.
* `quant`: Data type in TensorQuant format.
* `value`: Static input value for parameters, or `None` for runtime inputs.
* `axi_offset`: Vivado AXI-Lite window offset of the input DMA control interface.

For runtime inputs, generated drivers allocate a DMA input buffer and program the DMA mapped at `axi_offset`. For static inputs, the same metadata identifies initialization data that can be loaded before normal inference.

#### `output_map`

Dictionary describing output tensors.

The structure mirrors `input_map`, but each entry describes an output DMA interface. `axi_offset` is the Vivado AXI-Lite window offset of the output DMA control interface.

#### `buffer_map`

Dictionary describing internal tensors that are materialized in external DDR instead of remaining entirely in on-chip streams.

Each entry is keyed by the buffer name and includes metadata required by HLS, bitstream generation, simulation, and generated drivers.

One logical DDR buffer is exposed to HLS as two pointer arguments:

* `<buffer_name>_read`
* `<buffer_name>_write`

The generated HLS ports use AXI master interfaces to access DDR. Their pointer registers are exposed through the single HLS `s_axi_control` AXI-Lite interface.

Common fields are:

* `hls_type`: C++/HLS type string used by the generated pointer arguments.
* `depth`: Number of HLS elements in the buffer, currently stored as a string in generated metadata.
* `size_bytes`: Integer buffer size in bytes, used by generated PYNQ tests to allocate byte-addressed memory.
* `read_axi_offset`: Integer byte offset of `<buffer_name>_read` in the HLS `s_axi_control` register map.
* `write_axi_offset`: Integer byte offset of `<buffer_name>_write` in the HLS `s_axi_control` register map.
* `axi_offset`: Compatibility alias for `read_axi_offset`; new code should prefer `read_axi_offset` and `write_axi_offset`.

`read_axi_offset` and `write_axi_offset` are relative register offsets inside the HLS IP `s_axi_control` register map. They are not Vivado address-window offsets and they are not physical addresses. Vivado assigns one AXI-Lite address window to the whole HLS `s_axi_control` interface. In PYNQ, writes such as `ol.<top_name>_0.write(offset, value)` use these relative offsets because PYNQ already maps the IP base address.

The read and write pointer offsets are recovered after Vitis HLS export from the generated Linux driver header:

```text
<hls_output_dir>/impl/ip/drivers/<top_name>_v1_0/src/x<top_name>_hw.h
```

For example, Vitis HLS may emit:

```c
#define XRESNET8_CONTROL_ADDR_BUFFER_0_READ_DATA  0x10
#define XRESNET8_CONTROL_ADDR_BUFFER_0_WRITE_DATA 0x1c
```

`GenerateBitstream` parses these macros and stores the values in `read_axi_offset` and `write_axi_offset`. Offset recovery fails if either register offset is missing, because generating a driver with guessed pointer offsets would be unsafe.

### Build and Target Metadata

Build and target metadata is serialized with the package so later transforms and deployment tools do not need to rely only on transient model metadata.

```python
work_dir = "work/resnet8"
board_name = "KRIA"
top_name = "resnet8"
frequency = "200"
hls_version = "2025.1"
simulation = "csim"
```

#### `work_dir`

Working directory associated with the accelerator build. Generated HLS, Vivado, and deployment artifacts are placed under this tree.

#### `board_name`

Target FPGA board identifier used to recover board-specific part names, board part names, and AXI interface widths.

#### `top_name`

Name of the HLS top function and generated accelerator IP.

#### `frequency`

Target accelerator clock frequency in MHz.

#### `hls_version`

Version of the HLS tool used to select the correct Vitis HLS invocation and output directory layout.

#### `simulation`

Simulation mode/type, for example `csim` or `cosim`.

---

## Buffer Metadata Lifecycle

1. `OptimizeFifo` selects streams to move to DDR and creates `buffer_map` entries with `hls_type`, `depth`, and `size_bytes`.
2. HLS code generation emits `<buffer_name>_read` and `<buffer_name>_write` pointer arguments plus AXI master pragmas.
3. Vitis HLS export generates the Linux driver header containing the `s_axi_control` register map.
4. `GenerateBitstream` parses the header and updates `buffer_map` with `read_axi_offset`, `write_axi_offset`, and `axi_offset`.
5. Driver generation uses the recovered offsets to program DDR buffer addresses.

---

## Required Fields

The class defines a set of required fields:

```python
REQUIRED_FIELDS = {
    "hls_code_b64",
    "bitstream_b64",
    "hwh_b64",
    "input_map",
    "output_map",
    "buffer_map",
    "work_dir",
    "board_name",
    "top_name",
    "frequency",
    "hls_version",
    "simulation",
}
```

When loading from JSON, all fields must be present.

Required means that the JSON key must exist for reconstruction. Some fields may be empty during earlier compiler stages. For example, `bitstream_b64` and `hwh_b64` are expected to be empty before bitstream generation.

Missing fields result in:

```python
ValueError
```

---

## Serialization API

### `to_json()`

Serializes the dataclass fields into a JSON string. The method writes the current field values as-is; it does not validate nested map schemas.

Used for:

* Saving accelerator packages
* Transferring between components

---

### `from_json(json_str)`

Reconstructs an `AcceleratorPackage` from JSON. The method validates required top-level field presence, but it does not validate inner map schemas or field types.

Steps:

1. Parse JSON
2. Validate required fields
3. Instantiate object

Raises:

* `ValueError` if required fields are missing

---

## Base64 Helper API

### `set_hls_code(code)`

Encodes plain-text HLS code into base64.

### `get_hls_code()`

Returns decoded HLS code as a string.

---

### `set_bitstream(path)`

Reads a binary `.bit` file and encodes it into base64.

### `get_bitstream()`

Returns decoded bitstream as `bytes`.

---

### `set_hwh(path)`

Reads a `.hwh` file and encodes it into base64.

### `get_hwh()`

Returns decoded `.hwh` content as text.

---

## Example

```python
from nn2fpga.compiler.core.acceleratorpackage import AcceleratorPackage

# Buffer offsets are recovered from the Vitis HLS generated x<top_name>_hw.h header.
pkg = AcceleratorPackage(
    input_map={
        "in0": {
            "name": "input_0",
            "new_name": "global_in",
            "index": 0,
            "shape": [1, 3, 224, 224],
            "layout": None,
            "quant": "Q[8,0,1.0,0,0,ROUND]",
            "value": None,
            "axi_offset": 0x1000,
        }
    },
    output_map={
        "out0": {
            "name": "output_0",
            "new_name": "global_out",
            "index": 0,
            "shape": [1, 1000],
            "layout": None,
            "quant": "Q[8,0,1.0,0,0,ROUND]",
            "value": None,
            "axi_offset": 0x2000,
        }
    },
    buffer_map={
        "buffer_0": {
            "hls_type": "ap_uint<128>",
            "depth": "1024",
            "size_bytes": 16384,
            "read_axi_offset": 0x10,
            "write_axi_offset": 0x1C,
            "axi_offset": 0x10,
        }
    },
    work_dir="work/resnet50",
    board_name="KRIA",
    top_name="resnet8",
    frequency="200",
    hls_version="2025.1",
    simulation="csim",
)

json_blob = pkg.to_json()
```

---

## Notes

### Serialization

* All artifacts are encoded in base64 to ensure safe JSON serialization
* `set_bitstream()` handles binary data, while others operate on text
* `from_json()` validates presence of fields but does not enforce schema on maps

### AXI-Lite Offsets

* Input and output `axi_offset` values identify Vivado AXI-Lite windows for DMA control interfaces
* `read_axi_offset` and `write_axi_offset` are offsets relative to the HLS `s_axi_control` register map
* Generated drivers should not assume a fixed distance between read and write pointer registers

### Compatibility

* `axi_offset` is preserved for compatibility but should not be used by new code when both explicit offsets are available

---

## Limitations

* `input_map`, `output_map`, and `buffer_map` are loosely typed (`Dict[str, Any]`)
* Field types are not enforced by the dataclass
* Documented map schemas are not enforced at construction time
* Offset recovery depends on Vitis HLS generating the expected driver header
* No validation of tensor metadata structure
* No size checks for embedded binary data
* No compression of large artifacts

---

## Possible Improvements

* Introduce structured dataclasses for input, output, and buffer metadata
* Add schema validation for `input_map`, `output_map`, and `buffer_map`
* Add explicit metadata versioning for serialized packages
* Support compression before base64 encoding
* Add helpers to export decoded artifacts to disk
* Add helpers for buffer lookup and AXI-Lite offset validation
* Add validation for field values (not just presence)

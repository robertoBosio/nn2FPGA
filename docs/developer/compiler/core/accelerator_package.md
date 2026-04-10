# AcceleratorPackage

## Overview

`AcceleratorPackage` is a dataclass that encapsulates all artifacts and metadata required to represent an FPGA accelerator in a portable and serializable form.

It serves as a unified container for:

* Generated HLS source code
* Compiled FPGA bitstream
* Hardware handoff description (`.hwh`)
* Input/output tensor name mappings
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

Base64-encoded HLS source code.

* Use `set_hls_code()` to store plain text
* Use `get_hls_code()` to retrieve decoded text

#### `bitstream_b64`

Base64-encoded FPGA bitstream.

* Use `set_bitstream()` to load from a file
* Use `get_bitstream()` to retrieve raw binary data

#### `hwh_b64`

Base64-encoded `.hwh` file content.

* Use `set_hwh()` to load from a file
* Use `get_hwh()` to retrieve decoded text

---

### Tensor Interface Metadata

#### `input_map`

Dictionary describing input tensors. Since names of the inputs may change during the flow, it is necessary to maintain a mapping between the wrapper model and the nn2FPGA one.

It includes:

* Original tensor names.
* Internal accelerator names.
* Shapes.
* Data types in the TensorQuant format.
* Values in the case of static inputs (e.g. parameters).

#### `output_map`

Dictionary describing output tensors.

Same structure and purpose as `input_map`.

---

### Build and Target Metadata

#### `work_dir`

Working directory associated with the accelerator build.

#### `board_name`

Target FPGA board.

#### `top_name`

Name of the HLS top function.

#### `frequency`

Target clock frequency (MHz).

#### `hls_version`

Version of the HLS tool used.

#### `simulation`

Simulation mode/type (e.g., `csim`, `cosim`).

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
    "work_dir",
    "board_name",
    "top_name",
    "frequency",
    "hls_version",
    "simulation",
}
```

When loading from JSON, all fields must be present.

Missing fields result in:

```python
ValueError
```

---

## Serialization API

### `to_json()`

Serializes the object into a JSON string.

Used for:

* Saving accelerator packages
* Transferring between components

---

### `from_json(json_str)`

Reconstructs an `AcceleratorPackage` from JSON.

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

pkg = AcceleratorPackage(
    input_map={
        "in0": {
            "name": "input_0",
            "index": 0,
            "shape": [1, 3, 224, 224],
            "quant": None,
            "value": None,
        }
    },
    output_map={
        "out0": {
            "name": "output_0",
            "index": 0,
            "shape": [1, 1000],
            "quant": None,
            "value": None,
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

* All artifacts are encoded in base64 to ensure safe JSON serialization
* `set_bitstream()` handles binary data, while others operate on text
* `from_json()` validates presence of fields but does not enforce schema on maps

---

## Limitations

* `input_map` and `output_map` are loosely typed (`Dict[str, Any]`)
* No validation of tensor metadata structure
* No size checks for embedded binary data
* No compression of large artifacts

---

## Possible Improvements

* Introduce structured types for tensor metadata
* Add schema validation for input/output maps
* Support compression before base64 encoding
* Add helpers to export decoded artifacts to disk
* Add validation for field values (not just presence)

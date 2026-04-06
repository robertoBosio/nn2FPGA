import base64
import json
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, ClassVar


@dataclass
class AcceleratorPackage:
    """
    Represents an accelerator package containing all necessary information for simulation and compilation.
    This includes HLS code, driver code, bitstream, input/output maps, and other metadata.
    """

    # HLS code of the accelerator, encoded in base64.
    hls_code_b64: str = ""

    # Bitstream of the accelerator, encoded in base64.
    bitstream_b64: str = ""

    # .hwh file of the accelerator, encoded in base64.
    hwh_b64: str = ""

    # Mapping of input tensor names to their metadata.
    # The dictionary contains the mapping between the original tensor names and the new names used inside the accelerator,
    # as well as their shapes and data types.
    input_map: Dict[str, Any] = field(default_factory=dict)

    # Mapping of output tensor names to their metadata.
    # The dictionary contains the mapping between the original tensor names and the new names used inside the accelerator,
    # as well as their shapes and data types.
    output_map: Dict[str, Any] = field(default_factory=dict)

    # Working directory for the accelerator.
    work_dir: str = ""

    # Target board name.
    board_name: str = ""

    # HLS top function name.
    top_name: str = ""

    # Target clock frequency in MHz.
    frequency: str = ""

    # Version of HLS used for compilation.
    hls_version: str = ""

    REQUIRED_FIELDS: ClassVar[set] = {
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
    }

    # --- Serialization ---
    @classmethod
    def from_json(cls, json_str: str) -> "AcceleratorPackage":
        """Load from a JSON string and validate required fields."""
        data = json.loads(json_str)
        missing = cls.REQUIRED_FIELDS - data.keys()
        if missing:
            raise ValueError(f"Missing required fields: {missing}")
        return cls(**data)

    def to_json(self) -> str:
        """Dump to JSON string."""
        return json.dumps(asdict(self))

    # --- Base64 helpers ---
    def get_hls_code(self) -> str:
        """Get decoded HLS code as text."""
        return base64.b64decode(self.hls_code_b64).decode()

    def set_hls_code(self, code: str) -> None:
        """Set HLS code from text (will encode to base64)."""
        self.hls_code_b64 = base64.b64encode(code.encode()).decode()

    def get_bitstream(self) -> bytes:
        """Get decoded bitstream as binary data."""
        return base64.b64decode(self.bitstream_b64)

    def set_bitstream(self, bitstream_path: str) -> None:
        """Set bitstream from binary data (will encode to base64)."""
        with open(bitstream_path, "rb") as f:
            bitstream = f.read()
        self.bitstream_b64 = base64.b64encode(bitstream).decode()
    
    def get_hwh(self) -> str:
        """Get decoded .hwh file as text."""
        return base64.b64decode(self.hwh_b64).decode()

    def set_hwh(self, hwh_path: str) -> None:
        """Set .hwh file from text (will encode to base64)."""
        with open(hwh_path, "r") as f:
            hwh_content = f.read()
        self.hwh_b64 = base64.b64encode(hwh_content.encode()).decode()

import re
import base64
from typing import Optional
from qonnx.util import basic as qonnx_basic
from onnx import TensorAnnotation, StringStringEntryProto
from qonnx.core.modelwrapper import ModelWrapper

def _b64url_nopad_encode(s: str) -> str:
    """Base64url encode without '=' padding."""
    return base64.urlsafe_b64encode(s.encode("utf-8")).decode("ascii").rstrip("=")

def _b64url_nopad_decode(s: str) -> str:
    """Decode base64url string that may be missing '=' padding."""
    pad = (-len(s)) % 4
    if pad:
        s = s + ("=" * pad)
    return base64.urlsafe_b64decode(s.encode("ascii")).decode("utf-8")

class TensorFifo:
    """
    FIFO stream metadata:
      - depth: int
      - hls_type: arbitrary string (stored as base64url without padding)
      - n_array: int (cardinality of the array)

    Canonical encoding:
      FIFO_META[depth=<int>,hls_type_b64url=<b64url-no-pad>,n_array=<int>]
    """
    def __init__(self, depth: int, hls_type: str = "", n_array: int = 1):
        self.depth = int(depth)
        self.hls_type = str(hls_type)
        self.n_array = int(n_array)

    def get_canonical_name(self) -> str:
        hls_b64 = _b64url_nopad_encode(self.hls_type) if self.hls_type else ""
        return f"FIFO_META[depth={self.depth},hls_type_b64url={hls_b64},n_array={self.n_array}]"

    @staticmethod
    def from_canonical_name(s: str) -> "TensorFifo":
        m = re.fullmatch(
            r"FIFO_META\[depth=(\d+),hls_type_b64url=([-A-Za-z0-9_]*),n_array=(\d+)\]",
            s,
        )
        if not m:
            raise ValueError(f"Invalid FIFO meta string: {s}")
        depth = int(m.group(1))
        hls_b64 = m.group(2)
        hls_type = _b64url_nopad_decode(hls_b64) if hls_b64 else ""
        n_array = int(m.group(3))
        return TensorFifo(depth=depth, hls_type=hls_type, n_array=n_array)

    def __repr__(self):
        return f"<TensorFifo {self.get_canonical_name()}>"

def set_custom_tensor_fifo_metadata(
    model: ModelWrapper, tensor_name: str, fifo_meta: Optional[TensorFifo]
):
    """
    Store/update FIFO metadata under key 'fifo_meta' inside graph.quantization_annotation.
    If fifo_meta is None, remove the entry for this tensor (if present).
    """
    graph = model._model_proto.graph
    qnt_annotations = graph.quantization_annotation
    ret = qonnx_basic.get_by_name(qnt_annotations, tensor_name, "tensor_name")

    if ret is not None:
        # Find existing kv with key 'fifo_meta'
        idx = next((i for i, kv in enumerate(ret.quant_parameter_tensor_names)
                    if kv.key == "fifo_meta"), None)
        if fifo_meta is None:
            if idx is not None:
                del ret.quant_parameter_tensor_names[idx]
        else:
            if idx is not None:
                ret.quant_parameter_tensor_names[idx].value = fifo_meta.get_canonical_name()
            else:
                fd = StringStringEntryProto()
                fd.key = "fifo_meta"
                fd.value = fifo_meta.get_canonical_name()
                ret.quant_parameter_tensor_names.append(fd)
    elif fifo_meta is not None:
        qa = TensorAnnotation()
        qa.tensor_name = tensor_name
        fd = StringStringEntryProto()
        fd.key = "fifo_meta"
        fd.value = fifo_meta.get_canonical_name()
        qa.quant_parameter_tensor_names.append(fd)
        qnt_annotations.append(qa)

def get_custom_tensor_fifo_metadata(model: ModelWrapper, tensor_name: str) -> Optional[TensorFifo]:
    """
    Read FIFO metadata (depth, hls_type, is_array). Returns None if not present.
    """
    graph = model._model_proto.graph
    qnt_annotations = graph.quantization_annotation
    ret = qonnx_basic.get_by_name(qnt_annotations, tensor_name, "tensor_name")
    if ret is None:
        return None

    kv = qonnx_basic.get_by_name(ret.quant_parameter_tensor_names, "fifo_meta", "key")
    if kv is None or not kv.value:
        return None

    return TensorFifo.from_canonical_name(kv.value)

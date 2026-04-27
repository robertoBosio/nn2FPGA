from qonnx.util import basic as qonnx_basic
from onnx import TensorAnnotation, StringStringEntryProto
from qonnx.core.modelwrapper import ModelWrapper
import re

class TensorLayout:
    """
    Represents the axis ordering of a tensor relative to ONNX canonical order.

    The layout is stored as a permutation tuple, where each element indicates
    which ONNX canonical axis corresponds to the current position.
    For example, (0,2,3,1) means the tensor is stored in NHWC order
    while ONNX canonical is NCHW.

    The layout is stored in a canonical string format:
    L[ax0,ax1,...,axN]

    Where each axI is the ONNX canonical axis index at position I.

    Methods:
        __init__(perm: tuple):
            Initializes a TensorLayout instance with the given permutation.
        from_canonical_name(s: str) -> TensorLayout:
            Parses a canonical layout string and returns a TensorLayout instance.
        __eq__(other) -> bool:
            Checks equality with another TensorLayout instance.
        get_canonical_name() -> str:
            Returns a canonical string representation of the layout.
        is_identity() -> bool:
            Returns True if the layout matches ONNX canonical order.
        inverse() -> TensorLayout:
            Returns the inverse permutation.
        compose(other: TensorLayout) -> TensorLayout:
            Composes this permutation with another.
        __repr__() -> str:
            Returns a string representation of the TensorLayout instance.
    """

    def __init__(self, perm: tuple):
        if not perm:
            raise ValueError("Permutation cannot be empty.")
        if sorted(perm) != list(range(len(perm))):
            raise ValueError(f"Invalid permutation: {perm}. Must be a permutation of 0..N-1.")
        self.perm = tuple(perm)

    @classmethod
    def identity(cls, rank: int) -> "TensorLayout":
        """Returns the identity layout for a tensor of the given rank."""
        return cls(tuple(range(rank)))

    @classmethod
    def from_canonical_name(cls, s: str) -> "TensorLayout":
        m = re.fullmatch(r"L\[(\d+(?:,\d+)*)\]", s)
        if not m:
            raise ValueError(f"Invalid layout annotation string: {s}")
        return cls(tuple(int(x) for x in m.group(1).split(",")))

    def __eq__(self, other):
        if not isinstance(other, TensorLayout):
            return False
        return self.perm == other.perm

    def get_canonical_name(self) -> str:
        return "L[" + ",".join(str(i) for i in self.perm) + "]"

    def is_identity(self) -> bool:
        return self.perm == tuple(range(len(self.perm)))

    def inverse(self) -> "TensorLayout":
        """Returns the inverse permutation layout."""
        inv = [0] * len(self.perm)
        for i, p in enumerate(self.perm):
            inv[p] = i
        return TensorLayout(tuple(inv))

    def compose(self, other: "TensorLayout") -> "TensorLayout":
        """
        Composes this layout with another.
        The result represents applying self first, then other.
        Used to compute the transpose needed between two layouts:
            current_layout.inverse().compose(target_layout)
        """
        if len(self.perm) != len(other.perm):
            raise ValueError("Cannot compose layouts of different ranks.")
        return TensorLayout(tuple(self.perm[i] for i in other.perm))

    def __repr__(self) -> str:
        return f"<TensorLayout {self.get_canonical_name()}>"


def set_custom_tensor_layout(model: ModelWrapper, tensor_name: str, tensor_layout: TensorLayout):
    """Sets the TensorLayout of a tensor with the given name."""
    graph = model._model_proto.graph
    qnt_annotations = graph.quantization_annotation
    ret = qonnx_basic.get_by_name(qnt_annotations, tensor_name, "tensor_name")

    if ret is not None:
        ret_dt = qonnx_basic.get_by_name(ret.quant_parameter_tensor_names, "tensor_layout", "key")
        if ret_dt is not None:
            if tensor_layout is None:
                ret_dt.Clear()
            else:
                ret_dt.value = tensor_layout.get_canonical_name()
        elif tensor_layout is not None:
            dt = StringStringEntryProto()
            dt.key = "tensor_layout"
            dt.value = tensor_layout.get_canonical_name()
            ret.quant_parameter_tensor_names.append(dt)
    elif tensor_layout is not None:
        qa = TensorAnnotation()
        qa.tensor_name = tensor_name
        dt = StringStringEntryProto()
        dt.key = "tensor_layout"
        dt.value = tensor_layout.get_canonical_name()
        qa.quant_parameter_tensor_names.append(dt)
        qnt_annotations.append(qa)


def get_custom_tensor_layout(model: ModelWrapper, tensor_name: str) -> TensorLayout | None:
    """Gets the TensorLayout of a tensor with the given name.
    Returns None if not found.
    """
    graph = model._model_proto.graph
    qnt_annotations = graph.quantization_annotation
    ret = qonnx_basic.get_by_name(qnt_annotations, tensor_name, "tensor_name")
    if ret is None:
        return None

    ret_dt = qonnx_basic.get_by_name(ret.quant_parameter_tensor_names, "tensor_layout", "key")
    if ret_dt is None:
        return None

    try:
        return TensorLayout.from_canonical_name(ret_dt.value)
    except Exception as e:
        raise ValueError(f"Invalid TensorLayout string for tensor {tensor_name}: {ret_dt.value}") from e
    
def require_tensor_layout(model: ModelWrapper, tensor_name: str) -> TensorLayout:
    """Gets the TensorLayout of a tensor with the given name.
    Raises an error if not found.
    """
    layout = get_custom_tensor_layout(model, tensor_name)
    if layout is None:
        raise ValueError(f"Tensor layout for tensor '{tensor_name}' not found in model.")
    return layout
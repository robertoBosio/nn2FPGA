from onnx import NodeProto, helper
from qonnx.custom_op.registry import getCustomOp

def check_par_attributes(node: NodeProto) -> bool:
    """Check if a node has parallelization attributes."""

    par_attributes = get_par_attributes(node)
    return any(par_attributes[attr] is not None for attr in par_attributes.keys())


def get_par_attributes(node: NodeProto) -> dict:
    """Get parallelization attributes from a node."""
    attr_dict = {a.name: a for a in node.attribute}

    par_attributes = {
        "in_ch_par": attr_dict.get(f"in_ch_par", None),
        "out_ch_par": attr_dict.get(f"out_ch_par", None),
        "out_w_par": attr_dict.get(f"out_w_par", None),
        "in_w_par": attr_dict.get(f"in_w_par", None),
    }

    par_attributes = {
        "in_ch_par": (
            int(par_attributes["in_ch_par"].i) if par_attributes["in_ch_par"] else None
        ),
        "out_ch_par": (
            int(par_attributes["out_ch_par"].i)
            if par_attributes["out_ch_par"]
            else None
        ),
        "in_w_par": (
            int(par_attributes["in_w_par"].i) if par_attributes["in_w_par"] else None
        ),
        "out_w_par": (
            int(par_attributes["out_w_par"].i) if par_attributes["out_w_par"] else None
        ),
    }

    return par_attributes

def set_par_attributes(node: NodeProto, par_attributes: dict) -> None:
    """Set parallelization attributes on a node."""

    for attr_name, attr_value in par_attributes.items():
        getCustomOp(node).set_nodeattr(attr_name, attr_value)

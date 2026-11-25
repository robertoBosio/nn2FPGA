import math


def get_axi_word_count(shape, data_bitwidth: int, axi_bitwidth: int) -> int:
    """Return the number of AXI words needed to store a tensor."""
    data_per_word = axi_bitwidth // data_bitwidth
    if data_per_word == 0:
        raise ValueError("AXI bitwidth must be at least the tensor bitwidth.")
    return (math.prod(shape) + data_per_word - 1) // data_per_word

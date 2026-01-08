import json
import math

def read_board_info(board):
    """ Read the board json file and returns a dictionary with the available resources"""
    file_path = f"/workspace/NN2FPGA/nn2fpga/boards/{board}.json"

    # Opening JSON file with board resources
    with open(file_path) as f:
        board_dict = json.load(f)

    # Right now consider the board as a monolithic block
    board_res = {"uram" : 0, "bram" : 0, "dsp" : 0, "lut" : 0, "ff" : 0}

    # Check that the board has the required fields
    if "resource" not in board_dict:
        raise ValueError(f"Board {board} does not have a 'resource' field in its JSON file.")

    if "axi_bitwidth" not in board_dict or not isinstance(board_dict["axi_bitwidth"], int):
        raise ValueError(f"Board {board} does not have a valid 'axi_bitwidth' field in its JSON file.")

    if "PLL_frequency" not in board_dict or not isinstance(board_dict["PLL_frequency"], int):
        raise ValueError(f"Board {board} does not have a valid 'PLL_frequency' field in its JSON file.")

    if "board_part" not in board_dict or not isinstance(board_dict["board_part"], str):
        raise ValueError(f"Board {board} does not have a valid 'board_part' field in its JSON file.")

    if "part" not in board_dict or not isinstance(board_dict["part"], str):
        raise ValueError(f"Board {board} does not have a valid 'part' field in its JSON file.")

    if not isinstance(board_dict["resource"], list):
        raise ValueError(f"Board {board} 'resource' field must be a list of blocks (dictionaries).")

    for resource in board_dict["resource"]:
        if not isinstance(resource, dict):
            raise ValueError(f"Board {board} 'resource' field must contain dictionaries.")

        if "bram" not in resource or not isinstance(resource["bram"], int):
            raise ValueError(f"Board {board} does not have a valid 'bram' field in its 'resource' dictionary.")

        if "uram" not in resource or not isinstance(resource["uram"], int):
            raise ValueError(f"Board {board} does not have a valid 'uram' field in its 'resource' dictionary.")

        if "dsp" not in resource or not isinstance(resource["dsp"], int):
            raise ValueError(f"Board {board} does not have a valid 'dsp' field in its 'resource' dictionary.")

        if "lut" not in resource or not isinstance(resource["lut"], int):
            raise ValueError(f"Board {board} does not have a valid 'lut' field in its 'resource' dictionary.")

        if "ff" not in resource or not isinstance(resource["ff"], int):
            raise ValueError(f"Board {board} does not have a valid 'ff' field in its 'resource' dictionary.")

    for block in board_dict['resource']:
        for res in block.keys():
            if res in board_res:
                board_res[res] += block[res]
    board_res["axi_bitwidth"] = board_dict["axi_bitwidth"]
    board_res["PLL_frequency"] = board_dict["PLL_frequency"]
    board_res["board_part"] = board_dict["board_part"]
    board_res["part"] = board_dict["part"]

    return board_res


def packing_feature(operands_bitwidth, par, silvia_packing):
    """Returns the number of operation that can be packed in a single DSP.

    Arguments:
        operands_bitwidth: Tuple containing information about the bitwidth of the operands.
        par: Tuple containing the unroll factors of the matrix multiplication.
             Given a matrix multiplication AxB, par[0] is the unroll factor for A rows, par[1] for B columns.
             If A is the activation matrix and B the weight matrix, par[0] is the width parallelism, i.e.,
             the number of feature windows processed in parallel, and par[1] is the number of output channels processed in parallel.
        silvia_packing: Boolean indicating whether to use Silvia's packing strategy.

    Returns:
        int: The number of operations that can be packed in a single DSP.
        tuple: The packing for each dimension.
    """

    operand_bits = max(operands_bitwidth)
    if (operand_bits == 8):
        if (par[1] % 2 == 0):
            return 2, (1, 2)
        elif (par[0] % 2 == 0):
            return 2, (2, 1)
    elif (operand_bits == 4):
        if (silvia_packing):
            if (par[0] % 4 == 0):
                return 4, (4, 1)
            if (par[1] % 4 == 0):
                return 4, (1, 4)
        if (par[1] % 2 == 0):
            return 2, (1, 2)
        elif (par[0] % 2 == 0):
            return 2, (2, 1)
    return 1, (1, 1)


def bram_usage_evaluator(word_width, words, word_parallelism):
    """Compute the number of BRAMs needed to store a memory with a 
    widht of word_widths x word_parallelism bits, and a depth of words // word_parallelism

    Args:
        word_width (int): Width of a single word in bits.
        words (int): Total number of words to be stored.
        word_parallelism (int): Number of words read/written in parallel.
    Returns:
        int: Minimum number of BRAMs needed to store the memory.
    """

    bram9 = bram_consumption(word_width, words, word_parallelism, WIDTH=9)
    bram18 = bram_consumption(word_width, words, word_parallelism, WIDTH=18)
    bram36 = bram_consumption(word_width, words, word_parallelism, WIDTH=36)
    bram72 = bram_consumption(word_width, words, word_parallelism, WIDTH=72)

    # return min(bram9, bram18, bram36, bram72)
    return bram72

def bram_consumption(word_bits, words, word_parallelism, WIDTH=36):
    """Compute the number of BRAMs needed to store the weights, given the parallelism """

    # Useful space in BRAM18. Each BRAM18 is 18kb with a maximum word width of
    # 36 bits, in which 4 bits are reserved to ECC code
    SIZE_BRAM18 = (18 * 1024)
    
    # Useful space in BRAM36, composed by two BRAM18.
    SIZE_BRAM36 = SIZE_BRAM18 * 2

    WIDTH_BRAM36 = WIDTH

    # Assuming is implemented using LUTRAM
    if (words * word_bits) <= SIZE_BRAM18:
        return 0
    
    very_long_word = word_parallelism * word_bits
    mem_width = very_long_word // WIDTH_BRAM36
    mem_width_rem = very_long_word % WIDTH_BRAM36
    word_depth = words // word_parallelism
    mem_depth = int(math.ceil(word_depth / (SIZE_BRAM36 // WIDTH_BRAM36)))
    tot_bram = mem_width * mem_depth

    rem_bram = 0
    if (mem_width_rem > 36):
        rem_bram = int(math.ceil(word_depth / (SIZE_BRAM36 // 72)))
    elif (mem_width_rem > 18 and mem_width_rem <= 36):
        rem_bram = int(math.ceil(word_depth / (SIZE_BRAM36 // 36)))
    elif (mem_width_rem > 8 and mem_width_rem <= 18):
        rem_bram = int(math.ceil(word_depth / (SIZE_BRAM36 // 18)))
    elif (mem_width_rem > 0 and mem_width_rem <= 8):
        rem_bram = int(math.ceil(word_depth / (SIZE_BRAM36 // 9)))
    
    tot_bram += rem_bram
    
    return tot_bram

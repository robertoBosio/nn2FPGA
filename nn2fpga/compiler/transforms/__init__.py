from nn2fpga.compiler.transforms.add_streaming_params import AddStreamingParams
from nn2fpga.compiler.transforms.adjust_conv_scale import AdjustConvScale
from nn2fpga.compiler.transforms.adjust_streaming_comunication import (
    AdjustStreamingCommunication,
)
from nn2fpga.compiler.transforms.balance_computation import BalanceComputation
from nn2fpga.compiler.transforms.compute_fifo_depth import ComputeFifoDepth
from nn2fpga.compiler.transforms.convert_to_QCDQ import ConvertToQCDQ
from nn2fpga.compiler.transforms.custom_infershape import CustomInferShapes
from nn2fpga.compiler.transforms.embed_hls_code import EmbedHLSCode
from nn2fpga.compiler.transforms.fold_asymmetric_act_quant import FoldAsymmetricActQuant
from nn2fpga.compiler.transforms.fold_quant import FoldQuant
from nn2fpga.compiler.transforms.fold_reshape_into_initializer import (
    FoldReshapeIntoInitializer,
)
from nn2fpga.compiler.transforms.fullyconnected_to_conv import FullyConnectedToPointwise
from nn2fpga.compiler.transforms.fuse_elementwise_op import FuseElementwiseOps
from nn2fpga.compiler.transforms.generate_bitstream import GenerateBitstream
from nn2fpga.compiler.transforms.generate_driver import GenerateDriver
from nn2fpga.compiler.transforms.infer_quant import InferQuant
from nn2fpga.compiler.transforms.insert_axi_converters import InsertAXIConverters
from nn2fpga.compiler.transforms.insert_streaming_line_buffer import (
    InsertStreamingLineBuffer,
)
from nn2fpga.compiler.transforms.insert_tensor_duplicator import InsertTensorDuplicator
from nn2fpga.compiler.transforms.lower_to_HLS import LowerToHLS
from nn2fpga.compiler.transforms.lower_to_nn2fpga_layers import LowerToNN2FPGALayers
from nn2fpga.compiler.transforms.optimize_bitwidth import OptimizeBitwidth
from nn2fpga.compiler.transforms.propagate_quant import PropagateQuant
from nn2fpga.compiler.transforms.remove_noop_nodes import RemoveNoopNodes
from nn2fpga.compiler.transforms.remove_redundant_quant import RemoveRedundantQuant
from nn2fpga.compiler.transforms.remove_squeeze import RemoveSqueeze
from nn2fpga.compiler.transforms.set_dynamic_batchsize import SetDynamicBatchSize
from nn2fpga.compiler.transforms.slices_to_split_tree import SlicesToSplitTree
from nn2fpga.compiler.transforms.split_concat import SplitConcat
from nn2fpga.compiler.transforms.supported_partition import SupportedPartition

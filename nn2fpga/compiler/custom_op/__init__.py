from nn2fpga.compiler.custom_op.bandwidthadjust import (
    BandwidthAdjustIncreaseStreams, BandwidthAdjustDecreaseWord,
    BandwidthAdjustIncreaseWord, BandwidthAdjustDecreaseStreams,
)
from nn2fpga.compiler.custom_op.hlskernel import HLSKernel
from nn2fpga.compiler.custom_op.axitostream import AXIToStream
from nn2fpga.compiler.custom_op.nn2fpgapartition import nn2fpgaPartition
from nn2fpga.compiler.custom_op.streamingadd import StreamingAdd
from nn2fpga.compiler.custom_op.streamingaveragepool import StreamingAveragePool
from nn2fpga.compiler.custom_op.streamingconcat import StreamingConcat
from nn2fpga.compiler.custom_op.streamingconv import StreamingConv
from nn2fpga.compiler.custom_op.streamingdepthwiseconv import StreamingDepthwiseConv
from nn2fpga.compiler.custom_op.streamingglobalaveragepool import StreamingGlobalAveragePool
from nn2fpga.compiler.custom_op.streamingleakyrelu import StreamingLeakyReLU
from nn2fpga.compiler.custom_op.streaminglinebuffer import StreamingLineBuffer
from nn2fpga.compiler.custom_op.streamingmaxpool import StreamingMaxPool
from nn2fpga.compiler.custom_op.streamingmemory import StreamingMemory
from nn2fpga.compiler.custom_op.streamingmul import StreamingMul
from nn2fpga.compiler.custom_op.streamingrelu import StreamingReLU
from nn2fpga.compiler.custom_op.streamingreshape import StreamingReshape
from nn2fpga.compiler.custom_op.streamingsigmoid import StreamingSigmoid
from nn2fpga.compiler.custom_op.streamingsoftmax import StreamingSoftmax
from nn2fpga.compiler.custom_op.streamingsplit import StreamingSplit
from nn2fpga.compiler.custom_op.streamingswish import StreamingSwish
from nn2fpga.compiler.custom_op.streamingupsample import StreamingUpsample
from nn2fpga.compiler.custom_op.streamingyoloattention import StreamingYoloAttention
from nn2fpga.compiler.custom_op.streamingyoloheadsoftmax import StreamingYoloHeadSoftmax
from nn2fpga.compiler.custom_op.streamtonhwc import StreamToNHWC
from nn2fpga.compiler.custom_op.tensorduplicator import TensorDuplicator

custom_op = {
    "BandwidthAdjustDecreaseWord": BandwidthAdjustDecreaseWord,
    "BandwidthAdjustDecreaseStreams": BandwidthAdjustDecreaseStreams,
    "BandwidthAdjustIncreaseWord": BandwidthAdjustIncreaseWord,
    "BandwidthAdjustIncreaseStreams": BandwidthAdjustIncreaseStreams,
    "HLSKernel": HLSKernel,
    "AXIToStream": AXIToStream,
    "nn2fpgaPartition": nn2fpgaPartition,
    "StreamingAdd": StreamingAdd,
    "StreamingAveragePool": StreamingAveragePool,
    "StreamingConcat": StreamingConcat,
    "StreamingConv": StreamingConv,
    "StreamingDepthwiseConv": StreamingDepthwiseConv,
    "StreamingGlobalAveragePool": StreamingGlobalAveragePool,
    "StreamingLeakyReLU": StreamingLeakyReLU,
    "StreamingLineBuffer": StreamingLineBuffer,
    "StreamingMaxPool": StreamingMaxPool,
    "StreamingMemory": StreamingMemory,
    "StreamingMul": StreamingMul,
    "StreamingReLU": StreamingReLU,
    "StreamingReshape": StreamingReshape,
    "StreamingSigmoid": StreamingSigmoid,
    "StreamingSoftmax": StreamingSoftmax,
    "StreamingSplit": StreamingSplit,
    "StreamingSwish": StreamingSwish,
    "StreamingUpsample": StreamingUpsample,
    "StreamingYoloAttention": StreamingYoloAttention,
    "StreamingYoloHeadSoftmax": StreamingYoloHeadSoftmax,
    "StreamToNHWC": StreamToNHWC,
    "TensorDuplicator": TensorDuplicator,
}
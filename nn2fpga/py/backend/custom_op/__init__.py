from backend.custom_op.bandwidthadjust import (
    BandwidthAdjustIncreaseChannels, BandwidthAdjustDecreaseChannels,
    BandwidthAdjustIncreaseStreams, BandwidthAdjustDecreaseStreams,
)
from backend.custom_op.hlskernel import HLSKernel
from backend.custom_op.nhwctostream import NHWCToStream
from backend.custom_op.nn2fpgapartition import nn2fpgaPartition
from backend.custom_op.streaminglinebuffer import StreamingLineBuffer
from backend.custom_op.streamingconv import StreamingConv
from backend.custom_op.streamingdepthwiseconv import StreamingDepthwiseConv
from backend.custom_op.streamingglobalaveragepool import StreamingGlobalAveragePool
from backend.custom_op.streamingadd import StreamingAdd
from backend.custom_op.streamingmemory import StreamingMemory
from backend.custom_op.streamingrelu import StreamingRelu
from backend.custom_op.streamtonhwc import StreamToNHWC
from backend.custom_op.tensorduplicator import TensorDuplicator

custom_op = {
    "BandwidthAdjustDecreaseChannels": BandwidthAdjustDecreaseChannels,
    "BandwidthAdjustDecreaseStreams": BandwidthAdjustDecreaseStreams,
    "BandwidthAdjustIncreaseChannels": BandwidthAdjustIncreaseChannels,
    "BandwidthAdjustIncreaseStreams": BandwidthAdjustIncreaseStreams,
    "HLSKernel": HLSKernel,
    "NHWCToStream": NHWCToStream,
    "nn2fpgaPartition": nn2fpgaPartition,
    "StreamingAdd": StreamingAdd,
    "StreamingConv": StreamingConv,
    "StreamingDepthwiseConv": StreamingDepthwiseConv,
    "StreamingGlobalAveragePool": StreamingGlobalAveragePool,
    "StreamingLineBuffer": StreamingLineBuffer,
    "StreamingMemory": StreamingMemory,
    "StreamingRelu": StreamingRelu,
    "StreamToNHWC": StreamToNHWC,
    "TensorDuplicator": TensorDuplicator,
}
import os
import logging
import numpy as np
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    GiveUniqueParameterTensors,
)
import nn2fpga.compiler.transforms as transformation
from nn2fpga.compiler.utils.compare_models import test_transformation_equivalence


def nn2fpga_compile(config_dict: dict):
    """Compile an ONNX model for FPGA using nn2FPGA flow.
    Args:
        config_dict (dict): Configuration dictionary containing:
            - onnx_path (str): Path to the ONNX model file.
            - board (str): Target FPGA board name.
            - prj_root (str): Project root directory.
            - top_name (str): Top module name.
            - frequency (str): Target frequency.
            - hls_version (str): HLS version.
            - other options as needed.
    Returns:
        None
    """

    # Change the working directory to the project root.
    os.chdir(config_dict["prj_root"])

    # Logging setup.
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler("nn2FPGA_compile.log", mode="w")
    console_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[console_handler, file_handler])

    # Load the original ONNX model.
    original_model = ModelWrapper(config_dict["onnx_path"])

    # Save the model before any transformations.
    model = original_model

    # Save metadata properties.
    model.set_metadata_prop("board_name", config_dict["board"])
    model.set_metadata_prop("top_name", config_dict["top_name"])
    model.set_metadata_prop("frequency", config_dict["frequency"])
    model.set_metadata_prop("hls_version", config_dict["hls_version"])
    model.set_metadata_prop("axilite_address", str(0xA0000000))
    model.set_metadata_prop("axilite_size", str(0x10000))
    model.set_metadata_prop("design_id", str(np.random.randint(1, 2**31 - 1)))
    model.set_metadata_prop("silvia_packing", str(config_dict["silvia_packing"]))
    model.set_metadata_prop("simulation", str(config_dict["simulation"]))

    # Optional parameters
    dsp_limit = config_dict.get("dsp_limit")
    if dsp_limit is not None:
        model.set_metadata_prop("dsp_limit", str(dsp_limit))

    # Clean up the model.
    model.cleanup()
    model = model.transform(InferShapes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    original_model = model

    # Propagate quantization through quantization invariant nodes.
    model = model.transform(transformation.SplitConcat())
    model = model.transform(transformation.RemoveNoopNodes())
    model = model.transform(transformation.PropagateQuant())

    # Extract implementable partition.
    nn2fpga_model = model.transform(
        transformation.SupportedPartition(config_dict["prj_root"])
    )

    nn2fpga_model = nn2fpga_model.transform(transformation.SlicesToSplitTree())
    nn2fpga_model = nn2fpga_model.transform(transformation.FullyConnectedToPointwise())
    nn2fpga_model = nn2fpga_model.transform(transformation.FoldReshapeIntoInitializer())
    nn2fpga_model = nn2fpga_model.transform(transformation.RemoveSqueeze())
    nn2fpga_model = nn2fpga_model.transform(transformation.RemoveRedundantQuant())
    nn2fpga_model = nn2fpga_model.transform(transformation.CustomInferShapes())
    if config_dict["steps"].get("OptimizeBitwidth", True):
        nn2fpga_model = nn2fpga_model.transform(transformation.OptimizeBitwidth())
    nn2fpga_model = nn2fpga_model.transform(transformation.AdjustConvScale())
    nn2fpga_model = nn2fpga_model.transform(transformation.LowerToNN2FPGALayers())
    nn2fpga_model = nn2fpga_model.transform(transformation.InsertTensorDuplicator())
    nn2fpga_model = nn2fpga_model.transform(transformation.InsertAXIConverters())
    nn2fpga_model = nn2fpga_model.transform(transformation.PropagateQuant())
    nn2fpga_model = nn2fpga_model.transform(transformation.RemoveRedundantQuant())
    nn2fpga_model = nn2fpga_model.transform(transformation.CustomInferShapes())
    nn2fpga_model = nn2fpga_model.transform(GiveReadableTensorNames())

    # Start of the backend.
    nn2fpga_model = nn2fpga_model.transform(transformation.FuseElementwiseOps())
    nn2fpga_model = nn2fpga_model.transform(transformation.FoldQuant())
    nn2fpga_model = nn2fpga_model.transform(transformation.FoldAsymmetricActQuant())
    nn2fpga_model = nn2fpga_model.transform(
        transformation.BalanceComputation(nn2fpga_root=config_dict["prj_root"])
    )

    # Handle streaming.
    nn2fpga_model = nn2fpga_model.transform(
        transformation.AdjustStreamingCommunication()
    )
    nn2fpga_model = nn2fpga_model.transform(transformation.InsertStreamingLineBuffer())
    nn2fpga_model = nn2fpga_model.transform(transformation.InferQuant())

    if config_dict["steps"].get("AddStreamingParams", True):
        nn2fpga_model = nn2fpga_model.transform(
            transformation.AddStreamingParams(nn2fpga_root=config_dict["prj_root"])
        )
    nn2fpga_model = nn2fpga_model.transform(GiveUniqueNodeNames())
    nn2fpga_model = nn2fpga_model.transform(GiveReadableTensorNames())
    nn2fpga_model = nn2fpga_model.transform(transformation.LowerToHLS())

    if config_dict["steps"].get("ComputeFifoDepth", True):
        nn2fpga_model = nn2fpga_model.transform(
            transformation.ComputeFifoDepth(
                work_root=config_dict["prj_root"], erase=True, ste_already_done=False
            )
        )

    # Check file existence before embedding HLS code.
    wrapper_model_dir = os.path.join(config_dict["prj_root"], "wrapper_model.onnx")
    if not os.path.exists(wrapper_model_dir):
        logging.error(f"Wrapper model file '{wrapper_model_dir}' does not exist.")
        raise FileNotFoundError(f"Wrapper model file '{wrapper_model_dir}' does not exist.")
    model = ModelWrapper("wrapper_model.onnx")

    # Embed HLS code into the model.
    model = model.transform(
        transformation.EmbedHLSCode(
            nn2fpga_model=nn2fpga_model, work_root=config_dict["prj_root"], erase=False
        )
    )

    # Simulate the model to check correctness.
    if config_dict["steps"].get("Simulate", True):
        model.save("final_model_before_sim.onnx")
        original_model.save("original_model_for_sim.onnx")
        test_transformation_equivalence(original_model, model)

    # Generate the bitstream.
    if config_dict["steps"].get("Deploy", True):
        model = model.transform(
            transformation.GenerateBitstream(
                work_dir=config_dict["prj_root"],
                already_exported=False,
                only_synthesize=False,
            )
        )
        model.save("bitstream_generated.onnx")

        # Deploy converted model and original model.
        model = model.transform(
            transformation.GenerateDriver(
                work_dir=config_dict["prj_root"], original_model=original_model
            )
        )

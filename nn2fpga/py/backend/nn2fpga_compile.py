import os
import logging
import copy
import numpy as np
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    GiveUniqueParameterTensors,
)
import backend.transformation as transformation
from backend.util.compare_models import test_transformation_equivalence


def nn2fpga_compile(config_dict: dict):
    """Compile an ONNX model for FPGA using nn2FPGA flow.
    Args:
        config_dict (dict): Configuration dictionary containing:
            - onnx_model (str): Path to the ONNX model file.
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
    # Create handlers
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler("nn2fpga_compile.log", mode="w")

    # Set levels
    console_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)

    # Define a formatter (optional but recommended)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Attach formatter to handlers
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Get the root logger and attach handlers
    logging.basicConfig(level=logging.INFO, handlers=[console_handler, file_handler])

    original_model = ModelWrapper(config_dict["onnx_path"])
    generate_report_file = f"{config_dict['prj_root']}/generate_{config_dict['top_name']}_{config_dict['board']}.rpt"
    # If the file generate_report_file exists, delete it
    if os.path.exists(generate_report_file):
        os.remove(generate_report_file)

    # Save the model before any transformations.
    model = original_model

    # Save target board name in metadata properties.
    model.set_metadata_prop("board_name", config_dict["board"])
    model.set_metadata_prop("top_name", config_dict["top_name"])
    model.set_metadata_prop("frequency", config_dict["frequency"])
    model.set_metadata_prop("hls_version", config_dict["hls_version"])
    model.set_metadata_prop("axilite_address", str(0xA0000000))
    model.set_metadata_prop("axilite_size", str(0x10000))
    model.set_metadata_prop("design_id", str(np.random.randint(1, 2**31 - 1)))
    model.set_metadata_prop("silvia_packing", str(config_dict["silvia_packing"]))

    # Optional parameters
    if config_dict["dsp_limit"] is not None:
        model.set_metadata_prop("dsp_limit", str(config_dict["dsp_limit"]))

    # Clean up the model.
    model.cleanup()
    model = model.transform(InferShapes())
    model = model.transform(GiveUniqueParameterTensors())
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

    # Insert custom nodes.
    nn2fpga_model = nn2fpga_model.transform(transformation.SlicesToSplitTree())
    nn2fpga_model = nn2fpga_model.transform(transformation.FullyConnectedToPointwise())
    nn2fpga_model = nn2fpga_model.transform(transformation.FoldReshapeIntoInitializer())
    nn2fpga_model = nn2fpga_model.transform(transformation.RemoveSqueeze())
    nn2fpga_model = nn2fpga_model.transform(transformation.RemoveRedundantQuant())
    nn2fpga_model = nn2fpga_model.transform(transformation.CustomInferShapes())

    # Handle quantization.
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

    # nn2fpga_model.save("dse_baseline.onnx")
    # nn2fpga_model = ModelWrapper("dse_baseline.onnx")
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
        nn2fpga_model.save("post_fifo_depth.onnx")

    model = ModelWrapper("wrapper_model.onnx")
    # model.save("dse_wrapper_model.onnx")
    # model = ModelWrapper("dse_wrapper_model.onnx")
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

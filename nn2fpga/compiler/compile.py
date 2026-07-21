import copy
import logging
import os
from contextlib import contextmanager

import numpy as np
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
)
import nn2fpga.compiler.transforms as transformation
from nn2fpga.compiler.utils.compare_models import test_transformation_equivalence


def _metadata_bool(value: bool) -> str:
    return "true" if value else "false"


def _artifact_path(prj_root: str, artifact_name: str) -> str:
    return os.path.join(prj_root, artifact_name)


def _save_intermediate(model: ModelWrapper, enabled: bool, path: str) -> None:
    if enabled:
        model.save(path)


@contextmanager
def _compile_logging_context(log_path: str):
    root_logger = logging.getLogger()
    previous_handlers = list(root_logger.handlers)
    previous_level = root_logger.level

    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_path, mode="w")
    console_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    for handler in previous_handlers:
        root_logger.removeHandler(handler)

    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    try:
        yield logging.getLogger(__name__)
    finally:
        root_logger.removeHandler(console_handler)
        root_logger.removeHandler(file_handler)
        console_handler.close()
        file_handler.close()
        for handler in previous_handlers:
            root_logger.addHandler(handler)
        root_logger.setLevel(previous_level)


def _load_and_prepare_model(config_dict: dict) -> tuple[ModelWrapper, ModelWrapper]:
    model = ModelWrapper(config_dict["onnx_path"])
    model.set_metadata_prop("board_name", config_dict["board"])
    model.set_metadata_prop("top_name", config_dict["top_name"])
    model.set_metadata_prop("frequency", config_dict["frequency"])
    model.set_metadata_prop("hls_version", config_dict["hls_version"])
    model.set_metadata_prop("axilite_address", str(0xA0000000))
    model.set_metadata_prop("axilite_size", str(0x10000))
    model.set_metadata_prop("design_id", str(np.random.randint(1, 2**31 - 1)))
    model.set_metadata_prop(
        "silvia_packing", _metadata_bool(config_dict["silvia_packing"])
    )
    model.set_metadata_prop("simulation", str(config_dict["simulation"]))

    dsp_limit = config_dict.get("dsp_limit")
    if dsp_limit is not None:
        model.set_metadata_prop("dsp_limit", str(dsp_limit))

    model.cleanup()
    model = model.transform(InferShapes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    return model, model


def _apply_frontend_transforms(
    model: ModelWrapper, prj_root: str, steps: dict
) -> ModelWrapper:
    model = model.transform(transformation.SplitConcat())
    model = model.transform(transformation.RemoveNoopNodes())
    model = model.transform(transformation.PropagateQuant())

    nn2fpga_model = model.transform(transformation.SupportedPartition(prj_root))
    nn2fpga_model = nn2fpga_model.transform(transformation.SlicesToSplitTree())
    nn2fpga_model = nn2fpga_model.transform(transformation.FullyConnectedToPointwise())
    nn2fpga_model = nn2fpga_model.transform(transformation.FoldReshapeIntoInitializer())
    nn2fpga_model = nn2fpga_model.transform(transformation.RemoveSqueeze())
    nn2fpga_model = nn2fpga_model.transform(transformation.RemoveRedundantQuant())
    nn2fpga_model = nn2fpga_model.transform(transformation.CustomInferShapes())
    if steps.get("OptimizeBitwidth", True):
        nn2fpga_model = nn2fpga_model.transform(transformation.OptimizeBitwidth())
    nn2fpga_model = nn2fpga_model.transform(transformation.AdjustConvScale())
    nn2fpga_model = nn2fpga_model.transform(transformation.LowerToNN2FPGALayers())
    nn2fpga_model = nn2fpga_model.transform(transformation.InsertTensorDuplicator())
    nn2fpga_model = nn2fpga_model.transform(transformation.InsertAXIConverters())
    nn2fpga_model = nn2fpga_model.transform(transformation.PropagateQuant())
    nn2fpga_model = nn2fpga_model.transform(transformation.RemoveRedundantQuant())
    nn2fpga_model = nn2fpga_model.transform(transformation.CustomInferShapes())
    nn2fpga_model = nn2fpga_model.transform(GiveReadableTensorNames())
    return nn2fpga_model


def _apply_backend_transforms(model: ModelWrapper, prj_root: str) -> ModelWrapper:
    model = model.transform(transformation.FuseElementwiseOps())
    model = model.transform(transformation.FoldQuant())
    model = model.transform(transformation.FoldAsymmetricActQuant())
    model = model.transform(transformation.InferLayouts())
    model = model.transform(transformation.BalanceComputation(nn2fpga_root=prj_root))
    model = model.transform(transformation.AdjustStreamingCommunication())
    model = model.transform(transformation.InsertStreamingLineBuffer())
    model = model.transform(transformation.InferQuant())
    return model


def _generate_lightningsim_artifacts(
    model: ModelWrapper,
    prj_root: str,
    store_intermediate: bool,
) -> None:
    lightning_source_model = copy.deepcopy(model)
    lightning_source_model = lightning_source_model.transform(GiveUniqueNodeNames())
    lightning_source_model = lightning_source_model.transform(GiveReadableTensorNames())
    lightning_source_model = lightning_source_model.transform(
        transformation.InferLayouts()
    )
    _save_intermediate(
        lightning_source_model,
        store_intermediate,
        _artifact_path(prj_root, "lightningsim_model.onnx"),
    )

    lightning_hls_model = lightning_source_model.transform(
        transformation.LowerToHLS(
            infer_fifo_depth=False,
            optimize_fifo_storage=False,
            prj_root=prj_root,
        )
    )
    _save_intermediate(
        lightning_hls_model,
        store_intermediate,
        _artifact_path(prj_root, "lightningsim_hls_model.onnx"),
    )
    lightning_hls_model.transform(
        transformation.GenerateLightningSimCode(work_root=prj_root)
    )


def _finalize_nn2fpga_model(
    model: ModelWrapper, prj_root: str, steps: dict
) -> ModelWrapper:
    if steps.get("AddStreamingParams", True):
        model = model.transform(
            transformation.AddStreamingParams(nn2fpga_root=prj_root)
        )
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(transformation.InferLayouts())
    return model


def _lower_to_hls(model: ModelWrapper, prj_root: str, steps: dict) -> ModelWrapper:
    return model.transform(
        transformation.LowerToHLS(
            infer_fifo_depth=steps.get("ComputeFifoDepth", True),
            ste_already_done=False,
            optimize_fifo_storage=steps.get("OptimizeFifo", True),
            prj_root=prj_root,
        )
    )


def _load_wrapper_model(prj_root: str, logger: logging.Logger) -> ModelWrapper:
    wrapper_model_path = _artifact_path(prj_root, "wrapper_model.onnx")
    if not os.path.exists(wrapper_model_path):
        logger.error("Wrapper model file '%s' does not exist.", wrapper_model_path)
        raise FileNotFoundError(
            f"Wrapper model file '{wrapper_model_path}' does not exist."
        )
    return ModelWrapper(wrapper_model_path)


def _run_simulation_check(
    original_model: ModelWrapper,
    model: ModelWrapper,
    prj_root: str,
    store_intermediate: bool,
) -> None:
    _save_intermediate(
        model,
        store_intermediate,
        _artifact_path(prj_root, "final_model_before_sim.onnx"),
    )
    _save_intermediate(
        original_model,
        store_intermediate,
        _artifact_path(prj_root, "original_model_for_sim.onnx"),
    )
    qcdq_original_model = original_model.transform(transformation.ConvertToQCDQ())
    _save_intermediate(
        qcdq_original_model,
        store_intermediate,
        _artifact_path(prj_root, "original_model_qcdq_for_sim.onnx"),
    )
    test_transformation_equivalence(qcdq_original_model, model)


def _generate_bitstream(
    model: ModelWrapper,
    prj_root: str,
) -> ModelWrapper:
    model = model.transform(
        transformation.GenerateBitstream(
            work_dir=prj_root,
            already_exported=False,
            only_synthesize=False,
            vivado_already_done=False,
        )
    )
    model.save(_artifact_path(prj_root, "bitstream_generated.onnx"))
    return ModelWrapper(_artifact_path(prj_root, "bitstream_generated.onnx"))


def _load_bitstream_generated_model(
    prj_root: str, logger: logging.Logger
) -> ModelWrapper:
    bitstream_model_path = _artifact_path(prj_root, "bitstream_generated.onnx")
    if not os.path.exists(bitstream_model_path):
        logger.error("Bitstream model file '%s' does not exist.", bitstream_model_path)
        raise FileNotFoundError(
            f"Bitstream model file '{bitstream_model_path}' does not exist. "
            "Enable GenerateBitstream or provide a previously generated bitstream model."
        )
    return ModelWrapper(bitstream_model_path)


def _deploy_driver(
    model: ModelWrapper,
    original_model: ModelWrapper,
    prj_root: str,
) -> ModelWrapper:
    return model.transform(
        transformation.GenerateDriver(work_dir=prj_root, original_model=original_model)
    )


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

    prj_root = os.path.abspath(config_dict["prj_root"])
    onnx_path = os.path.abspath(config_dict["onnx_path"])
    steps = config_dict["steps"]
    store_intermediate = config_dict.get("store_intermediate", False)

    compile_config = dict(config_dict)
    compile_config["prj_root"] = prj_root
    compile_config["onnx_path"] = onnx_path

    with _compile_logging_context(
        _artifact_path(prj_root, "nn2FPGA_compile.log")
    ) as logger:
        original_model, model = _load_and_prepare_model(compile_config)
        nn2fpga_model = _apply_frontend_transforms(model, prj_root, steps)
        nn2fpga_model = _apply_backend_transforms(nn2fpga_model, prj_root)

        if steps.get("GenerateLightningSim", False):
            _generate_lightningsim_artifacts(
                nn2fpga_model,
                prj_root,
                store_intermediate,
            )

        nn2fpga_model = _finalize_nn2fpga_model(nn2fpga_model, prj_root, steps)
        _save_intermediate(
            nn2fpga_model,
            store_intermediate,
            _artifact_path(prj_root, "nn2fpga_model.onnx"),
        )

        hls_model = _lower_to_hls(nn2fpga_model, prj_root, steps)
        _save_intermediate(
            hls_model,
            store_intermediate,
            _artifact_path(prj_root, "hls_model.onnx"),
        )

        model = _load_wrapper_model(prj_root, logger)
        model = model.transform(
            transformation.EmbedHLSCode(
                hls_model=hls_model,
                work_root=prj_root,
                erase=False,
            )
        )

        if steps.get("Simulate", True):
            _run_simulation_check(
                original_model,
                model,
                prj_root,
                store_intermediate,
            )

        generated_bitstream = steps.get("GenerateBitstream", True)
        if generated_bitstream:
            model = _generate_bitstream(model, prj_root)

        if steps.get("Deploy", True):
            if not generated_bitstream:
                model = _load_bitstream_generated_model(prj_root, logger)
            _deploy_driver(model, original_model, prj_root)

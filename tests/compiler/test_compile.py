from pathlib import Path

import pytest

import nn2fpga.compiler.compile as compile_module


class FakeModel:
    registry = {}
    events = []

    def __init__(self, name, path=None, metadata=None):
        self.name = name
        self.path = path
        self.metadata = dict(metadata or {})

    def __deepcopy__(self, memo):
        return FakeModel(f"{self.name}_copy", path=self.path, metadata=self.metadata)

    def cleanup(self):
        return self

    def set_metadata_prop(self, key, value):
        self.metadata[key] = value

    def get_metadata_prop(self, key):
        return self.metadata.get(key)

    def transform(self, transform):
        FakeModel.events.append((self.name, getattr(transform, "name", type(transform).__name__)))
        if hasattr(transform, "apply"):
            result = transform.apply(self)
            if isinstance(result, tuple):
                return result[0]
            return result
        return self

    def save(self, path):
        path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.name, encoding="utf-8")
        FakeModel.registry[path] = FakeModel(self.name, path=path, metadata=self.metadata)
        FakeModel.events.append((self.name, "save", path))


class FakeModelWrapper:
    loaded_paths = []

    def __new__(cls, model_or_path):
        if isinstance(model_or_path, FakeModel):
            return model_or_path

        path = str(model_or_path)
        cls.loaded_paths.append(path)
        if path in FakeModel.registry:
            stored = FakeModel.registry[path]
            return FakeModel(stored.name, path=stored.path, metadata=stored.metadata)
        return FakeModel(Path(path).stem, path=path)


class NamedTransform:
    def __init__(self, name):
        self.name = name

    def apply(self, model):
        return model, False


class SupportedPartitionTransform:
    name = "SupportedPartition"

    def __init__(self, partition_directory):
        self.partition_directory = partition_directory

    def apply(self, model):
        wrapper_path = Path(self.partition_directory) / "wrapper_model.onnx"
        wrapper_model = FakeModel("wrapper_model", metadata=model.metadata)
        wrapper_model.save(wrapper_path)
        return FakeModel("nn2fpga_model", metadata=model.metadata), False


class LowerToHLSTransform:
    name = "LowerToHLS"

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def apply(self, model):
        return FakeModel("hls_model", metadata=model.metadata), False


class EmbedHLSCodeTransform:
    name = "EmbedHLSCode"

    def __init__(self, hls_model, work_root, erase):
        self.hls_model = hls_model
        self.work_root = work_root
        self.erase = erase

    def apply(self, model):
        model.metadata["embedded_hls_model"] = self.hls_model.name
        return model, False


class GenerateBitstreamTransform:
    name = "GenerateBitstream"

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def apply(self, model):
        return model, False


class GenerateDriverTransform:
    name = "GenerateDriver"

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def apply(self, model):
        return model, False


@pytest.fixture(autouse=True)
def reset_fakes():
    FakeModel.registry = {}
    FakeModel.events = []
    FakeModelWrapper.loaded_paths = []
    yield


@pytest.fixture
def fake_compile(monkeypatch):
    monkeypatch.setattr(compile_module, "ModelWrapper", FakeModelWrapper)
    monkeypatch.setattr(compile_module, "InferShapes", lambda: NamedTransform("InferShapes"))
    monkeypatch.setattr(
        compile_module,
        "GiveUniqueNodeNames",
        lambda: NamedTransform("GiveUniqueNodeNames"),
    )
    monkeypatch.setattr(
        compile_module,
        "GiveReadableTensorNames",
        lambda: NamedTransform("GiveReadableTensorNames"),
    )
    monkeypatch.setattr(
        compile_module,
        "test_transformation_equivalence",
        lambda original_model, model: FakeModel.events.append(
            (original_model.name, "equivalence", model.name)
        ),
    )
    monkeypatch.setattr(compile_module.np.random, "randint", lambda low, high: 7)

    fake_transforms = type("FakeTransforms", (), {})()
    for name in [
        "SplitConcat",
        "RemoveNoopNodes",
        "PropagateQuant",
        "SlicesToSplitTree",
        "FullyConnectedToPointwise",
        "FoldReshapeIntoInitializer",
        "RemoveSqueeze",
        "RemoveRedundantQuant",
        "CustomInferShapes",
        "OptimizeBitwidth",
        "AdjustConvScale",
        "LowerToNN2FPGALayers",
        "InsertTensorDuplicator",
        "InsertAXIConverters",
        "FuseElementwiseOps",
        "FoldQuant",
        "FoldAsymmetricActQuant",
        "InferLayouts",
        "AdjustStreamingCommunication",
        "InsertStreamingLineBuffer",
        "InferQuant",
        "ConvertToQCDQ",
    ]:
        setattr(fake_transforms, name, lambda name=name, **kwargs: NamedTransform(name))

    setattr(
        fake_transforms,
        "BalanceComputation",
        lambda **kwargs: NamedTransform("BalanceComputation"),
    )
    setattr(
        fake_transforms,
        "AddStreamingParams",
        lambda **kwargs: NamedTransform("AddStreamingParams"),
    )
    setattr(
        fake_transforms,
        "GenerateLightningSimCode",
        lambda **kwargs: NamedTransform("GenerateLightningSimCode"),
    )
    setattr(fake_transforms, "SupportedPartition", SupportedPartitionTransform)
    setattr(fake_transforms, "LowerToHLS", LowerToHLSTransform)
    setattr(fake_transforms, "EmbedHLSCode", EmbedHLSCodeTransform)
    setattr(fake_transforms, "GenerateBitstream", GenerateBitstreamTransform)
    setattr(fake_transforms, "GenerateDriver", GenerateDriverTransform)
    monkeypatch.setattr(compile_module, "transformation", fake_transforms)


def build_config(
    tmp_path,
    onnx_path,
    store_intermediate=False,
    generate_bitstream=False,
    deploy=False,
):
    prj_root = tmp_path / "project"
    prj_root.mkdir()
    return {
        "onnx_path": onnx_path,
        "board": "board",
        "prj_root": str(prj_root),
        "top_name": "top",
        "frequency": "100",
        "hls_version": "2024.2",
        "silvia_packing": True,
        "simulation": "csim",
        "dsp_limit": None,
        "store_intermediate": store_intermediate,
        "steps": {
            "OptimizeBitwidth": True,
            "AddStreamingParams": True,
            "ComputeFifoDepth": True,
            "OptimizeFifo": True,
            "GenerateLightningSim": False,
            "Simulate": True,
            "GenerateBitstream": generate_bitstream,
            "Deploy": deploy,
        },
    }


def test_compile_resolves_relative_onnx_path_without_changing_cwd(tmp_path, monkeypatch, fake_compile):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    onnx_path = source_dir / "model.onnx"
    onnx_path.write_text("fake", encoding="utf-8")
    monkeypatch.chdir(source_dir)

    config = build_config(tmp_path, "model.onnx")
    compile_module.nn2fpga_compile(config)

    assert Path.cwd() == source_dir
    assert FakeModelWrapper.loaded_paths[0] == str(onnx_path.resolve())


def test_compile_only_saves_mid_pipeline_artifacts_when_enabled(tmp_path, fake_compile):
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_text("fake", encoding="utf-8")
    config = build_config(tmp_path, str(onnx_path), store_intermediate=False)

    compile_module.nn2fpga_compile(config)

    prj_root = Path(config["prj_root"])
    assert (prj_root / "wrapper_model.onnx").exists()
    assert not (prj_root / "nn2fpga_model.onnx").exists()
    assert not (prj_root / "hls_model.onnx").exists()
    assert not (prj_root / "final_model_before_sim.onnx").exists()
    assert not (prj_root / "original_model_for_sim.onnx").exists()
    assert not (prj_root / "original_model_qcdq_for_sim.onnx").exists()


def test_compile_saves_mid_pipeline_artifacts_when_enabled(tmp_path, fake_compile):
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_text("fake", encoding="utf-8")
    config = build_config(tmp_path, str(onnx_path), store_intermediate=True)

    compile_module.nn2fpga_compile(config)

    prj_root = Path(config["prj_root"])
    assert (prj_root / "nn2fpga_model.onnx").exists()
    assert (prj_root / "hls_model.onnx").exists()
    assert (prj_root / "final_model_before_sim.onnx").exists()
    assert (prj_root / "original_model_for_sim.onnx").exists()
    assert (prj_root / "original_model_qcdq_for_sim.onnx").exists()

    saved_original = FakeModel.registry[str(prj_root / "original_model_for_sim.onnx")]
    assert saved_original.metadata["silvia_packing"] == "true"


def test_compile_can_generate_bitstream_without_deploy(tmp_path, fake_compile):
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_text("fake", encoding="utf-8")
    config = build_config(tmp_path, str(onnx_path), generate_bitstream=True, deploy=False)

    compile_module.nn2fpga_compile(config)

    prj_root = Path(config["prj_root"])
    assert (prj_root / "bitstream_generated.onnx").exists()
    assert any(event[1] == "GenerateBitstream" for event in FakeModel.events)
    assert not any(event[1] == "GenerateDriver" for event in FakeModel.events)


def test_compile_can_deploy_from_existing_bitstream_model(tmp_path, fake_compile):
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_text("fake", encoding="utf-8")
    config = build_config(tmp_path, str(onnx_path), generate_bitstream=False, deploy=True)
    prj_root = Path(config["prj_root"])
    bitstream_model_path = prj_root / "bitstream_generated.onnx"
    FakeModel("bitstream_model").save(bitstream_model_path)
    FakeModel.events.clear()

    compile_module.nn2fpga_compile(config)

    assert str(bitstream_model_path) in FakeModelWrapper.loaded_paths
    assert not any(event[1] == "GenerateBitstream" for event in FakeModel.events)
    assert ("bitstream_model", "GenerateDriver") in FakeModel.events

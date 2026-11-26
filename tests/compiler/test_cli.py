import pytest

import nn2fpga.compiler.cli as cli


def build_config(tmp_path):
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_text("fake", encoding="utf-8")
    return {
        "top_name": "top",
        "onnx_path": str(onnx_path),
        "prj_root": str(tmp_path / "project"),
        "hls_version": "2024.2",
        "hls_backend": "vitis",
        "board": "board",
        "frequency": "100",
        "xilinx_version": "2024.2",
        "silvia_packing": False,
        "simulation": "csim",
        "store_intermediate": False,
        "use_dma": True,
        "steps": {
            "GenerateLightningSim": False,
            "GenerateBitstream": False,
            "Deploy": False,
        },
    }


@pytest.mark.parametrize("backend", ["vitis", "catapult"])
def test_load_config_uses_requested_hls_backend(monkeypatch, backend):
    config = {
        "project": {
            "top_name": "top",
            "onnx_path": "model.onnx",
            "project_root": "project",
            "hls_version": 2024.2,
        },
        "platform": {"board": "board", "frequency": 100},
    }
    if backend == "catapult":
        config["project"]["hls_backend"] = backend

    monkeypatch.setattr(cli.toml, "load", lambda path: config)
    monkeypatch.setattr(cli, "read_board_info", lambda board: None)
    monkeypatch.setenv("XILINX_VERSION", "2024.2")

    assert cli.load_config("config.toml")["hls_backend"] == backend


def test_check_config_rejects_unknown_hls_backend(tmp_path):
    config = build_config(tmp_path)
    config["hls_backend"] = "unknown"

    with pytest.raises(ValueError, match="either 'vitis' or 'catapult'"):
        cli.check_config(config)


def test_check_config_accepts_supported_catapult_configuration(tmp_path):
    config = build_config(tmp_path)
    config["hls_backend"] = "catapult"

    cli.check_config(config)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda config: config.update(use_dma=False), "requires 'use_dma = true'"),
        (
            lambda config: config.update(silvia_packing=True),
            "does not support 'silvia_packing = true'",
        ),
        (
            lambda config: config["steps"].update(GenerateLightningSim=True),
            "does not support LightningSim generation",
        ),
        (
            lambda config: config["steps"].update(GenerateBitstream=True),
            "does not support bitstream generation",
        ),
        (
            lambda config: config["steps"].update(Deploy=True),
            "does not support deployment",
        ),
    ],
)
def test_check_config_rejects_unsupported_catapult_options(tmp_path, change, message):
    config = build_config(tmp_path)
    config["hls_backend"] = "catapult"
    change(config)

    with pytest.raises(ValueError, match=message):
        cli.check_config(config)

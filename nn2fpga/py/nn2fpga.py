import os
import sys
import toml
import argparse
from backend.util.qonnx_patch import patch_qonnx_ops
patch_qonnx_ops()
from backend.nn2fpga_compile import nn2fpga_compile
from backend.util.board_util import read_board_info

def load_config(config_path: str) -> dict:
    """
    Load configuration from a TOML file.
    Args:
        config_path (str): Path to the TOML configuration file.
    Returns:
        dict: Parsed configuration 

    """
    config_dict = {}
    try:
        config = toml.load(config_path)
    except Exception as e:
        print(f"Failed to load TOML config: {e}")
        sys.exit(1)

    try:
        config_dict["top_name"] = config["project"]["top_name"]
        config_dict["onnx_path"] = config["project"]["onnx_path"]
        config_dict["prj_root"] = config["project"]["project_root"]
        config_dict["hls_version"] = str(config["project"]["hls_version"])
        config_dict["board"] = config["platform"]["board"]
        config_dict["frequency"] = config["platform"]["frequency"]

        # Steps
        config_dict["steps"] = config.get("steps", {
            "OptimizeBitwidth": True,
            "AddStreamingParams": True,
            "ComputeFifoDepth": True,
            "Simulate": True,
            "Deploy": True,
        })  
        
        # Optional fields with defaults
        config_dict["silvia_packing"] = config.get("options", {}).get("silvia_packing", False)
        config_dict["dsp_limit"] = config.get("options", {}).get("dsp_limit", None)


    except KeyError as e:
        print(f"Missing configuration field: {e}")
        sys.exit(1)

    # Check if the board is valid
    _ = read_board_info(config_dict["board"])

    if "XILINX_VERSION" not in os.environ:
        raise EnvironmentError(
            """XILINX_VERSION environment variable is not set.
            It should be set automatically by the script running the Docker
            container. Please ensure you are running nn2FPGA Docker 
            container correctly."""
        )
    config_dict["xilinx_version"] = os.environ["XILINX_VERSION"]
    return config_dict

def check_config(config_dict: dict):
    """
    Check if the configuration is valid.
    Args:
        config_dict (dict): Configuration dictionary.
    Raises:
        ValueError: If any required field is missing or invalid.
    """

    # Check that the project root exists, otherwise create it
    if not os.path.exists(config_dict["prj_root"]):
        try:
            os.makedirs(config_dict["prj_root"])
        except Exception as e:
            raise ValueError(f"Failed to create project root directory: {e}")

    # Check that top_name is a valid string
    if not isinstance(config_dict["top_name"], str) or not config_dict["top_name"].strip():
        raise ValueError("Invalid 'top_name' in configuration. It must be a non-empty string.") 
    
    # Check that onnx_path is a valid file path
    if not isinstance(config_dict["onnx_path"], str) or not os.path.isfile(config_dict["onnx_path"]):
        raise ValueError(f"Invalid 'onnx_path' in configuration: {config_dict['onnx_path']}. It must be a valid file path.")
    
    # Check that the frequency is a string that can be converted to an integer, and is greater than 0 and less than 330
    if not isinstance(config_dict["frequency"], (int, str)):
        raise ValueError("Invalid 'frequency' in configuration. It must be an integer or a string that can be converted to an integer.")
    try:
        frequency = int(config_dict["frequency"])
        if frequency <= 0 or frequency >= 330:
            raise ValueError("Frequency must be greater than 0 and less than 330 MHz.")
    except ValueError:
        raise ValueError("Invalid 'frequency' in configuration. It must be a valid integer or string that can be converted to an integer.")
    
    # Must be stored as a string in the config_dict
    config_dict["frequency"] = str(frequency)

    # Check that hls_version is a valid string with a format like "2024.2"
    if not isinstance(config_dict["xilinx_version"], str) or not config_dict["xilinx_version"].strip():
        raise ValueError("Invalid 'xilinx_version' in configuration. It must be a non-empty string.")
    if not config_dict["xilinx_version"].replace('.', '', 1).isdigit():
        raise ValueError("Invalid 'xilinx_version' in configuration. It must be a string representing a version number (e.g., '2024.2').")


def main():
    parser = argparse.ArgumentParser(
        description="Compile an ONNX model to FPGA using nn2fpga."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the TOML config file",
    )
    args = parser.parse_args()

    config_dict = load_config(args.config)
    check_config(config_dict)

    nn2fpga_compile(
        config_dict
    )

if __name__ == '__main__':
    main()

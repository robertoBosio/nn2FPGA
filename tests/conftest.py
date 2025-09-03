import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--hls-steps",
        action="store",
        default="csim,csynth,cosim",
        help="Comma-separated HLS steps to run (choose from: csim, csynth, cosim)",
    )

@pytest.fixture
def hls_steps(request):
    return request.config.getoption("--hls-steps")
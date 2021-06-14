import pytest

@pytest.fixture(scope="function")
def test_network_run_time():
    return 5

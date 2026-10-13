import os
import socket
import pytest

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def pytest_configure(config):
    config.addinivalue_line("markers", "network: mark test as requiring network access")


@pytest.fixture(autouse=True)
def guard_network(request, monkeypatch):
    if request.node.get_closest_marker("network"):
        return

    def offline_connect(self, address):
        raise RuntimeError("Network access attempted in offline test")

    monkeypatch.setattr(socket.socket, "connect", offline_connect)

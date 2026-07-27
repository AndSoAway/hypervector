from __future__ import annotations

import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(_REPO_ROOT / "pyhypervec"))

from pyhypervec.uri import parse_uri
from pyhypervec._grpc_transport import grpc_max_message_bytes


@pytest.mark.parametrize("scheme", ["tcp", "grpc"])
def test_grpc_uri(scheme):
    parsed = parse_uri(f"{scheme}://localhost:50051")
    assert parsed.transport == "grpc"
    assert parsed.address == "localhost:50051"
    assert parsed.http_base is None


def test_bare_host_port_uses_grpc():
    assert parse_uri("localhost:50051").transport == "grpc"


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("http://localhost", "http://localhost:80"),
        ("https://localhost", "https://localhost:443"),
        ("http://localhost:8081", "http://localhost:8081"),
        ("http://localhost:8081/api", "http://localhost:8081/api"),
    ],
)
def test_http_uri_preserves_http_semantics(uri, expected):
    assert parse_uri(uri).http_base == expected


def test_ipv6_addresses_are_bracketed():
    assert parse_uri("grpc://[::1]:50051").address == "[::1]:50051"
    assert parse_uri("http://[::1]:8081").http_base == "http://[::1]:8081"


@pytest.mark.parametrize(
    "uri",
    [
        "",
        "ftp://localhost:21",
        "grpc://:50051",
        "grpc://localhost:70000",
        "grpc://user:password@localhost:50051",
        "grpc://localhost:50051/path",
    ],
)
def test_invalid_uri_rejected(uri):
    with pytest.raises(ValueError):
        parse_uri(uri)


def test_grpc_message_limit_environment(monkeypatch):
    monkeypatch.setenv("HYPERVEC_GRPC_MAX_MESSAGE_MB", "32")
    assert grpc_max_message_bytes() == 32 * 1024 * 1024

    monkeypatch.setenv("HYPERVEC_GRPC_MAX_MESSAGE_MB", "0")
    with pytest.raises(Exception, match="must be positive"):
        grpc_max_message_bytes()

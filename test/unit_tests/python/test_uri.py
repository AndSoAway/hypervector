"""Unit tests for pyhypervec.uri — no server required."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3] / "pyhypervec"))

import pytest
from pyhypervec.uri import parse_uri


def test_tcp_scheme():
    p = parse_uri("tcp://localhost:50051")
    assert p.transport == "grpc"
    assert p.host == "localhost"
    assert p.port == 50051
    assert p.address == "localhost:50051"
    assert p.http_base is None


def test_grpc_scheme():
    p = parse_uri("grpc://myhost:9000")
    assert p.transport == "grpc"
    assert p.host == "myhost"
    assert p.port == 9000
    assert p.address == "myhost:9000"
    assert p.http_base is None


def test_http_scheme():
    p = parse_uri("http://localhost:8080")
    assert p.transport == "http"
    assert p.host == "localhost"
    assert p.port == 8080
    assert p.address == "localhost:8080"
    assert p.http_base == "http://localhost:8080"


def test_https_scheme():
    p = parse_uri("https://myhost:443")
    assert p.transport == "https"
    assert p.host == "myhost"
    assert p.port == 443
    assert p.http_base == "https://myhost:443"


def test_bare_hostport_treated_as_grpc():
    p = parse_uri("localhost:50051")
    assert p.transport == "grpc"
    assert p.address == "localhost:50051"
    assert p.http_base is None


def test_default_ports():
    assert parse_uri("tcp://somehost").port == 50051
    assert parse_uri("grpc://somehost").port == 50051
    assert parse_uri("http://somehost").port == 8080
    assert parse_uri("https://somehost").port == 443


def test_invalid_scheme_raises():
    with pytest.raises(ValueError, match="Unsupported URI scheme"):
        parse_uri("ftp://localhost:21")


def test_missing_host_raises():
    with pytest.raises(ValueError, match="Missing host"):
        parse_uri("tcp://:50051")


def test_client_uri_sets_grpc_transport(monkeypatch):
    """HypervecClient with tcp:// URI sets _grpc, not HTTP self.uri."""
    import pyhypervec._grpc_transport as _mod

    class FakeGrpc:
        def __init__(self, address, *, timeout):
            self.address = address

        def close(self):
            pass

    monkeypatch.setattr(_mod, "GrpcTransport", FakeGrpc)

    from pyhypervec import HypervecClient

    c = HypervecClient("tcp://localhost:50051")
    assert c._grpc is not None
    assert c._grpc.address == "localhost:50051"


def test_client_uri_sets_http_transport():
    """HypervecClient with http:// URI sets _grpc=None and self.uri as base URL."""
    from pyhypervec import HypervecClient

    c = HypervecClient("http://localhost:8080")
    assert c._grpc is None
    assert c.uri == "http://localhost:8080/"


def test_client_uri_signature_unchanged():
    """__init__ signature must be (uri, token=None, timeout=30.0, http2=False)."""
    from pyhypervec import HypervecClient
    import inspect

    sig = inspect.signature(HypervecClient.__init__)
    params = list(sig.parameters.keys())
    assert params == ["self", "uri", "token", "timeout", "http2"]
    assert sig.parameters["token"].default is None
    assert sig.parameters["timeout"].default == 30.0
    assert sig.parameters["http2"].default is False

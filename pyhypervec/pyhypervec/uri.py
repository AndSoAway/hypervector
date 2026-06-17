from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

_GRPC_SCHEMES = frozenset({"tcp", "grpc"})
_HTTP_SCHEMES = frozenset({"http", "https"})
_SUPPORTED = _GRPC_SCHEMES | _HTTP_SCHEMES

_DEFAULT_PORTS: dict[str, int] = {
    "tcp": 50051,
    "grpc": 50051,
    "http": 8080,
    "https": 443,
}


@dataclass(frozen=True)
class ParsedURI:
    transport: str       # "grpc" | "http" | "https"
    host: str
    port: int
    address: str         # "host:port"      — used as gRPC channel target
    http_base: str | None  # "scheme://host:port" — used as HTTP base URL


def parse_uri(uri: str) -> ParsedURI:
    """Parse and normalise a HyperVec server URI.

    Supported schemes
    -----------------
    tcp://host[:port]   → gRPC insecure, default port 50051
    grpc://host[:port]  → gRPC insecure, default port 50051
    http://host[:port]  → HTTP REST, default port 8080
    https://host[:port] → HTTPS REST, default port 443
    host:port           → treated as tcp:// (gRPC)
    """
    uri = uri.strip()

    if "://" not in uri:
        uri = "tcp://" + uri

    parsed = urlparse(uri)
    scheme = (parsed.scheme or "").lower()

    if scheme not in _SUPPORTED:
        raise ValueError(
            f"Unsupported URI scheme {scheme!r}. "
            f"Supported: {sorted(_SUPPORTED)}"
        )

    host = parsed.hostname or ""
    if not host:
        raise ValueError(f"Missing host in URI: {uri!r}")

    port = parsed.port or _DEFAULT_PORTS[scheme]
    address = f"{host}:{port}"

    if scheme in _GRPC_SCHEMES:
        return ParsedURI(
            transport="grpc",
            host=host,
            port=port,
            address=address,
            http_base=None,
        )

    return ParsedURI(
        transport=scheme,
        host=host,
        port=port,
        address=address,
        http_base=f"{scheme}://{address}",
    )

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse, urlunparse


_GRPC_SCHEMES = frozenset({"tcp", "grpc"})
_HTTP_SCHEMES = frozenset({"http", "https"})
_SUPPORTED_SCHEMES = _GRPC_SCHEMES | _HTTP_SCHEMES
_DEFAULT_PORTS = {
    "tcp": 50051,
    "grpc": 50051,
    "http": 80,
    "https": 443,
}


@dataclass(frozen=True)
class ParsedURI:
    transport: str
    host: str
    port: int
    address: str
    http_base: str | None


def _host_port(host: str, port: int) -> str:
    rendered_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
    return f"{rendered_host}:{port}"


def parse_uri(uri: str) -> ParsedURI:
    """Parse a HyperVector endpoint and select its transport.

    ``tcp://`` and ``grpc://`` use an insecure gRPC channel. Explicit
    ``http://`` and ``https://`` endpoints retain HTTP behavior. A bare
    ``host:port`` endpoint is treated as gRPC for Milvus-style compatibility.
    """

    if not isinstance(uri, str) or not uri.strip():
        raise ValueError("HyperVector URI must be a non-empty string.")

    normalized = uri.strip()
    if "://" not in normalized:
        normalized = f"tcp://{normalized}"

    parsed = urlparse(normalized)
    scheme = parsed.scheme.lower()
    if scheme not in _SUPPORTED_SCHEMES:
        raise ValueError(
            f"Unsupported URI scheme {scheme!r}. "
            f"Supported schemes: {', '.join(sorted(_SUPPORTED_SCHEMES))}."
        )
    if parsed.username or parsed.password:
        raise ValueError("Credentials must be provided with the token parameter, not in the URI.")
    host = parsed.hostname or ""
    if not host:
        raise ValueError(f"Missing host in URI: {uri!r}")

    try:
        port = parsed.port or _DEFAULT_PORTS[scheme]
    except ValueError as exc:
        raise ValueError(f"Invalid port in URI: {uri!r}") from exc
    if not 1 <= port <= 65535:
        raise ValueError(f"Port must be between 1 and 65535: {port}")

    address = _host_port(host, port)
    if scheme in _GRPC_SCHEMES:
        if parsed.path not in ("", "/") or parsed.params or parsed.query or parsed.fragment:
            raise ValueError("gRPC URI must not contain a path, query, or fragment.")
        return ParsedURI("grpc", host, port, address, None)

    path = parsed.path.rstrip("/")
    http_base = urlunparse((scheme, address, path, "", "", ""))
    return ParsedURI(scheme, host, port, address, http_base)

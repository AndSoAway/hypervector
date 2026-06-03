from __future__ import annotations


class HypervecClientError(Exception):
    """Base exception for pyhypervec client failures."""


class HypervecHTTPError(HypervecClientError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = status_code
        self.message = message

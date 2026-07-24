# Copyright (c) 2024 HyperVec Authors. All rights reserved.
#
# This source code is licensed under the Mulan Permissive Software License v2 (the "License") found in the
# LICENSE file in the root directory of this source tree.

"""Typed startup configuration for the HyperVector HTTP server.

This module owns configuration metadata, parsing, precedence, validation,
sample generation, and HyperVector Python logger setup. Business modules
consume the immutable config objects and do not read INI or CLI inputs.
"""

from __future__ import annotations

import configparser
from dataclasses import dataclass
import logging
from pathlib import Path
import re
import sys
from typing import Callable, Dict, Mapping, Optional, Union


ConfigValue = Optional[Union[bool, int, str]]
ConfigOverrides = Dict[str, Dict[str, ConfigValue]]


class ConfigError(ValueError):
    """A user-facing configuration error with optional source context."""

    def __init__(
        self,
        message: str,
        *,
        path: Path | None = None,
        section: str | None = None,
        key: str | None = None,
        value: object | None = None,
    ) -> None:
        self.message = message
        self.path = path
        self.section = section
        self.key = key
        self.value = value

        location = str(path) if path is not None else ""
        if section is not None:
            option_name = f"[{section}]"
            if key is not None:
                option_name += f".{key}"
            location = f"{location} {option_name}".strip()

        super().__init__(f"{location}: {message}" if location else message)


@dataclass(frozen=True)
class ServerConfig:
    """Validated process and ASGI server startup settings."""

    data_root: str | None
    host: str
    port: int
    server: str
    enable_http2: bool
    certfile: str | None
    keyfile: str | None


@dataclass(frozen=True)
class LoggingConfig:
    """Logging policy consumed by the centralized logging initializer."""

    enable_logging: bool
    log_level: str
    log_to_stderr: bool
    log_to_file: bool
    log_file_path: str | None


@dataclass(frozen=True)
class IndexDefaultsConfig:
    """Reserved collection defaults; loading does not apply them to requests yet."""

    default_index_type: str
    default_metric_type: str


@dataclass(frozen=True)
class HypervecConfig:
    """Immutable configuration snapshot passed to application code."""

    server: ServerConfig
    defaults: IndexDefaultsConfig
    logging: LoggingConfig


@dataclass(frozen=True)
class ConfigOption:
    """Metadata used to parse, validate, document, and override one option."""

    section: str
    key: str
    field_path: tuple[str, str]
    value_type: type
    default: ConfigValue
    description: str
    cli_dest: str | None
    choices: tuple[str, ...] = ()
    validator: Callable[[ConfigValue], None] | None = None
    optional: bool = False
    is_path: bool = False


def _validate_non_empty(value: ConfigValue) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"value {value!r} must be a non-empty string")


def _validate_port(value: ConfigValue) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 65535:
        raise ValueError(f"value {value!r} must be an integer in the range 1..65535")


# This table is the single source of truth for defaults, supported keys,
# type conversion, validation, CLI mapping, and generated sample content.
CONFIG_OPTIONS: tuple[ConfigOption, ...] = (
    ConfigOption(
        section="server",
        key="data_root",
        field_path=("server", "data_root"),
        value_type=str,
        default=None,
        description=(
            "Root directory for collection metadata, scalar data, and indexes; "
            "required at server startup."
        ),
        cli_dest="data_root",
        optional=True,
        is_path=True,
    ),
    ConfigOption(
        section="server",
        key="host",
        field_path=("server", "host"),
        value_type=str,
        default="127.0.0.1",
        description="Host name or IP address on which the HTTP server listens.",
        cli_dest="host",
        validator=_validate_non_empty,
    ),
    ConfigOption(
        section="server",
        key="port",
        field_path=("server", "port"),
        value_type=int,
        default=8080,
        description="TCP port on which the HTTP server listens (1..65535).",
        cli_dest="port",
        validator=_validate_port,
    ),
    ConfigOption(
        section="server",
        key="server",
        field_path=("server", "server"),
        value_type=str,
        default="hypercorn",
        description="ASGI server implementation.",
        cli_dest="server",
        choices=("hypercorn", "uvicorn"),
    ),
    ConfigOption(
        section="server",
        key="enable_http2",
        field_path=("server", "enable_http2"),
        value_type=bool,
        default=True,
        description="Enable HTTP/2 when supported by the selected ASGI server.",
        cli_dest="enable_http2",
    ),
    ConfigOption(
        section="server",
        key="certfile",
        field_path=("server", "certfile"),
        value_type=str,
        default=None,
        description="TLS certificate file; configure together with keyfile.",
        cli_dest="certfile",
        optional=True,
        is_path=True,
    ),
    ConfigOption(
        section="server",
        key="keyfile",
        field_path=("server", "keyfile"),
        value_type=str,
        default=None,
        description="TLS private key file; configure together with certfile.",
        cli_dest="keyfile",
        optional=True,
        is_path=True,
    ),
    ConfigOption(
        section="defaults",
        key="default_index_type",
        field_path=("defaults", "default_index_type"),
        value_type=str,
        default="hnswflat",
        description="Default index type reserved for collection creation defaults.",
        cli_dest="default_index_type",
        choices=(
            "flat",
            "ivfflat",
            "ivflvq",
            "ivfpq",
            "hnswflat",
            "hnswlvq",
            "hnswpq",
        ),
    ),
    ConfigOption(
        section="defaults",
        key="default_metric_type",
        field_path=("defaults", "default_metric_type"),
        value_type=str,
        default="l2",
        description="Default distance metric reserved for collection creation defaults.",
        cli_dest="default_metric_type",
        choices=("l2", "ip", "cosine"),
    ),
    ConfigOption(
        section="logging",
        key="enable_logging",
        field_path=("logging", "enable_logging"),
        value_type=bool,
        default=True,
        description="Enable logging for the HyperVector Python logger namespace.",
        cli_dest="enable_logging",
    ),
    ConfigOption(
        section="logging",
        key="log_level",
        field_path=("logging", "log_level"),
        value_type=str,
        default="info",
        description="Minimum level emitted by HyperVector logging.",
        cli_dest="log_level",
        choices=("debug", "info", "warning", "error", "critical"),
    ),
    ConfigOption(
        section="logging",
        key="log_to_stderr",
        field_path=("logging", "log_to_stderr"),
        value_type=bool,
        default=True,
        description="Write HyperVector log records to standard error.",
        cli_dest="log_to_stderr",
    ),
    ConfigOption(
        section="logging",
        key="log_to_file",
        field_path=("logging", "log_to_file"),
        value_type=bool,
        default=False,
        description="Write HyperVector log records to a file.",
        cli_dest="log_to_file",
    ),
    ConfigOption(
        section="logging",
        key="log_file_path",
        field_path=("logging", "log_file_path"),
        value_type=str,
        default=None,
        description="Log file path used when log_to_file is enabled.",
        cli_dest="log_file_path",
        optional=True,
        is_path=True,
    ),
)


def _build_option_index(attribute: str) -> dict[object, ConfigOption]:
    """Build an immutable-at-runtime lookup and reject metadata collisions."""

    index: dict[object, ConfigOption] = {}
    for option in CONFIG_OPTIONS:
        key: object
        if attribute == "name":
            key = (option.section, option.key)
        else:
            key = getattr(option, attribute)
        if key is None:
            continue
        if key in index:
            raise RuntimeError(f"duplicate configuration metadata for {key!r}")
        index[key] = option
    return index


_OPTIONS_BY_NAME = _build_option_index("name")
_OPTIONS_BY_CLI_DEST = _build_option_index("cli_dest")
_INTEGER_PATTERN = re.compile(r"[+-]?[0-9]+\Z")
_BOOLEAN_VALUES = {
    "true": True,
    "yes": True,
    "on": True,
    "1": True,
    "false": False,
    "no": False,
    "off": False,
    "0": False,
}
_LOG_HANDLER_MARKER = "_hypervec_config_handler"
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

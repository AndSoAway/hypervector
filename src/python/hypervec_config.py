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


def _option_error(
    option: ConfigOption,
    message: str,
    *,
    path: Path | None = None,
    value: object | None = None,
) -> ConfigError:
    return ConfigError(
        message,
        path=path,
        section=option.section,
        key=option.key,
        value=value,
    )


def _validate_option_value(
    option: ConfigOption,
    value: ConfigValue,
    *,
    path: Path | None = None,
) -> None:
    if value is None:
        if option.optional:
            return
        raise _option_error(option, "value may not be empty", path=path, value=value)

    valid_type = isinstance(value, option.value_type)
    if option.value_type in (bool, int):
        valid_type = type(value) is option.value_type
    if not valid_type:
        raise _option_error(
            option,
            f"value {value!r} must have type {option.value_type.__name__}",
            path=path,
            value=value,
        )

    if option.choices and value not in option.choices:
        choices = ", ".join(option.choices)
        raise _option_error(
            option,
            f"invalid value {value!r}; expected one of: {choices}",
            path=path,
            value=value,
        )

    if option.validator is not None:
        try:
            option.validator(value)
        except ValueError as exc:
            raise _option_error(option, str(exc), path=path, value=value) from exc


def _coerce_option_value(
    option: ConfigOption,
    raw_value: object,
    *,
    base_dir: Path,
    path: Path | None = None,
) -> ConfigValue:
    """Convert one raw value using its metadata before final validation."""

    value: ConfigValue
    if raw_value is None:
        value = None
    elif option.value_type is bool:
        if type(raw_value) is bool:
            value = raw_value
        elif isinstance(raw_value, str) and raw_value.strip().lower() in _BOOLEAN_VALUES:
            value = _BOOLEAN_VALUES[raw_value.strip().lower()]
        else:
            raise _option_error(
                option,
                f"invalid boolean value {raw_value!r}",
                path=path,
                value=raw_value,
            )
    elif option.value_type is int:
        if type(raw_value) is int:
            value = raw_value
        elif isinstance(raw_value, str) and _INTEGER_PATTERN.fullmatch(raw_value.strip()):
            value = int(raw_value.strip(), 10)
        else:
            raise _option_error(
                option,
                f"invalid integer value {raw_value!r}",
                path=path,
                value=raw_value,
            )
    elif option.value_type is str:
        if not isinstance(raw_value, str):
            raise _option_error(
                option,
                f"value must have type str, got {type(raw_value).__name__}",
                path=path,
                value=raw_value,
            )
        value = raw_value.strip()
        if not value and option.optional:
            value = None
        elif option.choices:
            value = value.lower()
    else:  # pragma: no cover - guarded by the static metadata table
        raise RuntimeError(f"unsupported configuration type {option.value_type!r}")

    if option.is_path and value is not None:
        # INI paths use the config directory; CLI paths use the process cwd.
        value_path = Path(value).expanduser()
        if not value_path.is_absolute():
            value_path = base_dir / value_path
        value = str(value_path.resolve(strict=False))

    _validate_option_value(option, value, path=path)
    return value


def _default_values() -> ConfigOverrides:
    """Build validated defaults through the same conversion path as user input."""

    values: ConfigOverrides = {}
    for option in CONFIG_OPTIONS:
        value = _coerce_option_value(
            option,
            option.default,
            base_dir=Path.cwd(),
        )
        values.setdefault(option.section, {})[option.key] = value
    return values


def _build_config(values: ConfigOverrides) -> HypervecConfig:
    """Materialize the nested value map as immutable typed config objects."""

    return HypervecConfig(
        server=ServerConfig(**values["server"]),
        defaults=IndexDefaultsConfig(**values["defaults"]),
        logging=LoggingConfig(**values["logging"]),
    )


def default_config() -> HypervecConfig:
    """Build the complete default snapshot from CONFIG_OPTIONS."""

    config = _build_config(_default_values())
    validate_config(config, require_data_root=False)
    return config


def load_config_file(path: str | Path) -> ConfigOverrides:
    """Read and validate explicit values from an INI configuration file."""

    config_path = Path(path).expanduser().resolve(strict=False)
    if not config_path.exists():
        raise ConfigError("configuration file does not exist", path=config_path)
    if not config_path.is_file():
        raise ConfigError("configuration path is not a regular file", path=config_path)

    parser = configparser.ConfigParser(
        interpolation=None,
        strict=True,
        allow_no_value=False,
        empty_lines_in_values=False,
    )
    # Preserve key case so misspelled capitalization is rejected, not normalized.
    parser.optionxform = str
    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            parser.read_file(config_file)
    except (OSError, UnicodeError, configparser.Error) as exc:
        raise ConfigError(f"unable to read configuration: {exc}", path=config_path) from exc

    if parser.defaults():
        key = next(iter(parser.defaults()))
        raise ConfigError(
            "the DEFAULT section is not supported",
            path=config_path,
            section=parser.default_section,
            key=key,
        )

    overrides: ConfigOverrides = {}
    allowed_sections = {option.section for option in CONFIG_OPTIONS}
    for section in parser.sections():
        if section not in allowed_sections:
            raise ConfigError(
                "unknown configuration section",
                path=config_path,
                section=section,
            )
        for key, raw_value in parser.items(section, raw=True):
            option = _OPTIONS_BY_NAME.get((section, key))
            if option is None:
                raise ConfigError(
                    "unknown configuration option",
                    path=config_path,
                    section=section,
                    key=key,
                    value=raw_value,
                )
            value = _coerce_option_value(
                option,
                raw_value,
                base_dir=config_path.parent,
                path=config_path,
            )
            overrides.setdefault(section, {})[key] = value
    return overrides


def resolve_config(
    config_path: str | Path | None = None,
    cli_overrides: Mapping[str, object] | None = None,
) -> HypervecConfig:
    """Merge defaults, file values, and explicit CLI values, then validate."""

    # Later layers overwrite earlier layers: defaults < INI < explicit CLI.
    values = _default_values()
    if config_path is not None:
        for section, section_values in load_config_file(config_path).items():
            values[section].update(section_values)

    for cli_dest, raw_value in (cli_overrides or {}).items():
        option = _OPTIONS_BY_CLI_DEST.get(cli_dest)
        if option is None:
            raise ConfigError(f"unknown CLI configuration override {cli_dest!r}")
        values[option.section][option.key] = _coerce_option_value(
            option,
            raw_value,
            base_dir=Path.cwd(),
        )

    config = _build_config(values)
    validate_config(config)
    return config


def validate_config(
    config: HypervecConfig,
    *,
    require_data_root: bool = True,
) -> None:
    """Validate option values and constraints that span multiple options."""

    # Revalidate materialized objects so direct callers cannot bypass metadata.
    for option in CONFIG_OPTIONS:
        if option.section == "logging":
            continue
        section_value = getattr(config, option.field_path[0])
        value = getattr(section_value, option.field_path[1])
        _validate_option_value(option, value)

    if require_data_root and not config.server.data_root:
        raise ConfigError(
            "value is required to start the HTTP server",
            section="server",
            key="data_root",
        )
    if bool(config.server.certfile) != bool(config.server.keyfile):
        raise ConfigError(
            "certfile and keyfile must be configured together",
            section="server",
        )
    _validate_logging_config(config.logging)


def _validate_logging_config(config: LoggingConfig) -> None:
    for option in CONFIG_OPTIONS:
        if option.section == "logging":
            _validate_option_value(option, getattr(config, option.key))

    if config.enable_logging and not (config.log_to_stderr or config.log_to_file):
        raise ConfigError(
            "at least one logging output must be enabled",
            section="logging",
        )
    if config.log_to_file and not config.log_file_path:
        raise ConfigError(
            "log_file_path is required when log_to_file is enabled",
            section="logging",
            key="log_file_path",
        )


def configure_logging(config: LoggingConfig) -> None:
    """Apply handlers owned by HyperVector without modifying the root logger."""

    _validate_logging_config(config)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT)
    new_handlers: list[logging.Handler] = []

    # Build every replacement handler first. A file-open failure must leave the
    # currently active logger configuration untouched.
    if config.enable_logging:
        if config.log_to_stderr:
            stderr_handler = logging.StreamHandler(sys.stderr)
            stderr_handler.setFormatter(formatter)
            setattr(stderr_handler, _LOG_HANDLER_MARKER, True)
            new_handlers.append(stderr_handler)

        if config.log_to_file:
            log_path = Path(config.log_file_path or "").expanduser()
            try:
                file_handler = logging.FileHandler(
                    log_path,
                    mode="a",
                    encoding="utf-8",
                )
            except OSError as exc:
                for handler in new_handlers:
                    handler.close()
                raise ConfigError(
                    f"unable to open log file: {exc}",
                    path=log_path,
                    section="logging",
                    key="log_file_path",
                ) from exc
            file_handler.setFormatter(formatter)
            setattr(file_handler, _LOG_HANDLER_MARKER, True)
            new_handlers.append(file_handler)

    logger = logging.getLogger("hypervec")
    # Only replace handlers created by this module; embedded callers may own
    # additional handlers on the same logger namespace.
    for handler in list(logger.handlers):
        if getattr(handler, _LOG_HANDLER_MARKER, False):
            logger.removeHandler(handler)
            handler.close()

    logger.disabled = not config.enable_logging
    logger.setLevel(
        getattr(logging, config.log_level.upper())
        if config.enable_logging
        else logging.CRITICAL + 1
    )
    logger.propagate = False
    for handler in new_handlers:
        logger.addHandler(handler)

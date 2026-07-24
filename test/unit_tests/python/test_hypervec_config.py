from __future__ import annotations

from dataclasses import asdict
import importlib.util
from pathlib import Path
import sys

import pytest


def load_config_module():
    module_path = Path(__file__).parents[3] / "src" / "python" / "hypervec_config.py"
    module_name = "hypervec_config_under_test"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


config_module = load_config_module()


def test_module_loads_without_compiled_hypervec_extension():
    assert config_module.__file__.endswith("src/python/hypervec_config.py")


def test_metadata_defines_each_supported_option_once():
    expected_options = {
        "server.data_root": None,
        "server.host": "127.0.0.1",
        "server.port": 8080,
        "server.server": "hypercorn",
        "server.enable_http2": True,
        "server.certfile": None,
        "server.keyfile": None,
        "defaults.default_index_type": "hnswflat",
        "defaults.default_metric_type": "l2",
        "logging.enable_logging": True,
        "logging.log_level": "info",
        "logging.log_to_stderr": True,
        "logging.log_to_file": False,
        "logging.log_file_path": None,
    }

    actual_options = {
        f"{option.section}.{option.key}": option.default
        for option in config_module.CONFIG_OPTIONS
    }
    assert actual_options == expected_options
    assert len({option.field_path for option in config_module.CONFIG_OPTIONS}) == 14
    assert len({option.cli_dest for option in config_module.CONFIG_OPTIONS}) == 14


def test_metadata_indexes_are_unique_and_complete():
    assert len(config_module._OPTIONS_BY_NAME) == len(config_module.CONFIG_OPTIONS)
    assert len(config_module._OPTIONS_BY_CLI_DEST) == len(
        config_module.CONFIG_OPTIONS
    )


def test_config_error_includes_source_context():
    error = config_module.ConfigError(
        "invalid value",
        path=Path("hypervec.ini"),
        section="server",
        key="port",
        value="invalid",
    )

    assert "hypervec.ini [server].port: invalid value" == str(error)
    assert error.value == "invalid"


@pytest.mark.parametrize("value", ["", "   ", None, 1])
def test_non_empty_validator_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="non-empty string"):
        config_module._validate_non_empty(value)


@pytest.mark.parametrize("value", [0, 65536, -1, True, "8080"])
def test_port_validator_rejects_values_outside_integer_range(value):
    with pytest.raises(ValueError, match="1..65535"):
        config_module._validate_port(value)


@pytest.mark.parametrize(
    ("option_name", "raw_value", "expected"),
    [
        (("server", "enable_http2"), "OFF", False),
        (("logging", "enable_logging"), "yes", True),
        (("server", "port"), "+9090", 9090),
        (("server", "server"), "UVICORN", "uvicorn"),
        (("defaults", "default_metric_type"), "COSINE", "cosine"),
    ],
)
def test_coerce_option_value_supports_typed_input(option_name, raw_value, expected):
    option = config_module._OPTIONS_BY_NAME[option_name]

    assert (
        config_module._coerce_option_value(
            option,
            raw_value,
            base_dir=Path.cwd(),
        )
        == expected
    )


@pytest.mark.parametrize(
    ("option_name", "raw_value", "expected"),
    [
        (("server", "port"), "8080.5", "invalid integer value"),
        (("server", "server"), "gunicorn", "hypercorn, uvicorn"),
        (("logging", "enable_logging"), "maybe", "invalid boolean value"),
        (("logging", "log_level"), "verbose", "debug, info"),
    ],
)
def test_coerce_option_value_reports_invalid_input(option_name, raw_value, expected):
    option = config_module._OPTIONS_BY_NAME[option_name]

    with pytest.raises(config_module.ConfigError) as error:
        config_module._coerce_option_value(
            option,
            raw_value,
            base_dir=Path.cwd(),
        )

    assert expected in str(error.value)
    assert f"[{option.section}].{option.key}" in str(error.value)


def test_coerce_paths_use_the_caller_base_directory(tmp_path):
    option = config_module._OPTIONS_BY_NAME[("server", "data_root")]

    value = config_module._coerce_option_value(
        option,
        "data",
        base_dir=tmp_path,
    )

    assert value == str((tmp_path / "data").resolve())


def test_default_values_and_build_config_are_typed():
    values = config_module._default_values()
    config = config_module._build_config(values)

    assert asdict(config) == values
    assert type(config.server.port) is int
    assert type(config.server.enable_http2) is bool
    assert type(config.logging.enable_logging) is bool

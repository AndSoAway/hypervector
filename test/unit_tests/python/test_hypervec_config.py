from __future__ import annotations

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

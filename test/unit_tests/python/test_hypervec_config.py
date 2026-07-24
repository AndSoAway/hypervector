from __future__ import annotations

from dataclasses import asdict, replace
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


def test_default_config_is_complete_and_typed():
    config = config_module.default_config()

    assert asdict(config) == {
        "server": {
            "data_root": None,
            "host": "127.0.0.1",
            "port": 8080,
            "server": "hypercorn",
            "enable_http2": True,
            "certfile": None,
            "keyfile": None,
        },
        "defaults": {
            "default_index_type": "hnswflat",
            "default_metric_type": "l2",
        },
        "logging": {
            "enable_logging": True,
            "log_level": "info",
            "log_to_stderr": True,
            "log_to_file": False,
            "log_file_path": None,
        },
    }


def test_validation_can_defer_but_not_skip_required_data_root():
    config = config_module.default_config()

    config_module.validate_config(config, require_data_root=False)
    with pytest.raises(config_module.ConfigError, match="data_root"):
        config_module.validate_config(config)


def test_validation_rejects_incomplete_cross_field_combinations(tmp_path):
    config = config_module.default_config()
    server = replace(config.server, data_root=str(tmp_path), certfile="server.crt")
    with pytest.raises(config_module.ConfigError, match="configured together"):
        config_module.validate_config(replace(config, server=server))

    logging_config = replace(config.logging, log_to_stderr=False)
    with pytest.raises(config_module.ConfigError, match="logging output"):
        config_module.validate_config(
            replace(
                config,
                server=replace(config.server, data_root=str(tmp_path)),
                logging=logging_config,
            )
        )

    file_logging = replace(config.logging, log_to_file=True, log_file_path=None)
    with pytest.raises(config_module.ConfigError, match="log_file_path"):
        config_module.validate_config(
            replace(
                config,
                server=replace(config.server, data_root=str(tmp_path)),
                logging=file_logging,
            )
        )


def test_file_and_cli_values_merge_over_defaults(tmp_path, monkeypatch):
    config_path = tmp_path / "hypervec.ini"
    config_path.write_text(
        """\
[server]
data_root = data
port = 8081
server = UVICORN
enable_http2 = false

[defaults]
default_index_type = IVFFLAT
default_metric_type = COSINE

[logging]
log_level = WARNING
""",
        encoding="utf-8",
    )
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    monkeypatch.chdir(work_dir)

    config = config_module.resolve_config(
        config_path,
        {"port": 9090, "enable_http2": True, "default_metric_type": "ip"},
    )

    assert config.server.data_root == str((tmp_path / "data").resolve())
    assert config.server.port == 9090
    assert config.server.server == "uvicorn"
    assert config.server.enable_http2 is True
    assert config.defaults.default_index_type == "ivfflat"
    assert config.defaults.default_metric_type == "ip"
    assert config.logging.log_level == "warning"
    assert config.server.host == "127.0.0.1"


def test_only_data_root_uses_all_other_metadata_defaults(tmp_path):
    config = config_module.resolve_config(None, {"data_root": str(tmp_path / "data")})
    expected = asdict(config_module.default_config())
    expected["server"]["data_root"] = str((tmp_path / "data").resolve())

    assert asdict(config) == expected


def test_missing_non_file_and_unreadable_config_paths_fail(tmp_path, monkeypatch):
    missing_path = tmp_path / "missing.ini"
    with pytest.raises(config_module.ConfigError, match=str(missing_path)):
        config_module.load_config_file(missing_path)

    with pytest.raises(config_module.ConfigError, match="not a regular file"):
        config_module.load_config_file(tmp_path)

    unreadable_path = tmp_path / "unreadable.ini"
    unreadable_path.write_text("", encoding="utf-8")
    original_open = Path.open

    def deny_open(path, *args, **kwargs):
        if path == unreadable_path:
            raise PermissionError("permission denied")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", deny_open)
    with pytest.raises(config_module.ConfigError, match="permission denied"):
        config_module.load_config_file(unreadable_path)


def test_empty_file_has_no_overrides_but_still_requires_data_root(tmp_path):
    config_path = tmp_path / "empty.ini"
    config_path.write_text("", encoding="utf-8")

    assert config_module.load_config_file(config_path) == {}
    resolved = config_module.resolve_config(
        config_path, {"data_root": str(tmp_path / "data")}
    )
    assert resolved.server.port == 8080
    assert resolved.logging.log_level == "info"

    with pytest.raises(config_module.ConfigError, match="data_root"):
        config_module.resolve_config(config_path)


@pytest.mark.parametrize(
    "contents",
    [
        "port = 8080\n",
        "[server]\nport = 8080\nport = 8081\n",
        "[server]\nport = 8080\n[server]\nhost = localhost\n",
    ],
)
def test_invalid_ini_and_duplicates_report_the_source_path(tmp_path, contents):
    config_path = tmp_path / "invalid.ini"
    config_path.write_text(contents, encoding="utf-8")

    with pytest.raises(config_module.ConfigError) as error:
        config_module.load_config_file(config_path)
    assert str(config_path) in str(error.value)


@pytest.mark.parametrize(
    ("contents", "expected"),
    [
        ("[unknown]\nvalue = 1\n", "unknown configuration section"),
        ("[server]\nunknown = 1\n", "[server].unknown"),
        ("[Server]\nport = 8080\n", "[Server]"),
        ("[server]\nPort = 8080\n", "[server].Port"),
        ("[DEFAULT]\nport = 8080\n", "DEFAULT"),
    ],
)
def test_unknown_sections_and_options_are_rejected(tmp_path, contents, expected):
    config_path = tmp_path / "unknown.ini"
    config_path.write_text(contents, encoding="utf-8")

    with pytest.raises(config_module.ConfigError) as error:
        config_module.load_config_file(config_path)
    assert expected in str(error.value)

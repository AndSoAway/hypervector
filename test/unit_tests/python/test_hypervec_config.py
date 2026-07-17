from __future__ import annotations

from dataclasses import asdict, replace
import importlib.util
import io
import logging
from pathlib import Path
from types import SimpleNamespace
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


@pytest.fixture
def hypervec_logger():
    logger = logging.getLogger("hypervec")
    marker = config_module._LOG_HANDLER_MARKER
    for handler in list(logger.handlers):
        if getattr(handler, marker, False):
            logger.removeHandler(handler)
            handler.close()

    original_handlers = list(logger.handlers)
    original_disabled = logger.disabled
    original_level = logger.level
    original_propagate = logger.propagate
    yield logger

    for handler in list(logger.handlers):
        if handler not in original_handlers:
            logger.removeHandler(handler)
            handler.close()
    logger.disabled = original_disabled
    logger.setLevel(original_level)
    logger.propagate = original_propagate


def logging_config(**changes):
    return replace(config_module.default_config().logging, **changes)


def test_module_loads_without_compiled_hypervec_extension():
    assert config_module.__file__.endswith("src/python/hypervec_config.py")
    assert config_module.default_config().server.server == "hypercorn"


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
    assert type(config.server.port) is int
    assert type(config.server.enable_http2) is bool
    assert type(config.logging.enable_logging) is bool


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

    server = replace(config.server, data_root=str(tmp_path), keyfile="server.key")
    with pytest.raises(config_module.ConfigError, match="configured together"):
        config_module.validate_config(replace(config, server=server))

    logging = replace(config.logging, log_to_stderr=False)
    with pytest.raises(config_module.ConfigError, match="logging output"):
        config_module.validate_config(
            replace(config, server=replace(config.server, data_root=str(tmp_path)), logging=logging)
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


def test_omitted_cli_options_do_not_override_file_values(tmp_path):
    config_path = tmp_path / "hypervec.ini"
    config_path.write_text(
        """\
[server]
data_root = data
host = file-host
port = 8181
server = uvicorn
""",
        encoding="utf-8",
    )

    config = config_module.resolve_config(config_path, {"log_level": "error"})

    assert config.server.host == "file-host"
    assert config.server.port == 8181
    assert config.server.server == "uvicorn"
    assert config.logging.log_level == "error"


def test_namespace_extraction_and_sample_rendering_are_metadata_driven():
    namespace = SimpleNamespace(port=9000, log_to_stderr=False, config="ignored.ini")
    assert config_module.cli_overrides_from_namespace(namespace) == {
        "port": 9000,
        "log_to_stderr": False,
    }

    sample = config_module.render_sample_config()
    assert sample.count("[server]") == 1
    assert sample.count("[defaults]") == 1
    assert sample.count("[logging]") == 1
    assert "host = 127.0.0.1" in sample
    assert "port = 8080" in sample
    assert "enable_http2 = true" in sample
    assert "default_index_type = hnswflat" in sample
    assert "default_metric_type = l2" in sample
    assert "enable_logging = true" in sample
    assert "log_file_path =\n" in sample
    assert sample.endswith("\n")


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
    resolved = config_module.resolve_config(config_path, {"data_root": str(tmp_path / "data")})
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


def test_supported_types_and_optional_values_are_parsed(tmp_path):
    config_path = tmp_path / "types.ini"
    config_path.write_text(
        """\
[server]
data_root =
host = localhost
port = +9090
enable_http2 = OFF
certfile =

[defaults]
default_index_type = HNSWPQ
default_metric_type = IP

[logging]
enable_logging = YES
log_to_stderr = on
log_to_file = 0
log_file_path =
""",
        encoding="utf-8",
    )

    values = config_module.load_config_file(config_path)
    assert values == {
        "server": {
            "data_root": None,
            "host": "localhost",
            "port": 9090,
            "enable_http2": False,
            "certfile": None,
        },
        "defaults": {
            "default_index_type": "hnswpq",
            "default_metric_type": "ip",
        },
        "logging": {
            "enable_logging": True,
            "log_to_stderr": True,
            "log_to_file": False,
            "log_file_path": None,
        },
    }


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("true", True),
        ("TRUE", True),
        ("yes", True),
        ("on", True),
        ("1", True),
        ("false", False),
        ("FALSE", False),
        ("no", False),
        ("off", False),
        ("0", False),
    ],
)
def test_all_supported_boolean_tokens(tmp_path, raw_value, expected):
    config_path = tmp_path / "boolean.ini"
    config_path.write_text(
        f"[logging]\nenable_logging = {raw_value}\n",
        encoding="utf-8",
    )

    values = config_module.load_config_file(config_path)
    assert values["logging"]["enable_logging"] is expected


@pytest.mark.parametrize("port", ["0", "65536", "-1"])
def test_port_range_boundaries_are_rejected(tmp_path, port):
    config_path = tmp_path / "invalid-port.ini"
    config_path.write_text(f"[server]\nport = {port}\n", encoding="utf-8")

    with pytest.raises(config_module.ConfigError) as error:
        config_module.load_config_file(config_path)
    message = str(error.value)
    assert "[server].port" in message
    assert port in message
    assert "1..65535" in message


@pytest.mark.parametrize(
    ("section", "key", "value", "expected"),
    [
        ("server", "port", "8080.5", "invalid integer value"),
        ("server", "port", "8080x", "invalid integer value"),
        ("server", "port", "70000", "1..65535"),
        ("server", "server", "gunicorn", "hypercorn, uvicorn"),
        ("defaults", "default_index_type", "diskann", "flat, ivfflat"),
        ("defaults", "default_metric_type", "manhattan", "l2, ip, cosine"),
        ("logging", "enable_logging", "maybe", "invalid boolean value"),
        ("logging", "log_level", "verbose", "debug, info, warning, error, critical"),
    ],
)
def test_invalid_values_report_option_value_and_expectation(
    tmp_path, section, key, value, expected
):
    config_path = tmp_path / "invalid-value.ini"
    config_path.write_text(f"[{section}]\n{key} = {value}\n", encoding="utf-8")

    with pytest.raises(config_module.ConfigError) as error:
        config_module.load_config_file(config_path)
    message = str(error.value)
    assert f"[{section}].{key}" in message
    assert value in message
    assert expected in message


def test_config_paths_are_file_relative_and_cli_paths_are_cwd_relative(
    tmp_path, monkeypatch
):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "hypervec.ini"
    config_path.write_text(
        """\
[server]
data_root = data
certfile = tls/server.crt
keyfile = tls/server.key
""",
        encoding="utf-8",
    )
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    monkeypatch.chdir(work_dir)

    from_file = config_module.resolve_config(config_path)
    assert from_file.server.data_root == str((config_dir / "data").resolve())
    assert from_file.server.certfile == str((config_dir / "tls/server.crt").resolve())

    from_cli = config_module.resolve_config(
        config_path,
        {"data_root": "cli-data", "certfile": None, "keyfile": None},
    )
    assert from_cli.server.data_root == str((work_dir / "cli-data").resolve())


def test_no_config_path_does_not_search_the_working_directory(tmp_path, monkeypatch):
    (tmp_path / "hypervec.ini").write_text("[server]\nport = 1234\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    config = config_module.resolve_config(None, {"data_root": "data"})
    assert config.server.port == 8080


def test_export_sample_matches_repository_golden_file_and_refuses_overwrite(tmp_path):
    golden_path = Path(__file__).parents[3] / "configs" / "hypervec.ini.sample"
    expected = golden_path.read_text(encoding="utf-8")
    assert config_module.render_sample_config() == expected

    output_path = tmp_path / "hypervec.ini"
    config_module.export_sample_config(output_path)
    assert output_path.read_text(encoding="utf-8") == expected

    with pytest.raises(config_module.ConfigError, match="already exists"):
        config_module.export_sample_config(output_path)


def test_configure_logging_can_disable_all_owned_output(hypervec_logger, capsys):
    config_module.configure_logging(logging_config(enable_logging=False))

    assert hypervec_logger.disabled
    assert not [
        handler
        for handler in hypervec_logger.handlers
        if getattr(handler, config_module._LOG_HANDLER_MARKER, False)
    ]
    logging.getLogger("hypervec.test.disabled").error("disabled message")
    assert "disabled message" not in capsys.readouterr().err


def test_configure_logging_stderr_only_sets_level_and_preserves_external_handlers(
    hypervec_logger, capsys
):
    external_stream = io.StringIO()
    external_handler = logging.StreamHandler(external_stream)
    hypervec_logger.addHandler(external_handler)
    root_handlers = list(logging.getLogger().handlers)

    config_module.configure_logging(logging_config(log_level="warning"))
    test_logger = logging.getLogger("hypervec.test.stderr")
    test_logger.info("filtered message")
    test_logger.warning("stderr message")

    captured = capsys.readouterr().err
    assert "filtered message" not in captured
    assert "stderr message" in captured
    assert external_handler in hypervec_logger.handlers
    assert logging.getLogger().handlers == root_handlers
    owned_handlers = [
        handler
        for handler in hypervec_logger.handlers
        if getattr(handler, config_module._LOG_HANDLER_MARKER, False)
    ]
    assert len(owned_handlers) == 1
    assert type(owned_handlers[0]) is logging.StreamHandler


def test_configure_logging_file_only_uses_utf8_append_and_is_idempotent(
    tmp_path, hypervec_logger, capsys
):
    log_path = tmp_path / "hypervec.log"
    log_path.write_text("existing line\n", encoding="utf-8")
    config = logging_config(
        log_to_stderr=False,
        log_to_file=True,
        log_file_path=str(log_path),
    )

    config_module.configure_logging(config)
    logging.getLogger("hypervec.test.file").info("first file message")
    config_module.configure_logging(config)
    logging.getLogger("hypervec.test.file").info("second file message")

    contents = log_path.read_text(encoding="utf-8")
    assert contents.startswith("existing line\n")
    assert contents.count("first file message") == 1
    assert contents.count("second file message") == 1
    assert capsys.readouterr().err == ""
    owned_handlers = [
        handler
        for handler in hypervec_logger.handlers
        if getattr(handler, config_module._LOG_HANDLER_MARKER, False)
    ]
    assert len(owned_handlers) == 1
    assert isinstance(owned_handlers[0], logging.FileHandler)
    assert owned_handlers[0].encoding.lower().replace("-", "") == "utf8"


def test_configure_logging_can_write_stderr_and_file(
    tmp_path, hypervec_logger, capsys
):
    log_path = tmp_path / "hypervec.log"
    config_module.configure_logging(
        logging_config(log_to_file=True, log_file_path=str(log_path))
    )

    logging.getLogger("hypervec.test.both").error("双路日志")

    assert "双路日志" in capsys.readouterr().err
    assert "双路日志" in log_path.read_text(encoding="utf-8")
    owned_handlers = [
        handler
        for handler in hypervec_logger.handlers
        if getattr(handler, config_module._LOG_HANDLER_MARKER, False)
    ]
    assert len(owned_handlers) == 2


def test_configure_logging_reports_invalid_targets_and_file_open_errors(
    tmp_path, hypervec_logger
):
    with pytest.raises(config_module.ConfigError, match="logging output"):
        config_module.configure_logging(
            logging_config(log_to_stderr=False, log_to_file=False)
        )

    with pytest.raises(config_module.ConfigError, match="log_file_path"):
        config_module.configure_logging(
            logging_config(log_to_stderr=False, log_to_file=True, log_file_path=None)
        )

    log_path = tmp_path / "missing-parent" / "hypervec.log"
    with pytest.raises(config_module.ConfigError) as error:
        config_module.configure_logging(
            logging_config(
                log_to_stderr=False,
                log_to_file=True,
                log_file_path=str(log_path),
            )
        )
    assert str(log_path) in str(error.value)
    assert "unable to open log file" in str(error.value)


def test_configure_logging_does_not_modify_an_injected_external_logger(
    hypervec_logger,
):
    injected_logger = logging.getLogger("application.injected-engine")
    injected_handler = logging.StreamHandler(io.StringIO())
    injected_logger.addHandler(injected_handler)
    original_level = injected_logger.level
    original_disabled = injected_logger.disabled
    try:
        config_module.configure_logging(logging_config())
        assert injected_handler in injected_logger.handlers
        assert injected_logger.level == original_level
        assert injected_logger.disabled == original_disabled
    finally:
        injected_logger.removeHandler(injected_handler)
        injected_handler.close()

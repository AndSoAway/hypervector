from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


class FakeEngine:
    def health(self):
        return {"status": "ok"}

    def list_collections(self):
        return ["demo"]

    def has_collection(self, collection_name):
        return collection_name == "demo"

    def describe_collection(self, collection_name):
        return {"collection_name": collection_name}

    def describe_collections(self):
        return [
            {"collection_name": "demo"},
            {"collection_name": "other"},
        ]

    def supported_index_examples(self):
        return [
            {
                "index_type": "IndexIVFFlat",
                "cpp_class": "hypervec.IndexIVFFlat",
                "metric_types": ["L2", "IP", "COSINE"],
                "code_size": "d * sizeof(float)",
                "params": [{"name": "nlist", "type": "int", "default": 1024, "required": False}],
                "description": "IVF with raw float vectors stored per list.",
            }
        ]

    def get_version(self, collection_name):
        return {
            "collection_name": collection_name,
            "version": 2,
            "updated_at": 1.0,
            "index_checksum": "sha256:abc",
            "index_size_bytes": 4,
        }

    def sync_check(self, collection_name, *, client_version, client_checksum=None):
        return {
            "needs_sync": client_version != 2 or client_checksum != "sha256:abc",
            "server_version": 2,
            "client_version": client_version,
        }

    def index_path_for_download(self, collection_name):
        return Path(__file__)

    def upload_index(self, collection_name, source_path, *, version=None, checksum=None):
        return {"uploaded": True, "collection_name": collection_name, "version": version}


def load_http_module():
    root = Path(__file__).parents[3] / "src" / "python"
    package = type(sys)("hypervec")
    package.__path__ = [str(root)]
    sys.modules.setdefault("hypervec", package)
    spec = importlib.util.spec_from_file_location(
        "hypervec.hypervec_http_server",
        root / "hypervec_http_server.py",
        submodule_search_locations=[str(root)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["hypervec.hypervec_http_server"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_hypervec_http_server_sync_routes(tmp_path):
    import pytest

    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    module = load_http_module()
    client = TestClient(module.create_app(data_root=str(tmp_path), engine=FakeEngine()))

    described = client.get("/collections/describe")
    assert described.json()["collections"] == [
        {"collection_name": "demo"},
        {"collection_name": "other"},
    ]

    assert client.get("/collections/demo/version").json()["version"] == 2
    sync = client.post(
        "/collections/demo/sync-check",
        json={"client_version": 1, "client_checksum": "sha256:old"},
    )
    assert sync.json()["needs_sync"]

    download = client.get("/collections/demo/index")
    assert download.status_code == 200
    assert download.headers["x-hypervec-collection-version"] == "2"

    upload = client.put("/collections/demo/index?version=3", content=b"fake-index")
    assert upload.json()["uploaded"]


def test_hypervec_http_server_examples_route(tmp_path):
    import pytest

    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    module = load_http_module()
    client = TestClient(module.create_app(data_root=str(tmp_path), engine=FakeEngine()))

    res = client.get("/examples")
    payload = res.json()

    assert res.status_code == 200
    assert [item["index_type"] for item in payload["examples"]] == ["IndexIVFFlat"]
    assert payload["examples"][0]["cpp_class"] == "hypervec.IndexIVFFlat"


def test_config_cli_only_exposes_explicit_business_overrides():
    module = load_http_module()
    parser = module.build_argument_parser()
    help_text = parser.format_help()

    for option in (
        "--config",
        "--export-sample-config",
        "--data-root",
        "--enable-http2",
        "--no-enable-http2",
        "--default-index-type",
        "--default-metric-type",
        "--enable-logging",
        "--no-enable-logging",
        "--log-to-stderr",
        "--no-log-to-stderr",
        "--log-to-file",
        "--no-log-to-file",
        "--log-file-path",
    ):
        assert option in help_text

    defaults = parser.parse_args([])
    assert module.cli_overrides_from_namespace(defaults) == {}

    args = parser.parse_args(
        [
            "--data-root",
            "data",
            "--host",
            "localhost",
            "--port",
            "9090",
            "--server",
            "uvicorn",
            "--no-enable-http2",
            "--default-index-type",
            "ivfpq",
            "--default-metric-type",
            "cosine",
            "--log-level",
            "warning",
            "--certfile",
            "server.crt",
            "--keyfile",
            "server.key",
            "--no-enable-logging",
            "--no-log-to-stderr",
            "--log-to-file",
            "--log-file-path",
            "hypervec.log",
        ]
    )
    assert module.cli_overrides_from_namespace(args) == {
        "data_root": "data",
        "host": "localhost",
        "port": 9090,
        "server": "uvicorn",
        "enable_http2": False,
        "certfile": "server.crt",
        "keyfile": "server.key",
        "default_index_type": "ivfpq",
        "default_metric_type": "cosine",
        "enable_logging": False,
        "log_level": "warning",
        "log_to_stderr": False,
        "log_to_file": True,
        "log_file_path": "hypervec.log",
    }


def test_main_keeps_legacy_cli_only_startup_compatible(tmp_path, monkeypatch):
    module = load_http_module()
    started = []
    configured_logging = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "run_server", started.append)
    monkeypatch.setattr(module, "configure_logging", configured_logging.append)

    module.main(
        [
            "--data-root",
            "data",
            "--host",
            "legacy-host",
            "--port",
            "9090",
            "--server",
            "uvicorn",
            "--log-level",
            "warning",
            "--certfile",
            "server.crt",
            "--keyfile",
            "server.key",
        ]
    )

    assert len(started) == 1
    config = started[0]
    assert config.server.data_root == str((tmp_path / "data").resolve())
    assert config.server.host == "legacy-host"
    assert config.server.port == 9090
    assert config.server.server == "uvicorn"
    assert config.server.enable_http2 is True
    assert config.defaults.default_index_type == "hnswflat"
    assert config.defaults.default_metric_type == "l2"
    assert config.server.certfile == str((tmp_path / "server.crt").resolve())
    assert config.server.keyfile == str((tmp_path / "server.key").resolve())
    assert config.logging.log_level == "warning"
    assert configured_logging == [config.logging]


def test_main_merges_config_file_and_cli_before_starting(tmp_path, monkeypatch):
    module = load_http_module()
    config_path = tmp_path / "hypervec.ini"
    config_path.write_text(
        """\
[server]
data_root = data
host = config-host
port = 8081
server = hypercorn
enable_http2 = false

[defaults]
default_index_type = hnswlvq
default_metric_type = l2

[logging]
log_level = info
""",
        encoding="utf-8",
    )
    captured = []
    configured_logging = []
    monkeypatch.setattr(module, "run_server", captured.append)
    monkeypatch.setattr(module, "configure_logging", configured_logging.append)

    module.main(
        [
            "--config",
            str(config_path),
            "--host",
            "cli-host",
            "--port",
            "9090",
            "--server",
            "uvicorn",
            "--enable-http2",
            "--default-index-type",
            "flat",
            "--default-metric-type",
            "ip",
            "--log-level",
            "error",
        ]
    )

    assert len(captured) == 1
    config = captured[0]
    assert config.server.data_root == str((tmp_path / "data").resolve())
    assert config.server.host == "cli-host"
    assert config.server.port == 9090
    assert config.server.server == "uvicorn"
    assert config.server.enable_http2 is True
    assert config.defaults.default_index_type == "flat"
    assert config.defaults.default_metric_type == "ip"
    assert config.logging.log_level == "error"
    assert configured_logging == [config.logging]


def test_main_starts_uvicorn_using_only_config_file(tmp_path, monkeypatch):
    module = load_http_module()
    config_path = tmp_path / "hypervec.ini"
    config_path.write_text(
        """\
[server]
data_root = data
host = uvicorn-host
port = 8443
server = uvicorn
certfile = tls/server.crt
keyfile = tls/server.key

[logging]
log_level = error
""",
        encoding="utf-8",
    )
    app = object()
    app_data_roots = []
    run_calls = []
    uvicorn = ModuleType("uvicorn")
    uvicorn.run = lambda *args, **kwargs: run_calls.append((args, kwargs))
    monkeypatch.setattr(
        module,
        "create_app",
        lambda *, data_root: app_data_roots.append(data_root) or app,
    )
    monkeypatch.setattr(module, "configure_logging", lambda config: None)
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)

    module.main(["--config", str(config_path)])

    assert app_data_roots == [str((tmp_path / "data").resolve())]
    assert run_calls == [
        (
            (app,),
            {
                "host": "uvicorn-host",
                "port": 8443,
                "log_level": "error",
                "access_log": True,
                "ssl_certfile": str((tmp_path / "tls/server.crt").resolve()),
                "ssl_keyfile": str((tmp_path / "tls/server.key").resolve()),
            },
        )
    ]


def test_main_starts_hypercorn_using_only_config_file(tmp_path, monkeypatch):
    module = load_http_module()
    config_path = tmp_path / "hypervec.ini"
    config_path.write_text(
        """\
[server]
data_root = data
host = hypercorn-host
port = 9443
server = hypercorn
certfile = tls/server.crt
keyfile = tls/server.key

[logging]
log_level = critical
""",
        encoding="utf-8",
    )
    app = object()
    app_data_roots = []
    served = []

    class FakeHypercornConfig:
        pass

    async def serve(fake_app, fake_config):
        served.append((fake_app, fake_config))

    hypercorn = ModuleType("hypercorn")
    hypercorn.__path__ = []
    hypercorn_asyncio = ModuleType("hypercorn.asyncio")
    hypercorn_asyncio.serve = serve
    hypercorn_config = ModuleType("hypercorn.config")
    hypercorn_config.Config = FakeHypercornConfig
    monkeypatch.setitem(sys.modules, "hypercorn", hypercorn)
    monkeypatch.setitem(sys.modules, "hypercorn.asyncio", hypercorn_asyncio)
    monkeypatch.setitem(sys.modules, "hypercorn.config", hypercorn_config)
    monkeypatch.setattr(
        module,
        "create_app",
        lambda *, data_root: app_data_roots.append(data_root) or app,
    )
    monkeypatch.setattr(module, "configure_logging", lambda config: None)

    module.main(["--config", str(config_path)])

    assert app_data_roots == [str((tmp_path / "data").resolve())]
    assert len(served) == 1
    fake_app, fake_config = served[0]
    assert fake_app is app
    assert fake_config.bind == ["hypercorn-host:9443"]
    assert fake_config.loglevel == "critical"
    assert fake_config.certfile == str((tmp_path / "tls/server.crt").resolve())
    assert fake_config.keyfile == str((tmp_path / "tls/server.key").resolve())
    assert fake_config.alpn_protocols == ["h2", "http/1.1"]


def test_main_reports_configuration_errors_with_argparse_exit(
    tmp_path, monkeypatch, capsys
):
    import pytest

    module = load_http_module()
    monkeypatch.setattr(
        module,
        "run_server",
        lambda config: (_ for _ in ()).throw(AssertionError("server started")),
    )

    cases = [
        ([], "[server].data_root"),
        (
            ["--data-root", str(tmp_path / "data"), "--certfile", "server.crt"],
            "configured together",
        ),
        (["--config", str(tmp_path / "missing.ini")], "configuration file does not exist"),
    ]
    for argv, expected in cases:
        with pytest.raises(SystemExit) as error:
            module.main(argv)
        assert error.value.code == 2
        stderr = capsys.readouterr().err
        assert "usage:" in stderr
        assert expected in stderr
        assert "Traceback" not in stderr


def test_export_sample_cli_does_not_start_server(tmp_path, monkeypatch):
    module = load_http_module()
    output_path = tmp_path / "hypervec.ini"

    def fail_if_started(config):
        raise AssertionError(f"server unexpectedly started with {config!r}")

    monkeypatch.setattr(module, "run_server", fail_if_started)
    module.main(["--export-sample-config", str(output_path)])

    golden_path = Path(__file__).parents[3] / "configs" / "hypervec.ini.sample"
    assert output_path.read_text(encoding="utf-8") == golden_path.read_text(encoding="utf-8")


def test_export_sample_cli_reports_existing_target(tmp_path, capsys):
    import pytest

    module = load_http_module()
    output_path = tmp_path / "hypervec.ini"
    output_path.write_text("existing", encoding="utf-8")

    with pytest.raises(SystemExit) as error:
        module.main(["--export-sample-config", str(output_path)])

    assert error.value.code == 2
    assert "already exists" in capsys.readouterr().err
    assert output_path.read_text(encoding="utf-8") == "existing"


def test_run_server_passes_logging_level_and_switch_to_uvicorn(tmp_path, monkeypatch):
    module = load_http_module()
    config = module.resolve_config(
        None,
        {
            "data_root": str(tmp_path),
            "server": "uvicorn",
            "log_level": "error",
            "enable_logging": False,
        },
    )
    app = object()
    run_calls = []
    uvicorn = ModuleType("uvicorn")
    uvicorn.run = lambda *args, **kwargs: run_calls.append((args, kwargs))
    monkeypatch.setattr(module, "create_app", lambda **kwargs: app)
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)

    module.run_server(config)

    assert run_calls == [
        (
            (app,),
            {
                "host": "127.0.0.1",
                "port": 8080,
                "log_level": "error",
                "access_log": False,
            },
        )
    ]


def test_run_server_passes_logging_level_and_switch_to_hypercorn(
    tmp_path, monkeypatch
):
    module = load_http_module()
    config = module.resolve_config(
        None,
        {
            "data_root": str(tmp_path),
            "log_level": "critical",
            "enable_logging": False,
            "enable_http2": False,
        },
    )
    app = object()
    served = []

    class FakeHypercornConfig:
        accesslog = "unchanged"

    async def serve(fake_app, fake_config):
        served.append((fake_app, fake_config))

    hypercorn = ModuleType("hypercorn")
    hypercorn.__path__ = []
    hypercorn_asyncio = ModuleType("hypercorn.asyncio")
    hypercorn_asyncio.serve = serve
    hypercorn_config = ModuleType("hypercorn.config")
    hypercorn_config.Config = FakeHypercornConfig
    monkeypatch.setitem(sys.modules, "hypercorn", hypercorn)
    monkeypatch.setitem(sys.modules, "hypercorn.asyncio", hypercorn_asyncio)
    monkeypatch.setitem(sys.modules, "hypercorn.config", hypercorn_config)
    monkeypatch.setattr(module, "create_app", lambda **kwargs: app)

    module.run_server(config)

    assert len(served) == 1
    fake_app, fake_config = served[0]
    assert fake_app is app
    assert fake_config.bind == ["127.0.0.1:8080"]
    assert fake_config.loglevel == "critical"
    assert fake_config.alpn_protocols == ["http/1.1"]
    assert fake_config.accesslog is None

"""Persistent startup configuration and a real isolated service launch."""

import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import httpx
import pytest

from rag_mcp.config import RagConfigError
from rag_mcp.startup_config import DEFAULT_CONFIG, load_service_config


def test_first_start_creates_config_without_overwriting_edits(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(os, "environ", {})
    path = load_service_config()
    assert path == tmp_path / ".atlas-rag" / "config.toml"
    assert os.environ["RAG_BACKEND"] == "qdrant"
    edited = DEFAULT_CONFIG.replace('"8011"', '"8099"')
    path.write_text(edited, encoding="utf-8")
    os.environ.clear()
    load_service_config()
    assert os.environ["RAG_SERVICE_PORT"] == "8099"
    assert path.read_text(encoding="utf-8") == edited


@pytest.mark.parametrize("content", ['[environment\n', '[environment]\nRAG_SERVICE_PORRT = "8011"',
                                     '[environment]\nRAG_SERVICE_PORT = 8011'])
def test_invalid_config_is_rejected(tmp_path, monkeypatch, content):
    path = tmp_path / "config.toml"
    path.write_text(content, encoding="utf-8")
    monkeypatch.setattr(os, "environ", {"RAG_CONFIG_FILE": str(path)})
    with pytest.raises(RagConfigError):
        load_service_config()


def test_rename_keeps_existing_config_and_implicit_data_root(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(os, "environ", {})
    legacy = tmp_path / ".praxis-rag" / "config.toml"
    legacy.parent.mkdir()
    content = '[environment]\nRAG_SERVICE_PORT = "8099"\nQDRANT_PATH = "existing-db"\n'
    legacy.write_text(content, encoding="utf-8")

    assert load_service_config() == legacy
    assert os.environ["RAG_SERVICE_PORT"] == "8099"
    assert os.environ["QDRANT_PATH"] == "existing-db"
    assert os.environ["RAG_SERVICE_DATA_ROOT"] == str(legacy.parent / "data")
    assert legacy.read_text(encoding="utf-8") == content
    assert not (tmp_path / ".atlas-rag").exists()


def test_new_config_and_explicit_override_take_precedence(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(os, "environ", {})
    for directory, port in ((".praxis-rag", "8099"), (".atlas-rag", "8100")):
        config = tmp_path / directory / "config.toml"
        config.parent.mkdir()
        config.write_text(f'[environment]\nRAG_SERVICE_PORT = "{port}"\n', encoding="utf-8")

    assert load_service_config() == tmp_path / ".atlas-rag" / "config.toml"
    assert os.environ["RAG_SERVICE_PORT"] == "8100"
    explicit = tmp_path / "explicit.toml"
    explicit.write_text('[environment]\nRAG_SERVICE_PORT = "8101"\n', encoding="utf-8")
    os.environ.clear()
    os.environ["RAG_CONFIG_FILE"] = str(explicit)
    os.environ["RAG_SERVICE_DATA_ROOT"] = str(tmp_path / "custom-data")
    assert load_service_config() == explicit
    assert os.environ["RAG_SERVICE_PORT"] == "8101"
    assert os.environ["RAG_SERVICE_DATA_ROOT"] == str(tmp_path / "custom-data")


def test_real_service_reads_config_and_environment_override(tmp_path):
    path = tmp_path / "config.toml"
    data = tmp_path / "data"
    path.write_text(DEFAULT_CONFIG.replace("~/.atlas-rag/data", data.as_posix()), encoding="utf-8")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("RAG_", "QDRANT_"))}
    env.update(RAG_CONFIG_FILE=str(path), RAG_SERVICE_PORT=str(port), QDRANT_PATH=str(tmp_path / "qdrant"))
    log = tmp_path / "service.log"
    with log.open("w", encoding="utf-8") as output:
        python = os.getenv("ATLAS_TEST_PYTHON", sys.executable)
        process = subprocess.Popen([python, "-m", "rag_mcp.http_service"], env=env,
                                   stdout=output, stderr=subprocess.STDOUT)
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", trust_env=False, timeout=2) as client:
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline and process.poll() is None:
                    try:
                        response = client.get("/health")
                        response.raise_for_status()
                        break
                    except httpx.HTTPError:
                        time.sleep(0.2)
                else:
                    pytest.fail(log.read_text(encoding="utf-8"))
                assert response.json()["backend"] == "qdrant"
                assert client.get("/console").status_code == 200
                assert data.is_dir()
                assert (tmp_path / "qdrant").is_dir()
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)

import base64
import hashlib
import io
import json
import tarfile
from types import SimpleNamespace

import pytest
from flask import Flask, render_template

from qhana_plugin_runner.markdown import register_markdown_filter
from qhana_plugin_runner.util.download_mathjax import (
    _verify_path_traversal,
    download_mathjax,
)
from qhana_plugin_runner.util.jinja_helpers import register_helpers
from tests.utils import MockResponse

MATHJAX_VERSION = "3.2.2"
METADATA_URL = f"https://registry.npmjs.org/mathjax/{MATHJAX_VERSION}"
TARBALL_URL = f"https://registry.npmjs.org/mathjax/-/mathjax-{MATHJAX_VERSION}.tgz"


@pytest.fixture
def mathjax_app(tmp_path):
    """Minimal Flask app whose ``static_folder`` points at a temp directory.

    ``download_mathjax`` only reads ``app.static_folder`` and ``app.config``,
    so a lightweight real app avoids touching the packaged static assets.
    """
    flask_app = Flask(__name__, static_folder=str(tmp_path))
    flask_app.config.update(
        MATHJAX_VERSION=MATHJAX_VERSION,
        MATHJAX_METADATA_URL=METADATA_URL,
    )
    return flask_app


@pytest.fixture
def template_app():
    """Lightweight app for rendering ``simple_template.html``.

    ``Flask("qhana_plugin_runner")`` resolves the packaged ``templates`` and
    ``static`` folders automatically. Only the Jinja helpers and markdown
    filter used by the template are registered, so this avoids the costly
    plugin discovery performed by ``create_app``.
    """
    flask_app = Flask("qhana_plugin_runner")
    register_helpers(flask_app)
    register_markdown_filter(flask_app)

    flask_app.config["MATHJAX_SCRIPT_LOCATION"] = "/static/mathjax/es5/tex-mml-chtml.js"
    flask_app.config["MATHJAX_SCRIPT_INTEGRITY_HASH"] = "sha384-test-integrity-hash"

    return flask_app


def make_open_url(metadata: dict, tarball: bytes):
    """Build an ``open_url`` replacement dispatching on the requested URL."""
    responses = {
        METADATA_URL: MockResponse(METADATA_URL, "application/json", json_data=metadata),
        metadata["dist"]["tarball"]: MockResponse(
            metadata["dist"]["tarball"], "application/gzip", content=tarball
        ),
    }

    def open_url(url, *args, **kwargs):
        try:
            return responses[url]
        except KeyError:
            raise AssertionError(f"unexpected url requested: {url}")

    return open_url


def build_tarball(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(name=f"package/{name}")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def integrity_for(data: bytes, algo: str = "sha512") -> str:
    """Compute an npm-style Subresource Integrity string for ``data``."""
    digest = hashlib.new(algo, data).digest()
    return f"{algo}-{base64.b64encode(digest).decode('utf-8')}"


# --- Path Traversal Standalone Tests ---


def test_verify_path_traversal_allows_valid_paths(tmp_path):
    """Verify that safe paths staying inside the base directory do not raise an error."""
    base_dir = tmp_path / "mathjax"
    target_file = base_dir / "es5" / "tex-mml-chtml.js"

    # Resolves fine because target path stays inside the base dir
    _verify_path_traversal(target_file, base_dir.resolve())


def test_verify_path_traversal_catches_attacks(tmp_path):
    """Verify that any path trying to escape the base directory throws a ValueError."""
    base_dir = tmp_path / "mathjax"

    # Classic path traversal breakout payload
    malicious_target = base_dir / "../../escaped.js"

    with pytest.raises(ValueError, match="escapes the base directory"):
        _verify_path_traversal(malicious_target, base_dir.resolve())


# --- Download and Extraction Tests ---


def test_download_mathjax_extracts_full_package(tmp_path, mathjax_app, monkeypatch):
    dummy_content = b"Dummy content of file. This test only checks whether the file is located correctly and contains this."
    tarball = build_tarball({"es5/tex-mml-chtml.js": dummy_content})
    metadata = {"dist": {"integrity": integrity_for(tarball), "tarball": TARBALL_URL}}

    monkeypatch.setattr(
        "qhana_plugin_runner.requests.open_url",
        make_open_url(metadata, tarball),
    )

    download_mathjax(mathjax_app)

    downloaded_file = tmp_path / "mathjax" / "es5" / "tex-mml-chtml.js"
    assert downloaded_file.exists()
    assert downloaded_file.read_bytes() == dummy_content

    # the version manifest is written so subsequent runs can be skipped
    version_file = tmp_path / "mathjax" / ".version.json"
    assert json.loads(version_file.read_text(encoding="utf-8")) == {
        "version": MATHJAX_VERSION
    }


def test_download_mathjax_rejects_tampered_tarball(tmp_path, mathjax_app, monkeypatch):
    """A tarball whose bytes do not match the advertised integrity hash must
    be rejected instead of extracted."""
    tarball = build_tarball({"es5/tex-mml-chtml.js": b"real content"})
    metadata = {
        "dist": {
            "integrity": integrity_for(b"a completely different payload"),
            "tarball": TARBALL_URL,
        }
    }

    monkeypatch.setattr(
        "qhana_plugin_runner.requests.open_url",
        make_open_url(metadata, tarball),
    )

    with pytest.raises(ValueError, match="Integrity check failed"):
        download_mathjax(mathjax_app)

    assert not (tmp_path / "mathjax" / "es5" / "tex-mml-chtml.js").exists()


def test_download_mathjax_blocks_path_traversal_integration(
    tmp_path, mathjax_app, monkeypatch
):
    """An integration test to ensure a malicious tarball with traversal targets

    triggers the path-traversal guard, aborts the installation, and writes absolutely
    no files outside the directory.
    """
    # Malicious tarball structure attempting to write outside the mathjax directory
    tarball = build_tarball({"../../escaped.js": b"pwned"})
    metadata = {"dist": {"integrity": integrity_for(tarball), "tarball": TARBALL_URL}}

    monkeypatch.setattr(
        "qhana_plugin_runner.requests.open_url",
        make_open_url(metadata, tarball),
    )

    with pytest.raises(ValueError, match="escapes the base directory"):
        download_mathjax(mathjax_app)

    # Ensure the file was blocked and NEVER written to the host filesystem anywhere
    assert not (tmp_path / "mathjax" / "escaped.js").exists()
    assert not (tmp_path / "escaped.js").exists()


def test_download_mathjax_skips_when_version_current(tmp_path, mathjax_app, monkeypatch):
    """If the local version manifest already matches, no network request is
    made and the existing assets are left untouched."""
    mathjax_dir = tmp_path / "mathjax"
    mathjax_dir.mkdir()
    (mathjax_dir / ".version.json").write_text(
        json.dumps({"version": MATHJAX_VERSION}), encoding="utf-8"
    )

    def fail_open_url(*args, **kwargs):
        raise AssertionError("no network access expected for an up-to-date version")

    monkeypatch.setattr("qhana_plugin_runner.requests.open_url", fail_open_url)

    download_mathjax(mathjax_app)  # must return early without raising


# --- Template Rendering Tests ---


def test_mathjax_loader_script_is_templated_correctly(template_app):
    """Verify that the template outputs the inline MathJax loader script
    with its browser-side safety check and configuration blocks.
    """
    with template_app.test_request_context():
        html = render_template(
            "simple_template.html",
            schema=SimpleNamespace(fields={}),
            values={},
            errors={},
            valid=False,
        )

    assert "document.body?.textContent?.match" in html
    assert (
        "window.MathJax = window.MathJax || { tex: { inlineMath: { '[+]': [['$', '$']] } } }"
        in html
    )
    assert 'script.src = "/static/mathjax/es5/tex-mml-chtml.js";' in html
    assert 'script.integrity = "sha384-test-integrity-hash";' in html

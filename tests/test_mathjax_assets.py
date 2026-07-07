import base64
import hashlib
import io
import tarfile
from types import SimpleNamespace

from qhana_plugin_runner import download_mathjax


class DummyResponse:
    def __init__(self, content: bytes, status_code: int = 200):
        self.content = content
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("request failed")

    def close(self):
        return None


def build_tarball(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(name=f"package/{name}")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def test_download_mathjax_extracts_full_package(tmp_path, monkeypatch):
    dummy_content = b"Dummy content of file. This test only checks whether the file is located correctly and contains this."
    tarball = build_tarball({"es5/tex-mml-chtml.js": dummy_content})

    sha384_hash = hashlib.sha384(tarball).digest()
    dummy_hash = f"sha384-{base64.b64encode(sha384_hash).decode('utf-8')}"

    monkeypatch.setattr(
        "qhana_plugin_runner.requests.open_url",
        lambda *args, **kwargs: DummyResponse(tarball),
    )

    app = SimpleNamespace(static_folder=str(tmp_path))
    download_mathjax(app, dummy_hash)

    downloaded_file = tmp_path / "mathjax" / "es5" / "tex-mml-chtml.js"
    assert downloaded_file.exists()
    assert downloaded_file.read_bytes() == dummy_content

import io
import json
import base64
import hashlib
import tarfile
import shutil
from pathlib import Path

from .. import requests


def _clear_old_mathjax_files(mathjax_dir: Path) -> None:
    """Safely remove known MathJax files/folders keeping whitelisted local assets."""
    if not mathjax_dir.exists():
        return

    whitelist = {".gitignore", "check-for-tex.js", ".version.json"}

    for path in mathjax_dir.iterdir():
        if path.name in whitelist:
            continue

        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def download_mathjax(app) -> None:
    """Download and extract MathJax frontend assets from the npm registry.

    Checks the local version manifest to prevent redundant downloads. If an update is
    required, it safely clears old assets (preserving whitelisted files), fetches the
    tarball metadata from npm, verifies the package integrity dynamically via SHA hashes,
    and extracts the assets using path traversal protection.
    """
    mathjax_version = app.config.get("MATHJAX_VERSION")
    mathjax_dir = Path(app.static_folder) / "mathjax"
    version_file = mathjax_dir / ".version.json"

    if version_file.exists():
        try:
            metadata = json.loads(version_file.read_text(encoding="utf-8"))
            if metadata.get("version") == mathjax_version:
                return
        except (json.JSONDecodeError, KeyError):
            pass

    with requests.open_url(
        app.config.get("MATHJAX_METADATA_URL"), raise_on_error_status=True
    ) as meta_response:
        pkg_data = meta_response.json()

    expected_integrity = pkg_data["dist"]["integrity"]
    tarball_url = pkg_data["dist"]["tarball"]

    _clear_old_mathjax_files(mathjax_dir)
    mathjax_dir.mkdir(parents=True, exist_ok=True)

    with requests.open_url(
        tarball_url, raise_on_error_status=True, stream=True
    ) as response:
        is_sha512 = "sha512-" in expected_integrity
        hasher = hashlib.sha512() if is_sha512 else hashlib.sha384()
        prefix = "sha512-" if is_sha512 else "sha384-"

        buffer = io.BytesIO()
        for chunk in response.iter_content(chunk_size=65536):
            if chunk:
                hasher.update(chunk)
                buffer.write(chunk)

        actual_hash = f"{prefix}{base64.b64encode(hasher.digest()).decode('utf-8')}"
        if actual_hash != expected_integrity:
            raise ValueError(
                f"Integrity check failed! Expected {expected_integrity}, got {actual_hash}"
            )

        buffer.seek(0)
        with tarfile.open(fileobj=buffer, mode="r:gz") as archive:
            package_prefix = "package/"
            for member in archive.getmembers():
                if not member.name.startswith(package_prefix):
                    continue
                relative_path = member.name[len(package_prefix) :]
                if not relative_path or relative_path.startswith("."):
                    continue

                target_path = mathjax_dir / relative_path
                target_path.absolute().relative_to(
                    mathjax_dir.absolute()
                )  # Path Traversal protection

                if member.isdir():
                    target_path.mkdir(parents=True, exist_ok=True)
                    continue
                if member.isfile():
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    target_path.write_bytes(archive.extractfile(member).read())

        version_file.write_text(
            json.dumps({"version": mathjax_version}), encoding="utf-8"
        )

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

    whitelist = {".gitignore", ".version.json"}

    for path in mathjax_dir.iterdir():
        if path.name in whitelist:
            continue

        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _verify_path_traversal(target_path: Path, abs_base_directory: Path) -> None:
    """Verifies that the target path remains inside the base directory.

    Raises a ValueError if a path traversal attempt is detected.
    """
    # Resolve to absolute path to eliminate '..' and symlinks
    abs_target = target_path.resolve(strict=False)

    if not abs_target.is_relative_to(abs_base_directory):
        raise ValueError(
            f"Path traversal detected! Target path '{abs_target}' "
            f"escapes the base directory '{abs_base_directory}'."
        )


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
        known_hashfunctions = {"sha512-": hashlib.sha512, "sha384-": hashlib.sha384}
        prefix, hasher = None, None
        for p, f in known_hashfunctions.items():
            if expected_integrity.startswith(p):  # use startswith, not in!
                prefix = p
                hasher = f()
        if hasher is None:
            raise ValueError("unknon hash function")

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
            abs_mathjax_dir = mathjax_dir.resolve()
            for member in archive.getmembers():
                if not member.name.startswith(package_prefix):
                    continue
                relative_path = member.name[len(package_prefix) :]
                if not relative_path:
                    continue

                target_path = mathjax_dir / relative_path
                _verify_path_traversal(target_path, abs_mathjax_dir)

                if target_path.name.startswith("."):
                    continue

                if member.isdir():
                    target_path.mkdir(parents=True, exist_ok=True)
                    continue
                if member.isfile():
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    target_path.write_bytes(archive.extractfile(member).read())

        version_file.write_text(
            json.dumps({"version": mathjax_version}), encoding="utf-8"
        )

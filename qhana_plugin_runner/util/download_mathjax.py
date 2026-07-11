import io
import json
import base64
import hashlib
import tarfile
import shutil
from pathlib import Path
import requests

# Nur noch die gewünschte Version wird definiert. Keine manuellen Hashes!
MATHJAX_VERSION = "3.2.2"


def _clear_old_mathjax_files(mathjax_dir: Path) -> None:
    """Safely remove known MathJax files/folders."""
    if not mathjax_dir.exists():
        return

    whitelist = {".gitignore", "check-for-tex.js"}

    for path in mathjax_dir.iterdir():
        if path.name in whitelist:
            continue

        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def download_mathjax(app) -> None:
    mathjax_dir = Path(app.static_folder) / "mathjax"
    version_file = mathjax_dir / ".version.json"

    if version_file.exists():
        try:
            metadata = json.loads(version_file.read_text(encoding="utf-8"))
            if metadata.get("version") == MATHJAX_VERSION:
                return  # Bereits aktuell!
        except (json.JSONDecodeError, KeyError):
            pass

    metadata_url = f"https://registry.npmjs.org/mathjax/{MATHJAX_VERSION}"
    meta_response = requests.get(metadata_url)
    meta_response.raise_for_status()
    pkg_data = meta_response.json()

    expected_integrity = pkg_data["dist"]["integrity"]
    tarball_url = pkg_data["dist"]["tarball"]

    _clear_old_mathjax_files(mathjax_dir)
    mathjax_dir.mkdir(parents=True, exist_ok=True)

    with requests.get(tarball_url, stream=True) as response:
        response.raise_for_status()

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
                )  # Path Traversal Schutz

                if member.isdir():
                    target_path.mkdir(parents=True, exist_ok=True)
                    continue
                if member.isfile():
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    target_path.write_bytes(archive.extractfile(member).read())

        version_file.write_text(
            json.dumps({"version": MATHJAX_VERSION}), encoding="utf-8"
        )

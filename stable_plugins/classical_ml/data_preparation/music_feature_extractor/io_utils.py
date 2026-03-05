# Copyright 2026 QHAna plugin runner contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Input loading helpers for music feature extraction."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse
from zipfile import BadZipFile, ZipFile

from qhana_plugin_runner.requests import get_mimetype, open_url

SOURCE_MODE_OPTIONS = ("auto", "single", "zip")
DECLARED_FORMAT_OPTIONS = (
    "auto",
    "musicxml",
    "musicxml-xml",
    "xml",
    "musicxml-mxl",
    "mxl",
    "midi",
)

_XML_FORMATS = {"musicxml", "musicxml-xml", "xml"}

_EXTENSION_TO_FORMAT = {
    ".musicxml": "musicxml",
    ".xml": "xml",
    ".mxl": "mxl",
    ".mid": "midi",
    ".midi": "midi",
}

_FORMAT_TO_EXTENSIONS = {
    "musicxml": {".musicxml", ".xml"},
    "musicxml-xml": {".musicxml", ".xml"},
    "xml": {".xml"},
    "musicxml-mxl": {".mxl"},
    "mxl": {".mxl"},
    "midi": {".mid", ".midi"},
}

_MUSICXML_MIME_HINTS = {
    "application/xml",
    "text/xml",
    "application/vnd.recordare.musicxml+xml",
    "application/vnd.recordare.musicxml",
}

_MIDI_MIME_HINTS = {"audio/midi", "audio/x-midi", "audio/mid"}


@dataclass(frozen=True)
class LoadedMusicSource:
    source_name: str
    format: str
    content: str


def normalize_source_mode(mode: str | None) -> str:
    result = (mode or "auto").strip().lower()
    if result not in SOURCE_MODE_OPTIONS:
        raise ValueError(f"Invalid source mode '{mode}'.")
    return result


def normalize_declared_format(fmt: str | None) -> str:
    result = (fmt or "auto").strip().lower()
    if result not in DECLARED_FORMAT_OPTIONS:
        raise ValueError(f"Invalid declared format '{fmt}'.")
    return result


def resolve_source_mode(
    source_url: str,
    requested_mode: str,
    content_type: str | None = None,
    filename: str | None = None,
    payload: bytes | None = None,
) -> str:
    normalized = normalize_source_mode(requested_mode)
    if normalized != "auto":
        return normalized

    suffix = _suffix(filename)
    if suffix == ".zip":
        return "zip"

    content_type = (content_type or "").split(";")[0].strip().lower()
    if content_type == "application/zip":
        return "single" if suffix == ".mxl" else "zip"

    source_suffix = _suffix(source_url)
    if source_suffix == ".zip":
        return "zip"

    if payload and payload.startswith(b"PK\x03\x04"):
        return "single" if suffix == ".mxl" else "zip"

    return "single"


def load_music_sources(
    source_url: str,
    source_mode: str,
    declared_format: str,
    max_files: int,
) -> tuple[list[LoadedMusicSource], dict[str, object]]:
    mode = normalize_source_mode(source_mode)
    fmt = normalize_declared_format(declared_format)
    if max_files < 1:
        raise ValueError("max_files must be at least 1.")

    with open_url(source_url, stream=True) as response:
        payload = response.content
        content_type = get_mimetype(response)
        filename = _guess_filename(response.url)

    resolved_mode = resolve_source_mode(
        source_url=source_url,
        requested_mode=mode,
        content_type=content_type,
        filename=filename,
        payload=payload,
    )

    if resolved_mode == "single":
        source_format = _resolve_single_format(fmt, filename, content_type, payload)
        if source_format is None:
            raise ValueError(
                "Could not infer source format. Set 'declared format' explicitly."
            )
        source_name = filename or "music-source"
        content = _encode_content(payload, source_format)
        sources = [
            LoadedMusicSource(
                source_name=source_name,
                format=source_format,
                content=content,
            )
        ]
        summary = {
            "resolved_mode": resolved_mode,
            "source_name": source_name,
            "source_count": 1,
            "candidate_count": 1,
            "preview_names": [source_name],
            "truncated": False,
            "content_type": content_type,
        }
        return sources, summary

    sources, summary = _load_sources_from_zip(
        payload=payload,
        declared_format=fmt,
        max_files=max_files,
    )
    summary["resolved_mode"] = resolved_mode
    summary["source_name"] = filename
    summary["content_type"] = content_type
    return sources, summary


def _resolve_single_format(
    declared_format: str,
    filename: str | None,
    content_type: str | None,
    payload: bytes,
) -> str | None:
    if declared_format != "auto":
        return declared_format

    ext = _suffix(filename)
    if ext in _EXTENSION_TO_FORMAT:
        return _EXTENSION_TO_FORMAT[ext]

    mime = (content_type or "").split(";")[0].strip().lower()
    if mime in _MUSICXML_MIME_HINTS:
        return "musicxml"
    if mime in _MIDI_MIME_HINTS:
        return "midi"

    if payload.startswith(b"MThd"):
        return "midi"
    if payload.startswith(b"PK\x03\x04"):
        return "mxl"

    stripped = payload.lstrip()
    if stripped.startswith(b"<"):
        return "musicxml"
    return None


def _load_sources_from_zip(
    payload: bytes,
    declared_format: str,
    max_files: int,
) -> tuple[list[LoadedMusicSource], dict[str, object]]:
    try:
        archive = ZipFile(BytesIO(payload))
    except BadZipFile as exc:
        raise ValueError("Input is not a valid ZIP file.") from exc

    sources: list[LoadedMusicSource] = []
    candidate_count = 0
    skipped_count = 0
    preview_names: list[str] = []
    truncated = False

    with archive:
        for name in archive.namelist():
            if name.endswith("/"):
                continue

            fmt = _resolve_zip_entry_format(name, declared_format)
            if fmt is None:
                skipped_count += 1
                continue

            candidate_count += 1
            if len(preview_names) < 10:
                preview_names.append(name)

            if len(sources) >= max_files:
                truncated = True
                continue

            raw = archive.read(name)
            content = _encode_content(raw, fmt)
            sources.append(
                LoadedMusicSource(
                    source_name=name,
                    format=fmt,
                    content=content,
                )
            )

    if not sources:
        raise ValueError("No supported music files found in ZIP source.")

    summary = {
        "source_count": len(sources),
        "candidate_count": candidate_count,
        "skipped_count": skipped_count,
        "preview_names": preview_names,
        "truncated": truncated,
    }
    return sources, summary


def _resolve_zip_entry_format(name: str, declared_format: str) -> str | None:
    ext = _suffix(name)
    if declared_format == "auto":
        return _EXTENSION_TO_FORMAT.get(ext)

    valid_extensions = _FORMAT_TO_EXTENSIONS.get(declared_format, set())
    if ext not in valid_extensions:
        return None
    return declared_format


def _encode_content(data: bytes, fmt: str) -> str:
    if fmt in _XML_FORMATS:
        return data.decode("utf-8", errors="replace")
    return base64.b64encode(data).decode("ascii")


def _guess_filename(url: str) -> str:
    path_name = Path(urlparse(url).path).name
    return path_name or "music-source"


def _suffix(path_or_name: str | None) -> str:
    if not path_or_name:
        return ""
    return Path(path_or_name).suffix.lower()

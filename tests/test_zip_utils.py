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

import io
import zipfile

from qhana_plugin_runner.plugin_utils.zip_utils import (
    get_file_responses_from_zip,
    get_files_from_zip_url,
)

from .utils import MockResponse


def _make_zip(members: dict) -> bytes:
    """Build the raw bytes of a zip archive from a ``{name: text}`` mapping."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, text in members.items():
            archive.writestr(name, text)
    return buffer.getvalue()


class TestGetFileResponsesFromZip:
    def test_yields_one_response_per_file(self):
        zip_bytes = _make_zip({"a.json": "[]", "b.csv": "ID\ne1"})

        responses = list(get_file_responses_from_zip(zip_bytes))

        assert len(responses) == 2

    def test_response_content_matches_member(self):
        zip_bytes = _make_zip({"a.json": '{"ID": "e1"}'})

        (response,) = list(get_file_responses_from_zip(zip_bytes))

        assert response.text == '{"ID": "e1"}'
        assert response.json() == {"ID": "e1"}

    def test_response_url_is_member_name(self):
        zip_bytes = _make_zip({"vectors.csv": "ID\ne1"})

        (response,) = list(get_file_responses_from_zip(zip_bytes))

        assert response.url == "vectors.csv"

    def test_content_type_guessed_from_extension(self):
        zip_bytes = _make_zip({"a.json": "[]", "b.csv": "ID\ne1"})

        by_name = {r.url: r for r in get_file_responses_from_zip(zip_bytes)}

        assert by_name["a.json"].headers["Content-Type"] == "application/json"
        assert by_name["b.csv"].headers["Content-Type"] == "text/csv"

    def test_no_content_type_for_unknown_extension(self):
        zip_bytes = _make_zip({"member.abc": "ID\ne1", "member": "ID\ne2"})

        responses = list(get_file_responses_from_zip(zip_bytes))

        assert {r.headers["Content-Type"] for r in responses} == {None}

    def test_content_type_default_applies_to_unknown_extensions(self):
        zip_bytes = _make_zip({"a.json": "[]", "member": "ID\ne1"})

        responses = list(
            get_file_responses_from_zip(zip_bytes, default_content_type="text/csv")
        )

        assert {r.headers["Content-Type"] for r in responses} == {
            "application/json",  # JSON detected
            "text/csv",
        }

    def test_members_are_yielded_in_sorted_order(self):
        zip_bytes = _make_zip({"c.csv": "3", "a.csv": "1", "b.csv": "2"})

        names = [r.url for r in get_file_responses_from_zip(zip_bytes)]

        assert names == ["a.csv", "b.csv", "c.csv"]

    def test_directory_entries_are_skipped(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("folder/", "")
            archive.writestr("folder/a.csv", "ID\ne1")
        zip_bytes = buffer.getvalue()

        names = [r.url for r in get_file_responses_from_zip(zip_bytes)]

        assert names == ["folder/a.csv"]

    def test_empty_zip_yields_no_responses(self):
        zip_bytes = _make_zip({})

        assert list(get_file_responses_from_zip(zip_bytes)) == []

    def test_status_code_is_ok(self):
        zip_bytes = _make_zip({"a.csv": "ID\ne1"})

        (response,) = list(get_file_responses_from_zip(zip_bytes))

        assert response.status_code == 200


class TestGetFilesFromZipUrl:
    def test_text_mode_yields_readable_text_files(self, monkeypatch):
        url = "http://example.com/data.zip"
        response = MockResponse.from_zip(url, {"a.txt": "hello"})
        monkeypatch.setattr(
            "qhana_plugin_runner.plugin_utils.zip_utils.open_url",
            lambda *a, **k: response,
        )

        results = [
            (file_name, file_like.read())
            for file_like, file_name in get_files_from_zip_url(url)
        ]

        assert results == [("a.txt", "hello")]

    def test_binary_mode_yields_readable_byte_files(self, monkeypatch):
        url = "http://example.com/data.zip"
        response = MockResponse.from_zip(url, {"a.bin": "hello"})
        monkeypatch.setattr(
            "qhana_plugin_runner.plugin_utils.zip_utils.open_url",
            lambda *a, **k: response,
        )

        results = [
            (file_name, file_like.read())
            for file_like, file_name in get_files_from_zip_url(url, mode="b")
        ]

        assert results == [("a.bin", b"hello")]

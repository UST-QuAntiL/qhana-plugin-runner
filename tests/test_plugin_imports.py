# Copyright 2024 QHAna plugin runner contributors.
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

"""Tests for bad import statements in plugins."""

import re
from pathlib import Path

PLUGIN_FOLDERS = {
    "stable_plugins",
    "plugins",
}

IMPORT_PATH_REGEX = re.compile(r"^\s*(?:from|import)\s+(?P<path>[a-zA-Z_]\S+)(?:\s+|$)")


def test_plugin_imports():
    base_path = Path(".").resolve()

    bad_imports = []

    bad_import_paths = tuple(f"{folder}." for folder in PLUGIN_FOLDERS)

    for folder in PLUGIN_FOLDERS:
        for file_ in base_path.rglob(f"{folder}/**/*.py"):
            if not file_.exists() or file_.is_dir():
                continue
            relative_path = str(file_.relative_to(base_path))
            content = file_.read_text()
            lines = enumerate(content.splitlines(True))
            absolute_imports = (
                (i, match, line)
                for i, line in lines
                if (match := IMPORT_PATH_REGEX.match(line))
            )
            for line_nr, match, line in absolute_imports:
                import_path = match.group("path")
                if import_path in PLUGIN_FOLDERS or import_path.startswith(
                    bad_import_paths
                ):
                    bad_imports.append(
                        (relative_path, line_nr, import_path, line.rstrip("\n"))
                    )

    has_bad_imports = bool(bad_imports)

    bad_imports_warning = "The following plugin files contain bad absolute imports!\n\n"

    bad_imports_warning += "\n".join(
        f'    {file_} L{line}: "{import_statement}"'
        for file_, line, import_path, import_statement in bad_imports
    )

    assert not has_bad_imports, bad_imports_warning

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

"""Shared exception types for the state-analysis plugins.

Plugins should raise these instead of the generic ``ValueError`` /
``RuntimeError`` so that the API layer can map them to appropriate HTTP
responses and so error logs are searchable by category.
"""


class PluginError(Exception):
    """Base class for state-analysis plugin errors."""


class PluginInputError(PluginError):
    """Raised when caller-provided input is malformed or invariant-violating.

    Maps conceptually to HTTP 4xx — the request is wrong, not the server.
    """


class PluginExecutionError(PluginError):
    """Raised when a downstream call or computation fails at runtime.

    Maps conceptually to HTTP 5xx / task failure — the inputs were accepted
    but execution could not complete.
    """

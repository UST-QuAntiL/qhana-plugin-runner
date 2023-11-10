# Copyright 2023 QHAna plugin runner contributors.
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


"""Utilities to safely render request body and header templates."""

from typing import Any, Dict

from jinja2.sandbox import ImmutableSandboxedEnvironment


class TemplateRenderingError(ValueError):
    pass


SANDBOX = ImmutableSandboxedEnvironment()


def render_template_sandboxed(template: str, context: Dict[str, Any]) -> str:
    try:
        return SANDBOX.from_string(template).render(context)
    except Exception as err:
        # merge all exceptions into one single exception
        raise TemplateRenderingError from err

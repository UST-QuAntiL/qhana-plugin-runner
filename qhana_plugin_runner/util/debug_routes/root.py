# Copyright 2021 QHAna plugin runner contributors.
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

# originally from <https://github.com/buehlefs/flask-template/>

"""Module for the root endpoint of the debug routes.
Contains the blueprint to avoid circular dependencies."""

from json import dumps

from flask import Blueprint, current_app, render_template, url_for
from werkzeug.routing import converters

DEBUG_BLP = Blueprint(
    "debug-routes",
    __name__,
    template_folder="templates",
    static_folder="static",
    url_prefix="/debug",
)


@DEBUG_BLP.route("/")
@DEBUG_BLP.route("/index")
def index():
    return render_template("debug/index.html", title="QHAna Plugin Runner – Debug")


MESSAGE_TEMPLATES = {
    "prevent submit": "ui-prevent-submit",
    "allow submit": "ui-allow-submit",
    "get submit status": "ui-submit-status",
    "load css": {
        "type": "load-css",
        "urls": [],
    },
    "data url response": {
        "type": "data-url-response",
        "inputKey": "",
        "href": "",
        "dataType": "entity/list",
        "contentType": "text/csv",
        "filename": "test.csv",
        "version": "v1",
    },
    "plugin url response": {
        "type": "plugin-url-response",
        "inputKey": "",
        "pluginUrl": "",
        "pluginName": "Hello World",
        "pluginVersion": "v1.0.0",
    },
    "implementations response": {
        "type": "implementations-response",
        "implementations": [
            {"name": "test.qasm", "download": "http://", "version": 1, "type": "qasm??"}
        ],
    },
    "autofill": {
        "type": "autofill-response",
        "value": "",
        "encoding": "application/x-www-form-urlencoded",
    },
    "plugin context": {
        "type": "plugin-context",
        "experimentId": "",
        "location": "navigation|workspace|experiment-navigation|timeline-step|data-preview",
    },
}


@DEBUG_BLP.route("/frontend-debugger")
def frontend_debugger():
    """Debugger for Micro Frontends."""

    micro_frontend_urls = []  # TODO

    for rule in current_app.url_map.iter_rules():
        if (
            not rule.rule.startswith("/plugins/")
            or rule.rule.count("/") < 3
            or not rule.rule.split("/", maxsplit=3)[3]
            or "GET" not in rule.methods
        ):
            continue

        rule_converters = rule._converters

        def replaced(argument: str):
            argument_converter = rule_converters[argument]
            if isinstance(argument_converter, converters.NumberConverter):
                return 1
            replacement = f"{{{{{argument}}}}}"

            # check converter
            argument_converter.to_url(replacement)
            return replacement

        try:
            arguments = {a: replaced(a) for a in rule.arguments}
        except Exception:
            continue  # cannot build this URL, skip
        url = url_for(rule.endpoint, **arguments, _external=True)
        micro_frontend_urls.append(url)

    if not MESSAGE_TEMPLATES["load css"]["urls"]:
        MESSAGE_TEMPLATES["load css"]["urls"].append(
            url_for(
                "debug-routes.static",
                filename="microfrontend-debug.css",
                _external=True,
            )
        )

    message_templates = {k: dumps(v, indent=4) for k, v in MESSAGE_TEMPLATES.items()}
    return render_template(
        "debug/microfrontend-debugger.html",
        title="QHAna Plugin Runner – Micro Frontend Debugger",
        micro_frontend_urls=micro_frontend_urls,
        message_templates=message_templates,
    )

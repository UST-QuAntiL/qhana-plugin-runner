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

from flask import Blueprint, Flask, render_template

LICENSE_BLP = Blueprint("licenses", __name__, url_prefix="/licenses")


@LICENSE_BLP.route("/")
def show_licenses():
    return render_template("included_licenses.html")


def register_licenses(app: Flask):
    app.register_blueprint(LICENSE_BLP)

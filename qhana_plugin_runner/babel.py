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

"""Module for setting up Babel support for flask app."""

from typing import Optional
from flask import Flask, request, g
from flask_babel import Babel, refresh as flask_babel_refresh
from werkzeug.datastructures import LanguageAccept

"""The list of locales to support."""
SUPPORTED_LOCALES = ["de", "en"]


def get_locale():
    if "lang" in g:
        # check LanguageAccept option in g context
        accepted_languages = g.get("lang")
        if accepted_languages and isinstance(accepted_languages, LanguageAccept):
            # g context takes precedent over request accept_languages
            return accepted_languages.best_match(SUPPORTED_LOCALES)
    # try to guess the language from the user accept
    # header the browser transmits. We support SUPPORTED_LOCALES The best match wins.
    return request.accept_languages.best_match(SUPPORTED_LOCALES)


BABEL = Babel(locale_selector=get_locale)


def inject_lang_from_header():
    """Inject the language defined in the custom 'lang' Hader into the g context.

    This method can be used in a before request callback to read the custom
    header and use that to set the language for example for API requests.
    """
    lang: Optional[str] = request.headers.get("lang")
    if lang:
        old_accepts: LanguageAccept = request.accept_languages
        if "lang" in g:
            # check LanguageAccept option in g context
            accepted_languages = g.get("lang")
            if accepted_languages and isinstance(accepted_languages, LanguageAccept):
                old_accepts = accepted_languages
        # inject language from custom header as first choice into request
        values = (lang, 10), *old_accepts
        g.lang = LanguageAccept(values)
        # Force refresh to make sure that the change is applied
        flask_babel_refresh()


def register_babel(app: Flask):
    """Register babel to enable translations for this app."""
    BABEL.init_app(app)

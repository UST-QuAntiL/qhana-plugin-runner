# Copyright 2022 QHAna plugin runner contributors.
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


"""This module provides a helper class :py:class:`MutableJSON` that can be used as a column in SQL Alchemy. SQL Alchemy's built-in JSON type is immutable and hence inflexible for usage as a key-value storage. :py:class:`MutableJSON` supports all primitive types in JSON (``str``, ``float``, ``int`` and ``bool``) as well nested dicts and lists with change tracking."""

from sqlalchemy import event, inspect
from sqlalchemy.ext.mutable import Mutable
from sqlalchemy.types import JSON
from sqlalchemy_json import NestedMutableDict, NestedMutableList


class NestedMutable(Mutable):
    """SQLAlchemy `mutable` extension with nested change tracking."""

    @classmethod
    def _listen_on_attribute(cls, attribute, coerce, parent_cls):
        """Overwrite method of base class in order to accept primitive (immutable) types.

        Code is identical to the method MutableBase._listen_on_attribute from SQLAlchemy apart from handling
        ``str``, ``float`` and ``int`` (which includes ``bool`` as ``bool`` is a subclass of ``int``).

        SQLAlchemy is licensed under the MIT license:

        Copyright 2005-2022 SQLAlchemy authors and contributors <see AUTHORS file>.

        Permission is hereby granted, free of charge, to any person obtaining a copy of
        this software and associated documentation files (the "Software"), to deal in
        the Software without restriction, including without limitation the rights to
        use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
        of the Software, and to permit persons to whom the Software is furnished to do
        so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE."""
        key = attribute.key
        if parent_cls is not attribute.class_:
            return

        # rely on "propagate" here
        parent_cls = attribute.class_

        listen_keys = cls._get_listen_keys(attribute)

        def load(state, *args):
            """Listen for objects loaded or refreshed.
            Wrap the target data member's value with
            ``Mutable``.
            """
            val = state.dict.get(key, None)
            if val is not None and not isinstance(val, (str, float, int)):
                if coerce:
                    val = cls.coerce(key, val)
                    state.dict[key] = val
                val._parents[state] = key

        def load_attrs(state, ctx, attrs):
            if not attrs or listen_keys.intersection(attrs):
                load(state)

        def set_(target, value, oldvalue, initiator):
            """Listen for set/replace events on the target
            data member.
            Establish a weak reference to the parent object
            on the incoming value, remove it for the one
            outgoing.
            """
            if value is oldvalue:
                return value
            if isinstance(value, (str, float, int)):
                return value

            if not isinstance(value, cls):
                value = cls.coerce(key, value)
            if value is not None:
                value._parents[target] = key
            if isinstance(oldvalue, cls):
                oldvalue._parents.pop(inspect(target), None)
            return value

        def pickle(state, state_dict):
            val = state.dict.get(key, None)
            if val is not None:
                if "ext.mutable.values" not in state_dict:
                    state_dict["ext.mutable.values"] = []
                state_dict["ext.mutable.values"].append(val)

        def unpickle(state, state_dict):
            if "ext.mutable.values" in state_dict:
                for val in state_dict["ext.mutable.values"]:
                    val._parents[state] = key

        event.listen(parent_cls, "load", load, raw=True, propagate=True)
        event.listen(parent_cls, "refresh", load_attrs, raw=True, propagate=True)
        event.listen(parent_cls, "refresh_flush", load_attrs, raw=True, propagate=True)
        event.listen(attribute, "set", set_, raw=True, retval=True, propagate=True)
        event.listen(parent_cls, "pickle", pickle, raw=True, propagate=True)
        event.listen(parent_cls, "unpickle", unpickle, raw=True, propagate=True)

    @classmethod
    def coerce(cls, key, value):
        """Convert plain JSON structure to NestedMutable."""
        if value is None:
            return value
        if isinstance(value, (str, int, float)):
            return value
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return NestedMutableDict.coerce(key, value)
        if isinstance(value, list):
            return NestedMutableList.coerce(key, value)
        return super().coerce(key, value)


MutableJSON = NestedMutable.as_mutable(JSON)

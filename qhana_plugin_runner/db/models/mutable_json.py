# Copyright (c) 2014, Elmer de Looff <elmer.delooff@gmail.com>
# All rights reserved.

# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""This module contains the tracked object classes.

TrackedObject forms the basis for both the TrackedDict and the TrackedList.

A function for automatic conversion of dicts and lists to their tracked
counterparts is also included.
"""

from itertools import chain

from six import iteritems
from sqlalchemy.ext.mutable import Mutable
from sqlalchemy import event, inspect
from sqlalchemy.types import JSON

__all__ = "MutableJson"


class TrackedObject(object):
    """A base class for delegated change-tracking."""

    _type_mapping = {}

    def __new__(cls, *args, **kwds):
        tracked = super().__new__(cls, *args, **kwds)
        tracked.parent = None
        return tracked

    def changed(self):
        """Marks the object as changed.

        If a `parent` attribute is set, the `changed()` method on the parent
        will be called, propagating the change notification up the chain.

        The message (if provided) will be debug logged.
        """
        if self.parent is not None:
            self.parent.changed()
        elif isinstance(self, Mutable):
            super(TrackedObject, self).changed()

    @classmethod
    def register(cls, origin_type):
        """Decorator for mutation tracker registration.

        The provided `origin_type` is mapped to the decorated class such that
        future calls to `convert()` will convert the object of `origin_type`
        to an instance of the decorated class.
        """

        def decorator(tracked_type):
            """Adds the decorated class to the `_type_mapping` dictionary."""
            cls._type_mapping[origin_type] = tracked_type
            return tracked_type

        return decorator

    @classmethod
    def convert(cls, obj, parent):
        """Converts objects to registered tracked types

        This checks the type of the given object against the registered tracked
        types. When a match is found, the given object will be converted to the
        tracked type, its parent set to the provided parent, and returned.

        If its type does not occur in the registered types mapping, the object
        is returned unchanged.
        """
        replacement_type = cls._type_mapping.get(type(obj))
        if replacement_type is not None:
            new = replacement_type(obj)
            new.parent = parent
            return new
        return obj

    def convert_iterable(self, iterable):
        """Generator to `convert` every member of the given iterable."""
        return (self.convert(item, self) for item in iterable)

    def convert_items(self, items):
        """Generator like `convert_iterable`, but for 2-tuple iterators."""
        return ((key, self.convert(value, self)) for key, value in items)

    def convert_mapping(self, mapping):
        """Convenience method to track either a dict or a 2-tuple iterator."""
        if isinstance(mapping, dict):
            return self.convert_items(iteritems(mapping))
        return self.convert_items(mapping)


@TrackedObject.register(dict)
class TrackedDict(TrackedObject, dict):
    """A TrackedObject implementation of the basic dictionary."""

    def __init__(self, source=(), **kwds):
        super(TrackedDict, self).__init__(
            chain(self.convert_mapping(source), self.convert_mapping(kwds))
        )

    def __ior__(self, other):
        self.changed()
        return super(TrackedDict, self).__ior__(self.convert(other, self))

    def __setitem__(self, key, value):
        self.changed()
        super(TrackedDict, self).__setitem__(key, self.convert(value, self))

    def __delitem__(self, key):
        self.changed()
        super(TrackedDict, self).__delitem__(key)

    def clear(self):
        self.changed()
        super(TrackedDict, self).clear()

    def pop(self, *key_and_default):
        self.changed()
        return super(TrackedDict, self).pop(*key_and_default)

    def popitem(self):
        self.changed()
        return super(TrackedDict, self).popitem()

    def update(self, source=(), **kwds):
        self.changed()
        super(TrackedDict, self).update(
            chain(self.convert_mapping(source), self.convert_mapping(kwds))
        )

    def setdefault(self, key, default=None):
        if key in self:
            return self[key]

        # this calls __setitem__, which converts the value and calls changed()
        self[key] = default
        # the value at self[key] may be a new TrackedObject, so return
        # self[key] instead of default
        return self[key]


@TrackedObject.register(list)
class TrackedList(TrackedObject, list):
    """A TrackedObject implementation of the basic list."""

    def __init__(self, iterable=()):
        super(TrackedList, self).__init__(self.convert_iterable(iterable))

    def __setitem__(self, key, value):
        self.changed()
        super(TrackedList, self).__setitem__(key, self.convert(value, self))

    def __delitem__(self, key):
        self.changed()
        super(TrackedList, self).__delitem__(key)

    def append(self, item):
        self.changed()
        super(TrackedList, self).append(self.convert(item, self))

    def extend(self, iterable):
        self.changed()
        super(TrackedList, self).extend(self.convert_iterable(iterable))

    def remove(self, value):
        self.changed()
        return super(TrackedList, self).remove(value)

    def pop(self, index):
        self.changed()
        return super(TrackedList, self).pop(index)

    def sort(self, *, key=None, reverse=False):
        self.changed()
        super(TrackedList, self).sort(key=key, reverse=reverse)


class NestedMutableDict(TrackedDict, Mutable):
    @classmethod
    def coerce(cls, key, value):
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(value)
        return super(cls).coerce(key, value)


class NestedMutableList(TrackedList, Mutable):
    @classmethod
    def coerce(cls, key, value):
        if isinstance(value, cls):
            return value
        if isinstance(value, list):
            return cls(value)
        return super(cls).coerce(key, value)


# NOTE: the following deviates from the original sourcecode to support primitive types (int, float, str, bool, None) without nesting.
class NestedMutable(Mutable):
    """SQLAlchemy `mutable` extension with nested change tracking."""

    @classmethod
    def _listen_on_attribute(cls, attribute, coerce, parent_cls):
        """Overwrite method of base class in order to accept primitive (immutable) types. Code is identical apart from handling ``str``, ``float`` and ``int`` (which includes ``bool`` as ``bool`` is a subclass of ``int``)."""
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
        return super(cls).coerce(key, value)


MutableJSON = NestedMutable.as_mutable(JSON)

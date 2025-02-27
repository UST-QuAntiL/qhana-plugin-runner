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

"""Module containing extra marshmallow field classes to be used in plugin micro frontends."""

from collections import OrderedDict
from enum import Enum
from typing import Any, Iterable, List, Mapping, Optional, Set, Type
from warnings import warn

from marshmallow.exceptions import ValidationError
from marshmallow.fields import Field
from marshmallow.utils import resolve_field_instance
from marshmallow.validate import OneOf


class OneOfEnum(OneOf):
    """Validator for validating enum based choices.

    Succeeds if a ``value`` is a member of the enumeration.

    If choices contains the empty string ``""`` then ``None`` will also validate.

    Args:
        enum (Type[Enum]): the enum type the choices are based on
        choices (Iterable[str]): the names of the enum items that are valid choices (can be a subset of the whole enum; can include ``""``)
        labels (Optional[Iterable[str]]): the labels for the individual choices
        error (Optional[str]): Error message to raise in case of a validation error. Can be interpolated with ``{input}``, ``{choices}`` and ``{labels}``.
        use_value (Optional[bool]): If true, the enum value is used for choices instead of the enum name. Use labels to provide meaningful labels in this case.

    Raises:
        ValueError: if the choices cannot be mapped to the given enum
    """

    def __init__(
        self,
        enum: Type[Enum],
        choices: Iterable[str],
        labels: Optional[Iterable[str]],
        *,
        error: Optional[str],
        use_value: bool = False,
    ):
        for choice in choices:
            if choice != "":
                try:
                    if use_value:
                        enum(choice)
                    else:
                        enum[choice]
                except KeyError as err:
                    raise ValueError(
                        f"Choice '{choice}' is not a valid enum value of Enum {enum}!"
                    ) from err
        super().__init__(choices=choices, labels=labels, error=error)
        self.enum_choices: Set[Optional[Enum]]
        if use_value:
            self.enum_choices = {c for c in enum if c.value in choices}
        else:
            self.enum_choices = {c for c in enum if c.name in choices}
        if "" in self.choices:
            self.enum_choices.add(None)

    def __call__(self, value: Optional[Enum]) -> Optional[Enum]:
        if value not in self.enum_choices:
            raise ValidationError(self._format_error(value))

        return value


# field is registered in qhana_plugin_runner.api module
class EnumField(Field):
    #: Default error messages.
    default_error_messages = {"invalid": "Not a valid choice."}

    def __init__(self, enum_type: Type[Enum], use_value: bool = False, **kwargs):
        metadata = kwargs.pop("metadata", {})
        enum_meta = OrderedDict({e.name: e.value for e in enum_type})
        options = metadata.get("options", {})
        for name, value in options.items():
            if isinstance(name, enum_type):
                name = name.name
            if name in enum_meta or name == "":
                # prefer manual override for values; allow to specify None ("") value and name
                enum_meta[name] = value
            else:
                # unknown enum member
                warn(
                    f"The enum field got an option {name} that is not present on the enum {enum_type}!"
                )
        if "" in enum_meta:
            # None option should always be first
            enum_meta.move_to_end("", last=False)
        metadata["options"] = enum_meta

        super().__init__(metadata=metadata, **kwargs)

        self.enum_type: Type[Enum] = enum_type
        self.use_value = use_value

        if use_value:
            # switch the roles of name and value of the enum
            labels, choices = zip(*enum_meta.items())
        else:
            choices, labels = zip(*enum_meta.items())
        validator = OneOfEnum(
            enum=enum_type,
            choices=choices,
            labels=labels,
            error=self.error_messages["invalid"],
            use_value=use_value,
        )
        self.validators.insert(0, validator)

    def _serialize(self, value: Enum, attr: str, obj, **kwargs):
        if value is None:
            return None

        return value.name

    def _deserialize(
        self, value: str, attr: Optional[str], data: Optional[Mapping[str, Any]], **kwargs
    ):
        if value == "":
            return None
        try:
            if self.use_value:
                return self.enum_type(value)
            return self.enum_type[value]
        except KeyError as error:
            raise self.make_error("invalid", input=value) from error


class CSVList(Field):
    """Validator for validating comma separated lists.

    Args:
        element_type (Type[Field]): the field type of list elements

    Raises:
        ValueError: if fields are of an invalid type
    """

    def __init__(self, element_type: Field, **kwargs):
        super().__init__(**kwargs)
        self.element_type = resolve_field_instance(element_type)

    def _serialize(self, value: List[Any], attr: str, obj: Any, **kwargs) -> str:
        return ",".join(
            [self.element_type._serialize(v, attr, obj, **kwargs) for v in value]
        )

    def _deserialize(
        self, value: str, attr: Optional[str], data: Optional[Mapping[str, Any]], **kwargs
    ):
        if not value:
            return []
        return [self.element_type.deserialize(v) for v in value.split(",")]

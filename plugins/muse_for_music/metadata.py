from collections import Counter
from collections.abc import Collection, Sequence
from inspect import get_annotations
from typing import _AnnotatedAlias, _GenericAlias, _UnionGenericAlias, List  # type: ignore

from .util import OpusEntity, PartEntity, PersonEntity

from qhana_plugin_runner.plugin_utils.attributes import AttributeMetadata


def get_attribute_metadata(entity_class) -> List[AttributeMetadata]:
    metadata = []
    annotations = get_annotations(entity_class)
    for name in entity_class._fields:
        annotation = annotations[name]
        extra = None
        optional = False
        if isinstance(annotation, _AnnotatedAlias):
            extra = annotation.__metadata__
            annotation = annotation.__origin__
        if isinstance(annotation, _UnionGenericAlias):
            union_types = annotation.__args__
            if len(union_types) == 2:
                if union_types[0] is type(None):
                    optional = True
                    annotation = union_types[1]
                if union_types[1] is type(None):
                    optional = True
                    annotation = union_types[0]
        attr_type: str = "string"
        multiple = False

        if isinstance(annotation, _GenericAlias):
            base_class = annotation.__origin__
            if issubclass(base_class, Sequence):
                multiple = True
                annotation = getattr(annotation, "__args__", [str])[0]
            elif issubclass(base_class, Collection):
                multiple = True
                annotation = getattr(annotation, "__args__", [str])[0]

        if issubclass(annotation, str):
            attr_type = "string"
        if issubclass(annotation, int):
            attr_type = "integer"
        if issubclass(annotation, float):
            attr_type = "number"

        tax_name = None
        ref_target = None
        if extra and isinstance(extra[0], dict):
            tax_name = extra[0].get("taxonomy")
            if tax_name:
                ref_target = f"taxonomies.zip:{tax_name}.json"
            else:
                ref_target = extra[0].get("ref_target")
        metadata.append(
            AttributeMetadata(
                ID=name,
                title=name,
                attribute_type=attr_type,
                multiple=multiple,
                separator=";",
                ref_target=ref_target,
                extra={"taxonomy_name": tax_name} if tax_name else {},
            )
        )

    return metadata


# FIXME remove test code later
if __name__ == "__main__":
    print("\n".join(str(m) for m in get_attribute_metadata(PersonEntity)))
    print("\n")
    print("\n".join(str(m) for m in get_attribute_metadata(OpusEntity)))
    print("\n")
    print("\n".join(str(m) for m in get_attribute_metadata(PartEntity)))
    print("\n")

    # Find attributes that are used multiple times
    attrs = Counter(m.ID for m in get_attribute_metadata(PersonEntity))
    attrs += Counter(m.ID for m in get_attribute_metadata(OpusEntity))
    attrs += Counter(m.ID for m in get_attribute_metadata(PartEntity))

    print([n for n, c in attrs.items() if c > 1])

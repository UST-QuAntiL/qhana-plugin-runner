from functools import reduce
from json import dumps
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypeAlias,
    TypedDict,
    Union,
    cast,
)

from prance import BaseParser, ResolvingParser

OpenapiSpec: TypeAlias = Union[str, BaseParser, ResolvingParser]


def parse_spec(
    spec: OpenapiSpec, strict: bool = False, resolve_rels: bool = False
) -> Union[BaseParser, ResolvingParser]:
    if isinstance(spec, (BaseParser, ResolvingParser)):
        return spec
    if resolve_rels:
        return ResolvingParser(spec, strict=False)
    return BaseParser(spec, strict=strict)


def _get_endpoint_paths_2(specification: dict) -> Sequence[str]:
    return tuple(specification["paths"].keys())


def _get_endpoint_paths_3(specification: dict) -> Sequence[str]:
    return tuple(specification["paths"].keys())


def get_endpoint_paths(spec: OpenapiSpec) -> Sequence[str]:
    parser = parse_spec(spec)

    specification = parser.specification
    assert isinstance(specification, dict), "type assertion"

    match parser.version_parsed:
        case (2, _) | (2, _, _):
            return _get_endpoint_paths_2(specification)
        case (3, 0 | 1) | (3, 0 | 1, _):
            return _get_endpoint_paths_3(specification)
        case version:
            print(f"unsupported version {version}")  # FIXME throw error?
    return []


def _get_endpoint_methods_2(specification: dict, path: str) -> Sequence[str]:
    return tuple(method.lower() for method in specification["paths"].get(path, {}).keys())


def _get_endpoint_methods_3(specification: dict, path: str) -> Sequence[str]:
    return tuple(method.lower() for method in specification["paths"].get(path, {}).keys())


def get_endpoint_methods(spec: OpenapiSpec, path: str) -> Sequence[str]:
    parser = parse_spec(spec)

    specification = parser.specification
    assert isinstance(specification, dict), "type assertion"

    match parser.version_parsed:
        case (2, _) | (2, _, _):
            return _get_endpoint_methods_2(specification, path)
        case (3, 0 | 1) | (3, 0 | 1, _):
            return _get_endpoint_methods_3(specification, path)
        case version:
            print(f"unsupported version {version}")  # FIXME throw error?
    return []


def _get_endpoint_method_summary_2(
    specification: dict, path: str, method: str
) -> str | None:
    return (
        specification["paths"].get(path, {}).get(method.lower(), {}).get("summary", None)
    )


def _get_endpoint_method_summary_3(
    specification: dict, path: str, method: str
) -> str | None:
    return (
        specification["paths"].get(path, {}).get(method.lower(), {}).get("summary", None)
    )


def get_endpoint_method_summary(spec: OpenapiSpec, path: str, method: str) -> str | None:
    parser = parse_spec(spec)

    specification = parser.specification
    assert isinstance(specification, dict), "type assertion"

    match parser.version_parsed:
        case (2, _) | (2, _, _):
            return _get_endpoint_method_summary_2(specification, path, method)
        case (3, 0 | 1) | (3, 0 | 1, _):
            return _get_endpoint_method_summary_3(specification, path, method)
        case version:
            print(f"unsupported version {version}")  # FIXME throw error?
    return None


EndpointParameter = TypedDict(
    "EndpointParameter",
    {
        "name": str,
        "in": Literal["query", "header", "path", "cookie"],
        "description": str,
        "required": bool,
        "deprecated": bool,
        "allowEmptyValue": bool,
        "style": str,
        "explode": bool,
        "allowReserved": bool,
        "schema": Dict,
        "example": Any,
        "examples": Dict[str, Any],
        "content": Dict[str, Any],
    },
    total=False,
)


def _get_endpoint_parameters_3(
    specification: dict, path: str, method: str
) -> List[EndpointParameter]:
    parameters: List = (
        specification["paths"].get(path, {}).get(method.lower(), {}).get("parameters", {})
    )
    resolved_params = []

    for param in parameters:
        match param:
            case {"name": _, "in": _}:
                resolved_params.append(param)
            case {"$ref": ref}:
                resolved_params.append(_resolve_json_ref(specification, ref))
            case _:
                raise ValueError

    return resolved_params


def get_endpoint_parameters(
    spec: OpenapiSpec, path: str, method: str
) -> List[EndpointParameter]:
    parser = parse_spec(spec)

    specification = parser.specification
    assert isinstance(specification, dict), "type assertion"

    match parser.version_parsed:
        case (2, _) | (2, _, _):
            # TODO: Implement parameter parsing for 2.0. Parameters were quite different in 2.0.
            raise NotImplementedError
        case (3, 0 | 1) | (3, 0 | 1, _):
            return _get_endpoint_parameters_3(specification, path, method)
        case version:
            print(f"unsupported version {version}")  # FIXME throw error?
    return []


def get_query_variables(
    spec: OpenapiSpec, path: str, method: str
) -> List[Tuple[str, str]]:
    parsed = parse_spec(spec)
    params = get_endpoint_parameters(parsed, path, method)

    variables: List[Tuple[str, str]] = []

    for param in params:
        if param.get("in") != "query":
            continue
        name = param.get("name")
        if not name:
            continue
        value: str = ""
        schema = param.get("schema")
        if schema:
            value = _get_json_example_from_schema(parsed.specification, schema)
        variables.append((name, value))

    return variables


SecuritySchema = TypedDict(
    "SecuritySchema",
    {
        "type": Literal["apiKey", "http", "mututalTLS", "oauth2", "openIdConnect"],
        "description": str,
        "name": str,
        "in": Literal["query", "header", "cookie"],
        "scheme": str,
        "bearerFormat": str,
        "flows": Any,
        "openIdConnectUrl": str,
    },
    total=False,
)


def _get_security_schemes_3(specification: Dict) -> Dict[str, SecuritySchema]:
    security_schemes = specification.get("components", {}).get("securitySchemes", {})

    return security_schemes


def get_security_schemes(spec: OpenapiSpec) -> Dict:
    parser = parse_spec(spec)

    specification = parser.specification
    assert isinstance(specification, dict), "type assertion"

    match parser.version_parsed:
        case (2, _) | (2, _, _):
            raise NotImplementedError
        case (3, 0 | 1) | (3, 0 | 1, _):
            return _get_security_schemes_3(specification)
        case version:
            print(f"unsupported version {version}")  # FIXME throw error?
    return {}


def _get_endpoint_security_requirements_3(
    specification: Dict, path: str, method: str
) -> Optional[List[Dict[str, str]]]:
    # Get globally defined ones ("security" field of "OpenAPI Object")
    # overwrite them if the "security" field of the "Operation Object" is specified
    security_requirements: Optional[List[Dict[str, str]]] = specification.get(
        "components", {}
    ).get("security", None)
    local_security_requirements: Optional[List[Dict[str, str]]] = (
        specification["paths"].get(path, {}).get(method.lower(), {}).get("security", None)
    )

    if local_security_requirements is not None:
        security_requirements = local_security_requirements

    return security_requirements


def get_endpoint_security_requirements(
    spec: OpenapiSpec, path: str, method: str
) -> Optional[List[Dict[str, str]]]:
    parser = parse_spec(spec)

    specification = parser.specification
    assert isinstance(specification, dict), "type assertion"

    match parser.version_parsed:
        case (2, _) | (2, _, _):
            raise NotImplementedError
        case (3, 0 | 1) | (3, 0 | 1, _):
            return _get_endpoint_security_requirements_3(specification, path, method)
        case version:
            print(f"unsupported version {version}")  # FIXME throw error?
    return None


def _resolve_json_ref(specification: dict, ref: str) -> Optional[dict]:
    if not ref.startswith("#/"):
        return None  # TODO support outside refs?
    path = ref[2:].split("/")
    if not path:
        return None
    data = specification
    for ref in path:
        data = data.get(ref, None)
        if data is None:
            return None
    return data


def _get_json_example_from_schema(specification: dict, schema: dict) -> Any:
    match schema:
        case {"example": example}:
            return example
        case {"examples": [example, *_]}:
            return example
        case {"$ref": ref}:
            new_schema = _resolve_json_ref(specification, ref)
            if new_schema:
                return _get_json_example_from_schema(specification, new_schema)
        case {"anyOf": [new_schema, *_], **rest} | {"oneOf": [new_schema, *_], **rest}:
            data = _get_json_example_from_schema(specification, new_schema)
            if isinstance(data, dict):
                extra_data = _get_json_example_from_schema(specification, rest)
                if isinstance(extra_data, dict):
                    data.update(extra_data)
            return data
        case {"allOf": [*schemas], **rest}:
            schemas += [rest]
            data = [
                _get_json_example_from_schema(specification, new_schema)
                for new_schema in schemas
            ]
            if all(isinstance(d, dict) for d in data):
                data = cast(Sequence[dict], data)
                return reduce(
                    lambda d1, d2: d1.update(d2) if d1 is not None else d2, data, {}
                )
            return {"allOf": data}
        case {"type": type_, "properties": dict(props)} if "object" in type_:
            return {
                k: _get_json_example_from_schema(specification, prop_schema)
                for k, prop_schema in props.items()
                if not prop_schema.get("readOnly")
            }
        case {"type": type_, "items": dict(item)} if "array" in type_:
            return [
                _get_json_example_from_schema(specification, item)
                for _ in range(max(schema.get("minItems", 1), 1))
            ]
        case {"type": type_, "items": [*items]} if "array" in type_:
            return [
                _get_json_example_from_schema(specification, item_schema)
                for item_schema in items
            ]
        case {"enum": [const, *_]} | {"const": const} | {"default": const}:
            return const
        case {"type": type_, "pattern": pattern} if "string" in type_:
            return f"pattern: {pattern}"
        case {"type": type_} if "string" in type_:
            data = "string"
            if "minLength" in schema:
                data += f" minLength={schema['minLength']}"
            if "maxLength" in schema:
                data += f" maxLength={schema['maxLength']}"
            return data
        case {"type": type_} if "integer" in type_ or "number" in type_:
            number = schema.get("minimum", 0)
            if (excl_min := schema.get("exclusiveMinimum")) is not None:
                if excl_min is True:
                    excl_min = number
                if excl_min is not False:
                    number = excl_min + 1
            if isinstance(multiple := schema.get("multipleOf"), (int, float)):
                number += multiple - (number % multiple)
            if "integer" in type_:
                return int(number)
            else:
                return float(number)
        case {"type": type_} if "boolean" in type_:
            return True
        case {"nullable": True}:
            return None
        case {"type": type_} if "null" in type_:
            return None
        case _:
            print("Unsupported Schema:", schema)
    return None


def _extract_example(
    specification: dict, content: dict, content_type: str
) -> Optional[str]:
    match (content, content_type):
        case ({"example": example}, "application/json"):
            return dumps(example, indent="    ")
        case ({"examples": dict(example_dict)}, "application/json"):
            examples = list(example_dict.values())
            if examples:
                if "$ref" in examples[0]:
                    return dumps(
                        _resolve_json_ref(specification, examples[0]["$ref"]),
                        indent="    ",
                    )
                return dumps(examples[0]["value"], indent="    ")
        case ({"examples": dict(example_dict)}, "application/xml" | "application/html"):
            examples = list(example_dict.values())
            if examples and isinstance(example := examples[0]["value"], str):
                return example
    return None


def _get_example_body_2(specification: dict, path: str, method: str) -> str:
    if method in ("delete", "get"):
        return ""
    data = specification["paths"].get(path, {}).get(method.lower(), None)
    if data is None:
        return ""
    content = data.get("requestBody", {}).get("content", {})
    if "application/json" in content:
        example = content.get("application/json")
        if example is not None:
            return dumps(example, indent="    ")
        return dumps(
            _get_json_example_from_schema(
                specification, content["application/json"]["schema"]
            ),
            indent="    ",
        )
    elif "application/xml" in content:
        example = content.get("application/xml")
        if isinstance(example, str):
            return example
    elif "text/html" in content:
        example = content.get("text/html")
        if isinstance(example, str):
            return example
    else:
        print(content)
    return ""


def _get_example_body_3(specification: dict, path: str, method: str) -> str:
    if method in ("delete", "get"):
        return ""
    data = specification["paths"].get(path, {}).get(method.lower(), None)
    if data is None:
        return ""
    content = data.get("requestBody", {}).get("content", {})
    if "application/json" in content:
        example = _extract_example(
            specification, content["application/json"], "application/json"
        )
        if example is not None:
            return dumps(example, indent="    ")
        return dumps(
            _get_json_example_from_schema(
                specification, content["application/json"]["schema"]
            ),
            indent="    ",
        )
    elif "application/xml" in content:
        example = _extract_example(
            specification, content["application/xml"], "application/xml"
        )
        if example is not None:
            return example
    elif "application/html" in content:
        example = _extract_example(
            specification, content["application/html"], "application/html"
        )
        if example is not None:
            return example
    else:
        print(content)
    return ""


def get_example_body(spec: OpenapiSpec, path: str, method: str) -> str:
    parser = parse_spec(spec)

    specification = parser.specification
    assert isinstance(specification, dict), "type assertion"

    body = ...

    match parser.version_parsed:
        case (2, _) | (2, _, _):
            body = _get_example_body_2(specification, path, method)
        case (3, 0 | 1) | (3, 0 | 1, _):
            body = _get_example_body_3(specification, path, method)
        case version:
            print(f"unsupported version {version}")  # FIXME throw error?

    if body == ...:
        return ""
    return body


def _get_example_files_3(specification: dict, path: str, method: str) -> Sequence[str]:
    if method in ("delete", "get"):
        return []
    data = specification["paths"].get(path, {}).get(method.lower(), None)
    if data is None:
        return []

    content_types = set(data.get("requestBody", {}).get("content", {}).keys())

    content_types -= {
        "application/json",
        "application/xml",
        "application/html",
        "application/x-www-form-urlencoded",
    }

    return list(content_types)


def get_upload_files(spec: OpenapiSpec, path: str, method: str) -> Sequence[str]:
    parser = parse_spec(spec)

    specification = parser.specification
    assert isinstance(specification, dict), "type assertion"

    files = []

    match parser.version_parsed:
        case (2, _) | (2, _, _):
            pass  # TODO maybe implement
        case (3, 0 | 1) | (3, 0 | 1, _):
            files = _get_example_files_3(specification, path, method)
        case version:
            print(f"unsupported version {version}")  # FIXME throw error?

    return files


if __name__ == "__main__":
    # FIXME remove later
    spec = parse_spec(
        "https://raw.githubusercontent.com/swagger-api/swagger-petstore/master/src/main/resources/openapi.yaml"
    )
    endpoints = get_endpoint_paths(spec)
    print(endpoints)
    methods = get_endpoint_methods(spec, "/pet")
    print(methods)
    summary = get_endpoint_method_summary(spec, "/pet", "post")
    print(summary)
    body = get_example_body(spec, "/pet", "put")
    print(body)
    files = get_upload_files(spec, "/pet/{petId}/uploadImage", "post")
    print(files)
    params = get_endpoint_parameters(spec, "/pet/{petId}", "post")
    print(params)
    variables = get_query_variables(spec, "/pet/{petId}", "post")
    print(variables)
    security_schemes = get_security_schemes(spec)
    print(security_schemes)
    security_requirements = get_endpoint_security_requirements(spec, "/pet", "post")
    print(security_requirements)

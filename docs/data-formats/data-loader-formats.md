# Data Loader Formats

Data loaders load data and metadata from different sources in a specified data format that conforms to the {doc}`data-model`.

## Loading Entities

File Type Tag: `entity/list`, `entity/stream`, `entity/numeric`, `entity/vector`...

Entities can be serialized in two different formats (JSON or CSV).

The names of attributes **must** be unique for all attributes from the same data loader.

```{note}
The plugin runner provides utilities to read and write the entities from file formats specified here.
The utilities can be found in the module {py:mod}`qhana_plugin_runner.plugin_utils.entity_marshalling`.
```

The examples here use the following entities:

```{csv-table}
ID,href,color
paintA,example.com/paints/paintA,#8a2be2
paintB,example.com/paints/paintA,#e9322d
```

### Entities ({mimetype}`text/csv`)

The first column **must** be the ID column (named `ID`).
If the entity has a `href` attribute then it **must** be the second column.
All other columns are entity attributes.

The CSV file must contain a header row with all attribute names.
The attribute names (except href and ID) can then be used to lookup the attribute metadata.

Example:


```{code-block} text
ID,href,color
paintA,example.com/paints/paintA,#8a2be2
paintB,example.com/paints/paintA,#e9322d
```

### Entities ({mimetype}`application/json` or {mimetype}`application/X-lines+json`)

Entites serialized as JSON do not need a specific order for their attributes.
To support streaming parsing of files containing many entities the entites should be JSON objects with one object per line.
Files with one JSON object per line should use the {mimetype}`application/X-lines+json` mimetype.
Files using the {mimetype}`application/json` mimetype must only contain one valid JSON construct (e.g. a list or an object).

Example {mimetype}`application/json`:

```{code-block} json
[
    {"ID": "paintA","href": "example.com/paints/paintA","color": "#8a2be2"},
    {"ID": "paintB","href": "example.com/paints/paintB","color": "#e9322d"}
]
```

Example {mimetype}`application/X-lines+json`:

```{code-block} json
{"ID": "paintA","href": "example.com/paints/paintA","color": "#8a2be2"}
{"ID": "paintB","href": "example.com/paints/paintB","color": "#e9322d"}
```


## Attribute Metadata

File Type Tag: `entity/attribute-metadata`

```{note}
The plugin runner provides utilities to read attribute metadata and use it to serialize/de-serialize entity attributes.
The utilities can be found in the module {py:mod}`qhana_plugin_runner.plugin_utils.attributes`.
```

The attributes of entities (and relations) can be described by attribute metadata.
The metadata of an attribute is expressed as an entity with the following attributes:

ID
:   The name of the attribute (as used in the entity serializations)

title
:   A human readable title for the attribute

description
:   A human readable description of the attribute

type
:   The type of the scalar values of the attribute (e.g. one of `null`, `boolean`, `integer`, `number`, `string`, `url`, `ref` or a user defined type)

multiple
:   `True` if the attribute contains more that one scalar value. (Default is `False`)

ordered
:   `True` if the order of the values is important. (Default is `False`)

separator
:   A character sequence that separates the scalar values. (only used in serialization formats that do not natively support lists for attributes e.g. csv)

refTarget
:   A filename that contains the entities referenced in this attribute (when type is `ref`). If empty all entites must be searched for in all available files.

schema
:   An URL to a schema that can be used to validate the scalar values. (e.g. a json schema)


For attributes with boolean values the following values are allowed (if the serialization does not support booleans natively)

| value | serialization                                              |
|:-----:|:-----------------------------------------------------------|
| true  | `1`, `true`, `t`, `yes`, `y`, `on`                         |
| flase | `0`, `false`, `f`, `no`, `n`, `off`, `null`, `nil`, `none` |

Case and surrounding whitespace must be ignored for boolean attributes.

A data loader should produce one file containing the attribute metadata for all attributes in the data source.
Attributes already specified in {doc}`data-model` may be omitted from the attribute metadata.


### Attribute Metadata ({mimetype}`text/csv`)

The attribute metadata for the example entities:

```{code-block} text
ID,title,description,type
ID,Entity ID,the unique id of the entity,ref
href,Entity Link,link to the entity in the original data source,url
color,Color,the color of the paint,string
```

## Graphs

File Type Tag: `graph/*`

Taxonomies and other entity structures that form a graph should be serialized as a graph.
Formats like CSV are unsuitable to serialize graphs as they would need at least two files, one for the entities and one for the relations.

### Graph ({mimetype}`text/json`)

```{code-block} json
{
    "GRAPH_ID": "graphA",
    "type": "tree",
    "ref-target": "example-entities.csv",
    "entities": [
        "paintA",
        "paintB"
    ],
    "relations": [
        {"source": "paintA", "target": "paintB"}
    ]
}
```

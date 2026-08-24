# Allowed Data Types and Content Types

This page contains the complete list of the data types and content types (mimetypes) a QHAna plugin may declare in its `dataInput` and `dataOutput` metadata.

:::{important}
When adding a new type, it must be registered in **both** places: the enums in
{source}`tests/allowed_types.py` and this page. {source}`tests/test_allowed_types_docs.py`
will fail if the two get out of sync.

Before adding a new type, check whether an existing one already fits your case.
See the [data file examples](examples/index.rst) for detailed format descriptions,
and the [Input Data Model](data-model.rst) for the underlying entity, relation, and
graph model.
:::

## Data Types

A data type describes the meaning of  data.
It is independent of the serialization format, which is given by the content type.

### Namespaced Data Types

The regular form of a data type is `namespace/name`.

(dt-wildcard)=
`*`
: Wildcard. The plugin accepts (or produces) data of any type.
  Use it only when the plugin genuinely does not care about the meaning of the data, for example in generic file conversion or SQL processing plugins.

(dt-custom-clusters)=
`custom/clusters`
: Cluster assignments produced by the clustering plugins.
  Details: [custom/clusters](examples/custom.rst#custom-clusters)

(dt-custom-hello-world-output)=
`custom/hello-world-output`
: Demo text output of the hello world plugins.
  Details: [custom/hello-world-output](examples/custom.rst#custom-hello-world-output)

(dt-custom-kernel-matrix)=
`custom/kernel-matrix`
: Kernel matrix produced by the quantum kernel estimation plugins.
  Details: [custom/kernel-matrix](examples/custom.rst#custom-kernel-matrix)

(dt-custom-nisq-analyzer-result)=
`custom/nisq-analyzer-result`
: Selection result produced by the NISQ analyzer plugin.
  Details: [custom/nisq-analyzer-result](examples/custom.rst#custom-nisq-analyzer-result)

(dt-custom-pca-metadata)=
`custom/pca-metadata`
: Metadata describing a fitted principal component analysis.
  Details: [custom/pca-metadata](examples/custom.rst#custom-pca-metadata)

(dt-custom-plot)=
`custom/plot`
: Purely presentational plots (e.g. html output from pyplot).
  Details: [custom/plot](examples/custom.rst#custom-plot)

(dt-entity-wildcard)=
`entity/*`
: Any kind of entity data. Use it when a plugin works on entities without requiring a
  specific entity type.
  Details: [Entities](examples/entities.rst)

(dt-entity-attribute-metadata)=
`entity/attribute-metadata`
: Entities describing properties of attributes of other entities.
  Details: [entity/attribute-metadata](examples/entities.rst#entity-attribute-metadata)

(dt-entity-dimension-mapping)=
`entity/dimension-mapping`
: Entities recording the origin of each dimension of an `entity/vector` file.
  Details: [entity/dimension-mapping](examples/entities.rst#entity-dimension-mapping)

(dt-entity-label)=
`entity/label`
: Entities with a single `label` attribute holding a class label.
  Details: [entity/label](examples/entities.rst#entity-label)

(dt-entity-list)=
`entity/list`
: A plain list of entities.
  Details: [entity/list](examples/entities.rst#entity-list)

(dt-entity-matrix)=
`entity/matrix`
: A numeric matrix indexed by entity id in both dimensions (row first).
  Details: [entity/matrix](examples/entities.rst#entity-matrix)

(dt-entity-shaped-vector)=
`entity/shaped_vector`
: Like `entity/vector`, but each vector additionally carries a shape.
  Details: [entity/shaped_vector](examples/entities.rst#entity-shaped-vector)

(dt-entity-vector)=
`entity/vector`
: Entities whose attributes (aside from `ID` and `href`) are all single numbers, i.e. feature
  vectors.
  Details: [entity/vector](examples/entities.rst#entity-vector)

(dt-executable-circuit)=
`executable/circuit`
: A full quantum circuit, possibly parameterized.
  Details: [executable/circuit](examples/executables.rst#executable-circuit)

(dt-graph-taxonomy)=
`graph/taxonomy`
: A taxonomy tree whose nodes may carry numeric mappings.
  Details: [graph/taxonomy](examples/graphs.rst#graph-taxonomy)

(dt-image-html)=
`image/html`
: A self-contained html document rendering an image or plot, produced by the visualization
  plugins (histogram, cluster scatter). Always served as `text/html`.

(dt-provenance-execution-options)=
`provenance/execution-options`
: Options for an execution, e.g. the number of shots or the target backend. May contain
  nested values, so it must be serialized as json.
  Details: [provenance/execution-options](examples/provenance.rst#provenance-execution-options)

(dt-provenance-trace)=
`provenance/trace`
: Metadata about a single execution of an executable artifact.
  Details: [provenance/trace](examples/provenance.rst#provenance-trace)

(dt-relation-attribute-distances)=
`relation/attribute-distances`
: Pairwise entity distances per attribute, one json file per attribute in a zip archive.
  Details: [relation/attribute-distances](examples/relations.rst#relation-attribute-distances)

(dt-relation-attribute-similarities)=
`relation/attribute-similarities`
: Pairwise entity similarities per attribute, one json file per attribute in a zip archive.
  Details: [relation/attribute-similarities](examples/relations.rst#relation-attribute-similarities)

(dt-relation-element-distances)=
`relation/element-distances`
: Pairwise distances between attribute values (elements) based on their taxonomy.
  Details: [relation/element-distances](examples/relations.rst#relation-element-distances)

(dt-relation-element-similarities)=
`relation/element-similarities`
: Pairwise similarities between attribute values (elements) based on their taxonomy.
  Details: [relation/element-similarities](examples/relations.rst#relation-element-similarities)

(dt-relation-entity-distances)=
`relation/entity-distances`
: The aggregated distance between two entities across multiple attributes.
  Details: [relation/entity-distances](examples/relations.rst#relation-entity-distances)

(dt-table-html)=
`table/html`
: A self-contained html document rendering a table, produced by the confusion matrix visualization plugin.
  Always served as `text/html`.

### Data Types Without a Namespace

These types predate the `namespace/name` convention and are kept only because existing plugins still declare them.

:::{danger}
Do not use these types in new plugins.
See [Current Non-Standard Custom Types](examples/custom.rst#current-non-standard-custom-types) for the replacements.
:::

(dt-circuit)=
`circuit`
: A quantum circuit, consumed by the zxcalculus visualization plugin.
  Use [`executable/circuit`](#dt-executable-circuit) instead.

(dt-plot)=
`plot`
: A rendered plot, usually an html document.
  Use [`custom/plot`](#dt-custom-plot) instead.

(dt-qnn-weights)=
`qnn-weights`
: Trained weights of a quantum neural network, serialized as json.

(dt-representative-circuit)=
`representative-circuit`
: An example circuit illustrating the circuit a plugin executed, usually as QASM.
  Use [`executable/circuit`](#dt-executable-circuit) with a data name starting with `representative-circuit` instead.

(dt-txt)=
`txt`
: Plain text output, used by the objective function and minimizer plugins.

(dt-vqc-metadata)=
`vqc-metadata`
: Metadata of a trained variational quantum classifier.

## Content Types

A content type is the mimetype of the serialized data.
It defines *how* the data is encoded.

(ct-wildcard)=
`*`
: Wildcard. The plugin accepts (or produces) any serialization format.

(ct-application-x-lines-json)=
`application/X-lines+json`
: One json object per line. The streamable counterpart of `application/json`.
  Details: [Entities application/X-lines+json](examples/entities.rst#entities-application-x-lines-json)

(ct-application-json)=
`application/json`
: Json. Details: [Entities application/json](examples/entities.rst#entities-application-json)

(ct-application-octet-stream)=
`application/octet-stream`
: Raw binary data.

(ct-application-qasm)=
`application/qasm`
: OpenQASM source of a quantum circuit.

(ct-application-vnd-recordare-musicxml-xml)=
`application/vnd.recordare.musicxml+xml`
: MusicXML.

(ct-application-xml)=
`application/xml`
: Generic XML.

(ct-application-zip)=
`application/zip`
: A zip archive bundling several files.

(ct-audio-midi)=
`audio/midi`
: MIDI. The registered mimetype.

(ct-audio-x-midi)=
`audio/x-midi`
: MIDI.

(ct-image-svg-xml)=
`image/svg+xml`
: SVG vector graphics.

(ct-text-csv)=
`text/csv`
: CSV with a header row. 
  Details: [Entities text/csv](examples/entities.rst#entities-text-csv)

(ct-text-html)=
`text/html`
: A html document.

(ct-text-plain)=
`text/plain`
: Unstructured text.

(ct-text-xml)=
`text/xml`
: XML declared as text.

(ct-text-x-qasm)=
`text/x-qasm`
: OpenQASM declared as text.

# Wu-Palmer Pipeline

```{note}
This page documents the **legacy Wu-Palmer pipeline**. A new Wu-Palmer pipeline will be created in the new {ref}`Router plugin <router>` and will use slightly different plugin steps.

For a concrete example using the MUSE dataset, see [Using the Mini MUSE dataset](https://qhana.readthedocs.io/en/latest/muse.html#using-the-mini-muse-dataset).
```

This documentation describes the legacy Wu-Palmer pipeline in QHAna. The pipeline transforms taxonomy-based attributes step by step from categorical values into numerical vectors, which can subsequently be used, for example, for clustering methods.

The documentation is structured by plugins. New plugins or alternative processing steps can therefore be added as additional sections.

## Goal and Requirements

The Wu-Palmer pipeline is suitable for attributes whose possible values are arranged in a **tree-shaped taxonomy**. The closer two values are located to each other in this tree, the more similar they are considered.

The pipeline requires the following input data:

* a list of the {ref}`entities <data-formats/data-model:entities>` to be compared,
* metadata describing the attributes of the entities,
* a ZIP archive containing the corresponding taxonomies.

For each selected attribute, the attribute metadata must contain a `refTarget` pointing to the corresponding taxonomy file. The {ref}`Wu Palmer similarities plugin <wu-palmer>` only supports taxonomies with `"type": "tree"`.

## Pipeline Overview

```{mermaid}
flowchart TD
    A[Data Loader]
    B[Wu Palmer similarities]
    C[Sym Max Mean attribute comparer]
    D[Similarities to distances transformers]
    E[Aggregators]
    F[Multidimensional Scaling]
    G[Clustering or other ML plugins]

    A -->|entity/list| B
    A -->|entity/attribute-metadata| B
    A -->|graph/taxonomy| B
    B -->|relation/element-similarities| C
    A -->|entity/list| C
    C -->|relation/attribute-similarities| D
    D -->|relation/attribute-distances| E
    E -->|relation/entity-distances| F
    F -->|entity/vector| G
```

| Step | Plugin                                                                             | Input Data Types                                                                                                                                                                                                                  | Output Data Type                                                                                         |
| ---- | ---------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| 1    | {ref}`Wu Palmer similarities <wu-palmer>`                                          | {ref}`entity/list <data-formats/examples/entities:entity/list>`, {ref}`entity/attribute-metadata <data-formats/examples/entities:entity/attribute-metadata>`, {ref}`graph/taxonomy <data-formats/examples/graphs:graph/taxonomy>` | {ref}`relation/element-similarities <data-formats/examples/relations:relation/element-similarities>`     |
| 2    | {ref}`Sym Max Mean attribute comparer <sym-max-mean>`                              | {ref}`entity/list <data-formats/examples/entities:entity/list>`, {ref}`relation/element-similarities <data-formats/examples/relations:relation/element-similarities>`                                                             | {ref}`relation/attribute-similarities <data-formats/examples/relations:relation/attribute-similarities>` |
| 3    | {ref}`Similarities to distances transformers <attr-sim-to-attr-dist-transformers>` | {ref}`relation/attribute-similarities <data-formats/examples/relations:relation/attribute-similarities>`                                                                                                                          | {ref}`relation/attribute-distances <data-formats/examples/relations:relation/attribute-distances>`       |
| 4    | {ref}`Aggregators <distance-aggregator>`                                           | {ref}`relation/attribute-distances <data-formats/examples/relations:relation/attribute-distances>`                                                                                                                                | {ref}`relation/entity-distances <data-formats/examples/relations:relation/entity-distances>`             |
| 5    | {ref}`Multidimensional Scaling (MDS) <mds>`                                        | {ref}`relation/entity-distances <data-formats/examples/relations:relation/entity-distances>`                                                                                                                                      | {ref}`entity/vector <data-formats/examples/entities:entity/vector>`                                      |

## Meaning of the Data Types

In QHAna, the **data type** describes the semantic meaning of a file. The **content type**, on the other hand, describes its technical serialization, for example JSON or ZIP. Two files can therefore have the same content type but different data types.

The general QHAna data model distinguishes between {ref}`entities <data-formats/data-model:entities>` and {ref}`relations <data-formats/data-model:relations>`. More detailed definitions and examples of the specific types used by this pipeline are linked below.

| Data Type                                                                                                | Content Type                                                  | Meaning                                                                                                                                |
| -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| {ref}`entity/list <data-formats/examples/entities:entity/list>`                                          | `application/json`, `application/X-lines+json`, or `text/csv` | List of the actual data records. Each entity has at least a unique `ID` and the attributes to be processed.                            |
| {ref}`entity/attribute-metadata <data-formats/examples/entities:entity/attribute-metadata>`              | `application/json`, `application/X-lines+json`, or `text/csv` | Describes the attributes, including their type, whether they can contain multiple values, and which taxonomy they reference.           |
| {ref}`graph/taxonomy <data-formats/examples/graphs:graph/taxonomy>`                                      | `application/zip`                                             | ZIP archive containing one JSON file per taxonomy. For Wu-Palmer, the taxonomy must be a tree.                                         |
| {ref}`relation/element-similarities <data-formats/examples/relations:relation/element-similarities>`     | `application/zip`                                             | One JSON file per attribute containing similarities between individual attribute values.                                               |
| {ref}`relation/attribute-similarities <data-formats/examples/relations:relation/attribute-similarities>` | `application/zip`                                             | One JSON file per attribute containing `source`, `target`, and the similarity between the two entities with respect to this attribute. |
| {ref}`relation/attribute-distances <data-formats/examples/relations:relation/attribute-distances>`       | `application/zip`                                             | One JSON file per attribute containing `source`, `target`, and the distance between the two entities.                                  |
| {ref}`relation/entity-distances <data-formats/examples/relations:relation/entity-distances>`             | `application/json`                                            | One file containing `source`, `target`, and the distance aggregated over all selected attributes for each entity pair.                 |
| {ref}`entity/vector <data-formats/examples/entities:entity/vector>`                                      | `application/json`                                            | Numerical vector for each entity. All attributes except `ID` and `href` are numerical dimensions.                                      |

## Input Data

A data loader, for example the {ref}`Costume loader <costume-loader>`, produces the three input files required by the pipeline.

### Entities: `entity/list`

The {ref}`entity/list <data-formats/examples/entities:entity/list>` file contains the entities that are actually compared. Attributes can contain either a single value or multiple values.

```json
[
  {
    "ID": "entity-1",
    "href": "",
    "dominanteFarbe": ["Hellgrau", "Dunkelgrau"]
  },
  {
    "ID": "entity-2",
    "href": "",
    "dominanteFarbe": ["Dunkelbraun"]
  }
]
```

### Attribute Metadata: `entity/attribute-metadata`

The {ref}`entity/attribute-metadata <data-formats/examples/entities:entity/attribute-metadata>` describes how an attribute should be interpreted. The following fields are particularly relevant for taxonomy-based attributes:

* `ID`: name of the attribute in the entity file,
* `type`: `ref` if the values are nodes of a referenced taxonomy,
* `multiple`: `true` if an entity can have multiple values,
* `ordered`: specifies whether the order of multiple values is relevant,
* `separator`: separator for multi-valued CSV fields,
* `refTarget`: reference to the taxonomy file inside the ZIP archive.

```json
[
  {
    "ID": "dominanteFarbe",
    "title": "Dominante Farbe",
    "type": "ref",
    "multiple": true,
    "ordered": false,
    "separator": ";",
    "refTarget": "taxonomies.zip:dominanteFarbe.json"
  }
]
```

### Taxonomies: `graph/taxonomy`

The {ref}`graph/taxonomy <data-formats/examples/graphs:graph/taxonomy>` ZIP archive contains one taxonomy as a JSON file for each attribute. The `relations` describe directed parent-child relationships: `source` is the parent node and `target` is the child node.

```json
{
  "GRAPH_ID": "dominanteFarbe",
  "type": "tree",
  "entities": ["Farbe", "Grau", "Braun", "Hellgrau", "Dunkelbraun"],
  "relations": [
    {"source": "Farbe", "target": "Grau"},
    {"source": "Farbe", "target": "Braun"},
    {"source": "Grau", "target": "Hellgrau"},
    {"source": "Braun", "target": "Dunkelbraun"}
  ]
}
```

## 1. Wu Palmer similarities

**Purpose**

The {ref}`Wu Palmer similarities plugin <wu-palmer>` calculates the semantic similarity between individual values of a selected attribute. The calculation is based on the positions of the values in the corresponding tree taxonomy.

**Inputs, Output and Parameters**

| Input, Output or Parameter                      | Meaning                                                                                                                                                                 |
| ----------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Entities URL**                                | File of type {ref}`entity/list <data-formats/examples/entities:entity/list>`. It provides the attribute values that actually occur in the entities.                     |
| **Entities Attribute Metadata URL**             | File of type {ref}`entity/attribute-metadata <data-formats/examples/entities:entity/attribute-metadata>`. The corresponding taxonomy is determined through `refTarget`. |
| **Taxonomies URL**                              | ZIP file of type {ref}`graph/taxonomy <data-formats/examples/graphs:graph/taxonomy>`.                                                                                   |
| **Attributes**                                  | Names of the attributes to process, each on a separate line.                                                                                                            |
| **Consider root node as part of the hierarchy** | Determines whether the root node itself has semantic meaning and should be included in the depth calculation.                                                           |
| **Output data type**                            | {ref}`relation/element-similarities <data-formats/examples/relations:relation/element-similarities>`                                                                    |
| **Output content type**                         | `application/zip`                                                                                                                                                       |

**Processing**

For two taxonomy nodes `a` and `b`, their lowest common ancestor is determined. Similarity increases when the common ancestor is located deeper in the tree and the two nodes therefore share a longer path.

```text
WuPalmer(a, b) =
    2 * depth(LCS(a, b))
    / (depth(a) + depth(b))
```

`LCS` stands for *Lowest Common Subsumer*, meaning the lowest common ancestor. The root-node option influences the level at which depth counting starts.

The plugin also supports multi-valued attributes. It calculates the required similarities for the attribute values that occur in the entities. Because Wu-Palmer similarity is symmetric, the actual calculation for reversed value pairs is cached and reused. Therefore, `sim(a, b) = sim(b, a)`.

**Output**

The output has the data type {ref}`relation/element-similarities <data-formats/examples/relations:relation/element-similarities>` and the content type `application/zip`. The archive contains one file named `<attribute>.json` for each selected attribute.

```json
[
  {
    "source": "Hellgrau",
    "target": "Dunkelbraun",
    "similarity": 0.25
  }
]
```

The value describes only the similarity between the two **attribute values**, not yet the similarity between two complete entities.

## 2. Sym Max Mean attribute comparer

**Purpose**

The {ref}`Sym Max Mean attribute comparer <sym-max-mean>` transforms element similarities into attribute similarities between two entities. This step is particularly important for multi-valued attributes because two entities can each contain sets of different attribute values.

**Inputs, Output and Parameters**

| Input, Output or Parameter   | Meaning                                                                                                                                                                                  |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Entities URL**             | File of type {ref}`entity/list <data-formats/examples/entities:entity/list>`.                                                                                                            |
| **Element similarities URL** | ZIP file of type {ref}`relation/element-similarities <data-formats/examples/relations:relation/element-similarities>`, produced by the {ref}`Wu Palmer similarities plugin <wu-palmer>`. |
| **Attributes**               | Attributes to compare, each on a separate line.                                                                                                                                          |
| **Output data type**         | {ref}`relation/attribute-similarities <data-formats/examples/relations:relation/attribute-similarities>`                                                                                 |
| **Output content type**      | `application/zip`                                                                                                                                                                        |

**Processing**

Let `A` and `B` be the sets of values of an attribute for two entities. For each value in `A`, the highest similarity to a value in `B` is determined and the mean of these maximum similarities is calculated. The same process is then performed in the opposite direction. The mean of both directions is the symmetric max-mean similarity.

```text
SMM(A, B) = 0.5 * (
    mean over a in A of max(sim(a, b) for b in B)
    +
    mean over b in B of max(sim(b, a) for a in A)
)
```

This accounts for different numbers of attribute values without favoring either comparison direction.

Missing values are handled as follows:

* If the complete attribute is `null` for at least one of the two entities, the similarity is `null`.
* If both value sets are empty, the similarity is `null`.
* If only one value set is empty, the similarity is `0`.
* If a required element similarity is missing, `0` is used for that combination.

**Output**

The output has the data type {ref}`relation/attribute-similarities <data-formats/examples/relations:relation/attribute-similarities>` and the content type `application/zip`. The archive again contains one JSON file per attribute.

```json
[
  {
    "source": "entity-1",
    "target": "entity-2",
    "similarity": 0.375
  }
]
```

The value now describes the similarity of a complete **entity pair with respect to exactly one attribute**.

## 3. Similarities to distances transformers

**Purpose**

The {ref}`Similarities to distances transformers plugin <attr-sim-to-attr-dist-transformers>` converts each attribute similarity into an attribute distance because many subsequent methods work with distances instead of similarities. A high similarity should correspond to a small distance.

**Inputs, Output and Parameters**

| Input, Output or Parameter     | Meaning                                                                                                                    |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------- |
| **Attribute similarities URL** | ZIP file of type {ref}`relation/attribute-similarities <data-formats/examples/relations:relation/attribute-similarities>`. |
| **Attributes**                 | Attributes to transform, each on a separate line.                                                                          |
| **Transformer**                | Mathematical function used to convert similarities into distances.                                                         |
| **Output data type**           | {ref}`relation/attribute-distances <data-formats/examples/relations:relation/attribute-distances>`                         |
| **Output content type**        | `application/zip`                                                                                                          |

**Available Transformations**

For a similarity `s`, the following transformations are available:

| Transformer             | Distance calculation `d` |
| ----------------------- | ------------------------ |
| **Linear Inverse**      | `d = 1 - s`              |
| **Exponential Inverse** | `d = exp(-s)`            |
| **Gaussian Inverse**    | `d = exp(-s²)`           |
| **Polynomial Inverse**  | `d = 1 / (1 + s)`        |
| **Square Inverse**      | `d = sqrt(1 - s)`        |

The transformer expects a numerical similarity value. If a similarity is `null`, the transformation fails with an error.

For normalized Wu-Palmer values, **Linear Inverse** is the most direct transformation: a similarity of `1` results in a distance of `0`, while a similarity of `0` results in a distance of `1`.

**Output**

The output has the data type {ref}`relation/attribute-distances <data-formats/examples/relations:relation/attribute-distances>` and the content type `application/zip`. The `source` and `target` fields are preserved, while the `similarity` field is replaced by `distance`.

```json
[
  {
    "source": "entity-1",
    "target": "entity-2",
    "distance": 0.625
  }
]
```

## 4. Aggregators

**Purpose**

The {ref}`Aggregators plugin <distance-aggregator>` combines the separate attribute distances between two entities into exactly one overall distance for each entity pair.

**Inputs, Output and Parameters**

| Input, Output or Parameter  | Meaning                                                                                                              |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **Attribute distances URL** | ZIP file of type {ref}`relation/attribute-distances <data-formats/examples/relations:relation/attribute-distances>`. |
| **Aggregator**              | Function used to combine the available attribute distances.                                                          |
| **Missing data handling**   | Strategy for handling `null` distances.                                                                              |
| **Output data type**        | {ref}`relation/entity-distances <data-formats/examples/relations:relation/entity-distances>`                         |
| **Output content type**     | `application/json`                                                                                                   |

**Aggregation Methods**

| Aggregator | Meaning                                                            |
| ---------- | ------------------------------------------------------------------ |
| **Mean**   | Arithmetic mean of all attribute distances.                        |
| **Median** | Median of all attribute distances.                                 |
| **Max**    | Largest attribute distance; emphasizes strong differences.         |
| **Min**    | Smallest attribute distance; emphasizes the most similar property. |

**Handling Missing Data**

| Strategy   | Behavior                                                                                               |
| ---------- | ------------------------------------------------------------------------------------------------------ |
| **ignore** | `null` values are removed for the affected entity pair. Aggregation uses only the available distances. |
| **mean**   | A `null` value is replaced by the mean of all available distances of the same attribute.               |
| **max**    | A `null` value is replaced by the largest available distance of the same attribute.                    |

If an attribute contains only `null` values, the `mean` and `max` strategies cannot determine a replacement value and the plugin terminates with an error.

If the `ignore` strategy is used and an entity pair contains only `null` distances across all attributes, this entity pair does not appear in the output.

**Output**

The output has the data type {ref}`relation/entity-distances <data-formats/examples/relations:relation/entity-distances>` and the content type `application/json`. It contains exactly one aggregated distance value for each entity pair.

```json
[
  {
    "source": "entity-1",
    "target": "entity-2",
    "distance": 0.5133333333333334
  }
]
```

## 5. Multidimensional Scaling (MDS)

**Purpose**

The {ref}`Multidimensional Scaling (MDS) plugin <mds>` creates points in a low-dimensional space from the pairwise entity distances. Entities with a small distance should be located close to each other in the result, while entities with a large distance should be farther apart.

**Inputs, Output and Parameters**

| Input, Output or Parameter | Meaning                                                                                                         |
| -------------------------- | --------------------------------------------------------------------------------------------------------------- |
| **Entity distances URL**   | JSON file of type {ref}`relation/entity-distances <data-formats/examples/relations:relation/entity-distances>`. |
| **Dimensions**             | Number of dimensions in the resulting vector. Default value in the user interface: `2`.                         |
| **Metric**                 | **Metric MDS** considers the numerical distance values. **Nonmetric MDS** primarily considers their ranking.    |
| **SMACOF executions**      | Number of different initializations of the SMACOF optimization procedure. Default value: `4`.                   |
| **SMACOF max iterations**  | Maximum number of optimization steps per execution. Default value: `300`.                                       |
| **Output data type**       | {ref}`entity/vector <data-formats/examples/entities:entity/vector>`                                             |
| **Output content type**    | `application/json`                                                                                              |

**Processing**

The plugin first creates a symmetric distance matrix. For every entity pair, the same value is entered at positions `(i, j)` and `(j, i)`; the diagonal remains `0`.

MDS then optimizes the coordinates so that the distances between the resulting points represent the input distances as closely as possible. SMACOF is executed multiple times with different initial values to find a solution with lower stress, meaning a smaller representation error.

The individual dimensions do not have a direct semantic meaning. The relative positions and distances between the entities in the generated space are what matter.

**Output**

The output has the data type {ref}`entity/vector <data-formats/examples/entities:entity/vector>` and the content type `application/json`. Each entity receives numerical dimensions such as `dim0` and `dim1`.

```json
[
  {
    "ID": "entity-1",
    "href": "",
    "dim0": 0.143,
    "dim1": -0.527
  },
  {
    "ID": "entity-2",
    "href": "",
    "dim0": -0.311,
    "dim1": 0.408
  }
]
```

These vectors can be passed directly to clustering, classification, or visualization plugins, provided that they accept {ref}`entity/vector <data-formats/examples/entities:entity/vector>`.

## Interpretation of the Complete Data Flow

The meaning of the data changes at every step:

1. {ref}`entity/list <data-formats/examples/entities:entity/list>`: domain-specific data records containing categorical attribute values,
2. {ref}`relation/element-similarities <data-formats/examples/relations:relation/element-similarities>`: similarity between individual taxonomy nodes,
3. {ref}`relation/attribute-similarities <data-formats/examples/relations:relation/attribute-similarities>`: similarity between two entities with respect to one attribute,
4. {ref}`relation/attribute-distances <data-formats/examples/relations:relation/attribute-distances>`: distance between two entities with respect to one attribute,
5. {ref}`relation/entity-distances <data-formats/examples/relations:relation/entity-distances>`: distance between two entities aggregated across all attributes,
6. {ref}`entity/vector <data-formats/examples/entities:entity/vector>`: numerical position of an entity in a feature space.

The linked data-type pages contain the corresponding schema descriptions and examples used throughout the QHAna Plugin Runner documentation.

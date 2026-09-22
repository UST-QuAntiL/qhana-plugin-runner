(feature-engineering-pipeline-guide)=

# Feature Engineering Pipeline

This page describes how to use the **Feature Engineering Pipeline** plugin. It is written for users who want to turn a heterogeneous data set into numerical feature vectors without assembling a plugin chain by hand.

```{note}
The older, manually assembled pipeline is documented on the page {doc}`wu-palmer-pipeline`. That page is still useful for understanding the individual steps in detail. This page describes the automated replacement.
```

## What the plugin does

The Feature Engineering Pipeline takes one data set and produces one numerical vector per entity.

The problem it solves is that real data sets are heterogeneous. One row can contain

* single-valued references into a **tree taxonomy** (for example the biological class of an animal),
* multi-valued references into a taxonomy whose nodes carry a **mapping vector** instead of a meaningful tree shape (for example a habitat that is described by temperature and humidity),
* **numeric** values (for example a weight),
* multi-valued numeric values, that is a **vector per cell** (for example a temperature range).

Each of these kinds needs a different treatment before it can become part of a feature vector. The plugin asks you once, per attribute, how the attribute should be treated, and then runs the required sub-plugins for you, in the right order, with the right intermediate data types. Afterwards it merges the per-attribute results into one common vector space.

You therefore only interact with two forms: a parameter form and a routing form.

## Input data

The plugin expects the same three files that the legacy pipeline expects.

| Data type | Content type | Purpose |
| --- | --- | --- |
| {ref}`entity/list <data-formats/examples/entities:entity/list>` | `application/json`, `application/X-lines+json` or `text/csv` | The actual data records. Each entity has at least a unique `ID`. |
| {ref}`entity/attribute-metadata <data-formats/examples/entities:entity/attribute-metadata>` | `application/json`, `application/X-lines+json` or `text/csv` | Describes each attribute: its data type, whether it may contain multiple values, the separator, and which taxonomy it references. |
| {ref}`graph/taxonomy <data-formats/examples/graphs:graph/taxonomy>` | `application/zip` | One JSON file per taxonomy. |

A data loader produces these three files. The {ref}`Costume loader <costume-loader>` does this for the MUSE data, and the {ref}`Demo data loader <demo-data-loader>` ships the demo data set described below.

### What makes an attribute routable

The plugin inspects the attribute metadata and decides which options it can offer for an attribute. Two fields decide this.

| Metadata field | Value | Consequence |
| --- | --- | --- |
| `refTarget` | `taxonomies.zip:<file>.json` and the file exists in the taxonomy archive | The attribute is a **taxonomy attribute**. You get a dropdown with the available taxonomy pipelines. |
| `description` | one of `number`, `integer`, `int`, `float`, `double` | The attribute is a **numeric attribute**. You get a checkbox. |

```{note}
In the attribute metadata the field `description` carries the *data type* and the field `type` carries the attribute name. This is the convention used by the MUSE data and by all plugins in this repository. An attribute is only recognised as numeric if its `description` is one of the values listed above.
```

Attributes that are neither of the two, and attributes that appear in the metadata but not in any entity, are listed as unused and cannot be routed.

## Operating the plugin

### Step 1: parameters

The first form collects the three input files and the settings for all sub-plugins that may be started later. You fill this form once, even if you later route attributes through four different paths.

| Field | Meaning |
| --- | --- |
| **Entities URL** | The {ref}`entity/list <data-formats/examples/entities:entity/list>` file. |
| **Entities Attribute Metadata URL** | The {ref}`entity/attribute-metadata <data-formats/examples/entities:entity/attribute-metadata>` file. |
| **Taxonomies URL** | The {ref}`graph/taxonomy <data-formats/examples/graphs:graph/taxonomy>` archive. |
| **Include intermediate results** | If checked, every intermediate file produced by a sub-plugin is attached to the output of the run. |
| **Consider root node as part of the hierarchy** | Wu-Palmer setting. If checked, direct children of the root are considered similar to a degree instead of completely dissimilar. |
| **Distance Metric** | Metric used by the mapping path (Euclidean, Manhattan, Chebyshev, Cosine). |
| **Transformer** | How Wu-Palmer similarities are turned into distances. |
| **Dimensions** (MDS) | Number of dimensions each attribute embedding gets. |
| **Metric**, **SMACOF executions**, **SMACOF max iterations**, **Missing distances** | MDS settings. |
| **Concat output** | If checked, the per-attribute vectors are concatenated into one vector per entity. |
| **Output Format** | `CSV`, `JSON` or `JSON Lines`. |
| **Reduce dimensions with PCA** and the PCA settings | Optional dimensionality reduction after concatenation. |

```{note}
The MDS **Dimensions** setting applies to every attribute. With `2` dimensions and six routed attributes the concatenated vector has twelve dimensions.
```

### Step 2: routing

The second form lists every attribute the plugin found and lets you decide what happens to it.

| Attribute kind | Control | Options |
| --- | --- | --- |
| Taxonomy attribute | dropdown | `None`, `Wu-Palmer`, `One-Hot`, `Mapping` |
| Numeric attribute | checkbox | include or skip |

For taxonomy attributes the plugin shows a **recommendation**. The recommendation is derived from the taxonomy itself: if the taxonomy nodes carry mapping values, `Mapping` is recommended, otherwise `Wu-Palmer`.

Numeric attributes do not need a dropdown. Whether a numeric attribute is turned into a feature vector directly or is compared through the mapping plugin follows from the metadata, namely from whether the attribute is single- or multi-valued.

Attributes set to `None`, and numeric attributes whose checkbox is not checked, are skipped and do not appear in the result.

## The four processing paths

```{mermaid}
flowchart TD
    A[Attribute] --> B{Taxonomy or numeric?}
    B -->|taxonomy, tree| WP[Wu-Palmer path]
    B -->|taxonomy, mapping| MP[Mapping path]
    B -->|numeric, single-valued| NV[Direct feature vector]
    B -->|numeric, multi-valued| NM[Numeric mapping path]
    WP --> N[Normalization]
    MP --> N
    NV --> N
    NM --> N
    N --> C[vector-concat]
    C --> P[optional PCA]
    P --> R[entity/vector]
```

Every path ends with one {ref}`entity/vector <data-formats/examples/entities:entity/vector>` per attribute. The merge step is the same for all paths.

### Wu-Palmer path

Use this for attributes whose taxonomy is a real tree and where the position in the tree carries the meaning. Two values are similar if their lowest common ancestor is deep in the tree.

| Step | Plugin | Input | Output |
| --- | --- | --- | --- |
| 1 | {ref}`Wu Palmer similarities <wu-palmer>` | {ref}`entity/list <data-formats/examples/entities:entity/list>`, {ref}`entity/attribute-metadata <data-formats/examples/entities:entity/attribute-metadata>`, {ref}`graph/taxonomy <data-formats/examples/graphs:graph/taxonomy>` | {ref}`relation/element-similarities <data-formats/examples/relations:relation/element-similarities>` |
| 2 | {ref}`Similarities to distances transformers <element_sim-to-element_dist-transformers>` | {ref}`relation/element-similarities <data-formats/examples/relations:relation/element-similarities>` | {ref}`relation/element-distances <data-formats/examples/relations:relation/element-distances>` |
| 3 | {ref}`Attribute distance aggregator <attribute-distance-aggregator>` | {ref}`entity/list <data-formats/examples/entities:entity/list>`, {ref}`relation/element-distances <data-formats/examples/relations:relation/element-distances>` | {ref}`relation/attribute-distances <data-formats/examples/relations:relation/attribute-distances>` |
| 4 | {ref}`Attribute distance MDS <attribute-distance-mds>` | {ref}`relation/attribute-distances <data-formats/examples/relations:relation/attribute-distances>` | {ref}`entity/vector <data-formats/examples/entities:entity/vector>` |

### Mapping path

Use this for attributes whose taxonomy nodes carry a numeric mapping vector. The tree shape is then irrelevant; what counts is the position of a value in the mapping space.

| Step | Plugin | Input | Output |
| --- | --- | --- | --- |
| 1 | {ref}`Mapping distances <mapping-distances>` | {ref}`entity/list <data-formats/examples/entities:entity/list>`, {ref}`entity/attribute-metadata <data-formats/examples/entities:entity/attribute-metadata>`, {ref}`graph/taxonomy <data-formats/examples/graphs:graph/taxonomy>` | {ref}`relation/attribute-distances <data-formats/examples/relations:relation/attribute-distances>` |
| 2 | {ref}`Attribute distance MDS <attribute-distance-mds>` | {ref}`relation/attribute-distances <data-formats/examples/relations:relation/attribute-distances>` | {ref}`entity/vector <data-formats/examples/entities:entity/vector>` |

The distances are calculated with the **Distance Metric** chosen in step 1. Multi-valued cells are handled by the mapping plugin, so a habitat list with two entries and one with three entries can still be compared.

### Numeric path, single-valued

A single numeric value per entity is already a one-dimensional position. There is nothing to compare and nothing to embed, so the plugin writes the value directly as dimension `dim0` of an {ref}`entity/vector <data-formats/examples/entities:entity/vector>`. No sub-plugin is started; this path runs synchronously inside the pipeline plugin.

This is why a weight in kilograms enters the feature space as a raw kilogram value. See the section on normalization below.

### Numeric path, multi-valued

A cell that contains several numbers is a vector. These vectors are fed to the mapping plugin directly, that is the values themselves are used as the mapping coordinates.

| Step | Plugin | Input | Output |
| --- | --- | --- | --- |
| 1 | {ref}`Mapping distances <mapping-distances>` | {ref}`entity/list <data-formats/examples/entities:entity/list>` with the numeric values as vectors | {ref}`relation/attribute-distances <data-formats/examples/relations:relation/attribute-distances>` |
| 2 | {ref}`Attribute distance MDS <attribute-distance-mds>` | {ref}`relation/attribute-distances <data-formats/examples/relations:relation/attribute-distances>` | {ref}`entity/vector <data-formats/examples/entities:entity/vector>` |

```{warning}
If the cells of a multi-valued numeric attribute do not all contain the same number of values, the shorter vectors are padded with zeros. A padded zero is not "no value", it is the coordinate zero, and it shifts the distances. Prepare such attributes deliberately.
```

### Normalization

The four paths produce vectors on completely different scales. An MDS embedding usually lives roughly within a unit range, while a raw weight column can run from `0.03` to `750`. If such columns are concatenated without normalization, the largest column dominates every distance computed afterwards, and clustering effectively only sees that one attribute.

Normalization is therefore configured per attribute:

* **Input range** — either given manually, or `auto`, which takes the lowest and the highest value actually occurring in the column.
* **Output range** — `[0..1]`, `[-1..1]` or `[0..100]`.

Choose the same output range for all attributes you intend to concatenate. Use a manual input range when you want several runs or several data sets to stay comparable, because `auto` depends on the data of the current run.

### Merging

Finally the per-attribute vectors are merged.

| Step | Plugin | Input | Output |
| --- | --- | --- | --- |
| 1 | {ref}`Vector concatenation <vector-concat>` | all {ref}`entity/vector <data-formats/examples/entities:entity/vector>` files | one {ref}`entity/vector <data-formats/examples/entities:entity/vector>` |
| 2 (optional) | {ref}`PCA <pca>` | the concatenated {ref}`entity/vector <data-formats/examples/entities:entity/vector>` | a reduced {ref}`entity/vector <data-formats/examples/entities:entity/vector>` |

Concatenation only happens if **Concat output** is checked. Otherwise the run ends with one vector file per attribute, which is useful if you want to inspect the attributes separately.

## Difference to the legacy Wu-Palmer pipeline

The Feature Engineering Pipeline is not simply an automation of the legacy pipeline. It aggregates and embeds at different levels.

| Step | Legacy Wu-Palmer pipeline | Feature Engineering Pipeline |
| --- | --- | --- |
| 1 | wu-palmer produces similarities at **element** level | identical |
| 2 | Sym Max Mean aggregates to **attribute** similarities | similarity to distance conversion stays at **element** level |
| 3 | similarity to distance conversion at **attribute** level | aggregation of element distances into **attribute** distances |
| 4 | the distance aggregator collapses everything into **one** entity distance matrix | omitted |
| 5 | MDS is applied **once** to that matrix | MDS is applied **per attribute**; every attribute becomes its own vector space |
| 6 | — | vector-concat merges all attribute vectors, optionally followed by PCA |

The practical consequence is that each attribute keeps its own geometry until the very end. Merging data sets is possible with the legacy pipeline too, it is just not built into it.

```{warning}
Do not mix the two pipelines. The legacy chain uses the attribute-level transformer and the entity distance aggregator; the Feature Engineering Pipeline uses the element-level transformer and the attribute distance aggregator. The plugin names are similar, the data types are not interchangeable.
```

## The demo data set

The {ref}`Demo data loader <demo-data-loader>` ships a small data set called **Animals**: 16 entities and six attributes. It is deliberately built so that every one of the four processing paths occurs at least once.

| Attribute | Kind | Path | Notes |
| --- | --- | --- | --- |
| `taxonomicClass` | taxonomy, single-valued | Wu-Palmer | Deep tree: animal, vertebrate/invertebrate, mammal/bird/fish, down to ruminant, big cat, songbird, and so on. |
| `diet` | taxonomy, multi-valued | Wu-Palmer | Tree: plant food (grass, leaves, fruit, seeds) versus animal food (meat, fish, crustaceans, insects). |
| `habitat` | taxonomy, multi-valued | Mapping | Flat taxonomy whose nodes carry a two-dimensional mapping: mean temperature and relative humidity. |
| `activityTime` | taxonomy, single-valued | Mapping | Flat taxonomy with a one-dimensional ordinal mapping: nocturnal `0.0`, crepuscular `0.5`, diurnal `1.0`. |
| `weightKg` | numeric, single-valued | direct feature vector | Ranges from `0.03` (sparrow) to `750` (cow). |
| `temperatureRangeC` | numeric, multi-valued | numeric mapping | Two values per entity: lower and upper bound of the tolerated temperature. |

The entities are grouped so that the result is easy to judge: farm animals, predators, birds, and sea animals. The dolphin is an intentional edge case, because the taxonomy places it with the mammals while habitat, diet, and temperature place it with the sea animals. How the dolphin ends up in the feature space depends on which attributes you route and how you normalize them.

## A demo run, step by step

1. Run the {ref}`Demo data loader <demo-data-loader>` and select the data set `animals`. It produces the three input files.
2. Start the Feature Engineering Pipeline and select those three files.
3. Set **Dimensions** to `2`, check **Concat output**, and choose the output format you prefer.
4. Submit the form. The plugin analyses the data and shows the routing step.
5. Route the attributes:
   * `taxonomicClass` and `diet` to `Wu-Palmer`,
   * `habitat` and `activityTime` to `Mapping`,
   * check `weightKg` and `temperatureRangeC`.
6. Normalize `weightKg` with input range `auto` and output range `[0..1]`. Without this the weight column alone would dominate the concatenated vector. Normalize the remaining attributes to the same output range.
7. Start the run and wait until it finishes. The result is one {ref}`entity/vector <data-formats/examples/entities:entity/vector>` with twelve dimensions per animal.
8. Pass the result to a clustering plugin, for example k-Means with `k = 4`, and visualise it with the cluster scatter plot. The four groups should separate.

Then change one decision and run again. Setting `habitat` to `Wu-Palmer` instead of `Mapping` is instructive, because the habitat taxonomy is flat: Wu-Palmer can then only answer "identical" or "not identical" and the habitat information collapses.

## Reading intermediate results

If you check **Include intermediate results** in the first form, all files produced by the sub-plugins are attached to the run output. This is the fastest way to answer questions such as

* which values did Wu-Palmer actually compare (look at the element similarities),
* are the distances of one attribute all identical (then the attribute contributes nothing),
* did a numeric column enter the pipeline with the expected magnitude (look at the attribute vector before concatenation).

Without the option, only the final result is kept.

## Limits and things to prepare

The first entry is functionality that does not exist yet. The remaining entries are properties of the data you should check before starting a run.

| Limit | What that means for you |
| --- | --- |
| One-Hot encoding is not implemented | The option is selectable, but the attribute is only logged and then skipped. Do not rely on it. |
| Numeric columns must be complete | Every entity needs a value. A column that is empty for all entities must not be routed. |
| Multi-valued numeric attributes of differing length are zero-padded | The padding is a real coordinate and shifts distances. Either make the lengths equal or accept the effect knowingly. |
| Boolean attributes are not routable | Neither single- nor multi-valued. Convert them to `0`/`1` and declare them as `number` in the attribute metadata. |
| The recommendation is a heuristic | It only checks whether the taxonomy carries mapping values. It does not check whether the tree shape is meaningful. |
| There is no attribute weighting | Every routed attribute contributes the same number of dimensions. Weighting only happens implicitly through normalization. |
| PCA runs only after concatenation | You cannot reduce a single attribute before merging. |
| The pipelines run sequentially | Many routed attributes mean a long run. |
| There is no resume | If one sub-plugin fails, the whole run aborts and has to be started again. |
| MDS is not deterministic | Two runs on the same data can produce mirrored or rotated embeddings. Distances are preserved, absolute coordinates are not. |
| Flat taxonomies and empty mappings degrade silently | A flat taxonomy makes Wu-Palmer produce only `0` and `1`. An empty mapping value produces a maximum distance instead of an error. |

## Planned extensions

* Per-attribute settings for Wu-Palmer, MDS, and Mapping, so that one attribute can be embedded in two dimensions and another in five within the same run. This is in progress and not yet available.
* One-Hot encoding as a real fourth option for taxonomy attributes.
* Attribute weighting, so that an attribute can contribute more or less to the merged vector.

## MUSE4Music

MUSE4Music is the data set the Feature Engineering Pipeline was designed for, and it is a good illustration of why the four paths exist.

The data is nested: an opus consists of parts, a part of subparts, and a subpart of voices. Analyses usually run on one of these levels, most often on the voices. A single voice row already mixes all four processing paths:

* single- and multi-valued **taxonomy references**, for example `instrumentation` with values such as `t_Instrument_17;t_Instrument_27;…`, or `intervallik`,
* single-valued **counts**, for example `nr_repetitions_1_2` with a value such as `"5"`,
* multi-valued **numeric vectors**, for example `intervall_vector` with ten dimensions such as `"1;5;5;0;2;4;0;3;3;4"`, or `sequence_beats` with a variable length such as `"4;2"` for one voice and `"1;6;3;7"` for another.

Some of these taxonomies are real trees where the position carries meaning, others are flat and only useful through their mapping values. Before, you had to assemble a plugin chain per attribute kind and then find a way to bring the results together. The Feature Engineering Pipeline brings all of it into one common vector space in a single run, which is what makes it possible to compare voices across opera at all.

Three practical notes when working with MUSE4Music:

* Clarify `_na` nodes and empty cells before the run. A numeric column with gaps must not be routed (limit two above).
* `sequence_beats` has a different length per voice, so zero padding applies (limit three above). Decide whether that is acceptable or whether the attribute should be reduced to a fixed-length summary beforehand.
* Boolean columns such as `has_melody`, `is_polymetric`, or `sequence_is_exact_repetition` (`"True;False;True;True"`) have to be converted to numbers if you want to use them (limit four above).

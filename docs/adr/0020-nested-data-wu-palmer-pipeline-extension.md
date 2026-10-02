# Pipeline adaptation for comparing hierarchical Subparts (Muse4Music)

* Status: [proposed]
* Deciders: [Fabian Bühler, EnPro 2026 Team]
* Date: [2026-05-08]

## Context and Problem Statement

Adapt the existing Wu-Palmer-based plugin pipeline (originally designed for the Muse dataset) so that it can compare *Subparts* of the Muse4Music dataset, where each Subpart is a composite entity consisting of its own taxonomy attributes plus a variable number of child Voices (each *Voice* itself carrying taxonomy attributes).

(This ADR assumes that all attributes of the Subparts and Voices can be handled like a tree taxanomy for the Wu-Palmer Pipeline and therefore does not consider a restructuring **inside** the Wu-Palmer Pipeline.)

The existing pipeline is a chain of plugins (*database reader* → Wu-Palmer similarity → Sym Max Mean attibute comparer → Similarities to Distances Transformer → Distance Aggregator → *feature vector*) that operates on flat entities with a fixed attribute schema. But a Subpart is a *hierarchical/composite* object: it owns taxonomy attributes **and** contains a variable number of Voices that themselves own taxonomy attributes. The current pipeline has no concept of such nested structures. How should the pipeline be adapted conceptually so that Subparts (including their Voices) can be compared and turned into a feature vector suitable for downstream ML clustering?

A secondary, non-blocking concern is *explainability* of the resulting clustering: it should later be possible to reason about which Voice (or which group of attributes) influenced the clustering of a given Subpart. The chosen pipeline structure should not destroy this information prematurely.

## Decision Drivers

* Reuse of the existing Wu-Palmer pipeline and its plugins should be maximized.
* The adaptation must handle a *variable* number of Voices per Subpart (N ≠ M between two compared Subparts).
* Information about individual Voices should be preserved as long as possible to support later explainability of the clustering.
* The solution should generalize to further hierarchical levels (Parts contain Subparts) without a structural rewrite.
* Conceptual clarity and modularity of the plugin graph.

## Considered Options

* Option A: Pre-aggregation / flattening of Voices before the pipeline
* Option B: Nested Wu-Palmer with pairwise Voice matching inside a new Subpart-similarity plugin
* Option C: Bottom-up feature composition (concatenate / pool per-Voice feature vectors)
* Option D: Multi-level pipeline with two parallel tracks (native Subpart attributes vs. Voice-induced features)
* Option B + D combined: Outer Subpart pipeline using the existing pipeline as an inner sub-pipeline, with a dedicated nested-similarity plugin contributing a Voice-derived attribute to the outer pipeline

## Decision Outcome

Chosen option: **"Option B + D combined"**, because it maximally reuses the existing pipeline (it becomes an inner sub-pipeline), avoids any premature information loss across Voices, naturally handles variable Voice counts via an explicit matching step, and structurally preserves the Voice-to-Voice matching matrix as an artifact that can later be exploited for clustering explainability. It also generalizes recursively to higher levels of the data hierarchy (Part → Subpart → Voice).

Concretely:

* The **outer Subpart pipeline** processes the native Subpart taxonomy attributes the same way the existing pipeline already processes flat entities.
* A new **`NestedSimilarityPlugin`** invokes the existing Wu-Palmer pipeline as a sub-pipeline for Voices and performs **Voice-to-Voice matching** (e.g. Hungarian / optimal assignment, or greedy best-match) between the Voice sets of the two Subparts being compared, including a defined penalty for unmatched Voices when N ≠ M.
* The matching result (a similarity value, or a small sub-vector) is fed into the outer pipeline as an additional Subpart-level attribute and combined with the native Subpart attributes by the usual aggregator into the final Subpart feature vector.

### Positive Consequences

* The existing Wu-Palmer pipeline is reused unchanged as a sub-pipeline; no rewrite of existing plugins is needed.
* Each Voice similarity is considered individually — no information is collapsed before any comparison happens.
* The Voice-to-Voice similarity matrix and its matching are explicit, inspectable intermediate artifacts.
* Native Subpart-level attributes and Voice-induced attributes remain separable in the outer pipeline (Option D aspect), so their respective contributions to the final feature vector can be isolated.
* The same recursive pattern applies one level higher (Parts contain Subparts), so the architecture scales with the data hierarchy.
* The recursive pattern can also be used for complex attributes of Subparts as well.

### Negative Consequences

* Higher pipeline complexity: new `Plugins` must be introduced and wired.
* The matching step (e.g. Hungarian algorithm) introduces additional computational cost, especially as the number of Voices per Subpart grows.
* A clear convention for handling unmatched Voices (penalty / default similarity) must be defined and documented.
* Reducing the matching result to a single similarity value before the outer aggregator may lose some nuance; the exact shape of the matching output (scalar vs. small vector) is a follow-up decision.

## Pros and Cons of the Options

### Option A: Pre-aggregation / flattening of Voices before the pipeline

A new plugin (e.g. `VoiceAggregator`) runs *before* the Wu-Palmer plugin and reduces the N Voices of a Subpart into a fixed representation that is treated as additional Subpart attributes (e.g. per Voice taxonomy attribute → multiset / histogram / majority value across Voices).

* Good, because it is the smallest possible change to the existing pipeline.
* Good, because Wu-Palmer and all downstream plugins remain unchanged and operate on a flat object.
* Bad, because the identity of individual Voices is lost *before* Wu-Palmer ever runs, so a lot of information is destroyed early.
* Bad, because explainability is severely limited: once Voices are merged, it is no longer possible to attribute clustering effects back to a specific Voice.

### Option B: Nested Wu-Palmer with pairwise Voice matching

A new Subpart-level plugin runs an *inner* Wu-Palmer pipeline that first computes pairwise Voice-to-Voice similarities between the Voices of the two Subparts being compared (essentially calling the existing pipeline as a sub-pipeline). A **matching plugin** (e.g. Hungarian / optimal assignment / greedy best-match) then turns the N×M Voice similarity matrix into a single Subpart-level Voice-similarity value, which is fed into the outer pipeline as if it were an additional Subpart attribute, alongside the native Subpart attributes.

* Good, because the existing pipeline is reused as a building block (recursive Wu-Palmer that mirrors the natural data hierarchy).
* Good, because the Voice-to-Voice matching matrix is an explicit, inspectable artifact.
* Good, because variable Voice counts (N ≠ M) are handled cleanly by the matching step plus a penalty for unmatched Voices.
* Bad, because the matching output is a single similarity value, not directly a feature vector (acceptable, since downstream distance/aggregator plugins already consume similarities).
* Bad, because the optimal assignment step adds computational complexity.

### Option C: Bottom-up feature composition

The existing pipeline is run end-to-end *per Voice*, producing one feature vector per Voice. A new Subpart-level aggregator plugin then combines these per-Voice vectors (concatenation, mean, pooling, set embedding, …) together with the native Subpart features into a Subpart feature vector.
<!--Not completely sure if it works this way with single feature vectors PER VOICE.-->

* Good, because it cleanly separates concerns: the Voice pipeline is unchanged; the Subpart pipeline is one layer on top.
* Good, because it is highly modular and easy to reason about per layer.
* Bad, because *concatenation* requires a canonical Voice ordering and a fixed Voice count, which is unrealistic for this dataset.
* Bad, because *mean / pooling* aggregations destroy per-Voice information again (similar weakness as Option A).
* Bad, because set-based embeddings (e.g. Deep Sets) are significantly more complex and pull the architecture away from the existing plugin paradigm.
* Mixed for explainability: good for concatenation (each dimension has a clear Voice origin), poor for pooling.

### Option D: Multi-level pipeline with two parallel tracks

Two parallel pipeline tracks per Subpart:

1. Subpart-native attributes → Wu-Palmer → distances → partial vector A
2. Voices → Wu-Palmer → distances → aggregation → partial vector B

The final Subpart feature vector is the combination of A and B. This is essentially a generalization of B and C and can be combined with either of them.

* Good, because native and Voice-induced information remain separately traceable in the final feature vector.
* Good, because it is very flexible and composable with Option B (B providing the Voice track) or Option C (C providing the Voice track).
* Bad, because it adds more pipeline complexity and more plugin wiring.
* Bad on its own, because it does not by itself prescribe *how* the Voice track aggregates variable-length Voice sets — that question is delegated to B or C.

## Links

* Refines the existing [Wu-Palmer plugin pipeline](https://qhana.readthedocs.io/en/latest/muse.html) used for the Muse dataset.

> Note: drafted with the help of Claude Opus 4.7.

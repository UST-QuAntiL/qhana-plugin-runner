# Pipeline adaption for non-tree taxonomies

* Status: [proposed]
* Deciders: [Fabian Bühler, EnPro 2026 Team]
* Date: [2026-05-11]

Technical Story: [Adaptation of Wu-Palmer Pipeline for MUSE4MUSIC nominal/ordinal taxonomies]

## Context and Problem Statement

The first step of the existing Qhana pipeline uses the Wu-Palmer algorithm to compute similarity scores between attributes. However, Wu-Palmer mathematically requires a single-inheritance tree structure to calculate depth and Lowest Common Ancestors (LCA). The new MUSE4MUSIC dataset contains numerous attributes (e.g., Notenwert, Ausdruck, Bewegung im Tonraum) that are flat nominal lists or ordinal scales, not trees. If flat lists are fed into a Wu-Palmer algorithm, it evaluates them as sibling nodes with the same depth, degrading the calculation into a binary exact-match metric (all non-identical values get the exact same low similarity score).

How should the pipeline be adapted to accurately calculate similarities for non-tree taxonomies without breaking the downstream plugins (Sym-Max-Mean comparer, Distance Transformers, Aggregators)?

## Decision Drivers

* Domain Accuracy: The resulting similarity scores must reflect actual musical relationships (e.g., a "Halbe" note is mathematically closer to a "4-tel" than to a "64-tel").
* Pipeline Compatibility: Downstream plugins (Step 2 (Sym-Max-Mean comparer) onwards) must remain unaffected; they expect an attribute similarity score between 0.0 and 1.0.
* Modularity: The solution should cleanly separate the logic for structural comparisons (trees) from statistical/explicit comparisons (flat lists).
* Maintainability: Avoid hardcoding dataset-specific rules deep within calculation algorithms.

## Considered Options

* Option A: Synthetic Taxonomy Generation (Pre-processing flat lists into artificial trees)
* Option B: Type-Aware Router Plugin with Parallel Metric Sub-Plugins
* Option C: Pre-Pipeline Data Transformation (Replacing nominal strings with numeric values in the dataset)

## Decision Outcome

Chosen option: **"Option B: Type-Aware Router Plugin with Parallel Metric Sub-Plugins and a Combiner"**, because it provides the highest musical accuracy while maintaining a clean pipeline architecture.

Instead of forcing a single metric (Wu-Palmer) to handle data it wasn't designed for, Step 1 of the pipeline is expanded into a scatter-gather pattern:

1. A new **Router Plugin** reads the taxonomy type metadata of the incoming attribute.
2. It routes the data to parallel sub-plugins: `Tree` types go to the existing **Wu-Palmer Plugin**, while `Nominal/Ordinal` types go to a new **MatrixLookupPlugin** (which loads pre-defined CSV/JSON similarity matrices, e.g., for Ausdruck) or **NumericDistancePlugin** (for mathematically mappable values like Notenwert).
3. A new Similarity Combiner Plugin waits for these parallel calculations to finish. It takes the separate output files from the Wu-Palmer and non-tree plugins and merges them into a single, unified similarity output file.

This single file with similarity scores (0.0 to 1.0) then seamlessly feeds into the unmodified Sym-Max-Mean attribute comparer (Step 2).

### Positive Consequences

* **High Domain Accuracy:** Musicology experts can define exact similarity matrices for subjective nominal lists, and numeric metrics can handle fractional note values accurately.
* **Downstream Stability:** Because the Combiner Plugin formats the merged data exactly like the original Wu-Palmer output, plugins 2 through 5 require zero modifications.
* **Extensibility:** If a new data type is introduced in the future, a new sub-plugin can easily be added to the Router and caught by the Combiner.

### Negative Consequences

* **Increased Configuration Burden:** Requires the manual creation and maintenance of lookup matrices for nominal taxonomies.
* **Pipeline Complexity:** Step 1 becomes a composite scatter-gather workflow (Router $\rightarrow$ Parallel Metrics $\rightarrow$ Combiner) rather than a single algorithm step, increasing the orchestration overhead.

## Pros and Cons of the Options

### Option A: Synthetic Taxonomy Generation

A pre-processing script imposes an artificial hierarchy onto the flat lists (e.g., grouping Notenwert into "Long", "Medium", "Short" parent categories) so they can be processed by the unmodified Wu-Palmer plugin.

* Good, because it requires zero architectural changes to the existing pipeline plugins.
* Bad, because the generated trees are arbitrary and subjective.
* Bad, because Wu-Palmer still cannot capture exact mathematical proportions (e.g., that a half note is exactly twice as long as a quarter note).

### Option B: Type-Aware Router Plugin with Parallel Metric Sub-Plugins

Replaces the strict Wu-Palmer entry point with a routing layer, specialized metric calculation plugins, and a final combiner plugin to unify the outputs.

* Good, because it uses the mathematically correct tool for each specific data type.
* Good, because the Combiner Plugin ensures the structural contract with the downstream pipeline is perfectly maintained.
* Bad, because it requires developing and orchestrating three new plugin types (Router, Specific Metrics, Combiner) instead of just one.

### Option C: Pre-Pipeline Data Transformation

Modify the raw MUSE4MUSIC dataset before it enters Qhana, mapping all nominal values to pre-calculated numerical vectors. The pipeline only handles numeric distances.

* Good, because the pipeline logic remains incredibly simple.
* Bad, because it alters the raw data, destroying the human-readable nominal strings early in the process.
* Bad, because it harms explainability; users analyzing the clustering results will see arbitrary vector numbers instead of terms like "homophon" or "polyphon".

## Links

Refines the first stage of the Wu-Palmer pipeline (Similarity Calculation).

Relates to ADR: ["Pipeline adaptation for comparing hierarchical Subparts"](0020-nested-data-wu-palmer-pipeline-extension.md) (handles the nested routing).

> Note: drafted with the help of Gemini 3.1 Pro
# All Plugins

:::{list-table} Plugin Overview
:header-rows: 1
:width: 100%
:widths: 30 10 30

* - Plugin
  - Type
  - Tags
* - [Aggregators (@v0.2.1)](#distance-aggregator)

    distance-aggregator@v0-2-1
  - processing
  - distance-calculation

    preprocessing
* - [Multidimensional Scaling (MDS) (@v0.2.1)](#mds)

    mds@v0-2-1
  - processing
  - distance-calculation

    embedding

    feature-engineering

    preprocessing

:::

## Overview

**Used tags:** `ML`, `classical`, `distance-calculation`, `embedding`, `feature-engineering`, `json`, `preprocessing`

**Input formats:** `application/json`, `application/zip`\
**Output formats:** `application/json`, `application/zip`

**Input datatypes:** `custom/attribute-distances`, `custom/entity-distances`, `entity/label`, `entity/vector`\
**Output datatypes:** `custom/attribute-distances`, `custom/entity-distances`, `entity/vector`

## Plugins

(distance-aggregator)=
### Aggregators (@v0.2.1)

processing – distance-calculation, preprocessing\
*Path:* {file}`stable_plugins/classical_ml/data_preparation/aggregators.py`

Aggregates attribute distances to entity distances.

**Inputs:**

| Data Type | Content Type | Required |
|-----------|--------------| :------: |
|custom/attribute-distances|application/zip|✓|


**Outputs:**

| Data Type | Content Type | Always |
|-----------|--------------| :----: |
|custom/entity-distances|application/zip|✓|


(mds)=
### Multidimensional Scaling (MDS) (@v0.2.1)

processing – distance-calculation, embedding, feature-engineering, preprocessing\
*Path:* {file}`stable_plugins/classical_ml/scikit_ml/mds.py`

Converts distance values (distance matrix) to points in a space.

**Inputs:**

| Data Type | Content Type | Required |
|-----------|--------------| :------: |
|custom/entity-distances|application/json|✓|


**Outputs:**

| Data Type | Content Type | Always |
|-----------|--------------| :----: |
|entity/vector|application/json|✓|




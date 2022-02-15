# Preprocessing metadata [undecided]

* Status: [undecided]
* Deciders: 
* Date: [2022-02-14]


## Context and Problem Statement

If training data got preprocessed to train a machine learning model, then previously unknown data points have to undergo the same preprocessing steps, before the model is able to process them.

## Decision Drivers <!-- optional -->

* There needs to be a method of saving preprocessing steps, inorder for them to be applied to previously unknown data.

## Considered Options

* Each preprocessing plugin provides its own metadata file.
* A unified preprocessing metadata file, referencing functions.
* A unified preprocessing metadata file, referencing additional plugins.

## Decision Outcome

Chosen option: None yet.

### Positive Consequences <!-- optional -->

* /

### Negative Consequences <!-- optional -->

* /

## Pros and Cons of the Options <!-- optional -->

### Each preprocessing plugin provides its own metadata file

As the title mentions, each preprocessing plugin (e.g. scaling or PCA) provides its own metadata file.
Such a metadata file could look like the following for PCA:
```{code-block} json
{
    "mean":[0.01317,0.002229],
    "components":[[-1.0,0.0]]
}
```

* Good, because it is fast to implement.
* Bad, because there is no way to do multiple preprocessing steps and execute them in the correct order.

### A unified preprocessing metadata file, referencing functions

A unified preprocessing metadata file, in which the functions (e.g. for multiplication or subtraction) and their necessary parameters are referenced in the correct order of execution.
Such a metadata file could look like the following for first scaling and then doing a PCA:
```{code-block} json
[
    {
        "title":"Min"
        "description":"A vector with the minimum of each dimension",
        "function":"subtract",
        "params":{x1: x, x2: [2,1]}
    }
    {
        "title":"Max"
        "description":"A vector with the maximum of each dimension, after subtracting the minimum",
        "function":"divide",
        "params":{x1: x, x2: [10,10]}
    }
    {
        "title":"Mean",
        "description":"A vector containing the mean of each feature",
        "function":"subtract",
        "params":{x1: x, x2: [0.01317,0.002229]}
    },
    {
        "title":"Transposed Principle Components",
        "description":"A matrix containing the principle components of a PCA, but transposed",
        "function":"matmul",
        "params":{x1: x, x2: [[-1.0], [0.0]]}
    }
]
```

* Good, because it allows for multiple different preprocessing steps to be executed.
* Bad, because it would probably take a lot of effort to implement it.

### A unified preprocessing metadata file, referencing additional plugins

Each preprocessing method needs two plugins. One that computes and saves the components needed for the preprocessing 
(e.g. the principle components for PCA) and another plugin that transforms previously 
unknown points with the help of the saved components. The second type of plugins and the necessary parameters would then 
be referenced, in order, in a unified preprocessing metadata file.
Such a metadata file could look like the following for first scaling and then doing a PCA:
```{code-block} json
[
    {
        "plugin":"scaling",
        "type":"minmax",
        "params":{x: x, min: [2,1], max: [10,10]}
    }
    {
        "title":"pca",
        "type":"normal",
        "params":{x: x, mean: [0.01317,0.002229], components: [[-1.0,0.0]]}
    }
]
```

* Good, because it allows for multiple different preprocessing steps to be executed.
* Good, because it is more flexible and less has to be implemented, e.g. use kernels from scikit-learn!
* Bad, because there would be two plugins for each preprocessing option.

<!-- markdownlint-disable-file MD013 -->

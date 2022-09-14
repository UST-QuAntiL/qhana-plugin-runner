Writing Templates
=================

A QHAna template is a way to display and organize plugins that are related somehow.
For example you can write a template that displays the plugins needed for a specific task.

Template Definition
-------------------

You can define a template by creating a `.json` file in the folder `templates`.

A template has a name, description, and a list of categories.
A category has a name, description, and a plugin filter expression.

Plugin Filter Expressions
-------------------------

A plugin filter expression can either be:

* a string. It matches plugin tags, names, and identifiers. So the expression is true for a plugin, if either the string is one of the tags, the name, or the identifier.
* an or. This is represented by a object, that has only one attribute called `"or"`. This attribute is a list of plugin filter expressions.
* an and. This is represented by a object, that has only one attribute called `"and"`. This attribute is a list of plugin filter expressions.
* a not. This is represented by a object, that has only one attribute called `"not"`. This attribute is a single plugin filter expression.

.. code-block::
    :caption: An example to show the syntax

    {
        "or": [
            "thistag",
            "thattag",
            {
                "and": [
                    "specificname",
                    "anothertag"
                ]
            }
        ]
    }

This plugin filter expression will match all plugins, that either have `thistag`, or `thattag` as a tag, name, or identifier.
But it will also match plugins, that have `specificname` and `anothertag` as a tag, name, or identifier.

Putting it toghether
--------------------

.. code-block::
    :caption: A complete example

    {
        "name": "MUSE",
        "description": "Template for MUSE workflow",
        "categories": [
            {
              "name": "Load data",
              "description": "Plugin for loading costume data",
              "filter": "costume-loader"
            },
            {
              "name": "Data Preperation",
              "description": "Plugins for data Perperation",
              "filter": {
                    "or": [
                        "similarity-cache-generation",
                        "wu-palmer",
                        "attribute-similarity-calculation",
                        "sim-to-dist",
                        "aggregator",
                        "dist-to-points"
                    ]
                }
            },
            {
              "name": "Quantum Part",
              "description": "Plugin for Quantum Algorithm",
              "filter": "points-to-clusters"
            },
            {
              "name": "Visualization",
              "description": "Plugin for visualization",
              "filter": "visualization@v0-1-0"
            }
        ]
    }


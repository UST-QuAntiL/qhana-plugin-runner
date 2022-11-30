Plugin Recipes
==============

Plugin recipes provide a way to describe how to execute a plugin or a sequence of plugins.

.. code-block:: json

    {
        "recipeId": "someId",
        "plugin": "plugin-name",
        "version": "v1.0.0",
        "inputs": {
            "userInputA": {
                "type": "string"
            },
            "userInputB": {
                "type": "data",
                "dataType": "entity/stream",
                "contentTypes": [
                    "text/csv",
                    "application/x-lines+json"
                ]
            }
        },
        "captureOutputs": {
            "pluginOutput": {
                "name": { "pattern": "output(\\.csv)?" },
                "dataType": "entity/stream",
                "contentTypes": [
                    "text/csv",
                    "application/x-lines+json"
                ],
                "isOptional": false
            },
            "visualization": {
                "name": "plot.html",
                "dataType": "custom/plot",
                "contentType": "text/html",
                "isOptional": true
            }
        },
        "steps": [
            {
                "ID": "main",
                "inputs": {
                    "example": "stringConstant",
                    "data": {
                        "var": "userInputB"
                    }
                }
            },
            {
                "ID": "demo",
                "inputs": {
                    "fields": {
                        "var": "userInputB"
                    }
                }
            }
        ]
    }


.. code-block:: json

    {
        "recipeId": "compoundRecipe",
        "recipes": {
            "someId": "http://.../recipes/someId",
            "otherRecipe": {
                "recipeId": "inlineRecipe",
                "...": "..."
            }
        }
        "inputs": {
            "userInputA": {
                "$ref": "#someId/userInputA"
            },
            "userInputOther": {
                "const": "http://..."
            }
        },
        "captureOutputs": {
            "recipeOutput": {
                "$ref": "#someId/pluginOutput"
            },
            "visualization": {
                "$ref": "#someId/visualization"
            }
        },
        "dataFlow": {
            "userInputOther": ["#otherRecipe/input"],
            "#otherRecipe/output": ["#someId/userInputB"],
            "userInputA": ["#someId/userInputA"]
        }
    }

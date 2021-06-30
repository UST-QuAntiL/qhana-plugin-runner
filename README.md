# A Runner for QHAna Plugins

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![GitHub license](https://img.shields.io/github/license/UST-QuAntiL/qhana-plugin-runner)](https://github.com/UST-QuAntiL/qhana-plugin-runner/blob/main/LICENSE)
![Python: >= 3.7](https://img.shields.io/badge/python-^3.7-blue)
[![Documentation Status](https://readthedocs.org/projects/qhana-plugin-runner/badge/?version=latest)](https://qhana-plugin-runner.readthedocs.io/en/latest/?badge=latest)

This package uses Poetry ([documentation](https://python-poetry.org/docs/)).

Original template repository: <https://github.com/buehlefs/flask-template/>

## VSCode

For vscode install the python extension and add the poetry venv path to the folders the python extension searches for venvs.

On linux:

```json
{
    "python.venvFolders": [
        "~/.cache/pypoetry/virtualenvs"
    ]
}
```

## Development

Run `poetry install` to install dependencies.

If an environment variable specified in `.flaskenv` should be changed locally add a `.env` file with the corresponding variables set.

Run the development server with

```bash
poetry run flask run
```

Start a redis instance in a docker container and start the worker process used for executing background tasks with

```bash
poetry run invoke start-broker
poetry run invoke worker  # use strg+c to stop worker
```

### Trying out the Plugin-Runner

Start the plugin runner using the instructions above.

#### The API:

<http://localhost:5005/>

#### OpenAPI Documentation:

Configured in `qhana_plugin_runner/util/config/smorest_config.py`.

  * Redoc (view only): <http://localhost:5005/redoc>
  * Rapidoc (recommended): <http://localhost:5005/rapidoc>
  * Swagger-UI: <http://localhost:5005/swagger-ui>
  * OpenAPI Spec (JSON): <http://localhost:5005/api-spec.json>

#### Debug pages:

  * Index: <http://localhost:5005/debug/>
  * Registered Routes: <http://localhost:5005/debug/routes>\
    Useful for looking up which endpoint is served under a route or what routes are available.



## What this Repository contains

This plugin runner uses the following libraries to build a rest app with a database on top of flask.

 *  Flask ([documentation](https://flask.palletsprojects.com/en/2.0.x/))
 *  Flask-Cors ([documentation](https://flask-cors.readthedocs.io/en/latest/))\
    Used to provide cors headers.\
    Can be configured or removed in `qhana_plugin_runner/__init__.py`.
 *  flask-babel ([documentation](https://flask-babel.tkte.ch), [babel documentation](http://babel.pocoo.org/en/latest/))\
    Used to provide translations.\
    Can be configured in `qhana_plugin_runner/babel.py` and `babel.cfg`.\
    Translation files and Folders: `translations` (and `messages.pot` currently in .gitignore)
 *  Flask-SQLAlchemy ([documentation](https://flask-sqlalchemy.palletsprojects.com/en/2.x/), [SQLAlchemy documentation](https://docs.sqlalchemy.org/en/14/))\
    ORM Mapper for many SQL databases.\
    Models: `qhana_plugin_runner/db/models`\
    Config: `qhana_plugin_runner/util/config/sqlalchemy_config.py` and `qhana_plugin_runner/db/db.py`
 *  Flask-Migrate ([documentation](https://flask-migrate.readthedocs.io/en/latest/), [Alembic documentation](https://alembic.sqlalchemy.org/en/latest/index.html))\
    Provides automatic migration support based on alembic.\
    Migrations: `migrations`
 *  flask-smorest ([documentation](https://flask-smorest.readthedocs.io/en/latest/), [marshmallow documentation](https://marshmallow.readthedocs.io/en/stable/), [apispec documentation](https://apispec.readthedocs.io/en/latest/), [OpenAPI spec](http://spec.openapis.org/oas/v3.0.2))\
    Provides the API code and generates documentation in form of a OpenAPI specification.\
    API: `qhana_plugin_runner/api`\
    Config: `qhana_plugin_runner/util/config/smorest_config.py` and `qhana_plugin_runner/api/__init__.py`
 *  Flask-JWT-Extended ([documentation](https://flask-jwt-extended.readthedocs.io/en/stable/))\
    Provides authentication with JWT tokens.\
    Config: `qhana_plugin_runner/util/config/smorest_config.py` and `qhana_plugin_runner/api/jwt.py`
 *  Sphinx ([documentation](https://www.sphinx-doc.org/en/master/index.html))\
    The documentation generator.\
    Config: `pyproject.toml` and `docs/conf.py` (toml config input is manually configured in `conf.py`)
 *  sphinxcontrib-redoc ([documantation](https://sphinxcontrib-redoc.readthedocs.io/en/stable/))\
    Renders the OpenAPI spec with redoc in sphinx html output.\
    Config: `docs/conf.py` (API title is read from spec)
 *  Celery ([documentation](https://docs.celeryproject.org/en/stable/index.html))\
    A task queue for handling asynchrounous background tasks and scheduled/periodic tasks.\
    Config: `qhana_plugin_runner/config/celery_config.py` and `qhana_plugin_runner/celery.py`\
    Module to start celery as worker: `qhana_plugin_runner/celery_worker.py` (use `invoke worker` command)
 *  redis (dependency for Celery)\
    can be started via an invoke task
 *  packaging ([documentation](https://packaging.pypa.io/en/latest/index.html))\
    for parsing and sorting PEP 440 version string (the plugin versions)
 *  invoke ([documentation](http://www.pyinvoke.org))\
    small cli for easier dev environments
    Tasks: `tasks.py`

Additional files and folders:

 *  `default.nix` and `shell.nix`\
    For use with the [nix](https://nixos.org) ecosystem.
 *  `pyproject.toml`\
    Poetry package config and config for the [black](https://github.com/psf/black) formatter (also sphinx and other python tools).
 *  `.flaskenv`\
    environment variables used for the python project (no secrets here!)
 *  `.env`\
    local environment variable overrides (in `.gitignore`)
 *  `.flake8`\
    Config for the [flake8](https://flake8.pycqa.org/en/latest/) linter
 *  `.editorconfig`
 *  `tests`\
    Reserved for unit tests.
 *  `instance` (in .gitignore)\
    See <https://flask.palletsprojects.com/en/2.0.x/config/#instance-folders>
 *  `qhana_plugin_runner/templates` and `qhana_plugin_runner/static` (currently empty)\
    Templates and static files of the flask app
 *  `docs`\
    Folder containing a sphinx documentation
 *  `typings`\
    Python typing stubs for libraries that have no type information.
    Mostly generated with the pylance extension of vscode. (currently empty)
 *  `tasks.py`\
    Tasks that can be executed with `invoke` (see [celery background tasks](#celery-background-tasks))
 *  `plugins`\
    A folder to place plugins in during initial development. Mature plugins should be relocated into seperate repositories eventually.


## Babel

```bash
# initial
poetry run pybabel extract -F babel.cfg -o messages.pot .
# create language
poetry run pybabel init -i messages.pot -d translations -l en
# compile translations to be used
poetry run pybabel compile -d translations
# extract updated strings
poetry run pybabel update -i messages.pot -d translations
```

## SQLAlchemy

```bash
# create dev db (this will NOT run migrations!)
poetry run flask create-db
# drop dev db
poetry run flask drop-db
```

## Migrations

```bash
# create a new migration after changes in the db (Always manually review the created migration!)
poetry run flask db migrate -m "Initial migration."
# upgrade db to the newest migration
poetry run flask db upgrade
# help
poetry run flask db --help
```

## Celery background tasks

Use invoke to run celery commands (e.g. to start the celery worker), as it will automatically apply `.flaskenv` and `.env` environment variables.

```bash
# Start a redis instance in a docker container with
poetry run invoke start-broker

# start a worker instance
poetry run invoke worker

# stop redis container
poetry run invoke stop-broker

# remove existing redis container (e.g. to set a new port)
poetry run invoke reset-broker
poetry run invoke start-broker --port="6379"

# get help for available commands
poetry run invoke --list
poetry run invoke worker --help
```


## Compiling the Documentation

```bash
poetry shell
cd docs
make html

# export/update requirements.txt from poetry dependencies (for readthedocs build)
poetry export --dev --format requirements.txt --output docs/requirements.txt
```

Update the python source documentation

```bash
poetry run sphinx-apidoc --separate --force -o docs/source . ./tasks.py docs plugins migrations
```



## Acknowledgements

Current development is supported by the [Federal Ministry for Economic Affairs and Energy](http://www.bmwi.de/EN) as part of the [PlanQK](https://planqk.de) project (01MK20005N).

## Haftungsausschluss

Dies ist ein Forschungsprototyp.
Die Haftung für entgangenen Gewinn, Produktionsausfall, Betriebsunterbrechung, entgangene Nutzungen, Verlust von Daten und Informationen, Finanzierungsaufwendungen sowie sonstige Vermögens- und Folgeschäden ist, außer in Fällen von grober Fahrlässigkeit, Vorsatz und Personenschäden, ausgeschlossen.

## Disclaimer of Warranty

Unless required by applicable law or agreed to in writing, Licensor provides the Work (and each Contributor provides its Contributions) on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied, including, without limitation, any warranties or conditions of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR PURPOSE.
You are solely responsible for determining the appropriateness of using or redistributing the Work and assume any risks associated with Your exercise of permissions under this License.

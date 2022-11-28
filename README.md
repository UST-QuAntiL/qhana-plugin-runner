# A Runner for QHAna Plugins

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![GitHub license](https://img.shields.io/github/license/UST-QuAntiL/qhana-plugin-runner)](https://github.com/UST-QuAntiL/qhana-plugin-runner/blob/main/LICENSE)
![Python: >= 3.8](https://img.shields.io/badge/python-^3.8-blue)
[![Documentation Status](https://readthedocs.org/projects/qhana-plugin-runner/badge/?version=latest)](https://qhana-plugin-runner.readthedocs.io/en/latest/?badge=latest)

This package uses Poetry ([documentation](https://python-poetry.org/docs/)).

Original template repository: <https://github.com/buehlefs/flask-template/>

> This repository also contains a Dockerfile and has pre-built docker containers available as GitHub packages.
> Please see the [Docker](#docker) section for more information.

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

````{note}
First start only:

```bash
# create development database
poetry run flask create-db

# install requirements of plugins
poetry run flask install
```
````

Run the development server with

```bash
poetry run flask run
```

Start a redis instance in a docker container and start the worker process used for executing background tasks with

```bash
poetry run invoke start-broker
poetry run invoke worker  # use strg+c to stop worker
```


### Debugging with VSCode

There is a default launch configuration for vscode that should work on all platforms.
To use the configuration copy `default-launch.json` to `.vscode/launch.json`.
The `all` configuration starts the API and the worker process.
For code changes both debugging sessions must be restarted as they do not autoreload code!


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


#### Plugins:

Plugin requirements can be installed with the following command:

```bash
poetry run flask install # --skip-runner-requirements

# only check generated requirements file:
poetry run flask install --dry-run
```


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
 *  Requests ([documentation](https://docs.python-requests.org/en/master/))\
    For interacting with http apis and loading files from URLs.
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


## Poetry Commands

```bash
# install dependencies from lock file in a virtualenv
poetry install

# open a shell in the virtualenv
poetry shell

# update dependencies
poetry update
poetry run invoke update-dependencies # to update other dependencies in the repository

# run a command in the virtualenv (replace cmd with the command to run without quotes)
poetry run cmd
```

## Invoke Tasks

[Invoke](http://www.pyinvoke.org) is a python tool for scripting cli commands.
It allows to define complex commands in simple python functions in the `tasks.py` file.

:warning: Make sure to update the module name in `tasks.py` after renaming the `qhana_plugin_registry` module!

```bash
# list available commands
poetry run invoke --list

# update dependencies (requirements.txt in ./docs and licenses template)
poetry run invoke update-dependencies

# Compile the documentation
poetry run invoke doc

# Open the documentation in the default browser
poetry run invoke browse-doc
```


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

ℹ️ Try to minimize the number of migrations and only create a new one when your changes are likely final.
Altenatively merge all your new migration into one before submitting a pull request.

If you have added new mapped dataclasses or modified existing ones, a migration script needs to be added.
This script updates the tables and columns of the database to match the mapped dataclasses.
To generate the migration script you need to do the following steps:

```bash
# delete the database
rm instance/qhana_plugin_runner.db
# upgrade the database to the latest migration
poetry run flask db upgrade
# generate a new migration script for the changes you made (always manually review the created migration!)
poetry run flask db migrate -m "changelog message"
# upgrade the database to reflect your changes
poetry run flask db upgrade
# if you need help with the commands
poetry run flask db --help
```

The migrations are handled by [flask-migrate](https://flask-migrate.readthedocs.io/en/latest/index.html) which is based on [alembic](https://alembic.sqlalchemy.org/en/latest/index.html)

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

### Worker arguments

- `--pool=...`
  - for possible values see [celery docs](https://celery-safwan.readthedocs.io/en/latest/reference/cli.html#cmdoption-celery-worker-P)
  - don't use `solo` if you want multiple tasks to be able to be executed concurrently
- `--concurrency=...`
  - number of tasks that can be executed concurrently
- `--log_level=...`
  - see [Python docs](https://docs.python.org/3/howto/logging.html) for possible values
- `--periodic-scheduler`
  - add this flag to run the Celery beat scheduler alongside the worker
  - this is needed for periodic tasks
  - If a plugin is run by multiple workers, only one of these workers should start with a celery beat scheduler,
  otherwise the periodic tasks get scheduled by all of these schedulers and executed too many times.


## Compiling the Documentation

```bash
# compile documentation
poetry run invoke doc

# update source code documentation
poetry run invoke update-source-doc

# Open the documentation in the default browser
poetry run invoke browse-doc

# Find reference targets defined in the documentation
poetry run invoke doc-index --filter=searchtext

# export/update requirements.txt from poetry dependencies (for readthedocs build)
poetry run invoke update-dependencies
```

Update the python source documentation

```bash
poetry run sphinx-apidoc --separate --force -o docs/source . ./tasks.py docs plugins migrations
rm docs/source/modules.rst  # delete modules file as this repository only contains one module
```


## Updating the Third-Party Licenses

```bash
# list all licenses
poetry run invoke list-licenses

# update licenses in repository
poetry run invoke update-dependencies
```

The third party licenses will be stored in the `qhana_plugin_runner/templates/licenses.html` file.


## Unit Tests

The unit tests use [pytest](https://docs.pytest.org/en/latest/contents.html) and [hypothesis](https://hypothesis.readthedocs.io/en/latest/index.html).

```bash
# Run all unit tests
poetry run pytest

# Run failed tests
poetry run pytest --last-failed  # see also --stepwise

# Run new tests
poetry run pytest --new-first

# List all tests
poetry run pytest --collect-only

# Run with hypothesis explanations
poetry run pytest --hypothesis-explain

# Run with coverage
poetry run pytest -p pytest_cov --cov=qhana_plugin_runner

# With html report (open htmlcov/index.html in browser)
poetry run pytest -p pytest_cov --cov=qhana_plugin_runner --cov-report=html
```

## Docker

This repository contains a `Dockerfile` and a `docker-compose.yml` that can be used to run the project.
Pre-built Docker containers are also available via GitHub packages.


### Using the Docker Container

The docker container by default does not include any plugin.
There are three ways to include plugins:

 *  Build a new docker image with plugins.\
    Creating a new container with plugins and installing their dependencies at build time can improve the startup time of the container.
 *  Map in plugins with a bind mount.\
    Plugins can be mapped into the bare container from the local file system using a bind mount.
    This is useful for developing or testing plugins.
 *  Load plugins from a git repository.\
    The container can load plugins from git repositories specified in the `GIT_PLUGINS` environment variable.

    Specify a newline separated list of git repositories to load plugins from in the GIT_PLUGINS environment variable.
    Each line should contain a git URL following the same format as in requirements.txt used by pip.

    Examples:
    
    ```
    git+<<url to git repo>[@<branch/tag/commit hash>][#subdirectory=<directory in git repo holding the plugins>]
    git+https://github.com/UST-QuAntiL/qhana-plugin-runner.git@main#subdirectory=/plugins
    ```

The plugin runner is configured to look for plugins in the folders `/app/plugins/`, `/app/git-plugins/` and `/app/extra-plugins/`.
Plugins loaded from git will be placed in the `git-plugins` folder.
Plugins mapped into the container from outside should use the `extra plugins` folder and plugins included in the container build should go into `plugins`.

The container starts in server mode by default.
For background tasks of plugins to work a container with the exact same plugins configured needs to be started in worker mode by setting `CONTAINER_MODE=worker`.
Currently the local file system is used as result store by default, meaning that all server and worker containers using the same set of plugins should share one file system under `/app/instance/`.
This restriction also applies if the default sqlite database is used.

For communication between server and worker containers, a redis or amqp broker need to be setup.
Server and worker containers with the same plugin configuration need to use the same broker and queue name.
Server and worker containers that use different sets of plugins need to use different brokers or the same broker but different queue names.
The broker can be configured using the `BROKER_URL` and the `RESULT_BACKEND` environment variable.
The environment variable `CELERY_QUEUE` can be used to set the queue name.

The database to use can be configured using the `SQLALCHEMY_DATABASE_URI` environment variable.
SQLAlchemy is used which supports SQLite, Postgres and MariaDB/MySQL databases given that the [correct drivers](https://docs.sqlalchemy.org/en/14/core/engines.html#supported-databases) are installed.
Database drivers can be installed by using plugins that specify that driver as an install requirement.

The default file store can be configured with the `DEFAULT_FILE_STORE` environment variable.
This defaults to `local_filesystem`.

When a worker (or plugin in the worker) tries to generate a URL with `flask.url_for` and `_external=True`, it can fail with the error `Application was not able to create a URL adapter for request independent URL generation. You might be able to fix this by setting the SERVER_NAME config variable.`.
You can set the environment variable `SERVER_NAME` for the worker container and the value will be set in the flask configuration.

### Running the Plugin-Runner with Docker Compose

Start the docker compose with:

```
docker-compose up
```

Delete the containers with:

```
docker-compose down
```

To also delete the volume containing the output files add the flag `-v`.


## Acknowledgements

Current development is supported by the [Federal Ministry for Economic Affairs and Energy](http://www.bmwi.de/EN) as part of the [PlanQK](https://planqk.de) project (01MK20005N).

## Haftungsausschluss

Dies ist ein Forschungsprototyp.
Die Haftung für entgangenen Gewinn, Produktionsausfall, Betriebsunterbrechung, entgangene Nutzungen, Verlust von Daten und Informationen, Finanzierungsaufwendungen sowie sonstige Vermögens- und Folgeschäden ist, außer in Fällen von grober Fahrlässigkeit, Vorsatz und Personenschäden, ausgeschlossen.

## Disclaimer of Warranty

Unless required by applicable law or agreed to in writing, Licensor provides the Work (and each Contributor provides its Contributions) on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied, including, without limitation, any warranties or conditions of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR PURPOSE.
You are solely responsible for determining the appropriateness of using or redistributing the Work and assume any risks associated with Your exercise of permissions under this License.

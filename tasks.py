# Copyright 2021 QHAna plugin runner contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from os import environ
from pathlib import Path
from shlex import join
from typing import cast

from dotenv import load_dotenv, set_key, unset_key
from invoke import task
from invoke.context import Context
from invoke.runners import Result

load_dotenv(".flaskenv")
load_dotenv(".env")

MODULE_NAME = "qhana_plugin_runner"
CELERY_WORKER = f"{MODULE_NAME}.celery_worker:CELERY"


# a list of allowed licenses, dependencies with other licenses will trigger an error in the list-licenses command
ALLOWED_LICENSES = [
    "3-Clause BSD License",
    "Apache 2.0",
    "Apache License, Version 2.0",
    "Apache Software License",
    "BSD License",
    "BSD",
    "GNU Lesser General Public License v2 or later (LGPLv2+)",
    "GNU Library or Lesser General Public License (LGPL)",
    "GPLv3",
    "MIT License",
    "MIT",
    "Mozilla Public License 2.0 (MPL 2.0)",
    "new BSD",
    "Python Software Foundation License",
]


@task
def stop_broker(c):
    """Stop the previously started redis broker container with docker or podman.

    Discovers the container id from the environment variable REDIS_CONTAINER_ID.
    If the variable is not set ``--latest`` is used (this assumes that the latest
    created container is the broker!).

    To use podman instead of docker set the DOCKER_CMD environment variable to "podman".

    Args:
        c (Context): task context
    """
    c = cast(Context, c)
    docker_cmd = environ.get("DOCKER_CMD", "docker")
    container_id = environ.get("REDIS_CONTAINER_ID", "--latest")
    c.run(join([docker_cmd, "stop", container_id]))


@task(stop_broker)
def reset_broker(c):
    """Remove the current redis container and unset the REDIS_CONTAINER_ID variable.

    Discovers the container id from the environment variable REDIS_CONTAINER_ID.
    If the variable is not set this task does nothing.

    To use podman instead of docker set the DOCKER_CMD environment variable to "podman".

    Args:
        c (Context): task context
    """
    c = cast(Context, c)
    docker_cmd = environ.get("DOCKER_CMD", "docker")
    container_id = environ.get("REDIS_CONTAINER_ID")
    if not container_id:
        return
    c.run(join([docker_cmd, "rm", container_id]), echo=True)
    dot_env_path = Path(".env")
    unset_key(dot_env_path, "REDIS_CONTAINER_ID")


@task
def start_broker(c, port=None):
    """Start a redis broker container with docker or podman.

    Resuses an existing container if the environment variable REDIS_CONTAINER_ID is set.
    The reused container ignores the port option!
    Sets the environemnt variable in the .env file if a new container is created.

    Redis port is optionally read from REDIS_PORT environment variable. Use the
    ``reset-broker`` task to remove the old container to create a new container
    with a different port.

    To use podman instead of docker set the DOCKER_CMD environment variable to "podman".

    Args:
        c (Context): task context
        port (str, optional): outside port for connections to redis. Defaults to "6379".
    """
    c = cast(Context, c)
    docker_cmd = environ.get("DOCKER_CMD", "docker")
    container_id = environ.get("REDIS_CONTAINER_ID", None)

    if container_id:
        res: Result = c.run(join([docker_cmd, "restart", container_id]), echo=True)
        if res.failed:
            print(f"Failed to start container with id {container_id}.")
        return

    if not port:
        port = environ.get("REDIS_PORT", "6379")
    c.run(join([docker_cmd, "run", "-d", "-p", f"{port}:6379", "redis"]), echo=True)
    result: Result = c.run(join([docker_cmd, "ps", "-q", "--latest"]), hide=True)
    result_container_id = result.stdout.strip()
    dot_env_path = Path(".env")
    if not dot_env_path.exists():
        dot_env_path.touch()
    set_key(dot_env_path, "REDIS_CONTAINER_ID", result_container_id)


@task
def worker(c, pool="solo", concurrency=1, dev=False, loglevel="INFO"):
    """Run the celery worker, optionally starting the redis broker.

    Args:
        c (Context): task context
        pool (str, optional): the executor pool to use for celery workers (defaults to "solo" for development on linux and windows)
        concurrency (int, optional): the number of concurrent workers (defaults to 1 for development)
        dev (bool, optional): If true the redis docker container will be started before the worker and stopped after the workers finished. Defaults to False.
        loglevel (str, optional): The loglevel of the celery logger in the worker (DEBUG|INFO|WARNING|ERROR|CRITICAL|FATAL). Defaults to "INFO".
    """
    if dev:
        start_broker(c)
    c = cast(Context, c)
    cmd = [
        "celery",
        "--app",
        CELERY_WORKER,
        "worker",
        f"--pool={pool}",
        "--concurrency",
        str(concurrency),
        "--loglevel",
        loglevel,
    ]
    c.run(join(cmd), echo=True)
    if dev:
        stop_broker(c)


@task
def celery_status(c):
    """Show the status of celery workers.

    Args:
        c (Context): task context
    """
    c = cast(Context, c)
    c.run(
        join(["celery", "--app", CELERY_WORKER, "status"]),
        echo=True,
        hide="err",
        warn=True,
    )


@task
def celery_queues(c):
    """Show the queus of the celery instance.

    Args:
        c (Context): task context
    """
    c = cast(Context, c)
    c.run(
        join(["celery", "--app", CELERY_WORKER, "list", "bindings"]),
        echo=True,
        hide="err",
        warn=True,
    )


@task(celery_queues)
def celery_inspect(c):
    """Show a detailed status report of the running workers and queues.

    Args:
        c (Context): task context
    """
    c.run(
        join(["celery", "--app", CELERY_WORKER, "inspect", "report"]),
        echo=True,
        hide="err",
        warn=True,
    )
    c.run(
        join(["celery", "--app", CELERY_WORKER, "inspect", "stats"]),
        echo=True,
        hide="err",
        warn=True,
    )


@task
def celery_enabe_events(c):
    """Enable celery worker events events.

    Args:
        c (Context): task context
    """
    c.run(
        join(["celery", "--app", CELERY_WORKER, "control", "enable_events"]),
        echo=True,
        hide="err",
        warn=True,
    )


@task
def celery_disable_events(c):
    """Disable celery worker events events.

    Args:
        c (Context): task context
    """
    c.run(
        join(["celery", "--app", CELERY_WORKER, "control", "disable_events"]),
        echo=True,
        hide="err",
        warn=True,
    )


@task(pre=(celery_enabe_events,), post=(celery_disable_events,))
def celery_monitor(c):
    """Show current events.

    Args:
        c (Context): task context
    """
    c.run(
        join(["celery", "--app", CELERY_WORKER, "events"]),
        pty=True,
        hide="err",
        warn=True,
    )


@task
def purge_task_queues(c):
    """Purge all task queues. Deletes tasks forever!

    Args:
        c (Context): task context
    """
    answer = input(
        "This action cannot be undone. Type in 'purge' to purge all task queues:"
    )
    if answer != "purge":
        print("Not purging task queues.")
        return
    c.run(
        join(["celery", "--app", CELERY_WORKER, "purge"]),
        echo=True,
        hide="err",
        warn=True,
    )


@task
def list_licenses(
    c, format_="json", include_installed=False, summary=False, short=False, echo=False
):
    """List licenses of dependencies.

    By default only the direct (and transitive) dependencies of the plugin runner are included.

    Args:
        c (Context): task context
        format_ (str, optional): The output format (json, html, markdown, plain, plain-vertical, rst, confluence, json-license-finder, csv). Defaults to "json".
        include_installed (bool, optional): If true all currently installed packages are considered dependencies. Defaults to False.
        summary (bool, optional): If true output a summary of found licenses. Defaults to False.
        short (bool, optional): If true only name, version, license and authors of a apackage are printed. Defaults to False.
        echo (bool, optional): If true the command used to generate the license output is printed to console. Defaults to False.
    """
    packages = []
    if not include_installed:
        packages_output = c.run(
            join(["poetry", "export", "--dev", "--without-hashes"]),
            echo=False,
            hide="both",
        )
        packages = [p.split("=", 1)[0] for p in packages_output.stdout.splitlines() if p]
    cmd = [
        "pip-licenses",
        "--format",
        format_,
        "--with-authors",
        "--allow-only",
        ";".join(ALLOWED_LICENSES),
    ]
    if not short:
        cmd += [
            "--with-urls",
            "--with-description",
            "--with-license-file",
            "--no-license-path",
            "--with-notice-file",
        ]
    if summary:
        cmd.append("--summary")
    if not include_installed:
        cmd += [
            "--packages",
            *packages,
        ]
    c.run(
        join(cmd),
        echo=echo,
        warn=True,
    )


@task
def update_licenses(c, include_installed=False):
    """Update the licenses template to include all licenses.

    By default only the direct (and transitive) dependencies of the plugin runner are included.

    Args:
        c (Context): task context
        include_installed (bool, optional): Include all currently installed libraries. Defaults to False.
    """
    packages = []
    if not include_installed:
        packages_output = c.run(
            join(["poetry", "export", "--dev", "--without-hashes"]),
            echo=False,
            hide="both",
        )
        packages = [p.split("=", 1)[0] for p in packages_output.stdout.splitlines() if p]
    cmd = [
        "pip-licenses",
        "--format",
        "html",
        "--output-file",
        str(Path("./qhana_plugin_runner/templates/licenses.html")),
        "--with-authors",
        "--with-urls",
        "--with-description",
        "--with-license-file",
        "--no-license-path",
        "--with-notice-file",
        "--allow-only",
        ";".join(ALLOWED_LICENSES),
    ]
    if not include_installed:
        cmd += [
            "--packages",
            *packages,
        ]
    c.run(
        join(cmd),
        echo=True,
        hide="err",
        warn=True,
    )


@task(update_licenses)
def update_dependencies(c):
    """Update dependencies that are derived from the pyproject.toml dependencies (e.g. doc dependencies and licenses).

    Args:
        c (Context): task context
    """
    c.run(
        join(
            [
                "poetry",
                "export",
                "--dev",
                "--format",
                "requirements.txt",
                "--output",
                str(Path("./docs/requirements.txt")),
            ]
        ),
        echo=True,
        hide="err",
        warn=True,
    )

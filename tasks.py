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


@task
def stop_broker(c):
    """Stop the previously started redis broker container with docker or podman.

    Discovers the conteiner id from the environment variable REDIS_CONTAINER_ID.
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

    Discovers the conteiner id from the environment variable REDIS_CONTAINER_ID.
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
    c.run(join([docker_cmd, "rm", container_id]))
    dot_env_path = Path(".env")
    unset_key(dot_env_path, "REDIS_CONTAINER_ID")


@task
def start_broker(c, port=None):
    """Start a redis broker container with docker or podman.

    Resuses an existing container if the environment variable REDIS_CONTAINER_ID is set.
    The reused container ignores the port option!
    Sets the environemnt variable in the .env file if a new container is created.

    Redis port is optionally red from REDIS_PORT environment variable. Use the
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
        res: Result = c.run(join([docker_cmd, "restart", container_id]))
        if res.failed:
            print(f"Failed to start container with id {container_id}.")
        return

    if not port:
        port = environ.get("REDIS_PORT", "6379")
    c.run(join([docker_cmd, "run", "-d", "-p", f"{port}:6379", "redis"]))
    result: Result = c.run(join([docker_cmd, "ps", "-q", "--latest"]))
    result_container_id = result.stdout.strip()
    dot_env_path = Path(".env")
    if not dot_env_path.exists():
        dot_env_path.touch()
    set_key(dot_env_path, "REDIS_CONTAINER_ID", result_container_id)


@task
def worker(c, dev=False, loglevel="INFO"):
    """Run the celery worker, optionally starting the redis broker.

    Args:
        c (Context): task context
        dev (bool, optional): If true the redis docker container will be started before the worker and stopped after the workers finished. Defaults to False.
        loglevel (str, optional): The loglevel of the celery logger in the worker (DEBUG|INFO|WARNING|ERROR|CRITICAL|FATAL). Defaults to "INFO".
    """
    if dev:
        start_broker(c)
    c = cast(Context, c)
    c.run(join(["celery", "--app", CELERY_WORKER, "worker", "--loglevel", loglevel]))
    if dev:
        stop_broker(c)


@task
def celery_status(c):
    """Show the status of celery workers.

    Args:
        c (Context): task context
    """
    c.run(join(["celery", "--app", CELERY_WORKER, "status"]))

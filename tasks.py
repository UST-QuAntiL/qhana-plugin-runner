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
from os import execvpe as replace_process
from os import urandom
from pathlib import Path
from re import match
from shlex import join
from shutil import copytree
from typing import List, Optional, cast

from dotenv import load_dotenv, set_key, unset_key
from invoke import UnexpectedExit, call, task
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
def stop_camunda(c):
    """Stop the previously started camunda container with docker or podman.

    Discovers the container id from the environment variable CAMUNDA_CONTAINER_ID.
    If the variable is not set ``--latest`` is used (this assumes that the latest
    created container is the camunda container!).

    To use podman instead of docker set the DOCKER_CMD environment variable to "podman".

    Args:
        c (Context): task context
    """
    c = cast(Context, c)
    docker_cmd = environ.get("DOCKER_CMD", "docker")
    container_id = environ.get("CAMUNDA_CONTAINER_ID", "--latest")
    c.run(join([docker_cmd, "stop", container_id]))
    if docker_cmd == "podman":
        reset_camunda(c)


@task(stop_broker)
def reset_camunda(c):
    """Remove the current camunda container and unset the CAMUNDA_CONTAINER_ID variable.

    Discovers the container id from the environment variable CAMUNDA_CONTAINER_ID.
    If the variable is not set this task does nothing.

    To use podman instead of docker set the DOCKER_CMD environment variable to "podman".

    Args:
        c (Context): task context
    """
    c = cast(Context, c)
    docker_cmd = environ.get("DOCKER_CMD", "docker")
    container_id = environ.get("CAMUNDA_CONTAINER_ID")
    if not container_id:
        return
    c.run(join([docker_cmd, "rm", container_id]), echo=True, warn=True)
    dot_env_path = Path(".env")
    unset_key(dot_env_path, "CAMUNDA_CONTAINER_ID")


@task
def start_camunda(c, port=None):
    """Start the camunda container with docker or podman.

    Resuses an existing container if the environment variable CAMUNDA_CONTAINER_ID is set.
    The reused container ignores the port option!
    Sets the environemnt variable in the .env file if a new container is created.

    Camunda port is optionally read from CAMUNDA_PORT environment variable. Use the
    ``reset-camunda`` task to remove the old container to create a new container
    with a different port.

    To use podman instead of docker set the DOCKER_CMD environment variable to "podman".

    Args:
        c (Context): task context
        port (str, optional): outside port for connections to camunda. Defaults to "8080".
    """
    c = cast(Context, c)
    docker_cmd = environ.get("DOCKER_CMD", "docker")
    container_id = environ.get("CAMUNDA_CONTAINER_ID", None)

    if container_id and docker_cmd == "podman":
        reset_camunda(c)
        container_id = None
    if container_id:
        res: Result = c.run(join([docker_cmd, "restart", container_id]), echo=True)
        if res.failed:
            print(f"Failed to start container with id {container_id}.")
        return

    if not port:
        port = environ.get("CAMUNDA_PORT", "8080")
    c.run(
        join(
            [
                docker_cmd,
                "run",
                "-d",
                "-p",
                f"{port}:8080",
                "camunda/camunda-bpm-platform:run-latest",
            ]
        ),
        echo=True,
    )
    result: Result = c.run(join([docker_cmd, "ps", "-q", "--latest"]), hide=True)
    result_container_id = result.stdout.strip()
    dot_env_path = Path(".env")
    if not dot_env_path.exists():
        dot_env_path.touch()
    set_key(dot_env_path, "CAMUNDA_CONTAINER_ID", result_container_id)


@task(stop_broker, stop_camunda)
def stop_containers(c):
    """Stop both the camunda and the redis broker container."""
    pass


@task(start_broker, start_camunda)
def start_containers(c):
    """Start both the camunda and the redis broker container."""
    pass


@task
def worker(
    c, pool="solo", concurrency=1, dev=False, log_level="INFO", periodic_scheduler=False
):
    """Run the celery worker, optionally starting the redis broker.

    Args:
        c (Context): task context
        pool (str, optional): the executor pool to use for celery workers (defaults to "solo" for development on linux and windows)
        concurrency (int, optional): the number of concurrent workers (defaults to 1 for development)
        dev (bool, optional): If true the redis docker container will be started before the worker and stopped after the workers finished. Defaults to False.
        log_level (str, optional): The log level of the celery logger in the worker (DEBUG|INFO|WARNING|ERROR|CRITICAL|FATAL). Defaults to "INFO".
        periodic_scheduler (bool, optional): If true a celery beat scheduler will be started alongside the worker. This is needed for periodic tasks. Should only be set to True for one worker otherwise the periodic tasks get executed too often (see readme file).
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
        log_level.upper(),
        "-E",
    ]

    if periodic_scheduler:
        cmd += ["-B"]

    if dev:
        c.run(join(cmd), echo=True)
        stop_broker(c)
    else:
        # if not in dev mode completely replace the current process with the started process
        print(join(cmd))
        replace_process(cmd[0], cmd, environ)


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
def start_gunicorn(c, workers=1, log_level="info", docker=False):
    """Start the gunicorn server.

    This task is intended to be run in docker.
    The gunicorn server port defaults to 8080 but can be changed by setting
    the SERVER_PORT environment variable.

    Args:
        c (Context): task context
        workers (int, optional): The number of parallel workers (set this to around <nr_of_cores>*2 + 1). Defaults to 1.
        log_level (str, optional): the log level to output in console. Defaults to "info".
        docker (bool, optional): set this to True if running inside of docker. Defaults to false.
    """
    server_port: str = environ.get("SERVER_PORT", "8080")
    assert match(
        r"[1-9][0-9]*", server_port
    ), f"The given server port '{server_port}' does not have the right format! (must be a valid port number)"
    cmd = [
        "python",
        "-m",
        "gunicorn",
        "--pythonpath",
        ".",
        "--worker-tmp-dir",
        "/dev/shm" if docker else "/tmp",  # use in memory file system for heartbeats
        "-w",
        environ.get("GUNICORN_WORKERS", str(workers)),
        "-b",
        f"0.0.0.0:{server_port}",
        "--log-level",
        log_level.lower(),
        "--error-logfile=-",
        f"{MODULE_NAME}:create_app()",
    ]

    print(join(cmd))

    # replaces the current process with the subprocess!
    replace_process(cmd[0], cmd, environ)


def git_url_to_folder(url: str) -> str:
    """Extract a sensible and stable repository name from a git url"""
    # roughly matches …[/<organization]/<repository-name>[.git][/]
    url_match = match(r".*(?:\/(?P<orga>[^\/.]+))?\/(?P<repo>[^\/]+)(?:\.git)\/?$", url)
    if not url_match:
        raise ValueError(f"Url '{url}' could not be parsed!", url_match)
    if url_match["orga"]:
        return f"{url_match['orga']}__{url_match['repo']}"
    else:
        return url_match["repo"]


@task
def load_git_plugins(c, plugins_path="./git-plugins"):
    """Load plugins from git repositories (specified via GIT_PLUGINS env var).

    Specify a newline separated list of git repositories to load plugins from
    in the GIT_PLUGINS environment variable. Each line should contain a git
    URL following the same format as in requirements.txt used by pip.

    Examples:
    git+<<url to git repo>[@<branch/tag/commit hash>][#subdirectory=<directory in git repo holding the plugins>]
    git+https://github.com/UST-QuAntiL/qhana-plugin-runner.git@main#subdirectory=/plugins

    Args:
        c (Context): task context
        plugins_path (str, optional): the folder to load plugins into.
    """
    git_plugins = environ.get("GIT_PLUGINS")
    if not git_plugins:
        return

    repositories_path = Path(plugins_path) / Path(".repositories")

    if not repositories_path.exists():
        repositories_path.mkdir(parents=True, exist_ok=True)

    for git_plugin in git_plugins.splitlines():
        plugin_match = match(
            # roughly matches <vcs=git>+<repo_url>[@<ref>][#…subdirectory=<sub_dir>…]
            r"^(?P<vcs>git)\+(?P<url>[^@#\n]+)(?:@(?P<ref>[^#\n]+))?(?:#(?:[^&\n]+&)?subdirectory=\/?(?P<dir>[^&\n]+)[^\n]*)?$",
            git_plugin,
        )
        if not plugin_match:
            print(f"Could not recognise git url '{git_plugin}' – skipping")
            continue
        if plugin_match["vcs"] != "git":
            print(f"Only git is supported (got '{plugin_match['vcs']}') – skipping")
            continue
        url: str = plugin_match["url"]
        ref: Optional[str] = plugin_match["ref"]
        sub_dir: Optional[str] = plugin_match["dir"]

        cmd = [
            "git",
            "clone",
        ]

        shallow_cmd = [*cmd, "--depth=1"]

        if ref:
            shallow_cmd.append(f"--branch={ref}")

        folder = git_url_to_folder(url)
        if (Path(plugins_path) / Path(".repositories") / Path(folder)).exists():
            print(f"Repository '{url}' is already checked out – skipping")
            continue  # todo better handling for checked out repositories

        with c.cd(str(repositories_path)):
            try:
                # try a shallow clone (only branch and tag refs will work)
                c.run(join(shallow_cmd + [url, folder]))
            except UnexpectedExit:
                # fall back to full clone and checkout ref after cloning
                c.run(join(cmd + [url, folder]), warn=True)
                if ref:
                    with c.cd(folder):
                        c.run(join(["git", "checkout", ref]), warn=True)
            if sub_dir:
                plugin_folder = repositories_path / Path(folder) / Path(sub_dir)
            else:
                plugin_folder = repositories_path / Path(folder)

            if plugin_folder.exists() and plugin_folder.is_dir():
                # copy all files into the plugins directory
                copytree(plugin_folder, Path(plugins_path), dirs_exist_ok=True)


@task
def install_plugin_dependencies(c):
    """Install all plugin dependencies."""
    c.run(join(["python", "-m", "flask", "install"]), echo=True, warn=True)


@task
def await_db(c):
    """Docker specific task. Do not call."""
    c.run("/wait", echo=True, warn=False)


@task
def upgrade_db(c):
    """Upgrade the datzabase to the newest migration."""
    c.run(join(["python", "-m", "flask", "db", "upgrade"]), echo=True, warn=True)


@task
def ensure_paths(c):
    """Docker specific task. Do not call."""
    Path("/app/instance").mkdir(parents=True, exist_ok=True)


@task(ensure_paths)
def start_docker(c):
    """Docker entry point task. Do not call!"""

    def execute_pre_tasks(do_upgrade_db=False):
        for task in (load_git_plugins, install_plugin_dependencies, await_db):
            task(c)
        if do_upgrade_db:
            upgrade_db(c)

    if not environ.get("QHANA_SECRET_KEY"):
        environ["QHANA_SECRET_KEY"] = urandom(32).hex()

    log_level = environ.get("DEFAULT_LOG_LEVEL", "INFO")
    concurrency_env = environ.get("CONCURRENCY", "1")
    concurrency = int(concurrency_env) if concurrency_env.isdigit() else 1
    if environ.get("CONTAINER_MODE", "").lower() == "server":
        execute_pre_tasks(do_upgrade_db=True)
        start_gunicorn(c, workers=concurrency, log_level=log_level, docker=True)
    elif environ.get("CONTAINER_MODE", "").lower() == "worker":
        execute_pre_tasks()
        worker_pool = environ.get("CELERY_WORKER_POOL", "threads")
        periodic_scheduler = bool(environ.get("PERIODIC_SCHEDULER", False))
        worker(
            c,
            concurrency=concurrency,
            pool=worker_pool,
            log_level=log_level,
            periodic_scheduler=periodic_scheduler,
        )
    else:
        raise ValueError(
            "Environment variable 'CONTAINER_MODE' must be set to either 'server' or 'worker'!"
        )


@task
def doc(c, format_="html", all_=False, color=True):
    """Build the documentation.

    Args:
        c (Context): task context
        format_ (str, optional): the format to build. Defaults to "html".
        all (bool, optional): build all files new. Defaults to False.
        color (bool, optional): color output. Defaults to True.
    """
    cmd = ["sphinx-build", "-b", format_]
    if all_:
        cmd.append("-a")
    if color:
        cmd.append("--color")
    else:
        cmd.append("--no-color")
    cmd += [".", "_build"]
    with c.cd(str(Path("./docs"))):
        c.run(join(cmd), echo=True)


@task
def update_source_doc(c):
    """Update the autogenerated source documentation files.

    Args:
        c (Context): task context
    """
    cmd = [
        "sphinx-apidoc",
        "--separate",
        "--force",
        "-o",
        "docs/source",
        ".",
        "./tasks.py",
        "docs",
        "plugins",  # TODO exclude all known plugin folders!
        "migrations",
    ]

    c.run(join(cmd), echo=True)

    # remove unwanted files
    for p in (
        Path("docs/source/modules.rst"),
        Path("docs/source/qhana_plugin_runner.celery_worker.rst"),
    ):
        if p.exists():
            p.unlink()


@task
def browse_doc(c):
    """Open the documentation in the browser.

    Args:
        c (Context): task context
    """
    index_path = Path("./docs/_build/html/index.html")
    if not index_path.exists():
        doc(c)

    print(f"Open: file://{index_path.resolve()}")
    import webbrowser

    webbrowser.open_new_tab(str(index_path.resolve()))


@task
def doc_index(c, filter_=""):
    """Search the index of referencable sphinx targets in the documentation.

    Args:
        c (Context): task context
        filter_ (str, optional): an optional filter string. Defaults to "".
    """
    inv_path = Path("./docs/_build/html/objects.inv")
    if not inv_path.exists():
        doc(c)

    if filter_:
        filter_ = filter_.lower()

    with c.cd(str(Path("./docs"))):
        output: Result = c.run(
            join(["python", "-m", "sphinx.ext.intersphinx", "_build/html/objects.inv"]),
            echo=True,
            hide="stdout",
        )
        print(
            "".join(
                l
                for l in output.stdout.splitlines(True)
                if (l and not l[0].isspace()) or (not filter_) or (filter_ in l.lower())
            ),
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
    packages: List[str] = []
    if not include_installed:
        packages_output: Result = c.run(
            join(["poetry", "export", "--dev", "--without-hashes"]),
            echo=False,
            hide="both",
        )
        packages = [p.split("=", 1)[0] for p in packages_output.stdout.splitlines() if p]
    cmd: List[str] = [
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
    packages: List[str] = []
    if not include_installed:
        packages_output: Result = c.run(
            join(["poetry", "export", "--dev", "--without-hashes"]),
            echo=False,
            hide="both",
        )
        packages = [p.split("=", 1)[0] for p in packages_output.stdout.splitlines() if p]
    cmd: List[str] = [
        "pip-licenses",
        "--format",
        "html",
        "--output-file",
        str((Path(".") / Path(MODULE_NAME) / Path("templates/licenses.html")).resolve()),
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
                "--without-hashes",  # with hashes fails because pip is to strict with transitive dependencies
                "--output",
                str(Path("./docs/requirements.txt")),
            ]
        ),
        echo=True,
        hide="err",
        warn=True,
    )

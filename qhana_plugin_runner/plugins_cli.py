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

"""CLI functions for plugins."""

import sys
from io import TextIOWrapper
from pathlib import Path
from subprocess import CalledProcessError, run
from tempfile import TemporaryDirectory
from typing import cast

import click
from flask import Blueprint, Flask, current_app
from flask.cli import AppGroup, with_appcontext

from .util.logging import get_logger
from .util.plugins import QHAnaPluginBase

PLUGIN_CLI_BLP = Blueprint("plugins_cli", __name__, cli_group=None)
PLUGIN_CLI = cast(
    AppGroup, PLUGIN_CLI_BLP.cli
)  # expose as attribute for autodoc generation

PLUGIN_COMMAND_LOGGER = "plugins"


@click.option(
    "--dry-run",
    "-d",
    default=False,
    is_flag=True,
    help="Does not install the requirements but prints them to stdout.",
)
@PLUGIN_CLI.command("install")
@with_appcontext
def install_plugin_dependencies(dry_run):
    """Gather and install all plugin dependencies."""
    with TemporaryDirectory(prefix="qhana") as temp_dir:
        temp_dir = Path(temp_dir)
        requirements_file = temp_dir / Path("requirements.txt")
        with requirements_file.open(mode="w") as requirements:
            append_runner_dependencies(current_app, requirements)
            for plugin in QHAnaPluginBase.get_plugins().values():
                append_plugin_dependencies(plugin, current_app, requirements)
        if dry_run:
            with requirements_file.open() as r:
                click.echo("\n\n")
                for line in r:
                    click.echo(line, nl=False)
            return
        click.echo("Finished gathering requirements, installing requirements.")
        try:
            run(
                [
                    "python",
                    "-m",
                    "pip",
                    "install",
                    "--requirement",
                    str(requirements_file),
                ],
                check=True,
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
        except CalledProcessError:
            click.echo("Installing plugin requirements failed!")
            return
        click.echo("Successfully installed all plugin requirements.")


def append_runner_dependencies(app: Flask, requirements: TextIOWrapper):
    """Append the current plugin runner dependencies to the requirements file by exporting them from poetry."""
    get_logger(app, PLUGIN_COMMAND_LOGGER).info(
        "Gathering QHAna plugin runner dependencies."
    )
    requirements.write("# qhana_plugin_runner requirements\n")
    requirements.flush()
    run(["poetry", "export"], check=True, stdout=requirements)
    requirements.write("\n\n")


def append_plugin_dependencies(
    plugin: QHAnaPluginBase, app: Flask, requirements: TextIOWrapper
):
    """Append the plugin dependencies to the requirement file if the plugin reports its dependencies."""
    logger = get_logger(app, PLUGIN_COMMAND_LOGGER)
    logger.info(f"Gathering dependencies for plugin {plugin.name} @{plugin.version}.")
    try:
        dependencies = plugin.get_requirements()
        requirements.write(
            f"# {plugin.name}@{plugin.version}\n{dependencies.strip()}\n\n\n"
        )
    except NotImplementedError:
        logger.info("Plugin {plugin.name} @{plugin.version} defines no dependencies.")


def register_plugin_cli_blueprint(app: Flask):
    """Method to register the plugins CLI blueprint."""
    app.register_blueprint(PLUGIN_CLI_BLP)
    app.logger.info("Registered plugin cli blueprint.")

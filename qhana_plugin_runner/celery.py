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

from logging import Logger

from celery import Celery
from celery.platforms import IS_WINDOWS
from celery.signals import after_setup_task_logger
from flask.app import Flask

if IS_WINDOWS:
    # celery does not support windows, but should work with this fix applied
    from os import environ

    # solution taken from https://stackoverflow.com/questions/37255548/how-to-run-celery-on-windows
    environ.setdefault("FORKED_BY_MULTIPROCESSING", "1")


def _configure_celery_logger(logger: Logger, **kwargs):
    """Ensure that the logger propagates logs."""
    logger.propagate = True


after_setup_task_logger.connect(_configure_celery_logger)


CELERY = Celery(__name__)


def register_celery(app: Flask):
    """Load the celery config from the app instance."""
    CELERY.conf.update(app.config.get("CELERY", {}))
    app.logger.info(
        f"Celery settings:\n{CELERY.conf.humanize(with_defaults=False, censored=True)}\n"
    )

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

from celery import Celery, Task
from celery.signals import after_setup_task_logger
from flask import globals as flask_flobals
from flask.app import Flask


def _configure_celery_logger(logger: Logger, **kwargs):
    """Ensure that the logger propagates logs."""
    logger.propagate = True


after_setup_task_logger.connect(_configure_celery_logger)


class FlaskTask(Task):
    """Flask app context aware task base class."""

    def __call__(self, *args, **kwargs):
        """Execute task with an app context."""
        if flask_flobals._app_ctx_stack.top is not None:
            # app context already established
            return self.run(*args, **kwargs)
        with self.app.flask_app.app_context():
            # run task with app context
            return self.run(*args, **kwargs)


CELERY = Celery(__name__, flask_app=None, task_cls=FlaskTask)


def register_celery(app: Flask):
    """Load the celery config from the app instance."""
    CELERY.conf.update(
        app.config.get("CELERY", {}),
        beat_schedule={},
    )
    CELERY.flask_app = app  # set flask_app attribute used by FlaskTask
    app.logger.info(
        f"Celery settings:\n{CELERY.conf.humanize(with_defaults=False, censored=True)}\n"
    )

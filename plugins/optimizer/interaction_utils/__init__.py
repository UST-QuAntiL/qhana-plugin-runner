from plugins.optimizer.interaction_utils.signal_handlers import (
    task_status_changed_handler,
)
from qhana_plugin_runner.tasks import TASK_STATUS_CHANGED
import logging

# connect signal handlers
TASK_STATUS_CHANGED.connect(task_status_changed_handler)

# Create or get a logger
BENCHMARK_LOGGER = logging.getLogger("benchmark")

# Set log level
BENCHMARK_LOGGER.setLevel(logging.INFO)

# Create a file handler and set the level to log info
file_handler = logging.FileHandler("benchmark.log")
file_handler.setLevel(logging.INFO)

# Create a formatter
formatter = logging.Formatter("%(asctime)s - %(message)s")
file_handler.setFormatter(formatter)

# Add the handler to the logger
BENCHMARK_LOGGER.addHandler(file_handler)

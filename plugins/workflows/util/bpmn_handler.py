import logging

logger = logging.getLogger(__name__)


class BpmnHandler:
    def __init__(self, filename):
        self.filename = filename
        self.bpmn = self.load()

    def load(self):
        """
        Loads the BPMN model at the specified location
        :return:
        """
        return open(
            "plugins/workflows/bpmn/" + self.filename, "rb"
        )  # FIXME use instance relative locations (or code relative locations)!

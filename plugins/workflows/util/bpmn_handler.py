import logging

logger = logging.getLogger(__name__)


class BpmnHandler:
    def __init__(self, filename):
        self.filename = filename
        self.bpmn = None
        self.load()

    def load(self):
        """
        Loads the BPMN model at the specified location
        :return:
        """
        self.bpmn = open("plugins/workflows/bpmn_models/" + self.filename, 'rb')

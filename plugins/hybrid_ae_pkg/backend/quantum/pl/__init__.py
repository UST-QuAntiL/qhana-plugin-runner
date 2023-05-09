from enum import Enum


class QNNEnum(Enum):
    qnn1 = "QNN1"
    qnn2 = "QNN2"
    qnn3 = "QNN3"

    def get_qnn(self) -> str:
        return str(self.value)

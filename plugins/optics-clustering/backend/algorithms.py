from enum import Enum


class AlgorithmEnum(Enum):
    auto = "Auto"
    ball_tree = "Ball Tree"
    kd_tree = "KD Tree"
    brute = "Brute"

    def get_algorithm(self):
        return str(self.name)

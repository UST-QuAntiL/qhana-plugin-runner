from enum import Enum


class InitEnum(Enum):
    random = "Random"
    heuristic = "Heuristic"
    k_medoids_pp = "k-Medoids++"
    build = "Build"

    def get_init(self):
        if self == InitEnum.random:
            return "random"
        elif self == InitEnum.heuristic:
            return "heuristic"
        elif self == InitEnum.k_medoids_pp:
            return "k-medoids++"
        elif self == InitEnum.build:
            return "build"

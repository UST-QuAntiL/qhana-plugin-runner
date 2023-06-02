from enum import Enum


class MethodEnum(Enum):
    xi = "xi"
    dbscan = "dbscan"

    def get_method(self) -> str:
        return str(self.name)

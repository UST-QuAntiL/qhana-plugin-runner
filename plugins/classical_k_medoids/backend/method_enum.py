from enum import Enum


class MethodEnum(Enum):
    alternate = "Alternate"
    pam = "Pam"

    def get_method(self):
        return str(self.name)

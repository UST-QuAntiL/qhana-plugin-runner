from enum import Enum


class EntanglementPatternEnum(Enum):
    linear = "Linear"
    circular = "Circular"
    full = "Full"

    def get_entanglement_pattern(self) -> str:
        return str(self.name)

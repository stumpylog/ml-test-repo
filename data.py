import enum
from dataclasses import dataclass


class Tag(enum.IntEnum):
    ComputerScience = 1
    Physics = 2
    Mathematics = 3
    Statistics = 4
    QuantBiology = 5
    QuantFinance = 6

    def __repr__(self):
        return str(self.name)


@dataclass(frozen=True)
class Paper:
    title: str
    abstract: str
    tags: list[Tag]

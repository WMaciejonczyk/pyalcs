from __future__ import annotations

from lcs import TypedList
from lcs.agents.acs2er.ReplayMemorySample import ReplayMemorySample


class ReplayMemory(TypedList):
    """
    Represents the replay memory buffer
    """

    def __init__(self, *args, max_size: int) -> None:
        super().__init__(*args, oktypes=(tuple,))
        self.max_size = max_size

    def update(self, trajectory) -> None:
        if len(self) >= self.max_size:
            self.pop(0)

        self.append(trajectory)

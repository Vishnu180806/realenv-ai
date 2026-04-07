from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple
from env.models import AgentAction, EnvironmentState

class BaseGrader(ABC):
    @abstractmethod
    def grade_step(
        self, action: AgentAction, state: EnvironmentState
    ) -> Tuple[float, str, bool]:
        """
        Returns: (reward, feedback, done)
        """
        pass

    @abstractmethod
    def final_score(self, state: EnvironmentState) -> float:
        """
        Returns final episode score [0, 1].
        """
        pass

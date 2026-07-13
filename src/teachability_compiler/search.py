"""Search interfaces for curriculum planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt

from .state import CurriculumAction, LearningState


@dataclass(slots=True)
class EdgeStatistics:
    visits: int = 0
    value_sum: float = 0.0
    value_square_sum: float = 0.0
    prior: float = 0.0
    information_bonus: float = 0.0
    risk: float = 0.0

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass(slots=True)
class SearchNode:
    state: LearningState
    remaining_token_budget: int
    visits: int = 0
    edges: dict[CurriculumAction, EdgeStatistics] = field(default_factory=dict)


def puct_score(
    node: SearchNode,
    stats: EdgeStatistics,
    *,
    c_puct: float,
    uncertainty_weight: float,
    risk_weight: float,
) -> float:
    """Risk- and information-aware PUCT score."""
    exploration = c_puct * stats.prior * sqrt(max(node.visits, 1)) / (1 + stats.visits)
    return (
        stats.mean_value
        + exploration
        + uncertainty_weight * stats.information_bonus
        - risk_weight * stats.risk
    )

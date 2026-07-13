import pytest

from teachability_compiler.state import CurriculumAction


def test_action_requires_normalized_mixture() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        CurriculumAction(
            cluster_ids=("a", "b"),
            mixture_weights=(0.2, 0.2),
            optimizer_steps=1,
            token_budget=128,
        )

"""Greedy and MCTS curriculum policies."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from .search import EdgeStatistics, SearchNode, puct_score
from .simulator import TransitionSimulator
from .state import CurriculumAction, LearningState

ValueFunction = Callable[[LearningState], float]


def plan_greedy(
    simulator: TransitionSimulator,
    initial_state: LearningState,
    action_space: Sequence[CurriculumAction],
    horizon: int,
    value_fn: ValueFunction,
) -> list[CurriculumAction]:
    """Plan by repeatedly taking the action with best predicted one-step value."""

    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    if not action_space:
        raise ValueError("action_space cannot be empty")

    state = initial_state
    plan: list[CurriculumAction] = []
    for _ in range(horizon):
        action = _best_one_step_action(simulator, state, action_space, value_fn)
        plan.append(action)
        state = simulator.predict(state, action).next_state_mean
    return plan


def plan_mcts(
    simulator: TransitionSimulator,
    initial_state: LearningState,
    action_space: Sequence[CurriculumAction],
    horizon: int,
    value_fn: ValueFunction,
    *,
    simulations_per_step: int = 160,
    c_puct: float = 1.5,
    uncertainty_weight: float = 0.05,
    risk_weight: float = 0.5,
) -> list[CurriculumAction]:
    """Plan a curriculum with PUCT Monte Carlo tree search through a simulator."""

    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    if simulations_per_step <= 0:
        raise ValueError("simulations_per_step must be positive")
    if not action_space:
        raise ValueError("action_space cannot be empty")

    state = initial_state
    plan: list[CurriculumAction] = []

    for step_index in range(horizon):
        depth_remaining = horizon - step_index
        root = SearchNode(
            state=state,
            remaining_token_budget=_remaining_budget(action_space, depth_remaining),
        )
        children: dict[tuple[int, CurriculumAction], SearchNode] = {}

        for _ in range(simulations_per_step):
            _simulate(
                simulator=simulator,
                node=root,
                action_space=action_space,
                depth_remaining=depth_remaining,
                value_fn=value_fn,
                children=children,
                c_puct=c_puct,
                uncertainty_weight=uncertainty_weight,
                risk_weight=risk_weight,
            )

        chosen_action = _best_root_action(root, action_space)
        plan.append(chosen_action)
        state = simulator.predict(state, chosen_action).next_state_mean

    return plan


def _simulate(
    *,
    simulator: TransitionSimulator,
    node: SearchNode,
    action_space: Sequence[CurriculumAction],
    depth_remaining: int,
    value_fn: ValueFunction,
    children: dict[tuple[int, CurriculumAction], SearchNode],
    c_puct: float,
    uncertainty_weight: float,
    risk_weight: float,
) -> float:
    node.visits += 1
    if depth_remaining <= 0:
        return value_fn(node.state)

    if not node.edges:
        _expand_edges(node, action_space)

    action = _select_action(
        node,
        action_space,
        c_puct=c_puct,
        uncertainty_weight=uncertainty_weight,
        risk_weight=risk_weight,
    )
    stats = node.edges[action]
    prediction = simulator.predict(node.state, action)
    stats.information_bonus = prediction.novelty
    stats.risk = prediction.forgetting_risk

    key = (id(node), action)
    child = children.get(key)
    if child is None:
        child = SearchNode(
            state=prediction.next_state_mean,
            remaining_token_budget=max(
                node.remaining_token_budget - action.token_budget,
                0,
            ),
        )
        children[key] = child

    if stats.visits == 0:
        value = _greedy_rollout_value(
            simulator,
            prediction.next_state_mean,
            action_space,
            depth_remaining - 1,
            value_fn,
        )
    else:
        value = _simulate(
            simulator=simulator,
            node=child,
            action_space=action_space,
            depth_remaining=depth_remaining - 1,
            value_fn=value_fn,
            children=children,
            c_puct=c_puct,
            uncertainty_weight=uncertainty_weight,
            risk_weight=risk_weight,
        )

    stats.visits += 1
    stats.value_sum += value
    stats.value_square_sum += value * value
    return value


def _expand_edges(
    node: SearchNode,
    action_space: Sequence[CurriculumAction],
) -> None:
    prior = 1.0 / len(action_space)
    for action in action_space:
        node.edges[action] = EdgeStatistics(prior=prior)


def _select_action(
    node: SearchNode,
    action_space: Sequence[CurriculumAction],
    *,
    c_puct: float,
    uncertainty_weight: float,
    risk_weight: float,
) -> CurriculumAction:
    unvisited = [action for action in action_space if node.edges[action].visits == 0]
    if unvisited:
        return unvisited[0]

    return max(
        action_space,
        key=lambda action: (
            puct_score(
                node,
                node.edges[action],
                c_puct=c_puct,
                uncertainty_weight=uncertainty_weight,
                risk_weight=risk_weight,
            ),
            action.cluster_ids[0],
        ),
    )


def _best_root_action(
    root: SearchNode,
    action_space: Sequence[CurriculumAction],
) -> CurriculumAction:
    if not root.edges:
        raise RuntimeError("root was not searched")
    return max(
        action_space,
        key=lambda action: (
            root.edges[action].mean_value,
            root.edges[action].visits,
            action.cluster_ids[0],
        ),
    )


def _greedy_rollout_value(
    simulator: TransitionSimulator,
    state: LearningState,
    action_space: Sequence[CurriculumAction],
    depth_remaining: int,
    value_fn: ValueFunction,
) -> float:
    rollout_state = state
    for _ in range(max(depth_remaining, 0)):
        action = _best_one_step_action(simulator, rollout_state, action_space, value_fn)
        rollout_state = simulator.predict(rollout_state, action).next_state_mean
    return value_fn(rollout_state)


def _best_one_step_action(
    simulator: TransitionSimulator,
    state: LearningState,
    action_space: Sequence[CurriculumAction],
    value_fn: ValueFunction,
) -> CurriculumAction:
    return max(
        action_space,
        key=lambda action: (
            value_fn(simulator.predict(state, action).next_state_mean),
            action.cluster_ids[0],
        ),
    )


def _remaining_budget(
    action_space: Sequence[CurriculumAction],
    depth_remaining: int,
) -> int:
    max_token_budget = max(action.token_budget for action in action_space)
    return depth_remaining * max_token_budget

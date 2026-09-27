import copy
from sam_ai.benchmarks.arc_agi3_agent import (
    ARC3InteractiveEnvironment,
    ARC3InteractiveAgent,
    ARC3Action,
    ActionType,
)


def test_arc3_env_mechanics():
    # 3x3 grid with empty cells, player at (0, 0), goal at (2, 2)
    grid = [
        [8, 0, 0],
        [0, 0, 0],
        [0, 0, 3],
    ]
    # Goal: reach bottom-right corner
    env = ARC3InteractiveEnvironment(grid, goal_condition=lambda g: g[2][2] == 8)
    obs = env.reset()

    assert obs.step_num == 0
    assert obs.grid[0][0] == 8
    assert not obs.done

    # Test CLICK action
    obs = env.step(ARC3Action(ActionType.CLICK, x=1, y=1, color=5))
    assert obs.grid[1][1] == 5
    assert obs.step_num == 1


def test_arc3_agent_interactive_solving():
    # 4x4 grid: Player (8) at (0, 0), Target (3) at (2, 0)
    grid = [
        [8, 0, 0, 0],
        [0, 0, 0, 0],
        [3, 0, 0, 0],
        [0, 0, 0, 0],
    ]

    # Goal is player reaches row 2, col 0
    env = ARC3InteractiveEnvironment(grid, goal_condition=lambda g: g[2][0] == 8)
    agent = ARC3InteractiveAgent(action_budget=10)

    result = env.step(ARC3Action(ActionType.MOVE_DOWN)) # moves to (1, 0)
    assert result.grid[1][0] == 8
    assert result.grid[0][0] == 0

    # Test autonomous solving
    episode_result = agent.run_episode(env)
    assert episode_result["won"] is True
    assert episode_result["steps_taken"] <= 5
    assert episode_result["final_reward"] == 1.0

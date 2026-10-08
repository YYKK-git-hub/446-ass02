"""
================================================================================
BFS FOR VACUUM CLEANER — QUICK START GUIDE
================================================================================

Description of Implementation

1. Define Classes:
   - Percept: Represents the agent's perception of the environment.
   - Action: Represents the action taken by the agent.
   - Sensor: Abstract base class for sensors.
   - FullSensor: A sensor that can see the entire environment.
   - Agent: Abstract base class for agents.
   - RandomAgent: An agent that chooses actions randomly.
   - SearchNode: Represents a node in the search tree.
   - VacuumSearchProblem: Defines the search problem.
   - BFSAgent: An agent that uses Breadth-First Search to plan actions.

2. Implement Methods:
   - Percept Class:
     - __init__: Initializes the percept with position, status, bump, and visible cells.
     - __repr__: Returns a string representation of the percept.
   - Action Class:
     - VALID_MOVES: A set of valid moves.
     - __init__: Initializes the action with a clean flag and move.
     - __repr__: Returns a string representation of the action.
   - Sensor Class:
     - Abstract method read that must be implemented by subclasses.
   - FullSensor Class:
     - Implements the read method to return a percept with the full state of the environment.
   - Agent Class:
     - Abstract method decide that must be implemented by subclasses.
   - RandomAgent Class:
     - Implements the decide method to choose a random action.
   - SearchNode Class:
     - __init__: Initializes the search node with state, parent, action, and path cost.
     - get_path: Returns the path from the root to the current node.
   - VacuumSearchProblem Class:
     - __init__: Initializes the search problem with initial position, dirty cells, and grid size.
     - is_goal: Checks if the current state is the goal state.
     - get_actions: Returns a list of possible actions from the current state.
     - transition_model: Returns the next state after taking an action.
   - BFSAgent Class:
     - Implements the _compute_plan method to compute the plan using BFS.
     - Implements the decide method to use the computed plan to decide the next action.

3. Test the Implementation:
   - Create an instance of VacuumSearchProblem with the initial state.
   - Use an instance of BFSAgent to compute the plan and verify that it correctly solves the problem.
   - Compare the results with RandomAgent to see the difference in performance.

Example Usage

# Create an instance of VacuumSearchProblem
initial_position = (0, 0)
dirty_cells = {(1, 1), (2, 2)}
problem = VacuumSearchProblem(initial_position, dirty_cells)

# Create an instance of BFSAgent
bfs_agent = BFSAgent()

# Compute the plan
plan = bfs_agent._compute_plan(problem)
print("Plan:", plan)

# Decide the next action
percept = FullSensor().read(problem)  # Assuming problem has a method to simulate the environment
action = bfs_agent.decide(percept)
print("Action:", action)

"""

from abc import ABC, abstractmethod
from collections import deque
from typing import List, Tuple, Set, Dict, Any, FrozenSet, Optional
import random

# Type aliases
Position = Tuple[int, int]
SearchState = Tuple[Position, FrozenSet[Position]]

# ==============================================================================
# 1. INTERFACE DEFINITIONS (DO NOT MODIFY)
# ==============================================================================

class Percept:
    def __init__(self, position: Position, status: str, bump: Optional[bool] = None, visible_cells: Optional[Dict[Position, str]] = None):
        self.position = position
        self.status = status
        self.bump = bump
        self.visible_cells = visible_cells

    def __repr__(self) -> str:
        return f"Percept(pos={self.position}, status={self.status}, dirty_count={sum(1 for s in self.visible_cells.values() if s == 'Dirty') if self.visible_cells else 0})"


class Action:
    VALID_MOVES = {"Up", "Down", "Left", "Right", "NoOp", "Suck"}

    def __init__(self, clean: bool, move: str):
        if move not in self.VALID_MOVES:
            raise ValueError(f"Invalid move: '{move}'")
        self.clean = clean
        self.move = move

    def __repr__(self) -> str:
        return f"Action(clean={self.clean}, move={self.move})"


class Sensor(ABC):
    @abstractmethod
    def read(self, env: 'Any') -> Percept:
        pass


class FullSensor(Sensor):
    def read(self, env: 'Any') -> Percept:
        return Percept(
            position=env.agent_position(),
            status=env.cell_status(env.agent_position()),
            visible_cells=env.all_cells()
        )


class Agent(ABC):
    def __init__(self, sensor: Sensor):
        self.sensor = sensor

    @abstractmethod
    def decide(self, percept: Percept) -> Action:
        pass


class RandomAgent(Agent):
    def __init__(self):
        super().__init__(sensor=FullSensor())

    def decide(self, percept: Percept) -> Action:
        clean = (percept.status == "Dirty")
        available_moves = Action.VALID_MOVES if clean else Action.VALID_MOVES - {"Suck"}
        move = random.choice(list(available_moves))
        return Action(clean=clean, move=move)


# ==============================================================================
# 2. STUDENT IMPLEMENTATION SECTION
# ==============================================================================

class SearchNode:
    """Represents a node in the search tree."""
    def __init__(self, state: SearchState, parent: Optional['SearchNode'] = None, action: Optional[str] = None, path_cost: int = 0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost

    def get_path(self) -> List[str]:
        """Reconstructs the action path from the root to this node."""
        actions = []  # create a list for the actions
        node = self # create a node to trace back where it was at

        while node.parent is not None:  # just in case that the node is the starting point
            actions.append(node.action) # save the action
            node = node.parent  # reverse back to the parent node

        actions.reverse()   # reverse it to get the correct sequence
        return actions  # return what we've done

class VacuumSearchProblem:
    """Defines the state space and rules for the Vacuum Cleaner environment."""
    def __init__(self, initial_position: Position, dirty_cells: Set[Position], grid_size: int = 5):
        self.grid_size = grid_size
        self.initial: SearchState = (initial_position, frozenset(dirty_cells))

    def is_goal(self, state: SearchState) -> bool:
        """Returns True if the given state satisfies the goal condition."""
        position, dirty_cells = state
        return len(dirty_cells) == 0    # To check if all the dirty cells have been cleaned up

    def get_actions(self, state: SearchState) -> List[str]:
        """Returns valid actions from the given state without stepping out of grid bounds."""

        (x, y), dirty_cells = state         # deliver current state
        actions = []                        # a list to store the actions

        if y > 0:                            # not at the top, allow the vacuum to move up
            actions.append("Up")

        if y < self.grid_size - 1:           # not at the bottom, allow the vacuum to move down
            actions.append("Down")

        if x > 0:                            # not at the leftest, allow the vacuum to move right
            actions.append("Left")

        if x < self.grid_size - 1:           # not at the rightest, allow the vacuum to move left
            actions.append("Right")

        if (x, y) in dirty_cells:            # return "suck" as the vacuum will clean the dirty cell
            actions.append("Suck")

        return actions

    def transition_model(self, state: SearchState, action: str) -> SearchState:
        """Applies an action to a state and returns the resulting next state."""

        (x, y), dirty_cells = state          # deliver current state

        if action == "Up":                   # Execute the corresponding action
            y = y - 1
        elif action == "Down":
            y = y + 1
        elif action == "Left":
            x = x - 1
        elif action == "Right":
            x = x + 1
        elif action == "Suck":
            dirty_cells = dirty_cells - frozenset({(x, y)})     # delete current position after cleaning

        return ((x, y), dirty_cells)        # return a new state


class BFSAgent(Agent):
    """An agent that uses Breadth-First Search to find the optimal plan."""
    def __init__(self):
        super().__init__(sensor=FullSensor())

    def _compute_plan(self, problem: VacuumSearchProblem) -> List[str]:
        """Performs Breadth-First Search on the problem and returns a list of action strings."""
        # TODO: Implement BFS algorithm using deque for frontier and a set for reached states
        # # some code
        # ###*********************************************************
        # I actually don't know how to do this part, I don't know what this algorithm is.
        # After asking AI, AI helped me on this, and this is what it should be:
        # ###********************************************************

        # root = SearchNode(problem.initial)
        # frontier = deque([root])
        # reached = {root.state}
        #
        # while frontier:
        #     node = frontier.popleft()
        #
        #     if problem.is_goal(node.state):
        #         return node.get_path()
        #
        #     for action in problem.get_actions(node.state):
        #         next_state = problem.transition_model(node.state, action)
        #
        #         if next_state not in reached:
        #             reached.add(next_state)
        #
        #             child = SearchNode(
        #                 state=next_state,
        #                 parent=node,
        #                 action=action,
        #                 path_cost=node.path_cost + 1
        #             )
        #
        #             frontier.append(child)
        #
        # return []

        # ###***************************************************
        raise NotImplementedError("Implement BFSAgent._compute_plan()")

    def decide(self, percept: Percept) -> Action:
        """Parses the percept, computes a search plan, and returns the next immediate Action."""
        # TODO: Instantiate VacuumSearchProblem from percept, obtain plan, and pop first action
        # # some code
        # ###*********************************************************
        # So does this part
        # ###********************************************************

        # dirty_cells = set()
        #
        # for position, status in percept.visible_cells.items():
        #     if status == "Dirty":
        #         dirty_cells.add(position)
        #
        # problem = VacuumSearchProblem(percept.position, dirty_cells)
        # plan = self._compute_plan(problem)
        #
        # if not plan:
        #     return Action(clean=False, move="NoOp")
        #
        # next_action = plan.pop(0)
        #
        # if next_action == "Suck":
        #     return Action(clean=True, move="Suck")
        # else:
        #     return Action(clean=False, move=next_action)
        #
        # # ###***************************************************
        raise NotImplementedError("Implement BFSAgent.decide()")


# ==============================================================================
# 3. VERIFICATION & TESTING ENVIRONMENT
# ==============================================================================

class MockEnvironment:
    """Simple grid environment for testing."""
    def __init__(self, agent_pos: Position, dirty_cells: Set[Position], grid_size: int = 5):
        self.pos = agent_pos
        self.dirty = set(dirty_cells)
        self.size = grid_size

    def agent_position(self) -> Position:
        return self.pos

    def cell_status(self, pos: Position) -> str:
        return "Dirty" if pos in self.dirty else "Clean"

    def all_cells(self) -> Dict[Position, str]:
        return {(c, r): "Dirty" if (c, r) in self.dirty else "Clean" 
                for c in range(self.size) for r in range(self.size)}


if __name__ == "__main__":
    initial_pos = (0, 0)
    dirty_set = {(1, 1), (2, 2)}
    
    print("Testing Vacuum Search Problem Formulation...")
    # Add your test cases here

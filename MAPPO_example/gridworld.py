import torch
import pdb
import vmas
import os

from vmas.simulator.core import Agent, World, Landmark, Sphere, Box
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color

import numpy as np

COLORS = [
    "white",
    "black",
    "green",
    "red",
    "darkorange",
    "springgreen",
    "yellow",
    "brown",
    "aquamarine",
    "skyblue"
]
OBJECTS = [
        "EMPTY",
        "WALL",
        "GOAL",
        "TRAP",
        "OBSTACLE",
        "EXIT",
        "KEY",
        "DOOR",
        "BAIT",
        "PORTAL",
        "AGENT"
]

class GridWorld(BaseScenario):
    """Grid World"""

    ACTION_INDEX_TO_STR = {
        0: "UP",
        1: "DOWN",
        2: "LEFT",
        3: "RIGHT",
    }
    ACTION_INDEX_TO_CHAR = {
        0: "^",
        1: "v",
        2: "<",
        3: ">",
    }
    OBJECT_TO_INDEX = {
        "EMPTY": 0,
        "WALL": 1,
        "GOAL": 2,
        "TRAP": 3,
        "LAVA": 4,
        "EXIT": 5,
        "KEY": 6,
        "DOOR": 7,
        "BAIT": 8,
        "PORTAL": 9,
        "AGENT": 10,
    }
    OBJECT_INDEX_TO_CHAR = {
        0: " ",
        1: "#",
        2: "G",
        3: "T",
        4: "L",
        5: "E",
        6: "K",
        7: "D",
        8: "B",
        9: "P",
        10: "A",
    }

    def __init__(
        self,
        maze_file: str,
        goal_reward: float = 1,
        trap_reward: float = -1,
        step_reward: float = -1,
        exit_reward: float = 0.1,
        bait_reward: float = 1,
        lava_reward: float = -1,
        bait_step_penalty: float = -0.25,
        max_step: int = 1000,
    ):
        """Constructor for GridWorld

        Args:
            maze_file (str): Path to the maze file
            goal_reward (float, optional): Reward in the goal state. Defaults to 1.
            trap_reward (float, optional): Reward in the trap state. Defaults to -1.
            step_reward (float, optional): Reward in the step state. Defaults to -1.
            exit_reward (float, optional): Reward in the exit state. Defaults to 0.1.
            bait_reward (float, optional): Reward in the bait state. Defaults to 1.
            bait_step_penalty (float, optional): Penalty in the bait state. Defaults to -0.25.
            max_step (int, optional): Maximum number of steps. Defaults to 1000.
        """
        super().__init__()
        self._goal_reward = goal_reward
        self._trap_reward = trap_reward
        self._step_reward = step_reward
        self._exit_reward = exit_reward
        self._bait_reward = bait_reward
        self._lava_reward = lava_reward
        self._bait_step_penalty = bait_step_penalty
        self.step_reward = self._step_reward
        self._step_count = 0
        self._maze = np.array([])
        self._state_list = []
        self._agent_location = 0
        self.max_step = max_step
        self.maze_name = os.path.split(maze_file)[1].replace(".txt", "").capitalize()
        self._read_maze(maze_file)
        self._obstacle_count = self._count_obstacle()
        self.render_init(self.maze_name)

        # if min_y is None you can initialize the agent in any state
        # if min_y is not None, you can initialize the agent in the state left to min_y
        min_y = None

        # obstacle init
        self.obstacle_states = []
        for state in range(self.get_grid_space()):
            if self._is_obstacle_state(self._state_list[state]):
                self.obstacle_states.append(self._state_list[state])

        if len(self.obstacle_states) > 0:
            # get the leftest coordinate of obstacle states
            min_y = min(self.obstacle_states, key=lambda x: x[1])[1]

        self._init_states = []
        for state in range(self.get_grid_space()):
            if min_y is not None and self._state_list[state][1] < min_y:
                self._init_states.append(state)
            elif min_y is None:
                self._init_states.append(state)

        assert len(self._init_states) > 0

    def _read_maze(self, maze_file: str) -> None:
        """Read the maze file

        Returns:
            np.ndarray: Maze
        """
        self._maze = np.loadtxt(maze_file, dtype=np.uint8)
        for i in range(self._maze.shape[0]):
            for j in range(self._maze.shape[1]):
                if self._maze[i, j] != 1:
                    self._state_list.append((i, j))

    def _count_obstacle(self) -> int:
        """Count obstacle numbers

        Returns:
            int: 
        """
        return np.count_nonzero(self.maze == 4)
    
    def _is_obstacle_state(self, state_coord: tuple) -> bool:
        """Check if the state is a obstacle state

        Args:
            state_coord (tuple)

        Returns:
            bool
        """
        return self._maze[state_coord[0], state_coord[1]] == self.OBJECT_TO_INDEX["OBSTACLE"]
    
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        # Pass any kwargs you desire when creating the environment

        n_agents = kwargs.get("n_agents", 10) #kwargs會抓n_agents的數，如果沒有就直接給5
        # Create world
        self._world = World(batch_dim, device, dt=0.1, drag=0.25, dim_c=0)
        # Add agents
        for i in range(n_agents):
            agent = Agent(
                name=f"agent {i}",
                collide=True,
                mass=1.0,
                shape=Box(length=0.1,width=0.1),
                max_speed=None,
                color=Color.BLUE,
                action_size = 2,
                discrete_action_nvec = [2, 2], #List長度表示action的維度，數值表示這個維度有幾個動作。例如[2, 2, 2]表示有三個維度(例如x, y, z)，2表示可以向左或向右2個選項
                u_range=1.0,
                #render_action=True
            )
            self._world.add_agent(agent)
        # Add obstacles
        for i in range(self._obstacle_count):
            obstacles = Landmark(
                name=f"obstacle {i}",
                collide=True,
                movable=False,
                shape=Box(length=0.1,width=0.1),
                color=Color.ORANGE,
            )
            self._world.add_landmark(obstacles)
        return self._world

    def reset_world_at(self, env_index):
       for i, agent in enumerate(self.world.agents):
           agent.set_pos(
               torch.tensor(
                    list(self._state_list(np.random.choice(self._init_states))),
                    dtype=torch.float32,
                    device=self.world.device,
               ),
                batch_index=env_index,
           )
       for i, landmark in enumerate(self.world.landmarks):
           landmark.set_pos(
               torch.tensor(
                    list(self.obstacle_states[i]),
                    dtype=torch.float32,
                    device=self.world.device,
               ),
                batch_index=env_index,
           )
    def observation(self, agent):
        # get positions of all landmarks in this agent's reference frame
        landmark_rel_poses = []
        for landmark in self.world.landmarks:
            landmark_rel_poses.append(landmark.state.pos - agent.state.pos)
        return torch.cat([agent.state.pos, agent.state.vel, *landmark_rel_poses], dim=-1)


    def reward(self, agent):
        # reward every agent proportionally to distance from first landmark
        rew = -torch.linalg.vector_norm(agent.state.pos - self.world.landmarks[0].state.pos, dim=-1)
        return rew

kwargs = {
"n_agents" : 1,
}

Grid = GridWorld()# Grid是BaseScenario class
env = vmas.make_env(Grid, num_envs = 1, continuous_actions = False, max_steps = 100, terminated_truncated = True, **kwargs)
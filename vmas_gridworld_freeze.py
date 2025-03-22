# Torch
import torch
import pdb
import vmas
import os
# Tensordict modules
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import multiprocessing
from PIL import Image
# Vmas module
from vmas.simulator.core import Agent, World, Landmark, Sphere, Box
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils, X, Y

# Data collection
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage

# Env
from torchrl.envs import RewardSum, TransformedEnv
from vmas.simulator.core import Agent, Entity, Landmark, Sphere, World
from torchrl.envs.libs.vmas import VmasEnv
from vmas.simulator.sensors import Lidar

# Multi-agent network
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal

# Loss
from torchrl.objectives import ClipPPOLoss, ValueEstimators

# Utils
torch.manual_seed(0)
from matplotlib import pyplot as plt
from tqdm import tqdm
from typing import Callable, Dict, List

import numpy as np

#Define Gridworld Environment
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
        "LAVA",
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

    #def __init__(
    #    self,
    #    maze_file: str,
    #    goal_reward: float = 1,
    #    trap_reward: float = -1,
    #    step_reward: float = -1,
    #    exit_reward: float = 0.1,
    #    bait_reward: float = 1,
    #    lava_reward: float = -1,
    #    bait_step_penalty: float = -0.25,
    #    max_step: int = 1000,
    #    **kwargs
    #):
    #    """Constructor for GridWorld
#
    #    Args:
    #        maze_file (str): Path to the maze file
    #        goal_reward (float, optional): Reward in the goal state. Defaults to 1.
    #        trap_reward (float, optional): Reward in the trap state. Defaults to -1.
    #        step_reward (float, optional): Reward in the step state. Defaults to -1.
    #        exit_reward (float, optional): Reward in the exit state. Defaults to 0.1.
    #        bait_reward (float, optional): Reward in the bait state. Defaults to 1.
    #        bait_step_penalty (float, optional): Penalty in the bait state. Defaults to -0.25.
    #        max_step (int, optional): Maximum number of steps. Defaults to 1000.
    #    """
    #    super().__init__()
    #    self._goal_reward = goal_reward
    #    self._trap_reward = trap_reward
    #    self._step_reward = step_reward
    #    self._exit_reward = exit_reward
    #    self._bait_reward = bait_reward
    #    self._lava_reward = lava_reward
    #    self._bait_step_penalty = bait_step_penalty
    #    self.step_reward = self._step_reward
    #    self._step_count = 0
    #    self._maze = np.array([])
    #    self._state_list = []
    #    self._agent_location = 0
    #    self.max_step = max_step
    #    self.maze_name = os.path.split(maze_file)[1].replace(".txt", "").capitalize()
    #    self._read_maze(maze_file)
    #    self.render_init(self.maze_name)

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        # Pass any kwargs you desire when creating the environment

        self.plot_grid = True
        self.grid_spacing = 0.1
        self.n_agents = kwargs.pop("n_agents", 2)
        self.n_obstacles = kwargs.pop("n_obstacles", 72)
        self.collisions = kwargs.pop("collisions", True)        
        self.world_spawning_x = kwargs.pop(
            "world_spawning_x", 1
        )  # X-coordinate limit for entities spawning
        self.world_spawning_y = kwargs.pop(
            "world_spawning_y", 1
        )  # Y-coordinate limit for entities spawning
        self.enforce_bounds = kwargs.pop(
            "enforce_bounds", True
        )  # If False, the world is unlimited; else, constrained by world_spawning_x and world_spawning_y.

        self.agents_with_same_goal = kwargs.pop("agents_with_same_goal", 1)
        self.split_goals = kwargs.pop("split_goals", False)
        self.observe_all_goals = kwargs.pop("observe_all_goals", False)

        self.lidar_range = kwargs.pop("lidar_range", 0.35)
        self.agent_radius = kwargs.pop("agent_radius", 0.1)
        self.comms_range = kwargs.pop("comms_range", 0)
        self.n_lidar_rays = kwargs.pop("n_lidar_rays", 10)

        self.shared_rew = kwargs.pop("shared_rew", True)
        self.pos_shaping_factor = kwargs.pop("pos_shaping_factor", 1)
        self.final_reward = kwargs.pop("final_reward", 1)

        self.agent_collision_penalty = kwargs.pop("agent_collision_penalty", -1)
        ScenarioUtils.check_kwargs_consumed(kwargs)

        self.min_distance_between_entities = self.agent_radius * 2 + 0.05
        self.min_collision_distance = 0.5

        if self.enforce_bounds:
            self.x_semidim = self.world_spawning_x
            self.y_semidim = self.world_spawning_y
        else:
            self.x_semidim = None
            self.y_semidim = None

        assert 1 <= self.agents_with_same_goal <= self.n_agents
        if self.agents_with_same_goal > 1:
            assert (
                not self.collisions
            ), "If agents share goals they cannot be collidables"
        # agents_with_same_goal == n_agents: all agent same goal
        # agents_with_same_goal = x: the first x agents share the goal
        # agents_with_same_goal = 1: all independent goals
        if self.split_goals:
            assert (
                self.n_agents % 2 == 0
                and self.agents_with_same_goal == self.n_agents // 2
            ), "Splitting the goals is allowed when the agents are even and half the team has the same goal"

        # Make world
        world = World(
            batch_dim,
            device,
            substeps=2,
            x_semidim=self.x_semidim,
            y_semidim=self.y_semidim,
            collision_force = 800
        )

        known_colors = [
            (0.22, 0.49, 0.72),
            (1.00, 0.50, 0),
            (0.30, 0.69, 0.29),
            (0.97, 0.51, 0.75),
            (0.60, 0.31, 0.64),
            (0.89, 0.10, 0.11),
            (0.87, 0.87, 0),
        ]
        colors = torch.randn(
            (max(self.n_agents - len(known_colors), 0), 3), device=device
        )
        entity_filter_agents: Callable[[Entity], bool] = lambda e: isinstance(e, Agent)

        # Add agents
        for i in range(self.n_agents):
            color = (
                known_colors[i]
                if i < len(known_colors)
                else colors[i - len(known_colors)]
            )

            # Constraint: all agents have same action range and multiplier
            agent = Agent(
                name=f"agent_{i}",
                rotatable = False,
                density = 10,
                mass = 4,
                collide= True,
                color=color,
                shape=Box(length = 0.1, width = 0.1),
                render_action=True,
                sensors=(
                    [
                        Lidar(
                            world,
                            n_rays=self.n_lidar_rays,
                            max_range=self.lidar_range,
                            entity_filter=entity_filter_agents,
                        ),
                    ]
                    if self.collisions
                    else None
                ),
            )
            agent.pos_rew = torch.zeros(batch_dim, device=device)
            agent.agent_collision_rew = agent.pos_rew.clone()
            world.add_agent(agent)
            #agent.goal = obstacles

        #Add obstacles  
        for i in range(self.n_obstacles):

            obstacles = Landmark(
                name=f"obstacle {i}",
                shape=Box(length = 0.1, width = 0.1),
                collide=True,
                mass = 300,
                color=Color.BLACK,
            )
            world.add_landmark(obstacles)
            

        self.pos_rew = torch.zeros(batch_dim, device=device)
        self.final_rew = self.pos_rew.clone()

        return world

    def reset_world_at(self, env_index):
       #set agents position
        for i, agent in enumerate(self.world.agents):
           agent.set_pos(
               torch.tensor(
                    [0, 0.1*i],
                    dtype=torch.float32,
                    device=self.world.device,
               ),
                batch_index=env_index,
           )

        #set obstacles position
        
        # obstacles x 和 y 軸的範圍
        obstacles_x_range = torch.linspace(-0.4, 0.4, steps=19)
        obstacles_y_range = torch.linspace(-0.4, 0.4, steps=19)
        # 生成 obstacles 位置
        obstacles_positions = []
        # 上排
        for x in obstacles_x_range:
            obstacles_positions.append((x, 0.4))
        # 下排
        for x in obstacles_x_range:
            obstacles_positions.append((x, -0.4))
        # 左列（去掉上下角）
        for y in obstacles_y_range[1:-1]:  # 去掉 -0.4 和 0.4
            obstacles_positions.append((-0.4, y))
        # 右列（去掉上下角）
        for y in obstacles_y_range[1:-1]:  # 去掉 -0.4 和 0.4
            obstacles_positions.append((0.4, y))            
        # 放在 obstacles 位置
        for i, (x, y) in enumerate(obstacles_positions):
            landmark = self.world.landmarks[i]  # 取出對應的 Landmark
            landmark.set_pos(
                torch.tensor([x, y], dtype=torch.float32, device=self.world.device),
                batch_index=env_index
            )
                #landmark.set_rot(
                #    torch.tensor(
                #         [torch.pi / 4 if i % 2 else -torch.pi / 4],
                #         dtype=torch.float32,
                #         device=self.world.device,
                #    ),
                #     batch_index=env_index,
                #)
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

# Devices
is_fork = multiprocessing.get_start_method() == "fork"
device = torch.device("cuda")
vmas_device = device  # The device where the simulator is run (VMAS can run on GPU)

# Sampling
frames_per_batch = 6_000  # Number of team frames collected per training iteration
n_iters = 10  # Number of sampling and training iterations
total_frames = frames_per_batch * n_iters

# Training
num_epochs = 30  # Number of optimization steps per training iteration
minibatch_size = 400  # Size of the mini-batches in each optimization step
lr = 3e-4  # Learning rate
max_grad_norm = 1.0  # Maximum norm for the gradients

# PPO
clip_epsilon = 0.2  # clip value for PPO loss
gamma = 0.99  # discount factor
lmbda = 0.9  # lambda for generalised advantage estimation
entropy_eps = 1e-4  # coefficient of the entropy term in the PPO loss

max_steps = 300  # Episode steps before done
num_vmas_envs = (
    frames_per_batch // max_steps
)  # Number of vectorized envs. frames_per_batch should be divisible by this number
scenario_name = "gridworld"
n_agents = 3
grid_world = GridWorld()
env = VmasEnv(
    scenario=grid_world,
    num_envs=num_vmas_envs,
    continuous_actions=True,  # VMAS supports both continuous and discrete actions
    max_steps=max_steps,
    device=vmas_device,
    # Scenario kwargs
    n_agents=n_agents,  # These are custom kwargs that change for each VMAS scenario, see the VMAS repo to know more.
)

def _get_deterministic_action(agent: Agent, continuous: bool, env: VmasEnv):
    if continuous:
        action = -agent.action.u_range_tensor.expand(env.batch_dim, env.n_agents, agent.action_size)
        action[:,:,1] = 0
    else:
        action = (
            torch.tensor([1], device=env.device, dtype=torch.long)
            .unsqueeze(-1)
            .expand(env.batch_dim, 1)
        )
    return action.clone()


def rendering_callback(env, td):
    env.frames.append(Image.fromarray(env.render(mode="rgb_array")))
env.frames = []
frame_list = []
reset = env.reset()
with torch.no_grad():
    for _ in range(max_steps):
        actions = []
        for i, agent in enumerate(env.agents):
            action = _get_deterministic_action(agent, True, env)
            deterministic_action = env.rand_action(reset).clone()
            #print(deterministic_action)
            deterministic_action['agents', 'action'] = action
            #print(deterministic_action['agents','action'])
        step_data = env.step(deterministic_action)
        frame = env.render(
            mode="rgb_array",
            agent_index_focus=None,  # Can give the camera an agent index to focus on
            )

        frame_list.append(frame)

    from moviepy.editor import ImageSequenceClip
    fps=30
    clip = ImageSequenceClip(frame_list, fps=fps)
    clip.write_gif(f'{scenario_name}.gif', fps=fps)
          
   #env.rollout(
   #    max_steps=max_steps,
   #    callback=rendering_callback,
   #    auto_cast_to_device=True,
   #    break_when_any_done=False,
   #)
#env.frames[0].save(
#    f"{scenario_name}.gif",
#    save_all=True,
#    append_images=env.frames[1:],
#    duration=3,
#    loop=0,
#)
fuck_you = 1
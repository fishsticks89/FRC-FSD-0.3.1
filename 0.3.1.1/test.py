import pybullet as p
import pybullet_data
import time
import random
import matplotlib
import matplotlib.pyplot as plt
# Initialize PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import namedtuple, deque
from itertools import count
import math

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

class RobotLocomotionEnv(gym.Env):
    def __init__(self):
        # Action space: 5 discrete actions
        self.action_space = spaces.Discrete(5)

        # Environment constants
        self.max_steps = 280
        self.target = np.array([0.0, 0.0])
        self.success_reward = 100
        self.action_cost = -0.1 #! todo implement

        # Simulation parameters
        self.force_magnitude = 500  # Magnitude of the applied force
        self.reset()

    def reset(self):
        self.steps = 0

        # Set up the simulation environment
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        p.setPhysicsEngineParameter(enableConeFriction=1)

        def _create_box(start_pos, start_orientation=[0, 0, 0, 1]):
            """Helper function to create a box with standard parameters"""
            collision_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.65, height=1.0)
            body = p.createMultiBody(baseMass=1,
                                baseCollisionShapeIndex=collision_id,
                                basePosition=start_pos,
                                baseOrientation=start_orientation)
            p.changeDynamics(body, -1, lateralFriction=0.5, spinningFriction=0.5, rollingFriction=0.5)

            return collision_id, body

        # Create adversary box at center
        self.adv_id, self.adv_body = _create_box([0, 0, 0.5])

        # Create agent box at random position
        agent_x = random.uniform(0.5, 10) * np.random.choice([-1, 1])
        agent_y = random.uniform(0.5, 10) * np.random.choice([-1, 1])
        self.ag_id, self.ag_body = _create_box([agent_x, agent_y, 0.5])
        return self.get_state()

    def step(self, ag_action, adv_action):
        self.steps += 1
        if self.steps > self.max_steps or self.reached_goal():
            raise ValueError("Max steps reached")

        def apply_action(body_id, action):
            # Map discrete actions to force vectors
            action_to_force = {
                0: [0, self.force_magnitude, 0],  # Forward
                1: [0, -self.force_magnitude, 0], # Backward
                2: [-self.force_magnitude, 0, 0], # Left
                3: [self.force_magnitude, 0, 0],  # Right
                4: [0, 0, 0]                      # No movement
            }

            if action not in action_to_force:
                raise ValueError("Invalid action")

            force = action_to_force[action]
            pos = p.getBasePositionAndOrientation(body_id)[0]

            p.applyExternalForce(
                objectUniqueId=body_id,
                linkIndex=-1,
                forceObj=force,
                posObj=pos,
                flags=p.WORLD_FRAME
            )

        # Apply forces to both agent and adversary
        apply_action(self.ag_body, ag_action)
        apply_action(self.adv_body, adv_action)

        p.stepSimulation()
        state = self.get_state()

        return (
            state,
            self.get_agent_reward(),
            self.get_adversary_reward(),
            self.reached_goal(),
            self.steps >= self.max_steps
        )

    def get_state(self):
        ag_pos, ag_orientation = p.getBasePositionAndOrientation(self.ag_body)
        adv_pos, adv_orientation = p.getBasePositionAndOrientation(self.adv_body)
        ag_vel, ag_angular_vel = p.getBaseVelocity(self.ag_body)
        adv_vel, adv_angular_vel = p.getBaseVelocity(self.adv_body)
        return [ag_pos[0], ag_pos[1], ag_vel[0], ag_vel[1], adv_pos[0], adv_pos[1], adv_vel[0], adv_vel[1]]

    def dist_between_entities(self):
        position, orientation = p.getBasePositionAndOrientation(self.ag_body)
        adv_position, adv_orientation = p.getBasePositionAndOrientation(self.adv_body)
        return np.linalg.norm(np.array([position[0], position[1]]) - np.array([adv_position[0], adv_position[1]]))

    def ag_dist_to_target(self):
        position, orientation = p.getBasePositionAndOrientation(self.ag_body)
        return np.linalg.norm(np.array([position[0], position[1]]) - self.target)

    def get_adversary_reward(self):
        if self.reached_goal():
            return -self.success_reward
        # penalties are normalized 0 to -1 or 0 to 1
        dist_to_target_penalty = self.ag_dist_to_target() / (20 * 2**0.5) - 1
        dist_between_entities_penalty = -self.dist_between_entities() / (20 * 2**0.5)
        time_penalty = 1
        # penalties are multiplied by fractions of the success reward
        return (time_penalty * 0.7 + dist_between_entities_penalty * 0.3) / self.max_steps * self.success_reward

    def get_agent_reward(self):
        if self.reached_goal():
            return self.success_reward

        # penalties are normalized 0 to -1
        dist_to_target_penalty = -self.ag_dist_to_target() / (20 * 2**0.5)
        time_penalty = -1
        # penalties are multiplied by fractions of the success reward
        return (time_penalty *1 + dist_to_target_penalty * 0.2) / self.max_steps * self.success_reward

    def reached_goal(self):
        return self.ag_dist_to_target() < 0.5

env = RobotLocomotionEnv()

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

Transition = namedtuple('Transition',
                        ('state', 'ag_action', 'adv_action', 'next_state', 'ag_reward', 'adv_reward'))


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


# Get number of actions from gym action space
n_actions = 5
# Get the number of state observations
env.reset()
state = env.get_state()
n_observations = len(state)

policy_net_ag = DQN(n_observations, n_actions).to(device)

policy_net_adv = DQN(n_observations, n_actions).to(device)

def select_action(state, agent):
    with torch.no_grad():
        # t.max(1) will return the largest column value of each row.
        # second column on max result is index of where max element was
        # found, so we pick action with the larger expected reward.
        if agent == "ag":
            return policy_net_ag(state).max(1).indices.view(1, 1)
        else:
            return policy_net_adv(state).max(1).indices.view(1, 1)


policy_net_ag.load_state_dict(torch.load("./0.3.1.1/policy_net_agent.pth", map_location=device))
policy_net_adv.load_state_dict(torch.load("./0.3.1.1/policy_net_adversary.pth", map_location=device))

import time
import torch
import pybullet as p

# -------------------------------------------------------------------------
# Define a single constant to control who is user-controlled:
#   "AGENT"    -> user controls main agent
#   "ADVERSARY"-> user controls adversary
#   "NONE"     -> both agent and adversary are controlled by networks
# -------------------------------------------------------------------------
CONTROL_MODE = "NONE"  

# -------------------------------------------------------------------------
# Assume you have defined the following objects/functions above:
#   env = RobotLocomotionEnv()
#   policy_net_ag (the learned agent network)
#   policy_net_adv (the adversarial policy network, if needed)
#   device
#   select_action function or inline policy call
# -------------------------------------------------------------------------

time_step = 1.0 / 240.0  # Adjust for comfortable visualization
state = env.reset()

while True:
    # Check for reset key press
    keys = p.getKeyboardEvents()
    if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
        print("Manual reset triggered")
        state = env.reset()
        continue

    # 1) Get an action for the agent
    if CONTROL_MODE == "AGENT":
        # Use keyboard for main agent
        ag_action = 4  # default "do nothing" = 4
        
        if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
            ag_action = 0  # forward
        elif p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
            ag_action = 1  # backward
        elif p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
            ag_action = 2  # left
        elif p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
            ag_action = 3  # right

    else:
        # Use policy network for main agent
        state_t = torch.tensor([state], device=device, dtype=torch.float32)
        with torch.no_grad():
            ag_action = policy_net_ag(state_t).max(1).indices.item()

    # 2) Get an action for the adversary
    if CONTROL_MODE == "ADVERSARY":
        # Use keyboard for adversary
        adv_action = 4  # default "do nothing" = 4

        if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
            adv_action = 0  # forward
        elif p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
            adv_action = 1  # backward
        elif p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
            adv_action = 2  # left
        elif p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
            adv_action = 3  # right

    else:
        # Use (optional) policy network for adversary
        # If policy_net_adv does not exist or adversary is not needed,
        # you can choose a default (like 4) or skip entirely.
        if policy_net_adv is not None:
            state_t = torch.tensor([state], device=device, dtype=torch.float32)
            with torch.no_grad():
                adv_action = policy_net_adv(state_t).max(1).indices.item()
        else:
            adv_action = 4  # fallback if no adversarial policy

    # 3) Environment step
    try:
        next_state, ag_reward, adv_reward, reached_goal, done = env.step(ag_action, adv_action)
    except ValueError:
        # If the environment code raises ValueError when max steps is reached:
        print("Episode finished (max steps). Resetting environment.")
        state = env.reset()
        time.sleep(time_step)
        continue

    # 4) Check terminal conditions
    if reached_goal or done:
        print("Episode finished (goal reached or done). Resetting environment.")
        state = env.reset()
    else:
        state = next_state

    # 5) Sleep to throttle the loop for real-time visualization
    time.sleep(time_step)

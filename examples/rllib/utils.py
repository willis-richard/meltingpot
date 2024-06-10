# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MeltingPotEnv as a MultiAgentEnv wrapper to interface with RLLib."""

from typing import List, Optional, Tuple

import dm_env
import dmlab2d
from gymnasium import spaces
from meltingpot import substrate
from meltingpot.utils.policies import policy
from ml_collections.config_dict import ConfigDict
import numpy as np
import pandas as pd
from ray.rllib import algorithms
from ray.rllib.env import multi_agent_env
from ray.rllib.policy import Policy, sample_batch
from ray.rllib.utils.typing import MultiAgentDict

from ..gym import utils

PLAYER_STR_FORMAT = "player_{index}"


class MeltingPotEnv(multi_agent_env.MultiAgentEnv):
  """An adapter between the Melting Pot substrates and RLLib MultiAgentEnv."""

  def __init__(self, env_config: ConfigDict):
    """Initialize the instance

    Args:
      env_config: An environment config
    """
    self.setup(env_config)

    self._action_space_in_preferred_format = True
    self._obs_space_in_preferred_format = True

    super().__init__()

  def setup(self, env_config: ConfigDict):
    substrate_config = env_config["substrate_config"]
    roles = env_config["roles"]

    self._individual_obs = substrate_config["individual_observation_names"]
    self._env = substrate.build_from_config(substrate_config, roles=roles)

    self._num_players = len(self._env.observation_spec())
    self._ordered_agent_ids = [
        PLAYER_STR_FORMAT.format(index=index)
        for index in range(self._num_players)
    ]
    # RLLib requires environments to have the following member variables:
    # observation_space, action_space, and _agent_ids
    self._agent_ids = set(self._ordered_agent_ids)
    # RLLib expects a dictionary of agent_id to observation or action,
    # Melting Pot uses a tuple, so we convert
    self.observation_space = self._convert_spaces_tuple_to_dict(
        utils.spec_to_space(self._env.observation_spec()))
    for k in self.observation_space:
      self.observation_space[k] = utils.remove_world_observations_from_space(
          self.observation_space[k], self._individual_obs)

    self.action_space = self._convert_spaces_tuple_to_dict(
        utils.spec_to_space(self._env.action_spec()))

    self_interest = env_config.get("self-interest")
    if self_interest is not None:
      off_diag_val = (1 - self_interest) / (self._num_players - 1)
      rtm = np.full((self._num_players, self._num_players), off_diag_val)
      np.fill_diagonal(rtm, self_interest)
      rtm = pd.DataFrame(data=rtm,
                        index=self._ordered_agent_ids,
                        columns=self._ordered_agent_ids,
                        dtype=float)
      self._rtm = rtm
    else:
      self._rtm = None

  def reset(self, *args, **kwargs) -> Tuple[MultiAgentDict, MultiAgentDict]:
    """See base class."""
    timestep = self._env.reset()
    obs = utils.timestep_to_observations(timestep, self._individual_obs)
    return obs, {}

  def step(
      self, action_dict
  ) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict,
             MultiAgentDict]:
    """See base class."""
    actions = [action_dict[agent_id] for agent_id in self._ordered_agent_ids]
    timestep = self._env.step(actions)

    if self._rtm is not None:
      rewards = {
            aid:
            np.sum(np.array(timestep.reward, dtype=float) * self._rtm[aid])
            for aid in self._ordered_agent_ids
        }
    else:
      rewards = {
          aid: timestep.reward[idx]
          for idx, aid in enumerate(self._ordered_agent_ids)
      }
    # gymnasium split done into terminated and truncated
    terminated = {"__all__": timestep.last()}
    truncated = {"__all__": False}
    infos = {}
    obs = utils.timestep_to_observations(timestep, self._individual_obs)
    return obs, rewards, terminated, truncated, infos

  def close(self):
    """See base class."""
    self._env.close()

  def get_dmlab2d_env(self):
    """Return the underlying DM Lab2D environment."""
    return self._env

  # Metadata is required by the gymnasium `Env` class that we are extending, to
  # show which modes the `render` method supports.
  metadata = {'render.modes': ['rgb_array']}

  def render(self) -> np.ndarray:
    """Render the environment.

    This allows you to set `record_env` in your training config, to record
    videos of gameplay.

    Returns:
        np.ndarray: This returns a numpy.ndarray with shape (x, y, 3),
        representing RGB values for an x-by-y pixel image, suitable for turning
        into a video.
    """
    observation = self._env.observation()
    world_rgb = observation[0]['WORLD.RGB']

    # RGB mode is used for recording videos
    return world_rgb

  def _convert_spaces_tuple_to_dict(self,
                                    input_tuple: spaces.Tuple) -> spaces.Dict:
    """Returns spaces tuple converted to a dictionary.

    Args:
      input_tuple: tuple to convert.
    """
    return spaces.Dict({
        agent_id: (input_tuple[i])
        for i, agent_id in enumerate(self._ordered_agent_ids)
    })


def env_creator(env_config: ConfigDict) -> MeltingPotEnv:
  """Outputs an environment for registering."""
  return MeltingPotEnv(env_config)


class RayModelPolicy(policy.Policy[policy.State]):
  """Wraps an RLLib algorithm for inference.

  Note: Currently only supports a single input, batching is not enabled
  """

  def __init__(self,
               model: algorithms.Algorithm,
               individual_obs: List[str],
               policy_id: str = sample_batch.DEFAULT_POLICY_ID) -> None:
    """Initialize a policy instance.

    Args:
      model: An rllib.trainer.Trainer checkpoint.
      individual_obs: observation keys for the agent (not global observations)
      policy_id: Which policy to use (if trained in multi_agent mode)
    """
    self._model = model
    self._individual_obs = individual_obs
    self._policy_id = policy_id
    self._prev_action = 0

  def step(self, timestep: dm_env.TimeStep,
           prev_state: policy.State) -> Tuple[int, policy.State]:
    """See base class."""
    observations = {
        key: value
        for key, value in timestep.observation.items()
        if key in self._individual_obs
    }

    action, state, _ = self._model.compute_single_action(
        observations,
        prev_state,
        policy_id=self._policy_id,
        prev_action=self._prev_action,
        prev_reward=timestep.reward)

    self._prev_action = action
    return action, state

  def initial_state(self) -> policy.State:
    """See base class."""
    self._prev_action = 0
    return self._model.get_policy(self._policy_id).get_initial_state()

  def close(self) -> None:
    """See base class."""


class RayPolicy(policy.Policy):
  """Wraps an RLLib policy for inference.

  Loads the policy from a Policy checkpoint and filters observations.

  Note: Currently only supports a single input, batching is not enabled
  """

  def __init__(self,
               checkpoint_path: str,
               individual_obs: List[str],
               policy_id: str = sample_batch.DEFAULT_POLICY_ID) -> None:
    """Initialize a policy instance.

    Args:
      checkpoint_path: An rllib.trainer.Trainer checkpoint.
      individual_obs: observation keys for the agent (not global observations)
      policy_id: Which policy to use (if trained in multi_agent mode)
    """
    policy_path = f'{checkpoint_path}/policies/{policy_id}'
    self._policy = Policy.from_checkpoint(policy_path)
    self._individual_obs = individual_obs
    self._prev_action = 0

  def initial_state(self) -> policy.State:
    """See base class."""
    self._prev_action = 0
    state = self._policy.get_initial_state()
    self._prev_state = state
    return state

  def step(self, timestep: dm_env.TimeStep,
           prev_state: policy.State) -> Tuple[int, policy.State]:
    """See base class."""
    observations = {
        key: value
        for key, value in timestep.observation.items()
        if key in self._individual_obs
    }

    # We want the logic to be stateless so don't use prev_state from input
    action, state, _ = self._policy.compute_single_action(
        observations,
        self._prev_state,
        prev_action=self._prev_action,
        prev_reward=timestep.reward)

    self._prev_action = action
    self._prev_state = state
    return action, state

  def close(self) -> None:
    """See base class."""

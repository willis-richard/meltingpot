from copy import deepcopy
import logging
import numpy as np
from pandas import DataFrame
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import AgentID, EnvType, PolicyID
from typing import Dict, List, Optional, Tuple, Type, Union

# Set up logging
logging.basicConfig(filename="example.log", level=logging.INFO)

class MyCallbacks(DefaultCallbacks):
  def __init__(self):
    super().__init__()

  transfer_map: Dict[AgentID, float] = {f"policy_{i}": 1 - i/6 for i in range(6)}

  @classmethod
  def set_transfer_map(cls, new_map: Dict[AgentID, float]):
    cls.transfer_map = new_map

  def on_postprocess_trajectory(
    self,
    *,
    worker: "RolloutWorker",
    episode: EpisodeV2,  # I think a typo that it isn't Episodev2
    agent_id: AgentID,
    policy_id: PolicyID,
    policies: Dict[PolicyID, Policy],
    postprocessed_batch: SampleBatch,
    original_batches: Dict[AgentID, Tuple[Policy, SampleBatch]],
    **kwargs,
  ) -> None:
    # Do only once:
    # 1. add custom metrics which are the original game rewards
    # 2. Compute the post reward transfer rewards for all agents and store in user data
    if not episode.user_data:
      episode.custom_metrics["pre_transfer_reward"] = deepcopy(episode.agent_rewards)

      agent_ids = episode.get_agents()
      n = len(episode.get_agents())
      policy_ids = [episode.policy_for(aid) for aid in agent_ids]

      # create the reward transfer matrix
      self_interests = np.array([MyCallbacks.transfer_map[pid] for pid in policy_ids])
      off_diags = (1 - self_interests) / (n - 1)
      rtm = np.tile(off_diags, (n, 1))
      np.fill_diagonal(rtm, self_interests)
      rtm = DataFrame(rtm, index=agent_ids, columns=agent_ids)

      # find the post-transfer rewards
      rewards = DataFrame([episode._agent_reward_history[aid] for aid in agent_ids], index=agent_ids)
      ptr = rtm.dot(rewards)
      episode.user_data["ptr"] = ptr

      # logging.info(f"{episode.episode_id} \n agent_ids \n {agent_ids}")
      # logging.info(f"{episode.episode_id} \n policy_ids \n {policy_ids}")
      # logging.info(f"{episode.episode_id} \n rtm \n {rtm.T}")
      # logging.info(f"{episode.episode_id} \n agent_rewards \n {episode.agent_rewards}")
      # logging.info(f"{episode.episode_id} \n rewards \n {rewards}")
      # logging.info(f"{episode.episode_id} \n ptr \n {ptr}")

    # update the postprocessed_batch
    # logging.info(f"{episode.episode_id} \n before \n {postprocessed_batch[SampleBatch.REWARDS]}")
    postprocessed_batch[SampleBatch.REWARDS] = episode.user_data["ptr"].loc[agent_id].values
    # logging.info(f"{episode.episode_id} \n after \n {postprocessed_batch[SampleBatch.REWARDS]}")

  def on_train_result(
    self,
    *,
    algorithm: "Algorithm",
    result: dict,
    **kwargs,
  ) -> None:
      pass

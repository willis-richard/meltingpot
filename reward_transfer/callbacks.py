import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import AgentID
from ray.rllib.utils.typing import PolicyID

logging.basicConfig(filename="callbacks.log", level=logging.INFO)


class RewardTransferCallback(DefaultCallbacks):

  def __init__(self):
    super().__init__()

  transfer_map: Dict[str, float] = {"default": 1.}
  log: bool = False

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
      # This gives the dict values of Dict[Tuple[AgentID, PolicyID], float] instead of float
      # Would need to extract the values
      # episode.custom_metrics["pre_transfer_reward"] = deepcopy(
      #     episode.agent_rewards)

      agent_ids = episode.get_agents()
      n = len(episode.get_agents())
      assert n > 1, "Reward transfer requires two or more agents"
      policy_ids = [episode.policy_for(aid) for aid in agent_ids]

      # create the reward transfer matrix
      self_interests = np.array([self.transfer_map[pid] for pid in policy_ids],
                                dtype=float)
      off_diags = (1 - self_interests) / (n - 1)
      rtm = np.tile(off_diags, (n, 1))
      np.fill_diagonal(rtm, self_interests)
      rtm = pd.DataFrame(rtm, index=agent_ids, columns=agent_ids)

      # find the post-transfer rewards
      rewards = pd.DataFrame(
          [episode._agent_reward_history[aid] for aid in agent_ids],
          index=agent_ids)
      ptr = rtm.dot(rewards).astype(np.float32)
      episode.user_data["post_transfer_reward"] = ptr

    # update the postprocessed_batch
    if self.log:
      logging.info("%s \n before \n %s", episode.episode_id,
                   postprocessed_batch[SampleBatch.REWARDS])

    postprocessed_batch[SampleBatch.REWARDS] = episode.user_data[
        "post_transfer_reward"].loc[agent_id].values

    if self.log:
      logging.info("%s \n after \n %s", episode.episode_id,
                   postprocessed_batch[SampleBatch.REWARDS])

  # def on_algorithm_init(
  #     self,
  #     *,
  #     algorithm: "Algorithm",
  #     **kwargs,
  # ) -> None:
  #   self.transfer_map = deepcopy(algorithm.config.transfer_map)


def make_rt_callback(tm: Dict[str, float], l: bool):

  class NewCallbacksClass(RewardTransferCallback):
    transfer_map = tm
    log = l

  return NewCallbacksClass


class LoadPolicyCallback(DefaultCallbacks):

  def __init__(self):
    super().__init__()

  def on_algorithm_init(
      self,
      *,
      algorithm: "Algorithm",
      **kwargs,
  ) -> None:
    policy = Policy.from_checkpoint(algorithm.config.get("policy_checkpoint"))
    for pid in algorithm.config.get("policies").keys():
      algorithm.get_policy(policy_id=pid).set_weights(policy.get_weights())

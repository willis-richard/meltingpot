import json
import logging
import os

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID


logging.basicConfig(filename="callbacks.log", level=logging.INFO)
logger = logging.getLogger(__name__)


class LoadPolicyCallback(DefaultCallbacks):

  def __init__(self):
    super().__init__()

  def on_create_policy(self, *, policy_id: PolicyID, policy: Policy) -> None:
    """Callback run whenever a new policy is added to an algorithm.

    Args:
        policy_id: ID of the newly created policy.
        policy: The policy just created.
    """
    policy_checkpoint = policy.config.get("policy_checkpoint")

    if policy_checkpoint is not None:
      pretrained_path = os.path.join(policy_checkpoint, "policies", policy_id)

      # If we are pre-training and using independent training-mode, the new
      # player_n will not have a policy to start from. In this case start them
      # as a copy of player_1
      if not os.path.isdir(pretrained_path) and policy.config.get("training-mode") == "independent":
        pretrained_path = os.path.join(policy_checkpoint, "policies", "player_0")

      if os.path.isdir(pretrained_path):
        logger.info("on_create_policy::Process %s:Load pretrained policy from %s for policy %s",
                    os.getpid(), pretrained_path, policy_id)
        pretrained_policy = Policy.from_checkpoint(pretrained_path)
        pretrained_weights = pretrained_policy.get_weights()
        policy.set_weights(pretrained_weights)
      else:
        logger.warn("on_create_policy::Process %s:Pretrained policy %s does not exist",
                    os.getpid(), pretrained_path)



class SaveResultsCallback(DefaultCallbacks):

  def __init__(self):
    super().__init__()

  def on_train_result(self, *, algorithm, metrics_logger, result, **kwargs) -> None:

    results_filepath = os.path.join(algorithm.config["working_folder"], "results.json")

    info = {}
    info["training_iteration"] = result["training_iteration"]
    self_interest = algorithm.config.env_config.get("self-interest")
    info["self-interest"] = 1 if self_interest is None else self_interest
    info["num_players"] = len(algorithm.config.env_config["roles"])
    info["training-mode"] = algorithm.config.get("training-mode")
    info.update(result["env_runners"]["hist_stats"])

    with open(results_filepath, mode="a", encoding="utf8") as f:
      json.dump(info, f)
      f.write("\n")
      logger.debug("on_train_result::%s", info)

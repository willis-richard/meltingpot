import json
import logging
from typing import Tuple

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
      logger.info("on_create_policy::Load pretrained policy from %s", policy_checkpoint)
      pretrained_policy = Policy.from_checkpoint(policy_checkpoint)
      pretrained_weights = pretrained_policy.get_weights()
      logger.debug("on_create_policy::Loaded weights from checkpoint: %s", pretrained_weights)

      logger.debug("on_create_policy::Current weights for policy %s: %s", policy_id, policy.get_weights())

      policy.set_weights(pretrained_weights)

      logger.debug("on_create_policy::New weights for policy %s: %s", policy_id, policy.get_weights())


class UpdateTrainingCallback(DefaultCallbacks):

  def __init__(self):
    super().__init__()

  def on_train_result(self, *, algorithm, metrics_logger, result, **kwargs) -> None:
    # update lr and num players
    update_every = algorithm.config["epochs_per_curriculum"]
    logger.info("on_train_result::algorithm config roles: %s", algorithm.config["env_config"]["roles"])
    logger.info("on_train_result::algorithm config lr: %s", algorithm.config["lr"])

    if result["training_iteration"] % update_every == 0:
      default_player_roles = algorithm.config["env_config"]["substrate_config"]["default_player_roles"]
      n = 1 + result["training_iteration"] // update_every

      algorithm.config["env_config"]["roles"] = default_player_roles[0:n]
      algorithm.config["lr"] = algorithm.config["LR"] / n

      algorithm.setup(algorithm.config)


class SaveResultsCallback(DefaultCallbacks):

  def __init__(self):
    super().__init__()

  def on_train_result(self, *, algorithm, metrics_logger, result, **kwargs) -> None:
    with open(algorithm.config["results_filepath"], 'a') as f:
      json.dump(result["env_runners"]["hist_stats"], f)
      f.write('\n')

  # def on_evaluate_end(self, *, algorithm, metrics_logger, evaluation_metrics, **kwargs) -> None:

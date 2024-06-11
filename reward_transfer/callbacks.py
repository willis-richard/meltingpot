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
      logger.info("on_create_policy::Load pretrained policy from %s", policy_checkpoint)
      pretrained_policy = Policy.from_checkpoint(policy_checkpoint)
      pretrained_weights = pretrained_policy.get_weights()

      logger.debug("on_create_policy::Loaded weights from checkpoint: %s", pretrained_weights)
      logger.debug("on_create_policy::Current weights for policy %s: %s", policy_id, policy.get_weights())

      policy.set_weights(pretrained_weights)

      logger.debug("on_create_policy::New weights for policy %s: %s", policy_id, policy.get_weights())


class SaveResultsCallback(DefaultCallbacks):

  def __init__(self):
    super().__init__()

  def on_train_result(self, *, algorithm, metrics_logger, result, **kwargs) -> None:

    results_filepath = os.path.join(algorithm.config["working_folder"], "results.json")

    info = {}
    info["training_iteration"] = result["training_iteration"]
    self_interest = algorithm.config.env_config.get("self-interest")
    info["self-interest"] = 0 if self_interest is None else self_interest
    info["num_players"] = len(algorithm.config.env_config["roles"])
    info.update(result["env_runners"]["hist_stats"])

    with open(results_filepath, mode="a", encoding="utf8") as f:
      json.dump(info, f)
      f.write("\n")
      logger.debug("on_train_result::%s", info)

  # def on_evaluate_end(self, *, algorithm, metrics_logger, evaluation_metrics, **kwargs) -> None:


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

      algorithm.config["lr"] = algorithm.config["LR"] / n

      new_env_config = algorithm.config["env_config"]
      new_env_config["roles"] = default_player_roles[0:n]

      algorithm.config.environment(env_config=new_env_config)
      algorithm.workers.foreach_worker(lambda worker: worker.config.environment(env_config=new_env_config))

      algorithm.workers.reset([])
      algorithm.workers._setup(
        validate_env=algorithm.config.validate_env_runners_after_construction,
        config=algorithm.config,
        num_env_runners=algorithm.config.num_env_runners,
        local_env_runner=True
      )

      # also do lr: schedule linearly interpolates... which could be fine... LR to LR/8 in 8*n_iterations
      # algorithm.workers.foreach_env(lambda env: env.setup(new_env_config))

      # algorithm.workers.add_workers(
      #   algorithm.config.num_env_runners,
      #   validate=algorithm.config.validate_env_runners_after_construction)

      # algorithm.setup(algorithm.config)


      # undo these changes

      # missing lr -> do as a schedule is safer

      # update the algorithm config

      # policy = algorithm.get_policy("default")

      # other_policy = algorithm.get_policy("default")
      # assert (policy.get_weights() & other_policy.get_weights()).all(), f"POLICY={policy.get_weights()}\nOTHER={other_policy.get_weights()}"

      # algorithm.remove_policy("default")
      # algorithm.add_policy(policy_id="default", policy=policy)

      # local_tf_session_args = algorith.config.tf_session_args.copy()
      # local_tf_session_args.update(config.local_tf_session_args)
      # self._local_config = config.copy(copy_frozen=False).framework(
      #     tf_session_args=local_tf_session_args
      # )
      # algorithm.workers.


      # or get the env_runners to reset() and possibly add_workers

      # algorithm.workers.reset()
      # algorithm.workers.add_workers()

      # algorithm._counters["current_roles"] = new_roles

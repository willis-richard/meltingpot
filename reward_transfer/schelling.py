"""
Plots a Schelling diagram for an environment given checkpoints with policies
said to implement cooperation and defection
"""

import argparse
import logging
import os

import json
import ray
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.utils.checkpoints import get_checkpoint_info
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy.policy import Policy, PolicyID, PolicySpec
from ray.tune.registry import register_env
from ray.tune.analysis.experiment_analysis import ExperimentAnalysis

from examples.rllib import utils
from meltingpot import substrate


def main():
  print("Start")
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--experiment_state",
      type=str,
      required=True,
      help="ray.tune experiment_state to load. The default setting will load"
      " the last training run created by self_play_train.py. If you want to use"
      " a specific run, provide a path, expected to be of the format "
      " ~/ray_results/PPO/experiment_state-DATETIME.json")
  parser.add_argument(
      "--cooperation_checkpoint",
      type=str,
      required=True,
      help="If provided, use this checkpoint instead of the last checkpoint")
  parser.add_argument(
      "--defection_checkpoint",
      type=str,
      required=True,
      help="If provided, use this checkpoint instead of the last checkpoint")

  args = parser.parse_args()

  print("Calling init")
  ray.init(address="local",
           num_cpus=4,
           logging_level=logging.ERROR)

  register_env("meltingpot", utils.env_creator)

  # fuck it: get the config from the env state and just update the number of players and self interest as needed
  # will need to load in the policies
  print("Loading ExperimentAnalysis")
  experiment = ExperimentAnalysis(
      args.experiment_state,
      default_metric="env_runners/episode_reward_mean",
      default_mode="max")

  config = PPOConfig.from_dict(experiment.best_config)

  substrate_config = substrate.get_config(config["env_config"]["substrate"])

  config["env_config"]["self-interest"] = 1
  config["env_config"]["roles"] = substrate_config.default_player_roles

  base_env = utils.env_creator(config["env_config"])
  aids = base_env._ordered_agent_ids

  config = config.resources(num_gpus=0)
  config = config.env_runners(
      num_env_runners=0,
      num_envs_per_env_runner=1,
      # create_env_on_local_worker=True,
  )
  config = config.evaluation(
      evaluation_duration=1,
      evaluation_num_env_runners=2,
      evaluation_interval=1,
  )
  # # wtf wtf wtf
  # config = config.framework("tf2")

  # load all the policies...
  # Start with all defect
  print("Loading defect policies")
  policies = dict((aid, Policy.from_checkpoint(os.path.join(args.defection_checkpoint, "policies", aid))) for aid in aids)

  class LoadPolicyCallback(DefaultCallbacks):

    def __init__(self):
      super().__init__()

    def on_create_policy(self, *, policy_id: PolicyID, policy: Policy) -> None:
      """Callback run whenever a new policy is added to an algorithm.

      Args:
          policy_id: ID of the newly created policy.
          policy: The policy just created.
      """
      policy.set_weights(policies[policy_id].get_weights())


  config = config.callbacks(LoadPolicyCallback)
  # config = config.multi_agent(policies=policies)

  #            new="AlgorithmConfig.num_cpus_for_main_process",


  print("Building PPO instance")
  ppo = config.build()

  print("Running evaluate()")
  results = ppo.evaluate()
  print(results)

  i = 0

  with open(f"n_c_{i}.json", mode="w", encoding="utf8") as f:
    json.dump(results, f)

  # sweep over the possible policy pairings
  for aid in aids:
    print(f"Update cooperate policy {role}")
    policies[aid] = Policy.from_checkpoint(os.path.join(args.cooperation_checkpoint, "policies", aid))
    # config = config.multi_agent(policies=policies)
    ppo = config.build()

    print("Running evaluate()")
    results = ppo.evaluate()
    print(results)

    i += 1
    with open(f"n_c_{i}.json", mode="w", encoding="utf8") as f:
      json.dump(results, f)


if __name__ == "__main__":
  main()

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
from ray.rllib.policy.policy import Policy
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
  config["env_config"]["roles"] = substrate_config.default_player_roles[0:2]

  base_env = utils.env_creator(config["env_config"])
  aids = base_env._ordered_agent_ids[0:2]

  # load all the policies...

  # load the algorithm state, so we have access to the config
  # DO FROM_CHECKPOINT AND THEN ITERATIVELY ADD POLICY


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
  #            new="AlgorithmConfig.num_cpus_for_main_process",

  print("Building PPO instance")
  ppo = config.build()

  # Start with all defect
  for role in aids:
    print(f"Update defect policy {role}")
    ppo.remove_policy(role)
    policy = Policy.from_checkpoint(os.path.join(args.defection_checkpoint, "policies", role))
    ppo.add_policy(policy_id=role, policy=policy)
  # ppo.load_checkpoint(args.defection_checkpoint)

  # update the config resources
  # do I need to update env_runners?

  # sweep over the possible policy pairings

  # update the policy mapping functions and call evaluate

  print("Running evaluate()")
  results = ppo.evaluate()
  print(results)

  i = 0

  with open(f"n_c_{i}.json", mode="w", encoding="utf8") as f:
    json.dump(results, f)

  for role in aids:
    print(f"Update cooperate policy {role}")
    ppo.remove_policy(role)
    i += 1
    policy = Policy.from_checkpoint(os.path.join(args.cooperation_checkpoint, "policies", role))
    ppo.add_policy(policy_id=role, policy=policy)

    print("Running setup()")
    ppo.setup()
    print("Running evaluate()")
    results = ppo.evaluate()
    print(results)

    with open(f"n_c_{i}.json", mode="w", encoding="utf8") as f:
      json.dump(results, f)


if __name__ == "__main__":
  main()

"""Run experiments"""

import argparse
from collections import defaultdict
import importlib
import json
import numpy as np
import os
import pandas as pd
from typing import Any, Tuple, Dict, List, Mapping, Callable, Sequence, Optional

from meltingpot import substrate
from ml_collections.config_dict import ConfigDict
import ray
from ray import tune
from ray.air import CheckpointConfig
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.tune.experiment import Trial
from ray.tune.registry import register_env
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

from examples.rllib import utils
from reward_transfer.callbacks import LoadPolicyCallback, SaveResultsCallback

LOGGING_LEVEL = "WARN"
VERBOSE = 0  # 0: silent, 1: status


def parse_arguments() -> argparse.Namespace:
  """Parse command line arguments."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--substrate",
      type=str,
      required=True,
      help="Which substrate to train on. e.g. 'clean_up'")
  parser.add_argument(
      "--n_iterations",
      type=int,
      required=True,
      help="number of training iterations to use")
  parser.add_argument(
      "--framework",
      type=str,
      required=True,
      choices=["torch", "tf", "tf2"],
      help="which deep learning framework to use")
  parser.add_argument(
      "--num_cpus", type=int, required=True, help="number of CPUs to use")
  parser.add_argument(
      "--num_gpus", type=int, default=0, help="number of GPUs to use")
  parser.add_argument(
      "--local_dir",
      type=str,
      required=True,
      help="The path the results will be saved to")
  parser.add_argument(
      "--tmp_dir",
      type=str,
      default=None,
      help="Custom tmp location for temporary ray logs")
  parser.add_argument(
      "--rollout_workers",
      type=int,
      required=True,
      help="Number of rollout workers, should be in [0, num_cpus]")
  parser.add_argument(
      "--envs_per_worker",
      type=int,
      default=1,
      help="Number of episodes each worker runs in parallel")
  parser.add_argument(
      "--episodes_per_worker",
      type=int,
      default=1,
      help="Number of episodes per each worker in a training batch"
      " (not including parallelism)")
  parser.add_argument(
      "--max_concurrent_trials",
      type=int,
      default=None,
      help="maximum number of concurrent trials to run")
  parser.add_argument(
      "--wandb", type=str, default=None, help="wandb project name")

  subparsers = parser.add_subparsers(
      dest="training", required=True, help="Training mode")

  parser_optimise = subparsers.add_parser(
      "optimise", help="Optimise hyperparameters")
  parser_optimise.add_argument(
      "--num_samples",
      type=int,
      default=100,
      help="Number of samples to run for hyperparameter optimisation")

  parser_pretraining = subparsers.add_parser(
      "pre-training", help="Pre-train a policy")
  parser_pretraining.add_argument(
      "--trial_id",
      type=str,
      default=None,
      help="Trial id to resume training from (if we had an error)")
  parser_pretraining.add_argument(
      "--training_mode",
      type=str,
      choices=["independent", "self-play"],
      required=True,
      help="self-play enables parameter sharing")

  parser_training = subparsers.add_parser(
      "training", help="Iteratively decrease the self-interest")
  parser_training.add_argument(
      "--trial_id",
      type=str,
      required=True,
      help="The pre-training trial_id to continue training from")
  parser_training.add_argument(
      "--num_players", type=int, required=True, help="Number of players")
  parser_training.add_argument(
      "--training_mode",
      type=str,
      choices=["independent", "self-play"],
      required=True,
      help="self-play enables parameter sharing")

  parser_scratch = subparsers.add_parser(
      "scratch", help="Validate from scratch")
  parser_scratch.add_argument(
      "--num_players", type=int, required=True, help="Number of players")
  parser_scratch.add_argument(
      "--self_interest",
      type=float,
      required=True,
      help="Self-interest level (resuming crashed training)")
  parser_scratch.add_argument(
      "--num_seeds",
      type=int,
      default=4,
      help="Number of samples to run for scratch training")
  parser_scratch.add_argument(
      "--training_mode",
      type=str,
      choices=["independent", "self-play"],
      required=True,
      help="self-play enables parameter sharing")

  return parser.parse_args()


def setup_environment(
    args: argparse.Namespace) -> Tuple[Dict[str, Any], int, int]:
  """Register the environment creator function with rllib, and create the default
  environment config."""
  register_env("meltingpot", utils.env_creator)

  substrate_config = substrate.get_config(args.substrate)
  env_module = importlib.import_module(
      f"meltingpot.configs.substrates.{args.substrate}")
  substrate_definition = env_module.build(substrate_config.default_player_roles,
                                          substrate_config)

  env_config = ConfigDict({
      "substrate": args.substrate,
      "substrate_config": substrate_config,
      "roles": substrate_config.default_player_roles,
      "scaled": 1
  })

  return env_config, substrate_definition["spriteSize"], substrate_definition[
      "maxEpisodeLengthFrames"]


def create_model_config(base_env: utils.MeltingPotEnv,
                        sprite_size) -> Dict[str, Any]:
  """Create the neural network config, as per rllib.models.catalogue

    Args:
        base_env: The environment created using MeltingPot, so that we may
                  access the observation space
        sprite_size: The size of the pixel represenation of a grid square

    Returns:
        A dictionary containing the model configuration.
  """
  rgb_shape = base_env.observation_space["player_0"]["RGB"].shape
  sprite_x, sprite_y = rgb_shape[0], rgb_shape[1]

  if sprite_size == 8:
    conv_filters = [[16, [8, 8], 8], [32, [4, 4], 1],
                    [64, [sprite_x // sprite_size, sprite_y // sprite_size], 1]]
  elif sprite_size == 1:
    conv_filters = [[16, [3, 3], 1], [32, [3, 3], 1],
                    [64, [sprite_x, sprite_y], 1]]
  else:
    raise ValueError(f"Unsuppoerted sprite_size {sprite_size}")

  return {
      "conv_filters": conv_filters,
      "conv_activation": "relu",
      "post_fcnet_hiddens": [64, 64],
      "post_fcnet_activation": "relu",
      "vf_share_layers": True,
      "use_lstm": True,
      "lstm_use_prev_action": False,
      "lstm_use_prev_reward": False,
      "lstm_cell_size": 128,
  }


def create_ppo_config(args: argparse.Namespace, model: Mapping[str, Any],
                      train_batch_size: int, policy_mapping_fn: Callable[str,
                                                                         str],
                      env_config: Mapping[str, Any]) -> PPOConfig:
  return PPOConfig().training(
      gamma=0.999,
      train_batch_size=train_batch_size,
      model=model,
      lambda_=0.99,
      sgd_minibatch_size=min(10000, train_batch_size),
      num_sgd_iter=12,
      vf_loss_coeff=0.8,
      entropy_coeff=1e-3,
      clip_param=0.32,
      vf_clip_param=2,
  ).env_runners(
      num_env_runners=args.rollout_workers,
      num_envs_per_env_runner=args.envs_per_worker,
      rollout_fragment_length=400,
      batch_mode="complete_episodes",
  ).environment(
      env_config=env_config,
      env="meltingpot",
  ).multi_agent(
      count_steps_by="env_steps",
      policy_mapping_fn=policy_mapping_fn,
  ).fault_tolerance(
      recreate_failed_env_runners=True,
      num_consecutive_env_runner_failures_tolerance=3,
  ).debugging(
      log_level=LOGGING_LEVEL
  ).resources(
      num_gpus=min(args.num_gpus, 1)
  ).framework(
      framework=args.framework
  ).reporting(
      metrics_num_episodes_for_smoothing=1
  )


def create_tune_callbacks(
    args: argparse.Namespace) -> Optional[List[WandbLoggerCallback]]:
  """Create WandB callbacks for logging. The environment needs to have set a
  WANDB_API_KEY environment variable"""
  if args.wandb is not None:
    return [
        WandbLoggerCallback(
            project=args.wandb,
            api_key=os.environ["WANDB_API_KEY"],
            log_config=False)
    ]
  return None


def run_optimise(args: argparse.Namespace, config: PPOConfig, env_config: Mapping[str, Any]) -> None:
  """Run hyper-parameter optimisation in a single-agent environment for PPO"""

  def custom_trial_name_creator(trial: Trial) -> str:
    """Create a custom name that includes hyperparameters."""
    attributes = ("sgd_minibatch_size", "num_sgd_iter", "lr", "lambda",
                  "vf_loss_coeff", "clip_param")
    attributes_str = [
        f"{trial.config[a]:.5f}".rstrip("0").rstrip(".") for a in attributes
    ]
    return f"{trial.trainable_name}_{trial.trial_id}_{','.join(attributes_str)}"

  assert config.get(
      "train_batch_size"
  ) >= 10000, f"train_batch_size must be greater than 10000. Suggest increasing --episodes_per_worker"

  config = config.training(
      sgd_minibatch_size=tune.qrandint(5000, 10000, 2500),
      num_sgd_iter=tune.qrandint(8, 14, 2),
      lr=tune.qloguniform(5e-5, 3e-4, 1e-5),
      lambda_=tune.quniform(0.95, 1.0, 0.01),
      vf_loss_coeff=tune.quniform(0.75, 1, 0.05),
      clip_param=tune.quniform(0.28, 0.36, 0.02),
  ).multi_agent(policies={"default": PolicySpec()})

  env_config["roles"] = env_config["roles"][:1]

  config = config.environment(env_config=env_config)

  search_alg = OptunaSearch(
      metric="env_runners/episode_reward_mean",
      mode="max",
  )

  scheduler = ASHAScheduler(
      time_attr="training_iteration",
      metric="env_runners/episode_reward_mean",
      mode="max",
      max_t=args.n_iterations,
      grace_period=max(1, args.n_iterations // 2),
      reduction_factor=2,
      brackets=1,
  )

  tune.run(
      run_or_experiment="PPO",
      name=os.path.join(args.substrate, "optimise"),
      stop={"training_iteration": args.n_iterations},
      config=config,
      num_samples=args.num_samples,
      storage_path=args.local_dir,
      search_alg=search_alg,
      scheduler=scheduler,
      verbose=VERBOSE,
      trial_name_creator=custom_trial_name_creator,
      log_to_file=False,
      callbacks=create_tune_callbacks(args),
      max_concurrent_trials=args.max_concurrent_trials,
  )


def setup_logging_utils(
    args: argparse.Namespace,
    config: PPOConfig) -> Tuple[PPOConfig, str, Callable[[Trial], str], str]:
  """Set up logging utilities for the experiment.
  Args:
      args: Parsed command-line arguments.
      config: PPO configuration.

  Returns:
      A tuple containing:
      - Updated PPO configuration
      - Experiment name
      - Custom trial name creator function
      - Path to the checkpoints log file
  """

  def custom_trial_name_creator(trial: Trial) -> str:
    """Create a custom name that includes number of players and self-interest."""
    trial_name = f"{trial.trainable_name}_{trial.config['trial_id']}"
    trial_name += f"_n={len(trial.config['env_config'].get('roles'))}"

    self_interest = trial.config["env_config"].get("self-interest")
    trial_name += f"_s={self_interest:.3f}" if self_interest is not None else "_s=1.000"

    trial_name += f"_{trial.config.get('training-mode')}"

    return trial_name

  config = config.callbacks(
      ray.rllib.algorithms.callbacks.make_multi_callbacks(
          [SaveResultsCallback, LoadPolicyCallback]))

  trial_id = args.trial_id if hasattr(
      args, 'trial_id') and args.trial_id is not None else Trial.generate_id()
  config["trial_id"] = trial_id
  name = os.path.join(args.substrate, trial_id)
  working_folder = os.path.join(args.local_dir, name)
  config["working_folder"] = working_folder
  checkpoints_log_filepath = os.path.join(working_folder, "checkpoints.json")

  config["training-mode"] = args.training_mode

  return config, name, custom_trial_name_creator, checkpoints_log_filepath


def create_lr_and_policies(
    args: argparse.Namespace, num_players: int,
    ordered_agent_ids: Sequence[str]) -> Tuple[float, Dict[str, PolicySpec]]:
  """Create learning rate and policies based on the training mode and number of players.

  Args:
      args: Parsed command-line arguments.
      n: Number of players.
      ordered_agent_ids: Sequence of ordered agent IDs.

  Returns:
      A tuple containing:
      - Learning rate (float)
      - Dictionary of policies (Dict[str, PolicySpec])
  """
  if args.training_mode == "independent":
    lr = 7e-5
    policies = {aid: PolicySpec() for aid in ordered_agent_ids[:num_players]}
  else:  # self-play
    lr = 7e-5 / num_players
    policies = {"default": PolicySpec()}

  return lr, policies


def log_checkpoint_info(config: PPOConfig,
                        checkpoints_log_filepath: str) -> None:
  """Log checkpoint information to a file."""
  info = {
      "self-interest": config.env_config.get("self-interest", 1),
      "num_players": len(config.env_config["roles"]),
      "policy_checkpoint": config["policy_checkpoint"],
      "training-mode": config.get("training-mode")
  }
  with open(checkpoints_log_filepath, mode="a", encoding="utf8") as f:
    json.dump(info, f)
    f.write("\n")


def run_pretraining(args: argparse.Namespace, config: PPOConfig,
                    env_config: Mapping[str, Any],
                    ordered_agent_ids: Sequence[str]) -> None:
  """Train a policy (either independent or with parameter sharing) while
  iteratively increasing the number of players in the environment, up to the
  maximum

  Args:
      args: Parsed command-line arguments.
      config: Initial PPO configuration.
      env_config: Environment configuration.
      ordered_agent_ids: Sequence of ordered agent IDs that the Melting Pot
                         environment expects.
  """
  config, name, custom_trial_name_creator, checkpoints_log_filepath = setup_logging_utils(
      args, config)

  start_n = 1
  # a passed in trial_id means we wish to resume from the last checkpoint
  if args.trial_id is not None:
    with open(checkpoints_log_filepath, mode="r", encoding="utf8") as f:
      info = json.loads(f.readlines()[-1])
      if info["self-interest"] != 1:
        env_config["self-interest"] = info["self_interest"]
      start_n += info["num_players"]
      config["policy_checkpoint"] = info["policy_checkpoint"]

  default_player_roles = env_config["substrate_config"]["default_player_roles"]
  for n in range(start_n, len(default_player_roles) + 1):
    env_config["roles"] = default_player_roles[:n]

    lr, policies = create_lr_and_policies(args, n, ordered_agent_ids)

    config = config.training(lr=lr).environment(
        env_config=env_config,).multi_agent(
            policies=policies,)

    experiment = tune.run(
        run_or_experiment="PPO",
        name=name,
        metric="env_runners/episode_reward_mean",
        mode="max",
        stop={"training_iteration": args.n_iterations},
        config=config,
        storage_path=args.local_dir,
        checkpoint_config=CheckpointConfig(checkpoint_at_end=True),
        verbose=VERBOSE,
        trial_name_creator=custom_trial_name_creator,
        trial_dirname_creator=custom_trial_name_creator,
        log_to_file=False,
        callbacks=create_tune_callbacks(args),
        max_concurrent_trials=args.max_concurrent_trials,
    )

    policy_checkpoint = experiment.trials[-1].checkpoint.path
    config["policy_checkpoint"] = policy_checkpoint

    log_checkpoint_info(config, checkpoints_log_filepath)


def run_training(args: argparse.Namespace, config: PPOConfig,
                 env_config: Mapping[str, Any],
                 ordered_agent_ids: Sequence[str]) -> None:
  """Starting from a pre-trained policy, continue training while iteratively
     decreasing the self-interest of the agents.

  Args:
      args: Parsed command-line arguments.
      config: Initial PPO configuration.
      env_config: Environment configuration.
      ordered_agent_ids: Sequence of ordered agent IDs that the Melting Pot
                         environment expects.
  """
  config, name, custom_trial_name_creator, checkpoints_log_filepath = setup_logging_utils(
      args, config)

  n = args.num_players
  env_config["roles"] = env_config["substrate_config"][
      "default_player_roles"][:n]

  df = pd.read_json(checkpoints_log_filepath, lines=True)
  condition = (df["num_players"] == n) & \
    (df["training-mode"] == args.training_mode)
  assert df[condition][
      "self-interest"].is_unique, f"Duplicate checkpoints found in {df[condition]}"
  policy_checkpoint = df.loc[df[condition]
                             ["self-interest"].idxmin()]["policy_checkpoint"]
  config["policy_checkpoint"] = policy_checkpoint

  lr, policies = create_lr_and_policies(args, n, ordered_agent_ids)

  config = config.training(lr=lr).multi_agent(policies=policies,)

  ratio = [20, 10, 5, 3, 5 / 2, 2, 5 / 3, 4 / 3, 1]
  # If we are resuming
  n_completed = len(df[condition]["self-interest"])
  for s in [r / (n + r - 1) for r in ratio[(n_completed - 1):]]:
    env_config["self-interest"] = s

    config = config.environment(env_config=env_config)

    experiment = tune.run(
        run_or_experiment="PPO",
        name=name,
        metric="env_runners/episode_reward_mean",
        mode="max",
        stop={"training_iteration": args.n_iterations},
        config=config,
        storage_path=args.local_dir,
        checkpoint_config=CheckpointConfig(checkpoint_at_end=True),
        verbose=VERBOSE,
        trial_name_creator=custom_trial_name_creator,
        trial_dirname_creator=custom_trial_name_creator,
        log_to_file=False,
        callbacks=create_tune_callbacks(args),
        max_concurrent_trials=args.max_concurrent_trials,
    )

    policy_checkpoint = experiment.trials[-1].checkpoint.path
    config["policy_checkpoint"] = policy_checkpoint
    log_checkpoint_info(config, checkpoints_log_filepath)


def run_scratch(args: argparse.Namespace, config: ConfigDict,
                env_config: Mapping[str, Any],
                ordered_agent_ids: Sequence[str]) -> None:
  """Run an experiment starting from a newly initialised policy, and holding the
     self-interest level constant.

  Args:
      args: Parsed command-line arguments.
      config: Initial PPO configuration.
      env_config: Environment configuration.
      ordered_agent_ids: Sequence of ordered agent IDs that the Melting Pot
                         environment expects.
  """
  config, name, custom_trial_name_creator, checkpoints_log_filepath = setup_logging_utils(
      args, config)

  n = args.num_players

  env_config["roles"] = env_config["substrate_config"][
      "default_player_roles"][:n]

  lr, policies = create_lr_and_policies(args, n, ordered_agent_ids)

  env_config["self-interest"] = args.self_interest

  config = config.training(lr=lr).multi_agent(
      policies=policies,).environment(env_config=env_config)

  tune.run(
      run_or_experiment="PPO",
      name=name,
      metric="env_runners/episode_reward_mean",
      mode="max",
      stop={"training_iteration": args.n_iterations},
      config=config,
      storage_path=args.local_dir,
      checkpoint_config=CheckpointConfig(checkpoint_at_end=True),
      verbose=VERBOSE,
      trial_name_creator=custom_trial_name_creator,
      trial_dirname_creator=custom_trial_name_creator,
      log_to_file=False,
      callbacks=create_tune_callbacks(args),
      max_concurrent_trials=args.max_concurrent_trials,
      num_samples=args.num_seeds,
  )


def main():
  args = parse_arguments()

  ray.init(
      address="local",
      num_cpus=args.num_cpus,
      num_gpus=args.num_gpus,
      logging_level=LOGGING_LEVEL,
      _temp_dir=args.tmp_dir)

  env_config, sprite_size, horizon = setup_environment(args)

  base_env = utils.env_creator(env_config)

  model = create_model_config(base_env, sprite_size)

  train_batch_size = (
      max(1, args.rollout_workers) * args.envs_per_worker * horizon *
      args.episodes_per_worker)

  if args.training == "optimise" or args.training_mode == "self-play":
    policy_mapping_fn = lambda aid, *_, **__: "default"
  else:
    policy_mapping_fn = lambda aid, *_, **__: aid

  config = create_ppo_config(args, model, train_batch_size, policy_mapping_fn,
                             env_config)

  if args.training == "optimise":
    run_optimise(args, config, env_config)
  elif args.training == "pre-training":
    run_pretraining(args, config, env_config, base_env._ordered_agent_ids)
  elif args.training == "training":
    run_training(args, config, env_config, base_env._ordered_agent_ids)
  elif args.training == "scratch":
    run_scratch(args, config, env_config, base_env._ordered_agent_ids)

  ray.shutdown()


if __name__ == "__main__":
  main()

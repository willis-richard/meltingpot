"""Run experiments"""

import argparse
from collections import defaultdict
import importlib
import json
import numpy as np
import os
import pandas as pd

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

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--substrate",
    type=str,
    required=True,
    help="Which substrate to train on. e.g. 'clean_up'"
  )
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
    help="Number of rollout workers, should be in [0,num_cpus]")
  parser.add_argument(
    "--num_samples", type=int, default=3, help="Number of samples to run for scratch training")
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
    "--policy_checkpoint",
    type=str,
    default=None,
    help="A path to a policy checkpoint to load the weights from")
  parser.add_argument(
    "--wandb",
    type=str,
    default=None,
    help="wandb project name")
  parser.add_argument(
    "--resume",
    action="store_true",
    help="Resume the last trial with local_dir/name")
  parser.add_argument(
    "--trial_id",
    type=str,
    default=None,
    help="Trial id to resume training from (if we had an error)")
  parser.add_argument(
    "--training",
    type=str,
    choices=["training", "pre-training", "scratch"],
    required=True,
    help="Whether to pre-train a policy, iteratively decrease the self-interest, or validate from scratch")
  parser.add_argument(
    "--training_mode",
    type=str,
    choices=["independent", "self-play"],
    required=True,
    help="self-play enables parameter sharing")
  parser.add_argument(
    "--n",
    type=int,
    default=None,
    help="Number of players")
  parser.add_argument(
    "--s",
    type=float,
    default=None,
    help="Pre-trained self-interest level (resuming crashed training)")

  args = parser.parse_args()

  ray.init(
      address="local",
      num_cpus=args.num_cpus,
      num_gpus=args.num_gpus,
      logging_level=LOGGING_LEVEL,
      _temp_dir=args.tmp_dir)

  register_env("meltingpot", utils.env_creator)

  substrate_config = substrate.get_config(args.substrate)
  default_player_roles = substrate_config.default_player_roles
  env_module = importlib.import_module(
      f"meltingpot.configs.substrates.{args.substrate}")
  substrate_definition = env_module.build(default_player_roles, substrate_config)

  horizon = substrate_definition["maxEpisodeLengthFrames"]
  sprite_size = substrate_definition["spriteSize"]

  env_config = ConfigDict({
      "substrate": args.substrate,
      "substrate_config": substrate_config,
      "roles": default_player_roles,
      "scaled": 1
  })

  base_env = utils.env_creator(env_config)

  rgb_shape = base_env.observation_space["player_0"]["RGB"].shape
  sprite_x = rgb_shape[0]
  sprite_y = rgb_shape[1]

  if sprite_size == 8:
    conv_filters = [[16, [8, 8], 8], [32, [4, 4], 1],
                    [64, [sprite_x // sprite_size, sprite_y // sprite_size], 1]]
  elif sprite_size == 1:
    conv_filters = [[16, [3, 3], 1], [32, [3, 3], 1],
                    [64, [sprite_x, sprite_y], 1]]
  else:
    assert False, "Unknown sprite_size of {sprite_size}"

  DEFAULT_MODEL = {
      "conv_filters": conv_filters,
      "conv_activation": "relu",
      "post_fcnet_hiddens": [64, 64],
      "post_fcnet_activation": "relu",
      "vf_share_layers": True,
      "use_lstm": True,
      "lstm_use_prev_action": False,
      # "lstm_use_prev_action": True,
      "lstm_use_prev_reward": False,
      "lstm_cell_size": 128,
  }

  train_batch_size = max(1,
                         args.rollout_workers) * args.envs_per_worker * horizon * args.episodes_per_worker

  if args.training_mode == "independent":
    def policy_mapping_fn(aid, *_, **__):
      return aid
  else:
    # pylint: disable=unused-argument
    def policy_mapping_fn(aid, *_, **__):
      return "default"

  config = PPOConfig().training(
    gamma=0.999,
    train_batch_size=train_batch_size,
    model=DEFAULT_MODEL,
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
    # observation_filter="MeanStdFilter",
  ).multi_agent(
    count_steps_by="env_steps",
    policy_mapping_fn=policy_mapping_fn,
  ).fault_tolerance(
    recreate_failed_env_runners=True,
    num_consecutive_env_runner_failures_tolerance=3,
  ).environment(
    env="meltingpot",
  ).debugging(
    log_level=LOGGING_LEVEL,
  ).resources(
    num_gpus=min(args.num_gpus, 1),
  ).framework(
    framework=args.framework,
  ).reporting(
    metrics_num_episodes_for_smoothing=1,
  )

  def custom_trial_name_creator(trial: Trial) -> str:
    """Create a custom name that includes hyperparameters."""
    trial_name = f"{trial.trainable_name}_{trial.config['TRIAL_ID']}"
    trial_name += f"_n={len(trial.config['env_config'].get('roles'))}"

    self_interest = trial.config["env_config"].get("self-interest")
    if self_interest is not None:
      trial_name += f"_s={self_interest:.3f}"
    else:
      trial_name += "_s=1.000"

    # This is here because my first self-play training did not include "_self-play"
    if trial.config.get("training-mode") == "independent":
      trial_name += "_independent"

    if trial.config.get("training") == "scratch":
      trial_name += "_scratch"

    return trial_name

  checkpoint_config = CheckpointConfig(
      num_to_keep=1,
      checkpoint_frequency=100,
      checkpoint_at_end=True)

  tune_callbacks = [
      WandbLoggerCallback(
          project=args.wandb,
          api_key=os.environ["WANDB_API_KEY"],
          log_config=False)
  ] if args.wandb is not None else None


  config = config.callbacks(
    ray.rllib.algorithms.callbacks.make_multi_callbacks(
      [SaveResultsCallback, LoadPolicyCallback])
  )


  TRIAL_ID = args.trial_id if args.trial_id else Trial.generate_id()
  config["TRIAL_ID"] = TRIAL_ID
  name = os.path.join(args.substrate, TRIAL_ID)
  working_folder = os.path.join(args.local_dir, name)
  config["working_folder"] = working_folder
  checkpoints_log_filepath = os.path.join(working_folder, "checkpoints.json")

  config["training-mode"] = args.training_mode

  if args.training == "pre-training":
    # a passed in trial_id means training had an error and we wish to resume
    if args.trial_id is not None:
      with open(checkpoints_log_filepath, mode="r", encoding="utf8") as f:
        info = json.loads(f.readlines()[-1])
        self_interest = info["self-interest"]
        if self_interest != 1:
          env_config["self-interest"] = self_interest
        start_n = info["num_players"] + 1
        config["policy_checkpoint"] = info["policy_checkpoint"]
    else:
      start_n = 1

    for n in range(start_n, len(default_player_roles) + 1):
      env_config["roles"] = substrate_config.default_player_roles[0:n]

      if args.training_mode == "independent":
        lr = 7e-5
        policies = {aid: PolicySpec() for aid in base_env._ordered_agent_ids[0:n]}
      else:
        lr = 7e-5 / n
        policies = {"default": PolicySpec()}

      config = config.training(
        lr=lr
      ).environment(
        env_config=env_config,
      ).multi_agent(
        policies=policies,
      )

      experiment = tune.run(
        run_or_experiment="PPO",
        name=name,
        metric="env_runners/episode_reward_mean",
        mode="max",
        stop={"training_iteration": args.n_iterations},
        config=config,
        storage_path=args.local_dir,
        checkpoint_config=checkpoint_config,
        verbose=VERBOSE,
        trial_name_creator=custom_trial_name_creator,
        trial_dirname_creator=custom_trial_name_creator,
        log_to_file=False,
        callbacks=tune_callbacks,
        max_concurrent_trials=args.max_concurrent_trials,
        # resume=args.resume,
      )

      policy_checkpoint = experiment.trials[-1].checkpoint.path
      config["policy_checkpoint"] = policy_checkpoint

      # checkpoint logging
      info = {}
      self_interest = config.env_config.get("self-interest")
      info["self-interest"] = 1 if self_interest is None else self_interest
      info["num_players"] = len(config.env_config["roles"])
      info["policy_checkpoint"] = policy_checkpoint
      info["training-mode"] = config.get("training-mode")
      with open(checkpoints_log_filepath, mode="a", encoding="utf8") as f:
        json.dump(info, f)
        f.write("\n")

  if args.training == "training":
    assert args.trial_id is not None
    assert args.n is not None
    n = args.n
    env_config["roles"] = substrate_config.default_player_roles[0:n]

    df = pd.read_json(checkpoints_log_filepath, lines=True)
    condition = (df["num_players"] == n) & \
      (df["training-mode"] == args.training_mode)
    assert df[condition]["self-interest"].is_unique, f"Duplicate checkpoints found in {df[condition]}"
    policy_checkpoint = df.loc[df[condition]["self-interest"].idxmin()]["policy_checkpoint"]
    config["policy_checkpoint"] = policy_checkpoint

    if args.training_mode == "independent":
      lr = 7e-5
      policies = dict((aid, PolicySpec()) for aid in base_env._ordered_agent_ids[0:n])
    else:
      lr = 7e-5 / n
      policies = {"default": PolicySpec()}

    config = config.training(
      lr=lr
    ).multi_agent(
      policies=policies,
    )

    ratio = [20, 10, 5, 3, 5/2, 2, 5/3, 4/3, 1]
    # ratio = [20, 10, 5, 3, 5/2]
    # ratio = [2, 5/3, 4/3, 1]
    for s in [r / (n + r - 1) for r in ratio]:
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
        checkpoint_config=checkpoint_config,
        verbose=VERBOSE,
        trial_name_creator=custom_trial_name_creator,
        trial_dirname_creator=custom_trial_name_creator,
        log_to_file=False,
        callbacks=tune_callbacks,
        max_concurrent_trials=args.max_concurrent_trials,
        # resume=args.resume,
      )

      policy_checkpoint = experiment.trials[-1].checkpoint.path
      config["policy_checkpoint"] = policy_checkpoint

      # checkpoint logging
      info = {}
      self_interest = config.env_config.get("self-interest")
      info["self-interest"] = 1 if self_interest is None else self_interest
      info["num_players"] = len(config.env_config["roles"])
      info["policy_checkpoint"] = policy_checkpoint
      info["training-mode"] = args.training_mode
      with open(checkpoints_log_filepath, mode="a", encoding="utf8") as f:
        json.dump(info, f)
        f.write("\n")

  if args.training == "scratch":
    assert args.n is not None
    assert args.s is not None
    n = args.n
    env_config["roles"] = substrate_config.default_player_roles[0:n]

    if args.training_mode == "independent":
      lr = 7e-5
      policies = dict((aid, PolicySpec()) for aid in base_env._ordered_agent_ids[0:n])
    else:
      lr = 7e-5 / n
      policies = {"default": PolicySpec()}

    config = config.training(
      lr=lr
    ).multi_agent(
      policies=policies,
    )

    env_config["self-interest"] = args.s

    config = config.environment(env_config=env_config)

    experiment = tune.run(
      run_or_experiment="PPO",
      name=name,
      metric="env_runners/episode_reward_mean",
      mode="max",
      stop={"training_iteration": args.n_iterations},
      config=config,
      storage_path=args.local_dir,
      checkpoint_config=checkpoint_config,
      verbose=VERBOSE,
      trial_name_creator=custom_trial_name_creator,
      trial_dirname_creator=custom_trial_name_creator,
      log_to_file=False,
      callbacks=tune_callbacks,
      max_concurrent_trials=args.max_concurrent_trials,
      num_samples=args.num_samples,
    )

  ray.shutdown()

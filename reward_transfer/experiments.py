"""Run experiments"""

import argparse
from collections import defaultdict
import importlib
import os
from typing import Dict

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

LOGGING_LEVEL = "INFO"
VERBOSE = 1

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
  # parser.add_argument(
  #   "--num_samples", type=int, default=1, help="Number of samples to run")
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
    help="Resume the last trial with name/local_dir")

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
    policies={"default": PolicySpec()},
    policy_mapping_fn=lambda aid, *args, **kwargs: "default",
    count_steps_by="env_steps",
  ).fault_tolerance(
    recreate_failed_env_runners=True,
    num_consecutive_env_runner_failures_tolerance=3,
  ).environment(
    env="meltingpot",
  ).debugging(
    log_level=LOGGING_LEVEL,
  ).resources(
    num_gpus=args.num_gpus,
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
      trial_name += "_s=0.000"

    return trial_name

  checkpoint_config = CheckpointConfig(
      num_to_keep=None,
      checkpoint_frequency=args.n_iterations,
      checkpoint_at_end=True)

  tune_callbacks = [
      WandbLoggerCallback(
          project=args.wandb,
          api_key=os.environ["WANDB_API_KEY"],
          log_config=False)
  ] if args.wandb is not None else None


  config["results_folder"] = os.path.join(args.local_dir, args.substrate)

  config = config.callbacks(
    ray.rllib.algorithms.callbacks.make_multi_callbacks(
      [SaveResultsCallback, LoadPolicyCallback])
  )


  config["TRIAL_ID"] = Trial.generate_id()
  for n in range(1, len(default_player_roles) + 1):
    env_config["roles"] = substrate_config.default_player_roles[0:n]

    config = config.training(
      lr=7e-5 / n,
    ).environment(
      env_config=env_config,
    )

    experiment = tune.run(
      run_or_experiment="PPO",
      name=args.substrate,
      metric="env_runners/episode_reward_mean",
      mode="max",
      stop={"training_iteration": args.n_iterations},
      config=config,
      storage_path=args.local_dir,
      checkpoint_config=checkpoint_config,
      verbose=VERBOSE,
      trial_name_creator=custom_trial_name_creator,
      log_to_file=False,
      callbacks=tune_callbacks,
      max_concurrent_trials=args.max_concurrent_trials,
      # resume=args.resume,
    )

    checkpoint = experiment.trials[-1].checkpoint

    config["policy_checkpoint"] = os.path.join(checkpoint.path, "policies/default")


  # run_config = RunConfig(
  #     name=args.substrate,
  #     local_dir=args.local_dir,
  #     stop={"training_iteration": args.n_iterations},
  #     checkpoint_config=checkpoint_config,
  #     verbose=VERBOSE)

  # tune_config = tune.TuneConfig(
  #   num_samples=args.num_samples,
  #   search_alg=optuna_search,
  #   scheduler=asha_scheduler)

  # tuner = tune.Tuner(
  #     "PPO", param_space=config, tune_config=tune_config, run_config=run_config)

  # results = tuner.fit()

  # best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
  # print(best_result)

  ray.shutdown()

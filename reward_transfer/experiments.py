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
from ray.tune.registry import register_env
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

from examples.rllib import utils
from reward_transfer.callbacks import make_rt_callback, LoadPolicyCallback

# Use Tuner.fit() to gridsearch over exchange values
# Thus I need to stick a custom parameter in the config and hope I can access this in the callback
# worry about loading pre-training later

LOGGING_LEVEL = "WARN"
VERBOSE = 1
KEEP_CHECKPOINTS_NUM = None  # Default None
CHECKPOINT_FREQ = 0  # Default 0

EXPLORE_EVAL = False
# TODO: Fix evaluation at end of training
EVAL_DURATION = 80

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
    "--num_samples", type=int, default=1, help="Number of samples to run")
  parser.add_argument(
    "--envs_per_worker",
    type=int,
    default=1,
    help="Number of episodes each worker runs in parallel")
  parser.add_argument(
    "--episodes_per_worker",
    type=int,
    default=1,
    help="Number of episodes per each worker in a training batch (not including parallelism)")
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
    "--independent",
    action="store_true",
    help="Training is independent (not self-play) with n policies")
  parser.add_argument(
    "--resume",
    action="store_true",
    help="Resume the last trial with name/local_dir")
  parser.add_argument(
    "--reward_transfer",
    type=float,
    default=None,
    help="Self-interest of the agents")
  parser.add_argument(
    "--optimiser",
    action="store_true",
    help="Use an optimiser for hyper parameter tuning")

  args = parser.parse_args()

  ray.init(
      address="local",
      num_cpus=args.num_cpus,
      num_gpus=args.num_gpus,
      logging_level=LOGGING_LEVEL,
      _temp_dir=args.tmp_dir)

  register_env("meltingpot", utils.env_creator)

  substrate_config = substrate.get_config(args.substrate)
  player_roles = substrate_config.default_player_roles
  num_players = len(player_roles)

  # 1. import the module.
  # 2. call build
  env_module = importlib.import_module(
      f"meltingpot.configs.substrates.{args.substrate}")
  substrate_definition = env_module.build(player_roles, substrate_config)

  horizon = substrate_definition["maxEpisodeLengthFrames"]
  sprite_size = substrate_definition["spriteSize"]

  # TODO: SPRITE_SIZE
  env_config = ConfigDict({
      "substrate": args.substrate,
      "substrate_config": substrate_config,
      "roles": player_roles,
      "scaled": 1
  })

  base_env = utils.env_creator(env_config)

  unique_roles: Dict[str, list] = defaultdict(list)
  for i, (role, pid) in enumerate(zip(player_roles, base_env._ordered_agent_ids)):
    unique_roles[role].append(pid)


  if args.independent:
    policies = dict((aid, PolicySpec()) for aid in base_env._ordered_agent_ids)

    def policy_mapping_fn(aid, *args, **kwargs):
      return aid
  else:
    policies = dict((role, PolicySpec()) for role in unique_roles)

    def policy_mapping_fn(aid, *args, **kwargs):
      for role, pids in unique_roles.items():
        if aid in pids:
          return role
      assert False, f"Agent id {aid} not found in unique roles {unique_roles}"

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

  parallelism = max(1, args.num_cpus // (1 + args.rollout_workers))

  train_batch_size = max(1,
                         args.rollout_workers) * args.envs_per_worker * horizon * args.episodes_per_worker

  config = PPOConfig().training(
    model=DEFAULT_MODEL,
    train_batch_size=train_batch_size,
    entropy_coeff=1e-3,
  ).rollouts(
    batch_mode="complete_episodes",
    num_rollout_workers=args.rollout_workers,
    rollout_fragment_length=100,
    num_envs_per_worker=args.envs_per_worker,
  ).multi_agent(
    policies=policies,
    policy_mapping_fn=policy_mapping_fn,
    count_steps_by="env_steps",
  ).fault_tolerance(
    recreate_failed_workers=True,
    num_consecutive_worker_failures_tolerance=3,
  ).environment(
    env="meltingpot",
    env_config=env_config,
  ).debugging(
    log_level=LOGGING_LEVEL,
  ).resources(
    num_gpus=args.num_gpus / parallelism,
  ).framework(
    framework=args.framework,
  ).reporting(
    metrics_num_episodes_for_smoothing=1,
  ).evaluation(
    evaluation_interval=None,  # don't evaluate unless we call evaluation()
    evaluation_config={
        "explore": EXPLORE_EVAL,
    },
    evaluation_duration=EVAL_DURATION,
  ).experimental(
    # will be set to true in future versions of Ray, was True in baselines
    # I don't know how to get this to work though - and I can't use the baselines
    # policy wrapper either without it
  _disable_preprocessor_api=False)

  algo_callbacks = []

  if args.reward_transfer:
    if args.independent:
      tm = dict((aid, args.reward_transfer) for aid in base_env._ordered_agent_ids)
    else:
      tm = dict((role, args.reward_transfer) for role in unique_roles)
    rt_callback = make_rt_callback(tm, False)
    algo_callbacks.append(rt_callback)

  if args.policy_checkpoint:
    config["policy_checkpoint"] = args.policy_checkpoint
    algo_callbacks.append(LoadPolicyCallback)

  if algo_callbacks:
    if len(algo_callbacks) > 1:
      CombinedCallback = type("CombinedCallback", tuple(algo_callbacks), {})
      config = config.callbacks(CombinedCallback)
    else:
      config = config.callbacks(algo_callbacks[0])

  checkpoint_config = CheckpointConfig(
      num_to_keep=KEEP_CHECKPOINTS_NUM,
    checkpoint_frequency=CHECKPOINT_FREQ,
      checkpoint_at_end=True)

  if args.optimiser:
    config = config.training(
      sgd_minibatch_size=tune.qrandint(10000, 30000, 10000),
      num_sgd_iter=tune.qlograndint(5, 20, 1),
      lr=tune.qloguniform(1e-5, 1e-3, 1e-5),
      lambda_=tune.quniform(0.9, 1.0, 0.05),
      vf_loss_coeff=tune.quniform(0.5, 1, 0.1),
      clip_param=tune.quniform(0.1, 0.5, 0.05),
      vf_clip_param=tune.qlograndint(1, 20, 1),
    )

    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        metric="episode_reward_mean",
        mode="max",
        max_t=args.n_iterations,
        grace_period=max(1, args.n_iterations // 2),
        reduction_factor=2,
        brackets=1,
    )

    search_alg = OptunaSearch(
      metric="episode_reward_mean",
      mode="max",
      points_to_evaluate=[
        {"sgd_minibatch_size": 20000, "num_sgd_iter": 12, "lr": 0.000126, "lambda": 0.95, "vf_loss_coeff": 0.7, "clip_param": 0.25, "vf_clip_param": 2},
        {"sgd_minibatch_size": 10000, "num_sgd_iter": 13, "lr": 0.000217, "lambda": 0.90, "vf_loss_coeff": 0.7, "clip_param": 0.25, "vf_clip_param": 5},
      ],
    )

    def trial_name_string(trial: ray.tune.experiment.Trial):
      """Create a custom name that includes hyperparameters."""
      attributes = ["sgd_minibatch_size", "num_sgd_iter", "lr", "lambda",
                    "vf_loss_coeff", "clip_param", "vf_clip_param"]
      attributes_str = [f"{trial.config[a]:.5f}".rstrip("0").rstrip(".") for a in attributes]
      return f"{trial.trainable_name}_{trial.trial_id}_{','.join(attributes_str)}"

    metric = None
    mode = None
  else:
    config = config.training(
      sgd_minibatch_size=200,
      num_sgd_iter=12,
      lr=1e-5,
      lambda_=0.925,
      vf_loss_coeff=0.75,
      clip_param=0.25,
      vf_clip_param=5,
    )

    metric = "episode_reward_mean"
    mode = "max"
    search_alg = None
    scheduler = None
    trial_name_string = None


  tune_callbacks = [
      WandbLoggerCallback(
          project=args.wandb,
          api_key=os.environ["WANDB_API_KEY"],
          log_config=False)
  ] if args.wandb is not None else None

  experiment = tune.run(
    run_or_experiment="PPO",
    name=args.substrate,
    metric=metric,
    mode=mode,
    stop={"training_iteration": args.n_iterations},
    config=config,
    num_samples=args.num_samples,
    storage_path=args.local_dir,
    search_alg=search_alg,
    scheduler=scheduler,
    checkpoint_config=checkpoint_config,
    verbose=VERBOSE,
    trial_name_creator=trial_name_string,
    log_to_file=False,
    callbacks=tune_callbacks,
    max_concurrent_trials=args.max_concurrent_trials,
    resume=args.resume,
  )

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

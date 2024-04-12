"""Run experiments"""


import argparse
from collections import defaultdict
from typing import Dict

from meltingpot import substrate
from ml_collections.config_dict import ConfigDict
import ray
from ray import tune
from ray.air import CheckpointConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

from examples.rllib import utils
from reward_transfer.callbacks import MyCallbacks

# Use Tuner.fit() to gridsearch over exchange values
# Thus I need to stick a custom parameter in the config and hope I can access this in the callback
# worry about loading pre-training later

LOGGING_LEVEL = "WARN"
VERBOSE = 1
KEEP_CHECKPOINTS_NUM = 1  # Default None
CHECKPOINT_FREQ = 50  # Default 0

NUM_ENVS_PER_WORKER = 1
SGD_MINIBATCH_SIZE = 250  # 256 = minimum for efficient CPU training
LR = 2e-4
VF_CLIP_PARAM = 2.0
NUM_SGD_ITER = 10
EXPLORE_EVAL = False
ENTROPY_COEFF = 0.003
# TODO: Fix evaluation at end of training
EVAL_DURATION = 80

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--substrate",
      type=str,
      required=True,
      help="Which substrate to train on. e.g. 'coins' or 'allelopathic_harvest__open'.")
  parser.add_argument(
      "--n_iterations",
      type=int,
      required=True,
      help="number of training iterations to use")
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
      "--num_samples",
      type=int,
      default=1,
      help="Number of samples to run")
  parser.add_argument(
      "--episodes_per_worker",
      type=int,
      default=1,
      help="Number of episodes per each rollout worker in a training batch")
  parser.add_argument(
      "--max_concurrent_trials",
      type=int,
      default=None,
      help="maximum number of concurrent trials to run")
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

  # TODO: Fix if multiple roles

  substrate_config = substrate.get_config(args.substrate)
  player_roles = substrate_config.default_player_roles
  num_players = len(player_roles)
  unique_roles: Dict[str, list] = defaultdict(list)
  for i, role in enumerate(player_roles):
    unique_roles[role].append(f"player_{i}")

    # 1. import the module.
    # 2. call build
  import importlib

  env_module = importlib.import_module(f"meltingpot.configs.substrates.{args.substrate}")
  substrate_definition = env_module.build(player_roles, substrate_config)
  horizon = substrate_definition["maxEpisodeLengthFrames"]
  sprite_size = substrate_definition["spriteSize"]

  def policy_mapping_fn(aid, *args, **kwargs):
    for role, pids in unique_roles.items():
      if aid in pids:
        return role
    assert False

  # TODO: SPRITE_SIZE
  env_config = ConfigDict({
      "substrate": args.substrate,
      "substrate_config": substrate_config,
      "roles": player_roles,
      "scaled": 1
  })

  base_env = utils.env_creator(env_config)
  policies = {}
  for i, role in enumerate(unique_roles):
    rgb_shape = base_env.observation_space[f"player_{i}"]["RGB"].shape
    sprite_x = rgb_shape[0]
    sprite_y = rgb_shape[1]

    policies[role] = PolicySpec(
        observation_space=base_env.observation_space[f"player_{i}"],
        action_space=base_env.action_space[f"player_{i}"],
        config={
        })

  if sprite_size == 8:
    conv_filters = [[16, [8, 8], 8],
                    [32, [4, 4], 1],
                    [64, [sprite_x // sprite_size, sprite_y // sprite_size], 1]]
  elif sprite_size == 1:
    conv_filters = [[16, [3, 3], 1],
                    [32, [3, 3], 1],
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

  assert args.episodes_per_worker >= NUM_ENVS_PER_WORKER

  # TODO: Get maxEpisodeLengthFrames from substrate definition
  train_batch_size = max(
      1, args.rollout_workers) * args.episodes_per_worker * horizon

  config = PPOConfig().training(
      model=DEFAULT_MODEL,
      train_batch_size=train_batch_size,
      sgd_minibatch_size=min(SGD_MINIBATCH_SIZE, train_batch_size),
      num_sgd_iter=NUM_SGD_ITER,
      # lr=LR,
      # lr=tune.loguniform(1e-5, 5e-4),
      lr=tune.grid_search([1e-5, 3e-5, 1e-4, 3e-4]),
      # lambda_=0.80,
      # lambda_=tune.uniform(0.5, 1),
      lambda_=tune.grid_search([0.5, 1]),
      # vf_loss_coeff=0.5,
      # vf_loss_coeff=tune.uniform(0.2, 1),
      vf_loss_coeff=tune.grid_search([0.2, 0.6, 1]),
      # entropy_coeff=ENTROPY_COEFF,
      # entropy_coeff=tune.loguniform(3e-4, 3e-2),
      entropy_coeff=tune.grid_search([3e-4, 3e-2]),
      # clip_param=0.2,
      # clip_param=tune.uniform(0.2, 0.4),
      clip_param=tune.grid_search([0.2, 0.4]),
      # vf_clip_param=VF_CLIP_PARAM,
      # vf_clip_param=tune.uniform(1, 20),
      vf_clip_param=tune.grid_search([1, 20]),
  ).rollouts(
      batch_mode="complete_episodes",
      num_rollout_workers=args.rollout_workers,
      rollout_fragment_length=100,
      num_envs_per_worker=NUM_ENVS_PER_WORKER,
  ).multi_agent(
      policies=policies,
      policy_mapping_fn=policy_mapping_fn,
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
      num_gpus_per_learner_worker=args.num_gpus / parallelism,
  ).framework(
      framework="tf",
      eager_tracing=True,
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
    _disable_preprocessor_api=False
  )

  # each worker will get its own copy of MyCallbacks
  # and careful about setting the value of a mutable class member

  # MyCallbacks.set_transfer_map = {"default": 0.5}
  # for role in unique_roles:
  #   MyCallbacks.transfer_map[role] = 0.5
  config = config.callbacks(MyCallbacks)

  checkpoint_config = CheckpointConfig(
      num_to_keep=KEEP_CHECKPOINTS_NUM,
      checkpoint_frequency=CHECKPOINT_FREQ,
      checkpoint_at_end=True)

  asha_scheduler = ASHAScheduler(
      time_attr='training_iteration',
      metric='episode_reward_mean',
      mode='max',
      max_t=args.n_iterations,
      grace_period=max(1, args.n_iterations//4),
      reduction_factor=2,
      brackets=1,
  )

  optuna_search = OptunaSearch(
      metric='episode_reward_mean',
      mode='max',
  )

  experiment = tune.run(
    run_or_experiment="PPO",
    name=args.substrate,
    # metric="episode_reward_mean",
    # mode="max",
    stop={"training_iteration": args.n_iterations},
    config=config,
    num_samples=args.num_samples,
    storage_path=args.local_dir,
    # search_alg=optuna_search,
    # scheduler=asha_scheduler,
    checkpoint_config=checkpoint_config,
    verbose=VERBOSE,
    log_to_file=False,
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

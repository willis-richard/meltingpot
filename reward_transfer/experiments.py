# Copyright 2020 DeepMind Technologies Limited.

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Runs an example of a self-play training experiment."""

import argparse
import copy
import math
from typing import List, Optional

import numpy as np
import pandas as pd
import ray
from ray import tune
from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.policy.policy import Policy, PolicySpec
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.tune.registry import register_env


from examples.rllib import utils
from reward_transfer.common import create_env_config_coins, CUSTOM_MODEL, DEFAULT_MODEL, LOGGING_LEVEL, VERBOSE


def make_2p_rs(k_0: float, k_1: Optional[float] = None):
  if k_1 is None:
    k_1 = k_0
  return [[1 - k_0, k_1], [k_0, 1 - k_1]]


# Can always see these choices in the params.json file of a trial.
ALGO = "PPO"
NUM_WORKERS = 0
NUM_ENVS_PER_WORKER = 8
NUM_EPISODES_PER_WORKER = 1

N_SAMPLES = 5
EVAL_DURATION = 80
KEEP_CHECKPOINTS_NUM = None  # Default None
CHECKPOINT_FREQ = 0  # Default 0

NUM_GPUS = 0
SGD_MINIBATCH_SIZE = 4096  # 256 = minimum for efficient CPU training
LR = 2e-4
VF_CLIP_PARAM = 2.0
NUM_SGD_ITER = 10
EXPLORE_EVAL = False
ENTROPY_COEFF = 0.003


def main():
  """Evaluate different values of reward transfer."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--num_cpus", type=int, required=True, help="number of CPUs to use")
  parser.add_argument(
      "--local_dir",
      type=str,
      required=True,
      help="This is the path the results will be saved to, "
      "defaults to ~/ray_results/")
  parser.add_argument(
      "--selfp_episodes",
      type=int,
      default=0,
      help="number of initial self play pre-training episodes")
  parser.add_argument(
      "--indepdndent_episodes",
      type=int,
      default=4000,
      help="number of training episodes")
  parser.add_argument(
      "--regrowth_probability",
      type=float,
      default=0.15,
      help="probability of an apple re-spawning")
  parser.add_argument(
      "--gifting",
      action="store_true",
      help="Whether to use reward gifting (default: reward exchange)")
  parser.add_argument(
      "--evaluate",
      action="store_true",
      help="Whether to evaluate the trained models")
  parser.add_argument(
      "--tmp_dir",
      type=str,
      default=None,
      help="Custom tmp location for temporary ray logs")

  args = parser.parse_args()

  ks = np.round(np.arange(0.0, 1.01, 0.1), 2) if args.gifting else np.round(
      np.arange(0.0, 0.51, 0.05), 2)

  ray.init(
      address="local",
      num_cpus=args.num_cpus,
      num_gpus=NUM_GPUS,
      logging_level=LOGGING_LEVEL,
      _temp_dir=args.tmp_dir)

  register_env("meltingpot", utils.env_creator)

  env_eval_config = create_env_config_coins(make_2p_rs(0))

  # Extract space dimensions
  player_roles = env_eval_config.default_player_roles
  test_env = utils.env_creator(env_eval_config)

  POLICIES = {}
  for i in range(len(player_roles)):
    rgb_shape = test_env.observation_space[f"player_{i}"]["RGB"].shape
    sprite_x = rgb_shape[0] // 8
    sprite_y = rgb_shape[1] // 8

    POLICIES[f"player_{i}"] = PolicySpec(
        policy_class=None,  # use default policy
        observation_space=test_env.observation_space[f"player_{i}"],
        action_space=test_env.action_space[f"player_{i}"],
        config={
            # "model": {
            #     "conv_filters": [[16, [8, 8], 8],
            #                      [128, [sprite_x, sprite_y], 1]],
            # },
        })

  # No longer in the config (was lab2d_settings)
  # horizon = env_eval_config.substrate_definition["maxEpisodeLengthFrames"]
  horizon = 500
  train_batch_size = max(
      1, NUM_WORKERS) * NUM_ENVS_PER_WORKER * NUM_EPISODES_PER_WORKER * horizon

  max_concurrent_algos = min(
      len(ks) * N_SAMPLES, math.floor(args.num_cpus / (1 + NUM_WORKERS)))
  num_gpus_per_algo = NUM_GPUS / max_concurrent_algos

  config = PPOConfig().training(
      model=DEFAULT_MODEL,
      lr=LR,
      train_batch_size=train_batch_size,
      lambda_=0.80,
      vf_loss_coeff=0.5,
      entropy_coeff=ENTROPY_COEFF,
      clip_param=0.2,
      vf_clip_param=VF_CLIP_PARAM,
      sgd_minibatch_size=min(SGD_MINIBATCH_SIZE, train_batch_size),
      num_sgd_iter=NUM_SGD_ITER,

      # grad_clip=40, # THIS
      # # microbatch_size=min(SGD_MINIBATCH_SIZE, train_batch_size),
  ).rollouts(
      batch_mode="complete_episodes",
      num_rollout_workers=NUM_WORKERS,
      rollout_fragment_length=100,
      num_envs_per_worker=NUM_ENVS_PER_WORKER,
  ).fault_tolerance(
      recreate_failed_workers=True,
      num_consecutive_worker_failures_tolerance=3,
  ).environment(
    env="meltingpot",
    env_config=env_eval_config,
  ).debugging(log_level=LOGGING_LEVEL,
  ).resources(num_gpus=num_gpus_per_algo,
  ).framework(framework="tf",
  ).reporting(metrics_num_episodes_for_smoothing=1,
  ).evaluation(
      evaluation_interval=None,  # don't evaluate unless we call evaluation()
      evaluation_config={
          "explore": EXPLORE_EVAL,
          "env_config": env_eval_config,
      },
      evaluation_duration=EVAL_DURATION,
  )


  if args.selfp_episodes:
    num_epochs = math.ceil(args.selfp_episodes /
                           (max(1, NUM_WORKERS) * NUM_ENVS_PER_WORKER))

    @ray.remote
    def run_selfp(k: float, num_epochs: int, config: AlgorithmConfig,
                  trial_id: int):
      name = f"{ALGO}_selfp_{k}_{trial_id}"
      checkpoints = []

      for player, spec in POLICIES.items():
        config = config.multi_agent(
          policies={player: spec},
          policy_mapping_fn=lambda aid, episode, worker, **kwargs: player)

        if args.gifting and player == "player_1":
          config = config.environment(
              env_config=create_env_config_coins(make_2p_rs(0)))
        else:
          config = config.environment(
              env_config=create_env_config_coins(make_2p_rs(k)))

        trial = tune.run(
            ALGO,
            stop={"training_iteration": num_epochs},
            checkpoint_at_end=True,
            config=config,
            metric="episode_reward_mean",
            mode="max",
            log_to_file=False,
            local_dir=args.local_dir,
            name=name,
            verbose=VERBOSE,
        ).trials[-1]

        checkpoints += [trial.checkpoint.dir_or_data]

      return checkpoints

    checkpoints = ray.get([
        run_selfp.remote(ks[i], num_epochs, copy.deepcopy(config), j)
        for i in range(len(ks)) for j in range(N_SAMPLES)
    ])

  else:
    checkpoints = [None] * N_SAMPLES * len(ks)

  # train independent agents
  @ray.remote
  def run_indep(k: float, checkpoints: Optional[List[str]], num_epochs: int,
                config: AlgorithmConfig, trial_id: int):
    name = f"{ALGO}_indep_{k}_{trial_id}"

    if args.gifting:
      config = config.environment(
          env_config=create_env_config_coins(make_2p_rs(k_0=k, k_1=0)))
    else:
      config = config.environment(
          env_config=create_env_config_coins(make_2p_rs(k)))

    if checkpoints is not None:
      class MyCallbacks(DefaultCallbacks):
        def __init__(self):
          super().__init__()
          self.checkpoints = checkpoints
          self.policy_mapping_fn = lambda aid, episode, worker, **kwargs: aid

        def on_algorithm_init(
            self,
            *,
            algorithm: "Algorithm",
            **kwargs,
        ) -> None:
          for checkpoint in self.checkpoints:
            policy = Policy.from_checkpoint(checkpoint)
            for p_id, p in policy.items():
              algorithm.add_policy(p_id, policy=p)

          algorithm.remove_policy(DEFAULT_POLICY_ID,
                                  policy_mapping_fn=self.policy_mapping_fn,
                                  policies_to_train=list(POLICIES.keys()))

      config = config.callbacks(MyCallbacks)
    else:
      config = config.multi_agent(
          policies=POLICIES,
          policy_mapping_fn=lambda aid, episode, worker, **kwargs: aid)

    return tune.run(
        ALGO,
        stop={"training_iteration": num_epochs},
        keep_checkpoints_num=KEEP_CHECKPOINTS_NUM,
        checkpoint_freq=CHECKPOINT_FREQ,
        checkpoint_at_end=True,
        config=config,
        metric="episode_reward_mean",
        mode="max",
        log_to_file=False,
        local_dir=args.local_dir,
        name=name,
        verbose=VERBOSE,
    ).trials[-1]

  num_epochs = math.ceil(args.independent_episodes /
                         (max(1, NUM_WORKERS) * NUM_ENVS_PER_WORKER))

  trials = ray.get([
      run_indep.remote(ks[i], checkpoints[i * N_SAMPLES + j],
                       num_epochs, copy.deepcopy(config), j)
      for i in range(len(ks)) for j in range(N_SAMPLES)
  ])

  if args.evaluate:
    # # TODO: necessary? was only a copy that was edited
    # # cleanup callbacks if used
    # config = config.callbacks(DefaultCallbacks)

    @ray.remote
    def run_evaluate(checkpoint):
      algo = config.build()
      algo.load_checkpoint(checkpoint)
      return algo.evaluate()["evaluation"]

    # evaluate agents
    results = {}
    names = []
    checkpoints = []
    for i, trial in enumerate(trials):
      k = ks[math.floor(i / N_SAMPLES)]
      for j, checkpoint in enumerate(trial.get_trial_checkpoints()):
        # k, trial, checkpoint
        names.append(f"{k}_{i % N_SAMPLES}_{j}")
        checkpoints.append(checkpoint.dir_or_data)

    results = ray.get(
        [run_evaluate.remote(c) for c in checkpoints])
    results_dict = dict(((n, r) for n, r in zip(names, results)))

    identifier = "stoch" if EXPLORE_EVAL else "det"
    pd.DataFrame(results_dict).to_csv(
        f"{args.local_dir}/df_indep_{identifier}.csv")

  ray.shutdown()


if __name__ == "__main__":
  main()

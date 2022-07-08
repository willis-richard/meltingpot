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
import random

from ray import init
from ray.rllib.agents.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.tune import run, sample_from
from ray.tune.registry import register_env
from ray.tune.schedulers.pb2 import PB2

from examples.rllib import config_creator, utils
from meltingpot.python import substrate

from tensorflow.keras.optimizers import RMSprop

DEFAULT_ALGORITHM = "A3C"

# Setup for the neural network.
# The strides of the first convolutional layer were chosen to perfectly line
# up with the sprites, which are 8x8.
# The final layer must be chosen specifically so that its output is
# [B, 1, 1, X]. See the explanation in
# https://docs.ray.io/en/latest/rllib-models.html#built-in-models. It is
# because rllib is unable to flatten to a vector otherwise.
# The a3c models used as baselines in the meltingpot paper were not run using
# rllib, so they used a different configuration for the second convolutional
# layer. It was 32 channels, [4, 4] kernel shape, and stride = 1.

# examples conv nets are in ray.rllib.models.utils.get_filter_config
custom_model = {
    "conv_filters": [[16, [8, 8], 8], [32, [3, 3], 1], [64, [5, 5], 1]],
    "conv_activation": "relu",
    "post_fcnet_hiddens": [64, 64],  # [64, 64]
    "post_fcnet_activation": "relu",
    "use_lstm": True,
    "lstm_use_prev_action": False,  # False
    "lstm_use_prev_reward": False,  # False
    "lstm_cell_size": 128,  # 128
    # This it a linear layer after the CNN that resizes the number
    # of outputs to match the post_fcnet_hiddens number of neurons.
    # I am removing it because I flatten the convnet output in the last filter
    # If False, the final number of channels in the conv net
    # must be the same as post_fcnet_hiddens
    "no_final_linear": True  # True
}

# TODO: skip first conv net by using the layers, say WORLD.LAYER?
# TODO: print out model and check where lstm is
# TODO: Be able to colour agents by policy
# TODO: Improve logging of results
# TODO: make apple re-spawn deterministic

# https://iq.opengenus.org/same-and-valid-padding/
# rllib gives padding "same" for the first n-1 layers,
# then padding valid on the last layer, before a possible final layer that
# changes the output before the hidden layers.
# see ray.rllib.models.torch.misc.same_padding to understand the padding


def main():
  """
  Example of a self-play training experiment with RLLib
  """

  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--algorithm",
      type=str,
      default=DEFAULT_ALGORITHM,
      help="ray.rllib.agents algorithm to use")
  parser.add_argument(
      "--cpus", type=int, default=7, help="number of cpus ray can use")
  parser.add_argument(
      "--restore",
      type=str,
      default=None,
      help="ray.tune checkpoint. point to the checkpoint that has an "
      "accompanying .tune_metadata. e.g. ~/ray_results/"
      f"{DEFAULT_ALGORITHM}/{DEFAULT_ALGORITHM}/"
      "checkpoint_XXX/checkpoint-XXX")
  parser.add_argument(
      "--use_scheduler",
      action="store_true",
      help="whether to use the preconfigured scheduler")
  parser.add_argument(
      "--use_optimiser",
      action="store_true",
      help="whether to use the preconfigured scheduler")
  parser.add_argument(
      "--local_dir",
      type=str,
      default=None,
      help="This is the path the results will be saved to, "
      "defaults to ~/ray_results/.")

  args = parser.parse_args()

  # Initialize ray
  init(configure_logging=True, num_cpus=args.cpus, logging_level="info")

  # Setup the training environment (MeltingPot subscrate)
  substrate_name = "commons_harvest_open_simple"
  env_config = substrate.get_config(substrate_name)
  register_env("meltingpot", utils.env_creator)

  # After n steps, force reset simulation
  # Each unroll happens exactly over one episode, from
  # beginning to end. Data collection will not stop unless the episode
  # terminates or a configured horizon (hard or soft) is hit.
  horizon = env_config.lab2d_settings["maxEpisodeLengthFrames"]
  lr = 5e-5 if args.algorithm == "PPO" else 4e-4

  # Extract space dimensions
  test_env = utils.env_creator(env_config)

  policies = {
      "av":
          PolicySpec(
              policy_class=None,  # use default policy
              observation_space=test_env.observation_space["player_0"],
              action_space=test_env.action_space["player_0"],
              config={}),
  }

  optimiser = RMSprop(
      learning_rate=lr,
      rho=0.99,  # discount factor/decay
      momentum=0.0,
      epsilon=1e-5) if args.use_optimiser else None


  config = PPOConfig().framework(
      framework="tf2",
      eager_tracing=True  # only applies if using tf2
  ).training(
      model=custom_model,
      lr=sample_from(lambda spec: random.uniform(1e-3, 1e-5)),
      gamma=0.99,  # Default=0.99
      train_batch_size=12 * horizon,  # Default=4000
      optimizer=optimiser,
      # PPO specifics
      use_critic=True,
      use_gae=True,
      lambda_=sample_from(lambda spec: random.uniform(0.9, 0.99)),
      kl_coeff=0.2,
      sgd_minibatch_size=128,  # Default=128
      num_sgd_iter=30,  # Default=30
      shuffle_sequences=True,  # recommended to be True
      vf_loss_coeff=1.0,  # coefficient of the value function loss. not used?
      # TODO: find out what a sensible value should be -> log out loss stats
      vf_clip_param=2.0,  # N.B. sensitive to reward scale.
      entropy_coeff=sample_from(lambda spec: random.uniform(0.03, 0.0003)),
      # entropy_coeff = grid_search([0.03, 0.01, 0.003, 0.001, 0.0003]),
      entropy_coeff_schedule=None,
      clip_param=sample_from(lambda spec: random.uniform(0.1, 0.5)),
      grad_clip=None,
      kl_target=0.01).rollouts(
          batch_mode="complete_episodes",
          horizon=horizon,
          num_rollout_workers=0,
          rollout_fragment_length=100,
      ).environment(
          env_config=env_config,
          env="meltingpot",
      ).multi_agent(
          count_steps_by="env_steps",
          policies=policies,
          policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: "av",
      ).debugging(log_level="INFO").to_dict()

  scheduler = PB2(
      time_attr="training_iteration",
      metric="episode_reward_mean",
      mode="max",
      perturbation_interval=10,
      hyperparam_bounds={
          "lr": [1e-3, 1e-5],
          "entropy_coeff": [0.03, 0.0003],
          "lambda": [0.9, 0.99],
          "clip_param": [0.1, 0.5],
      },
      quantile_fraction=0.25,
      log_config=False,  # True if you want to reconstruct the schedule
  )

  # TODO, what is observation_filter in trainer?
  # TODO: add pb2_ppo_example df analysis

  run(
      args.algorithm,
      stop={"training_iteration": 250},
      checkpoint_at_end=True,
      checkpoint_freq=10,
      config=config,
      # metric="episode_reward_mean",
      # mode="max",
      log_to_file=True,
      num_samples=4 if args.use_scheduler else 1,
      scheduler=scheduler if args.use_scheduler else None,
      # resources_per_trial = {"cpu": 2, "gpu": 0},
      # reuse_actors=True,
      # restore=args.restore,
      resume="LOCAL",
      local_dir=args.local_dir,
  )


if __name__ == "__main__":
  main()

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

from ray import init
from ray.rllib.policy.policy import PolicySpec
from ray.tune import tune
from ray.tune.registry import register_env
from ray.tune.schedulers import PopulationBasedTraining

from examples.rllib import config_creator
from examples.rllib import utils
from meltingpot.python import substrate

from tensorflow.keras.optimizers import RMSprop

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
    # final number of channels must be the same as post_fcnet_hiddens
    "conv_filters": [[16, [8, 8], 8], [32, [3, 3], 1], [64, [5, 5], 1]],
    "conv_activation": "relu",
    "post_fcnet_hiddens": [128, 128],  # [64, 64]
    "post_fcnet_activation": "relu",
    "use_lstm": True,
    "lstm_use_prev_action": True,  # False
    "lstm_use_prev_reward": True,  # False
    "lstm_cell_size": 256,  # 128
    # This comes after the CNN before the post_fc_hiddens stuff
    # I am removing it because I have that flattening convnet layer anyway
    "no_final_linear": False  # True
}

# TODO: skip first conv net by using the layers, say WORLD.LAYER?
# TODO: print out model and check where lstm is
# TODO: provide the timestep as a model input

# https://iq.opengenus.org/same-and-valid-padding/
# rllib gives padding "same" for the first n-1 layers,
# then padding valid on the last layer, before a possible final layer that
# changes the output before the hidden layers.
# see ray.rllib.models.torch.misc.same_padding to understand the padding
# same effectively adds padding equal to the kernel height
# Must disable this in both the net and the value net part


def main():
  """
  Example of a self-play training experiment with RLLib
  """

  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--algorithm",
      type=str,
      default="A3C",
      help="ray.rllib.agents algorithm to use")
  parser.add_argument(
      "--cpus", type=int, default=7, help="number of cpus ray can use")
  parser.add_argument(
      "--restore",
      type=str,
      default=None,
      help="ray.tune checkpoint. point to the checkpoint that"
      "has an accompanying .tune_metadata. e.g."
      "~/ray_results/A3C/A3C/checkpoint_XXX/checkpoint-XXX")
  parser.add_argument(
      "--use_scheduler",
      action="store_true",
      help="whether to use the preconfigured scheduler")

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

  rmsprop = RMSprop(
      learning_rate=lr,
      rho=0.99,  # discount factor/decay
      momentum=0.0,
      epsilon=1e-5)

  config = config_creator.generate_config(args.algorithm, custom_model, env_config,
                                          args.cpus, policies, horizon, lr,
                                          rmsprop)

  scheduler = PopulationBasedTraining(
      burn_in_period=25,
      hyperparam_mutations={
          "lr": [1e-4, 3e-5, 1e-5, 3e-6, 1e-6],
          "entropy_coeff": [0.01, 0.003, 0.001],
      },
      log_config=True,
      perturbation_interval=25,
      quantile_fraction=0.25,
      resample_probability=0.25,
      time_attr="training_iteration",
  )

  # Tune
  # https://docs.ray.io/en/latest/tune/api_docs/execution.html#tune-run
  tune.run(
      args.algorithm,
      stop={"training_iteration": 1000},
      checkpoint_at_end=True,
      checkpoint_freq=25,
      config=config,
      metric="episode_reward_mean",
      mode="max",
      log_to_file=True,
      num_samples=4 if args.use_scheduler else 1,
      scheduler=scheduler if args.use_scheduler else None,
      # resources_per_trial = {"cpu": 2, "gpu": 0},
      # reuse_actors=True,
      restore=args.restore)


if __name__ == "__main__":
  main()

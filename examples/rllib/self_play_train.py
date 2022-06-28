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
import os

from ray import init
from ray.rllib.agents.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.tune import tune
from ray.tune.registry import register_env
from ray.tune.schedulers import PopulationBasedTraining

from examples.rllib import utils
from meltingpot.python import substrate

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

# in order to understand the choices, look at ray.rllib.models.torch.misc.same_padding
# examples ones are in ray.rllib.models.utils.get_filter_config
custom_model = {
    # final number of channels must be the same as post_fcnet_hiddens
    "conv_filters": [[16, [8, 8], 8], [32, [3, 3], 1], [64, [5, 5], 1]],
    "conv_activation": "relu",
    "post_fcnet_hiddens":
        [64],  # They used 2x64 MLP, but my extra conv net counts for one...?
    "post_fcnet_activation": "relu",
    "use_lstm": True,
    "lstm_use_prev_action": False,
    "lstm_use_prev_reward": False,
    # TODO: is this wrapping the CNN rather than the FC?
    "lstm_cell_size": 128,
    # This comes after the CNN before the post_fc_hiddens stuff
    # I am removing it because I have that flattening convnet layer anyway
    "no_final_linear": True
}

# TODO: understand that final linear thing in the conv net
# TODO: A3C their setup e.g. keras RMSProp
# TODO: SAC with lstms
# TODO: skip first conv net by using the layers, say WORLD.LAYER?

# https://iq.opengenus.org/same-and-valid-padding/
# rllib gives padding "same" for the first n-1 layers, then padding valid on the last layer, before a possible final layer that changes the output before the hidden layers
# same effectively adds padding equal to the kernel height
# Must disable this in both the net and the value net part


def main():
  """
  Example of a self-play training experiment with RLLib
  """

  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--restore",
      type=str,
      default=None,
      help="ray.tune checkpoint. point to the checkpoint that"
      "has an accompanying .tune_metadata. e.g."
      "~/ray_results/PPO/PPO_meltingpot_XXX/checkpoint_XXX/checkpoint-XXX")

  args = parser.parse_args()

  # Initialize ray
  init(
      configure_logging=True,
      # num_cpus=4,
      logging_level="info")

  # Setup the training environment (MeltingPot subscrate)
  substrate_name = "commons_harvest_open_simple"
  env_config = substrate.get_config(substrate_name)
  register_env("meltingpot", utils.env_creator)

  # A3CConfig
  # learning rate
  # config["lr"] = 1e-4
  # config["lr_schedule"] = None
  # # discount rate
  # config["gamma"] = 0.99

  # # use a critic as a baseline (required for using GAE).
  # config["use_critic"] = True
  # # use the Generalized Advantage Estimator (GAE) with a value function
  # config["use_gae"] = True
  # # size of rollout batch
  # config["rollout_fragment_length"] = 50  # see later note about this being larger
  # # GAE(gamma) parameter
  # config["lambda"] = 1.0
  # # Max global norm for each gradient calculated by worker
  # config["grad_clip"] = 40.0
  # # Value Function Loss coefficient
  # config["vf_loss_coeff"] =  0.5
  # # coefficient of the entropy regularizer.
  # config["entropy_coeff"] = 0.005
  # config["entropy_coeff_schedule"] = None
  # # Workers sample async. Note that this increases the effective
  # # rollout_fragment_length by up to 5x due to async buffering of batches.
  # config["sample_async"] = True
  # # Use the Trainer's `training_iteration` function instead of `execution_plan`.
  # # Fixes a severe performance problem with A3C. Setting this to True leads to a
  # # speedup of up to 3x for a large number of workers and heavier
  # # gradient computations (e.g. ray/rllib/tuned_examples/a3c/pong-a3c.yaml)).
  # config["_disable_execution_plan_api"] = True

  # Configure the horizon
  # After n steps, force reset simulation
  # Each unroll happens exactly over one episode, from
  # beginning to end. Data collection will not stop unless the episode
  # terminates or a configured horizon (hard or soft) is hit.
  horizon = env_config.lab2d_settings["maxEpisodeLengthFrames"]

  # Extract space dimensions
  test_env = utils.env_creator(env_config)

  # Configure the PPO Alogrithm
  # https://github.com/ray-project/ray/blob/master/rllib/algorithms/algorithm_config.py
  # https://github.com/ray-project/ray/blob/master/rllib/algorithms/ppo/ppo.py
  # PPO does not share value function layers
  config = PPOConfig().framework(
      framework="torch",
      eager_tracing=True  # only applies if using tf2
  ).training(
      model=custom_model,
      lr=5e-5,  # Default=5e-5 for PPO
      gamma=0.99,  # Default=0.99
      train_batch_size=12 * horizon,  # Default=4000
      sgd_minibatch_size=128,  # Default=128
      # PPO specifics
      use_critic=True,
      use_gae=True,
      lambda_=1.0,
      kl_coeff=0.2,
      num_sgd_iter=30,  # Default=30
      shuffle_sequences=False,
      vf_loss_coeff=1.0,  # coefficient of the value function loss. not used?
      entropy_coeff=0.003,
      entropy_coeff_schedule=None,
      clip_param=0.3,
      vf_clip_param=10.0,  # N.B. sensitive to reward scale
      grad_clip=None,
      kl_target=0.01).exploration(exploration_config={
          "type": "StochasticSampling"
      }).evaluation(
          evaluation_interval=25,
          evaluation_duration=6,
          always_attach_evaluation_results=False
      ).resources(
          num_cpus_per_worker=1,
          # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
          num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")),
      ).rollouts(
          batch_mode="complete_episodes",
          horizon=horizon,
          num_rollout_workers=6,
          num_envs_per_worker=1,
          rollout_fragment_length=100,
      ).environment(
          clip_rewards=False,
          env_config=env_config,
          env="meltingpot",
          normalize_actions=True,
      ).multi_agent(
          count_steps_by="env_steps",
          policies={
              "av":
                  PolicySpec(
                      policy_class=None,  # use default policy
                      observation_space=test_env.observation_space["player_0"],
                      action_space=test_env.action_space["player_0"],
                      config={}),
          },
          policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: "av",
      ).reporting(
          metrics_num_episodes_for_smoothing=1,).debugging(log_level="INFO")

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
      "PPO",
      stop={"training_iteration": 1000},
      checkpoint_at_end=True,
      checkpoint_freq=25,
      config=config.to_dict(),
      metric="episode_reward_mean",
      mode="max",
      log_to_file=True,
      num_samples=4,  # Number of hyperparameter trials
      # scheduler=scheduler,
      # resources_per_trial = {"cpu": 2, "gpu": 0},
      # reuse_actors=True,
      # restore from a checkpoint, that has the .tune metadatafile
      # restore=args.restore
  )


if __name__ == "__main__":
  main()

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
"""Generates a config for an RLLib trainer."""

import copy
from typing import Dict
import os

from ray.rllib.agents.ppo import PPOConfig
from ray.rllib.agents.registry import get_trainer_class
from ray.tune import grid_search


def generate_config(algo: str, model, env_config: Dict, num_cpus: int, policies: Dict,
                    horizon: int, lr: float, optimizer) -> Dict:
  if algo == "PPO":
    # Configure the PPO Algorithm
    # https://github.com/ray-project/ray/blob/master/rllib/algorithms/algorithm_config.py
    # https://github.com/ray-project/ray/blob/master/rllib/algorithms/ppo/ppo.py
    # PPO does not share value function layers
    config = PPOConfig().framework(
        framework="tf2",
        eager_tracing=True  # only applies if using tf2
    ).training(
        model=model,
        lr=lr,  # Default=5e-5 for PPO
        gamma=0.99,  # Default=0.99
        train_batch_size=12 * horizon,  # Default=4000
        optimizer=optimizer,
        # PPO specifics
        use_critic=True,
        use_gae=True,
        lambda_=1.0,
        kl_coeff=0.2,
        sgd_minibatch_size=128,  # Default=128
        num_sgd_iter=30,  # Default=30
        shuffle_sequences=True,  # recommended to be True
        vf_loss_coeff=1.0,  # coefficient of the value function loss. not used?
        # TODO: find out what a sensible value should be -> log out loss stats
        vf_clip_param=2.0,  # N.B. sensitive to reward scale.
        entropy_coeff=0.003,
        # entropy_coeff = grid_search([0.03, 0.01, 0.003, 0.001, 0.0003]),
        entropy_coeff_schedule=None,
        clip_param=0.3,
        grad_clip=None,
        kl_target=0.01).exploration(exploration_config={
            "type": "StochasticSampling"
        }).resources(
            num_cpus_per_worker=1,
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        ).rollouts(
            batch_mode="complete_episodes",
            horizon=horizon,
            num_rollout_workers=num_cpus-1,
            num_envs_per_worker=1,
            rollout_fragment_length=100,
        ).environment(
            clip_rewards=False,
            env_config=env_config,
            env="meltingpot",
            normalize_actions=True,
        ).multi_agent(
            count_steps_by="env_steps",
            policies=policies,
            policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: "av",
        ).debugging(log_level="INFO")

    return config.to_dict()

  elif algo == "A3C":
    config = copy.deepcopy(get_trainer_class(algo).get_default_config())
    config["framework"] = "tf2"
    config["eager_tracing"] = True  # only applies if using tf2
    config["env"] = "meltingpot"
    config["env_config"] = env_config
    config["model"] = model
    config["multiagent"] = {
        "policies": policies,
        "policy_mapping_fn": lambda agent_id, episode, worker, **kwargs: "av",
        "count_steps_by": "env_steps",
    }

    config["num_gpus"] = int(os.environ.get("RLLIB_NUM_GPUS", "0"))
    config["log_level"] = "INFO"
    config["num_workers"] = num_cpus-1
    # config["evaluation_interval"] = 25
    # config["evaluation_duration"] = num_cpus-1

    config["horizon"] = horizon
    config["batch_mode"] = "complete_episodes"
    config["train_batch_size"] = 12 * horizon

    config["lr"] = lr
    config["optimizer"] = optimizer
    # discount rate
    config["gamma"] = 0.99

    # size of rollout batch. see later note about this being larger in effect
    config["rollout_fragment_length"] = 50
    config["vf_loss_coeff"] = 0.5
    config["entropy_coeff"] = 0.003
    # config["entropy_coeff"] = grid_search([0.03, 0.01, 0.003, 0.001, 0.0003])
    # Workers sample async. Note that this increases the effective
    # rollout_fragment_length by up to 5x due to async buffering of batches.
    config["sample_async"] = True
    return config

  else:
    raise ValueError(f"Unknown algorithm: {algo}")

# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Runs the bots trained in self_play_train.py and renders in pygame.
"""

import argparse

import dm_env
from dmlab2d.ui_renderer import pygame
import numpy as np
from ray.rllib.algorithms.registry import _get_algorithm_class
from ray.tune.analysis.experiment_analysis import ExperimentAnalysis
from ray.tune.registry import register_env

from examples.rllib.utils import env_creator, RayModelPolicy


def get_human_action():
  a = None
  for event in pygame.event.get():
    if event.type == pygame.KEYDOWN:
      key_pressed = pygame.key.get_pressed()
      if key_pressed[pygame.K_SPACE]:
        a = 0
      if key_pressed[pygame.K_UP]:
        a = 1
      if key_pressed[pygame.K_DOWN]:
        a = 2
      if key_pressed[pygame.K_LEFT]:
        a = 3
      if key_pressed[pygame.K_RIGHT]:
        a = 4
      if key_pressed[pygame.K_z]:
        a = 5
      if key_pressed[pygame.K_x]:
        a = 6
      break  # removing this did not solve the bug
  return a

def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--experiment_state",
      type=str,
      required=True,
      help="ray.tune experiment_state to load. The default setting will load"
      " the last training run created by self_play_train.py. If you want to use"
      " a specific run, provide a path, expected to be of the format "
      " ~/ray_results/PPO/experiment_state-DATETIME.json")
  parser.add_argument(
      "--trial",
      type=str,
      default=None,
      help="If provided, use this trial instead of the best trial")
  parser.add_argument(
      "--checkpoint",
      type=str,
      default=None,
      help="If provided, use this checkpoint instead of the last checkpoint")
  parser.add_argument(
      "--human", action="store_true", help="a human controls one of the bots")

  args = parser.parse_args()

  agent_algorithm = "PPO"

  register_env("meltingpot", env_creator)

  experiment = ExperimentAnalysis(
      args.experiment_state,
      default_metric="episode_reward_mean",
      default_mode="max")

  if args.trial:
    trial = None
    for idx, trial_path in enumerate(experiment._get_trial_paths()):
      if args.trial in trial_path:
        trial = experiment.trials[idx]
    assert trial is not None
  else:
    trial = experiment.get_best_trial(scope="last")

  checkpoint_path = args.checkpoint if args.checkpoint else trial.checkpoint.path

  config = trial.config
  # checkpoint_path = experiment.get_trial_checkpoints_paths(best_trial)[-1][0]
  # config = experiment.get_best_config()
  # checkpoint_path = experiment.get_best_checkpoint()
  # TODO: Do I need a serious evaluation during these passes?

  config["explore"] = False
  config["num_rollout_workers"] = 0

  trainer = _get_algorithm_class(agent_algorithm)(config=config)
  trainer.load_checkpoint(checkpoint_path)

  # Create a new environment to visualise
  env_config = config["env_config"]
  env = env_creator(env_config).get_dmlab2d_env()

  roles = env_config["roles"]
  num_players = len(roles)
  bots = [
      RayModelPolicy(
        trainer,
        env_config["substrate_config"]["individual_observation_names"],
        role) for role in roles
  ]
  bots = bots[1:] if args.human else bots

  timestep = env.reset()
  states = [bot.initial_state() for bot in bots]
  actions = [0] * len(bots)


  obs_spec = env.observation_spec()
  shape = obs_spec[0]["WORLD.RGB"].shape

  # Configure the pygame display
  pygame.init()
  scale = 1000 // max(int(shape[0]), int(shape[1]))
  fps = 8
  game_display = pygame.display.set_mode(
      (int(shape[1] * scale), int(shape[0] * scale)))
  clock = pygame.time.Clock()
  pygame.display.set_caption("DM Lab2d")

  total_rewards = np.zeros(num_players)

  for _ in range(500):
    obs = timestep.observation[0]["WORLD.RGB"]
    obs = np.transpose(obs, (1, 0, 2))
    surface = pygame.surfarray.make_surface(obs)
    rect = surface.get_rect()
    surf = pygame.transform.scale(surface,
                                  (int(rect[2] * scale), int(rect[3] * scale)))

    game_display.blit(surf, dest=(0, 0))
    pygame.display.update()
    clock.tick(fps)

    if args.human:
      while True:
        a = get_human_action()
        # TODO: fix bug where two quick presses, e.g. 1,2, are counted as 1,1

        if a is not None:
          break

      human_action = [a]
    else:
      human_action = []

    for i, bot in enumerate(bots):
      timestep_bot = dm_env.TimeStep(
          step_type=timestep.step_type,
          reward=timestep.reward[i],
          discount=timestep.discount,
          observation=timestep.observation[i])

      actions[i], states[i] = bot.step(timestep_bot, states[i])

    timestep = env.step(actions + human_action)
    print(actions + human_action, timestep.reward)
    total_rewards = total_rewards + timestep.reward

  print("Total rewards: {}".format(total_rewards))


if __name__ == "__main__":
  main()

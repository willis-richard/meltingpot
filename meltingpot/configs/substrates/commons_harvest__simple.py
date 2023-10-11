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
"""Configuration for Commons Harvest: Simple.

Example video: https://youtu.be/lZ-qpPP4BNE

Apples are spread around the map and can be consumed for a reward of 1. Apples
that have been consumed regrow with a per-step probability that depends on the
number of uneaten apples in a `L2` norm neighborhood of radius 2 (by default).
After an apple has been eaten and thus removed, its regrowth probability depends
on the number of uneaten apples still in its local neighborhood. With standard
parameters, it the grown rate decreases as the number of uneaten apples in the
neighborhood decreases and when there are zero uneaten apples in the
neighborhood then the regrowth rate is zero. As a consequence, a patch of apples
that collectively doesn't have any nearby apples, can be irrevocably lost if all
apples in the patch are consumed. Therefore, agents must exercise restraint when
consuming apples within a patch. Notice that in a single agent situation, there
is no incentive to collect the last apple in a patch (except near the end of the
episode). However, in a multi-agent situation, there is an incentive for any
agent to consume the last apple rather than risk another agent consuming it.
This creates a tragedy of the commons from which the substrate derives its name.

This mechanism was first described in Janssen et al (2010) and adapted for
multi-agent reinforcement learning in Perolat et al (2017).

Janssen, M.A., Holahan, R., Lee, A. and Ostrom, E., 2010. Lab experiments for
the study of social-ecological systems. Science, 328(5978), pp.613-617.

Perolat, J., Leibo, J.Z., Zambaldi, V., Beattie, C., Tuyls, K. and Graepel, T.,
2017. A multi-agent reinforcement learning model of common-pool
resource appropriation. In Proceedings of the 31st International Conference on
Neural Information Processing Systems (pp. 3646-3655).
"""

from typing import Any, Dict, List, Mapping, Sequence

from ml_collections import config_dict
import numpy as np

from meltingpot.configs.substrates import _validated
from meltingpot.utils.substrates import shapes
from meltingpot.utils.substrates import specs

# Warning: setting `_ENABLE_DEBUG_OBSERVATIONS = True` may cause slowdown.
_ENABLE_DEBUG_OBSERVATIONS = False

APPLE_RESPAWN_RADIUS = 1.0
REGROWTH_PROBABILITIES = [0.0, 0.15]
SPRITE_SIZE = 1

ASCII_MAP = """
P   P
  A  
 AAA 
  A  
P   P
"""

# `prefab` determines which prefab game object to use for each `char` in the
# ascii map.
CHAR_PREFAB_MAP = {
    "P": {"type": "all", "list": ["floor", "spawn_point"]},
    " ": "floor",
    "A": {"type": "all", "list": ["grass", "apple"]},
}

_COMPASS = ["N", "E", "S", "W"]

FLOOR = {
    "name": "floor",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "floor",
                "stateConfigs": [{
                    "state": "floor",
                    "layer": "background",
                    "sprite": "Floor",
                }],
            }
        },
        {
            "component": "Transform",
        },
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["Floor"],
                "spriteShapes": [shapes.GRAINY_FLOOR],
                "palettes": [{"*": (0, 0, 0, 255),
                              "+": (0, 0, 0, 255),}],
                "noRotates": [False]
            }
        },
    ]
}

GRASS = {
    "name":
        "grass",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState":
                    "grass",
                "stateConfigs": [
                    {
                        "state": "grass",
                        "layer": "background",
                        "sprite": "Grass"
                    },
                    {
                        "state": "dessicated",
                        "layer": "background",
                        "sprite": "Floor"
                    },
                ],
            }
        },
        {
            "component": "Transform",
        },
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["Grass", "Floor"],
                "spriteShapes": [
                    shapes.GRASS_STRAIGHT, shapes.GRAINY_FLOOR
                ],
                "palettes": [{
                    "*": (158, 194, 101, 255),
                    "@": (158, 194, 101, 255)
                }, {
                    "*": (220, 205, 185, 255),
                    "+": (210, 195, 175, 255),
                }],
                "noRotates": [False, False]
            }
        },
    ]
}

SPAWN_POINT = {
    "name": "spawnPoint",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "spawnPoint",
                "stateConfigs": [{
                    "state": "spawnPoint",
                    "layer": "alternateLogic",
                    "groups": ["spawnPoints"]
                }],
            }
        },
        {
            "component": "Transform",
        },
    ]
}

# Primitive action components.
# pylint: disable=bad-whitespace
# pyformat: disable
NOOP       = {"move": 0}
FORWARD    = {"move": 1}
BACKWARD   = {"move": 2}
STEP_LEFT  = {"move": 3}
STEP_RIGHT = {"move": 4}
# pyformat: enable
# pylint: enable=bad-whitespace

ACTION_SET = (
    NOOP,
    FORWARD,
    BACKWARD,
    STEP_LEFT,
    STEP_RIGHT,
)

TARGET_SPRITE_SELF = {
    "name": "Self",
    "shape": shapes.CUTE_AVATAR,
    "palette": shapes.get_palette((50, 100, 200)),
    "noRotate": True,
}


def create_scene():
  """Creates the scene with the provided args controlling apple regrowth."""
  scene = {
      "name": "scene",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": "scene",
                  "stateConfigs": [{
                      "state": "scene",
                  }],
              }
          },
          {
              "component": "Transform",
          },
          {
              "component": "Neighborhoods",
              "kwargs": {}
          },
      ]
  }

  return scene


def create_apple_prefab(regrowth_radius=-1.0,  # pylint: disable=dangerous-default-value
                        regrowth_probabilities=[0, 0.0, 0.0, 0.0]):
  """Creates the apple prefab with the provided settings."""
  growth_rate_states = [
      {
          "state": "apple",
          "layer": "lowerPhysical",
          "sprite": "Apple",
          "groups": ["apples"]
      },
      {
          "state": "appleWait",
          "layer": "logic",
          "sprite": "AppleWait",
      },
  ]
  # Enumerate all possible states for a potential apple. There is one state for
  # each regrowth rate i.e., number of nearby apples.
  upper_bound_possible_neighbors = np.floor(np.pi*regrowth_radius**2+1)+1
  for i in range(int(upper_bound_possible_neighbors)):
    growth_rate_states.append(dict(state="appleWait_{}".format(i),
                                   layer="logic",
                                   groups=["waits_{}".format(i)],
                                   sprite="AppleWait"))

  apple_prefab = {
      "name": "apple",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": "apple",
                  "stateConfigs": growth_rate_states,
              }
          },
          {
              "component": "Transform",
          },
          {
              "component": "Appearance",
              "kwargs": {
                  "renderMode": "ascii_shape",
                  "spriteNames": ["Apple", "AppleWait"],
                  "spriteShapes": [shapes.APPLE, shapes.FILL],
                  "palettes": [
                      {"x": (0, 0, 0, 0),
                       "*": (214, 88, 88, 255),
                       "#": (194, 79, 79, 255),
                       "o": (53, 132, 49, 255),
                       "|": (102, 51, 61, 255)},
                      {"i": (0, 0, 0, 0)}],
                  "noRotates": [True, True]
              }
          },
          {
              "component": "Edible",
              "kwargs": {
                  "liveState": "apple",
                  "waitState": "appleWait",
                  "rewardForEating": 1.0,
              }
          },
          {
              "component": "DensityRegrow",
              "kwargs": {
                  "liveState": "apple",
                  "waitState": "appleWait",
                  "radius": regrowth_radius,
                  "regrowthProbabilities": regrowth_probabilities,
              }
          },
      ]
  }

  return apple_prefab


def create_prefabs(regrowth_radius=-1.0,
                   # pylint: disable=dangerous-default-value
                   regrowth_probabilities=[0, 0.0, 0.0, 0.0]):
  """Returns a dictionary mapping names to template game objects."""
  prefabs = {
      "floor": FLOOR,
      "grass": GRASS,
      "spawn_point": SPAWN_POINT,
  }
  prefabs["apple"] = create_apple_prefab(
      regrowth_radius=regrowth_radius,
      regrowth_probabilities=regrowth_probabilities)
  return prefabs


def create_avatar_object(player_idx: int,
                         target_sprite_self: Dict[str, Any],
                         spawn_group: str) -> Dict[str, Any]:
  """Create an avatar object that always sees itself as blue."""
  # Lua is 1-indexed.
  lua_index = player_idx + 1

  # Setup the self vs other sprite mapping.
  source_sprite_self = "Avatar" + str(lua_index)
  custom_sprite_map = {source_sprite_self: target_sprite_self["name"]}

  live_state_name = "player{}".format(lua_index)
  avatar_object = {
      "name": "avatar",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": live_state_name,
                  "stateConfigs": [
                      {"state": live_state_name,
                       "layer": "upperPhysical",
                       "sprite": source_sprite_self,
                       "contact": "avatar",
                       "groups": ["players"]},

                      {"state": "playerWait",
                       "groups": ["playerWaits"]},
                  ]
              }
          },
          {
              "component": "Transform",
          },
          {
              "component": "Appearance",
              "kwargs": {
                  "renderMode": "ascii_shape",
                  "spriteNames": [source_sprite_self],
                  "spriteShapes": [shapes.CUTE_AVATAR],
                  # for training we want the agents to always see the other
                  # player as the same colour independent of player_idx
                  #  "palettes": [shapes.get_palette(
                  #      colors.human_readable[player_idx])],
                  "palettes": [shapes.get_palette((150, 100, 50))],
                  "noRotates": [True]
              }
          },
          {
              "component": "AdditionalSprites",
              "kwargs": {
                  "renderMode": "ascii_shape",
                  "customSpriteNames": [target_sprite_self["name"]],
                  "customSpriteShapes": [target_sprite_self["shape"]],
                  "customPalettes": [target_sprite_self["palette"]],
                  "customNoRotates": [target_sprite_self["noRotate"]],
              }
          },
          {
              "component": "Avatar",
              "kwargs": {
                  "index": lua_index,
                  "aliveState": live_state_name,
                  "waitState": "playerWait",
                  "speed": 1.0,
                  "spawnGroup": spawn_group,
                  "postInitialSpawnGroup": "spawnPoints",
                  "actionOrder": ["move"],
                  "actionSpec": {
                      "move": {"default": 0, "min": 0, "max": len(_COMPASS)},
                  },
                  "view": {
                      "left": 0,
                      "right": 4,
                      "forward": 0,
                      "backward": 4,
                      "centered": False
                  },
                  "fullObservations": True,
                  "useAbsoluteCoordinates": True,
                  "spriteMap": custom_sprite_map,
                  "randomizeInitialOrientation": False
              }
          },
      ]
  }
  if _ENABLE_DEBUG_OBSERVATIONS:
    avatar_object["components"].append({
        "component": "LocationObserver",
        "kwargs": {"objectIsAvatar": True, "alsoReportOrientation": True},
    })

  return avatar_object


def create_avatar_objects(num_players):
  """Returns list of avatar objects of length 'num_players'."""
  avatar_objects = []
  for player_idx in range(0, num_players):
    spawn_group = "spawnPoints"

    game_object = create_avatar_object(player_idx,
                                       TARGET_SPRITE_SELF,
                                       spawn_group=spawn_group)
    avatar_objects.append(game_object)

  return avatar_objects


def get_config(num_players: int=2,
        regrowth_probabilities: List[float]=REGROWTH_PROBABILITIES):
  """Default configuration for training on the commons_harvest level."""
  config = config_dict.ConfigDict()

  # Basic configuration.
  config.num_players = num_players
  # Action set configuration.
  config.action_set = ACTION_SET
  # Observation format configuration.
  config.individual_observation_names = [
      "RGB",
  ]
  config.global_observation_names = [
      "WORLD.RGB",
  ]

  # The specs of the environment (from a single-agent perspective).
  config.action_spec = specs.action(len(ACTION_SET))
  config.timestep_spec = specs.timestep({
      "RGB": specs.OBSERVATION["RGB"],
      "WORLD.RGB": specs.rgb(5*SPRITE_SIZE, 5*SPRITE_SIZE),
  })

  # The roles assigned to each player.
  config.valid_roles = frozenset({"default"})
  config.default_player_roles = ("default",) * num_players
  config.regrowth_probabilities = regrowth_probabilities
  config.lab2d_settings_builder = _validated(build)

  return config


def build(
    roles: Sequence[str],
    config: config_dict.ConfigDict,
) -> Mapping[str, Any]:
  """Build substrate definition given player roles."""
  num_players = len(roles)
  # Build the rest of the substrate definition.
  substrate_definition = dict(
      levelName="commons_harvest",
      levelDirectory="meltingpot/lua/levels",
      numPlayers=num_players,
      # Deterministic end
      maxEpisodeLengthFrames=500,
      spriteSize=SPRITE_SIZE,
      topology="BOUNDED",  # Choose from ["BOUNDED", "TORUS"],
      simulation={
          "map": ASCII_MAP,
          "gameObjects": create_avatar_objects(num_players),
          "prefabs": create_prefabs(APPLE_RESPAWN_RADIUS,
                                    config.regrowth_probabilities),
          "charPrefabMap": CHAR_PREFAB_MAP,
          "scene": create_scene(),
      },
  )
  del config
  return substrate_definition

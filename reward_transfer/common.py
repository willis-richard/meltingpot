"""Defines some helper constants and functions."""
from typing import List, Optional

from meltingpot.python.configs.substrates.commons_harvest__simple import get_config as get_config_harvest
from meltingpot.python.configs.substrates.coins import get_config as get_config_coins

LOGGING_LEVEL = "WARN"
VERBOSE = 1

# The final layer must be chosen specifically so that its output is
# [B, 1, 1, X]. See the explanation in
# https://docs.ray.io/en/latest/rllib-models.html#built-in-models. It is
# because rllib is unable to flatten to a vector otherwise.
CUSTOM_MODEL = {
    "conv_filters": [[16, [3, 3], 1], [32, [3, 3], 1], [64, [5, 5], 1]],
    "conv_activation": "relu",
    "post_fcnet_hiddens": [64, 64],
    "post_fcnet_activation": "relu",
    "no_final_linear": True,
    # needs vf_loss_coeff to be tuned if True
    "vf_share_layers": True,
    "use_lstm": True,
    "lstm_cell_size": 128,
    "lstm_use_prev_action": False,
    "lstm_use_prev_reward": False,
}

DEFAULT_MODEL = {
    "conv_filters": [[16, [8, 8], 8], [128, [11, 11], 1]],
    "fcnet_hiddens": [64, 64],
    "fcnet_activation": "relu",
    "conv_activation": "relu",
    "post_fcnet_hiddens": [256,],
    "post_fcnet_activation": "relu",
    "use_lstm": True,
    "lstm_use_prev_action": True,
    "lstm_use_prev_reward": False,
    "lstm_cell_size": 256,
}

def create_env_config_harvest(reward_transfer: List[List],
                              regrowth_probability: float = 0.15):
  """Create the commons_harvest__simple config."""
  config = get_config_harvest(
      regrowth_probabilities=[0, regrowth_probability])

  config["substrate"] = "commons_harvest__simple"
  config["roles"] = config.default_player_roles

  config["reward_transfer"] = reward_transfer
  config.lock()
  return config


def create_env_config_coins(reward_transfer: List[List]):
  """Create the commons_harvest__simple config."""
  config = get_config_coins()

  config["substrate"] = "commons_harvest__simple"
  config["roles"] = config.default_player_roles

  config["reward_transfer"] = reward_transfer
  config.lock()
  return config

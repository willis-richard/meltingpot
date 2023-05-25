"""Defines some helper constants and functions."""
from typing import List, Optional

from meltingpot.python.configs.substrates.commons_harvest__simple import get_config

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

def create_env_config(num_players: int,
                      regrowth_probability: float = 0.15,
                      reward_transfer: Optional[List[List]] = None):
  """Create the commons_harvest__simple config."""
  config = get_config(
      num_players=num_players,
      regrowth_probabilities=[0, regrowth_probability])

  config["substrate"] = "commons_harvest__simple"
  config["roles"] = config.default_player_roles

  if num_players == 1:
    config["reward_transfer"] = [[1]]
  elif num_players == 2:
    assert reward_transfer is not None, "must provide reward_transfer"
    config["reward_transfer"] = reward_transfer
  else:
    assert False, "num_player must be 1 or 2"
  config.lock()
  return config

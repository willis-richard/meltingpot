'''Plot the results of experiments.py'''

import argparse
import ast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    '--data_path',
    type=str,
    required=True,
    help='This is csv produced by experiments.py')
parser.add_argument(
    '--plot_name', type=str, required=True, help='name to save plot')
parser.add_argument(
    '--regrowth_probability',
    type=float,
    default=0.15,
    help='apple regrowth probability')
parser.add_argument(
    '--gifting',
    action='store_true',
    help='If true, this represents player_0 gifting to player_1')

args = parser.parse_args()

X_MAX = 1.0 if args.gifting else 0.5
Y_MAX = {0.05: 100, 0.15: 250, 0.45: 500}[args.regrowth_probability]

PLAYERS = ['player_0', 'player_1']


def gini(x):
  x = np.sort(x)
  total = 0
  for i, xi in enumerate(x[:-1], 1):
    total += np.sum(np.abs(xi - x[i:]))
  return total / (len(x)**2 * np.mean(x)) if total else 0


def compute_gini(df):
  rewards = df.loc['hist_stats'].apply(
      lambda x: [ast.literal_eval(x)[f'policy_{p}_reward'] for p in PLAYERS])
  rewards = rewards.rename('episode_rewards')
  df = pd.concat([df, rewards.to_frame().transpose()], axis='index')

  g = df.loc['episode_rewards'].apply(
      lambda x: np.mean([gini(a) for a in zip(x[0], x[1])])).rename('gini')
  df = pd.concat([df, g.to_frame().transpose()], axis='index')
  return df


def tidy_df(df):
  if not args.gifting:
    df = compute_gini(df)
    df = df.drop(['episode_rewards'])
  keep = ['episode_reward_max', 'episode_reward_min', 'episode_reward_mean', 'episodes_this_iter', 'policy_reward_mean', 'gini']
  df = df.drop([i for i in df.index if i not in keep])
  k = pd.DataFrame(
      dict(zip(df.columns,
               df.columns.str.split('_').str[0])), index=['k'])
  trial = pd.DataFrame(
      dict(zip(df.columns,
               df.columns.str.split('_').str[1])), index=['trial'])
  checkpoint = pd.DataFrame(
      dict(zip(df.columns,
               df.columns.str.split('_').str[2])),
      index=['checkpoint'])
  df = pd.concat([k, trial, checkpoint, df], axis='index')

  policy_reward_mean = df.loc['policy_reward_mean'].apply(
      lambda x: ast.literal_eval(x))
  policy_reward_mean = pd.concat(
      [policy_reward_mean.apply(lambda x: x[p]) for p in PLAYERS],
      axis='columns')
  policy_reward_mean.columns = PLAYERS

  df = df.drop('policy_reward_mean')
  df = df.T
  df = pd.concat([df, policy_reward_mean], axis='columns')

  for c in df.columns:
    df[c] = pd.to_numeric(df[c])
  df = df.reset_index(drop=True)
  return df


def plot(df, ax):
  if args.gifting:
    df['player_1_gift'] = df.player_1 + (df.player_0 * df.k)
    df['player_0_gift'] = df.player_0 * (1 - df.k)
    df.groupby('k').player_0_gift.mean().plot(ax=ax)
    df.groupby('k').player_1_gift.mean().plot(ax=ax, linestyle='--')
  else:
    df.groupby('k').episode_reward_mean.mean().plot(ax=ax)
  labels = ['episode_reward_mean'] + PLAYERS
  ks = df.groupby(['k', 'trial'])[labels].mean()
  ks = pd.DataFrame({k: ks[k].values for k in labels},
                    index=ks.index.droplevel('trial')).reset_index()
  ks.plot(
      x='k',
      y='episode_reward_mean',
      kind='scatter',
      s=20,
      marker='x',
      ax=ax,
      xlabel='',
      ylabel='')

  ax.set_ylim(0, Y_MAX)
  ax.set_xlim(-0.01, X_MAX + 0.01)

  if not args.gifting:
    ax_r = df.groupby('k').gini.mean().plot(
        secondary_y=True,
        color='g',
        ax=ax,
        xlabel='',
        ylabel='',
        linestyle='-.')
    ax.right_ax.set_ylim(0, 0.5)


def format_labels_legend(ax):
  ax.set_ylabel('reward')
  if args.gifting:
    ax.set_xlabel('reward gifting, $g$')
    ax.legend(['gifter (mean)', 'opponent (mean)', 'utilitarian'],
              loc='upper left',
              bbox_to_anchor=(-0.1, -0.25),
              ncol=3)
  else:
    ax.set_xlabel('reward exchange, $e$')
    ax.legend(['mean utilitarian', 'by seed'],
              loc='upper left',
              bbox_to_anchor=(-0.1, -0.25),
              ncol=2)
  try:
    ax.right_ax.legend(['mean gini (right)'],
                       loc='center left',
                       bbox_to_anchor=(0.5, -0.37),
                       ncol=1)
    ax.right_ax.set_ylabel('gini value (unmodified rewards)')
  except:
    pass


if __name__ == '__main__':
  df = pd.read_csv(args.data_path, index_col=0)

  df = tidy_df(df)

  plt.rcParams.update({'font.size': 18})

  fig, ax = plt.subplots(figsize=(12, 5), facecolor='white')

  plot(df, ax)

  format_labels_legend(ax)

  fig.subplots_adjust(bottom=0.3)

  ax.axvline(0.5, color='gray', linestyle=(0, (2, 6)))

  fig.savefig(args.plot_name)

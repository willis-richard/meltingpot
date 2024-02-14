import argparse
from ray import tune

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--restore",
      type=str,
      default=None,
      help="Path to Tuner state")
  args = parser.parse_args()

  ea = tune.ExperimentAnalysis(args.restore)

  print(ea.results_df)

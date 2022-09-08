import wandb
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import argparse

def get_df(runs, rew_column):
  print("This may takes up to a few minutes, depends on the number of runs.")
  df = pd.DataFrame()
  df = runs[0].history()[["timesteps_total"]]
  df["machine_type"] = runs[0].config['env_config']['machine_type']
  for run in runs:
    df["raw_rew_mean"+run.id] = run.history()[rew_column]
  return df.melt(id_vars=['timesteps_total', 'machine_type'])

def main(entity, project, group, save_path):
  api = wandb.Api()
  runs_getput = api.runs(f"{entity}/{project}", filters={"group": group, "tags":{"$in": ["get-put"]}})
  runs_trivial = api.runs(f"{entity}/{project}", filters={"group": group, "tags":{"$in": ["trivial"]}})
  print("trivial/get-put #runs:", len(runs_trivial), len(runs_getput))

  draw_df = get_df(runs_trivial, "episode_reward_mean")
  draw_df = pd.concat([draw_df,get_df(runs_getput, "policy_reward_mean/ACTUAL")], axis=0)

  sns.set_theme()
  ax = sns.lineplot(data=draw_df, x="timesteps_total", y="value", hue='machine_type', errorbar='sd')
  plt.legend(bbox_to_anchor=(0.5, 1.05), loc='center', ncol=2, frameon=False, borderaxespad=0.)
  plt.ylabel('mean reward')
  start, end = ax.get_xlim()
  range = str(int(end/5))
  tick_size = int(range[0]+"0"*(len(range)-1))
  ax.xaxis.set_ticks(np.arange(0, end, tick_size))
  plt.xticks(rotation = 20)
  plt.savefig(save_path, bbox_inches='tight')

if __name__=="__main__":
  print("start")
  parser = argparse.ArgumentParser()
  parser.add_argument('entity', type=str, nargs='?')
  parser.add_argument('project', type=str, nargs='?')
  parser.add_argument('group', type=str, nargs='?')
  parser.add_argument('save_path', type=str, nargs='?')
  args = parser.parse_args()
  main(args.entity, args.project, args.group, args.save_path)
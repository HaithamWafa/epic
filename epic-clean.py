
import argparse

import importlib
spec = importlib.util.find_spec("evaluating_rewards")

#if spec is None:
   # !pip install --quiet git+git://github.com/HumanCompatibleAI/evaluating-rewards

# Turn off distracting warnings and logging
import warnings
warnings.filterwarnings("ignore")

#import tensorflow as tf
import logging
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)

# Import rest of the dependencies
#import gym
import pandas as pd
#from stable_baselines.common import vec_env

#from evaluating_rewards import datasets, serialize
from evaluating_rewards.analysis import util
from evaluating_rewards.distances import epic_sample, tabular


from getRewards import getRewards ###importing my custom function###
##### terminal stuff #####
parser = argparse.ArgumentParser(description='Generate sweep and save results to CSV.')
parser.add_argument('--power', nargs='+', type=float, required=True, help='List of power values')
parser.add_argument('--ips', nargs='+', type=float, required=True, help='List of ips values')
parser.add_argument('--output', type=str, default='epic_results.csv', help='Output CSV file name')
# Parse the command-line arguments
args = parser.parse_args()

#bypassing deshaped_rew and using raw rewards for test purposes#
#deshaped_rew = {'P(45%)_IPS(25%)': getRewards(0.45, 0.25),'P(60%)_IPS(25%)': getRewards(0.6, 0.25), 'P(75%)_IPS(25%)': getRewards(0.75, 0.25), 'P(100%)_IPS(25%)': getRewards(1.0, 0.25),   'P6I5': getRewards(0.6, 0.5), 'P4I5': getRewards(0.4, 0.5), 'P1I5': getRewards(0.1, 0.5),'P10I5': getRewards(1, 0.5)}
#automating the generation of sweep - use epic_df.to_csv('name of file.csv') to save data

deshaped_rew = {}

#now required to have a terminal input
#power_range = [0.45, 0.6, 0.75, 0.9]
#ips_range = [0.25, 0.5, 0.75, 1]

for power in args.power:
    for ips in args.ips:
        key = f'P({int(power * 100)}%)_IPS({int(ips * 100)}%)'
        deshaped_rew[key] = getRewards(power, ips)
        
##################

# Now, let's compute the Pearson distance between these canonicalized rewards.
# The canonicalized rewards are quantized to `n_samples` granularity, so we can
# then compute the Pearson distance on this (finite approximation) exactly.
epic_distance = util.cross_distance(deshaped_rew, deshaped_rew, tabular.pearson_distance, parallelism=1)
epic_df = pd.Series(epic_distance).unstack()

epic_df


######
# Save results to CSV
epic_df.to_csv(args.output, index=True)

print(f'Results saved to {args.output}')
##########
import argparse
# Install library if it's not already installed
import importlib




spec = importlib.util.find_spec("evaluating_rewards")

#if spec is None:
   # !pip install --quiet git+git://github.com/HumanCompatibleAI/evaluating-rewards

# Turn off distracting warnings and logging
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
import logging
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)

# Import rest of the dependencies
import gym
import pandas as pd
from stable_baselines.common import vec_env

from evaluating_rewards import datasets, serialize
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
#######
n_samples = 512  # number of samples to take final expectation over
n_mean_samples = 512  # number of samples to use to canonicalize potential
env_name = "evaluating_rewards/PointMassLine-v0"  # the environment to compare in
# The reward models to load.
model_kinds = (
    "evaluating_rewards/PointMassSparseWithCtrl-v0",
    "evaluating_rewards/PointMassDenseWithCtrl-v0",
    "evaluating_rewards/PointMassGroundTruth-v0"
)
seed = 42  # make results deterministic



venv = vec_env.DummyVecEnv([lambda: gym.make(env_name)])
sess = tf.Session()
with sess.as_default():
    tf.set_random_seed(seed)
    models = {kind: serialize.load_reward(reward_type=kind, reward_path="dummy", venv=venv) 
              for kind in model_kinds}

# Visitation distribution (obs,act,next_obs) is IID sampled from spaces
with datasets.transitions_factory_iid_from_sample_dist_factory(
    obs_dist_factory=datasets.sample_dist_from_space,
    act_dist_factory=datasets.sample_dist_from_space,
    obs_kwargs={"space": venv.observation_space},
    act_kwargs={"space": venv.action_space},
    seed=seed,
) as iid_generator:
    batch = iid_generator(n_samples)

with datasets.sample_dist_from_space(venv.observation_space, seed=seed+1) as obs_dist:
    next_obs_samples = obs_dist(n_mean_samples)
with datasets.sample_dist_from_space(venv.action_space, seed=seed+2) as act_dist:
    act_samples = act_dist(n_mean_samples)
    
# Finally, let's compute the EPIC distance between these models.
# First, we'll canonicalize the rewards.

###need to overwrite the parameters models, rewards(overwritten inside), next_obs, act_samples###
#### models ####

#### batch ####

#### act_samples ####

#### next_obs #####
with sess.as_default():
    deshaped_rew = epic_sample.sample_canon_shaping(
        models=models,
        batch=batch,
        next_obs_samples=next_obs_samples,
        act_samples=act_samples,
        # You can also specify the discount rate and the type of norm,
        # but defaults are fine for most use cases.
    )

###################
#bypassing deshaped_rew and using raw rewards for test purposes#
#deshaped_rew = {'P(45%)_IPS(25%)': getRewards(0.45, 0.25),'P(60%)_IPS(25%)': getRewards(0.6, 0.25), 'P(75%)_IPS(25%)': getRewards(0.75, 0.25), 'P(100%)_IPS(25%)': getRewards(1.0, 0.25),   'P6I5': getRewards(0.6, 0.5), 'P4I5': getRewards(0.4, 0.5), 'P1I5': getRewards(0.1, 0.5),'P10I5': getRewards(1, 0.5)}
#automating the generation of sweep - use epic_df.to_csv('name of file.csv') to save data

deshaped_rew = {}

power_range = [0.45, 0.6, 0.75, 0.9]
ips_range = [0.25, 0.5, 0.75, 1]

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
#epic_df.index = epic_df.index.str.replace(r'evaluating_rewards/PointMass(.*)-v0', r'\1')
#epic_df.columns = epic_df.columns.str.replace(r'evaluating_rewards/PointMass(.*)-v0', r'\1')

epic_df


######
# Save results to CSV
epic_df.to_csv(args.output, index=True)

print(f'Results saved to {args.output}')
##########
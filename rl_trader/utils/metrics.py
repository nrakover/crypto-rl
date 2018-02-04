'''
Utils for evaluating episodes and experiments.
'''

import math

SECOND_PER_DAY = 60*60*24

def compute_total_growth(rewards):
    total_reward = sum(rewards) # sum of logs of growth factors
    return math.exp(total_reward)

def compute_avg_growth_per_day(episode, rewards):
    total_growth = compute_total_growth(rewards)

    episode_start = episode.history[0].get_timestamp()
    episode_end = episode.history[-1].get_timestamp()
    num_days = 1 + (episode_end - episode_start) / SECOND_PER_DAY

    return float(total_growth) / num_days

def total_growth_relative_to_best_pick(episode, rewards):
    best_asset = None
    best_growth = 0.0
    for asset_index in range(len(episode.starting_allocation)):
        starting_value = episode.history[0].get_asset_value(asset_index)
        ending_value = episode.history[-1].get_asset_value(asset_index)
        asset_growth = ending_value / starting_value

        if best_asset is None or asset_growth > best_growth:
            best_asset = asset_index
            best_growth = asset_growth

    actual_growth = compute_total_growth(rewards)
    return actual_growth / best_growth

class MetricsLogger:

    def __init__(self):
        self.episodes = []

    def new_episode(self, episode):
        self.episodes.append((episode, []))

    def log_reward(self, reward):
        _, rewards = self.episodes[-1]
        rewards.append(reward)

class EmptyLogger(MetricsLogger):
    def new_episode(self, episode):
        pass

    def log_reward(self, reward):
        pass

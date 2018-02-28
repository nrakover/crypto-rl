'''
Utils for evaluating episodes and experiments.
'''

import numpy as np

SECOND_PER_DAY = 60*60*24

def compute_total_growth(rewards):
    total_reward = np.sum(rewards) # sum of logs of growth factors
    return np.exp(total_reward)

def compute_episode_length_in_days(episode):
    episode_start = episode.history[0].get_timestamp()
    episode_end = episode.history[-1].get_timestamp()
    num_days = 1 + (episode_end - episode_start) / SECOND_PER_DAY
    return num_days

def compute_avg_growth_per_day(episode, rewards):
    total_reward = np.sum(rewards)
    num_days = compute_episode_length_in_days(episode)
    reward_per_day = total_reward / num_days
    return np.exp(reward_per_day)

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
        self.summaries = []
        self.current_episode = None
        self.current_rewards = []

    def new_episode(self, episode):
        # compute summary stats
        if self.current_episode is not None:
            stats = self._compute_summary_stats(self.current_episode, self.current_rewards)
            self.summaries.append(stats)

        # start new episode
        self.current_episode = episode
        self.current_rewards.clear()

    def log_reward(self, reward):
        self.current_rewards.append(reward)

    def finalize(self):
        stats = self._compute_summary_stats(self.current_episode, self.current_rewards)
        self.summaries.append(stats)

    def clear(self):
        self.summaries.clear()
        self.current_episode = None
        self.current_rewards.clear()

    @staticmethod
    def _compute_summary_stats(episode, rewards):
        episode_length = compute_episode_length_in_days(episode)

        total_growth = compute_total_growth(rewards)
        avg_growth_per_day = compute_avg_growth_per_day(episode, rewards)
        relative_growth = total_growth_relative_to_best_pick(episode, rewards)

        return episode_length, total_growth, avg_growth_per_day, relative_growth

class EmptyLogger(MetricsLogger):
    def new_episode(self, episode):
        pass

    def log_reward(self, reward):
        pass

    def finalize(self):
        pass

    def clear(self):
        pass

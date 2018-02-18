'''
APIs for an RL agent.
'''

import random

import numpy as np

class StochasticAgent:
    '''
    An agent from which we can sample actions both on- and off-policy.
    '''

    def __init__(self, num_assets, policy):
        self.num_assets = num_assets
        self.policy = policy

    def sample_action_uniform(self):
        sells = np.random.randint(0, 2, size=(self.num_assets,)).tolist()
        buys = [0.0] * (self.num_assets + 1)
        asset_to_buy = np.random.randint(0, self.num_assets)
        buys[asset_to_buy] = 1.0
        return (sells, buys)

    def sample_action_on_policy(self, state):
        sell_dist, buy_dist = self.policy.forward(state)
        sampled_sells = []
        for i in range(self.num_assets):
            policy_threshold = sell_dist[i, 0]
            if random.random() < policy_threshold:
                sampled_sells.append(1.0)
            else:
                sampled_sells.append(0.0)

        sampled_buys = [0.0] * (self.num_assets + 1)
        asset_to_buy = self._choose_weighted(buy_dist)
        sampled_buys[asset_to_buy] = 1.0

        return (sampled_sells, sampled_buys)

    def get_best_action(self, state):
        sell, buy = self.policy.forward(state)
        return sell.squeeze().tolist(), buy.squeeze().tolist()

    @staticmethod
    def _choose_weighted(distribution):
        p = random.random()
        cumulative_threshold = 0.0
        for i in range(distribution.shape[0]):
            cumulative_threshold += distribution[i, 0]
            if p < cumulative_threshold:
                return i
        return distribution.shape[0] - 1

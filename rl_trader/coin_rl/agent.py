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
        sells = np.random.randint(0, 2, size=(self.num_assets,))
        buys = [0.0] * (self.num_assets + 1)
        asset_to_buy = np.random.randint(0, self.num_assets)
        buys[asset_to_buy] = 1.0
        return (sells, buys)

    def sample_action_on_policy(self, state):
        action_distribution = self.policy.forward(state)
        sampled_sells = []
        for i in range(self.num_assets):
            policy_threshold = action_distribution[i]
            if random.random() < policy_threshold:
                sampled_sells.append(1.0)
            else:
                sampled_sells.append(0.0)

        sampled_buys = [0.0] * (self.num_assets + 1)
        asset_to_buy = self._choose_weighted(action_distribution[0, self.num_assets+1::])
        sampled_buys[asset_to_buy] = 1.0

        return (sampled_sells, sampled_buys)

    def get_best_action(self, state):
        distribution = self.policy.forward(state)
        return self._distribution_to_action(distribution, self.num_assets)

    @staticmethod
    def _distribution_to_action(distribution, num_assets):
        sells = distribution[0, 0:num_assets]
        buys = distribution[0, num_assets::]
        return (sells, buys)

    @staticmethod
    def _choose_weighted(distribution):
        p = random.random()
        cumulative_threshold = 0.0
        for i in range(distribution.shape[1]):
            cumulative_threshold += distribution[0, i]
            if p < cumulative_threshold:
                return i
        return distribution.shape[1] - 1

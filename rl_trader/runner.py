'''
Scripts for training and evaluating RL agents.
'''

import random

from .coin_rl.agent import StochasticAgent
from .coin_rl.environment import Env
from .coin_rl.policy_gradient import PGNetwork

from .utils.metrics import MetricsLogger

def train(env_config, num_assets, num_iterations, network_layers, discount_gamma=0.999, verbose=True):
    # init PG network
    policy = PGNetwork(num_assets, network_layers, discount_gamma)
    # init agent
    agent = StochasticAgent(num_assets, policy)

    # training iterations
    for iteration in range(num_iterations):
        # init environment
        logger = MetricsLogger()
        env = Env(env_config, metrics_logger=logger)

        # learning schedule
        probs_off_policy_sample = 1.0 / (1+iteration)

        # run all episodes
        paths = []
        while env.has_next_episode():
            state = env.reset()
            done = False
            current_path = []
            while not done:
                # sample action
                action = None
                if random.random() < probs_off_policy_sample:
                    action = agent.sample_action_uniform()
                else:
                    action = agent.sample_action_on_policy(state)
                # act
                new_state, reward, done = env.step(action)
                # collect observation
                current_path.append((state, action, reward))
                # next
                state = new_state

            paths.append(current_path)

        # update agent parameters
        cost = policy.take_gradient_step(paths)

        if verbose:
            print ("Iteration {0}:  cost = {1}".format(iteration, cost))

    return policy

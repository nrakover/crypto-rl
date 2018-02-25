'''
Scripts for training and evaluating RL agents.
'''

import random

import tensorflow as tf

from .coin_rl.agent import StochasticAgent
from .coin_rl.environment import Env
from .coin_rl.policy_gradient import PGNetwork

from .utils.metrics import MetricsLogger

def train(env_config, num_assets, num_iterations, network_layers, discount_gamma=0.999, target_buffer_size=32000, verbose=True, tracemalloc=None, num_malloc_stats=5):
    # init PG network
    sess = tf.Session()
    policy = PGNetwork(sess, num_assets, network_layers, discount_gamma)
    policy.initialize()

    # training iterations
    for iteration in range(num_iterations):
        # init environment
        env = Env(env_config)

        # init agent
        exploration_policy = PGNetwork.clone(policy)
        agent = StochasticAgent(num_assets, exploration_policy)

        # learning schedule
        probs_off_policy_sample = 1.0 / (1+iteration)

        # run all episodes
        num_episodes_run = 0
        buffer_size = 0
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
            num_episodes_run += 1
            buffer_size += len(current_path)

            if verbose:
                print("finished episode {}".format(num_episodes_run-1))
            if tracemalloc is not None:
                snapshot = tracemalloc.take_snapshot()
                stats = snapshot.statistics("lineno")
                for stat in stats[0:num_malloc_stats]:
                    print(stat)

            # flush experience buffer and update policy parameters
            if buffer_size >= target_buffer_size:
                _do_policy_update(policy, paths, verbose)
                paths.clear()
                buffer_size = 0

        # update policy parameters
        _do_policy_update(policy, paths, verbose)

        if verbose:
            print("Finished Iteration {0}".format(iteration))
            print("---------------------------------")

    return policy

def _do_policy_update(policy, paths, verbose):
    if verbose:
        print(">>>>>>>> updating policy")
    if len(paths) == 0:
        return
    policy.take_gradient_step(paths)

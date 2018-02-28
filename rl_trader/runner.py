'''
Scripts for training and evaluating RL agents.
'''

import random

import numpy as np
import tensorflow as tf

from .coin_rl.agent import StochasticAgent
from .coin_rl.environment import Env
from .coin_rl.policy_gradient import PGNetwork

from .utils.metrics import MetricsLogger

def train(env_config, num_assets, num_iterations, network_layers, discount_gamma=0.999, target_buffer_size=32000, eval_frequency=0, verbose=True, tracemalloc=None, num_malloc_stats=5):
    # init PG network
    sess = tf.Session()
    policy = PGNetwork(sess, num_assets, network_layers, discount_gamma)
    policy.initialize()

    # establish performance baseline
    if eval_frequency > 0:
        print(">>>>> Pre-training baseline performance")
        eval(env_config, num_assets, policy, detailed_report=verbose)
        print("---------------------------------")

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

            print("finished episode {}".format(num_episodes_run-1))
            if tracemalloc is not None:
                snapshot = tracemalloc.take_snapshot()
                stats = snapshot.statistics("lineno")
                for stat in stats[0:num_malloc_stats]:
                    print(stat)

            # flush experience buffer and update policy parameters
            if buffer_size >= target_buffer_size:
                _do_policy_update(policy, paths)
                paths.clear()
                buffer_size = 0

        # update policy parameters
        _do_policy_update(policy, paths)

        # report progres
        print("Finished Iteration {0}".format(iteration))
        if eval_frequency > 0 and iteration % eval_frequency == 0:
            eval(env_config, num_assets, policy, detailed_report=verbose)
        print("---------------------------------")

    return policy

def _do_policy_update(policy, paths):
    print(">>>>> updating policy")
    if len(paths) == 0:
        return
    policy.take_gradient_step(paths)

def eval(env_config, num_assets, policy, detailed_report=True):
    # init environment
    logger = MetricsLogger()
    env = Env(env_config, metrics_logger=logger)

    # init agent
    agent = StochasticAgent(num_assets, policy)

    rewards = []
    while env.has_next_episode():
        state = env.reset()
        done = False
        while not done:
            # choose best action
            action = agent.get_best_action(state)
            # act
            state, reward, done = env.step(action)
            rewards.append(reward)
    logger.finalize()

    # report results
    print("Number of steps: {}".format(len(rewards)))
    print("Mean reward: {}".format(np.mean(rewards)))
    if detailed_report:
        _produce_detailed_report(logger.summaries)

def _produce_detailed_report(summary_stats):
    print("Number of episodes: {}".format(len(summary_stats)))

    mean_total_growth = np.mean([x for (_, x, _, _) in summary_stats])
    std_total_growth = np.std([x for (_, x, _, _) in summary_stats])
    min_total_growth = np.min([x for (_, x, _, _) in summary_stats])
    print("MEAN total growth: {}".format(mean_total_growth))
    print("STD total growth: {}".format(std_total_growth))
    print("MIN total growth: {}".format(min_total_growth))

    mean_avg_daily_growth = np.mean([x for (_, _, x, _) in summary_stats])
    std_avg_daily_growth = np.std([x for (_, _, x, _) in summary_stats])
    min_avg_daily_growth = np.min([x for (_, _, x, _) in summary_stats])
    print("MEAN avg daily growth: {}".format(mean_avg_daily_growth))
    print("STD avg daily growth: {}".format(std_avg_daily_growth))
    print("MIN avg daily growth: {}".format(min_avg_daily_growth))

    mean_relative_growth = np.mean([x for (_, _, _, x) in summary_stats])
    std_relative_growth = np.std([x for (_, _, _, x) in summary_stats])
    min_relative_growth = np.min([x for (_, _, _, x) in summary_stats])
    print("MEAN relative growth: {}".format(mean_relative_growth))
    print("STD relative growth: {}".format(std_relative_growth))
    print("MIN relative growth: {}".format(min_relative_growth))

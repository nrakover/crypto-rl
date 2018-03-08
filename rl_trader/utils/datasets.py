'''
Utils for working with datasets
'''

import datetime as dt
import random

import numpy as np

from ..coin_rl.episode import Config

TEST_BLOCK_FRACTION = 0.15 # as fraction of total range
MIN_EPISODE_LENGTH = 4 # in days
MAX_EPISODE_LENGTH = 21 # in days
PROBS_ALL_BASE_CURRENCY = 0.2
NUM_EPISODES_AS_SPAN_FRACTION = 0.75

def generate_train_test_configurations(root_dir, train_step_size, test_step_size, start_date, end_date, assets, test_block_fraction=TEST_BLOCK_FRACTION):
    '''
    Generates two lists -- train and test -- of random episode configurations within the given bounds.

    Args:
        root_dir            (string) path to root directory for episodes data
        train_step_size     (integer) episode step size for training configs
        test_step_size      (integer) episode step size for test configs
        start_date          (dt.date) lower bound for episode dates
        end_date            (dt.date) upper bound for episode dates
        assets              (list) asset identifiers for each configuration
        test_block_fraction (float in (0,1)) size of the test block as a fraction of total range

    Returns:
        list of train episode configurations
        list of test episode configurations
    '''
    total_span = (end_date - start_date).days
    test_block_size = max(round(total_span * test_block_fraction), MIN_EPISODE_LENGTH)

    test_block_start_index, test_block_end_index = _get_test_block_indices(total_span, test_block_size)
    test_block_start_date = dt.date.fromordinal(start_date.toordinal() + test_block_start_index)
    test_block_end_date = dt.date.fromordinal(start_date.toordinal() + test_block_end_index)

    # build test configurations
    num_test_episodes = int(NUM_EPISODES_AS_SPAN_FRACTION * test_block_size)
    test_configs = generate_random_configurations(root_dir, test_step_size, test_block_start_date, test_block_end_date, assets, num_test_episodes)

    # build train configurations
    lower_train_block_start_date = start_date
    lower_train_block_end_date = test_block_start_date
    lower_train_block_num_episodes = int(NUM_EPISODES_AS_SPAN_FRACTION * test_block_start_index)
    train_configs = generate_random_configurations(root_dir, train_step_size, lower_train_block_start_date, lower_train_block_end_date, assets, lower_train_block_num_episodes)

    upper_train_block_start_date = test_block_end_date
    upper_train_block_end_date = end_date
    upper_train_block_num_episodes = int(NUM_EPISODES_AS_SPAN_FRACTION * (total_span - test_block_end_index))
    upper_train_block_configs = generate_random_configurations(root_dir, train_step_size, upper_train_block_start_date, upper_train_block_end_date, assets, upper_train_block_num_episodes)
    train_configs.extend(upper_train_block_configs)

    return train_configs, test_configs

def _get_test_block_indices(total_span, test_block_size):
    valid_start_indices = total_span - test_block_size - 2 * (MIN_EPISODE_LENGTH - 1)

    start = random.randint(0, valid_start_indices)
    # adjust lower end -- if 0th index then keep in place, otherwise shift up to leave
    # enought space below for a full episode
    if start > 0 and start < total_span - test_block_size - (MIN_EPISODE_LENGTH - 1):
        start += (MIN_EPISODE_LENGTH - 1)
    elif start > 0:
        start = total_span - test_block_size

    end = start + test_block_size

    return start, end

def generate_random_configurations(root_dir, step_size, start_date, end_date, assets, num_episodes):
    '''
    Generates a list of random episode configurations within the given bounds.

    Args:
        root_dir            (string) path to root directory for episodes data
        step_size           (integer) episode step size
        start_date          (dt.date) lower bound (inclusive) for episode dates
        end_date            (dt.date) upper bound (exclusive) for episode dates
        assets              (list) asset identifiers for each configuration
        num_episodes        (integer) number of episodes to generate

    Returns:
        list of episode configurations
    '''
    ranges = generate_episode_ranges(start_date, end_date, num_episodes)
    allocations = generate_random_allocations(len(assets), num_episodes)
    configurations = []
    for i in range(num_episodes):
        start, end = ranges[i]
        starting_allocation = allocations[i]
        configurations.append(Config(root_dir, start, end, assets, step_size, starting_allocation))
    return configurations

def generate_random_allocations(num_assets, num_episodes):
    '''
    Generates a list of random assset allocations.

    Args:
        num_assets      (integer) number of assets to allocate to
        num_episodes    (integer) number of episodes for which to generate allocations

    Returns:
        list of allocations, each a list representing a distribution over the assets and base currency
    '''
    allocations = []
    for _ in range(num_episodes):
        if random.random() < PROBS_ALL_BASE_CURRENCY:
            allocations.append([1.0] + [0.0] * num_assets)
        else:
            pre_normalized = np.random.uniform(size=(num_assets+1))
            distribution = pre_normalized / np.sum(pre_normalized)
            allocations.append(distribution.tolist())
    return allocations

def generate_episode_ranges(start_date, end_date, num_episodes):
    '''
    Generates a list of random episode ranges within the given bounds.

    Args:
        start_date          (dt.date) lower bound (inclusive) for episode dates
        end_date            (dt.date) upper bound (exclusive) for episode dates
        num_episodes        (integer) number of episodes to generate

    Returns:
        list of episode ranges as tuples (start, end) of dt.date
    '''
    total_span = (end_date - start_date).days
    valid_start_indices = total_span - MIN_EPISODE_LENGTH
    start_date_ordinal = start_date.toordinal()
    ranges = []
    for _ in range(num_episodes):
        range_start_index = random.randint(0, valid_start_indices)
        range_end_index = random.randint(range_start_index + MIN_EPISODE_LENGTH, min(range_start_index + MAX_EPISODE_LENGTH, total_span))

        range_start_date = dt.date.fromordinal(start_date_ordinal + range_start_index)
        range_end_date = dt.date.fromordinal(start_date_ordinal + range_end_index)
        ranges.append((_date_to_str(range_start_date), _date_to_str(range_end_date)))

    return ranges

def _date_to_str(date):
    return "{0}-{1}-{2}".format(date.year, date.month, date.day)

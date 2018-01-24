'''
API for a single RL episode.
'''

import csv
import datetime as dt
import math
import os

from . import market
from .state import State

class Episode:
    '''
    Abstraction for a single executable RL episode.

    Specified by a segment of market history, a starting asset allocation, and a step size.
    '''

    def __init__(self, history, starting_allocation, step_size):
        self.history = history
        self.starting_allocation = starting_allocation
        self.step_size = step_size

        self.t = None
        self.allocation = None

    def reset(self):
        '''
        Returns the starting state of the episode.
        '''
        self.t = State.find_starting_timestep(self.history)
        self.allocation = self.starting_allocation[:]
        return State.create(self.history, self.t, self.allocation)

    def step(self, action):
        '''
        Performs the action at the current state.

        Args
            action: agent action

        Returns
            state:      new state
            reward:     reward for action given state transition
            terminal:   boolean indicating whether the new state is terminal
        '''
        assert self.t is not None
        assert self.allocation is not None

        pre_act_allocation = self.allocation[:]
        post_act_allocation = self._do_action(pre_act_allocation, action)

        next_t = self.t + self.step_size
        post_market_allocation, reward = self._compute_portfolio_change(
            post_act_allocation, self.history[self.t], self.history[next_t])
        self.t = next_t
        self.allocation = post_market_allocation
        new_state = State.create(self.history, self.t, self.allocation)

        return new_state, reward, (self.t + self.step_size >= len(self.history))

    @staticmethod
    def _do_action(allocation, action):
        (sell, buy) = action
        new_allocation = [0] * len(allocation)
        new_allocation[0] = allocation[0] # init with base currency allocation

        # process sales
        for asset_indx in range(len(sell)):
            amount_sold = allocation[asset_indx + 1] * sell[asset_indx]
            remaining = allocation[asset_indx + 1] - amount_sold

            new_allocation[0] += amount_sold            # update base currency amount
            new_allocation[asset_indx + 1] = remaining  # set remaining asset amount

        # process purchases
        pre_buy_base_currency_amount = new_allocation[0]    # redistribute the base currency post sales
        new_allocation[0] = 0
        for asset_indx in range(len(buy)):
            new_allocation[asset_indx] += pre_buy_base_currency_amount * buy[asset_indx]

        return new_allocation

    @staticmethod
    def _compute_portfolio_change(allocation, starting_market_snapshot, ending_market_snapshot):
        growth_scaled_allocation = []
        for asset_indx in range(len(allocation)):
            starting_value = starting_market_snapshot.get_asset_value(asset_indx)
            ending_value = ending_market_snapshot.get_asset_value(asset_indx)
            growth_factor = ending_value / starting_value

            growth_scaled_allocation.append(allocation[asset_indx] * growth_factor)
        
        net_growth = sum(growth_scaled_allocation)
        new_allocation = [value/net_growth for value in growth_scaled_allocation] # normalize
        
        return new_allocation, math.log(net_growth) # reward is the log of the net growth factor
    
    @staticmethod
    def build(config):
        start_date = Episode._parse_date(config.start)
        end_date = Episode._parse_date(config.end)

        histories = [[]] * len(config.assets)
        current_date = start_date
        while current_date < end_date:
            current_date_string = "{0}-{1}-{2}".format(current_date.year, current_date.month, current_date.day)
            for (asset_indx, asset) in enumerate(config.assets):
                bucket_path = os.path.join(config.root_dir, asset, current_date_string) + '.csv'
                assert os.path.exists(bucket_path)
                with open(bucket_path, 'r') as bucket_file:
                    reader = csv.reader(bucket_file)
                    for line in reader:
                        histories[asset_indx].append(market.Candle.from_list(line))

            current_date = dt.date.fromordinal(current_date.toordinal() + 1)

        combined_history = market.collate_multi_asset_history(histories)
        return Episode(combined_history, config.starting_allocation, config.step_size)

    @staticmethod
    def _parse_date(string):
        return dt.datetime.strptime(string, '%Y-%m-%d').replace(tzinfo=dt.timezone.utc).date()

class Config:
    '''
    Episode configuration.
    '''

    SETTING_DELIMITER = '|'
    ASSETS_DELIMITER = ','
    ALLOCATION_DELIMITER = ':'

    def __init__(self, root_dir, start, end, assets, step_size, starting_allocation):
        assert len(assets) == len(starting_allocation) - 1

        self.root_dir = root_dir
        self.start = start
        self.end = end
        self.assets = assets
        self.step_size = step_size
        self.starting_allocation = starting_allocation

    @staticmethod
    def parse(config_string):
        assert isinstance(config_string, str)
        settings = config_string.strip().split(Config.SETTING_DELIMITER)

        root_dir = settings[0]
        start = settings[1]
        end = settings[2]
        assets = settings[3].split(Config.ASSETS_DELIMITER)
        step_size = int(settings[4])
        starting_allocation = [float(x) for x in settings[5].split(Config.ALLOCATION_DELIMITER)]

        return Config(root_dir, start, end, assets, step_size, starting_allocation)

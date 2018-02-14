'''
API for working with agent state-space.
'''

import numpy as np

class State:
    '''
    Abstraction for a state object.
    '''

    NUM_SNAPSHOTS_IN_STATE_WINDOW = 1440 #1 day

    def __init__(self, snapshot, allocation, features):
        self.snapshot = snapshot
        self.allocation = allocation
        self.features = features

    def get_features(self):
        '''
        Returns a numpy array (dtype=np.float) representing a feature column.
        '''
        return self.features

    def __str__(self):
        return "{0}::{1}".format(self.snapshot.get_timestamp(), self.allocation)

    @staticmethod
    def create(history, timestep, allocation):
        return State(history[timestep], allocation, State._compute_features(history, timestep, allocation))

    @staticmethod
    def _compute_features(history, timestep, allocation):
        features = []
        for t in range(timestep, timestep + State.NUM_SNAPSHOTS_IN_STATE_WINDOW):
            snapshot = history[t]
            for asset_indx in range(snapshot.get_num_assets()):
                candle = snapshot.get_asset_candle(asset_indx)
                features.extend(State._get_candle_features(candle))
        features.extend(allocation)
        return np.array(features, dtype=np.float).reshape((-1, 1))

    @staticmethod
    def _get_candle_features(candle):
        return [candle.open, candle.close, candle.high, candle.low, candle.volume]

    @staticmethod
    def get_starting_timestep():
        '''
        Get the earliest starting timestep in a history such that the state window can be computed.
        '''
        return State.NUM_SNAPSHOTS_IN_STATE_WINDOW - 1

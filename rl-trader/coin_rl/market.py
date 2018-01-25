'''
API for working with market snapshots.
'''

class Snapshot:
    '''
    A snapshot in market history.
    '''

    def __init__(self, timestamp, asset_candles):
        assert all([c.timestamp == timestamp for c in asset_candles]) # make sure all assets' timestamps line up
        self.timestamp = timestamp
        self.asset_candles = asset_candles
    
    def get_timestamp(self):
        '''
        Returns the snapshot's timestamp.
        '''
        return self.timestamp

    def get_asset_value(self, asset_index):
        '''
        Returns the asset's market value.

        Computed as the midpoint between the high and low for that period.
        '''
        if asset_index == 0:
            return 1.0

        low = self.asset_candles[asset_index-1].low
        high = self.asset_candles[asset_index-1].high
        return (low + high) / 2.0

class Candle:
    '''
    A market "candle", encoding the low, high, opening and closing prices as well as
    the volume and starting timestamp of some window of time.
    '''

    def __init__(self, timestamp, low_price, high_price, open_price, close_price, volume, window_size=60):
        self.timestamp = timestamp
        self.low = low_price
        self.high = high_price
        self.open = open_price
        self.close = close_price
        self.volume = volume
        self.window_size = window_size

    def __str__(self):
        return "[{0},{1},{2}]".format(self.timestamp, self.low, self.high)

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def from_list(fields_list):
        '''
        Instantiate Candle from a list containing the [timestamp, low, high, open, close, volume]
        '''
        return Candle(
            int(fields_list[0]),
            float(fields_list[1]),
            float(fields_list[2]),
            float(fields_list[3]),
            float(fields_list[4]),
            float(fields_list[5])
        )


def _interpolate_candles(candle1, candle2, timestamp, granularity):
    scaling = (timestamp - candle1.timestamp) / (candle2.timestamp - candle1.timestamp)
    return Candle(
        timestamp,
        (candle2.low - candle1.low) * scaling + candle1.low,
        (candle2.high - candle1.high) * scaling + candle1.high,
        (candle2.open - candle1.open) * scaling + candle1.open,
        (candle2.close - candle1.close) * scaling + candle1.close,
        (candle2.volume - candle1.volume) * scaling + candle1.volume,
        granularity
    )

def collate_multi_asset_history(histories, granularity=60):
    num_assets = len(histories)
    snapshots = []

    first_timestamp = max([h[0].timestamp for h in histories])
    max_timestamp = min([h[-1].timestamp for h in histories])

    asset_indices = [0] * num_assets
    current_timestamp = first_timestamp
    while current_timestamp <= max_timestamp:
        asset_candles = []
        for asset in range(num_assets):
            asset_history = histories[asset]

            # move up the history until we reach the last timestamp prior to current_timestamp
            while asset_history[asset_indices[asset] + 1].timestamp < current_timestamp:
                asset_indices[asset] = asset_indices[asset] + 1

            # get stradling candles
            asset_indx = asset_indices[asset]
            candle = asset_history[asset_indx]
            next_candle = asset_history[min(len(asset_history), asset_indx+1)]
            # interpolate them at the target timestamp
            interpolated_candle = _interpolate_candles(candle, next_candle, current_timestamp, granularity)
            asset_candles.append(interpolated_candle)

        snapshots.append(Snapshot(current_timestamp, asset_candles))
        current_timestamp = current_timestamp + granularity

    return snapshots

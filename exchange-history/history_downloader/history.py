import datetime as dt
import time

import requests

from .client import ExchangeClient

class History:
    '''
    API to access the GDAX historical records.
    '''

    MAX_CANDLES_SIZE = 200
    SLEEP_TIME_BETWEEN_REQUESTS_SEC = 0.2
    STARTING_SLEEP_TIME_AFTER_REJECTION_SEC = 1.0

    def __init__(self, exchange_client):
        assert isinstance(exchange_client, ExchangeClient)
        self.client = exchange_client

    def stream_candles(self, product, start, end, granularity=60):
        '''
        Generator function for candles within the specified range.

        Args
            product (string): the product identifier
            start (datetime): starting time, inclusive
            end (datetime): ending time, exclusive
            granularity (int): the candle size in seconds, defaults to 60
        
        Yields instances of Candle
        '''
        assert start < end
        range_start = start.timestamp()
        range_end = min(end.timestamp(), range_start + History.MAX_CANDLES_SIZE * granularity)

        while range_end <= end.timestamp():
            candles = self._get_candles_range(product, range_start, range_end, granularity)
            for c in candles:
                yield c
            
            range_start = range_end
            range_end = max(range_start + granularity, min(end.timestamp(), range_start + History.MAX_CANDLES_SIZE * granularity))
            time.sleep(History.SLEEP_TIME_BETWEEN_REQUESTS_SEC)
    
    def _get_candles_range(self, product, start, end, granularity):
        '''
        Returns a list of the candles in the specified range.

        Args
            product (string): the product identifier
            start (int, timestamp): start time, inclusive
            end (int, timestamp): end time, exclusive
            granularity (int)
        '''
        assert (end - start) / granularity <= 200
        iso_start = dt.datetime.fromtimestamp(start, dt.timezone.utc).isoformat()
        iso_end = dt.datetime.fromtimestamp(end, dt.timezone.utc).isoformat()
        raw_candles = None
        sleep_time = History.STARTING_SLEEP_TIME_AFTER_REJECTION_SEC
        attempt_count = 1
        while raw_candles is None:
            try:
                raw_candles = self.client.get_candles(product, start=iso_start, end=iso_end, granularity=granularity)
            except requests.exceptions.HTTPError as err:
                print ('Failed attempt {}:'.format(attempt_count), err)
                time.sleep(sleep_time)
                sleep_time = min(10, sleep_time * 2)
                attempt_count += 1
                raw_candles = None
            
        return reversed([Candle(raw) for raw in raw_candles])

class Candle:
    '''
    Abstraction for the candle entries returned by the GDAX client.
    '''
    def __init__(self, data_list):
        assert len(data_list) == 6, data_list
        self.timestamp = data_list[0]
        self.low = data_list[1]
        self.high = data_list[2]
        self.open = data_list[3]
        self.close = data_list[4]
        self.volume = data_list[5]
    
    def as_list(self):
        return [self.timestamp, self.low, self.high, self.open, self.close, self.volume]
    
    def __str__(self):
        return "(timestamp={0}, low={1}, high={2}, open={3}, close={4}, volume={5})".format(
            self.timestamp, self.low, self.high, self.open, self.close, self.volume
        )

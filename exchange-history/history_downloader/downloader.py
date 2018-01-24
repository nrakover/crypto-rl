
import csv
import datetime as dt
import os

from . import history
from .client import ExchangeClient

class Downloader:

    def __init__(self, exchange_client):
        assert isinstance(exchange_client, ExchangeClient)
        self.client = exchange_client

    def download(self, product, start, end, target_dir, bucket_by='day', candle_granularity=60, verbose=True, print_frequency=10):
        # check that target dir exists, create it if missing
        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)

        # buffer for a single bucket's snapshots
        buffer = []
        current_bucket = Downloader._compute_bucket(start.timestamp(), bucket_by)

        exchange_history = history.History(self.client)
        bucket_counter = 0
        debug_start_time = dt.datetime.now()
        # consume stream of snapshots
        for candle in exchange_history.stream_candles(product, start, end, granularity=candle_granularity):
            # test if it belongs to the current bucket
            if Downloader._compute_bucket(candle.timestamp, bucket_by) == current_bucket:
                buffer.append(candle)
            else:
                # flush buffer
                Downloader._write_bucket(buffer, current_bucket, target_dir)
                buffer.clear()

                # show progress
                bucket_counter += 1
                if verbose and bucket_counter % print_frequency == 0:
                    elapsed_time = (dt.datetime.now() - debug_start_time).total_seconds()
                    print (current_bucket, '(time elapsed {0}:{1})'.format(int(elapsed_time / 60), elapsed_time % 60))
                
                # move on to next bucket
                current_bucket = Downloader._compute_bucket(candle.timestamp, bucket_by)
                buffer.append(candle)
        
        # flush final bucket
        Downloader._write_bucket(buffer, current_bucket, target_dir)

    @staticmethod
    def _compute_bucket(timestamp, bucket_by):
        full_dt = dt.datetime.fromtimestamp(timestamp, dt.timezone.utc)
        if bucket_by == 'day':
            return '{0}-{1}-{2}'.format(full_dt.year, full_dt.month, full_dt.day)
        elif bucket_by == 'month':
            return '{0}-{1}'.format(full_dt.year, full_dt.month)
        else:
            raise NotImplementedError()
    
    @staticmethod
    def _write_bucket(buffer, bucket, parent_dir):
        bucket_file = os.path.join(parent_dir, bucket) + '.csv'
        with open(bucket_file, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            for candle in buffer:
                writer.writerow(candle.as_list())

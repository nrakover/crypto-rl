import requests

from history_downloader.client import ExchangeClient

class GDAXClient(ExchangeClient):

    URL = 'https://api.gdax.com'

    @staticmethod
    def get_candles(product, start, end, granularity):
        params = {}
        if start is not None:
            params['start'] = start
        if end is not None:
            params['end'] = end
        if granularity is not None:
            params['granularity'] = granularity
        r = requests.get(GDAXClient.URL + '/products/{}/candles'
                         .format(product), params=params)
        r.raise_for_status()
        return r.json()
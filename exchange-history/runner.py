
import datetime as dt

import history_downloader as history
import gdax

def run_downloader(product, start, end, target_dir):
    history.downloader.Downloader(gdax.GDAXClient()).download(product, start, end, target_dir)

def parse_datetime(string):
    return dt.datetime.strptime(string, '%Y-%m-%d').replace(tzinfo=dt.timezone.utc)

if __name__ == '__main__':
    from sys import argv
    assert len(argv) == 5, "expected arguments: 'product', 'start', 'end', 'target dir'"
    run_downloader(argv[1], parse_datetime(argv[2]), parse_datetime(argv[3]), argv[4])

import datetime

import time


def estimate_eta(start_time, current_n, all_n):
    elapsed_time = time.time() - start_time
    remaining_time = elapsed_time * 1.0 / current_n * (all_n - current_n)
    eta = datetime.timedelta(seconds=remaining_time)
    return eta

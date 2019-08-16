import os


def request_linear_algebra_single_threaded():
    if str(os.environ.get('OMP_NUM_THREADS', None)) != '1' or str(os.environ.get('MKL_NUM_THREADS', None)) != '1' or \
            str(os.environ.get('OPENBLAS_NUM_THREADS', None)) != '1':
        raise ValueError('You must set OMP_NUM_THREADS,MKL_NUM_THREADS,OPENBLAS_NUM_THREADS as "1" '
                         'in your environment variables!')

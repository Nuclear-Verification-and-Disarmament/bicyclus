"""Collection of logging functions.

Please note that these are currently written for usage on RWTH Aachen's
SLURM-based high-performance compute cluster. SLURM environment variables may
be different or non-existent on other HPCs or PCs.
"""
from datetime import datetime

import random
import string
import sys
import os
from os import path


def task_identifier():
    return '{}_{}_{:03d}'.format(
        os.environ.get('SLURM_JOB_NAME', 'local'),
        os.environ.get('SLURM_ARRAY_JOB_ID',
                       os.environ.get('SLURM_JOB_ID', '0')),
        int(os.environ.get('SLURM_ARRAY_TASK_ID', '0')))


def log_file_path(outpath=None, run='', typ='', ending='log'):
    if outpath is None:
        outpath = os.environ.get('WORK', os.environ.get('HOME'))
    os.makedirs(outpath, exist_ok=True)
    outfile = path.join(
        outpath, 'sampling_log_{}_{}_{}.{}'.format(run, typ, task_identifier(),
                                                   ending))
    return outfile


def write_to_log_file(outpath=None, run=''):
    outfile = log_file_path(outpath, run)
    fh = open(outfile, 'w', buffering=1)
    print("write_to_log_file(): Writing to program-defined log at {}".format(
        outfile))
    sys.stdout = fh
    sys.stderr = fh

    log_init_debug_info()


def log_init_debug_info():
    print('Environment:')
    for (k, v) in sorted(os.environ.items(), key=lambda t: t[0]):
        print(k, '=', v)


def log_print(*args):
    print(datetime.now(), '::', *args)

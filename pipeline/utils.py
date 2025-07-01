import pandas as pd
import numpy as np
import time
import psutil
import os
import sys
import gc
import logging
from mostlyai.qa._accuracy import bin_data, calculate_univariates, calculate_bivariates, calculate_trivariates

logger = logging.getLogger(__name__)

def time_it(func):
    """Decorator to time a function's execution."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Function '{func.__name__}' executed in {(end_time - start_time)/60:.2f} minutes.")
        return result
    return wrapper


def print_memory_consumption():
    """
    Prints current memory consumption and the largest objects in memory,
    with accurate reporting for Pandas DataFrames.
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    rss_in_gb = memory_info.rss / (1024 * 1024 * 1024)
    logger.info(f"Current memory consumption: {rss_in_gb:.2f} GB")

def calculate_accuracy(original_data, synthetic_data, variate_level=3):
    ori_bin, bins = bin_data(df=original_data, bins=10)
    syn_bin, _ = bin_data(df=synthetic_data, bins=bins)

    # mimick mostly columns
    ori_bin.columns = ["tgt::" + c for c in ori_bin.columns]
    syn_bin.columns = ["tgt::" + c for c in syn_bin.columns]

    res = {}

    if variate_level >= 1:
        acc_uni = calculate_univariates(ori_bin, syn_bin)
        res["univariate_accuracy"] = acc_uni["accuracy"].mean()
    if variate_level >= 2:
        acc_biv = calculate_bivariates(ori_bin, syn_bin)
        res["bivariate_accuracy"] = acc_biv["accuracy"].mean()
    if variate_level >= 3:
        acc_triv = calculate_trivariates(ori_bin, syn_bin)
        res["trivariate_accuracy"] = acc_triv["accuracy"].mean()

    res["overall_accuracy"] = np.mean(list(res.values()))
    return res
import os
import re
import pandas as pd
from dataset import DS_NAMES


def get_latest_results_directory(dir_path="./results"):
    '''
    Utility function for finding the directory of the latest run

    Parameters
    ----------
    dir_path : the path to a directory containing the results folders (which have names 'run-000', 'run-001', etc)

    Returns
    -------
    max_run_id : the integer ID of the most recent run
    max_run_path : the absolute path to the most recent run directory
    '''
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    # find the next availible run_id in the specified results directory
    max_run_id = 0
    dir_names = os.listdir(dir_path)
    for name in dir_names:
        x = re.match("run-(\d{3})", name)
        if x is not None:
            found_run_id = int(x.group(1))
            if found_run_id > max_run_id:
                max_run_id = found_run_id

    # return the run_id and a path to that folder
    max_run_dir = "run-{:03d}".format(max_run_id)
    max_run_path = os.path.join(os.path.abspath(dir_path), max_run_dir)

    return max_run_id, max_run_path


def make_fig_directory(run_id):
    fig_output_path = "run-{:03d}".format(run_id)
    try:
        os.makedirs(fig_output_path)
    except FileExistsError:
        pass

    return fig_output_path


def load_results(run_path, group_fields=["n_features"]):
    file_names = os.listdir(run_path)
    found_ds = []
    for name in file_names:
        x = re.match("(.*)[.]csv", name)
        if x is not None:
            match_text = x.group(1)
            if match_text in DS_NAMES:
                found_ds.append(match_text)

    results = {}
    for ds in found_ds:
        results[ds] = pd.read_csv(run_path + "/" + ds + ".csv").groupby(group_fields).mean().reset_index()

    return found_ds, results

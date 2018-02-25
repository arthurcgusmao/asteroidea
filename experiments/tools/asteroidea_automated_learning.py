"""Reads all files named 'dataset_...' in the folder given as first argument
and learns the optimal parameters for them using asteroidea. The structure file
to be considered should be named 'structure.pl'.

The dataset files should be named 'dataset_XXX.pl' or 'dataset_XXX.csv'. This
script will automatically read the file type and know if the dataset is
relational or propositional.
"""

# add the parent folder to python path
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
# other imports
import time
import logging
import pandas as pd
from asteroidea.missing_learner import Learner


# configure logging
logging.basicConfig(filename=('./logs/asteroidea_autolearn_{}.log'.format(int(time.time()))),
                    format='%(asctime)s | %(levelname)s : %(message)s',
                    level=logging.INFO)

_, experiment_dir = sys.argv
# experiment_dir = os.getcwd() + '/' + experiment_dir
experiment_dir = os.path.abspath(experiment_dir)

def find_dataset_and_structure_files(dataset_prefix="dataset_"):
    logging.info("Finding dataset and structure files for folder: '{}'...".format(experiment_dir))
    files = os.listdir(experiment_dir)
    dataset_files = []
    dataset_filetype = None
    # read dataset files
    for f in files:
        if dataset_prefix in f:
            if 'problog_' in f:
                continue
            dataset_files.append(f)
            f_type = f.split('.')[-1]
            if dataset_filetype == None:
                dataset_filetype = f_type
            elif dataset_filetype != f_type:
                raise Exception("Not all datasets are of the same filetype.")
            dataset_filetype = dataset_files[0].split('.')[1]
    dataset_files.sort()
    if dataset_filetype == 'csv':
        relational_data = False
    elif dataset_filetype == 'pl':
        relational_data = True
    else:
        raise Exception("Filetype of datasets should be either '.csv' or '.pl'")

    # run the function again if its relational data, but now getting the files with prefix problog_
    ## if relational_data == True and dataset_prefix == "dataset_":
    ##     return find_dataset_and_structure_files(dataset_prefix="problog_dset_")

    # checks if structure file is present
    structure_file = "structure.pl"
    if structure_file not in files:
        raise Exception("'structure.pl' was not found in experiment directory.")
    logging.info("Ok")
    structure_filepath = experiment_dir + "/" + structure_file
    dataset_filepaths = [experiment_dir + "/" + dataset_file for dataset_file in dataset_files]
    return structure_filepath, dataset_filepaths, relational_data


def save_results(df, dataset_name, time_):
    experiment_dir_name = os.path.basename(experiment_dir)
    results_filepath = './results/{}/asteroidea/{}/asteroidea___{}___{}___{}.csv'.format(
        experiment_dir_name, time_, experiment_dir_name, dataset_name, time_)
    dirname = os.path.dirname(results_filepath)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    # save file
    df.to_csv(results_filepath)


def run():
    logging.info("STARTING AUTOMATED LEARNING")
    structure_filepath, dataset_filepaths, relational_data = find_dataset_and_structure_files()
    results = {'filename': [],
               'time': [],
               'log-likelihood': []}
    exp_time_ = int(time.time())
    for dataset_filepath in dataset_filepaths:
        dataset_filename = dataset_filepath.split('/')[-1]
        logging.info("Learning for dataset '{}'...".format(dataset_filename))

        start_time = time.time()

        learner = Learner(structure_filepath,
                  dataset_filepath=dataset_filepath,
                  relational_data=relational_data)
        learning_info=learner.learn_parameters(epsilon=0.00001)
        print (learning_info)
        ll=float(learning_info['df'].iloc[-1:]['Log-Likelihood'])
        model=learner.model

        end_time = time.time()
        learning_time = end_time - start_time

        logging.info("Learned:\nDATASET '{}'\nTime: {}\nLog-Likelihood: {}\nModel: {}".format(dataset_filename, learning_time, ll, model))
        save_results(learner.info['df'], dataset_filename, exp_time_)
    return results
run()

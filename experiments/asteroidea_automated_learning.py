"""Reads all files named 'dataset_...' in the folder given as first argument
and learns the optimal parameters for them using asteroidea. The structure file
to be considered should be named 'structure.pl'.

The dataset files should be named 'dataset_XXX.pl' or 'dataset_XXX.csv'. This
script will automatically read the file type and know if the dataset is
relational or propositional.
"""

# add the parent folder to python path
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# other imports
import time
import logging
import pandas as pd
from asteroidea.missing_learner import Learner


# configure logging
logging.basicConfig(filename=(__file__ + '.log'),
                    format='%(asctime)s | %(levelname)s : %(message)s',
                    level=logging.INFO)

_, experiment_dir = sys.argv
experiment_dir = os.getcwd() + '/' + experiment_dir

def find_dataset_and_structure_files(dataset_prefix="dataset_"):
    logging.info("Finding dataset and structure files for folder: '{}'...".format(experiment_dir))
    files = os.listdir(experiment_dir)
    dataset_files = []
    dataset_filetype = None
    # read dataset files
    for f in files:
        if dataset_prefix in f:
            if '_problog.pl' in f:
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


def get_dataset(dataset_filepath, relational_data):
    if relational_data:
        dataset = dataset_filepath
    else:
        print(dataset_filepath)
        dataset = pd.read_csv(dataset_filepath)
    return dataset



def store_results(results, dataset_filename, learning_time, ll, model):
    results['filename'].append(dataset_filename)
    results['time'].append(learning_time)
    results['log-likelihood'].append(ll)
    params = []
    for head in model:
        for rule in model[head]['rules']:
            clause_string = rule['clause_string']
            if clause_string not in results:
                results[clause_string] = []
            results[clause_string].append(rule['parameter'])
    return results


def write_results_to_file(results, filename):
    df = pd.DataFrame(results)
    df.to_csv(filename)


def save_results(df, dataset_name):
    experiment_dir_name = experiment_dir.split('/')[-2]
    results_folder = 'asteroidea_results/' + experiment_dir_name + '/'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    dataset_name = dataset_name.split('.')[0] + '.csv'
    df.to_csv(results_folder + dataset_name)


def run():
    logging.info("STARTING AUTOMATED LEARNING")
    structure_filepath, dataset_filepaths, relational_data = find_dataset_and_structure_files()
    results = {'filename': [],
               'time': [],
               'log-likelihood': []}
    results_filename=__file__ +'.results.'+ str(int(time.time())) + '.csv'
    for dataset_filepath in dataset_filepaths:
        dataset_filename = dataset_filepath.split('/')[-1]
        logging.info("Learning for dataset '{}'...".format(dataset_filename))
        dataset = get_dataset(dataset_filepath, relational_data)

        start_time = time.time()

        # learner = Learner(structure_filepath, relational_data=relational_data)
        # ll, model = learner.learn_parameters(dataset)
        learner = Learner(structure_filepath,
                  dataset_filepath=dataset_filepath,
                  relational_data=True)
        learning_info=learner.learn_parameters(epsilon=0.0000001)
        print (learning_info)
        ll=float(learning_info['df'].iloc[-1:]['Log-Likelihood'])
        model=learner.model

        end_time = time.time()
        learning_time = end_time - start_time

        logging.info("Learned:\nDATASET '{}'\nTime: {}\nLog-Likelihood: {}\nModel: {}".format(dataset_filename, learning_time, ll, model))
        # results = store_results(results, dataset_filename, learning_time, ll, learner.model)
        # logging.debug("Result added: {}".format(results))
        # write_results_to_file(results, results_filename)
        save_results(learner.info['df'], dataset_filename)
    return results
run()

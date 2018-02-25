"""Reads all files named 'problog_dataset_...' in the folder given as first argument
and learns the optimal parameters for them using problog. The structure file
to be considered should be named 'problog_structure.pl'.

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

from problog.logic import Term
from problog.program import PrologString
from problog.learning import lfi
# needed to read the final output of problog
# from problog_parse_structure_wo_file import read_structure_not_file
from asteroidea.parser import *


# configure logging
logging.basicConfig(filename=('./logs/problog_autolearn_{}.log'.format(int(time.time()))),
                    format='%(asctime)s | %(levelname)s : %(message)s',
                    level=logging.INFO)

# globals
_, experiment_dir = sys.argv
# experiment_dir = os.getcwd() + '/' + experiment_dir
experiment_dir = os.path.abspath(experiment_dir)


def find_dataset_and_structure_files(dataset_prefix="problog_dataset_",
                                     structure_filename='problog_structure.pl'):
    logging.info("Finding dataset and structure files for folder: '{}'...".format(experiment_dir))
    files = os.listdir(experiment_dir)
    dataset_files = []
    dataset_filetype = None
    # read dataset files
    for f in files:
        if dataset_prefix in f:
            dataset_files.append(f)
            f_type = f.split('.')[-1]
            if dataset_filetype == None:
                dataset_filetype = f_type
            elif dataset_filetype != f_type:
                raise Exception("Not all datasets are of the same filetype.")
            # dataset_filetype = dataset_files[0].split('.')[1]

    # if no dataset file was found, then the dataset may be propositional. Call the function again,
    # but this time using only 'dataset_' as prefix.
    if len(dataset_files) == 0:
        return find_dataset_and_structure_files(dataset_prefix='dataset_',
                                                structure_filename='structure.pl')

    dataset_files.sort()
    if dataset_filetype == 'csv':
        relational_data = False
    elif dataset_filetype == 'pl':
        relational_data = True
    else:
        raise Exception("Filetype of datasets should be either '.csv' or '.pl'")

    # checks if structure file is present
    if structure_filename not in files:
        raise Exception("'{}' was not found in experiment directory.".format(structure_file))
    logging.info("Ok")
    structure_filepath = experiment_dir + "/" + structure_filename
    dataset_filepaths = [experiment_dir + "/" + dataset_file for dataset_file in dataset_files]
    return structure_filepath, dataset_filepaths, relational_data



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


def save_results_to_file(results, results_filepath):
    df = pd.DataFrame(results)
    # ensure directory where file should be saved exists
    dirname = os.path.dirname(results_filepath)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    # save file
    df.to_csv(results_filepath)


def read_structure_from_file(structure_filepath):
    f = open(structure_filepath, 'r')
    content = f.read()
    f.close()
    return content


def parse_dataset_to_problog(dataset):
    """Parses a propositional dataset to be fed into problog's lfi.
    """
    columns_list = list(dataset.columns)
    terms_dict = {}
    for column in columns_list:
        term = Term(column)
        terms_dict[column] = term
    examples = []
    for index, row in dataset.iterrows():
        example = []
        for column, value in row.iteritems():
            if value == 1:
                example.append((terms_dict[column], True))
            if value == 0:
                example.append((terms_dict[column], False))
        examples.append(example)
    return examples


def parse_structure_to_problog(structure):
    """Parses a default structure file to be fed into problog's lfi for the propositional case.
    """
    problog_structure = ''
    for line in structure.split('\n'):
        # comment syntax
        if '%' in line:
            continue
        # remove whitespace
        line = line.replace(' ', '')
        # skip empty lines
        if line == '':
            continue
        # modify line
        if '::' in line:
            param, rest = line.split('::')
            param = "t(%s)" % param
            line = param +'::'+ rest
            problog_structure += line + '\n'
    return problog_structure



# this is needed to interpret the final model produced by problog
def read_structure_not_file(structure, relational_data=False):
    """Reads a file containing a structure for the PLP program and returns a
    model dict. The structure should be written accordingly to ProbLog's
    syntax. The structure is stored in a dict called model where each key is
    the head of the rules and the value another dict consisting of two
    elements: rules dict (a list of rules for that head) and parents set (the
    set of parents for that head).

    Each rule in the rules list corresponds to a dict with the following
    key-value pairs:

    parameter -- the parameter of the rule
    parameter_name -- a generated name for the parameter
    body -- a dict where each key is a variable and its value is 1 if the
            variable is non-negated or 0 if it is negated
    clause_string -- the original string for that clause disconsidering the
                     parameters
    """
    model = {}
    for i, line in enumerate(structure.split('\n')):
        # comment syntax
        if '%' in line:
            continue
        # remove whitespace and end of line
        line = line.replace(' ', '').replace('.\n', '').replace('\n', '')
        # skip empty lines
        if line == '':
            continue
        # if line does not have a probability associated with it pass
        if not '::' in line:
            continue
        # parse line
        parameter, clause = line.split('::', 1)
        if ':-' in clause:
            head, body = clause.split(':-')
            if not relational_data:
                body = body.split(',')
            else:
                body = body.split('),')
                body = [body_var + ')' for body_var in body]
                body[-1] = body[-1][:-1]
            body_dict = {}
            for body_var in body:
                if '\+' in body_var:
                    body_var = body_var.replace('\+', '')
                    body_dict[body_var] = 0
                else:
                    body_dict[body_var] = 1
        else:
            head = clause
            body_dict = {}
        # update rules
        if not head in model:
            model[head] = {'rules': [],
                           'parents': set()}
            param_index = 0
        # generate parameter name
        predicate, _ = parse_relational_var(head)
        param_name = 'theta_' + predicate + '_' + str(param_index)
        param_index += 1
        model[head]['rules'].append({'parameter': float(parameter),
                                     'parameter_name': param_name,
                                     'body': body_dict,
                                     'clause_string': clause,
                                     'line': i})
        # update parents
        for parent in body_dict:
            model[head]['parents'].add(parent)
    # dumb var for probabilistic observation
    for head in model:
        dumb_var = 'y'
        while dumb_var in model.keys():
            dumb_var += dumb_var
        predicate, _ = parse_relational_var(head)
        prob_dumb_var = dumb_var +'_'+ predicate
        model[head]['prob_dumb'] = {
            'var': prob_dumb_var,
            'weight_0': 'theta_' + prob_dumb_var +'_0',
            'weight_1': 'theta_' + prob_dumb_var +'_1'}
    return model



def run():
    logging.info("STARTING AUTOMATED LEARNING FOR PROBLOG")
    structure_filepath, dataset_filepaths, relational_data = find_dataset_and_structure_files()
    results = {'filename': [],
               'time': [],
               'log-likelihood': []}
    # results_filename = __file__ +'.results.'
    # if relational_data:
    #     results_filename += 'relational.'
    # else:
    #     results_filename += 'propositional.'
    # results_filename += str(int(time.time())) + '.csv'
    # experiment_dir_name = experiment_dir.split('/')[-2]
    experiment_dir_name = os.path.basename(experiment_dir)
    results_filepath = './results/{}/problog/problog__{}__{}.csv'.format(
        experiment_dir_name, experiment_dir_name, int(time.time()))


    problog_structure = read_structure_from_file(structure_filepath)
    min_improv = 0.001

    for problog_dataset_filepath in dataset_filepaths:
        dataset_filename = problog_dataset_filepath.split('/')[-1]
        logging.info("Learning for dataset '{}'...".format(dataset_filename))

        # begin countin time
        start_time = time.time()

        if relational_data:
            # RELATIONAL CASE
            ll, weights, atoms, iteration, lfi_problem = lfi.run_lfi(
                    PrologString(problog_structure),
                    lfi.read_examples(problog_dataset_filepath),
                    min_improv=min_improv)
        else:
            # PROPOSITIONAL CASE
            problog_structure = parse_structure_to_problog(problog_structure)
            dataset = pd.read_csv(problog_dataset_filepath)
            dataset = parse_dataset_to_problog(dataset)

            ll, weights, atoms, iteration, lfi_problem = lfi.run_lfi(
                    PrologString(problog_structure),
                    dataset,
                    min_improv=min_improv)

        string_model = lfi_problem.get_model()
        model = read_structure_not_file(string_model)
        # learner = Learner(structure_filepath, relational_data=relational_data)
        # ll, model = learner.learn_parameters(dataset)

        # end counting time
        end_time = time.time()
        learning_time = end_time - start_time

        logging.info("Learned:\nDATASET '{}'\nTime: {}\nLog-Likelihood: {}\nModel: {}".format(dataset_filename, learning_time, ll, string_model))
        results = store_results(results, dataset_filename, learning_time, ll, model)
        # logging.debug("Result added: {}".format(results))

        save_results_to_file(results, results_filepath)
    return results
run()

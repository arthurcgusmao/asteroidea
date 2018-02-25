"""This file builds the datasets from a structure file (that should be named
`original_structure.pl`) and a queries file (that should be named `queries.pl` and contain a query
for each variable in the problem). Default datasets and initial structure for learning will be
created, for both ProbLog and Asteroidea.
"""
import os
import sys
import pandas as pd
from random import random
from problog.tasks import sample
from problog.program import PrologString


def read_model(experiment_path):
    original_structure_filename = 'original_structure.pl'
    queries_filename = 'queries.pl'

    original_structure_path = "{}/{}".format(experiment_path, original_structure_filename)
    queries_path = "{}/{}".format(experiment_path, queries_filename)

    f = open(original_structure_path, 'r')
    original_structure_text = f.read()
    f.close()
    f = open(queries_path, 'r')
    queries_text = f.read()
    f.close()

    modeltext = original_structure_text + queries_text
    model = PrologString(modeltext)
    return original_structure_text, model


def create_structure(experiment_path, original_structure_text):
    """This function creates a `structure.pl` file into the experiment directory based on the original structure of the problem. What it does is to substitute the original parameters for 0.5, so that we have a default learning structure for both asteroidea and problog.
    """
    lines = []
    for line in original_structure_text.split('\n'):
        if '::' in line:
            idx = line.index('::')
            lines.append(''.join(['0.5', line[idx:], '\n']))
        else:
            lines.append(line)
    learn_structure_text = ''.join(lines)

    learn_structure_filepath = "{}/{}".format(experiment_path, 'structure.pl')
    f = open(learn_structure_filepath, 'w+')
    f.write(learn_structure_text)
    f.close()
    return


def generate_dataset(model, size, missing_rate):
    datapoints = []
    result = sample.sample(model, n=size, format='dict')
    for s in result:
        datapoints.append(s)
    output = pd.DataFrame(datapoints)
    output = output.applymap(lambda x: 1 if x==True else 0)
    # applying missing rate
    if missing_rate != 0:
        # transform values greater than 1 to percentage
        if missing_rate >= 1:
            missing_rate = float(missing_rate) / 100.
        # missing values are represented as NaN
        output = output.applymap(lambda x: x if random() > missing_rate else float('NaN'))
    return output


def create_propositional_datasets(experiment_path, size_range, missing_range):
    # remove last "/" in experiment path if present
    if experiment_path[-1] == '/':
        experiment_path = experiment_path[:-1]
    # read model
    original_structure_text, model = read_model(experiment_path)
    create_structure(experiment_path, original_structure_text)

    for n in size_range:
        for m in missing_range:
            filename = '{}/dataset_{:03d}size_{:03d}missing.csv'.format(experiment_path, n, m)
            df = generate_dataset(model, n, m)
            df.to_csv(filename, index=False)

# size_range = [5, 10, 15, 20, 25, 30]
# missing_range = [0, 1, 2, 5, 10, 20, 30]

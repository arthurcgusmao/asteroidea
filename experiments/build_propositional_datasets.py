"""This file builds the datasets from a structure file (that should be named
`original_structure.pl`) and a queries file (that should be named `queries.pl` and contain a query
for each variable in the problem). Default datasets and initial structure for learning will be
created, for both ProbLog and Asteroidea. Call this script from the command line as pass as
argument the path of the experiment folder where the mentioned files are present.
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
    asteroidea_structure_filename = 'structure.pl'

    # remove last "/" in experiment path if present
    if experiment_path[-1] == '/':
        experiment_path = experiment_path[:-1]
    original_structure_path = "{}/{}".format(experiment_path, original_structure_filename)
    queries_path = "{}/{}".format(experiment_path, queries_filename)
    asteroidea_structure_path = "{}/{}".format(experiment_path, asteroidea_structure_filename)

    f = open(original_structure_path, 'r')
    original_structure_text = f.read()
    f.close()
    f = open(queries_path, 'r')
    queries_text = f.read()
    f.close()

    modeltext = original_structure_text + queries_text
    model = PrologString(modeltext)
    return model


def generate_prob_line(model, randm=False):
    result = sample.sample(model, n=5, format='dict')
    count = {}
    total = 0
    for s in result:
        total += 1
        for var in s:
            if not var in count.keys():
                count[var] = 0
            if s[var] == True:
                count[var] += 1
    for var in count:
        count[var] /= float(total)
        count[var] += (random() - 0.5) / 5
        count[var] = max(count[var], 0)
        count[var] = min(count[var], 1)
    return count


def generate_missing_line(model, missing_rate):
    result = sample.sample(model, n=1, format='dict')
    count = {}
    for s in result:
        for var in s:
            if not var in count.keys():
                count[var] = 0
            if s[var] == True:
                count[var] += 1
    for var in count:
        count[var] /= float(total)
        count[var] += (random() - 0.5) / 5
        count[var] = max(count[var], 0)
        count[var] = min(count[var], 1)
    return count


def generate_dataset(model, size, missing_rate=0.0):
    datapoints = []
    if missing_rate != 0:
        for i in range(size):
            datapoints.append(generate_prob_line(model, randm=randm))
        output = pd.DataFrame(datapoints)
    else:
        result = sample.sample(model, n=size, format='dict')
        for s in result:
            datapoints.append(s)
        output = pd.DataFrame(datapoints)
        output = output.applymap(lambda x: 1 if x==True else 0)
    return output


def generate_and_write_datasets(experiment_path, start_size, end_size, delta=100):
    model = read_model(experiment_path)
    for i in range(int(start_size/delta), int(end_size/delta + 1)):
        size=delta*i
        dataset = generate_dataset(model, size)
        file_name='dataset_{}.csv'.format(str(size))
        dataset.to_csv(file_name, index=False)

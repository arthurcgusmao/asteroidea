"""This file builds the datasets from a structure file, that should be named
`original_structure.pl`. Default datasets and initial structure for learning will be created, for
both ProbLog and Asteroidea.
"""
import os
import sys
import pandas as pd
from random import random
from problog.tasks import sample
from problog.program import PrologString





def create_problog_structure_and_dataset(experiment_path,
                                         asteroidea_structure_filename,
                                         asteroidea_structure_path,
                                         dataset_filename,
                                         problog_prefix='problog'):
    dataset_filepath = "{}/{}".format(experiment_path, dataset_filename)

    problog_dataset_filename = "{}_{}".format(problog_prefix, dataset_filename)
    problog_structure_filename = "{}_{}".format(problog_prefix, asteroidea_structure_filename)

    problog_dataset_filepath = "{}/{}".format(experiment_path, problog_dataset_filename)
    problog_structure_filepath = "{}/{}".format(experiment_path, problog_structure_filename)

    problog_structure = ''
    problog_dataset = ''

    # read structure file
    f = open(asteroidea_structure_path, 'r')
    structure = f.read()
    f.close()

    # read dataset file
    f = open(dataset_filepath, 'r')
    dataset = f.read()
    f.close()

    # parse structure to problog
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

    # retrieve facts from dataset (in the alarm case, it was person(a) and same_person(a,a))
    for line in dataset.split('\n'):
        # comment syntax
        if '%' in line:
            continue
        # remove whitespace
        line = line.replace(' ', '')
        # skip empty lines
        if line == '':
            continue
        # check if line contains something that is not an evidence
        if not 'evidence' in line:
            problog_structure += line + '\n'
        else:
            problog_dataset += line + '\n'

    # save problog structure to file
    f = open(problog_structure_filepath, 'w+')
    for line in problog_structure.split('\n'):
        f.write(line + '\n')
    f.close()

    # save problog dataset to file
    f = open(problog_dataset_filepath, 'w+')
    for line in problog_dataset.split('\n'):
        f.write(line + '\n')
    f.close()

    return


def create_dataset_file(experiment_path, original_structure_text, number_of_houses, missing_rate):
    """Some hardcoded things here. This function supposes that the original_structure is from the
    fire/burglary/alarm example.
    """
    modeltext = original_structure_text
    output = ''
    constant = 'p'
    type_ = 'person'

    for i in range(1, (number_of_houses+1)):
        output += '{}({}_{}).\n'.format(type_, constant, i)
        output += 'same_{}({}_{},{}_{}).\n'.format(type_, constant, i, constant, i)
        modeltext += '{}({}_{}).\n'.format(type_, constant, i)
        modeltext += 'same_{}({}_{},{}_{}).\n'.format(type_, constant, i, constant, i)

        modeltext += 'query(fire({}_{})).\n'.format(constant, i)
        modeltext += 'query(alarm({}_{})).\n'.format(constant, i)
        modeltext += 'query(burglary({}_{})).\n'.format(constant, i)

        for j in range(1, (number_of_houses+1)):
            modeltext += 'query(calls({}_{},{}_{})).\n'.format(constant, i, constant, j)
            modeltext += 'query(cares({}_{},{}_{})).\n'.format(constant, i, constant, j)

    output += missing_sample(modeltext, missing_rate)
    # saving output to file
    filename = "dataset_{:03d}size_{:03d}missing.pl".format(number_of_houses, missing_rate)
    filepath = "{}/{}".format(experiment_path, filename)
    f = open(filepath, 'w+')
    f.write(output)
    f.close()
    return


def missing_sample(modeltext, missing_rate):
    """Returns a relational observation sampled from modeltext (which should be a structure in
    ProbLog syntax).
    """
    model = PrologString(modeltext)
    result = sample.sample(model, n=1, format='dict')
    output = ""
    for s in result:
        for q in s:
            value = s[q]
            if random() > missing_rate/100:
                if value == True:
                    output += "evidence({},true).\n".format(str(q))
                else:
                    output += "evidence({},false).\n".format(str(q))
            else:
                output += "evidence({},none).\n".format(str(q))
    return output


def create_asteroidea_structure(original_structure_text, asteroidea_structure_path):
    lines = []
    for line in original_structure_text.split('\n'):
        if '::' in line:
            idx = line.index('::')
            lines.append(''.join(['0.5', line[idx:], '\n']))
        else:
            lines.append(line)
    structure_text = ''.join(lines)

    f = open(asteroidea_structure_path, 'w+')
    f.write(structure_text)
    f.close()
    return


def create_relational_datasets(experiment_path, size_range, missing_range):
    original_structure_filename = 'original_structure.pl'
    asteroidea_structure_filename = 'structure.pl'

    # remove last "/" in experiment path if present
    if experiment_path[-1] == '/':
        experiment_path = experiment_path[:-1]
    original_structure_path = "{}/{}".format(experiment_path, original_structure_filename)
    asteroidea_structure_path = "{}/{}".format(experiment_path, asteroidea_structure_filename)

    f = open(original_structure_path, 'r')
    original_structure_text = f.read()
    f.close()

    # create asteroidea structure
    create_asteroidea_structure(original_structure_text, asteroidea_structure_path)

    # create all asteroidea datasets
    for h in size_range:
        for m in missing_range:
            create_dataset_file(experiment_path, original_structure_text, h, m)

    # create all problog datasets and structure
    files = os.listdir(experiment_path)
    for file_name in files:
        if 'dataset' in file_name:
            if not 'problog' in file_name:
                create_problog_structure_and_dataset(experiment_path, asteroidea_structure_filename, asteroidea_structure_path, file_name)


# size_range = [5, 10, 15, 20, 25, 30]
# missing_range = [0, 1, 2, 5, 10, 20, 30]

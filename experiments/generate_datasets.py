"""This file generates the 'alarm' and 'ship_energy_plant' datasets.
"""

# other imports
import os
import tarfile
from tools.build_propositional_datasets import create_propositional_datasets
from tools.build_relational_datasets import create_relational_datasets


# We are going to create 8 instances of each dataset. In order to do that, we'll copy the original folders to 8 different instances and run the function for each one. This ensures that the datasets will be different, due to the pseudo randomicity of the sample function.

def create_all_dataset_instances(dataset_name, relational, size_range, missing_range):
    print('Creating {} datasets...'.format(dataset_name))
    dataset_paths = []
    for i in range(1, 9):
        path = "./datasets/{}_{:02d}".format(dataset_name, i)
        if os.path.exists(path):
            raise RuntimeError('Path already exists: {}'.format(path))
        dataset_paths.append(path)

    tarf = tarfile.open('./datasets/{}.tar.gz'.format(dataset_name))
    for path in dataset_paths:
        tarf.extractall(path=path)
    tarf.close()

    for path in dataset_paths:
        print(os.path.abspath(path))
        if not relational:
            create_propositional_datasets(path, size_range, missing_range)
        else:
            create_relational_datasets(path, size_range, missing_range)
    print('Done.')


# PROPOSITIONAL DATASETS

create_all_dataset_instances(
    dataset_name='ship_energy_plant',
    relational=False,
    size_range=[2,4],
    missing_range=[0, 10])


# RELATIONAL DATASETS

create_all_dataset_instances(
    dataset_name='alarm',
    relational=True,
    size_range=[1,2],
    missing_range=[0, 10])

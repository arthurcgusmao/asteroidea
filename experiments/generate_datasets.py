"""This file generates the 'alarm' and 'ship_energy_plant' datasets.
"""

# other imports
import os
import tarfile
from tools.build_propositional_datasets import create_propositional_datasets
# from build_relational_datasets import create_relational_datasets


# PROPOSITIONAL DATASETS

# We are going to create 8 instances of each dataset. In order to do that, we'll copy the original folders to 8 different instances and run the function for each one. This ensures that the datasets will be different, due to the pseudo randomicity of the sample function.

dataset_paths = []
for i in range(1, 9):
    path = "./datasets/ship_energy_plant_{:02d}".format(i)
    if os.path.exists(path):
        raise RuntimeError('Path already exists: {}'.format(path))
    dataset_paths.append(path)

tarf = tarfile.open('./datasets/ship_energy_plant.tar.gz')
for path in dataset_paths:
    tarf.extractall(path=path)
tarf.close()

size_range = [2,4]
missing_range = [0, 10]

for path in dataset_paths:
    create_propositional_datasets(path, size_range, missing_range)

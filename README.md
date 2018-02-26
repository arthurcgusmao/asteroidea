
# Asteroidea

Asteroidea is a parameter learning algorithm for probabilistic logic programs.


## Setting up a working environment

At the root of this project there is an `environment.yml` file that contains information about the [conda](http://conda.io) environment that was used to run this program. Use the command below to create this environment using conda:

```bash
conda env create -f environment.yml
```


## Running experiments


### 1) Generate new datasets

In order to reproduce the experiments you must first enter into the `experiments` directory and run `generate_datasets.py`:

```bash
git clone https://github.com/arthurcgusmao/asteroidea/tree/master
cd asteroidea/experiments
python generate_datasets.py
```


### 2) Learn parameters

Next, you can perform the experiments by running any of the `run_*` shell script files. For instance:

```bash
./run_asteroidea_alarm
```


### Reproducing the original experiments

Notice: when following the steps outlined above, please keep in mind that the results may differ from the ones published on the paper because by executing the steps above we are **generating new datasets**. We have, however, uploaded the files that were used in the original experiment into `experiments/datasets/original_datasets.tar.gz`. If you want to run the tests with them, simply extract the compressed tarball file:

```bash
cd experiments/datasets
tar -zxvf original_datasets.tar.gz
```

Next, jump right into step 2 described above.

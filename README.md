
# Asteroidea

Asteroidea is a parameter learning algorithm for probabilistic logic programs licensed under the [GNU General Public License v3.0](LICENSE).


## Setting up a working environment

At the root of this project there is an `environment.yml` file that contains information about the [conda](http://conda.io) environment that was used to run this program. Use the command below to create this environment using conda:

```bash
conda env create -f environment.yml
```

Activate the environment:
```bash
source activate problog # (the name of the environment is problog)
```
Note: we named the environment "ProbLog". At this stage, it has nothing to do with the ProbLog package itself, which we'll install next.

Finally, install ProbLog in this environment (which is not directly handled by conda):

```bash
pip install problog
```

## Running experiments


### 1) Generate new datasets

Enter into the `experiments` directory and run `generate_datasets.py` to generate new datasets (i.e., sample from the models we have):

```bash
git clone https://github.com/arthurcgusmao/asteroidea/tree/master
cd asteroidea/experiments
python generate_datasets.py
```


### 2) Learn parameters

Perform the experiments by running any of the `run_*` shell script files, which will automatically run the respective algorithm for all instances, sizes and missing rates for the respective dataset. For instance, to run Asteroidea on all the alarm datasets:

```bash
./run_asteroidea_alarm
```


### Reproducing the original experiments

Notice: when following the steps outlined above, please keep in mind that the results may differ a bit from the ones we published paper because by executing the steps above we are **generating new datasets**. However, we have uploaded the datasets that were used in the original experiments: `experiments/datasets/experiments_ijar_alarm.tar.gz`. If you want to run the tests with them, simply extract the compressed tarball file:

```bash
cd experiments/datasets
tar -zxvf experiments_ijar_alarm.tar.gz
```

Next, jump right into step 2 described above. Now the results likely will be more consistent. However, as there are still fonts of randomicity, the results should still not be exactly the same.


## Using Asteroidea (quick intro)

This section is meant to be a quick intro so you can use the algorithm in your own dataset.

Asteroidea's module is located in the `asteroidea` folder at the root of this project. You can import the module that contains the class `Learner`:

```python
from asteroidea.missing_learner import Learner
```

Next you instantiate the class and pass the arguments:

```python
learner = Learner(structure_filepath=structure_filepath,
                  dataset_filepath=dataset_filepath,
                  relational_data=relational_data)
```

Arguments description:

- `structure_filepath`: (string) the path to the file containing the structure of your probabilistic logic program, written in ProbLog syntax, with the initial parameters (the parameters that the algorithm will start from). See the end of this section for examples of structure file.
- `dataset_filepath`: (string) the path to the file containing the observations for your problem. If it is relational, it should be written using ProbLog syntax. If it is propositional, it should be a CSV file. See the end of this section for examples of dataset file.
- `relational_data`: (bool) should be `true` if the data you are dealing with is relational and `false` if propositional.

After you have instantiated the class, you are able to make it learn the parameters with the `learn_parameters` method. A value for the stopping criteria (epsilon) can also be passed:

```python
learning_info = learner.learn_parameters(epsilon=0.001)
```

After the learning function ends it returns a dictionary that contains information about which rules were not exact solutions (`learning_info['no_exact_solution']`) (i.e., which rules could not be calculated in closed-form in each iteration), a time log (`learning_info['time_log']`) that categorizes the time consumption into different categories, and a pandas dataframe (`learning_info['df']`) that contains all iterations, the parameter values reach at each iteration, and the respective log-likelihood.

Please check the [`missing_learner.py`](asteroidea/missing_learner.py), [`asteroidea_automated_learning.py`](experiments/tools/asteroidea_automated_learning.py), and [`run_asteroidea_alarm.sh`](experiments/run_asteroidea_alarm.sh) files contained in this repository to see the implementation and examples on how to automate the learning process.


### Examples of structure file

Relational:
```
0.5::fire(X):-person(X).
0.5::burglary(X):-person(X).
0.5::alarm(X):-fire(X).
0.5::alarm(X):-burglary(X).
0.5::cares(X,Y):-person(X),person(Y).
0.5::calls(X,Y):-cares(X,Y),alarm(Y),\+same_person(X,Y).
```

Propositional:
```
0.5::high_load:-ll2,ll3,pl2,pl3.
0.5::low_load:-high_load.
0.5::low_load:-ll1,pl1.
0.5::low_load:-ll2,pl2.
0.5::low_load:-ll3,pl3.
0.5::high_supply:-a2,a3.
0.5::high_supply:-a2,a4.
0.5::high_supply:-a3,a4.
0.5::low_supply:-a1.
0.5::low_supply:-high_supply.
0.5::failure:-high_load,\+high_supply.
0.5::failure:-low_load,\+low_supply.
0.5::emergency:-\+a3,\+a4.
0.5::ll1:-emergency.
0.5::pl1:-emergency.
0.5::a2:-a3.
0.5::a2:-a4.
0.5::a1.
0.5::a3.
0.5::a4.
0.5::ll2.
0.5::pl2.
0.5::ll3.
0.5::pl3.
```

### Examples of dataset file

Relational:
```
person(p_1).
same_person(p_1,p_1).
person(p_2).
same_person(p_2,p_2).
person(p_3).
same_person(p_3,p_3).
person(p_4).
same_person(p_4,p_4).
person(p_5).
same_person(p_5,p_5).
evidence(fire(p_1),false).
evidence(alarm(p_1),false).
evidence(burglary(p_1),false).
evidence(calls(p_1,p_1),false).
evidence(cares(p_1,p_1),true).
evidence(calls(p_1,p_2),false).
evidence(cares(p_1,p_2),true).
evidence(calls(p_1,p_3),false).
evidence(cares(p_1,p_3),false).
evidence(calls(p_1,p_4),true).
evidence(cares(p_1,p_4),true).
evidence(calls(p_1,p_5),false).
evidence(cares(p_1,p_5),false).
evidence(fire(p_2),false).
evidence(alarm(p_2),false).
evidence(burglary(p_2),false).
evidence(calls(p_2,p_1),false).
evidence(cares(p_2,p_1),true).
evidence(calls(p_2,p_2),false).
evidence(cares(p_2,p_2),true).
evidence(calls(p_2,p_3),false).
evidence(cares(p_2,p_3),true).
evidence(calls(p_2,p_4),true).
evidence(cares(p_2,p_4),true).
evidence(calls(p_2,p_5),false).
evidence(cares(p_2,p_5),true).
evidence(fire(p_3),false).
evidence(alarm(p_3),false).
evidence(burglary(p_3),false).
evidence(calls(p_3,p_1),false).
evidence(cares(p_3,p_1),false).
evidence(calls(p_3,p_2),false).
evidence(cares(p_3,p_2),true).
evidence(calls(p_3,p_3),false).
evidence(cares(p_3,p_3),true).
evidence(calls(p_3,p_4),true).
evidence(cares(p_3,p_4),true).
evidence(calls(p_3,p_5),false).
evidence(cares(p_3,p_5),true).
evidence(fire(p_4),false).
evidence(alarm(p_4),true).
evidence(burglary(p_4),true).
evidence(calls(p_4,p_1),false).
evidence(cares(p_4,p_1),true).
evidence(calls(p_4,p_2),false).
evidence(cares(p_4,p_2),true).
evidence(calls(p_4,p_3),false).
evidence(cares(p_4,p_3),true).
evidence(calls(p_4,p_4),false).
evidence(cares(p_4,p_4),false).
evidence(calls(p_4,p_5),false).
evidence(cares(p_4,p_5),true).
evidence(fire(p_5),true).
evidence(alarm(p_5),false).
evidence(burglary(p_5),false).
evidence(calls(p_5,p_1),false).
evidence(cares(p_5,p_1),false).
evidence(calls(p_5,p_2),false).
evidence(cares(p_5,p_2),true).
evidence(calls(p_5,p_3),false).
evidence(cares(p_5,p_3),true).
evidence(calls(p_5,p_4),false).
evidence(cares(p_5,p_4),false).
evidence(calls(p_5,p_5),false).
evidence(cares(p_5,p_5),true).
```
To make a datapoint missing in your relational dataset change `true` or `false` to `none`. For instance:
```
evidence(calls(p_4,p_1),none).
```


Propositional:
```
low_load,low_supply,high_load,high_supply,emergency,failure,a1,a2,a3,a4,ll1,ll2,ll3,pl1,pl2,pl3
1,1,1,1,0,0,0,1,0,1,0,1,1,0,1,1
1,1,0,1,0,0,1,1,1,1,0,0,1,0,1,1
1,1,0,1,0,0,1,1,1,1,0,1,1,0,1,1
1,1,1,1,0,0,0,1,1,1,0,1,1,0,1,1
1,1,0,1,0,0,1,1,1,1,0,1,1,0,1,1
1,1,1,1,0,0,1,1,1,1,0,1,1,0,1,1
1,1,1,1,0,0,1,1,1,1,0,1,1,0,1,1
1,1,0,1,0,0,1,1,1,1,0,0,1,0,1,1
1,1,1,1,0,0,1,1,1,1,0,1,1,0,1,1
0,1,0,1,0,0,1,1,1,1,0,1,1,0,0,0
```
To make a datapoint missing in your propositional dataset simply erase the value in the column. For instance:
```
low_load,low_supply,high_load
1,,0
1,1,
0,1,0
```

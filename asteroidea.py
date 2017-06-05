import math
import numpy as np
import pandas as pd
import itertools
from functools import partial
from scipy.optimize import basinhopping, minimize
from problog.program import PrologString
from problog import get_evaluatable

class plp(object):

    def __init__(self):
        print('hello, I\'m a plp bro.')

        
    def read_structure(self, filepath):
        """Reads a file containing a structure for the PLP program. The
        structure should be written accordingly to ProbLog's syntax.

        The structure is stored in a dict called self.model where each key is
        the head of the rules and the value is a rules list (for the given
        head).

        Each rule in the rules list corresponds to a dict with the following
        key-value pairs:
        parameter -- the parameter of the rule;
        body -- a dict where each key is a variable and its value is True if
                the variable is non-negated or False if it is negated.

        An example: for the set of rules

            0.6::X1.
            0.7::X2.
            0.8::X3:-X1,\+X2.
            0.9::X3:-\+X1.
        
        we would have the following structure:
        
        self.model = {
            'X1': [{'parameter': 0.6, 'body': {}}],
            'X2': [{'parameter': 0.7, 'body': {}}],
            'X3': [{'parameter': 0.8, 'body': {'X1': 1, 'X2': 0}},
                   {'parameter': 0.9, 'body': {'X1': 0}}]}


        Also, another dict called self.parents is created. It stores for each
        head a set of its parents. In our example, we would have:
        
        self.parents = {
            'X1': set(),
            'X2': set(),
            'X3': {'X1', 'X2'}}
        """
        self.model = {}
        self.parents = {}
        temp_file = open(filepath, 'r+')
        for i, line in enumerate(temp_file):
            # remove end of line and whitespace
            line = line.replace('.\n', '').replace(' ', '')
            # parse line
            parameter, head = line.split('::', 1)
            if ':-' in head:
                head, body = head.split(':-')
                body = body.split(',')
                body_dict = {}
                for body_var in body:
                    if '\+' in body_var:
                        body_var = body_var.replace('\+', '')
                        body_dict[body_var] = 0
                    else:
                        body_dict[body_var] = 1
            else:
                body_dict = {}
            # update self.model
            if not head in self.model:
                self.model[head] = []
            self.model[head].append({'parameter': float(parameter),
                                     'body': body_dict})
            # update self.parents
            if not head in self.parents:
                self.parents[head] = set()
            for parent in body_dict:
                self.parents[head].add(parent)
        temp_file.close()


    def learn_parameters(self, dataset, epsilon=0.01):
        """Find the (exact or approximated) optimal parameters for the dataset.
        Before running this function, make sure you have read a structure file
        compatible with the variables present in the dataset.

        Keyword arguments:
        dataset -- Pandas DataFrame containing the observations
        epsilon -- stopping criteria

        Missing values in the dataset should be represented as NaN.
        """
        # ensure dataset is a pandas dataframe
        if not isinstance(dataset, pd.DataFrame):
            raise TypeError('dataset should be a Pandas DataFrame')
        # ensure all variables in the dataset are head of a rule
        for column in dataset.columns:
            if not column in self.model:
                # what to do if column in dataset is not head of any rule?
                raise Exception('Column %s in dataset is not head of any rule.'
                                % column)
        # pre-build configurations tables
        self._build_configs_tables()

        # begin EM cycle
        old_ll = 0
        while True:
            for head in self.model:
                # in this inner loop, both E and M steps are performed for each
                # variable, instead of for all variables at once.
                rules = self.model[head]
                configs_table = self.configs_tables[head]
                
                ### E step ###
                configs_table.loc[:, 'count'] = 0 # zero all counts
                for i, row in dataset.iterrows():
                    for c, config in configs_table.iterrows():
                        prob = self.inference(query=config, evidence=row)
                        configs_table.loc[c, 'count'] += prob # update count

                ### M step ###
                # @TODO: check if there is an exact solution for the parameters
                if(False):
                    # there is an exact solution, calculate it
                    optimal_params = self._exact_ll_maximization() # NOT IMPLEMENTED YET
                else:
                    # there is no exact solution, run optimization method
                    initial_guess = []
                    for i, rule in enumerate(rules):
                        initial_guess[i] = rule['parameter']
                    res = minimize(
                            partial(self._family_log_likelihood, x),
                            initial_guess,
                            method = 'L-BFGS-B',
                            bounds = bnds,
                            options = {'disp': True ,'eps' : 1e-7})
                    optimal_params = res.x
                
                # update parameters of the model
                for i, rule in enumerate(rules):
                    rules[i]['parameter'] = optimal_params.pop(i)
                self.model[head] = rules
            
            # EM cycle stopping criteria
            if (abs(ll - old_ll) / ll) < epsilon:
                break
            old_ll = ll
                
        
    def _family_log_likelihood(self, head, parameters):
        return output

    
    def _build_configs_tables(self):
        """Builds self.configs_tables dict, which for each head is associated a
        Pandas DataFrame that represents the configurations table for that head.

        Columns of each dataframe are:
        head variable
        body variable(s)
        count -- expected number of times the configuration is observed
        likelihood -- the likelihood of the head given parents in that config
        """
        self.configs_tables = {}
        for head in self.model:
            rules = self.model[head]
            columns = [head]
            columns.extend(self.parents[head])
            configs_values = list(
                map(list, itertools.product([0, 1], repeat=len(columns))))
            df = pd.DataFrame(configs_values, columns=columns)
            df.loc[:,'count'] = pd.Series(np.nan, index=df.index)
            df.loc[:,'likelihood'] = pd.Series(np.nan, index=df.index)
            self.configs_tables[head] = df


    def inference(self, query, evidence=pd.Series()):
        """Computes inference for a set of query variables, given the model and
        evidences.

        Keyword arguments:
        query -- a Panda Series containing queried values
        evidence -- a Panda Series containing observed values

        For both query and evidence arguments the indexes of the Pandas Series
        should be the variable names. Values containing NaN will be interpreted
        as missing and be disconsidered.
        """
        # generate model string accordingly to ProbLog's syntax
        # model =
        
        # @TODO: remove missing values in row
        return prob

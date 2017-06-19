"""Learns the optimal parameters for a complete dataset."""

from asteroidea import parser
from asteroidea import calculations

import time
import math
import numpy as np
import pandas as pd
from scipy.optimize import basinhopping, minimize

    
class Learner(object):

    def __init__(self, structure_filepath):
        self.model = parser.read_structure(structure_filepath)
        self.configs_tables = parser.build_configs_tables(self.model)

            
    def learn_parameters(self, dataset):
        """Find the (exact or approximated) optimal parameters for the dataset.
        Before running this function, make sure you have read a structure file
        compatible with the variables present in the dataset.
        Keyword arguments:
        dataset -- Pandas DataFrame containing the observations
        epsilon -- stopping criteria
        Missing values in the dataset should be represented as NaN.
        """
        model = self.model
        configs_tables = self.configs_tables
        # ensure dataset is a pandas dataframe
        if not isinstance(dataset, pd.DataFrame):
            raise TypeError('dataset should be a Pandas DataFrame')
        # ensure all variables in the dataset are head of a rule
        for column in dataset.columns:
            if not column in model:
                # what to do if column in dataset is not head of any rule?
                raise Exception('Column %s in dataset is not head of any rule.'
                                % column)

        # count the number of occurences of each configuration for each family
        for i, row in dataset.iterrows():
            for head in configs_tables:
                configs_table = configs_tables[head]
                config_vars = {head: row[head]}
                for parent in model[head]['parents']:
                    config_vars[parent] = row[parent]
                # get the index of the configs_table that corresponds to the
                # configuration of the current example
                df = configs_table
                for var in config_vars:
                    value = config_vars[var]
                    df = df.loc[df[var] == value]
                index = df.index.values[0]
                # updates the count in configs_table
                configs_table.loc[index, 'count'] += 1
        

        # maximize the likelihood by finding the optimal parameters
        ll = 0
        new_params = {}
        for head in model:
            rules = model[head]['rules']
            configs_table = configs_tables[head]
            optimal_params = calculations.exact_optimization(head, configs_table)
            if optimal_params == False:
                self.info['no_exact_solution'].add(head)
                # there is no exact solution, run optimization method
                initial_guess = []
                for rule in rules:
                    initial_guess.append(rule['parameter'])
                res = minimize(
                        calculations.head_log_likelihood,
                        initial_guess,
                        args = (head, model, configs_table, -1.0),
                        method = 'L-BFGS-B',
                        bounds = [(0.0, 1.0)]*len(initial_guess),
                        # bounds = [(0.001,0.999)]*len(initial_guess),
                        options = {'disp': True ,'eps' : 1e-7})
                optimal_params = res.x.tolist()

            # update log-likelihood
            ll += calculations.head_log_likelihood(optimal_params, head, model,
                                                   configs_table)
            # store new parameters
            new_params[head] = optimal_params
        
        return {'log-likelihood': ll,
                'optimal parameters': new_params}

"""Learns the optimal parameters for a complete dataset."""

from asteroidea import parser
from asteroidea import calculations

import time
import math
import numpy as np
import pandas as pd
from scipy.optimize import basinhopping, minimize

    
class Learner(object):

    def __init__(self, structure_filepath, relational_data=False):
        self.relational_data = relational_data
        self.model = parser.read_structure(structure_filepath, relational_data=relational_data)
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
        self.dataset = dataset
        if self.relational_data:
            self._update_count_relational()
        else:
            self._verify_propositional_dataset()
            self._update_count_propositional()
        return self._find_optimal_parameters()

    
    def _update_count_propositional(self):
        # count the number of occurences of each configuration for each family
        for i, row in self.dataset.iterrows():
            for head in self.configs_tables:
                configs_table = self.configs_tables[head]
                config_vars = {head: row[head]}
                for parent in self.model[head]['parents']:
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


    def _update_count_relational(self):
        observations, constants = parser.parse_relational_dataset(self.dataset)
        for head in self.configs_tables:
            configs_table = self.configs_tables[head]
            config_atoms = [head]
            for parent in self.model[head]['parents']:
                config_atoms.append(parent)

            substitutions = parser.generate_substitutions(config_atoms, constants)
            for substitution in substitutions:
                df = configs_table
                for atom in config_atoms:
                    substituted_atom = parser.apply_substitution(atom, substitution)
                    if substituted_atom in observations:
                        value = 1
                    else:
                        value = 0
                    df = df.loc[df[atom] == value]
                # we are left with only one row in the df, which is the right
                # row where we should increase the count
                index = df.index.values[0]
                configs_table.loc[index, 'count'] += 1


    def _find_optimal_parameters(self):
        # maximize the likelihood by finding the optimal parameters
        ll = 0
        new_params = {}
        for head in self.model:
            rules = self.model[head]['rules']
            configs_table = self.configs_tables[head]
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
                        args = (head, self.model, configs_table, -1.0),
                        method = 'L-BFGS-B',
                        bounds = [(0.0, 1.0)]*len(initial_guess),
                        # bounds = [(0.001,0.999)]*len(initial_guess),
                        options = {'disp': True ,'eps' : 1e-7})
                optimal_params = res.x.tolist()
            # update log-likelihood
            ll += calculations.head_log_likelihood(optimal_params, head,
                                                   self.model, configs_table)
            # store new parameters
            new_params[head] = optimal_params
        return {'log-likelihood': ll,
                'optimal parameters': new_params}


    def _verify_propositional_dataset(self):
        # ensure dataset is a pandas dataframe
        if not isinstance(self.dataset, pd.DataFrame):
            raise TypeError('dataset should be a Pandas DataFrame')
        # ensure all variables in the dataset are head of a rule
        for column in self.dataset.columns:
            if not column in self.model:
                # what to do if column in dataset is not head of any rule?
                raise Exception('Column %s in dataset is not head of any rule.'
                                % column)

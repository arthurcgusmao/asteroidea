"""Learns the optimal parameters for a complete dataset."""

from asteroidea import parser
from asteroidea import calculations

import time
import math
import logging
import numpy as np
import pandas as pd
from scipy.optimize import basinhopping, minimize


class Learner(object):

    def __init__(self, structure_filepath, relational_data=False):
        self.logger = logging.getLogger('asteroidea')
        self.info = {'no_exact_solution': set()}
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
        ll, optimal_params = self._find_optimal_parameters()
        self._update_model(optimal_params)
        return ll, parser.pretty_print_model(self.model)
        
    
    def _update_count_propositional(self):
        self.logger.info('Updating count for propositional dataset...')
        # get ordered configuration variables for each head
        config_vars_head = {}
        for head in self.configs_tables:
            number_of_vars = len(self.model[head]['parents']) + 1
            columns = self.configs_tables[head].columns.tolist()
            config_vars_head[head] = columns[0:number_of_vars]
        # count the number of occurences of each configuration for each family
        for i, row in self.dataset.iterrows():
            for head in self.configs_tables:
                config_vars = config_vars_head[head]
                number_of_vars = len(config_vars)
                # calculate the index of the configs_table that corresponds to
                # the configuration of the current example
                index = 0
                for v, var in enumerate(config_vars):
                    value = row[var]
                    index += 2**(number_of_vars - (v + 1)) * value
                # updates the count in configs_table
                self.configs_tables[head].loc[index, 'count'] += 1
        self.logger.info('Ok')


    def _update_count_relational(self):
        self.logger.info('Updating count for relational dataset...')
        observations, constants = parser.parse_relational_dataset(self.dataset)
        # get ordered configuration variables for each head
        config_atoms_head = {}
        for head in self.configs_tables:
            number_of_atoms = len(self.model[head]['parents']) + 1
            columns = self.configs_tables[head].columns.tolist()
            config_atoms = columns[0:number_of_atoms]
            
            groundings = parser.generate_groundings(config_atoms, constants)
            for grounding in groundings:
                index = 0
                for g, grounded_atom in enumerate(grounding):
                    if grounded_atom in observations:
                        value = observations[grounded_atom]
                    else:
                        value = 0 # closed-world assumption
                    index += 2**(number_of_atoms - (g + 1)) * value
                self.configs_tables[head].loc[index, 'count'] += 1
            self.logger.debug('Updated counts for head {} in configs_table.'.format(head))
        self.logger.info('Ok')


    def _find_optimal_parameters(self):
        # maximize the likelihood by finding the optimal parameters
        self.logger.info('Finding optimal parameters...')
        ll = 0
        new_params = {}
        for head in self.model:
            rules = self.model[head]['rules']
            configs_table = self.configs_tables[head]
            optimal_params = calculations.exact_optimization(head, configs_table)
            if optimal_params == False:
                # there is no exact solution, run optimization method
                self.logger.debug('Variable %s has no exact solution, approximate maximization will be run.' % head)
                self.info['no_exact_solution'].add(head)
                initial_guess = []
                for rule in rules:
                    initial_guess.append(rule['parameter'])
                res = minimize(
                        calculations.head_log_likelihood,
                        initial_guess,
                        args = (head, self.model, configs_table, -1.0),
                        method = 'L-BFGS-B',
                        bounds = [(0.001, 0.999)]*len(initial_guess),
                        # bounds = [(0.001,0.999)]*len(initial_guess),
                        options = {'disp': True ,'eps' : 1e-7})
                optimal_params = res.x.tolist()
                self.logger.debug('Approximate maximization run successfully for variable %s.' % head)
            # update log-likelihood
            ll += calculations.head_log_likelihood(optimal_params, head,
                                                   self.model, configs_table)
            # store new parameters
            new_params[head] = optimal_params
        self.logger.info('Optimal parameters found.')
        return ll, new_params


    def _verify_propositional_dataset(self):
        self.logger.info("Verifying propositional dataset against model...")
        # ensure dataset is a pandas dataframe
        if not isinstance(self.dataset, pd.DataFrame):
            raise TypeError('dataset should be a Pandas DataFrame')
        # ensure all variables in the dataset are head of a rule
        for column in self.dataset.columns:
            if not column in self.model:
                # what to do if column in dataset is not head of any rule?
                raise Exception('Column %s in dataset is not head of any rule.'
                                % column)
        self.logger.info("Ok")


    def _update_model(self, parameters):
        for head in self.model:
            rules = self.model[head]['rules']
            for i, rule in enumerate(rules):
                rules[i]['parameter'] = parameters[head][i]

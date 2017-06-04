import numpy as np
import pandas as pd
from functools import partial
from scipy.optimize import basinhopping, minimize

class plp(object):

    def __init__(self):
        print('hello, I\'m a plp bro.')

        
    def read_structure(self, filepath):
        """Reads a file containing a structure for the PLP program. The
        structure should be written according to ProbLog's syntax.
        """
        self.rules = [] # list of all rules in the model
        self.head_vars = [] # list of all variables that are head of a rule
        temp_file = open(filepath, 'r+')
        for i, line in enumerate(temp_file):
            # remove end of line and whitespace
            line = line.replace('.\n', '').replace(' ', '')

            parameter, head = line.split('::', 1)
            if ':-' in head:
                head, body = head.split(':-')
                body = body.split(',')
            else:
                body = None
            self.rules.append({'parameter': float(parameter),
                               'head': head,
                               'body': body})
            # build helper "heads" list
            if head not in self.head_vars:
                self.heads.append(head)
        temp_file.close()
        # self.rules_df = pd.DataFrame(rules)


    def learn_parameters(self, dataset, epsilon=0.01):
        # ensure dataset is a pandas dataframe
        if not isinstance(dataset, pd.DataFrame):
            raise TypeError('dataset should be a Pandas DataFrame')
        # ensure all variables in the dataset are head of a rule
        for column in dataset.columns:
            if not column in self.heads:
                # what to do if column in dataset is not head of any rule?
                raise Exception('Column %s in dataset is not head of any rule.'
                                % column)
        
        while True:
            # each iteration in this loop is an EM cycle

            # E step
            for x in self.head_vars:
                initial_guess = []
                for rule in self.rules:
                    if rule.head == x:
                        initial_guess.append(rule['parameter'])
                res = minimize(
                        partial(self._family_log_likelihood, x),
                        initial_guess,
                        method='L-BFGS-B',
                        bounds=bnds,
                        options={'disp': True ,'eps' : 1e-7})
                optimal_params = res.x
                for rule in self.rules:
                    if rule.head == x:
                        rule['parameter'] = optimal_params
            # M step


            # stopping criteria
            if (abs(ll - old_ll) / ll) < epsilon:
                break
                
        
    def _family_log_likelihood(self, head, parameters):
        return output

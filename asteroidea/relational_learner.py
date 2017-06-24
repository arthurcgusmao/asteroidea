from asteroidea import parser
from asteroidea import calculations
from asteroidea.inference import Inference

import time
import math
import numpy as np
import pandas as pd
from scipy.optimize import basinhopping, minimize

    
class Learner(object):

    def __init__(self, structure_filepath, dataset_filepath=None,
                 probabilistic_data=False, sampling=False, relational=False):
        self.info = {'df': None,
                     'time_log': {}}
        self._log_time('Building models', start=True)
        self.relational = relational
        if relational:
            if dataset_filepath == None:
                raise ValueError("""When learning a relational dataset you
                                 should pass `dataset_filepath` argument.""")
        
        self.model = parser.read_structure(structure_filepath)
        self.configs_tables = parser.build_configs_tables(self.model)
        self.problog_model_str = parser.build_problog_model_str(
                                    self.model, self.configs_tables,
                                    probabilistic_data=probabilistic_data,
                                    suppress_evidences=(sampling or relational),
                                    relational_dataset_path=dataset_filepath)
        if sampling:
            print('not implemented.')
        else:
            self.knowledge = Inference(self.problog_model_str,
                                       probabilistic_data=probabilistic_data)

            
    def learn_parameters(self, epsilon=0.01):
        """Find the (exact or approximated) optimal parameters for the dataset.
        Before running this function, make sure you have read a structure file
        compatible with the variables present in the dataset.
        Keyword arguments:
        dataset -- Pandas DataFrame containing the observations
        epsilon -- stopping criteria
        Missing values in the dataset should be represented as NaN.
        """
        self._log_time('Others')
        model = self.model
        configs_tables = self.configs_tables

        # begin EM cycle
        old_ll = None
        while True:
            ll = 0
            new_params = {}

            ### E step ###
            self._log_time('E step')
            for head in configs_tables:
                # reset configurations tables
                configs_tables[head].loc[:, 'count'] = 0             
            self.knowledge.update_weights(model)
            if not self.relational:
                for i, row in self.dataset.iterrows():
                    res = self.knowledge.eval(evidence=row)
                    for head in configs_tables:
                        configs_table = configs_tables[head]
                        for c, config in configs_table.iterrows():
                            update_in_count = res[config['dumb_var']]
                            configs_table.loc[c, 'count'] += update_in_count
            else:
                res = self.knowledge.eval()
                for head in configs_tables:
                    configs_table = configs_tables[head]
                    # for c, config in configs_table.iterrows():
                    #     update_in_count = res[config['dumb_var']]
                    #     configs_table.loc[c, 'count'] += update_in_count
                    for query in res:
                        prob = res[query]
                        dumb_var = query.split('__')[0]
                        # print(configs_tables[head])
                        configs_table.loc[configs_table['dumb_var'] == dumb_var, 'count'] += prob
                        # print(configs_tables[head])

            ### updating the initial ll value in learning info ###
            if old_ll == None:
                old_ll = calculations.log_likelihood(model, configs_tables)
                self._update_learning_info(old_ll, begin_em=True)

            ### M step ###
            self._log_time('M step')
            for head in model:
                rules = model[head]['rules']
                configs_table = configs_tables[head]
                optimal_params = calculations.exact_optimization(head, configs_table)
                if optimal_params == False:
                    print("Iteration", len(self._learning_data), "had no exact solution for head", head, "bro.")
                    # there is no exact solution, run optimization method
                    initial_guess = []
                    for rule in rules:
                        initial_guess.append(rule['parameter'])
                    res = minimize(
                            calculations.head_log_likelihood,
                            initial_guess,
                            args = (head, model, configs_table, -1.0),
                            method = 'L-BFGS-B',
                            bounds = [(0.00001, 0.99999)]*len(initial_guess),
                            # bounds = [(0.001,0.999)]*len(initial_guess),
                            options = {'disp': True ,'eps' : 1e-7})
                    optimal_params = res.x.tolist()
                        
                # update log-likelihood
                ll += calculations.head_log_likelihood(optimal_params, head, model, configs_table)
                # store new parameters
                new_params[head] = optimal_params

            self._log_time('Others')
            # update parameters of the model
            for head in model:
                rules = model[head]['rules']
                for i, rule in enumerate(rules):
                    rules[i]['parameter'] = new_params[head][i]
            # EM cycle stopping criteria
            if abs((ll - old_ll) / ll) < epsilon:
                learning_info = self._update_learning_info(ll, end_em=True)
                break
            old_ll = ll
            # update data about iterations
            self._update_learning_info(ll)
        return learning_info
    
        
    def _update_learning_info(self, log_likelihood,
                              begin_em=False, end_em=False):
        """This function stores information about parameters, log-likelihood
        and elapsed time during parameter learning. At the end of the cycle,
        it'll put this information into a DataFrame accessible by
        self.learn_parameters_info.
        Keyword arguments:
        log_likelihood -- the expected-value of the log-likelihood of the whole
                          model given the dataset
        begin_em -- indicates that the EM cycle is beginning
        end_em -- indicates that the EM cycle is ending
        """
        if begin_em:
            self._start_time = time.time()
            self._learning_data = []
        iter_data = []
        for head in self.model:
            for rule in self.model[head]['rules']:
                iter_data.append(rule['parameter'])
        elapsed_time = time.time() - self._start_time
        iter_data.extend([elapsed_time, log_likelihood])
        self._learning_data.append(iter_data)
        if end_em:
            columns = []
            for head in self.model:
                for rule in self.model[head]['rules']:
                    columns.append(rule['clause_string'])
            columns.extend(['Elapsed Time', 'Log-Likelihood'])
            self.info['df'] = pd.DataFrame(self._learning_data,
                                           columns=columns)

            
    def _log_time(self, activity, start=False):
        if not start:
            last_activity = self._log_time__last_activity
            last_time = self._log_time__last_time
            if not last_activity in self.info['time_log']:
                self.info['time_log'][last_activity] = 0
            self.info['time_log'][last_activity] += time.time() - last_time
        self._log_time__last_activity = activity
        self._log_time__last_time = time.time()

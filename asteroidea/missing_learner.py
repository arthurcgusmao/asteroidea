from asteroidea import parser
from asteroidea import calculations
from asteroidea.inference import Inference

import time
import math
import numpy as np
import pandas as pd
from scipy.optimize import basinhopping, minimize
from problog.errors import InconsistentEvidenceError

import logging


class Learner(object):

    def __init__(self, structure_filepath, dataset_filepath=None,
                 probabilistic_data=False, sampling=False, relational_data=False):
        self.info = {'df': None,
                     'time_log': {},
                     'no_exact_solution': set()}
        self._log_time('Building models', start=True)
        self.logger = logging.getLogger('asteroidea')
        self.dataset_filepath = dataset_filepath
        self.relational_data = relational_data

        if relational_data:
            if dataset_filepath == None:
                raise ValueError("""When learning a relational dataset you
                                 should pass `dataset_filepath` argument.""")

        self.model = parser.read_structure(structure_filepath, relational_data=relational_data)

        if not relational_data:
            self._read_propositional_dataset()

        self.configs_tables = parser.build_configs_tables(self.model)
        self.problog_model_str=parser.build_problog_model_str(
                                self.model, self.configs_tables,
                                probabilistic_data=probabilistic_data,
                                suppress_evidences=(sampling or relational_data),
                                relational_data=relational_data,
                                relational_dataset_path=dataset_filepath,
                                typed=True,
                                consistency_test=False)
        self.logger.debug("Problog model string builded:\n{}".format(self.problog_model_str))
        if sampling:
            raise NotImplemented
        else:
            self.knowledge = Inference(self.problog_model_str,
                                       probabilistic_data=probabilistic_data,
                                       relational_data=relational_data)
            self.knowledge.update_weights(self.model)


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

        ### updating the initial ll value in learning info ###
        # update weights
        self.knowledge.update_weights(model)
        # calculate observed log likelihood
        old_obs_ll = self._calculate_log_likelihood()
        self._update_learning_info(old_obs_ll, begin_em=True)

        # begin EM cycle
        step = 0
        while True:
            expec_ll = 0
            new_params = {}

            ### E step ###
            self.logger.info("Starting E step #{}".format(step))
            self._log_time('E step')
            # reset configurations tables
            for head in configs_tables:
                configs_table = configs_tables[head]
                for c, config in configs_table.iterrows():
                    configs_table.loc[c, 'count'] = configs_table.loc[c, 'real_count']
            # update (probabilistic, total) counts
            if not self.relational_data:
                self._update_count_propositional()
            else:
                res = self.knowledge.eval()
                for head in configs_tables:
                    configs_table = configs_tables[head]
                    for query in res:
                        prob = res[query]
                        dumb_var = query.split('__')[0]
                        configs_table.loc[configs_table['dumb_var'] == dumb_var, 'count'] += prob

            ### M step ###
            self.logger.info("Starting M step #{}".format(step))
            self._log_time('M step')
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
                            calculations.expected_head_log_likelihood,
                            initial_guess,
                            args = (head, model, configs_table, -1.0),
                            method = 'L-BFGS-B',
                            bounds = [(0.000001, 0.999999)]*len(initial_guess),
                            options = {'disp': True ,'eps' : 1e-7})
                    optimal_params = res.x.tolist()
                self.logger.debug("Optimal params for head {}: {}".format(head, optimal_params))

                # update log-likelihood
                # expec_ll += calculations.expected_head_log_likelihood(optimal_params, head, model, configs_table)

                # consistency check --- in case we won't do inference for complete rows in the complete case
                # if expec_ll == float('-inf'):
                #     raise InconsistentEvidenceError()

                # store new parameters
                new_params[head] = optimal_params

            self._log_time('Others')
            # update parameters of the model
            for head in model:
                rules = model[head]['rules']
                for i, rule in enumerate(rules):
                    rules[i]['parameter'] = new_params[head][i]

            # update weights
            self.knowledge.update_weights(model)
            # calculate observed log likelihood
            obs_ll = self._calculate_log_likelihood()
            # EM cycle stopping criteria
            self.logger.info("Log-likelihood for step #{} step of EM: {}".format(step, obs_ll))
            if (obs_ll - old_obs_ll) < epsilon and (obs_ll - old_obs_ll) >= 0:
                learning_info = self._update_learning_info(obs_ll, end_em=True)
                break
            old_obs_ll = obs_ll
            step += 1
            # update data about iterations
            self._update_learning_info(obs_ll)
        return learning_info


    def _calculate_log_likelihood(self):
        """Calculates the observed log likelihood for the current model.
        """
        if self.relational_data:
            obs_ll = math.log(self.knowledge.p_evidence()[0])
        else:
            obs_ll = 0
            for i, row in self.propositional_dataset.iterrows():
                obs_ll += math.log(self.knowledge.p_evidence(evidence=row))
        return obs_ll


    def _update_learning_info(self, log_likelihood,
                              begin_em=False, end_em=False):
        """This function stores information about parameters, log-likelihood
        and elapsed time during parameter learning. At the end of the cycle,
        it'll put this information into a DataFrame accessible by
        self.learn_parameters_info.
        Keyword arguments:
        log_likelihood -- the observed-value of the log-likelihood of the whole
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
        return self.info


    def _log_time(self, activity, start=False):
        if not start:
            last_activity = self._log_time__last_activity
            last_time = self._log_time__last_time
            if not last_activity in self.info['time_log']:
                self.info['time_log'][last_activity] = 0
            self.info['time_log'][last_activity] += time.time() - last_time
        self._log_time__last_activity = activity
        self._log_time__last_time = time.time()


    def _read_propositional_dataset(self):
        self.logger.info("Reading and checking propositional dataset...")
        # ensure dataset is a pandas dataframe
        self.propositional_dataset = pd.read_csv(self.dataset_filepath)
        # ensure all variables in the dataset are head of a rule
        for column in self.propositional_dataset.columns:
            if not column in self.model:
                # what to do if column in dataset is not head of any rule?
                raise Exception('Column %s in dataset is not head of any rule.'
                                % column)
        self.logger.info("Ok")


    def check_if_complete(self,row):
        if row.isnull().values.any():
            return False
        else:
            return True


    def _update_count_propositional(self):
        configs_tables=self.configs_tables
        self.logger.info('Updating count for propositional dataset...')
        # get ordered configuration variables for each head
        config_vars_head = {}
        for head in configs_tables:
            number_of_vars = len(self.model[head]['parents']) + 1
            columns = self.configs_tables[head].columns.tolist()
            config_vars_head[head] = columns[0:number_of_vars]
        # count the number of occurences of each configuration for each family
        for i, row in self.propositional_dataset.iterrows():
            complete = self.check_if_complete(row)
            if not complete:
                res = self.knowledge.eval(evidence=row)
                for head in configs_tables:
                    configs_table = configs_tables[head]
                    for c, config in configs_table.iterrows():
                        if config['dumb_var'] in res:
                            update_in_count = res[config['dumb_var']]
                            configs_table.loc[c, 'count'] += update_in_count
            else:
                for head in configs_tables:
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

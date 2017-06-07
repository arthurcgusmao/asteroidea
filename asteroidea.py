import time
import math
import numpy as np
import pandas as pd
import itertools
from scipy.optimize import basinhopping, minimize
from problog.program import PrologString
from problog import get_evaluatable
from problog.errors import InconsistentEvidenceError

class Plp(object):

    def __init__(self):
        print('')

        
    def read_structure(self, filepath):
        """Reads a file containing a structure for the PLP program. The
        structure should be written accordingly to ProbLog's syntax.

        The structure is stored in a dict called self.model where each key is
        the head of the rules and the value is a rules list (for the given
        head).

        Each rule in the rules list corresponds to a dict with the following
        key-value pairs:
        parameter -- the parameter of the rule;
        body -- a dict where each key is a variable and its value is 1 if the
                variable is non-negated or 0 if it is negated.

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
            parameter, clause = line.split('::', 1)
            if ':-' in clause:
                head, body = clause.split(':-')
                body = body.split(',')
                body_dict = {}
                for body_var in body:
                    if '\+' in body_var:
                        body_var = body_var.replace('\+', '')
                        body_dict[body_var] = 0
                    else:
                        body_dict[body_var] = 1
            else:
                head = clause
                body_dict = {}
            # update self.model
            if not head in self.model:
                self.model[head] = []
            self.model[head].append({'parameter': float(parameter),
                                     'body': body_dict,
                                     'clause_string': clause})
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
        old_ll = None
        self._update_learning_info(np.nan, begin_em=True)
        while True:
            ll = 0
            for head in self.model:
                # in this inner loop, both E and M steps are performed for each
                # variable, instead of for all variables at once.
                # this implies that the E step for a later head variable will
                # use the already updated parameters from the last variable.
                # is this better or worse than the classical EM cycle?
                # does it still converge performing the updates this way?
                rules = self.model[head]
                configs_table = self.configs_tables[head]
                config_vars = [head]
                for parent in self.parents[head]:
                    config_vars.append(parent)
                
                ### E step ###
                configs_table.loc[:, 'count'] = 0 # zero all counts
                for i, row in dataset.iterrows():
                    for c, config in configs_table.iterrows():
                        prob = self.inference(
                                query=config.filter(items=config_vars),
                                evidence=row)
                        configs_table.loc[c, 'count'] += prob # update count

                ### M step ###
                optimal_params = self._exact_ll_maximization()
                if optimal_params == False:
                    # there is no exact solution, run optimization method
                    initial_guess = []
                    for rule in rules:
                        initial_guess.append(rule['parameter'])
                    res = minimize(
                            self._head_log_likelihood,
                            initial_guess,
                            args = (head, -1.0),
                            method = 'L-BFGS-B',
                            bounds = [(0.00001, 0.99999)]*len(initial_guess),
                            # bounds = [(0.001,0.999)]*len(initial_guess),
                            options = {'disp': True ,'eps' : 1e-7})
                    optimal_params = res.x.tolist()
                
                # update log-likelihood
                ll += self._head_log_likelihood(optimal_params, head)
                # update parameters of the model
                for i, rule in enumerate(rules):
                    rules[i]['parameter'] = optimal_params[i]
                self.model[head] = rules
            # EM cycle stopping criteria
            if old_ll != None:
                if abs((ll - old_ll) / ll) < epsilon:
                    learning_info = self._update_learning_info(ll, end_em=True)
                    break
            old_ll = ll
            # update data about iterations
            self._update_learning_info(ll)
        return learning_info
                
        
    def _head_log_likelihood(self, parameters, head, sign=+1.0):
        """Returns the expected-value of the log-likelihood of a head variable
        given its parents, for all possible configurations the examples of a
        dataset can take. In other words, it is the function that the M step
        tries to maximize in the EM cycle. It is implict that the model and the
        dataset are given, and that the appropriated calculations in
        self.configs_tables[head] were made.

        Keyword arguments:
        parameters -- the parameters for the set of rules which head is head
        head -- the variable that is head of the rules
        sign -- sign of the output. Default is 1.0, use -1.0 for
                minus-log-likelihood
        """
        # parameters are only of rules which head is head
        rules = list(self.model[head]) # make a copy of rules list
        for i, rule in enumerate(rules):
            rules[i]['parameter'] = parameters[i]
        model = {head: rules}
        parents = self.parents[head]
        configs_table = self.configs_tables[head]
        # update column likelihood of configs_table using given parameters.
        # we only need to consider configurations which count > 0
        for c, config in configs_table[configs_table['count'] > 0].iterrows():
            # we only need to update the value of the likelihood for the cases
            # where the parameters influence it (i.e., when there are active
            # rules).
            if config['active_rules'] != '':
                # convert active_rules string to list
                active_rules = config['active_rules'].split(',')
                active_rules = [int(r) for r in active_rules]
                # calculate likelihood
                prob = 0
                for rule_index in active_rules:
                    param = rules[rule_index]['parameter']
                    prob += param - prob * param
                if config[head] == 0:
                    prob = 1 - prob
                # prob = self.inference(query = config.filter(items=[head]),
                #                       evidence = config.filter(items=parents),
                #                       model = model)
                configs_table.loc[c, 'likelihood'] = prob
        # calculate the sum of all log-likelihood * count for table
        output = 0
        for c, config in configs_table[configs_table['count'] > 0].iterrows():
            output += config['count'] * math.log10(config['likelihood'])
        return sign*output

    
    def _build_configs_tables(self):
        """Builds self.configs_tables dict, which for each head is associated a
        Pandas DataFrame that represents the configurations table for that head.

        Columns of each dataframe are:
        head variable -- value the head variable takes for the configuration
        body variable(s) -- value the parents of head take for the config
        count -- expected number of times the configuration is observed, given
                 a model and dataset. It should be updated each time the
                 learning algorithms passes throught the E-step
        likelihood -- the likelihood of the head given parents in that config
        active_rules -- the rule's indexes that are active for that config
        """
        self.configs_tables = {}
        for head in self.model:
            rules = self.model[head]
            init_columns = [head]
            init_columns.extend(self.parents[head])
            configs_values = list(
                map(list, itertools.product([0, 1], repeat=len(init_columns))))
            df = pd.DataFrame(configs_values, columns=init_columns)
            df.loc[:,'count'] = pd.Series(np.nan, index=df.index)
            df.loc[:,'likelihood'] = pd.Series(np.nan, index=df.index)
            df.loc[:,'active_rules'] = pd.Series(None, index=df.index)
            for c, config in df.iterrows():
                # fill active_rules column
                active_rules = []
                for r, rule in enumerate(rules):
                    body = rule['body']
                    rule_active = True
                    for parent in body:
                        if not body[parent] == config[parent]:
                            rule_active = False
                    if rule_active:
                        active_rules.append(str(r))
                # converts list to string
                df.loc[c, 'active_rules'] = ','.join(active_rules)
                # pre-compute likelihood values that doesn't depend on the
                # parameters (they are defined by the structure)
                if len(active_rules) == 0:
                    df.loc[c, 'likelihood'] = 1 - config[head]
            self.configs_tables[head] = df


    def inference(self, query, evidence=pd.Series(), model=None):
        """Computes inference for a set of query variables, given the model and
        evidences.

        Keyword arguments:
        query -- a Panda Series containing queried values
        evidence -- a Panda Series containing observed values
        model -- a model (structure + parameters) to replace self.model

        For both query and evidence arguments the indexes of the Pandas Series
        should be the variable names. Values different from 0 or 1 will be
        interpreted as missing and be disconsidered.
        """
        if model == None:
            model = self.model
        # generate model string accordingly to ProbLog's syntax
        model_str = """"""
        for head in model:
            rules = model[head]
            for rule in rules:
                model_str += str(rule['parameter']) + '::' + head
                if len(rule['body']) > 0:
                    model_str += ':-'
                    for parent in rule['body']:
                        model_str += parent + ','
                else:
                    model_str += ','
                model_str = model_str[:-1]
                model_str += '.\n'
        # add evidence
        for var, value in evidence.iteritems():
            if value == 1:
                model_str += "evidence(%s, true).\n" % var
            if value == 0:
                model_str += "evidence(%s, false).\n" % var
        # add query
        dumb_var = "y"
        while dumb_var in model.keys():
            dumb_var += "y"
        model_str += "%s:-" % dumb_var
        for var, value in query.iteritems():
            if value == 1:
                model_str += "%s," % var
            if value == 0:
                model_str += "\+%s," % var
        model_str = model_str[:-1]
        model_str += '.\n'
        model_str += "query(%s).\n" % dumb_var
        # make inference using problog
        pl_model = PrologString(model_str.lower())
        try:
            res = get_evaluatable().create_from(pl_model).evaluate()
            for key in res:
                output = res[key]
        except InconsistentEvidenceError:
            raise InconsistentEvidenceError("""This error may have occured
                because some observation in the dataset is impossible given the
                model structure.""")
        return output


    def _exact_ll_maximization(self):
        # return False if there is no exact solution
        return False


    def _update_learning_info(self, log_likelihood, begin_em=False, end_em=False):
        if begin_em:
            self._start_time = time.time()
            self._learning_info = []
        iter_data = []
        for head in self.model:
            for rule in self.model[head]:
                iter_data.append(rule['parameter'])
        elapsed_time = time.time() - self._start_time
        iter_data.extend([elapsed_time, log_likelihood])
        self._learning_info.append(iter_data)
        if end_em:
            columns = []
            for head in self.model:
                for rule in self.model[head]:
                    columns.append(rule['clause_string'])
            columns.extend(['Elapsed Time', 'Log-Likelihood'])
            return pd.DataFrame(self._learning_info, columns=columns)

        

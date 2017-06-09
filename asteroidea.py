import time
import math
import numpy as np
import pandas as pd
import itertools
from scipy.optimize import basinhopping, minimize
from problog.program import PrologString
from problog import get_evaluatable
from problog.evaluator import SemiringLogProbability
from problog.logic import Term, Constant
from problog.errors import InconsistentEvidenceError



class CustomSemiring(SemiringLogProbability):
    # this class is used to make inference faster for the same structure but
    # changing parameters using ProbLog
    
    def __init__(self, weights):
        SemiringLogProbability.__init__(self)
        self.weights = weights

        
    def value(self, a):
        # Argument 'a' contains ground term. Look up its probability in the
        # weights dictionary.
        return SemiringLogProbability.value(self, self.weights.get(a, a))


    
class Plp(object):

    def __init__(self):
        self.problog_knowledge_sr = None

        
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
            new_params = {}
                
            ### E step ###
            for head in self.model:
                # reset configurations tables
                self.configs_tables[head].loc[:, 'count'] = 0
            dumb_var = "y"
            while dumb_var in self.model.keys():
                dumb_var += "y"
            for i, row in dataset.iterrows():
                res = self.inference(evidence=row)
                for head in self.model:
                    configs_table = self.configs_tables[head]
                    config_vars = [head].extend(self.parents[head])
                    for c, config in configs_table.iterrows():
                        config_dumb_var = dumb_var + '_' + head + '_' + str(c)
                        update_in_count = res[config_dumb_var.lower()]
                        configs_table.loc[c, 'count'] += update_in_count

            ### M step ###
            for head in self.model:
                rules = self.model[head]
                optimal_params = self._exact_ll_maximization(head)
                if optimal_params == False:
                    print("Iteration", len(self._learning_data), "had no exact solution for head", head, "bro.")
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
                # store new parameters
                new_params[head] = optimal_params
            
            # update parameters of the model
            for head in self.model:
                rules = self.model[head]
                for i, rule in enumerate(rules):
                    rules[i]['parameter'] = new_params[head][i]
            # EM cycle stopping criteria
            if old_ll != None:
                if abs((ll - old_ll) / ll) < epsilon:
                    learning_info = self._update_learning_info(ll, end_em=True)
                    break
            old_ll = ll
            # update data about iterations
            self._update_learning_info(ll)
        return learning_info
                
        
    def _head_log_likelihood(self, parameters, head, sign=1):
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
                    prob += param - (prob * param)
                if config[head] == 0:
                    prob = 1 - prob
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


    def inference(self, evidence=pd.Series()):
        """Computes inference for all configurations of all families of head
        variables, given the current model and evidences.

        Keyword arguments:
        evidence -- a Panda Series containing observed values. The indexes of
                    the Pandas Series should be the variable names. Values
                    different from 0 or 1 will be interpreted as missing
                    and be disconsidered.
        """
        model = self.model
        if self.problog_knowledge_sr == None:
            # generate model string accordingly to ProbLog's syntax
            model_str = """"""
            evidences_str = """"""
            queries_str = """"""
            self.params_strings = []
            dumb_var = "y"
            while dumb_var in model.keys():
                dumb_var += "y"
            for head in model:
                # add clauses -- we use a variable named theta_head_index to
                # change the model's parameters dinamically
                rules = model[head]
                for i, rule in enumerate(rules):
                    param_str = 'theta_' + head + '_' + str(i)
                    model_str += param_str + '::' + rule['clause_string'] + '.\n'
                    self.params_strings.append(param_str.lower())
                # add evidence and query -- all variables should be
                # evidence/queries because only then we can avoid recompiling
                # the model for different evidences/queries. There is no
                # problem in setting every evidence to true because later these
                # values are discarded.
                # evidence:
                evidences_str += "evidence(%s, true).\n" % head
                # queries (each configuration is a query):
                configs_table = self.configs_tables[head]
                config_vars = [head]
                for parent in self.parents[head]:
                    config_vars.append(parent)
                for c, config in configs_table.iterrows():
                    query = config.filter(items=config_vars)
                    config_dumb_var = dumb_var + '_' + head + '_' + str(c)
                    queries_str += config_dumb_var + ':-'
                    for var, value in query.iteritems():
                        if value == 1:
                            queries_str += "%s," % var
                        if value == 0:
                            queries_str += "\+%s," % var
                    queries_str = queries_str[:-1]
                    queries_str += '.\n'
                    queries_str += "query(%s).\n" % config_dumb_var
            model_str += evidences_str + queries_str
            model_str = model_str.lower()
            self.model_str = model_str

            # parse the Prolog string
            pl_model_sr = PrologString(model_str)
            # compile the Prolog model
            self.problog_knowledge_sr = get_evaluatable().create_from(pl_model_sr)

        # change model weights
        custom_weights = {}
        for x in self.problog_knowledge_sr.get_weights().values():
            if getattr(x, "functor", None):
                for head in self.model:
                    rules = self.model[head]
                    for i, rule in enumerate(rules):
                        param_str = 'theta_' + head + '_' + str(i)
                        param_str = param_str.lower()
                        if x.functor == param_str:
                            custom_weights[x] = rule['parameter']

        # change evidence
        evidence_dict = {}
        for var, value in evidence.iteritems():
            var = var.lower()
            if value == 1:
                term = Term(var)
                evidence_dict[term] = True
            if value == 0:
                term = Term(var)
                evidence_dict[term] = False
                
        # make inference
        try:
            res = self.problog_knowledge_sr.evaluate(
                    evidence=evidence_dict,
                    # keep_evidence=False,
                    semiring=CustomSemiring(custom_weights)),
            output = {}
            for key in res[0]:
                output[str(key)] = res[0][key]
            # output = res[0]
        except InconsistentEvidenceError:
            raise InconsistentEvidenceError("""This error may have occured
                because some observation in the dataset is impossible given the
                model structure.""")
        return output


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
            for rule in self.model[head]:
                iter_data.append(rule['parameter'])
        elapsed_time = time.time() - self._start_time
        iter_data.extend([elapsed_time, log_likelihood])
        self._learning_data.append(iter_data)
        if end_em:
            columns = []
            for head in self.model:
                for rule in self.model[head]:
                    columns.append(rule['clause_string'])
            columns.extend(['Elapsed Time', 'Log-Likelihood'])
            self.learn_parameters_info = pd.DataFrame(self._learning_data,
                                                      columns=columns)

            
    def _exact_ll_maximization(self,head):
        """Returns the optimal parameters that maximize the likelihood for a
        number of structures. If for the given structure it is not possible to
        find the optimal paramters using this method, returns False.
        """
        configs_table = self.configs_tables[head]
        rules_combinations = set(configs_table['active_rules'].tolist())

        #Combinations of 1 rule
        if set(["1"])==rules_combinations: #Exact sollution
            A1=configs_table.loc[configs_table[head]==0,'count'].sum(axis=1)
            A0=configs_table.loc[configs_table[head]==1,'count'].sum(axis=1)

            aux_a=np.float64(A0)/(A0+A1)
            if A0+A1==0:
                aux_a=0.5
            probabilities_list[0]=max(min(aux_a,0.999),0.001) #r1

            return probabilities_list

        #Combinations of 2 rules
        if set(["1","2"])==rules_combinations: #Exact sollution
            coefficients=self.calculate_coefficients(["1","2"])
            A1=coefficients[0]
            A0=coefficients[1]
            B1=coefficients[2]
            B0=coefficients[3] 

            aux_a=np.float64(A0)/(A0+A1)
            if A0+A1==0:
                aux_a=0.5

            aux_b=np.float64(B0)/(B0+B1)
            if B0+B1==0:
                aux_b=0.5

            probabilities_list[0]=max(min(aux_a,0.999),0.001) #r1
            probabilities_list[1]=max(min(aux_b,0.999),0.001) #r2

            return probabilities_list

        if set(["1","1,2"])==rules_combinations: #Exact sollution
            coefficients=self.calculate_coefficients(["1","1,2"])
            A1=coefficients[0]
            A0=coefficients[1]
            B1=coefficients[2]

            exact_sollution=True  

            aux_a=np.float64(A0)/(A0+A1)

            aux_b=np.float64(A1*B0-A0*B1)/(A1*B0+A1*B1)

            if A0+A1==0:
                aux_a=0.0
                aux_b=np.float64(B0)/(B0+B1)
                if B0+B1==0:
                    aux_a=0.5
                    aux_b=0.5

            if A1*B0+A1*B1==0:
                if A1==0:
                    if B0==0 and B1==0:
                        aux_b=0.5
                    if B0==0 and B1!=0:
                        aux_b=0.0
                    if B0!=0 and B1==0:
                        aux_b=1.0
                    if B0!=0 and B1!=0:
                        aux_b=B0/(B0+B1)
                        if exact_sollution==True:
                            exact_sollution=False  
                else: #A1!=0       
                    if B0==0 and B1==0:
                        aux_b=0.5

            probabilities_list[0]=max(min(aux_a,0.999),0.001) #r1
            probabilities_list[1]=max(min(aux_b,0.999),0.001) #r2  

            if not exact_sollution:
                return False

            return probabilities_list

        if set(["1","2","1,2"])==rules_combinations: #No exact sollution
            return False

        #Combinations of 3 rules
        if set(["1","2","3"])==rules_combinations: #Exact sollution
            coefficients=self.calculate_coefficients(["1","2","3"])
            A1=coefficients[0]
            A0=coefficients[1]
            B1=coefficients[2]
            B0=coefficients[3]
            C1=coefficients[4]
            C0=coefficients[5]     

            aux_a=np.float64(A0)/(A0+A1)
            if A0+A1==0:
                aux_a=0.5

            aux_b=np.float64(B0)/(B0+B1)
            if B0+B1==0:
                aux_b=0.5

            aux_c=np.float64(C0)/(C0+C1)
            if C0+C1==0:
                aux_c=0.5

            probabilities_list[0]=max(min(aux_a,0.999),0.001) #r1
            probabilities_list[1]=max(min(aux_b,0.999),0.001) #r2
            probabilities_list[2]=max(min(aux_c,0.999),0.001) #r3         
        
            return probabilities_list

        if set(["1","2","1,2","3"])==rules_combinations: #No exact sollution      
            return False

        if set(["1","1,2","1,3","1,2,3"])==rules_combinations: #No exact sollution
            return False

        if set(["1","2","1,3","2,3"])==rules_combinations: #No exact sollution
            return False

        if set(["2","3","1,2","1,3"])==rules_combinations: #No exact sollution
            return False

        if set(["1","1,2","1,3"])==rules_combinations: #Exact sollution
            coefficients=self.calculate_coefficients(["1","1,2","1,3"])
            A1=coefficients[0]
            A0=coefficients[1]
            B1=coefficients[2]
            B0=coefficients[3]
            C1=coefficients[4]
            C0=coefficients[5]          

            exact_sollution=True

            aux_a=np.float64(A0)/(A0+A1)
            if A0+A1==0:
                aux_a=0.5

            aux_b=np.float64(A1*B0-A0*B1)/(A1*B0+A1*B1)

            aux_c=np.float64(A1*C0-A0*C1)/(A1*C0+A1*C1)

            if A1*B0+A1*B1==0 and A1*C0+A1*C1==0:
                if A1==0:
                    if B0==0 and B1==0:
                        aux_b=0.5
                    if B0==0 and B1!=0:
                        aux_b=0.0
                    if B0!=0 and B1==0:
                        aux_b=1.0
                    if B0!=0 and B1!=0:                       
                        aux_b=B0/(B0+B1)
                        if exact_sollution==True:
                            exact_sollution=False

                    if C0==0 and C1==0:
                        aux_c=0.5
                    if C0==0 and C1!=0:
                        aux_c=0.0
                    if C0!=0 and C1==0:
                        aux_c=1.0
                    if C0!=0 and C1!=0:                       
                        aux_c=B0/(C0+C1)
                        if exact_sollution==True:
                            exact_sollution=False

                else:# -> B0=B1=C0=C1=0
                    aux_b=0.5
                    aux_c=0.5

            if A1*B0+A1*B1==0 and A1*C0+A1*C1!=0:# -> B0=B1=0
                aux_b=0.5

            if A1*B0+A1*B1!=0 and A1*C0+A1*C1==0:# -> C0=C1=0
                aux_c=0.5

            probabilities_list[0]=max(min(aux_a,0.999),0.001) #r1
            probabilities_list[1]=max(min(aux_b,0.999),0.001) #r2
            probabilities_list[2]=max(min(aux_c,0.999),0.001) #r3    

            if not exact_sollution:
                return False

            return probabilities_list

        #Combinations of 4 rules
        if set(["1","2","3","4"])==rules_combinations: #Exact sollution
            coefficients=self.calculate_coefficients(["1","2","3","4"])
            A1=coefficients[0]
            A0=coefficients[1]
            B1=coefficients[2]
            B0=coefficients[3]
            C1=coefficients[4]
            C0=coefficients[5]    
            D1=coefficients[6]
            D0=coefficients[7]   

            aux_a=np.float64(A0)/(A0+A1) 
            if A0+A1==0:
                aux_a=0.5
            
            aux_b=np.float64(B0)/(B0+B1) 
            if B0+B1==0:
                aux_b=0.5           

            aux_c=np.float64(C0)/(C0+C1)
            if C0+C1==0:
                aux_c=0.5 

            aux_d=np.float64(D0)/(D0+D1)
            if D0+D1==0:
                aux_d=0.5 

            probabilities_list[0]=max(min(aux_a,0.999),0.001) #r1
            probabilities_list[1]=max(min(aux_b,0.999),0.001) #r2
            probabilities_list[2]=max(min(aux_c,0.999),0.001) #r3  
            probabilities_list[3]=max(min(aux_d,0.999),0.001) #r4

            return probabilities_list

        return False  

    def calculate_coefficients(self,head,pattern_list):

        configs_table = self.configs_tables[head]
        coefficients=[]
        for i, i_element in enumerate(pattern_list):
            coeffcients.extend([0.0,0.0])
            coefficients[2*i]=configs_table.loc[configs_table['active_rules']==i_element and configs_table[head]==0,'count'].sum(axis=1)
            coefficients[2*i+1]=configs_table.loc[configs_table['active_rules']==i_element and configs_table[head]==1,'count'].sum(axis=1)

        #print "                coefficents=",coefficients
        return coefficients

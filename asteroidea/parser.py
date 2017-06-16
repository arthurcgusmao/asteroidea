import itertools
import numpy as np
import pandas as pd

def read_structure(filepath):
    """Reads a file containing a structure for the PLP program and returns a
    model dict. The structure should be written accordingly to ProbLog's
    syntax. The structure is stored in a dict called model where each key is
    the head of the rules and the value another dict consisting of two
    elements: rules dict (a list of rules for that head) and parents set (the
    set of parents for that head).

    Each rule in the rules list corresponds to a dict with the following
    key-value pairs:

    parameter -- the parameter of the rule
    parameter_name -- a generated name for the parameter
    body -- a dict where each key is a variable and its value is 1 if the
            variable is non-negated or 0 if it is negated
    clause_string -- the original string for that clause disconsidering the
                     parameters
    """
    model = {}
    temp_file = open(filepath, 'r+')
    for i, line in enumerate(temp_file):
        # comment syntax
        if '%' in line:
            continue
        # remove whitespace and end of line
        line = line.replace(' ', '').replace('.\n', '')
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
        # update rules
        if not head in model:
            model[head] = {'rules': [],
                           'parents': set()}
            param_index = 0
        # generate parameter name
        predicate, _ = parse_relational_var(head)
        param_name = 'theta_' + predicate + '_' + str(param_index)
        param_index += 1
        model[head]['rules'].append({'parameter': float(parameter),
                                     'parameter_name': param_name,
                                     'body': body_dict,
                                     'clause_string': clause})
        # update parents
        for parent in body_dict:
            model[head]['parents'].add(parent)
    # dumb var for probabilistic observation
    for head in model:
        dumb_var = 'y'
        while dumb_var in model.keys():
            dumb_var += dumb_var
        predicate, _ = parse_relational_var(head)
        prob_dumb_var = dumb_var +'_'+ predicate
        model[head]['prob_dumb'] = {
            'var': prob_dumb_var,
            'weight_0': 'theta_' + prob_dumb_var +'_0',
            'weight_1': 'theta_' + prob_dumb_var +'_1'}
    temp_file.close()
    return model


def build_configs_tables(model):
    """Builds a configuration tables dict, which for each head is associated a
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
    dumb_var = 'c'
    while dumb_var in model.keys():
        dumb_var += dumb_var
        
    configs_tables = {}
    for head in model:
        rules = model[head]['rules']
        init_columns = [head]
        init_columns.extend(model[head]['parents'])
        configs_values = list(
            map(list, itertools.product([0, 1], repeat=len(init_columns))))
        df = pd.DataFrame(configs_values, columns=init_columns)
        df.loc[:,'count'] = pd.Series(np.nan, index=df.index)
        df.loc[:,'likelihood'] = pd.Series(np.nan, index=df.index)
        df.loc[:,'active_rules'] = pd.Series(None, index=df.index)
        df.loc[:,'dumb_var'] = pd.Series(None, index=df.index)
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
            # generate dumb_var names
            predicate, _  = parse_relational_var(head)
            df.loc[c, 'dumb_var'] = dumb_var + '_' + predicate + '_' + str(c)
        configs_tables[head] = df
    return configs_tables


def build_problog_model_str(model, configs_tables, probabilistic_data=False,
                            suppress_evidences=False, relational_dataset=None):
    """Parses a set of rules and configuration tables and creates a model
    ready to make inference.

    Keyword arguments (see Model class for more information):
    model -- model dict 
    config_tables -- a list of configuration tables
    """
    # generate model string accordingly to ProbLog's syntax
    model_str = ''
    rules_str = ''
    prob_str = ''
    configs_str = ''
    evidences_str = ''
    queries_str = ''

    relational_data = False
    if relational_dataset != None:
        relational_data = True
        dataset_str, constants = parse_relational_dataset(dataset_filepath)
    
    for head in model:
        # add clauses -- we use a variable named theta_head_index to
        # change the model's parameters dinamically
        rules = model[head]['rules']
        parents = model[head]['parents']
        for i, rule in enumerate(rules):
            rules_str += rule['parameter_name'] + '::' + rule['clause_string'] + '.\n'
        # add probabilistic observations string
        if probabilistic_data:
            prob_dumb_var = model[head]['prob_dumb']['var']
            prob_dumb_weight_0 = model[head]['prob_dumb']['weight_0']
            prob_dumb_weight_1 = model[head]['prob_dumb']['weight_1']
            prob_str += prob_dumb_weight_0 +'::'+ prob_dumb_var +':-\+'+ head +'.\n'
            prob_str += prob_dumb_weight_1 +'::'+ prob_dumb_var +':-'+ head +'.\n'
            if not suppress_evidences:
                evidences_str += 'evidence('+ prob_dumb_var +', true).\n'
        # add evidence and query -- all variables should be
        # evidence/queries because only then we can avoid recompiling
        # the model for different evidences/queries. There is no
        # problem in setting every evidence to true because later these
        # values are discarded.
        # evidence:
        if not suppress_evidences:
            evidences_str += "evidence(%s, true).\n" % head
        # queries (each configuration is a query):
        configs_table = configs_tables[head]
        config_vars = [head]
        for parent in parents:
            config_vars.append(parent)
            
        if relational_data:
            # dealing with relational data
            substitutions = generate_substitutions(config_vars, constants)
            for c, config in configs_table.iterrows():
                query = config.filter(items=config_vars)
                for s, substitution in enumerate(substitutions):
                    config_dumb_var = configs_table.loc[c,:]['dumb_var'] +'__'+ str(s)
                    configs_str += config_dumb_var + ':-'
                    for var, value in query.iteritems():
                        predicate, arguments = parse_relational_var(var)
                        var_str = predicate+'('
                        for argument in arguments:
                            var_str += substitution[argument] + ','
                        var_str = var_str[:-1] + ')'
                        if value == 1:
                            configs_str += "%s," % var_str
                        if value == 0:
                            configs_str += "\+%s," % var_str
                    configs_str = configs_str[:-1]
                    configs_str += '.\n'
                    queries_str += "query(%s).\n" % config_dumb_var
        else:
            for c, config in configs_table.iterrows():
                query = config.filter(items=config_vars)
                config_dumb_var = configs_table.loc[c,:]['dumb_var']
                configs_str += config_dumb_var + ':-'
                for var, value in query.iteritems():
                    if value == 1:
                        configs_str += "%s," % var
                    if value == 0:
                        configs_str += "\+%s," % var
                configs_str = configs_str[:-1]
                configs_str += '.\n'
                queries_str += "query(%s).\n" % config_dumb_var
                
    model_str += rules_str + configs_str + prob_str + queries_str
    if not suppress_evidences:
        model_str += evidences_str
    if relational_data:
        output = dataset_str + model_str
    else:
        output = model_str
    return output


def parse_relational_var(var):
    if '(' in var:
        predicate, arguments = var.split('(')
        arguments = arguments[:-1].split(',')
    else:
        predicate = var
        arguments = []
    return predicate, arguments


def generate_substitutions(atoms, constants):
    #@TODO: create substitutions with replacement
    arguments_set = set()
    for atom in atoms:
        _, arguments = parse_relational_var(atom)
        arguments_set = arguments_set.union(set(arguments))
    substitutions = []
    combinations = itertools.permutations(constants, len(arguments_set))
    for combination in combinations:
        substitution = {}
        for i, argument in enumerate(arguments_set):
            # print(argument, combination)
            substitution[argument] = combination[i]
        substitutions.append(substitution)
    return substitutions

    
def parse_relational_dataset(filepath):
    constants = set()
    temp_file = open(filepath, 'r+')
    # dataset_str = temp_file.read()
    dataset_str = ''
    for i, line in enumerate(temp_file):
        dataset_str += line
        # comment syntax
        if '%' in line:
            continue
        # remove whitespace and end of line
        line = line.replace(' ', '').replace('\n', '').replace('.', '')
        # parse line
        _, arguments = parse_relational_var(line)
        constants = constants.union(set(arguments))
    temp_file.close()
    return dataset_str, constants
    

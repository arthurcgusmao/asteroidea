import itertools


def generate_rules_combinations(head, parents_list):
    number_of_parents = len(parents_list)
    rules_combinations = []
    if number_of_parents == 0:
        rules_combinations.append([{'parameter': 0.5,
                                    'body': {},
                                    'head': head}])
    
    if number_of_parents == 1:
        parent1 = parents_list[0]
        # COMBINATIONS (list, see case 2 below)
        combinations = [
            [[1]],
            [[0]],
            [[1], [0]]
        ]
        for combination in combinations:
            rules_combination = []
            for rule_repr in combination:
                body_dict = {parent1: rule_repr[0]}
                rule = {'parameter': 0.5,
                        'body': body_dict,
                        'head': head}
                rules_combination.append(rule)
            rules_combinations.append(rules_combination)

    if number_of_parents == 2:
        parent1 = parents_list[0]
        parent2 = parents_list[1]
        # COMBINATIONS (list)
        # each combination is a list of rules representations
        # each rules representation is an activation for the parents:
        # [0, 1] means parent1 activates when 0 and parent2 activates when 1
        combinations = [
            [[1, 1]],
            [[1, 0]],
            [[0, 1]],
            [[0, 0]],
            
            [[None, None], [1, 1]],
            [[None, None], [1, 0]],
            [[None, None], [0, 1]],
            [[None, None], [0, 0]],
            
            [[1, None], [None, 1]],
            [[1, None], [None, 0]],
            [[0, None], [None, 1]],
            [[0, None], [None, 0]],
            
            [[1, None], [0, 1]],
            [[1, None], [0, 0]],
            [[0, None], [1, 1]],
            [[0, None], [1, 0]],
            
            [[None, 1], [1, 0]],
            [[None, 1], [0, 0]],
            [[None, 0], [1, 1]],
            [[None, 0], [0, 1]],
            
            [[1, 1], [1, 0]],
            [[1, 1], [0, 1]],
            [[1, 1], [0, 0]],
            [[1, 0], [0, 0]],
            [[1, 0], [0, 1]],
            [[0, 1], [0, 0]],
            
            [[None, None], [1, None], [None, 1]],
            [[None, None], [1, None], [None, 0]],
            [[None, None], [0, None], [None, 1]],
            [[None, None], [0, None], [None, 0]],
            [[None, None], [1, 1], [0, 0]],
            [[None, None], [1, 0], [0, 1]],
            
            [[1, None], [0, None], [None, 1]],
            [[1, None], [0, None], [None, 0]],
            [[1, None], [None, 1], [None, 0]],
            [[0, None], [None, 1], [None, 0]],
            
            [[1, None], [None, 1], [0, 0]],
            [[1, None], [None, 0], [0, 1]],
            [[0, None], [None, 1], [1, 0]],
            [[0, None], [None, 0], [1, 1]],
            
            [[0, None], [1, 1], [1, 0]],
            [[1, None], [0, 1], [0, 0]],
            [[None, 1], [1, 0], [0, 0]],
            [[None, 0], [1, 1], [0, 1]],

            [[1, 1], [1, 0], [0, 1]],
            [[1, 1], [0, 1], [0, 0]],
            [[1, 1], [1, 0], [0, 0]],
            [[1, 0], [0, 1], [0, 0]],

            [[1, 1], [1, 0], [0, 1], [0, 0]]
        ]
        
        for combination in combinations:
            rules_combination = []
            for rule_repr in combination:
                body_dict = {parent1: rule_repr[0],
                             parent2: rule_repr[1]}
                rule = {'parameter': 0.5,
                        'body': body_dict,
                        'head': head}
                rules_combination.append(rule)
            rules_combinations.append(rules_combination)
    return rules_combinations


def generate_vars_combinations(vars):
    """Vars should be a list of variables"""
    combs = []
    for r in range(1, 4):
        combs.extend(itertools.combinations(vars, r))
    output = []
    for comb in combs:
        output.append(list(comb))
    return output


def generate_possible_families(vars_combination):
    families = []
    for head in vars_combination:
        body_list = list(vars_combination)
        body_list.remove(head)
        # each family in a combination of variables has as head variable one
        # variable and the others are in the body
        families.append({'rules_combinations': generate_rules_combinations(head, body_list),
                         'head': head})
    return families


def generate_all_models(variables):
    """Generates all models that will be considered in the M step. The returned
    list is structured in the following manner:

    all models of all combinations of variables
    [
        all possible families for each combination of variables
        {'vars_combination': ...,
         'possible_families': ...}
        {'vars_combination': ...,
         'possible_families': ...}
        {'vars_combination': ['x0', 'x1'],
         'possible_families': [
                                  {'head': 'x0',
                                   'rules_combinations': ...},
                                  {'head': 'x1',
                                   'rules_combinations': [list of rules]}
                              ]
        }
    ]
    """
    vars_combinations = generate_vars_combinations(variables)
    all_models = []
    for vars_combination in vars_combinations:
        all_models.append({'possible_families': generate_possible_families(vars_combination),
                           'vars_combination': vars_combination})
    return all_models


def build_configs_tables(all_models):
    """Builds all configurations tables for all considered models. It returns a
    list of configurations tables which indexes follow the same order of the
    argument `all_models`.
    """
    configs_tables = []
    for vars_combination_models in all_models:
        # build one config table for each vars combination
        vars_combination = vars_combination_models['vars_combination']
        configs_values = list(
            map(list, itertools.product([0, 1], repeat=len(vars_combination))))
        df = pd.DataFrame(configs_values, columns=vars_combination)
        df.loc[:,'count'] = pd.Series(np.nan, index=df.index)
        df.loc[:,'dumb_var'] = pd.Series(None, index=df.index)

        # for each rules combination, create two coluns in the dataframe:
        # likelihood and active rules.
        df.loc[:,'likelihood'] = pd.Series(np.nan, index=df.index)
        df.loc[:,'active_rules'] = pd.Series(None, index=df.index)





# ITERATOR EXAMPLE
# for vars_combination_models in all_models:
#     vars_combination = vars_combination_models['vars_combination']
#     possible_families_models = vars_combination_models['possible_families']
#     for family_models in possible_families_models:
#         # head = family_models['head']
#         rules_combinations = family_models['rules_combinations']
#         for rule_combination in rules_combinations:
#             for rule in rule_combination:
#                 head = rule['head']
#                 body = rule['body']
#                 parameter = rule['parameter']

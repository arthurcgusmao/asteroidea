import math
import numpy as np


def head_log_likelihood(parameters, head, model, configs_table, sign=1):
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
    rules = list(model[head]['rules']) # make a copy of rules list
    for i, rule in enumerate(rules):
        rules[i]['parameter'] = parameters[i]
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


def log_likelihood(model, configs_tables, sign=1):
    """Returns the expected-value of the log-likelihood of the whole model.
    """
    ll = 0
    for head in model:
        configs_table = configs_tables[head]
        rules = model[head]['rules']
        params = []
        for rule in rules:
            params.append(rule['parameter'])
        ll += head_log_likelihood(params, head, model, configs_table)
    return ll


def exact_optimization(head, configs_table):
    """Returns the optimal parameters that maximize the likelihood for a
    number of structures. If for the given structure it is not possible to
    find the optimal paramters using this method, returns False.
    """
    # return False
    # print(configs_table)
    rules_combinations = set(configs_table['active_rules'].tolist()) - set([""])

    #Combinations of 1 rule
    if set(["0"])==rules_combinations: #Exact sollution
        probabilities_list=[0.0]
        coefficients=calculate_coefficients(head, configs_table,["0"])
        A1=coefficients[0]
        A0=coefficients[1]
        # print ("A0=",A0)
        # print ("A1=",A1)            
        aux_a=np.float64(A0)/(A0+A1)
        if A0+A1==0:
            aux_a=0.5
        probabilities_list[0]=max(min(aux_a,0.999),0.001) #r1

        return probabilities_list

    #Combinations of 2 rules
    if set(["0","1"])==rules_combinations: #Exact sollution
        probabilities_list=[0.0]*2
        coefficients=calculate_coefficients(head, configs_table,["0","1"])
        A1=coefficients[0]
        A0=coefficients[1]
        B1=coefficients[2]
        B0=coefficients[3] 
        # print ("A0=",A0)
        # print ("A1=",A1) 
        # print ("B0=",B0)
        # print ("B1=",B1) 
        aux_a=np.float64(A0)/(A0+A1)
        if A0+A1==0:
            aux_a=0.5

        aux_b=np.float64(B0)/(B0+B1)
        if B0+B1==0:
            aux_b=0.5

        probabilities_list[0]=max(min(aux_a,0.999),0.001) #r1
        probabilities_list[1]=max(min(aux_b,0.999),0.001) #r2

        return probabilities_list

    if set(["0","0,1"])==rules_combinations: #Exact sollution
        probabilities_list=[0.0]*2
        coefficients=calculate_coefficients(head, configs_table,["0","0,1"])
        A1=coefficients[0]
        A0=coefficients[1]
        B1=coefficients[2]
        B0=coefficients[3]

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

        # print ("A0=",A0)
        # print ("A1=",A1) 
        # print ("B0=",B0)
        # print ("B1=",B1) 

        probabilities_list[0]=max(min(aux_a,0.999),0.001) #r1
        probabilities_list[1]=max(min(aux_b,0.999),0.001) #r2  

        if not exact_sollution:
            return False

        return probabilities_list

    if set(["0","1","0,1"])==rules_combinations: #No exact sollution
        return False

    #Combinations of 3 rules
    if set(["0","1","2"])==rules_combinations: #Exact sollution
        probabilities_list=[0.0]*3
        coefficients=calculate_coefficients(head, configs_table,["0","1","2"])
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

        # print ("A0=",A0)
        # print ("A1=",A1) 
        # print ("B0=",B0)
        # print ("B1=",B1) 
        # print ("C0=",C0)
        # print ("C1=",C1) 

        probabilities_list[0]=max(min(aux_a,0.999),0.001) #r1
        probabilities_list[1]=max(min(aux_b,0.999),0.001) #r2
        probabilities_list[2]=max(min(aux_c,0.999),0.001) #r3         

        return probabilities_list

    if set(["0","1","0,1","2"])==rules_combinations: #No exact sollution      
        return False

    if set(["0","0,1","0,2","0,1,2"])==rules_combinations: #No exact sollution
        return False

    if set(["0","1","0,2","1,2"])==rules_combinations: #No exact sollution
        return False

    if set(["1","2","0,1","0,2"])==rules_combinations: #No exact sollution
        return False

    if set(["0","0,1","0,2"])==rules_combinations: #Exact sollution
        probabilities_list=[0.0]*3
        coefficients=calculate_coefficients(head, configs_table,["0","0,1","0,2"])
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

        # print ("A0=",A0)
        # print ("A1=",A1) 
        # print ("B0=",B0)
        # print ("B1=",B1) 
        # print ("C0=",C0)
        # print ("C1=",C1) 

        probabilities_list[0]=max(min(aux_a,0.999),0.001) #r1
        probabilities_list[1]=max(min(aux_b,0.999),0.001) #r2
        probabilities_list[2]=max(min(aux_c,0.999),0.001) #r3    

        if not exact_sollution:
            return False

        return probabilities_list

    #Combinations of 4 rules
    if set(["0","1","2","3"])==rules_combinations: #Exact sollution
        probabilities_list=[0.0]*4
        coefficients=calculate_coefficients(head, configs_table,["0","1","2","3"])
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

        # print ("A0=",A0)
        # print ("A1=",A1) 
        # print ("B0=",B0)
        # print ("B1=",B1) 
        # print ("C0=",C0)
        # print ("C1=",C1) 
        # print ("C0=",C0)
        # print ("C1=",C1) 

        probabilities_list[0]=max(min(aux_a,0.999),0.001) #r1
        probabilities_list[1]=max(min(aux_b,0.999),0.001) #r2
        probabilities_list[2]=max(min(aux_c,0.999),0.001) #r3  
        probabilities_list[3]=max(min(aux_d,0.999),0.001) #r4

        return probabilities_list

    return False  
  


def calculate_coefficients(head, configs_table, pattern_list):
    """"""
    coefficients=[]
    for i, i_element in enumerate(pattern_list):
        coefficients.extend([0.0,0.0])
        aux=configs_table[configs_table['active_rules']==i_element]
        aux=aux[aux[head]==0]['count']
        coefficients[2*i]=aux.sum()
        aux=configs_table[configs_table['active_rules']==i_element]
        aux=aux[aux[head]==1]['count']
        coefficients[2*i+1]=aux.sum()
    #print "                coefficents=",coefficients
    return coefficients

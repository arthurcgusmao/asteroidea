from problog.tasks import sample
from problog.program import PrologString
from problog.logic import Clause, Term, Constant



class Sample(object):

    
    def __init__(self, model_str, probabilistic_data=False, sample_size=5):
        self.probabilistic_data = probabilistic_data
        self.sample_size = sample_size
        # parse the Prolog string
        self.pl_model = PrologString(model_str)


    def update_weights(self, model):
        # """Updates the weights of the model."""
        self.model = model
        # self.custom_weights = {}
        self.pl_model_items = {}
        for item in self.pl_model:
            if getattr(item, "functor", None):
                self.custom_weights_items[x.functor] = x
        for head in model:
            rules = model[head]['rules']
            for i, rule in enumerate(rules):
                x = self.custom_weights_items[rule['parameter_name']]
                self.custom_weights[x] = rule['parameter']


    def eval(self, evidence=pd.Series()):
        # change evidence (and weights in case evidence is probabilistic)
        model = self.model
        # change evidence (and weights in case evidence is probabilistic)
        evidence_dict = {}
        for var, value in evidence.iteritems():
            term = Term(var)
            if value == 1:
                evidence_dict[term] = True
            if value == 0:
                evidence_dict[term] = False
            # MANAGING PROBABILISTIC CASE
            if self.probabilistic_data:
                # initialize all probability dumb variables custom weights
                x_0 = self.custom_weights_items[model[var]['prob_dumb']['weight_0']]
                x_1 = self.custom_weights_items[model[var]['prob_dumb']['weight_1']]
                self.custom_weights[x_1] = 0.5
                self.custom_weights[x_0] = 0.5
                if value > 0 and value < 1:
                    # if observation is probabilistic, insert evidence for dumb var
                    prob_term = Term(model[var]['prob_dumb']['var'])
                    evidence_dict[prob_term] = True
                    # and weights for probabilistic dumb rules
                    self.custom_weights[x_1] = value
                    self.custom_weights[x_0] = 1 - value
        # sample
        res = sample.sample(self.pl_model, n=self.sample_size, format='dict')
        output = {}
        for s in res:
            for config in s:
                c = str(config)
                if not c in output.keys():
                    output[c] = 0
                if res[config] == True:
                    output[c] += 1
        for c in output:
            output[c] /= float(self.sample_size)
        return output
        
    


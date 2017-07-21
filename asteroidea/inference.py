import parser
import pandas as pd
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



class Inference(object):


    def __init__(self, model_str, probabilistic_data=False, relational_data=False):
        # parse the Prolog string
        pl_model_sr = PrologString(model_str)
        # compile the Prolog model
        self.problog_knowledge_sr = get_evaluatable().create_from(pl_model_sr)
        self.probabilistic_data = probabilistic_data
        self.relational_data = relational_data


    def update_weights(self, model):
        """Updates the weights of the model."""
        self.model = model
        self.custom_weights = {}
        self.custom_weights_items = {}
        xs = self.problog_knowledge_sr.get_weights().values()
        for x in xs:
            if getattr(x, "functor", None):
                self.custom_weights_items[x.functor] = x
        for head in model:
            rules = model[head]['rules']
            for i, rule in enumerate(rules):
                if rule['parameter_name'] in self.custom_weights_items.keys():
                    x = self.custom_weights_items[rule['parameter_name']]
                    self.custom_weights[x] = rule['parameter']


    def eval(self, evidence=pd.Series()):
        """Returns the query for all configurations of all head variables
        in the model, given the evidence.

        Keyword arguments:
        evidence
        """
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

        # make inference
        try:
            if not self.relational_data:
                res = self.problog_knowledge_sr.evaluate(
                            evidence=evidence_dict,
                            keep_evidence=False,
                            semiring=CustomSemiring(self.custom_weights)),
            else:
                res = self.problog_knowledge_sr.evaluate(
                            # evidence=evidence_dict,
                            keep_evidence=True,
                            semiring=CustomSemiring(self.custom_weights)),
            output = {}
            for key in res[0]:
                output[str(key)] = res[0][key]
        except InconsistentEvidenceError:
            raise InconsistentEvidenceError("""This error may have occured
                because some observation in the dataset is impossible given
                the model structure.""")
        return output


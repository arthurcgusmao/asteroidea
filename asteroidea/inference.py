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

    
    def __init__(self, model_str):
        # parse the Prolog string
        pl_model_sr = PrologString(model_str)
        # compile the Prolog model
        self.problog_knowledge_sr = get_evaluatable().create_from(pl_model_sr)

        
    def eval(self, model, evidence=pd.Series()):
        """Returns the query for all configurations of all head variables
        in the model, given the evidence.

        Keyword arguments:
        evidence
        """
        # change model weights
        custom_weights = {}
        for x in self.problog_knowledge_sr.get_weights().values():
            if getattr(x, "functor", None):
                for head in model:
                    rules = model[head]['rules']
                    for i, rule in enumerate(rules):
                        if x.functor == rule['parameter_name']:
                            custom_weights[x] = rule['parameter']

        # change evidence
        evidence_dict = {}
        for var, value in evidence.iteritems():
            term = Term(var)
            if value == 1:
                evidence_dict[term] = True
            if value == 0:
                evidence_dict[term] = False

        # make inference
        try:
            res = self.problog_knowledge_sr.evaluate(
                        evidence=evidence_dict,
                        keep_evidence=False,
                        semiring=CustomSemiring(custom_weights)),
            output = {}
            for key in res[0]:
                output[str(key)] = res[0][key]
                # output = res[0]
        except InconsistentEvidenceError:
            raise InconsistentEvidenceError("""This error may have occured
                because some observation in the dataset is impossible given
                the model structure.""")
        return output


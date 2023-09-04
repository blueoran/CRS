from crsgpt.communicate.communicate import *
from crsgpt.prompter.prompter import *
from crsgpt.utils.utils import *

class Evaluator:
    def __init__(self,logger,product_types,verbose):
        self.logger=logger
        self.product_types=product_types
        self.context=[]
        self.verbose=verbose

    def evaluate(self,context,agent_response,preference,products):
        result = general_json_chat(
            self.logger,self.verbose,
            compose_messages(
                {'s':compose_system_prompts(
                    {'prompt':Prompter.SYSTEM_PROMPT},
                    {'prompt':Prompter.VALIDATE_ANSWER_PROMPT},
                    {'attribute':'Context','content':context},
                    {'attribute':'User Preference','content':preference},
                    {'attribute':'Products','content':products},
                    {'attribute':'Recommendation System Output','content':agent_response}
                )}
            ),
            "evaluate",max_tokens=400
        )
        return result

class Scorer:
    def __init__(self,logger,verbose,context):
        self.file_logger=logger
        self.verbose=verbose
        self.context=context

    def score(self):
        evaluation = general_json_chat(
            self.file_logger,self.verbose,
            compose_messages(
                {'s':compose_system_prompts(
                    {'prompt':SCORE_PROMPT},
                    {'attribute':'Conversation','content':self.context},
                )}
            ),
            "scorer"
        )
        if type(evaluation) == str:
            score=find_first_number(evaluation)
            explanation=evaluation[evaluation.find("score_explanation")+len("score_explanation: ")+3:-2]
        else:
            score = evaluation["score"]
            explanation = evaluation["score_explanation"]
        return score, explanation

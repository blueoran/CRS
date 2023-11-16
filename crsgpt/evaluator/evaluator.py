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
    def __init__(self,logger,verbose,context,explicit):
        self.file_logger=logger
        self.verbose=verbose
        self.context=context
        self.explicit=explicit

    def score(self,testcase=None):
        evaluation = general_json_chat(
            self.file_logger,self.verbose,
            compose_messages(
                {'s':compose_system_prompts(
                    {'prompt':SCORE_PROMPT_EXPLICIT if self.explicit  else SCORE_PROMPT},
                    {'attribute':'Testcase Description','content':TestPrompter.TEST_CASES[testcase][:TestPrompter.TEST_CASES[testcase].find('Example')]} if testcase is not None else {},
                    {'attribute':'Conversation','content':self.context},
                )}
            ),
            "scorer_explicit" if self.explicit is True else "scorer", model = "gpt-4"
        )
        if self.explicit:
            if type(evaluation) == str:
                score=find_first_number(evaluation)
                explanation=evaluation[evaluation.find("score_explanation")+len("score_explanation: ")+3:-2]
            else:
                score = evaluation["score"]
                explanation = evaluation["score_explanation"]
        else:
            try:
                score = evaluation["score"]
            except:
                score = evaluation
            explanation = ""
        return score, explanation

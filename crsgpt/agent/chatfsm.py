from crsgpt.communicate.communicate import *
from crsgpt.utils.utils import *

class RecommendFSM:

    def __init__(self,product_type_set,file_logger,verbose,explicit=False):
        self.file_logger=file_logger
        self.verbose=verbose
        self.explicit=explicit
        self.product_type_set = product_type_set
        self.status="Chatbot"

    def is_capable(self,context):
        capable = general_json_chat(
            self.file_logger,self.verbose,
            compose_messages(
                {'s':compose_system_prompts(
                    {'prompt':Prompter.SYSTEM_PROMPT},
                    {'prompt':Prompter.CAPABLE_RECOMMEND_RATING_PROMPT_EXPLICIT if self.explicit else Prompter.CAPABLE_RECOMMEND_RATING_PROMPT},
                    {'attribute':'Chat History','content':context[:-1]},
                    {'attribute':'User Input','content':context[-1]},
                    {'attribute':'Product Types','content':list(self.product_type_set)}
                )}
            ),
            "capable_explicit" if self.explicit else "capable","capable"
        )
        self.file_logger.info(f"capable={capable}")
        return capable=="True" or capable is True


    def recommend_fsm(self, context):
        rec_score = general_json_chat(
            self.file_logger,self.verbose,
            compose_messages(
                {'s':compose_system_prompts(
                    {'prompt':Prompter.SYSTEM_PROMPT},
                    {'prompt':Prompter.RECOMMEND_RATING_PROMPT_EXPLICIT if self.explicit else Prompter.RECOMMEND_RATING_PROMPT},
                    {'attribute':'Chat History','content':context[:-1]},
                    {'attribute':'User Input','content':context[-1]}
                )}
            ),
            "rating_recommend_explicit" if self.explicit else "rating_recommend","recommend score"
        )
        con_score = general_json_chat(
            self.file_logger,self.verbose,
            compose_messages(
                {'s':compose_system_prompts(
                    {'prompt':Prompter.SYSTEM_PROMPT},
                    {'prompt':Prompter.CONSISTENCY_RATING_PROMPT_EXPLICIT if self.explicit else Prompter.CONSISTENCY_RATING_PROMPT},
                    {'attribute':'Chat History','content':context[:-1]},
                    {'attribute':'User Input','content':context[-1]}
                )}
            ),
            "rating_consistency_explicit" if self.explicit else "rating_consistency","consistency score"
        )

        rec_score=find_first_number(rec_score)
        con_score=find_first_number(con_score)

        self.file_logger.info(f'status={self.status}')
        self.file_logger.info(f'rec_score={rec_score} {" >5" if rec_score>5 else " <=5"}')
        self.file_logger.info(f'con_score={con_score} {" >5" if con_score>5 else " <=5"}')

        if self.status == "Chatbot":
            if rec_score > 5:
                capable = self.is_capable(context)
                if capable is True:
                    self.status = "Recommend"
                else:
                    self.status = "Chatbot"
            else:
                self.status = "Chatbot"
        elif self.status == "Recommend":
            if rec_score <= 5:
                self.status = "Chatbot"
            else:
                if con_score > 5:
                    self.status = "Recommend"
                else:
                    capable = self.is_capable(context)
                    if capable is True:
                        self.status = "Recommend"
                    else:
                        self.status = "Chatbot"
        self.file_logger.info(f'status={self.status}')
        if self.verbose:
            print(f"Status = {self.status}")
        return self.status

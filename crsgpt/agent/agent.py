import openai
import logging
import json
import time
import os
import pandas as pd
from crsgpt.communicate.communicate import *
from crsgpt.evaluator.evaluator import *
from crsgpt.utils.utils import *
from crsgpt.prompter.prompter import *
from crsgpt.agent.chatfsm import *




class Agent:
    def __init__(
        self,
        products,
        preference,
        evaluator,
        file_logger,
        explicit=False,
        verbose=False
    ):
        self.file_logger=file_logger


        self.context = []
        self.start_chat = -1

        self.products=products
        self.preference=preference
        self.verbose=verbose
        self.output = ""
        self.product_selected=""

        self.explicit=explicit
        self.evaluator=evaluator
        self.resources = {'product':'','preference':''}
        self.fsm=RecommendFSM(self.products.product_type_set,self.file_logger,
                              self.verbose,self.explicit)


    def user_interactive(self,user_message):

        self.file_logger.info(f"User: {user_message}")
        start_time = time.time()
        self.context.append(f"User: {user_message}")

        prev_status = self.fsm.status
        cur_status = self.fsm.recommend_fsm(self.context)
        if cur_status == "Chatbot":
            if prev_status == "Recommend":
                self.start_chat = len(self.context)-1

            self.output = general_json_chat(
                self.file_logger,self.verbose,
                compose_messages(
                    {'s':compose_system_prompts(
                        {'prompt':Prompter.SYSTEM_PROMPT},
                        {'prompt':Prompter.CHAT_PROMPT},
                        {'attribute':'Chat History','content':self.context[:-1]},
                    )},
                    {'u':user_message}
                ),
                "chatbot","response",complete_mode=True,
                context=f'{self.context}'
            )

        elif cur_status == "Recommend":
            if prev_status == "Chatbot":
                self.context = self.context[: self.start_chat]
                self.context.append(
                    f"User: {user_message}"
                )
                self.file_logger.info(self.context)
                self.start_chat = -1

            self.preference.update_preference(self.context)
            self.resources["preference"] = self.preference.user_preference

            goal = general_json_chat(
                self.file_logger,self.verbose,
                compose_messages(
                    {'s':compose_system_prompts(
                        {'prompt':Prompter.SYSTEM_PROMPT},
                        {'prompt':Prompter.GOAL_PROMPT},
                        {'attribute':'Chat History','content':self.context[:-1]},
                        {'attribute':'User Input','content':self.context[-1]},
                    )},
                ),
                "goal","goal"
            )

            instruction = general_json_chat(
                self.file_logger,self.verbose,
                compose_messages(
                    {'s':compose_system_prompts(
                        {'prompt':Prompter.SYSTEM_PROMPT},
                        {'prompt':Prompter.ACHIEVE_PROMPT},
                        {'attribute':'Goal','content':goal}
                    )},
                ),
                "instruction","goal_instruction",strict_mode=False,max_tokens=150,
            )

            required_resource = general_json_chat(
                self.file_logger,self.verbose,
                compose_messages(
                    {'s':compose_system_prompts(
                        {'prompt':Prompter.SYSTEM_PROMPT},
                        {'prompt':Prompter.RESOURCES_PROMPT},
                        {'attribute':'Goal','content':goal},
                        {'attribute':'Instruction','content':instruction},
                        {'attribute':'User Preference','content':self.preference.user_preference},
                        {'attribute':'Chat History','content':self.context[:-1]},
                        {'attribute':'User Input','content':self.context[-1]}
                    )},
                ),
                "required_resource"
            )


            if required_resource["need_product_details"] is True:
                self.product_selected = self.products.select_products_large(goal,instruction,self.preference.user_preference,self.context)
                self.resources["product"] = self.product_selected

            if required_resource["need_user_preference"] is True:
                self.output = self.preference.ask_preference(goal,instruction,self.resources,self.context)

            else:
                result={'faults/hallucinations': True, 'feedback': '', 'suggestions': ''}
                previous_output=""
                for i in range(3):
                    if type(result) is str:
                        result = {'faults/hallucinations': "false" in result.lower(), 'feedback': result, 'suggestions': result}

                    if result['faults/hallucinations'] is False:
                        break
                    
                    self.output = general_json_chat(
                        self.file_logger,self.verbose,
                        compose_messages(
                            {'s':compose_system_prompts(
                                {'prompt':Prompter.SYSTEM_PROMPT},
                                {'prompt':Prompter.RECOMMEND_PROMPT_EXPLICIT if self.explicit else Prompter.RECOMMEND_PROMPT},
                                {'attribute':'Goal','content':goal},
                                {'attribute':'Instruction','content':instruction},
                                {'attribute':'User Preference','content':self.preference.user_preference},
                                {'attribute':'Product that you should focus on','content':self.product_selected},
                                {'attribute':'Chat History','content':self.context[:-1]},
                                {'attribute':'User Input','content':self.context[-1]},
                            )},
                            # {'u':user_message},
                            {'s':compose_system_prompts(
                                {'prompt':'**Please refer to the feedback and suggestion to the following possible output from a recommendation system to generate a better response**'},
                                {'attribute':'Possible Output','content':previous_output},
                                {'attribute':'Feedback','content':result['feedback']},
                                {'attribute':'Suggestions','content':result['suggestions']},
                            )} if result['feedback']!='' else {},
                        ),
                        "response_explicit" if self.explicit else "response","response"
                        ,max_tokens=400,complete_mode=True,
                        context=self.context
                    )

                    result=self.evaluator.evaluate(self.context,self.output,self.preference.user_preference,self.product_selected)
                    previous_output=self.output

        self.context.append(f"System: {self.output}")
        self.file_logger.info(f"System: {self.output}")
        self.file_logger.info(f"Time elapsed: {time.time()-start_time}")
        print(f"Time elapsed: {time.time()-start_time}")
        return self.output

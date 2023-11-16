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
from crsgpt.component.web import *




class Agent:
    def __init__(
        self,
        products,
        preference,
        evaluator,
        file_logger,
        explicit=False,
        verbose=False,
        test=False,
        web=False,
    ):
        self.file_logger=file_logger


        self.context = []
        self.start_chat = -1

        self.products=products
        self.preference=preference
        self.verbose=verbose
        self.test=test
        self.web=web
        if self.web:
            self.web_resource = WEB(file_logger,verbose)
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
        prev_time = start_time
        self.context.append(f"User: {user_message}")

        prev_status = self.fsm.status
        cur_status = self.fsm.recommend_fsm(self.context)
        self.file_logger.info(f"FSM Time elapsed: {time.time()-start_time}")
        prev_time = time.time()
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
            self.file_logger.info(f"Chatbot Time elapsed: {time.time()-prev_time}")

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
            
            # self.products.update_appeared_products(self.context)

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


            resource_prompt = ""
            resource_template = ""
            if self.web is True:
                if self.explicit is True:
                    resource_prompt = Prompter.RESOURCES_PROMPT_WEB_EXPLICIT
                    resource_template = "required_resource_web_explicit"
                else:
                    resource_prompt = Prompter.RESOURCES_PROMPT_WEB
                    resource_template = "required_resource_web"
            else:
                if self.explicit is True:
                    resource_prompt = Prompter.RESOURCES_PROMPT_EXPLICIT
                    resource_template = "required_resource_explicit"
                else:
                    resource_prompt = Prompter.RESOURCES_PROMPT
                    resource_template = "required_resource"
            required_resource = general_json_chat(
                self.file_logger,self.verbose,
                compose_messages(
                    {'s':compose_system_prompts(
                        {'prompt':Prompter.SYSTEM_PROMPT},
                        {'prompt':resource_prompt},
                        {'attribute':'Goal','content':goal},
                        {'attribute':'Instruction','content':instruction},
                        {'attribute':'User Preference','content':self.preference.user_preference},
                        # {'attribute':'Resources','content':self.resources},
                        {'attribute':'Chat History','content':self.context[:-1]},
                        {'attribute':'User Input','content':self.context[-1]}
                    )},
                ),
                resource_template,
                max_tokens=300 if self.explicit else 150,model="gpt-4"
            )
            self.file_logger.info(f"Basic info Time elapsed: {time.time()-prev_time}")
            prev_time = time.time()


            if required_resource.get("need_product_lists") is True:
                self.product_selected = self.products.select_products_large(goal,instruction,self.preference.user_preference,self.context)
                self.resources["product"] = self.product_selected
                self.file_logger.info(f"Product selection Time elapsed: {time.time()-prev_time}")
                prev_time = time.time()
                
            if required_resource.get("need_web_resources") is True:
                self.resources["web"] = self.web_resource.get_relevant_documents(goal,instruction,self.preference.user_preference,self.context)
                self.file_logger.info(f"Web resource Time elapsed: {time.time()-prev_time}")
                prev_time = time.time()

            if required_resource.get("need_user_preference") is True:
                self.output = self.preference.ask_preference(goal,instruction,self.resources,self.context)

            else:
                result={'faults/hallucinations': True, 'feedback': '', 'suggestions': ''}
                previous_output=""
                for i in range(1):
                # while True:
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
                                {'attribute':'Product that you should focus on','content':self.resources["product"] if required_resource.get("need_product_lists") is True else None},
                                {'attribute':'Web Resources','content':self.resources.get("web",None) if required_resource.get("need_web_resources") is True else None},
                                {'attribute':'Chat History','content':self.context[:-1]},
                                {'attribute':'User Input','content':self.context[-1]},
                            )},
                            {'s':compose_system_prompts(
                                {'prompt':'**Please refer to the feedback and suggestion to the following possible output from a recommendation system to generate a better response**'},
                                {'attribute':'Possible Output','content':previous_output},
                                {'attribute':'Feedback','content':result['feedback']},
                                {'attribute':'Suggestions','content':result['suggestions']},
                            )} if result['feedback']!='' else {},
                        ),
                        "response_explicit" if self.explicit else "response","response"
                        ,max_tokens=400,complete_mode=True,
                        context=self.context, temperature=1 if result['feedback']!='' else 0
                    )

                    # result=self.evaluator.evaluate(self.context,self.output,self.preference.user_preference,self.product_selected)
                    previous_output=self.output
                    self.file_logger.info(f"Recommendation Time elapsed: {time.time()-prev_time}")
                    
                self.products.update_past_selected_products()

        self.context.append(f"System: {self.output}")
        self.file_logger.info(f"System: {self.output}")
        self.file_logger.info(f"Time elapsed: {time.time()-start_time}")
        print(f"Time elapsed: {time.time()-start_time}")
        if self.test:
            return self.output, time.time()-start_time
        else:
            return self.output

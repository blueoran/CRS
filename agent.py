import openai
import logging
import json
from json_utilities import extract_json_from_response, validate_json, llm_response_schema
import goal
import time

resource_list=['user_preference','product_details','whole_chat_histories']

prompt_start = (
    "You mustn't recommend products that are not given in the resources. Play to your strengths as an LLM and pursue"
    " simple strategies with no legal complications."
    ""
)
PRODUCT_PROMPT="Given the following descriptions of products, please extract and summarize [three] most important key features as short and simplified as possible that for making a successful recommendation"

GOAL_PROMPT="Given the Chat History below, generate the most important step for the chatbot to formulate a response that addresses the user's needs and maintains a coherent conversation flow. Then interpret this as a goal prompt to chatgpt to generate response. Don't use 'I' in the prompt."
ACHIEVE_PROMPT="what aspect should I consider to achieve the goal of "
RESOURCES_PROMPT=f"Given the goal and the chat history provided, determine the least resources needed for you to generate a response that effectively achieves the goal and addresses the user's needs. Remember you mustn't recommend products that are not given from the database. You should first consider to make the full use of the given context, determine whether you need more information from the user side and from database, then request what you really need only among [ {', '.join(resource_list)} ] to assist in achieving the goal."
USER_PREFERENCE_PROMPT="Considering the goal and the resources required, generate a question that should ask the user to gather their preferences effectively. This question should aim to obtain crucial information aligned with the goal and enable the chatbot to provide personalized recommendations based on the user's preferences."
UPDATE_USER_PREFERENCE_PROMPT="Given the user's response to the question and the history preferences, update the user's preferences accordingly."
SUMMARY_PROMPT="Given the goal, context, and the details of the resource provided, select the key points or items that are relevant to the user's needs. Then, generate a concise summary that effectively captures the essence of the resource while aligning with the goal and maintaining coherence with the conversation."

RECOMMEND_PROMPT="Considering the context, goal, and the given resources, generate a response that provides a successful recommendation tailored to the user's needs. Ensure that the reply incorporates relevant information, aligns with the user's preferences, and maintains coherence with the conversation."



constraints = [
  '~4000 word limit for short term memory. Your short term memory is short, so immediately save important information to files.',
  'If you are unsure how you previously did something or want to recall past events, thinking about similar events will help you remember.',
  'No user assistance',
]
resources = [

]
performance_evaluations = [
  'Continuously review and analyze your actions to ensure you are performing to the best of your abilities.',
  'Constructively self-criticize your big-picture behavior constantly.',
  'Reflect on past decisions and strategies to refine your approach.',
  'Every command has a cost, so be smart and efficient. Aim to complete tasks in the least number of steps.'
]





class Agent:
    def __init__(self,product_type,products,api_key,log_file='./log/chatgpt.log',log_level=logging.DEBUG):
        self.product_type=product_type
        self.products=products
        self.product_info=self.products.apply(lambda x: ';'.join([f'{k}:{v}' for k,v in x.to_dict().items()]),axis=1)
        self.product_string="\n".join([f"Product{i}: {n}" for i,n in enumerate(self.product_info)])
        openai.api_key = api_key
        self.context=[]

        self.file_logger=logging.getLogger('file_logger')
        self.file_logger.setLevel(log_level)
        self.file_hander=logging.FileHandler(log_file)
        self.file_logger.addHandler(self.file_hander)
        self.SYSTEM_PROMPT=f"You are a Socially Intelligent {self.product_type} Recommender. Recommend items that align with the user's social attributes and preferences."# Engage in conversational interactions, gathering information about their social context, values, and interests. Personalize recommendations based on the user's social connections and feedback. Foster trust and understanding by using empathetic language and acknowledging emotions. Continuously refine recommendations to enhance user satisfaction."
        
        self.product_detail=[f'{self.products.iloc[i]["title"]}({self.products.iloc[i]["year"]}): '+str(self.json_chat(f"{self.SYSTEM_PROMPT}\n{PRODUCT_PROMPT}\n\nProduct: {product}",'product','','key_features',max_tokens=50)) for i,product in enumerate(self.product_info)]
        print(self.product_detail)
        
        self.resource_dict={i:'' for i in resource_list}
        self.resource_dict['product_details']="\n".join(self.product_detail)
        self.output=""
        self.ask_preference=False
        
        
    def chat(self,message,schema_name,**kwargs):
        self.file_logger.info(f"Chat: {message}")
        while True:
            try:
                completion = openai.ChatCompletion.create(
                messages=message,
                
                
                model=kwargs.get('model','gpt-3.5-turbo'),
                temperature=kwargs.get('temperature',0.1),
                top_p=kwargs.get('top_p',1),
                n=kwargs.get('n',1),
                stream=kwargs.get('stream',False),
                stop=kwargs.get('stop',None),
                max_tokens=kwargs.get('max_tokens',150),
                presence_penalty=kwargs.get('presence_penalty',0),
                frequency_penalty=kwargs.get('frequency_penalty',0),
                logit_bias=kwargs.get('logit_bias',{}),
                )
                break
            except BaseException as e:
                print(e)
                continue
        completion=completion["choices"][0]["message"]["content"]
        self.file_logger.info(f"Completion: {completion}")
        
        try:
            assistant_reply_json = extract_json_from_response(completion)
            validate_json(assistant_reply_json,schema_name)
        except json.JSONDecodeError as e:
            self.file_logger.error(f"Exception while validating assistant reply JSON: {e}")
            assistant_reply_json = {}
        return assistant_reply_json
        
    def json_chat(self,sys_message,schema_name,user_message,concerened_key=None,**kwargs):
        if user_message!='':
            message=[{"role":"system","content":sys_message},
                    {"role":"user","content": user_message},
                    {"role":"system","content":f"\n\nRespond with only valid JSON conforming to the following schema: \n{llm_response_schema(schema_name)}\n"}]
        else:
            message=[{"role":"system","content":sys_message},
                    {"role":"system","content":f"\n\nRespond with only valid JSON conforming to the following schema: \n{llm_response_schema(schema_name)}\n"}]
        
        
        assistant_reply_json=self.chat(message,schema_name,**kwargs)
        while assistant_reply_json=={}:
            assistant_reply_json=self.chat(message,schema_name,**kwargs)
            time.sleep(1)
        if concerened_key!=None:
            try:
                if concerened_key in assistant_reply_json.keys():
                    return str(assistant_reply_json[concerened_key])
                else:
                    return str(assistant_reply_json["properties"][concerened_key])
            except:
                return str(assistant_reply_json)
        return assistant_reply_json
    
    
    
        
    def main_loop(self):
        while True:
            start_time=time.time()
            context_string="\n".join(self.context[-6:])
            user_message=input("User: ")
            self.context.append(f"User: {user_message}")
            if user_message=='exit':
                break
            
            if self.ask_preference==True:
                self.resource_dict['user_preference']=self.json_chat(f"{self.SYSTEM_PROMPT}\n{UPDATE_USER_PREFERENCE_PROMPT}\nOriginal {'user_preference'}: {self.resource_dict['user_preference']}\n",'preference_sum',user_message,'preference_summary')
                print(f"User Preference: {self.resource_dict['user_preference']}")
                self.ask_preference=False
            
            
            goal=self.json_chat(f"{self.SYSTEM_PROMPT}\n{GOAL_PROMPT}\nChat History:\n{context_string}\n{self.context[-1]}",'goal','')['goal']
            print(f'Goal:{goal}')
            instruction=self.json_chat(f"{self.SYSTEM_PROMPT}\n{ACHIEVE_PROMPT} {goal}?",'instruction','','goal_instruction')
            print(f'Instrction:{instruction}')
            required_resource=self.json_chat(f"{self.SYSTEM_PROMPT}\n{RESOURCES_PROMPT}\nGoal:{goal}\nInstruction:{instruction}\nChat History:\n{context_string}\n{self.context[-1]}\n",'required_resource','')
            print(f'Required Resource:{required_resource}')

            # for k in required_resource.keys():
            #     if required_resource[k]==True:
            #         resource_dict[k]=self.json_chat(f"{self.SYSTEM_PROMPT}\n{SUMMARY_PROMPT}\nGoal:{goal}\nChat History:\n{context_string}\n",'resource',user_message)[k]
            
            # required_resource={k:True if k in required_resource else k:False for k in re}
            
            if required_resource['need_user_preference']==True: # need user's specified information
                
                resources='\n'.join([f"{k}:{v}" for k,v in self.resource_dict.items()])
                self.output=self.json_chat(f"{self.SYSTEM_PROMPT}\n{USER_PREFERENCE_PROMPT}\nGoal:{goal}\nInstruction:{instruction}\nResources:\n\n{resources}",'preference',user_message,'question_for_preference')
                self.ask_preference=True
            else:
                
                resources=self.resource_dict['user_preference']
                if required_resource['need_whole_chat_histories']==True:
                    whole_context_string="\n".join(self.context)
                    resources+=f'\n\nWhole Chat History:\n{whole_context_string}'
                    # self.resource_dict['whole_chat_histories']=self.json_chat(f"{self.SYSTEM_PROMPT}\n{SUMMARY_PROMPT}\nGoal:{goal}\nResources:{resources}\n\nWhole Chat History:\n{whole_context_string}\nYou should summarize the {'whole_chat_histories'}",'chat_history',user_message)['chat_history_sum']
                    
                if required_resource['need_product_details']==True:
                    resources+=f'\n\nWhole product details:\n{self.product_string}'
                    # self.resource_dict['product_details']=self.json_chat(f"{self.SYSTEM_PROMPT}\n{SUMMARY_PROMPT}\nGoal:{goal}\nResources:{resources}\n\nWhole prosuct details:{self.product_string}\nYou should summarize the {'product_details'}",'temp1',user_message)['result']
                    
                self.output=self.json_chat(f"{self.SYSTEM_PROMPT}\n{RECOMMEND_PROMPT}\nGoal:{goal}\nInstruction:{instruction}\nResources:{resources}\nContext: {context_string}",'response',user_message,'response')
            
            print(f"System: {self.output}")
            self.context.append(f"System: {self.output}")
            print(f"Time elapsed: {time.time()-start_time}")
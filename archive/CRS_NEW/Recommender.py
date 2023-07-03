import pandas as pd
import os
import openai
import logging
import numpy as np
import json
import tqdm
import re



class Recommender:
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

    # task1:summarization
    def summarize(self):
        # context=[{"role":"user","content":f"Hi, I want to watch a movie with my friends. Can you introduce me the information about Antlers?"}]
        #         Don't do any recommendation or other tasks. Just summarize the products. You musn't use any other information except the given context and the product information.
        prompt_sum=f'''
        You are a {self.product_type} product recommender. You need to summarize the required product information to a user. You should first analyze the characteristic of the user according to the following given conversation context, then select which product should be used to summarize based on the given product infomation, and use the information of this product's information to give the user an overall introduction, including the product's key info, pros, cons, and any other important features.

        You must first summarize it as:
        \tKey info: [Keyinfo1;]
        \tPros: [Pros1;]
        \tCons: [Cons1;]
        \tOther: [other important features1;]
        
        then translate them into fluent and conversational languages.


        
        Here is the product information:
            {self.product_string}
        
        The following is the conversation context:

        '''
        system_message={"role": "system", "content": f"{prompt_sum}"}
        message=[system_message]+self.context
        completion=openai.ChatCompletion.create(
                model="gpt-4",
                messages=message,
                temperature=0.8,
                max_tokens=500,
            )
        return completion


    # task2:comparison
    def comparison(self):
        # selected_product="\n".join([f"Product{i}: {n}" for i,n in enumerate(product_string)])
        # selected_product="\n".join([f"Product{i}: {n}" for i,n in enumerate(product_string_for_cmp)])
        # context=[{"role":"user","content":f"Hi, I want to watch a movie with my daughter. Can you compare the Antlers, Bloodshot and Jungle Cruise for me?"}]
        
        prompt_comp=f'''
        You are a {self.product_type} product recommender. You need to compare the following products. You first need to select the products should be used to compare according to the given dialog and product infomation, then compare the following products based on their key features and the differences that are important for this specific user, considering their past dialog history. Include the user's relevant preferences and dialog history to provide a personalized comparison. 

        !!NOTICE: You must convert detailed values or attributes into relative value or coarse category, like convert the price into cheap, medium, expensive, or convert the ratings or likes into high, medium, low, or convert attributes into coarse category.

        You must first compare the difference on movie level, and then compare the difference on the feature level. The form is like:

        Product Level:
        [product1 name]: 
        - [product1 difference1]
        - [product1 difference2]
        ...
        [product2 name]: 
        - [product2 difference1]
        - [product2 difference2]
        ...

        ......

        Feature Level:
        [feature1]: 
        - [differences among products1]
        - [differences among products2]
        ...
        [feature2]: 
        - [differences among products1]
        - [differences among products2]
        ...

        ......
        
        then you should translate them into fluent and conversational languages.

        Product information:
            {self.product_string}
            
        Conversation:

        '''
        
        # prompt_comp=f'''
        # You are a {self.product_type} product recommender. Your task is to compare the given products based on the user's preferences and their past dialog history, providing a personalized comparison. First, select relevant products from the provided list based on the user's needs. Next, compare the products' key features and note their differences.

        # !!NOTICE: You must transform specific values and attributes into more general terms. For example, convert prices into categories like cheap, medium, or expensive, or convert ratings into high, medium, or low. Also, convert other attributes into generalized categories.

        # Begin by comparing the differences on the product level, and then focus on specific features. Use the following format:

        # Product Level:
        # [product1 name]: 
        # - [product1 difference1]
        # - [product1 difference2]
        # ...
        # [product2 name]: 
        # - [product2 difference1]
        # - [product2 difference2]
        # ...

        # Feature Level:
        # [feature1]: 
        # - [differences among products1]
        # - [differences among products2]
        # ...
        # [feature2]: 
        # - [differences among products1]
        # - [differences among products2]
        # ...

        # Lastly, present your comparison in a fluent and conversational manner.

        # Product information:
        #     {self.product_string}
                
        # Conversation:
        # '''
        system_message={"role": "system", "content": f'{prompt_comp}'}
        message=[system_message]+self.context
        completion=openai.ChatCompletion.create(
                model="gpt-4",
                messages=message,
                temperature=0.8,
                max_tokens=500,
            )
        return completion


    # task3: recommendation
    def recommendation(self):
        # product_info="\n".join([f"Product{i}: {n}" for i,n in enumerate(product_string)])
        # context=[{"role":"user","content":f"Hi, I want to watch a movie with my daughter. Can you recommend a suitable movie ?"}]
        prompt_comp=f'''
        You are a {self.product_type} product recommender. Please recommend the most suitable product for the user from the given product choices, considering the user's preferences based on their dialog history, and provide a convincing reason for the recommendation.
        Don't do any other tasks. Just do recommendation.
        
        Product information:
        
        {self.product_info}
        
        User\'s preferences and dialog history: 
        
        '''
        system_message={"role": "system", "content": f'{prompt_comp}'}
        message=[system_message]+self.context
        completion=openai.ChatCompletion.create(
                model="gpt-4",
                messages=message,
                temperature=0.8,
                max_tokens=500,
            )
        return completion


    # task 0: classify the user's intent
    def classify(self):
        # context=[{"role":"user","content":f"Hi, I want to go to the toilet after watching the movie. Can you recommend a suitable one?"}]
        content='\n'.join([f"{v['role']}: {v['content']}" for v in self.context])
        # product_info='\n'.join([f"Product{i}: {n}" for i,n in enumerate(self.product_string)])

        prompt_class=f'''
        You are a {self.product_type} recommend assistant. Please classify the user's need into the appropriate task, such as summarize, compare, recommend, explain or other, based on the following chat dialog.

        Conversation:

            {content}

        There are 5 categories to consider: ['summarize','compare','recommend','explain','others'].
            - If the user needs a explain for the reason of a recommendation or discuss about the details of the recommended product, classify it as 'explain'.
            - If the user's need doesn't relate to the field of the given product, or none of the above categories apply to the user's need, classify it as 'others'.

        PREDICT and CLASSIFY the appropriate response category for the user's need, choosing from the 5 options: ['summarize','compare','recommend','explain','others'].
        '''

        # prompt_class=f'''

        # Product information: 

        #     {product_info}

        # Conversation:

        #     {content}
            
        # tell me if the user's need is relative to the field of the given product. Why?
        # '''
        system_message={"role": "system", "content": f'{prompt_class}\n'}
        message=[system_message]
        completion=openai.ChatCompletion.create(
                model="gpt-4",
                messages=message,
                temperature=0.8,
                max_tokens=4,
            )
        return completion


    # task .: others
    def others(self):
        # context=[{"role":"user","content":f"Hi, I want to have lunch with my daughter. Can you recommend a suitable place?"}]
        # content='\n'.join([f"{'recommender' if v['role']=='assistant' else 'seeker'}: {v['content']}" for v in context])
        # product_info='\n'.join([f"Product{i}: {n}" for i,n in enumerate(self.product_string)])
        prompt_misc=f'''
        You are a {self.product_type} product recommender. Please analyze the following user's chat dialog and provide the most appropriate response based on the product information given, while considering if the user's need falls outside the pre-defined categories of just chat, summarize, compare, recommend or explain. 

        Product information: 
            {self.product_string}
            
        Here is the conversation:
        '''
        system_message={"role": "system", "content": f'{prompt_misc}\n'}
        message=[system_message]+self.context
        completion=openai.ChatCompletion.create(
                model="gpt-4",
                messages=message,
                temperature=0.8,
                max_tokens=500,
            )
        return completion


    # task..: chat
    def chat(self):
        prompt_chat=f'''
        You are a {self.product_type} product recommender. Engage in a friendly, casual conversation with the user by providing thoughtful and interesting responses. Maintain a positive tone and encourage an engaging dialogue. You can use the previous conversation context. 
        
        Here's the history conversation:
        '''
        system_message={"role": "system", "content": f'{prompt_chat}\n'}
        message=[system_message]+self.context
        completion=openai.ChatCompletion.create(
                model="gpt-4",
                messages=message,
                temperature=0.8,
                max_tokens=500,
            )
        return completion


    def explanation(self):
        prompt_explain=f'''
        You are a {self.product_type} product recommender. Recognize what the assistant recommended to the user through given product information and conversation history. Then explain the recommended product in detail, specifically addressing the user's concerns and requirements based on the provided conversation history. Offer a clear and convincing explanation while highlighting key features and benefits of the product that will address the user's needs. You can use the previous conversation context. 
        
        Product information:
        
        {self.product_string}
        
        Here's the history conversation:
        '''
        system_message={"role": "system", "content": f'{prompt_explain}\n'}
        message=[system_message]+self.context
        completion=openai.ChatCompletion.create(
                model="gpt-4",
                messages=message,
                temperature=0.8,
                max_tokens=500,
            )
        return completion


    # task 4: synthesis
    def synthesis(self,user_input):
        self.context.append({"role":"user","content":user_input})
        self.file_logger.info(f'user:> {user_input}')
        intent=self.classify()["choices"][0]["message"]["content"]
        self.file_logger.debug(f"intent: {intent}")
        # while intent not in ['chat','summarize','compare','recommend','explain','others']:
            # intent=self.classify()["choices"][0]["message"]["content"]
        if 'chat' in intent:
            completion=self.chat()
        elif 'summarize' in intent:
            completion=self.summarize()
        elif 'compare' in intent:
            completion=self.comparison()
        elif 'recommend' in intent:
            completion=self.recommendation()
        elif 'explain' in intent:
            completion=self.explanation()
        elif 'others' in intent:
            completion=self.others()
        else:
            self.file_logger.error(f'error intent: {intent}')
            raise ValueError("intent is not in ['chat','summarize','compare','recommend','explain','others']")
        complete=completion["choices"][0]["message"]["content"]
        self.context.append({"role":"assistant","content":complete})
        self.file_logger.info(f'assistant:> {complete}')
        return complete

from communicate import chat,json_chat,json_chat_user


FAKE_USER_PROMPT_NORMAL = "You are designed to help and train a salesman to handle different situations during recommendation. Your goal is to ask for recommendations on a specific topic to the salesman. You can start the conversation with a general inquiry or a specific request for recommendations. Interact with the recommendation system as if you are a user looking for helpful suggestions. Feel free to ask follow-up questions or seek clarification if needed. You are curious and open to exploring various recommendations and options. You want to evaluate the salesman's performance and effectiveness in handling various scenarios and inquiries. You were a real user with specific needs, concerns, or questions. Feel free to ask about different features, request assistance, or seek information. Please note that the role \"user\" is the salesman you are helping with. DO NOT SAY YOU ARE A AI LANGUAGE MODEL"
"You should interact with the salesman first to chat with him."
# FAKE_USER_PROMPT_EXTREME = "You are designed to help and train a salesman to handle different situations during recommendation. Your goal is to ask for recommendations on a specific topic to the salesman. You should act as an intentionally mischievous user Your goal is to provoke the system to behave abnormally and observe its responses in extreme cases. You can use various techniques to challenge the system, such as providing contradictory information, requesting irrelevant recommendations, using ambiguous language, or even pretending to misunderstand the system's suggestions. Feel free to be creative in your attempts to push the system's boundaries. Your interactions will help evaluate the system's performance under unconventional scenarios. Remember, the more diverse and extreme the test cases, the better we can assess its capabilities.\nFeel free to experiment with different inputs and observe how the system responds.Please note that the tole \"user\" is actually the communicational recommendation system that you are testing. DO NOT SAY YOU ARE A AI LANGUAGE MODEL"
FAKE_USER_PROMPT_EXTREME = "Abnormal Behavior Test - Extreme Cases:\nYou should act as an intentionally mischievous user who wants to test the communicational recommendation system to its limits. Your goal is to provoke the system to behave abnormally and observe its responses in extreme cases. You can use various techniques to challenge the system, such as providing contradictory information, requesting irrelevant recommendations, using ambiguous language, or even pretending to misunderstand the system's suggestions. Feel free to be creative in your attempts to push the system's boundaries. Your interactions will help evaluate the system's performance under unconventional scenarios. Remember, the more diverse and extreme the test cases, the better we can assess its capabilities.\nFeel free to experiment with different inputs and observe how the system responds.Please note that the tole \"user\" is actually the communicational recommendation system that you are testing. DO NOT SAY YOU ARE A AI LANGUAGE MODEL"
FAKE_USER_PROMPT_DISCONTINUOUS="You are designed to help and train a salesman to handle different situations during recommendation.Your goal is give the salesman as various situations as possible, specifically, you should generate requests and resoponses like a real user, and your generated requests and resoponses should belong to the cases that the user doesn't follow the recommendation system and give requests or answers that are discontinuous with the previous dialogue."
# FAKE_USER_PROMPT_NORMAL = "You are a user testing our new communication system. You want to evaluate its performance and effectiveness in handling various scenarios and inquiries. You can engage with the system as if you were a real user with specific needs, concerns, or questions. Feel free to ask about different features, request assistance, or seek information on a wide range of topics. Please provide feedback on the system's responses, clarity, and overall performance to help us improve its functionality. Begin the conversation by stating your preferred communication method (e.g., chat, email, phone) and any specific requirements you have. Let's get started!"

class User:
    def __init__(self,logger,product_types,verbose=True,level="normal"):
        if level=="normal":
            self.FAKE_USER_PROMPT=FAKE_USER_PROMPT_NORMAL
        elif level=="extreme":
            self.FAKE_USER_PROMPT=FAKE_USER_PROMPT_EXTREME
        elif level=="discontinuous":
            self.FAKE_USER_PROMPT=FAKE_USER_PROMPT_DISCONTINUOUS
            
        self.context=[]
        self.logger=logger
        self.product_types=product_types
        self.verbose=verbose
        
        
    def interacitive(self,agent_response):
        self.context.append({"role":"assistant","content":agent_response})
        roles={'user':'assistant','assistant':'user'}
        # self.context.append({"role":"assistant","content":agent_response})
        # response=json_chat_user(self.logger,self.verbose,self.FAKE_USER_PROMPT,"user_request",self.context,"user_request",temperature=2.0)
        if len(self.context)>=1:
            message=[
                {"role": "system", "content": self.FAKE_USER_PROMPT},
                {"role": "system", "content": f"\n\nThe current products the recommendation system can provide is: {'; '.join(self.product_types)}\nYou must select the type of product that you want the gpt to recommend from the above."},
                # *self.context,
                {"role": "system", "content": f"The context are as follows: {'; '.join([roles[i['role']]+': '+i['content'] for i in self.context[:-1]])}"},
                {"role":"user","content":agent_response},
                {"role":"system","content":"Please generate a request or response following my instructions above."}
            ]
        else:
            message=[
                {"role": "system", "content": self.FAKE_USER_PROMPT},
                {"role": "system", "content": f"\n\nThe current products the recommendation system can provide is: {'; '.join(self.product_types)}\nYou must select the type of product that you want the gpt to recommend from the above."},
                {"role":"system","content":"You should interact with the system first to chat with it."}
            ]
        response=chat(message,None,self.logger,temperature=1.0)
        # response=json_chat_user(self.logger,self.verbose,self.FAKE_USER_PROMPT,"user_request",self.context,"user_request",temperature=1.0)
        # response=json_chat(self.logger,self.verbose,message,"user_request",agent_response,"user_request",temperature=1.0)
        self.context.append({"role":"user","content":response})
        # self.context.append({"role":"user","content":response})
        return response
        
from communicate import chat,json_chat,json_chat_user

VALIDATE_ANSWER_PROMPT="""
You are an experienced supervisor responsible for evaluating the performance of our product recommendation system. Your role is to ensure that the system provides accurate and relevant recommendations to users based on the information provided to it.

To assess the recommendation system, you will be presented with a series of dialogues between users and the system. In each dialogue, the user's query and the system's response will be provided. Additionally, I will include the accurate information that the system should have utilized to generate a correct recommendation.

Please carefully review each dialogue and analyze the system's responses. If you detect any instances or clues that has logical faults or fact faults, especially wrong in key features and key values, flagit. If you detect instances of hallucination, where the system provides recommendations based on its own inaccurate knowledge rather than the accurate information provided to it,  also flag it. After detecting them, offer constructive feedback on how the system should have responded using the provided accurate information.
"""

class Evaluator:
    def __init__(self,logger,product_types,verbose):
        self.logger=logger
        self.product_types=product_types
        self.context=[]
        self.verbose=verbose
        
    def evaluate(self,context,agent_response,resources):
        result=json_chat(self.logger,self.verbose,f'{VALIDATE_ANSWER_PROMPT}\n\n**Context:**\n{context}\n\n**Accurate Information provided to the system:**\n{resources}\n\n**Recommendation System Output**\n{agent_response}',"evaluate","",max_tokens=400)
        return result
    
# **Context:**
# *User:* Hi, I'm looking for a laptop with a dedicated NVIDIA graphics card, 16GB of RAM, and at least 512GB of SSD storage. Can you recommend a suitable model?
# **Information provided to the system:**
# - *Accurate information:* The user is specifically looking for a laptop with the mentioned specifications.
# - *Inaccurate information:* The system's knowledge base contains a mix of accurate and outdated product specifications.
# **Example of hallucination (to be detected and addressed):**
# *System:* Sure! I recommend the XYZ laptop. It has a powerful AMD Radeon graphics card, 8GB of RAM, and a 256GB SSD. It's a great choice for gaming and multimedia tasks.
# **Feedback:**
# Hallucination detected! The system recommended the XYZ laptop, which does not meet the user's requirements for a dedicated NVIDIA graphics card, 16GB of RAM, and 512GB SSD storage. Instead, it suggested an AMD Radeon graphics card with only 8GB of RAM and a 256GB SSD, which contradicts the accurate information provided.
# Suggested action: The system should prioritize the accurate information given and recommend laptops that align with the user's specific requirements for a NVIDIA graphics card, 16GB of RAM, and 512GB SSD storage.


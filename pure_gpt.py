from communicate import chat,json_chat,json_chat_user



class PureGPT:
    def __init__(self,logger):
        self.context=[]
        self.logger=logger
        
    def interacitive(self,context):
        response=chat(context,None,self.logger,temperature=1.0)
        return response
        
from crsgpt.communicate.communicate import *


class PureGPT:
    def __init__(self,logger):
        self.context=[]
        self.logger=logger
        
    def interactive(self,context):
        response=chat(context,None,self.logger,temperature=1.0)
        return response['val']
        
class ProductGPT:
    def __init__(
        self,
        products,
        file_logger,
        explicit=False,
        verbose=False
    ):
        self.file_logger=file_logger

        self.products=products
        self.verbose=verbose
        self.output = ""
        self.product_selected=""
        self.explicit=explicit

    def user_interactive(self,context):
        self.product_selected = self.products.select_products_large("Find products that can help to satisfy the user's request",None,None,context)
        self.output = general_json_chat(
            self.file_logger,self.verbose,
            compose_messages(
                {'s':compose_system_prompts(
                    {'prompt':Prompter.SYSTEM_PROMPT},
                    {'prompt':Prompter.RECOMMEND_PROMPT_EXPLICIT if self.explicit else Prompter.RECOMMEND_PROMPT},
                    {'attribute':'Product that you should focus on','content':self.product_selected},
                    {'attribute':'Chat History','content':context[:-1]},
                    {'attribute':'User Input','content':context[-1]},
                )},
            ),
            "response_explicit" if self.explicit else "response","response"
            ,max_tokens=400,complete_mode=True,
            context=context
        )
        return self.output


from crsgpt.communicate.communicate import *

class Preference:
    def __init__(self,file_logger,verbose):
        self.user_preference = ""
        self.current_focus=""
        self.should_update=False
        self.file_logger=file_logger
        self.verbose=verbose

    def ask_preference(self,goal,instruction,resources,context):
        question = general_json_chat(
            self.file_logger,self.verbose,
            compose_messages(
                {'s':compose_system_prompts(
                    {'prompt':Prompter.SYSTEM_PROMPT},
                    {'prompt':Prompter.USER_PREFERENCE_PROMPT},
                    {'attribute':'Goal','content':goal},
                    {'attribute':'Instruction','content':instruction},
                    {'attribute':'Resources','content':resources},
                    {'attribute':'Chat History','content':context[:-1]},
                    {'attribute':'User Input','content':context[-1]},
                )}
            ),
            "preference",concerened_key="question_for_preference"
        )

        return question

    def update_preference(self,context):
        ori_preference = self.user_preference
        preference_result=general_json_chat(
            self.file_logger,self.verbose,
            compose_messages(
                {'s':compose_system_prompts(
                    {'prompt':Prompter.SYSTEM_PROMPT},
                    {'prompt':Prompter.UPDATE_USER_PREFERENCE_PROMPT},
                    {'attribute':'Chat History','content':context[:-1]},
                    {'attribute':'User Input','content':context[-1]},
                    {'attribute':'User\'s Former Preference','content':self.user_preference}
                )}
            ),
            "preference_sum",
            strict_mode=False,
        )

        self.user_preference = preference_result.get('preference_summary',"")
        if self.verbose:
            print(f"Update User Preference: {ori_preference} -> {self.user_preference}")
        return self.user_preference

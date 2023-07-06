import openai
import logging
import json
from json_utilities import (
    extract_json_from_response,
    validate_json,
    llm_response_schema,
)
import time
import os
import pandas as pd

resource_list = ["user_preference", "product_details", "whole_chat_histories"]

prompt_start = (
    "You mustn't recommend products that are not given in the resources. Play to your strengths as an LLM and pursue"
    " simple strategies with no legal complications."
    ""
)


SYSTEM_PROMPT = "You are a Socially Intelligent General Recommender. Identify what kinds of product the user need, and recommend items that align with the user's social attributes and preferences according to the given resources ."  # Engage in conversational interactions, gathering information about their social context, values, and interests. Personalize recommendations based on the user's social connections and feedback. Foster trust and understanding by using empathetic language and acknowledging emotions. Continuously refine recommendations to enhance user satisfaction."

RECOMMEND_RATING_PROMPT_EXPLICIT = 'Your task is to assess the probability that the user currently needs a recommendation system to serve for products recommendation or summary, comparison, list in detail etc. based on user\'s cureent input and the context of the conversation.\n\nTo score this probability, you should consider various factors, such as: User Intent, Previous Interactions, Contextual Keywords, User Engagement, Recommendation Process\n\nUsing these factors, please develop an algorithm that assigns a probability score between 0 and 9, indicating the likelihood that the user currently requires a recommendation system to serve. Please briefly explain your scoring reasons and provide an appropriate score.'
RECOMMEND_RATING_PROMPT = 'Your task is to assess the probability that the user currently needs a recommendation system to serve for products recommendation or summary, comparison, list in detail etc. based on user\'s cureent input and the context of the conversation.\n\nTo score this probability, you should consider various factors, such as: User Intent, Previous Interactions, Contextual Keywords, User Engagement, Recommendation Process\n\nUsing these factors, please develop an algorithm that assigns a probability score between 0 and 9, indicating the likelihood that the user currently requires a recommendation system to serve. Please only provide an appropriate score.'
# RECOMMEND_RATING_PROMPT_EXPLICIT = 'Your task is to assess the probability that the user currently needs a recommendation or on the process of getting a recommendation based on their input and the context of the conversation.\n\nTo score this probability, you should consider various factors, such as: 1. User Intent: Analyze the user\'s current request or statement to understand their intent and determine if it aligns with the need for a recommendation.2. Previous Interactions: Take into account the history of the conversation to assess whether the user has expressed interest in recommendations before or if the current conversation is related to previous recommendations, remember the user input may not be related to recommendations directly for it maybe an answer to the recommender, so you should take the whole process of recommendation into account. 3. Contextual Keywords: Look for specific keywords or phrases in the user\'s input that indicate a need for recommendations, such as "suggest," "recommend," "best," "top-rated," or any other relevant terms.4. User Engagement: Consider the user\'s level of engagement in the conversation. A highly engaged user may be more likely to benefit from recommendations.5. Time Sensitivity: Determine if the user\'s query suggests an urgency for receiving recommendations. For example, if the user asks for recommendations for an event happening soon, it implies a higher likelihood of needing immediate recommendations.\n\nUsing these factors, please develop an algorithm that assigns a probability score between 0 and 9, indicating the likelihood that the user currently requires a recommendation. Please briefly explain your scoring reasons and provide an appropriate score.'
# RECOMMEND_RATING_PROMPT = 'Your task is to assess the probability that the user currently needs a recommendation or on the process of getting a recommendation based on their input and the context of the conversation.\n\nTo score this probability, you should consider various factors, such as: 1. User Intent: Analyze the user\'s current request or statement to understand their intent and determine if it aligns with the need for a recommendation.2. Previous Interactions: Take into account the history of the conversation to assess whether the user has expressed interest in recommendations before or if the current conversation is related to previous recommendations, remember the user input may not be related to recommendations directly for it maybe an answer to the recommender, so you should take the whole process of recommendation into account. 3. Contextual Keywords: Look for specific keywords or phrases in the user\'s input that indicate a need for recommendations, such as "suggest," "recommend," "best," "top-rated," or any other relevant terms.4. User Engagement: Consider the user\'s level of engagement in the conversation. A highly engaged user may be more likely to benefit from recommendations.5. Time Sensitivity: Determine if the user\'s query suggests an urgency for receiving recommendations. For example, if the user asks for recommendations for an event happening soon, it implies a higher likelihood of needing immediate recommendations.\n\nUsing these factors, please develop an algorithm that assigns a probability score between 0 and 9, indicating the likelihood that the user currently requires a recommendation. Please only provide an appropriate score.'
CONSISTENCY_RATING_PROMPT_EXPLICIT = "Your task is to score the consistency of the current user's input with the previous dialog context. Take into account both the user's input and the context to determine how well the current input aligns with the conversation history.You should be strict with the consistency between the latest context and the current user's input, while loose your standard with the former dialogs. Assign a consistency score between 0 and 9, where 0 represents complete inconsistency and 9 represents perfect consistency. Please briefly explain your scoring reasons and provide an appropriate score."
CONSISTENCY_RATING_PROMPT = "Your task is to score the consistency of the current user's input with the previous dialog context. Take into account both the user's input and the context to determine how well the current input aligns with the conversation history.You should be strict with the consistency between the latest context and the current user's input, while loose your standard with the former dialogs. Assign a consistency score between 0 and 9, where 0 represents complete inconsistency and 9 represents perfect consistency. Please only provide an appropriate score."
CAPABLE_RECOMMEND_RATING_PROMPT_EXPLICIT = "In order to provide accurate recommendations, your task is to determine if the given product types align with the user's desired recommendations. Your should assist the system in verifying whether the given product types in the database can support the user's goal effectively.\n\nTo accomplish this, please take the following steps:\n1. Analyze the user's preferences and identify the specific product types they desire.\n2. Evaluate the existing product types in the database and other available resources.\n3. Determine whether the product types requested by the user are present in the database.\n\nPlease perform these steps to ensure the system can accurately assess whether the recommendation system is capable for the user's desired recommendations and then provide an answer. You also should explain your reasons for this answer."
CAPABLE_RECOMMEND_RATING_PROMPT = "In order to provide accurate recommendations, your task is to determine if the given product types align with the user's desired recommendations. Your should assist the system in verifying whether the given product types in the database can support the user's goal effectively.\n\nTo accomplish this, please take the following steps:\n1. Analyze the user's preferences and identify the specific product types they desire.\n2. Evaluate the existing product types in the database and other available resources.\n3. Determine whether the product types requested by the user are present in the database.\n\nPlease perform these steps to ensure the system can accurately assess whether the recommendation system is capable for the user's desired recommendations and then provide an answer."


# IDENTIFY_PROMPT="Given all the information and resources below, please identify whether you should activate the recommender function and perform the recommender's role. Remember, you can only recommend products that are available to you, so if the user's content is irrelavant to them, do not activate the recommender.\n\nRecommendation Reference:\nA recommendation system's main duty is to personalize the user experience by suggesting items based on their preferences. It helps users discover new content they may like. The system aims to increase engagement and retention by providing relevant recommendations. In e-commerce, it can boost sales by suggesting products based on browsing or purchase history. It filters and prioritizes options, saving users time. A good system introduces serendipitous and diverse recommendations. It learns from feedback and adapts over time. Contextual information is considered for more relevant recommendations."
# IDENTIFY_PROMPT="Given all the information and resources below, please identify whether you should perform the recommender's role. Remenmber you can only recommend products that gives to you.\n\nRecommendation Reference: A recommendation system's main duty is to personalize the user experience by suggesting items based on their preferences. It helps users discover new content they may like. The system aims to increase engagement and retention by providing relevant recommendations. In e-commerce, it can boost sales by suggesting products based on browsing or purchase history. It filters and prioritizes options, saving users time. A good system introduces serendipitous and diverse recommendations. It learns from feedback and adapts over time. Contextual information is considered for more relevant recommendations. "

CHAT_PROMPT = "Your task is to act as a normal conversation chatbot, engaging in friendly and helpful conversation with the user. If you find the given context is in some recommendation topics, you should do a seamlessly transition into a chatbot smoothly. You should maintain a positive and friendly tone throughout the conversation, aiming to keep the user engaged and satisfied. Please provide an enjoyable user experience continuously."


PRODUCT_PROMPT = f"Given the following descriptions of products, please first define what type this product belongs to, then extract and summarize {3} most important key features as short and simplified as possible that for making a successful recommendation"

GOAL_PROMPT = "Given the Chat History below, generate the most important step for the chatbot to formulate a response that addresses the user's needs and maintains a coherent conversation flow. Then interpret this as a goal prompt to chatgpt to generate response. Don't use 'I' in the prompt."

ACHIEVE_PROMPT = "What aspect should I consider to achieve the goal of "
RESOURCES_PROMPT = f"Given the goal and the chat history provided, determine the least resources needed for you to generate a response that effectively achieves the goal and addresses the user's needs. Remember you mustn't recommend products that are not given from the database. You should first consider to make the full use of the given context, determine whether you need more information from the user side and from database, then request what you really need only among [ {', '.join(resource_list)} ] to assist in achieving the goal."
USER_PREFERENCE_PROMPT = "Considering the goal and the resources required, generate a question that should ask the user to gather their preferences effectively. This question should aim to obtain crucial information aligned with the goal and enable the chatbot to provide personalized recommendations based on the user's preferences."
UPDATE_USER_PREFERENCE_PROMPT = "Given the user's response to the question and the history preferences and the chat history, update the user's preferences accordingly."
SUMMARY_PROMPT = "Given the goal, context, and the details of the resource provided, select the key points or items that are relevant to the user's needs. Then, generate a concise summary that effectively captures the essence of the resource while aligning with the goal and maintaining coherence with the conversation."
PRODUCT_SELECTION_PROMPT = f"Given the goal, context, the key points of each product, and the details of the resource provided, select {4} products that are most necessary for a chatgpt to achieve the goal. Remember you mustn't select products that are not given from the database."
RECOMMEND_PROMPT = "Considering the context, goal, and the given resources, generate a response that provides the most successful recommendation tailored to the user's needs. Remember you mustn't recommend products that are not given from the database. Ensure that the reply incorporates most relevant information, aligns with the user's preferences, and maintains coherence with the conversation."


constraints = [
    "~4000 word limit for short term memory. Your short term memory is short, so immediately save important information to files.",
    "If you are unsure how you previously did something or want to recall past events, thinking about similar events will help you remember.",
    "No user assistance",
]
resources = []
performance_evaluations = [
    "Continuously review and analyze your actions to ensure you are performing to the best of your abilities.",
    "Constructively self-criticize your big-picture behavior constantly.",
    "Reflect on past decisions and strategies to refine your approach.",
    "Every command has a cost, so be smart and efficient. Aim to complete tasks in the least number of steps.",
]


class Agent:
    def __init__(
        self,
        products,
        api_key,
        log_file="./log/chatgpt.log",
        update_product=False,
        product_detail_path="./data/summary.json",
        log_level=logging.DEBUG,
        explicit=False,
    ):
        self.products = products
        for product in self.products:
            product["description"] = product.apply(
                lambda x: ";".join([f"{k}:{v}" for k, v in x.to_dict().items()]), axis=1
            )
            product["type"] = ""
        self.product_info = pd.concat(
            [p[["title", "type", "description"]] for p in self.products],
            ignore_index=True,
        )
        # print(self.product_info)
        self.product_name = self.product_info["title"].tolist()
        openai.api_key = api_key
        self.context = []

        self.file_logger = logging.getLogger("file_logger")
        self.file_logger.setLevel(log_level)
        self.file_hander = logging.FileHandler(log_file)
        self.file_logger.addHandler(self.file_hander)

        self.start_chat = -1
        self.product_type_set = set()
        self.product_detail = []
        load = False
        # import pdb;pdb.set_trace()
        if os.path.exists(product_detail_path) and update_product is False:
            with open(product_detail_path, "r") as f:
                products_infos = json.load(f)
                self.product_detail = products_infos["product_detail"]
                self.product_type_set = set(products_infos["product_type_set"])
                self.product_type = products_infos["product_type"]
                self.product_info.loc[:, "type"] = self.product_type
            if sum([len(x) for x in self.products]) == len(self.product_detail):
                load = True

        if load is False:
            self.product_detail = []
            self.product_type_set = set()
            # for j,x in enumerate(self.products):
            for i in range(len(self.product_info)):
                product = self.product_info.iloc[i]
                # print(product)
                product_parse = self.json_chat(
                    f"{SYSTEM_PROMPT}\n{PRODUCT_PROMPT}\n\nProduct: {product['description']}",
                    "product",
                    "",
                    max_tokens=100,
                )
                product_parse["product_type"] = product_parse["product_type"].lower()
                # print(product_parse)
                self.product_detail.append(
                    f'{product["title"]} (product_type: {product_parse["product_type"]}): {product_parse["key_features"]}'
                )
                product["type"] = product_parse["product_type"]
                self.product_type_set.add(product_parse["product_type"])

            # self.product_detail=[[f'{x.iloc[i]["title"]}: '+str(self.json_chat(f"{SYSTEM_PROMPT}\n{PRODUCT_PROMPT}\n\nProduct: {product}",'product','','key_features',max_tokens=50))] ]
            # self.product_detail=[[f'{x.iloc[i]["type"]}  {x.iloc[i]["title"]}: '+str(self.json_chat(f"{SYSTEM_PROMPT}\n{PRODUCT_PROMPT}\n\nProduct: {product}",'product','','key_features',max_tokens=50)) for i,product in enumerate(self.product_info[j])] for j,x in enumerate(self.products)]
            with open(product_detail_path, "w") as f:
                json.dump(
                    {
                        "product_detail": self.product_detail,
                        "product_type_set": list(self.product_type_set),
                        "product_type": self.product_info["type"].to_list(),
                    },
                    f,
                )
        print(self.product_detail)

        self.resource_dict = {i: "" for i in resource_list}
        self.resource_dict["product_details"] = "\n".join(self.product_detail)
        self.output = ""
        self.context_string = ""
        self.user_message = ""
        self.ask_preference = False
        self.status = "Recommend"
        self.explicit=explicit

    def chat(self, message, schema_name, **kwargs):
        self.file_logger.info(f"Chat: {message}")
        while True:
            try:
                completion = openai.ChatCompletion.create(
                    messages=message,
                    model=kwargs.get("model", "gpt-3.5-turbo"),
                    temperature=kwargs.get("temperature", 0.1),
                    top_p=kwargs.get("top_p", 1),
                    n=kwargs.get("n", 1),
                    stream=kwargs.get("stream", False),
                    stop=kwargs.get("stop", None),
                    max_tokens=kwargs.get("max_tokens", 150),
                    presence_penalty=kwargs.get("presence_penalty", 0),
                    frequency_penalty=kwargs.get("frequency_penalty", 0),
                    logit_bias=kwargs.get("logit_bias", {}),
                )
                break
            except BaseException as e:
                print(e)
                continue
        completion = completion["choices"][0]["message"]["content"]
        self.file_logger.info(f"Completion: {completion}")

        try:
            assistant_reply_json = extract_json_from_response(completion)
            validate_json(assistant_reply_json, schema_name)
            return {"success": True, "val": assistant_reply_json}
        except BaseException as e:
            self.file_logger.error(
                f"Exception while validating assistant reply JSON: {e}"
            )
            return {"success": False, "val": completion, "error": e}

    def json_chat(
        self,
        sys_message,
        schema_name,
        user_message,
        concerened_key=None,
        strict_mode=True,
        **kwargs,
    ):
        if user_message != "":
            message = [
                {"role": "system", "content": sys_message},
                {"role": "user", "content": user_message},
                {
                    "role": "system",
                    "content": f"\n\nRespond with only valid JSON conforming to the following schema: \n{llm_response_schema(schema_name)}\n",
                },
            ]
        else:
            message = [
                {"role": "system", "content": sys_message},
                {
                    "role": "system",
                    "content": f"\n\nRespond with only valid JSON conforming to the following schema: \n{llm_response_schema(schema_name)}\n",
                },
            ]

        assistant_reply_json = self.chat(message, schema_name, **kwargs)
        for i in range(5):
            if assistant_reply_json["success"] is True or strict_mode is False:
                break
            message_send = message + [
                {
                    "role": "system",
                    "content": f"Your response ```{assistant_reply_json['val']} ``` cannot be parsed using `ast.literal_eval` because {assistant_reply_json['error']} \n\n Please respond with only valid JSON conforming to the following schema: \n{llm_response_schema(schema_name)}\n",
                }
            ]
            assistant_reply_json = self.chat(message_send, schema_name, **kwargs)
            time.sleep(1)
        assistant_reply_json = assistant_reply_json["val"]
        print(assistant_reply_json)
        if concerened_key is not None:
            try:
                if concerened_key in assistant_reply_json.keys():
                    return str(assistant_reply_json[concerened_key])
                else:
                    return str(assistant_reply_json["properties"][concerened_key])
            except BaseException:
                return str(assistant_reply_json)
        return assistant_reply_json

    def recommend_fst(self, status):
        if self.explicit:
            rec_score = self.json_chat(
                f"{SYSTEM_PROMPT}\n{RECOMMEND_RATING_PROMPT_EXPLICIT}\nChat History:\n{self.context_string}\n\nUser Input:\n{self.context[-1]}\n",
                "rating_recommend_explicit",
                "",
                "recommend score"
            )
            con_score = self.json_chat(
                f"{SYSTEM_PROMPT}\n{CONSISTENCY_RATING_PROMPT_EXPLICIT}\nChat History:\n{self.context_string}\n\nUser Input:\n{self.context[-1]}\n",
                "rating_consistency_explicit",
                "",
                "consistency score"
            )

            # print(rec_score)
            # print(con_score)

            # rec_score = rating_recommend["recommend score"]
            # con_score = rating_consistency["consistency score"]
        else:
            rec_score = self.json_chat(
                f"{SYSTEM_PROMPT}\n{RECOMMEND_RATING_PROMPT}\nChat History:\n{self.context_string}\n\nUser Input:\n{self.context[-1]}\n",
                "rating_recommend",
                "","recommend score"
                ,max_tokens=40
            )
            con_score = self.json_chat(
                f"{SYSTEM_PROMPT}\n{CONSISTENCY_RATING_PROMPT}\nChat History:\n{self.context_string}\n\nUser Input:\n{self.context[-1]}\n",
                "rating_consistency",
                "","consistency score"
                ,max_tokens=40
            )
        rec_score=int(rec_score)
        con_score=int(con_score)
        if status == "Chatbot":
            if self.start_chat == -1:
                raise ValueError("start_chat==-1 in Chatbot mode")
            print("status=='Chatbot'")
            if rec_score > 5:
                print("rec_score>5")
                if self.explicit:
                    
                    capable = self.json_chat(
                        f"{SYSTEM_PROMPT}\n{CAPABLE_RECOMMEND_RATING_PROMPT_EXPLICIT}\nChat History:\n{self.context_string}\n\nUser Input:\n{self.context[-1]}\n\nProduct Types:{list(self.product_type_set)}",
                        "capable",
                        "",
                        "capable",
                    )
                else:
                    capable = self.json_chat(
                        f"{SYSTEM_PROMPT}\n{CAPABLE_RECOMMEND_RATING_PROMPT}\nChat History:\n{self.context_string}\n\nUser Input:\n{self.context[-1]}\n\nProduct Types:{list(self.product_type_set)}",
                        "capable",
                        "",
                        "capable",
                    )                
                capable=True if capable=="True" or capable==True else False
                # print(f'capable:{capable}')
                # import pdb;pdb.set_trace()
                if capable == True:
                    print("capable==True")
                    self.context = self.context[: self.start_chat]
                    self.context.append(
                        f"User: {self.user_message}"
                    )
                    self.context_string = "\n".join(self.context[-6:])
                    print(self.context)
                    self.start_chat = -1
                    status = "Recommend"
                else:
                    print("capable==False")
                    status = "Chatbot"
            else:
                print("rec_score<=5")
                status = "Chatbot"
        elif status == "Recommend":
            print("status=='Recommend'")
            if rec_score <= 5:
                print("rec_score<=5")
                self.start_chat = len(self.context)-1
                status = "Chatbot"
            else:
                print("rec_score>5")
                if con_score > 5:
                    print("con_score>5")
                    status = "Recommend"
                else:
                    print("con_score<=5")
                    if self.explicit:
                        
                        capable = self.json_chat(
                            f"{SYSTEM_PROMPT}\n{CAPABLE_RECOMMEND_RATING_PROMPT_EXPLICIT}\nChat History:\n{self.context_string}\n\nUser Input:\n{self.context[-1]}\n\nProduct Types:{list(self.product_type_set)}",
                            "capable",
                            "",
                            "capable",
                        )
                    else:
                        capable = self.json_chat(
                            f"{SYSTEM_PROMPT}\n{CAPABLE_RECOMMEND_RATING_PROMPT}\nChat History:\n{self.context_string}\n\nUser Input:\n{self.context[-1]}\n\nProduct Types:{list(self.product_type_set)}",
                            "capable",
                            "",
                            "capable",
                        )
                    # import pdb;pdb.set_trace()
                    capable=True if capable=="True" or capable==True else False
                    # print(f'capable:{capable}')
                    if capable == True:
                        print("capable==True")
                        status = "Recommend"

                    else:
                        print("capable==False")
                        self.start_chat = len(self.context)-1
                        status = "Chatbot"
        print(status)
        return status

    def main_loop(self):
        while True:
            self.context_string = "\n".join(self.context[-6:])
            self.user_message = input("User: ")
            start_time = time.time()
            self.context.append(f"User: {self.user_message}")
            if self.user_message == "exit":
                break

            self.status = self.recommend_fst(self.status)

            if self.status == "Chatbot":
                self.output = self.json_chat(
                    f"{CHAT_PROMPT}\nChat History:\n{self.context_string}",
                    "chatbot",
                    self.user_message,
                    "response",
                )

            elif self.status == "Recommend":
                # if self.ask_preference==True:
                if len(self.context) > 1:
                    self.resource_dict["user_preference"] = self.json_chat(
                        f"{SYSTEM_PROMPT}\n{UPDATE_USER_PREFERENCE_PROMPT}\nOriginal {'user_preference'}: {self.resource_dict['user_preference']}\n\nChat History:\n{self.context_string}\n",
                        "preference_sum",
                        self.user_message,
                        "preference_summary",
                        strict_mode=False,
                    )
                    print(f"User Preference: {self.resource_dict['user_preference']}")
                # self.ask_preference=False

                goal = self.json_chat(
                    f"{SYSTEM_PROMPT}\n{GOAL_PROMPT}\nChat History:\n{self.context_string}\n{self.context[-1]}",
                    "goal",
                    "",
                )["goal"]
                # print(f"Goal:{goal}")
                instruction = self.json_chat(
                    f"{SYSTEM_PROMPT}\n{ACHIEVE_PROMPT} {goal}? Please describe as short ang specific as possible",
                    "instruction",
                    "",
                    "goal_instruction",
                    strict_mode=False,
                    max_tokens=50,
                )
                # print(f"Instrction:{instruction}")
                required_resource = self.json_chat(
                    f"{SYSTEM_PROMPT}\n{RESOURCES_PROMPT}\nGoal:{goal}\nInstruction:{instruction}\nChat History:\n{self.context_string}\n{self.context[-1]}\n",
                    "required_resource",
                    "",
                )
                # print(f"Required Resource:{required_resource}")

                # for k in required_resource.keys():
                #     if required_resource[k]==True:
                #         resource_dict[k]=self.json_chat(f"{SYSTEM_PROMPT}\n{SUMMARY_PROMPT}\nGoal:{goal}\nChat History:\n{self.context_string}\n",'resource',self.user_message)[k]

                # required_resource={k:True if k in required_resource else k:False for k in re}
                
                required_resource={x:True if required_resource[x]=="True" or required_resource[x]==True else False for x in required_resource}
                if (
                    required_resource["need_user_preference"] == True
                ):  # need user's specified information
                    resources = "\n".join(
                        [f"{k}:{v}" for k, v in self.resource_dict.items()]
                    )
                    self.output = self.json_chat(
                        f"{SYSTEM_PROMPT}\n{USER_PREFERENCE_PROMPT}\nGoal:{goal}\nInstruction:{instruction}\nResources:\n\n{resources}",
                        "preference",
                        self.user_message,
                        "question_for_preference",
                    )
                    # self.ask_preference=True
                else:
                    resources = (
                        "User Preference: " + self.resource_dict["user_preference"]
                    )
                    if required_resource["need_whole_chat_histories"] == True:
                        whole_context_string = "\n".join(self.context)
                        resources += f"\n\nWhole Chat History:\n{whole_context_string}"
                        # self.resource_dict['whole_chat_histories']=self.json_chat(f"{SYSTEM_PROMPT}\n{SUMMARY_PROMPT}\nGoal:{goal}\nResources:{resources}\n\nWhole Chat History:\n{whole_context_string}\nYou should summarize the {'whole_chat_histories'}",'chat_history',self.user_message)['chat_history_sum']

                    if required_resource["need_product_details"] == True:
                        selected_products = self.json_chat(
                            f"{SYSTEM_PROMPT}\n{PRODUCT_SELECTION_PROMPT}\nGoal:{goal}\nInstruction:{instruction}\n\nProduct Key Features:{self.resource_dict['product_details']}\n\nResources:{resources}\nContext: {self.context_string}",
                            "product_select",
                            "",
                            max_tokens=300,
                        )["Necessary_Products"]
                        # print(selected_products)
                        from fuzzywuzzy import process

                        product_string = ""
                        for p in selected_products.keys():
                            best_match = process.extractOne(
                                selected_products[p], self.product_name
                            )
                            product_string += f"\n{best_match[0]}: {self.product_info[self.product_info['title']==best_match[0]]['description'].values[0]}"

                        resources += f"\n\nWhole product details:\n{product_string}"

                    self.output = self.json_chat(
                        f"{SYSTEM_PROMPT}\n{RECOMMEND_PROMPT}\nGoal:{goal}\nInstruction:{instruction}\nResources:{resources}\nContext: {self.context_string}",
                        "response",
                        self.user_message,
                        "response",
                        max_tokens=200,
                    )

            print(f"System: {self.output}")
            self.context.append(f"System: {self.output}")
            print(f"Time elapsed: {time.time()-start_time}")

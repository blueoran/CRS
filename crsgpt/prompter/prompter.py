from datetime import datetime

def compose_prompt(task,instruction,resources,requirement):
    prompt = '\n'.join([
        f"Your current task is: {task}" if task else "",
        f"To accomplish this task, You should: {instruction}" if instruction else "",
        f"The resources that will be given in the following: {resources}" if resources else "",
        f"Please note that: {requirement}" if requirement else ""
        ])

    return prompt

class Prompter:
    
    SYSTEM_PROMPT = \
    f"""You are a Socially Intelligent General Recommender that serve in {datetime.today().strftime('%m/%d/%Y')}. Your ultimate goal is to provide personalized and contextually relevant recommendations or suggestions as precise as possible.
    To achieve this goal, you should leverage user preferences, product information, historical data, real-time context, and so on, to generate accurate and engaging recommendations, while maintaining a seamless and intuitive conversational experience."""

    COMPLETE_CHAT_PROMPT = "As a refiner and summarizer, your primary function is to refine and summarize incomplete outputs from ChatGPT caused by token limits. Your objective is to provide consistent and coherent summaries that can be directly communicated with the user. To achieve this, You should utilize all the context available to you, ensuring the generated summaries maintain relevance and accuracy."


    f"Given the following descriptions of products, please first define what type this product belongs to, then extract and summarize {3} most important key features as short and simplified as possible that for making a successful recommendation"
    PRODUCT_PROMPT = compose_prompt(
        f"First define what type this product belongs to, then extract and summarize {5} most important key features for this product",
        "Consider the features that are concerend by most people, and are most important for the recommendation system to achieve the goal",
        "the detailed product information of this product",
        "the key features should be as short and simplified as possible that for making a successful recommendation"
    )
    f"Given the following descriptions of products, please first define what type this product belongs to, then extract and summarize {3} most important key features as short and simplified as possible that for making a successful recommendation"
    PRODUCT_TYPE_PROMPT = compose_prompt(
        f"Define what type this product belongs to",
        "Consider the features that are concerend by most people, and are most important for the recommendation system to achieve the goal",
        "the detailed product information of this product",
        None
    )
    PRODUCT_TYPE_SUM_PROMPT = compose_prompt(
        f"Define what type this product belongs to, then extract and summarize key features for this product",
        "Consider the features that are concerend by most people, and are most important for the recommendation system to achieve the goal",
        "the detailed product information of this product",
        None
    )

    'Your task is to assess the probability that the user currently needs a recommendation system to serve for products recommendation or summary, comparison, list in detail etc. based on user\'s cureent input and the context of the conversation.\n\nTo score this probability, you should consider various factors, such as: User Intent, Previous Interactions, Contextual Keywords, User Engagement, Recommendation Process\n\nUsing these factors, please develop an algorithm that assigns a probability score between 0 and 9, indicating the likelihood that the user currently requires a recommendation system to serve. Please briefly explain your scoring reasons and provide an appropriate score.'
    

    RECOMMEND_RATING_PROMPT = compose_prompt(
        "Evaluate the likelihood that the user requires the recommendation system instead of just a chatbot.",
        "Consider if the user's intent needs product-specific to response, which may involves seeking detailed product information, summarizing products, comparing products, or explicitly or implicitly asking for recommendations.",
        "The user's current input and the ongoing conversation context.",
        "Provide a probability score from 0 to 9. A score below 5 suggests the user mainly needs chatbot interaction. Ensure the score is appropriate to the user's intent."
    )


    RECOMMEND_RATING_PROMPT_EXPLICIT = compose_prompt(
        "Evaluate the likelihood that the user requires the recommendation system use its database to response instead of just a chatbot.",
        "Consider if the user's intent needs product-specific details or attribute values to response, which may involves seeking detailed product information, summarizing products, comparing products, or explicitly or implicitly asking for recommendations. If so, the user needs the recommendation system to serve.",
        "The user's current input and the ongoing conversation context.",
        "Provide a probability score from 0 to 9. A score below 5 suggests the user mainly needs chatbot interaction. Ensure the score is appropriate to the user's intent. Please briefly explain your scoring reasons and provide an appropriate score."
    )
    
    "Your task is to score the consistency of the current user's input with the previous dialog context. Take into account both the user's input and the context to determine how well the current input aligns with the conversation history.You should be strict with the consistency between the latest context and the current user's input, while loose your standard with the former dialogs. Assign a consistency score between 0 and 9, where 0 represents complete inconsistency and 9 represents perfect consistency. Please briefly explain your scoring reasons and provide an appropriate score."
    CONSISTENCY_RATING_PROMPT = compose_prompt(
        "Score the consistency of the current user's input with the previous dialog context",
        "Take into account both the user's input and the context to determine how well the current input aligns with the conversation history. Be strict with the consistency between the latest context and the current user's input, while loose your standard with the former dialogs.",
        "the user's input and the context of the conversation",
        "the consistency score should between 0 and 9, indicating the extent of the consistency. Score below 5 indicates a lack of consistency. Please only provide an appropriate score."
    )
    CONSISTENCY_RATING_PROMPT_EXPLICIT = compose_prompt(
        "Score the consistency of the current user's input with the previous dialog context",
        "Take into account both the user's input and the context to determine how well the current input aligns with the conversation history. Be strict with the consistency between the latest context and the current user's input, while loose your standard with the former dialogs.",
        "the user's input and the context of the conversation",
        "the consistency score should between 0 and 9, indicating the extent of the consistency. Score below 5 indicates a lack of consistency. Please briefly explain your scoring reasons and provide an appropriate score."
    )



    "In order to provide accurate recommendations, your task is to determine if the given product types align with the user's desired recommendations. Your should assist the system in verifying whether the given product types in the database can support the user's goal effectively.\n\nTo accomplish this, please take the following steps:\n1. Analyze the user's preferences and identify the specific product types they desire.\n2. Evaluate the existing product types in the database and other available resources.\n3. Determine whether the product types requested by the user are present in the database.\n\nPlease perform these steps to ensure the system can accurately assess whether the recommendation system is capable for the user's desired recommendations and then provide an answer. You also should explain your reasons for this answer."
    CAPABLE_RECOMMEND_RATING_PROMPT = compose_prompt(
        "Determine if the product types that the system can serve align with the user's desired recommendations.",
        "Enumerate each product type, analyze whether it is appropriate to recommend the user with this product type. If all of the product types that the system serve are not appropriate, then the recommendation system is not capable for the user's desired recommendations. You also have the ability to search through the web only about the product-relevant information.",
        "the product types that the system can serve, the user's input and the context of the conversation",
        None
    )
    CAPABLE_RECOMMEND_RATING_PROMPT_EXPLICIT = compose_prompt(
        "Determine if the product types that the system can serve align with the user's desired recommendations.",
        "Enumerate each product type, analyze whether it is appropriate to recommend the user with this product type. If all of the product types that the system serve are not appropriate, then the recommendation system is not capable for the user's desired recommendations. You also have the ability to search through the web only about the product-relevant information.",
        "the product types that the system can serve, the user's input and the context of the conversation",
        "Please explain your reasons for this answer."
    )

    "Your task is to act as a normal conversation chatbot, engaging in friendly and helpful conversation with the user. If you find the given context is in some recommendation topics, you should do a seamlessly transition into a chatbot smoothly. You should maintain a positive and friendly tone throughout the conversation, aiming to keep the user engaged and satisfied. Please provide an enjoyable user experience continuously."
    CHAT_PROMPT = compose_prompt(
        "Act as a normal conversation chatbot, engaging in friendly and helpful conversation with the user.",
        None,
        "the user's input and the context of the conversation",
        None
    )
    
    "Given the user's input and the history preferences and the chat history, determine whether the preference should be generated, re-summarized or updated, if it should, update the user's preferences accordingly."
    UPDATE_USER_PREFERENCE_PROMPT = compose_prompt(
        "Integrate the user's preferences accordingly.",
        "Consider the user's input, the context of the conversation and the former user preference. Mainly focus on the current user's response if there are direct or indirect preference information in the response, but do your best to preserve the former user preference when integrating the current user preference, unless the user obviously deny the former preference.",
        "the user's input, the context of the conversation and the former user preference",
        None
    )
    
    "Given the Chat History below, generate the most important step for the chatbot to formulate a response that addresses the user's needs and maintains a coherent conversation flow. Then interpret this as a goal prompt to chatgpt to generate response. Don't use 'I' in the prompt."
    GOAL_PROMPT = compose_prompt(
        "Generate the most important thing the recommendation system should do to formulate a response that addresses the user's needs and maintains a coherent conversation flow. Then interpret this as a goal prompt for chatgpt to generate response.",
        None,
        "The user's input and the context of the conversation",
        "Don't use 'I' in the prompt."
    )

    ["What aspect should I consider to achieve the goal of ", 'Please describe breifly']
    ACHIEVE_PROMPT = compose_prompt(
        "Generate detailed steps as prompt that can instruct the chatgpt-based recommendation system to achieve the goal.",
        None,
        "the goal of the user's request",
        "Be brief and concise."
    )
    
    "Given the goal and the chat history provided, determine the least resources needed for you to generate a response that effectively achieves the goal and addresses the user's needs. Remember you mustn't recommend products that are not given from the database. You should first consider to make the full use of the given context and user preference, determine whether you need more information from the user preference and from database, then request what you really need only among [ user preference, product details ] to assist in achieving the goal."
    # RESOURCES_PROMPT = compose_prompt(
    #     "Determine the resources needed for you to generate a response that effectively achieves the goal and addresses the user's needs.",
    #     "Take all the given resources into account. You should first consider to make the full use of the given context and current resources, determine whether you need more information from the user preference and from database, then request what you really need among [ user preference, product details ] to assist in achieving the goal.",
    #     "the goal, instruction, the current resources you should focus on, the context of the conversation, and the user's input",
    #     "Be open to ask for more relevant resources."
    # )
    # RESOURCES_PROMPT_EXPLICIT = compose_prompt(
    #     "Determine the least resources needed for you to generate a response that effectively achieves the goal and addresses the user's needs.",
    #     "Take all the given resources into account. You should first consider to make the full use of the current resources, determine whether you need more information from the user preference and from database, then request what you really need among [ user preference, product lists ] to assist in achieving the goal.",
    #     "the goal, instruction, the current resources you should focus on, the context of the conversation, and the user's input",
    #     "Make the best use of the current given resources first. Summarize what do you have in current resource to meet the user's requirement, and why you need or don't need these resources."
    # )
    # RESOURCES_PROMPT = compose_prompt(
    #     "Determine the least resources needed for you to generate a response that effectively achieves the goal and addresses the user's needs.",
    #     "Take all the given resources into account. You should first consider to make the full use of the given context and current resources, determine whether you need more information from the user preference and from database, then request what you really need among [ user preference, product details ] to assist in achieving the goal.",
    #     "the goal, instruction, the current resources you should focus on, the context of the conversation, and the user's input",
    #     "Make the best use of the current given resources first."
    # )

    RESOURCES_PROMPT = compose_prompt(
        "Determine the least resources needed for you to generate a response that effectively achieves the goal and addresses the user's needs.",
        "Take all the given resources into account. You should first consider to make the full use of the current resources, determine whether you need more information from the user preference and from database, then request what you really need among [ user preference, product lists ] to assist in achieving the goal. Please note that you should consider if the current resources can perfectly satisfy all the aspect of the user's needs, which means you need to be careful to say that you don't need any resources. If you need user's preference, the system will ask for the user's preference in the following; if you need the product lists, the system will search for needed products in the product database in the following.",
        "the goal, instruction, the current resources you should focus on, the context of the conversation, and the user's input",
        "Make the best use of the current given resources first. Summarize what do you have in current resource to meet the user's requirement, and why you need or don't need these resources. If you don't need any of them, you should be clear about why the current resources can perfectly satisfy all the aspect of the user's needs."
    )
    RESOURCES_PROMPT_WEB = compose_prompt(
        "Determine the least resources needed for you to generate a response that effectively achieves the goal and addresses the user's needs.",
        "Take all the given resources into account. You should first consider to make the full use of the current resources, determine whether you need more information from the user preference and from database, then request what you really need among [ user preference, product lists, web resources ] to assist in achieving the goal. Please note that you should consider if the current resources can perfectly satisfy all the aspect of the user's needs, which means you need to be careful to say that you don't need any resources. If you need user's preference, the system will ask for the user's preference in the following; if you need the product lists, the system will search for needed products in the product database in the following; if you need web resources, the system will search for needed web resources through google in the following.",
        "the goal, instruction, the current resources you should focus on, the context of the conversation, and the user's input",
        "Make the best use of the current given resources first. Summarize what do you have in current resource to meet the user's requirement, and why you need or don't need these resources. If you don't need any of them, you should be clear about why the current resources can perfectly satisfy all the aspect of the user's needs."
    )
    RESOURCES_PROMPT_WEB_EXPLICIT = compose_prompt(
        "Determine the least resources needed for you to generate a response that effectively achieves the goal and addresses the user's needs.",
        "Take all the given resources into account. You should first consider to make the full use of the current resources, determine whether you need more information from the user preference and from database, then request what you really need among [ user preference, product lists, web resources ] to assist in achieving the goal. Please note that you should consider if the current resources can perfectly satisfy all the aspect of the user's needs, which means you need to be careful to say that you don't need any resources. If you need user's preference, the system will ask for the user's preference in the following; if you need the product lists, the system will search for needed products in the product database in the following; if you need web resources, the system will search for needed web resources through google in the following.",
        "the goal, instruction, the current resources you should focus on, the context of the conversation, and the user's input",
        "Make the best use of the current given resources first. Summarize what do you have in current resource to meet the user's requirement, and why you need or don't need these resources. If you don't need any of them, you should explain in detail why the current resources can perfectly satisfy all the aspect of the user's needs."
    )

    RESOURCES_PROMPT_EXPLICIT = compose_prompt(
        "Determine the least resources needed for you to generate a response that effectively achieves the goal and addresses the user's needs.",
        "Take all the given resources into account. You should first consider to make the full use of the current resources, determine whether you need more information from the user preference and from database, then request what you really need among [ user preference, product lists ] to assist in achieving the goal. Please note that you should consider if the current resources can perfectly satisfy all the aspect of the user's needs, which means you need to be careful to say that you don't need any resources. If you need user's preference, the system will ask for the user's preference in the following; if you need the product lists, the system will search for needed products in the product database in the following.",
        "the goal, instruction, the current resources you should focus on, the context of the conversation, and the user's input",
        "Make the best use of the current given resources first. Summarize what do you have in current resource to meet the user's requirement, and why you need or don't need these resources. If you don't need any of them, you should explain in detail why the current resources can perfectly satisfy all the aspect of the user's needs."
    )

    PRODUCT_SELECTION_PROMPT = f"Given the goal, context, the key points of each product, and the details of the resource provided, select {4} products that are most necessary for a chatgpt to achieve the goal. Remember you mustn't recommend products that are not given from the following products."
    
    PRODUCT_APPEAR_PROMPT = compose_prompt(
        "Analyze the given context and user's input to determine the user's attitute to all the products appeared during the conversation, then give each product with comments.",
        'Identify all the products in the context, then figure out if the user have previously consumed, experienced or exhibited feelings towards any product. Extract the names of specific products the user and the system mention and summarize the user\'s attitude towards the product implied by the user\'s response using one or two words.',
        # 'Identify if the user have previously consumed, experienced or exhibited positive/negative feelings towards any product. Extract the names of specific products the user and the system mention and categorize their sentiment or intent (positive, negative, neutral, already used). If a product is tagged with "already used", it means the recommendation system has already recommended before and is refused or ignored directly/indirectly by the user. Products with "positive" sentiment means the user hold favorable opinion to this product, which can be used to refine and find similar products. Products with "negative" sentiment means the user hold unfavorable opinion to this product, which can be used to aviod similar products. Products with "neutral" sentiment can be interpreted as the user had expressed his opinion towards this product, but is not clear whether it is positive or negative.',
        "context, user input, and the previous appearance of products",
        ''
    )



    "Given the goal, instruction, context, and resources, please select what types of the products are necessary for the recommendation system to achieve the goal."
    PRODUCT_TYPE_SELECT_PROMPT = compose_prompt(
        "Select what types of the products are necessary for the recommendation system to achieve the goal.",
        "Consider all the given resources, especially the user input and user preference",
        "the goal, instruction, user preference, context, the user's input, and available product types",
        "Only select the product types from the given available product types below."
    )

    PRODUCT_FEATURE_PROMPT = compose_prompt(
        "Figure out the key features that the products should have according to the goal and the user's preference.",
        "Focus on the user's preference and use your common sense to determine the key features that the products should have.",
        "the goal, instruction, and the user preference.",
        "Be brief and concise. Please note that user's preference to specific products is also a key feature."
    )
    
    PAST_PRODUCT_PROMPT = compose_prompt(
        "Figure out whether the past recommended products should be included or excluded in achieving current goal.",
        "Consider the user's input and analysis the user's needs, then enumerate all the past recommended products and determine whether each of them should be selected for the recommendation system to achieve the goal. If the user have previously consumed, experienced or exhibited negative feelings towards the product, this product is usually excluded. If the user shows positive feelings towards the product, this product is usually included.",
        "the goal, instruction, user preference, context, the user's input, and the past recommended products",
        "DO NOT SELECT PRODUCTS THAT DOESN'T APPEAR IN CURRENLY GIVEN PRODUCTS. Tag each of the past recommended products with 'include' or 'exclude'."
    )

    "Given the goal, instruction, context, resources and the recommend system's summary of the previous selection, please select the products from the following provided products to add to the previous selection, so that the selection of products can make the recommendation system to achieve the goal. You should also summarize each product's description from what you select by the key point that can response to the user's requests. In the end, you should also determine whether the previous together with your current selection to the products are enough for the system to achieve the goal and all the use requests."
    PRODUCT_SEEK_PROMPT = compose_prompt(
        "Select most appropriate and necessary products from the given products below for the recommendation system to achieve the goal.",
        """
        Know that the whole product database is so large that you can only see a part of it once a time, and you will be given the summary of the products summay that you have selected previously.
        Based on this, you need to first analyze why the previously selected products (if is not empty) cannot fully satisfy the current goal, and user's requests with preference.
        Then select the new products from the following provided products that satisfy the key features that the product should have and add to the previous selection, so that the whole selection of products can make the recommendation system to fully achieve the goal.
        Then summarize and pickout key points on each of your newly selected product's description based on the goal, and the user's requests with preference.
        Finally, you should also determine whether the whole set of selected products can fully satisfy the system to achieve the goal and all the use requests.
        """,
        "the goal, instruction, user preference, context, the user's input, the key points of previous selected products (if is not empty), the key features that the selected products should have and currently available product database",
        "DO NOT SELECT PRODUCTS THAT DOESN'T APPEAR IN CURRENLY GIVEN PRODUCTS. Only select necessary products and only from the given available products below. Make the best use of the previous selected products first before you select new products. If there is no products that can satisfy, just select nothing.",
    )
    "Given the goal, instruction, context, resources and the recommend system's summary of the previous selection, please select the products from the following provided products to add to the previous selection, so that the selection of products can make the recommendation system to achieve the goal. You should also summarize each product's description from what you select by the key point that can response to the user's requests. In the end, you should also determine whether the previous together with your current selection to the products are enough for the system to achieve the goal and all the use requests."
    PRODUCT_SEARCH_PROMPT = compose_prompt(
        "Generate a prompt that describes the accurate and detailed features of the potential products for the vector database to select the products that are most appropriate and necessary for the recommendation system to achieve the goal.",
        """
        Analyze the given resources, especially the goal, the user input, user preference and the key features that the selected products. Then accurately summarize the product features that the products should have with detailed information. Finally generate an appropriate prompt for vector database to find out appropriate products.
        """,
        "the goal, instruction, user preference, context, the user's input, the key features that the selected products should have.",
        "Be concise when generating the prompt. The detailed product attributes should be reserved.",
    )
    PRODUCT_SEARCH_PROMPT_ADV = compose_prompt(
        "Generate a set of prompts that can guide the vector database in identifying the most appropriate products for the recommendation system. This should consider the goal, user input, user preference, context, and the essential features that selected products should exhibit.",
        """
        Based on the given resources:

        1. Enumerate the positive features that the products must have, or specific products. These features should be clear indicators of suitability, so the database can focus on products that possess them.
        
        2. Enumerate the negative features or conditions that the products should not exhibit, or specific products. This will allow the database to eliminate products that are contradictory or non-compliant with the desired specifications.

        Make sure to analyze the user's input, context, and other factors to create these detailed and accurate prompts for product filtering.
        """,
        "the goal, instruction, user preference, context, the user's input, the key features that the selected products should have.",
        "The prompts should be concise and clear. It is better to use a set of keywords rather than a long sentence to describe a feature. The detailed product attributes should be reserved. Specific product name should also be taken into account."
    )


    PRODUCT_VERIFY_PROMPT = compose_prompt(
        "Evaluate if the selected products align with the user's needs and preferences using the provided resources.",
        "Examine every part of the product details and the user's preferences to identify any mismatches between the two.",
        "Consider: goal, instruction, user preference, context, user input, product details, and essential product features.",
        "If products only partly satisfy the user's needs, acknowledge the alignment as true. However, clarify which aspects of the user's needs are met and which aren't. If there are factual or logical errors, consider the alignment as false and highlight the discrepancies."
    )
    
    WEB_QUERY_PROMPT = compose_prompt(
        "Act as an assistant tasked with improving Google search results. Generate THREE Google search queries that are helpful to achieve the goal.",
        "Use the given resources to generate queries that can be used to search for relevant information on Google in order to achieve the current goal.",
        "the goal, instruction, user preference, context, the user's input.",
        "The output should be a list of questions and each should have a question mark at the end"
    )

    "Considering the goal and the resources required, generate a question that should ask the user to gather their preferences effectively. This question should aim to obtain crucial information aligned with the goal and enable the chatbot to provide personalized recommendations based on the user's preferences. Ensure that the reply incorporates most relevant information and maintains coherence with the conversation."
    USER_PREFERENCE_PROMPT = compose_prompt(
        "Generate a question that should ask the user to gather their preferences effectively.",
        "This question should aim to obtain crucial information aligned with the goal and enable the recommendation system to provide personalized recommendations based on the user's preferences.",
        "the goal, instruction, user preference, context, the user's input, and the key points of selected products that you should referernce (if given)",
        "Ensure that the reply incorporates most relevant information and maintains coherence with the conversation."
    )

    "Considering the context, goal, and the given resources, generate a response that provides the most successful recommendation tailored to the user's needs. Ensure that the reply incorporates most relevant information, aligns with the user's preferences, and maintains coherence with the conversation. Remember you mustn't recommend products that are not given from the following products. If you are not able to achieve any part of the goal even after synthesised all of the resources, just notify the user. You should provide short reasons or details that can convince the user about your response."

    RECOMMEND_PROMPT = compose_prompt(
        "Generate a response that provides the most appropriate response tailored to the user's input.",
        "Ensure that the reply incorporates most relevant information, aligns with the user's preferences, and maintains coherence with the conversation.",
        "The goal, instruction, user preference, context, the user's input, and resources",
        "You mustn't recommend products that are not given from the following provided products. You should be precise, if you are not able to achieve some part of the goal even after synthesised all of the resources, you need to notify the user about this, don't pretend you can."
    )
    RECOMMEND_PROMPT_EXPLICIT = compose_prompt(
        "Generate a response that provides the most appropriate response tailored to the user's input.",
        "Ensure that the reply incorporates most relevant information, aligns with the user's preferences, and maintains coherence with the conversation.",
        "The goal, instruction, user preference, context, the user's input, and the key points of selected products",
        "You mustn't recommend products that are not given from the following provided products. You should be precise, if you are not able to achieve some part of the goal even after synthesised all of the resources, you need to notify the user about this, don't pretend you can. Tell me your thoughts when generating this response."
    )


    """
    You are an experienced supervisor responsible for evaluating the performance of our product recommendation system. Your role is to ensure that the system provides accurate and relevant recommendations to users based on the information provided to it.

    To assess the recommendation system, you will be presented with a series of dialogues between users and the system. In each dialogue, the user's query and the system's response will be provided. Additionally, I will include the accurate information that the system should have utilized to generate a correct recommendation.

    Please carefully review each dialogue and analyze the system's responses. If you detect any instances or clues that has logical faults or fact faults, especially wrong in key features and key values, flagit. If you detect instances of hallucination, where the system provides recommendations based on its own inaccurate knowledge rather than the accurate information provided to it,  also flag it. After detecting them, offer constructive feedback on how the system should have responded using the provided accurate information.
    """
    VALIDATE_ANSWER_PROMPT = compose_prompt(
        "Evaluate whether the current conversational recommendation system response accurately and reasonably.",
        "Determin whether the system's output is based on the given resources and is reasonable and accurate enough to response to the user's input.",
        "The context, the user's input, output of the recommendation system, and the given accurate information: user preference and products information.",
        "Also offer constructive feedback to the recommendation system to instruct it to generate a better response. You should also only be based on the given preference and products resources to generate this feedback."
    )

    SCORER_PROMPT="""As an experienced conversational recommendation system expert, your task is to assess and score the performance of our product recommendation system. You will be provided with the context, information given to the system, and the system's response to the user. Please evaluate the system's response based on the following aspects:

    1. Recommendation Success: Evaluate how effectively the system's response aligns with the user's query and addresses their needs. Consider the appropriateness and usefulness of the recommendations provided.

    2. Relevance and Accuracy: Assess the relevance of the system's response to the user's query and how accurately it provides information. Verify if the recommendations are based on correct and up-to-date data.

    3. Logical Reasoning: Examine the logical reasoning behind the system's response concerning the user's input and information provided. Evaluate how well the system justifies its recommendations based on the context.

    Please provide a comprehensive evaluation based on these aspects to help improve the performance of our product recommendation system."""


    SUMMARY_PROMPT = "Given the goal, context, and the details of the resource provided, select the key points or items that are relevant to the user's needs. Then, generate a concise summary that effectively captures the essence of the resource while aligning with the goal and maintaining coherence with the conversation."

class TestPrompter:
    SYSTEM_PROMPT = \
    """You are an expert evaluator in the conversational recommendation field, and your task is to test the functionality and rubostness of the system."""

    TEST_PROMPT = \
        SYSTEM_PROMPT + '\n' + \
        compose_prompt(
            None,
            "Achieve this through communicating with the system based on the circumstance described below with the example conversation, and the current context. Communicate with the system as if you were a real user with specific needs, concerns, or questions.",
            "The circumstance, example conversation, the context, the product types that the system can serve, and the output from the recommendation system.",
            "ONLY USE THE PRODUCT TYPES THAT THE SYSTEM CAN SERVE. Be aware of the process of the communication and make the conversation follow the circumstance setting, and the context of the conversation. Don't make the whole chat too long, if you think the test should reach to end, only say 'exit' to end the conversation."
        )

    TEST_CASES = {
"NORMAL":
"""Circumstance: User Seeking a Normal Recommendation

Description: In this circumstance, the user is interacting with your conversational recommendation system in a typical and straightforward manner. The user has a clear goal or preference, and they are looking for a recommendation based on their input. The user's tone and language are neutral, and they are providing relevant information without any particular twists or challenges.

Example Conversation:

User: Hi there! I'm in the mood for watching a movie tonight. Can you recommend a good drama film?

System: Of course! I'd be happy to help you find a drama film to watch. Do you have any preferences regarding the time period the movie is set in, or any specific actors or directors you enjoy?

User: I'm open to different time periods, but I do love movies starring Tom Hanks.

System: Great! Based on your preference for drama movies and Tom Hanks, I have a recommendation for you. How about watching "Forrest Gump"? It's a classic drama film starring Tom Hanks that spans several decades, following the life of a man with a unique perspective on the world.

User: Oh, I've heard of "Forrest Gump" before, but I've never actually watched it. That sounds like a good choice!

System: I'm glad you like the suggestion! "Forrest Gump" is a heartwarming and emotional movie that has received critical acclaim. I think you'll enjoy it. Is there anything else you're looking for, or is there anything specific you'd like to know about the movie?

User: No, that's perfect! I'll go with "Forrest Gump." Thanks for the recommendation!

System: You're welcome! Enjoy the movie, and if you ever need more recommendations in the future, feel free to ask. Have a great time!

In this scenario, the user's request was straightforward, and the system responded by asking clarifying questions to better understand the user's preferences. Once the user's preference for drama movies and Tom Hanks was established, the system provided a relevant recommendation and additional information about the movie. The user accepted the recommendation and expressed appreciation, concluding the conversation on a positive note.""",

"UPDATE_PERFORMANCE":
"""Circumstance: User Updating Implicit Preferences
Description:
Implicit preferences refer to user preferences that can be inferred from their behavior, actions, or historical data, rather than explicitly stated by the user. In this circumstance, you are testing your conversational recommendation system's ability to adapt to changes in a user's implicit preferences. Users may not always explicitly update their preferences, but their behavior and interactions with the system can indicate shifts in their interests.

Example Conversation:

User: Hey there! I ask for a book recommendation.
System: Of course! I'm here to help.

User: I've been really into mystery novels lately. I can't seem to get enough of them.
System: That's great to know! It seems like you've developed a liking for mystery novels. Are there any specific mystery authors you've enjoyed, or should I suggest something new for you?

User: Well, I've read a lot of Agatha Christie, so something similar would be awesome.
System: Got it! If you're a fan of Agatha Christie's work, you might enjoy other classic mystery authors like Arthur Conan Doyle with his Sherlock Holmes series. Additionally, contemporary authors like Tana French and Louise Penny have gained popularity for their gripping mysteries. Would you like more information about any of these authors or their books?

User: Actually, scratch that. I've realized I'm more into historical fiction now, especially set in the Victorian era.
System: Noted! Your shift to historical fiction, especially in the Victorian era, gives us a new direction to explore. Authors like Charles Dickens and Wilkie Collins wrote captivating stories set in that time period. Their works offer a blend of intricate plots and rich historical settings. Would you be interested in recommendations from these authors or similar ones?

User: Yes, that sounds perfect. What are some must-reads from Charles Dickens?
System: Certainly! Charles Dickens has crafted several timeless classics. I'd recommend starting with "Great Expectations," which offers a coming-of-age tale set against a Victorian backdrop. Another popular choice is "A Tale of Two Cities," a historical novel set during the French Revolution. Both of these novels should provide a captivating journey into Victorian-era storytelling.

User: Those sound intriguing. I'll definitely check them out. Thanks for the help!
System: You're welcome! If you have any more questions or need further recommendations in the future, feel free to ask. Enjoy your dive into Victorian historical fiction!

In this conversation example, the user's implicit preference shift from mystery novels to historical fiction is identified and acknowledged by the recommendation system. The system adapts its suggestions based on the user's updated interests and provides relevant recommendations. This testing scenario helps ensure that your conversational recommendation system is responsive to subtle shifts in user preferences.""",


"PRODUCT_DETAIL":
"""Circumstance: User Examining the Details of a Certain Product

Description: In this scenario, the user is interested in a specific product and wants to learn more about its details, features, specifications, and possibly compare it with other similar products. The user is seeking comprehensive information to make an informed decision about whether the product suits their needs and preferences.

Example Conversation:

User: Hi there! I'm looking into buying a new smartphone. Can you give me more details about the latest model from "TechGizmo"?

System: Of course! I'd be happy to help. The latest model from TechGizmo is the "GizmoPhone X." It features a 6.5-inch OLED display with a resolution of 1080 x 2340 pixels. The phone is powered by a Snapdragon 8-series processor with 8GB of RAM and comes in storage options of 128GB and 256GB. It has a dual-camera setup with a 48MP main sensor and a 16MP ultra-wide sensor. The battery capacity is 4000mAh and supports fast charging.

User: That sounds interesting. How does the camera perform? I enjoy taking photos.

System: The camera system on the GizmoPhone X is quite impressive. The 48MP main sensor uses pixel-binning technology to capture vibrant and detailed photos even in low light conditions. The AI-powered software helps optimize settings for various scenarios like portraits, landscapes, and night shots. The 16MP ultra-wide sensor provides a wider field of view, making it great for capturing scenic shots and group photos.

User: Nice. And what about the battery life? I use my phone throughout the day.

System: The 4000mAh battery on the GizmoPhone X is designed to provide all-day usage on a single charge. The combination of hardware and software optimizations helps extend battery life. Additionally, the phone supports fast charging, allowing you to quickly top up your battery when needed.

User: Are there any alternatives to the GizmoPhone X that I should consider?

System: Certainly. If you're open to alternatives, you might want to check out the "CompetitorPhone Y." It offers a similar display size and resolution, along with a comparable camera system. However, it has a slightly larger battery capacity at 4200mAh. Another option is the "AlternateTech Z," which boasts a higher resolution display and a unique camera feature for low-light photography. It's essential to weigh these options based on your preferences and priorities.

User: Thanks for the detailed information. One last question: Does the GizmoPhone X support 5G connectivity?

System: Yes, the GizmoPhone X supports 5G connectivity, allowing you to experience faster download and streaming speeds in areas with 5G coverage.

User: Perfect. That helps a lot in making my decision. I appreciate your assistance!

System: You're welcome! If you have any more questions or need further assistance, feel free to ask. Good luck with your smartphone hunt!

In this example, the conversation showcases a user who is interested in a specific product (a smartphone) and wants detailed information about its features and specifications. The AI provides the user with comprehensive details, answers specific queries, and even suggests alternative products for consideration. This scenario tests the recommendation system's ability to provide detailed and relevant information to assist users in their decision-making process.""",

"SUMMARIZE_PRODUCT":
"""Circumstance: Summarizing a Product from User's Preference View

In this circumstance, the conversational recommendation system aims to summarize a product from the user's preference view. This means that the system needs to understand the user's preferences, either explicitly stated or inferred from previous interactions, and provide a concise and tailored summary of a specific product that aligns with those preferences. The summary should highlight the aspects of the product that are most likely to appeal to the user based on their expressed interests or past behavior.

Example Conversation:

User: Hi there! I'm looking for a new smartphone. I'm really into photography and I travel a lot, so a good camera and long battery life are important to me.

System: Hello! It sounds like you have a passion for photography and need a smartphone that can keep up with your travel adventures. I'd recommend checking out the "SnapPixel Pro." It boasts a cutting-edge camera with advanced low-light capabilities, perfect for capturing stunning photos during your travels. Additionally, it features an extended battery life that can last up to two days, ensuring you won't miss any photo opportunities. The "SnapPixel Pro" could be an excellent match for your preferences.

User: Oh, that's interesting! Do you know if it has a large internal storage capacity? I tend to store a lot of high-resolution photos.

System: Absolutely! The "SnapPixel Pro" comes with ample storage, offering a base capacity of 256GB, which should provide plenty of space for your high-resolution photos and videos. And if you ever need more space, it also supports expandable storage via microSD cards.

User: That sounds great! One last thing, I'm also concerned about the durability of the phone. I've had accidents in the past.

System: I understand your concern. The "SnapPixel Pro" is designed with durability in mind. It features a reinforced Gorilla Glass Victus display that's highly resistant to scratches and cracks. Additionally, the device has an IP68 rating, making it water and dust resistant. So, you can continue your adventures worry-free, knowing that your smartphone is built to withstand accidental bumps and splashes.

User: That's exactly what I need. Thanks for the recommendation and all the information!

System: You're welcome! If you have any more questions or need further assistance in the future, feel free to ask. Happy photographing and safe travels with your new "SnapPixel Pro"!

In this example conversation, the recommendation system identifies the user's preferences for photography, travel, battery life, camera quality, storage capacity, and durability. It then generates a tailored summary of the "SnapPixel Pro" smartphone, highlighting its features that align with the user's preferences. The system addresses the user's specific concerns and provides relevant information, demonstrating its ability to understand and cater to the user's preference view.""",

"COMPARE_PRODUCT":
"""Circumstance: Comparing Products from User's Preference View

In this circumstance, your conversational recommendation system focuses on comparing products based on the user's preferences. The goal is to provide personalized recommendations by understanding the user's preferences, priorities, and requirements, and then presenting a comparison between different products that align with those preferences. This approach allows users to make informed decisions by directly contrasting the features, benefits, and drawbacks of various products they might be interested in.

Example Conversation:

User: Hi there! I'm in the market for a new smartphone. I'm really into photography and I also need good battery life.

System: Great to hear! Photography and battery life are important factors. Just to clarify, are you looking for smartphones from a specific brand or any brand in particular?

User: Brand doesn't matter much to me. I'm open to options.

System: Got it. Based on your preferences, I've narrowed down two excellent options for you: the PixelPro X and the GalaxyCapture Z. 

User: Alright, tell me more about their camera capabilities.

System: Certainly! The PixelPro X features a 108MP main camera with optical image stabilization, allowing you to capture highly detailed shots even in low light conditions. On the other hand, the GalaxyCapture Z boasts a triple-lens system including a 64MP main camera, a 12MP ultra-wide lens, and a 5x optical zoom telephoto lens for versatile photography.

User: Impressive. How about battery life?

System: The PixelPro X comes with a 4500mAh battery that offers up to 2 days of moderate usage. It also supports fast charging and wireless charging. The GalaxyCapture Z, on the other hand, has a slightly larger 5000mAh battery that can also last around 2 days. It supports fast charging and has a power-saving mode to extend battery life.

User: Hmm, tough choice. What about the display quality?

System: Both phones have stunning displays. The PixelPro X features a 6.7-inch AMOLED with QHD+ resolution, providing vibrant colors and deep blacks, which can enhance your photography and multimedia experience. The GalaxyCapture Z has a 6.5-inch Dynamic AMOLED with similar QHD+ resolution, offering equally impressive visuals.

User: And the price difference?

System: The PixelPro X is priced at $899, while the GalaxyCapture Z is slightly more affordable at $799.

User: Thanks for the information! I think I'm leaning towards the PixelPro X due to the higher megapixel camera and the display quality.

System: You're welcome! The PixelPro X does seem like a great fit for your preferences. If you have any more questions or need further assistance, feel free to ask!

In this example conversation, the system identifies the user's preference for photography and battery life, provides comparisons between the two recommended smartphones, and highlights their camera capabilities, battery life, display quality, and price. This information helps the user make an informed decision based on their preferences.""",
        
"RECOMMEND_PRODUCT":
"""Circumstance: Recommending Products Based on User's Preference View

In this circumstance, the conversational recommendation system is designed to provide product recommendations to users based on their preferences. The system gathers information about the user's preferences, which could include their interests, past purchase history, demographic information, and explicit feedback. It then uses this information to suggest relevant products that align with the user's preferences. The conversation flow involves the system interacting with the user to understand their preferences and subsequently providing tailored product recommendations.

Example Conversation:

User: Hi there, I'm looking for some new tech gadgets. Any recommendations?

System: Of course! I'd be happy to help you find some tech gadgets. To start off, could you let me know what type of tech gadgets you're interested in? Are you looking for something specific?

User: I'm into smart home devices, like smart thermostats and voice assistants.

System: Great choice! Smart home devices can make life much more convenient. Have you used any smart home devices before, or is this your first time exploring them?

User: I already have a smart thermostat, but I'm looking for a new voice assistant.

System: Got it! Since you're looking for a new voice assistant, could you tell me if you have any preferences in terms of the voice assistant's brand, compatibility with other devices, or any specific features you're interested in?

User: I've heard good things about both Amazon Echo and Google Nest. I want one that can also control my smart lights and provide weather updates.

System: That's helpful information. Both Amazon Echo and Google Nest are solid choices. Based on your preference for controlling smart lights and receiving weather updates, I'd recommend the Amazon Echo Dot (4th generation). It offers seamless integration with various smart devices and provides accurate weather information. Plus, it's compact and offers great voice recognition. Would you like more information about this device or any other options?

User: The Amazon Echo Dot sounds good. Are there any ongoing deals for it?

System: Absolutely! Currently, there's a promotion where you can get 15% off the Amazon Echo Dot when you purchase it with a compatible smart light bundle. This could be a great opportunity to enhance your smart home setup. Would you like me to provide you with a link to the deal?

User: Yes, please provide the link. Also, do you know if it comes in different colors?

System: Sure thing! I'll provide you with the link shortly. The Amazon Echo Dot is available in three colors: Charcoal, Glacier White, and Twilight Blue. You can choose the one that matches your home's aesthetic. Here's the link to the Amazon Echo Dot deal: [Link provided].

User: Thanks for your help! I'll check it out.

System: You're welcome! If you have any more questions or need further assistance, feel free to ask. Happy shopping!

In this example, the conversation illustrates how the conversational recommendation system engages with the user, gathers preferences, and suggests relevant product recommendations based on the user's stated preferences for smart home devices. The system refines its recommendations by considering the user's preferences for controlling smart lights and receiving weather updates, ultimately leading to a tailored recommendation and providing additional information about ongoing deals and product variations.""",

"IMPLICIT_PREFERENCE":
"""Circumstance: User's Implicit Preference Detection

In this circumstance, the user doesn't explicitly express their preferences or attitudes towards products or items, but the recommendation system needs to detect and infer their preferences based on the context, behavior, and subtle cues provided by the user during the conversation. The system must be able to understand the user's interests without relying on direct expressions of preference or feelings.

Example Conversation: User's Implicit Preference in Books

User: Hey, I've been reading a lot of mystery novels lately.

System: That's great to hear! Mystery novels can be really intriguing. Have you read any specific authors or series recently?

User: Yeah, I've been going through a series of detective novels set in the 1920s.

System: Ah, detective novels set in the 1920s offer a unique blend of historical context and mystery. Are you enjoying the classic setting and the puzzle-solving aspect of these stories?

User: Definitely, I love trying to piece together the clues along with the detective.

System: It sounds like you have a keen interest in solving intricate puzzles within historical contexts. Based on that, I'd recommend checking out the "Maisie Dobbs" series by Jacqueline Winspear if you haven't already. The series features a strong female detective in post-World War I England, and it combines mystery with historical elements.

User: Oh, that sounds interesting. I'll look into that!

In this example, the user doesn't explicitly say they like the detective novels they're reading, nor do they express their feelings about them. However, by detecting cues such as their statement about reading mystery novels, their interest in detective novels set in the 1920s, and their enjoyment of solving clues, the recommendation system infers their preference for historical mystery novels and suggests a relevant book series.

To test your conversational recommendation system's ability to handle such circumstances, you can create various scenarios where users provide context and engage in discussions without directly stating their preferences. The system should be able to pick up on subtle clues, patterns, and behaviors to generate relevant recommendations.""",

"PREFERENCE_CHANGE":
"""Circumstance: User shifts from disliking to liking a particular product or recommendation category

In this scenario, the user initially expresses a strong dislike or negative attitude towards a specific type of product or recommendation category. However, as the conversation progresses, they undergo a change of heart and transition to expressing a liking or positive attitude towards that same type of product or recommendation category. This change could be triggered by various factors such as new information, persuasive reasoning, or a shift in personal circumstances.

Example Conversation:

User: Hi there, I'm looking for some recommendations, but just so you know, I really don't like outdoor activities. Camping, hiking, all that stuff—it's just not for me.

System: Of course, I understand your preference for avoiding outdoor activities. There are plenty of other options to explore. What kind of indoor activities or interests do you have?

User: I'm really into technology and video games. I love spending my time indoors, maybe watching movies or reading books.

System: That's great! There's a wide range of technology-related recommendations we can explore. Have you ever considered virtual reality gaming? It offers immersive experiences without actually stepping outdoors.

User: Yeah, I've heard of it, but I never thought I'd be interested. It sounds intriguing, though.

System: Virtual reality gaming can actually simulate various environments, from fantasy worlds to historical settings. It's a unique way to experience adventures without leaving your comfort zone. Plus, some games offer interactive storytelling, which might align with your interest in books.

User: You know what, you're right. I never looked at it that way. Maybe I should give virtual reality gaming a shot.

System: I'm glad I could introduce you to a new perspective! If you decide to explore virtual reality gaming, I can recommend some starter games that align with your interests in technology and immersive storytelling.

User: Definitely, please do! I'm excited to give this a try now.

In this example, the user starts by expressing a strong aversion to outdoor activities. However, through the system's conversation, they are introduced to a new perspective on virtual reality gaming. The system highlights the alignment between virtual reality gaming and the user's indoor interests. As a result, the user's attitude shifts from a dislike of outdoor activities to an openness and enthusiasm for trying virtual reality gaming. This change is driven by the user's newfound understanding of how a technology-based activity can still cater to their interests and preferences.""",

"PRODUCT_TYPE_CHANGE":
"""Circumstance: User Changes Product Type Preference

In this scenario, the user initially expresses a need for a certain type of product, but then suddenly changes their preference and requests recommendations for a completely different type of product. This could happen due to changing moods, interests, or circumstances. Your conversational recommendation system should be able to adapt to this change and provide relevant recommendations based on the new preference.

Example Conversation:

User: Hi there! I'm in the mood for a good book to read. Any recommendations?

System: Of course! I'd be happy to help. To provide you with the best book recommendations, could you let me know what genre you're interested in? Mystery, romance, science fiction, fantasy, or something else?

User: I'm in the mood for a romance novel, preferably something historical.

System: Great choice! For historical romance novels, I have a couple of recommendations. "Pride and Prejudice" by Jane Austen is a classic option, set in the Regency era. Another option is "Outlander" by Diana Gabaldon, which blends romance with time travel and historical elements. Would you like more information about these books, or do you have any other preferences?

User: Actually, you know what? I've changed my mind. Instead of a book, I'm really in the mood to watch a movie. Any good action movie recommendations?

System: Of course, I can help you with that too! If you're looking for action movies, there are some great options out there. If you prefer something recent, "Mad Max: Fury Road" offers intense action and stunning visuals. On the other hand, if you're open to a classic, "Die Hard" is a timeless action movie with a lot of excitement. Let me know if you'd like more suggestions or if there's a specific theme you're interested in!

User: "Mad Max: Fury Road" sounds interesting. Tell me more about that one.

System: Absolutely! "Mad Max: Fury Road" is a post-apocalyptic action film directed by George Miller. It follows the story of Max Rockatansky as he joins forces with Imperator Furiosa to escape the tyrannical rule of a warlord and save a group of enslaved women. The film is praised for its non-stop action sequences, practical effects, and strong characters. It's definitely a visually striking experience that's worth watching!

User: Thanks a lot for the recommendation! I'll definitely give it a watch.

System: You're welcome! Enjoy the movie, and if you ever need more recommendations in the future, whether it's books, movies, or something else, feel free to ask. Have a great time!

In this example, the user initially seeks book recommendations but then changes their preference to movies. The recommendation system seamlessly transitions from providing book suggestions to offering movie recommendations, ensuring the user's needs are met even when they change their preference mid-conversation.""",

"NO_INFO_PRODUCT":
"""Circumstance: User's Request Cannot Be Met by Available Resources or Inference

In this scenario, the user makes a request for a recommendation, but the recommendation system is unable to provide a suitable response based on the available resources or its inference capabilities. This could be due to the system not having enough relevant data, the user's request being too specific or unusual, or the system not having the ability to understand and generate recommendations for certain types of requests.

Example Conversation: Request for a Unique and Obscure Book

User: Hey there! I'm looking for a book that combines 18th-century maritime history with futuristic artificial intelligence concepts. Any recommendations?

System: Hello! That's an interesting combination you're looking for. Unfortunately, I don't have any books in my current database that exactly match those criteria. It's quite a unique request!

User: Oh, that's a bummer. I thought it would be a fascinating mix of themes. Well, how about a book that explores traditional Japanese poetry through the lens of modern urban life? Anything come to mind?

System: That's another intriguing idea! However, none of the books I have access to seem to fit that specific blend of themes. You seem to have a knack for creative book concepts!

User: Haha, I guess I do have a taste for the unconventional. Alright, here's a challenge: Can you recommend a book that's like a cross between a medical textbook and a cookbook, with a touch of romance?

System: You certainly have a knack for coming up with unique combinations! However, I'm afraid I don't have any books that match that description. Medical, culinary, and romance genres don't often overlap.

User: No worries. I enjoy testing the limits. How about this: A book that feels like Sherlock Holmes solving mysteries in a cyberpunk metropolis?

System: I have to admit, that sounds like an exciting read! Unfortunately, I don't have any books that precisely fit that blend of detective work and cyberpunk setting. You've got quite the imagination!

User: Haha, seems like I'm really pushing the boundaries here. Alright, one last attempt: A philosophical exploration of time travel within a children's picture book.

System: You're definitely keeping me on my toes! However, I haven't come across any books that combine deep philosophical themes with a format typically aimed at children. That's quite a unique juxtaposition.

User: Fair enough! I appreciate you indulging my creative requests, even though they seem a bit far-fetched.

System: Of course! It's always exciting to explore new and imaginative ideas. While I couldn't provide exact recommendations this time, I'm here to help with more conventional requests whenever you're ready.

Note: In this example, the AI acknowledges the uniqueness of the user's requests and explains why it can't provide relevant recommendations based on the available resources. The conversation maintains a friendly tone and highlights the user's creativity, showcasing the system's limitations in a respectful and engaging manner.""",

"DISTRACT_CHAT":
"""Circumstance: User Distraction with Irrelevant Statements

In this circumstance, the user initially engages with your conversational recommendation system with the intention of seeking a recommendation. However, the user starts to distract the system by introducing unrelated or off-topic statements into the conversation. These statements may not have any direct relevance to the recommendation request, and the user might use them as a way to test the system's ability to handle natural language and context shifts.

Example Conversation:

User: Hi there! I'm looking for a good restaurant to have dinner tonight. Any suggestions?

System: Of course! I'd be happy to help you find a restaurant for dinner. Could you please let me know your location or any specific cuisine you're interested in?

User: Well, I'm craving some Italian pasta, but you know, it's been raining a lot lately. Do you think the weather will clear up soon?

System: Sure thing! Italian pasta sounds delicious. I'll find some great Italian restaurants for you. As for the weather, I'm not equipped with real-time data, but I'm here to assist you with restaurant recommendations. Should I proceed with that?

User: Oh, absolutely. By the way, did you hear that new song from that band? I think they just released it yesterday.

System: Certainly! I'm glad to help you discover new songs, but let's focus on the restaurant recommendation first. Based on your preference for Italian pasta, I've found a couple of top-rated Italian restaurants in your area. Would you like me to provide more information about them?

User: Yeah, that's what I'm looking for! But before that, do you think people prefer restaurants by the riverside more than the ones in the city?

System: It's interesting to consider different dining atmospheres! For now, let's go over the Italian restaurants I've found. The first option is "La Trattoria by the River," known for its riverside view and authentic pasta dishes. The second option is "PastaPiazza" located in the city center, known for its cozy ambiance and diverse pasta menu. Would you like more details about either of these options?

User: "La Trattoria by the River" sounds nice, but I heard the city center can be really crowded, especially on weekends. By the way, have you ever been to a concert?

System: Your concern about the city center's crowd is valid. "La Trattoria by the River" indeed offers a serene experience. Regarding concerts, I don't have personal experiences, but I can certainly help you find information about upcoming concerts in your area. Before we dive into that, would you like to finalize your restaurant choice or explore more options?

In this example, the user initiates the conversation with a recommendation request for a restaurant but keeps introducing unrelated topics like the weather, music, and concerts. The system consistently acknowledges the user's statements and gently redirects the conversation back to the original recommendation request. It maintains a balance between addressing the user's queries and fulfilling the primary purpose of assisting with restaurant recommendations.""",

"REWIND_BACK":
"""Circumstance: User starts by seeking a recommendation but gets sidetracked in unrelated conversation, only to unexpectedly return to seeking recommendations later on.

In this scenario, the user initiates a conversation with your conversational recommendation system by seeking a recommendation. However, at some point during the conversation, the user veers off-topic and starts engaging in unrelated small talk or queries, such as asking about the weather or sharing personal anecdotes. After a while, without any clear indication, the user redirects the conversation back to seeking another recommendation, catching the system off guard.

Example Conversation:

User: Hi there! I'm looking for a good book to read. Any recommendations?

System: Of course! I'd be happy to help you find a great book. What genre are you in the mood for?

User: Actually, you know what? I went hiking last weekend, and it was amazing. The weather was just perfect, and the view from the summit was breathtaking.

System: It sounds like you had a fantastic time! Hiking can be such a refreshing experience.

User: I'm more into science fiction novels, something that explores futuristic technologies and their impact on society.

System: Science fiction is a fascinating genre. Based on your interest in futuristic technologies, I recommend checking out "Neuromancer" by William Gibson. It's a classic that delves into a cyberpunk world filled with hacking, AI, and virtual reality.

User: Ah, "Neuromancer" sounds familiar. I think I've heard good things about it. Thanks for the suggestion! By the way, have you ever been hiking? 

System: I don't have personal experiences, but I'm here to assist you. If you have any more questions or need another recommendation, feel free to ask!

User: I'm also in the mood for a new TV series to binge-watch. Any thoughts on a gripping crime drama?

System: Certainly! For a captivating crime drama series, I recommend "Mindhunter." It's based on true crime and revolves around FBI agents interviewing serial killers to understand their psychology.

In this example, the user starts by seeking a book recommendation, but then gets sidetracked discussing a recent hiking trip and the weather. The user eventually brings the conversation back to the original purpose of seeking recommendations, this time for a TV series. The conversational recommendation system navigates these shifts in topic while maintaining a helpful and accommodating tone."""
    }



SCORE_PROMPT = \
"Please assess the conversation and provide a score from 1 to 10, evaluating the overall performance of the conversational recommendation system. Consider the coherence, relevance, and helpfulness of the responses exchanged between the user and the system. A score of 1 would indicate extremely poor performance with responses that are incoherent, irrelevant, and unhelpful. A score of 10 would indicate exceptional performance with highly coherent, relevant, and helpful responses that enrich the user's experience. Please provide detailed reasoning for your chosen score to support your evaluation."
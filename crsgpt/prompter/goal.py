
prompt_start = (
    "Your decisions must always be made independently without"
    " seeking user assistance. Play to your strengths as an LLM and pursue"
    " simple strategies with no legal complications."
    ""
)
GOAL_PROMPT=f"Given the chat history below, generate goals for the chatbot to formulate a response that addresses the user's needs and maintains a coherent conversation flow."
RESOURCES_PROMPT=f"Given the chat history and the goal provided, determine the least resources needed for you to generate a response that effectively achieves the goal and addresses the user's needs. You should first consider to make the full use of the given context, then request what you really need, like relevant features from the user's preferences, selected products, or additional chat history to assist in achieving the goal."
USER_PREFERENCE_PROMPT=f"Considering the goal and the resources required, generate a question that should ask the user to gather their preferences effectively. This question should aim to obtain crucial information aligned with the goal and enable the chatbot to provide personalized recommendations based on the user's preferences."
UPDATE_USER_PREFERENCE_PROMPT=f"Given the user's response to the question and the history preferences, update the user's preferences accordingly."
SUMMARY_PROMPT=f"Given the goal, context, and the details of the resource provided, select the key points or items that are relevant to the user's needs. Then, generate a concise summary that effectively captures the essence of the resource while aligning with the goal and maintaining coherence with the conversation."

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



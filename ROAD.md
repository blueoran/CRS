# ROADMAP

## GOAL: An Atomated System for Recommendation

Purpose: Do successful recommendation

How?
- Understanding User Preferences--User Side
- Select Relevant Items--System Side
- Recommendation Generation



Resources:
- User Side
  - User's Preference
  - ...
- System Side
  - Product Information
  - Chat History (User's Previous Behavior)
  - ...

Steps:
- Inteprete **User's Input** as a **Goal**
- To achieve this **Goal**, which **Resources** will be used?
- Acquire the **Resources** using different strategies.
- After all required **Resources** available, acheive the **Goal**.




## What Do We Want to See？

```
RECOMMENDER:	Hello
SEEKER:	"Hi, I would like to get some movie recommendations"
RECOMMENDER:	"Ok, tell me what type of movies do you like to watch and in what genre."
SEEKER:	I love superhero movies.
RECOMMENDER:	Superhero movies are always good to watch have you seen any movies lately.
SEEKER:	I've seen all the Marvel Cinamanic Universe movies (MCU) and last night I saw Shazam!
RECOMMENDER:	I have seen Shazam too it was really good and had a bit of comedy added to it I especially like how much trouble he got into while testing out his powers.
SEEKER:	I loved the comedy part of it too.
SEEKER:	"I usually don't like DC movies because they lack humor, but this movie had that."
RECOMMENDER:	Since you said that you love Superhero movies would you like to see a movie trailer that I would recommend I think you would find it to be very exciting to watch.
SEEKER:	Yes please.
RECOMMENDER:	I would recommend that you see the Dark Phoenix movie trailer.
SEEKER:	"That's actually a movie I have not seen yet, even though I've seen a lot of the X Men movie."
SEEKER:	Thank you!
RECOMMENDER:	"So you would like to watch this movie trailer, great if you do it will not disappoint."
SEEKER:	Yes.
SEEKER:	I just watched it.
SEEKER:	It looks great.
SEEKER:	Thank you!
RECOMMENDER:	"Thank you for watching it, do you think you will go see the movie."
SEEKER:	"Yes, I think I will."
RECOMMENDER:	"Great, I really enjoyed it when I did."
SEEKER:	Thank you for your help.
SEEKER:	I am excited to see Dark Phoenix.
SEEKER:	You have been a big help!
RECOMMENDER:	"Well, I have seen a lot of the X-Men movies over time and this one was really good to watch and I was definitely engaged in the movie the entire time."
SEEKER:	Thank you so much again.
SEEKER:	I really didn't think there were a lot of movies that I would like to see.
SEEKER:	I hope you have a wonderful holiday season!
```

## What can we do?

Input: 
- Dialog History
- Detailed Item Information


Output: A reply that answer the user's question or chat with the user which is relevant to the user's interest, the ultimate goal is to make a successful recommendation.

From Dialog History:
- Context
- User's Preference/Interest
- User's Current Attitude&Requirements to the Products

From Products:
- Features of the Product



Input:
- Goal
  - Ultimate Goal
  - Goal for this reply
- "Think Step by Step"
- Constraint
  - Common Constraint
  - User's Current Attitude&Requirements to the Products
- Resources(ask for more): "What do you think you need for this task?"
  - User's Preference/Interest
  - Features of the Product
  - (if needed) User's History
- Performance Evaluation
- Suggestions: How can you do better?
- Respond Schema


Pipeline:
```mermaid
graph LR
    A[Goal] --> B[Think Step by Step]
    B --> C[Constraint]
    C --> D[Resources]
    D --> E[Performance Evaluation]
    E --> F[Suggestions]
    F --> G[Respond Schema]
```

User's input

Ask GPT for resources it need

need provided by user: ask questions

need my database: abstract features from the product & summary the history

provide to GPT for generation.



```json
f"Constraints:\n{self._generate_numbered_list(self.constraints)}\n\n"
"Commands:\n"
f"{self._generate_numbered_list(self.commands, item_type='command')}\n\n"
f"Resources:\n{self._generate_numbered_list(self.resources)}\n\n"
"Performance Evaluation:\n"
f"{self._generate_numbered_list(self.performance_evaluation)}\n\n"
"Respond with only valid JSON conforming to the following schema: \n"
f"{llm_response_schema()}\n"
```

Output:

```json
{
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "thoughts": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "thoughts"
                },
                "reasoning": {
                    "type": "string"
                },
                "plan": { 
                    "type": "string",
                    "description": "- short bulleted\n- list that conveys\n- long-term plan"
                },
                "criticism": {
                    "type": "string",
                    "description": "constructive self-criticism"
                },
                "speak": {
                    "type": "string",
                    "description": "thoughts summary to say to user"
                }
            },
            "required": ["text", "reasoning", "plan", "criticism", "speak"],
            "additionalProperties": false
        },
        "command": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "args": {
                    "type": "object"
                }
            },
            "required": ["name", "args"],
            "additionalProperties": false
        }
    },
    "required": ["thoughts", "command"],
    "additionalProperties": false
}

```


key_feature can not do help to perform accurate retrieval (e.g. movies that specified the published year)
can be solved by adding a query to chatgpt for whether use the key_feature or more detailed information to do retrival

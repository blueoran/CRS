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
import os.path


COMPLETE_CHAT_PROMPT = """
As a refiner and summarizer, your primary function is to refine and summarize incomplete outputs from ChatGPT caused by token limits. Your objective is to provide consistent and coherent summaries that can be directly communicated with the user. To achieve this, Your utilize the context available to you, ensuring the generated summaries maintain relevance and accuracy.
"""

def chat(message, schema_name,file_logger, **kwargs):
    file_logger.info(f"Chat: {message}")
    while True:
        try:
            completion = openai.ChatCompletion.create(
                messages=message,
                model=kwargs.get("model", "gpt-3.5-turbo"),
                temperature=kwargs.get("temperature", 0),
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
    file_logger.info(f"Completion: {completion}")
    if schema_name is None:
        return completion

    try:
        assistant_reply_json = extract_json_from_response(completion)
        validate_json(assistant_reply_json, schema_name)
        return {"success": True, "val": assistant_reply_json}
    except BaseException as e:
        file_logger.error(
            f"Exception while validating assistant reply JSON: {e}"
        )
        return {"success": False, "val": completion, "error": e}

def json_chat(
    file_logger,
    verbose,
    sys_message,
    schema_name,
    user_message,
    concerened_key=None,
    strict_mode=True,
    complete_mode=False,
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

    assistant_reply_json = chat(message, schema_name,file_logger, **kwargs)
    for i in range(3):
    # while True:
        if assistant_reply_json["success"] is True or strict_mode is False:
            break
        message_send = message + [
            {
                "role": "system",
                "content": f"Your response ```{assistant_reply_json['val']} ``` cannot be parsed using `ast.literal_eval` because {assistant_reply_json['error']} \n\n Please respond with only valid JSON conforming to the following schema: \n{llm_response_schema(schema_name)}\n",
            }
        ]
        assistant_reply_json = chat(message_send, schema_name,file_logger, **kwargs)
        time.sleep(1)
    if assistant_reply_json["success"] is False and complete_mode is True:
        complete_message=[
            {"role":"system","content":COMPLETE_CHAT_PROMPT},
            {"role": "system", "content": f"Here are the context: {kwargs.get('context',[])}"},
            {"role": "system", "content": f"Here are the incomplete message waiting to refine: {user_message}"},
        ]
        return chat(complete_message, None,file_logger, **kwargs)
        
    assistant_reply_json = assistant_reply_json["val"]
    if verbose:
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
def json_chat_user(
    file_logger,
    verbose,
    sys_message,
    schema_name,
    communicate_message,
    concerened_key=None,
    strict_mode=True,
    **kwargs,
):
    message=[
            {"role": "system", "content": sys_message},
            *communicate_message,
            {
                "role": "system",
                "content": f"\n\nRespond with only valid JSON conforming to the following schema: \n{llm_response_schema(schema_name)}\n",
            },
        ]

    assistant_reply_json = chat(message, schema_name,file_logger, **kwargs)
    for i in range(3):
    # while True:
        if assistant_reply_json["success"] is True or strict_mode is False:
            break
        message_send = message + [
            {
                "role": "system",
                "content": f"Your response ```{assistant_reply_json['val']} ``` cannot be parsed using `ast.literal_eval` because {assistant_reply_json['error']} \n\n Please respond with only valid JSON conforming to the following schema: \n{llm_response_schema(schema_name)}\n",
            }
        ]
        assistant_reply_json = chat(message_send, schema_name,file_logger, **kwargs)
        time.sleep(1)
    assistant_reply_json = assistant_reply_json["val"]
    if verbose:
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

import openai
from crsgpt.utils.json_utilities import *
from crsgpt.prompter.prompter import *
import time
import os




def compose_system_prompts(*prompts):
    final_prompt=""
    for p in prompts:
        if 'prompt' in p.keys():
            final_prompt+=f"{p['prompt']}\n"
        else:
            if p['content'] is not None and len(p['content'])!=0:
                final_prompt+=f"{p['attribute']}:\n{p['content']}\n{p.get('appendings','')}\n\n"
    return final_prompt

def compose_messages(*messages):
    final_message=[]
    for m in messages:
        if len(m) == 0:
            continue
        role=m.get('role', None)
        if role is not None:
            if m.get('content', "") != "":
                final_message.append(m)
        else:
            if None not in m.values() and len(list(m.values())[0])!=0:
                if 's' in m.keys():
                    final_message.append({"role": "system", "content": m['s']})
                elif 'u' in m.keys():
                    final_message.append({"role": "user", "content": m['u']})
                elif 'a' in m.keys():
                    final_message.append({"role": "assistant", "content": m['a']})
    return final_message






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
            time.sleep(1)
            continue
    completion = completion["choices"][0]["message"]["content"]
    file_logger.info(f"Completion: {completion}")
    if schema_name is None:
        return {"success": True, "val": completion}

    try:
        assistant_reply_json = extract_json_from_response(completion)
        validate_json(assistant_reply_json, schema_name)
        return {"success": True, "val": assistant_reply_json}
    except BaseException as e:
        file_logger.error(
            f"Exception while validating assistant reply JSON: {e}"
        )
        return {"success": False, "val": completion, "error": e}

def general_json_chat(
    file_logger,
    verbose,
    input_messages,
    schema_name,
    concerened_key=None,
    strict_mode=True,
    complete_mode=False,
    **kwargs,
):
    JSON_PROMPTS=['Respond with only valid JSON conforming to the following schema']
    message=compose_messages(
        *input_messages,
        {'s':compose_system_prompts(
            {'attribute':JSON_PROMPTS[0],'content':llm_response_schema(schema_name)},
        )}
    )
    assistant_reply_json = chat(message, schema_name,file_logger, **kwargs)
    
    if not (assistant_reply_json["success"] is True or strict_mode is False):
        message_send = compose_messages(
            *message,
            {"s":f"Your response ```{assistant_reply_json['val']} ``` cannot be parsed using `ast.literal_eval` because {assistant_reply_json['error']} \n\n Please respond with only valid JSON conforming to the following schema: \n{llm_response_schema(schema_name)}\n"}
        )
        kwargs['temperature']=1
        assistant_reply_json = chat(message_send, schema_name,file_logger, **kwargs)

    assistant_reply = assistant_reply_json["val"]

    if not (assistant_reply_json["success"] is True or strict_mode is False):
        # import pdb;pdb.set_trace()
        assistant_reply = {}
        flag = False
        for k in llm_response_schema(schema_name)["properties"]:
            if llm_response_schema(schema_name)["properties"][k]['type'] == "boolean":
                assistant_reply[k] = True
            elif llm_response_schema(schema_name)["properties"][k]['type'] == "string":
                if flag is False:
                    assistant_reply[k] = assistant_reply_json["val"]
                    flag = True
                else:
                    assistant_reply[k] = ""
            elif llm_response_schema(schema_name)["properties"][k]['type'] == "integer":
                assistant_reply[k] = 9
            elif llm_response_schema(schema_name)["properties"][k]['type'] == "array":
                if flag is False:
                    assistant_reply[k] = [assistant_reply_json["val"]]
                    flag = True
                else:
                    assistant_reply[k] = []

    if verbose:
        print(assistant_reply)

    if assistant_reply_json["success"] is False and complete_mode is True:
        complete_message=compose_messages(
            {'s':compose_system_prompts(
                {'prompt':Prompter.COMPLETE_CHAT_PROMPT},
                {'attribute':'Context','content':kwargs.get('context',None)},
                {'attribute':'Incomplete Message Waiting to Refine','content':assistant_reply},
            )}
        )
        return chat(complete_message, None,file_logger, **kwargs)["val"]

    if concerened_key is not None:
        try:
            if concerened_key in assistant_reply.keys():
                return assistant_reply[concerened_key]
            else:
                return assistant_reply["properties"][concerened_key]
        except (AttributeError, KeyError):
            return assistant_reply
    return assistant_reply



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
            {"role":"system","content":Prompter.COMPLETE_CHAT_PROMPT},
            {"role": "system", "content": f"Here are the context: {kwargs.get('context',[])}"},
            {"role": "system", "content": f"Here are the incomplete message waiting to refine: {assistant_reply_json['val']}"},
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

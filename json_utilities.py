"""Utilities for the json_fixes package."""
import ast
import json
import os.path
from typing import Any
import re

from jsonschema import Draft7Validator


LLM_DEFAULT_RESPONSE_FORMAT = "llm_response_format_1"


def extract_json_from_response(response_content: str) -> dict:
    # Sometimes the response includes the JSON in a code block with ```
    response_content=response_content.replace('true','True').replace('false','False').replace('\n','')
    # contents=response_content.strip('\n }{').split(',')
    # # print(contents)
    # content=[]
    # neirong=""
    # attr=""
    # for c in contents:
    #     if len(c.split(':'))==1:
    #         neirong+=','+c
    #         continue
    #     ori_attr=attr
    #     ori_neirong=neirong
    #     attr=c.split(':')[0].split(' ')[-1]
    #     neirong=':'.join(c.split(':')[1:])
    #     ori_attr=ori_attr.strip('\'\" \n').replace('\"','').replace('\'','')
    #     ori_neirong+=' '.join(c.split(':')[0].split(' ')[:-1])
    #     ori_neirong=ori_neirong.strip('\'\" \n').replace('\"','').replace('\'','')
    #     if ori_attr!="":
    #         content.append(f'\"{ori_attr}\": \"{ori_neirong}\"')
    # attr=attr.strip('\'\" \n').replace('\"','').replace('\'','')
    # neirong=neirong.strip('\'\" \n').replace('\"','').replace('\'','')
    # content.append(f'\"{attr}\": \"{neirong}\"')
    # response_content='{ '+', '.join(content)+' }'
    response_content=response_content.replace('\"True\"','True').replace('\"False\"','False')
    # .replace('\'','\"')
    # response_content = re.sub(r"':([^']+)'", r'\1', response_content)
    # print(response_content)
    if response_content.startswith("```") and response_content.endswith("```"):
        # Discard the first and last ```, then re-join in case the response naturally included ```
        response_content = "```".join(response_content.split("```")[1:-1])

    # response content comes from OpenAI as a Python `str(content_dict)`, literal_eval reverses this
    try:
        # return json.loads(response_content)
        return ast.literal_eval(response_content)
    except BaseException as e:
        # print(f"Error parsing JSON response with literal_eval {e}")
        # TODO: How to raise an error here without causing the program to exit?
        raise e


def llm_response_schema(
    schema_name: str = LLM_DEFAULT_RESPONSE_FORMAT,
) -> dict[str, Any]:
    filename = os.path.join(os.path.dirname(__file__), f"json_schema/{schema_name}.json")
    with open(filename, "r") as f:
        return json.load(f)


def validate_json(
    json_object: object, schema_name: str = LLM_DEFAULT_RESPONSE_FORMAT
) -> bool:
    """
    :type schema_name: object
    :param schema_name: str
    :type json_object: object

    Returns:
        bool: Whether the json_object is valid or not
    """
    schema = llm_response_schema(schema_name)
    validator = Draft7Validator(schema)

    if errors := sorted(validator.iter_errors(json_object), key=lambda e: e.path):
        # for error in errors:
        #     print(f"JSON Validation Error: {error}")
            # logger.error(f"JSON Validation Error: {error}")

        return False

    # print("The JSON object is valid.")
    # logger.debug("The JSON object is valid.")

    return True

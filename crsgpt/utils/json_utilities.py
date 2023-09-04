"""Utilities for the json_fixes package."""
import ast
import json
from typing import Any

from jsonschema import Draft7Validator


LLM_DEFAULT_RESPONSE_FORMAT = "llm_response_format_1"


def extract_json_from_response(response_content: str) -> dict:
    # Sometimes the response includes the JSON in a code block with ```
    response_content=response_content.replace('true','True').replace('false','False').replace('\n','')
    response_content=response_content.replace('\"True\"','True').replace('\"False\"','False')
    if response_content.startswith("```") and response_content.endswith("```"):
        response_content = "```".join(response_content.split("```")[1:-1])

    try:
        return ast.literal_eval(response_content)
    except BaseException as e:
        raise e


def llm_response_schema(
    schema_name: str = LLM_DEFAULT_RESPONSE_FORMAT,
) -> dict[str, Any]:
    if schema_name==None:
        return None
    filename = f"./json_schema/{schema_name}.json"
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
        return False

    return True

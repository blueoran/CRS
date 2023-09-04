import re

def find_first_number(input_string):
    if type(input_string) != str:
        return input_string
    pattern = r'\d+'  # 正则表达式匹配数字
    match = re.search(pattern, input_string)
    if match:
        return int(match.group())  # 将匹配到的数字转换为整数
    else:
        return -1  # 如果找不到数字，返回None或者其他你认为合适的值
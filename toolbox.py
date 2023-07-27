import markdown
import mdtex2html
import threading
import importlib
import traceback
import inspect
import re
from latex2mathml.converter import convert as tex2mathml
from functools import wraps, lru_cache
import agg

############################### 插件输入输出接驳区 #######################################


# class ChatBotWithCookies(list):
#     def __init__(self, cookie):
#         self._cookies = cookie

#     def write_list(self, list):
#         for t in list:
#             self.append(t)

#     def get_list(self):
#         return [t for t in self]

#     def get_cookies(self):
#         return self._cookies


def ArgsGeneralWrapper(f):
    """
        装饰器函数，用于重组输入参数，改变输入参数的顺序与结构。
    """

    def decorated(txt, chatbot, history, *args):

        # 引入一个有cookie的chatbot

        yield from f(txt, chatbot, history, *args)
    return decorated


def update_ui(chatbot, history, msg='正常', **kwargs):  # 刷新界面
    """
    刷新用户界面
    """

    yield chatbot, history, msg
############################### ################## #######################################
##########################################################################################


def format_io(self, y):
    """
        将输入和输出解析为HTML格式。将y中最后一项的输入部分段落化，并将输出部分的Markdown和数学公式转换为HTML格式。
    """
    if y is None or y == []:
        return []
    i_ask, gpt_reply = y[-1]
    i_ask = text_divide_paragraph(i_ask)  # 输入部分太自由，预处理一波
    gpt_reply = close_up_code_segment_during_stream(
        gpt_reply)  # 当代码输出半截的时候，试着补上后个```
    y[-1] = (
        None if i_ask is None else markdown.markdown(
            i_ask, extensions=['fenced_code', 'tables']),
        None if gpt_reply is None else markdown_convertion(gpt_reply)
    )
    return y


def find_free_port():
    """
        返回当前系统中可用的未使用端口。
    """
    import socket
    from contextlib import closing
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def chat_with_awesome(inputs, chatbot, history=[], stream=True):

    if stream:
        raw_input = inputs
        chatbot.append((inputs, ""))
        # 刷新界面
        yield from update_ui(chatbot=chatbot, history=history, msg="等待响应")

    history.append(inputs)
    history.append(" ")

    # import random
    gpt_replying_buffer = agg.communicate(inputs)
    history[-1] = gpt_replying_buffer
    chatbot[-1] = (history[-2], history[-1])
    # 刷新界面
    yield from update_ui(chatbot=chatbot, history=history, msg="Updated")

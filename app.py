from crsgpt.agent.agent import *
import logging
import argparse
from crsgpt.evaluator.fake_user import *
from crsgpt.evaluator.pure_gpt import *
from crsgpt.component.product import *
from crsgpt.component.preference import *
import openai
from flask import Flask, render_template, request
import pandas as pd
import logging

app = Flask(__name__)


top_K=50
openai.api_key="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"
update_product=False
product_detail_path="./data/summary.json"
log_level=logging.DEBUG
explicit=False
verbose=False
testcase=None

product_dict = {
    './data/2022_movie.csv':'Title',
    './data/2022_book.csv':'Name_of_the_Book',
    './data/2023_phone.csv':'name',
    './data/2023_movie.csv':'name'
}


# 存储用户实例对象的字典
user_instances = {}
user_history = {}
user_context = {}
log_file={}

def create_instance(user_id):
    file_logger = logging.getLogger("file_logger")
    file_logger.setLevel(log_level)
    file_hander = logging.FileHandler(f"./logs/{user_id}.log")
    file_logger.addHandler(file_hander)

    user_instances[user_id] = {
        'log_file':file_logger,
        'product':Product(top_K,file_logger,product_detail_path,update_product,verbose,product_dict),
        'preference':Preference(file_logger,verbose),
        'gpt_rec':PureGPT(file_logger)
    }
    user_instances[user_id]['evaluator'] = Evaluator(file_logger,user_instances[user_id]['product'].product_type_set,verbose)
    user_instances[user_id]['rec'] = Agent(user_instances[user_id]['product'],user_instances[user_id]['preference'],user_instances[user_id]['evaluator'],file_logger,explicit,verbose)

    user_history[user_id] = []
    user_context[user_id] = []


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 获取用户输入的字符串
        user_input = request.form['user_input']
        
        # 获取当前用户的唯一标识（可以使用session、cookie或其他方式）
        # user_id = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)   # 使用IP地址作为标识
        user_id = request.access_route[0]
        # 如果当前用户没有实例对象，则创建一个新的实例对象
        if user_id not in user_instances:
            create_instance(user_id)
        user_history[user_id].append({'role': 'user', 'content': f'[[User]]: {user_input}'})
        user_context[user_id].append({'role': 'user', 'content': user_input})
        # 调用class的interactive方法，传入用户输入，获取返回结果
        result_rec = user_instances[user_id]['rec'].user_interactive(user_input)
        result_gpt = user_instances[user_id]['gpt_rec'].interactive(user_context[user_id])
        user_history[user_id].append({'role': 'class_rec', 'content': f'[[Recommender]]: {result_rec}'})
        user_history[user_id].append({'role': 'class_gpt', 'content': f'[[ChatGPT]]: {result_gpt}'})
        user_context[user_id].append({"role":"assistant","content":result_rec})
        
        return render_template('index.html', result=(result_rec,result_gpt), history=user_history[user_id])
 # Check if the 'reset' query parameter is present in the URL
    if request.args.get('reset') == 'true':
        # Reset the user's history and delete the user instancelog_file=f"./logs/{user_id}.log"
        # user_id = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)   
        user_id = request.access_route[0]
        if user_id not in user_instances:
            create_instance(user_id)
        file_logger = user_instances[user_id]['log_file']
        user_instances[user_id] = {
            'log_file':file_logger,
            'product':Product(top_K,file_logger,product_detail_path,update_product,verbose,product_dict),
            'preference':Preference(file_logger,verbose),
            'gpt_rec':PureGPT(file_logger)
        }
        user_instances[user_id]['evaluator'] = Evaluator(file_logger,user_instances[user_id]['product'].product_type_set,verbose)
        user_instances[user_id]['rec'] = Agent(user_instances[user_id]['product'],user_instances[user_id]['preference'],user_instances[user_id]['evaluator'],file_logger,explicit,verbose)

        user_history[user_id] = []
        user_context[user_id] = []

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)

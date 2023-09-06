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
import signal

app = Flask(__name__)

#### Configurations
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
if not os.path.exists('./logs'):
    os.makedirs('./logs')
####

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
    }
    user_instances[user_id]['evaluator'] = Evaluator(file_logger,user_instances[user_id]['product'].product_type_set,verbose)
    user_instances[user_id]['rec'] = Agent(user_instances[user_id]['product'],user_instances[user_id]['preference'],user_instances[user_id]['evaluator'],file_logger,explicit,verbose)

    user_history[user_id] = []
    user_context[user_id] = []


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['user_input']
        user_id = request.access_route[0]
        print(user_id)

        if user_id not in user_instances:
            create_instance(user_id)
        
        result_rec = user_instances[user_id]['rec'].user_interactive(user_input).replace('\n','<br />')
        user_history[user_id].append({'role': 'user', 'content': f'[[User]]: {user_input}'})
        user_history[user_id].append({'role': 'class_rec', 'content': f'[[Recommender]]: {result_rec}'})
        return render_template('index.html', result=(f'[[Recommender]]: {result_rec}'), history=user_history[user_id])


    if request.args.get('reset') == 'true':
        user_id = request.access_route[0]
        create_instance(user_id)

    return render_template('index.html')

if __name__ == '__main__':
    app.run()

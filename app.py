from crsgpt.agent.agent import *
import logging
import argparse
from crsgpt.evaluator.fake_user import *
from crsgpt.evaluator.pure_gpt import *
from crsgpt.component.product import *
from crsgpt.component.preference import *
import openai
from flask import Flask, render_template, request, make_response
import pandas as pd
import logging
import uuid
import signal
from timeout_decorator import timeout
from timeout_decorator.timeout_decorator import TimeoutError
import time
import signal
import requests
import threading


app = Flask(__name__)
app.config['PERMANENT_SESSION_LIFETIME'] = 1 
#### Configurations
top_K=5
head_K=50
update_product=False
embedding_cache_path="./data/recommendations_embeddings_cache.pkl"
log_level=logging.DEBUG
explicit=False
verbose=False
testcase=None
api_init()

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
    file_logger = logging.getLogger(f'{user_id}')
    file_logger.setLevel(log_level)
    file_handler_exists = any(isinstance(handler, logging.FileHandler) and (f"logs/{user_id}.log" in handler.baseFilename) for handler in file_logger.handlers)
    if not file_handler_exists:
        file_handler = logging.FileHandler(f"./logs/{user_id}.log")
        file_logger.addHandler(file_handler)


    user_instances[user_id] = {
        'log_file': file_logger,
        'product': Product(head_K, file_logger, embedding_cache_path, top_K, update_product, verbose, product_dict),
        'preference': Preference(file_logger, verbose),
    }
    user_instances[user_id]['evaluator'] = Evaluator(file_logger, user_instances[user_id]['product'].product_type_set, verbose)
    user_instances[user_id]['rec'] = Agent(user_instances[user_id]['product'], user_instances[user_id]['preference'],
                                           user_instances[user_id]['evaluator'], file_logger, explicit, verbose)

    user_history[user_id] = []
    user_context[user_id] = []


worker_function_result = None
def worker_function(user_id, user_input):
    global worker_function_result
    try:
        result_rec = user_instances[user_id]['rec'].user_interactive(user_input).replace('\n', '<br />')
    except Exception as e:
        result_rec = f"[[Recommender]]: An error occurred: {str(e)}"
    worker_function_result = result_rec

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['user_input']
        user_id = request.cookies.get('user_id')
        if user_id is None:
            user_id = str(uuid.uuid4())
            create_instance(user_id)
        else:
            if user_id not in user_instances:
                create_instance(user_id)
        print(user_id)

        try:
            # Create a thread to execute the worker function
            from copy import copy
            copied_user_instances = copy(user_instances[user_id]['rec'])
            worker_thread = threading.Thread(target=worker_function, args=(user_id, user_input))
            worker_thread.start()
            worker_thread.join(timeout=50)  # Set a timeout for the thread execution
            if worker_thread.is_alive():
                # Handle the timeout here
                result_rec = f"Sorry, the response is taking too long. Please try again."
                user_instances[user_id]['rec'] = copied_user_instances
                
            else:
                # Thread has completed, get the return value
                result_rec = worker_function_result
                # Now you can use 'response' as needed
            user_history[user_id].append({'role': 'user', 'content': f'[[User]]: {user_input}'})
            user_history[user_id].append({'role': 'class_rec', 'content': f'[[Recommender]]: {result_rec}'})
            response = make_response(render_template('index.html', result=(f'[[Recommender]]: {result_rec}'), history=user_history[user_id]))
            response.set_cookie('user_id', user_id)
            return response

        except Exception as e:
            print(f"An error occurred: {str(e)}")

    if request.args.get('reset') == 'true':
        user_id = request.cookies.get('user_id')
        if user_id:
            create_instance(user_id)

    return render_template('index.html')


if __name__ == '__main__':
    app.run()
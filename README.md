# Conversational Recommender System using ChatGPT

## Requirements

```
python=3.10
pandas
numpy
openai
fuzzywuzzy
jsonschema
flask=2.3.2
```

## Usage

under `CRS` directory:

```bash
$ python main.py (--verbose) (--explicit)
```

### Arguments
```bash
options:
  -h, --help            show this help message and exit
  --top_K TOP_K         top K products to recommend
  --api_key API_KEY     openai api key
  --log_file LOG_FILE   log file path
  --update_product      whether to update product details
  --product_detail_path PRODUCT_DETAIL_PATH
                        product details json save path
  --log_level LOG_LEVEL
                        log level
  --explicit            whether to explicitly show the thinking of the recommendation gpt
  --verbose             whether to show the process of the recommendation

  --pure_gpt            whether to use pure gpt to generate response for comparison
  --product_gpt         whether to use product gpt to generate response for comparison

```

> Note: Type `Ctrl+Z` to suspend the process then kill it mannuanly if it stuck during the process instead of `Ctrl+C`, as the Error Handling part has not been perfectly implemented yet.

## Deployment
The web is based on Flask. To deploy the web, run the following command under `CRS` directory:
```bash
$ python app.py
```

P.S. I used ngrok as reverse proxy to expose my local server to the Internet.

### Interface

In `app.py`, The recommendation agent instance is created for each user, distinguished by the user's IP address. Each instance is saved in the `user_instances` dictionary.

`create_instance(user_id)`: initialize a user instance and save it to the user_instances dictionary


```python
user_instances (Dict): {
  user_id: {
    'rec':Agent,

    'log_file':file_logger,
    'product':Product,
    'preference':Preference,
    'gpt_rec':PureGPT,
    'evaluator':Evaluator,
    ......
  }
}
```

We can use `Agent.user_interactive(user_input)` to get the response of the recommendation agent. The `user_history` dictionary is used to save the conversation history of each user to show in the web page as shown in the following code:


```python
if user_id not in user_instances:
    create_instance(user_id)
user_history[user_id].append({'role': 'user', 'content': f'[[User]]: {user_input}'})
result_rec = user_instances[user_id]['rec'].user_interactive(user_input)
user_history[user_id].append({'role': 'class_rec', 'content': f'[[Recommender]]: {result_rec}'})
```
# Conversational Recommender System using ChatGPT

## Requirements
- Using `python=3.10.9`
- Install the requirements by running the following command under `CRS` directory:
  ```bash
  $ pip install -r requirements.txt
  ```
- List api-keys that you want the program to use in `api.cfg`
- Move your product data in `.csv` format in `data`, modify the `product_dict` in `main.py` to include the products you want to recommend. The key is the file path and the value is the column name that represents the product's name.

## Usage

under `CRS` directory:

```bash
$ python main.py
```

### Options
#### General Options
- `-h, --help`: Show this help message and exit.
- `--log_file LOG_FILE`: Define the log file path. The default path is "./logs/new.log".
- `--update_product`: Flag to decide whether to update product details or not. This is not enabled by default.
- `--log_level LOG_LEVEL`: Set the logging level. Default is set to DEBUG.
- `--explicit`: Enable this to explicitly show the thinking process of the recommendation GPT. This is not enabled by default.
- `--verbose`: Enable this to show the detailed process of the recommendation. This is not enabled by default.

#### Comparison Options
- `--pure_gpt`: Enable to use a pure GPT model for generating response for comparison.
- `--product_gpt`: Enable to use a product-specific GPT model for generating response for comparison.
- `--web`: Enable this to activate web-based features.
- `--hallucination`: Enable this to include hallucination-detection features in the GPT responses.

#### Additional Options
- `--top_K TOP_K`: Specify the top K products to recommend. The default value is 5.
- `--head_K`: Specify the head K products in the system. The default value is 50.
- `--embedding_cache_path`: Set the path for caching embeddings. The default is `./data/recommendations_embeddings_cache.pkl`.
- `--testcase`: Provide specific test cases. Default is None.


## Deployment
The web is based on Flask. To deploy the web, run the following command under `CRS` directory:
```bash
$ python app.py
```
# Conversational Recommender System using ChatGPT

## Usage

under `CRS` directory:

```bash
$ conda activate openai
$ python3 main.py (--verbose) (--explicit)
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

```

> Note: Type `Ctrl+Z` to suspend the process then kill it mannuanly if it stuck during the process instead of `Ctrl+C`, as the Error Handling part has not been perfectly implemented yet.

## TODO

- A better product retrieval method
- More structured code for further development (more functionalities and attributes)
- Role-Goal-Task based dialogue management
- Error Handling
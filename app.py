from flask import Flask, render_template, request
import agent
import pandas as pd
import os
import openai
import logging
import numpy as np
import json
import tqdm
import re
import argparse


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    # parser = argparse.ArgumentParser(description = 'test')

    # parser.add_argument('--top_K', type=int, default=5, help='top K products to recommend')
    # parser.add_argument('--api_key', type=str, default="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",help='openai api key')
    # parser.add_argument('--log_file', type=str, default="./log/chatgpt.log",help='log file path')
    # parser.add_argument('--update_product', action='store_true', default=False,help='whether to update product details')
    # parser.add_argument('--product_detail_path', type=str, default="./data/summary.json",help='product details json save path')
    # parser.add_argument('--log_level', type=int, default=logging.DEBUG,help='log level')
    # parser.add_argument('--explicit', action='store_true',help='whether to explicitly show the thinking of the recommendation gpt')
    # parser.add_argument('--verbose', action='store_true',help='whether to show the process of the recommendation')

    # args = parser.parse_args()

    # Load your API key from an environment variable or secret management service
    top_K=20


    products=pd.read_csv('./data/Inspired/data/movie_database_small.tsv',sep='\t')
    products_1=products.iloc[:top_K].copy()

    products=pd.read_csv('./data/phones_data.csv')
    products['title']=products['brand_name']+' '+products['model_name']
    products.drop(columns=['Unnamed: 0'],inplace=True)
    products_2=products.iloc[:top_K].copy()

    products=pd.read_csv('./data/StockX-Data-Contest-2019-3.csv')
    products['title']=products['Sneaker Name']
    products_3=products.iloc[:top_K].copy()

    rec=agent.Agent([products_1,products_2,products_3],'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',update_product=True,explicit=True,verbose=True)
    if request.method == 'POST':
        user_input = request.form['user_input']
        recommendation = rec.user_interactive(user_input)  # 调用您的智能推荐系统函数
        return render_template('index.html', recommendation=recommendation)
    return render_template('index.html')



if __name__ == '__main__':
    app.run()

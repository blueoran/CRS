import transformers
import torch
import pandas as pd
import os
import openai
import logging
import numpy as np
import json
import tqdm
import re
import logging
import agent


# Load your API key from an environment variable or secret management service
openai.proxy = "socks5h://localhost:7890"

top_K=10

products=pd.read_csv('./data/Inspired/data/movie_database_small.tsv',sep='\t')
need_cols=['title','year','trailer_duration','actors','awards','box_office','country','director','genre','imdb_votes','language','long_plot','movie_runtime','production','rated','rating','release_date','writer','youtube_comment','youtube_dislike','youtube_favorite','youtube_like','youtube_view']
products_1=products[need_cols].iloc[:top_K].copy()

products=pd.read_csv('./data/phones_data.csv')
products['title']=products['brand_name']+' '+products['model_name']
products.drop(columns=['Unnamed: 0'],inplace=True)
products_2=products.iloc[:top_K].copy()

products=pd.read_csv('./data/StockX-Data-Contest-2019-3.csv')
products['title']=products['Sneaker Name']
products_3=products.iloc[:top_K].copy()

products=pd.read_csv('./data/2023_phone.csv')
products['title']=products['name']
products_4=products.iloc[:top_K].copy()

products=pd.read_csv('./data/2023_movie.csv')
products['title']=products['name']
products_5=products.iloc[:top_K].copy()


top_K=10
api_key="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"

update_product=False
product_detail_path="./data/summary.json"
log_level=logging.DEBUG
explicit=False
verbose=False
# explicit=True
# verbose=True
# 存储用户实例对象的字典
user_instances = {}
user_history = {}
log_file=f"./logs/frontend.log"
rec = agent.Agent([products_1,products_2,products_3,products_4,products_5],api_key,log_file,update_product,product_detail_path,log_level,explicit,verbose) # 在这里实例化你的class

def communicate(user_input):
    return rec.user_interactive(user_input)


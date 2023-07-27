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
import fake_user
import pure_gpt

parser = argparse.ArgumentParser(description = 'test')

parser.add_argument('--top_K', type=int, default=10, help='top K products to recommend')
parser.add_argument('--api_key', type=str, default="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",help='openai api key')
parser.add_argument('--log_file', type=str, default="./logs/chatgpt.log",help='log file path')
parser.add_argument('--update_product', action='store_true', default=False,help='whether to update product details')
parser.add_argument('--product_detail_path', type=str, default="./data/summary.json",help='product details json save path')
parser.add_argument('--log_level', type=int, default=logging.DEBUG,help='log level')
parser.add_argument('--explicit', action='store_true',help='whether to explicitly show the thinking of the recommendation gpt')
parser.add_argument('--verbose', action='store_true',help='whether to show the process of the recommendation')

args = parser.parse_args()

# Load your API key from an environment variable or secret management service
top_K=args.top_K


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

if __name__=='__main__':
    print(args)
    rec=agent.Agent([products_1,products_2,products_3,products_4,products_5],args.api_key,args.log_file,args.update_product,args.product_detail_path,args.log_level,args.explicit,args.verbose)
    gpt_rec=pure_gpt.PureGPT(rec.file_logger)
    user=fake_user.User(rec.file_logger,list(rec.product_type_set),verbose=args.verbose,level="extreme")
    user_input=""
    rec_response=""
    context=[]
    while True:
        # user_input=user.interacitive(rec_response)
        # print(f'[[User]]: {user_input}')
        user_input=input("User: ")
        context.append({"role":"user","content":user_input})
        rec_response=rec.user_interactive(user_input)
        print(f'[[Rec]]: {rec_response}')
        gpt_rec_response=gpt_rec.interacitive(context)
        print(f'[[ChatGPT]]: {gpt_rec_response}')
        context.append({"role":"assistant","content":rec_response})

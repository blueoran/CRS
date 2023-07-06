import agent
import pandas as pd
import os
import openai
import logging
import numpy as np
import json
import tqdm
import re



# Load your API key from an environment variable or secret management service
top_K=5
API_KEY = "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"


cols=['title','year','trailer_duration','actors','awards','box_office','country','director','dvd_release','genre','imdb_id','imdb_type','imdb_votes','language','long_plot','movie_runtime','poster','production','rated','rating','release_date','short_plot','video_id','writer','youtube_comment','youtube_dislike','youtube_favorite','youtube_like','youtube_link','youtube_view']
need_cols=['title','year','trailer_duration','actors','awards','box_office','country','director','genre','imdb_votes','language','long_plot','movie_runtime','production','rated','rating','release_date','writer','youtube_comment','youtube_dislike','youtube_favorite','youtube_like','youtube_view']


products=pd.read_csv('./data/Inspired/data/movie_database_small.tsv',sep='\t')
# products['type']='movie'
products_1=products.iloc[:top_K].copy()
products=pd.read_csv('./data/phones_data.csv')
products['title']=products['brand_name']+' '+products['model_name']
products.drop(columns=['Unnamed: 0'],inplace=True)
# products['type']='cellphone'
products_2=products.iloc[:top_K].copy()
products=pd.read_csv('./data/StockX-Data-Contest-2019-3.csv')
products['title']=products['Sneaker Name']
# products.drop(columns=['Sneaker Name'],inplace=True)
# products['type']='sneaker'
products_3=products.iloc[:top_K].copy()




stream_logger=logging.getLogger('stream_logger')
stream_logger.setLevel(logging.INFO)
stream_logger.addHandler(logging.StreamHandler())

if __name__=='__main__':
    rec=agent.Agent([products_1,products_2,products_3],API_KEY,'./log/synthesis.log',explicit=True)
    rec.main_loop()


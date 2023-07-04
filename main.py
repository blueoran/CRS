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
top_K=100
API_KEY = "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"


cols=['title','year','trailer_duration','actors','awards','box_office','country','director','dvd_release','genre','imdb_id','imdb_type','imdb_votes','language','long_plot','movie_runtime','poster','production','rated','rating','release_date','short_plot','video_id','writer','youtube_comment','youtube_dislike','youtube_favorite','youtube_like','youtube_link','youtube_view']
need_cols=['title','year','trailer_duration','actors','awards','box_office','country','director','genre','imdb_votes','language','long_plot','movie_runtime','production','rated','rating','release_date','writer','youtube_comment','youtube_dislike','youtube_favorite','youtube_like','youtube_view']


products=pd.read_csv('./data/Inspired/data/movie_database_small.tsv',sep='\t')
products=products[need_cols].iloc[:top_K]




stream_logger=logging.getLogger('stream_logger')
stream_logger.setLevel(logging.INFO)
stream_logger.addHandler(logging.StreamHandler())

if __name__=='__main__':
    rec=agent.Agent('movie',products,API_KEY,'./log/synthesis.log')
    rec.main_loop()


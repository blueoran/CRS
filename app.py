import agent
from flask import Flask, render_template, request
import pandas as pd
import logging
top_K=10

app = Flask(__name__)
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


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 获取用户输入的字符串
        user_input = request.form['user_input']
        
        # 获取当前用户的唯一标识（可以使用session、cookie或其他方式）
        user_id = request.remote_addr  # 使用IP地址作为标识
        log_file=f"./logs/{user_id}.log"
        # 如果当前用户没有实例对象，则创建一个新的实例对象
        if user_id not in user_instances:
            user_instances[user_id] = agent.Agent([products_1,products_2,products_3,products_4,products_5],api_key,log_file,update_product,product_detail_path,log_level,explicit,verbose) # 在这里实例化你的class
            user_history[user_id] = []
        
        user_history[user_id].append({'role': 'user', 'content': user_input})
        # 调用class的interactive方法，传入用户输入，获取返回结果
        result = user_instances[user_id].user_interactive(user_input)
        user_history[user_id].append({'role': 'class', 'content': result})
        
        return render_template('index.html', result=result, history=user_history[user_id])
 # Check if the 'reset' query parameter is present in the URL
    if request.args.get('reset') == 'true':
        # Reset the user's history and delete the user instancelog_file=f"./logs/{user_id}.log"
        user_id = request.remote_addr
        log_file=f"./logs/{user_id}.log"
        user_instances[user_id]= agent.Agent([products_1,products_2,products_3,products_4,products_5],api_key,log_file,update_product,product_detail_path,log_level,explicit,verbose)
        user_history[user_id] = []
    
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)

import os
import json
import pandas as pd
from fuzzywuzzy import process
from crsgpt.communicate.communicate import *
from crsgpt.prompter.prompter import *
from concurrent.futures import ThreadPoolExecutor
import pickle
import numpy as np
from tqdm import trange
from openai.embeddings_utils import (
    get_embedding,
    distances_from_embeddings,
    indices_of_nearest_neighbors_from_distances,
)
import queue




class Product:
    def __init__(self,
                 head_k,
                 file_logger,
                 embedding_cache_path,
                 top_k,
                 update_product,
                 verbose,
                 files,
                 hallucination=False):
        self.head_k=head_k
        self.top_k=top_k
        self.file_logger=file_logger
        self.embedding_cache_path=embedding_cache_path
        self.update_product=update_product
        self.verbose=verbose
        self.file_to_title=files
        self.hallucination=hallucination
        self.past_selected_products=[]
        self.selected_products = []

        self.files_pandas=[]
        for f,t in self.file_to_title.items():
            file=pd.read_csv(f)
            file.rename(columns={t:"name"},inplace=True)
            if self.head_k is not None:
                file=file.iloc[:self.head_k].copy()
            self.files_pandas.append(file)


        for product in self.files_pandas:
            product["description"] = product.apply(
                lambda x: ";".join([f"{k}:{v}" for k, v in x.to_dict().items()]), axis=1
            )
            product["type"] = ""
        self.product_info = pd.concat(
            [p[["name", "type", "description"]] for p in self.files_pandas],
            ignore_index=True,
        )


        self.product_type_set = set()
        self.product_embedding = []
        try:
            self.embedding_cache = pd.read_pickle(self.embedding_cache_path)
        except FileNotFoundError:
            self.embedding_cache = {}
        with open(self.embedding_cache_path, "wb") as self.embedding_cache_file:
            pickle.dump(self.embedding_cache, self.embedding_cache_file)

        for i in trange(len(self.product_info)):
            product = self.product_info.iloc[i]
            product_type, product_embedding, product_sum = self.embedding_from_string(product["name"],product["description"])
            product["type"] = product_type
            self.product_type_set.add(product_type)
            self.product_embedding.append(product_embedding)

    def summarize_product_type(self,product):
        product_parse = general_json_chat(
            self.file_logger,self.verbose,
            compose_messages(
                {'s':compose_system_prompts(
                    {'prompt':Prompter.SYSTEM_PROMPT},
                    {'prompt':Prompter.PRODUCT_TYPE_SUM_PROMPT},
                    {'attribute':'Product Detail','content':product},
                )}
            ),
            "product_type_sum",max_tokens=250
        )
        return product_parse



    # define a function to retrieve embeddings from the cache if present, and otherwise request via the API
    def embedding_from_string(
        self,
        name: str,
        product_str: str,
        model: str = "text-embedding-ada-002",
    ) -> list:
        """Return embedding of given string, using a cache to avoid recomputing."""
        if (product_str, model) not in self.embedding_cache.keys():
            product_type_sum = self.summarize_product_type(product_str)
            product = f"Product Name: [[{name}]]; Product Type: [[{product_type_sum['product_type']}]]; Product Summary: [[{product_type_sum['product_summary']}]]; Product Details: [[{product_str}]]"
            success = False
            while not success:
                try:
                    self.embedding_cache[(product_str, model)] = (product_type_sum['product_type'].lower(),get_embedding(product, model),product)
                    success = True
                except:
                    pass
            with open(self.embedding_cache_path, "wb") as self.embedding_cache_file:
                pickle.dump(self.embedding_cache, self.embedding_cache_file)
        return self.embedding_cache[(product_str, model)]

    def select_product_type(self,goal,instruction,preference,context):
        selected_product_type = general_json_chat(
            self.file_logger,self.verbose,
            compose_messages(
                {'s':compose_system_prompts(
                    {'prompt':Prompter.SYSTEM_PROMPT},
                    {'prompt':Prompter.PRODUCT_TYPE_SELECT_PROMPT},
                    {'attribute':'Goal','content':goal},
                    {'attribute':'Instruction','content':instruction},
                    {'attribute':'User Preference','content':preference},
                    {'attribute':'Context','content':context[:-1]},
                    {'attribute':'User Input','content':context[-1]},
                    {'attribute':'Available Product Types','content':self.product_type_set},
                )}
            ),
            "product_type_select",max_tokens=50,concerened_key="Product Types"
        )
        self.file_logger.info(f"Selected product type: {selected_product_type}")
        return selected_product_type

    def recommendations_from_strings(
        self,
        query_string: str,
        part_product_index,
        model: str = "text-embedding-ada-002",
    ) -> list[int]:
        """Print out the k nearest neighbors of a given string."""
        query_embedding = get_embedding(query_string,model)
        product_embedding = np.array(self.product_embedding)[part_product_index]
        distances = distances_from_embeddings(query_embedding, product_embedding, distance_metric="cosine")
        indices_of_nearest_neighbors = indices_of_nearest_neighbors_from_distances(distances)
        return part_product_index[indices_of_nearest_neighbors]

    def process_selected_product(self, goal, instruction, preference, context, idx, product_features):
        verify_selected_product_answer = general_json_chat(
            self.file_logger, self.verbose,
            compose_messages(
                {'s': compose_system_prompts(
                    {'prompt': Prompter.SYSTEM_PROMPT},
                    {'prompt': Prompter.PRODUCT_VERIFY_PROMPT},
                    {'attribute': 'Goal', 'content': goal},
                    {'attribute': 'Instruction', 'content': instruction},
                    {'attribute': 'Chat History', 'content': context[:-1]},
                    {'attribute': 'User Input', 'content': context[-1]},
                    {'attribute': 'User Preference', 'content': preference},
                    {'attribute': 'Selected Products', 'content': self.product_info['description'].iloc[idx]},
                    {'attribute': 'Key features that the selected products should have', 'content': product_features},
                )}
            ),
            "product_verify", max_tokens=100
        )
        alignment = verify_selected_product_answer['Alignment']
        comment = verify_selected_product_answer['Comments']
        return {self.product_info["name"].iloc[idx]:f'{self.product_info["description"].iloc[idx]}\n***Alignment: {alignment}***\n***Comment: {comment}***\n'}

    def update_appeared_products(self,context):
        appeared_product = general_json_chat(
            self.file_logger,self.verbose,
            compose_messages(
                {'s':compose_system_prompts(
                    {'prompt':Prompter.SYSTEM_PROMPT},
                    {'prompt':Prompter.PRODUCT_APPEAR_PROMPT},
                    {'attribute':'Chat History','content':context[:-1]},
                    {'attribute':'User Input','content':context[-1]},
                    {'attribute':'Previously Appeared Products','content':self.appeared_products},
                )}
            ),
            "product_appearance","product_appear",max_tokens=300
        )
        self.appeared_products=appeared_product['ProductsAppeared']
        

    def filter_products(self,goal,instruction,preference,context,product_features,selected_products_index,past_product_selection):
            search_filter_prompts = general_json_chat(
                self.file_logger,self.verbose,
                compose_messages(
                    {'s':compose_system_prompts(
                        {'prompt':Prompter.SYSTEM_PROMPT},
                        {'prompt':Prompter.PRODUCT_SEARCH_PROMPT_ADV},
                        {'attribute':'Goal','content':goal},
                        {'attribute':'Instruction','content':instruction},
                        {'attribute':'Chat History','content':context[:-1]},
                        {'attribute':'User Input','content':context[-1]},
                        {'attribute':'User Preference','content':preference},
                        {'attribute':'Key features that the selected products should have','content':product_features},
                    )}
                ),
                "product_search_ad",max_tokens=100
            )
            
            positive_filter_prompts = ', '.join(search_filter_prompts['PositivePrompts'])
            negative_filter_prompts = ', '.join(search_filter_prompts['NegativePrompts'])
            
            positive_selected_products = list(self.recommendations_from_strings(positive_filter_prompts, selected_products_index)) if len(search_filter_prompts['PositivePrompts']) > 0 else []
            negative_selected_products = list(self.recommendations_from_strings(negative_filter_prompts, selected_products_index)) if len(search_filter_prompts['NegativePrompts']) > 0 else []
            # import pdb;pdb.set_trace()
            included_products_idx = []
            excluded_products_idx = []
            for tag_prod in past_product_selection:
                if tag_prod['Name'] not in self.past_selected_products:
                    continue
                if tag_prod['Tag'] == 'include':
                    included_product=tag_prod['Name']
                    try:
                        included_products_idx.append(self.product_info[self.product_info['name']==included_product].index[0])
                    except:
                        pass
                elif tag_prod['Tag'] == 'exclude':
                    excluded_product=tag_prod['Name']
                    try:
                        excluded_products_idx.append(self.product_info[self.product_info['name']==excluded_product].index[0])
                    except:
                        pass
            for idx in excluded_products_idx:
                try:
                    positive_selected_products.remove(idx)
                except:
                    pass
            
            selected_products = []
            for i in range(self.top_k,5*self.top_k):
                selected_products = list(set(positive_selected_products[:i+1])-set(negative_selected_products[:i+1]))
                if len(selected_products) >= self.top_k:
                    break
            
            for idx in included_products_idx:
                if idx not in selected_products:
                    selected_products.append(idx)
            
            return selected_products


    def seek_products_multi(self, goal, instruction, preference, context, selected_products_index,max_workers=16):  # max_workers can be adjusted as needed
        product_features = general_json_chat(
            self.file_logger,self.verbose,
            compose_messages(
                {'s':compose_system_prompts(
                    {'prompt':Prompter.SYSTEM_PROMPT},
                    {'prompt':Prompter.PRODUCT_FEATURE_PROMPT},
                    {'attribute':'Goal','content':goal},
                    {'attribute':'Instruction','content':instruction},
                    {'attribute':'User Preference','content':preference},
                )}
            ),
            "product_feature",max_tokens=300,concerened_key="Product Feature"
        )
        
        past_product_selection = general_json_chat(
            self.file_logger,self.verbose,
            compose_messages(
                {'s':compose_system_prompts(
                    {'prompt':Prompter.SYSTEM_PROMPT},
                    {'prompt':Prompter.PAST_PRODUCT_PROMPT},
                    {'attribute':'Goal','content':goal},
                    {'attribute':'Instruction','content':instruction},
                    {'attribute':'User Preference','content':preference},
                    {'attribute':'Past Selected Products','content':self.past_selected_products},
                )}
            ),
            "past_product_selection",max_tokens=1000, concerened_key="TaggedProducts"
        )

        selected_description = {}
        self.selected_products = self.filter_products(goal,instruction,preference,context,product_features,selected_products_index,past_product_selection)
        self.file_logger.info(f"Selected products: {self.product_info.iloc[self.selected_products]}")
        
        if self.hallucination:
        
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(self.process_selected_product, goal, instruction, preference, context, idx, product_features) for idx in self.selected_products]

            for future in futures:
                res = future.result()
                selected_description.update(res)
        else:

            for idx in self.selected_products:
                selected_description[self.product_info["name"].iloc[idx]]=f'{self.product_info["description"].iloc[idx]}\n'

        return selected_description
    def update_past_selected_products(self):
        for idx in self.selected_products:
            name = self.product_info["name"].iloc[idx]
            if name not in self.past_selected_products[:-2*self.top_k]:
                self.past_selected_products.append(name)
    def select_products_large(self,goal,instruction,preference,context):
        selected_product_type=self.select_product_type(goal,instruction,preference,context)
        filtered_product=self.product_info[self.product_info['type'].isin(selected_product_type)]
        filtered_product_index=filtered_product.index
        selected_product_description=self.seek_products_multi(goal,instruction,preference,context,filtered_product_index)
        product_string = '\n'.join([f'{k}: {v}' for k, v in selected_product_description.items()])
        self.file_logger.info(f"Selected products: {product_string}")
        return product_string
    

import os
import json
import pandas as pd
from fuzzywuzzy import process
from crsgpt.communicate.communicate import *
from crsgpt.prompter.prompter import *
from concurrent.futures import ThreadPoolExecutor
import pickle
import numpy as np
from openai.embeddings_utils import (
    get_embedding,
    distances_from_embeddings,
    indices_of_nearest_neighbors_from_distances,
)





class Product:
    def __init__(self,
                 head_k,
                 file_logger,
                 embedding_cache_path,
                 top_k,
                 update_product,
                 verbose,
                 files):
        self.head_k=head_k
        self.top_k=top_k
        self.file_logger=file_logger
        self.embedding_cache_path=embedding_cache_path
        self.update_product=update_product
        self.verbose=verbose
        self.file_to_title=files

        self.files_pandas=[]
        for f,t in self.file_to_title.items():
            file=pd.read_csv(f)
            file.rename(columns={t:'title'},inplace=True)
            if self.head_k is not None:
                file=file.iloc[:self.head_k].copy()
            self.files_pandas.append(file)


        for product in self.files_pandas:
            product["description"] = product.apply(
                lambda x: ";".join([f"{k}:{v}" for k, v in x.to_dict().items()]), axis=1
            )
            product["type"] = ""
        self.product_info = pd.concat(
            [p[["title", "type", "description"]] for p in self.files_pandas],
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

        for i in range(len(self.product_info)):
            product = self.product_info.iloc[i]
            product_type, product_embedding = self.embedding_from_string(product["description"])
            product["type"] = product_type
            self.product_type_set.add(product_type)
            self.product_embedding.append(product_embedding)

    def summarize_product_type(self,product):
        product_parse = general_json_chat(
            self.file_logger,self.verbose,
            compose_messages(
                {'s':compose_system_prompts(
                    {'prompt':Prompter.SYSTEM_PROMPT},
                    {'prompt':Prompter.PRODUCT_TYPE_PROMPT},
                    {'attribute':'Product Detail','content':product},
                )}
            ),
            "product_type",max_tokens=100,concerened_key="product_type"
        ).lower()
        return product_parse



    # define a function to retrieve embeddings from the cache if present, and otherwise request via the API
    def embedding_from_string(
        self,
        product: str,
        model: str = "text-embedding-ada-002",
    ) -> list:
        """Return embedding of given string, using a cache to avoid recomputing."""
        if (product, model) not in self.embedding_cache.keys():
            product_type = self.summarize_product_type(product)
            self.embedding_cache[(product, model)] = (product_type,get_embedding(product, model))
            with open(self.embedding_cache_path, "wb") as self.embedding_cache_file:
                pickle.dump(self.embedding_cache, self.embedding_cache_file)
        return self.embedding_cache[(product, model)]

    def select_product_type(self,goal,instruction,preference,context):
        selected_product_type = general_json_chat(
            self.file_logger,self.verbose,
            compose_messages(
                {'s':compose_system_prompts(
                    {'prompt':Prompter.SYSTEM_PROMPT},
                    {'prompt':Prompter.PRODUCT_TYPE_PROMPT},
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
        return part_product_index[indices_of_nearest_neighbors[:self.top_k]]

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
        return {self.product_info["title"].iloc[idx]:f'{self.product_info["description"].iloc[idx]}\n***Alignment: {alignment}***\n***Comment: {comment}***\n'}



    def seek_products_multi(self, goal, instruction, preference, context, selected_products_index):  # max_workers can be adjusted as needed
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

        search_prompt = general_json_chat(
            self.file_logger,self.verbose,
            compose_messages(
                {'s':compose_system_prompts(
                    {'prompt':Prompter.SYSTEM_PROMPT},
                    {'prompt':Prompter.PRODUCT_SEARCH_PROMPT},
                    {'attribute':'Goal','content':goal},
                    {'attribute':'Instruction','content':instruction},
                    {'attribute':'Chat History','content':context[:-1]},
                    {'attribute':'User Input','content':context[-1]},
                    {'attribute':'User Preference','content':preference},
                    {'attribute':'Key features that the selected products should have','content':product_features},
                )}
            ),
            "product_search",max_tokens=100,concerened_key="Prompt"
        )
        selected_description = {}
        selected_products = self.recommendations_from_strings(search_prompt,selected_products_index)
        self.file_logger.info(f"Selected products: {self.product_info.iloc[selected_products]}")
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(self.process_selected_product, goal, instruction, preference, context, idx, product_features) for idx in selected_products]

        for future in futures:
            res = future.result()
            selected_description.update(res)

        return selected_description

    def select_products_large(self,goal,instruction,preference,context):
        selected_product_type=self.select_product_type(goal,instruction,preference,context)
        filtered_product=self.product_info[self.product_info['type'].isin(selected_product_type)]
        filtered_product_index=filtered_product.index
        selected_product_description=self.seek_products_multi(goal,instruction,preference,context,filtered_product_index)
        product_string = '\n'.join([f'{k}: {v}' for k, v in selected_product_description.items()])
        self.file_logger.info(f"Selected products: {product_string}")
        return product_string
    

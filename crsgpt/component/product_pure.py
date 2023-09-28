import os
import json
import pandas as pd
from fuzzywuzzy import process
from crsgpt.communicate.communicate import *
from crsgpt.prompter.prompter import *

import asyncio
from aiohttp import ClientSession
from concurrent.futures import ThreadPoolExecutor





class Product:
    def __init__(self,top_k,file_logger,product_detail_path,update_product,verbose,files):
        self.top_k=top_k
        self.file_logger=file_logger
        self.product_detail_path=product_detail_path
        self.update_product=update_product
        self.verbose=verbose
        self.file_to_title=files

        self.files_pandas=[]
        for f,t in self.file_to_title.items():
            file=pd.read_csv(f)
            file.rename(columns={t:'title'},inplace=True)
            if self.top_k is not None:
                file=file.iloc[:self.top_k].copy()
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
        self.product_name = self.product_info["title"].tolist()


        self.product_type_set = set()
        self.product_summary = []
        load = False
        if os.path.exists(product_detail_path) and update_product is False:
            with open(product_detail_path, "r") as f:
                products_infos = json.load(f)
                self.product_summary = products_infos["product_detail"]
                self.product_type_set = set(products_infos["product_type_set"])
                self.product_type = products_infos["product_type"]
            if sum([len(x) for x in self.files_pandas]) == len(self.product_summary):
                load = True
                self.product_info.loc[:, "type"] = self.product_type

        if load is False:
            self.product_summary = []
            self.product_type_set = set()
            for i in range(len(self.product_info)):
                product = self.product_info.iloc[i]
                product_parse=self.summarize_products(product)
                self.product_summary.append(
                    f'{product["title"]} (product_type: {product_parse["product_type"]}): {product_parse["key_features"]}'
                )
                product["type"] = product_parse["product_type"]
                self.product_type_set.add(product_parse["product_type"])

            with open(product_detail_path, "w") as f:
                json.dump(
                    {
                        "product_detail": self.product_summary,
                        "product_type_set": list(self.product_type_set),
                        "product_type": self.product_info["type"].to_list(),
                    },
                    f,
                )
        self.current_product_sum=""
        if self.verbose:
            print(self.product_summary)

    def summarize_products(self,product):
        product_parse = general_json_chat(
            self.file_logger,self.verbose,
            compose_messages(
                {'s':compose_system_prompts(
                    {'prompt':Prompter.SYSTEM_PROMPT},
                    {'prompt':Prompter.PRODUCT_PROMPT},
                    {'attribute':'Product Detail','content':product['description']},
                )}
            ),
            "product",max_tokens=100
        )
        product_parse["product_type"] = product_parse["product_type"].lower()
        return product_parse

    def fuzzy_match(self,gpt_output,standard_match):
        fuzzy_selected={}
        for p in gpt_output:
            best_match = process.extractOne(
                p, standard_match.tolist(), score_cutoff=90
            )
            if best_match is not None:
                fuzzy_selected[best_match[0]]=gpt_output.index(p)
                # fuzzy_selected[best_match[0]]=standard_match[standard_match["title"]==best_match[0]].index.values[0]
        return fuzzy_selected

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


    def seek_products(self,goal,instruction,preference,context,selected_products,batch_size=10):

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


        satisfied=False
        selected_product_summary={}
        selected_product_description={}
        for i in range(len(selected_products)//batch_size+1):
            if satisfied and len(selected_product_description)!=0:
                break
            if (i+1)*batch_size>len(selected_products):
                end=len(selected_products)
            else:
                end=(i+1)*batch_size
            part_select_products=selected_products.iloc[i*batch_size:end]
            part_selected_product_answer=general_json_chat(
                self.file_logger,self.verbose,
                compose_messages(
                    {'s':compose_system_prompts(
                        {'prompt':Prompter.SYSTEM_PROMPT},
                        {'prompt':Prompter.PRODUCT_SEEK_PROMPT},
                        {'attribute':'Goal','content':goal},
                        {'attribute':'Instruction','content':instruction},
                        {'attribute':'Chat History','content':context[:-1]},
                        {'attribute':'User Input','content':context[-1]},
                        {'attribute':'User Preference','content':preference},
                        {'attribute':'Key points of products that previously selected from the database',
                        'content':'\n'.join(list(selected_product_summary.values())) if len(selected_product_summary)>0 else 'None'},
                        {'attribute':'Key features that the selected products should have','content':product_features},
                        {'attribute':'Current products that you should select from',
                         'content':'\n'.join((part_select_products['title']+': '+part_select_products['description']).tolist())},
                    )}
                ),
                "product_seek_large",max_tokens=500
            )
            # import pdb;pdb.set_trace()
            satisfied=part_selected_product_answer['Can Satisfy']
            part_selected_product_title=part_selected_product_answer['Products Selected from Current Given Products']
            part_selected_product_summary=part_selected_product_answer['Products Summary']
            actually_selected=self.fuzzy_match(part_selected_product_title,selected_products['title'].iloc[i*batch_size:end])
            actually_selected_sum={product:part_selected_product_summary[actually_selected[product]] for product in actually_selected.keys()}
            actually_selected_description={product:part_select_products[part_select_products['title']==product].iloc[0]['description'] for product in actually_selected.keys()}
            
            if len(actually_selected_description)==0:
                continue
            
            
            actually_selected_description_filtered = {}
            actually_selected_product_sum_filtered = {}
            for p in actually_selected_description:
                verify_selected_product_answer=general_json_chat(
                    self.file_logger,self.verbose,
                    compose_messages(
                        {'s':compose_system_prompts(
                            {'prompt':Prompter.SYSTEM_PROMPT},
                            {'prompt':Prompter.PRODUCT_VERIFY_PROMPT},
                            {'attribute':'Goal','content':goal},
                            {'attribute':'Instruction','content':instruction},
                            {'attribute':'Chat History','content':context[:-1]},
                            {'attribute':'User Input','content':context[-1]},
                            {'attribute':'User Preference','content':preference},
                            {'attribute':'Selected Products','content':actually_selected_description[p]},
                            {'attribute':'Key features that the selected products should have','content':product_features},
                        )}
                    ),
                    "product_verify",max_tokens=100
                )
                alignment=verify_selected_product_answer['Alignment']
                comment=verify_selected_product_answer['Comments']
                # if alignment==True:
                actually_selected_description_filtered[p] = \
                    f'{actually_selected_description[p]}\n***Alignment: {alignment}***\n***Comment: {comment}***\n'
                actually_selected_product_sum_filtered[p]=actually_selected_sum[p]
                
            
            selected_product_summary=dict(selected_product_summary, **actually_selected_product_sum_filtered)
            selected_product_description=dict(selected_product_description, **actually_selected_description_filtered)
        return selected_product_summary, selected_product_description



    def process_batch(self, i, batch_size, selected_products, goal, instruction, context, preference, product_features):
        if (i+1)*batch_size>len(selected_products):
            end=len(selected_products)
        else:
            end=(i+1)*batch_size

        part_select_products=selected_products.iloc[i*batch_size:end]
        part_selected_product_answer=general_json_chat(
            self.file_logger,self.verbose,
            compose_messages(
                {'s':compose_system_prompts(
                    {'prompt':Prompter.SYSTEM_PROMPT},
                    {'prompt':Prompter.PRODUCT_SEEK_PROMPT},
                    {'attribute':'Goal','content':goal},
                    {'attribute':'Instruction','content':instruction},
                    {'attribute':'Chat History','content':context[:-1]},
                    {'attribute':'User Input','content':context[-1]},
                    {'attribute':'User Preference','content':preference},
                    {'attribute':'Key points of products that previously selected from the database', 'content':'None'},
                    {'attribute':'Key features that the selected products should have','content':product_features},
                    {'attribute':'Current products that you should select from',
                        'content':'\n'.join((part_select_products['title']+': '+part_select_products['description']).tolist())},
                )}
            ),
            "product_seek_large",max_tokens=500
        )
        part_selected_product_title=part_selected_product_answer['Products Selected from Current Given Products']
        # part_selected_product_summary=part_selected_product_answer['Products Summary']
        actually_selected=self.fuzzy_match(part_selected_product_title,selected_products['title'].iloc[i*batch_size:end])
        # actually_selected_sum={product:part_selected_product_summary[actually_selected[product]] for product in actually_selected.keys()}
        actually_selected_description={product:part_select_products[part_select_products['title']==product].iloc[0]['description'] for product in actually_selected.keys()}
        
        
        actually_selected_description_filtered = {}
        actually_selected_product_sum_filtered = {}
        for p in actually_selected_description:
            verify_selected_product_answer=general_json_chat(
                self.file_logger,self.verbose,
                compose_messages(
                    {'s':compose_system_prompts(
                        {'prompt':Prompter.SYSTEM_PROMPT},
                        {'prompt':Prompter.PRODUCT_VERIFY_PROMPT},
                        {'attribute':'Goal','content':goal},
                        {'attribute':'Instruction','content':instruction},
                        {'attribute':'Chat History','content':context[:-1]},
                        {'attribute':'User Input','content':context[-1]},
                        {'attribute':'User Preference','content':preference},
                        {'attribute':'Selected Products','content':actually_selected_description[p]},
                        {'attribute':'Key features that the selected products should have','content':product_features},
                    )}
                ),
                "product_verify",max_tokens=100
            )
            alignment=verify_selected_product_answer['Alignment']
            comment=verify_selected_product_answer['Comments']
            actually_selected_description_filtered[p] = \
                f'{actually_selected_description[p]}\n***Alignment: {alignment}***\n***Comment: {comment}***\n'
            # actually_selected_product_sum_filtered[p]=actually_selected_sum[p]

        return actually_selected_product_sum_filtered, actually_selected_description_filtered


    def seek_products_multi(self, goal, instruction, preference, context, selected_products, batch_size=10, max_workers=5):  # max_workers can be adjusted as needed
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

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.process_batch, i, batch_size, selected_products, goal, instruction, context, preference, product_features) for i in range(len(selected_products)//batch_size+1)]

        combined_product_summary = {}
        combined_product_description = {}
        for future in futures:
            res = future.result()
            combined_product_summary.update(res[0])
            combined_product_description.update(res[1])

        return combined_product_summary, combined_product_description

    def select_products_large(self,goal,instruction,preference,context):
        selected_product_type=self.select_product_type(goal,instruction,preference,context)
        # import pdb;pdb.set_trace()
        filtered_product=self.product_info[self.product_info['type'].isin(selected_product_type)]
        selected_product_summary,selected_product_description=self.seek_products_multi(goal,instruction,preference,context,filtered_product)
        # product_string='\n'.join(list(selected_product_summary.values()))
        product_string = '\n'.join([f'{k}: {v}' for k, v in selected_product_description.items()])
        self.file_logger.info(f"Selected products: {product_string}")
        # final_products=filtered_product.iloc[list(product_selected.values())]
        # product_string='\n'.join((final_products['title']+': '+final_products['description']).tolist())
        # product_summary_list=self.product_summary[self.product_name.isin(product_selected)]
        # self.current_product_sum='\n'.join((product_summary_list['title']+': '+product_summary_list['description']).tolist())
        return product_string
    


    def select_products(self,goal,instruction,preference,context):
        selected_products = general_json_chat(
            self.file_logger,self.verbose,
            compose_messages(
                {'s':compose_system_prompts(
                    {'prompt':Prompter.SYSTEM_PROMPT},
                    {'prompt':Prompter.PRODUCT_SELECTION_PROMPT},
                    {'attribute':'Goal','content':goal},
                    {'attribute':'Instruction','content':instruction},
                    {'attribute':'Selected Product','content':self.product_summary},
                    {'attribute':'User Preference','content':preference},
                    {'attribute':'Context','content':context},
                )}
            ),
            "product_seek",max_tokens=300,concerened_key="Necessary Products"
        )
        real_select=self.fuzzy_match(selected_products,self.product_name)
        real_select_products=self.product_info.iloc[list(real_select.values())]
        product_string = '\n'.join((real_select_products['title']+': '+real_select_products['description']).tolist())
        
        product_resources = compose_system_prompts(
            {'attribute':'Selected Products that you should only refer from','content':'; '.join(selected_products.values())},
            {'attribute':'Whole product details','content':product_string},
        )
        return product_resources

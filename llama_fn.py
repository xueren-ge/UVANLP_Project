
import argparse 
import ipdb
import os
import pandas as pd
from multiprocessing.pool import ThreadPool as Pool
from multiprocessing import cpu_count

from tqdm import tqdm
import openai
import re
import time
import argparse
import numpy as np
import re
import random
import transformers
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import json
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# model_name_or_path = "m42-health/Llama3-Med42-70B"
# model_name_or_path = "ProbeMedicalYonseiMAILab/medllama3-v20"
# model_name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_name_or_path = "meta-llama/Meta-Llama-3-70B-Instruct"
# model_name_or_path = "meta-llama/Meta-Llama-3.1-70B-Instruct"
# model_name_or_path = "meta-llama/Meta-Llama-3.1-405B-Instruct"
# model_name_or_path = "meta-llama/Meta-Llama-3.1-405B-Instruct-FP8"


class LLAMA:
    def __init__(self, model_name_or_path):
        self.pipeline = transformers.pipeline(
            "text-generation", 
            model=model_name_or_path, 
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    
    def apply_medllama3(self, messages, temperature=0.7, max_tokens=-1, top_k=150, top_p=0.75):
        prompt = self.pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        if max_tokens != -1:
            outputs = self.pipeline(
                prompt,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            response = outputs[0]["generated_text"][len(prompt) :]
        else:
            outputs = self.pipeline(
                prompt,
                max_new_tokens=2048,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            response = outputs[0]["generated_text"][len(prompt) :]
        return response

    def extract_json(self, response, pattern=r'\{.*\}'):
        if type(response) == dict:
            # data = json.loads(response)
            return None, response

        # Search for the pattern in the text
        # match = re.search(pattern, response, re.DOTALL)
        matches = re.findall(pattern, response, re.DOTALL)

        if not matches:
            print("No JSON object found in the text.")
            print(response)
            # print(json_data)
            return None, None

        json_data = matches[0] if len(matches) == 1 else matches[-1]
        
        try:
            # Load the JSON data
            # data = json.loads(json_data)
            # return None, data

            python_list = json.loads(json_data.replace("'", '"'))
            return None, python_list
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            # print(response)
            # print(json_data)
            return e, json_data

    def handle_error(self, messages, next_response, pattern):
        error, next_response_dict = self.extract_json(next_response, pattern)
        print(error)
        print(next_response)
        ################################ no json file, regenerate ################################
        cnt = 1
        while error == None and next_response_dict == None and next_response:
            print(f"No json, repeat generating for the {cnt} time")
            next_response = self.apply_medllama3(messages, temperature=0.7)
            error, next_response_dict = self.extract_json(next_response, pattern)
            cnt += 1

        ################################ json file incorrect ################################
        cnt = 1
        while error and cnt < 10:
            print(f"fix error for the {cnt} time")
            prompt = f"""There is an Error decoding JSON: {error} in the following json data
            {next_response_dict}, Can you fix the error and return the correct json format. Directly return the json without explanation.
            """
            messages = [{"role": "user", "content": prompt}]
            new_response = self.apply_medllama3(messages, temperature=0.3)
            print(new_response)
            error, next_response_dict = self.extract_json(new_response, pattern)
            cnt += 1
        
        if error:
            prompt = f"""There is an Error decoding JSON: {error} in the following json data
            {next_response_dict}, Can you fix the error and return the correct json format. Make sure it can be loaded using python (json.loads()). Directly return the json without explanation.
            """
            messages = [{"role": "user", "content": prompt}]
            new_response = self.apply_medllama3(messages, temperature=0.3)
            next_response_dict = json.loads(new_response)
        return next_response_dict

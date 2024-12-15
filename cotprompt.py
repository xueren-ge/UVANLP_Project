
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


with open("/scratch/zar8jw/data/EMS_Protocol.json", 'r') as f:
    ems_labels = json.load(f)

# model_name_or_path = "m42-health/Llama3-Med42-70B"
model_name_or_path = "ProbeMedicalYonseiMAILab/medllama3-v20"
# model_name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_name_or_path = "meta-llama/Meta-Llama-3-70B-Instruct"
# model_name_or_path = "meta-llama/Meta-Llama-3.1-70B-Instruct"
# model_name_or_path = "meta-llama/Meta-Llama-3.1-405B-Instruct"
# model_name_or_path = "meta-llama/Meta-Llama-3.1-405B-Instruct-FP8"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_name_or_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

def apply_medllama3(messages, temperature=0.7, max_tokens=-1, top_k=150, top_p=0.75):
    prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    if max_tokens != -1:
        outputs = pipeline(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        response = outputs[0]["generated_text"][len(prompt) :]
    else:
        outputs = pipeline(
            prompt,
            max_new_tokens=2048,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        response = outputs[0]["generated_text"][len(prompt) :]
    return response

def extract_json(response):

    if type(response) == dict:
        data = json.loads(json_data)
        return None, data

    # Regular expression pattern to match JSON content
    # pattern = r'\{.*\}'
    pattern = r'\[.*?\]'

    # Search for the pattern in the text
    # match = re.search(pattern, response, re.DOTALL)
    matches = re.findall(pattern, response, re.DOTALL)

    if not matches:
        print("No JSON object found in the text.")
        print(response)
        # print(json_data)
        return None, None

    json_data = matches[0] if len(matches) == 1 else matches[-1]
    print(json_data)

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

def handleError(messages, next_response):
    
    error, next_response_dict = extract_json(next_response)
    print(error)
    print(next_response)
    ################################ no json file, regenerate ################################
    cnt = 1
    while error == None and next_response_dict == None and next_response:
        print(f"No json, repeat generating for the {cnt} time")
        next_response = apply_medllama3(messages, temperature=0.7)
        error, next_response_dict = extract_json(next_response)
        cnt += 1

    ################################ json file incorrect ################################
    cnt = 1
    error_list = [str(error)]
    while error and cnt < 10:
        error_string = ";".join(error_list)
        print(f"fix error for the {cnt} time")

        prompt = f"""
        There is a text, read the text, organize and return the predictions in the python list format.

        Double check your returned json and avoid the known errors, for example, {error_string}

        Text: {next_response}
        """

        messages = [{"role": "user", "content": prompt}]
        new_response = apply_medllama3(messages, temperature=0.3)
        error, next_response_dict = extract_json(new_response)
        if error not in error_list:
            error_list.append(str(error))
        cnt += 1
    
    # if error:
    #     prompt = f"""There is an Error decoding JSON: {error} in the following json data
    #     {next_response_dict}, Can you fix the error and return the correct json format. Make sure it can be loaded using python (json.loads()). Directly return the json without explanation.
    #     """
    #     messages = [{"role": "user", "content": prompt}]
    #     new_response = apply_medllama3(messages, temperature=0.3)
    #     next_response_dict = json.loads(new_response)
    return next_response_dict


def chat(idx, ePCR, name, max_epochs=20):
    prompt = f"""Based on the medic note, recommend EMS protocols for first responder to save the patient. You can only return your result one time.
    
    Here are all EMS Protocols to choose from: {ems_labels}
    
    medic note: {ePCR}

    Use the following step-by-step thinking to recommend EMS protocols
    
    Let's think step by step
    Step1: Symptom Abstraction: analyzing patient symptoms
    Step2: Candidate Disease Recall
    Step3: From Step2, reorganize the result in the required format. You must return the recommended EMS protocols in the python list format (e.g.: ["your result"]).
    
    ---

    RESPONSE:   
    """

    messages = [{"role": "system", "content": "you are an enmergency medical service professional."}]
    messages = [{"role": "user", "content": prompt}]
    # messages.append({"role": "user", "content": 'ePCR:' + ePCR})
    response = apply_medllama3(messages, temperature=0.3)

    if not os.path.exists(f'./log/{name}'):
        os.makedirs(f'./log/{name}')

    error, status = extract_json(response)

    if error:
        response = handleError(messages, response)
        error, status = extract_json(response)

    with open(f'./log/{name}/{idx}.json', 'w') as f:
        json.dump(response, f, indent=4)

    return status


if __name__ == "__main__":
    name = "medllama3-v20_cot_prompt"
    if not os.path.exists('./results'):
        os.makedirs('./results')
    if not os.path.exists(f"./log/{name}/"):
        os.makedirs(f"./log/{name}/")

    # df = pd.read_csv('./dataset/RAA_processed_all.csv')
    df = pd.read_csv("/scratch/zar8jw/data/test.csv")
    ret = []
    for i in tqdm(range(len(df))):

        if f'{i}.json' in os.listdir(f"./log/{name}/"):
            with open(f"./log/{name}/{i}.json", 'r') as f:
                response = json.load(f)
            error, cur_ret = extract_json(response)
            ret.append(cur_ret)
            continue

        text = ''.join(df['Medic Notes'][i])
        cur_ret = chat(i, text, name)

        if cur_ret == None or cur_ret == [] or not cur_ret:
            print(i)
            break

        ret.append(cur_ret)

        if not i % 10:
            with open(f'./results/{name}.json', 'w') as f:
                json.dump(ret, f, indent=4)

    with open(f'./results/{name}.json', 'w') as f:
        json.dump(ret, f, indent=4)
import json

from collections import Counter
import os

import dataclasses
import fire
import random
import torch
import torch.optim as optim
import numpy as np
from peft import get_peft_model, PeftModel
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy
)
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.optim.lr_scheduler import StepLR
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoProcessor, 
    LlamaForCausalLM,
    MllamaForConditionalGeneration,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.mllama.modeling_mllama import  MllamaSelfAttentionDecoderLayer,MllamaCrossAttentionDecoderLayer,MllamaVisionEncoderLayer

from llama_recipes.configs import fsdp_config as FSDP_CONFIG
from llama_recipes.configs import train_config as TRAIN_CONFIG
from llama_recipes.configs import quantization_config  as QUANTIZATION_CONFIG
from llama_recipes.data.concatenator import ConcatDataset
from llama_recipes.policies import AnyPrecisionAdamW, apply_fsdp_checkpointing
from llama_recipes.inference.model_utils import load_model
from llama_recipes.utils import fsdp_auto_wrap_policy
from llama_recipes.utils.config_utils import (
    update_config,
    generate_peft_config,
    generate_dataset_config,
    get_dataloader_kwargs,
    check_fsdp_config,
)
from llama_recipes.utils.dataset_utils import get_preprocessed_dataset,get_custom_data_collator

from llama_recipes.utils.fsdp_utils import hsdp_device_mesh
from llama_recipes.utils.train_utils import (
    train,
    freeze_transformer_layers,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    print_model_size,
    get_policies,
)
from accelerate.utils import is_xpu_available
from warnings import warn
from tqdm import tqdm
import pandas as pd
import re

with open("/scratch/zar8jw/data/EMS_Protocol.json", 'r') as f:
    ems_labels = json.load(f)




def extract_json(response):

    if type(response) == dict:
        data = json.loads(json_data)
        return None, data

    # Regular expression pattern to match JSON content
    # pattern = r'\{.*\}'
    pattern = r'\[.*\]'

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

        python_list = json.loads(json_data.replace("'", '"'))
        return None, python_list
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        print(response)
        print(json_data)
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
    while error and cnt < 10:
        print(f"fix error for the {cnt} time")
        prompt = f"""There is an Error decoding JSON: {error} in the following json data
        {next_response_dict}, Can you fix the error and return the correct json format. Directly return the json without explanation.
        """
        messages = [{"role": "user", "content": prompt}]
        new_response = apply_medllama3(messages, temperature=0.3)
        print(new_response)
        error, next_response_dict = extract_json(new_response)
        cnt += 1
    
    if error:
        prompt = f"""There is an Error decoding JSON: {error} in the following json data
        {next_response_dict}, Can you fix the error and return the correct json format. Make sure it can be loaded using python (json.loads()). Directly return the json without explanation.
        """
        messages = [{"role": "user", "content": prompt}]
        new_response = apply_medllama3(messages, temperature=0.3)
        next_response_dict = json.loads(new_response)
    return next_response_dict




if __name__ == "__main__":
    model_id = "meta-llama/Llama-3.1-8B"
    # model_id = "meta-llama/Llama-3.1-70B"

    # peft_model_id = "/scratch/zar8jw/llama-recipes/models/llama-3.1-8B-instruct"
    # peft_model_id = "/scratch/zar8jw/llama-recipes/models/llama-3.1-70B-instruct"
    # peft_model_id = "/scratch/zar8jw/llama-recipes/models/llama-3.1-8B-instruct-directresp"
    peft_model_id = "/scratch/zar8jw/llama-recipes/models/llama-3.1-8B"

    # folder_name = "3.1-8B-finetune"
    # folder_name = "3.1-70B-finetune"
    folder_name = "3.1-8B-finetune-woinstruct"

    quantization = None
    model = load_model(model_id, quantization, use_fast_kernels=False)
    model = PeftModel.from_pretrained(model, peft_model_id)
    # print(model)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if not tokenizer.pad_token_id: 
        tokenizer.pad_token_id = tokenizer.eos_token_id
    

    df = pd.read_csv("/scratch/zar8jw/data/test.csv")
    ret = []

    if not os.path.exists(f"./log/{folder_name}"):
        os.makedirs(f"./log/{folder_name}")


    for i in tqdm(range(len(df))):

        if f'{i}.json' in os.listdir(f"./log/{folder_name}"):
            with open(f"./log/{folder_name}/{i}.json", 'r') as f:
                generated_text = json.load(f)
            response = generated_text.split("---\n\nRESPONSE:")[1].strip()
            error, cur_ret = extract_json(response)
            ret.append(cur_ret)
            continue


        input_ = ''.join(df['Medic Notes'][i])
        # input_ = tokenizer(text, return_tensors="pt")

        prompt = f"""Based on the medic note, recommend EMS protocols for first responder to save the patient. Return the result in the json format defined as follows. If there are multiple recommendations, separate each with comma. You can only return your result one time. \n[\n"prediction1", "prediction2", ...\n]\nHere are all EMS Protocols to choose from: {ems_labels}\nmedic note: {input_}\n\n---\n\nRESPONSE:"""

        # print(prompt)
        batch = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **batch,
                max_new_tokens=4096,
                do_sample=True,
                top_p=1.0,
                temperature=0.8,
                min_length=None,
                use_cache=True,
                top_k=50,
                repetition_penalty=1.2,
                length_penalty=1,
            )

        # print(outputs)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text.split("---\n\nRESPONSE:")[1].strip()
        error, json_data = extract_json(response)
        
        if not error:
            ret.append(json_data)

            if not os.path.exists(f'./log/{folder_name}'):
                os.makedirs(f"./log/{folder_name}")

            with open(f'./log/{folder_name}/{i}.json', 'w') as f:
                json.dump(generated_text, f, indent=4)
        else:
            break
    
    with open(f'./results/{folder_name}.json', 'w') as f:
        json.dump(ret, f, indent=4)
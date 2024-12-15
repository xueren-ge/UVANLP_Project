import os
import json
from datasets import load_dataset
from torch.utils.data import Dataset

with open("/scratch/zar8jw/data/EMS_Protocol.json", 'r') as f:
    ems_labels = json.load(f)

class ems(Dataset):
    def __init__(self, tokenizer, partition):
        self.partition = partition
        print(partition)
        self.dataset = load_dataset(
                "csv",
                data_files={f"{self.partition}": f"/scratch/zar8jw/data/{partition}.csv"},
                delimiter=",",
            )
        self.tokenizer = tokenizer
    
    def __len__(self):
        return self.dataset[self.partition].shape[0]
    
    def convert_to_features(self, example_batch):
        # Create prompt and tokenize contexts and questions
        
        # print("Input Text: ", self.clean_text(example_batch["text"]))

        input_ = example_batch["Medic Notes"]
        impression_ = example_batch["Impression"]
        target_ = example_batch["Protocols"]

        prompt = f"""Based on the medic note, recommend EMS protocols for first responder to save the patient. Return the result in the json format defined as follows. If there are multiple recommendations, separate each with comma. You can only return your result one time. \n[\n"prediction1", "prediction2", ...\n]\nHere are all EMS Protocols to choose from: {ems_labels}\nmedic note: {input_}\n\n---\n\nRESPONSE:"""

        # response_ = f"""Let's think step by step, \nStep1: Symptom Abstraction: analyzing patient symptoms\n{impression_}\nStep2: Candidate Disease Recall\n{target_}\nStep3: Organize the format from Step2 in the defined json format\n{[t.strip() for t in target_.split(";")]}"""
        response_ = f"""{[t.strip() for t in target_.split(";")]}"""


        prompt_ids = self.tokenizer.encode(self.tokenizer.bos_token + prompt, add_special_tokens=False)
        label_ids = self.tokenizer.encode(response_ + self.tokenizer.eos_token, add_special_tokens=False)

        sample = {
            "input_ids": prompt_ids + label_ids,
            "attention_mask": [1] * len(prompt_ids + label_ids),
            "labels": [-100] * len(prompt_ids) + label_ids
        }

        return sample
    
    def __getitem__(self, index):
        return self.convert_to_features(self.dataset[self.partition][int(index)])


def get_dataset(dataset_config, tokenizer, partition):
    """cover function for handling loading the working dataset"""
    """dataset loading"""

    dataset = ems(
        tokenizer=tokenizer,
        partition=partition
    )

    return dataset
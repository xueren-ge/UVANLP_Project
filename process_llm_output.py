import os
import json
import numpy as np
import re
import pandas as pd
from llama_fn import LLAMA
from sklearn import metrics
from sklearn.metrics import multilabel_confusion_matrix
from tqdm import tqdm

# model_name_or_path = "m42-health/Llama3-Med42-70B"
model_name_or_path = "ProbeMedicalYonseiMAILab/medllama3-v20"
# model_name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_name_or_path = "meta-llama/Meta-Llama-3-70B-Instruct"
# model_name_or_path = "meta-llama/Llama-3.1-8B-Instruct"
# model_name_or_path = "meta-llama/Llama-3.1-70B-Instruct"
# model_name_or_path = "meta-llama/Meta-Llama-3.1-70B-Instruct"
# model_name_or_path = "meta-llama/Meta-Llama-3.1-405B-Instruct"
# model_name_or_path = "meta-llama/Meta-Llama-3.1-405B-Instruct-FP8"
lm = LLAMA(model_name_or_path=model_name_or_path)

with open("/scratch/zar8jw/data/EMS_Protocol.json", 'r') as f:
    ems_labels = json.load(f)


def find_most_similar(protocol):
    prompt = f"""The {protocol} is not in the EMS Protocols. In the provided EMS Protocols, what is the most similar one to {protocol}? Your result MUST be exactly in the provided EMS Protocols.\nYou must return the the most similar one in a python list format (e.g.: ["your result"]). You can only return your result in one time.\nEMS Protocols: {ems_labels}\n---\nRESPONSE:"""
    messages = [{"role": "system", "content": "you are an enmergency medical service professional."}]
    messages = [{"role": "user", "content": prompt}]
    response = lm.apply_medllama3(messages)
    print(response)
    error, json_data = lm.extract_json(response, pattern=r'\[.*?\]')
    if error:
        json_data = lm.handle_error(messages, response, pattern=r'\[.*?\]')

    return json_data[0]


def encode(pred):
    one_hot = [0] * len(ems_labels)
    for p in pred:
        index = ems_labels.index(p)
        one_hot[index] = 1
    return one_hot

def check_revise(raw_pred):
    pred = []
    for p in raw_pred:
        if p in ems_labels:
            pred.append(p)
        else:
            p_star = find_most_similar(p)
            while p_star not in ems_labels:
                p_star = find_most_similar(p_star)
            pred.append(p_star)
    return pred

def evaluate_metrics(y_true, y_pred):
    confusion_matrix = multilabel_confusion_matrix(np.array(y_true), np.array(y_pred))
    TP = [i[1][1] for i in confusion_matrix]
    FP = [i[0][1] for i in confusion_matrix]
    FN = [i[1][0] for i in confusion_matrix]
    TN = [i[0][0] for i in confusion_matrix]

    report = metrics.classification_report(np.array(y_true), np.array(y_pred), target_names=ems_labels, output_dict=True)
    for i in range(len(ems_labels)):
        pname = ems_labels[i]
        report[pname]['TP'] = int(TP[i])
        report[pname]['FP'] = int(FP[i])
        report[pname]['FN'] = int(FN[i])
        report[pname]['TN'] = int(TN[i])
    return report


if __name__ == "__main__":
    # name = "3.1-8B-finetune"
    # name = "3.1-8B-finetune-directresp"
    # name = "cot_prompt"
    # name = "medllama3-v20_direct_prompt"
    name = "medllama3-v20_cot_prompt"

    with open(f"./results/{name}.json", 'r') as f:
        res = json.load(f)
    
    df = pd.read_csv("/scratch/zar8jw/data/test.csv")
    # ret = []

    assert len(res) == len(df), print(len(res), len(df))
    y_true = []
    y_pred = []
    if not os.path.exists(f"./log/report/{name}"):
        os.makedirs(f"./log/report/{name}")
    
    for i in tqdm(range(len(df))):
        print(i)

        if f"{i}.json" in os.listdir(f"./log/report/{name}"):
            with open(f"./log/report/{name}/{i}.json", "r") as f:
                cur_res = json.load(f)
            y_true.append(cur_res["gt"])
            y_pred.append(cur_res["pred"])
            continue

        ps = [p.strip().lower() for p in df["Protocols"][i].split(';')]
        gt = encode(ps)

        pred_raw = check_revise(res[i])
        pred = encode(pred_raw)
        
        y_true.append(gt)
        y_pred.append(pred)

        cur_res = {
            "gt": gt,
            "pred": pred
        }
        with open(f"./log/report/{name}/{i}.json", 'w') as f:
            json.dump(cur_res, f, indent=4)


    report = evaluate_metrics(y_true, y_pred)
    print(report)
    with open(f"./results/{name}_report.json", 'w') as f:
        json.dump(report, f, indent=4)
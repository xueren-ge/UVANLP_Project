# UVANLP Project

We will use test multiple different prompts to study the differences of prompt engineering on diagnosis prediction, and we also compare prompt engineering methods with parameter-efficient fine-tuning method.

* Prompts
We use two large language models Llama3.1-70b and medllama3-v20 as our base models, and tried two different prompts: direct prompts and chain-of-diagnosis (CoD) prompts. In direct prompt (directprompt.py), we ask LLMs to select the recommended EMS protocols from the given list. In the CoD prompt(cot prompt.py), we ask LLMs to first abstract the symptoms in the EHR and then do the disease recall.

* Finetuning
Refer the finetuning folder for training the model. For the inference, use finetune_inference.py.

* Report Generation
Use process_llm_output.py to generate the final report for performance evaluation. Please specify the model id and the path in the file.
import os
import json
import time
import torch
import torch.nn as nn
import random
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm
from vllm import LLM, SamplingParams
from openai import OpenAI
from transformers import BertModel,BertPreTrainedModel
from torch.nn import CrossEntropyLoss

# 自动下载模型时，指定使用modelscope; 否则，会从HuggingFace下载
os.environ['VLLM_USE_MODELSCOPE']='True'

def llm_parser_args():
    parser = ArgumentParser()
    parser.add_argument("--data",type=str,default="data/new_split")
    parser.add_argument("--split",type=str,required=True) # 第几个划分
    parser.add_argument("--shot",type=int,required=True) # 样例的数量
    parser.add_argument("--retrieve",type=str,required=True) # 样例检索的策略,random,similarity
    parser.add_argument("--prompt",type=str,default="prompt/interaction.txt") # 提示词
    parser.add_argument("--model",type=str,required=True) # 大模型
    parser.add_argument("--moda",type=str,default='greedy') # 解码策略
    parser.add_argument("--output_dir",type=str,default="output/llm-interaction")
    return parser.parse_args()

# def load_llms(model_name,moda,max_tokens=1024,max_model_len=4096):
#     model_dir = ""
#     stop_token_ids = []
#     if model_name.startswith("qwen2.5-7b"):
#         print("Loading Qwen2.5-7B-Instruct")
#         model_dir = "Qwen/Qwen2.5-7B-Instruct"
#     elif model_name.startswith("glm4-9b"):
#         print("Loading glm-4-9b")
#         model_dir = "THUDM/glm-4-9b"
#         stop_token_ids = [151329]
#     elif model_name.startswith("gemma-2-9b"):
#         print("Loading gemma-2-9b")
#         model_dir = "LLM-Research/gemma-2-9b-it"
#     elif model_name.startswith("llama3.1-8b"):
#         print("Loading Llama-3.1-8B-Instruct")
#         model_dir = "LLM-Research/Meta-Llama-3.1-8B-Instruct" # 这里使用的modelscope，如果使用huggingface，请改成meta-llama/
#         # model_dir = "/home/jindongming/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct"
#         stop_token_ids = [128001,128009]
#     elif model_name.startswith("mistral-7b"):
#         print("Loading mistral-7b")
#         model_dir = "mistralai/Mistral-7B-v0.3"
#         stop_token_ids = []
#     elif model_name.startswith("deepseek"):
#         print("Loading deepseek-LLama3-8B")
#         model_dir = "/home/jindongming/.cache/huggingface/hub/DeepSeek-R1-Distill-Llama-8B"
#         stop_token_ids = [151329, 151336, 151338]
#     if moda=='greedy':
#         sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens, stop_token_ids=stop_token_ids,n=1)
#     else:
#         sampling_params = SamplingParams(temperature=0.4, top_p=0.95, max_tokens=max_tokens, n=20)
#     os.environ['VLLM_USE_MODELSCOPE']='True'
#     model = LLM(model=model_dir,tokenizer=None,max_model_len=max_model_len,trust_remote_code=True,gpu_memory_utilization=0.9)
#     return model,sampling_params

def gpt_completion(args,prompt):
    client = OpenAI(
        api_key = "sk-sR8RiK6YYrtk8Rss1b29047069804d108211285c7a25356c", # 填写上api-key
        base_url= "https://api.yesapikey.com/v1"
    )
    flag = False
    while not flag:
        try:
            response = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=args.model,
                    temperature=0   
            )
            flag=True
        except Exception as e:
            print(e)
            time.sleep(0.5)
    return response.choices[0].message.content

# def llm_reason():
#     args = llm_parser_args()
#     model,sampling_params = load_llms(args.model,args.moda)
#     # construct prompt
#     with open(os.path.join(args.data,f"split_{args.split}/train.json"),'r',encoding='utf-8') as file:
#         train_data = json.load(file)
#     with open(os.path.join(args.data,f"split_{args.split}/test.json"),'r',encoding='utf-8') as file:
#         test_data = json.load(file)
#     with open(args.prompt,'r',encoding='utf-8') as prompt_file:
#         prompt_template = prompt_file.read()
#     if args.retrieve=="random":
#         prompt_list = []
#         for test_item in test_data:
#             input_req = test_item["text"]
#             entities = test_item["entities"]
#             example_prompt = "\n## Examples\n"
#             examples = random.sample(train_data, args.shot)
#             for shot_item in examples:
#                 shot_input = shot_item["text"]
#                 shot_entity = shot_item["entity"]
#                 shot_answer = shot_item["relation"]
#                 example_prompt += f"Input:{shot_input}\n"
#                 example_prompt += f"Entity: {shot_entity}\n"
#                 example_prompt += f"Answer:{shot_answer}\n"
#             prompt = prompt_template.format(examples=example_prompt,input_req=input_req,entities=entities)
#             prompt_list.append(prompt)
#     elif args.retrieve=="similarity":
#         simi_file = os.path.join(args.data,f"split_{args.split}/simi.json")
#         with open(simi_file,'r',encoding='utf-8') as shot_file:
#             simi_data = json.load(shot_file)
#         prompt_list = []
#         for i,test_item in enumerate(test_data):
#             input_req = test_item["text"]
#             entities = test_item["entity"]
#             example_prompt = "\n## Examples\n"
#             examples = simi_data[i]["id"][:args.shot]
#             for shot_id in examples:
#                 shot_item = train_data[shot_id]
#                 shot_input = shot_item["text"]
#                 shot_entity = shot_item["entity"]
#                 shot_answer = shot_item["relation"]
#                 example_prompt += f"Input:{shot_input}\n"
#                 example_prompt += f"Entity: {shot_entity}\n"
#                 example_prompt += f"Answer:{shot_answer}\n"
#             prompt = prompt_template.format(examples=example_prompt,input_req=input_req,entity_list=entities)
#             prompt_list.append(prompt)
    
#     predict_list = []
#     for prompt in tqdm(prompt_list, desc='generate answer'):
#         predict = model.generate([prompt],sampling_params)
#         predict = predict[0].outputs[0].text
#         predict_dict = {
#             "predict":predict
#         }
#         predict_list.append(predict_dict)
    
#     # save results
#     out_dir = os.path.join(args.output_dir,f"{args.model}_{args.retrieve}_{args.shot}")
#     if not os.path.exists(out_dir):
#         os.makedirs(out_dir)
#     out_path = os.path.join(out_dir,f"{args.split}.json")
#     json_data = json.dumps(predict_list,ensure_ascii=False,indent=2)
#     with open(out_path,'w',encoding='utf-8') as output_file:
#         output_file.write(json_data)

def gpt_reason():
    args = llm_parser_args()
    # construct prompt
    with open(os.path.join(args.data,f"split_{args.split}/train.json"),'r',encoding='utf-8') as file:
        train_data = json.load(file)
    with open(os.path.join(args.data,f"split_{args.split}/test.json"),'r',encoding='utf-8') as file:
        test_data = json.load(file)
    with open(args.prompt,'r',encoding='utf-8') as prompt_file:
        prompt_template = prompt_file.read()
    if args.retrieve=="random":
        prompt_list = []
        for test_item in test_data:
            input_req = test_item["text"]
            entities = test_item["entity"]
            example_prompt = "\n## Examples\n"
            examples = random.sample(train_data, args.shot)
            for shot_item in examples:
                shot_input = shot_item["text"]
                shot_entity = shot_item["entity"]
                shot_answer = shot_item["relation"]
                example_prompt += f"Input:{shot_input}\n"
                example_prompt += f"Entity: {shot_entity}\n"
                example_prompt += f"Answer:{shot_answer}\n"
            prompt = prompt_template.format(examples=example_prompt,input_req=input_req,entity_list=entities)
            prompt_list.append(prompt)
    elif args.retrieve=="similarity":
        simi_file = os.path.join(args.data,f"split_{args.split}/simi.json")
        with open(simi_file,'r',encoding='utf-8') as shot_file:
            simi_data = json.load(shot_file)
        prompt_list = []
        for i,test_item in enumerate(test_data):
            input_req = test_item["text"]
            entities = test_item["entity"]
            example_prompt = "\n## Examples\n"
            examples = simi_data[i]["id"][:args.shot]
            for shot_id in examples:
                shot_item = train_data[shot_id]
                shot_input = shot_item["text"]
                shot_entity = shot_item["entity"]
                shot_answer = shot_item["relation"]
                example_prompt += f"Input:{shot_input}\n"
                example_prompt += f"Entity: {shot_entity}\n"
                example_prompt += f"Answer:{shot_answer}\n"
            prompt = prompt_template.format(examples=example_prompt,input_req=input_req,entity_list=entities)
            prompt_list.append(prompt)
    
    predict_list = []
    for prompt in tqdm(prompt_list, desc='generate answer'):
        predict = gpt_completion(args,prompt)
        predict_dict = {
            "predict":predict
        }
        predict_list.append(predict_dict)
    
    # save results
    out_dir = os.path.join(args.output_dir,f"{args.model}_{args.retrieve}_{args.shot}")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = os.path.join(out_dir,f"{args.split}.json")
    json_data = json.dumps(predict_list,ensure_ascii=False,indent=2)
    with open(out_path,'w',encoding='utf-8') as output_file:
        output_file.write(json_data)

def main():
    gpt_reason()

if __name__=="__main__":
    main()
    pass
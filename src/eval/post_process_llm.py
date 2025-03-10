import os
import json
import re
import ast

def deepseek_process(result):
    kong = "{'Software System': [], 'Physical Device': [], 'Environment Object': [], 'External System': [], 'System Requirements': [], 'Shared Phenomena': []}"
    if result[:2] == "[]":
        return kong
    print("======")
    print(result)
    # 使用正则表达式找到第一个字典
    match = re.search(r"{[^}]*}", result)
    if match:
        dict_str = match.group(0)
    print("**********")
    print(dict_str)
    return dict_str

def gemma_process(result):
    kong = "{'Software System': [], 'Physical Device': [], 'Environment Object': [], 'External System': [], 'System Requirements': [], 'Shared Phenomena': []}"
    if result==" \n\n\n" or result==" \n" or result=="\n\n\n" or result == "\n\n\n\n" or result == " \n\n\n\n":
        return kong
    print("======")
    # print(result)
    print("dongming")
    print(repr(result))
    print("dongming")
    # 使用正则表达式找到第一个字典
    match = re.search(r"{[^}]*}", result)
    if match:
        dict_str = match.group(0)
    print("**********")
    print(dict_str)
    return dict_str

def llama_process(result):
    # kong = "{'Software System': [], 'Physical Device': [], 'Environment Object': [], 'External System': [], 'System Requirements': [], 'Shared Phenomena': []}"
    # if result==" \n\n\n" or result==" \n":
    #     return kong
    print("======")
    print(result)
    # 使用正则表达式找到第一个字典
    match = re.search(r"{[^}]*}", result)
    if match:
        dict_str = match.group(0)
    print("**********")
    print(dict_str)
    return dict_str

def qwen_process(result):
    # kong = "{'Software System': [], 'Physical Device': [], 'Environment Object': [], 'External System': [], 'System Requirements': [], 'Shared Phenomena': []}"
    # if result==" \n\n\n" or result==" \n":
    #     return kong
    print("======")
    print(result)
    # 使用正则表达式找到第一个字典
    match = re.search(r"{[^}]*}", result)
    if match:
        dict_str = match.group(0)
    print("**********")
    print(dict_str)
    return dict_str

def gpt_process(result):
    # kong = "{'Software System': [], 'Physical Device': [], 'Environment Object': [], 'External System': [], 'System Requirements': [], 'Shared Phenomena': []}"
    # if result==" \n\n\n" or result==" \n":
    #     return kong
    print("======")
    print(result)
    # 使用正则表达式找到第一个字典
    match = re.search(r"{[^}]*}", result)
    if match:
        dict_str = match.group(0)
    print("**********")
    print(dict_str)
    return dict_str

def parser_predict_entity():
    retrieve = "similarity"
    shot_list = [16]
    sys_list = [1]
    for shot in shot_list:
        for sys in sys_list:
            result = []
            out_dir = f"metric/llm/qwen_{retrieve}_{shot}"
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            model_dir = os.path.join(out_dir,"predict")
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            input_file = f"output/llm/qwen2.5-7b_{retrieve}_{shot}/{sys}.json"
            with open(input_file,'r',encoding='utf-8') as file:
                data = json.load(file)
            for i,predict in enumerate(data):
                predict = predict["predict"]
                predict = qwen_process(predict)
                print(f"going on qwen_{retrieve}_{shot}/{sys}.json")
                predict = ast.literal_eval(predict)
                result.append({"entity":predict})
            out_path = os.path.join(model_dir,f"{sys}.json")
            json_data = json.dumps(result,ensure_ascii=False,indent=2)
            with open(out_path,'w',encoding='utf-8') as output_file:
                output_file.write(json_data)

def deepseek_process_interaction(result):
    kong = "{'Interface': [], 'Requirements Reference': [], 'Requirements Constraint': []}"
    if result.startswith(" \n\nWait, the") or result.startswith(" \nAnswer:[]\n\nWait"):
        return kong
    if result.startswith(" \n\nWait, but") or result.startswith(" \nWait, the entities"):
        return kong
    if result.startswith(" \nSentence: "):
        return kong
    if result.startswith(" \n\nThe task is"):
        return kong
    print("======")
    print(result)
    # 使用正则表达式找到第一个字典
    match = re.search(r"{[^}]*}", result)
    if match:
        dict_str = match.group(0)
    print("**********")
    print(dict_str)
    return dict_str

def llama_process_interaction(result):
    kong = "{'Interface': [], 'Requirements Reference': [], 'Requirements Constraint': []}"
    if result.startswith(" \nAnswer:[]"):
        return kong
    print("======")
    print(result)
    # 使用正则表达式找到第一个字典
    match = re.search(r"{[^}]*}", result)
    if match:
        dict_str = match.group(0)
    print("**********")
    print(dict_str)
    return dict_str

def qwen_process_interaction(result):
    # kong = "{'Interface': [], 'Requirements Reference': [], 'Requirements Constraint': []}"
    # if result.startswith(" \nAnswer:[]"):
    #     return kong
    print("======")
    print(result)
    # 使用正则表达式找到第一个字典
    match = re.search(r"{[^}]*}", result)
    if match:
        dict_str = match.group(0)
    print("**********")
    print(dict_str)
    return dict_str

def gpt4_process_interaction(result):
    # kong = "{'Interface': [], 'Requirements Reference': [], 'Requirements Constraint': []}"
    # if result.startswith(" \nAnswer:[]"):
    #     return kong
    print("======")
    print(result)
    # 使用正则表达式找到第一个字典
    match = re.search(r"{[^}]*}", result)
    if match:
        dict_str = match.group(0)
    print("**********")
    print(dict_str)
    return dict_str

def parser_predict_interaction():
    retrieve = "similarity"
    shot_list = [16]
    sys_list = [1]
    for shot in shot_list:
        for sys in sys_list:
            result = []
            out_dir = f"metric/llm-interaction/deepseek_{retrieve}_{shot}"
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            model_dir = os.path.join(out_dir,"predict")
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            input_file = f"output/llm-interaction/deepseek_{retrieve}_{shot}/{sys}.json"
            with open(input_file,'r',encoding='utf-8') as file:
                data = json.load(file)
            for i,predict in enumerate(data):
                predict = predict["predict"]
                predict = deepseek_process_interaction(predict)
                print(f"going on deepseek_{retrieve}_{shot}/{sys}.json")
                predict = ast.literal_eval(predict)
                result.append({"interaction":predict})
            out_path = os.path.join(model_dir,f"{sys}.json")
            json_data = json.dumps(result,ensure_ascii=False,indent=2)
            with open(out_path,'w',encoding='utf-8') as output_file:
                output_file.write(json_data)

if __name__=="__main__":
    # parser_predict_entity()
    parser_predict_interaction()
            

            
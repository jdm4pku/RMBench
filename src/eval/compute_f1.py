import os
import json
from argparse import ArgumentParser

def get_parser_args():
    parser = ArgumentParser()
    parser.add_argument("--input_dir",type=str,default="output/sm/hmm")
    parser.add_argument("--split",type=str,default=1) # 第几个划分
    parser.add_argument("--output_dir",type=str,default="metric/sm/hmm/f1")
    return parser.parse_args()

def compute_f1(predict_path,human_path):
    f1_result = {
        "Software System":[0,0,0],
        "Physical Device":[0,0,0],
        "Environment Object":[0,0,0],
        "External System":[0,0,0],
        "System Requirements":[0,0,0],
        "Shared Phenomena":[0,0,0]
    }
    predict_model = {
        "Software System":None,
        "Physical Device":None,
        "Environment Object":None,
        "External System":None,
        "System Requirements":None,
        "Shared Phenomena":None
    }
    ground_model = {
        "Software System":None,
        "Physical Device":None,
        "Environment Object":None,
        "External System":None,
        "System Requirements":None,
        "Shared Phenomena":None
    }
    with open(predict_path,'r',encoding='utf-8') as file:
        predict_data = json.load(file)
    with open(human_path,'r',encoding='utf-8') as file:
        human_data = json.load(file)
    total_tp = 0
    total_fp = 0
    total_fn = 0
    for key in f1_result:
        flat_predict = []
        flat_ground = []
        for i,predict in enumerate(predict_data):
            if key not in predict['entity']:
                continue
            predict = predict['entity'][key]
            ground = human_data[i]['entity'][key]
            for item in predict:
                if item not in flat_predict:
                    flat_predict.append(item)
            for item in ground:
                if item not in flat_ground:
                    flat_ground.append(item)
        predict_model[key] = flat_predict
        ground_model[key] = flat_ground
        TP = len(set(flat_ground).intersection(set(flat_predict)))
        FP = len(set(flat_predict)) - TP
        FN = len(set(flat_ground)) - TP
        f1_result[key][0]=TP
        f1_result[key][1]=FP
        f1_result[key][2]=FN
        precision = TP / (TP + FP) if TP + FP != 0 else 0
        recall = TP / (TP + FN) if TP + FN != 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall !=0 else 0
        f1_result[key].append(precision)
        f1_result[key].append(recall)
        f1_result[key].append(f1)
        total_fp +=FP
        total_tp +=TP
        total_fn +=FN
    total_p = total_tp / (total_tp + total_fp) if total_tp + total_fp != 0 else 0
    total_r = total_tp / (total_tp + total_fn) if total_tp + total_fn !=0 else 0
    total_f1 = 2 * (total_p * total_r) / (total_p + total_r) if total_p + total_r!=0 else 0
    f1_result["total"] = [total_tp,total_fp,total_fn,total_p,total_r,total_f1]
    return f1_result,predict_model,ground_model


if __name__=="__main__":
    args = get_parser_args()
    predict_path = os.path.join(args.input_dir,f"{args.split}.json")
    ground_path = os.path.join(f"data/split_v1/split_{args.split}","test.json")
    f1_result,predict_model,ground_model = compute_f1(predict_path,ground_path)
    print(f1_result)
    f1_result_dir = f"{args.output_dir}/f1"
    if not os.path.exists(f1_result_dir):
        os.makedirs(f1_result_dir)
    output_file = f"{f1_result_dir}/{args.split}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(f1_result, f, ensure_ascii=False, indent=4)
    
    model_result_dir = f"{args.output_dir}/model"
    if not os.path.exists(model_result_dir):
        os.makedirs(model_result_dir)
    predict_output_file = f"{model_result_dir}/{args.split}_predict.json"
    with open(predict_output_file, "w", encoding="utf-8") as f:
        json.dump(predict_model, f, ensure_ascii=False, indent=4)
    ground_output_file = f"{model_result_dir}/{args.split}_ground.json"
    with open(ground_output_file, "w", encoding="utf-8") as f:
        json.dump(ground_model, f, ensure_ascii=False, indent=4)
import os
import json
from argparse import ArgumentParser
from simcse import SimCSE
# import faiss
import numpy as np
from tqdm import tqdm

def llm_parser_args():
    parser = ArgumentParser()
    parser.add_argument("--data",type=str,default="data/split_v1")
    parser.add_argument("--split",type=str,required=True) # 第几个划分
    parser.add_argument("--shot",type=int,required=True) # 样例的数量
    return parser.parse_args()

def get_few_shot(args):
    args = llm_parser_args()
    # construct prompt
    with open(os.path.join(args.data,f"split_{args.split}/train.json"),'r',encoding='utf-8') as file:
        train_data = json.load(file)
    with open(os.path.join(args.data,f"split_{args.split}/test.json"),'r',encoding='utf-8') as file:
        test_data = json.load(file)
    emb_model = SimCSE("models/sup-simcse-bert-base-uncased")
    test_sentence = [item["text"] for item in test_data]
    train_sentence = [item["text"] for item in train_data]
    train_embedding = np.array(emb_model.encode(train_sentence))
    test_embedding = np.array(emb_model.encode(test_sentence))
    # Calculate cosine similarity
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    indices = []
    for test_emb in test_embedding:
        similarities = [cosine_similarity(test_emb, train_emb) for train_emb in train_embedding]
        sorted_indices = np.argsort(similarities)[-args.shot:][::-1]
        indices.append(sorted_indices)
    simi_data = []
    for i in tqdm(range(len(test_sentence)), desc='Processing sentences'):
        sentence = test_sentence[i]
        text = sentence
        id = indices[i].tolist()
        example = [train_sentence[idx] for idx in indices[i]]
        item = {
            "text":text,
            "id":id,
            "example":example
        }
        simi_data.append(item)
    json_data = json.dumps(simi_data,ensure_ascii=False,indent=2)
    out_path = os.path.join(args.data,f"split_{args.split}/simi.json")
    with open(out_path, 'w', encoding='utf-8') as output_file:
        output_file.write(json_data)
    print(f"Finish writing split_{args.split}")

if __name__=="__main__":
    args = llm_parser_args()
    get_few_shot(args)
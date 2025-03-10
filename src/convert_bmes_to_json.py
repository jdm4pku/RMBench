import os
import json



def save_predict_result(test_word_lists,pred_tag_lists,output_path):
    result = []
    # 定义各类实体
    entity_categories = {
        "Software System": [],
        "Physical Device": [],
        "Environment Object": [],
        "External System": [],
        "System Requirements": [],
        "Shared Phenomena": []
    }
    for i,test_word in enumerate(test_word_lists):
        text = " ".join(test_word)
        entities = {category: [] for category in entity_categories}
        current_entity = None
        for j, (word,tag) in enumerate(zip(test_word,pred_tag_lists[i])):
            entity_type = tag[2:]
            if tag.startswith('B-'):
                if current_entity:
                    entities[current_entity["label"]].append(current_entity["text"])
                current_entity = {
                    "start": j,
                    "end": j,
                    "label": entity_type,
                    "text": word
                }
            elif tag.startswith('M-'):
                if current_entity:
                    current_entity["end"] = j
                    current_entity["text"] += " " + word
            elif tag.startswith('E-'):  # 实体结束
                if current_entity:
                    current_entity["end"] = j
                    current_entity["text"] += " " + word
                    # 保存实体并重置当前实体
                    entities[current_entity["label"]].append(current_entity["text"])
                    current_entity = None
            elif tag.startswith('S-'):  # 单一实体
                entities[entity_type].append(word)
        result.append({
            "text": text + "\n",  # 保留原始文本并加换行符
            "entity": entities
        })
    json_data = json.dumps(result,ensure_ascii=False,indent=2)
    with open(output_path,'w',encoding='utf-8') as output_file:
        output_file.write(json_data)


model_list = ["bert_softmax","bert_crf"]

for model in model_list:
    for split in range(1,13):
        bems_data = []
        bems_path = f"output/bert/{model}/split_{split}/test_prediction.json"
        with open(bems_path, 'r') as file:
            for line in file:
                bems_data.append(json.loads(line))
        bems_tag_data = []
        for item in bems_data:
            tag = item["tag_seq"].split(" ")
            bems_tag_data.append(tag)
        test_file = f"data/new_split/split_{split}/test.char.bmes"
        test_data = []
        current_sentence = []
        with open(test_file,'r',encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:
                    word = line.split()[0]
                    current_sentence.append(word)
                else:
                    if current_sentence:
                        test_data.append(current_sentence)  # 将当前句子加入列表
                        current_sentence = []
            if current_sentence:
                test_data.append(current_sentence)
        # assert len(test_data)==len(bems_tag_data)
        i = 0
        for test_item,tag_item in zip(test_data,bems_tag_data):
            print(i)
        save_path = f"output/bert/{model}/predict/{split}.json"
        save_predict_result(test_data,bems_tag_data,save_path)
            




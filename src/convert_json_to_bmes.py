import json
import re
import os


def add_spaces_to_punctuation(text):
    # 使用正则表达式匹配所有标点符号前加上空格
    # return re.sub(r'([.,!?;:(){}[\]"\'<>/’])', r' \1', text)
    return re.sub(r'([.,!?;:(){}[\]"\'<>’/‘’“”\-~%_*=\\#&+，])', r' \1 ', text).strip()

# 将 JSON 数据转化为 BMES 格式
def convert_to_bmes(text, entities):
    text = add_spaces_to_punctuation(text)
    words = text.split()  # 将文本按空格分割成单词
    bmes_tags = ['O'] * len(words)  # 初始化标签为 'O'，表示没有实体

    # 将实体和对应的标签赋值
    for entity_type, entity_list in entities.items():
        for entity in entity_list:
            entity = add_spaces_to_punctuation(entity)
            print(entity)
            print("*****************")
            start_idx = text.find(entity)  # 查找实体的起始位置
            end_idx = start_idx + len(entity)  # 查找实体的结束位置
            if start_idx != -1:
                # 根据实体的起始和结束位置标记 BMES 标签
                entity_words = entity.split()
                if len(entity_words)==1:
                    word_idx = text.split().index(entity_words[0])
                    bmes_tags[word_idx] = f"S-{entity_type}"
                    continue
                for i, word in enumerate(entity_words):
                    word_idx = text.split().index(word)  # 查找当前单词的位置
                    if i == 0:
                        bmes_tags[word_idx] = f"B-{entity_type}"  # 第一个单词是 'B-'
                    elif i == len(entity_words) - 1:
                        bmes_tags[word_idx] = f"E-{entity_type}"  # 最后一个单词是 'E-'
                    else:
                        bmes_tags[word_idx] = f"M-{entity_type}"  # 中间的单词是 'M-'
    return list(zip(words, bmes_tags))  # 返回单词和对应标签的列表

def get_bems_data():
    for i in range(1,8):
        data_dir = f"data/split_v1/split_{i}"
        with open(os.path.join(data_dir,"train.json"),'r',encoding='utf-8') as file:
            train_data = json.load(file)
        with open(os.path.join(data_dir,'train.char.bmes'), 'w', encoding='utf-8') as f:
            for item in train_data:
                text = item['text']
                print(text)
                print("========================")
                entities = item["entity"]
                bmes = convert_to_bmes(text,entities)
                for word, tag in bmes:
                    f.write(f"{word}\t{tag}\n")
                # 写入空行来分隔不同的样本
                f.write("\n")
        with open(os.path.join(data_dir,"test.json"),'r',encoding='utf-8') as file:
            train_data = json.load(file)
        with open(os.path.join(data_dir,'test.char.bmes'), 'w', encoding='utf-8') as f:
            for item in train_data:
                text = item['text']
                entities = item["entity"]
                bmes = convert_to_bmes(text,entities)
                for word, tag in bmes:
                    f.write(f"{word}\t{tag}\n")
                # 写入空行来分隔不同的样本
                f.write("\n")

def main():
    get_bems_data()

if __name__=="__main__":
    main()
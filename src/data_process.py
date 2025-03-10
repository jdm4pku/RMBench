import os
import json

def split_dataset(in_dir,out_dir):
    # json_files = ["C2C.json","CCS.json","CTS.json","HCS.json","MES.json","SFS.json","TCS.json"]
    json_files = [f for f in os.listdir(in_dir)]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for idx, test_file in enumerate(json_files,start=1):
        split_dir = os.path.join(out_dir, f"split_{idx}")
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)
        train_files = [f for f in json_files if f!=test_file]
        train_data = []
        for train_file in train_files:
            with open(os.path.join(in_dir,train_file),'r',encoding='utf-8') as f:
                train_data.extend(json.load(f))
        with open(os.path.join(split_dir,'train.json'),'w',encoding='utf-8') as f:
            json.dump(train_data,f,indent=4)
        with open(os.path.join(in_dir, test_file), 'r') as f:
            test_data = json.load(f)
        with open(os.path.join(split_dir, 'test.json'), 'w') as f:
            json.dump(test_data, f, indent=4)
        print(f"划分 {idx} 完成,测试集包含 {test_file}")

def replace_key_words(in_dir,out_dir):
    def replace_rule(path):
        with open(path,'r',encoding='utf-8') as f:
            data = json.load(f)
        new_data = []
        for old_item in data:
            print(old_item)
            print("=================")
            new_item = {"text":"","entity":{},"relation":{}}
            new_item["text"] = old_item['text']
            new_item["entity"]["Software System"] = old_item["entity"]["Machine Domain"]
            new_item["entity"]["Physical Device"] = old_item["entity"]["Physical Device"]
            new_item["entity"]["Environment Object"] = old_item["entity"]["Environment Entity"]
            new_item["entity"]["External System"] = old_item["entity"]["Design Domain"]
            new_item["entity"]["System Requirements"] = old_item["entity"]["Requirements"]
            new_item["entity"]["Shared Phenomena"] = old_item["entity"]["Shared Phenomena"]
            new_item["relation"]["Phenomena Interface"] = old_item["relation"]["interface"]
            new_item["relation"]["requirements reference"] = old_item["relation"]["requirements reference"]
            new_item["relation"]["requirements constraint"] = old_item["relation"]["requirements constraints"]
            new_data.append(new_item)
        return new_data
    for i in range(1,13):
        train_path = os.path.join(in_dir,f"split_{i}/train.json")
        test_path = os.path.join(in_dir,f"split_{i}/test.json")
        new_train = replace_rule(train_path)
        new_test = replace_rule(test_path)
        new_split_dir = os.path.join(out_dir,f"split_{i}")
        if not os.path.exists(new_split_dir):
            os.makedirs(new_split_dir)
        with open(os.path.join(out_dir,f"split_{i}/train.json"), 'w') as f:
            json.dump(new_train, f, indent=4)
        with open(os.path.join(out_dir,f"split_{i}/test.json"), 'w') as f:
            json.dump(new_test, f, indent=4)

def main():
    # in_dir = "data/total"
    # out_dir = "data/split_v2"
    # split_dataset(in_dir,out_dir)
    in_dir = "data/split_v2"
    out_dir = "data/split_v2"
    replace_key_words(in_dir,out_dir)

if __name__=="__main__":
    main()
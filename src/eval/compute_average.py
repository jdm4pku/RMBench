import os
import json
import argparse

"""
每个key汇总求平均
"""
def compute_f1_average(directory):
    average_result = {
        "Software System": [0, 0, 0],
        "Physical Device": [0, 0, 0],
        "Environment Object": [0, 0, 0],
        "External System": [0, 0, 0],
        "System Requirements": [0, 0, 0],
        "Shared Phenomena": [0, 0, 0],
        "total": [0, 0, 0]
    }
    json_file_index = [1,2,3,4,5,7]
    for i in json_file_index:
        json_file = os.path.join(directory, f"{i}.json")
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
        for key in average_result:
            average_result[key][0] += data[key][3]
            average_result[key][1] += data[key][4]
            average_result[key][2] += data[key][5]
    for key in average_result:
        average_result[key][0] /= len(json_file_index)
        average_result[key][1] /= len(json_file_index)
        average_result[key][2] /= len(json_file_index)
    out_dir = os.path.join(directory, "average.json")
    json_data = json.dumps(average_result, ensure_ascii=False, indent=2)
    with open(out_dir, 'w', encoding='utf-8') as output_file:
        output_file.write(json_data)
    print(f"Average result has been saved in {out_dir}")

def compute_rough_average(directory):
    average_result = {
        "Software System": [0, 0, 0],
        "Physical Device": [0, 0, 0],
        "Environment Object": [0, 0, 0],
        "External System": [0, 0, 0],
        "System Requirements": [0, 0, 0],
        "Shared Phenomena": [0, 0, 0]
    }
    json_file_index = [1,2,3,4,5,7]
    for i in json_file_index:
        json_file = os.path.join(directory, f"{i}.json")
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
        for key in average_result:
            average_result[key][0] += data[key][0]
            average_result[key][1] += data[key][1]
            average_result[key][2] += data[key][2]
    for key in average_result:
        average_result[key][0] /= len(json_file_index)
        average_result[key][1] /= len(json_file_index)
        average_result[key][2] /= len(json_file_index)
    out_dir = os.path.join(directory, "average.json")
    json_data = json.dumps(average_result, ensure_ascii=False, indent=2)
    with open(out_dir, 'w', encoding='utf-8') as output_file:
        output_file.write(json_data)
    print(f"Average result has been saved in {out_dir}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compute average results from JSON files in a directory.")
    parser.add_argument("--directory", type=str, help="The directory containing the JSON files.")
    args = parser.parse_args()

    # compute_f1_average(args.directory)
    compute_rough_average(args.directory)

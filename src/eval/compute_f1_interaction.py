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
        "interface":[0,0,0],
        "requirements reference":[0,0,0],
        "requirements constraints":[0,0,0],
    }
    predict_path
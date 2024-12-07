import json
import os
import numpy as np
from tqdm import tqdm
import time
import re
import csv
import pandas as pd
import argparse
from utils import utils
from yacs.config import CfgNode
from langchain.schema import HumanMessage

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config_file", type=str, default="config/convert_character.yaml", help="Path to config file"
    )
    parser.add_argument(
        "-l", "--log_file", type=str, default="conver_character.log", help="Path to log file"
    )
    parser.add_argument(
        "-n", "--log_name", type=str, default=str(os.getpid()), help="Name of logger"
    )
    args = parser.parse_args()
    return args





def get_num_tokens(text: str) -> int:
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    return len(tokenizer.encode(text))


if __name__ == '__main__':

    
    args = parse_args()
    
    # create config
    config = CfgNode(new_allowed=True)
    config = utils.add_variable_to_config(config, "log_file", args.log_file)
    config = utils.add_variable_to_config(config, "log_name", args.log_name)
    config.merge_from_file(args.config_file)

    narrator=config['narrator_llm']
    
    character_llms= [
       "gpt-3.5",
       "gpt-4"
    ]  
    character_llms=[x.split("/")[-1] for x in character_llms]
    zh_titles=[ '西游记','三国演义','红楼梦', '还珠格格', '笑傲江湖']
    en_titles=['Harry_Potter','The_Lord_of_the_Rings',  'The_Matrix', 'Twilight','A_Song_of_Ice_and_Fire' ]
    datas=[]
    utils.ensure_dir("output/data/character/")
    SAVE_PATH=f"output/data/character/character_data.json"
    if os.path.exists(SAVE_PATH):
        with open(SAVE_PATH, "r") as f:
            datas=json.load(f)

    zh_character_llms=character_llms.copy()
    en_character_llms=character_llms.copy()
    zh_character_llms.append('Qwen2.5-14B-Chat')
    en_character_llms.append('Meta-Llama-3-8B-Instruct')

    cnt=0
    tot=0
    print(f"Start Total: {len(datas)}")
    for title in tqdm(zh_titles):
        
        for character in zh_character_llms:      
            RECORD_PATH=f"output/record/{title}/character/{narrator}_{character}_character.jsonl"
            if not os.path.exists(RECORD_PATH):
                print(f"{RECORD_PATH} does not exist!")
                continue
           
            with open(RECORD_PATH, "r") as f:
                lines=f.readlines()
                for line in lines:
                    record=json.loads(line)
                    record=record['record']
                    for k,v in record.items():
                        for action in v:
                            template=action['detail']['prompt']['template']
                            try:
                                prompt=template.format(**action['detail'])
                                data={
                                    "language":"zh",
                                    "title":title,
                                    "narrator":narrator,
                                    "model":character,
                                    "type": action['type'],
                                    "instruction":prompt,
                                    "input":"",
                                    "output":action['detail']['text'],
                                    "history":[]
                                } 
                                datas.append(data)
                            except:
                                print(f"Error: {title}, {character}")
                                print(template)
                                print(action['detail'])
            print(f"Finish {title}: {character}, total: {len(datas)}")

    for title in tqdm(en_titles):
        for character in en_character_llms:      
            RECORD_PATH=f"output/record/{title}/character/{narrator}_{character}_character.jsonl"
            if not os.path.exists(RECORD_PATH):
                continue
           
            with open(RECORD_PATH, "r") as f:
                lines=f.readlines()
                for line in lines:
                    record=json.loads(line)
                    record=record['record']
                    for k,v in record.items():
                        for action in v:
                            template=action['detail']['prompt']['template']
                            prompt=template.format(**action['detail'])
                            data={
                                "language":"en",
                                "title":title,
                                "narrator":narrator,
                                "model":character,
                                "type": action['type'],
                                "instruction":prompt,
                                "input":"",
                                "output":action['detail']['text'],
                                "history":[]
                            } 
                            datas.append(data)
            print(f"Finish {title}: {character}, total: {len(datas)}")
        print(f"Total: {len(datas)}")
        with open(SAVE_PATH,"w") as f:
            json.dump(datas,f,ensure_ascii=False,indent=4)
                
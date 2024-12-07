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
        "-c", "--config_file", type=str, default="config/convert_narrator.yaml", help="Path to config file"
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
        "gpt-4",
        "gpt-3.5",
        "Qwen/Qwen2.5-7B-Chat",
        "Qwen/Qwen2.5-14B-Chat",
        "baichuan-inc/Baichuan2-13B-Chat",
        "baichuan-inc/Baichuan2-7B-Chat",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "microsoft/Phi-3.5-mini-128k-instruct"
    ]  
    character_llms=[x.split("/")[-1] for x in character_llms]
    zh_titles=[ '西游记','三国演义','红楼梦', '还珠格格', '笑傲江湖']
    en_titles=['Harry_Potter','The_Lord_of_the_Rings',  'The_Matrix', 'Twilight','A_Song_of_Ice_and_Fire' ]
    datas=[]
    utils.ensure_dir("output/data/narrator/")
    SAVE_PATH=f"output/data/narrator/narrator_data.json"
    if os.path.exists(SAVE_PATH):
        with open(SAVE_PATH, "r") as f:
            datas=json.load(f)

    cnt=0
    tot=0
    print(f"Start Total: {len(datas)}")
    for title in tqdm(zh_titles):
        
        for character in character_llms:     
            RECORD_PATH=f"output/record/{title}/narrator/{narrator}_{character}_narrator.jsonl"
            if not os.path.exists(RECORD_PATH):
                print(f"{RECORD_PATH} does not exist!")
                continue
           
            with open(RECORD_PATH, "r") as f:
                lines=f.readlines()
                for line in lines:
                    d=json.loads(line)
                    records=d['record']
                    for action in records:
                        template=action['detail']['prompt']['template']
                        try:
                            prompt=template.format(**action['detail'])
                            
                            if action['type']=='Update Scene' :
                                if 'event' in action['detail']['text']:
                                    pass
                                try:
                                    scene=json.loads(action['detail']['text'])
                                 
                                    output=f"- Time:{scene['time']}\n- Location:{scene['location']}\n- Description:{scene['description']}"
                                except:
                                    output=action['detail']['text']
                            else:
                                output=action['detail']['text']
                            data={
                                "language":"zh",
                                "title":title,
                                "narrator":narrator,
                                "model":character,
                                "type": action['type'],
                                "instruction":prompt,
                                "input":"",
                                "output":output,
                                "history":[]
                            } 
                                
                            datas.append(data)
                        except:
                            #print(1)
                            print(f"Error: {title}, {character}")
                            print(template)
                            print(action['detail'])
            print(f"Finish {title}: {character}, total: {len(datas)}")

    for title in tqdm(en_titles):
        for character in character_llms:      
            RECORD_PATH=f"output/record/{title}/narrator/{narrator}_{character}_narrator.jsonl"
            if not os.path.exists(RECORD_PATH):
                continue
           
            with open(RECORD_PATH, "r") as f:
                lines=f.readlines()
                for line in lines:
                    d=json.loads(line)
                    records=d['record']
                    for action in records:
                        
                        template=action['detail']['prompt']['template']
                        prompt=template.format(**action['detail'])
                        output=action['detail']['text']
                        data={
                            "language":"zh",
                            "title":title,
                            "narrator":narrator,
                            "model":character,
                            "type": action['type'],
                            "instruction":prompt,
                            "input":"",
                            "output":output,
                            "history":[]
                        } 
                        datas.append(data)
            print(f"Finish {title}: {character}, total: {len(datas)}")
        print(f"Total: {len(datas)}")
        with open(SAVE_PATH,"w") as f:
            json.dump(datas,f,ensure_ascii=False,indent=4)
                
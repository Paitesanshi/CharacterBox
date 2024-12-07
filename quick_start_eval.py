import os
import time
import json
import math
import copy
import logging
from datetime import datetime
from collections import OrderedDict


import argparse
import threading
from tqdm import tqdm
import concurrent.futures
from yacs.config import CfgNode

from utils import utils



from fastapi import FastAPI, Request
import json
import os
from evaluate import run_eval_detail
import threading
import csv
import pandas as pd
app = FastAPI()


lock = threading.Lock()




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
         "--character_llm", type=str,default=None, help="Path to config file"
    )
    parser.add_argument(
         "--scene_path", type=str,default=None, help="Path to config file"
    )
    parser.add_argument(
        "-c", "--config_file", type=str, default="config/evaluate.yaml", help="Path to config file"
    )
    parser.add_argument(
        "-o", "--output_file", type=str,default="message.json", help="Path to output file"
    )
    parser.add_argument(
        "-l", "--log_file", type=str, default="", help="Path to log file"
    )
    parser.add_argument(
        "-n", "--log_name", type=str, default=str(os.getpid()), help="Name of logger"
    )
    parser.add_argument(
        "-p",
        "--play_role",
        type=int,
        default=-1,
        help="Add a user controllable role",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    args = parser.parse_args()
    return args

def run_eval_with_config(config, logger,scene,SAVE_PATH,done):
    
    scene_id=run_eval_detail(config, logger,scene,SAVE_PATH,done)
    return scene_id

def run_thread(thread, title,model,scene_id):
    try:
        thread.join()
    except Exception as e:
        print(f"Error in {title}-{model}: {scene_id}!!!!!!!!!!!!!!!!!!!!") 




if __name__ == "__main__":
    args = parse_args()
    config = CfgNode(new_allowed=True)
    output_file = os.path.join("output/message", args.output_file)
    config = utils.add_variable_to_config(config, "output_file", output_file)
    config = utils.add_variable_to_config(config, "log_file", args.log_file)
    config = utils.add_variable_to_config(config, "log_name", args.log_name)
    config = utils.add_variable_to_config(config, "play_role", args.play_role)
    config.merge_from_file(args.config_file)
    global_logger = utils.set_logger("quick_start.log", "quick_start")
    # titles=[ '西游记','三国演义','红楼梦', '还珠格格', '笑傲江湖', 'Harry_Potter','The_Lord_of_the_Rings',  'The_Matrix', 'Twilight','A_Song_of_Ice_and_Fire' ]
    titles=['Harry_Potter']
    # character_llms= [
        
    #     "gpt-4",
    #     "gpt-3.5",
    #      "Qwen/Qwen2.5-7B-Chat",
    #      "Qwen/Qwen2.5-14B-Chat",
    #      "baichuan-inc/Baichuan2-13B-Chat",
    #      "baichuan-inc/Baichuan2-7B-Chat",
    #      "meta-llama/Meta-Llama-3-8B-Instruct",
    #      "mistralai/Mistral-7B-Instruct-v0.2",
    #      "microsoft/Phi-3.5-mini-128k-instruct"
    # ]  

    character_llms=['gpt-4o-mini']
    max_scenes=config['max_scenes'] if 'max_scenes' in config else 5
        
    
    scene_ids=[i for i in range(0,5)]    
    scene_ids=[0]
    all_st=time.time()
    for model_path in tqdm(character_llms):
        print("Current Model:",model_path)
        llm_time=time.time()
        for title in tqdm(titles):
            print("Current Title:",title)
            
            character=model_path.split("/")[-1]
            config['title']=title
            
            threads = []
            judger=config['judger_llm']
            narrator=config['narrator_llm']
            config['character_llm']=character
            utils.ensure_dir("output/evaluation/detail/"+title)
            RECORD_PATH=f"output/record/{title}/character/{narrator}_{character}_character.jsonl"
            SAVE_PATH=f"output/evaluation/detail/{title}/{judger}_{narrator}_character_evaluation_detail.csv"
            eval_done={}
            for character_llm in character_llms:
                character=character_llm.split("/")[-1]
                eval_done[character]={}
                for scene in scene_ids:
                    eval_done[character][scene]=[]

            if not os.path.exists(SAVE_PATH):
                with open(SAVE_PATH, 'a', newline='', encoding='utf-8') as csvfile:
                    fieldnames = ['Title','Judger','Narrator','Model','SceneID', 'Round','SceneInfo','CharacterInfo','Critic','Actions', 'Knowledge Accuracy', 'Emotional Expression', 'Personality Traits', 'Behavioral Accuracy', 'Immersion', 'Adaptability', 'Behavioral Coherence']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    csvfile.seek(0, 2)  # Move to the end of the file
                    if csvfile.tell() == 0:  # If file is empty, write headers
                        writer.writeheader()
            else:
                with open(SAVE_PATH,'r') as f:
                    df=pd.read_csv(f)
                    for index, row in df.iterrows():
                        if row['Model'] not in eval_done:
                            eval_done[row['Model']]={}
                        if row['SceneID'] not in eval_done[row['Model']]:
                            eval_done[row['Model']][row['SceneID']]=[]
                        eval_done[row['Model']][row['SceneID']].append(row['CharacterInfo'])
            character_record = []
            utils.ensure_dir("output/log/evaluation/detail/"+title)
            log_file=f"evaluation/detail/{title}/{judger}_{narrator}_{character}_evaluation_detail.log"
            scenes={}
            with open(RECORD_PATH, 'r', encoding='utf-8') as f:
                lines=f.readlines()
                for id in range(len(lines)-1,-1,-1):
                    record=json.loads(lines[id])
                    if record['scene_id'] not in scenes:
                        scenes[record['scene_id']]=record
            

            st=max(0,len(scenes)-max_scenes)
            title_time=time.time()
            futures = []
            with concurrent.futures.ThreadPoolExecutor() as executor_eval:
                for id in scene_ids:
                    scene=scenes[id]
                    config['scene_id']=scene['scene_id']
                    log_file=f"evaluation/detail/{title}/{config['narrator_llm']}_{config['character_llm']}_{config['scene_id']}_evaluation.log"
                    logger = utils.set_logger(log_file, args.log_name)
                    logger.info(f"os.getpid()={os.getpid()}")
                    logger.info(f"\n{config}")
                    
                    future = executor_eval.submit(run_eval_with_config, config.copy(), logger, scene['record'], SAVE_PATH,eval_done[config['character_llm']][config['scene_id']])
                    futures.append(future)

                
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result() 
                        print(f"SCENE DONE: {result}")
                    except Exception as exc:
                        print(f'Generated an exception: {exc}')
            title_ed=time.time()
            print(f"Title {title} done in {title_ed-title_time} seconds, average {(title_ed-title_time)/len(scenes)} seconds per scene")
            logger.info(f"The evaluation result has been saved to {SAVE_PATH}")
        llm_ed=time.time()
        print(f"Model {model_path} done in {llm_ed-llm_time} seconds, average {(llm_ed-llm_time)/len(titles)} seconds per title")
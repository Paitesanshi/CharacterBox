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
        "-c", "--config_file", type=str, default="config/convert_reward.yaml", help="Path to config file"
    )
    parser.add_argument(
        "-l", "--log_file", type=str, default="", help="Path to log file"
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
    judger=config['judger_llm']
    character_llms= [
        "gpt-4",
        "gpt-3.5",
        "Qwen/Qwen2.5-7B-Chat",
        "Qwen/Qwen2.5-14B-Chat",
        "baichuan-inc/Baichuan2-13B-Chat",
        "baichuan-inc/Baichuan2-7B-Chat",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "microsoft/Phi-3.5-mini-instruct"
    ]  
    character_llms=[x.split("/")[-1] for x in character_llms]
    zh_titles=[ '西游记','三国演义','红楼梦', '还珠格格', '笑傲江湖']
    en_titles=['Harry_Potter','The_Lord_of_the_Rings',  'The_Matrix', 'Twilight','A_Song_of_Ice_and_Fire' ]

    datas=[]
    SAVE_PATH=f"output/data/reward/evaluation.json"
    if os.path.exists(SAVE_PATH):
        with open(SAVE_PATH, "r") as f:
            datas=json.load(f)  
    
    metrics=["Knowledge Accuracy","Emotional Expression","Personality Traits","Behavioral Accuracy","Immersion","Adaptability","Behavioral Coherence"]
    metric_descriptions = {
    "Knowledge Accuracy": "Knowledge Accuracy evaluates the model's proficiency in providing accurate and relevant information consistent with the character's background. It assesses the depth of understanding demonstrated by the model regarding the character's historical or professional context.",
    "Emotional Expression": "Emotional Expression assesses the model's ability to convey emotions effectively in dialogue, reflecting the character's emotional states. It captures the richness and authenticity of emotional portrayal, enhancing the immersion and engagement of users.",
    "Personality Traits": "Personality Traits measure the consistency of the model's behavior with the character's established personality traits. It evaluates the model's ability to maintain the character's unique identity and distinct characteristics throughout the interaction.",
    "Behavioral Accuracy": "Behavioral Accuracy evaluates the fidelity of the model's portrayal of the character's specific behaviors and linguistic habits. It assesses the precision and comprehensiveness of the model's replication of the character's mannerisms, enhancing the authenticity of the interaction.",
    "Immersion": "Immersion assesses the consistency of the character's portrayal, which impacts the user's ability to engage with and understand the character. It evaluates the degree to which the model's portrayal enhances user immersion and comprehension.",
    "Adaptability": "Adaptability measures the model's ability to respond to new situations and dialogue developments while maintaining character consistency. It assesses the model's flexibility in handling unexpected scenarios, contributing to dynamic and realistic interaction.",
    "Behavioral Coherence": "Behavioral Coherence evaluates the logical consistency of the character's behavior and responses throughout the interaction. It assesses the alignment of the character's actions and responses with the dialogue and plot progression, enhancing the coherence and believability of the interaction."
}
    critic="""
<Note>:
Please identify any issues based on these aspects:
1. Factual Accuracy: Identify and point out any elements that do not accurately match the historical or factual backdrop.
2. Character Consistency: Explicitly highlight inconsistencies between the character's actions, dialogues, and their predefined traits and goals.
3. Logical Coherence: Point out any logical fallacies or actions that contradict the established context or character logic.
4. Content Redundancy: Identify repetitions in dialogue or action that could detract from engagement and realism.
5. Emotional Expression: Assess whether emotional responses and expressions are appropriate and convincingly portrayed, highlighting any discrepancies.
6. Interaction Adaptability: Critique the character's interactions with others, noting any unnatural or contextually inappropriate responses.
7. Creativity and Originality: Evaluate the creativity of responses and actions, pointing out generic or unoriginal content.
8. Detail Handling: Scrutinize the level of detail in scene setting and character enactment, marking areas lacking depth or accuracy.
9. Style Consistency: Ensure that the narrative and linguistic style remains consistent, identifying any deviations.
10. Fluency and Quality: Critically assess the smoothness and quality of the text, highlighting any grammatical errors or awkward phrasings.

"""

    for title in tqdm(zh_titles):
        
        RECORD_PATH=f"output/evaluation/detail/{title}/{judger}_{narrator}_character_evaluation_detail.csv"
        utils.ensure_dir("output/data/reward/")
        max_retries = config['max_retries']
        max_rounds=config['max_rounds']
        character_record = []
        if args.log_file=="":
            utils.ensure_dir("output/log/evaluation/data/reward/"+title)
            args.log_file=f"evaluation/data/reward/{title}/{narrator}_generation.log"
        logger = utils.set_logger(args.log_file, args.log_name)
        logger.info(f"os.getpid()={os.getpid()}")
        logger.info(f"\n{config}")

        eval_result=pd.read_csv(RECORD_PATH,encoding='gbk')
        print(f"{title}:{len(eval_result)}")
        err_cnt=0
        be_cnt=0
        for index, row in eval_result.iterrows():

            scene=row['SceneInfo']
            character_static_info=row['CharacterInfo']
            actions_text=row['Actions']
            if not isinstance(actions_text,str):
                print(actions_text)
                continue
            observations = re.findall(r"Observation:(.*?)(?=Observation:|Action:|$)", actions_text, flags=re.S)
            actions = re.findall(r"Action:(.*?)(?=Observation:|Action:|$)", actions_text, flags=re.S)
            behaviors=""
            num=min(len(observations),len(actions))
            for j in range(num):
                round_action=f"Observation:\n{observations[j]}\nAction:\n{actions[j]}\n"
                behaviors+=round_action
            prompt=f"Please evaluate the role-playing ability of the character based on actions across multiple turns based on scene, character information and evaluation criteria.\n<Scene>:\n{scene}\n<Character>:\n{character_static_info}\n<Actions>:{actions}\n"
            prompt+=critic
            criteria=f"<Criteria>:\n"
            for id,metric in enumerate(metrics):
                criteria+=f"{id+1}. {metric}: {metric_descriptions[metric]}\n"
            
            criteria+=f"<Score>:\n"
            output=""
            for metric in metrics:
                output+=f"{metric}: {row[metric]}\n"
            
            data={
                "language":"zh",
                "title":title,
                "judger":judger,
                "narrator":narrator,
                "model":row['Model'],
                "criteria":metric,
                "instruction":prompt+criteria,
                "input":"",
                "output":output,
                "history":[]
            }     
                
            datas.append(data)
    print("Total Data:",len(datas))
    for title in tqdm(en_titles):
        
        RECORD_PATH=f"output/evaluation/detail/{title}/{judger}_{narrator}_character_evaluation_detail.csv"
        utils.ensure_dir("output/data/reward/")
        max_retries = config['max_retries']
        max_rounds=config['max_rounds']
        character_record = []
        if args.log_file=="":
            utils.ensure_dir("output/log/evaluation/data/reward/"+title)
            args.log_file=f"evaluation/data/reward/{title}/{narrator}_generation.log"
        logger = utils.set_logger(args.log_file, args.log_name)
        logger.info(f"os.getpid()={os.getpid()}")
        logger.info(f"\n{config}")

        eval_result=pd.read_csv(RECORD_PATH,encoding='gbk')
        print(f"{title}:{len(eval_result)}")
        for index, row in eval_result.iterrows():
            
            scene=row['SceneInfo']
            character_static_info=row['CharacterInfo']
            actions_text=row['Actions']
            observations = re.findall(r"Observation:(.*?)(?:Observation:|Action:|$)", actions_text, flags=re.S)
            actions = re.findall(r"Action:(.*?)(?:Observation:|Action:|$)", actions_text, flags=re.S)
            behaviors=""
            num=min(len(observations),len(actions))
            for j in range(num):
                actions[j]=actions[j].strip('\n')
                round_action=f"Observation:\n{observations[j]}\nAction:\n{actions[j]}\n"
                behaviors+=round_action
            prompt=f"Please evaluate the role-playing ability of the character based on actions across multiple turns based on scene, character information and evaluation criteria.\n<Scene>:\n{scene}\n<Character>:\n{character_static_info}\n<Actions>:{behaviors}\n"
            prompt+=critic
            
            criteria=f"<Criteria>:\n"
            for id,metric in enumerate(metrics):
                criteria+=f"{id+1}. {metric}: {metric_descriptions[metric]}\n"
            
            criteria+=f"<Score>:\n"
            output=""
            for metric in metrics:
                output+=f"{metric}: {row[metric]}\n"
            
            data={
                "language":"zh",
                "title":title,
                "judger":judger,
                "narrator":narrator,
                "model":row['Model'],
                "criteria":metric,
                "instruction":prompt+criteria,
                "input":"",
                "output":output,
                "history":[]
            }     
            datas.append(data)

    print("Total Data:",len(datas))
    with open(SAVE_PATH,"w") as f:
        json.dump(datas,f,ensure_ascii=False,indent=4)
            
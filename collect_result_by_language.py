import pandas as pd
import csv
from utils import utils
import os

zh_titles=[ '西游记','三国演义','红楼梦', '还珠格格', '笑傲江湖']
en_titles=['Harry_Potter','The_Lord_of_the_Rings',  'The_Matrix', 'Twilight','A_Song_of_Ice_and_Fire' ]
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
narrator="gpt-3.5"
judger="gpt-4"





fieldnames = ['Judger','Narrator','Model', 'Knowledge Accuracy', 'Emotional Expression', 'Personality Traits', 'Behavioral Accuracy', 'Immersion', 'Adaptability', 'Behavioral Coherence','Average']

utils.ensure_dir("output/evaluation/avg/")

ZH_SAVE_PATH=f"output/evaluation/avg/{judger}_{narrator}_character_evaluation_zh.csv"    

if not os.path.exists(ZH_SAVE_PATH):
    with open(ZH_SAVE_PATH, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
zh_avg=pd.DataFrame(columns = fieldnames)
detail=pd.DataFrame(columns = [
    'Title', 'Judger', 'Narrator', 'Model', 'SceneID', 'Round', 'SceneInfo',
    'CharacterInfo', 'Critic', 'Actions', 'Knowledge Accuracy',
    'Emotional Expression', 'Personality Traits', 'Behavioral Accuracy',
    'Immersion', 'Adaptability', 'Behavioral Coherence'
])
for title in zh_titles:
    

    RESULT_PATH=f"output/evaluation/detail/{title}/{judger}_{narrator}_character_evaluation_detail.csv"
    if not os.path.exists(RESULT_PATH):
        print("Not exist:",RESULT_PATH)
        narrator='gpt-3.5'
        RESULT_PATH=f"output/evaluation/detail/{title}/{judger}_{narrator}_character_evaluation_detail.csv"
        if not os.path.exists(RESULT_PATH):
            print("Not exist both:",RESULT_PATH)
            continue
    df2=pd.read_csv(RESULT_PATH)
    detail=pd.concat([detail, df2], ignore_index=True)
grouped_detail = detail.groupby([ 'Judger', 'Narrator', 'Model']).agg({
'Knowledge Accuracy': 'mean',
'Emotional Expression': 'mean',
'Personality Traits': 'mean',
'Behavioral Accuracy': 'mean',
'Immersion': 'mean',
'Adaptability': 'mean',
'Behavioral Coherence': 'mean'
}).reset_index()

grouped_detail['Average'] = grouped_detail[['Knowledge Accuracy',
                                                'Emotional Expression',
                                                'Personality Traits',
                                                'Behavioral Accuracy',
                                                'Immersion',
                                                'Adaptability',
                                                'Behavioral Coherence']].mean(axis=1)

print(grouped_detail)

grouped_detail = grouped_detail.sort_values(by='Average', ascending=False)
zh_avg=pd.concat([zh_avg, grouped_detail], ignore_index=True)

zh_avg.to_csv(ZH_SAVE_PATH, index=False)  



EN_SAVE_PATH=f"output/evaluation/avg/{judger}_{narrator}_character_evaluation_en.csv"    

if not os.path.exists(EN_SAVE_PATH):
    with open(EN_SAVE_PATH, 'w', newline='') as csvfile:
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
en_avg=pd.DataFrame(columns = fieldnames)
detail=pd.DataFrame(columns = [
    'Title', 'Judger', 'Narrator', 'Model', 'SceneID', 'Round', 'SceneInfo',
    'CharacterInfo', 'Critic', 'Actions', 'Knowledge Accuracy',
    'Emotional Expression', 'Personality Traits', 'Behavioral Accuracy',
    'Immersion', 'Adaptability', 'Behavioral Coherence'
])
for title in en_titles:
    

    RESULT_PATH=f"output/evaluation/detail/{title}/{judger}_{narrator}_character_evaluation_detail.csv"
    if not os.path.exists(RESULT_PATH):
        print("Not exist:",RESULT_PATH)
        continue
    df2=pd.read_csv(RESULT_PATH)
    detail=pd.concat([detail, df2], ignore_index=True)
grouped_detail = detail.groupby([ 'Judger', 'Narrator', 'Model']).agg({
'Knowledge Accuracy': 'mean',
'Emotional Expression': 'mean',
'Personality Traits': 'mean',
'Behavioral Accuracy': 'mean',
'Immersion': 'mean',
'Adaptability': 'mean',
'Behavioral Coherence': 'mean'
}).reset_index()

grouped_detail['Average'] = grouped_detail[['Knowledge Accuracy',
                                                'Emotional Expression',
                                                'Personality Traits',
                                                'Behavioral Accuracy',
                                                'Immersion',
                                                'Adaptability',
                                                'Behavioral Coherence']].mean(axis=1)

print(grouped_detail)

grouped_detail = grouped_detail.sort_values(by='Average', ascending=False)
en_avg=pd.concat([en_avg, grouped_detail], ignore_index=True)
en_avg.to_csv(EN_SAVE_PATH, index=False)  
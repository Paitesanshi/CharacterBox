
import os
import re
import csv
import time
import json
from tqdm import tqdm
from openai import OpenAI
from transformers import GPT2TokenizerFast
import argparse
from utils import utils
from yacs.config import CfgNode
from langchain.schema import HumanMessage


def extract_scores(response):
    regex=r"Quantity Appropriateness: \[?(\d+)\]?.*?Quality Accuracy: \[?(\d+)\]?.*?Relevance to Context: \[?(\d+)\]?.*?Clarity and Order: \[?(\d+)\]?"
    # Search for matches
    match = re.search(regex, response, re.DOTALL)

    # Extract the scores if found
    if match:
        quantity_appropriateness,quality_accuracy,relevance_to_context,clarity_and_order  = match.groups()
    else:
        quantity_appropriateness,quality_accuracy,relevance_to_context,clarity_and_order=-1,-1,-1,-1

    return quantity_appropriateness,quality_accuracy,relevance_to_context,clarity_and_order




def get_num_tokens(text: str) -> int:
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    return len(tokenizer.encode(text))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config_file", type=str, default="config/evaluate_narrator.yaml", help="Path to config file"
    )
    parser.add_argument(
        "-l", "--log_file", type=str, default="evaluate_narrator.log", help="Path to log file"
    )
    parser.add_argument(
        "-n", "--log_name", type=str, default=str(os.getpid()), help="Name of logger"
    )
    args = parser.parse_args()
    return args



if __name__ == '__main__':

    args = parse_args()
    
    # create config
    config = CfgNode(new_allowed=True)
    config = utils.add_variable_to_config(config, "log_file", args.log_file)
    config = utils.add_variable_to_config(config, "log_name", args.log_name)
    config.merge_from_file(args.config_file)

    judger=config['judger_llm']
    narrator=config['narrator_llm']
    narrator_llms=['gpt-3.5']
    character='gpt-3.5'
    titles=['Harry_Potter']
    types=['Action Influence','Analyze Result','Update Character','Update Scene']
    for title in titles:
        TITLE_SAVE_PATH=f"output/evaluation/narrator/{title}/narrator_evaluation.csv"
        print(f"current title: {title}")
        print(f"TITLE_SAVE_PATH: {TITLE_SAVE_PATH}")
        for narrator in narrator_llms:
            
            print(f"current narrator: {narrator}")
            title_avg_quality=0.
            title_avg_quantity=0.
            title_avg_relevance=0.
            title_avg_manner=0.
            
            title_quantity={k:0. for k in types}
            title_quality={k:0. for k in types}
            title_relevance={k:0. for k in types}
            title_manner={k:0. for k in types}
            title_cnt_types={k:0 for k in types}


            for scene_id in range(5,10):
                utils.ensure_dir("output/evaluation/narrator/"+title)
                RECORD_PATH=f"output/record/{title}/narrator/{narrator}_{character}_narrator_unseen.jsonl"
                SAVE_PATH=f"output/evaluation/narrator/{title}/{character}_{scene_id}_narrator_evaluation.csv"
                if args.log_file=="":
                    utils.ensure_dir("output/log/evaluation/narrator/"+title)
                    args.log_file=f"evaluation/narrator/{title}/{judger}_{narrator}_{character}_{scene_id}_character_evaluation.log"
                
                logger = utils.set_logger(args.log_file, args.log_name)
                logger.info(f"os.getpid()={os.getpid()}")
                logger.info(f"\n{config}")
                max_retries = config['max_retries']
                narrator_record = []

                with open(RECORD_PATH, 'r', encoding='utf-8') as f:
                    lines=f.readlines()
                    record = json.loads(lines[0])
                    narrator_record = record['record']
                print(type(narrator_record))
                
                
                with open(SAVE_PATH, 'a', newline='', encoding='utf-8') as csvfile:
                    fieldnames = ['Title','Judger','Narrator','Model', 'Action Type', 'Quality', 'Quantity','Relevance','Manner']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    csvfile.seek(0, 2)  # Move to the end of the file
                    if csvfile.tell() == 0:  # If file is empty, write headers
                        writer.writeheader()
                    avg_rows=[]
                    LLM=utils.get_llm(judger,config,logger,config['api_key'],config['api_base'])
                
                    avg_quantity={k:0. for k in types}
                    avg_quality={k:0. for k in types}
                    avg_relevance={k:0. for k in types}
                    avg_manner={k:0. for k in types}
                    cnt_types={k:0 for k in types}
                    valid_action=0
                    tot = 0
                    cnt = 0
                    for id in tqdm(range(len(narrator_record))):

                        actions = narrator_record[id]
                        detail=narrator_record[id]['detail']
                        inputs={}
                        for key in detail['prompt']['input_variables']:
                            inputs[key]=detail[key]
                        prompt=detail['prompt']['template'].format(**inputs)
                        logger.info(f"Prompt: {prompt}")
                        
                        output=detail['text']
                        
                        criteria="""
[Evaluation Criteria]:
1. Quantity Appropriateness: Evaluate whether the output provides an appropriate amount of information, balancing between being informative and concise. This includes avoiding overwhelming the audience with too much detail and not omitting crucial information for understanding the scenario. Rate from 1 to 5, where 1 indicates an inappropriate level of information provided and 5 indicates an optimal balance.

2. Quality Accuracy: Assess the accuracy and reliability of the information within the output, ensuring that the statements are factual or plausible within the context of the role-playing scenario. Consider whether the output includes speculations or errors that could mislead or confuse. Rate from 1 to 5, where 1 indicates frequent inaccuracies or implausibilities, and 5 represents high accuracy and reliability.

3. Relevance to Context: Measure how the output’s content directly relates to the ongoing narrative and role-playing scenario. The content should contribute effectively to the advancement of the storyline or development of the scene. The output should align with the specific details and structure requested in the prompt. Rate from 1 to 5, where 1 indicates content that is off-topic or irrelevant, and 5 indicates content that is highly pertinent and engaging.

4. Clarity and Order: Judge the output for its clarity, precision, and the logical organization of information. The narrative should be easy to follow and understand, avoiding ambiguity and ensuring a smooth flow of ideas. Rate from 1 to 5, where 1 indicates poor clarity and disorganization, and 5 indicates exceptional clarity and logical structure.

[Evaluation Steps]:
1. Prompt Review: Thoroughly read and understand the prompt provided to the LLM, focusing on the specific instructions and expected output format.
2. Output Review: Examine the LLM's generated output, noting how well it adheres to the prompt and the rationality of its content.
3. Criteria-Based Assessment: Systematically evaluate the output using the criteria of Instruction Adherence and Content Rationality.

[Response Format]:
Quantity Appropriateness: [1-5]
Quality Accuracy: [1-5]
Relevance to Context: [1-5]
Clarity and Order: [1-5]

Please provide detailed justifications for your ratings to ensure a comprehensive assessment.\n
            """
                        prompt = f"You have been assigned to assess the output of a language model (LLM) based on its adherence to the provided prompt and the rationality of the content produced. \n [Prompt]: {prompt}\n [Output]:{output}\n Please evaluate the output using the following criteria:{criteria}"
                        cnt += 1
                        print(f"tot: \n{tot}\n")
                        print(f"cnt: \n{cnt}\n")
                        print(f"average: \n{tot/cnt}\n")
                        if 'Chat' in type(LLM).__name__:
                            response = LLM([HumanMessage(content=prompt)]).content
                        else:
                            response = LLM(prompt)
                        
                        logger.info(f"Response: \n{response}\n")
                        quantity_appropriateness, quality_accuracy, relevance_to_context, clarity_and_order = extract_scores(response)
                        logger.info(f"Quantity Appropriateness: {quantity_appropriateness}")
                        logger.info(f"Quality Accuracy: {quality_accuracy}")
                        logger.info(f"Relevance to Context: {relevance_to_context}")
                        logger.info(f"Clarity and Order: {clarity_and_order}")


                        if quantity_appropriateness == -1:
                            continue
                        action_type = actions['type']
                        cnt_types[action_type] += 1
                        title_cnt_types[action_type]+=1
                        avg_quantity[action_type]+=float(quantity_appropriateness)
                        avg_quality[action_type]+=float(quality_accuracy)
                        avg_relevance[action_type]+=float(relevance_to_context)
                        avg_manner[action_type]+=float(clarity_and_order)


                        title_quantity[action_type]+=float(quantity_appropriateness)
                        title_quality[action_type]+=float(quality_accuracy)
                        title_relevance[action_type]+=float(relevance_to_context)
                        title_manner[action_type]+=float(clarity_and_order)


                        
                    scene_avg_quality=sum(avg_quality.values())/sum(cnt_types.values())
                    scene_avg_quantity=sum(avg_quantity.values())/sum(cnt_types.values())
                    scene_avg_relevance=sum(avg_relevance.values())/sum(cnt_types.values())
                    scene_avg_manner=sum(avg_manner.values())/sum(cnt_types.values())
                    for key in avg_quality:
                        if cnt_types[key]==0:
                            continue
                        writer.writerow({
                            'Title':title,
                            'Judger': judger,
                            'Narrator': narrator,
                            'Model': character,
                            'Action Type': key,
                            'Quality': avg_quality[key]/cnt_types[key],
                            'Quantity': avg_quantity[key]/cnt_types[key],
                            'Relevance': avg_relevance[key]/cnt_types[key],
                            'Manner':avg_manner[key]/cnt_types[key]
                        })
                        print('Model:', character)
                        print("Action Type:",key)
                        print("Quality:",avg_quality[key]/cnt_types[key])
                        print("Quantity:",avg_quantity[key]/cnt_types[key])
                        print("Relevance:",avg_relevance[key]/cnt_types[key])
                        print("Manner:",avg_manner[key]/cnt_types[key])


                    writer.writerow({
                            'Title':title,
                            'Judger': judger,
                            'Narrator': narrator,
                            'Model': character,
                            'Action Type': "Average",
                            'Quality': scene_avg_quality,
                            'Quantity': scene_avg_quantity,
                            'Relevance': scene_avg_relevance,
                            'Manner':scene_avg_manner
                    })
            with open(TITLE_SAVE_PATH, 'a', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['Title','Judger','Narrator','Model', 'Action Type', 'Quality', 'Quantity','Relevance','Manner']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                csvfile.seek(0, 2)  # Move to the end of the file
                if csvfile.tell() == 0:  # If file is empty, write headers
                    writer.writeheader()
                title_avg_quality=sum(title_quality.values())/sum(title_cnt_types.values())
                title_avg_quantity=sum(title_quantity.values())/sum(title_cnt_types.values())
                title_avg_relevance=sum(title_relevance.values())/sum(title_cnt_types.values())
                title_avg_manner=sum(title_manner.values())/sum(title_cnt_types.values())
                print("Title avg Quality:",title_avg_quality)
                print("Title avg Quantity:",title_avg_quantity)
                print("Title avg Relevance:",title_avg_relevance)
                print("Title avg Manner:",title_avg_manner)
                for key in avg_quality:
                    if title_cnt_types[key]==0:
                        continue
                    writer.writerow({
                        'Title':title,
                        'Judger': judger,
                        'Narrator': narrator,
                        'Model': character,
                        'Action Type': key,
                        'Quality': title_quality[key]/title_cnt_types[key],
                        'Quantity': title_quantity[key]/title_cnt_types[key],
                        'Relevance': title_relevance[key]/title_cnt_types[key],
                        'Manner':title_manner[key]/title_cnt_types[key]
                    })

                writer.writerow({
                        'Title':title,
                        'Judger': judger,
                        'Narrator': narrator,
                        'Model': character,
                        'Action Type': "Average",
                        'Quality': title_avg_quality,
                        'Quantity': title_avg_quantity,
                        'Relevance': title_avg_relevance,
                        'Manner':title_avg_manner
                })
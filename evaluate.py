import json
import os
import numpy as np
from tqdm import tqdm
from openai import OpenAI
import time
import re
import csv
import argparse
from utils import utils
from yacs.config import CfgNode
from langchain.schema import HumanMessage
from filelock import FileLock
import threading


lock = threading.Lock()

def extract_scores(response):
    #Pattern to match the scores and reasons
    regex = r"Knowledge Accuracy:\s*\[?\s*(\d+)\s*\]?.*?Emotional Expression:\s*\[?\s*(\d+)\s*\]?.*?Personality Traits:\s*\[?\s*(\d+)\s*\]?.*?Behavioral Accuracy:\s*\[?\s*(\d+)\s*\]?.*?Immersion:\s*\[?\s*(\d+)\s*\]?.*?Adaptability:\s*\[?\s*(\d+)\s*\]?.*?Behavioral Coherence:\s*\[?\s*(\d+)\s*\]?"

    # Search for matches
    match = re.search(regex, response, re.DOTALL)

    # Extract the scores if found
    if match:
        accuracy, expression,traits,behavior,immersion,adaptability,coherence = match.groups()
    else:
        accuracy, expression,traits,behavior,immersion,adaptability,coherence = -1, -1,-1,-1,-1,-1,-1
    

    return int(accuracy),int(expression), int(traits),int(behavior),int(immersion),int(adaptability),int(coherence)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config_file", type=str, default="config/evaluate.yaml", help="Path to config file"
    )
    parser.add_argument(
        "-l", "--log_file", type=str, default="", help="Path to log file"
    )
    parser.add_argument(
        "-n", "--log_name", type=str, default=str(os.getpid()), help="Name of logger"
    )
    args = parser.parse_args()
    return args


def critic(LLM,scene,character,actions):
    prompt = f"""
Please execute the following role-play and identify any issues based on these strict evaluation criteria:

- Scene Description: \n{scene}\n
- Character Description:\n {character}\n
- Character Actions: \n{actions}\n

Strict Evaluation Criteria:
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
Condense the issues into one paragraph.
    """
    if 'Chat' in type(LLM).__name__:
        response = LLM([HumanMessage(content=prompt)]).content
    else:
        response = LLM(prompt)  
    return response


def get_num_tokens(text: str) -> int:
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    return len(tokenizer.encode(text))


def run_eval_detail(config, logger,character_record,SAVE_PATH,done):
    
    judger=config['judger_llm']
    narrator=config['narrator_llm']
    character=config['character_llm']
    scene_id=config['scene_id']
    max_rounds=config['max_rounds']
    title=config['title']
    api_key=config['api_key']
    api_base=config['api_base']
    os.environ['OPENAI_API_KEY']=api_key
    os.environ['OPENAI_API_BASE']=api_base
    LLM=utils.get_llm(judger,config,logger,config['api_key'],config['api_base'])
    ids=list(character_record.keys())
    print(f"Scene ID: {scene_id}\n Characters:{character_record.keys()}")
    lock_path = SAVE_PATH + '.lock'
    lock = FileLock(lock_path)
    with open(SAVE_PATH, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Title','Judger','Narrator','Model','SceneID', 'Round','SceneInfo','CharacterInfo','Critic','Actions', 'Knowledge Accuracy', 'Emotional Expression', 'Personality Traits', 'Behavioral Accuracy', 'Immersion', 'Adaptability', 'Behavioral Coherence']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        metrics=['Knowledge Accuracy', 'Emotional Expression', 'Personality Traits', 'Behavioral Accuracy', 'Immersion', 'Adaptability', 'Behavioral Coherence']
        avg_scores={k:0.0 for k in metrics}
        valid_count=0
        for id in tqdm(ids,desc=f"Scene {scene_id} Characters"):
            actions = character_record[id]
            name=actions[0]['character_name']
            print("Processing character: ", actions[0]['character_name'])
            event=actions[0]["detail"]['event']
            scene_time=actions[0]["detail"]['time']
            location=actions[0]["detail"]['location']
            description=actions[0]["detail"]['description']
            scene = (
            f"Scenario Information:\n"
            f"Event: {event}\n"
            f"Time: {scene_time}\n"
            f"Location: {location}\n"
            f"Description: {description}\n"
            )

            character_static_info=(
                f"Name: {name}\n"
                f"Description: {actions[0]['detail']['character_description']}\n")
            if character_static_info in done:
                print(f"{character}: Skipping character {name} as it is already evaluated.")
                continue
            prompt = f"Please evaluate the role-playing ability of the character based on actions across multiple turns based on scene, character information, critique and evaluation criteria.\n[Scene]:\n{scene}\n[Character]:\n{character_static_info}\n[Multi-turn Actions]:\n"

            last_round=0
            behaviors=""
            for action in actions:
                round=action['round']
                if round>max_rounds:
                    break
                if round!=last_round:
                    last_round=round
                    prompt+="Round: "+str(round)+"\n"
                trail_token=0
                character_info=(
                    f"Observation: {action['detail']['observation']}\n"
                )

                
                
                behavior="Action:\n"
                if action['type']=='dialogue':
                    behavior+=f"{name}: {action['detail']['text']}\n"
                else:
                    behavior+=f"{action['detail']['text']}\n"
                prompt+=character_info+f"{behavior}\n"
                behaviors+=character_info+f"{behavior}\n"   

                

            problem=critic(LLM,scene,character_static_info,behaviors)
            prompt+=f"[Critique]:\n{problem}\n"
            criteria="""
[Scoring Criteria]:
1. Knowledge Accuracy:
   - 1 Point: Information is often incorrect or irrelevant, significantly inconsistent with the character's background.
   - 3 Points: Information is generally accurate, though there are occasional errors or some details are not very relevant to the character's background.
   - 5 Points: Information is always accurate and highly relevant to the character's historical or professional background, demonstrating deep knowledge and skills.
2. Emotional Expression:
   - 1 Point: Emotional expression is monotonous or inappropriate, not aligning with the dialogue content and context.
   - 3 Points: Emotional expression is moderately varied, usually matching the content, but lacks depth and subtlety.
   - 5 Points: Emotional expression is rich and profound, highly consistent with the dialogue content and context, showing complex and nuanced emotions.
3. Personality Traits:
   - 1 Point: The displayed personality traits often conflict with or lack consistency with the character's setup.
   - 3 Points: Personality traits generally match the character's setup, though there are occasional inconsistencies.
   - 5 Points: Consistently displays behavior and language choices that match the core personality traits of the character, showcasing the character's uniqueness.
4. Behavioral Accuracy:
   - 1 Point: The model fails to capture or reproduce the character's unique behaviors and linguistic habits.
   - 3 Points: The model reflects the character's behaviors and linguistic habits to some extent, but not precisely or completely.
   - 5 Points: The model accurately mimics and reproduces the character's specific behaviors, linguistic habits, and catchphrases.
5. Immersion:
   - 1 Point: Character portrayal is often inconsistent, making it difficult for users to immerse or understand the character.
   - 3 Points: Character is mostly consistent, but occasional contradictions slightly affect immersion.
   - 5 Points: Character portrayal is always consistent, enhancing user immersion and effectively showing self-awareness and character limitations.
6. Adaptability:
   - 1 Point: Character portrayal lacks adaptability in the development of the dialogue, unable to handle new situations reasonably.
   - 3 Points: Character adapts to changes in the dialogue in most situations, though occasionally it may not be very flexible.
   - 5 Points: Character flexibly handles any new situations in the dialogue, always maintaining character consistency and adjusting to new directions.
7. Behavioral Coherence:
   - 1 Point: Character's behavior and responses are often logically disordered, not matching the dialogue or plot development.
   - 3 Points: Character's behavior and responses are generally logically coherent, though occasionally there may be unreasonable aspects.
   - 5 Points: Character's behavior and responses are always logically consistent, reasonably adjusting based on the progression of the dialogue and plot development.

[Evaluation Steps]:
1. Contextual Understanding: Examine the character's profile and background information thoroughly to fully grasp the nuances of their context, motivations, and historical background.
2. Behavioral Observation: Monitor how the character reacts across different scenarios, paying special attention to their decisions, dialogues, and interactions.
3. Critique Analysis: Evaluate the character based on issues and weaknesses identified in external critiques, ensuring a focused assessment that considers these specific points.
4. Criteria-Based Assessment: Analyze each observed behavior using the criteria to systematically evaluate the consistency and effectiveness of the character's portrayal. Your resposne must follow the format provided below.

[Response Format]:
Knowledge Accuracy: [1-5]
Emotional Expression: [1-5]
Personality Traits: [1-5]
Behavioral Accuracy: [1-5]
Immersion: [1-5]
Adaptability: [1-5]
Behavioral Coherence: [1-5]

[Response Format Example]:
Knowledge Accuracy: 3
Emotional Expression: 3
Personality Traits: 3
Behavioral Accuracy: 3
Immersion: 3
Adaptability: 3
Behavioral Coherence: 3

[Response]:\n
"""#immersion or fidelity?
            prompt+=criteria
            cnt=0
            accuracy=-1
            while accuracy==-1:
                cnt+=1
                if cnt==5:
                    break
                if 'Chat' in type(LLM).__name__:
                    response = LLM([HumanMessage(content=prompt)]).content
                else:
                    response = LLM(prompt)

                logger.info(f"Prompt: \n{prompt}\n")
                logger.info(f"Response: \n{response}\n")
                print(f"Attempt: {cnt}")
                accuracy, expression,traits,behavior,immersion,adaptability,coherence = extract_scores(response)

            print(f"Scores: Knowledge Accuracy: {accuracy}\nEmotional Expression: {expression}\nPersonality Traits: {traits}\nBehavioral Accuracy: {behavior}\nImmersion: {immersion}\nAdaptability: {adaptability}\nBehavioral Coherence: {coherence}\n")
            if accuracy==-1:
                continue
            data_row = {
                'Title': title,
                'Judger': judger,
                'Narrator': narrator,
                'Model': character,
                'SceneID': scene_id,
                'Round':last_round,
                'SceneInfo': scene,
                'CharacterInfo': character_static_info,
                'Critic': problem,
                'Actions': behaviors,
                'Knowledge Accuracy': accuracy,
                'Emotional Expression': expression,
                'Personality Traits': traits,
                'Behavioral Accuracy': behavior,
                'Immersion': immersion,
                'Adaptability': adaptability,
                'Behavioral Coherence': coherence,
            }
            
            writer.writerow(data_row)
    return scene_id


if __name__ == '__main__':

    
    args = parse_args()
    
    # create config
    config = CfgNode(new_allowed=True)
    config = utils.add_variable_to_config(config, "log_file", args.log_file)
    config = utils.add_variable_to_config(config, "log_name", args.log_name)
    config.merge_from_file(args.config_file)

    judger=config['judger_llm']
    narrator=config['narrator_llm']
    character=config['character_llm']
    title=config['title']
    scene_id=config['scene_id']
    utils.ensure_dir("output/evaluation/multi/"+title)
    RECORD_PATH=f"output/record/{title}/{narrator}_{character}_{scene_id}_character.json"
    SAVE_PATH=f"output/evaluation/multi/{title}/{narrator}_{scene_id}_character_evaluation_avg.csv"
    max_retries = config['max_retries']
    max_rounds=config['max_rounds']
    character_record = []
    if args.log_file=="":
        utils.ensure_dir("output/log/evaluation/multi/"+title)
        args.log_file=f"evaluation/multi/{title}/{narrator}_{character}_{scene_id}_character_evaluation_avg.log"
    logger = utils.set_logger(args.log_file, args.log_name)
    logger.info(f"os.getpid()={os.getpid()}")
    logger.info(f"\n{config}")

    with open(RECORD_PATH, 'r', encoding='utf-8') as f:
        character_record = json.load(f)
    
    LLM=utils.get_llm(judger,config,logger,config['api_key'],config['api_base'])
    ids=list(character_record.keys())
    with open(SAVE_PATH, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Title','Judger','Narrator','Model','SceneID', 'Round', 'Knowledge Accuracy', 'Emotional Expression', 'Personality Traits', 'Behavioral Accuracy', 'Immersion', 'Adaptability', 'Behavioral Coherence','Average']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csvfile.seek(0, 2)  # Move to the end of the file
        if csvfile.tell() == 0:  # If file is empty, write headers
            writer.writeheader()
        metrics=['Knowledge Accuracy', 'Emotional Expression', 'Personality Traits', 'Behavioral Accuracy', 'Immersion', 'Adaptability', 'Behavioral Coherence']
        avg_scores={k:0.0 for k in metrics}
        valid_count=0
        for id in tqdm(ids):
            actions = character_record[id]
            name=actions[0]['character_name']
            print("Processing character: ", actions[0]['character_name'])
            event=actions[0]["detail"]['event']
            scene_time=actions[0]["detail"]['time']
            location=actions[0]["detail"]['location']
            description=actions[0]["detail"]['description']
            scene = (
            f"Scenario Information:\n"
            f"Event: {event}\n"
            f"Time: {scene_time}\n"
            f"Location: {location}\n"
            f"Description: {description}\n"
        )
            avg_rationality=.0
            avg_relevance=.0
            character_static_info=(
                f"Name: {name}\n"
                f"Description: {actions[0]['detail']['character_description']}\n")
            prompt = f"please evaluate the role-playing ability of the character based on actions across multiple turns based on scene, character information, critique and evaluation criteria.\n{scene}\n{character_static_info}\nMulti-turn Actions as follows:\n"
            last_round=0
            behaviors=""
            for action in actions:
                round=action['round']
                if round>max_rounds:
                    break
                if round!=last_round:
                    last_round=round
                    prompt+="Round: "+str(round)+"\n"
                character_info=(
                    f"Observation: {action['detail']['observation']}\n"
                )
                behavior="Action:\n"
                if action['type']=='dialogue':
                    behavior+=f"{name}: {action['detail']['text']}\n"
                else:
                    behavior+=f"{action['detail']['text']}\n"
                prompt+=character_info+f"{behavior}\n"
                behaviors+=character_info+f"{behavior}\n"   

            problem=critic(LLM,scene,character_static_info,behaviors)
            prompt+=f"[Critique]:\n{problem}\n"
            criteria="""
[Scoring Criteria]:
1. Knowledge Accuracy:
   - 1 Point: Information is often incorrect or irrelevant, significantly inconsistent with the character's background.
   - 3 Points: Information is generally accurate, though there are occasional errors or some details are not very relevant to the character's background.
   - 5 Points: Information is always accurate and highly relevant to the character's historical or professional background, demonstrating deep knowledge and skills.
2. Emotional Expression:
   - 1 Point: Emotional expression is monotonous or inappropriate, not aligning with the dialogue content and context.
   - 3 Points: Emotional expression is moderately varied, usually matching the content, but lacks depth and subtlety.
   - 5 Points: Emotional expression is rich and profound, highly consistent with the dialogue content and context, showing complex and nuanced emotions.
3. Personality Traits:
   - 1 Point: The displayed personality traits often conflict with or lack consistency with the character's setup.
   - 3 Points: Personality traits generally match the character's setup, though there are occasional inconsistencies.
   - 5 Points: Consistently displays behavior and language choices that match the core personality traits of the character, showcasing the character's uniqueness.
4. Behavioral Accuracy:
   - 1 Point: The model fails to capture or reproduce the character's unique behaviors and linguistic habits.
   - 3 Points: The model reflects the character's behaviors and linguistic habits to some extent, but not precisely or completely.
   - 5 Points: The model accurately mimics and reproduces the character's specific behaviors, linguistic habits, and catchphrases.
5. Immersion:
   - 1 Point: Character portrayal is often inconsistent, making it difficult for users to immerse or understand the character.
   - 3 Points: Character is mostly consistent, but occasional contradictions slightly affect immersion.
   - 5 Points: Character portrayal is always consistent, enhancing user immersion and effectively showing self-awareness and character limitations.
6. Adaptability:
   - 1 Point: Character portrayal lacks adaptability in the development of the dialogue, unable to handle new situations reasonably.
   - 3 Points: Character adapts to changes in the dialogue in most situations, though occasionally it may not be very flexible.
   - 5 Points: Character flexibly handles any new situations in the dialogue, always maintaining character consistency and adjusting to new directions.
7. Behavioral Coherence:
   - 1 Point: Character's behavior and responses are often logically disordered, not matching the dialogue or plot development.
   - 3 Points: Character's behavior and responses are generally logically coherent, though occasionally there may be unreasonable aspects.
   - 5 Points: Character's behavior and responses are always logically consistent, reasonably adjusting based on the progression of the dialogue and plot development.

[Evaluation Steps]:
1. Contextual Understanding: Examine the character's profile and background information thoroughly to fully grasp the nuances of their context, motivations, and historical background.
2. Behavioral Observation: Monitor how the character reacts across different scenarios, paying special attention to their decisions, dialogues, and interactions.
3. Critique Analysis: Evaluate the character based on issues and weaknesses identified in external critiques, ensuring a focused assessment that considers these specific points.
4. Criteria-Based Assessment: Analyze each observed behavior using the criteria to systematically evaluate the consistency and effectiveness of the character's portrayal. Your resposne must follow the format provided below.

[Response Format]:
Knowledge Accuracy: [1-5]
Emotional Expression: [1-5]
Personality Traits: [1-5]
Behavioral Accuracy: [1-5]
Immersion: [1-5]
Adaptability: [1-5]
Behavioral Coherence: [1-5]\n"""#immersion or fidelity?
            prompt+=criteria
            
            if 'Chat' in type(LLM).__name__:
                response = LLM([HumanMessage(content=prompt)]).content
            else:
                response = LLM(prompt)
            accuracy, expression,traits,behavior,immersion,adaptability,coherence = extract_scores(response)
            print(f"Scores: Knowledge Accuracy: {accuracy}\nEmotional Expression: {expression}\nPersonality Traits: {traits}\nBehavioral Accuracy: {behavior}\nImmersion: {immersion}\nAdaptability: {adaptability}\nBehavioral Coherence: {coherence}\n")
            if accuracy==-1:
                continue
            avg_scores['Knowledge Accuracy']+=accuracy
            avg_scores['Emotional Expression']+=expression
            avg_scores['Personality Traits']+=traits
            avg_scores['Behavioral Accuracy']+=behavior
            avg_scores['Immersion']+=immersion
            avg_scores['Adaptability']+=adaptability
            avg_scores['Behavioral Coherence']+=coherence
            valid_count+=1
        avg_accuracy=avg_scores['Knowledge Accuracy']/valid_count
        avg_expression=avg_scores['Emotional Expression']/valid_count
        avg_traits=avg_scores['Personality Traits']/valid_count
        avg_behavior=avg_scores['Behavioral Accuracy']/valid_count
        avg_immersion=avg_scores['Immersion']/valid_count
        avg_adaptability=avg_scores['Adaptability']/valid_count
        avg_coherence=avg_scores['Behavioral Coherence']/valid_count
        avg=(avg_accuracy+avg_expression+avg_traits+avg_behavior+avg_immersion+avg_adaptability+avg_coherence)/7
        writer.writerow({
        'Title':title,
        'Judger': judger,
        'Narrator': narrator,
        'Model': character,
        'SceneID': scene_id,
        'Round':last_round,
        'Knowledge Accuracy': avg_accuracy,
        'Emotional Expression': avg_expression,
        'Personality Traits': avg_traits,
        'Behavioral Accuracy': avg_behavior,
        'Immersion': avg_immersion,
        'Adaptability': avg_adaptability,
        'Behavioral Coherence': avg_coherence,
        'Average': avg
    })
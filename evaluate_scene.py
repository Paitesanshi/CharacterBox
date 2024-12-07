
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


    regex=r"Creativity: \[?(\d+)\]?.*?Coherence: \[?(\d+)\]?.*?Conformity: \[?(\d+)\]?.*?Detail Handling: \[?(\d+)\]?.*?Language Style: \[?(\d+)\]?"
    # Search for matches
    match = re.search(regex, response, re.DOTALL)

    # Extract the scores if found
    if match:
        creativity, coherence, conformity, detail_handling, language_style = match.groups()
    else:
        creativity, coherence, conformity, detail_handling, language_style = -1, -1, -1, -1, -1

    return creativity, coherence, conformity, detail_handling, language_style




def get_num_tokens(text: str) -> int:
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    return len(tokenizer.encode(text))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config_file", type=str, default="config/scene_evaluate.yaml", help="Path to config file"
    )
    parser.add_argument(
        "-l", "--log_file", type=str, default="", help="Path to log file"
    )
    parser.add_argument(
        "-n", "--log_name", type=str, default=str(os.getpid()), help="Name of logger"
    )
    args = parser.parse_args()
    return args



def get_critic(title,scene):
    prompt = f"""
Given Scene based on {title}: 
{scene}

Please provide a detailed critique of the scene based on the following criteria:

1. **Creativity**: Analyze the level of creativity exhibited in the scene. Discuss whether the scene introduces innovative ideas or takes unexpected turns. Comment on the originality of the scenario and its elements.

2. **Coherence**: Critically assess the logical coherence of the scene. Determine if the events, character behaviors, and dialogues are consistent and believable within the context of the scene. Highlight any discrepancies or particularly smooth transitions.

3. **Conformity**: Evaluate how well the scene adheres to the established world and character rules of the original narrative, if applicable. Point out any deviations from the expected character developments or the established world rules.

4. **Detail Handling**: Examine the attention to detail in the scene's description. Consider whether the environmental details, character descriptions, and actions are vivid and contribute to the overall scene's impact.

5. **Language Style**: Review the language style used in the scene. Assess its suitability for the scene’s setting and characters, including the dialogue realism and narrative voice consistency.

Condense the critique into one paragraph.
"""
    
    if 'Chat' in type(LLM).__name__:
        response = LLM([HumanMessage(content=prompt)]).content
    else:
        response = LLM(prompt)
    return response


def evaluate_scene(title,scene):

    criteria="""
[Evaluation Criteria]:
1. Creativity: Evaluate the novelty and originality of the generated scene. Consider how the scene introduces unique perspectives or plot developments while maintaining the style and themes of the original work. Rate from 1 to 5, where 1 indicates minimal creativity and 5 indicates highly creative and original content.

2. Coherence: Assess the logical coherence of the scene both at the sentence level and in its overall structure. Determine if the scene seamlessly integrates into the existing storyline, or serves as a plausible extension. Rate from 1 to 5, where 1 indicates poor coherence and 5 indicates excellent coherence.

3. Conformity: Examine whether the generated scene adheres to the world-building, character consistency, and narrative progression of the original work. Consider the fidelity to the background setting and the consistency in characters’ dialogues and actions. Rate from 1 to 5, where 1 indicates poor conformity and 5 indicates high fidelity to the original work.

4. Detail Handling: Evaluate the precision and richness of the scene’s descriptive details. Determine if these details enhance the authenticity and depth of the scene. Rate from 1 to 5, where 1 indicates poor detail handling and 5 indicates highly detailed and enriching descriptions.

5. Language Style: Review the language style of the scene. Check if it matches the original work’s style in terms of vocabulary selection, sentence construction, and overall expression. Rate from 1 to 5, where 1 indicates a poor match in style and 5 indicates a perfect match in style.

[Evaluation Steps]:
1. Prompt Review: Thoroughly read and understand the prompt provided to the LLM, focusing on the specific expectations regarding creativity, coherence, conformity, detail handling, and language style.
2. Scene Review: Examine the LLM's generated scene, noting how well it adheres to the style and requirements of the original work.
3. Criteria-Based Assessment: Systematically evaluate the scene using the criteria of Creativity, Coherence, Conformity, Detail Handling, and Language Style.

[Response Format]:
Creativity: [1-5]
Coherence: [1-5]
Conformity: [1-5]
Detail Handling: [1-5]
Language Style: [1-5]\n"""
    critique=get_critic(title,scene)
    prompt = f"You have been assigned to assess the generated new scene based on {title}. \n [Generated Scene]:\n {scene}\n [Critique]:\n{critique}\nPlease use the following criteria to evaluate the generated scene based on the provided critique.:{criteria}"

    
    if 'Chat' in type(LLM).__name__:
        response = LLM([HumanMessage(content=prompt)]).content
    else:
        response = LLM(prompt)
    
    logger.info(f"Response: \n{response}\n")

    creativity, coherence, conformity, detail_handling, language_style = extract_scores(response)
    return creativity, coherence, conformity, detail_handling, language_style


def get_all_files_in_folder(folder_path):
    file_names = []
    # 遍历指定文件夹下的所有文件和子文件夹
    for root, dirs, files in os.walk(folder_path):
        # 遍历当前文件夹中的所有文件
        for file in files:
            file_names.append(os.path.join(root, file))
    return file_names




if __name__ == '__main__':

    args = parse_args()
    titles=['Titanic', 'Avatar', 'The_Great_Gatsby', 'The_Matrix', 'Harry_Potter', '三体', '西游记', '还珠格格', '三国演义', 'The_Lord_of_the_Rings']
    # create config
    config = CfgNode(new_allowed=True)
    config = utils.add_variable_to_config(config, "log_file", args.log_file)
    config = utils.add_variable_to_config(config, "log_name", args.log_name)
    config.merge_from_file(args.config_file)

    judger=config['judger_llm']
    generator=config['generator_llm']
    utils.ensure_dir("output/evaluation/scene/")
    SCENE_DIR=config['scene_dir']
    SAVE_PATH=f"output/evaluation/scene/{judger}_scene_evaluation.csv"
    if args.log_file=="":
        utils.ensure_dir("output/log/evaluation/scene/")
        args.log_file=f"evaluation/scene/{judger}_{generator}_scene_evaluation.log"
    
    logger = utils.set_logger(args.log_file, args.log_name)
    logger.info(f"os.getpid()={os.getpid()}")
    logger.info(f"\n{config}")
    max_retries = config['max_retries']
    narrator_record = []

    result={"Judger":judger}
    with open(SAVE_PATH, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Judger','Generator', 'Creativity', 'Coherence', 'Conformity', 'Detail Handling', 'Language Style', 'Average']
        files=get_all_files_in_folder(SCENE_DIR)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csvfile.seek(0, 2)  # Move to the end of the file
        if csvfile.tell() == 0:  # If file is empty, write headers
            writer.writeheader()
        
        metrics = ['Creativity', 'Coherence', 'Conformity', 'Detail_Handling', 'Language_Style']
        avg_scores={k:0. for k in metrics}
        valid_scene=0
        LLM=utils.get_llm(judger,config,logger,config['api_key'],config['api_base'])
        
        for file in tqdm(files):
            title=file.split('/')[-1].split('.')[0]
            with open(file, 'r', encoding='utf-8') as f:
                scene_record = json.load(f)

            for scene in scene_record:

                creativity, coherence, conformity, detail_handling, language_style = evaluate_scene(title,scene)
                logger.info(f"Creativity: {creativity}\nCoherence: {coherence}\nConformity: {conformity}\nDetail Handling: {detail_handling}\nLanguage Style: {language_style}\n")
                print(f"Creativity: {creativity}\nCoherence: {coherence}\nConformity: {conformity}\nDetail Handling: {detail_handling}\nLanguage Style: {language_style}\n")
                if creativity==-1:
                    continue
                valid_scene+=1
                avg_scores['Creativity']+=int(creativity)
                avg_scores['Coherence']+=int(coherence)
                avg_scores['Conformity']+=int(conformity)
                avg_scores['Detail_Handling']+=int(detail_handling)
                avg_scores['Language_Style']+=int(language_style)
        
            
        writer.writerow({
                'Judger': judger,
                'Generator': generator,
                'Creativity': avg_scores['Creativity']/valid_scene,
                'Coherence': avg_scores['Coherence']/valid_scene,
                'Conformity': avg_scores['Conformity']/valid_scene,
                'Detail Handling': avg_scores['Detail_Handling']/valid_scene,
                'Language Style': avg_scores['Language_Style']/valid_scene,
                'Average': (avg_scores['Creativity']+avg_scores['Coherence']+avg_scores['Conformity']+avg_scores['Detail_Handling']+avg_scores['Language_Style'])/5/valid_scene
            
            })
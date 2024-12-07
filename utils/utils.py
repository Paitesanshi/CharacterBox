import json
import logging
from typing import Any, Dict, Optional, List, Tuple
import re
import random
from llm import *
from yacs.config import CfgNode
import os
from langchain_openai import ChatOpenAI

# logger
def set_logger(log_file, name="default"):
    """
    Set logger.
    Args:
        log_file (str): log file path
        name (str): logger name
    """

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    output_folder = "output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create the 'log' folder if it doesn't exist
    log_folder = os.path.join(output_folder, "log")
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    # Create the 'message' folder if it doesn't exist
    message_folder = os.path.join(output_folder, "message")
    if not os.path.exists(message_folder):
        os.makedirs(message_folder)
    log_file = os.path.join(log_folder, log_file)
    handler = logging.FileHandler(log_file, mode="w")
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(handler)
    return logger


# json
def load_json(json_file: str, encoding: str = "utf-8") -> Dict:
    with open(json_file, "r", encoding=encoding) as fi:
        data = json.load(fi)
    return data


def save_json(
    json_file: str,
    obj: Any,
    encoding: str = "utf-8",
    ensure_ascii: bool = False,
    indent: Optional[int] = None,
    **kwargs,
) -> None:
    with open(json_file, "w", encoding=encoding) as fo:
        json.dump(obj, fo, ensure_ascii=ensure_ascii, indent=indent, **kwargs)


def bytes_to_json(data: bytes) -> Dict:
    return json.loads(data)


def dict_to_json(data: Dict) -> str:
    return json.dumps(data)


# cfg
def load_cfg(cfg_file: str, new_allowed: bool = True) -> CfgNode:
    """
    Load config from file.
    Args:
        cfg_file (str): config file path
        new_allowed (bool): whether to allow new keys in config
    """
    with open(cfg_file, "r") as fi:
        cfg = CfgNode.load_cfg(fi)
    cfg.set_new_allowed(new_allowed)
    return cfg


def add_variable_to_config(cfg: CfgNode, name: str, value: Any) -> CfgNode:
    """
    Add variable to config.
    Args:
        cfg (CfgNode): config
        name (str): variable name
        value (Any): variable value
    """
    cfg.defrost()
    cfg[name] = value
    cfg.freeze()
    return cfg


def merge_cfg_from_list(cfg: CfgNode, cfg_list: list) -> CfgNode:
    """
    Merge config from list.
    Args:
        cfg (CfgNode): config
        cfg_list (list): a list of config, it should be a list like
        `["key1", "value1", "key2", "value2"]`
    """
    cfg.defrost()
    cfg.merge_from_list(cfg_list)
    cfg.freeze()
    return cfg




def ensure_dir(dir_path):
    """
    Make sure the directory exists, if it does not exist, create it
    Args:
        dir_path (str): The directory path.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def generate_id(dir_name):
    ensure_dir(dir_name)
    existed_id = set()
    for f in os.listdir(dir_name):
        existed_id.add(f.split("-")[0])
    id = random.randint(1, 999999999)
    while id in existed_id:
        id = random.randint(1, 999999999)
    return id


def get_llm(model,config, logger, api_key=None, api_base=None):
    """
    Get the large language model.
    Args:
        config (CfgNode): The config.
        logger (Logger): The logger.
        api_key (str): The API key.
    """
    if  api_base is not None and "http" not in api_base and api_base != "xxx":
        LLM=LocalLLM(model_path=api_base,max_token=config['max_token'],temperature=config['temperature'],logger=logger)
    else :
        LLM = ChatOpenAI(
            max_tokens=config["max_token"],
            temperature=config["temperature"],
            api_key=api_key,
            base_url=api_base,
            model=model,
            max_retries=config["max_retries"],
        )
    return LLM


def count_files_in_directory(target_directory:str):
    """Count the number of files in the target directory"""
    return len(os.listdir(target_directory))

def get_avatar_url(id:int,gender:str,type:str="origin",role=False):
    if role:
        target='/asset/img/avatar/role/'+gender+'/'
        return target+str(id%10)+'.png'
    target='/asset/img/avatar/'+type+"/"+gender+'/'
    return target+str(id%10)+'.png'


def detect_language(text):
    
    if re.search("[\u4e00-\u9fff]", text):
        return "Chinese"
    elif re.search("[A-Za-z]", text):
        return "English"
    else:
        return "Unknown"


def get_token_num(text):
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2',max_length=8192)
    return len(tokenizer.encode(text))

def get_cost(llm,prompt,output):
    input_price=0
    output_price=0
    llm=llm.lower()
    if 'gpt-4' in llm:
        input_price=10
        output_price=30
    elif 'gpt-3.5' in llm or 'gpt-35' in llm:
        input_price=0.5
        output_price=1.5
    
    input_length=get_token_num(prompt)
    output_length=get_token_num(output)
    return input_price*input_length/1000000+output_price*output_length/1000000,input_length,output_length


def calc_cost(narrator_llm,character_llm,actions):

    total_cost=0
    total_duration=0
    for action in actions:
        if action['character'] == 'Narrator':
            cost,_,_=get_cost(narrator_llm,action['prompt'],action['response'])
        else:
            cost,_,_=get_cost(character_llm,action['prompt'],action['response'])
        total_cost+=cost
        total_duration+=action['duration']
    return total_cost,total_duration
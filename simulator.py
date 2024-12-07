import os
import time
import json
import math
import copy
import logging
from datetime import datetime
from collections import OrderedDict


import faiss
import argparse
import threading
from tqdm import tqdm
import concurrent.futures
from yacs.config import CfgNode
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain_experimental.generative_agents import GenerativeAgentMemory
from langchain_community.embeddings import HuggingFaceEmbeddings


from dataloader.data import Data
from utils import utils
from utils.message import Message
from utils.character import SceneInfo
from agents import Character, HumanPlayer, Narrator
from langchain.globals import set_debug


# set_debug(True)


logging.basicConfig(level=logging.ERROR)
lock = threading.Lock()


class Simulator:
    """
    Simulator class for running the simulation.
    """
    def __init__(self, config: CfgNode, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.round_cnt = 0
        self.file_name_path: list[str] = []
        self.play_event = threading.Event()
        self.now = datetime.now().replace(hour=8, minute=0, second=0)
        self.round_message:list[Message] = []
        self.round_obs = {}
        self.acting = OrderedDict()
        self.acted = OrderedDict()
        self.last_round = ""
        self.character_record = {}
        self.round_record = []
        self.plot_record = {}
        self.narrator_record=[]
        self.records = []
        self.duration=0.

    def load_simulator(self):
        """Load and initiate the simulator."""
        self.round_cnt = 0
        self.data = Data(self.config, self.logger)
        # self.round_obs = {i:[] for i in self.data.characters.keys()}
        self.narrator, self.characters = self.agents_creation()
        
        self.character_record = {i:[] for i in self.characters.keys()}
        utils.ensure_dir(self.config["record_dir"])
        title = self.config['scene_path'].split("/")[-1].split(".")[0]
        utils.ensure_dir(os.path.join(self.config["record_dir"], title))
        self.plot_record = self.data.scenes[self.config['scene_id']]
        self.plot_record['generated_plot'] = []
        self.plot_record['synopsis'] = []
        for i in self.characters.keys():
            self.acting[i] = True
        self.logger.info("Simulator loaded.")



    def relevance_score_fn(self, score: float) -> float:
        """Return a similarity score on a scale [0, 1]."""

        return 1.0 - score / math.sqrt(2)

    def create_new_memory_retriever(self):
        """Create a new vector store retriever unique to the agent."""
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {'device': 'cpu'}
        embeddings_model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
        embedding_size=768
        #embeddings_model=OpenAIEmbeddings()
        #embedding_size = 1536
        # Initialize the vectorstore as empty
        
        index = faiss.IndexFlatL2(embedding_size)
        vectorstore = FAISS(
            embeddings_model.embed_query,
            index,
            InMemoryDocstore({}),
            {},
            relevance_score_fn=self.relevance_score_fn,
        )


        return TimeWeightedVectorStoreRetriever(vectorstore=vectorstore, other_score_keys=["importance"], k=5)

    def pause(self):
        self.play_event.clear()

    def play(self):
        self.play_event.set()

    def get_round_info(self):
        history=""
        for i in range(len(self.round_record)):
            history += self.round_record[i]['character_name']+":"+self.round_record[i]['detail']['text']+'\n'
        return history
    
    def get_character_obs(self, agent_id):
        return ". ".join(self.round_obs[agent_id])
    
    def save_record(self):
        title = self.config['scene_path'].split("/")[-1].split(".")[0]

        # 记录结果
        if self.config['save_record']:
            utils.ensure_dir(os.path.join(self.config["record_dir"], title,"plot"))
            with open(os.path.join(self.config["record_dir"], title,"plot", self.config['narrator_llm']+"_"+self.config['character_llm']+"_"+self.config["plot_record_path"]), "a", encoding='utf8') as f:
                data={"title":title,"scene_id":self.config['scene_id'],"scene":self.data.init_scene,"characters":self.data.init_characters,"record":self.plot_record}
                json.dump(data, f, default=lambda o: o.__dict__,  ensure_ascii=False)
                f.write("\n")

            print("Plot record saved in:",os.path.join(self.config["record_dir"], title,"plot", self.config['narrator_llm']+"_"+self.config['character_llm']+"_"+self.config["plot_record_path"]))

            utils.ensure_dir(os.path.join(self.config["record_dir"], title,"character"))
            with open(os.path.join(self.config["record_dir"], title,"character", self.config['narrator_llm']+"_"+self.config['character_llm']+"_"+self.config["character_record_path"]), "a", encoding='utf8') as f:
                data={"title":title,"scene_id":self.config['scene_id'],"scene":self.data.init_scene,"characters":self.data.init_characters,"record":self.character_record}
                json.dump(data, f, default=lambda o: o.__dict__,  ensure_ascii=False)
                f.write("\n")

            print("Character record saved in:",os.path.join(self.config["record_dir"], title,"character", self.config['narrator_llm']+"_"+self.config['character_llm']+"_"+self.config["character_record_path"]))

            utils.ensure_dir(os.path.join(self.config["record_dir"], title,"round"))
            with open(os.path.join(self.config["record_dir"], title,"round", self.config['narrator_llm']+"_"+self.config['character_llm']+"_"+self.config["round_record_path"]), "a", encoding='utf8') as f:
                data={"title":title,"scene_id":self.config['scene_id'],"scene":self.data.init_scene,"characters":self.data.init_characters,"record":self.records}
                json.dump(data, f, default=lambda o: o.__dict__,  ensure_ascii=False)
                f.write("\n")
                
            print("Round record saved in:",os.path.join(self.config["record_dir"], title,"round", self.config['narrator_llm']+"_"+self.config['character_llm']+"_"+self.config["round_record_path"]))

            utils.ensure_dir(os.path.join(self.config["record_dir"], title,"narrator"))
            with open(os.path.join(self.config["record_dir"], title,"narrator", self.config['narrator_llm']+"_"+self.config['character_llm']+"_"+self.config["narrator_record_path"]), "a", encoding='utf8') as f:
                data={"title":title,"scene_id":self.config['scene_id'],"scene":self.data.init_scene,"characters":self.data.init_characters,"record":self.narrator_record}
                json.dump(data, f, default=lambda o: o.__dict__, ensure_ascii=False)
                f.write("\n")
            print("Narrator record saved in:",os.path.join(self.config["record_dir"], title,"narrator", self.config['narrator_llm']+"_"+self.config['character_llm']+"_"+self.config["narrator_record_path"]))


            utils.ensure_dir(os.path.join(self.config["record_dir"], title,"all"))
            with open(os.path.join(self.config["record_dir"], title,"all", self.config['narrator_llm']+"_"+self.config['character_llm']+"_"+self.config["all_record_path"]), "a", encoding='utf8') as f:
                all_record=self.narrator.all_actions
                for id,agent in self.characters.items():
                    all_record+=agent.all_actions
                total_cost,duration=utils.calc_cost(self.config['narrator_llm'],self.config['character_llm'],all_record)
                self.logger.info(f"Total Cost:{total_cost}, Duration:{duration}")
                data={"title":title,"scene_id":self.config['scene_id'],"scene":self.data.init_scene,"characters":self.data.init_characters,"total_cost":total_cost,"duration":self.duration,"record":all_record}
                json.dump(data, f, default=lambda o: o.__dict__,  ensure_ascii=False)
                f.write("\n")
    # Run one step of an agent.
    def one_step(self, agent_id):
        self.play_event.wait()
        agent = self.characters[agent_id]
        name = agent.name
        message = []

        action, detail = agent.take_action(self.get_character_obs(agent_id),self.synopsis[name], self.now)
        action_info = {"round":self.round_cnt, "character_id":agent_id, "character_name":name, "type":"action", "detail":detail}
        self.round_record.append(action_info)
        self.character_record[agent_id].append(action_info)
        action_message = Message(
                agent_id=0,
                action="POST",
                content=f"{name}: {action}."
            )
        message.append(action_message)
        self.round_message.append(action_message)
        # target, operation = self.narrator.analyze_action(action)
        self.logger.info(f"{name}: {action}")
        
        response, detail = agent.generate_dialogue( action,self.synopsis[name], self.now)
        self.logger.info(f"{name}: \"{response}\" ")
        response_message = Message(
                agent_id=agent_id,
                action="POST",
                content=f"{name}: \"{response}\""
            )
        message.append(response_message)
        self.round_message.append(response_message)
        self.round_obs[agent_id].clear() 
        action_info={"round":self.round_cnt, "character_id":agent_id, "character_name":name, "type":"dialogue", "detail":detail}   
        self.character_record[agent_id].append(action_info)
        self.round_record.append(action_info)

        # 分析行为对其他角色的影响并记录
        influence_id, inf_action,detail = self.narrator.analyze_action_influence(agent.name, action, self.now,verbose=True)
        action_info={"round":self.round_cnt, "character_id":-1, "character_name":"Narrator", "type":"Action Influence", "detail":detail}   
        self.narrator_record.append(action_info)
        print(f"influence_id:{influence_id}, inf_action:{inf_action}")
        
        reactions=[]
        if influence_id is not agent.id and influence_id is not None:
            
            inf_agent=self.characters[influence_id]
            reactions.append({"Actor:":name,"Action:":action})
            self.logger.info(f"{name} influenced {inf_agent.name}: {inf_action}")
           
            self.round_obs[influence_id].append(f"{name}:{action}")
            self.round_obs[influence_id].append(f"{name}: {response}")
            self.round_obs[influence_id].append(f"{inf_action}")
            reaction, detail = inf_agent.take_reaction(self.get_character_obs(influence_id),"", self.now)
            inf_agent.memory.add_memory(f"{name}: {action}", now=self.now)
            inf_agent.memory.add_memory(f"{name}: {inf_action}", now=self.now)
            inf_agent.memory.add_memory(f"{name}: {response}", now=self.now)
            inf_agent.memory.add_memory(f"{inf_agent.name}: {reaction}", now=self.now)
    
            if influence_id not in self.acted:
                self.acted[influence_id] = True
                self.acting.pop(influence_id)
            action_info = {"round":self.round_cnt, "character_id":influence_id, "character_name":inf_agent.name, "type":"reaction", "detail":detail}
            self.character_record[influence_id].append(action_info)
            self.round_record.append(action_info)
            self.logger.info(f"{inf_agent.name} react to {name}: {reaction}")
            
            # Narrator analyzes the result of the interaction
            reactions.append({"Actor:":inf_agent.name,"Action:":reaction})
            reactions_info=";".join([f"{r['Actor:']}:{r['Action:']}" for r in reactions])
            result,detail = self.narrator.analyze_result(reactions_info,self.now,verbose=True)
            action_info = {"round":self.round_cnt, "character_id":-1, "character_name":"Narrator", "type":"Analyze Result", "detail":detail}
            
            self.round_record.append(action_info)
            self.narrator_record.append(action_info)
            self.logger.info(f"Interaction Result:{result}")
            
           
            agent.memory.add_memory(result, now=self.now)
            inf_agent.memory.add_memory(result, now=self.now)

             # Update Characters
            inf_position, inf_states,detail = self.narrator.update_character(self.characters[influence_id].name,reactions_info+";"+result,self.now,verbose=True)
            self.round_obs[influence_id].clear()
            action_info={"round":self.round_cnt, "character_id":-1, "character_name":"Narrator", "type":"Update Character", "detail":detail}   
            self.narrator_record.append(action_info)
            self.logger.info(f"update_character: name:{self.characters[influence_id].name}\n position:{inf_position}\n states:{inf_states}\n")
            
           
            
            position, states,detail = self.narrator.update_character(self.characters[agent_id].name, reactions_info+";"+result, self.now,verbose=True)
            action_info={"round":self.round_cnt, "character_id":-1, "character_name":"Narrator", "type":"Update Character", "detail":detail}   
            self.narrator_record.append(action_info)
            self.logger.info(f"update_character: name:{self.characters[agent_id].name}\n position:{position}\n states:{states}\n")
            if position is not None and states is not None:
                for char_obj in [self.characters[influence_id], self.narrator.characters[influence_id], self.data.characters[influence_id], self.data.characters_list[influence_id]]:
                    char_obj.position = inf_position
                    char_obj.states = inf_states
                for char_obj in [self.characters[agent_id], self.narrator.characters[agent_id], self.data.characters[agent_id], self.data.characters_list[agent_id]]:
                    char_obj.position = position
                    char_obj.states = states
       
        else:
            position, states,detail = self.narrator.update_character(self.characters[agent_id].name,action ,self.now,verbose=True)
            action_info={"round":self.round_cnt, "character_id":-1, "character_name":"Narrator", "type":"Update Character", "detail":detail}   
            self.narrator_record.append(action_info)
            self.logger.info(f"update_character: name:{self.characters[agent_id].name}\n position:{position}\n states:{states}\n")
            for char_obj in [self.characters[agent_id], self.narrator.characters[agent_id], self.data.characters[agent_id], self.data.characters_list[agent_id]]:
                    char_obj.position = position
                    char_obj.states = states
       
        return message
    
    # Run many steps
    def round(self):
        """
        Run one step for all agents.
        """
        messages = []
        futures = []
        self.logger.info("Round {} start.".format(self.round_cnt))
        for i in self.characters.keys():
            #self.characters[i].scene = new_scene
            self_belief = self.characters[i].update_self_belief(self.last_round, self.now)
            other_character = self.get_other_agents(self.data.characters_list, i)
            env_belief = self.characters[i].update_env_belief(self.last_round, other_character, self.now)

        sequence=[self.characters[i].name for i in self.acting.keys()]
        #self.synopsis=self.narrator.generate_synopsis(self.last_round,sequence,self.now)
        self.synopsis=self.characters[0].generate_synopsis(self.last_round,sequence,self.plot_record['synopsis'],self.now)
        for i in self.characters.keys():
            if self.characters[i].name not in self.synopsis:
                self.synopsis[self.characters[i].name]=""
        self.plot_record["synopsis"].append({"round":self.round_cnt, "synopsis":self.synopsis})
        self.logger.info(f"Round Synopsis:{self.synopsis}")
        while len(self.acted)<len(self.characters):
            actor = self.acting.popitem(last=False)[0]           
            msgs = self.one_step(actor)
            messages.append(msgs)
            self.acted[actor] = True


        
        self.last_round = self.get_round_info()
        new_scene,detail = self.narrator.update_scene(self.last_round,verbose=True)
        action_info={"round":self.round_cnt, "character_id":-1, "character_name":"Narrator", "type":"Update Scene", "detail":detail}   
        self.narrator_record.append(action_info)
        self.logger.info(f"Scene:{new_scene}")
        round_plot = self.narrator.summary_plot(self.last_round)
        self.plot_record["generated_plot"].append({"round":self.round_cnt, "plot":round_plot})
        self.logger.info(f"Round Plot:{round_plot}")
        

        for i in self.characters.keys():
            self.characters[i].scene = new_scene

        self.records.append(copy.copy(self.round_record))
        self.round_message.clear()
        self.round_record.clear()
        self.acting, self.acted = self.acted, self.acting
        
        return messages

    def create_character(self, i, api_key, api_base="") -> Character:
        """
        Create an agent with the given id.
        """

        LLM = utils.get_llm(self.config['character_llm'], config=self.config, logger=self.logger, api_key=api_key,api_base=api_base)
        agent_memory = GenerativeAgentMemory(
            llm=LLM,
            memory_retriever = self.create_new_memory_retriever(),
            verbose = False,
            reflection_threshold = 10,
            max_tokens_limit=2500,
            importance_weight=0.1
        )

        agent = Character(
                id = self.data.characters[i].id,
                name = self.data.characters[i].name,
                scene = SceneInfo(**self.data.scenes[self.config['scene_id']]),
                position = self.data.characters[i].position,
                states = self.data.characters[i].states,
                traits = self.data.characters[i].description,
                llm = LLM,
                memory = agent_memory,
                status="",
        )
        
        return agent

    def create_human_player(self, id, api_key, api_base=None):

        LLM = utils.get_llm(self.config['character_llm'],config=self.config, logger=self.logger, api_key=api_key, api_base=api_base)
        
      
        MemoryClass = GenerativeAgentMemory
        agent_memory = MemoryClass(
            llm=LLM,
            memory_retriever = self.create_new_memory_retriever(),
            verbose = False,
            reflection_threshold = 15,
        )
        
        agent = HumanPlayer(
            id=id,
            name=self.data.characters[id].name,
            gender=self.data.characters[id].gender,
            scene=SceneInfo(**self.data.scenes[self.config['scene_id']]),
            position=self.data.characters[id].position,
            states=self.data.characters[id].states,
            traits=self.data.characters[id].description,
            llm=LLM,
            memory=agent_memory,
            status="",
        )

        return agent

    def create_narrator(self, config, logger, api_key, api_base=None):

        LLM = utils.get_llm(self.config['narrator_llm'],config=self.config, logger=self.logger, api_key=api_key, api_base=api_base)
        
      
        MemoryClass = GenerativeAgentMemory
        agent_memory = MemoryClass(
            llm=LLM,
            memory_retriever = self.create_new_memory_retriever(),
            verbose = False,
            reflection_threshold = 15,
        )

        sid=config['scene_id']
        narrator = Narrator(
            id=0,
            name="Narrator",
            traits="",
            status="",
            llm=LLM,
            memory=agent_memory,
            scene=SceneInfo(**self.data.scenes[self.config['scene_id']]),
            characters=self.data.characters_list,
            
        )

        return narrator

    # 用于agents_creation的工具函数
    def get_other_agents(self, characters, agent_id):
        return [characters[i] for i in range(len(characters)) if characters[i].id != agent_id]

    def agents_creation(self):
        """
        Create agents in parallel
        """
        characters = {}
        character_api_key= self.config.get("character_api_key","")
        character_api_base = self.config.get("character_api_base","")
        # Add ONE human controllable role into the simulator if the flag is true.
        # We block the main thread when the user is creating the role.
        self.last_round=""
        for action in self.data.init_actions:
            self.last_round += action["character"] + ":" + action['action'] + ";" + action["character"] + ":" + action['dialogue'] + '\n'
        

        for i in tqdm(self.data.characters.keys()):
            if self.config["play_role"] is not None and i==self.config["play_role"]:
                role_id = self.config["play_role"]
                agent = self.create_human_player(role_id, character_api_key, character_api_base)
                characters[role_id] = agent
                self.data.role_id = role_id
                continue
            agent = self.create_character(i, character_api_key, character_api_base)
            characters[agent.id] = agent
            self.round_obs[agent.id]=[]
            self.round_obs[agent.id].append(self.last_round)
                
        narrator_api_key= self.config.get("narrator_api_key","")
        narrator_api_base = self.config.get("narrator_api_base","")
        narrator = self.create_narrator(self.config, self.logger, narrator_api_key, narrator_api_base)

        return narrator, characters

    def reset(self):
        # Reset the system
        self.pause()
        self.round_cnt = 0
        log_string = ""
        self.load_simulator()
        log_string = "The system is reset, and the historic records are removed."
        return log_string


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
         "--character_llm", type=str,default=None, help="Path to config file"
    )
    parser.add_argument(
         "--scene_path", type=str,default=None, help="Path to config file"
    )
    parser.add_argument(
        "-c", "--config_file", type=str, required=True, help="Path to config file"
    )
    parser.add_argument(
        "-i", "--scene_id", type=int, default=None, help="Path to config file"
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
    args = parser.parse_args()
    return args


def run_simulation(config, logger):

    role_agent = Simulator(config, logger)
    role_agent.load_simulator()

    messages = []
    role_agent.play()
    for _ in range(role_agent.round_cnt + 1, config["round"] + 1):
        role_agent.round_cnt = role_agent.round_cnt + 1
        role_agent.logger.info(f"Round {role_agent.round_cnt}")

        message = role_agent.round()
        messages.append(message)
        
    role_agent.save_record()

def main():
    args = parse_args()

    
    # create config
    config = CfgNode(new_allowed=True)
    config = utils.add_variable_to_config(config, "log_file", args.log_file)
    config = utils.add_variable_to_config(config, "log_name", args.log_name)
    config = utils.add_variable_to_config(config, "play_role", args.play_role)
    config.merge_from_file(args.config_file)
    if args.character_llm is not None:
        config['character_llm']=args.character_llm
    if args.scene_path is not None:
        config['scene_path']=args.scene_path
    if args.scene_id is not None:
        config['scene_id']=args.scene_id
    title=config['scene_path'].split("/")[-1].split(".")[0]
    if args.log_file=="":
        utils.ensure_dir("output/log/simulation/"+title)
        args.log_file=f"simulation/{title}/{config['narrator_llm']}_{config['character_llm']}_{config['scene_id']}_character_simulation.log"
    logger = utils.set_logger(args.log_file, args.log_name)
    logger.info(f"os.getpid()={os.getpid()}")
    logger.info(f"\n{config}")

    
    role_agent = Simulator(config, logger)
    role_agent.load_simulator()

    messages = []
    role_agent.play()
    st=time.time()
    for _ in range(role_agent.round_cnt + 1, config["round"] + 1):
        role_agent.round_cnt = role_agent.round_cnt + 1
        role_agent.logger.info(f"Round {role_agent.round_cnt}")

        message = role_agent.round()
        messages.append(message)
        

    ed=time.time()
    duration=ed-st
    role_agent.duration=duration
    logger.info(f"Duration:{duration}")
    print(f"Duration:{duration}")
    role_agent.save_record()

if __name__ == "__main__":
    main()

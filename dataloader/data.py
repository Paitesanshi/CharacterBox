import csv
import json
import copy
from utils.character import CharacterInfo,SceneInfo

class Data:
    """
    Data class for loading data from local files.
    """
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.scenes = {}
        self.init_scene=None
        self.init_characters=[]
        self.characters = {}
        self.characters_list = []
        self.name2id = {}
        self.groups = []
        self.db = None
        self.tot_relationship_num = 0
        self.netwerk_density = 0.0
        self.role_id = -1
        self.init_actions = []
        self.load_scene(config["scene_path"])
       
    def load_scene(self, file_path):
        """
        Load scene from local file.
        """
        with open(file_path, "r", newline="", encoding='utf-8') as file:
            work = json.load(file)
            scenes = work['scenes']
            scene = scenes[self.config['scene_id']]
            # scene['source'] = work['title']
            self.scenes[scene['id']] = scene
            self.init_scene = str(SceneInfo(**scene))
            for character in scene['characters']:
                c = CharacterInfo(**character)
                self.characters[character['id']] = c
                self.name2id[character['name']] = character['id']
                self.characters_list.append(c)
            if "actions" in scene:
                self.init_actions = scene['actions']
            
            self.init_characters = "\n".join(str(c) for c in self.characters_list)
            


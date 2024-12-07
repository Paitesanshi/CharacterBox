from datetime import datetime
from typing import Any, Dict, Optional


from langchain.schema import BaseMemory
from langchain.prompts import PromptTemplate
from langchain_experimental.generative_agents import GenerativeAgent,GenerativeAgentMemory


from utils.character import SceneInfo

class HumanPlayer(GenerativeAgent):
    id: int
    """The agent's unique identifier"""

    traits: str
    """The agent's traits"""

    position:str

    states: str

    scene: SceneInfo

    self_belief: str=""

    env_belief: str=""

    BUFFERSIZE = 10
    """The size of the agent's history buffer"""

    max_dialogue_token_limit: int = 2048
    """The maximum number of tokens to use in a dialogue"""

    memory: GenerativeAgentMemory
    """The memory module in Character."""


    def update_from_dict(self, data_dict: dict):
        for key, value in data_dict.items():
            setattr(self, key, value)

    def reset_agent(self):
        """
        Reset the agent attributes, including memory, watched_history and heared_history.
        """
        # Remove watched_history and heared_history
        self.watched_history = []
        self.heared_history = []

    def _generate_reaction(
        self, suffix: str, now: Optional[datetime] = None, **extra_kwargs
    ) -> str:
        """React to a given observation."""
        prompt = PromptTemplate.from_template(
            "Please act as {name} and respond using the tone, manner and vocabulary {name} would use.\n The context and {name}'s information is as follows:\n"
            +"Event: {event}\n"
            +"Time: {time}\n"
            +"Location: {location}\n"
            +"Description: {description}\n"
            +"Character Profile:\n"
            +"Name: {name}\n"
            +"Character Description: {character_description}\n"
            +"Position: {position}\n"
            +"Current Status: {states}\n"
            +"Self Belief: {self_belief}\n"
            +"Environment Belief: {env_belief}\n"
            +suffix
        )
        now = datetime.now() if now is None else now
        current_time_str = (
            datetime.now().strftime("%B %d, %Y, %I:%M %p")
            if now is None
            else now.strftime("%B %d, %Y, %I:%M %p")
        )
        kwargs: Dict[str, Any] = dict(
            # source=self.scene.source,
            event=self.scene.event,
            time=self.scene.time,
            location=self.scene.location,
            description=self.scene.description,
            name=self.name,
            states=self.states,
            character_description=self.traits,
            position=self.position,
            self_belief=self.self_belief,
            env_belief=self.env_belief,
             **extra_kwargs
        )
        result = self.chain(prompt=prompt).invoke(input=kwargs)["text"].strip()
        return result

    def generate_dialogue(self) -> str:
        """
        Controled by human
        """
        response=input("请输入你的发言：")
        return response

    def take_action(self) -> str:
        """
        Controled by human
        """
        response=input("请输入你的行动：")
        return response

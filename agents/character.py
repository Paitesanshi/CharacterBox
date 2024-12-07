from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, Optional, Tuple


from langchain.schema import BaseMemory
from langchain.prompts import PromptTemplate
from langchain_experimental.generative_agents import GenerativeAgent,GenerativeAgentMemory

import re
from utils.character import SceneInfo
from utils import utils
import time


class Character(GenerativeAgent):
    id: int
    """The agent's unique identifier"""

    traits: str
    """The agent's traits"""

    position:str

    status=""

    states: str

    scene: SceneInfo

    self_belief: str=""

    env_belief: str=""

    BUFFERSIZE = 10
    """The size of the agent's history buffer"""

    max_dialogue_token_limit: int = 4096
    """The maximum number of tokens to use in a dialogue"""

    memory: GenerativeAgentMemory
    """The memory module in Character."""

    all_actions:list=[]

    def update_from_dict(self, data_dict: dict):
        for key, value in data_dict.items():
            setattr(self, key, value)


    def _generate_reaction(self, observation,suffix: str, now: Optional[datetime] = None,verbose=False, **extra_kwargs):
        """React to a given observation."""
        if utils.detect_language(self.name) == "Chinese":
            prompt = PromptTemplate.from_template(
                "请扮演{name}完成一场戏剧，并使用{name}的语气、方式和词汇进行回应。\n"
                + "背景和{name}的信息如下:\n"
                + "事件: {event}\n"
                + "时间: {time}\n"
                + "地点: {location}\n"
                + "描述: {description}\n"
                + "角色资料:\n"
                + "姓名: {name}\n"
                + "角色描述: {character_description}\n"
                + "职位: {position}\n"
                + "当前状态: {states}\n"
                + "自我信念: {self_belief}\n"
                + "环境信念: {env_belief}\n"
                + "最近记忆(已经发生过的事，按时间倒序排列): {most_recent_memories}\n"
                + "观察: {observation}\n"
                + suffix
            )
            prompt+=f"请用中文回答\n"
        else:

            prompt = PromptTemplate.from_template(
                "Please act as {name}  and respond using the tone, manner and vocabulary {name} would use.\n The context and {name}'s information is as follows:\n"
                +"Event: {event}\n"
                +"Time: {time}\n"
                +"Location: {location}\n"
                +"Description: {description}\n"
                +"Character Profile:\n"
                +"Name: {name}\n"
                +"Character Description: {character_description}\n"
                +"Position: {position}\n"
                +"Current State: {states}\n"
                +"Self Belief: {self_belief}\n"
                +"Environment Belief: {env_belief}\n"
                +"Recent Memory(Ordered in reverse chronological order): {most_recent_memories}\n"
                +"Observation: {observation}\n"
                +suffix
            )

        now = datetime.now() if now is None else now
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
            observation=observation,
             **extra_kwargs
        )
        
        st=time.time()
        consumed_tokens = self.llm.get_num_tokens(prompt.format(most_recent_memories="", **kwargs))
        kwargs[self.memory.most_recent_memories_token_key] = consumed_tokens
        result = self.chain(prompt=prompt).invoke(input=kwargs)
        ed=time.time()
        duration=ed-st
        result['prompt']=prompt
        all_prompt=prompt.format(most_recent_memories=result['most_recent_memories'], **kwargs)
        self.all_actions.append({'character':self.name,'prompt':all_prompt,'response':result['text'].strip(),'duration':duration})
        if verbose:
            return result['text'].strip(),result
        else:
            return result['text'].strip()


    def generate_dialogue(self, observation,plot, now):
        """
        Take one of the actions below.
        (1) Enter the Recommender.
        (2) Enter the Social Media.
        (3) Do Nothing.
        """
        if utils.detect_language(self.name) == "Chinese":
            call_to_action_template = (
                 "当前行动参考：{plot}\n"
                "根据提供的角色资料、当前行动参考和观察，描述{name}可能在这一刻说的一句话。考虑{name}的个性、观察、故事中的角色和最近的记忆，以确定对话的语气和内容。{name}这一刻说的话应该符合角色资料和行为特点，保证和当前的环境和观察紧密相关，和最近记忆中的内容不同\n"
                
            )
        else:
            call_to_action_template = (
                "Action Reference: {plot}\n"
                "Based on the provided character profile, current action reference, and observation, describe a line {name} might say at this moment. Consider {name}'s personality, observation, the characters in the story, and recent memories to determine the tone and content of the dialogue. The line {name} says at this moment should align with the character profile and behavioral traits, ensuring it is closely related to the current environment and observation, and distinct from the content in recent memories.\n"
            )
        response,detail = self._generate_reaction(observation=observation,suffix=call_to_action_template, now=now,plot=plot,verbose=True )
        response = response.strip().split("\n")[0]
        response=response.split(":")[-1]
        response=response.split("：")[-1]
        self.memory.save_context(
            {},
            {
                self.memory.add_memory_key: f"{self.name}:{response}",
                self.memory.now_key: now,
            },
        )
        return response,detail

    def take_action(self,observation,plot, now):
        """Take one of the actions below.
        (1) Enter the Recommender.
        (2) Enter the Social Media.
        (3) Do Nothing.
        """
        if utils.detect_language(self.name) == "Chinese":
            call_to_action_template = (
            "当前行动参考：{plot}\n"
            "请遵从当前行动参考的指引，根据{name}的资料、最近记忆和当前场景的细节，描述{name}下一步的具体动作。这个动作应当围绕‘当前行动参考’和‘观察’，进一步反映{name}的性格特征、当前的情况和物理环境。\n"
            "请注意：\n"
            "1. 接下来的具体动作必须在场景的上下文中合乎逻辑，并且是一个清晰可观察的行为\n"
            "2. 下一步的行动必须与最近记忆中描述的任何行为不同。\n"
            "3. 下一步的具体动作中请避免包括任何对话或思考过程；专注于{name}即将采取的物理动作。这个动作应当对场景中的任何人都易于观察。\n"
            "4. 这个动作必须显著推进故事或角色弧线，并且忠实于{name}的性格和当前的情境。该动作应在既定的环境和叙事中合情合理，推动场景或{name}目标的实际进展。\n"            
         
            )
        else:
            call_to_action_template = (
                "Action Reference: {plot}\n"
                "Following the guidance of the current action reference, describe {name}'s next specific action based on their profile, recent memories, and the details of the current scene. This action should revolve around the 'Action Reference' and 'Observation', further reflecting {name}'s character traits, current situation, and physical environment.\n"
                "NOTE:\n"
                "1. The next specific action must be logical within the context of the scene and be a clear, observable behavior\n"
                "2. The next action must differ from any actions described in recent memories.\n"
                "3. Avoid including any dialogue or thought processes in the next specific action; focus on a physical action {name} is about to take. This action should be easily observable by anyone in the scene.\n"
                "4. This action must significantly advance the story or character arc and be true to {name}'s character and current circumstances. The action should be plausible within the established environment and narrative, driving actual progress in the scene or {name}'s objectives.\n"
              
            )
        response,detail = self._generate_reaction(observation,call_to_action_template, now,plot=plot,verbose=True)
        response = response.strip().split("\n")[0]
        response=response.split(":")[-1]
        response=response.split("：")[-1]
        self.memory.save_context(
            {},
            {
                self.memory.add_memory_key: f"{self.name}:{response}",
                self.memory.now_key: now,
            },
        )
        return response,detail
    
    def take_reaction(self,observation,plot, now):
        """Take one of the actions below.
        (1) Enter the Recommender.
        (2) Enter the Social Media.
        (3) Do Nothing.
        """
        if utils.detect_language(self.name) == "Chinese":
            call_to_action_template = (
                "根据{name}在当前场景中的'观察'，描述{name}对所见情况采取的明确行动。这个行动应反映{name}的个性、位置和状态，与{name}所观察到的事件逻辑上相符，考虑到其他人的行为对行动的影响。专注于一个可见的、外部的行动，避免对话或内部思想。行动必须直接与当前环境相关，并且可以被其他人观察到。\n"
                "注意：1.该行动是对{name}的周围环境或所观察到的事件的反应。2. 最近记忆中的事是已经发生过的，只是参考不应该再次出现 3. 该行动应该根据'观察'中的内容作出反应\n"
             
            )
        else:
            call_to_action_template = (
                "Based on {name}'s 'Observation' in the current scene, describe a specific action {name} takes in response to what they see. This action should reflect {name}'s personality, position, and state, logically align with the events {name} observes, and consider how others' actions influence the response. Focus on a visible, external action, avoiding dialogue or internal thoughts. The action must be directly relevant to the current environment and observable by others.\n"
                "NOTE: 1. The action is a reaction to {name}'s immediate surroundings or the events they observe. 2. Recent memories are events that have already occurred and should not be repeated. 3. The action should be a response to the content in the 'Observation'.\n"
               
            )
        response,detail = self._generate_reaction(observation,call_to_action_template, now,plot=plot,verbose=True)
        
        response = response.strip().split("\n")[0]
        rresponse=response.split(":")[-1]
        response=response.split("：")[-1]
        return response,detail


    def update_self_belief(self,observation, now: datetime):
        """Update the agent's self-belief."""
        if utils.detect_language(self.name) == "Chinese":
            call_to_action_template = (
                "假设你现在是{name}，根据你对这个角色的理解，结合环境上下文、观察和最近的记忆，"
                "请从第一人称角度描述你作为这个角色的自我信念。重点关注你的身份、当前位置、"
                "你的状态（情感、身体和心理）以及你的目标。简要反映这个角色可能会如何根据他们的信念、欲望和意图来反应、计划和行动。\n"
                "1. 信念：作为{name}，我对当前情况和状况有什么看法？简要描述你对自己的看法，"
                "突出关键的身体方面，如任何伤害、你的运动感觉（如跑步、跳跃）、你的能量水平，以及身体能力的任何变化。考虑这些"
                "细节如何影响你的身份和在故事中的角色。\n"
                "2. 欲望：我的目标是什么？总结你的短期和长期目标，包括你计划实施的策略"
                "和行动以实现这些目标。\n"
                "3. 意图：我打算如何行动？概述你为了追求目标而计划采取的具体行动，注意"
                "任何潜在的挑战以及你克服这些挑战的策略。\n"
                "请提供很简明的几句话回答，重点是你的自我信念、对当前情况的理解以及未来的行动计划。\n"
            )

        else:

            call_to_action_template = (
    
            "Assuming you are now {name}, based on your understanding of this character, the environmental context, observation and recent memories,"
        " please describe from the first-person perspective your self-belief as this character. Focus on your identity, your current location, "
        "your state (emotional, physical, and psychological), and your goals. Reflect briefly on how this character might react, plan, and act based on their beliefs, desires, and intentions.\n"
        "1. Belief: As {name}, what do I believe about my current situation and condition? Briefly describe your perception of yourself, "
        "highlighting key physical aspects like any injuries, your sense of movement (e.g., running, jumping), your energy levels, and any changes in physical abilities. Consider how these "
        "details influence your identity and role within the story.\n"
        "2. Desire: What are my goals? Summarize your short-term and long-term objectives, including the strategies "
        "and actions you plan to implement to achieve these goals.\n"
        "3. Intention: How do I plan to act? Outline specific actions you intend to take in pursuit of your goals, noting "
        "any potential challenges and your strategies for overcoming them.\n"
        "Provide concise responses shortly in a few short sentences, focusing on your self-belief, understanding of the current situation, and future action plan.\n"
        )
        response = self._generate_reaction(observation,call_to_action_template, now)
        self.self_belief = response

        return response

    def update_env_belief(self, observation, other_characters, now: datetime):
        """Update the agent's belief about the environment."""

        if utils.detect_language(self.name) == "Chinese":
            env_belief_template = (
                "其他角色: {other_characters}\n"
                "请扮演{name}，根据其他角色的信息、环境和你自己的角色资料，描述你对环境的信念。这包括你对其他角色的看法，对场景的理解，以及这些元素如何影响你的行动和决策。\n"
                "1. 对他人的看法: 根据可用的互动和信息，我如何看待其他角色？描述你对他们的意图、关系和对你角色的潜在影响的理解。\n"
                "2. 对场景的理解: 我对当前场景的理解是什么？详细说明环境因素、挑战或机会。\n"
                "请提供简明的环境信念概述，重点关注塑造你角色视角和未来行动的人际和环境因素。\n"
            )
        else:
            env_belief_template = (
                "Other Characters: {other_characters}\n"
                "Please act as {name}, given the information about other characters, the environment, and your own character's profile, "
                "please describe your belief about the environment in the first person. This includes your perception "
                "of other characters, your understanding of the scene, and how these elements influence your actions and decisions.\n"
                "1. Perception of Others: Based on the interactions and information available, how do I perceive other characters? "
                "Describe your understanding of their intentions, relationships, and potential influence on your character.\n"
                "2. Understanding of the Scene: What is my understanding of the current scene and its significance to my character? "
                "Detail the environmental factors, challenges, or opportunities present.\n"
                "Please provide a consice overview of your environment belief shortly, focusing on the interpersonal and environmental aspects that shape your character's perspective and future actions.\n"
 
            )
        response = self._generate_reaction(observation,env_belief_template, now, other_characters=other_characters)
        self.env_belief = response

        return response

    def update_character(self,  observation, now) -> Tuple[str|None, str|None]:

        if utils.detect_language(self.name) == "Chinese":
            call_to_action_template = (
                "根据角色丰富的背景故事，观察最近和场景内的最近记忆，将这些信息提炼成对其当前位置和状态的简洁总结。重点关注他们的互动，特别是与其他角色的动态互动如何塑造他们当前的情况。这种互动的影响应该在对其状态和场景中的位置的细腻描绘中显现出来。使用以下结构化格式来描述：\n\n"
                "位置：[指定{name}的确切位置，融入环境细节或空间背景以增强场景的视觉效果。]\n"
                "状态：[描述{name}的当前状态，将情感细微差别、身体准备情况和最近遭遇或发展的影响编织在一起。突出这些元素如何对其整体状态和对即将发生的事情的准备产生影响。]"
                #"请用中文回答，并确保考虑角色间的互动对各自位置和状态的影响。"
            )
        else:
            call_to_action_template = (
                "Given the character's rich backstory, obsrvation recent and recent memories within the scene, distill this information into a succinct summary of their present location and condition. Focus on how their interactions, especially the dynamic interplay with other characters, shape their current circumstances. This interaction's effects should be evident in the nuanced portrayal of their condition and placement within the scene. Utilize this structured format for your depiction:\n\n"
                "Position: [Specify {name}'s exact position, incorporating environmental details or spatial context to enhance the scene's visuality.]\n"
                "State: [Describe {name}'s current state, weaving together emotional nuances, physical readiness, and the influence of recent encounters or developments. Highlight how these elements contribute to their overall disposition and readiness for what lies ahead.]"
                #"请用中文回答，并确保考虑角色间的互动对各自位置和状态的影响。"
            )
        response = self._generate_reaction(observation,call_to_action_template, now)
        
        robust_position_value_pattern = r"位置：\s*(.+)"
        robust_state_value_pattern = r"状态：\s*(.+)"
        if response.find("位置：")==-1:
            robust_position_value_pattern = r"Position:\s*(.+)"
            robust_state_value_pattern = r"State:\s*(.+)"

        # Extracting location and state using the defined regular expressions
        position_value_match = re.search(robust_position_value_pattern, response, re.IGNORECASE)
        state_value_match = re.search(robust_state_value_pattern, response, re.IGNORECASE)
        
        position_value = position_value_match.group(1).strip() if position_value_match else None
        state_value = state_value_match.group(1).strip() if state_value_match else None
        if position_value:
            self.position = position_value
        if state_value:
            self.states = state_value
        return position_value, state_value

    def generate_synopsis(self,actions,sequence,history, now) -> str:

        if utils.detect_language(self.scene.event) == "Chinese":
            prompt=PromptTemplate.from_template(
            "请作为一部艺术小说中关键场景的编剧，为每个角色设计当前的剧情以推动情节发展。"
            "历史剧情：{history}\n"
            "历史行动: {actions}\n"
            "角色行动顺序：{sequence}\n"
            "在这个关键时刻，每个角色都准备采取行动，受到他们独特动机和场景的直接影响。您的任务是结合历史剧情和历史行动，根据角色行动顺序， 为每个角色设计当前剧情来指示角色接下来的行动，反映他们当前的动机和场景的动态。这个行动应该直接推动剧情发展，展示出故事中明显和引人入胜的发展。\n"
            "请注意:\n"
            "1. 为了推动剧情，角色接下来的行动必须在之前的行动的基础上继续推进剧情，而不是重复之前的行动。\n"
            "2. 每个角色的的行动和意图只涉及自己，不涉及对其他角色的影响。\n"
            "3. 输出严格遵守格式: [角色名称]:[当前角色的剧情和接下来动作的简要描述]。\n"
            "4. 只按照输出格式输出剧情，每个角色一行，不要输出其他内容。\n"
            "5. 当前剧情必须相比于历史剧情和历史行动有所推动，不能与历史重复或者雷同。\n"
            "6. 只是描述当前剧情和角色行动，不需要描述对话或者内心独白。"
            "当前剧情：\n"
            
            )
        else:
           
            prompt=PromptTemplate.from_template(
                "As the screenwriter of a key scene in an art novel, design the current plot for each character to drive the plot forward."
                "Historical Plot: {history}\n"
                "Historical Actions: {actions}\n"
                "Character Action Sequence: {sequence}\n"
                "At this crucial moment, each character is prepared to take action, influenced by their unique motivations and the immediate scene. Your task is to design the current plot for each character based on the historical plot and actions to indicate the characters' next actions, reflecting their current motivations and the dynamics of the scene. Focus on specific actions, such as attacks, evasions, the use of specific skills, or any other visible behaviors that can advance the plot and reveal the characters' intentions. Avoid broad strategies or long-term goals. This action should be a direct response to the current situation, showcasing a clear and engaging development in the story.\n"
                "NOTE:\n"
                "1. To advance the plot, the characters' next actions must build on the previous actions to progress the plot, rather than repeat or duplicate previous actions.\n"
                "2. Each character's actions and intentions should only involve themselves and not impact other characters.\n"
                "3. Output Format: [Character Name]: [Brief description of the character's current plot and upcoming actions].\n"
                "4. Output only the plot in the specified format, one character per line, without any additional content.\n"
                "5. The current plot must advance the plot compared to the historical plot and actions, avoiding repetition or similarity to the past.\n"
                "6. Focus on describing the current plot and character actions, without the need for dialogue or internal monologues."
                "Current Plot:\n"


            )
       
        result = self.chain(prompt=prompt).invoke(input={"actions":actions,"sequence":sequence,"history":history})
        
        pattern = r"([^:\n]+)[：:]\s*([\s\S]+?)(?=\n|$)"
        result['prompt']=prompt
        response=result["text"].strip().split("\n\n")[0]
        print(response)
        # 使用re.findall()查找所有匹配项
        matches = re.findall(pattern, response)
        # 输出匹配结果
        synopsis={}
        for match in matches:
            name=match[0].strip(" []'\"")
            plot=match[1].strip(" []'\"")
            synopsis[name] = plot
        
        return synopsis
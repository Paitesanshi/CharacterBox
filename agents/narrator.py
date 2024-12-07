from __future__ import annotations
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import json

from langchain.schema import BaseMemory
from langchain.prompts import PromptTemplate
from langchain_experimental.generative_agents import GenerativeAgent,GenerativeAgentMemory


from utils.character import CharacterInfo,SceneInfo
from utils import utils
import time

class Narrator(GenerativeAgent):
    id: int
    """The agent's unique identifier"""
  
    BUFFERSIZE = 10
    """The size of the agent's history buffer"""

    max_dialogue_token_limit: int = 600
    """The maximum number of tokens to use in a dialogue"""

    synopsis:List[str]=[]

    plots:List[str]=[]

    history: List[str] = []


    characters:List[CharacterInfo]


    scene: SceneInfo

    memory: GenerativeAgentMemory

    all_actions:list=[]
    
    round_plot_actions:List[Dict]=[]

    """The memory module in Narrator."""

    def update_from_dict(self, data_dict: dict):
        for key, value in data_dict.items():
            setattr(self, key, value)

    def _generate_reaction(self, suffix: str, now: Optional[datetime] = None,verbose=False, **extra_kwargs) -> str:
        """React to a given observation."""

        if utils.detect_language(self.scene.event) == "Chinese":
            prompt = PromptTemplate.from_template(
                "请作为一部艺术小说中关键场景的编剧，突出角色的独特特征并推动情节发展。\n"
                + "事件: {event}\n"
                + "时间: {time}\n"
                + "地点: {location}\n"
                + "描述: {description}\n"
                + "角色: {characters}\n"
                + "场景中除了上述角色外没有其他人。\n"
                + suffix
            )
        else:
            prompt = PromptTemplate.from_template(
                "Please act as a screenwriter for a key scene in an art novel that highlights the characters' unique traits and drives the plot forward\n"
                +"Event: {event}\n"
                +"Time: {time}\n"
                +"Location: {location}\n"
                +"Description: {description}\n"
                +"Characters: {characters}\n"
                +"There is no one else in the scene other than the above characters.\n"
                +suffix
            )
        now = datetime.now() if now is None else now
        kwargs: Dict[str, Any] = dict(
            event=self.scene.event,
            time=self.scene.time,
            location=self.scene.location,
            description=self.scene.description,
            characters=self.characters,
            #synopsis=self.synopsis,
            **extra_kwargs
        )

        st=time.time()
        result = self.chain(prompt=prompt).invoke(input=kwargs)
        ed=time.time()
        duration=ed-st
        result['prompt']=prompt
        all_prompt=prompt.format(**kwargs)
        self.all_actions.append({'character':self.name,'prompt':all_prompt,'response':result['text'].strip(),'duration':duration})
        if verbose:
            return result['text'].strip(),result
        else:
            return result['text'].strip()

    def update_character(self, name, observation, now,verbose=False):

        if utils.detect_language(self.scene.event) == "Chinese":
            call_to_action_template = (
                "观察：{observation}\n"
                "角色名称：{name}\n"
                "根据角色丰富的背景和场景中的观察，将这些信息提炼成一个关于他们当前位置和状态的简明总结。侧重于他们的互动，特别是与其他角色的动态互动，以及这些互动如何塑造了他们当前的情况。这种互动的影响应该在对其状态和场景中的位置的细致描绘中显现出来。利用以下结构化格式来描述：\n\n"
                "位置：[指定{name}的确切位置，结合环境细节或空间背景来增强场景的视觉效果。]\n"
                "状态：[描述{name}的当前状态，将情感细微差异、身体准备情况以及最近的相遇或发展的影响编织在一起。]"
                #+"请用中文回答，并确保考虑角色间的互动对各自位置和状态的影响。"
            )
        else:
            call_to_action_template = (
                "Observation: {observation}\n"
                "Character Name: {name}\n"
                "Given the character's rich backstory and observation within the scene, distill this information into a succinct summary of their present location and condition. Focus on how their interactions, especially the dynamic interplay with other characters, shape their current circumstances. This interaction's effects should be evident in the nuanced portrayal of their condition and placement within the scene. Utilize this structured format for your depiction:\n\n"
                "Position: [Specify {name}'s exact position, incorporating environmental details or spatial context to enhance the scene's visuality.]\n"
                "State: [Describe {name}'s current state, weaving together emotional nuances, physical readiness, and the influence of recent encounters or developments.]"
            # "请用中文回答，并确保考虑角色间的互动对各自位置和状态的影响。"
            )
        response,detail = self._generate_reaction(call_to_action_template, now, observation=observation, name=name,verbose=True)
        
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
        if verbose:
            return position_value, state_value,detail
        return position_value, state_value,None


    def analyze_action(self, actor,action, now) -> Tuple[int,str]:

        if utils.detect_language(self.scene.event) == "Chinese":
            call_to_action_template = (
                "动作: {action}\n"
                "执行者: {actor}\n"
                "在给定的上下文中分析动作和执行者，以确定在角色列表中最有可能对该动作做出反应的角色。评估并解释为什么该角色会受到影响，并描述可能的反应。如果该动作直接对目标角色产生物理影响，请识别并描述具体的影响动作，以及其原因和效果。响应格式应为[反应角色名]|[受影响动作]。如果没有角色可能做出反应，则返回执行者的名字。确保响应简洁明了。"
            )

        else:

            call_to_action_template = (
                "Action: {action}\n"
                "Actor: {actor}\n"
                "Analyze the action and the actor within the given context to determine which character from the Characters is most likely to respond to the action. Assess and explain why this character would be affected and describe the potential reaction. If the action directly physically impacts a target character, identify and describe the specific action affecting them, along with the reason and its effect. Responses should be formatted as [Reacting Character Name]|[Affected Action]. If no character is likely to react, return the actor's name. Ensure the response is concise and clear."
            )
        response = self._generate_reaction(call_to_action_template, now, action=action,actor=actor)
        pattern = re.compile(r"\s*(.*)\|\s*(.*)")

        match = re.match(pattern, response)
        target_id = None
        action = ""
        reason=""
        if match:
            target_id, action = match.groups()
            #target_id = int(target_id)
            for character in self.characters:
                if character.name==target_id:
                    target_id = character.id
                    break
        return target_id,action

    def analyze_action_influence(self, actor,action, now,verbose=False) :

        if utils.detect_language(self.scene.event) == "Chinese":
            call_to_action_template=(
                "动作: {action}\n"
                "执行者: {actor}\n"
                "请分析上述详细的物理动作和影响，特别关注对“角色”中列出的一个角色的影响。"
                "您的分析必须：\n"
                "1. 确定受影响的目标角色（必须来自“角色”列表）。\n"
                "2. 描述执行者发起的具体物理动作。\n"
                "3. 解释此动作对目标角色的物理状态或情况的具体影响。\n"
                "4. 您必须从“Characters”列表中选择一个角色。\n"
                "5. 强调物理互动或影响。如果某个动作不会对任何列出的角色产生物理影响，请将执行者的名字返回为目标名。\n"
                "6. 可感知的行为才有影响。如果角色与动作发生地点不在同一空间，或者感知不到动作，则不会产生影响，请将执行者的名字返回为目标名。\n"
                "7. 必须按照以下格式编写您的响应：[执行者];;[目标名];;[执行者对目标的详细物理影响]。\n"
                "确保响应简洁、准确，并符合指定的格式。"
            )
        else:

            call_to_action_template = (
    "Action: {action}\n"
        "Actor: {actor}\n"
        "Please analyze the physical actions and impacts detailed above, specifically focusing on the effects on ONLY one character listed in 'Characters'.\n"
        "Your analysis must:\n"
        "1. Identify the target character affected (must be from the 'Characters' list).\n"
        "2. Describe the specific physical action initiated by the actor.\n"
        "3. Explain the tangible impact of this action on the target character's physical state or circumstances.\n"
        "4. You must pick up ONLY ONE character from the 'Characters' list.\n"
        "5. Emphasize physical interactions or impacts. If an action does not physically affect any characters listed, return the actor's name as Target Name.\n"
        "6. Must format your response as follows: [Actor];;[Target Name];;[Detailed Physical Impact of {actor} on Target].\n"
        "Ensure responses are concise, precise, and adhere to the specified format.\n"
            )
        response,detail = self._generate_reaction(call_to_action_template, now, action=action,actor=actor,verbose=True)
        #print("influence response: ",response)
        pattern = r"\[?([^;\[\]]+)\]?\s*;;\s*\[?([^;\[\]]+)\]?\s*;;\s*\[?([^;\[\]]+)\]?"
        matches = re.search(pattern, response)
        target_id=None
        if matches:
            actor = matches.group(1)
            target_name = matches.group(2)
            action = matches.group(3)
            for character in self.characters:
                if character.name==target_name:
                    target_id = character.id
                    break
        else:
            print("No match found.")
        if verbose:
            return target_id,action,detail
        return target_id,action

    def analyze_result(self, actions, now,verbose=False):
        """Take one of the actions below.
        (1) Enter the Recommender.
        (2) Enter the Social Media.
        (3) Do Nothing.
        """

        if utils.detect_language(self.scene.event) == "Chinese":
            call_to_action_template = (
                "动作: {actions}\n"
                "指导: 请充当即时事件裁判，迅速分析推理判断指定角色之间互动的结果和影响。以简明的全知旁观者的语气叙述当前角色之间交互动作的直接结果。您的叙述应清晰直接地阐明动作之间的因果关系，强调当前动作的直接结果，不要深入未来的影响或延伸故事情节。"
                "非常重要的说明：\n"
                "1. 用即时性叙述结果，重点关注当前动作互动的直接结果。\n"
                "2. 用简明的全知旁观者的语气保持叙述风格，确保分析直截了当、简明扼要。\n"
                "3. 依据提供的角色描述和动作来叙述，避免插入推测性或冗余的细节。\n"
                "4. 严格禁止在结果中重复任何输入的角色动作，结果仅包含动作互动的直接结果。"
            )
        else:
            call_to_action_template = (
                "Actions: {actions}\n"
                "Instruction: Serve as an instant event adjudicator, swiftly analyzing the interactions between specified characters and their actions. Narrate the immediate outcomes in a concise omniscient narrator's voice, focusing exclusively on the direct consequences of these interactions at this very moment. Your narration should clearly and directly elucidate the cause-and-effect relationship between actions, emphasizing the instant outcomes without delving into any future implications or extended storylines."

                "Very Important Guidelines:"

            "1. Narrate the outcomes with immediacy, centering on the direct results of the current actions' interactions."
            "2. Use a concise omniscient narrator’s voice to maintain a narrative style while ensuring the analysis is straightforward and to the point."
            "3. Your analysis should be grounded in the character descriptions and actions provided, avoiding any speculative or unnecessary detail."
            "4. Do not repeat the Actions in the result. The result is only the result of the current action interaction."
            )
        response,detail = self._generate_reaction(call_to_action_template, now, actions=actions,verbose=True)
        if verbose:
            return response,detail
        return response

    def update_scene(self, obs,verbose=False):

        if utils.detect_language(self.scene.event) == "Chinese":
            prompt = PromptTemplate.from_template(
            """
            给定一个初始场景描述和观察结果，此任务要求你更新场景以体现观察到的任何直接和显著的物理环境变化。如果观察结果未显示任何重大的物理变化，原始场景描述应保持不变。在更新场景时，保持初始场景描述的结构，避免引入原始场景中没有的新属性。

            注意：
            1. 场景更新应仅涉及物理环境的变化，避免包含任何人物行动或互动。
            2. 保持“时间”和“地点”的元素不变，除非观察结果明确指示变化。
            3. 输出应结构化为“时间”、“地点”和“环境描述”，不包含额外文本或前缀。
            4. “环境描述”应仅限于环境的物理状态，不涉及任何人物的行动或对话。
            5. "环境描述"中不应该出现任何人物角色

            输入：
            - 时间：{time}
            - 地点：{location}
            - 环境描述：{description}

            观察：{observation}

            输出：
            - 时间：{time}
            - 地点：{location}
            - 环境描述：（根据观察结果对物理环境的变化进行描述）
            """
        )

        else:
            prompt = PromptTemplate.from_template(
                """
    Given an initial scene description, examine the provided observations to identify any direct and significant physical impacts on the environment. Update the scene based on these observations, focusing solely on changes to the physical environment. If the observations do not reveal any significant physical changes to the environment, the original scene description should remain unchanged. Ensure the updated scene retains the structure of the initial scene description and does not introduce new properties that were not part of the original scene description.

    Note:
    1. The scene description should focus solely on the physical environment and should not contain character actions or interactions.
    2. The elements 'time', 'location', and 'description' in the scene should not be changed unless the observation specifically indicates a change.
    3. The output should consist of structured elements for 'time', 'location', and 'description' without adding any extra text or prefixes.
    4. The description refers to the physical description of the environment and should not include character actions or interactions.

    Input:
    - Time: {time}
    - Location: {location}
    - Description: {description}

    Observation: {observation}

    Output:
    - Time: 
    - Location: 
    - Description:
    """
        )
        st=time.time()
        result = self.chain(prompt=prompt).invoke(input={"time":self.scene.time,"location":self.scene.location,"description":self.scene.description,"observation":obs})
        result['prompt']=prompt
        response=result["text"].strip()
        ed=time.time()
        duration=ed-st
        all_prompt=prompt.format( **{"time":self.scene.time,"location":self.scene.location,"description":self.scene.description,"observation":obs})
        self.all_actions.append({'character':self.name,'prompt':all_prompt,'response':result['text'].strip(),'duration':duration})
        if utils.detect_language(self.scene.event) == "Chinese":
            output_time_pattern = re.compile(r"- 时间：\s*\[?(.+?)\]?\n", re.IGNORECASE)
            output_location_pattern = re.compile(r"- 地点：\s*\[?(.+?)\]?\n", re.IGNORECASE)
            output_description_pattern = re.compile(r"- 环境描述：\s*\[?(.+?)\]?(?:\n|$)", re.IGNORECASE | re.DOTALL)
        else:
            output_time_pattern = re.compile(r"- Time:\s*\[?(.+?)\]?\n", re.IGNORECASE)
            output_location_pattern = re.compile(r"- Location:\s*\[?(.+?)\]?\n", re.IGNORECASE)
            output_description_pattern = re.compile(r"- Description:\s*\[?(.+?)\]?(?:\n|$)", re.IGNORECASE | re.DOTALL)

        # 提取输出信息
        updated_time = output_time_pattern.search(response)
        updated_location = output_location_pattern.search(response)
        updated_description = output_description_pattern.search(response)
        if updated_time:
            self.scene.time = updated_time.group(1)
            #print("updated_time: ",updated_time.group(1))
        if updated_location:
            self.scene.location = updated_location.group(1)
            #print("updated_location: ",updated_location.group(1))
        if updated_description:
            self.scene.description = updated_description.group(1)
            #print("updated_description: ",updated_description.group(1))
        
        if verbose:
            return self.scene,result
        else:
            return self.scene

    def summary_plot(self,actions) -> str:
        if utils.detect_language(self.scene.event) == "Chinese":
            prompt = PromptTemplate.from_template(
                """
                "动作: {actions}\n"
                "基于上述提供的角色动作，总结本段的关键情节。重点关注角色采取的主要行动，由此产生的任何重大变化或事件，以及这些行动的直接结果或含义。旨在用简明的总结捕捉叙述的精髓。"
                """
            )
        else:
            prompt = PromptTemplate.from_template(
                """
                "Actions: {actions}\n"
                Given ths list of actions performed by characters as above, create a concise paragraph summarizing these actions. Detail each action in the order they occurred, focusing solely on what was done without interpreting the motivations or outcomes. Ensure the summary is clear and maintains the logical sequence of events.

                """
            )
        st=time.time()
        result = self.chain(prompt=prompt).invoke(input={"actions":actions})
        result['prompt']=prompt
        response=result["text"].strip()
        ed=time.time()
        duration=ed-st
        all_prompt=prompt.format( **{"actions":actions})
        self.all_actions.append({'character':self.name,'prompt':all_prompt,'response':result['text'].strip(),'duration':duration})

        self.plots.append(response)
        return response
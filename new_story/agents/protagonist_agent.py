from typing import List, Dict, Any
from .role_agent import RoleAgent
from models import Character, CharacterProfile
from memory import MemoryManager


class ProtagonistAgent(RoleAgent):
    def __init__(self, character: Character):
        super().__init__(character)
        self.growth_points = 0
        self.growth_arc = []
        self.observations = []
        self.special_abilities = []
        self.narrative_perspective = "first_person"

    def add_special_ability(self, ability_name: str, description: str = ""):
        self.special_abilities.append({
            "name": ability_name,
            "description": description,
            "level": 1
        })

    def upgrade_ability(self, ability_name: str):
        for ability in self.special_abilities:
            if ability["name"] == ability_name:
                ability["level"] = min(3, ability["level"] + 1)
                break

    def observe_scene(self, scene_description: str, characters: List[str] = None) -> Dict[str, Any]:
        observation = {
            "scene": scene_description,
            "characters": characters if characters else [],
            "key_elements": [],
            "potential_clues": [],
            "emotional_response": ""
        }

        prompt = f"""请以主角'{self.character.profile.name}'的视角观察以下场景：

场景描述：{scene_description}

场景中的角色：{', '.join(characters) if characters else '无'}

请分析：
1. 场景中的关键元素
2. 可能的线索或重要信息
3. 主角对此场景的情感反应

请按照以下格式返回（JSON）：
{{
    "key_elements": ["元素1", "元素2"],
    "potential_clues": ["线索1", "线索2"],
    "emotional_response": "情感描述"
}}"""

        system_prompt = "你是一位专业的叙事作家，擅长从主角视角进行细致的场景观察和分析。"
        result = self.generate_response(system_prompt, prompt)

        try:
            import json
            import re
            json_match = re.search(r"\{[\s\S]*\}", result)
            if json_match:
                data = json.loads(json_match.group(0))
                observation["key_elements"] = data.get("key_elements", [])
                observation["potential_clues"] = data.get("potential_clues", [])
                observation["emotional_response"] = data.get("emotional_response", "")
        except:
            pass

        self.observations.append(observation)
        self.add_memory(f"我观察到：{scene_description}")

        return observation

    def drive_story(self, scene_context: str, dialogue_history: List[Dict] = None) -> str:
        if dialogue_history is None:
            dialogue_history = self.get_dialogue_history(limit=10)

        dialogue_text = ""
        for d in dialogue_history:
            if "record_type" in d and (d["record_type"] == "对话" or d["record_type"].startswith("对")):
                if "target" in d:
                    dialogue_text += f"{d['speaker']} 对 {d['target']} 说：{d['content']}\n"
                else:
                    dialogue_text += f"{d['speaker']}：{d['content']}\n"

        traits_text = "，".join(self.character.profile.traits) if self.character.profile.traits else "无特定性格特征"

        prompt = f"""请作为主角'{self.character.profile.name}'驱动故事发展：

主角信息：
- 年龄：{self.character.profile.age}
- 性别：{self.character.profile.gender}
- 性格特征：{traits_text}
- 特殊能力：{', '.join([a['name'] for a in self.special_abilities])}
- 当前成长阶段：{self.get_growth_stage()}

场景上下文：
{scene_context}

对话历史：
{dialogue_text}

请根据主角的性格、目标和成长弧线，生成一段能够推动剧情发展的对话或行动。对话应：
1. 符合主角性格特点
2. 推动情节向前发展
3. 可能揭示新信息或提出关键问题
4. 与之前的对话内容保持一致

请直接返回对话内容，不要包含角色名称或动作描述。"""

        system_prompt = "你是一位经验丰富的互动叙事作家，负责为故事主角生成推动剧情发展的对话。"
        response = self.generate_response(system_prompt, prompt)

        response = response.strip()
        if response.startswith('"') and response.endswith('"'):
            response = response[1:-1]

        return response

    def update_growth(self, experience: str, impact: float = 0.1):
        self.growth_points += impact

        growth_entry = {
            "experience": experience,
            "impact": impact,
            "total_points": self.growth_points,
            "stage": self.get_growth_stage()
        }
        self.growth_arc.append(growth_entry)

        self.add_memory(f"成长经历：{experience}")

        if len(self.special_abilities) > 0 and self.growth_points >= 1.0:
            self.upgrade_ability(self.special_abilities[0]["name"])

    def get_growth_stage(self) -> str:
        if self.growth_points < 0.3:
            return "初始阶段"
        elif self.growth_points < 0.6:
            return "成长阶段"
        elif self.growth_points < 0.9:
            return "成熟阶段"
        else:
            return "巅峰阶段"

    def get_growth_summary(self) -> Dict[str, Any]:
        return {
            "total_points": self.growth_points,
            "current_stage": self.get_growth_stage(),
            "growth_arc": self.growth_arc,
            "special_abilities": self.special_abilities
        }

    def set_narrative_perspective(self, perspective: str):
        valid_perspectives = ["first_person", "third_person_limited", "third_person_omniscient"]
        if perspective in valid_perspectives:
            self.narrative_perspective = perspective

    def generate_narration(self, scene_description: str) -> str:
        perspective_text = {
            "first_person": "第一人称视角",
            "third_person_limited": "第三人称有限视角",
            "third_person_omniscient": "第三人称全知视角"
        }

        prompt = f"""请以{perspective_text[self.narrative_perspective]}为'{self.character.profile.name}'生成一段场景叙述：

场景描述：{scene_description}

主角性格：{', '.join(self.character.profile.traits)}

请生成一段生动的叙述文字，体现主角的感受和观察。"""

        system_prompt = "你是一位专业的叙事作家，擅长撰写生动的场景叙述。"
        return self.generate_response(system_prompt, prompt)

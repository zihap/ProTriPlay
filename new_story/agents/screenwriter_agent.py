from typing import List, Dict, Any
import json
import re
from .base_agent import BaseAgent
from models import Scene, Dialog, Character


class ScreenwriterAgent(BaseAgent):
    def __init__(self):
        super().__init__("Screenwriter")
        self.scene_descriptions: Dict[str, str] = {}
        self.initial_script: Dict[str, Dict] = {}

    def load_initial_script(self, script_dict: Dict[str, Dict]):
        self.initial_script = script_dict
        for scene_id, scene_data in script_dict.items():
            if "description" in scene_data:
                self.scene_descriptions[scene_id] = scene_data["description"]

    def generate_scene_setting(self, scene_id: str, director=None, 
                              player_character: str = None) -> str:
        base_description = self.scene_descriptions.get(scene_id)
        if not base_description:
            if scene_id in self.initial_script and "description" in self.initial_script[scene_id]:
                base_description = self.initial_script[scene_id]["description"]
            else:
                return f"错误：找不到场景 {scene_id} 的描述"

        scene_characters = []
        characters_info = ""

        if scene_id in self.initial_script and "characters" in self.initial_script[scene_id]:
            scene_characters = self.initial_script[scene_id]["characters"]

            if director and hasattr(director, "characters"):
                for char_name in scene_characters:
                    if player_character and char_name == player_character:
                        continue

                    if char_name in director.characters:
                        character = director.characters[char_name]
                        char_info = f"- {char_name}：{character.profile.age}岁，{character.profile.gender}\n"

                        if character.profile.traits:
                            char_info += f"  性格特征：{', '.join(character.profile.traits)}\n"

                        if character.relationships:
                            char_info += "  与其他角色的关系：\n"
                            for other_name, rel in character.relationships.items():
                                if other_name in scene_characters:
                                    char_info += f"    - 与{other_name}：{rel.relation_type}（{rel.description}）\n"

                        characters_info += char_info

        prompt = f"""根据以下场景基础描述和角色信息，生成更加详细的场景描述，包括场景环境、场景中的物品和场景中的非玩家人物：

场景基础描述：{base_description}

"""

        if characters_info:
            prompt += f"""场景中的角色信息：
{characters_info}
"""

        if player_character:
            prompt += f"""注意：场景中的玩家角色"{player_character}"不需要被详细描述，因为玩家将自己控制这个角色。
"""

        prompt += """请提供：
1. 环境描述：包括空间布局、光线、声音、气味等
2. 物品描述：列出场景中的主要物品及其摆放位置
3. 非玩家角色描述：描述场景中的角色（不包括玩家角色）的外貌、姿态、当前行为和情绪状态，性格描述应与上述角色信息保持一致
4. 玩家角色目标描述：描述玩家角色在当前场景中的大致目标

请使用生动形象的语言，创造出沉浸式的场景体验。"""

        system_prompt = "你是一位经验丰富的戏剧编剧，擅长创造生动详实的场景描述。"
        detailed_description = self.generate_response(system_prompt, prompt)

        self.scene_descriptions[scene_id] = detailed_description
        return detailed_description

    def generate_script(self, scene_id: str, director=None) -> Scene:
        scene_desc = self.scene_descriptions.get(scene_id, "")
        characters = []

        if scene_id in self.initial_script and "characters" in self.initial_script[scene_id]:
            characters = self.initial_script[scene_id]["characters"]

        dialogue_history_text = ""
        recent_dialogues = self.get_dialogue_history(limit=20)
        for d in recent_dialogues:
            if "record_type" in d:
                record_type = d.get("record_type", "")
                if record_type == "对话" or record_type.startswith("对"):
                    if "target" in d:
                        dialogue_history_text += f"{d['speaker']} 对 {d.get('target')} 说：{d['content']}\n"
                    else:
                        dialogue_history_text += f"{d['speaker']}：{d['content']}\n"

        prompt = f"""作为编剧，请根据以下信息生成场景剧本：

场景描述：{scene_desc}

场景中的角色：{', '.join(characters)}

对话历史：
{dialogue_history_text}

请生成包含以下要素的场景剧本：
1. 场景标题
2. 出场人物列表
3. 环境描述
4. 动作指示
5. 角色对话

请严格按照以下JSON格式返回：
{{
    "title": "场景标题",
    "characters": ["角色名1", "角色名2"],
    "environment": "环境描述",
    "dialogs": [
        {{"character": "角色名", "action": "动作描述", "content": "对话内容"}}
    ]
}}

注意：
1. 对话内容应该符合角色性格特点并推动情节发展
2. JSON格式必须完全正确"""

        system_prompt = "你是一位经验丰富的编剧，专门为互动式戏剧创作剧本。"
        generated_text = self.generate_response(system_prompt, prompt)

        try:
            json_match = re.search(r"\{[\s\S]*\}", generated_text)
            if json_match:
                generated_text = json_match.group(0)

            script_data = json.loads(generated_text)

            dialogs = []
            for d in script_data.get("dialogs", []):
                dialogs.append(Dialog(
                    character=d.get("character", ""),
                    content=d.get("content", ""),
                    action=d.get("action", "")
                ))

            scene = Scene(
                scene_id=scene_id,
                description=script_data.get("environment", scene_desc),
                characters=script_data.get("characters", characters),
                dialogs=dialogs
            )

            return scene

        except Exception as e:
            return Scene(
                scene_id=scene_id,
                description=scene_desc,
                characters=characters
            )

    def generate_new_scene(self, current_scene_id: str, dialogue_history: List[Dict] = None) -> Scene:
        if dialogue_history is None:
            dialogue_history = self.get_dialogue_history(limit=20)

        try:
            scene_num_match = re.search(r"(\d+)", current_scene_id)
            if scene_num_match:
                scene_num = int(scene_num_match.group(1))
                next_scene_id = current_scene_id.replace(str(scene_num), str(scene_num + 1))
            else:
                next_scene_id = f"{current_scene_id}_1"
        except:
            import time
            next_scene_id = f"scene_{int(time.time())}"

        current_scene = self.initial_script.get(current_scene_id, {})
        scene_desc = self.scene_descriptions.get(current_scene_id, "")
        current_characters = current_scene.get("characters", [])

        dialogue_text = ""
        for d in dialogue_history:
            if isinstance(d, dict):
                speaker = d.get("speaker", "")
                record_type = d.get("record_type", "")
                content = d.get("content", "")
                if record_type == "对话" or record_type.startswith("对"):
                    if "target" in d:
                        dialogue_text += f"{speaker} 对 {d.get('target')} 说：{content}\n"
                    else:
                        dialogue_text += f"{speaker}：{content}\n"

        prompt = f"""作为编剧，请根据以下信息生成剧本的下一个场景：

当前场景：{scene_desc}

当前场景中的角色：{', '.join(current_characters)}

对话历史：
{dialogue_text}

请根据上述信息，创作剧本的下一个场景。特别注意：
1. 新场景必须与对话历史保持一致，不能出现矛盾
2. 角色的行为和对话必须符合其性格特点
3. 场景转换要自然，情节发展要合理

你必须严格按照以下JSON格式返回：
{{
    "description": "详细的场景描述",
    "characters": ["角色名1", "角色名2"],
    "dialogs": [
        {{"character": "角色名", "action": "动作描述", "content": "对话内容"}}
    ]
}}

注意：
1. JSON格式必须完全正确，不要有任何解释或额外文本
2. 对话内容应该符合角色特点并推动情节发展"""

        system_prompt = "你是一位经验丰富的编剧，专门为互动式戏剧创作剧本。"
        generated_text = self.generate_response(system_prompt, prompt)

        try:
            json_match = re.search(r"\{[\s\S]*\}", generated_text)
            if json_match:
                generated_text = json_match.group(0)

            new_scene_data = json.loads(generated_text)

            dialogs = []
            for d in new_scene_data.get("dialogs", []):
                dialogs.append(Dialog(
                    character=d.get("character", ""),
                    content=d.get("content", ""),
                    action=d.get("action", "")
                ))

            new_scene = Scene(
                scene_id=next_scene_id,
                description=new_scene_data.get("description", ""),
                characters=new_scene_data.get("characters", []),
                dialogs=dialogs
            )

            self.initial_script[next_scene_id] = new_scene_data
            self.scene_descriptions[next_scene_id] = new_scene_data.get("description", "")

            return new_scene

        except Exception as e:
            return None

    def generate_transition(self, from_scene: Scene, to_scene: Scene, 
                           last_interaction: str = "") -> str:
        prompt = f"""请根据以下信息提供场景转场描述：

当前场景描述：
{from_scene.description}

最后一次互动：
{last_interaction}

下一个场景描述：
{to_scene.description}

请提供一个自然流畅的转场描述，将当前场景连接到下一个场景。"""

        system_prompt = "你是一位经验丰富的编剧，专门创作场景转场。"
        return self.generate_response(system_prompt, prompt)

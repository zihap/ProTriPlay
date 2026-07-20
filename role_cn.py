import openai
import faiss
import numpy as np
from typing import List, Dict, Tuple
import pickle
import os
from openai import OpenAI
from config import ark_api_key, ark_base_url, ark_model, ark_embedding_model, ark_embedding_dim, openai_api_key, deepseek_api_key, qwen_api_key, http_proxy, https_proxy, use_model

os.environ["http_proxy"] = http_proxy
os.environ["https_proxy"] = https_proxy


def parse_ark_response(response):
    if hasattr(response, 'output_text') and response.output_text:
        return response.output_text
    if hasattr(response, 'output') and isinstance(response.output, list):
        for item in response.output:
            if hasattr(item, 'type') and item.type == 'output_text':
                if hasattr(item, 'text'):
                    return item.text
                if hasattr(item, 'content'):
                    return item.content
    return str(response)


def handle_stream_response(client, model, messages, extra_body=None):
    if extra_body is None:
        extra_body = {}

    if use_model == "ark":
        response = client.responses.create(
            model=ark_model,
            input=messages,
            extra_body=extra_body
        )
        return parse_ark_response(response)
    else:
        if use_model == "qwen3-235b-a22b":
            if use_model.startswith("qwen"):
                extra_body["enable_thinking"] = False
            response_stream = client.chat.completions.create(
                model=use_model,
                messages=messages,
                stream=True,
                extra_body=extra_body
            )
            response_content = ""
            for chunk in response_stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    response_content += chunk.choices[0].delta.content
            return response_content
        else:
            response = client.chat.completions.create(
                model=use_model,
                messages=messages,
                extra_body=extra_body
            )
            return response.choices[0].message.content


def get_ark_client():
    return OpenAI(
        api_key=ark_api_key,
        base_url=ark_base_url
    )


def get_client():
    if use_model == "gpt-4o-mini":
        return OpenAI(
            api_key=openai_api_key,
            base_url="https://api.openai.com/v1"
        )
    elif use_model == "deepseek-chat":
        return OpenAI(
            api_key=deepseek_api_key,
            base_url="https://api.deepseek.com/v1"
        )
    elif use_model == "qwen3-235b-a22b":
        return OpenAI(
            api_key=qwen_api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    elif use_model == "ark":
        return get_ark_client()
    return OpenAI(
        api_key=openai_api_key,
        base_url="https://api.openai.com/v1"
    )


class Actor:
    def __init__(self, name, age, gender, memory_path=None):
        self.name = name
        self.age = age
        self.gender = gender
        self.embedding_dim = ark_embedding_dim if use_model == "ark" else 1536
        self.memories = []
        self.memory_embeddings = None
        self.index = None
        self.relationships = {}
        self.traits = []

        if use_model == "ark":
            self.talk_client = get_ark_client()
            self.embedding_client = get_ark_client()
        else:
            self.embedding_client = OpenAI(
                api_key=openai_api_key,
                base_url="https://api.openai.com/v1"
            )
            self.talk_client = get_client()

        self._initialize_memory(memory_path)

    def _initialize_memory(self, memory_path):
        if memory_path and os.path.exists(memory_path):
            with open(memory_path, 'rb') as f:
                saved_data = pickle.load(f)
                self.memories = saved_data.get('memories', [])
                self.memory_embeddings = saved_data.get('embeddings')
                self.relationships = saved_data.get('relationships', {})
                self.traits = saved_data.get('traits', [])

        if self.memory_embeddings is None or len(self.memories) == 0:
            self.memory_embeddings = np.zeros((0, self.embedding_dim), dtype=np.float32)

        self.index = faiss.IndexFlatL2(self.embedding_dim)
        if len(self.memories) > 0:
            self.index.add(self.memory_embeddings)

    def __str__(self):
        return f"Actor(name={self.name}, age={self.age}, gender={self.gender})"

    def add_memory(self, memory_text: str):
        embedding = self._get_embedding(memory_text)
        self.memories.append(memory_text)
        if len(self.memories) == 1:
            self.memory_embeddings = embedding.reshape(1, -1)
        else:
            self.memory_embeddings = np.vstack([self.memory_embeddings, embedding])

        self.index.reset()
        self.index.add(self.memory_embeddings)
        return len(self.memories) - 1

    def _get_embedding(self, text: str) -> np.ndarray:
        if use_model == "ark":
            try:
                response = self.embedding_client.embeddings.create(
                    model=ark_embedding_model,
                    input=text
                )
                return np.array(response.data[0].embedding, dtype=np.float32)
            except Exception as e:
                print(f"火山方舟Embedding API调用失败: {str(e)}")
                np.random.seed(hash(text) % 4294967295)
                return np.random.rand(self.embedding_dim).astype(np.float32)
        else:
            try:
                response = self.embedding_client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=text
                )
                return np.array(response.data[0].embedding, dtype=np.float32)
            except Exception as e:
                print(f"Embedding API调用失败，使用本地随机向量替代: {str(e)}")
                np.random.seed(hash(text) % 4294967295)
                return np.random.rand(self.embedding_dim).astype(np.float32)

    def retrieve_relevant_memories(self, query: str, k: int = 3) -> List[str]:
        if len(self.memories) == 0:
            return []

        query_embedding = self._get_embedding(query).reshape(1, -1)
        scores, indices = self.index.search(query_embedding, min(k, len(self.memories)))
        return [self.memories[idx] for idx in indices[0]]

    def add_trait(self, trait_description):
        if trait_description not in self.traits:
            self.traits.append(trait_description)
            self.add_memory(f"我的性格特点是{trait_description}")
        return len(self.traits) - 1

    def get_traits(self):
        return self.traits

    def save_memories(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'memories': self.memories,
                'embeddings': self.memory_embeddings,
                'relationships': self.relationships,
                'traits': self.traits
            }, f)

    def add_relationship(self, other_actor, relationship_type, description=""):
        other_name = other_actor.name if hasattr(other_actor, 'name') else other_actor

        if other_name not in self.relationships:
            self.relationships[other_name] = []

        for i, rel in enumerate(self.relationships[other_name]):
            if rel['type'] == relationship_type:
                self.relationships[other_name][i] = {
                    'type': relationship_type,
                    'description': description
                }
                return

        self.relationships[other_name].append({
            'type': relationship_type,
            'description': description
        })

        memory_text = f"我与{other_name}的关系是{relationship_type}"
        if description:
            memory_text += f"：{description}"
        self.add_memory(memory_text)

    def get_relationship(self, other_actor):
        other_name = other_actor.name if hasattr(other_actor, 'name') else other_actor
        return self.relationships.get(other_name, [])

    def get_all_relationships(self):
        return self.relationships

    def speak(self, information, speaker=None, guidance=None):
        relevant_memories = self.retrieve_relevant_memories(information)

        speaker_name = speaker
        if hasattr(speaker, 'name'):
            speaker_name = speaker.name

        relationship_context = ""
        if speaker_name and speaker_name in self.relationships:
            relationships = self.get_relationship(speaker_name)
            if relationships:
                rel_texts = []
                for rel in relationships:
                    rel_type = rel['type']
                    rel_desc = rel['description']
                    if rel_desc:
                        rel_text = f"你与{speaker_name}的关系是{rel_type}：{rel_desc}"
                    else:
                        rel_text = f"你与{speaker_name}的关系是{rel_type}"
                    rel_texts.append(rel_text)
                relationship_context += "\n".join(rel_texts) + "\n"

        for name in self.relationships.keys():
            if name != speaker_name and name.lower() in information.lower():
                relationships = self.get_relationship(name)
                if relationships:
                    rel_texts = []
                    for rel in relationships:
                        rel_type = rel['type']
                        rel_desc = rel['description']
                        if rel_desc:
                            rel_text = f"你与{name}的关系是{rel_type}：{rel_desc}"
                        else:
                            rel_text = f"你与{name}的关系是{rel_type}"
                        rel_texts.append(rel_text)
                    relationship_context += "\n".join(rel_texts) + "\n"

        traits_context = ""
        if self.traits:
            traits_context = "你的性格特征是：" + "，".join(self.traits) + "。请根据这些性格特征来塑造你的回应。\n"

        context = ""
        if relevant_memories or relationship_context or speaker_name or traits_context:
            context = "根据以下信息回答：\n"
            if speaker_name:
                context += f"与你对话的人是：{speaker_name}\n\n"
            if relationship_context:
                context += relationship_context + "\n"
            if traits_context:
                context += traits_context + "\n"
            if relevant_memories:
                context += "\n".join(relevant_memories) + "\n"
            context += "\n问题："

        if guidance:
            context += "\n导演指导：" + guidance

        if speaker_name:
            self.add_memory(f"{speaker_name}对我说：{information}")

        messages = [
            {"role": "system", "content": f"你是一名戏剧演员，你在剧中扮演的角色叫{self.name}，一个{self.age}岁的{self.gender}。请根据对话者身份和你的角色特点做出合适的回应。\n\n你的回应必须按照以下格式：\"（{self.name}的表情与动作）对话内容\"。括号内必须包含你扮演角色的名字作为主语，清晰描述表情和动作，括号后直接跟对话内容。例如：\"（{self.name}紧张地握紧拳头）我不知道你在说什么。\"或\"（{self.name}微笑着点头）我很高兴见到你。\""},
            {"role": "user", "content": (context + information) if context else information}
        ]

        response_content = handle_stream_response(self.talk_client, use_model, messages)

        if not response_content.startswith(f"（{self.name}") and not response_content.startswith(f"({self.name}"):
            response_content = f"（{self.name}平静地说道）{response_content}"

        if speaker_name:
            self.add_memory(f"我对{speaker_name}说：{response_content}")

        return response_content

    def should_speak(self, context, speaker=None):
        relevant_memories = self.retrieve_relevant_memories(context)

        speaker_name = speaker
        if hasattr(speaker, 'name'):
            speaker_name = speaker.name

        prompt = f"情境：{context}\n\n"

        if speaker_name:
            prompt += f"说话者：{speaker_name}\n\n"

        if relevant_memories:
            prompt += "相关记忆：\n" + "\n".join(relevant_memories) + "\n\n"

        prompt += f"作为一个名叫{self.name}的角色，在这种情况下，我需要说话吗？请只回答'是'或'否'，并给出简短理由。"

        messages = [
            {"role": "system", "content": "你需要帮助角色判断在当前情境下是否应该说话。请只回答'是'或'否'，并给出简短理由。"},
            {"role": "user", "content": prompt}
        ]

        decision = handle_stream_response(self.talk_client, use_model, messages)

        should_speak = "是" in decision[:10] or "yes" in decision.lower()[:10]

        if should_speak:
            return self.speak(context, speaker)

        return None


class Director:
    def __init__(self):
        self.actors = {}
        self.script = {}
        self.current_scene = None

        self.client = get_client()

    def add_actor(self, actor):
        self.actors[actor.name] = actor

    def generate_actor_profile(self, character_name, scene_id, player_name):
        scene_desc = self.get_scene_description(scene_id)

        prompt = f"""根据以下场景描述和角色名称，生成一个完整的角色信息：

场景描述：{scene_desc}

角色名称：{character_name}

玩家角色名称：{player_name}

请生成包含以下内容的角色信息：
1. 角色年龄
2. 角色性别
3. 角色背景故事（3-5条）
4. 角色性格特征（2-3个）
5. 与玩家角色"{player_name}"的关系类型和描述

请以JSON格式返回结果：
{{
    "age": 数字,
    "gender": "性别",
    "background": ["记忆/背景1", "记忆/背景2", ...],
    "traits": ["性格特征1", "性格特征2", ...],
    "relationship": {{
        "type": "关系类型",
        "description": "关系描述"
    }}
}}"""

        messages = [
            {"role": "system", "content": "你是一位创造角色的AI助手，根据场景和角色名称生成符合情境的角色信息。"},
            {"role": "user", "content": prompt}
        ]

        response = handle_stream_response(self.client, use_model, messages)

        try:
            import json
            import re

            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
        except Exception as e:
            print(f"解析角色信息失败: {str(e)}")

        return {
            "age": 30,
            "gender": "未知",
            "background": [f"我是{character_name}，在当前场景中新出现的角色"],
            "traits": ["神秘"],
            "relationship": {
                "type": "陌生人",
                "description": "刚刚相遇"
            }
        }

    def ensure_all_characters_exist(self, scene_id, player_name):
        all_characters = self.get_scene_characters(scene_id=scene_id)

        for character_name in all_characters:
            if character_name == player_name:
                continue

            if character_name not in self.actors:
                print(f"\n发现新角色: {character_name}，正在使用AI生成角色信息...")

                profile = self.generate_actor_profile(character_name, scene_id, player_name)

                new_actor = Actor(character_name, profile.get("age", 30), profile.get("gender", "未知"))

                for memory in profile.get("background", [f"我是{character_name}，在当前场景中新出现的角色"]):
                    new_actor.add_memory(memory)

                for trait in profile.get("traits", ["神秘"]):
                    new_actor.add_trait(trait)

                relationship = profile.get("relationship", {"type": "陌生人", "description": "刚刚相遇"})
                new_actor.add_relationship(
                    player_name,
                    relationship.get("type", "陌生人"),
                    relationship.get("description", "刚刚相遇")
                )

                self.add_actor(new_actor)
                print(f"已创建角色: {character_name}，年龄: {profile.get('age', 30)}，性别: {profile.get('gender', '未知')}")
                print(f"性格特征: {', '.join(profile.get('traits', ['神秘']))}")
                print(f"与玩家关系: {relationship.get('type', '陌生人')} - {relationship.get('description', '刚刚相遇')}")

    def check_and_create_new_characters(self, scene_ids, current_scene_index, player_name):
        if current_scene_index < len(scene_ids):
            current_scene_id = scene_ids[current_scene_index]
            self.ensure_all_characters_exist(current_scene_id, player_name)

    def load_script(self, script_dict):
        self.script = script_dict

    def set_current_scene(self, scene_id):
        if scene_id in self.script:
            self.current_scene = scene_id
            return True
        return False

    def get_current_scene(self):
        return self.current_scene

    def get_scene_description(self, scene_id=None):
        scene = scene_id if scene_id else self.current_scene
        if scene in self.script:
            return self.script[scene].get("description", "")
        return ""

    def get_scene_characters(self, scene_id=None, player=None):
        scene = scene_id if scene_id else self.current_scene
        if scene not in self.script:
            return []

        characters = self.script[scene].get("characters", [])

        if player:
            player_name = player
            if hasattr(player, 'get_player_name'):
                player_name = player.get_player_name()
            elif hasattr(player, 'name'):
                player_name = player.name

            characters = [char for char in characters if char != player_name]

        return characters

    def guide_actor_from_player_speech(self, player_speech, actor_name):
        if self.current_scene is None or actor_name not in self.actors:
            return "无法指导：未设置当前场景或演员未找到"

        scene_info = self.script.get(self.current_scene, {})
        scene_desc = scene_info.get("description", "")

        dialogues = scene_info.get("dialogues", [])
        actor_dialogues = [d for d in dialogues if d.get("character") == actor_name]

        actor = self.actors.get(actor_name)
        actor_traits = actor.get_traits() if hasattr(actor, "get_traits") else []
        traits_text = "，".join(actor_traits) if actor_traits else "无特定性格特征"

        prompt = f"""作为戏剧导演，请根据以下信息为演员'{actor_name}'提供表演指导：

场景描述：{scene_desc}

演员的性格特征：{traits_text}

玩家刚才的发言："{player_speech}"

演员的台词：
"""
        for dialogue in actor_dialogues:
            prompt += f"- {dialogue.get('content')}\n"

        prompt += """
请分析玩家发言的意图、情感和可能的隐含含义，然后提供具体的表演指导，包括：
1. 情感表达建议
2. 肢体语言指导
3. 语调和节奏建议
4. 是否应该透露某些信息
5. 如何忠于角色性格特征

请直接给出指导内容，无需前置说明："""

        messages = [
            {"role": "system", "content": "你是一位经验丰富的戏剧导演，擅长分析玩家发言并为演员提供回应指导。"},
            {"role": "user", "content": prompt}
        ]

        return handle_stream_response(self.client, use_model, messages)

    def is_scene_continuing(self, last_dialogue, screenwriter=None, detailed_scene=None):
        if self.current_scene is None:
            return False

        scene_info = self.script.get(self.current_scene, {})
        dialogues = scene_info.get("dialogues", [])

        if not dialogues:
            return False

        player_goal = ""
        scene_description = ""

        if detailed_scene:
            scene_description = detailed_scene
        elif screenwriter and hasattr(screenwriter, 'scene_descriptions') and self.current_scene in screenwriter.scene_descriptions:
            scene_description = screenwriter.scene_descriptions.get(self.current_scene, "")

        if scene_description:
            import re
            goal_patterns = [
                r"玩家角色目标描述[：:](.*?)(?=\n\n|\Z)",
                r"玩家角色目标[：:](.*?)(?=\n\n|\Z)",
                r"玩家目标[：:](.*?)(?=\n\n|\Z)",
                r"4[\.。]\s*玩家角色目标[^：:]*[：:](.*?)(?=\n\n|\Z)"
            ]

            for pattern in goal_patterns:
                goal_match = re.search(pattern, scene_description, re.DOTALL)
                if goal_match:
                    player_goal = goal_match.group(1).strip()
                    break

        recent_dialogue_history = ""
        if screenwriter and hasattr(screenwriter, 'get_dialogue_history'):
            recent_dialogues = screenwriter.get_dialogue_history(limit=5)
            if recent_dialogues:
                recent_dialogue_history = "\n\n最近的对话历史：\n"
                for d in recent_dialogues:
                    if 'record_type' in d:
                        record_type = d['record_type']
                        if record_type == "对话" or record_type.startswith("对"):
                            if 'target' in d:
                                recent_dialogue_history += f"{d['speaker']} 对 {d['target']} 说：{d['content']}\n"
                            else:
                                recent_dialogue_history += f"{d['speaker']} ({record_type})：{d['content']}\n"
                        else:
                            recent_dialogue_history += f"{d['speaker']} ({record_type})：{d['content']}\n"
                    else:
                        recent_dialogue_history += f"{d['speaker']}：{d['content']}\n"

        prompt = f"请分析以下情况，判断剧本是否应该继续在当前场景：\n\n"
        prompt += f"场景描述：{scene_info.get('description', '')}\n\n"

        if player_goal:
            prompt += f"玩家角色在本场景的目标：{player_goal}\n\n"

        prompt += "预期对话内容:\n"
        for d in dialogues[-3:]:
            prompt += f"{d.get('character')}: {d.get('content')}\n"

        if recent_dialogue_history:
            prompt += recent_dialogue_history
        elif last_dialogue:
            prompt += f"\n实际最新对话：{last_dialogue}\n"

        prompt += "\n根据以下判断标准，确定当前场景是否应该继续：\n"
        prompt += "1. 玩家角色的目标是否已经达成\n"
        prompt += "2. 场景中的关键对话是否已经完成\n"
        prompt += "3. 对话是否自然到达了一个结束点\n"
        prompt += "4. 是否出现了明显的场景转换线索\n\n"
        prompt += "特别注意：玩家角色的目标是判断场景是否应该继续的重要条件，如果目标尚未达成且无明显转场信号，场景通常应该继续。\n\n"
        prompt += "请明确判断场景是否应该继续，先回答是或否，然后给出简短理由。"

        messages = [
            {"role": "system", "content": "你是一位经验丰富的戏剧导演，擅长分析剧本和表演。你需要判断当前场景是否应该继续，还是应该转入下一个场景。"},
            {"role": "user", "content": prompt}
        ]

        result = handle_stream_response(self.client, use_model, messages).lower()

        return "是" in result[:30] or "应该继续" in result[:30] or "继续" in result[:30] or "未达成" in result[:50]

    def should_generate_new_script(self, screenwriter, current_scene_id, next_scene_id=None):
        if self.is_scene_continuing(None, screenwriter):
            return False

        recent_dialogues = screenwriter.get_dialogue_history(limit=10)
        dialogue_text = ""
        for d in recent_dialogues:
            if 'record_type' in d:
                record_type = d['record_type']
                if record_type == "对话" or record_type.startswith("对"):
                    dialogue_text += f"{d['speaker']} 对 {record_type} 说：{d['content']}\n"
                else:
                    dialogue_text += f"{d['speaker']} ({record_type})：{d['content']}\n"

        current_scene = self.script.get(current_scene_id, {})
        current_scene_desc = current_scene.get("description", "")

        next_scene_info = ""
        if next_scene_id and next_scene_id in self.script:
            next_scene = self.script.get(next_scene_id, {})
            next_scene_desc = next_scene.get("description", "")
            next_scene_info = f"""计划中的下一个场景描述：
{next_scene_desc}"""

        prompt = f"""作为戏剧导演，请判断是否需要在当前场景与下一个计划场景之间插入新场景。

当前场景描述：
{current_scene_desc}

最近的对话历史：
{dialogue_text}

{next_scene_info}

请分析以下几点：
1. 当前场景的情节是否已经自然结束
2. 对话历史中是否出现了需要立即处理的新情节线索
3. 当前场景与下一个计划场景之间是否存在情节或场景跨度过大的问题
4. 是否有未解决的冲突或未完成的情节需要在新场景中处理

基于以上分析，请判断是否需要插入新场景？请只回答"是"或"否"，然后给出简短理由。"""

        messages = [
            {"role": "system", "content": "你是一位经验丰富的戏剧导演，擅长分析剧情发展和场景转换。"},
            {"role": "user", "content": prompt}
        ]

        result = handle_stream_response(self.client, use_model, messages)

        return "是" in result[:10] or "需要" in result[:20] or "应该" in result[:20]


class Player:
    def __init__(self, name, age, gender):
        self.name = name
        self.age = age
        self.gender = gender

    def talk_to_actor(self, actor, message, guidance=None):
        if not hasattr(actor, 'speak'):
            return f"错误：无法与此对象对话。"

        response = actor.speak(message, self.name, guidance)
        return response

    def interact_with_environment(self, screenwriter, action, current_scene_id=None):
        screenwriter.add_dialogue_record(self.name, "环境互动", action)

        if current_scene_id:
            updated_scene = screenwriter.transform_scene(
                current_scene_id,
                action
            )

            self.current_scene = updated_scene

        return updated_scene

    def get_player_name(self):
        return self.name


class Screenwriter:
    def __init__(self):
        self.client = get_client()

        self.dialogue_history = []
        self.scene_descriptions = {}
        self.initial_script = {}

    def load_initial_script(self, script_dict):
        self.initial_script = script_dict
        for scene_id, scene_data in script_dict.items():
            if "description" in scene_data:
                self.scene_descriptions[scene_id] = scene_data["description"]

    def generate_scene_description(self, scene_id, director=None, player_character=None):
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

            if director and hasattr(director, 'actors'):
                for char_name in scene_characters:
                    if player_character and char_name == player_character:
                        continue

                    if char_name in director.actors:
                        actor = director.actors[char_name]
                        char_info = f"- {char_name}：{actor.age}岁，{actor.gender}\n"

                        if hasattr(actor, 'get_traits') and actor.get_traits():
                            traits = actor.get_traits()
                            char_info += f"  性格特征：{', '.join(traits)}\n"

                        if hasattr(actor, 'get_all_relationships'):
                            relationships = actor.get_all_relationships()
                            if relationships:
                                char_info += "  与其他角色的关系：\n"
                                for other_name, rel_list in relationships.items():
                                    if other_name in scene_characters:
                                        for rel in rel_list:
                                            rel_type = rel.get('type', '')
                                            rel_desc = rel.get('description', '')
                                            if rel_desc:
                                                char_info += f"    - 与{other_name}：{rel_type}（{rel_desc}）\n"
                                            else:
                                                char_info += f"    - 与{other_name}：{rel_type}\n"

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

        messages = [
            {"role": "system", "content": "你是一位经验丰富的戏剧编剧，擅长创造生动详实的场景描述。"},
            {"role": "user", "content": prompt}
        ]

        detailed_description = handle_stream_response(self.client, use_model, messages)

        self.scene_descriptions[scene_id] = detailed_description

        return detailed_description

    def add_dialogue_record(self, speaker, record_type, content, target=None):
        record = {
            "time": len(self.dialogue_history),
            "speaker": speaker,
            "record_type": record_type,
            "content": content
        }

        if target and (record_type == "对话" or record_type.startswith("对")):
            record["target"] = target

        self.dialogue_history.append(record)

    def get_dialogue_history(self, limit=10):
        return self.dialogue_history[-limit:] if self.dialogue_history else []

    def get_all_dialogue_history(self):
        return self.dialogue_history if self.dialogue_history else []

    def generate_new_script(self, current_scene_id, player_feedback=None, max_retries=3, dialogue_history=None):
        try:
            import re
            scene_num_match = re.search(r'(\d+)', current_scene_id)
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
        if dialogue_history:
            for d in dialogue_history:
                if isinstance(d, dict):
                    speaker = d.get('speaker', '')
                    record_type = d.get('record_type', '')
                    content = d.get('content', '')
                    if record_type == "对话" or record_type.startswith("对"):
                        if 'target' in d:
                            dialogue_text += f"{speaker} 对 {d.get('target')} 说：{content}\n"
                        else:
                            dialogue_text += f"{speaker} ({record_type})：{content}\n"
                    else:
                        dialogue_text += f"{speaker} ({record_type})：{content}\n"

        else:
            recent_dialogues = self.get_dialogue_history(limit=20)
            for d in recent_dialogues:
                if 'record_type' in d:
                    record_type = d.get('record_type', '')
                    if record_type == "对话" or record_type.startswith("对"):
                        if 'target' in d:
                            dialogue_text += f"{d['speaker']} 对 {d.get('target')} 说：{d['content']}\n"
                        else:
                            dialogue_text += f"{d['speaker']} ({record_type})：{d['content']}\n"
                    else:
                        dialogue_text += f"{d['speaker']} ({record_type})：{d['content']}\n"
                else:
                    dialogue_text += f"{d['speaker']}：{d['content']}\n"

        for attempt in range(max_retries):
            prompt = f"""作为编剧，请根据以下信息生成剧本的下一个场景：

当前场景：{scene_desc}

当前场景中的角色：{', '.join(current_characters)}

对话历史：
{dialogue_text}

"""

            if player_feedback:
                prompt += f"""
玩家反馈：
{player_feedback}

"""

            prompt += """请根据上述信息，创作剧本的下一个场景。特别注意：
1. 新场景必须与对话历史保持一致，不能出现矛盾
2. 角色的行为和对话必须符合其性格特点
3. 场景转换要自然，情节发展要合理

你必须严格按照以下JSON格式返回，这是系统必须的格式：
{
    "description": "详细的场景描述",
    "characters": ["角色名1", "角色名2", ...],
    "dialogues": [
        {"character": "角色名1", "content": "（角色表情与动作）对话内容1"},
        {"character": "角色名2", "content": "（角色表情与动作）对话内容2"},
        ...
    ]
}

注意：
1. 只返回一个场景的内容
2. JSON格式必须完全正确，不要有任何解释或额外文本
3. 对话内容应该符合角色特点并推动情节发展
4. 不要生成scene_id字段，系统会自动处理
5. 确保新场景与之前的对话和情节保持连贯性"""

            messages = [
                {"role": "system", "content": "你是一位经验丰富的编剧，专门为互动式戏剧创作剧本。你必须按照要求的JSON格式返回结果，并确保新场景与历史对话和情节保持一致性。"},
                {"role": "user", "content": prompt}
            ]

            generated_text = handle_stream_response(self.client, use_model, messages)

            try:
                import json
                import re

                json_match = re.search(r'\{[\s\S]*\}', generated_text)
                if json_match:
                    generated_text = json_match.group(0)

                new_scene = json.loads(generated_text)

                if not all(key in new_scene for key in ["description", "dialogues"]):
                    raise ValueError("生成的JSON缺少必要字段")

                self.initial_script[next_scene_id] = {
                    "description": new_scene["description"],
                    "characters": new_scene["characters"],
                    "dialogues": new_scene["dialogues"]
                }

                self.scene_descriptions[next_scene_id] = new_scene["description"]

                return {next_scene_id: self.initial_script[next_scene_id]}

            except Exception as e:
                if attempt < max_retries - 1:
                    continue
                else:
                    return {"error": str(e), "generated_text": generated_text, "next_scene_id": next_scene_id}

    def generate_actor_response_suggestions(self, actor_name, player_action):
        recent_dialogues = self.get_dialogue_history(limit=5)
        dialogue_text = ""
        for d in recent_dialogues:
            if 'record_type' in d:
                record_type = d['record_type']
                if record_type == "对话" or record_type.startswith("对"):
                    if 'target' in d:
                        dialogue_text += f"{d['speaker']} 对 {d['target']} 说：{d['content']}\n"
                    else:
                        dialogue_text += f"{d['speaker']} ({record_type})：{d['content']}\n"
                else:
                    dialogue_text += f"{d['speaker']} ({record_type})：{d['content']}\n"
            else:
                dialogue_text += f"{d['speaker']}：{d['content']}\n"

        prompt = f"""请根据以下上下文为演员'{actor_name}'提供3-5个可能的回应：

玩家动作：{player_action}

最近对话历史：
{dialogue_text}

请提供符合角色性格和当前情境的回应建议。每个建议应包括：
1. 角色的表情和动作
2. 对话内容

每个建议格式为："（表情和动作）对话内容"

不要包含任何解释或额外文本。"""

        messages = [
            {"role": "system", "content": "你是一位经验丰富的编剧，专门为互动式戏剧创作演员回应。"},
            {"role": "user", "content": prompt}
        ]

        response = handle_stream_response(self.client, use_model, messages)

        suggestions = []
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line and ('(' in line and ')' in line):
                suggestions.append(line)

        return suggestions

    def transform_scene(self, scene_id, player_action):
        current_scene = self.scene_descriptions.get(scene_id, "")

        prompt = f"""请描述玩家动作后场景如何变化。

当前场景描述：
{current_scene}

玩家动作：{player_action}

请提供场景变化的详细描述，包括：
1. 环境变化
2. 物品位置或状态的变化
3. 角色反应或行为的变化
4. 出现的任何新元素

描述要简洁但生动。"""

        messages = [
            {"role": "system", "content": "你是一位经验丰富的编剧，专门描述场景变化。"},
            {"role": "user", "content": prompt}
        ]

        response = handle_stream_response(self.client, use_model, messages)

        self.scene_descriptions[scene_id] = response

        return response

    def end_scene(self, last_interaction, director, current_scene_id, next_scene_id):
        current_scene = director.script.get(current_scene_id, {})
        current_scene_desc = current_scene.get("description", "")

        next_scene_desc = ""
        if next_scene_id:
            next_scene = director.script.get(next_scene_id, {})
            next_scene_desc = next_scene.get("description", "")

        prompt = f"""请根据以下信息提供场景转场描述：

当前场景描述：
{current_scene_desc}

最后一次互动：
{last_interaction}

下一个场景描述（如果有）：
{next_scene_desc}

请提供一个自然流畅的转场描述，将当前场景连接到下一个场景（如果没有下一个场景，则连接到故事的结尾）。"""

        messages = [
            {"role": "system", "content": "你是一位经验丰富的编剧，专门创作场景转场。"},
            {"role": "user", "content": prompt}
        ]

        return handle_stream_response(self.client, use_model, messages)
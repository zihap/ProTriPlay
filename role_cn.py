import openai
import faiss
import numpy as np
from typing import List, Dict, Tuple
import pickle
import os
from openai import OpenAI
from config import openai_api_key, deepseek_api_key, qwen_api_key, http_proxy, https_proxy, use_model

# 设置代理
os.environ["http_proxy"] = http_proxy
os.environ["https_proxy"] = https_proxy

# 在文件顶部添加一个通用的流式处理助手函数
def handle_stream_response(client, model, messages, extra_body=None):
    """处理流式和非流式响应的通用助手函数
    
    参数:
        client: OpenAI客户端
        model: 模型名称
        messages: 消息列表
        extra_body: 额外的请求体参数
        
    返回:
        模型的响应文本
    """
    # 添加默认extra_body
    if extra_body is None:
        extra_body = {}
    
    # 对于qwen模型，添加enable_thinking参数
    if model.startswith("qwen"):
        extra_body["enable_thinking"] = False
    
    # 检查是否使用流式输出
    if model == "qwen3-235b-a22b":
        # 流式输出
        response_content = ""
        response_stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            extra_body=extra_body
        )
        
        # 收集流式响应
        for chunk in response_stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                response_content += chunk.choices[0].delta.content
        
        return response_content
    else:
        # 非流式输出
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            extra_body=extra_body
        )
        
        return response.choices[0].message.content

class Actor:
    def __init__(self, name, age, gender, memory_path=None):
        self.name = name
        self.age = age
        self.gender = gender
        self.embedding_dim = 1536  # OpenAI embedding维度
        self.memories = []
        self.memory_embeddings = None
        self.index = None
        self.relationships = {}  # 存储与其他Actor的关系
        self.traits = []  # 存储角色的性格特征
        
        # 创建两个不同的客户端
        self.embedding_client = OpenAI(
            api_key=openai_api_key,
            base_url="https://api.openai.com/v1"  # OpenAI的官方API端点
        )
        
        if use_model == "gpt-4o-mini":
            self.talk_client = OpenAI(
                api_key=openai_api_key,
                base_url="https://api.openai.com/v1"  # OpenAI的官方API端点
            )
        elif use_model == "deepseek-chat":
            self.talk_client = OpenAI(
                api_key=deepseek_api_key,
                base_url="https://api.deepseek.com/v1"  # Deepseek的API端点
            )
        elif use_model == "qwen3-235b-a22b":
            self.talk_client = OpenAI(
                api_key=qwen_api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # Qwen的API端点
            )
        
        self._initialize_memory(memory_path)
    
    def _initialize_memory(self, memory_path):
        """初始化记忆系统"""
        if memory_path and os.path.exists(memory_path):
            with open(memory_path, 'rb') as f:
                saved_data = pickle.load(f)
                self.memories = saved_data.get('memories', [])
                self.memory_embeddings = saved_data.get('embeddings')
                self.relationships = saved_data.get('relationships', {})
                self.traits = saved_data.get('traits', [])  # 加载性格特征
        
        if self.memory_embeddings is None or len(self.memories) == 0:
            self.memory_embeddings = np.zeros((0, self.embedding_dim), dtype=np.float32)
        
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        if len(self.memories) > 0:
            self.index.add(self.memory_embeddings)

    def __str__(self):
        return f"Actor(name={self.name}, age={self.age}, gender={self.gender})"
    
    def add_memory(self, memory_text: str):
        """添加新记忆到记忆库"""
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
        """获取文本的向量嵌入，使用OpenAI的API"""
        response = self.embedding_client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return np.array(response.data[0].embedding, dtype=np.float32)
    
    def retrieve_relevant_memories(self, query: str, k: int = 3) -> List[str]:
        """检索与查询相关的记忆"""
        if len(self.memories) == 0:
            return []
        
        query_embedding = self._get_embedding(query).reshape(1, -1)
        scores, indices = self.index.search(query_embedding, min(k, len(self.memories)))
        return [self.memories[idx] for idx in indices[0]]
    
    def add_trait(self, trait_description):
        """添加性格特征到角色中
        
        参数:
            trait_description: 性格特征的描述，例如"谨慎"、"冲动"、"善良"等
        """
        if trait_description not in self.traits:
            self.traits.append(trait_description)
            # 将性格特征也添加为记忆
            self.add_memory(f"我的性格特点是{trait_description}")
        return len(self.traits) - 1
    
    def get_traits(self):
        """获取角色的所有性格特征
        
        返回:
            性格特征列表
        """
        return self.traits
    
    def save_memories(self, path: str):
        """保存记忆、关系和性格特征到文件"""
        with open(path, 'wb') as f:
            pickle.dump({
                'memories': self.memories,
                'embeddings': self.memory_embeddings,
                'relationships': self.relationships,
                'traits': self.traits  # 保存性格特征
            }, f)
    
    def add_relationship(self, other_actor, relationship_type, description=""):
        """添加与其他Actor的关系
        
        参数:
            other_actor: 另一个Actor对象或Actor名称
            relationship_type: 关系类型，如"朋友"、"家人"、"同事"等
            description: 关系描述，可以提供更详细的关系信息
        """
        other_name = other_actor.name if hasattr(other_actor, 'name') else other_actor
        
        if other_name not in self.relationships:
            self.relationships[other_name] = []
            
        # 检查是否已存在相同类型的关系
        for i, rel in enumerate(self.relationships[other_name]):
            if rel['type'] == relationship_type:
                # 更新已有关系
                self.relationships[other_name][i] = {
                    'type': relationship_type,
                    'description': description
                }
                return
                
        # 添加新关系
        self.relationships[other_name].append({
            'type': relationship_type,
            'description': description
        })
        
        # 将关系添加为记忆
        memory_text = f"我与{other_name}的关系是{relationship_type}"
        if description:
            memory_text += f"：{description}"
        self.add_memory(memory_text)
    
    def get_relationship(self, other_actor):
        """获取与指定Actor的关系信息"""
        other_name = other_actor.name if hasattr(other_actor, 'name') else other_actor
        return self.relationships.get(other_name, [])
    
    def get_all_relationships(self):
        """获取所有关系信息"""
        return self.relationships
    
    def speak(self, information, speaker=None, guidance=None):
        """根据记忆和关系回答问题，使用Deepseek的API
        
        参数:
            information: 对话内容或问题
            speaker: 与角色交谈的人（可以是Player对象或字符串名称）
            guidance: 导演提供的表演指导（可选）
        """
        # 检索相关记忆
        relevant_memories = self.retrieve_relevant_memories(information)
        
        # 提取说话者信息
        speaker_name = speaker
        if hasattr(speaker, 'name'):
            speaker_name = speaker.name
        
        # 检查是否涉及关系查询
        relationship_context = ""
        # 检查与当前说话者的关系
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
        
        # 检查对话中提到的其他人物关系
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
        
        # 添加性格特征信息
        traits_context = ""
        if self.traits:
            traits_context = "你的性格特征是：" + "，".join(self.traits) + "。请根据这些性格特征来塑造你的回应。\n"
        
        # 构建上下文
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
        
        # 如果存在导演指导，则添加指导
        if guidance:
            context += "\n导演指导：" + guidance
        
        # 添加对话记录到记忆
        if speaker_name:
            self.add_memory(f"{speaker_name}对我说：{information}")

        # 构建消息列表
        messages = [
            {"role": "system", "content": f"你是一名戏剧演员，你在剧中扮演的角色叫{self.name}，一个{self.age}岁的{self.gender}。请根据对话者身份和你的角色特点做出合适的回应。\n\n你的回应必须按照以下格式：\"（{self.name}的表情与动作）对话内容\"。括号内必须包含你扮演角色的名字作为主语，清晰描述表情和动作，括号后直接跟对话内容。例如：\"（{self.name}紧张地握紧拳头）我不知道你在说什么。\"或\"（{self.name}微笑着点头）我很高兴见到你。\""},
            {"role": "user", "content": (context + information) if context else information}
        ]
        
        # 使用通用处理函数获取响应
        response_content = handle_stream_response(self.talk_client, use_model, messages)
        
        # 检查并修正响应格式
        if not response_content.startswith(f"（{self.name}") and not response_content.startswith(f"({self.name}"):
            # 如果响应不符合要求的格式，尝试添加默认格式
            response_content = f"（{self.name}平静地说道）{response_content}"
        
        # 添加回复记录到记忆
        if speaker_name:
            self.add_memory(f"我对{speaker_name}说：{response_content}")
            
        return response_content
    
    def should_speak(self, context, speaker=None):
        """判断Actor是否需要说话，如果需要则执行speak方法。（可选）
        
        参数:
            context: 当前对话或场景上下文
            speaker: 与角色交谈的人（可以是Player对象或字符串名称）
            
        返回:
            如果需要说话，返回speak方法的执行结果；否则返回None
        """
        # 检索相关记忆以辅助判断
        relevant_memories = self.retrieve_relevant_memories(context)
        
        # 提取说话者信息
        speaker_name = speaker
        if hasattr(speaker, 'name'):
            speaker_name = speaker.name
            
        # 构建提示内容，用于判断是否需要说话
        prompt = f"情境：{context}\n\n"
        
        if speaker_name:
            prompt += f"说话者：{speaker_name}\n\n"
            
        if relevant_memories:
            prompt += "相关记忆：\n" + "\n".join(relevant_memories) + "\n\n"
            
        prompt += f"作为一个名叫{self.name}的角色，在这种情况下，我需要说话吗？请只回答'是'或'否'，并给出简短理由。"
        
        # 构建消息列表
        messages = [
            {"role": "system", "content": "你需要帮助角色判断在当前情境下是否应该说话。请只回答'是'或'否'，并给出简短理由。"},
            {"role": "user", "content": prompt}
        ]
        
        # 使用通用处理函数获取响应
        decision = handle_stream_response(self.talk_client, use_model, messages)
        
        # 分析决策结果
        should_speak = "是" in decision[:10] or "yes" in decision.lower()[:10]
        
        # 如果应该说话，则执行speak方法
        if should_speak:
            return self.speak(context, speaker)
        
        return None

class Director:
    def __init__(self):
        """初始化导演"""
        self.actors = {}  # 使用字典存储演员，键为演员名称
        self.script = {}  # 存储剧本，按场景组织
        self.current_scene = None  # 当前场景
        
        if use_model == "gpt-4o-mini":
            self.client = OpenAI(
                api_key=openai_api_key,
                base_url="https://api.openai.com/v1"  # OpenAI的官方API端点
            )
        elif use_model == "deepseek-chat":
            self.client = OpenAI(
                api_key=deepseek_api_key,
                base_url="https://api.deepseek.com/v1"  # Deepseek的API端点
            )
        elif use_model == "qwen3-235b-a22b":
            self.client = OpenAI(
                api_key=qwen_api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # Qwen的API端点
            )
        
    def add_actor(self, actor):
        """添加演员到导演的管理中"""
        self.actors[actor.name] = actor
    
    def generate_actor_profile(self, character_name, scene_id, player_name):
        """使用AI生成角色的详细信息
        
        参数:
            character_name: 角色名称
            scene_id: 当前场景ID
            player_name: 玩家角色名称
            
        返回:
            包含角色信息的字典
        """
        # 获取当前场景描述
        scene_desc = self.get_scene_description(scene_id)
        
        # 构建提示
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

        # 构建消息列表
        messages = [
            {"role": "system", "content": "你是一位创造角色的AI助手，根据场景和角色名称生成符合情境的角色信息。"},
            {"role": "user", "content": prompt}
        ]
        
        # 使用通用处理函数获取响应
        response = handle_stream_response(self.client, use_model, messages)
        
        # 尝试解析JSON响应
        try:
            import json
            import re
            
            # 尝试从文本中提取JSON部分
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
        except Exception as e:
            print(f"解析角色信息失败: {str(e)}")
            
        # 如果解析失败，返回默认信息
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
        """检查场景中的角色是否都存在，如果不存在则创建
        
        参数:
            scene_id: 当前场景ID
            player_name: 玩家角色名称，用于排除玩家角色和建立关系
        """
        # 获取当前场景中的所有角色
        all_characters = self.get_scene_characters(scene_id=scene_id)
        
        # 检查每个角色是否在self.actors中存在
        for character_name in all_characters:
            # 跳过玩家角色
            if character_name == player_name:
                continue
                
            # 如果角色不存在，创建一个新的Actor实例
            if character_name not in self.actors:
                print(f"\n发现新角色: {character_name}，正在使用AI生成角色信息...")
                
                # 使用AI生成角色信息
                profile = self.generate_actor_profile(character_name, scene_id, player_name)
                
                # 创建Actor实例
                new_actor = Actor(character_name, profile.get("age", 30), profile.get("gender", "未知"))
                
                # 添加背景记忆
                for memory in profile.get("background", [f"我是{character_name}，在当前场景中新出现的角色"]):
                    new_actor.add_memory(memory)
                
                # 添加性格特征
                for trait in profile.get("traits", ["神秘"]):
                    new_actor.add_trait(trait)
                
                # 添加与玩家的关系
                relationship = profile.get("relationship", {"type": "陌生人", "description": "刚刚相遇"})
                new_actor.add_relationship(
                    player_name, 
                    relationship.get("type", "陌生人"), 
                    relationship.get("description", "刚刚相遇")
                )
                
                # 将新角色添加到导演管理中
                self.add_actor(new_actor)
                print(f"已创建角色: {character_name}，年龄: {profile.get('age', 30)}，性别: {profile.get('gender', '未知')}")
                print(f"性格特征: {', '.join(profile.get('traits', ['神秘']))}")
                print(f"与玩家关系: {relationship.get('type', '陌生人')} - {relationship.get('description', '刚刚相遇')}")
    
    def check_and_create_new_characters(self, scene_ids, current_scene_index, player_name):
        """检查并创建新场景中可能出现的新角色
        
        参数:
            scene_ids: 场景ID列表
            current_scene_index: 当前场景索引
            player_name: 玩家角色名称
        """
        if current_scene_index < len(scene_ids):
            current_scene_id = scene_ids[current_scene_index]
            self.ensure_all_characters_exist(current_scene_id, player_name)
        
    def load_script(self, script_dict):
        """加载剧本
        
        参数:
            script_dict: 包含剧本内容的字典，格式为:
            {
                "scene_1": {
                    "description": "场景描述",
                    "characters": ["角色名1", "角色名2", ...],
                    "dialogues": [
                        {"character": "角色名", "content": "对话内容"},
                        ...
                    ]
                },
                ...
            }
        """
        self.script = script_dict
        
    def set_current_scene(self, scene_id):
        """设置当前场景"""
        if scene_id in self.script:
            self.current_scene = scene_id
            return True
        return False
    
    def get_current_scene(self):
        """获取当前场景"""
        return self.current_scene
    
    def get_scene_description(self, scene_id=None):
        """获取指定场景的描述"""
        scene = scene_id if scene_id else self.current_scene
        if scene in self.script:
            return self.script[scene].get("description", "")
        return ""
    
    def get_scene_characters(self, scene_id=None, player=None):
        """获取指定场景的角色列表，不包括玩家角色
        
        参数:
            scene_id: 场景ID，如果为None则使用当前场景
            player: Player对象或玩家角色名称，将从返回结果中排除
            
        返回:
            不包含玩家角色的角色列表
        """
        scene = scene_id if scene_id else self.current_scene
        if scene not in self.script:
            return []
            
        characters = self.script[scene].get("characters", [])
        
        # 如果提供了player参数，从角色列表中排除玩家角色
        if player:
            player_name = player
            # 如果player是Player对象，获取其名称
            if hasattr(player, 'get_player_name'):
                player_name = player.get_player_name()
            elif hasattr(player, 'name'):
                player_name = player.name
                
            # 过滤掉玩家角色
            characters = [char for char in characters if char != player_name]
            
        return characters

    
    def guide_actor_from_player_speech(self, player_speech, actor_name):
        """直接从玩家发言生成演员表演指导，合并了generate_guidance_from_player_speech和guide_actor的功能
        
        参数:
            player_speech: 玩家的发言内容
            actor_name: 要指导的演员名称
            
        返回:
            演员表演指导
        """
        if self.current_scene is None or actor_name not in self.actors:
            return "无法指导：未设置当前场景或演员未找到"
        
        # 获取当前场景信息
        scene_info = self.script.get(self.current_scene, {})
        scene_desc = scene_info.get("description", "")
        
        # 获取该演员在当前场景中的对话
        dialogues = scene_info.get("dialogues", [])
        actor_dialogues = [d for d in dialogues if d.get("character") == actor_name]
        
        # 获取角色信息
        actor = self.actors.get(actor_name)
        actor_traits = actor.get_traits() if hasattr(actor, "get_traits") else []
        traits_text = "，".join(actor_traits) if actor_traits else "无特定性格特征"
        
        # 构建提示信息
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
        
        # 构建消息列表
        messages = [
            {"role": "system", "content": "你是一位经验丰富的戏剧导演，擅长分析玩家发言并为演员提供回应指导。"},
            {"role": "user", "content": prompt}
        ]
        
        # 使用通用处理函数获取响应
        return handle_stream_response(self.client, use_model, messages)
    
    
    def is_scene_continuing(self, last_dialogue, screenwriter=None, detailed_scene=None):
        """判断剧本是否还处于当前场景
        
        参数:
            last_dialogue: 最近的对话内容（可选，如果提供了screenwriter则优先使用screenwriter的对话历史）
            screenwriter: Screenwriter对象，用于获取对话历史（可选）
            detailed_scene: 详细场景描述（可选，优先使用）
            
        返回:
            布尔值，表示是否继续在当前场景
        """
        if self.current_scene is None:
            return False
        
        scene_info = self.script.get(self.current_scene, {})
        dialogues = scene_info.get("dialogues", [])
        
        # 检查是否所有对话都已完成
        if not dialogues:
            return False
        
        # 获取玩家角色目标描述
        player_goal = ""
        scene_description = ""
        
        # 优先使用传入的详细场景描述
        if detailed_scene:
            scene_description = detailed_scene
        # 如果没有提供详细场景描述，尝试从screenwriter获取
        elif screenwriter and hasattr(screenwriter, 'scene_descriptions') and self.current_scene in screenwriter.scene_descriptions:
            scene_description = screenwriter.scene_descriptions.get(self.current_scene, "")
        
        # 提取玩家角色目标描述
        if scene_description:
            # 典型格式是包含"玩家角色目标"或"玩家目标"的段落
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
        
        # 使用Screenwriter的对话历史（如果提供）
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
        
        # 构建提示
        prompt = f"请分析以下情况，判断剧本是否应该继续在当前场景：\n\n"
        prompt += f"场景描述：{scene_info.get('description', '')}\n\n"
        
        # 添加玩家目标信息（如果有）
        if player_goal:
            prompt += f"玩家角色在本场景的目标：{player_goal}\n\n"
            
        prompt += "预期对话内容:\n"
        for d in dialogues[-3:]:  # 仅使用最近几条对话作为上下文
            prompt += f"{d.get('character')}: {d.get('content')}\n"
        
        # 添加对话历史或最新对话
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
        
        # 构建消息列表
        messages = [
            {"role": "system", "content": "你是一位经验丰富的戏剧导演，擅长分析剧本和表演。你需要判断当前场景是否应该继续，还是应该转入下一个场景。"},
            {"role": "user", "content": prompt}
        ]
        
        # 使用通用处理函数获取响应
        result = handle_stream_response(self.client, use_model, messages).lower()
        
        # 简单判断回答是肯定还是否定
        return "是" in result[:30] or "应该继续" in result[:30] or "继续" in result[:30] or "未达成" in result[:50]

    def should_generate_new_script(self, screenwriter, current_scene_id, next_scene_id=None):
        """判断是否需要生成新剧本/场景
        
        参数:
            screenwriter: Screenwriter对象，用于获取对话历史和场景信息
            current_scene_id: 当前场景ID
            next_scene_id: 下一个计划场景ID（如果有）
            
        返回:
            布尔值，表示是否需要生成新场景
        """
        # 首先检查当前场景是否已经结束
        if self.is_scene_continuing(None, screenwriter):
            return False  # 当前场景还在继续，不需要生成新场景
        
        # 获取近期对话历史
        recent_dialogues = screenwriter.get_dialogue_history(limit=10)
        dialogue_text = ""
        for d in recent_dialogues:
            if 'record_type' in d:
                record_type = d['record_type']
                if record_type == "对话" or record_type.startswith("对"):
                    dialogue_text += f"{d['speaker']} 对 {record_type} 说：{d['content']}\n"
                else:
                    dialogue_text += f"{d['speaker']} ({record_type})：{d['content']}\n"
        
        # 获取当前场景信息
        current_scene = self.script.get(current_scene_id, {})
        current_scene_desc = current_scene.get("description", "")
        
        # 获取下一个场景信息（如果有）
        next_scene_info = ""
        if next_scene_id and next_scene_id in self.script:
            next_scene = self.script.get(next_scene_id, {})
            next_scene_desc = next_scene.get("description", "")
            next_scene_info = f"""计划中的下一个场景描述：
{next_scene_desc}"""
        
        # 构建提示
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

        # 构建消息列表
        messages = [
            {"role": "system", "content": "你是一位经验丰富的戏剧导演，擅长分析剧情发展和场景转换。"},
            {"role": "user", "content": prompt}
        ]
        
        # 使用通用处理函数获取响应
        result = handle_stream_response(self.client, use_model, messages)
        
        # 判断回答是否建议生成新场景
        return "是" in result[:10] or "需要" in result[:20] or "应该" in result[:20]

class Player:
    def __init__(self, name, age, gender):
        """初始化玩家角色
        
        参数:
            name: 玩家扮演的角色名称
            age: 玩家扮演的角色年龄
            gender: 玩家扮演的角色性别
        """
        self.name = name
        self.age = age
        self.gender = gender

    
    def talk_to_actor(self, actor, message, guidance=None):
        """与场景中的演员对话
        
        参数:
            actor: Actor对象
            message: 玩家输入的对话内容
            guidance: 导演生成的指导
            
        返回:
            演员的回应
        """
        if not hasattr(actor, 'speak'):
            return f"错误：无法与此对象对话。"
        
        # 直接将玩家的消息传递给演员，由演员生成回应
        response = actor.speak(message, self.name, guidance)
        return response
    
    def interact_with_environment(self, screenwriter, action, current_scene_id=None):
        """与当前场景环境或物品交互
        
        参数:
            screenwriter: 编剧对象，用于处理玩家行为
            action: 玩家想要执行的动作
            current_scene_id: 当前场景ID（可选）
            
        返回:
            交互结果描述
        """
        # 记录交互内容到对话历史
        screenwriter.add_dialogue_record(self.name, "环境互动", action)
        
        # 如果提供了场景ID，更新场景描述
        if current_scene_id:
            # 根据玩家行为更新场景
            updated_scene = screenwriter.transform_scene(
                current_scene_id,
                action
            )
            
            # 设置玩家当前场景
            self.current_scene = updated_scene
        
        return updated_scene
    
    def get_player_name(self):
        """获取玩家名称
        
        返回:
            玩家名称
        """
        return self.name

class Screenwriter:
    def __init__(self):
        """初始化编剧"""
        # 创建OpenAI客户端
        if use_model == "gpt-4o-mini":
            self.client = OpenAI(
                api_key=openai_api_key,
                base_url="https://api.openai.com/v1"
            )
        elif use_model == "deepseek-chat":
            self.client = OpenAI(
                api_key=deepseek_api_key,
                base_url="https://api.deepseek.com/v1"
            )
        elif use_model == "qwen3-235b-a22b":
            self.client = OpenAI(
                api_key=qwen_api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
        
        self.dialogue_history = []  # 存储对话历史
        self.scene_descriptions = {}  # 存储场景描述
        self.initial_script = {}  # 存储初始剧本
    
    def load_initial_script(self, script_dict):
        """加载初始剧本
        
        参数:
            script_dict: 初始剧本字典
        """
        self.initial_script = script_dict
        # 初始化场景描述
        for scene_id, scene_data in script_dict.items():
            if "description" in scene_data:
                self.scene_descriptions[scene_id] = scene_data["description"]
    
    def generate_scene_description(self, scene_id, director=None, player_character=None):
        """生成场景的详细描述
        
        参数:
            scene_id: 场景ID
            director: Director对象，用于获取角色信息（可选）
            player_character: 玩家控制的角色名称（可选），不会被详细描述
            
        返回:
            包含场景、物品和非玩家人物的详细描述
        """
        # 检查是否已有场景基础描述
        base_description = self.scene_descriptions.get(scene_id)
        if not base_description:
            if scene_id in self.initial_script and "description" in self.initial_script[scene_id]:
                base_description = self.initial_script[scene_id]["description"]
            else:
                return f"错误：找不到场景 {scene_id} 的描述"
        
        # 获取场景中的角色列表
        scene_characters = []
        characters_info = ""
        
        if scene_id in self.initial_script and "characters" in self.initial_script[scene_id]:
            scene_characters = self.initial_script[scene_id]["characters"]
            
            # 如果提供了Director对象，获取更详细的角色信息
            if director and hasattr(director, 'actors'):
                for char_name in scene_characters:
                    # 跳过玩家控制的角色
                    if player_character and char_name == player_character:
                        continue
                        
                    # 获取角色信息
                    if char_name in director.actors:
                        actor = director.actors[char_name]
                        char_info = f"- {char_name}：{actor.age}岁，{actor.gender}\n"
                        
                        # 添加性格特征
                        if hasattr(actor, 'get_traits') and actor.get_traits():
                            traits = actor.get_traits()
                            char_info += f"  性格特征：{', '.join(traits)}\n"
                            
                        # 添加与其他角色的关系
                        if hasattr(actor, 'get_all_relationships'):
                            relationships = actor.get_all_relationships()
                            if relationships:
                                char_info += "  与其他角色的关系：\n"
                                for other_name, rel_list in relationships.items():
                                    if other_name in scene_characters:  # 只添加场景中存在的角色关系
                                        for rel in rel_list:
                                            rel_type = rel.get('type', '')
                                            rel_desc = rel.get('description', '')
                                            if rel_desc:
                                                char_info += f"    - 与{other_name}：{rel_type}（{rel_desc}）\n"
                                            else:
                                                char_info += f"    - 与{other_name}：{rel_type}\n"
                        
                        characters_info += char_info
        
        # 构建提示
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

        # 构建消息列表
        messages = [
            {"role": "system", "content": "你是一位经验丰富的戏剧编剧，擅长创造生动详实的场景描述。"},
            {"role": "user", "content": prompt}
        ]
        
        # 使用通用处理函数获取响应
        detailed_description = handle_stream_response(self.client, use_model, messages)
        
        # 更新场景描述
        self.scene_descriptions[scene_id] = detailed_description
        
        return detailed_description
    
    def add_dialogue_record(self, speaker, record_type, content, target=None):
        """添加戏剧记录
        
        参数:
            speaker: 说话者或行动者名称
            record_type: 记录类型（如"对话"、"旁白"、"场景描述"、"环境互动"等）
            content: 内容
            target: 对话的接收者（可选，仅在record_type为"对话"时有意义）
        """
        record = {
            "time": len(self.dialogue_history),
            "speaker": speaker,
            "record_type": record_type,
            "content": content
        }
        
        # 如果提供了target且record_type为对话相关类型，则记录对话目标
        if target and (record_type == "对话" or record_type.startswith("对")):
            record["target"] = target
            
        self.dialogue_history.append(record)
    
    def get_dialogue_history(self, limit=10):
        """获取最近的对话历史
        
        参数:
            limit: 返回的最大对话数量
            
        返回:
            最近的对话历史列表
        """
        return self.dialogue_history[-limit:] if self.dialogue_history else []
    
    def get_all_dialogue_history(self):
        """获取所有的对话历史
        
        参数:
            limit: 返回的最大对话数量
            
        返回:
            所有的对话历史列表
        """
        return self.dialogue_history if self.dialogue_history else []
    
    

    def generate_new_script(self, current_scene_id, player_feedback=None, max_retries=3, dialogue_history=None):
        """根据历史对话生成新的剧本
        
        参数:
            current_scene_id: 当前场景ID
            player_feedback: 玩家提供的反馈（可选）
            max_retries: 最大重试次数
            dialogue_history: 完整的对话历史记录（可选）
            
        返回:
            符合script_dict格式的新剧本部分
        """
        # 自动生成下一个场景ID
        try:
            # 尝试从当前场景ID中提取数字部分
            import re
            scene_num_match = re.search(r'(\d+)', current_scene_id)
            if scene_num_match:
                scene_num = int(scene_num_match.group(1))
                next_scene_id = current_scene_id.replace(str(scene_num), str(scene_num + 1))
            else:
                # 如果当前场景ID没有数字，则添加_1
                next_scene_id = f"{current_scene_id}_1"
        except:
            # 如果提取失败，使用时间戳作为备选
            import time
            next_scene_id = f"scene_{int(time.time())}"
        
        # 获取当前场景信息
        current_scene = self.initial_script.get(current_scene_id, {})
        scene_desc = self.scene_descriptions.get(current_scene_id, "")
        
        # 获取当前场景的角色列表
        current_characters = current_scene.get("characters", [])
        
        # 获取对话历史
        dialogue_text = ""
        if dialogue_history:
            # 使用传入的完整对话历史
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
            # 如果没有传入对话历史，则获取最近的对话
            recent_dialogues = self.get_dialogue_history(limit=20)  # 增加历史记录数量
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
        
        # 生成剧本，最多尝试max_retries次
        for attempt in range(max_retries):
            # 构建提示
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

            # 构建消息列表
            messages = [
                {"role": "system", "content": "你是一位经验丰富的编剧，专门为互动式戏剧创作剧本。你必须按照要求的JSON格式返回结果，并确保新场景与历史对话和情节保持一致性。"},
                {"role": "user", "content": prompt}
            ]
            
            # 使用通用处理函数获取响应
            generated_text = handle_stream_response(self.client, use_model, messages)
            
            # 尝试解析JSON
            try:
                import json
                import re
                
                # 尝试从文本中提取JSON部分
                json_match = re.search(r'\{[\s\S]*\}', generated_text)
                if json_match:
                    generated_text = json_match.group(0)
                    
                new_scene = json.loads(generated_text)
                
                # 验证JSON结构
                if not all(key in new_scene for key in ["description", "dialogues"]):
                    raise ValueError("生成的JSON缺少必要字段")
                
                # 构建完整的场景信息
                self.initial_script[next_scene_id] = {
                    "description": new_scene["description"],
                    "characters": new_scene["characters"],
                    "dialogues": new_scene["dialogues"]
                }
                
                # 更新场景描述
                self.scene_descriptions[next_scene_id] = new_scene["description"]
                
                return {next_scene_id: self.initial_script[next_scene_id]}
                
            except Exception as e:
                if attempt < max_retries - 1:
                    # 如果不是最后一次尝试，继续下一次
                    continue
                else:
                    # 所有尝试都失败，返回错误信息
                    return {"error": str(e), "generated_text": generated_text, "next_scene_id": next_scene_id}
    
    def generate_actor_response_suggestions(self, actor_name, player_action):
        """生成演员回应玩家行为的建议
        
        参数:
            actor_name: 演员名称
            player_action: 玩家行为
            
        返回:
            演员可能的回应建议列表
        """
        # 获取相关对话历史
        recent_dialogues = self.get_dialogue_history(limit=5)
        dialogue_text = ""
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
        
        # 构建提示
        prompt = f"""玩家刚才对角色 {actor_name} 执行了以下行为：
        {player_action}

        最近的对话历史：
        {dialogue_text}

        根据角色特性和上下文，生成3种不同风格的回应建议：
        1. 友好回应
        2. 中性回应
        3. 冷淡回应"""

        # 构建消息列表
        messages = [
            {"role": "system", "content": "你是一位编剧，擅长为角色创作恰当自然的对白。"},
            {"role": "user", "content": prompt}
        ]
        
        # 使用通用处理函数获取响应
        return handle_stream_response(self.client, use_model, messages)
    
    def transform_scene(self, scene_id, player_action=None):
        """根据玩家与场景中物品/环境的交互，生成交互后的场景描述
        
        参数:
            scene_id: 当前场景ID
            player_action: 玩家具体的交互行为（例如"把玻璃杯推下桌面"）
            
        返回:
            交互后更新的场景描述
        """
        # 获取原始场景描述
        original_desc = self.scene_descriptions.get(scene_id, "")
        if not original_desc:
            return "错误：找不到场景描述"
        
        # 构建提示
        prompt = f"""原始场景描述：
{original_desc}

玩家在场景中执行了以下交互：
"""
        
        if player_action:
            prompt += f"""
{player_action}
"""
        
        prompt += """
请详细描述玩家交互后的场景状态变化，并以JSON格式返回：
1. 交互的物品现在处于什么状态（如果适用）
2. 交互对场景环境造成了什么变化
3. 场景中的其他物品受到了什么影响
4. 场景中的人物（如果有）对这一交互有何反应

请以JSON格式返回场景描述，格式如下：
{
    "scene_description": "场景环境和氛围的变化详细描述",
    "interactions": [
        {
            "object": "交互物品名称",
            "state": "交互后的物品状态描述"
        }
    ],
    "character_reactions": [
        {
            "character": "角色名",
            "action": "角色动作描述",
            "dialogue": "角色对此交互的对话反应(如果有)"
        }
    ]
}

请确保：
1. JSON格式完全正确，可被解析
2. 场景描述要生动且符合故事情节发展
3. 保持与原场景的连续性，只描述因玩家交互而发生的合理变化
4. 角色反应要符合其性格特点"""

        # 构建消息列表
        messages = [
            {"role": "system", "content": "你是一位擅长描述场景变化的编剧。当玩家与场景中的物品或环境交互时，你需要生动描述交互后的场景状态变化，并以JSON格式返回结果。"},
            {"role": "user", "content": prompt}
        ]
        
        # 使用通用处理函数获取响应
        json_response = handle_stream_response(self.client, use_model, messages)
        
        # 解析JSON响应
        import json
        import re
        
        try:
            # 尝试从可能的文本中提取JSON部分
            json_match = re.search(r'\{[\s\S]*\}', json_response)
            if json_match:
                json_str = json_match.group(0)
                scene_data = json.loads(json_str)
                
                # 构建人类可读的场景描述
                formatted_description = ""
                
                if "scene_description" in scene_data:
                    formatted_description += scene_data["scene_description"] + "\n\n"
                
                # 添加物品交互信息
                if "interactions" in scene_data and scene_data["interactions"]:
                    formatted_description += "交互的物品变化：\n"
                    for interaction in scene_data["interactions"]:
                        object_name = interaction.get("object", "")
                        state = interaction.get("state", "")
                        if object_name and state:
                            formatted_description += f"- {object_name}：{state}\n"
                    formatted_description += "\n"
                
                # 添加角色反应
                if "character_reactions" in scene_data and scene_data["character_reactions"]:
                    formatted_description += "场景中人物的反应：\n"
                    for reaction in scene_data["character_reactions"]:
                        character = reaction.get("character", "")
                        action = reaction.get("action", "")
                        dialogue = reaction.get("dialogue", "")
                        
                        if character and (action or dialogue):
                            reaction_text = f"- {character}："
                            if action:
                                reaction_text += f"{action}"
                                # 记录角色反应到对话历史
                                self.add_dialogue_record(character, "动作", action)
                            
                            if dialogue:
                                if action:
                                    reaction_text += f'，说道："{dialogue}"'
                                else:
                                    reaction_text += f'"{dialogue}"'
                                # 记录角色对话到对话历史
                                self.add_dialogue_record(character, "对话", dialogue)
                            
                            formatted_description += reaction_text + "\n"
                
                # 更新场景描述
                import time
                # 使用时间戳生成唯一的场景ID
                new_scene_id = f"{scene_id}_{int(time.time())}"
                self.scene_descriptions[new_scene_id] = formatted_description
                
                # 记录这次交互到对话历史
                self.add_dialogue_record("系统", "场景", f"场景变化：{player_action}")
                
                return formatted_description
                
        except Exception as e:
            # 如果JSON解析失败，使用原始响应
            print(f"JSON解析失败: {str(e)}，使用原始文本")
            
            # 更新场景描述
            import time
            # 使用时间戳生成唯一的场景ID
            new_scene_id = f"{scene_id}_{int(time.time())}"
            self.scene_descriptions[new_scene_id] = json_response
            
            # 记录这次交互到对话历史
            self.add_dialogue_record("系统", "场景", f"场景变化：{player_action}")
            
            return json_response
    
    
    def end_scene(self, last_interaction, director, current_scene_id, next_scene_id=None, dialogue_history=None):
        """生成场景结束描述
        
        参数:
            last_interaction: 最后一次互动内容
            director: Director对象，用于获取场景描述
            current_scene_id: 当前场景ID
            next_scene_id: 下一个场景ID（可选）
            dialogue_history: 对话历史记录列表（可选）
            
        返回:
            场景结束描述文本
        """
        # 错误检查
        if not director or not hasattr(director, 'get_scene_description'):
            return "错误：无效的导演对象"
        
        scene_desc = director.get_scene_description(current_scene_id)
        if not scene_desc:
            return "错误：无法获取场景描述"
        
        # 获取当前场景的角色列表
        scene_characters = []
        if hasattr(director, 'get_scene_characters'):
            scene_characters = director.get_scene_characters(current_scene_id)
        
        # 获取下一个场景信息（如果有）
        next_scene_info = ""
        if next_scene_id and hasattr(director, 'get_scene_description'):
            next_scene_desc = director.get_scene_description(next_scene_id)
            if next_scene_desc:
                next_scene_info = "下一个场景:\n" + next_scene_desc + "\n\n请确保你的场景结束描述能够自然地过渡到下一个场景。考虑角色如何从当前场景移动到下一个场景，环境如何变化，以及情节如何发展。"
        
        # 构建角色信息
        characters_info = ""
        if scene_characters:
            characters_info = "当前场景中的角色: " + ", ".join(scene_characters)
        
        # 构建对话历史信息
        dialogue_history_info = ""
        if dialogue_history:
            dialogue_history_info = "\n\n对话历史记录:\n"
            for dialogue in dialogue_history:
                if isinstance(dialogue, dict):
                    speaker = dialogue.get('speaker', '')
                    record_type = dialogue.get('record_type', '')
                    content = dialogue.get('content', '')
                    if record_type == "对话" or record_type.startswith("对"):
                        if 'target' in dialogue:
                            dialogue_history_info += f"{speaker} 对 {dialogue.get('target')} 说：{content}\n"
                        else:
                            dialogue_history_info += f"{speaker} ({record_type})：{content}\n"
                    else:
                        dialogue_history_info += f"{speaker} ({record_type})：{content}\n"
        
        # 构建提示，用于生成场景结束描述
        prompt = "请根据以下最后一次互动内容和对话历史，生成一个符合戏剧剧本格式的场景结束描述，使故事能够自然过渡到下一个场景：\n\n"
        prompt += "注意！生成的内容不能前后矛盾！必须与对话历史保持一致！\n\n"
        prompt += "最后一次互动:\n" + last_interaction + "\n\n"
        prompt += "当前场景:\n" + scene_desc + "\n\n"
        
        if characters_info:
            prompt += characters_info + "\n\n"
            
        if dialogue_history_info:
            prompt += dialogue_history_info + "\n\n"
            
        if next_scene_info:
            prompt += next_scene_info + "\n\n"
        
        prompt += """请以JSON格式返回场景结束描述，格式如下：
{
    "scene_description": "场景环境和氛围的变化描述",
    "dialogues": [
        {
            "character": "角色名1",
            "action": "动作描述(可选)",
            "content": "对话内容"
        },
        {
            "character": "角色名2",
            "action": "动作描述(可选)",
            "content": "对话内容"
        }
    ],
    "transition": "自然过渡到下一个场景的引导描述"
}

请确保：
1. JSON格式完全正确，可被解析
2. 角色名必须是场景中实际存在的角色，或者"旁白"
3. 所有对话必须包含在dialogues数组中
4. 场景描述要生动且符合故事情节发展
5. 生成的内容必须与对话历史保持一致，不能出现矛盾"""

        # 使用handle_stream_response函数获取场景结束描述
        messages = [
            {"role": "system", "content": '你是一位经验丰富的戏剧编剧，擅长创作符合戏剧剧本格式的场景过渡。你需要返回JSON格式的响应，包含场景描述、角色对话和场景过渡。特别注意保持与对话历史的一致性。'},
            {"role": "user", "content": prompt}
        ]
        json_response = handle_stream_response(self.client, use_model, messages)
        
        # 解析JSON响应
        import json
        import re
        
        try:
            # 尝试从可能的文本中提取JSON部分
            json_match = re.search(r'\{[\s\S]*\}', json_response)
            if json_match:
                json_str = json_match.group(0)
                scene_data = json.loads(json_str)
                
                # 提取对话并记录
                if "dialogues" in scene_data and isinstance(scene_data["dialogues"], list):
                    for dialogue in scene_data["dialogues"]:
                        character = dialogue.get("character", "").strip()
                        content = dialogue.get("content", "").strip()
                        action = dialogue.get("action", "").strip()
                        
                        # 检查角色是否在场景中或是旁白
                        if character and (character in scene_characters or character == "旁白"):
                            # 记录对话，如果有动作描述，添加到内容中
                            full_content = content
                            if action:
                                full_content = f"({action}) {content}"
                            self.add_dialogue_record(character, "场景结束", full_content)
                
                # 构建人类可读的场景结束描述
                formatted_description = ""
                
                if "scene_description" in scene_data:
                    formatted_description += scene_data["scene_description"] + "\n\n"
                
                if "dialogues" in scene_data:
                    for dialogue in scene_data["dialogues"]:
                        character = dialogue.get("character", "")
                        content = dialogue.get("content", "")
                        action = dialogue.get("action", "")
                        
                        if action:
                            formatted_description += f"{character}：({action}) {content}\n"
                        else:
                            formatted_description += f"{character}：{content}\n"
                    
                    formatted_description += "\n"
                
                if "transition" in scene_data:
                    formatted_description += scene_data["transition"]
                
                return formatted_description
            
        except Exception as e:
            # 如果解析失败，返回原始响应并尝试基本解析
            print(f"JSON解析失败: {str(e)}，使用基本文本处理")
            
            # 基本文本处理作为后备方案
            lines = json_response.split('\n')
            for line in lines:
                if '：' in line:
                    parts = line.split('：', 1)
                    if len(parts) == 2:
                        character = parts[0].strip()
                        content = parts[1].strip()
                        
                        # 检查角色是否在场景中或是旁白
                        if character in scene_characters or character == "旁白":
                            # 记录对话
                            self.add_dialogue_record(character, "场景结束", content)
            
            return json_response
import tkinter as tk
from tkinter import scrolledtext, ttk, messagebox
import threading
import sys
import io

# API密钥配置
openai_api_key = "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"
deepseek_api_key = "sk-7ff0c7d5cfe243e18da3f4f86affa409"
qwen_api_key = "sk-c3cbded760ac4591aad781c640ccdc5d"

# 代理配置
http_proxy = "http://localhost:7890"
https_proxy = "http://localhost:7890"

# 模型配置
use_model = "deepseek-chat"  # 可选: "gpt-4o-mini", "deepseek-chat", "qwen3-235b-a22b" 

# 测试配置
try_chance = 2  # 场景循环次数/尝试机会
max_new_scene_generations = 1  # 最大允许生成的新场景数 
max_inserted_scenes = 1  # 最大允许插入的新场景数

import openai
import faiss
import numpy as np
from typing import List, Dict, Tuple
import pickle
import os
from openai import OpenAI

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
    # 获取全局模型设置
    global use_model
    model = use_model
    
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
        
        # 使用全局use_model变量
        global use_model
        
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

        # 使用全局模型变量
        global use_model
        
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
        
        # 使用全局模型变量
        global use_model
        
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
        
        # 使用全局use_model变量
        global use_model
        
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
        # 使用全局use_model变量
        global use_model
        
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

class StdoutRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.buffer = ""

    def write(self, string):
        self.buffer += string
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)

    def flush(self):
        pass

class SettingsDialog:
    def __init__(self, parent):
        self.parent = parent
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("游戏设置")
        self.dialog.geometry("500x450")  # 调整合适的大小
        self.dialog.configure(bg="#1e1e2e")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # 设置变量
        self.model_var = tk.StringVar(value=use_model)
        self.try_chance_var = tk.IntVar(value=try_chance)
        self.max_inserted_scenes_var = tk.IntVar(value=max_inserted_scenes)
        self.max_new_scene_generations_var = tk.IntVar(value=max_new_scene_generations)
        
        # 创建设置界面
        self.create_widgets()
        
        # 返回值初始化
        self.result = {
            "model": use_model,
            "try_chance": try_chance,
            "max_inserted_scenes": max_inserted_scenes,
            "max_new_scene_generations": max_new_scene_generations
        }
        
        # 确保对话框显示在父窗口中心
        self.dialog.update_idletasks()
        width = self.dialog.winfo_width()
        height = self.dialog.winfo_height()
        x = parent.winfo_rootx() + (parent.winfo_width() - width) // 2
        y = parent.winfo_rooty() + (parent.winfo_height() - height) // 2
        self.dialog.geometry(f'{width}x{height}+{x}+{y}')
        
        # 强制在初始化后刷新窗口
        self.dialog.update()
        
        # 设置窗口最小大小，防止内容被裁剪
        self.dialog.minsize(400, 300)
        
        # 阻止窗口被关闭时游戏继续（必须通过按钮关闭）
        self.dialog.protocol("WM_DELETE_WINDOW", self.on_cancel)
        
    def create_widgets(self):
        main_frame = tk.Frame(self.dialog, bg="#1e1e2e", padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 模型选择
        model_frame = tk.LabelFrame(main_frame, text="模型设置", bg="#2a2a3e", fg="#ffffff", 
                                  font=("微软雅黑", 12, "bold"), padx=10, pady=10)
        model_frame.pack(fill=tk.X, pady=5)
        
        models = [("OpenAI (gpt-4o-mini)", "gpt-4o-mini"), 
                 ("DeepSeek (deepseek-chat)", "deepseek-chat"), 
                 ("Qianwen (qwen3-235b-a22b)", "qwen3-235b-a22b")]
        
        for i, (text, value) in enumerate(models):
            radio = tk.Radiobutton(model_frame, text=text, value=value, variable=self.model_var,
                          bg="#2a2a3e", fg="#ffffff", selectcolor="#3d3d60", 
                          activebackground="#2a2a3e", activeforeground="#ffffff",
                          font=("微软雅黑", 10))
            radio.pack(anchor=tk.W, pady=2)
            # 确保按钮立即显示
            radio.update()
        
        # 场景设置
        scene_frame = tk.LabelFrame(main_frame, text="场景设置", bg="#2a2a3e", fg="#ffffff", 
                                  font=("微软雅黑", 12, "bold"), padx=10, pady=10)
        scene_frame.pack(fill=tk.X, pady=10)
        
        # 使用Grid布局以确保对齐
        scene_frame.grid_columnconfigure(0, weight=3)
        scene_frame.grid_columnconfigure(1, weight=1)
        
        # 尝试机会
        row = 0
        tk.Label(scene_frame, text="每场景尝试机会:", bg="#2a2a3e", fg="#ffffff", 
               font=("微软雅黑", 10)).grid(row=row, column=0, sticky=tk.W, pady=5)
        
        try_chance_spinbox = tk.Spinbox(scene_frame, from_=1, to=5, textvariable=self.try_chance_var,
                                     width=5, bg="#2d2d40", fg="#ffffff", font=("微软雅黑", 10))
        try_chance_spinbox.grid(row=row, column=1, sticky=tk.W, padx=10)
        
        # 最大插入场景数
        row += 1
        tk.Label(scene_frame, text="最大插入场景数:", bg="#2a2a3e", fg="#ffffff", 
               font=("微软雅黑", 10)).grid(row=row, column=0, sticky=tk.W, pady=5)
        
        max_inserted_spinbox = tk.Spinbox(scene_frame, from_=0, to=3, textvariable=self.max_inserted_scenes_var,
                                       width=5, bg="#2d2d40", fg="#ffffff", font=("微软雅黑", 10))
        max_inserted_spinbox.grid(row=row, column=1, sticky=tk.W, padx=10)
        
        # 最大新场景生成数
        row += 1
        tk.Label(scene_frame, text="最大新场景生成数:", bg="#2a2a3e", fg="#ffffff", 
               font=("微软雅黑", 10)).grid(row=row, column=0, sticky=tk.W, pady=5)
        
        max_new_spinbox = tk.Spinbox(scene_frame, from_=0, to=3, textvariable=self.max_new_scene_generations_var,
                                   width=5, bg="#2d2d40", fg="#ffffff", font=("微软雅黑", 10))
        max_new_spinbox.grid(row=row, column=1, sticky=tk.W, padx=10)
        
        # 按钮区域
        button_frame = tk.Frame(main_frame, bg="#1e1e2e", pady=10)
        button_frame.pack(fill=tk.X)
        
        # 添加确定按钮
        ok_button = tk.Button(button_frame, text="确定", command=self.on_ok, 
                bg="#3d3d60", fg="#ffffff", font=("微软雅黑", 10),
                width=10)
        ok_button.pack(side=tk.RIGHT, padx=5)
        
        # 添加取消按钮
        cancel_button = tk.Button(button_frame, text="取消", command=self.on_cancel,
                bg="#3d3d60", fg="#ffffff", font=("微软雅黑", 10),
                width=10)
        cancel_button.pack(side=tk.RIGHT, padx=5)
        
        # 强制更新所有小部件
        self.dialog.update()
    
    def on_ok(self):
        self.result = {
            "model": self.model_var.get(),
            "try_chance": self.try_chance_var.get(),
            "max_inserted_scenes": self.max_inserted_scenes_var.get(),
            "max_new_scene_generations": self.max_new_scene_generations_var.get()
        }
        self.dialog.destroy()
    
    def on_cancel(self):
        # 当取消时，将结果设置为None
        self.result = None
        self.dialog.destroy()
    
    def show(self):
        # 在主窗口中等待对话框
        self.parent.wait_window(self.dialog)
        return self.result

class CoCGameGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("克苏鲁的呼唤 - 互动游戏")
        self.root.geometry("1200x800")
        self.root.configure(bg="#1e1e2e")
        
        # 在游戏初始化前显示设置对话框
        self.show_settings_dialog()
        
        self.setup_variables()
        self.create_ui()
        self.initialize_game()
        
    def show_settings_dialog(self):
        # 显示设置对话框前，先确保根窗口已经完全初始化
        self.root.update_idletasks()
        
        # 创建并显示设置对话框
        dialog = SettingsDialog(self.root)
        settings = dialog.show()
        
        # 检查用户是否取消了设置
        if not settings:
            # 如果取消，使用默认值
            settings = {
                "model": use_model,
                "try_chance": try_chance,
                "max_inserted_scenes": max_inserted_scenes,
                "max_new_scene_generations": max_new_scene_generations
            }
            print("\n用户取消了设置，将使用默认配置")
        
        # 保存设置
        self.settings = settings
        
        # 打印当前配置
        print(f"\n===== 游戏配置 =====")
        print(f"使用模型: {settings['model']}")
        print(f"场景尝试机会: {settings['try_chance']}")
        print(f"最大插入场景数: {settings['max_inserted_scenes']}")
        print(f"最大新场景生成数: {settings['max_new_scene_generations']}")
        print(f"====================\n")
        
    def setup_variables(self):
        # 游戏状态变量
        self.current_scene_index = 0
        self.script_ids = []
        self.scene_finished = False
        self.should_exit_game = False
        self.last_interaction = ""
        self.inserted_scene_count = 0
        self.new_scene_generation_count = 0
        self.characters = []
        
        # 角色与导演
        self.player = None
        self.director = None
        self.screenwriter = None

    def create_ui(self):
        # 创建主框架
        main_frame = tk.Frame(self.root, bg="#1e1e2e")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 分割窗口
        top_frame = tk.Frame(main_frame, bg="#1e1e2e")
        top_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        bottom_frame = tk.Frame(main_frame, bg="#1e1e2e", height=200)
        bottom_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 场景和对话显示区
        left_frame = tk.Frame(top_frame, bg="#1e1e2e", width=800)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 场景描述区域
        scene_frame = tk.LabelFrame(left_frame, text="场景描述", bg="#2a2a3e", fg="#ffffff", font=("微软雅黑", 12, "bold"))
        scene_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.scene_text = scrolledtext.ScrolledText(scene_frame, wrap=tk.WORD, bg="#2d2d40", fg="#ffffff", font=("微软雅黑", 11))
        self.scene_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 角色和动作区
        right_frame = tk.Frame(top_frame, bg="#1e1e2e", width=300)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5, pady=5)
        
        # 角色选择区域
        character_frame = tk.LabelFrame(right_frame, text="角色", bg="#2a2a3e", fg="#ffffff", font=("微软雅黑", 12, "bold"))
        character_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.character_listbox = tk.Listbox(character_frame, bg="#2d2d40", fg="#ffffff", font=("微软雅黑", 11), selectbackground="#3d3d60")
        self.character_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 动作按钮区域
        action_frame = tk.LabelFrame(right_frame, text="动作", bg="#2a2a3e", fg="#ffffff", font=("微软雅黑", 12, "bold"))
        action_frame.pack(fill=tk.BOTH, padx=5, pady=5)
        
        # 对话按钮
        self.talk_btn = tk.Button(action_frame, text="与角色对话", bg="#3d3d60", fg="#ffffff", font=("微软雅黑", 11),
                               command=self.talk_to_character)
        self.talk_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # 互动按钮
        self.interact_btn = tk.Button(action_frame, text="与环境互动", bg="#3d3d60", fg="#ffffff", font=("微软雅黑", 11),
                                  command=self.interact_with_environment)
        self.interact_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # 下一场景按钮
        self.next_btn = tk.Button(action_frame, text="下一场景", bg="#3d3d60", fg="#ffffff", font=("微软雅黑", 11),
                               command=self.go_to_next_scene)
        self.next_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # 退出按钮
        self.exit_btn = tk.Button(action_frame, text="退出游戏", bg="#3d3d60", fg="#ffffff", font=("微软雅黑", 11),
                               command=self.exit_game)
        self.exit_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # 对话输入区域
        input_frame = tk.LabelFrame(bottom_frame, text="输入", bg="#2a2a3e", fg="#ffffff", font=("微软雅黑", 12, "bold"))
        input_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.input_text = tk.Text(input_frame, height=3, bg="#2d2d40", fg="#ffffff", font=("微软雅黑", 11))
        self.input_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 添加提示文本
        tip_label = tk.Label(input_frame, text="请输入内容后点击'与角色对话'或'与环境互动'按钮", 
                           bg="#2a2a3e", fg="#b0b0c0", font=("微软雅黑", 9, "italic"))
        tip_label.pack(side=tk.RIGHT, padx=5, pady=2)
        
        # 游戏日志区域
        log_frame = tk.LabelFrame(main_frame, text="游戏日志", bg="#2a2a3e", fg="#ffffff", font=("微软雅黑", 12, "bold"))
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, bg="#2d2d40", fg="#ffffff", font=("微软雅黑", 10))
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 重定向标准输出到日志区域
        self.stdout_redirector = StdoutRedirector(self.log_text)
        sys.stdout = self.stdout_redirector

    def initialize_game(self):
        # 启动游戏初始化线程，避免UI卡顿
        threading.Thread(target=self._initialize_game_thread, daemon=True).start()
    
    def _initialize_game_thread(self):
        # 修改全局变量以应用设置
        global use_model, try_chance, max_inserted_scenes, max_new_scene_generations
        
        # 保存当前设置值
        selected_model = self.settings['model']
        selected_try_chance = self.settings['try_chance']
        selected_max_inserted_scenes = self.settings['max_inserted_scenes']
        selected_max_new_scene_generations = self.settings['max_new_scene_generations']
        
        # 直接修改全局变量
        use_model = selected_model
        try_chance = selected_try_chance
        max_inserted_scenes = selected_max_inserted_scenes
        max_new_scene_generations = selected_max_new_scene_generations
        
        print(f"\n===== 确认游戏配置已生效 =====")
        print(f"使用模型: {use_model}")
        print(f"场景尝试机会: {try_chance}")
        print(f"最大插入场景数: {max_inserted_scenes}")
        print(f"最大新场景生成数: {max_new_scene_generations}")
        print(f"============================\n")
        
        # 创建一个函数以强制使用选定的模型设置
        def create_actor_with_model(name, age, gender):
            actor = Actor(name, age, gender)
            # 根据用户选择的模型重新设置talk_client
            if selected_model == "gpt-4o-mini":
                actor.talk_client = OpenAI(
                    api_key=openai_api_key,
                    base_url="https://api.openai.com/v1"
                )
            elif selected_model == "deepseek-chat":
                actor.talk_client = OpenAI(
                    api_key=deepseek_api_key,
                    base_url="https://api.deepseek.com/v1"
                )
            elif selected_model == "qwen3-235b-a22b":
                actor.talk_client = OpenAI(
                    api_key=qwen_api_key,
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
                )
            return actor
        
        # 初始化角色
        # 创建调查员(玩家角色)
        self.player = Player("霍华德", 25, "男")
        
        # 使用自定义函数创建NPC角色，确保使用正确的模型
        librarian = create_actor_with_model("玛莎·迪尔", 57, "女")
        librarian.add_memory("我是海港镇图书馆的管理员，已经工作了30年")
        librarian.add_memory("最近镇上发生了一些奇怪的事情，特别是靠近海边的居民")
        librarian.add_memory("我收藏了一些关于古老神话的禁忌书籍，包括《伊波恩之书》")
        librarian.add_relationship(self.player.name, "谨慎", "知道他是个调查员，但不确定能否信任他")
        librarian.add_trait("谨慎小心")
        librarian.add_trait("博学多识")
        librarian.add_trait("对神秘学知识有着复杂的好奇和恐惧")

        professor = create_actor_with_model("威廉·阿克雷", 68, "男")
        professor.add_memory("我是密斯卡托尼克大学的前教授，研究古代文明和神话")
        professor.add_memory("我见证了十年前的海港镇事件，那些不可名状的存在")
        professor.add_memory("我的理智已经受到了损害，但我仍然试图阻止即将到来的灾难")
        professor.add_relationship(self.player.name, "盟友", "认为他可能是阻止仪式的关键人物")
        professor.add_relationship("玛莎·迪尔", "同谋", "她帮我保存了一些关键的古籍")
        professor.add_trait("精神不稳定")
        professor.add_trait("睿智但偏执")
        professor.add_trait("勇敢但已伤痕累累")

        cultist = create_actor_with_model("约瑟夫·马什", 45, "男")
        cultist.add_memory("我是深潜者的后裔，忠于达贡和海德拉")
        cultist.add_memory("我表面上是普通渔民，实际负责监视镇上的外来者")
        cultist.add_memory("我知道即将到来的仪式，我的血脉让我可以呼唤海中的存在")
        cultist.add_relationship(self.player.name, "敌对", "怀疑他想干扰我们的仪式")
        cultist.add_relationship("威廉·阿克雷", "仇恨", "他知道太多了，必须被处理掉")
        cultist.add_trait("狂热")
        cultist.add_trait("双面性格")
        cultist.add_trait("残忍无情")

        # 创建导演
        self.director = Director()

        # 添加演员到导演管理中
        self.director.add_actor(librarian)
        self.director.add_actor(professor)
        self.director.add_actor(cultist)

        # 创建编剧
        self.screenwriter = Screenwriter()

        # 加载CoC风格的剧本
        coc_script = {
            "scene_1": {
                "description": "你是一位美国联邦调查员，来海港镇调查人员神秘失踪事件。现在你在阴郁潮湿的海港镇图书馆。窗外大雨倾盆，雷声轰鸣。图书馆内昏暗的灯光下，古老的书架排列整齐，空气中弥漫着霉味和古籍的气息。角落里的老式座钟滴答作响，偶尔发出不协调的声音。",
                "characters": ["霍华德", "玛莎·迪尔"],
                "dialogues": [
                    {"character": "玛莎·迪尔", "content": "(神情紧张地整理着书架)这几天镇上不太平，先生。你来这里是为了什么？"},
                ]
            },
            "scene_2": {
                "description": "图书馆的地下室。一个狭小、昏暗的空间，墙壁上挂着老式油灯。中间是一张大木桌，上面摊开着几本古老的书籍和手稿。空气更加浑浊，墙上的水渍形成了奇怪的图案。角落里有一个锁着的铁箱。",
                "characters": ["霍华德", "玛莎·迪尔"],
                "dialogues": [
                    {"character": "玛莎·迪尔", "content": "(声音压低)这些是我们不对外开放的藏书。有些知识...最好永远不要被发现。"},
                ]
            },
            "scene_3": {
                "description": "阿克雷教授的小屋。位于海港镇郊外的一座孤立小屋，周围是茂密的树林。屋内满是书籍、笔记和奇怪的收藏品。墙上挂着神秘的符号和地图。壁炉里的火焰投下摇曳的影子。一股淡淡的海水和药草混合的气味弥漫在空气中。",
                "characters": ["霍华德", "威廉·阿克雷","玛莎·迪尔"],
                "dialogues": [
                    {"character": "威廉·阿克雷", "content": "(手微微颤抖，眼神飘忽)你找到《伊波恩之书》了吗？时间不多了，'它们'即将苏醒...(突然压低声音)你被跟踪了，小心那些'渔民'，他们不是人类..."},
                ]
            },
            "scene_4": {
                "description": "海港镇海滩，夜晚。月光被厚重的云层遮挡，只有零星的星光照亮沙滩。海浪拍打着岸边，发出低沉的声响。远处礁石上似乎站着几个人影，正在进行某种仪式。空气中弥漫着浓重的咸味和一种无法描述的异味。",
                "characters": ["霍华德", "约瑟夫·马什","威廉·阿克雷"],
                "dialogues": [
                    {"character": "约瑟夫·马什", "content": "(站在祭坛旁边，双手举起一个奇怪的雕像)外来者，你不该来这里。这片海域属于伟大的存在，而我们即将得到祂们的祝福。"},
                ]
            }
        }

        # 获取剧本id列表
        self.script_ids = list(coc_script.keys())

        # 导演加载剧本
        self.director.load_script(coc_script) 

        # 编剧也加载相同的剧本
        self.screenwriter.load_initial_script(coc_script)

        # 设置当前场景
        self.current_scene_index = 0
        self.director.set_current_scene(self.script_ids[self.current_scene_index])

        # 输出实际使用的模型
        print(f"\n===== 模型确认 =====")
        print(f"玛莎·迪尔使用的模型端点: {librarian.talk_client.base_url}")
        print(f"威廉·阿克雷使用的模型端点: {professor.talk_client.base_url}")
        print(f"约瑟夫·马什使用的模型端点: {cultist.talk_client.base_url}")
        print(f"====================\n")
        
        # 启动第一个场景
        self.root.after(0, self.start_scene)

    def start_scene(self):
        # 获取当前场景编号
        current_scene_id = self.script_ids[self.current_scene_index]
        # 确保当前场景设置正确
        self.director.set_current_scene(current_scene_id)
        
        # 检查并创建当前场景中的新角色
        self.director.ensure_all_characters_exist(current_scene_id, self.player.name)

        print(f"\n当前场景编号: {current_scene_id}")

        # 生成场景描述
        detailed_scene = self.screenwriter.generate_scene_description(current_scene_id, self.director, self.player.get_player_name())
        self.scene_text.delete(1.0, tk.END)
        self.scene_text.insert(tk.END, detailed_scene)
        
        # 将生成的场景描述加入到dialogue_history中
        self.screenwriter.add_dialogue_record("旁白", "场景描述", detailed_scene)

        # 判断当前场景中是否有初始dialogues，如果有，则先让NPC角色表演
        scene_info = self.director.script.get(current_scene_id, {})
        initial_dialogues = scene_info.get("dialogues", [])
        
        if initial_dialogues:
            print("\n==== 对话开始 ====")
            for dialogue in initial_dialogues:
                character_name = dialogue.get("character")
                content = dialogue.get("content")
                
                # 跳过玩家角色的对话
                if character_name == self.player.get_player_name():
                    continue
                    
                # 显示NPC的对话
                print(f"\n{character_name}: {content}")
                self.scene_text.insert(tk.END, f"\n\n{character_name}: {content}")
                
                # 记录对话到编剧的对话历史
                self.screenwriter.add_dialogue_record(character_name, "场景对话", content)

        # 更新可对话角色列表
        self.update_character_list()
        
        # 重置场景状态
        self.scene_finished = False
        self.last_interaction = ""
        
        # 初始化场景尝试次数计数器
        self.scene_try_count = 0
        
        # 显示场景尝试次数上限
        print(f"\n==== 当前场景尝试机会: {self.settings['try_chance']} 次 ====")

    def update_character_list(self):
        # 获取当前场景可对话角色
        self.characters = self.director.get_scene_characters(player=self.player)
        
        # 更新列表框
        self.character_listbox.delete(0, tk.END)
        for character in self.characters:
            self.character_listbox.insert(tk.END, character)

    def talk_to_character(self):
        if self.scene_finished:
            messagebox.showinfo("提示", "当前场景已结束，请进入下一个场景")
            return
            
        # 检查是否还有尝试次数
        if self.scene_try_count >= self.settings['try_chance'] and not self.scene_finished:
            messagebox.showinfo("提示", f"已达到场景尝试次数上限({self.settings['try_chance']}次)")
            self.handle_scene_timeout()
            return
            
        # 检查是否选择了角色
        selected_idx = self.character_listbox.curselection()
        if not selected_idx:
            messagebox.showinfo("提示", "请先选择一个角色")
            return
            
        selected_character = self.characters[selected_idx[0]]
        
        # 获取对话内容
        dialogue = self.input_text.get(1.0, tk.END).strip()
        if not dialogue:
            messagebox.showinfo("提示", "请输入对话内容")
            return
        
        # 增加场景尝试次数
        self.scene_try_count += 1
        print(f"\n==== 尝试次数: {self.scene_try_count}/{self.settings['try_chance']} ====")
            
        # 记录最后一次对话
        self.last_interaction = f"{self.player.name}对{selected_character}说：{dialogue}"
        
        # 添加玩家对话记录
        self.screenwriter.add_dialogue_record(self.player.name, "对话", dialogue, target=selected_character)
        
        # 在场景文本中添加玩家对话
        self.scene_text.insert(tk.END, f"\n\n{self.player.name}: {dialogue}")
        
        # 使用合并后的方法直接生成指导
        guide_message = self.director.guide_actor_from_player_speech(dialogue, selected_character)
        
        # 获取角色实例，而不仅仅是角色名称
        actor_instance = self.director.actors.get(selected_character)
        
        if actor_instance:
            # 演员对话，使用Actor实例
            response = self.player.talk_to_actor(actor_instance, dialogue, guide_message)
            
            # 更新最后一次互动记录
            self.last_interaction += f"\n{selected_character}回应：{response}"
            
            # 添加NPC对话记录
            self.screenwriter.add_dialogue_record(selected_character, "对话", response, target=self.player.name)
            
            # 在场景文本中添加NPC回应
            self.scene_text.insert(tk.END, f"\n\n{selected_character}: {response}")
            self.scene_text.see(tk.END)
            
            print(f"\n{selected_character}: {response}")

            # 判断是否继续当前场景
            if not self.director.is_scene_continuing(response):
                print("当前场景结束")
                self.scene_finished = True
                
                # 获取下一个场景ID（如果有）
                next_scene = self.script_ids[self.current_scene_index + 1] if self.current_scene_index + 1 < len(self.script_ids) else None
                
                # 判断是否需要插入新场景
                should_generate = self.director.should_generate_new_script(self.screenwriter, 
                                                                         self.script_ids[self.current_scene_index], 
                                                                         next_scene)
                
                if should_generate and self.inserted_scene_count < self.settings['max_inserted_scenes']:
                    print(f"\n==== 根据情节发展，需要插入新场景 ({self.inserted_scene_count+1}/{self.settings['max_inserted_scenes']}) ====")
                    
                    # 生成新场景
                    new_script = self.screenwriter.generate_new_script(self.script_ids[self.current_scene_index], 
                                                                   dialogue_history=self.screenwriter.get_dialogue_history())
                    
                    if "error" not in new_script:
                        # 更新导演的剧本
                        self.director.load_script(self.screenwriter.initial_script)
                        
                        # 重新获取并排序剧本ID列表
                        self.script_ids = list(self.screenwriter.initial_script.keys())
                        new_scene_id = list(new_script.keys())[0]
                        print(f"\n成功生成新场景: {new_scene_id}")
                        
                        # 设置新场景为下一个场景
                        next_scene = new_scene_id
                        
                        # 增加插入场景计数
                        self.inserted_scene_count += 1
                        
                        # 检查并创建新角色
                        self.director.check_and_create_new_characters(self.script_ids, 
                                                                   self.script_ids.index(new_scene_id), 
                                                                   self.player.name)
                    else:
                        print(f"\n生成新场景失败: {new_script.get('error')}")
                elif should_generate and self.inserted_scene_count >= self.settings['max_inserted_scenes']:
                    print(f"\n==== 已达到插入新场景次数限制({self.settings['max_inserted_scenes']}次)，继续使用原有剧本 ====")
                
                # 生成场景结束描述
                ending_description = self.screenwriter.end_scene(self.last_interaction, self.director, 
                                                              self.script_ids[self.current_scene_index], next_scene)
                
                # 显示场景结束描述
                self.scene_text.insert(tk.END, f"\n\n{ending_description}")
                self.scene_text.see(tk.END)
                print(f"\n{ending_description}")
                
                # 添加场景转场描述
                self.screenwriter.add_dialogue_record("旁白", "场景转场", 
                                                   f"从{self.script_ids[self.current_scene_index]}场景转场到{next_scene if next_scene else '故事结束'}")
                
                # 提示用户进入下一场景
                messagebox.showinfo("场景结束", "当前场景已结束，请点击'下一场景'按钮继续")
            else:
                print("当前场景继续")
                
                # 检查是否已达到尝试次数上限
                if self.scene_try_count >= self.settings['try_chance'] and not self.scene_finished:
                    print(f"\n==== 场景尝试机会已用完 ({self.scene_try_count}/{self.settings['try_chance']})，强制结束场景 ====")
                    self.handle_scene_timeout()
        else:
            print(f"错误：找不到角色 {selected_character} 的实例")
            
        # 清空输入框
        self.input_text.delete(1.0, tk.END)

    def interact_with_environment(self):
        if self.scene_finished:
            messagebox.showinfo("提示", "当前场景已结束，请进入下一个场景")
            return
            
        # 检查是否还有尝试次数
        if self.scene_try_count >= self.settings['try_chance'] and not self.scene_finished:
            messagebox.showinfo("提示", f"已达到场景尝试次数上限({self.settings['try_chance']}次)")
            self.handle_scene_timeout()
            return
            
        # 获取互动内容
        interaction = self.input_text.get(1.0, tk.END).strip()
        if not interaction:
            messagebox.showinfo("提示", "请输入互动内容")
            return
            
        # 增加场景尝试次数
        self.scene_try_count += 1
        print(f"\n==== 尝试次数: {self.scene_try_count}/{self.settings['try_chance']} ====")
        
        # 记录最后一次互动
        self.last_interaction = f"{self.player.name}与环境互动：{interaction}"
        
        # 添加玩家互动记录
        self.screenwriter.add_dialogue_record(self.player.name, "环境互动", interaction)
        
        # 在场景文本中添加玩家互动
        self.scene_text.insert(tk.END, f"\n\n{self.player.name} 行动: {interaction}")
        
        # 编剧处理玩家行动
        action_response = self.screenwriter.transform_scene(self.script_ids[self.current_scene_index], interaction)
        
        # 更新最后一次互动记录
        self.last_interaction += f"\n环境响应：{action_response}"

        # 在场景文本中添加环境响应
        self.scene_text.insert(tk.END, f"\n\n{action_response}")
        self.scene_text.see(tk.END)
        
        print(f"\n{action_response}")

        # 判断是否继续当前场景
        if not self.director.is_scene_continuing(action_response):
            print("当前场景结束")
            self.scene_finished = True
            
            # 获取下一个场景ID（如果有）
            next_scene = self.script_ids[self.current_scene_index + 1] if self.current_scene_index + 1 < len(self.script_ids) else None
            
            # 判断是否需要插入新场景
            should_generate = self.director.should_generate_new_script(self.screenwriter, 
                                                                     self.script_ids[self.current_scene_index], 
                                                                     next_scene)
            
            if should_generate and self.inserted_scene_count < self.settings['max_inserted_scenes']:
                print(f"\n==== 根据情节发展，需要插入新场景 ({self.inserted_scene_count+1}/{self.settings['max_inserted_scenes']}) ====")
                
                # 生成新场景
                new_script = self.screenwriter.generate_new_script(self.script_ids[self.current_scene_index], 
                                                               dialogue_history=self.screenwriter.get_dialogue_history())
                
                if "error" not in new_script:
                    # 更新导演的剧本
                    self.director.load_script(self.screenwriter.initial_script)
                    
                    # 重新获取并排序剧本ID列表
                    self.script_ids = list(self.screenwriter.initial_script.keys())
                    new_scene_id = list(new_script.keys())[0]
                    print(f"\n成功生成新场景: {new_scene_id}")
                    
                    # 设置新场景为下一个场景
                    next_scene = new_scene_id
                    
                    # 增加插入场景计数
                    self.inserted_scene_count += 1
                    
                    # 检查并创建新角色
                    self.director.check_and_create_new_characters(self.script_ids, 
                                                               self.script_ids.index(new_scene_id), 
                                                               self.player.name)
                else:
                    print(f"\n生成新场景失败: {new_script.get('error')}")
            elif should_generate and self.inserted_scene_count >= self.settings['max_inserted_scenes']:
                print(f"\n==== 已达到插入新场景次数限制({self.settings['max_inserted_scenes']}次)，继续使用原有剧本 ====")
            
            # 生成场景结束描述
            ending_description = self.screenwriter.end_scene(self.last_interaction, self.director, 
                                                          self.script_ids[self.current_scene_index], next_scene)
            
            # 显示场景结束描述
            self.scene_text.insert(tk.END, f"\n\n{ending_description}")
            self.scene_text.see(tk.END)
            print(f"\n{ending_description}")
            
            # 添加场景转场描述
            self.screenwriter.add_dialogue_record("旁白", "场景转场", 
                                               f"从{self.script_ids[self.current_scene_index]}场景转场到{next_scene if next_scene else '故事结束'}")
            
            # 提示用户进入下一场景
            messagebox.showinfo("场景结束", "当前场景已结束，请点击'下一场景'按钮继续")
        else:
            print("当前场景继续")
            
            # 检查是否已达到尝试次数上限
            if self.scene_try_count >= self.settings['try_chance'] and not self.scene_finished:
                print(f"\n==== 场景尝试机会已用完 ({self.scene_try_count}/{self.settings['try_chance']})，强制结束场景 ====")
                self.handle_scene_timeout()
            
        # 清空输入框
        self.input_text.delete(1.0, tk.END)

    def go_to_next_scene(self):
        # 获取当前场景ID
        current_scene_id = self.script_ids[self.current_scene_index]
        
        if not self.scene_finished:
            # 如果场景未结束，询问用户是否确定跳过
            if not messagebox.askyesno("确认", "当前场景尚未结束，确定要跳过吗？"):
                return
                
            # 手动结束当前场景
            print("手动结束当前场景，进入下一个场景")
            
            # 获取下一个场景ID（如果有）
            next_scene = self.script_ids[self.current_scene_index + 1] if self.current_scene_index + 1 < len(self.script_ids) else None
            
            if next_scene:
                # 创建简单的转场描述
                ending_description = self.screenwriter.end_scene(self.last_interaction or "玩家选择跳过当前场景", 
                                                              self.director, current_scene_id, next_scene)
                                                              
                # 显示场景结束描述
                self.scene_text.insert(tk.END, f"\n\n{ending_description}")
                self.scene_text.see(tk.END)
                print(f"\n{ending_description}")
                
                # 添加场景转场描述
                self.screenwriter.add_dialogue_record("旁白", "场景转场", f"从{current_scene_id}场景转场到{next_scene}")
            else:
                print("\n没有更多场景，故事结束")
                messagebox.showinfo("游戏结束", "恭喜您完成了所有场景！")
                return
                
        # 移动到下一个场景
        self.current_scene_index += 1
        
        # 检查是否还有场景
        if self.current_scene_index < len(self.script_ids):
            print(f"\n==== 进入下一个场景: {self.script_ids[self.current_scene_index]} ====")
            # 开始新场景
            self.start_scene()
        else:
            # 更新剧本ID列表（可能有新场景生成）
            self.script_ids = list(self.screenwriter.initial_script.keys())
            
            # 检查是否需要生成全新场景（所有场景均已完成）
            if self.current_scene_index >= len(self.script_ids):
                # 检查是否已达到场景生成次数限制
                if self.new_scene_generation_count >= self.settings['max_new_scene_generations']:
                    print(f"\n==== 已达到场景生成次数限制({self.settings['max_new_scene_generations']}次)，准备结束故事 ====")
                    # 生成结尾场景
                    ending_prompt = "这是故事的结局场景。请根据之前的剧情走向，提供一个令人满意、符合逻辑且有情感冲击力的结束。确保所有主要情节线索都得到适当的解决。"
                    
                    # 使用特殊标记告诉编剧这是结尾
                    new_script = self.screenwriter.generate_new_script(current_scene_id, ending_prompt, 
                                                                    dialogue_history=self.screenwriter.get_dialogue_history())
                    
                    if "error" not in new_script:
                        # 更新剧本ID列表
                        self.script_ids = list(self.screenwriter.initial_script.keys())
                        ending_scene_id = list(new_script.keys())[0]
                        print(f"\n成功生成结局场景: {ending_scene_id}")
                        
                        # 更新导演的剧本
                        self.director.load_script(self.screenwriter.initial_script)
                        
                        # 确保下一个场景是结局场景
                        self.current_scene_index = self.script_ids.index(current_scene_id) + 1
                        # 检查并创建新角色
                        self.director.check_and_create_new_characters(self.script_ids, self.current_scene_index, self.player.name)
                        
                        # 开始新场景
                        self.start_scene()
                    else:
                        print(f"\n生成结局场景失败: {new_script.get('error')}")
                        messagebox.showinfo("游戏结束", "故事已结束，感谢您的参与！")
                else:
                    print(f"\n==== 所有计划场景已完成，尝试生成新场景 ({self.new_scene_generation_count+1}/{self.settings['max_new_scene_generations']}) ====")
                    # 生成新场景
                    new_script = self.screenwriter.generate_new_script(current_scene_id, 
                                                                    dialogue_history=self.screenwriter.get_dialogue_history())
                    
                    if "error" not in new_script:
                        # 更新剧本ID列表
                        self.script_ids = list(self.screenwriter.initial_script.keys())
                        new_scene_id = list(new_script.keys())[0]
                        print(f"\n成功生成新场景: {new_scene_id}")
                        
                        # 更新导演的剧本
                        self.director.load_script(self.screenwriter.initial_script)
                        
                        # 增加场景生成计数
                        self.new_scene_generation_count += 1
                        
                        # 检查并创建新角色
                        self.director.check_and_create_new_characters(self.script_ids, self.current_scene_index, self.player.name)
                        
                        # 开始新场景
                        self.start_scene()
                    else:
                        print(f"\n生成新场景失败: {new_script.get('error')}")
                        messagebox.showinfo("游戏结束", "故事已结束，感谢您的参与！")
            else:
                # 还有场景，开始新场景
                self.start_scene()

    def exit_game(self):
        if messagebox.askyesno("确认退出", "确定要退出游戏吗？"):
            # 导出对话历史到JSON文件供测评使用
            import json
            import datetime

            # 创建带有时间戳的文件名
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            dialogue_history_file = f"dialogue_history_{timestamp}.json"

            # 将对话历史转换为可序列化格式
            dialogue_history = self.screenwriter.get_all_dialogue_history() # 获取全部对话历史

            # 保存对话历史到文件
            with open(dialogue_history_file, "w", encoding="utf-8") as f:
                json.dump(dialogue_history, f, ensure_ascii=False, indent=2)

            print(f"\n对话历史已导出到文件: {dialogue_history_file}")
            
            # 退出应用
            self.root.destroy()

    def handle_scene_timeout(self):
        # 获取当前场景ID
        current_scene_id = self.script_ids[self.current_scene_index]
        
        # 获取下一个场景ID（如果有）
        next_scene = self.script_ids[self.current_scene_index + 1] if self.current_scene_index + 1 < len(self.script_ids) else None
        
        # 判断是否需要插入新场景
        should_generate = self.director.should_generate_new_script(self.screenwriter, current_scene_id, next_scene)
        
        if should_generate and self.inserted_scene_count < self.settings['max_inserted_scenes']:
            print(f"\n==== 根据情节发展，需要插入新场景 ({self.inserted_scene_count+1}/{self.settings['max_inserted_scenes']}) ====")
            
            # 生成新场景
            new_script = self.screenwriter.generate_new_script(current_scene_id, 
                                                           dialogue_history=self.screenwriter.get_dialogue_history())
            
            if "error" not in new_script:
                # 更新导演的剧本
                self.director.load_script(self.screenwriter.initial_script)
                
                # 重新获取并排序剧本ID列表
                self.script_ids = list(self.screenwriter.initial_script.keys())
                new_scene_id = list(new_script.keys())[0]
                print(f"\n成功生成新场景: {new_scene_id}")
                
                # 设置新场景为下一个场景
                next_scene = new_scene_id
                
                # 增加插入场景计数
                self.inserted_scene_count += 1
                
                # 检查并创建新角色
                self.director.check_and_create_new_characters(self.script_ids, 
                                                           self.script_ids.index(new_scene_id), 
                                                           self.player.name)
            else:
                print(f"\n生成新场景失败: {new_script.get('error')}")
        elif should_generate and self.inserted_scene_count >= self.settings['max_inserted_scenes']:
            print(f"\n==== 已达到插入新场景次数限制({self.settings['max_inserted_scenes']}次)，继续使用原有剧本 ====")
        
        # 生成场景结束描述
        ending_description = self.screenwriter.end_scene(self.last_interaction or "场景尝试次数已用完", 
                                                      self.director, current_scene_id, next_scene)
        
        # 显示场景结束描述
        self.scene_text.insert(tk.END, f"\n\n{ending_description}")
        self.scene_text.see(tk.END)
        print(f"\n{ending_description}")
        
        # 添加场景转场描述
        self.screenwriter.add_dialogue_record("旁白", "场景转场", 
                                           f"从{current_scene_id}场景转场到{next_scene if next_scene else '故事结束'}")
        
        # 设置场景为已结束
        self.scene_finished = True
        
        # 提示用户进入下一场景
        messagebox.showinfo("场景结束", "场景尝试次数已用完，请点击'下一场景'按钮继续")

    # 添加回车键处理函数
    def on_enter_pressed(self, event):
        # 阻止回车键在文本框中插入换行符
        self.submit_input()
        return "break"  # 阻止默认行为

if __name__ == "__main__":
    root = tk.Tk()
    app = CoCGameGUI(root)
    root.mainloop() 
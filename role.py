import openai
import faiss
import numpy as np
from typing import List, Dict, Tuple
import pickle
import os
from openai import OpenAI
from config import ark_api_key, ark_base_url, ark_model, ark_embedding_model, ark_embedding_dim, openai_api_key, deepseek_api_key, qwen_api_key, http_proxy, https_proxy, use_model

# 配置网络代理环境变量
# 火山方舟作为国内API通常不需要代理，设置为空字符串
os.environ["http_proxy"] = http_proxy
os.environ["https_proxy"] = https_proxy


def parse_ark_response(response):
    """解析火山方舟API的响应，提取文本内容

    火山方舟API的响应格式与OpenAI不同，可能返回两种格式：
    1. response.output_text 直接包含文本内容
    2. response.output 列表中包含多个item，需要遍历找到type为'output_text'的项

    Args:
        response: 火山方舟API返回的响应对象

    Returns:
        str: 提取出的文本内容，如果无法解析则返回响应对象的字符串表示
    """
    # 优先尝试直接读取output_text属性
    if hasattr(response, 'output_text') and response.output_text:
        return response.output_text
    
    # 其次尝试从output列表中提取
    if hasattr(response, 'output') and isinstance(response.output, list):
        for item in response.output:
            if hasattr(item, 'type') and item.type == 'output_text':
                if hasattr(item, 'text'):
                    return item.text
                if hasattr(item, 'content'):
                    return item.content
    
    # 如果以上方式都失败，返回响应的字符串表示作为兜底
    return str(response)


def handle_stream_response(client, model, messages, extra_body=None):
    """处理API响应的通用函数，支持流式和非流式响应

    根据use_model配置自动选择API调用方式，支持火山方舟、OpenAI、Deepseek、Qwen等多种模型。
    对于火山方舟使用responses.create接口，其他模型使用chat.completions.create接口。

    Args:
        client: OpenAI兼容的客户端实例
        model: 模型名称（实际使用时会被use_model覆盖）
        messages: 消息列表，包含系统提示和用户输入
        extra_body: 额外的请求体参数，用于传递特定模型的配置

    Returns:
        str: 模型生成的响应文本
    """
    if extra_body is None:
        extra_body = {}

    # 使用全局配置的模型类型
    if use_model == "ark":
        # 火山方舟使用responses.create接口，传入input参数
        response = client.responses.create(
            model=ark_model,
            input=messages,
            extra_body=extra_body
        )
        return parse_ark_response(response)
    else:
        # Qwen模型需要禁用思考模式
        if use_model == "qwen3-235b-a22b":
            if use_model.startswith("qwen"):
                extra_body["enable_thinking"] = False
            
            # Qwen模型使用流式响应
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
            # 其他模型使用非流式响应
            response = client.chat.completions.create(
                model=use_model,
                messages=messages,
                extra_body=extra_body
            )
            return response.choices[0].message.content


def get_ark_client():
    """创建火山方舟API客户端

    使用配置文件中的ark_api_key和ark_base_url初始化OpenAI兼容客户端。

    Returns:
        OpenAI: 配置好的火山方舟API客户端实例
    """
    return OpenAI(
        api_key=ark_api_key,
        base_url=ark_base_url
    )


def get_client():
    """根据use_model配置获取对应的API客户端

    根据全局配置的use_model参数，返回相应的API客户端实例。
    支持的模型类型：gpt-4o-mini、deepseek-chat、qwen3-235b-a22b、ark

    Returns:
        OpenAI: 配置好的API客户端实例
    """
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
    
    # 默认返回OpenAI客户端
    return OpenAI(
        api_key=openai_api_key,
        base_url="https://api.openai.com/v1"
    )


class Actor:
    """演员类，代表戏剧中的一个角色

    Actor类负责管理角色的记忆系统、性格特征和社交关系，
    并根据这些信息生成符合角色设定的对话回应。

    Attributes:
        name: 角色名称
        age: 角色年龄
        gender: 角色性别
        embedding_dim: 向量维度，根据使用的模型类型确定
        memories: 记忆文本列表
        memory_embeddings: 记忆向量矩阵（FAISS索引使用）
        index: FAISS向量索引，用于快速检索相关记忆
        relationships: 与其他角色的关系字典
        traits: 性格特征列表
        talk_client: 对话API客户端
        embedding_client: 向量API客户端
    """

    def __init__(self, name, age, gender, memory_path=None):
        """初始化演员对象

        Args:
            name: 角色名称
            age: 角色年龄
            gender: 角色性别
            memory_path: 记忆文件路径，用于加载已保存的记忆数据（可选）
        """
        self.name = name
        self.age = age
        self.gender = gender
        
        # 根据模型类型选择向量维度
        self.embedding_dim = ark_embedding_dim if use_model == "ark" else 1536
        
        self.memories = []
        self.memory_embeddings = None
        self.index = None
        self.relationships = {}
        self.traits = []

        # 根据模型类型初始化API客户端
        if use_model == "ark":
            self.talk_client = get_ark_client()
            self.embedding_client = get_ark_client()
        else:
            self.embedding_client = OpenAI(
                api_key=openai_api_key,
                base_url="https://api.openai.com/v1"
            )
            self.talk_client = get_client()

        # 初始化记忆系统
        self._initialize_memory(memory_path)

    def _initialize_memory(self, memory_path):
        """初始化记忆系统

        从文件加载已保存的记忆数据，或初始化空的记忆系统。
        使用FAISS构建向量索引以支持快速记忆检索。

        Args:
            memory_path: 记忆文件路径，如果为None或文件不存在则初始化空记忆
        """
        # 如果提供了文件路径且文件存在，加载记忆数据
        if memory_path and os.path.exists(memory_path):
            with open(memory_path, 'rb') as f:
                saved_data = pickle.load(f)
                self.memories = saved_data.get('memories', [])
                self.memory_embeddings = saved_data.get('embeddings')
                self.relationships = saved_data.get('relationships', {})
                self.traits = saved_data.get('traits', [])

        # 如果没有加载到向量数据或记忆为空，初始化空向量矩阵
        if self.memory_embeddings is None or len(self.memories) == 0:
            self.memory_embeddings = np.zeros((0, self.embedding_dim), dtype=np.float32)

        # 初始化FAISS向量索引（使用L2距离）
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        if len(self.memories) > 0:
            self.index.add(self.memory_embeddings)

    def __str__(self):
        """返回演员对象的字符串表示"""
        return f"Actor(name={self.name}, age={self.age}, gender={self.gender})"

    def add_memory(self, memory_text: str):
        """添加新记忆到记忆库

        将文本转换为向量后存储，并更新FAISS索引。

        Args:
            memory_text: 记忆文本内容

        Returns:
            int: 新记忆在记忆列表中的索引位置
        """
        # 获取文本向量表示
        embedding = self._get_embedding(memory_text)
        
        # 添加到记忆列表
        self.memories.append(memory_text)
        
        # 更新向量矩阵
        if len(self.memories) == 1:
            self.memory_embeddings = embedding.reshape(1, -1)
        else:
            self.memory_embeddings = np.vstack([self.memory_embeddings, embedding])

        # 重建FAISS索引
        self.index.reset()
        self.index.add(self.memory_embeddings)
        
        return len(self.memories) - 1

    def _get_embedding(self, text: str) -> np.ndarray:
        """获取文本的向量表示

        调用Embedding API将文本转换为向量，支持火山方舟和OpenAI两种模型。
        如果API调用失败，使用基于文本哈希的随机向量作为降级方案。

        Args:
            text: 待转换的文本

        Returns:
            np.ndarray: 文本的向量表示（float32类型）
        """
        if use_model == "ark":
            try:
                # 使用火山方舟的doubao-embedding-vision模型
                response = self.embedding_client.embeddings.create(
                    model=ark_embedding_model,
                    input=text
                )
                return np.array(response.data[0].embedding, dtype=np.float32)
            except Exception as e:
                # 降级处理：使用随机向量替代
                print(f"火山方舟Embedding API调用失败: {str(e)}")
                np.random.seed(hash(text) % 4294967295)
                return np.random.rand(self.embedding_dim).astype(np.float32)
        else:
            try:
                # 使用OpenAI的text-embedding-ada-002模型
                response = self.embedding_client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=text
                )
                return np.array(response.data[0].embedding, dtype=np.float32)
            except Exception as e:
                # 降级处理：使用随机向量替代
                print(f"Embedding API调用失败，使用本地随机向量替代: {str(e)}")
                np.random.seed(hash(text) % 4294967295)
                return np.random.rand(self.embedding_dim).astype(np.float32)

    def retrieve_relevant_memories(self, query: str, k: int = 3) -> List[str]:
        """检索与查询相关的记忆

        使用FAISS向量索引进行相似度搜索，返回最相关的k条记忆。

        Args:
            query: 查询文本
            k: 返回的记忆数量，默认为3

        Returns:
            List[str]: 相关记忆文本列表
        """
        if len(self.memories) == 0:
            return []

        # 获取查询向量并进行搜索
        query_embedding = self._get_embedding(query).reshape(1, -1)
        scores, indices = self.index.search(query_embedding, min(k, len(self.memories)))
        
        # 返回对应的记忆文本
        return [self.memories[idx] for idx in indices[0]]

    def add_trait(self, trait_description):
        """添加性格特征到角色

        性格特征会同时存储在traits列表和记忆系统中，
        以便在生成对话时能够考虑角色的性格特点。

        Args:
            trait_description: 性格特征描述，如"cautious"、"kind"等

        Returns:
            int: 性格特征在列表中的索引位置
        """
        if trait_description not in self.traits:
            self.traits.append(trait_description)
            # 将性格特征添加为记忆
            self.add_memory(f"My character trait is {trait_description}")
        return len(self.traits) - 1

    def get_traits(self):
        """获取角色的所有性格特征

        Returns:
            List[str]: 性格特征列表
        """
        return self.traits

    def save_memories(self, path: str):
        """保存记忆、关系和性格特征到文件

        使用pickle序列化存储，以便下次加载时恢复状态。

        Args:
            path: 保存文件的路径
        """
        with open(path, 'wb') as f:
            pickle.dump({
                'memories': self.memories,
                'embeddings': self.memory_embeddings,
                'relationships': self.relationships,
                'traits': self.traits
            }, f)

    def add_relationship(self, other_actor, relationship_type, description=""):
        """添加与其他角色的关系

        关系信息会同时存储在relationships字典和记忆系统中，
        支持关系类型的更新（如果已存在相同类型的关系则更新描述）。

        Args:
            other_actor: 另一个Actor对象或角色名称
            relationship_type: 关系类型，如"friend"、"family"、"colleague"等
            description: 关系描述，提供更详细的关系信息（可选）
        """
        # 获取对方角色名称
        other_name = other_actor.name if hasattr(other_actor, 'name') else other_actor

        # 初始化关系列表（如果不存在）
        if other_name not in self.relationships:
            self.relationships[other_name] = []

        # 检查是否已存在相同类型的关系，存在则更新
        for i, rel in enumerate(self.relationships[other_name]):
            if rel['type'] == relationship_type:
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
        memory_text = f"My relationship with {other_name} is {relationship_type}"
        if description:
            memory_text += f": {description}"
        self.add_memory(memory_text)

    def get_relationship(self, other_actor):
        """获取与指定角色的关系信息

        Args:
            other_actor: 另一个Actor对象或角色名称

        Returns:
            List[Dict]: 关系信息列表，每项包含'type'和'description'字段
        """
        other_name = other_actor.name if hasattr(other_actor, 'name') else other_actor
        return self.relationships.get(other_name, [])

    def get_all_relationships(self):
        """获取所有关系信息

        Returns:
            Dict[str, List[Dict]]: 关系字典，键为角色名称，值为关系信息列表
        """
        return self.relationships

    def speak(self, information, speaker=None, guidance=None):
        """根据记忆和关系生成角色对话回应

        综合考虑相关记忆、与对话者的关系、角色性格特征以及导演指导，
        生成符合角色设定的对话回应。

        Args:
            information: 对话内容或问题
            speaker: 与角色交谈的人（可以是Player对象或字符串名称）
            guidance: 导演提供的表演指导（可选）

        Returns:
            str: 角色的对话回应，格式为"(角色表情与动作) 对话内容"
        """
        # 检索与当前对话相关的记忆
        relevant_memories = self.retrieve_relevant_memories(information)

        # 提取说话者名称
        speaker_name = speaker
        if hasattr(speaker, 'name'):
            speaker_name = speaker.name

        # 构建关系上下文
        relationship_context = ""
        
        # 添加与当前说话者的关系
        if speaker_name and speaker_name in self.relationships:
            relationships = self.get_relationship(speaker_name)
            if relationships:
                rel_texts = []
                for rel in relationships:
                    rel_type = rel['type']
                    rel_desc = rel['description']
                    if rel_desc:
                        rel_text = f"Your relationship with {speaker_name} is {rel_type}: {rel_desc}"
                    else:
                        rel_text = f"Your relationship with {speaker_name} is {rel_type}"
                    rel_texts.append(rel_text)
                relationship_context += "\n".join(rel_texts) + "\n"

        # 添加对话中提到的其他角色的关系
        for name in self.relationships.keys():
            if name != speaker_name and name.lower() in information.lower():
                relationships = self.get_relationship(name)
                if relationships:
                    rel_texts = []
                    for rel in relationships:
                        rel_type = rel['type']
                        rel_desc = rel['description']
                        if rel_desc:
                            rel_text = f"Your relationship with {name} is {rel_type}: {rel_desc}"
                        else:
                            rel_text = f"Your relationship with {name} is {rel_type}"
                        rel_texts.append(rel_text)
                    relationship_context += "\n".join(rel_texts) + "\n"

        # 构建性格特征上下文
        traits_context = ""
        if self.traits:
            traits_context = "Your character traits are: " + ", ".join(self.traits) + ". Please shape your response based on these character traits.\n"

        # 整合所有上下文信息
        context = ""
        if relevant_memories or relationship_context or speaker_name or traits_context:
            context = "Answer based on the following information:\n"
            if speaker_name:
                context += f"The person talking to you is: {speaker_name}\n\n"
            if relationship_context:
                context += relationship_context + "\n"
            if traits_context:
                context += traits_context + "\n"
            if relevant_memories:
                context += "\n".join(relevant_memories) + "\n"
            context += "\nQuestion: "

        # 添加导演指导（如果有）
        if guidance:
            context += "\nDirector's guidance: " + guidance

        # 将对话记录添加到记忆
        if speaker_name:
            self.add_memory(f"{speaker_name} said to me: {information}")

        # 构建API消息列表
        messages = [
            {"role": "system", "content": f"You are a theater actor, and you are playing the role of {self.name}, a {self.age}-year-old {self.gender} in the play. Please make appropriate responses based on the identity of the conversationalist and your character traits.\n\nYour response must follow the following format: \"({self.name}'s expression and action) Conversation content\". The parentheses must contain the name of the character you are playing as the subject, clearly describe the expression and action, and the content after the parentheses should directly follow the conversation content. For example: \"({self.name} nervously clenches fists) I don't know what you're talking about.\" or \"({self.name} smiles and nods) I'm glad to see you.\""},
            {"role": "user", "content": (context + information) if context else information}
        ]

        # 调用API获取响应
        response_content = handle_stream_response(self.talk_client, use_model, messages)

        # 确保响应格式符合要求
        if not response_content.startswith(f"({self.name}"):
            response_content = f"({self.name} calmly says) {response_content}"

        # 将回应记录添加到记忆
        if speaker_name:
            self.add_memory(f"I said to {speaker_name}: {response_content}")

        return response_content

    def should_speak(self, context, speaker=None):
        """判断角色是否需要在当前情境下说话

        通过调用模型分析当前情境，判断角色是否应该做出回应。
        如果应该说话，则执行speak方法；否则返回None。

        Args:
            context: 当前对话或场景上下文
            speaker: 与角色交谈的人（可以是Player对象或字符串名称）

        Returns:
            str or None: 如果需要说话则返回对话内容，否则返回None
        """
        # 检索相关记忆以辅助判断
        relevant_memories = self.retrieve_relevant_memories(context)

        # 提取说话者信息
        speaker_name = speaker
        if hasattr(speaker, 'name'):
            speaker_name = speaker.name

        # 构建判断提示
        prompt = f"Scenario: {context}\n\n"

        if speaker_name:
            prompt += f"Speaker: {speaker_name}\n\n"

        if relevant_memories:
            prompt += "Relevant memories:\n" + "\n".join(relevant_memories) + "\n\n"

        prompt += f"As a character named {self.name}, do I need to speak in this situation? Please only answer 'Yes' or 'No', and give a brief reason."

        # 构建消息列表
        messages = [
            {"role": "system", "content": "You need to help the character determine if they should speak in the current situation. Please only answer 'Yes' or 'No', and give a brief reason."},
            {"role": "user", "content": prompt}
        ]

        # 获取判断结果
        decision = handle_stream_response(self.talk_client, use_model, messages)

        # 解析判断结果
        should_speak = "Yes" in decision[:10] or "yes" in decision.lower()[:10]

        # 如果应该说话则执行speak方法
        if should_speak:
            return self.speak(context, speaker)

        return None


class Director:
    """导演类，负责管理剧本、演员和场景切换

    Director类协调整个戏剧的流程，包括：
    - 管理演员列表和剧本数据
    - 生成新角色信息
    - 判断场景是否应该继续或切换
    - 为演员提供表演指导

    Attributes:
        actors: 演员字典，键为演员名称
        script: 剧本字典，按场景组织
        current_scene: 当前场景ID
        client: API客户端实例
    """

    def __init__(self):
        """初始化导演对象"""
        self.actors = {}
        self.script = {}
        self.current_scene = None

        # 根据配置获取API客户端
        self.client = get_client()

    def add_actor(self, actor):
        """添加演员到导演管理中

        Args:
            actor: Actor对象
        """
        self.actors[actor.name] = actor

    def generate_actor_profile(self, character_name, scene_id, player_name):
        """使用AI生成角色的详细信息

        根据场景描述和角色名称，调用API生成完整的角色档案，
        包括年龄、性别、背景故事、性格特征和与玩家角色的关系。

        Args:
            character_name: 角色名称
            scene_id: 当前场景ID
            player_name: 玩家角色名称

        Returns:
            Dict: 包含角色信息的字典，格式见方法内prompt定义
        """
        # 获取场景描述作为上下文
        scene_desc = self.get_scene_description(scene_id)

        # 构建生成角色信息的prompt
        prompt = f"""Based on the following scene description and character name, generate a complete character profile:

Scene description: {scene_desc}

Character name: {character_name}

Player character name: {player_name}

Please generate character information including the following:
1. Character age
2. Character gender
3. Character background story (3 - 5 items)
4. Character traits (2 - 3 items)
5. Relationship type and description with the player character "{player_name}"

Please return the result in JSON format:
{{
    "age": number,
    "gender": "gender",
    "background": ["memory/background 1", "memory/background 2", ...],
    "traits": ["trait 1", "trait 2", ...],
    "relationship": {{
        "type": "relationship type",
        "description": "relationship description"
    }}
}}"""

        # 调用API生成角色信息
        messages = [
            {"role": "system", "content": "You are an AI assistant creating characters. Generate character information that fits the scenario based on the scene and character name."},
            {"role": "user", "content": prompt}
        ]

        response = handle_stream_response(self.client, use_model, messages)

        # 尝试解析JSON响应
        try:
            import json
            import re

            # 提取JSON部分
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
        except Exception as e:
            print(f"Failed to parse character information: {str(e)}")

        # 返回默认角色信息作为降级方案
        return {
            "age": 30,
            "gender": "unknown",
            "background": [f"I am {character_name}, a new character in the current scene"],
            "traits": ["mysterious"],
            "relationship": {
                "type": "stranger",
                "description": "Just met"
            }
        }

    def ensure_all_characters_exist(self, scene_id, player_name):
        """确保场景中的所有角色都已创建

        检查剧本中定义的角色是否都在actors字典中，
        如果不存在则使用AI生成角色信息并创建Actor实例。

        Args:
            scene_id: 当前场景ID
            player_name: 玩家角色名称，用于排除玩家角色和建立关系
        """
        # 获取场景中的所有角色
        all_characters = self.get_scene_characters(scene_id=scene_id)

        # 逐个检查并创建角色
        for character_name in all_characters:
            # 跳过玩家角色
            if character_name == player_name:
                continue

            # 如果角色不存在，创建新的Actor实例
            if character_name not in self.actors:
                print(f"\nNew character found: {character_name}, using AI to generate character information...")

                # 使用AI生成角色信息
                profile = self.generate_actor_profile(character_name, scene_id, player_name)

                # 创建Actor实例
                new_actor = Actor(character_name, profile.get("age", 30), profile.get("gender", "unknown"))

                # 添加背景记忆
                for memory in profile.get("background", [f"I am {character_name}, a new character in the current scene"]):
                    new_actor.add_memory(memory)

                # 添加性格特征
                for trait in profile.get("traits", ["mysterious"]):
                    new_actor.add_trait(trait)

                # 添加与玩家角色的关系
                relationship = profile.get("relationship", {"type": "stranger", "description": "Just met"})
                new_actor.add_relationship(
                    player_name,
                    relationship.get("type", "stranger"),
                    relationship.get("description", "Just met")
                )

                # 将新角色添加到导演管理中
                self.add_actor(new_actor)
                
                # 输出创建信息
                print(f"Character created: {character_name}, Age: {profile.get('age', 30)}, Gender: {profile.get('gender', 'unknown')}")
                print(f"Character traits: {', '.join(profile.get('traits', ['mysterious']))}")
                print(f"Relationship with player: {relationship.get('type', 'stranger')} - {relationship.get('description', 'Just met')}")

    def check_and_create_new_characters(self, scene_ids, current_scene_index, player_name):
        """检查并创建新场景中可能出现的角色

        预先检查下一个场景的角色，确保在进入场景前所有角色都已创建。

        Args:
            scene_ids: 场景ID列表
            current_scene_index: 当前场景索引
            player_name: 玩家角色名称
        """
        if current_scene_index < len(scene_ids):
            current_scene_id = scene_ids[current_scene_index]
            self.ensure_all_characters_exist(current_scene_id, player_name)

    def load_script(self, script_dict):
        """加载剧本

        Args:
            script_dict: 剧本字典，格式为：
                {
                    "scene_1": {
                        "description": "场景描述",
                        "characters": ["角色名1", "角色名2"],
                        "dialogues": [{"character": "角色名", "content": "对话内容"}]
                    }
                }
        """
        self.script = script_dict

    def set_current_scene(self, scene_id):
        """设置当前场景

        Args:
            scene_id: 场景ID

        Returns:
            bool: 如果场景存在则返回True，否则返回False
        """
        if scene_id in self.script:
            self.current_scene = scene_id
            return True
        return False

    def get_current_scene(self):
        """获取当前场景ID

        Returns:
            str or None: 当前场景ID
        """
        return self.current_scene

    def get_scene_description(self, scene_id=None):
        """获取场景描述

        Args:
            scene_id: 场景ID，如果为None则使用当前场景

        Returns:
            str: 场景描述文本
        """
        scene = scene_id if scene_id else self.current_scene
        if scene in self.script:
            return self.script[scene].get("description", "")
        return ""

    def get_scene_characters(self, scene_id=None, player=None):
        """获取场景中的角色列表

        可选地排除玩家角色。

        Args:
            scene_id: 场景ID，如果为None则使用当前场景
            player: Player对象或玩家角色名称，将从返回结果中排除

        Returns:
            List[str]: 角色名称列表
        """
        scene = scene_id if scene_id else self.current_scene
        if scene not in self.script:
            return []

        characters = self.script[scene].get("characters", [])

        # 如果提供了player参数，排除玩家角色
        if player:
            player_name = player
            if hasattr(player, 'get_player_name'):
                player_name = player.get_player_name()
            elif hasattr(player, 'name'):
                player_name = player.name

            characters = [char for char in characters if char != player_name]

        return characters

    def guide_actor_from_player_speech(self, player_speech, actor_name):
        """根据玩家发言为演员提供表演指导

        综合场景信息、角色性格和玩家发言，生成针对性的表演指导。

        Args:
            player_speech: 玩家的发言内容
            actor_name: 要指导的演员名称

        Returns:
            str: 表演指导内容
        """
        # 验证场景和演员是否存在
        if self.current_scene is None or actor_name not in self.actors:
            return "Unable to guide: Current scene not set or actor not found."

        # 获取场景信息
        scene_info = self.script.get(self.current_scene, {})
        scene_desc = scene_info.get("description", "")

        # 获取演员的台词
        dialogues = scene_info.get("dialogues", [])
        actor_dialogues = [d for d in dialogues if d.get("character") == actor_name]

        # 获取演员信息
        actor = self.actors.get(actor_name)
        actor_traits = actor.get_traits() if hasattr(actor, "get_traits") else []
        traits_text = ", ".join(actor_traits) if actor_traits else "No specific character traits."

        # 构建指导prompt
        prompt = f"""As a theater director, please provide acting guidance for the actor '{actor_name}' based on the following information:

Scene description: {scene_desc}

Actor's character traits: {traits_text}

Player's recent speech: "{player_speech}"

Actor's lines:
"""
        for dialogue in actor_dialogues:
            prompt += f"- {dialogue.get('content')}\n"

        prompt += """
Please analyze the intention, emotion, and possible implied meaning of the player's speech, and then provide specific acting guidance, including:
1. Emotional expression suggestions
2. Body language guidance
3. Tone and rhythm suggestions
4. Whether certain information should be revealed
5. How to stay true to the character's traits

Please directly provide the guidance content without any preamble:"""

        # 调用API生成指导
        messages = [
            {"role": "system", "content": "You are an experienced theater director, good at analyzing the player's speech and providing response guidance for the actor."},
            {"role": "user", "content": prompt}
        ]

        return handle_stream_response(self.client, use_model, messages)

    def is_scene_continuing(self, last_dialogue, screenwriter=None, detailed_scene=None):
        """判断当前场景是否应该继续

        根据场景描述、对话历史和玩家目标，判断场景是否应该继续进行。

        Args:
            last_dialogue: 最近的对话内容（可选）
            screenwriter: Screenwriter对象，用于获取对话历史（可选）
            detailed_scene: 详细场景描述（可选，优先使用）

        Returns:
            bool: 如果场景应该继续则返回True，否则返回False
        """
        if self.current_scene is None:
            return False

        scene_info = self.script.get(self.current_scene, {})
        dialogues = scene_info.get("dialogues", [])

        # 如果没有对话，场景无法继续
        if not dialogues:
            return False

        # 提取玩家目标描述
        player_goal = ""
        scene_description = ""

        # 优先使用传入的详细场景描述
        if detailed_scene:
            scene_description = detailed_scene
        elif screenwriter and hasattr(screenwriter, 'scene_descriptions') and self.current_scene in screenwriter.scene_descriptions:
            scene_description = screenwriter.scene_descriptions.get(self.current_scene, "")

        # 使用正则表达式提取玩家目标
        if scene_description:
            import re
            goal_patterns = [
                r"Player character goal description[：:](.*?)(?=\n\n|\Z)",
                r"Player character goal[：:](.*?)(?=\n\n|\Z)",
                r"Player goal[：:](.*?)(?=\n\n|\Z)",
                r"4[\.。]\s*Player character goal[^：:]*[：:](.*?)(?=\n\n|\Z)"
            ]

            for pattern in goal_patterns:
                goal_match = re.search(pattern, scene_description, re.DOTALL)
                if goal_match:
                    player_goal = goal_match.group(1).strip()
                    break

        # 获取对话历史
        recent_dialogue_history = ""
        if screenwriter and hasattr(screenwriter, 'get_dialogue_history'):
            recent_dialogues = screenwriter.get_dialogue_history(limit=5)
            if recent_dialogues:
                recent_dialogue_history = "\n\nRecent dialogue history:\n"
                for d in recent_dialogues:
                    if 'record_type' in d:
                        record_type = d['record_type']
                        if record_type == "Dialogue" or record_type.startswith("Dialogue"):
                            if 'target' in d:
                                recent_dialogue_history += f"{d['speaker']} said to {d['target']}: {d['content']}\n"
                            else:
                                recent_dialogue_history += f"{d['speaker']} ({record_type}): {d['content']}\n"
                        else:
                            recent_dialogue_history += f"{d['speaker']} ({record_type}): {d['content']}\n"
                    else:
                        recent_dialogue_history += f"{d['speaker']}: {d['content']}\n"

        # 构建判断prompt
        prompt = f"Please analyze the following situation and determine if the script should continue in the current scene:\n\n"
        prompt += f"Scene description: {scene_info.get('description', '')}\n\n"

        if player_goal:
            prompt += f"Player character's goal in this scene: {player_goal}\n\n"

        prompt += "Expected dialogue content:\n"
        for d in dialogues[-3:]:
            prompt += f"{d.get('character')}: {d.get('content')}\n"

        if recent_dialogue_history:
            prompt += recent_dialogue_history
        elif last_dialogue:
            prompt += f"\nActual latest dialogue: {last_dialogue}\n"

        prompt += "\nBased on the following criteria, determine if the current scene should continue:\n"
        prompt += "1. Whether the player character's goal has been achieved.\n"
        prompt += "2. Whether the key dialogues in the scene have been completed.\n"
        prompt += "3. Whether the dialogue has naturally reached an end point.\n"
        prompt += "4. Whether there are obvious scene transition clues.\n\n"
        prompt += "Especially note: The player character's goal is an important condition for determining if the scene should continue. If the goal has not been achieved and there are no obvious transition signals, the scene usually should continue.\n\n"
        prompt += "Please clearly determine if the scene should continue. First answer 'Yes' or 'No', then give a brief reason."

        # 调用API进行判断
        messages = [
            {"role": "system", "content": "You are an experienced theater director, good at analyzing scripts and performances. You need to determine if the current scene should continue or move on to the next scene."},
            {"role": "user", "content": prompt}
        ]

        result = handle_stream_response(self.client, use_model, messages).lower()

        # 判断结果
        return "Yes" in result[:30] or "Should continue" in result[:30] or "Continue" in result[:30] or "Not achieved" in result[:50]

    def should_generate_new_script(self, screenwriter, current_scene_id, next_scene_id=None):
        """判断是否需要在场景间插入新场景

        分析当前场景结束后的情节发展，判断是否需要生成过渡场景。

        Args:
            screenwriter: Screenwriter对象
            current_scene_id: 当前场景ID
            next_scene_id: 下一个计划场景ID（可选）

        Returns:
            bool: 如果需要生成新场景则返回True，否则返回False
        """
        # 如果当前场景还在继续，则不需要生成新场景
        if self.is_scene_continuing(None, screenwriter):
            return False

        # 获取对话历史
        recent_dialogues = screenwriter.get_dialogue_history(limit=10)
        dialogue_text = ""
        for d in recent_dialogues:
            if 'record_type' in d:
                record_type = d['record_type']
                if record_type == "Dialogue" or record_type.startswith("Dialogue"):
                    dialogue_text += f"{d['speaker']} said to {record_type}: {d['content']}\n"
                else:
                    dialogue_text += f"{d['speaker']} ({record_type}): {d['content']}\n"

        # 获取当前场景信息
        current_scene = self.script.get(current_scene_id, {})
        current_scene_desc = current_scene.get("description", "")

        # 获取下一个场景信息（如果有）
        next_scene_info = ""
        if next_scene_id and next_scene_id in self.script:
            next_scene = self.script.get(next_scene_id, {})
            next_scene_desc = next_scene.get("description", "")
            next_scene_info = f"""Planned next scene description:
{next_scene_desc}"""

        # 构建判断prompt
        prompt = f"""As a theater director, please determine if a new scene needs to be inserted between the current scene and the next planned scene.

Current scene description:
{current_scene_desc}

Recent dialogue history:
{dialogue_text}

{next_scene_info}

Please analyze the following points:
1. Whether the plot of the current scene has naturally ended.
2. Whether there are new plot clues in the dialogue history that need to be immediately addressed.
3. Whether there is a large plot or scene gap between the current scene and the next planned scene.
4. Whether there are unresolved conflicts or incomplete plots that need to be addressed in a new scene.

Based on the above analysis, please determine if a new scene needs to be inserted? Please only answer 'Yes' or 'No', then give a brief reason."""

        # 调用API进行判断
        messages = [
            {"role": "system", "content": "You are an experienced theater director, good at analyzing plot development and scene transitions."},
            {"role": "user", "content": prompt}
        ]

        result = handle_stream_response(self.client, use_model, messages)

        # 判断结果
        return "Yes" in result[:10] or "Need" in result[:20] or "Should" in result[:20]


class Player:
    """玩家类，代表玩家控制的角色

    Player类提供玩家与场景中其他角色交互的接口，
    包括对话和环境互动。

    Attributes:
        name: 玩家角色名称
        age: 玩家角色年龄
        gender: 玩家角色性别
        current_scene: 当前所在场景描述
    """

    def __init__(self, name, age, gender):
        """初始化玩家角色

        Args:
            name: 玩家扮演的角色名称
            age: 玩家扮演的角色年龄
            gender: 玩家扮演的角色性别
        """
        self.name = name
        self.age = age
        self.gender = gender

    def talk_to_actor(self, actor, message, guidance=None):
        """与场景中的演员对话

        将玩家消息传递给指定演员，获取演员的回应。

        Args:
            actor: Actor对象
            message: 玩家输入的对话内容
            guidance: 导演生成的指导（可选）

        Returns:
            str: 演员的回应内容
        """
        if not hasattr(actor, 'speak'):
            return f"Error: Unable to talk to this object."

        response = actor.speak(message, self.name, guidance)
        return response

    def interact_with_environment(self, screenwriter, action, current_scene_id=None):
        """与当前场景环境或物品交互

        记录玩家的环境互动行为，并更新场景描述。

        Args:
            screenwriter: Screenwriter对象，用于处理玩家行为
            action: 玩家想要执行的动作
            current_scene_id: 当前场景ID（可选）

        Returns:
            str: 场景变化描述
        """
        # 记录互动内容
        screenwriter.add_dialogue_record(self.name, "Environmental interaction", action)

        # 如果提供了场景ID，更新场景描述
        if current_scene_id:
            updated_scene = screenwriter.transform_scene(
                current_scene_id,
                action
            )
            self.current_scene = updated_scene

        return updated_scene

    def get_player_name(self):
        """获取玩家名称

        Returns:
            str: 玩家角色名称
        """
        return self.name


class Screenwriter:
    """编剧类，负责剧本生成和场景转换

    Screenwriter类处理：
    - 加载和管理剧本数据
    - 生成详细场景描述
    - 记录对话历史
    - 根据对话历史生成新场景
    - 处理场景转换描述

    Attributes:
        client: API客户端实例
        dialogue_history: 对话历史列表
        scene_descriptions: 场景描述字典
        initial_script: 初始剧本字典
    """

    def __init__(self):
        """初始化编剧对象"""
        self.client = get_client()

        self.dialogue_history = []
        self.scene_descriptions = {}
        self.initial_script = {}

    def load_initial_script(self, script_dict):
        """加载初始剧本

        将剧本数据加载到内部存储，并初始化场景描述。

        Args:
            script_dict: 初始剧本字典
        """
        self.initial_script = script_dict
        for scene_id, scene_data in script_dict.items():
            if "description" in scene_data:
                self.scene_descriptions[scene_id] = scene_data["description"]

    def generate_scene_description(self, scene_id, director=None, player_character=None):
        """生成场景的详细描述

        根据基础描述和角色信息，生成沉浸式的场景描述。

        Args:
            scene_id: 场景ID
            director: Director对象，用于获取角色详细信息（可选）
            player_character: 玩家控制的角色名称（可选），不会被详细描述

        Returns:
            str: 包含环境、物品和非玩家角色的详细场景描述
        """
        # 获取基础场景描述
        base_description = self.scene_descriptions.get(scene_id)
        if not base_description:
            if scene_id in self.initial_script and "description" in self.initial_script[scene_id]:
                base_description = self.initial_script[scene_id]["description"]
            else:
                return f"Error: No description found for scene {scene_id}"

        # 获取场景角色信息
        scene_characters = []
        characters_info = ""

        if scene_id in self.initial_script and "characters" in self.initial_script[scene_id]:
            scene_characters = self.initial_script[scene_id]["characters"]

            # 如果提供了Director，获取角色详细信息
            if director and hasattr(director, 'actors'):
                for char_name in scene_characters:
                    # 跳过玩家角色
                    if player_character and char_name == player_character:
                        continue

                    if char_name in director.actors:
                        actor = director.actors[char_name]
                        char_info = f"- {char_name}: {actor.age} years old, {actor.gender}\n"

                        # 添加性格特征
                        if hasattr(actor, 'get_traits') and actor.get_traits():
                            traits = actor.get_traits()
                            char_info += f"  Personality traits: {', '.join(traits)}\n"

                        # 添加与其他角色的关系
                        if hasattr(actor, 'get_all_relationships'):
                            relationships = actor.get_all_relationships()
                            if relationships:
                                char_info += "  Relationships with other characters:\n"
                                for other_name, rel_list in relationships.items():
                                    if other_name in scene_characters:
                                        for rel in rel_list:
                                            rel_type = rel.get('type', '')
                                            rel_desc = rel.get('description', '')
                                            if rel_desc:
                                                char_info += f"    - With {other_name}: {rel_type} ({rel_desc})\n"
                                            else:
                                                char_info += f"    - With {other_name}: {rel_type}\n"

                        characters_info += char_info

        # 构建生成场景描述的prompt
        prompt = f"""Based on the following base description of the scene and character information, generate a more detailed description of the scene, including the scene environment, items in the scene, and non-player characters:

Base description of the scene: {base_description}

"""

        if characters_info:
            prompt += f"""Character information in the scene:
{characters_info}
"""

        if player_character:
            prompt += f"""Note: The player character "{player_character}" in the scene does not need to be described in detail because the player will control this character themselves.
"""

        prompt += """Please provide:
1. Environment description: including spatial layout, lighting, sound, smell, etc.
2. Item description: list the main items in the scene and their placement
3. Non-player character description: describe the appearance, posture, current behavior, and emotional state of the characters in the scene (excluding the player character), and the personality description should be consistent with the above character information
4. Player character goal description: describe the general goal of the player character in the current scene

Please use vivid and vivid language to create an immersive scene experience."""

        # 调用API生成详细场景描述
        messages = [
            {"role": "system",
             "content": "You are an experienced drama screenwriter, good at creating vivid and detailed scene descriptions."},
            {"role": "user", "content": prompt}
        ]

        detailed_description = handle_stream_response(self.client, use_model, messages)

        # 更新场景描述存储
        self.scene_descriptions[scene_id] = detailed_description

        return detailed_description

    def add_dialogue_record(self, speaker, record_type, content, target=None):
        """添加对话记录

        记录对话或互动内容到对话历史中。

        Args:
            speaker: 说话者或行动者名称
            record_type: 记录类型，如"Dialogue"、"Narrative"等
            content: 内容
            target: 对话接收者（可选）
        """
        record = {
            "time": len(self.dialogue_history),
            "speaker": speaker,
            "record_type": record_type,
            "content": content
        }

        # 如果是对话类型且有目标，添加目标字段
        if target and (record_type == "dialogue" or record_type.startswith("dialogue")):
            record["target"] = target

        self.dialogue_history.append(record)

    def get_dialogue_history(self, limit=10):
        """获取最近的对话历史

        Args:
            limit: 返回的最大记录数，默认为10

        Returns:
            List[Dict]: 对话历史记录列表
        """
        return self.dialogue_history[-limit:] if self.dialogue_history else []

    def get_all_dialogue_history(self):
        """获取所有对话历史

        Returns:
            List[Dict]: 所有对话历史记录列表
        """
        return self.dialogue_history if self.dialogue_history else []

    def generate_new_script(self, current_scene_id, player_feedback=None, max_retries=3, dialogue_history=None):
        """根据历史对话生成新场景

        基于当前场景和对话历史，生成下一个场景的剧本。

        Args:
            current_scene_id: 当前场景ID
            player_feedback: 玩家反馈（可选）
            max_retries: 最大重试次数，默认为3
            dialogue_history: 对话历史记录（可选）

        Returns:
            Dict: 新场景剧本字典，格式为{scene_id: scene_data}
        """
        # 生成下一个场景ID
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

        # 获取当前场景信息
        current_scene = self.initial_script.get(current_scene_id, {})
        scene_desc = self.scene_descriptions.get(current_scene_id, "")

        # 获取当前场景角色
        current_characters = current_scene.get("characters", [])

        # 构建对话历史文本
        dialogue_text = ""
        if dialogue_history:
            for d in dialogue_history:
                if isinstance(d, dict):
                    speaker = d.get('speaker', '')
                    record_type = d.get('record_type', '')
                    content = d.get('content', '')
                    if record_type == "dialogue" or record_type.startswith("dialogue"):
                        if 'target' in d:
                            dialogue_text += f"{speaker} says to {d.get('target')}: {content}\n"
                        else:
                            dialogue_text += f"{speaker} ({record_type}): {content}\n"
                    else:
                        dialogue_text += f"{speaker} ({record_type}): {content}\n"
        else:
            recent_dialogues = self.get_dialogue_history(limit=20)
            for d in recent_dialogues:
                if 'record_type' in d:
                    record_type = d.get('record_type', '')
                    if record_type == "dialogue" or record_type.startswith("dialogue"):
                        if 'target' in d:
                            dialogue_text += f"{d['speaker']} says to {d.get('target')}: {d['content']}\n"
                        else:
                            dialogue_text += f"{d['speaker']} ({record_type}): {d['content']}\n"
                    else:
                        dialogue_text += f"{d['speaker']} ({record_type}): {d['content']}\n"
                else:
                    dialogue_text += f"{d['speaker']}: {d['content']}\n"

        # 最多重试max_retries次生成新场景
        for attempt in range(max_retries):
            prompt = f"""As a screenwriter, please generate the next scene of the script based on the following information:

Current scene: {scene_desc}

Current scene characters: {', '.join(current_characters)}

Dialogue history:
{dialogue_text}

"""

            if player_feedback:
                prompt += f"""
Player feedback:
{player_feedback}

"""

            prompt += """Please create the next scene of the script based on the above information. Pay special attention to:
1. The new scene must be consistent with the dialogue history and cannot have contradictions.
2. The actions and dialogues of the characters must conform to their personality traits.
3. The scene transition must be natural and the plot development must be reasonable.

You must strictly return in the following JSON format, which is required by the system:
{
    "description": "Detailed description of the scene",
    "characters": ["Character name 1", "Character name 2", ...],
    "dialogues": [
        {"character": "Character name 1", "content": "(Character's expression and action) Dialogue content 1"},
        {"character": "Character name 2", "content": "(Character's expression and action) Dialogue content 2"},
        ...
    ]
}

Note:
1. Only return the content of one scene.
2. The JSON format must be completely correct, without any explanations or extra text.
3. The dialogue content should conform to the character's characteristics and promote the plot development.
4. Do not generate the scene_id field, the system will handle it automatically.
5. Ensure that the new scene is consistent with the previous dialogue and plot."""

            # 调用API生成新场景
            messages = [
                {"role": "system",
                 "content": "You are an experienced screenwriter, specializing in creating scripts for interactive dramas. You must return the results in the required JSON format and ensure that the new scene is consistent with the historical dialogue and plot."},
                {"role": "user", "content": prompt}
            ]

            generated_text = handle_stream_response(self.client, use_model, messages)

            # 尝试解析生成的JSON
            try:
                import json
                import re

                json_match = re.search(r'\{[\s\S]*\}', generated_text)
                if json_match:
                    generated_text = json_match.group(0)

                new_scene = json.loads(generated_text)

                # 验证必需字段
                if not all(key in new_scene for key in ["description", "dialogues"]):
                    raise ValueError("Generated JSON is missing required fields")

                # 保存新场景
                self.initial_script[next_scene_id] = {
                    "description": new_scene["description"],
                    "characters": new_scene["characters"],
                    "dialogues": new_scene["dialogues"]
                }

                self.scene_descriptions[next_scene_id] = new_scene["description"]

                return {next_scene_id: self.initial_script[next_scene_id]}

            except Exception as e:
                # 如果还有重试机会则继续尝试
                if attempt < max_retries - 1:
                    continue
                else:
                    return {"error": str(e), "generated_text": generated_text, "next_scene_id": next_scene_id}

    def generate_actor_response_suggestions(self, actor_name, player_action):
        """为演员生成回应建议

        根据玩家动作和对话历史，生成3-5个可能的演员回应。

        Args:
            actor_name: 演员名称
            player_action: 玩家动作描述

        Returns:
            List[str]: 回应建议列表
        """
        # 获取对话历史
        recent_dialogues = self.get_dialogue_history(limit=5)
        dialogue_text = ""
        for d in recent_dialogues:
            if 'record_type' in d:
                record_type = d['record_type']
                if record_type == "dialogue" or record_type.startswith("dialogue"):
                    if 'target' in d:
                        dialogue_text += f"{d['speaker']} says to {d.get('target')}: {d['content']}\n"
                    else:
                        dialogue_text += f"{d['speaker']} ({record_type}): {d['content']}\n"
                else:
                    dialogue_text += f"{d['speaker']} ({record_type}): {d['content']}\n"
            else:
                dialogue_text += f"{d['speaker']}: {d['content']}\n"

        # 构建生成建议的prompt
        prompt = f"""Please provide 3-5 possible responses for the actor '{actor_name}' based on the following context:

Player action: {player_action}

Recent dialogue history:
{dialogue_text}

Please provide response suggestions that are consistent with the character's personality and the current situation. Each suggestion should include:
1. The character's expression and action
2. The dialogue content

Format each suggestion as: "(Expression and action) Dialogue content"

Do not include any explanations or extra text."""

        # 调用API生成建议
        messages = [
            {"role": "system",
             "content": "You are an experienced screenwriter who specializes in creating actor responses for interactive dramas."},
            {"role": "user", "content": prompt}
        ]

        response = handle_stream_response(self.client, use_model, messages)

        # 解析建议列表
        suggestions = []
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line and ('(' in line and ')' in line):
                suggestions.append(line)

        return suggestions

    def transform_scene(self, scene_id, player_action):
        """根据玩家动作转换场景描述

        更新场景描述以反映玩家的环境互动。

        Args:
            scene_id: 场景ID
            player_action: 玩家动作描述

        Returns:
            str: 更新后的场景描述
        """
        # 获取当前场景描述
        current_scene = self.scene_descriptions.get(scene_id, "")

        # 构建转换prompt
        prompt = f"""Please describe how the scene changes based on the player's action.

Current scene description:
{current_scene}

Player action: {player_action}

Please provide a detailed description of the changes in the scene, including:
1. Changes in the environment
2. Changes in the positions or states of items
3. Changes in the reactions or behaviors of characters
4. Any new elements that appear

Keep the description concise but vivid."""

        # 调用API生成场景变化描述
        messages = [
            {"role": "system",
             "content": "You are an experienced screenwriter who specializes in describing scene transformations."},
            {"role": "user", "content": prompt}
        ]

        response = handle_stream_response(self.client, use_model, messages)

        # 更新场景描述
        self.scene_descriptions[scene_id] = response

        return response

    def end_scene(self, last_interaction, director, current_scene_id, next_scene_id):
        """生成场景转场描述

        根据当前场景、最后互动和下一个场景，生成自然的转场描述。

        Args:
            last_interaction: 最后一次互动内容
            director: Director对象
            current_scene_id: 当前场景ID
            next_scene_id: 下一个场景ID（可选）

        Returns:
            str: 场景转场描述
        """
        # 获取场景信息
        current_scene = director.script.get(current_scene_id, {})
        current_scene_desc = current_scene.get("description", "")

        next_scene_desc = ""
        if next_scene_id:
            next_scene = director.script.get(next_scene_id, {})
            next_scene_desc = next_scene.get("description", "")

        # 构建转场描述prompt
        prompt = f"""Please provide a scene transition description based on the following information:

Current scene description:
{current_scene_desc}

Last interaction:
{last_interaction}

Next scene description (if any):
{next_scene_desc}

Please provide a natural and smooth transition description that connects the current scene to the next scene (or the end of the story if there is no next scene)."""

        # 调用API生成转场描述
        messages = [
            {"role": "system",
             "content": "You are an experienced screenwriter who specializes in creating scene transitions."},
            {"role": "user", "content": prompt}
        ]

        return handle_stream_response(self.client, use_model, messages)
from typing import List, Dict, Any
from .base_agent import BaseAgent
from models import Character, CharacterProfile, Relationship
from memory import MemoryManager


class RoleAgent(BaseAgent):
    def __init__(self, character: Character):
        super().__init__(character.profile.name)
        self.character = character
        self.memory_manager = MemoryManager(max_memories=20)

    def add_memory(self, memory_text: str):
        self.character.add_memory(memory_text)
        self.memory_manager.add_memory(memory_text)

    def add_trait(self, trait: str):
        self.character.profile.add_trait(trait)
        self.add_memory(f"我的性格特点是{trait}")

    def add_relationship(self, target_name: str, relation_type: str,
                         description: str = "", closeness: float = 0.5):
        self.character.add_relationship(target_name, relation_type, description, closeness)
        memory_text = f"我与{target_name}的关系是{relation_type}"
        if description:
            memory_text += f"：{description}"
        self.add_memory(memory_text)

    def update_relationship(self, target_name: str, delta: float, new_description: str = ""):
        self.character.update_relationship(target_name, delta, new_description)

    def make_decision(self, context: str) -> str:
        relevant_memories = self.memory_manager.retrieve_relevant_memories(context)

        traits_text = "，".join(self.character.profile.traits) if self.character.profile.traits else "无特定性格特征"

        prompt = f"""请作为角色'{self.character.profile.name}'分析当前情境并做出决策：

角色信息：
- 年龄：{self.character.profile.age}
- 性别：{self.character.profile.gender}
- 性格特征：{traits_text}

相关记忆：
{chr(10).join(relevant_memories) if relevant_memories else '无'}

当前情境：
{context}

请根据角色设定，决定角色应该采取什么行动或说什么话。请只回答决策内容，不要有任何解释。"""

        system_prompt = "你是一位专业的演员，擅长根据角色设定做出符合性格的决策。"
        return self.generate_response(system_prompt, prompt)

    def generate_dialogue(self, information: str, speaker=None, guidance: str = None) -> str:
        relevant_memories = self.memory_manager.retrieve_relevant_memories(information)

        speaker_name = speaker
        if hasattr(speaker, "name"):
            speaker_name = speaker.name

        relationship_context = ""
        if speaker_name and speaker_name in self.character.relationships:
            rel = self.character.relationships[speaker_name]
            rel_text = f"你与{speaker_name}的关系是{rel.relation_type}"
            if rel.description:
                rel_text += f"：{rel.description}"
            relationship_context = rel_text + "\n"

        for name in self.character.relationships.keys():
            if name != speaker_name and name.lower() in information.lower():
                rel = self.character.relationships[name]
                rel_text = f"你与{name}的关系是{rel.relation_type}"
                if rel.description:
                    rel_text += f"：{rel.description}"
                relationship_context += rel_text + "\n"

        traits_context = ""
        if self.character.profile.traits:
            traits_context = (
                "你的性格特征是："
                + "，".join(self.character.profile.traits)
                + "。请根据这些性格特征来塑造你的回应。\n"
            )

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

        system_prompt = f'你是一名戏剧演员，你在剧中扮演的角色叫{self.character.profile.name}，一个{self.character.profile.age}岁的{self.character.profile.gender}。请根据对话者身份和你的角色特点做出合适的回应。\n\n你的回应必须按照以下格式："（{self.character.profile.name}的表情与动作）对话内容"。括号内必须包含你扮演角色的名字作为主语，清晰描述表情和动作，括号后直接跟对话内容。'
        user_prompt = (context + information) if context else information

        response_content = self.generate_response(system_prompt, user_prompt)

        if not response_content.startswith(
            f"（{self.character.profile.name}"
        ) and not response_content.startswith(f"({self.character.profile.name}"):
            response_content = f"（{self.character.profile.name}平静地说道）{response_content}"

        if speaker_name:
            self.add_memory(f"我对{speaker_name}说：{response_content}")

        return response_content

    def should_speak(self, context: str, speaker=None) -> bool:
        relevant_memories = self.memory_manager.retrieve_relevant_memories(context)

        speaker_name = speaker
        if hasattr(speaker, "name"):
            speaker_name = speaker.name

        prompt = f"情境：{context}\n\n"

        if speaker_name:
            prompt += f"说话者：{speaker_name}\n\n"

        if relevant_memories:
            prompt += "相关记忆：\n" + "\n".join(relevant_memories) + "\n\n"

        prompt += f"作为一个名叫{self.character.profile.name}的角色，在这种情况下，我需要说话吗？请只回答'是'或'否'。"

        system_prompt = "你需要帮助角色判断在当前情境下是否应该说话。请只回答'是'或'否'。"
        decision = self.generate_response(system_prompt, prompt)

        return "是" in decision[:10] or "yes" in decision.lower()[:10]

    def get_character_info(self) -> Dict[str, Any]:
        return self.character.to_dict()

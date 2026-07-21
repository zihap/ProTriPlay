from typing import List, Dict, Any
from .base_agent import BaseAgent
from models import StoryOutline, StoryNode, Scene, SceneTransition, Character


class DirectorAgent(BaseAgent):
    def __init__(self):
        super().__init__("Director")
        self.story_outline = None
        self.scenes: Dict[str, Scene] = {}
        self.transitions: List[SceneTransition] = []
        self.current_scene_id = None
        self.characters: Dict[str, Character] = {}

    def load_story_outline(self, outline: StoryOutline):
        self.story_outline = outline

    def add_scene(self, scene: Scene):
        self.scenes[scene.scene_id] = scene

    def add_transition(self, transition: SceneTransition):
        self.transitions.append(transition)

    def add_character(self, character: Character):
        self.characters[character.profile.name] = character

    def set_current_scene(self, scene_id: str) -> bool:
        if scene_id in self.scenes:
            self.current_scene_id = scene_id
            return True
        return False

    def get_current_scene(self) -> Scene:
        if self.current_scene_id and self.current_scene_id in self.scenes:
            return self.scenes[self.current_scene_id]
        return None

    def schedule_characters(self, scene_id: str) -> List[str]:
        scene = self.scenes.get(scene_id)
        if scene:
            return scene.characters
        return []

    def determine_next_scene(self, dialogue_history: List[Dict] = None) -> str:
        if not self.current_scene_id:
            return None

        for transition in self.transitions:
            if transition.from_scene_id == self.current_scene_id:
                return transition.to_scene_id

        if self.story_outline:
            current_node = self.story_outline.get_current_node()
            if current_node and current_node.branches:
                return current_node.branches[0].get("next_scene")

        return None

    def provide_guidance(self, agent_type: str, context: str) -> str:
        system_prompt = "你是一位经验丰富的戏剧导演，擅长为编剧和演员提供创作指导。"
        user_prompt = f"""请为{agent_type}提供创作指导：

当前上下文：
{context}

请提供具体的指导建议，帮助{agent_type}更好地完成创作任务。"""

        return self.generate_response(system_prompt, user_prompt)

    def evaluate_scene_completion(self, scene_id: str, dialogue_history: List[Dict] = None) -> bool:
        scene = self.scenes.get(scene_id)
        if not scene:
            return False

        if dialogue_history is None:
            dialogue_history = self.get_dialogue_history()

        dialogue_text = "\n".join([f"{d['speaker']}: {d['content']}" for d in dialogue_history])

        system_prompt = "你是一位经验丰富的戏剧导演，擅长判断场景是否应该结束。"
        user_prompt = f"""请判断以下场景是否已经完成：

场景描述：{scene.description}
玩家目标：{scene.player_goal}

对话历史：
{dialogue_text}

请根据以下标准判断：
1. 玩家角色的目标是否已经达成
2. 场景中的关键对话是否已经完成
3. 对话是否自然到达了一个结束点

请只回答'是'或'否'，并给出简短理由。"""

        result = self.generate_response(system_prompt, user_prompt).lower()
        return "是" in result[:30] or "完成" in result[:30]

    def generate_character_schedule(self, scene_id: str) -> Dict[str, Any]:
        scene = self.scenes.get(scene_id)
        if not scene:
            return {}

        schedule = {
            "scene_id": scene_id,
            "characters": [],
            "entrance_order": [],
            "interactions": []
        }

        for char_name in scene.characters:
            character = self.characters.get(char_name)
            if character:
                schedule["characters"].append({
                    "name": char_name,
                    "traits": character.profile.traits,
                    "relationships": {
                        name: rel.relation_type
                        for name, rel in character.relationships.items()
                        if name in scene.characters
                    }
                })

        return schedule

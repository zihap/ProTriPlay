from typing import List, Dict, Any


class Relationship:
    def __init__(self, target_name: str, relation_type: str, description: str = "",
                 closeness: float = 0.5):
        self.target_name = target_name
        self.relation_type = relation_type
        self.description = description
        self.closeness = closeness

    def update_closeness(self, delta: float):
        self.closeness = max(0.0, min(1.0, self.closeness + delta))

    def update_type(self, new_type: str, new_description: str = ""):
        self.relation_type = new_type
        if new_description:
            self.description = new_description

    def to_dict(self):
        return {
            "target_name": self.target_name,
            "relation_type": self.relation_type,
            "description": self.description,
            "closeness": self.closeness
        }


class CharacterProfile:
    def __init__(self, name: str, age: int, gender: str, traits: List[str] = None,
                 background: List[str] = None, goals: List[str] = None):
        self.name = name
        self.age = age
        self.gender = gender
        self.traits = traits if traits else []
        self.background = background if background else []
        self.goals = goals if goals else []

    def add_trait(self, trait: str):
        if trait not in self.traits:
            self.traits.append(trait)

    def add_background(self, background: str):
        if background not in self.background:
            self.background.append(background)

    def add_goal(self, goal: str):
        if goal not in self.goals:
            self.goals.append(goal)

    def to_dict(self):
        return {
            "name": self.name,
            "age": self.age,
            "gender": self.gender,
            "traits": self.traits,
            "background": self.background,
            "goals": self.goals
        }


class Character:
    def __init__(self, profile: CharacterProfile):
        self.profile = profile
        self.relationships: Dict[str, Relationship] = {}
        self.memories: List[str] = []
        self.current_mood = "neutral"

    def add_relationship(self, target_name: str, relation_type: str,
                         description: str = "", closeness: float = 0.5):
        if target_name not in self.relationships:
            self.relationships[target_name] = Relationship(
                target_name, relation_type, description, closeness
            )
        else:
            self.relationships[target_name].update_type(relation_type, description)

    def update_relationship(self, target_name: str, delta: float, new_description: str = ""):
        if target_name in self.relationships:
            self.relationships[target_name].update_closeness(delta)
            if new_description:
                self.relationships[target_name].description = new_description
        else:
            self.add_relationship(target_name, "acquaintance", new_description, 0.3)

    def get_relationship(self, target_name: str):
        return self.relationships.get(target_name)

    def add_memory(self, memory_text: str):
        self.memories.append(memory_text)
        if len(self.memories) > 50:
            self.memories = self.memories[-50:]

    def get_recent_memories(self, limit: int = 10):
        return self.memories[-limit:]

    def to_dict(self):
        return {
            "profile": self.profile.to_dict(),
            "relationships": {name: rel.to_dict() for name, rel in self.relationships.items()},
            "memories": self.memories,
            "current_mood": self.current_mood
        }

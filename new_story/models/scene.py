from typing import List, Dict, Any


class Dialog:
    def __init__(self, character: str, content: str, action: str = ""):
        self.character = character
        self.content = content
        self.action = action

    def to_dict(self):
        return {
            "character": self.character,
            "content": self.content,
            "action": self.action
        }


class Scene:
    def __init__(self, scene_id: str, description: str, characters: List[str] = None,
                 dialogs: List[Dialog] = None, player_goal: str = ""):
        self.scene_id = scene_id
        self.description = description
        self.characters = characters if characters else []
        self.dialogs = dialogs if dialogs else []
        self.player_goal = player_goal

    def add_dialog(self, dialog: Dialog):
        self.dialogs.append(dialog)

    def add_character(self, character_name: str):
        if character_name not in self.characters:
            self.characters.append(character_name)

    def to_dict(self):
        return {
            "scene_id": self.scene_id,
            "description": self.description,
            "characters": self.characters,
            "dialogs": [dialog.to_dict() for dialog in self.dialogs],
            "player_goal": self.player_goal
        }


class SceneTransition:
    def __init__(self, from_scene_id: str, to_scene_id: str, condition: str = ""):
        self.from_scene_id = from_scene_id
        self.to_scene_id = to_scene_id
        self.condition = condition

    def to_dict(self):
        return {
            "from_scene_id": self.from_scene_id,
            "to_scene_id": self.to_scene_id,
            "condition": self.condition
        }

import pytest
import json
import os
from unittest.mock import patch, MagicMock
from agents import DirectorAgent, ScreenwriterAgent, RoleAgent, ProtagonistAgent
from models import Character, CharacterProfile, Scene, SceneTransition


@patch("agents.base_agent.handle_stream_response")
def test_full_story_generation(mock_handle):
    mock_handle.side_effect = [
        "A dark forest at dusk with shadows dancing between ancient trees.",
        '{"key_elements": ["trees", "shadows"], "mood": "mysterious"}',
        "I wander through the forest, sensing something ancient nearby.",
        "OldMan: Greetings, traveler. You seek the temple, do you not?",
        "Ancient temple entrance with massive stone pillars.",
        '{"key_elements": ["pillars", "entrance"], "mood": "majestic"}',
        "I stand before the temple, ready to face whatever lies within.",
        "Guardian: Only the worthy may enter. Prove your intentions."
    ]
    
    director = DirectorAgent()
    screenwriter = ScreenwriterAgent()
    
    script = {
        "scene_1": {
            "description": "A dark forest at dusk",
            "characters": ["Hero", "OldMan"],
            "dialogs": []
        },
        "scene_2": {
            "description": "Ancient temple entrance",
            "characters": ["Hero", "Guardian"],
            "dialogs": []
        }
    }
    
    screenwriter.load_initial_script(script)
    
    hero_profile = CharacterProfile("Hero", 25, "男", ["brave", "curious", "determined"])
    hero_character = Character(hero_profile)
    protagonist = ProtagonistAgent(hero_character)
    protagonist.add_special_ability("感知", "能够感知周围的危险")
    
    oldman_profile = CharacterProfile("OldMan", 70, "男", ["wise", "mysterious"])
    oldman_character = Character(oldman_profile)
    oldman = RoleAgent(oldman_character)
    
    guardian_profile = CharacterProfile("Guardian", 50, "男", ["noble", "protective"])
    guardian_character = Character(guardian_profile)
    guardian = RoleAgent(guardian_character)
    
    director.add_character(hero_character)
    director.add_character(oldman_character)
    director.add_character(guardian_character)
    
    scene1 = Scene("scene_1", "A dark forest at dusk", ["Hero", "OldMan"])
    scene2 = Scene("scene_2", "Ancient temple entrance", ["Hero", "Guardian"])
    director.add_scene(scene1)
    director.add_scene(scene2)
    director.add_transition(SceneTransition("scene_1", "scene_2"))
    
    scene_ids = ["scene_1", "scene_2"]
    story_output = []
    
    for scene_id in scene_ids:
        director.set_current_scene(scene_id)
        detailed_setting = screenwriter.generate_scene_setting(scene_id, director, "Hero")
        
        protagonist.observe_scene(detailed_setting, director.schedule_characters(scene_id))
        
        scene_data = {
            "scene_id": scene_id,
            "description": detailed_setting,
            "characters": director.schedule_characters(scene_id),
            "dialogues": []
        }
        
        scene_characters = [c for c in director.schedule_characters(scene_id) if c != "Hero"]
        
        for char_name in scene_characters:
            role_agent = oldman if char_name == "OldMan" else guardian
            
            hero_line = protagonist.drive_story(detailed_setting)
            scene_data["dialogues"].append({
                "speaker": "Hero",
                "content": hero_line
            })
            
            response = role_agent.generate_dialogue(hero_line, "Hero")
            scene_data["dialogues"].append({
                "speaker": char_name,
                "content": response
            })
            
            protagonist.update_growth(f"与{char_name}对话", 0.1)
        
        story_output.append(scene_data)
    
    assert len(story_output) == 2
    assert len(story_output[0]["dialogues"]) >= 2
    assert len(story_output[1]["dialogues"]) >= 2
    assert protagonist.growth_points > 0
    
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "story_output", "test_story.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(story_output, f, ensure_ascii=False, indent=2)
    
    assert os.path.exists(output_path)


@patch("agents.base_agent.handle_stream_response")
def test_character_consistency(mock_handle):
    mock_handle.return_value = "I face the danger with courage, determined to protect those around me."
    
    hero_profile = CharacterProfile("Hero", 25, "男", ["brave", "honest"])
    hero_character = Character(hero_profile)
    protagonist = ProtagonistAgent(hero_character)
    
    for _ in range(3):
        dialogue = protagonist.drive_story("You are in a dangerous situation")
        assert isinstance(dialogue, str)
        assert len(dialogue) > 0


def test_scene_progression():
    director = DirectorAgent()
    screenwriter = ScreenwriterAgent()
    
    script = {
        "scene_1": {"description": "Start", "characters": ["Hero"]},
        "scene_2": {"description": "Middle", "characters": ["Hero"]},
        "scene_3": {"description": "End", "characters": ["Hero"]}
    }
    
    screenwriter.load_initial_script(script)
    
    for i in range(1, 4):
        scene_id = f"scene_{i}"
        scene = Scene(scene_id, script[scene_id]["description"], ["Hero"])
        director.add_scene(scene)
        
        if i < 3:
            director.add_transition(SceneTransition(scene_id, f"scene_{i+1}"))
    
    director.set_current_scene("scene_1")
    
    for _ in range(2):
        next_scene = director.determine_next_scene()
        assert next_scene is not None
        director.set_current_scene(next_scene)
    
    assert director.current_scene_id == "scene_3"
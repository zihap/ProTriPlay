import pytest
from unittest.mock import patch, MagicMock
from agents import DirectorAgent, ScreenwriterAgent, RoleAgent, ProtagonistAgent
from models import Character, CharacterProfile, Scene, SceneTransition, StoryOutline, StoryNode


@patch("agents.base_agent.handle_stream_response")
def test_director_screenwriter_collaboration(mock_handle):
    mock_handle.return_value = "A mysterious room with ancient artifacts and dim candlelight."
    
    director = DirectorAgent()
    screenwriter = ScreenwriterAgent()
    
    script = {
        "scene_1": {
            "description": "A mysterious room",
            "characters": ["Hero", "Guard"],
            "dialogs": []
        }
    }
    
    screenwriter.load_initial_script(script)
    scene = Scene("scene_1", "A mysterious room", ["Hero", "Guard"])
    director.add_scene(scene)
    
    detailed_setting = screenwriter.generate_scene_setting("scene_1", director)
    assert isinstance(detailed_setting, str)
    assert len(detailed_setting) > 0
    
    director.set_current_scene("scene_1")
    assert director.current_scene_id == "scene_1"


@patch("agents.base_agent.handle_stream_response")
def test_role_protagonist_interaction(mock_handle):
    mock_handle.side_effect = [
        "I approach the guard and ask about the castle entrance.",
        "Guard: Halt! Who goes there? State your business!"
    ]
    
    hero_profile = CharacterProfile("Hero", 25, "男", ["brave", "curious"])
    hero_character = Character(hero_profile)
    protagonist = ProtagonistAgent(hero_character)
    
    guard_profile = CharacterProfile("Guard", 40, "男", ["loyal", "suspicious"])
    guard_character = Character(guard_profile)
    guard = RoleAgent(guard_character)
    
    protagonist.add_relationship("Guard", "stranger", "初次见面")
    guard.add_relationship("Hero", "stranger", "可疑的闯入者")
    
    hero_dialogue = protagonist.drive_story("You meet a guard at the gate")
    assert isinstance(hero_dialogue, str)
    
    guard_response = guard.generate_dialogue(hero_dialogue, "Hero")
    assert isinstance(guard_response, str)
    assert "Guard" in guard_response


def test_scene_transition_flow():
    director = DirectorAgent()
    
    scene1 = Scene("scene_1", "Room 1", ["Hero"])
    scene2 = Scene("scene_2", "Room 2", ["Hero", "NPC"])
    director.add_scene(scene1)
    director.add_scene(scene2)
    
    transition = SceneTransition("scene_1", "scene_2", "Hero opens the door")
    director.add_transition(transition)
    
    director.set_current_scene("scene_1")
    next_scene = director.determine_next_scene()
    assert next_scene == "scene_2"
    
    assert director.set_current_scene("scene_2")
    assert director.current_scene_id == "scene_2"


def test_story_outline_with_director():
    outline = StoryOutline("Adventure Story", "A fantasy adventure")
    node1 = StoryNode("node_1", "Beginning", [{"next_scene": "scene_1"}])
    node2 = StoryNode("node_2", "Climax", [{"next_scene": "scene_2"}])
    outline.add_node(node1)
    outline.add_node(node2)
    
    director = DirectorAgent()
    director.load_story_outline(outline)
    
    assert director.story_outline.title == "Adventure Story"
    assert director.story_outline.get_current_node().node_id == "node_1"


def test_character_relationship_development():
    hero_profile = CharacterProfile("Hero", 25, "男")
    hero_character = Character(hero_profile)
    protagonist = ProtagonistAgent(hero_character)
    
    ally_profile = CharacterProfile("Ally", 30, "女")
    ally_character = Character(ally_profile)
    ally = RoleAgent(ally_character)
    
    protagonist.add_relationship("Ally", "acquaintance", "刚认识", 0.3)
    ally.add_relationship("Hero", "acquaintance", "刚认识", 0.3)
    
    protagonist.update_relationship("Ally", 0.2, "一起经历了战斗")
    ally.update_relationship("Hero", 0.2, "一起经历了战斗")
    
    assert protagonist.character.relationships["Ally"].closeness == 0.5
    assert ally.character.relationships["Hero"].closeness == 0.5
    
    protagonist.update_relationship("Ally", 0.3, "成为了亲密的伙伴")
    assert protagonist.character.relationships["Ally"].closeness == 0.8
    assert protagonist.character.relationships["Ally"].description == "成为了亲密的伙伴"


@patch("agents.base_agent.handle_stream_response")
def test_full_scene_flow(mock_handle):
    mock_handle.side_effect = [
        "A cozy tavern with wooden tables and a roaring fire.",
        '{"key_elements": ["fire", "bar"], "mood": "warm"}',
        "I walk to the bar and order a drink.",
        "Bartender: Welcome, traveler! What'll it be?"
    ]
    
    director = DirectorAgent()
    screenwriter = ScreenwriterAgent()
    
    script = {
        "scene_1": {
            "description": "Tavern interior",
            "characters": ["Hero", "Bartender"],
            "dialogs": []
        }
    }
    
    screenwriter.load_initial_script(script)
    
    hero_profile = CharacterProfile("Hero", 25, "男", ["adventurous"])
    hero_character = Character(hero_profile)
    protagonist = ProtagonistAgent(hero_character)
    
    bartender_profile = CharacterProfile("Bartender", 45, "男", ["friendly"])
    bartender_character = Character(bartender_profile)
    bartender = RoleAgent(bartender_character)
    
    director.add_character(hero_character)
    director.add_character(bartender_character)
    
    scene = Scene("scene_1", "Tavern interior", ["Hero", "Bartender"])
    director.add_scene(scene)
    director.set_current_scene("scene_1")
    
    detailed_setting = screenwriter.generate_scene_setting("scene_1", director, "Hero")
    assert len(detailed_setting) > 0
    
    protagonist.observe_scene(detailed_setting, ["Bartender"])
    
    hero_line = protagonist.drive_story(detailed_setting)
    assert len(hero_line) > 0
    
    bartender_response = bartender.generate_dialogue(hero_line, "Hero")
    assert "Bartender" in bartender_response
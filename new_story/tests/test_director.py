import pytest
from agents import DirectorAgent
from models import Scene, SceneTransition, Character, CharacterProfile, StoryOutline, StoryNode


def test_director_init():
    director = DirectorAgent()
    assert director.name == "Director"
    assert len(director.scenes) == 0


def test_director_add_scene():
    director = DirectorAgent()
    scene = Scene("scene_1", "Test scene", ["Char1"])
    director.add_scene(scene)
    
    assert "scene_1" in director.scenes
    assert director.scenes["scene_1"].description == "Test scene"


def test_director_set_current_scene():
    director = DirectorAgent()
    scene = Scene("scene_1", "Test scene")
    director.add_scene(scene)
    
    assert director.set_current_scene("scene_1")
    assert director.current_scene_id == "scene_1"
    assert not director.set_current_scene("nonexistent")


def test_director_add_character():
    director = DirectorAgent()
    profile = CharacterProfile("TestChar", 25, "男")
    character = Character(profile)
    director.add_character(character)
    
    assert "TestChar" in director.characters


def test_director_add_transition():
    director = DirectorAgent()
    transition = SceneTransition("scene_1", "scene_2", "condition")
    director.add_transition(transition)
    
    assert len(director.transitions) == 1
    assert director.transitions[0].to_scene_id == "scene_2"


def test_director_schedule_characters():
    director = DirectorAgent()
    scene = Scene("scene_1", "Test scene", ["Char1", "Char2"])
    director.add_scene(scene)
    
    chars = director.schedule_characters("scene_1")
    assert len(chars) == 2
    assert "Char1" in chars


def test_director_determine_next_scene():
    director = DirectorAgent()
    scene1 = Scene("scene_1", "Test scene 1")
    scene2 = Scene("scene_2", "Test scene 2")
    director.add_scene(scene1)
    director.add_scene(scene2)
    director.add_transition(SceneTransition("scene_1", "scene_2"))
    director.set_current_scene("scene_1")
    
    next_scene = director.determine_next_scene()
    assert next_scene == "scene_2"

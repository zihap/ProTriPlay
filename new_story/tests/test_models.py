import pytest
from models import StoryNode, StoryOutline, Scene, Dialog, CharacterProfile, Character, Relationship


def test_story_node():
    node = StoryNode("node_1", "test node", [{"next": "node_2"}])
    assert node.node_id == "node_1"
    assert node.description == "test node"
    assert len(node.branches) == 1


def test_story_outline():
    outline = StoryOutline("Test Story", "Test Background")
    node1 = StoryNode("node_1", "Node 1")
    node2 = StoryNode("node_2", "Node 2")
    outline.add_node(node1)
    outline.add_node(node2)
    
    assert outline.title == "Test Story"
    assert len(outline.main_nodes) == 2
    assert outline.get_current_node().node_id == "node_1"
    assert outline.advance_node()
    assert outline.get_current_node().node_id == "node_2"


def test_scene():
    dialog = Dialog("Character1", "Hello", "waves")
    scene = Scene("scene_1", "Test scene", ["Character1", "Character2"], [dialog])
    
    assert scene.scene_id == "scene_1"
    assert len(scene.characters) == 2
    assert len(scene.dialogs) == 1
    assert scene.dialogs[0].character == "Character1"


def test_character_profile():
    profile = CharacterProfile("TestChar", 25, "男", ["brave"], ["background"], ["goal"])
    assert profile.name == "TestChar"
    assert profile.age == 25
    assert "brave" in profile.traits


def test_character_relationships():
    profile = CharacterProfile("TestChar", 25, "男")
    character = Character(profile)
    
    character.add_relationship("Other", "friend", "close friend", 0.8)
    assert "Other" in character.relationships
    assert character.relationships["Other"].relation_type == "friend"
    
    character.update_relationship("Other", 0.1)
    assert character.relationships["Other"].closeness == 0.9


def test_character_memories():
    profile = CharacterProfile("TestChar", 25, "男")
    character = Character(profile)
    
    character.add_memory("memory 1")
    character.add_memory("memory 2")
    assert len(character.memories) == 2
    assert "memory 1" in character.memories

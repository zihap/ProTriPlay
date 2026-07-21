import pytest
from unittest.mock import patch, MagicMock
from agents import ProtagonistAgent
from models import Character, CharacterProfile


def test_protagonist_init():
    profile = CharacterProfile("Hero", 25, "男")
    character = Character(profile)
    protagonist = ProtagonistAgent(character)
    
    assert protagonist.name == "Hero"
    assert protagonist.growth_points == 0


def test_protagonist_add_special_ability():
    profile = CharacterProfile("Hero", 25, "男")
    character = Character(profile)
    protagonist = ProtagonistAgent(character)
    
    protagonist.add_special_ability("洞察力", "能够发现隐藏的线索")
    assert len(protagonist.special_abilities) == 1
    assert protagonist.special_abilities[0]["name"] == "洞察力"


def test_protagonist_upgrade_ability():
    profile = CharacterProfile("Hero", 25, "男")
    character = Character(profile)
    protagonist = ProtagonistAgent(character)
    
    protagonist.add_special_ability("洞察力")
    protagonist.upgrade_ability("洞察力")
    assert protagonist.special_abilities[0]["level"] == 2


def test_protagonist_update_growth():
    profile = CharacterProfile("Hero", 25, "男")
    character = Character(profile)
    protagonist = ProtagonistAgent(character)
    
    protagonist.update_growth("经历了一场战斗", 0.3)
    assert protagonist.growth_points == 0.3
    assert protagonist.get_growth_stage() == "成长阶段"


def test_protagonist_growth_stages():
    profile = CharacterProfile("Hero", 25, "男")
    character = Character(profile)
    protagonist = ProtagonistAgent(character)
    
    assert protagonist.get_growth_stage() == "初始阶段"
    
    protagonist.update_growth("event1", 0.35)
    assert protagonist.get_growth_stage() == "成长阶段"
    
    protagonist.update_growth("event2", 0.35)
    assert protagonist.get_growth_stage() == "成熟阶段"
    
    protagonist.update_growth("event3", 0.35)
    assert protagonist.get_growth_stage() == "巅峰阶段"


@patch("agents.base_agent.handle_stream_response")
def test_protagonist_observe_scene(mock_handle):
    mock_handle.return_value = '{"key_elements": ["darkness", "door"], "mood": "suspenseful"}'
    
    profile = CharacterProfile("Hero", 25, "男")
    character = Character(profile)
    protagonist = ProtagonistAgent(character)
    
    observation = protagonist.observe_scene("A dark room", ["Enemy"])
    assert isinstance(observation, dict)
    assert "key_elements" in observation


@patch("agents.base_agent.handle_stream_response")
def test_protagonist_drive_story(mock_handle):
    mock_handle.return_value = "I walk towards the castle gates, determined to uncover the secrets within."
    
    profile = CharacterProfile("Hero", 25, "男")
    character = Character(profile)
    protagonist = ProtagonistAgent(character)
    
    dialogue = protagonist.drive_story("You are in a castle")
    assert isinstance(dialogue, str)
    assert len(dialogue) > 0


def test_protagonist_set_perspective():
    profile = CharacterProfile("Hero", 25, "男")
    character = Character(profile)
    protagonist = ProtagonistAgent(character)
    
    protagonist.set_narrative_perspective("third_person_limited")
    assert protagonist.narrative_perspective == "third_person_limited"
    
    protagonist.set_narrative_perspective("invalid")
    assert protagonist.narrative_perspective == "third_person_limited"
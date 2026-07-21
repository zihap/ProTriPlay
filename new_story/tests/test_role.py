import pytest
from unittest.mock import patch, MagicMock
from agents import RoleAgent
from models import Character, CharacterProfile


def test_role_agent_init():
    profile = CharacterProfile("TestChar", 25, "男")
    character = Character(profile)
    agent = RoleAgent(character)
    
    assert agent.name == "TestChar"
    assert agent.character.profile.name == "TestChar"


def test_role_agent_add_memory():
    profile = CharacterProfile("TestChar", 25, "男")
    character = Character(profile)
    agent = RoleAgent(character)
    
    agent.add_memory("test memory")
    assert len(agent.character.memories) == 1
    assert "test memory" in agent.character.memories


def test_role_agent_add_trait():
    profile = CharacterProfile("TestChar", 25, "男")
    character = Character(profile)
    agent = RoleAgent(character)
    
    agent.add_trait("brave")
    assert "brave" in agent.character.profile.traits


def test_role_agent_add_relationship():
    profile = CharacterProfile("TestChar", 25, "男")
    character = Character(profile)
    agent = RoleAgent(character)
    
    agent.add_relationship("Other", "friend", "close", 0.8)
    assert "Other" in agent.character.relationships
    assert agent.character.relationships["Other"].relation_type == "friend"


def test_role_agent_update_relationship():
    profile = CharacterProfile("TestChar", 25, "男")
    character = Character(profile)
    agent = RoleAgent(character)
    
    agent.add_relationship("Other", "friend", "close", 0.5)
    agent.update_relationship("Other", 0.2)
    assert agent.character.relationships["Other"].closeness == 0.7


@patch("agents.base_agent.handle_stream_response")
def test_role_agent_should_speak(mock_handle):
    mock_handle.return_value = "Yes"
    
    profile = CharacterProfile("TestChar", 25, "男")
    character = Character(profile)
    agent = RoleAgent(character)
    
    result = agent.should_speak("Someone says hello to you")
    assert isinstance(result, bool)


@patch("agents.base_agent.handle_stream_response")
def test_role_agent_make_decision(mock_handle):
    mock_handle.return_value = "Open the door"
    
    profile = CharacterProfile("TestChar", 25, "男")
    character = Character(profile)
    agent = RoleAgent(character)
    
    decision = agent.make_decision("You see a door")
    assert isinstance(decision, str)
    assert len(decision) > 0
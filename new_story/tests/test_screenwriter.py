import pytest
from unittest.mock import patch, MagicMock
from agents import ScreenwriterAgent
from models import Scene


def test_screenwriter_init():
    screenwriter = ScreenwriterAgent()
    assert screenwriter.name == "Screenwriter"
    assert len(screenwriter.scene_descriptions) == 0


def test_screenwriter_load_script():
    screenwriter = ScreenwriterAgent()
    script = {
        "scene_1": {
            "description": "Test scene",
            "characters": ["Char1"],
            "dialogs": []
        }
    }
    screenwriter.load_initial_script(script)
    
    assert "scene_1" in screenwriter.initial_script
    assert screenwriter.scene_descriptions["scene_1"] == "Test scene"


@patch("agents.base_agent.handle_stream_response")
def test_screenwriter_generate_scene_setting(mock_handle):
    mock_handle.return_value = "A detailed description of a simple room with wooden furniture."
    
    screenwriter = ScreenwriterAgent()
    script = {
        "scene_1": {"description": "A simple room", "characters": ["Char1"]}
    }
    screenwriter.load_initial_script(script)
    
    setting = screenwriter.generate_scene_setting("scene_1")
    assert isinstance(setting, str)
    assert len(setting) > 0
    assert "room" in setting.lower()


@patch("agents.base_agent.handle_stream_response")
def test_screenwriter_generate_script(mock_handle):
    mock_handle.return_value = "Scene: scene_1\nCharacters: Char1\nDescription: A simple room\nDialogs:\nChar1: Hello!"
    
    screenwriter = ScreenwriterAgent()
    script = {
        "scene_1": {"description": "A simple room", "characters": ["Char1"]}
    }
    screenwriter.load_initial_script(script)
    
    scene = screenwriter.generate_script("scene_1")
    assert isinstance(scene, Scene)
    assert scene.scene_id == "scene_1"


def test_screenwriter_add_dialogue_record():
    screenwriter = ScreenwriterAgent()
    screenwriter.add_dialogue_record("Char1", "对话", "Hello", "Char2")
    
    history = screenwriter.get_dialogue_history()
    assert len(history) == 1
    assert history[0]["speaker"] == "Char1"
    assert history[0]["target"] == "Char2"
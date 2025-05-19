This repository contains a series of Python scripts for ProTriPlay (A Trinity Framework for Professional Interactive Drama Based on LLMs), designed to interact with multiple AI models, manage dramatic characters, and simulate dialogue within a narrative structure. The scripts integrate AI functionalities such as OpenAI, DeepSeek, and Qwen for generating responses and maintaining character memories.

## File Overview

### 1. `config.py`
This configuration file contains API keys and model settings necessary for connecting to different AI services.

#### Key Variables:
- **openai_api_key**: Your OpenAI API key.
- **deepseek_api_key**: Your DeepSeek API key.
- **qwen_api_key**: Your Qwen API key.
- **http_proxy, https_proxy**: Proxy configuration.
- **use_model**: The AI model to use (`deepseek-chat`, `gpt-4o-mini`, `qwen3-235b-a22b`).
- **try_chance, max_new_scene_generations, max_inserted_scenes**: Configuration related to testing and scene generation.

### 2. `role.py`
This file defines the core logic for actor interactions, memory management, and scene dynamics. It includes the `Actor`, `Player`, and `Director` classes to manage character states and narrative.

#### Key Classes:
- **Actor**: The Actor Agent, manages character attributes, memories, relationships, and personality traits. Interacts with AI models to generate dialogue and responses.
- **Player**: Represents the player character, which can interact with NPCs (Actors).
- **Director**: The Director Agent manages the overall narrative, controlling scene flow and interactions between characters.
- **Screenwriter**: The Screenwriter Agent is responsible for script generation and creating dialogues based on predefined scripts.

### 3. `my_test.py`
This script simulates a role-playing scenario with memory and trait management, where an AI-driven player character interacts with NPCs.

#### Main Features:
- **Character Management**: Defines `Actor` and `Player` classes to manage character attributes, memories, and relationships.
- **Scene Simulation**: Uses the `Director` class to manage scenes, characters, and dialogues, simulating interactions and pushing the narrative forward.
- **Scene Generation**: Automatically generates new scenes based on character interactions.

### 4. `gui_test.py`
This script provides a graphical user interface (GUI) based on Tkinter, allowing users to interact with the AI system by sending and receiving messages.

#### Main Features:
- **API Key Management**: Allows configuring API keys for OpenAI, DeepSeek, and Qwen.
- **Proxy Configuration**: Allows setting up HTTP/HTTPS proxies.
- **Model Selection**: Users can choose between different AI models (DeepSeek, GPT-4, Qwen).
- **Memory and Response Handling**: Supports handling both streaming and non-streaming AI responses.

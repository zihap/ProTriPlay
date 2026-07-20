import tkinter as tk
from tkinter import scrolledtext, ttk, messagebox
import threading
import sys
import io
import os
from dotenv import load_dotenv

load_dotenv()

# API key configuration
openai_api_key = "<Your-API-KEY>"
deepseek_api_key = "<Your-API-KEY>"
qwen_api_key = "<Your-API-KEY>"
ark_api_key = os.getenv("ARK_API_KEY", "<Your-Ark-API-KEY>")
ark_base_url = os.getenv("API_URL", "https://ark.cn-beijing.volces.com/api/plan/v3")
ark_model = "deepseek-v4-pro"
ark_embedding_model = "doubao-embedding-vision"
ark_embedding_dim = 2048

# Proxy configuration
http_proxy = ""
https_proxy = ""

# Model configuration
use_model = "ark"  # Optional: "gpt-4o-mini", "deepseek-chat", "qwen3-235b-a22b", "ark"

# Test configuration
try_chance = 2  # Number of scenario loops/attempts
max_new_scene_generations = 1  # Maximum number of new scenarios allowed to be generated
max_inserted_scenes = 1  # Maximum number of new scenarios allowed to be inserted

import openai
import faiss
import numpy as np
from typing import List, Dict, Tuple
import pickle
import os
from openai import OpenAI

# Set proxy
os.environ["http_proxy"] = http_proxy
os.environ["https_proxy"] = https_proxy


def parse_ark_response(response):
    if hasattr(response, 'output_text') and response.output_text:
        return response.output_text
    if hasattr(response, 'output') and isinstance(response.output, list):
        for item in response.output:
            if hasattr(item, 'type') and item.type == 'output_text':
                if hasattr(item, 'text'):
                    return item.text
                if hasattr(item, 'content'):
                    return item.content
    return str(response)


def handle_stream_response(client, model, messages, extra_body=None):
    """General helper function to handle streaming and non-streaming responses

    Parameters:
        client: OpenAI client
        model: Model name
        messages: List of messages
        extra_body: Additional request body parameters

    Returns:
        Response text from the model
    """
    global use_model
    model = use_model

    if extra_body is None:
        extra_body = {}

    if use_model == "ark":
        response = client.responses.create(
            model=ark_model,
            input=messages,
            extra_body=extra_body
        )
        return parse_ark_response(response)
    else:
        if model.startswith("qwen"):
            extra_body["enable_thinking"] = False

        if model == "qwen3-235b-a22b":
            response_content = ""
            response_stream = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                extra_body=extra_body
            )

            for chunk in response_stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    response_content += chunk.choices[0].delta.content

            return response_content
        else:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                extra_body=extra_body
            )

            return response.choices[0].message.content


class Actor:
    def __init__(self, name, age, gender, memory_path=None):
        global use_model
        
        self.name = name
        self.age = age
        self.gender = gender
        self.embedding_dim = ark_embedding_dim if use_model == "ark" else 1536
        self.memories = []
        self.memory_embeddings = None
        self.index = None
        self.relationships = {}  # Store relationships with other Actors
        self.traits = []  # Store character traits

        if use_model == "ark":
            self.talk_client = OpenAI(
                api_key=ark_api_key,
                base_url=ark_base_url
            )
            self.embedding_client = OpenAI(
                api_key=ark_api_key,
                base_url=ark_base_url
            )
        else:
            self.embedding_client = OpenAI(
                api_key=openai_api_key,
                base_url="https://api.openai.com/v1"  # OpenAI's official API endpoint
            )

            if use_model == "gpt-4o-mini":
                self.talk_client = OpenAI(
                    api_key=openai_api_key,
                    base_url="https://api.openai.com/v1"  # OpenAI's official API endpoint
                )
            elif use_model == "deepseek-chat":
                self.talk_client = OpenAI(
                    api_key=deepseek_api_key,
                    base_url="https://api.deepseek.com/v1"  # Deepseek's API endpoint
                )
            elif use_model == "qwen3-235b-a22b":
                self.talk_client = OpenAI(
                    api_key=qwen_api_key,
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # Qwen's API endpoint
                )

        self._initialize_memory(memory_path)

    def _initialize_memory(self, memory_path):
        """Initialize the memory system"""
        if memory_path and os.path.exists(memory_path):
            with open(memory_path, 'rb') as f:
                saved_data = pickle.load(f)
                self.memories = saved_data.get('memories', [])
                self.memory_embeddings = saved_data.get('embeddings')
                self.relationships = saved_data.get('relationships', {})
                self.traits = saved_data.get('traits', [])  # Load character traits

        if self.memory_embeddings is None or len(self.memories) == 0:
            self.memory_embeddings = np.zeros((0, self.embedding_dim), dtype=np.float32)

        self.index = faiss.IndexFlatL2(self.embedding_dim)
        if len(self.memories) > 0:
            self.index.add(self.memory_embeddings)

    def __str__(self):
        return f"Actor(name={self.name}, age={self.age}, gender={self.gender})"

    def add_memory(self, memory_text: str):
        """Add a new memory to the memory bank"""
        embedding = self._get_embedding(memory_text)
        self.memories.append(memory_text)
        if len(self.memories) == 1:
            self.memory_embeddings = embedding.reshape(1, -1)
        else:
            self.memory_embeddings = np.vstack([self.memory_embeddings, embedding])

        self.index.reset()
        self.index.add(self.memory_embeddings)
        return len(self.memories) - 1

    def _get_embedding(self, text: str) -> np.ndarray:
        if use_model == "ark":
            try:
                response = self.embedding_client.embeddings.create(
                    model=ark_embedding_model,
                    input=text
                )
                return np.array(response.data[0].embedding, dtype=np.float32)
            except Exception as e:
                print(f"火山方舟Embedding API调用失败: {str(e)}")
                np.random.seed(hash(text) % 4294967295)
                return np.random.rand(self.embedding_dim).astype(np.float32)
        else:
            try:
                response = self.embedding_client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=text
                )
                return np.array(response.data[0].embedding, dtype=np.float32)
            except Exception as e:
                print(f"Embedding API调用失败，使用本地随机向量替代: {str(e)}")
                np.random.seed(hash(text) % 4294967295)
                return np.random.rand(self.embedding_dim).astype(np.float32)

    def retrieve_relevant_memories(self, query: str, k: int = 3) -> List[str]:
        """Retrieve memories relevant to the query"""
        if len(self.memories) == 0:
            return []

        query_embedding = self._get_embedding(query).reshape(1, -1)
        scores, indices = self.index.search(query_embedding, min(k, len(self.memories)))
        return [self.memories[idx] for idx in indices[0]]

    def add_trait(self, trait_description):
        """Add a character trait to the character

        Parameters:
            trait_description: Description of the character trait, e.g., "cautious", "impulsive", "kind"
        """
        if trait_description not in self.traits:
            self.traits.append(trait_description)
            # Add the character trait as a memory
            self.add_memory(f"My character trait is {trait_description}")
        return len(self.traits) - 1

    def get_traits(self):
        """Get all character traits of the character

        Returns:
            List of character traits
        """
        return self.traits

    def save_memories(self, path: str):
        """Save memories, relationships, and character traits to a file"""
        with open(path, 'wb') as f:
            pickle.dump({
                'memories': self.memories,
                'embeddings': self.memory_embeddings,
                'relationships': self.relationships,
                'traits': self.traits  # Save character traits
            }, f)

    def add_relationship(self, other_actor, relationship_type, description=""):
        """Add a relationship with another Actor

        Parameters:
            other_actor: Another Actor object or Actor name
            relationship_type: Type of relationship, e.g., "friend", "family", "colleague"
            description: Relationship description, which can provide more detailed relationship information
        """
        other_name = other_actor.name if hasattr(other_actor, 'name') else other_actor

        if other_name not in self.relationships:
            self.relationships[other_name] = []

        # Check if the same type of relationship already exists
        for i, rel in enumerate(self.relationships[other_name]):
            if rel['type'] == relationship_type:
                # Update the existing relationship
                self.relationships[other_name][i] = {
                    'type': relationship_type,
                    'description': description
                }
                return

        # Add a new relationship
        self.relationships[other_name].append({
            'type': relationship_type,
            'description': description
        })

        # Add the relationship as a memory
        memory_text = f"My relationship with {other_name} is {relationship_type}"
        if description:
            memory_text += f": {description}"
        self.add_memory(memory_text)

    def get_relationship(self, other_actor):
        """Get relationship information with the specified Actor"""
        other_name = other_actor.name if hasattr(other_actor, 'name') else other_actor
        return self.relationships.get(other_name, [])

    def get_all_relationships(self):
        """Get all relationship information"""
        return self.relationships

    def speak(self, information, speaker=None, guidance=None):
        """Answer questions based on memories and relationships using Deepseek's API

        Parameters:
            information: Conversation content or question
            speaker: Person talking to the character (can be a Player object or string name)
            guidance: Performance guidance provided by the director (optional)
        """
        # Retrieve relevant memories
        relevant_memories = self.retrieve_relevant_memories(information)

        # Extract speaker information
        speaker_name = speaker
        if hasattr(speaker, 'name'):
            speaker_name = speaker.name

        # Check if it involves a relationship query
        relationship_context = ""
        # Check the relationship with the current speaker
        if speaker_name and speaker_name in self.relationships:
            relationships = self.get_relationship(speaker_name)
            if relationships:
                rel_texts = []
                for rel in relationships:
                    rel_type = rel['type']
                    rel_desc = rel['description']
                    if rel_desc:
                        rel_text = f"Your relationship with {speaker_name} is {rel_type}: {rel_desc}"
                    else:
                        rel_text = f"Your relationship with {speaker_name} is {rel_type}"
                    rel_texts.append(rel_text)
                relationship_context += "\n".join(rel_texts) + "\n"

        # Check relationships of other characters mentioned in the conversation
        for name in self.relationships.keys():
            if name != speaker_name and name.lower() in information.lower():
                relationships = self.get_relationship(name)
                if relationships:
                    rel_texts = []
                    for rel in relationships:
                        rel_type = rel['type']
                        rel_desc = rel['description']
                        if rel_desc:
                            rel_text = f"Your relationship with {name} is {rel_type}: {rel_desc}"
                        else:
                            rel_text = f"Your relationship with {name} is {rel_type}"
                        rel_texts.append(rel_text)
                    relationship_context += "\n".join(rel_texts) + "\n"

        # Add character trait information
        traits_context = ""
        if self.traits:
            traits_context = "Your character traits are: " + ", ".join(
                self.traits) + ". Please shape your response based on these character traits.\n"

        # Build the context
        context = ""
        if relevant_memories or relationship_context or speaker_name or traits_context:
            context = "Answer based on the following information:\n"
            if speaker_name:
                context += f"The person talking to you is: {speaker_name}\n\n"
            if relationship_context:
                context += relationship_context + "\n"
            if traits_context:
                context += traits_context + "\n"
            if relevant_memories:
                context += "\n".join(relevant_memories) + "\n"
            context += "\nQuestion: "

        # If there is director guidance, add it
        if guidance:
            context += "\nDirector's guidance: " + guidance

        # Add the conversation record to memory
        if speaker_name:
            self.add_memory(f"{speaker_name} said to me: {information}")

        # Use the global model variable
        global use_model

        # Build the message list
        messages = [
            {"role": "system",
             "content": f"You are a theater actor, and you are playing the character named {self.name}, a {self.age}-year-old {self.gender}. Please make an appropriate response based on the identity of the conversationalist and your character traits.\n\nYour response must follow the following format: \"({self.name}'s expression and actions) Conversation content\". The parentheses must contain the name of the character you are playing as the subject, clearly describing the expression and actions, and directly follow the conversation content after the parentheses. For example: \"({self.name} nervously clenches fists) I don't know what you're talking about.\" or \"({self.name} smiles and nods) I'm glad to see you.\""},
            {"role": "user", "content": (context + information) if context else information}
        ]

        # Use the general processing function to get the response
        response_content = handle_stream_response(self.talk_client, use_model, messages)

        # Check and correct the response format
        if not response_content.startswith(f"({self.name}"):
            # If the response does not meet the required format, try to add the default format
            response_content = f"({self.name} calmly says) {response_content}"

        # Add the reply record to memory
        if speaker_name:
            self.add_memory(f"I said to {speaker_name}: {response_content}")

        return response_content

    def should_speak(self, context, speaker=None):
        """Determine if the Actor needs to speak. If so, execute the speak method. (Optional)

        Parameters:
            context: Current conversation or scenario context
            speaker: Person talking to the character (can be a Player object or string name)

        Returns:
            If the Actor needs to speak, return the result of the speak method; otherwise, return None
        """
        # Retrieve relevant memories to assist in the determination
        relevant_memories = self.retrieve_relevant_memories(context)

        # Extract speaker information
        speaker_name = speaker
        if hasattr(speaker, 'name'):
            speaker_name = speaker.name

        # Build the prompt content to determine if the Actor needs to speak
        prompt = f"Scenario: {context}\n\n"

        if speaker_name:
            prompt += f"Speaker: {speaker_name}\n\n"

        if relevant_memories:
            prompt += "Relevant memories:\n" + "\n".join(relevant_memories) + "\n\n"

        prompt += f"As a character named {self.name}, do I need to speak in this situation? Please only answer 'Yes' or 'No', and give a brief reason."

        # Use the global model variable
        global use_model

        # Build the message list
        messages = [
            {"role": "system",
             "content": "You need to help the character determine if they should speak in the current situation. Please only answer 'Yes' or 'No', and give a brief reason."},
            {"role": "user", "content": prompt}
        ]

        # Use the general processing function to get the response
        decision = handle_stream_response(self.talk_client, use_model, messages)

        # Analyze the decision result
        should_speak = "Yes" in decision[:10] or "yes" in decision.lower()[:10]

        # If the Actor should speak, execute the speak method
        if should_speak:
            return self.speak(context, speaker)

        return None


class Director:
    def __init__(self):
        """Initialize the director"""
        self.actors = {}  # Use a dictionary to store actors, with the actor name as the key
        self.script = {}  # Store the script, organized by scenario
        self.current_scene = None  # Current scenario

        global use_model

        if use_model == "ark":
            self.client = OpenAI(
                api_key=ark_api_key,
                base_url=ark_base_url
            )
        elif use_model == "gpt-4o-mini":
            self.client = OpenAI(
                api_key=openai_api_key,
                base_url="https://api.openai.com/v1"  # OpenAI's official API endpoint
            )
        elif use_model == "deepseek-chat":
            self.client = OpenAI(
                api_key=deepseek_api_key,
                base_url="https://api.deepseek.com/v1"  # Deepseek's API endpoint
            )
        elif use_model == "qwen3-235b-a22b":
            self.client = OpenAI(
                api_key=qwen_api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # Qwen's API endpoint
            )

    def add_actor(self, actor):
        """Add an actor to the director's management"""
        self.actors[actor.name] = actor

    def generate_actor_profile(self, character_name, scene_id, player_name):
        """Use AI to generate detailed information about the character

        Parameters:
            character_name: Character name
            scene_id: Current scenario ID
            player_name: Player character name

        Returns:
            Dictionary containing character information
        """
        # Get the current scenario description
        scene_desc = self.get_scene_description(scene_id)

        # Build the prompt
        prompt = f"""Based on the following scenario description and character name, generate a complete character profile:

Scenario description: {scene_desc}

Character name: {character_name}

Player character name: {player_name}

Please generate character information including the following:
1. Character age
2. Character gender
3. Character background story (3-5 items)
4. Character traits (2-3 items)
5. Relationship type and description with the player character "{player_name}"

Please return the result in JSON format:
{{
    "age": number,
    "gender": "gender",
    "background": ["memory/background 1", "memory/background 2", ...],
    "traits": ["trait 1", "trait 2", ...],
    "relationship": {{
        "type": "relationship type",
        "description": "relationship description"
    }}
}}"""

        # Build the message list
        messages = [
            {"role": "system",
             "content": "You are an AI assistant for creating characters. Generate character information that fits the scenario based on the scenario and character name."},
            {"role": "user", "content": prompt}
        ]

        # Use the general processing function to get the response
        response = handle_stream_response(self.client, use_model, messages)

        # Try to parse the JSON response
        try:
            import json
            import re

            # Try to extract the JSON part from the text
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
        except Exception as e:
            print(f"Failed to parse character information: {str(e)}")

        # If parsing fails, return default information
        return {
            "age": 30,
            "gender": "unknown",
            "background": [f"I am {character_name}, a newly appeared character in the current scenario"],
            "traits": ["mysterious"],
            "relationship": {
                "type": "stranger",
                "description": "Just met"
            }
        }

    def ensure_all_characters_exist(self, scene_id, player_name):
        """Check if all characters in the scenario exist. If not, create them

        Parameters:
            scene_id: Current scenario ID
            player_name: Player character name, used to exclude the player character and establish relationships
        """
        # Get all characters in the current scenario
        all_characters = self.get_scene_characters(scene_id=scene_id)

        # Check if each character exists in self.actors
        for character_name in all_characters:
            # Skip the player character
            if character_name == player_name:
                continue

            # If the character does not exist, create a new Actor instance
            if character_name not in self.actors:
                print(f"\nNew character found: {character_name}, using AI to generate character information...")

                # Use AI to generate character information
                profile = self.generate_actor_profile(character_name, scene_id, player_name)

                # Create an Actor instance
                new_actor = Actor(character_name, profile.get("age", 30), profile.get("gender", "unknown"))

                # Add background memories
                for memory in profile.get("background", [
                    f"I am {character_name}, a newly appeared character in the current scenario"]):
                    new_actor.add_memory(memory)

                # Add character traits
                for trait in profile.get("traits", ["mysterious"]):
                    new_actor.add_trait(trait)

                # Add the relationship with the player
                relationship = profile.get("relationship", {"type": "stranger", "description": "Just met"})
                new_actor.add_relationship(
                    player_name,
                    relationship.get("type", "stranger"),
                    relationship.get("description", "Just met")
                )

                # Add the new character to the director's management
                self.add_actor(new_actor)
                print(
                    f"Character created: {character_name}, age: {profile.get('age', 30)}, gender: {profile.get('gender', 'unknown')}")
                print(f"Character traits: {', '.join(profile.get('traits', ['mysterious']))}")
                print(
                    f"Relationship with the player: {relationship.get('type', 'stranger')} - {relationship.get('description', 'Just met')}")

    def check_and_create_new_characters(self, scene_ids, current_scene_index, player_name):
        """Check and create new characters that may appear in the new scenario

        Parameters:
            scene_ids: List of scenario IDs
            current_scene_index: Index of the current scenario
            player_name: Player character name
        """
        if current_scene_index < len(scene_ids):
            current_scene_id = scene_ids[current_scene_index]
            self.ensure_all_characters_exist(current_scene_id, player_name)

    def load_script(self, script_dict):
        """Load the script

        Parameters:
            script_dict: Dictionary containing the script content, in the following format:
            {
                "scene_1": {
                    "description": "Scenario description",
                    "characters": ["character name 1", "character name 2", ...],
                    "dialogues": [
                        {"character": "character name", "content": "Dialogue content"},
                        ...
                    ]
                },
                ...
            }
        """
        self.script = script_dict

    def set_current_scene(self, scene_id):
        """Set the current scenario"""
        if scene_id in self.script:
            self.current_scene = scene_id
            return True
        return False

    def get_current_scene(self):
        """Get the current scenario"""
        return self.current_scene

    def get_scene_description(self, scene_id=None):
        """Get the description of the specified scenario"""
        scene = scene_id if scene_id else self.current_scene
        if scene in self.script:
            return self.script[scene].get("description", "")
        return ""

    def get_scene_characters(self, scene_id=None, player=None):
        """Get the list of characters in the specified scenario, excluding the player character

        Parameters:
            scene_id: Scenario ID. If None, use the current scenario.
            player: Player object or player character name, which will be excluded from the return result.

        Returns:
            List of characters excluding the player character
        """
        scene = scene_id if scene_id else self.current_scene
        if scene not in self.script:
            return []

        characters = self.script[scene].get("characters", [])

        # If the player parameter is provided, exclude the player character from the character list
        if player:
            player_name = player
            # If player is a Player object, get its name
            if hasattr(player, 'get_player_name'):
                player_name = player.get_player_name()
            elif hasattr(player, 'name'):
                player_name = player.name

            # Filter out the player character
            characters = [char for char in characters if char != player_name]

        return characters

    def guide_actor_from_player_speech(self, player_speech, actor_name):
        """Generate performance guidance for the actor directly from the player's speech, combining the functions of generate_guidance_from_player_speech and guide_actor

        Parameters:
            player_speech: Content of the player's speech
            actor_name: Name of the actor to be guided

        Returns:
            Performance guidance for the actor
        """
        if self.current_scene is None or actor_name not in self.actors:
            return "Unable to guide: Current scenario not set or actor not found"

        # Get the current scenario information
        scene_info = self.script.get(self.current_scene, {})
        scene_desc = scene_info.get("description", "")

        # Get the actor's dialogues in the current scenario
        dialogues = scene_info.get("dialogues", [])
        actor_dialogues = [d for d in dialogues if d.get("character") == actor_name]

        # Get the character information
        actor = self.actors.get(actor_name)
        actor_traits = actor.get_traits() if hasattr(actor, "get_traits") else []
        traits_text = ", ".join(actor_traits) if actor_traits else "No specific character traits"

        # Build the prompt information
        prompt = f"""As a theater director, please provide performance guidance for the actor '{actor_name}' based on the following information:

Scenario description: {scene_desc}

Actor's character traits: {traits_text}

Player's recent speech: "{player_speech}"

Actor's lines:
"""
        for dialogue in actor_dialogues:
            prompt += f"- {dialogue.get('content')}\n"

        prompt += """
Please analyze the intention, emotion, and possible implied meaning of the player's speech, and then provide specific performance guidance, including:
1. Suggestions for emotional expression
2. Guidance on body language
3. Suggestions on intonation and rhythm
4. Whether certain information should be revealed
5. How to stay true to the character's traits

Please directly provide the guidance content without a preamble:"""

        # Build the message list
        messages = [
            {"role": "system",
             "content": "You are an experienced theater director who is good at analyzing the player's speech and providing response guidance for the actor."},
            {"role": "user", "content": prompt}
        ]

        # Use the general processing function to get the response
        return handle_stream_response(self.client, use_model, messages)

    def is_scene_continuing(self, last_dialogue, screenwriter=None, detailed_scene=None):
        """Determine if the script is still in the current scenario

        Parameters:
            last_dialogue: Content of the most recent dialogue (optional. If screenwriter is provided, the dialogue history of the screenwriter will be used first)
            screenwriter: Screenwriter object, used to get the dialogue history (optional)
            detailed_scene: Detailed scenario description (optional, used first)

        Returns:
            Boolean value indicating whether to continue in the current scenario
        """
        if self.current_scene is None:
            return False

        scene_info = self.script.get(self.current_scene, {})
        dialogues = scene_info.get("dialogues", [])

        # Check if all dialogues are completed
        if not dialogues:
            return False

        # Get the player character's goal description
        player_goal = ""
        scene_description = ""

        # Use the provided detailed scenario description first
        if detailed_scene:
            scene_description = detailed_scene
        # If no detailed scenario description is provided, try to get it from the screenwriter
        elif screenwriter and hasattr(screenwriter,
                                      'scene_descriptions') and self.current_scene in screenwriter.scene_descriptions:
            scene_description = screenwriter.scene_descriptions.get(self.current_scene, "")

        # Extract the player character's goal description
        if scene_description:
            # The typical format contains a paragraph with "Player character goal" or "Player goal"
            import re
            goal_patterns = [
                r"Player character goal description[：:](.*?)(?=\n\n|\Z)",
                r"Player character goal[：:](.*?)(?=\n\n|\Z)",
                r"Player goal[：:](.*?)(?=\n\n|\Z)",
                r"4[\.。]\s*Player character goal[^：:]*[：:](.*?)(?=\n\n|\Z)"
            ]

            for pattern in goal_patterns:
                goal_match = re.search(pattern, scene_description, re.DOTALL)
                if goal_match:
                    player_goal = goal_match.group(1).strip()
                    break

        # Use the screenwriter's dialogue history (if provided)
        recent_dialogue_history = ""
        if screenwriter and hasattr(screenwriter, 'get_dialogue_history'):
            recent_dialogues = screenwriter.get_dialogue_history(limit=5)
            if recent_dialogues:
                recent_dialogue_history = "\n\nRecent dialogue history:\n"
                for d in recent_dialogues:
                    if 'record_type' in d:
                        record_type = d['record_type']
                        if record_type == "Dialogue" or record_type.startswith("Dialogue"):
                            if 'target' in d:
                                recent_dialogue_history += f"{d['speaker']} said to {d['target']}: {d['content']}\n"
                            else:
                                recent_dialogue_history += f"{d['speaker']} ({record_type}): {d['content']}\n"
                        else:
                            recent_dialogue_history += f"{d['speaker']} ({record_type}): {d['content']}\n"
                    else:
                        recent_dialogue_history += f"{d['speaker']}: {d['content']}\n"

        # Build the prompt
        prompt = f"Please analyze the following situation to determine if the script should continue in the current scenario:\n\n"
        prompt += f"Scenario description: {scene_info.get('description', '')}\n\n"

        # Add the player's goal information (if any)
        if player_goal:
            prompt += f"Player character's goal in this scenario: {player_goal}\n\n"

        prompt += "Expected dialogue content:\n"
        for d in dialogues[-3:]:  # Only use the most recent few dialogues as context
            prompt += f"{d.get('character')}: {d.get('content')}\n"

        # Add the dialogue history or the latest dialogue
        if recent_dialogue_history:
            prompt += recent_dialogue_history
        elif last_dialogue:
            prompt += f"\nActual latest dialogue: {last_dialogue}\n"

        prompt += "\nBased on the following criteria, determine if the current scenario should continue:\n"
        prompt += "1. Whether the player character's goal has been achieved\n"
        prompt += "2. Whether the key dialogues in the scenario have been completed\n"
        prompt += "3. Whether the dialogue has naturally reached an end point\n"
        prompt += "4. Whether there are obvious clues for scenario transition\n\n"
        prompt += "Note: The player character's goal is an important condition for determining if the scenario should continue. If the goal has not been achieved and there are no obvious signals for transition, the scenario usually should continue.\n\n"
        prompt += "Please clearly determine if the scenario should continue. First answer Yes or No, then give a brief reason."

        # Build the message list
        messages = [
            {"role": "system",
             "content": "You are an experienced theater director who is good at analyzing scripts and performances. You need to determine if the current scenario should continue or if it should transition to the next scenario."},
            {"role": "user", "content": prompt}
        ]

        # Use the general processing function to get the response
        result = handle_stream_response(self.client, use_model, messages).lower()

        # Simply determine if the answer is affirmative or negative
        return "yes" in result[:30] or "should continue" in result[:30] or "continue" in result[
                                                                                         :30] or "not achieved" in result[
                                                                                                                   :50]

    def should_generate_new_script(self, screenwriter, current_scene_id, next_scene_id=None):
        """Determine if a new script/scenario needs to be generated

        Parameters:
            screenwriter: Screenwriter object, used to get the dialogue history and scenario information
            current_scene_id: Current scenario ID
            next_scene_id: ID of the next planned scenario (if any)

        Returns:
            Boolean value indicating whether a new scenario needs to be generated
        """
        # First, check if the current scenario has ended
        if self.is_scene_continuing(None, screenwriter):
            return False  # The current scenario is still continuing, no need to generate a new scenario

        # Get the recent dialogue history
        recent_dialogues = screenwriter.get_dialogue_history(limit=10)
        dialogue_text = ""
        for d in recent_dialogues:
            if 'record_type' in d:
                record_type = d['record_type']
                if record_type == "Dialogue" or record_type.startswith("Dialogue"):
                    dialogue_text += f"{d['speaker']} said to {record_type}: {d['content']}\n"
                else:
                    dialogue_text += f"{d['speaker']} ({record_type}): {d['content']}\n"

        # Get the current scenario information
        current_scene = self.script.get(current_scene_id, {})
        current_scene_desc = current_scene.get("description", "")

        # Get the next scenario information (if any)
        next_scene_info = ""
        if next_scene_id and next_scene_id in self.script:
            next_scene = self.script.get(next_scene_id, {})
            next_scene_desc = next_scene.get("description", "")
            next_scene_info = f"""Description of the next planned scenario:
{next_scene_desc}"""

        # Build the prompt
        prompt = f"""As a theater director, please determine if a new scenario needs to be inserted between the current scenario and the next planned scenario.

Current scenario description:
{current_scene_desc}

Recent dialogue history:
{dialogue_text}

{next_scene_info}

Please analyze the following points:
1. Whether the plot of the current scenario has naturally ended
2. Whether there are new plot clues in the dialogue history that need to be immediately addressed
3. Whether there is a large plot or scenario gap between the current scenario and the next planned scenario
4. Whether there are unresolved conflicts or incomplete plots that need to be addressed in a new scenario

Based on the above analysis, determine if a new scenario needs to be inserted? Please only answer 'Yes' or 'No', then give a brief reason."""

        # Build the message list
        messages = [
            {"role": "system",
             "content": "You are an experienced theater director who is good at analyzing plot development and scenario transitions."},
            {"role": "user", "content": prompt}
        ]

        # Use the general processing function to get the response
        result = handle_stream_response(self.client, use_model, messages)

        # Determine if the answer suggests generating a new scenario
        return "yes" in result[:10] or "need" in result[:20] or "should" in result[:20]


class Player:
    def __init__(self, name, age, gender):
        """Initialize the player character.

        Args:
            name: The name of the character the player is playing.
            age: The age of the character the player is playing.
            gender: The gender of the character the player is playing.
        """
        self.name = name
        self.age = age
        self.gender = gender

    def talk_to_actor(self, actor, message, guidance=None):
        """Talk to an actor in the scene.

        Args:
            actor: An Actor object.
            message: The content of the dialogue input by the player.
            guidance: Guidance generated by the director.

        Returns:
            The actor's response.
        """
        if not hasattr(actor, 'speak'):
            return "Error: Unable to talk to this object."

        # Directly pass the player's message to the actor to generate a response.
        response = actor.speak(message, self.name, guidance)
        return response

    def interact_with_environment(self, screenwriter, action, current_scene_id=None):
        """Interact with the current scene environment or items.

        Args:
            screenwriter: The screenwriter object to handle the player's actions.
            action: The action the player wants to perform.
            current_scene_id: The current scene ID (optional).

        Returns:
            A description of the interaction result.
        """
        # Record the interaction content in the dialogue history.
        screenwriter.add_dialogue_record(self.name, "Environment Interaction", action)

        # If a scene ID is provided, update the scene description.
        if current_scene_id:
            # Update the scene based on the player's action.
            updated_scene = screenwriter.transform_scene(
                current_scene_id,
                action
            )

            # Set the player's current scene.
            self.current_scene = updated_scene

        return updated_scene

    def get_player_name(self):
        """Get the player's name.

        Returns:
            The player's name.
        """
        return self.name


class Screenwriter:
    def __init__(self):
        """Initialize the screenwriter."""
        global use_model

        if use_model == "ark":
            self.client = OpenAI(
                api_key=ark_api_key,
                base_url=ark_base_url
            )
        elif use_model == "gpt-4o-mini":
            self.client = OpenAI(
                api_key=openai_api_key,
                base_url="https://api.openai.com/v1"
            )
        elif use_model == "deepseek-chat":
            self.client = OpenAI(
                api_key=deepseek_api_key,
                base_url="https://api.deepseek.com/v1"
            )
        elif use_model == "qwen3-235b-a22b":
            self.client = OpenAI(
                api_key=qwen_api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )

        self.dialogue_history = []  # Store the dialogue history.
        self.scene_descriptions = {}  # Store the scene descriptions.
        self.initial_script = {}  # Store the initial script.

    def load_initial_script(self, script_dict):
        """Load the initial script.

        Args:
            script_dict: A dictionary of the initial script.
        """
        self.initial_script = script_dict
        # Initialize the scene descriptions.
        for scene_id, scene_data in script_dict.items():
            if "description" in scene_data:
                self.scene_descriptions[scene_id] = scene_data["description"]

    def generate_scene_description(self, scene_id, director=None, player_character=None):
        """Generate a detailed description of the scene.

        Args:
            scene_id: The scene ID.
            director: A Director object to get character information (optional).
            player_character: The name of the character controlled by the player (optional), which will not be described in detail.

        Returns:
            A detailed description including the scene, items, and non-player characters.
        """
        # Check if there is a basic description of the scene.
        base_description = self.scene_descriptions.get(scene_id)
        if not base_description:
            if scene_id in self.initial_script and "description" in self.initial_script[scene_id]:
                base_description = self.initial_script[scene_id]["description"]
            else:
                return f"Error: Could not find the description of scene {scene_id}."

        # Get the list of characters in the scene.
        scene_characters = []
        characters_info = ""

        if scene_id in self.initial_script and "characters" in self.initial_script[scene_id]:
            scene_characters = self.initial_script[scene_id]["characters"]

            # If a Director object is provided, get more detailed character information.
            if director and hasattr(director, 'actors'):
                for char_name in scene_characters:
                    # Skip the character controlled by the player.
                    if player_character and char_name == player_character:
                        continue

                    # Get the character information.
                    if char_name in director.actors:
                        actor = director.actors[char_name]
                        char_info = f"- {char_name}: {actor.age} years old, {actor.gender}\n"

                        # Add personality traits.
                        if hasattr(actor, 'get_traits') and actor.get_traits():
                            traits = actor.get_traits()
                            char_info += f"  Personality traits: {', '.join(traits)}\n"

                        # Add relationships with other characters.
                        if hasattr(actor, 'get_all_relationships'):
                            relationships = actor.get_all_relationships()
                            if relationships:
                                char_info += "  Relationships with other characters:\n"
                                for other_name, rel_list in relationships.items():
                                    if other_name in scene_characters:  # Only add relationships with characters in the scene.
                                        for rel in rel_list:
                                            rel_type = rel.get('type', '')
                                            rel_desc = rel.get('description', '')
                                            if rel_desc:
                                                char_info += f"    - With {other_name}: {rel_type} ({rel_desc})\n"
                                            else:
                                                char_info += f"    - With {other_name}: {rel_type}\n"

                        characters_info += char_info

        # Build the prompt.
        prompt = f"""Based on the following basic scene description and character information, generate a more detailed scene description, including the scene environment, items in the scene, and non-player characters:

Basic scene description: {base_description}

"""

        if characters_info:
            prompt += f"""Character information in the scene:
{characters_info}
"""

        if player_character:
            prompt += f"""Note: The player character "{player_character}" in the scene does not need to be described in detail because the player will control this character themselves.
"""

        prompt += """Please provide:
1. Environment description: Including spatial layout, lighting, sound, smell, etc.
2. Item description: List the main items in the scene and their placement.
3. Non-player character description: Describe the appearance, posture, current actions, and emotional states of the characters in the scene (excluding the player character). The personality description should be consistent with the above character information.
4. Player character goal description: Describe the general goal of the player character in the current scene.

Please use vivid and imaginative language to create an immersive scene experience."""

        # Build the message list.
        messages = [
            {"role": "system", "content": "You are an experienced theater screenwriter, skilled at creating vivid and detailed scene descriptions."},
            {"role": "user", "content": prompt}
        ]

        # Use the general handling function to get the response.
        detailed_description = handle_stream_response(self.client, use_model, messages)

        # Update the scene description.
        self.scene_descriptions[scene_id] = detailed_description

        return detailed_description

    def add_dialogue_record(self, speaker, record_type, content, target=None):
        """Add a drama record.

        Args:
            speaker: The name of the speaker or actor.
            record_type: The type of record (e.g., "Dialogue", "Narrative", "Scene Description", "Environment Interaction", etc.).
            content: The content.
            target: The recipient of the dialogue (optional, only meaningful when record_type is "Dialogue").
        """
        record = {
            "time": len(self.dialogue_history),
            "speaker": speaker,
            "record_type": record_type,
            "content": content
        }

        # If a target is provided and record_type is related to dialogue, record the dialogue target.
        if target and (record_type == "Dialogue" or record_type.startswith("Dialogue")):
            record["target"] = target

        self.dialogue_history.append(record)

    def get_dialogue_history(self, limit=10):
        """Get the recent dialogue history.

        Args:
            limit: The maximum number of dialogues to return.

        Returns:
            A list of the recent dialogue history.
        """
        return self.dialogue_history[-limit:] if self.dialogue_history else []

    def get_all_dialogue_history(self):
        """Get all the dialogue history.

        Args:
            limit: The maximum number of dialogues to return.

        Returns:
            A list of all the dialogue history.
        """
        return self.dialogue_history if self.dialogue_history else []

    def generate_new_script(self, current_scene_id, player_feedback=None, max_retries=3, dialogue_history=None):
        """Generate a new script based on the historical dialogue.

        Args:
            current_scene_id: The current scene ID.
            player_feedback: Feedback provided by the player (optional).
            max_retries: The maximum number of retries.
            dialogue_history: A complete record of the dialogue history (optional).

        Returns:
            A new part of the script in the script_dict format.
        """
        # Automatically generate the next scene ID.
        try:
            # Try to extract the numeric part from the current scene ID.
            import re
            scene_num_match = re.search(r'(\d+)', current_scene_id)
            if scene_num_match:
                scene_num = int(scene_num_match.group(1))
                next_scene_id = current_scene_id.replace(str(scene_num), str(scene_num + 1))
            else:
                # If there is no number in the current scene ID, add _1.
                next_scene_id = f"{current_scene_id}_1"
        except:
            # If the extraction fails, use a timestamp as an alternative.
            import time
            next_scene_id = f"scene_{int(time.time())}"

        # Get the current scene information.
        current_scene = self.initial_script.get(current_scene_id, {})
        scene_desc = self.scene_descriptions.get(current_scene_id, "")

        # Get the list of characters in the current scene.
        current_characters = current_scene.get("characters", [])

        # Get the dialogue history.
        dialogue_text = ""
        if dialogue_history:
            # Use the provided complete dialogue history.
            for d in dialogue_history:
                if isinstance(d, dict):
                    speaker = d.get('speaker', '')
                    record_type = d.get('record_type', '')
                    content = d.get('content', '')
                    if record_type == "Dialogue" or record_type.startswith("Dialogue"):
                        if 'target' in d:
                            dialogue_text += f"{speaker} says to {d.get('target')}: {content}\n"
                        else:
                            dialogue_text += f"{speaker} ({record_type}): {content}\n"
                    else:
                        dialogue_text += f"{speaker} ({record_type}): {content}\n"

        else:
            # If no dialogue history is provided, get the recent dialogues.
            recent_dialogues = self.get_dialogue_history(limit=20)  # Increase the number of historical records.
            for d in recent_dialogues:
                if 'record_type' in d:
                    record_type = d.get('record_type', '')
                    if record_type == "Dialogue" or record_type.startswith("Dialogue"):
                        if 'target' in d:
                            dialogue_text += f"{d['speaker']} says to {d.get('target')}: {d['content']}\n"
                        else:
                            dialogue_text += f"{d['speaker']} ({record_type}): {d['content']}\n"
                    else:
                        dialogue_text += f"{d['speaker']} ({record_type}): {d['content']}\n"
                else:
                    dialogue_text += f"{d['speaker']}: {d['content']}\n"

        # Generate the script, trying up to max_retries times.
        for attempt in range(max_retries):
            # Build the prompt.
            prompt = f"""As a screenwriter, please generate the next scene of the script based on the following information:

Current scene: {scene_desc}

Characters in the current scene: {', '.join(current_characters)}

Dialogue history:
{dialogue_text}

"""

            if player_feedback:
                prompt += f"""
Player feedback:
{player_feedback}

"""

            prompt += """Please create the next scene of the script based on the above information. Pay special attention to:
1. The new scene must be consistent with the dialogue history and not contain contradictions.
2. The actions and dialogues of the characters must conform to their personality traits.
3. The scene transition must be natural, and the plot development must be reasonable.

You must strictly return in the following JSON format, which is required by the system:
{
    "description": "Detailed scene description",
    "characters": ["Character Name 1", "Character Name 2", ...],
    "dialogues": [
        {"character": "Character Name 1", "content": "(Character expression and action) Dialogue content 1"},
        {"character": "Character Name 2", "content": "(Character expression and action) Dialogue content 2"},
        ...
    ]
}

Note:
1. Only return the content of one scene.
2. The JSON format must be completely correct, without any explanations or extra text.
3. The dialogue content should conform to the character's characteristics and advance the plot.
4. Do not generate the scene_id field; the system will handle it automatically.
5. Ensure that the new scene is consistent with the previous dialogue and plot."""

            # Build the message list.
            messages = [
                {"role": "system",
                 "content": "You are an experienced screenwriter specializing in creating scripts for interactive dramas. You must return the result in the required JSON format and ensure that the new scene is consistent with the historical dialogue and plot."},
                {"role": "user", "content": prompt}
            ]

            # Use the general handling function to get the response.
            generated_text = handle_stream_response(self.client, use_model, messages)

            # Try to parse the JSON.
            try:
                import json
                import re

                # Try to extract the JSON part from the text.
                json_match = re.search(r'\{[\s\S]*\}', generated_text)
                if json_match:
                    generated_text = json_match.group(0)

                new_scene = json.loads(generated_text)

                # Validate the JSON structure.
                if not all(key in new_scene for key in ["description", "dialogues"]):
                    raise ValueError("The generated JSON is missing necessary fields.")

                # Build the complete scene information.
                self.initial_script[next_scene_id] = {
                    "description": new_scene["description"],
                    "characters": new_scene["characters"],
                    "dialogues": new_scene["dialogues"]
                }

                # Update the scene description.
                self.scene_descriptions[next_scene_id] = new_scene["description"]

                return {next_scene_id: self.initial_script[next_scene_id]}

            except Exception as e:
                if attempt < max_retries - 1:
                    # If it's not the last attempt, continue to the next one.
                    continue
                else:
                    # All attempts have failed, return an error message.
                    return {"error": str(e), "generated_text": generated_text, "next_scene_id": next_scene_id}

    def generate_actor_response_suggestions(self, actor_name, player_action):
        """Generate suggestions for the actor's response to the player's action.

        Args:
            actor_name: The name of the actor.
            player_action: The player's action.

        Returns:
            A list of suggestions for the actor's possible responses.
        """
        # Get the relevant dialogue history.
        recent_dialogues = self.get_dialogue_history(limit=5)
        dialogue_text = ""
        for d in recent_dialogues:
            if 'record_type' in d:
                record_type = d.get('record_type', '')
                if record_type == "Dialogue" or record_type.startswith("Dialogue"):
                    if 'target' in d:
                        dialogue_text += f"{d['speaker']} says to {d.get('target')}: {d['content']}\n"
                    else:
                        dialogue_text += f"{d['speaker']} ({record_type}): {d['content']}\n"
                else:
                    dialogue_text += f"{d['speaker']} ({record_type}): {d['content']}\n"
            else:
                dialogue_text += f"{d['speaker']}: {d['content']}\n"

        # Build the prompt.
        prompt = f"""The player has just performed the following action on the character {actor_name}:
        {player_action}

        Recent dialogue history:
        {dialogue_text}

        Based on the character's characteristics and context, generate 3 different styles of response suggestions:
        1. Friendly response
        2. Neutral response
        3. Cold response"""

        # Build the message list.
        messages = [
            {"role": "system", "content": "You are a screenwriter, skilled at creating appropriate and natural dialogues for characters."},
            {"role": "user", "content": prompt}
        ]

        # Use the general handling function to get the response.
        return handle_stream_response(self.client, use_model, messages)

    def transform_scene(self, scene_id, player_action=None):
        """Generate a description of the scene after the interaction based on the player's interaction with the items/environment in the scene.

        Args:
            scene_id: The current scene ID.
            player_action: The specific interaction action of the player (e.g., "Push the glass off the table").

        Returns:
            An updated description of the scene after the interaction.
        """
        # Get the original scene description.
        original_desc = self.scene_descriptions.get(scene_id, "")
        if not original_desc:
            return "Error: Could not find the scene description."

        # Build the prompt.
        prompt = f"""Original scene description:
{original_desc}

The player has performed the following interaction in the scene:
"""

        if player_action:
            prompt += f"""
{player_action}
"""

        prompt += """
Please describe in detail the changes in the scene state after the player's interaction and return in JSON format:
1. What state is the interacted item in now (if applicable)?
2. What changes has the interaction caused to the scene environment?
3. What impact has it had on other items in the scene?
4. How have the characters in the scene (if any) reacted to this interaction?

Please return the scene description in JSON format as follows:
{
    "scene_description": "Detailed description of the changes in the scene environment and atmosphere",
    "interactions": [
        {
            "object": "Name of the interacted item",
            "state": "Description of the state of the item after the interaction"
        }
    ],
    "character_reactions": [
        {
            "character": "Character name",
            "action": "Description of the character's action",
            "dialogue": "The character's dialogue reaction to this interaction (if any)"
        }
    ]
}

Please ensure:
1. The JSON format is completely correct and can be parsed.
2. The scene description should be vivid and consistent with the development of the story plot.
3. Maintain continuity with the original scene and only describe reasonable changes caused by the player's interaction.
4. The character reactions should conform to their personality traits."""

        # Build the message list.
        messages = [
            {"role": "system",
             "content": "You are a screenwriter skilled at describing scene changes. When the player interacts with the items or environment in the scene, you need to vividly describe the changes in the scene state after the interaction and return the result in JSON format."},
            {"role": "user", "content": prompt}
        ]

        # Use the general handling function to get the response.
        json_response = handle_stream_response(self.client, use_model, messages)

        # Parse the JSON response.
        import json
        import re

        try:
            # Try to extract the JSON part from the possible text.
            json_match = re.search(r'\{[\s\S]*\}', json_response)
            if json_match:
                json_str = json_match.group(0)
                scene_data = json.loads(json_str)

                # Build a human-readable scene description.
                formatted_description = ""

                if "scene_description" in scene_data:
                    formatted_description += scene_data["scene_description"] + "\n\n"

                # Add item interaction information.
                if "interactions" in scene_data and scene_data["interactions"]:
                    formatted_description += "Changes in the interacted items:\n"
                    for interaction in scene_data["interactions"]:
                        object_name = interaction.get("object", "")
                        state = interaction.get("state", "")
                        if object_name and state:
                            formatted_description += f"- {object_name}: {state}\n"
                    formatted_description += "\n"

                # Add character reactions.
                if "character_reactions" in scene_data and scene_data["character_reactions"]:
                    formatted_description += "Reactions of the characters in the scene:\n"
                    for reaction in scene_data["character_reactions"]:
                        character = reaction.get("character", "")
                        action = reaction.get("action", "")
                        dialogue = reaction.get("dialogue", "")

                        if character and (action or dialogue):
                            reaction_text = f"- {character}: "
                            if action:
                                reaction_text += f"{action}"
                                # Record the character's reaction in the dialogue history.
                                self.add_dialogue_record(character, "Action", action)

                            if dialogue:
                                if action:
                                    reaction_text += f', saying: "{dialogue}"'
                                else:
                                    reaction_text += f'"{dialogue}"'
                                # Record the character's dialogue in the dialogue history.
                                self.add_dialogue_record(character, "Dialogue", dialogue)

                            formatted_description += reaction_text + "\n"

                # Update the scene description.
                import time
                # Generate a unique scene ID using a timestamp.
                new_scene_id = f"{scene_id}_{int(time.time())}"
                self.scene_descriptions[new_scene_id] = formatted_description

                # Record this interaction in the dialogue history.
                self.add_dialogue_record("System", "Scene", f"Scene change: {player_action}")

                return formatted_description

        except Exception as e:
            # If JSON parsing fails, use the original response.
            print(f"JSON parsing failed: {str(e)}, using the original text.")

            # Update the scene description.
            import time
            # Generate a unique scene ID using a timestamp.
            new_scene_id = f"{scene_id}_{int(time.time())}"
            self.scene_descriptions[new_scene_id] = json_response

            # Record this interaction in the dialogue history.
            self.add_dialogue_record("System", "Scene", f"Scene change: {player_action}")

            return json_response

    def end_scene(self, last_interaction, director, current_scene_id, next_scene_id=None, dialogue_history=None):
        """Generate a description of the scene ending.

        Args:
            last_interaction: The content of the last interaction.
            director: A Director object to get the scene description.
            current_scene_id: The current scene ID.
            next_scene_id: The next scene ID (optional).
            dialogue_history: A list of the dialogue history records (optional).

        Returns:
            A text description of the scene ending.
        """
        # Error check.
        if not director or not hasattr(director, 'get_scene_description'):
            return "Error: Invalid director object."

        scene_desc = director.get_scene_description(current_scene_id)
        if not scene_desc:
            return "Error: Could not get the scene description."

        # Get the list of characters in the current scene.
        scene_characters = []
        if hasattr(director, 'get_scene_characters'):
            scene_characters = director.get_scene_characters(current_scene_id)

        # Get the next scene information (if available).
        next_scene_info = ""
        if next_scene_id and hasattr(director, 'get_scene_description'):
            next_scene_desc = director.get_scene_description(next_scene_id)
            if next_scene_desc:
                next_scene_info = "Next scene:\n" + next_scene_desc + "\n\nPlease ensure that your scene ending description can naturally transition to the next scene. Consider how the characters move from the current scene to the next scene, how the environment changes, and how the plot develops."

        # Build the character information.
        characters_info = ""
        if scene_characters:
            characters_info = "Characters in the current scene: " + ", ".join(scene_characters)

        # Build the dialogue history information.
        dialogue_history_info = ""
        if dialogue_history:
            dialogue_history_info = "\n\nDialogue history records:\n"
            for dialogue in dialogue_history:
                if isinstance(dialogue, dict):
                    speaker = dialogue.get('speaker', '')
                    record_type = dialogue.get('record_type', '')
                    content = dialogue.get('content', '')
                    if record_type == "Dialogue" or record_type.startswith("Dialogue"):
                        if 'target' in dialogue:
                            dialogue_history_info += f"{speaker} says to {dialogue.get('target')}: {content}\n"
                        else:
                            dialogue_history_info += f"{speaker} ({record_type}): {content}\n"
                    else:
                        dialogue_history_info += f"{speaker} ({record_type}): {content}\n"

        # Build the prompt to generate the scene ending description.
        prompt = "Please generate a scene ending description in the format of a drama script based on the following last interaction content and dialogue history, so that the story can naturally transition to the next scene:\n\n"
        prompt += "Note! The generated content cannot be contradictory! It must be consistent with the dialogue history!\n\n"
        prompt += "Last interaction:\n" + last_interaction + "\n\n"
        prompt += "Current scene:\n" + scene_desc + "\n\n"

        if characters_info:
            prompt += characters_info + "\n\n"

        if dialogue_history_info:
            prompt += dialogue_history_info + "\n\n"

        if next_scene_info:
            prompt += next_scene_info + "\n\n"

        prompt += """Please return the scene ending description in JSON format as follows:
{
    "scene_description": "Description of the changes in the scene environment and atmosphere",
    "dialogues": [
        {
            "character": "Character Name 1",
            "action": "Description of the action (optional)",
            "content": "Dialogue content"
        },
        {
            "character": "Character Name 2",
            "action": "Description of the action (optional)",
            "content": "Dialogue content"
        }
    ],
    "transition": "Description of the natural transition to the next scene"
}

Please ensure:
1. The JSON format is completely correct and can be parsed.
2. The character names must be actual characters in the scene or "Narrator".
3. All dialogues must be included in the dialogues array.
4. The scene description should be vivid and consistent with the development of the story plot.
5. The generated content must be consistent with the dialogue history and not contain contradictions."""

        # Use the handle_stream_response function to get the scene ending description.
        messages = [
            {"role": "system",
             "content": 'You are an experienced theater screenwriter, skilled at creating scene transitions in the format of a drama script. You need to return a JSON-formatted response containing the scene description, character dialogues, and scene transition. Pay special attention to maintaining consistency with the dialogue history.'},
            {"role": "user", "content": prompt}
        ]
        json_response = handle_stream_response(self.client, use_model, messages)

        # Parse the JSON response.
        import json
        import re

        try:
            # Try to extract the JSON part from the possible text.
            json_match = re.search(r'\{[\s\S]*\}', json_response)
            if json_match:
                json_str = json_match.group(0)
                scene_data = json.loads(json_str)

                # Extract and record the dialogues.
                if "dialogues" in scene_data and isinstance(scene_data["dialogues"], list):
                    for dialogue in scene_data["dialogues"]:
                        character = dialogue.get("character", "").strip()
                        content = dialogue.get("content", "").strip()
                        action = dialogue.get("action", "").strip()

                        # Check if the character is in the scene or is the narrator.
                        if character and (character in scene_characters or character == "Narrator"):
                            # Record the dialogue. If there is an action description, add it to the content.
                            full_content = content
                            if action:
                                full_content = f"({action}) {content}"
                            self.add_dialogue_record(character, "Scene Ending", full_content)

                # Build a human-readable scene ending description.
                formatted_description = ""

                if "scene_description" in scene_data:
                    formatted_description += scene_data["scene_description"] + "\n\n"

                if "dialogues" in scene_data:
                    for dialogue in scene_data["dialogues"]:
                        character = dialogue.get("character", "")
                        content = dialogue.get("content", "")
                        action = dialogue.get("action", "")

                        if action:
                            formatted_description += f"{character}: ({action}) {content}\n"
                        else:
                            formatted_description += f"{character}: {content}\n"

                    formatted_description += "\n"

                if "transition" in scene_data:
                    formatted_description += scene_data["transition"]

                return formatted_description

        except Exception as e:
            # If parsing fails, return the original response and try basic parsing.
            print(f"JSON parsing failed: {str(e)}, using basic text processing.")

            # Basic text processing as a fallback.
            lines = json_response.split('\n')
            for line in lines:
                if '：' in line:
                    parts = line.split('：', 1)
                    if len(parts) == 2:
                        character = parts[0].strip()
                        content = parts[1].strip()

                        # Check if the character is in the scene or is the narrator.
                        if character in scene_characters or character == "Narrator":
                            # Record the dialogue.
                            self.add_dialogue_record(character, "Scene Ending", content)

            return json_response

class StdoutRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.buffer = ""

    def write(self, string):
        self.buffer += string
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)

    def flush(self):
        pass


class SettingsDialog:
    def __init__(self, parent):
        self.parent = parent
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Game Settings")
        self.dialog.geometry("500x450")  # Adjust to appropriate size
        self.dialog.configure(bg="#1e1e2e")
        self.dialog.transient(parent)
        self.dialog.grab_set()

        # Set variables
        self.model_var = tk.StringVar(value=use_model)
        self.try_chance_var = tk.IntVar(value=try_chance)
        self.max_inserted_scenes_var = tk.IntVar(value=max_inserted_scenes)
        self.max_new_scene_generations_var = tk.IntVar(value=max_new_scene_generations)

        # Create the settings interface
        self.create_widgets()

        # Initialize the return value
        self.result = {
            "model": use_model,
            "try_chance": try_chance,
            "max_inserted_scenes": max_inserted_scenes,
            "max_new_scene_generations": max_new_scene_generations
        }

        # Ensure the dialog is displayed in the center of the parent window
        self.dialog.update_idletasks()
        width = self.dialog.winfo_width()
        height = self.dialog.winfo_height()
        x = parent.winfo_rootx() + (parent.winfo_width() - width) // 2
        y = parent.winfo_rooty() + (parent.winfo_height() - height) // 2
        self.dialog.geometry(f'{width}x{height}+{x}+{y}')

        # Force the window to refresh after initialization
        self.dialog.update()

        # Set the minimum window size to prevent content from being cropped
        self.dialog.minsize(400, 300)

        # Prevent the game from continuing when the window is closed (must be closed via button)
        self.dialog.protocol("WM_DELETE_WINDOW", self.on_cancel)

    def create_widgets(self):
        main_frame = tk.Frame(self.dialog, bg="#1e1e2e", padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Model selection
        model_frame = tk.LabelFrame(main_frame, text="Model Settings", bg="#2a2a3e", fg="#ffffff",
                                    font=("Microsoft YaHei", 12, "bold"), padx=10, pady=10)
        model_frame.pack(fill=tk.X, pady=5)

        models = [("OpenAI (gpt-4o-mini)", "gpt-4o-mini"),
                  ("DeepSeek (deepseek-chat)", "deepseek-chat"),
                  ("Qianwen (qwen3-235b-a22b)", "qwen3-235b-a22b")]

        for i, (text, value) in enumerate(models):
            radio = tk.Radiobutton(model_frame, text=text, value=value, variable=self.model_var,
                                   bg="#2a2a3e", fg="#ffffff", selectcolor="#3d3d60",
                                   activebackground="#2a2a3e", activeforeground="#ffffff",
                                   font=("Microsoft YaHei", 10))
            radio.pack(anchor=tk.W, pady=2)
            # Ensure the button is displayed immediately
            radio.update()

        # Scene settings
        scene_frame = tk.LabelFrame(main_frame, text="Scene Settings", bg="#2a2a3e", fg="#ffffff",
                                    font=("Microsoft YaHei", 12, "bold"), padx=10, pady=10)
        scene_frame.pack(fill=tk.X, pady=10)

        # Use Grid layout to ensure alignment
        scene_frame.grid_columnconfigure(0, weight=3)
        scene_frame.grid_columnconfigure(1, weight=1)

        # Try chances
        row = 0
        tk.Label(scene_frame, text="Tries per scene:", bg="#2a2a3e", fg="#ffffff",
                 font=("Microsoft YaHei", 10)).grid(row=row, column=0, sticky=tk.W, pady=5)

        try_chance_spinbox = tk.Spinbox(scene_frame, from_=1, to=5, textvariable=self.try_chance_var,
                                        width=5, bg="#2d2d40", fg="#ffffff", font=("Microsoft YaHei", 10))
        try_chance_spinbox.grid(row=row, column=1, sticky=tk.W, padx=10)

        # Maximum number of inserted scenes
        row += 1
        tk.Label(scene_frame, text="Max inserted scenes:", bg="#2a2a3e", fg="#ffffff",
                 font=("Microsoft YaHei", 10)).grid(row=row, column=0, sticky=tk.W, pady=5)

        max_inserted_spinbox = tk.Spinbox(scene_frame, from_=0, to=3, textvariable=self.max_inserted_scenes_var,
                                          width=5, bg="#2d2d40", fg="#ffffff", font=("Microsoft YaHei", 10))
        max_inserted_spinbox.grid(row=row, column=1, sticky=tk.W, padx=10)

        # Maximum number of new scene generations
        row += 1
        tk.Label(scene_frame, text="Max new scene generations:", bg="#2a2a3e", fg="#ffffff",
                 font=("Microsoft YaHei", 10)).grid(row=row, column=0, sticky=tk.W, pady=5)

        max_new_spinbox = tk.Spinbox(scene_frame, from_=0, to=3, textvariable=self.max_new_scene_generations_var,
                                     width=5, bg="#2d2d40", fg="#ffffff", font=("Microsoft YaHei", 10))
        max_new_spinbox.grid(row=row, column=1, sticky=tk.W, padx=10)

        # Button area
        button_frame = tk.Frame(main_frame, bg="#1e1e2e", pady=10)
        button_frame.pack(fill=tk.X)

        # Add OK button
        ok_button = tk.Button(button_frame, text="OK", command=self.on_ok,
                              bg="#3d3d60", fg="#ffffff", font=("Microsoft YaHei", 10),
                              width=10)
        ok_button.pack(side=tk.RIGHT, padx=5)

        # Add Cancel button
        cancel_button = tk.Button(button_frame, text="Cancel", command=self.on_cancel,
                                  bg="#3d3d60", fg="#ffffff", font=("Microsoft YaHei", 10),
                                  width=10)
        cancel_button.pack(side=tk.RIGHT, padx=5)

        # Force update all widgets
        self.dialog.update()

    def on_ok(self):
        self.result = {
            "model": self.model_var.get(),
            "try_chance": self.try_chance_var.get(),
            "max_inserted_scenes": self.max_inserted_scenes_var.get(),
            "max_new_scene_generations": self.max_new_scene_generations_var.get()
        }
        self.dialog.destroy()

    def on_cancel(self):
        # When canceling, set the result to None
        self.result = None
        self.dialog.destroy()

    def show(self):
        # Wait for the dialog in the main window
        self.parent.wait_window(self.dialog)
        return self.result

class CoCGameGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Call of Cthulhu - Interactive Game")
        self.root.geometry("1200x800")
        self.root.configure(bg="#1e1e2e")

        # Show the settings dialog before game initialization
        self.show_settings_dialog()

        self.setup_variables()
        self.create_ui()
        self.initialize_game()

    def show_settings_dialog(self):
        # Ensure the root window is fully initialized before showing the settings dialog
        self.root.update_idletasks()

        # Create and show the settings dialog
        dialog = SettingsDialog(self.root)
        settings = dialog.show()

        # Check if the user cancelled the settings
        if not settings:
            # If cancelled, use default values
            settings = {
                "model": use_model,
                "try_chance": try_chance,
                "max_inserted_scenes": max_inserted_scenes,
                "max_new_scene_generations": max_new_scene_generations
            }
            print("\nUser cancelled the settings, default configuration will be used")

        # Save the settings
        self.settings = settings

        # Print the current configuration
        print(f"\n===== Game Configuration =====")
        print(f"Model in use: {settings['model']}")
        print(f"Scene try chances: {settings['try_chance']}")
        print(f"Maximum inserted scenes: {settings['max_inserted_scenes']}")
        print(f"Maximum new scene generations: {settings['max_new_scene_generations']}")
        print(f"====================\n")

    def setup_variables(self):
        # Game state variables
        self.current_scene_index = 0
        self.script_ids = []
        self.scene_finished = False
        self.should_exit_game = False
        self.last_interaction = ""
        self.inserted_scene_count = 0
        self.new_scene_generation_count = 0
        self.characters = []

        # Characters and director
        self.player = None
        self.director = None
        self.screenwriter = None

    def create_ui(self):
        # Create the main frame
        main_frame = tk.Frame(self.root, bg="#1e1e2e")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Split the window
        top_frame = tk.Frame(main_frame, bg="#1e1e2e")
        top_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        bottom_frame = tk.Frame(main_frame, bg="#1e1e2e", height=200)
        bottom_frame.pack(fill=tk.X, padx=5, pady=5)

        # Scene and dialogue display area
        left_frame = tk.Frame(top_frame, bg="#1e1e2e", width=800)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Scene description area
        scene_frame = tk.LabelFrame(left_frame, text="Scene Description", bg="#2a2a3e", fg="#ffffff",
                                    font=("Microsoft YaHei", 12, "bold"))
        scene_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.scene_text = scrolledtext.ScrolledText(scene_frame, wrap=tk.WORD, bg="#2d2d40", fg="#ffffff",
                                                    font=("Microsoft YaHei", 11))
        self.scene_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Character and action area
        right_frame = tk.Frame(top_frame, bg="#1e1e2e", width=300)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5, pady=5)

        # Character selection area
        character_frame = tk.LabelFrame(right_frame, text="Characters", bg="#2a2a3e", fg="#ffffff",
                                        font=("Microsoft YaHei", 12, "bold"))
        character_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.character_listbox = tk.Listbox(character_frame, bg="#2d2d40", fg="#ffffff", font=("Microsoft YaHei", 11),
                                            selectbackground="#3d3d60")
        self.character_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Action button area
        action_frame = tk.LabelFrame(right_frame, text="Actions", bg="#2a2a3e", fg="#ffffff",
                                     font=("Microsoft YaHei", 12, "bold"))
        action_frame.pack(fill=tk.BOTH, padx=5, pady=5)

        # Dialogue button
        self.talk_btn = tk.Button(action_frame, text="Talk to Character", bg="#3d3d60", fg="#ffffff", font=("Microsoft YaHei", 11),
                                  command=self.talk_to_character)
        self.talk_btn.pack(fill=tk.X, padx=5, pady=5)

        # Interaction button
        self.interact_btn = tk.Button(action_frame, text="Interact with Environment", bg="#3d3d60", fg="#ffffff",
                                      font=("Microsoft YaHei", 11),
                                      command=self.interact_with_environment)
        self.interact_btn.pack(fill=tk.X, padx=5, pady=5)

        # Next scene button
        self.next_btn = tk.Button(action_frame, text="Next Scene", bg="#3d3d60", fg="#ffffff", font=("Microsoft YaHei", 11),
                                  command=self.go_to_next_scene)
        self.next_btn.pack(fill=tk.X, padx=5, pady=5)

        # Exit button
        self.exit_btn = tk.Button(action_frame, text="Exit Game", bg="#3d3d60", fg="#ffffff", font=("Microsoft YaHei", 11),
                                  command=self.exit_game)
        self.exit_btn.pack(fill=tk.X, padx=5, pady=5)

        # Dialogue input area
        input_frame = tk.LabelFrame(bottom_frame, text="Input", bg="#2a2a3e", fg="#ffffff",
                                    font=("Microsoft YaHei", 12, "bold"))
        input_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.input_text = tk.Text(input_frame, height=3, bg="#2d2d40", fg="#ffffff", font=("Microsoft YaHei", 11))
        self.input_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Add a hint text
        tip_label = tk.Label(input_frame, text="Please enter content and click the 'Talk to Character' or 'Interact with Environment' button",
                             bg="#2a2a3e", fg="#b0b0c0", font=("Microsoft YaHei", 9, "italic"))
        tip_label.pack(side=tk.RIGHT, padx=5, pady=2)

        # Game log area
        log_frame = tk.LabelFrame(main_frame, text="Game Log", bg="#2a2a3e", fg="#ffffff",
                                  font=("Microsoft YaHei", 12, "bold"))
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, bg="#2d2d40", fg="#ffffff",
                                                  font=("Microsoft YaHei", 10))
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Redirect standard output to the log area
        self.stdout_redirector = StdoutRedirector(self.log_text)
        sys.stdout = self.stdout_redirector

    def initialize_game(self):
        # Start the game initialization thread to avoid UI freezing
        threading.Thread(target=self._initialize_game_thread, daemon=True).start()

    def _initialize_game_thread(self):
        # Modify global variables to apply settings
        global use_model, try_chance, max_inserted_scenes, max_new_scene_generations

        # Save the current settings values
        selected_model = self.settings['model']
        selected_try_chance = self.settings['try_chance']
        selected_max_inserted_scenes = self.settings['max_inserted_scenes']
        selected_max_new_scene_generations = self.settings['max_new_scene_generations']

        # Directly modify global variables
        use_model = selected_model
        try_chance = selected_try_chance
        max_inserted_scenes = selected_max_inserted_scenes
        max_new_scene_generations = selected_max_new_scene_generations

        print(f"\n===== Confirmation: Game Configuration is in Effect =====")
        print(f"Model in use: {use_model}")
        print(f"Scene try chances: {try_chance}")
        print(f"Maximum inserted scenes: {max_inserted_scenes}")
        print(f"Maximum new scene generations: {max_new_scene_generations}")
        print(f"============================\n")

        # Create a function to force the use of the selected model settings
        def create_actor_with_model(name, age, gender):
            actor = Actor(name, age, gender)
            # Reset the talk_client according to the user's selected model
            if selected_model == "gpt-4o-mini":
                actor.talk_client = OpenAI(
                    api_key=openai_api_key,
                    base_url="https://api.openai.com/v1"
                )
            elif selected_model == "deepseek-chat":
                actor.talk_client = OpenAI(
                    api_key=deepseek_api_key,
                    base_url="https://api.deepseek.com/v1"
                )
            elif selected_model == "qwen3-235b-a22b":
                actor.talk_client = OpenAI(
                    api_key=qwen_api_key,
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
                )
            return actor

        # Initialize characters
        # Create the investigator (player character)
        self.player = Player("Howard", 25, "Male")

        # Use the custom function to create NPC characters, ensuring the correct model is used
        librarian = create_actor_with_model("Martha Deer", 57, "Female")
        librarian.add_memory("I am the administrator of the Harbor Town Library and have been working here for 30 years")
        librarian.add_memory("There have been some strange things happening in town recently, especially among the residents near the sea")
        librarian.add_memory("I have a collection of some forbidden books about ancient myths, including the 'Book of Eibon'")
        librarian.add_relationship(self.player.name, "Cautious", "I know he is an investigator, but I'm not sure if I can trust him")
        librarian.add_trait("Cautious")
        librarian.add_trait("Knowledgeable")
        librarian.add_trait("Has a complex curiosity and fear of occult knowledge")

        professor = create_actor_with_model("William Akeley", 68, "Male")
        professor.add_memory("I am a former professor at Miskatonic University, researching ancient civilizations and myths")
        professor.add_memory("I witnessed the Harbor Town incident ten years ago, those indescribable beings")
        professor.add_memory("My sanity has been damaged, but I still try to prevent the impending disaster")
        professor.add_relationship(self.player.name, "Ally", "I think he might be the key to stopping the ritual")
        professor.add_relationship("Martha Deer", "Accomplice", "She helped me preserve some key ancient books")
        professor.add_trait("Mentally unstable")
        professor.add_trait("Wise but paranoid")
        professor.add_trait("Brave but scarred")

        cultist = create_actor_with_model("Joseph Marsh", 45, "Male")
        cultist.add_memory("I am a descendant of the Deep Ones and am loyal to Dagon and Hydra")
        cultist.add_memory("On the surface, I'm an ordinary fisherman, but in reality, I'm responsible for monitoring outsiders in town")
        cultist.add_memory("I know about the upcoming ritual, and my blood allows me to call forth the beings from the sea")
        cultist.add_relationship(self.player.name, "Hostile", "I suspect he wants to interfere with our ritual")
        cultist.add_relationship("William Akeley", "Hate", "He knows too much and must be dealt with")
        cultist.add_trait("Fanatical")
        cultist.add_trait("Dual personality")
        cultist.add_trait("Merciless")

        # Create the director
        self.director = Director()

        # Add actors to the director's management
        self.director.add_actor(librarian)
        self.director.add_actor(professor)
        self.director.add_actor(cultist)

        # Create the screenwriter
        self.screenwriter = Screenwriter()

        # Load the CoC-style script
        coc_script = {
            "scene_1": {
                "description": "You are a federal investigator in the United States, coming to Harbor Town to investigate the mysterious disappearances of people. Now you are in the gloomy and damp Harbor Town Library. Heavy rain is pouring outside the window, and thunder is rumbling. Under the dim lights in the library, ancient bookshelves are neatly arranged, and the air is filled with the smell of mold and ancient books. An old-fashioned wall clock in the corner ticks, occasionally making an uncoordinated sound.",
                "characters": ["Howard", "Martha Deer"],
                "dialogues": [
                    {"character": "Martha Deer",
                     "content": "(Nervously organizing the bookshelves) These days have been tough in town, sir. What brings you here?"},
                ]
            },
            "scene_2": {
                "description": "The basement of the library. A small, dim space with old oil lamps hanging on the walls. In the middle is a large wooden table with several ancient books and manuscripts spread out on it. The air is even more turbid, and strange patterns are formed by the water stains on the walls. There is a locked iron box in the corner.",
                "characters": ["Howard", "Martha Deer"],
                "dialogues": [
                    {"character": "Martha Deer",
                     "content": "(In a hushed voice) These are our non-public collections. Some knowledge... is better left undiscovered."},
                ]
            },
            "scene_3": {
                "description": "Professor Akeley's cottage. An isolated cottage on the outskirts of Harbor Town, surrounded by dense woods. The cottage is filled with books, notes, and strange collectibles. Mysterious symbols and maps are hanging on the walls. The flames in the fireplace cast flickering shadows. A faint smell of seawater and herbs fills the air.",
                "characters": ["Howard", "William Akeley", "Martha Deer"],
                "dialogues": [
                    {"character": "William Akeley",
                     "content": "(Hands trembling slightly, eyes darting) Have you found the 'Book of Eibon'? Time is running out. 'They' are about to awaken... (Suddenly lowering his voice) You're being followed. Be careful of those 'fishermen'. They're not human..."},
                ]
            },
            "scene_4": {
                "description": "Harbor Town Beach, at night. The moonlight is blocked by thick clouds, and only sporadic starlight illuminates the beach. The waves crash against the shore, making a low sound. There seem to be several figures standing on the rocks in the distance, performing some kind of ritual. The air is filled with a strong smell of salt and an indescribable odor.",
                "characters": ["Howard", "Joseph Marsh", "William Akeley"],
                "dialogues": [
                    {"character": "Joseph Marsh",
                     "content": "(Standing next to the altar, holding a strange statue with both hands) Outsider, you shouldn't be here. This sea belongs to the great beings, and we are about to receive their blessing."},
                ]
            }
        }

        # Get the list of script IDs
        self.script_ids = list(coc_script.keys())

        # The director loads the script
        self.director.load_script(coc_script)

        # The screenwriter also loads the same script
        self.screenwriter.load_initial_script(coc_script)

        # Set the current scene
        self.current_scene_index = 0
        self.director.set_current_scene(self.script_ids[self.current_scene_index])

        # Output the actually used model
        print(f"\n===== Model Confirmation =====")
        print(f"Model endpoint used by Martha Deer: {librarian.talk_client.base_url}")
        print(f"Model endpoint used by William Akeley: {professor.talk_client.base_url}")
        print(f"Model endpoint used by Joseph Marsh: {cultist.talk_client.base_url}")
        print(f"====================\n")

        # Start the first scene
        self.root.after(0, self.start_scene)

    def start_scene(self):
        # Get the current scene ID
        current_scene_id = self.script_ids[self.current_scene_index]
        # Ensure the current scene is set correctly
        self.director.set_current_scene(current_scene_id)

        # Check and create new characters in the current scene
        self.director.ensure_all_characters_exist(current_scene_id, self.player.name)

        print(f"\nCurrent scene ID: {current_scene_id}")

        # Generate the scene description
        detailed_scene = self.screenwriter.generate_scene_description(current_scene_id, self.director,
                                                                      self.player.get_player_name())
        self.scene_text.delete(1.0, tk.END)
        self.scene_text.insert(tk.END, detailed_scene)

        # Add the generated scene description to the dialogue_history
        self.screenwriter.add_dialogue_record("Narrator", "Scene Description", detailed_scene)

        # Check if there are initial dialogues in the current scene. If so, let the NPC characters perform first
        scene_info = self.director.script.get(current_scene_id, {})
        initial_dialogues = scene_info.get("dialogues", [])

        if initial_dialogues:
            print("\n==== Dialogue Start ====")
            for dialogue in initial_dialogues:
                character_name = dialogue.get("character")
                content = dialogue.get("content")

                # Skip the player character's dialogue
                if character_name == self.player.get_player_name():
                    continue

                # Display the NPC's dialogue
                print(f"\n{character_name}: {content}")
                self.scene_text.insert(tk.END, f"\n\n{character_name}: {content}")

                # Record the dialogue in the screenwriter's dialogue history
                self.screenwriter.add_dialogue_record(character_name, "Scene Dialogue", content)

        # Update the list of characters available for dialogue
        self.update_character_list()

        # Reset the scene state
        self.scene_finished = False
        self.last_interaction = ""

        # Initialize the scene try count counter
        self.scene_try_count = 0

        # Display the scene try count limit
        print(f"\n==== Current scene try chances: {self.settings['try_chance']} times ====")

    def update_character_list(self):
        # Get the characters available for dialogue in the current scene
        self.characters = self.director.get_scene_characters(player=self.player)

        # Update the list box
        self.character_listbox.delete(0, tk.END)
        for character in self.characters:
            self.character_listbox.insert(tk.END, character)

    def talk_to_character(self):
        if self.scene_finished:
            messagebox.showinfo("Prompt", "The current scene has ended. Please proceed to the next scene")
            return

        # Check if there are any more try chances
        if self.scene_try_count >= self.settings['try_chance'] and not self.scene_finished:
            messagebox.showinfo("Prompt", f"You have reached the scene try count limit ({self.settings['try_chance']} times)")
            self.handle_scene_timeout()
            return

        # Check if a character is selected
        selected_idx = self.character_listbox.curselection()
        if not selected_idx:
            messagebox.showinfo("Prompt", "Please select a character first")
            return

        selected_character = self.characters[selected_idx[0]]

        # Get the dialogue content
        dialogue = self.input_text.get(1.0, tk.END).strip()
        if not dialogue:
            messagebox.showinfo("Prompt", "Please enter dialogue content")
            return

        # Increase the scene try count
        self.scene_try_count += 1
        print(f"\n==== Try count: {self.scene_try_count}/{self.settings['try_chance']} ====")

        # Record the last dialogue
        self.last_interaction = f"{self.player.name} said to {selected_character}: {dialogue}"

        # Add the player's dialogue record
        self.screenwriter.add_dialogue_record(self.player.name, "Dialogue", dialogue, target=selected_character)

        # Add the player's dialogue to the scene text
        self.scene_text.insert(tk.END, f"\n\n{self.player.name}: {dialogue}")

        # Use the merged method to directly generate guidance
        guide_message = self.director.guide_actor_from_player_speech(dialogue, selected_character)

        # Get the character instance, not just the character name
        actor_instance = self.director.actors.get(selected_character)

        if actor_instance:
            # The actor's dialogue, using the Actor instance
            response = self.player.talk_to_actor(actor_instance, dialogue, guide_message)

            # Update the last interaction record
            self.last_interaction += f"\n{selected_character} replied: {response}"

            # Add the NPC's dialogue record
            self.screenwriter.add_dialogue_record(selected_character, "Dialogue", response, target=self.player.name)

            # Add the NPC's response to the scene text
            self.scene_text.insert(tk.END, f"\n\n{selected_character}: {response}")
            self.scene_text.see(tk.END)

            print(f"\n{selected_character}: {response}")

            # Determine if the current scene should continue
            if not self.director.is_scene_continuing(response):
                print("The current scene has ended")
                self.scene_finished = True

                # Get the next scene ID (if available)
                next_scene = self.script_ids[self.current_scene_index + 1] if self.current_scene_index + 1 < len(
                    self.script_ids) else None

                # Determine if a new scene needs to be inserted
                should_generate = self.director.should_generate_new_script(self.screenwriter,
                                                                           self.script_ids[self.current_scene_index],
                                                                           next_scene)

                if should_generate and self.inserted_scene_count < self.settings['max_inserted_scenes']:
                    print(
                        f"\n==== According to the plot development, a new scene needs to be inserted ({self.inserted_scene_count + 1}/{self.settings['max_inserted_scenes']}) ====")

                    # Generate a new scene
                    new_script = self.screenwriter.generate_new_script(self.script_ids[self.current_scene_index],
                                                                       dialogue_history=self.screenwriter.get_dialogue_history())

                    if "error" not in new_script:
                        # Update the director's script
                        self.director.load_script(self.screenwriter.initial_script)

                        # Re - get and sort the script ID list
                        self.script_ids = list(self.screenwriter.initial_script.keys())
                        new_scene_id = list(new_script.keys())[0]
                        print(f"\nSuccessfully generated a new scene: {new_scene_id}")

                        # Set the new scene as the next scene
                        next_scene = new_scene_id

                        # Increase the inserted scene count
                        self.inserted_scene_count += 1

                        # Check and create new characters
                        self.director.check_and_create_new_characters(self.script_ids,
                                                                      self.script_ids.index(new_scene_id),
                                                                      self.player.name)
                    else:
                        print(f"\nFailed to generate a new scene: {new_script.get('error')}")
                elif should_generate and self.inserted_scene_count >= self.settings['max_inserted_scenes']:
                    print(
                        f"\n==== You have reached the limit of inserting new scenes ({self.settings['max_inserted_scenes']} times). Continuing with the original script ====")

                # Generate the scene ending description
                ending_description = self.screenwriter.end_scene(self.last_interaction, self.director,
                                                                 self.script_ids[self.current_scene_index], next_scene)

                # Display the scene ending description
                self.scene_text.insert(tk.END, f"\n\n{ending_description}")
                self.scene_text.see(tk.END)
                print(f"\n{ending_description}")

                # Add the scene transition description
                self.screenwriter.add_dialogue_record("Narrator", "Scene Transition",
                                                      f"Transition from {self.script_ids[self.current_scene_index]} scene to {next_scene if next_scene else 'End of the story'}")

                # Prompt the user to proceed to the next scene
                messagebox.showinfo("Scene End", "The current scene has ended. Please click the 'Next Scene' button to continue")
            else:
                print("The current scene continues")

                # Check if the try count limit has been reached
                if self.scene_try_count >= self.settings['try_chance'] and not self.scene_finished:
                    print(
                        f"\n==== Scene try chances have been used up ({self.scene_try_count}/{self.settings['try_chance']}). Forcing the end of the scene ====")
                    self.handle_scene_timeout()
        else:
            print(f"Error: Could not find an instance of the character {selected_character}")

        # Clear the input box
        self.input_text.delete(1.0, tk.END)

    def interact_with_environment(self):
        if self.scene_finished:
            messagebox.showinfo("Prompt", "The current scene has ended. Please proceed to the next scene")
            return

        # Check if there are any more try chances
        if self.scene_try_count >= self.settings['try_chance'] and not self.scene_finished:
            messagebox.showinfo("Prompt", f"You have reached the scene try count limit ({self.settings['try_chance']} times)")
            self.handle_scene_timeout()
            return

        # Get the interaction content
        interaction = self.input_text.get(1.0, tk.END).strip()
        if not interaction:
            messagebox.showinfo("Prompt", "Please enter interaction content")
            return

        # Increase the scene try count
        self.scene_try_count += 1
        print(f"\n==== Try count: {self.scene_try_count}/{self.settings['try_chance']} ====")

        # Record the last interaction
        self.last_interaction = f"{self.player.name} interacted with the environment: {interaction}"

        # Add the player's interaction record
        self.screenwriter.add_dialogue_record(self.player.name, "Environment Interaction", interaction)

        # Add the player's interaction to the scene text
        self.scene_text.insert(tk.END, f"\n\n{self.player.name} Action: {interaction}")

        # The screenwriter processes the player's action
        action_response = self.screenwriter.transform_scene(self.script_ids[self.current_scene_index], interaction)

        # Update the last interaction record
        self.last_interaction += f"\nEnvironment response: {action_response}"

        # Add the environment response to the scene text
        self.scene_text.insert(tk.END, f"\n\n{action_response}")
        self.scene_text.see(tk.END)

        print(f"\n{action_response}")

        # Determine if the current scene should continue
        if not self.director.is_scene_continuing(action_response):
            print("The current scene has ended")
            self.scene_finished = True

            # Get the next scene ID (if available)
            next_scene = self.script_ids[self.current_scene_index + 1] if self.current_scene_index + 1 < len(
                self.script_ids) else None

            # Determine if a new scene needs to be inserted
            should_generate = self.director.should_generate_new_script(self.screenwriter,
                                                                           self.script_ids[self.current_scene_index],
                                                                           next_scene)

            if should_generate and self.inserted_scene_count < self.settings['max_inserted_scenes']:
                print(
                    f"\n==== According to the plot development, a new scene needs to be inserted ({self.inserted_scene_count + 1}/{self.settings['max_inserted_scenes']}) ====")

                # Generate a new scene
                new_script = self.screenwriter.generate_new_script(self.script_ids[self.current_scene_index],
                                                                       dialogue_history=self.screenwriter.get_dialogue_history())

                if "error" not in new_script:
                    # Update the director's script
                    self.director.load_script(self.screenwriter.initial_script)

                    # Re - get and sort the script ID list
                    self.script_ids = list(self.screenwriter.initial_script.keys())
                    new_scene_id = list(new_script.keys())[0]
                    print(f"\nSuccessfully generated a new scene: {new_scene_id}")

                    # Set the new scene as the next scene
                    next_scene = new_scene_id

                    # Increase the inserted scene count
                    self.inserted_scene_count += 1

                    # Check and create new characters
                    self.director.check_and_create_new_characters(self.script_ids,
                                                                      self.script_ids.index(new_scene_id),
                                                                      self.player.name)
                else:
                    print(f"\nFailed to generate a new scene: {new_script.get('error')}")
            elif should_generate and self.inserted_scene_count >= self.settings['max_inserted_scenes']:
                    print(
                        f"\n==== You have reached the limit of inserting new scenes ({self.settings['max_inserted_scenes']} times). Continuing with the original script ====")

            # Generate the scene ending description
            ending_description = self.screenwriter.end_scene(self.last_interaction, self.director,
                                                                 self.script_ids[self.current_scene_index], next_scene)

            # Display the scene ending description
            self.scene_text.insert(tk.END, f"\n\n{ending_description}")
            self.scene_text.see(tk.END)
            print(f"\n{ending_description}")

            # Add the scene transition description
            self.screenwriter.add_dialogue_record("Narrator", "Scene Transition",
                                                      f"Transition from {self.script_ids[self.current_scene_index]} scene to {next_scene if next_scene else 'End of the story'}")

            # Prompt the user to proceed to the next scene
            messagebox.showinfo("Scene End", "The current scene has ended. Please click the 'Next Scene' button to continue")
        else:
            print("The current scene continues")

            # Check if the try count limit has been reached
            if self.scene_try_count >= self.settings['try_chance'] and not self.scene_finished:
                print(
                    f"\n==== Scene try chances have been used up ({self.scene_try_count}/{self.settings['try_chance']}). Forcing the end of the scene ====")
                self.handle_scene_timeout()

            # Clear the input box
            self.input_text.delete(1.0, tk.END)

    def go_to_next_scene(self):
        # Get the current scene ID
        current_scene_id = self.script_ids[self.current_scene_index]

        if not self.scene_finished:
            # If the scene is not finished, ask the user if they are sure to skip
            if not messagebox.askyesno("Confirmation", "The current scene is not finished. Are you sure you want to skip?"):
                return

            # Manually end the current scene
            print("Manually ending the current scene and proceeding to the next scene")

            # Get the next scene ID (if available)
            next_scene = self.script_ids[self.current_scene_index + 1] if self.current_scene_index + 1 < len(
                self.script_ids) else None

            if next_scene:
                # Create a simple transition description
                ending_description = self.screenwriter.end_scene(self.last_interaction or "Player chose to skip the current scene",
                                                                 self.director, current_scene_id, next_scene)

                # Display the scene ending description
                self.scene_text.insert(tk.END, f"\n\n{ending_description}")
                self.scene_text.see(tk.END)
                print(f"\n{ending_description}")

                # Add the scene transition description
                self.screenwriter.add_dialogue_record("Narrator", "Scene Transition", f"Transition from {current_scene_id} scene to {next_scene}")
            else:
                print("\nNo more scenes. The story has ended")
                messagebox.showinfo("Game Over", "Congratulations! You have completed all the scenes!")
                return

        # Move to the next scene
        self.current_scene_index += 1

        # Check if there are more scenes
        if self.current_scene_index < len(self.script_ids):
            print(f"\n==== Proceeding to the next scene: {self.script_ids[self.current_scene_index]} ====")
            # Start the new scene
            self.start_scene()
        else:
            # Update the script ID list (there may be new scenes generated)
            self.script_ids = list(self.screenwriter.initial_script.keys())

            # Check if a brand - new scene needs to be generated (all scenes are completed)
            if self.current_scene_index >= len(self.script_ids):
                # Check if the scene generation count limit has been reached
                if self.new_scene_generation_count >= self.settings['max_new_scene_generations']:
                    print(
                        f"\n==== You have reached the limit of scene generations ({self.settings['max_new_scene_generations']} times). Preparing to end the story ====")
                    # Generate the ending scene
                    ending_prompt = "This is the ending scene of the story. Please provide a satisfying, logical, and emotionally impactful ending based on the previous plot. Ensure that all major plot lines are appropriately resolved."

                    # Use a special marker to tell the screenwriter this is the ending
                    new_script = self.screenwriter.generate_new_script(current_scene_id, ending_prompt,
                                                                       dialogue_history=self.screenwriter.get_dialogue_history())

                    if "error" not in new_script:
                        # Update the script ID list
                        self.script_ids = list(self.screenwriter.initial_script.keys())
                        ending_scene_id = list(new_script.keys())[0]
                        print(f"\nSuccessfully generated the ending scene: {ending_scene_id}")

                        # Update the director's script
                        self.director.load_script(self.screenwriter.initial_script)

                        # Ensure the next scene is the ending scene
                        self.current_scene_index = self.script_ids.index(current_scene_id) + 1
                        # Check and create new characters
                        self.director.check_and_create_new_characters(self.script_ids, self.current_scene_index,
                                                                      self.player.name)

                        # Start the new scene
                        self.start_scene()
                    else:
                        print(f"\nFailed to generate the ending scene: {new_script.get('error')}")
                        messagebox.showinfo("Game Over", "The story has ended. Thank you for playing!")
                else:
                    print(f"\n==== All planned scenes have been completed. Trying to generate a new scene ({self.new_scene_generation_count + 1}/{self.settings['max_new_scene_generations']}) ====")
                    # Generate a new scene
                    new_script = self.screenwriter.generate_new_script(current_scene_id,
                                                                       dialogue_history=self.screenwriter.get_dialogue_history())

                    if "error" not in new_script:
                        # Update the script ID list
                        self.script_ids = list(self.screenwriter.initial_script.keys())
                        new_scene_id = list(new_script.keys())[0]
                        print(f"\nSuccessfully generated a new scene: {new_scene_id}")

                        # Update the director's script
                        self.director.load_script(self.screenwriter.initial_script)

                        # Increase the scene generation count
                        self.new_scene_generation_count += 1

                        # Check and create new characters
                        self.director.check_and_create_new_characters(self.script_ids, self.current_scene_index,
                                                                      self.player.name)

                        # Start a new scene
                        self.start_scene()
                    else:
                        print(f"\nFailed to generate the ending scene: {new_script.get('error')}")
                        messagebox.showinfo("Game Over", "The story has ended. Thank you for your participation!")

            else:
            # There are still scenes. Start a new scene.
                self.start_scene()


    def exit_game(self):
        if messagebox.askyesno("Confirm Exit", "Are you sure you want to exit the game?"):
            # Export the dialogue history to a JSON file for evaluation.
            import json
            import datetime

            # Create a file name with a timestamp.
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            dialogue_history_file = f"dialogue_history_{timestamp}.json"

            # Convert the dialogue history to a serializable format.
            dialogue_history = self.screenwriter.get_all_dialogue_history()  # Get the entire dialogue history.

            # Save the dialogue history to a file.
            with open(dialogue_history_file, "w", encoding="utf-8") as f:
                json.dump(dialogue_history, f, ensure_ascii=False, indent=2)

            print(f"\nThe dialogue history has been exported to the file: {dialogue_history_file}")

            # Exit the application.
            self.root.destroy()


    def handle_scene_timeout(self):
        # Get the current scene ID.
        current_scene_id = self.script_ids[self.current_scene_index]

        # Get the next scene ID (if available).
        next_scene = self.script_ids[self.current_scene_index + 1] if self.current_scene_index + 1 < len(self.script_ids) else None

        # Determine if a new scene needs to be inserted.
        should_generate = self.director.should_generate_new_script(self.screenwriter, current_scene_id, next_scene)

        if should_generate and self.inserted_scene_count < self.settings['max_inserted_scenes']:
            print(
                f"\n==== According to the plot development, a new scene needs to be inserted ({self.inserted_scene_count + 1}/{self.settings['max_inserted_scenes']}) ====")

            # Generate a new scene.
            new_script = self.screenwriter.generate_new_script(current_scene_id,
                                                               dialogue_history=self.screenwriter.get_dialogue_history())

            if "error" not in new_script:
                # Update the director's script.
                self.director.load_script(self.screenwriter.initial_script)

                # Re - get and sort the script ID list.
                self.script_ids = list(self.screenwriter.initial_script.keys())
                new_scene_id = list(new_script.keys())[0]
                print(f"\nSuccessfully generated a new scene: {new_scene_id}")

                # Set the new scene as the next scene.
                next_scene = new_scene_id

                # Increase the inserted scene count.
                self.inserted_scene_count += 1

                # Check and create new characters.
                self.director.check_and_create_new_characters(self.script_ids,
                                                                  self.script_ids.index(new_scene_id),
                                                                  self.player.name)
            else:
                print(f"\nFailed to generate a new scene: {new_script.get('error')}")
        elif should_generate and self.inserted_scene_count >= self.settings['max_inserted_scenes']:
            print(
                f"\n==== The limit on the number of inserted new scenes ({self.settings['max_inserted_scenes']}) has been reached. Continue using the original script. ====")

        # Generate the scene ending description.
        ending_description = self.screenwriter.end_scene(self.last_interaction or "Scene attempt limit reached",
                                                         self.director, current_scene_id, next_scene)

        # Display the scene ending description.
        self.scene_text.insert(tk.END, f"\n\n{ending_description}")
        self.scene_text.see(tk.END)
        print(f"\n{ending_description}")

        # Add the scene transition description.
        self.screenwriter.add_dialogue_record("Narrator", "Scene Transition",
                                              f"Transition from {current_scene_id} scene to {next_scene if next_scene else 'End of the story'}")

        # Set the scene as finished.
        self.scene_finished = True

        # Prompt the user to enter the next scene.
        messagebox.showinfo("Scene End", "Scene attempt limit reached. Please click the 'Next Scene' button to continue.")


    # Add an enter key handling function.
    def on_enter_pressed(self, event):
        # Prevent the enter key from inserting a new line in the text box.
        self.submit_input()
        return "break"  # Prevent the default behavior

if __name__ == "__main__":
    root = tk.Tk()
    app = CoCGameGUI(root)
    root.mainloop()

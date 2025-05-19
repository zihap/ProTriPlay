import openai
import faiss
import numpy as np
from typing import List, Dict, Tuple
import pickle
import os
from openai import OpenAI
from config import openai_api_key, deepseek_api_key, qwen_api_key, http_proxy, https_proxy, use_model

# Set proxy
os.environ["http_proxy"] = http_proxy
os.environ["https_proxy"] = https_proxy


# Add a generic streaming processing helper function at the top of the file
def handle_stream_response(client, model, messages, extra_body=None):
    """A generic helper function for handling both streaming and non-streaming responses

    Args:
        client: OpenAI client
        model: Model name
        messages: List of messages
        extra_body: Additional request body parameters

    Returns:
        The response text from the model
    """
    # Set default extra_body if not provided
    if extra_body is None:
        extra_body = {}

    # Add enable_thinking parameter for Qwen models
    if model.startswith("qwen"):
        extra_body["enable_thinking"] = False

    # Check if streaming output is used
    if model == "qwen3-235b-a22b":
        # Streaming output
        response_content = ""
        response_stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            extra_body=extra_body
        )

        # Collect streaming response
        for chunk in response_stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                response_content += chunk.choices[0].delta.content

        return response_content
    else:
        # Non-streaming output
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            extra_body=extra_body
        )

        return response.choices[0].message.content

class Actor:
    def __init__(self, name, age, gender, memory_path=None):
        self.name = name
        self.age = age
        self.gender = gender
        self.embedding_dim = 1536  # OpenAI embedding dimension
        self.memories = []
        self.memory_embeddings = None
        self.index = None
        self.relationships = {}  # Store relationships with other Actors
        self.traits = []  # Store character traits

        # Create two different clients
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
        """Get the vector embedding of the text using OpenAI's API"""
        response = self.embedding_client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    def retrieve_relevant_memories(self, query: str, k: int = 3) -> List[str]:
        """Retrieve memories relevant to the query"""
        if len(self.memories) == 0:
            return []

        query_embedding = self._get_embedding(query).reshape(1, -1)
        scores, indices = self.index.search(query_embedding, min(k, len(self.memories)))
        return [self.memories[idx] for idx in indices[0]]

    def add_trait(self, trait_description):
        """Add a character trait to the role

        Args:
            trait_description: Description of the character trait, e.g., "cautious", "impulsive", "kind", etc.
        """
        if trait_description not in self.traits:
            self.traits.append(trait_description)
            # Add the character trait as a memory
            self.add_memory(f"My character trait is {trait_description}")
        return len(self.traits) - 1

    def get_traits(self):
        """Get all character traits of the role

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

        Args:
            other_actor: Another Actor object or Actor name
            relationship_type: Type of relationship, e.g., "friend", "family", "colleague", etc.
            description: Relationship description, can provide more detailed relationship information
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
        """Get the relationship information with the specified Actor"""
        other_name = other_actor.name if hasattr(other_actor, 'name') else other_actor
        return self.relationships.get(other_name, [])

    def get_all_relationships(self):
        """Get all relationship information"""
        return self.relationships

    def speak(self, information, speaker=None, guidance=None):
        """Answer questions based on memories and relationships, using Deepseek's API

        Args:
            information: Conversation content or question
            speaker: The person talking to the role (can be a Player object or string name)
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

        # Check the relationships of other characters mentioned in the conversation
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
            traits_context = "Your character traits are: " + ", ".join(self.traits) + ". Please shape your response based on these character traits.\n"

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

        # Add the conversation record to the memory
        if speaker_name:
            self.add_memory(f"{speaker_name} said to me: {information}")

        # Build the message list
        messages = [
            {"role": "system", "content": f"You are a theater actor, and you are playing the role of {self.name}, a {self.age}-year-old {self.gender} in the play. Please make appropriate responses based on the identity of the conversationalist and your character traits.\n\nYour response must follow the following format: \"({self.name}'s expression and action) Conversation content\". The parentheses must contain the name of the character you are playing as the subject, clearly describe the expression and action, and the content after the parentheses should directly follow the conversation content. For example: \"({self.name} nervously clenches fists) I don't know what you're talking about.\" or \"({self.name} smiles and nods) I'm glad to see you.\""},
            {"role": "user", "content": (context + information) if context else information}
        ]

        # Use the general processing function to get the response
        response_content = handle_stream_response(self.talk_client, use_model, messages)

        # Check and correct the response format
        if not response_content.startswith(f"({self.name}"):
            # If the response does not meet the required format, try to add the default format
            response_content = f"({self.name} calmly says) {response_content}"

        # Add the reply record to the memory
        if speaker_name:
            self.add_memory(f"I said to {speaker_name}: {response_content}")

        return response_content

    def should_speak(self, context, speaker=None):
        """Determine if the Actor needs to speak. If so, execute the speak method. (Optional)

        Args:
            context: Current conversation or scene context
            speaker: The person talking to the role (can be a Player object or string name)

        Returns:
            If the Actor needs to speak, return the execution result of the speak method; otherwise, return None
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

        # Build the message list
        messages = [
            {"role": "system", "content": "You need to help the character determine if they should speak in the current situation. Please only answer 'Yes' or 'No', and give a brief reason."},
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
        self.script = {}  # Store the script, organized by scene
        self.current_scene = None  # Current scene

        if use_model == "gpt-4o-mini":
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

        Args:
            character_name: Character name
            scene_id: Current scene ID
            player_name: Player character name

        Returns:
            A dictionary containing character information
        """
        # Get the current scene description
        scene_desc = self.get_scene_description(scene_id)

        # Build the prompt
        prompt = f"""Based on the following scene description and character name, generate a complete character profile:

Scene description: {scene_desc}

Character name: {character_name}

Player character name: {player_name}

Please generate character information including the following:
1. Character age
2. Character gender
3. Character background story (3 - 5 items)
4. Character traits (2 - 3 items)
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
            {"role": "system", "content": "You are an AI assistant creating characters. Generate character information that fits the scenario based on the scene and character name."},
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
            "background": [f"I am {character_name}, a new character in the current scene"],
            "traits": ["mysterious"],
            "relationship": {
                "type": "stranger",
                "description": "Just met"
            }
        }

    def ensure_all_characters_exist(self, scene_id, player_name):
        """Check if all characters in the scene exist. If not, create them

        Args:
            scene_id: Current scene ID
            player_name: Player character name, used to exclude the player character and establish relationships
        """
        # Get all characters in the current scene
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
                for memory in profile.get("background", [f"I am {character_name}, a new character in the current scene"]):
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
                print(f"Character created: {character_name}, Age: {profile.get('age', 30)}, Gender: {profile.get('gender', 'unknown')}")
                print(f"Character traits: {', '.join(profile.get('traits', ['mysterious']))}")
                print(f"Relationship with player: {relationship.get('type', 'stranger')} - {relationship.get('description', 'Just met')}")

    def check_and_create_new_characters(self, scene_ids, current_scene_index, player_name):
        """Check and create new characters that may appear in new scenes

        Args:
            scene_ids: List of scene IDs
            current_scene_index: Index of the current scene
            player_name: Player character name
        """
        if current_scene_index < len(scene_ids):
            current_scene_id = scene_ids[current_scene_index]
            self.ensure_all_characters_exist(current_scene_id, player_name)

    def load_script(self, script_dict):
        """Load the script

        Args:
            script_dict: A dictionary containing the script content, formatted as:
            {
                "scene_1": {
                    "description": "Scene description",
                    "characters": ["Character name 1", "Character name 2", ...],
                    "dialogues": [
                        {"character": "Character name", "content": "Dialogue content"},
                        ...
                    ]
                },
                ...
            }
        """
        self.script = script_dict

    def set_current_scene(self, scene_id):
        """Set the current scene"""
        if scene_id in self.script:
            self.current_scene = scene_id
            return True
        return False

    def get_current_scene(self):
        """Get the current scene"""
        return self.current_scene

    def get_scene_description(self, scene_id=None):
        """Get the description of the specified scene"""
        scene = scene_id if scene_id else self.current_scene
        if scene in self.script:
            return self.script[scene].get("description", "")
        return ""

    def get_scene_characters(self, scene_id=None, player=None):
        """Get the list of characters in the specified scene, excluding the player character

        Args:
            scene_id: Scene ID. If None, use the current scene.
            player: Player object or player character name, which will be excluded from the result.

        Returns:
            A list of characters excluding the player character.
        """
        scene = scene_id if scene_id else self.current_scene
        if scene not in self.script:
            return []

        characters = self.script[scene].get("characters", [])

        # If the player parameter is provided, exclude the player character from the list.
        if player:
            player_name = player
            # If the player is a Player object, get its name.
            if hasattr(player, 'get_player_name'):
                player_name = player.get_player_name()
            elif hasattr(player, 'name'):
                player_name = player.name

            # Filter out the player character.
            characters = [char for char in characters if char != player_name]

        return characters

    def guide_actor_from_player_speech(self, player_speech, actor_name):
        """Generate acting guidance for an actor directly from the player's speech, combining the functions of generate_guidance_from_player_speech and guide_actor.

        Args:
            player_speech: The content of the player's speech.
            actor_name: The name of the actor to be guided.

        Returns:
            Acting guidance for the actor.
        """
        if self.current_scene is None or actor_name not in self.actors:
            return "Unable to guide: Current scene not set or actor not found."

        # Get the current scene information.
        scene_info = self.script.get(self.current_scene, {})
        scene_desc = scene_info.get("description", "")

        # Get the actor's dialogues in the current scene.
        dialogues = scene_info.get("dialogues", [])
        actor_dialogues = [d for d in dialogues if d.get("character") == actor_name]

        # Get the character information.
        actor = self.actors.get(actor_name)
        actor_traits = actor.get_traits() if hasattr(actor, "get_traits") else []
        traits_text = ", ".join(actor_traits) if actor_traits else "No specific character traits."

        # Build the prompt information.
        prompt = f"""As a theater director, please provide acting guidance for the actor '{actor_name}' based on the following information:

Scene description: {scene_desc}

Actor's character traits: {traits_text}

Player's recent speech: "{player_speech}"

Actor's lines:
"""
        for dialogue in actor_dialogues:
            prompt += f"- {dialogue.get('content')}\n"

        prompt += """
Please analyze the intention, emotion, and possible implied meaning of the player's speech, and then provide specific acting guidance, including:
1. Emotional expression suggestions
2. Body language guidance
3. Tone and rhythm suggestions
4. Whether certain information should be revealed
5. How to stay true to the character's traits

Please directly provide the guidance content without any preamble:"""

        # Build the message list.
        messages = [
            {"role": "system", "content": "You are an experienced theater director, good at analyzing the player's speech and providing response guidance for the actor."},
            {"role": "user", "content": prompt}
        ]

        # Use the general processing function to get the response.
        return handle_stream_response(self.client, use_model, messages)

    def is_scene_continuing(self, last_dialogue, screenwriter=None, detailed_scene=None):
        """Determine if the script is still in the current scene.

        Args:
            last_dialogue: The content of the most recent dialogue (optional. If the screenwriter is provided, the dialogue history from the screenwriter will be used first).
            screenwriter: Screenwriter object, used to get the dialogue history (optional).
            detailed_scene: Detailed scene description (optional. It will be used first if provided).

        Returns:
            A boolean value indicating whether to continue in the current scene.
        """
        if self.current_scene is None:
            return False

        scene_info = self.script.get(self.current_scene, {})
        dialogues = scene_info.get("dialogues", [])

        # Check if all dialogues have been completed.
        if not dialogues:
            return False

        # Get the player character's goal description.
        player_goal = ""
        scene_description = ""

        # Use the provided detailed scene description first.
        if detailed_scene:
            scene_description = detailed_scene
        # If no detailed scene description is provided, try to get it from the screenwriter.
        elif screenwriter and hasattr(screenwriter, 'scene_descriptions') and self.current_scene in screenwriter.scene_descriptions:
            scene_description = screenwriter.scene_descriptions.get(self.current_scene, "")

        # Extract the player character's goal description.
        if scene_description:
            # The typical format contains paragraphs with "Player character goal" or "Player goal".
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

        # Use the screenwriter's dialogue history (if provided).
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

        # Build the prompt.
        prompt = f"Please analyze the following situation and determine if the script should continue in the current scene:\n\n"
        prompt += f"Scene description: {scene_info.get('description', '')}\n\n"

        # Add the player's goal information (if available).
        if player_goal:
            prompt += f"Player character's goal in this scene: {player_goal}\n\n"

        prompt += "Expected dialogue content:\n"
        for d in dialogues[-3:]:  # Only use the most recent dialogues as context.
            prompt += f"{d.get('character')}: {d.get('content')}\n"

        # Add the dialogue history or the latest dialogue.
        if recent_dialogue_history:
            prompt += recent_dialogue_history
        elif last_dialogue:
            prompt += f"\nActual latest dialogue: {last_dialogue}\n"

        prompt += "\nBased on the following criteria, determine if the current scene should continue:\n"
        prompt += "1. Whether the player character's goal has been achieved.\n"
        prompt += "2. Whether the key dialogues in the scene have been completed.\n"
        prompt += "3. Whether the dialogue has naturally reached an end point.\n"
        prompt += "4. Whether there are obvious scene transition clues.\n\n"
        prompt += "Especially note: The player character's goal is an important condition for determining if the scene should continue. If the goal has not been achieved and there are no obvious transition signals, the scene usually should continue.\n\n"
        prompt += "Please clearly determine if the scene should continue. First answer 'Yes' or 'No', then give a brief reason."

        # Build the message list.
        messages = [
            {"role": "system", "content": "You are an experienced theater director, good at analyzing scripts and performances. You need to determine if the current scene should continue or move on to the next scene."},
            {"role": "user", "content": prompt}
        ]

        # Use the general processing function to get the response.
        result = handle_stream_response(self.client, use_model, messages).lower()

        # Simply judge whether the answer is affirmative or negative.
        return "Yes" in result[:30] or "Should continue" in result[:30] or "Continue" in result[:30] or "Not achieved" in result[:50]

    def should_generate_new_script(self, screenwriter, current_scene_id, next_scene_id=None):
        """Determine if a new script/scene needs to be generated.

        Args:
            screenwriter: Screenwriter object, used to get the dialogue history and scene information.
            current_scene_id: Current scene ID.
            next_scene_id: The ID of the next planned scene (if available).

        Returns:
            A boolean value indicating whether a new scene needs to be generated.
        """
        # First, check if the current scene has ended.
        if self.is_scene_continuing(None, screenwriter):
            return False  # The current scene is still ongoing, no need to generate a new scene.

        # Get the recent dialogue history.
        recent_dialogues = screenwriter.get_dialogue_history(limit=10)
        dialogue_text = ""
        for d in recent_dialogues:
            if 'record_type' in d:
                record_type = d['record_type']
                if record_type == "Dialogue" or record_type.startswith("Dialogue"):
                    dialogue_text += f"{d['speaker']} said to {record_type}: {d['content']}\n"
                else:
                    dialogue_text += f"{d['speaker']} ({record_type}): {d['content']}\n"

        # Get the current scene information.
        current_scene = self.script.get(current_scene_id, {})
        current_scene_desc = current_scene.get("description", "")

        # Get the next scene information (if available).
        next_scene_info = ""
        if next_scene_id and next_scene_id in self.script:
            next_scene = self.script.get(next_scene_id, {})
            next_scene_desc = next_scene.get("description", "")
            next_scene_info = f"""Planned next scene description:
{next_scene_desc}"""

        # Build the prompt.
        prompt = f"""As a theater director, please determine if a new scene needs to be inserted between the current scene and the next planned scene.

Current scene description:
{current_scene_desc}

Recent dialogue history:
{dialogue_text}

{next_scene_info}

Please analyze the following points:
1. Whether the plot of the current scene has naturally ended.
2. Whether there are new plot clues in the dialogue history that need to be immediately addressed.
3. Whether there is a large plot or scene gap between the current scene and the next planned scene.
4. Whether there are unresolved conflicts or incomplete plots that need to be addressed in a new scene.

Based on the above analysis, please determine if a new scene needs to be inserted? Please only answer 'Yes' or 'No', then give a brief reason."""

        # Build the message list.
        messages = [
            {"role": "system", "content": "You are an experienced theater director, good at analyzing plot development and scene transitions."},
            {"role": "user", "content": prompt}
        ]

        # Use the general processing function to get the response.
        result = handle_stream_response(self.client, use_model, messages)

        # Determine if the answer suggests generating a new scene.
        return "Yes" in result[:10] or "Need" in result[:20] or "Should" in result[:20]


class Player:
    def __init__(self, name, age, gender):
        """Initialize the player character.

        Args:
            name: The name of the character played by the player.
            age: The age of the character played by the player.
            gender: The gender of the character played by the player.
        """
        self.name = name
        self.age = age
        self.gender = gender

    def talk_to_actor(self, actor, message, guidance=None):
        """Talk to an actor in the scene.

        Args:
            actor: Actor object.
            message: The content of the dialogue input by the player.
            guidance: Guidance generated by the director.

        Returns:
            The actor's response.
        """
        if not hasattr(actor, 'speak'):
            return f"Error: Unable to talk to this object."

        # Directly pass the player's message to the actor to generate a response.
        response = actor.speak(message, self.name, guidance)
        return response

    def interact_with_environment(self, screenwriter, action, current_scene_id=None):
        """Interact with the environment or items in the current scene.

        Args:
            screenwriter: Screenwriter object, used to handle the player's actions.
            action: The action the player wants to perform.
            current_scene_id: Current scene ID (optional).

        Returns:
            A description of the interaction result.
        """
        # Record the interaction content in the dialogue history.
        screenwriter.add_dialogue_record(self.name, "Environmental interaction", action)


        # If a scene ID is provided, update the scene description
        if current_scene_id:
            # Update the scene based on the player's action
            updated_scene = screenwriter.transform_scene(
                current_scene_id,
                action
            )

            # Set the player's current scene
            self.current_scene = updated_scene

        return updated_scene


    def get_player_name(self):
        """Get the player's name

        Returns:
            The player's name
        """
        return self.name


class Screenwriter:
    def __init__(self):
        """Initialize the screenwriter"""
        # Create an OpenAI client
        if use_model == "gpt-4o-mini":
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

        self.dialogue_history = []  # Store the dialogue history
        self.scene_descriptions = {}  # Store the scene descriptions
        self.initial_script = {}  # Store the initial script

    def load_initial_script(self, script_dict):
        """Load the initial script

        Args:
            script_dict: The initial script dictionary
        """
        self.initial_script = script_dict
        # Initialize the scene descriptions
        for scene_id, scene_data in script_dict.items():
            if "description" in scene_data:
                self.scene_descriptions[scene_id] = scene_data["description"]

    def generate_scene_description(self, scene_id, director=None, player_character=None):
        """Generate a detailed description of the scene

        Args:
            scene_id: The scene ID
            director: The Director object to get character information (optional)
            player_character: The name of the player-controlled character (optional), which will not be described in detail

        Returns:
            A detailed description of the scene, including the scene, items, and non-player characters
        """
        # Check if there is a base description for the scene
        base_description = self.scene_descriptions.get(scene_id)
        if not base_description:
            if scene_id in self.initial_script and "description" in self.initial_script[scene_id]:
                base_description = self.initial_script[scene_id]["description"]
            else:
                return f"Error: No description found for scene {scene_id}"

        # Get the list of characters in the scene
        scene_characters = []
        characters_info = ""

        if scene_id in self.initial_script and "characters" in self.initial_script[scene_id]:
            scene_characters = self.initial_script[scene_id]["characters"]

            # If a Director object is provided, get more detailed character information
            if director and hasattr(director, 'actors'):
                for char_name in scene_characters:
                    # Skip the player-controlled character
                    if player_character and char_name == player_character:
                        continue

                    # Get the character information
                    if char_name in director.actors:
                        actor = director.actors[char_name]
                        char_info = f"- {char_name}: {actor.age} years old, {actor.gender}\n"

                        # Add personality traits
                        if hasattr(actor, 'get_traits') and actor.get_traits():
                            traits = actor.get_traits()
                            char_info += f"  Personality traits: {', '.join(traits)}\n"

                        # Add relationships with other characters
                        if hasattr(actor, 'get_all_relationships'):
                            relationships = actor.get_all_relationships()
                            if relationships:
                                char_info += "  Relationships with other characters:\n"
                                for other_name, rel_list in relationships.items():
                                    if other_name in scene_characters:  # Only add relationships with characters in the scene
                                        for rel in rel_list:
                                            rel_type = rel.get('type', '')
                                            rel_desc = rel.get('description', '')
                                            if rel_desc:
                                                char_info += f"    - With {other_name}: {rel_type} ({rel_desc})\n"
                                            else:
                                                char_info += f"    - With {other_name}: {rel_type}\n"

                        characters_info += char_info

        # Build the prompt
        prompt = f"""Based on the following base description of the scene and character information, generate a more detailed description of the scene, including the scene environment, items in the scene, and non-player characters:

Base description of the scene: {base_description}

"""

        if characters_info:
            prompt += f"""Character information in the scene:
{characters_info}
"""

        if player_character:
            prompt += f"""Note: The player character "{player_character}" in the scene does not need to be described in detail because the player will control this character themselves.
"""

        prompt += """Please provide:
1. Environment description: including spatial layout, lighting, sound, smell, etc.
2. Item description: list the main items in the scene and their placement
3. Non-player character description: describe the appearance, posture, current behavior, and emotional state of the characters in the scene (excluding the player character), and the personality description should be consistent with the above character information
4. Player character goal description: describe the general goal of the player character in the current scene

Please use vivid and vivid language to create an immersive scene experience."""

        # Build the message list
        messages = [
            {"role": "system",
             "content": "You are an experienced drama screenwriter, good at creating vivid and detailed scene descriptions."},
            {"role": "user", "content": prompt}
        ]

        # Use the general processing function to get the response
        detailed_description = handle_stream_response(self.client, use_model, messages)

        # Update the scene description
        self.scene_descriptions[scene_id] = detailed_description

        return detailed_description

    def add_dialogue_record(self, speaker, record_type, content, target=None):
        """Add a drama record

        Args:
            speaker: The name of the speaker or actor
            record_type: The type of record (e.g., "dialogue", "narrative", "scene description", "environmental interaction", etc.)
            content: The content
            target: The recipient of the dialogue (optional, only meaningful when record_type is "dialogue")
        """
        record = {
            "time": len(self.dialogue_history),
            "speaker": speaker,
            "record_type": record_type,
            "content": content
        }

        # If a target is provided and the record type is related to dialogue, record the dialogue target
        if target and (record_type == "dialogue" or record_type.startswith("dialogue")):
            record["target"] = target

        self.dialogue_history.append(record)

    def get_dialogue_history(self, limit=10):
        """Get the recent dialogue history

        Args:
            limit: The maximum number of dialogues to return

        Returns:
            The list of recent dialogue history
        """
        return self.dialogue_history[-limit:] if self.dialogue_history else []

    def get_all_dialogue_history(self):
        """Get all the dialogue history

        Args:
            limit: The maximum number of dialogues to return

        Returns:
            The list of all dialogue history
        """
        return self.dialogue_history if self.dialogue_history else []

    def generate_new_script(self, current_scene_id, player_feedback=None, max_retries=3, dialogue_history=None):
        """Generate a new script based on the historical dialogue

        Args:
            current_scene_id: The current scene ID
            player_feedback: The feedback provided by the player (optional)
            max_retries: The maximum number of retries
            dialogue_history: The complete dialogue history record (optional)

        Returns:
            The new script part in the script_dict format
        """
        # Automatically generate the next scene ID
        try:
            # Try to extract the numeric part from the current scene ID
            import re
            scene_num_match = re.search(r'(\d+)', current_scene_id)
            if scene_num_match:
                scene_num = int(scene_num_match.group(1))
                next_scene_id = current_scene_id.replace(str(scene_num), str(scene_num + 1))
            else:
                # If there is no number in the current scene ID, add _1
                next_scene_id = f"{current_scene_id}_1"
        except:
            # If the extraction fails, use the timestamp as an alternative
            import time
            next_scene_id = f"scene_{int(time.time())}"

        # Get the current scene information
        current_scene = self.initial_script.get(current_scene_id, {})
        scene_desc = self.scene_descriptions.get(current_scene_id, "")

        # Get the list of characters in the current scene
        current_characters = current_scene.get("characters", [])

        # Get the dialogue history
        dialogue_text = ""
        if dialogue_history:
            # Use the passed-in complete dialogue history
            for d in dialogue_history:
                if isinstance(d, dict):
                    speaker = d.get('speaker', '')
                    record_type = d.get('record_type', '')
                    content = d.get('content', '')
                    if record_type == "dialogue" or record_type.startswith("dialogue"):
                        if 'target' in d:
                            dialogue_text += f"{speaker} says to {d.get('target')}: {content}\n"
                        else:
                            dialogue_text += f"{speaker} ({record_type}): {content}\n"
                    else:
                        dialogue_text += f"{speaker} ({record_type}): {content}\n"

        else:
            # If no dialogue history is passed in, get the recent dialogues
            recent_dialogues = self.get_dialogue_history(limit=20)  # Increase the number of historical records
            for d in recent_dialogues:
                if 'record_type' in d:
                    record_type = d.get('record_type', '')
                    if record_type == "dialogue" or record_type.startswith("dialogue"):
                        if 'target' in d:
                            dialogue_text += f"{d['speaker']} says to {d.get('target')}: {d['content']}\n"
                        else:
                            dialogue_text += f"{d['speaker']} ({record_type}): {d['content']}\n"
                    else:
                        dialogue_text += f"{d['speaker']} ({record_type}): {d['content']}\n"
                else:
                    dialogue_text += f"{d['speaker']}: {d['content']}\n"

        # Generate the script, try up to max_retries times
        for attempt in range(max_retries):
            # Build the prompt
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
1. The new scene must be consistent with the dialogue history and cannot have contradictions.
2. The actions and dialogues of the characters must conform to their personality traits.
3. The scene transition must be natural and the plot development must be reasonable.

You must strictly return in the following JSON format, which is required by the system:
{
    "description": "Detailed description of the scene",
    "characters": ["Character name 1", "Character name 2", ...],
    "dialogues": [
        {"character": "Character name 1", "content": "(Character's expression and action) Dialogue content 1"},
        {"character": "Character name 2", "content": "(Character's expression and action) Dialogue content 2"},
        ...
    ]
}

Note:
1. Only return the content of one scene.
2. The JSON format must be completely correct, without any explanations or extra text.
3. The dialogue content should conform to the character's characteristics and promote the plot development.
4. Do not generate the scene_id field, the system will handle it automatically.
5. Ensure that the new scene is consistent with the previous dialogue and plot."""

            # Build the message list
            messages = [
                {"role": "system",
                 "content": "You are an experienced screenwriter, specializing in creating scripts for interactive dramas. You must return the result in the required JSON format and ensure that the new scene is consistent with the historical dialogue and plot."},
                {"role": "user", "content": prompt}
            ]

            # Use the general processing function to get the response
            generated_text = handle_stream_response(self.client, use_model, messages)

            # Try to parse the JSON
            try:
                import json
                import re

                # Try to extract the JSON part from the text
                json_match = re.search(r'\{[\s\S]*\}', generated_text)
                if json_match:
                    generated_text = json_match.group(0)

                new_scene = json.loads(generated_text)

                # Validate the JSON structure
                if not all(key in new_scene for key in ["description", "dialogues"]):
                    raise ValueError("The generated JSON is missing required fields")

                # Build the complete scene information
                self.initial_script[next_scene_id] = {
                    "description": new_scene["description"],
                    "characters": new_scene["characters"],
                    "dialogues": new_scene["dialogues"]
                }

                # Update the scene description
                self.scene_descriptions[next_scene_id] = new_scene["description"]

                return {next_scene_id: self.initial_script[next_scene_id]}

            except Exception as e:
                if attempt < max_retries - 1:
                    # If it's not the last attempt, continue to the next one
                    continue
                else:
                    # All attempts have failed, return the error information
                    return {"error": str(e), "generated_text": generated_text, "next_scene_id": next_scene_id}

    def generate_actor_response_suggestions(self, actor_name, player_action):
        """Generate suggestions for the actor's response to the player's action

        Args:
            actor_name: The name of the actor
            player_action: The player's action

        Returns:
            A list of suggestions for the actor's possible responses
        """
        # Get the relevant dialogue history
        recent_dialogues = self.get_dialogue_history(limit=5)
        dialogue_text = ""
        for d in recent_dialogues:
            if 'record_type' in d:
                record_type = d.get('record_type', '')
                if record_type == "dialogue" or record_type.startswith("dialogue"):
                    if 'target' in d:
                        dialogue_text += f"{d['speaker']} says to {d.get('target')}: {d['content']}\n"
                    else:
                        dialogue_text += f"{d['speaker']} ({record_type}): {d['content']}\n"
                else:
                    dialogue_text += f"{d['speaker']} ({record_type}): {d['content']}\n"
            else:
                dialogue_text += f"{d['speaker']}: {d['content']}\n"

        # Build the prompt
        prompt = f"""The player has just performed the following action on the character {actor_name}:
        {player_action}

        Recent dialogue history:
        {dialogue_text}

        Based on the character's characteristics and context, generate 3 different styles of response suggestions:
        1. Friendly response
        2. Neutral response
        3. Cold response"""

        # Build the message list
        messages = [
            {"role": "system",
             "content": "You are a screenwriter, good at creating appropriate and natural dialogues for characters."},
            {"role": "user", "content": prompt}
        ]

        # Use the general processing function to get the response
        return handle_stream_response(self.client, use_model, messages)

    def transform_scene(self, scene_id, player_action=None):
        """Generate the scene description after the interaction based on the player's interaction with the items/environment in the scene

        Args:
            scene_id: The current scene ID
            player_action: The specific interaction action of the player (e.g., "push the glass off the table")

        Returns:
            The updated scene description after the interaction
        """
        # Get the original scene description
        original_desc = self.scene_descriptions.get(scene_id, "")
        if not original_desc:
            return "Error: No scene description found"

        # Build the prompt
        prompt = f"""Original scene description:
{original_desc}

The player has performed the following interaction in the scene:
"""

        if player_action:
            prompt += f"""
{player_action}
"""

        prompt += """
Please describe in detail the changes in the scene state after the player's interaction and return it in JSON format:
1. What state is the interacted item in now (if applicable)?
2. What changes has the interaction brought to the scene environment?
3. What impact has it had on other items in the scene?
4. How have the characters in the scene (if any) reacted to this interaction?

Please return the scene description in JSON format, with the following structure:
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
2. The scene description should be vivid and conform to the development of the story plot.
3. Maintain continuity with the original scene and only describe the reasonable changes caused by the player's interaction.
4. The character reactions should conform to their personality traits."""

        # Build the message list
        messages = [
            {"role": "system",
             "content": "You are a screenwriter good at describing scene changes. When the player interacts with the items or environment in the scene, you need to vividly describe the changes in the scene state after the interaction and return the result in JSON format."},
            {"role": "user", "content": prompt}
        ]

        # Use the general processing function to get the response
        json_response = handle_stream_response(self.client, use_model, messages)

        # Parse the JSON response
        import json
        import re

        try:
            # Try to extract the JSON part from the possible text
            json_match = re.search(r'\{[\s\S]*\}', json_response)
            if json_match:
                json_str = json_match.group(0)
                scene_data = json.loads(json_str)

                # Build the human-readable scene description
                formatted_description = ""

                if "scene_description" in scene_data:
                    formatted_description += scene_data["scene_description"] + "\n\n"

                # Add the item interaction information
                if "interactions" in scene_data and scene_data["interactions"]:
                    formatted_description += "Changes in the interacted items:\n"
                    for interaction in scene_data["interactions"]:
                        object_name = interaction.get("object", "")
                        state = interaction.get("state", "")
                        if object_name and state:
                            formatted_description += f"- {object_name}: {state}\n"
                    formatted_description += "\n"

                # Add the character reactions
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
                                # Record the character's reaction in the dialogue history
                                self.add_dialogue_record(character, "action", action)

                            if dialogue:
                                if action:
                                    reaction_text += f', says: "{dialogue}"'
                                else:
                                    reaction_text += f'"{dialogue}"'
                                # Record the character's dialogue in the dialogue history
                                self.add_dialogue_record(character, "dialogue", dialogue)

                            formatted_description += reaction_text + "\n"

                # Update the scene description
                import time
                # Generate a unique scene ID using the timestamp
                new_scene_id = f"{scene_id}_{int(time.time())}"
                self.scene_descriptions[new_scene_id] = formatted_description

                # Record this interaction in the dialogue history
                self.add_dialogue_record("system", "scene", f"Scene change: {player_action}")

                return formatted_description

        except Exception as e:
            # If the JSON parsing fails, use the original response
            print(f"JSON parsing failed: {str(e)}, using the original text")

            # Update the scene description
            import time
            # Generate a unique scene ID using the timestamp
            new_scene_id = f"{scene_id}_{int(time.time())}"
            self.scene_descriptions[new_scene_id] = json_response

            # Record this interaction in the dialogue history
            self.add_dialogue_record("system", "scene", f"Scene change: {player_action}")

            return json_response

    def end_scene(self, last_interaction, director, current_scene_id, next_scene_id=None, dialogue_history=None):
        """Generate the scene end description

        Args:
            last_interaction: The content of the last interaction
            director: The Director object to get the scene description
            current_scene_id: The current scene ID
            next_scene_id: The next scene ID (optional)
            dialogue_history: The list of dialogue history records (optional)

        Returns:
            The text of the scene end description
        """
        # Error check
        if not director or not hasattr(director, 'get_scene_description'):
            return "Error: Invalid director object"

        scene_desc = director.get_scene_description(current_scene_id)
        if not scene_desc:
            return "Error: Unable to get the scene description"

        # Get the list of characters in the current scene
        scene_characters = []
        if hasattr(director, 'get_scene_characters'):
            scene_characters = director.get_scene_characters(current_scene_id)

        # Get the next scene information (if any)
        next_scene_info = ""
        if next_scene_id and hasattr(director, 'get_scene_description'):
            next_scene_desc = director.get_scene_description(next_scene_id)
            if next_scene_desc:
                next_scene_info = "Next scene:\n" + next_scene_desc + "\n\nPlease ensure that your scene end description can naturally transition to the next scene. Consider how the characters move from the current scene to the next scene, how the environment changes, and how the plot develops."

        # Build the character information
        characters_info = ""
        if scene_characters:
            characters_info = "Characters in the current scene: " + ", ".join(scene_characters)

        # Build the dialogue history information
        dialogue_history_info = ""
        if dialogue_history:
            dialogue_history_info = "\n\nDialogue history records:\n"
            for dialogue in dialogue_history:
                if isinstance(dialogue, dict):
                    speaker = dialogue.get('speaker', '')
                    record_type = dialogue.get('record_type', '')
                    content = dialogue.get('content', '')
                    if record_type == "dialogue" or record_type.startswith("dialogue"):
                        if 'target' in dialogue:
                            dialogue_history_info += f"{speaker} says to {dialogue.get('target')}: {content}\n"
                        else:
                            dialogue_history_info += f"{speaker} ({record_type}): {content}\n"
                    else:
                        dialogue_history_info += f"{speaker} ({record_type}): {content}\n"

        # Build the prompt to generate the scene end description
        prompt = "Please generate a scene end description in the drama script format based on the following last interaction content and dialogue history, so that the story can naturally transition to the next scene:\n\n"
        prompt += "Note! The generated content cannot be self - contradictory! It must be consistent with the dialogue history!\n\n"
        prompt += "Last interaction:\n" + last_interaction + "\n\n"
        prompt += "Current scene:\n" + scene_desc + "\n\n"

        if characters_info:
            prompt += characters_info + "\n\n"

        if dialogue_history_info:
            prompt += dialogue_history_info + "\n\n"

        if next_scene_info:
            prompt += next_scene_info + "\n\n"

        prompt += """Please return the scene end description in JSON format, with the following structure:
{
    "scene_description": "Description of the changes in the scene environment and atmosphere",
    "dialogues": [
        {
            "character": "Character name 1",
            "action": "Description of the action (optional)",
            "content": "Dialogue content"
        },
        {
            "character": "Character name 2",
            "action": "Description of the action (optional)",
            "content": "Dialogue content"
        }
    ],
    "transition": "Description to guide the natural transition to the next scene"
}

Please ensure:
1. The JSON format is completely correct and can be parsed.
2. The character names must be the actual characters in the scene or "narrative".
3. All dialogues must be included in the dialogues array.
4. The scene description should be vivid and conform to the development of the story plot.
5. The generated content must be consistent with the dialogue history and cannot have contradictions."""

        # Use the handle_stream_response function to get the scene end description
        messages = [
            {"role": "system",
             "content": 'You are an experienced drama screenwriter, good at creating scene transitions in the drama script format. You need to return a JSON - formatted response, including the scene description, character dialogues, and scene transition. Pay special attention to maintaining consistency with the dialogue history.'},
            {"role": "user", "content": prompt}
        ]
        json_response = handle_stream_response(self.client, use_model, messages)

        # Parse the JSON response
        import json
        import re

        try:
            # Try to extract the JSON part from the possible text
            json_match = re.search(r'\{[\s\S]*\}', json_response)
            if json_match:
                json_str = json_match.group(0)
                scene_data = json.loads(json_str)

                # Extract the dialogues and record them
                if "dialogues" in scene_data and isinstance(scene_data["dialogues"], list):
                    for dialogue in scene_data["dialogues"]:
                        character = dialogue.get("character", "").strip()
                        content = dialogue.get("content", "").strip()
                        action = dialogue.get("action", "").strip()

                        # Check if the character is in the scene or is a narrative
                        if character and (character in scene_characters or character == "narrative"):
                            # Record the dialogue, if there is an action description, add it to the content
                            full_content = content
                            if action:
                                full_content = f"({action}) {content}"
                            self.add_dialogue_record(character, "scene end", full_content)

                # Build the human-readable scene end description
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
            # If the parsing fails, return the original response and try basic parsing
            print(f"JSON parsing failed: {str(e)}, using basic text processing")

            # Basic text processing as a fallback
            lines = json_response.split('\n')
            for line in lines:
                if '：' in line:
                    parts = line.split('：', 1)
                    if len(parts) == 2:
                        character = parts[0].strip()
                        content = parts[1].strip()

                        # Check if the character is in the scene or is a narrative
                        if character in scene_characters or character == "narrative":
                            # Record the dialogue
                            self.add_dialogue_record(character, "scene end", content)

            return json_response

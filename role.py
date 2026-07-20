import openai
import faiss
import numpy as np
from typing import List, Dict, Tuple
import pickle
import os
from openai import OpenAI
from config import ark_api_key, ark_base_url, ark_model, ark_embedding_model, ark_embedding_dim, openai_api_key, deepseek_api_key, qwen_api_key, http_proxy, https_proxy, use_model

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
        if use_model == "qwen3-235b-a22b":
            if use_model.startswith("qwen"):
                extra_body["enable_thinking"] = False
            response_stream = client.chat.completions.create(
                model=use_model,
                messages=messages,
                stream=True,
                extra_body=extra_body
            )
            response_content = ""
            for chunk in response_stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    response_content += chunk.choices[0].delta.content
            return response_content
        else:
            response = client.chat.completions.create(
                model=use_model,
                messages=messages,
                extra_body=extra_body
            )
            return response.choices[0].message.content


def get_ark_client():
    return OpenAI(
        api_key=ark_api_key,
        base_url=ark_base_url
    )


def get_client():
    if use_model == "gpt-4o-mini":
        return OpenAI(
            api_key=openai_api_key,
            base_url="https://api.openai.com/v1"
        )
    elif use_model == "deepseek-chat":
        return OpenAI(
            api_key=deepseek_api_key,
            base_url="https://api.deepseek.com/v1"
        )
    elif use_model == "qwen3-235b-a22b":
        return OpenAI(
            api_key=qwen_api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    elif use_model == "ark":
        return get_ark_client()
    return OpenAI(
        api_key=openai_api_key,
        base_url="https://api.openai.com/v1"
    )


class Actor:
    def __init__(self, name, age, gender, memory_path=None):
        self.name = name
        self.age = age
        self.gender = gender
        self.embedding_dim = ark_embedding_dim if use_model == "ark" else 1536
        self.memories = []
        self.memory_embeddings = None
        self.index = None
        self.relationships = {}
        self.traits = []

        if use_model == "ark":
            self.talk_client = get_ark_client()
            self.embedding_client = get_ark_client()
        else:
            self.embedding_client = OpenAI(
                api_key=openai_api_key,
                base_url="https://api.openai.com/v1"
            )
            self.talk_client = get_client()

        self._initialize_memory(memory_path)

    def _initialize_memory(self, memory_path):
        if memory_path and os.path.exists(memory_path):
            with open(memory_path, 'rb') as f:
                saved_data = pickle.load(f)
                self.memories = saved_data.get('memories', [])
                self.memory_embeddings = saved_data.get('embeddings')
                self.relationships = saved_data.get('relationships', {})
                self.traits = saved_data.get('traits', [])

        if self.memory_embeddings is None or len(self.memories) == 0:
            self.memory_embeddings = np.zeros((0, self.embedding_dim), dtype=np.float32)

        self.index = faiss.IndexFlatL2(self.embedding_dim)
        if len(self.memories) > 0:
            self.index.add(self.memory_embeddings)

    def __str__(self):
        return f"Actor(name={self.name}, age={self.age}, gender={self.gender})"

    def add_memory(self, memory_text: str):
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
        if len(self.memories) == 0:
            return []

        query_embedding = self._get_embedding(query).reshape(1, -1)
        scores, indices = self.index.search(query_embedding, min(k, len(self.memories)))
        return [self.memories[idx] for idx in indices[0]]

    def add_trait(self, trait_description):
        if trait_description not in self.traits:
            self.traits.append(trait_description)
            self.add_memory(f"My character trait is {trait_description}")
        return len(self.traits) - 1

    def get_traits(self):
        return self.traits

    def save_memories(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'memories': self.memories,
                'embeddings': self.memory_embeddings,
                'relationships': self.relationships,
                'traits': self.traits
            }, f)

    def add_relationship(self, other_actor, relationship_type, description=""):
        other_name = other_actor.name if hasattr(other_actor, 'name') else other_actor

        if other_name not in self.relationships:
            self.relationships[other_name] = []

        for i, rel in enumerate(self.relationships[other_name]):
            if rel['type'] == relationship_type:
                self.relationships[other_name][i] = {
                    'type': relationship_type,
                    'description': description
                }
                return

        self.relationships[other_name].append({
            'type': relationship_type,
            'description': description
        })

        memory_text = f"My relationship with {other_name} is {relationship_type}"
        if description:
            memory_text += f": {description}"
        self.add_memory(memory_text)

    def get_relationship(self, other_actor):
        other_name = other_actor.name if hasattr(other_actor, 'name') else other_actor
        return self.relationships.get(other_name, [])

    def get_all_relationships(self):
        return self.relationships

    def speak(self, information, speaker=None, guidance=None):
        relevant_memories = self.retrieve_relevant_memories(information)

        speaker_name = speaker
        if hasattr(speaker, 'name'):
            speaker_name = speaker.name

        relationship_context = ""
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

        traits_context = ""
        if self.traits:
            traits_context = "Your character traits are: " + ", ".join(self.traits) + ". Please shape your response based on these character traits.\n"

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

        if guidance:
            context += "\nDirector's guidance: " + guidance

        if speaker_name:
            self.add_memory(f"{speaker_name} said to me: {information}")

        messages = [
            {"role": "system", "content": f"You are a theater actor, and you are playing the role of {self.name}, a {self.age}-year-old {self.gender} in the play. Please make appropriate responses based on the identity of the conversationalist and your character traits.\n\nYour response must follow the following format: \"({self.name}'s expression and action) Conversation content\". The parentheses must contain the name of the character you are playing as the subject, clearly describe the expression and action, and the content after the parentheses should directly follow the conversation content. For example: \"({self.name} nervously clenches fists) I don't know what you're talking about.\" or \"({self.name} smiles and nods) I'm glad to see you.\""},
            {"role": "user", "content": (context + information) if context else information}
        ]

        response_content = handle_stream_response(self.talk_client, use_model, messages)

        if not response_content.startswith(f"({self.name}"):
            response_content = f"({self.name} calmly says) {response_content}"

        if speaker_name:
            self.add_memory(f"I said to {speaker_name}: {response_content}")

        return response_content

    def should_speak(self, context, speaker=None):
        relevant_memories = self.retrieve_relevant_memories(context)

        speaker_name = speaker
        if hasattr(speaker, 'name'):
            speaker_name = speaker.name

        prompt = f"Scenario: {context}\n\n"

        if speaker_name:
            prompt += f"Speaker: {speaker_name}\n\n"

        if relevant_memories:
            prompt += "Relevant memories:\n" + "\n".join(relevant_memories) + "\n\n"

        prompt += f"As a character named {self.name}, do I need to speak in this situation? Please only answer 'Yes' or 'No', and give a brief reason."

        messages = [
            {"role": "system", "content": "You need to help the character determine if they should speak in the current situation. Please only answer 'Yes' or 'No', and give a brief reason."},
            {"role": "user", "content": prompt}
        ]

        decision = handle_stream_response(self.talk_client, use_model, messages)

        should_speak = "Yes" in decision[:10] or "yes" in decision.lower()[:10]

        if should_speak:
            return self.speak(context, speaker)

        return None


class Director:
    def __init__(self):
        self.actors = {}
        self.script = {}
        self.current_scene = None

        self.client = get_client()

    def add_actor(self, actor):
        self.actors[actor.name] = actor

    def generate_actor_profile(self, character_name, scene_id, player_name):
        scene_desc = self.get_scene_description(scene_id)

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

        messages = [
            {"role": "system", "content": "You are an AI assistant creating characters. Generate character information that fits the scenario based on the scene and character name."},
            {"role": "user", "content": prompt}
        ]

        response = handle_stream_response(self.client, use_model, messages)

        try:
            import json
            import re

            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
        except Exception as e:
            print(f"Failed to parse character information: {str(e)}")

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
        all_characters = self.get_scene_characters(scene_id=scene_id)

        for character_name in all_characters:
            if character_name == player_name:
                continue

            if character_name not in self.actors:
                print(f"\nNew character found: {character_name}, using AI to generate character information...")

                profile = self.generate_actor_profile(character_name, scene_id, player_name)

                new_actor = Actor(character_name, profile.get("age", 30), profile.get("gender", "unknown"))

                for memory in profile.get("background", [f"I am {character_name}, a new character in the current scene"]):
                    new_actor.add_memory(memory)

                for trait in profile.get("traits", ["mysterious"]):
                    new_actor.add_trait(trait)

                relationship = profile.get("relationship", {"type": "stranger", "description": "Just met"})
                new_actor.add_relationship(
                    player_name,
                    relationship.get("type", "stranger"),
                    relationship.get("description", "Just met")
                )

                self.add_actor(new_actor)
                print(f"Character created: {character_name}, Age: {profile.get('age', 30)}, Gender: {profile.get('gender', 'unknown')}")
                print(f"Character traits: {', '.join(profile.get('traits', ['mysterious']))}")
                print(f"Relationship with player: {relationship.get('type', 'stranger')} - {relationship.get('description', 'Just met')}")

    def check_and_create_new_characters(self, scene_ids, current_scene_index, player_name):
        if current_scene_index < len(scene_ids):
            current_scene_id = scene_ids[current_scene_index]
            self.ensure_all_characters_exist(current_scene_id, player_name)

    def load_script(self, script_dict):
        self.script = script_dict

    def set_current_scene(self, scene_id):
        if scene_id in self.script:
            self.current_scene = scene_id
            return True
        return False

    def get_current_scene(self):
        return self.current_scene

    def get_scene_description(self, scene_id=None):
        scene = scene_id if scene_id else self.current_scene
        if scene in self.script:
            return self.script[scene].get("description", "")
        return ""

    def get_scene_characters(self, scene_id=None, player=None):
        scene = scene_id if scene_id else self.current_scene
        if scene not in self.script:
            return []

        characters = self.script[scene].get("characters", [])

        if player:
            player_name = player
            if hasattr(player, 'get_player_name'):
                player_name = player.get_player_name()
            elif hasattr(player, 'name'):
                player_name = player.name

            characters = [char for char in characters if char != player_name]

        return characters

    def guide_actor_from_player_speech(self, player_speech, actor_name):
        if self.current_scene is None or actor_name not in self.actors:
            return "Unable to guide: Current scene not set or actor not found."

        scene_info = self.script.get(self.current_scene, {})
        scene_desc = scene_info.get("description", "")

        dialogues = scene_info.get("dialogues", [])
        actor_dialogues = [d for d in dialogues if d.get("character") == actor_name]

        actor = self.actors.get(actor_name)
        actor_traits = actor.get_traits() if hasattr(actor, "get_traits") else []
        traits_text = ", ".join(actor_traits) if actor_traits else "No specific character traits."

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

        messages = [
            {"role": "system", "content": "You are an experienced theater director, good at analyzing the player's speech and providing response guidance for the actor."},
            {"role": "user", "content": prompt}
        ]

        return handle_stream_response(self.client, use_model, messages)

    def is_scene_continuing(self, last_dialogue, screenwriter=None, detailed_scene=None):
        if self.current_scene is None:
            return False

        scene_info = self.script.get(self.current_scene, {})
        dialogues = scene_info.get("dialogues", [])

        if not dialogues:
            return False

        player_goal = ""
        scene_description = ""

        if detailed_scene:
            scene_description = detailed_scene
        elif screenwriter and hasattr(screenwriter, 'scene_descriptions') and self.current_scene in screenwriter.scene_descriptions:
            scene_description = screenwriter.scene_descriptions.get(self.current_scene, "")

        if scene_description:
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

        prompt = f"Please analyze the following situation and determine if the script should continue in the current scene:\n\n"
        prompt += f"Scene description: {scene_info.get('description', '')}\n\n"

        if player_goal:
            prompt += f"Player character's goal in this scene: {player_goal}\n\n"

        prompt += "Expected dialogue content:\n"
        for d in dialogues[-3:]:
            prompt += f"{d.get('character')}: {d.get('content')}\n"

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

        messages = [
            {"role": "system", "content": "You are an experienced theater director, good at analyzing scripts and performances. You need to determine if the current scene should continue or move on to the next scene."},
            {"role": "user", "content": prompt}
        ]

        result = handle_stream_response(self.client, use_model, messages).lower()

        return "Yes" in result[:30] or "Should continue" in result[:30] or "Continue" in result[:30] or "Not achieved" in result[:50]

    def should_generate_new_script(self, screenwriter, current_scene_id, next_scene_id=None):
        if self.is_scene_continuing(None, screenwriter):
            return False

        recent_dialogues = screenwriter.get_dialogue_history(limit=10)
        dialogue_text = ""
        for d in recent_dialogues:
            if 'record_type' in d:
                record_type = d['record_type']
                if record_type == "Dialogue" or record_type.startswith("Dialogue"):
                    dialogue_text += f"{d['speaker']} said to {record_type}: {d['content']}\n"
                else:
                    dialogue_text += f"{d['speaker']} ({record_type}): {d['content']}\n"

        current_scene = self.script.get(current_scene_id, {})
        current_scene_desc = current_scene.get("description", "")

        next_scene_info = ""
        if next_scene_id and next_scene_id in self.script:
            next_scene = self.script.get(next_scene_id, {})
            next_scene_desc = next_scene.get("description", "")
            next_scene_info = f"""Planned next scene description:
{next_scene_desc}"""

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

        messages = [
            {"role": "system", "content": "You are an experienced theater director, good at analyzing plot development and scene transitions."},
            {"role": "user", "content": prompt}
        ]

        result = handle_stream_response(self.client, use_model, messages)

        return "Yes" in result[:10] or "Need" in result[:20] or "Should" in result[:20]


class Player:
    def __init__(self, name, age, gender):
        self.name = name
        self.age = age
        self.gender = gender

    def talk_to_actor(self, actor, message, guidance=None):
        if not hasattr(actor, 'speak'):
            return f"Error: Unable to talk to this object."

        response = actor.speak(message, self.name, guidance)
        return response

    def interact_with_environment(self, screenwriter, action, current_scene_id=None):
        screenwriter.add_dialogue_record(self.name, "Environmental interaction", action)

        if current_scene_id:
            updated_scene = screenwriter.transform_scene(
                current_scene_id,
                action
            )

            self.current_scene = updated_scene

        return updated_scene

    def get_player_name(self):
        return self.name


class Screenwriter:
    def __init__(self):
        self.client = get_client()

        self.dialogue_history = []
        self.scene_descriptions = {}
        self.initial_script = {}

    def load_initial_script(self, script_dict):
        self.initial_script = script_dict
        for scene_id, scene_data in script_dict.items():
            if "description" in scene_data:
                self.scene_descriptions[scene_id] = scene_data["description"]

    def generate_scene_description(self, scene_id, director=None, player_character=None):
        base_description = self.scene_descriptions.get(scene_id)
        if not base_description:
            if scene_id in self.initial_script and "description" in self.initial_script[scene_id]:
                base_description = self.initial_script[scene_id]["description"]
            else:
                return f"Error: No description found for scene {scene_id}"

        scene_characters = []
        characters_info = ""

        if scene_id in self.initial_script and "characters" in self.initial_script[scene_id]:
            scene_characters = self.initial_script[scene_id]["characters"]

            if director and hasattr(director, 'actors'):
                for char_name in scene_characters:
                    if player_character and char_name == player_character:
                        continue

                    if char_name in director.actors:
                        actor = director.actors[char_name]
                        char_info = f"- {char_name}: {actor.age} years old, {actor.gender}\n"

                        if hasattr(actor, 'get_traits') and actor.get_traits():
                            traits = actor.get_traits()
                            char_info += f"  Personality traits: {', '.join(traits)}\n"

                        if hasattr(actor, 'get_all_relationships'):
                            relationships = actor.get_all_relationships()
                            if relationships:
                                char_info += "  Relationships with other characters:\n"
                                for other_name, rel_list in relationships.items():
                                    if other_name in scene_characters:
                                        for rel in rel_list:
                                            rel_type = rel.get('type', '')
                                            rel_desc = rel.get('description', '')
                                            if rel_desc:
                                                char_info += f"    - With {other_name}: {rel_type} ({rel_desc})\n"
                                            else:
                                                char_info += f"    - With {other_name}: {rel_type}\n"

                        characters_info += char_info

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

        messages = [
            {"role": "system",
             "content": "You are an experienced drama screenwriter, good at creating vivid and detailed scene descriptions."},
            {"role": "user", "content": prompt}
        ]

        detailed_description = handle_stream_response(self.client, use_model, messages)

        self.scene_descriptions[scene_id] = detailed_description

        return detailed_description

    def add_dialogue_record(self, speaker, record_type, content, target=None):
        record = {
            "time": len(self.dialogue_history),
            "speaker": speaker,
            "record_type": record_type,
            "content": content
        }

        if target and (record_type == "dialogue" or record_type.startswith("dialogue")):
            record["target"] = target

        self.dialogue_history.append(record)

    def get_dialogue_history(self, limit=10):
        return self.dialogue_history[-limit:] if self.dialogue_history else []

    def get_all_dialogue_history(self):
        return self.dialogue_history if self.dialogue_history else []

    def generate_new_script(self, current_scene_id, player_feedback=None, max_retries=3, dialogue_history=None):
        try:
            import re
            scene_num_match = re.search(r'(\d+)', current_scene_id)
            if scene_num_match:
                scene_num = int(scene_num_match.group(1))
                next_scene_id = current_scene_id.replace(str(scene_num), str(scene_num + 1))
            else:
                next_scene_id = f"{current_scene_id}_1"
        except:
            import time
            next_scene_id = f"scene_{int(time.time())}"

        current_scene = self.initial_script.get(current_scene_id, {})
        scene_desc = self.scene_descriptions.get(current_scene_id, "")

        current_characters = current_scene.get("characters", [])

        dialogue_text = ""
        if dialogue_history:
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
            recent_dialogues = self.get_dialogue_history(limit=20)
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

        for attempt in range(max_retries):
            prompt = f"""As a screenwriter, please generate the next scene of the script based on the following information:

Current scene: {scene_desc}

Current scene characters: {', '.join(current_characters)}

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

            messages = [
                {"role": "system",
                 "content": "You are an experienced screenwriter, specializing in creating scripts for interactive dramas. You must return the results in the required JSON format and ensure that the new scene is consistent with the historical dialogue and plot."},
                {"role": "user", "content": prompt}
            ]

            generated_text = handle_stream_response(self.client, use_model, messages)

            try:
                import json
                import re

                json_match = re.search(r'\{[\s\S]*\}', generated_text)
                if json_match:
                    generated_text = json_match.group(0)

                new_scene = json.loads(generated_text)

                if not all(key in new_scene for key in ["description", "dialogues"]):
                    raise ValueError("Generated JSON is missing required fields")

                self.initial_script[next_scene_id] = {
                    "description": new_scene["description"],
                    "characters": new_scene["characters"],
                    "dialogues": new_scene["dialogues"]
                }

                self.scene_descriptions[next_scene_id] = new_scene["description"]

                return {next_scene_id: self.initial_script[next_scene_id]}

            except Exception as e:
                if attempt < max_retries - 1:
                    continue
                else:
                    return {"error": str(e), "generated_text": generated_text, "next_scene_id": next_scene_id}

    def generate_actor_response_suggestions(self, actor_name, player_action):
        recent_dialogues = self.get_dialogue_history(limit=5)
        dialogue_text = ""
        for d in recent_dialogues:
            if 'record_type' in d:
                record_type = d['record_type']
                if record_type == "dialogue" or record_type.startswith("dialogue"):
                    if 'target' in d:
                        dialogue_text += f"{d['speaker']} says to {d.get('target')}: {d['content']}\n"
                    else:
                        dialogue_text += f"{d['speaker']} ({record_type}): {d['content']}\n"
                else:
                    dialogue_text += f"{d['speaker']} ({record_type}): {d['content']}\n"
            else:
                dialogue_text += f"{d['speaker']}: {d['content']}\n"

        prompt = f"""Please provide 3-5 possible responses for the actor '{actor_name}' based on the following context:

Player action: {player_action}

Recent dialogue history:
{dialogue_text}

Please provide response suggestions that are consistent with the character's personality and the current situation. Each suggestion should include:
1. The character's expression and action
2. The dialogue content

Format each suggestion as: "(Expression and action) Dialogue content"

Do not include any explanations or extra text."""

        messages = [
            {"role": "system",
             "content": "You are an experienced screenwriter who specializes in creating actor responses for interactive dramas."},
            {"role": "user", "content": prompt}
        ]

        response = handle_stream_response(self.client, use_model, messages)

        suggestions = []
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line and ('(' in line and ')' in line):
                suggestions.append(line)

        return suggestions

    def transform_scene(self, scene_id, player_action):
        current_scene = self.scene_descriptions.get(scene_id, "")

        prompt = f"""Please describe how the scene changes based on the player's action.

Current scene description:
{current_scene}

Player action: {player_action}

Please provide a detailed description of the changes in the scene, including:
1. Changes in the environment
2. Changes in the positions or states of items
3. Changes in the reactions or behaviors of characters
4. Any new elements that appear

Keep the description concise but vivid."""

        messages = [
            {"role": "system",
             "content": "You are an experienced screenwriter who specializes in describing scene transformations."},
            {"role": "user", "content": prompt}
        ]

        response = handle_stream_response(self.client, use_model, messages)

        self.scene_descriptions[scene_id] = response

        return response

    def end_scene(self, last_interaction, director, current_scene_id, next_scene_id):
        current_scene = director.script.get(current_scene_id, {})
        current_scene_desc = current_scene.get("description", "")

        next_scene_desc = ""
        if next_scene_id:
            next_scene = director.script.get(next_scene_id, {})
            next_scene_desc = next_scene.get("description", "")

        prompt = f"""Please provide a scene transition description based on the following information:

Current scene description:
{current_scene_desc}

Last interaction:
{last_interaction}

Next scene description (if any):
{next_scene_desc}

Please provide a natural and smooth transition description that connects the current scene to the next scene (or the end of the story if there is no next scene)."""

        messages = [
            {"role": "system",
             "content": "You are an experienced screenwriter who specializes in creating scene transitions."},
            {"role": "user", "content": prompt}
        ]

        return handle_stream_response(self.client, use_model, messages)
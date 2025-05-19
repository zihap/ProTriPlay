from role import Actor, Player, Screenwriter, Director
from config import max_inserted_scenes, max_inserted_scenes, try_chance

# Initialize characters
# Create the investigator (player character)
player = Player("Howard", 25, "Male")

# Create NPC characters
librarian = Actor("Martha Deer", 57, "Female")
librarian.add_memory("I'm the administrator of the Harbor Town Library and have worked here for 30 years.")
librarian.add_memory("There have been some strange things happening in town recently, especially among the residents near the sea.")
librarian.add_memory("I have a collection of taboo books about ancient myths, including the 'Book of Eibon'.")
librarian.add_relationship(player.name, "Cautious", "I know he's an investigator, but I'm not sure if I can trust him.")
librarian.add_trait("Cautious")
librarian.add_trait("Knowledgeable")
librarian.add_trait("Has a complex curiosity and fear of esoteric knowledge.")

professor = Actor("William Akeley", 68, "Male")
professor.add_memory("I'm a former professor at Miskatonic University, researching ancient civilizations and myths.")
professor.add_memory("I witnessed the Harbor Town incident ten years ago, those indescribable beings.")
professor.add_memory("My sanity has been damaged, but I still try to prevent the upcoming disaster.")
professor.add_relationship(player.name, "Ally", "I think he might be the key to stopping the ritual.")
professor.add_relationship("Martha Deer", "Accomplice", "She helped me preserve some key ancient books.")
professor.add_trait("Mentally unstable")
professor.add_trait("Wise but paranoid")
professor.add_trait("Brave but scarred")

cultist = Actor("Joseph Marsh", 45, "Male")
cultist.add_memory("I'm a descendant of the Deep Ones and am loyal to Dagon and Hydra.")
cultist.add_memory("I'm a fisherman on the surface, but actually I'm responsible for monitoring the outsiders in town.")
cultist.add_memory("I know about the upcoming ritual. My bloodline allows me to summon the beings in the sea.")
cultist.add_relationship(player.name, "Hostile", "I suspect he wants to interfere with our ritual.")
cultist.add_relationship("William Akeley", "Hate", "He knows too much and must be dealt with.")
cultist.add_trait("Fanatical")
cultist.add_trait("Dual personality")
cultist.add_trait("Cruel and heartless")

# Create the director
director = Director()

# Add actors to the director's management
director.add_actor(librarian)
director.add_actor(professor)
director.add_actor(cultist)

# Create the screenwriter
screenwriter = Screenwriter()

# Load the CoC-style script
coc_script = {
    "scene_1": {
        "description": "You're a federal investigator in the United States, coming to Harbor Town to investigate the mysterious disappearances. Now you're in the gloomy and damp Harbor Town Library. It's pouring rain outside, and thunder is rumbling. Inside the library, old bookshelves are neatly arranged under the dim lights. The air is filled with the smell of mold and ancient books. An old-fashioned wall clock in the corner ticks, occasionally making an out-of-tune sound.",
        "characters": ["Howard", "Martha Deer"],
        "dialogues": [
            {"character": "Martha Deer", "content": "(Nervously organizing the bookshelves) It's been an unsettling few days in town, sir. What brings you here?"}
        ]
    },
    "scene_2": {
        "description": "The basement of the library. A small, dim space with old oil lamps hanging on the walls. There's a large wooden table in the middle, with several ancient books and manuscripts spread out on it. The air is even more turbid, and strange patterns are formed by the water stains on the walls. There's a locked iron box in the corner.",
        "characters": ["Howard", "Martha Deer"],
        "dialogues": [
            {"character": "Martha Deer", "content": "(In a hushed voice) These are our non-public collections. Some knowledge... is better left undiscovered."}
        ]
    },
    "scene_3": {
        "description": "Professor Akeley's cottage. An isolated cottage on the outskirts of Harbor Town, surrounded by dense woods. The cottage is filled with books, notes, and strange collectibles. Mysterious symbols and maps are hanging on the walls. The flames in the fireplace cast flickering shadows. A faint smell of seawater and herbs fills the air.",
        "characters": ["Howard", "William Akeley", "Martha Deer"],
        "dialogues": [
            {"character": "William Akeley", "content": "(Hands trembling slightly, eyes darting) Have you found the 'Book of Eibon'? Time is running out. 'They' are about to awaken... (Suddenly lowering his voice) You're being followed. Be careful of those 'fishermen', they're not human..."}
        ]
    },
    "scene_4": {
        "description": "Harbor Town Beach, at night. The moonlight is blocked by thick clouds, and only sporadic starlight illuminates the beach. The waves are crashing against the shore, making a low sound. There seem to be several figures standing on the rocks in the distance, performing some kind of ritual. The air is filled with a strong salty smell and an indescribable odor.",
        "characters": ["Howard", "Joseph Marsh", "William Akeley"],
        "dialogues": [
            {"character": "Joseph Marsh", "content": "(Standing next to the altar, holding a strange statue with both hands) Outsider, you shouldn't be here. This sea belongs to the great beings, and we're about to receive their blessings."}
        ]
    }
}

# Get the list of script IDs
script_ids = list(coc_script.keys())

# Add scene control variables
current_scene_index = 0  # Current scene index
new_scene_generation_count = 0  # Count of new scene generations
max_new_scene_generations = 1  # Maximum number of new scenes allowed to be generated
inserted_scene_count = 0  # Count of scenes generated by "According to the plot development, a new scene needs to be inserted"
# max_inserted_scenes = 2  # Maximum number of new scenes allowed to be inserted

# The director loads the script
director.load_script(coc_script)

# The screenwriter also loads the same script
screenwriter.load_initial_script(coc_script)

# Set the current scene
director.set_current_scene(script_ids[current_scene_index])

print("\n==== Performance starts ====")

# In the while loop, use the methods in the Director class
while current_scene_index < len(script_ids):
    # Get the current scene ID
    current_scene_id = script_ids[current_scene_index]
    # Ensure the current scene is set correctly
    director.set_current_scene(current_scene_id)

    # Check and create new characters in the current scene
    director.ensure_all_characters_exist(current_scene_id, player.name)

    print(f"\nCurrent scene ID: {current_scene_id}")

    detailed_scene = screenwriter.generate_scene_description(current_scene_id, director, player.get_player_name())
    print("\nCurrent scene description:")
    print(detailed_scene)
    # Add the generated scene description to the dialogue_history
    screenwriter.add_dialogue_record("Narrator", "Scene description", detailed_scene)

    # The director gets the scene description
    # !!There's a BUG. The new description modified by the screenwriter is not updated in the director!!
    # print("\nThe director gets the current scene description:")
    scene_desc = director.get_scene_description()
    # print(scene_desc)

    # Check if there are initial dialogues in the current scene. If so, let the NPC characters perform first
    scene_info = director.script.get(current_scene_id, {})
    initial_dialogues = scene_info.get("dialogues", [])

    if initial_dialogues:
        print("\n==== Dialogue starts ====")
        for dialogue in initial_dialogues:
            character_name = dialogue.get("character")
            content = dialogue.get("content")

            # Skip the player character's dialogue
            if character_name == player.get_player_name():
                continue

            # Display the NPC's dialogue
            print(f"\n{character_name}: {content}")

            # Record the dialogue in the screenwriter's dialogue history
            screenwriter.add_dialogue_record(character_name, "Scene dialogue", content)

        # print("\n==== Initial dialogue ends ====")

    # Display the characters that can be interacted with in the current scene
    print("\nCharacters that can be interacted with in the current scene:")
    characters = director.get_scene_characters(player=player)
    print(characters)

    # Scene loop, with x chances
    scene_finished = False
    should_exit_game = False
    last_interaction = ""  # Record the last dialogue or interaction content
    # try_chance = 2

    for i in range(try_chance):
        if scene_finished:
            break

        print("\n==== Please choose your action ====")
        print("\nEnter 1 to talk to a character")
        print("\nEnter 2 to interact with the environment")
        # print("\nEnter next to go to the next scene")
        print("\nEnter next to go to the next scene")
        print("\nEnter esc to exit the drama")
        action = input("Please enter your choice:")
        if action == "1":
            # Talk to a character
            print("\n==== Please choose a character to talk to ====")
            for j, character in enumerate(characters):
                print(f"\n{j + 1}. {character}")
            choice = input("Please choose a character to talk to:")
            if choice.isdigit() and 1 <= int(choice) <= len(characters):
                selected_character = characters[int(choice) - 1]
                print(f"\n==== Please enter your dialogue ====")
                dialogue = input("Please enter your dialogue:")
                # Record the last dialogue
                last_interaction = f"{player.name} says to {selected_character}: {dialogue}"

                # Add the player's dialogue record
                screenwriter.add_dialogue_record(player.name, "Dialogue", dialogue, target=selected_character)
                # Use the merged method to directly generate guidance
                guide_message = director.guide_actor_from_player_speech(dialogue, selected_character)
                # Get the actor instance, not just the character name
                actor_instance = director.actors.get(selected_character)
                if actor_instance:
                    # The actor's dialogue, using the Actor instance
                    response = player.talk_to_actor(actor_instance, dialogue, guide_message)
                    # Update the last interaction record
                    last_interaction += f"\n{selected_character} replies: {response}"

                    # Add the NPC's dialogue record
                    screenwriter.add_dialogue_record(selected_character, "Dialogue", response, target=player.name)

                    print(f"\n{selected_character}: {response}")

                    # Check if the current scene should continue
                    if not director.is_scene_continuing(response):
                        print("The current scene ends")
                        scene_finished = True
                        # Get the next scene ID (if available)
                        next_scene = script_ids[current_scene_index + 1] if current_scene_index + 1 < len(script_ids) else None
                        # Generate the scene ending description
                        ending_description = screenwriter.end_scene(last_interaction, director, current_scene_id, next_scene)
                        print(f"\n{ending_description}")
                        # Add the scene transition description
                        screenwriter.add_dialogue_record("Narrator", "Scene transition", f"Transition from {current_scene_id} scene to {next_scene if next_scene else 'The end of the story'}")
                    else:
                        print("The current scene continues")
                else:
                    print(f"Error: Could not find an instance of the character {selected_character}")
            else:
                print("Invalid choice. Please try again.")

        elif action == "2":
            # Interact with the environment
            print("\n==== Please enter your interaction ====")
            interaction = input("Please enter your interaction:")
            # Record the last interaction
            last_interaction = f"{player.name} interacts with the environment: {interaction}"

            # Add the player's interaction record
            screenwriter.add_dialogue_record(player.name, "Environmental interaction", interaction)
            # The screenwriter processes the player's action
            action_response = screenwriter.transform_scene(current_scene_id, interaction)
            # Update the last interaction record
            last_interaction += f"\nThe environment responds: {action_response}"

            print(f"\n{action_response}")

            # Check if the current scene should continue
            if not director.is_scene_continuing(action_response):
                print("The current scene ends")
                scene_finished = True
                # Get the next scene ID (if available)
                next_scene = script_ids[current_scene_index + 1] if current_scene_index + 1 < len(script_ids) else None
                # Generate the scene ending description
                ending_description = screenwriter.end_scene(last_interaction, director, current_scene_id, next_scene)
                print(f"\n{ending_description}")
                # Add the scene transition description
                screenwriter.add_dialogue_record("Narrator", "Scene transition", f"Transition from {current_scene_id} scene to {next_scene if next_scene else 'The end of the story'}")
            else:
                print("The current scene continues")

        elif action.lower() == "next":
            # Manually go to the next scene
            print("Manually end the current scene and go to the next scene")
            scene_finished = True

            # Get the next scene ID (if available)
            next_scene = script_ids[current_scene_index + 1] if current_scene_index + 1 < len(script_ids) else None

            if next_scene:
                # Create a simple transition description
                ending_description = screenwriter.end_scene(last_interaction or "The player chooses to skip the current scene", director, current_scene_id, next_scene)
                print(f"\n{ending_description}")
                # Add the scene transition description
                screenwriter.add_dialogue_record("Narrator", "Scene transition", f"Transition from {current_scene_id} scene to {next_scene}")
            else:
                print("\nNo more scenes. The story ends")
                should_exit_game = True

        elif action.lower() == "esc":
            # Manually exit the entire drama
            # print("Exit the drama")
            should_exit_game = True
            break

        else:
            print("Invalid choice. Please try again.")

    # If the 10 chances are used up but the scene doesn't end normally, generate a forced ending description
    if not scene_finished and not should_exit_game:
        print("\n==== The scene dialogue chances are used up. Generate a forced scene ending ====")

        # Get the next scene ID (if available)
        next_scene = script_ids[current_scene_index + 1] if current_scene_index + 1 < len(script_ids) else None

        # Check if a new scene needs to be inserted
        should_generate = director.should_generate_new_script(screenwriter, current_scene_id, next_scene)

        if should_generate and inserted_scene_count < max_inserted_scenes:
            print(f"\n==== According to the plot development, a new scene needs to be inserted ({inserted_scene_count + 1}/{max_inserted_scenes}) ====")

            # Directly generate a new scene without player feedback
            new_script = screenwriter.generate_new_script(current_scene_id, dialogue_history=screenwriter.get_dialogue_history())

            if "error" not in new_script:
                # Update the director's script
                director.load_script(screenwriter.initial_script)

                # Re-get and sort the script ID list
                script_ids = list(screenwriter.initial_script.keys())
                new_scene_id = list(new_script.keys())[0]
                print(f"\nSuccessfully generated a new scene: {new_scene_id}")

                # Set the new scene as the next scene
                next_scene = new_scene_id

                # Increase the inserted scene count
                inserted_scene_count += 1

                # Check and create new characters
                director.check_and_create_new_characters(script_ids, script_ids.index(new_scene_id), player.name)
            else:
                print(f"\nFailed to generate a new scene: {new_script.get('error')}")
        elif should_generate and inserted_scene_count >= max_inserted_scenes:
            print(f"\n==== The limit of inserted new scenes ({max_inserted_scenes} times) has been reached. Continue using the original script ====")

        # Generate the scene ending description, using the updated next_scene
        ending_description = screenwriter.end_scene(last_interaction, director, current_scene_id, next_scene)
        print(f"\n{ending_description}")

        # Add the scene transition description
        screenwriter.add_dialogue_record("Narrator", "Scene transition", f"Transition from {current_scene_id} scene to {next_scene if next_scene else 'The end of the story'}")

        # Set the scene as finished
        scene_finished = True

    # Check if the drama should be exited
    if should_exit_game:
        print("Exit the drama")
        break

    # The current scene ends. Update the script ID list (there may be new scenes generated)
    script_ids = list(screenwriter.initial_script.keys())

    # Check if a brand new scene needs to be generated (all scenes are completed)
    # There's a limit to the number of brand new scenes generated, for example, 3 times. After generating 3 times, generate a forced ending description for the script
    if current_scene_index + 1 >= len(script_ids):
        # Check if the scene generation limit has been reached
        if new_scene_generation_count >= max_new_scene_generations:
            print(f"\n==== The scene generation limit ({max_new_scene_generations} times) has been reached. Preparing to end the story ====")
            # Generate the ending scene without player input
            ending_prompt = "This is the ending scene of the story. Please provide a satisfying, logical, and emotionally impactful ending based on the previous plot. Ensure that all major plot threads are appropriately resolved."

            # Use a special marker to tell the screenwriter this is the ending
            new_script = screenwriter.generate_new_script(current_scene_id, ending_prompt, dialogue_history=screenwriter.get_dialogue_history())

            if "error" not in new_script:
                # Update the script ID list
                script_ids = list(screenwriter.initial_script.keys())
                ending_scene_id = list(new_script.keys())[0]
                print(f"\nSuccessfully generated the ending scene: {ending_scene_id}")

                # Update the director's script
                director.load_script(screenwriter.initial_script)

                # Ensure the next scene is the ending scene
                current_scene_index = script_ids.index(current_scene_id)
                current_scene_index += 1
                # Check and create new characters
                director.check_and_create_new_characters(script_ids, current_scene_index, player.name)
            else:
                print(f"\nFailed to generate the ending scene: {new_script.get('error')}")
                break

            # Set the flag to indicate this is the last scene
            is_final_scene = True
        else:
            print(f"\n==== All planned scenes are completed. Trying to generate a new scene ({new_scene_generation_count + 1}/{max_new_scene_generations}) ====")
            # Generate a new scene without player input
            new_script = screenwriter.generate_new_script(current_scene_id, dialogue_history=screenwriter.get_dialogue_history())

            if "error" not in new_script:
                # Update the script ID list
                script_ids = list(screenwriter.initial_script.keys())
                new_scene_id = list(new_script.keys())[0]
                print(f"\nSuccessfully generated a new scene: {new_scene_id}")

                # Update the director's script
                director.load_script(screenwriter.initial_script)

                # Increase the scene generation count
                new_scene_generation_count += 1
                # Check and create new characters
                director.check_and_create_new_characters(script_ids, current_scene_index + 1, player.name)
            else:
                print(f"\nFailed to generate a new scene: {new_script.get('error')}")
                break

    # Move to the next scene
    current_scene_index += 1

    # Check if it's the last scene or there are no more scenes
    if current_scene_index < len(script_ids):
        print(f"\n==== Enter the next scene: {script_ids[current_scene_index]} ====")
    else:
        print("\n==== The story ends ====")
        break  # Ensure to exit the loop after the story ends

print("\n==== Performance ends ====")

# Export the dialogue history to a JSON file for evaluation
import json
import datetime

# Create a file name with a timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
dialogue_history_file = f"dialogue_history_{timestamp}.json"

# Convert the dialogue history to a serializable format
dialogue_history = screenwriter.get_all_dialogue_history()  # Get the entire dialogue history

# Save the dialogue history to a file
with open(dialogue_history_file, "w", encoding="utf-8") as f:
    json.dump(dialogue_history, f, ensure_ascii=False, indent=2)

print(f"\nThe dialogue history has been exported to the file: {dialogue_history_file}")
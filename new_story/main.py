import json
import os
import datetime
from agents import DirectorAgent, ScreenwriterAgent, RoleAgent, ProtagonistAgent
from models import Character, CharacterProfile, Scene, SceneTransition, StoryOutline, StoryNode


def create_story():
    print("==== 多Agent合作叙事系统 ====")
    print("正在初始化故事生成系统...")
    
    director = DirectorAgent()
    screenwriter = ScreenwriterAgent()
    
    print("\n==== 加载剧情大纲 ====")
    outline = StoryOutline(
        title="星辰秘钥",
        background="在遥远的星辰大陆，传说中有一把能够打开时空之门的秘钥。年轻的探险家林远踏上了寻找秘钥的旅程，在旅途中结识了不同的伙伴，揭开了一个古老的秘密..."
    )
    
    outline.add_node(StoryNode("node_1", "旅程开端：林远从故乡出发，踏上寻找秘钥的冒险"))
    outline.add_node(StoryNode("node_2", "结识伙伴：在旅途中遇到神秘的向导和博学的学者"))
    outline.add_node(StoryNode("node_3", "发现线索：找到关于秘钥下落的重要线索"))
    outline.add_node(StoryNode("node_4", "最终挑战：抵达秘钥所在之地，面对最后的考验"))
    
    director.load_story_outline(outline)
    
    print("\n==== 定义初始剧本 ====")
    script = {
        "scene_1": {
            "description": "晨光镇酒馆。温暖的木质酒馆，壁炉中燃烧着火焰。墙上挂着古老的地图和探险者的装备。角落里坐着一位神秘的老者。",
            "characters": ["林远", "神秘老者"],
            "dialogs": []
        },
        "scene_2": {
            "description": "迷雾森林入口。古老的森林笼罩在薄雾之中，参天大树遮天蔽日。一条蜿蜒的小径通向森林深处。",
            "characters": ["林远", "森林向导"],
            "dialogs": []
        },
        "scene_3": {
            "description": "星辰遗迹。古老的石质遗迹，墙上刻满神秘的符文。中央有一座祭坛，祭坛上放着发光的水晶球。",
            "characters": ["林远", "学者"],
            "dialogs": []
        },
        "scene_4": {
            "description": "时空之门所在的洞穴。巨大的洞穴深处，一座闪耀着七彩光芒的传送门悬浮在空中。周围散落着古老的魔法物品。",
            "characters": ["林远", "守护者"],
            "dialogs": []
        }
    }
    
    screenwriter.load_initial_script(script)
    
    print("\n==== 创建角色 ====")
    linyuan_profile = CharacterProfile(
        name="林远",
        age=22,
        gender="男",
        traits=["勇敢坚毅", "好奇心强", "乐于助人", "充满智慧"],
        background=["从小听着冒险故事长大", "立志成为伟大的探险家", "曾独自穿越过危险的山脉"],
        goals=["找到星辰秘钥", "揭开古老的秘密", "成为传奇探险家"]
    )
    linyuan_character = Character(linyuan_profile)
    protagonist = ProtagonistAgent(linyuan_character)
    protagonist.add_special_ability("感知", "能够感知周围的魔法能量")
    protagonist.set_narrative_perspective("third_person_limited")
    
    laozhe_profile = CharacterProfile(
        name="神秘老者",
        age=80,
        gender="男",
        traits=["神秘莫测", "智慧渊博", "言语简练"],
        background=["据说年轻时也是一位伟大的探险家", "在酒馆隐居多年", "知道许多不为人知的传说"],
        goals=["引导年轻的探险家", "守护古老的秘密"]
    )
    laozhe_character = Character(laozhe_profile)
    laozhe = RoleAgent(laozhe_character)
    
    xiangdao_profile = CharacterProfile(
        name="森林向导",
        age=35,
        gender="女",
        traits=["身手敏捷", "熟悉自然", "性格开朗"],
        background=["从小在森林中长大", "能够与动物沟通", "从未迷失在森林中"],
        goals=["帮助林远穿越森林", "保护森林的秘密"]
    )
    xiangdao_character = Character(xiangdao_profile)
    xiangdao = RoleAgent(xiangdao_character)
    
    xuezhe_profile = CharacterProfile(
        name="学者",
        age=45,
        gender="男",
        traits=["学识渊博", "严谨认真", "有些固执"],
        background=["研究古代文明多年", "发表过多篇学术论文", "一直在追寻星辰秘钥的线索"],
        goals=["解读遗迹中的符文", "找到秘钥的下落"]
    )
    xuezhe_character = Character(xuezhe_profile)
    xuezhe = RoleAgent(xuezhe_character)
    
    shouhuzhe_profile = CharacterProfile(
        name="守护者",
        age=50,
        gender="男",
        traits=["忠诚坚定", "力量强大", "责任心强"],
        background=["世代守护时空之门", "掌握古老的魔法", "等待着真正的继承者"],
        goals=["保护时空之门", "找到合格的继承者"]
    )
    shouhuzhe_character = Character(shouhuzhe_profile)
    shouhuzhe = RoleAgent(shouhuzhe_character)
    
    director.add_character(linyuan_character)
    director.add_character(laozhe_character)
    director.add_character(xiangdao_character)
    director.add_character(xuezhe_character)
    director.add_character(shouhuzhe_character)
    
    scene1 = Scene("scene_1", script["scene_1"]["description"], ["林远", "神秘老者"])
    scene2 = Scene("scene_2", script["scene_2"]["description"], ["林远", "森林向导"])
    scene3 = Scene("scene_3", script["scene_3"]["description"], ["林远", "学者"])
    scene4 = Scene("scene_4", script["scene_4"]["description"], ["林远", "守护者"])
    
    director.add_scene(scene1)
    director.add_scene(scene2)
    director.add_scene(scene3)
    director.add_scene(scene4)
    
    director.add_transition(SceneTransition("scene_1", "scene_2"))
    director.add_transition(SceneTransition("scene_2", "scene_3"))
    director.add_transition(SceneTransition("scene_3", "scene_4"))
    
    role_map = {
        "神秘老者": laozhe,
        "森林向导": xiangdao,
        "学者": xuezhe,
        "守护者": shouhuzhe
    }
    
    story_output = []
    scene_ids = ["scene_1", "scene_2", "scene_3", "scene_4"]
    
    print("\n==== 开始生成故事 ====")
    
    for scene_index, scene_id in enumerate(scene_ids):
        print(f"\n==== 场景 {scene_index + 1}: {scene_id} ====")
        
        director.set_current_scene(scene_id)
        
        detailed_setting = screenwriter.generate_scene_setting(scene_id, director, "林远")
        print(f"\n场景描述:\n{detailed_setting}")
        
        protagonist.observe_scene(detailed_setting, director.schedule_characters(scene_id))
        
        scene_data = {
            "scene_id": scene_id,
            "scene_number": scene_index + 1,
            "description": detailed_setting,
            "characters": director.schedule_characters(scene_id),
            "dialogues": [],
            "narrative": protagonist.generate_narration(detailed_setting)
        }
        
        screenwriter.add_dialogue_record("旁白", "场景描述", detailed_setting)
        
        scene_characters = [c for c in director.schedule_characters(scene_id) if c != "林远"]
        
        for round_num in range(2):
            for char_name in scene_characters:
                role_agent = role_map.get(char_name)
                if not role_agent:
                    continue
                
                hero_line = protagonist.drive_story(detailed_setting)
                print(f"\n林远: {hero_line}")
                
                scene_data["dialogues"].append({
                    "round": round_num + 1,
                    "speaker": "林远",
                    "content": hero_line,
                    "action": ""
                })
                screenwriter.add_dialogue_record("林远", "对话", hero_line, target=char_name)
                
                guidance = director.provide_guidance("演员", f"角色{char_name}需要回应林远的对话")
                response = role_agent.generate_dialogue(hero_line, "林远", guidance)
                print(f"{char_name}: {response}")
                
                scene_data["dialogues"].append({
                    "round": round_num + 1,
                    "speaker": char_name,
                    "content": response,
                    "action": ""
                })
                screenwriter.add_dialogue_record(char_name, "对话", response, target="林远")
                
                protagonist.update_growth(f"与{char_name}对话，获得新线索", 0.08)
                
                if char_name in protagonist.character.relationships:
                    protagonist.update_relationship(char_name, 0.1)
        
        story_output.append(scene_data)
        
        if scene_index < len(scene_ids) - 1:
            next_scene_id = scene_ids[scene_index + 1]
            next_scene = director.scenes.get(next_scene_id)
            transition_desc = screenwriter.generate_transition(
                director.scenes[scene_id],
                next_scene,
                f"林远与{', '.join(scene_characters)}的对话结束"
            )
            print(f"\n场景转场:\n{transition_desc}")
            
            scene_data["transition"] = transition_desc
            screenwriter.add_dialogue_record("旁白", "场景转场", transition_desc)
    
    print(f"\n==== 主角成长总结 ====")
    growth_summary = protagonist.get_growth_summary()
    print(f"成长点数: {growth_summary['total_points']}")
    print(f"当前阶段: {growth_summary['current_stage']}")
    print(f"特殊能力: {growth_summary['special_abilities']}")
    
    story_output.append({
        "type": "growth_summary",
        "data": growth_summary
    })
    
    print("\n==== 导出故事 ====")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join("story_output", f"story_{timestamp}.json")
    
    os.makedirs("story_output", exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(story_output, f, ensure_ascii=False, indent=2)
    
    print(f"故事已导出到: {output_path}")
    
    print("\n==== 故事生成完成 ====")
    
    return output_path


if __name__ == "__main__":
    create_story()

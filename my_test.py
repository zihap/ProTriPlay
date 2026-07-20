"""
CoC风格互动叙事游戏测试脚本
基于火山方舟Agent Plan API实现的文本冒险游戏测试
核心功能：角色对话、场景管理、动态剧本生成

文件结构：
1. 角色初始化：玩家角色和NPC角色创建
2. 剧本定义：4个预定义场景的CoC风格剧本
3. 游戏流程：场景循环、玩家交互、场景切换
4. 动态生成：根据情节发展插入新场景或生成结局
5. 结果导出：对话历史保存到JSON文件
"""

from role import Actor, Player, Screenwriter, Director
from config import max_inserted_scenes, try_chance


# ==================== 角色初始化 ====================

# 创建玩家角色（调查员）
player = Player("霍华德", 25, "男")

# 创建NPC角色：图书管理员玛莎·迪尔
# 角色设定：海港镇图书馆管理员，掌握禁忌知识，对调查员持谨慎态度
librarian = Actor("玛莎·迪尔", 57, "女")
librarian.add_memory("我是海港镇图书馆的管理员，已经工作了30年")
librarian.add_memory("最近镇上发生了一些奇怪的事情，特别是靠近海边的居民")
librarian.add_memory("我收藏了一些关于古老神话的禁忌书籍，包括《伊波恩之书》")
librarian.add_relationship(player.name, "谨慎", "知道他是个调查员，但不确定能否信任他")
librarian.add_trait("谨慎小心")
librarian.add_trait("博学多识")
librarian.add_trait("对神秘学知识有着复杂的好奇和恐惧")

# 创建NPC角色：前教授威廉·阿克雷
# 角色设定：密斯卡托尼克大学前教授，见证过超自然事件，精神受损但仍努力阻止灾难
professor = Actor("威廉·阿克雷", 68, "男")
professor.add_memory("我是密斯卡托尼克大学的前教授，研究古代文明和神话")
professor.add_memory("我见证了十年前的海港镇事件，那些不可名状的存在")
professor.add_memory("我的理智已经受到了损害，但我仍然试图阻止即将到来的灾难")
professor.add_relationship(player.name, "盟友", "认为他可能是阻止仪式的关键人物")
professor.add_relationship("玛莎·迪尔", "同谋", "她帮我保存了一些关键的古籍")
professor.add_trait("精神不稳定")
professor.add_trait("睿智但偏执")
professor.add_trait("勇敢但已伤痕累累")

# 创建NPC角色：邪教徒约瑟夫·马什
# 角色设定：深潜者后裔，表面渔民，实为邪教成员，敌视调查员
cultist = Actor("约瑟夫·马什", 45, "男")
cultist.add_memory("我是深潜者的后裔，忠于达贡和海德拉")
cultist.add_memory("我表面上是普通渔民，实际负责监视镇上的外来者")
cultist.add_memory("我知道即将到来的仪式，我的血脉让我可以呼唤海中的存在")
cultist.add_relationship(player.name, "敌对", "怀疑他想干扰我们的仪式")
cultist.add_relationship("威廉·阿克雷", "仇恨", "他知道太多了，必须被处理掉")
cultist.add_trait("狂热")
cultist.add_trait("双面性格")
cultist.add_trait("残忍无情")


# ==================== 游戏系统初始化 ====================

# 创建导演实例，负责管理场景切换和角色调度
director = Director()

# 将所有NPC角色添加到导演管理中
director.add_actor(librarian)
director.add_actor(professor)
director.add_actor(cultist)

# 创建编剧实例，负责剧本生成和对话历史管理
screenwriter = Screenwriter()


# ==================== CoC风格剧本定义 ====================
# 剧本包含4个场景，每个场景包含描述、出场角色和初始对话
coc_script = {
    "scene_1": {
        "description": "你是一位美国联邦调查员，来海港镇调查人员神秘失踪事件。现在你在阴郁潮湿的海港镇图书馆。窗外大雨倾盆，雷声轰鸣。图书馆内昏暗的灯光下，古老的书架排列整齐，空气中弥漫着霉味和古籍的气息。角落里的老式座钟滴答作响，偶尔发出不协调的声音。",
        "characters": ["霍华德", "玛莎·迪尔"],
        "dialogues": [
            {"character": "玛莎·迪尔", "content": "(神情紧张地整理着书架)这几天镇上不太平，先生。你来这里是为了什么？"},
        ]
    },
    "scene_2": {
        "description": "图书馆的地下室。一个狭小、昏暗的空间，墙壁上挂着老式油灯。中间是一张大木桌，上面摊开着几本古老的书籍和手稿。空气更加浑浊，墙上的水渍形成了奇怪的图案。角落里有一个锁着的铁箱。",
        "characters": ["霍华德", "玛莎·迪尔"],
        "dialogues": [
            {"character": "玛莎·迪尔", "content": "(声音压低)这些是我们不对外开放的藏书。有些知识...最好永远不要被发现。"},
        ]
    },
    "scene_3": {
        "description": "阿克雷教授的小屋。位于海港镇郊外的一座孤立小屋，周围是茂密的树林。屋内满是书籍、笔记和奇怪的收藏品。墙上挂着神秘的符号和地图。壁炉里的火焰投下摇曳的影子。一股淡淡的海水和药草混合的气味弥漫在空气中。",
        "characters": ["霍华德", "威廉·阿克雷", "玛莎·迪尔"],
        "dialogues": [
            {"character": "威廉·阿克雷", "content": "(手微微颤抖，眼神飘忽)你找到《伊波恩之书》了吗？时间不多了，'它们'即将苏醒...(突然压低声音)你被跟踪了，小心那些'渔民'，他们不是人类..."},
        ]
    },
    "scene_4": {
        "description": "海港镇海滩，夜晚。月光被厚重的云层遮挡，只有零星的星光照亮沙滩。海浪拍打着岸边，发出低沉的声响。远处礁石上似乎站着几个人影，正在进行某种仪式。空气中弥漫着浓重的咸味和一种无法描述的异味。",
        "characters": ["霍华德", "约瑟夫·马什", "威廉·阿克雷"],
        "dialogues": [
            {"character": "约瑟夫·马什", "content": "(站在祭坛旁边，双手举起一个奇怪的雕像)外来者，你不该来这里。这片海域属于伟大的存在，而我们即将得到祂们的祝福。"},
        ]
    }
}


# ==================== 游戏状态变量初始化 ====================

script_ids = list(coc_script.keys())          # 场景ID列表
current_scene_index = 0                       # 当前场景索引
new_scene_generation_count = 0                # 新场景生成次数（用于限制动态生成）
max_new_scene_generations = 1                 # 最大允许生成的新场景数
inserted_scene_count = 0                      # 插入场景次数（用于限制场景插入）


# ==================== 剧本加载 ====================

# 导演加载剧本，用于场景管理和角色调度
director.load_script(coc_script)

# 编剧加载剧本，用于场景描述生成和对话历史记录
screenwriter.load_initial_script(coc_script)

# 设置初始场景
director.set_current_scene(script_ids[current_scene_index])


# ==================== 游戏主循环 ====================

print("\n==== 表演开始 ====")

# 遍历所有场景，执行游戏流程
while current_scene_index < len(script_ids):
    # 获取当前场景ID并更新导演状态
    current_scene_id = script_ids[current_scene_index]
    director.set_current_scene(current_scene_id)
    
    # 检查并创建当前场景中尚未存在的角色
    director.ensure_all_characters_exist(current_scene_id, player.name)

    print(f"\n当前场景编号: {current_scene_id}")

    # 生成详细场景描述（由AI根据当前状态生成）
    detailed_scene = screenwriter.generate_scene_description(current_scene_id, director, player.get_player_name())
    print("\n当前场景描述:")
    print(detailed_scene)
    # 将场景描述记录到对话历史中
    screenwriter.add_dialogue_record("旁白", "场景描述", detailed_scene)

    # 获取导演端的场景描述（注意：此处存在同步问题，编剧修改的描述未更新到导演）
    scene_desc = director.get_scene_description()

    # 播放场景初始对话（NPC预设台词）
    scene_info = director.script.get(current_scene_id, {})
    initial_dialogues = scene_info.get("dialogues", [])
    
    if initial_dialogues:
        print("\n==== 对话开始 ====")
        for dialogue in initial_dialogues:
            character_name = dialogue.get("character")
            content = dialogue.get("content")
            
            # 跳过玩家角色的预设对话（玩家对话由用户输入）
            if character_name == player.get_player_name():
                continue
                
            # 显示NPC初始对话
            print(f"\n{character_name}: {content}")
            
            # 记录到对话历史
            screenwriter.add_dialogue_record(character_name, "场景对话", content)

    # 显示当前场景可交互的角色列表
    print("\n当前场景可对话角色:")
    characters = director.get_scene_characters(player=player)
    print(characters)


    # ==================== 场景内交互循环 ====================
    # 每个场景玩家有try_chance次交互机会
    scene_finished = False       # 场景是否已结束
    should_exit_game = False     # 是否需要退出游戏
    last_interaction = ""        # 记录最后一次交互内容（用于场景结束描述）
    
    for i in range(try_chance):
        if scene_finished:
            break
            
        # 显示操作菜单
        print("\n==== 请选择你的行动 ====")
        print("1. 与人物对话")
        print("2. 与环境互动")
        print("next. 进入下一个场景")
        print("esc. 退出戏剧")
        action = input("请输入你的选择: ")

        # 处理对话操作
        if action == "1":
            print("\n==== 请选择对话对象 ====")
            for j, character in enumerate(characters):
                print(f"{j+1}. {character}")
            choice = input("请选择对话对象: ")
            
            # 验证玩家输入是否有效
            if choice.isdigit() and 1 <= int(choice) <= len(characters):
                selected_character = characters[int(choice) - 1]
                print(f"\n==== 请输入对话内容 ====")
                dialogue = input("请输入对话内容: ")
                
                # 记录交互内容
                last_interaction = f"{player.name}对{selected_character}说：{dialogue}"
                
                # 添加玩家对话到历史记录
                screenwriter.add_dialogue_record(player.name, "对话", dialogue, target=selected_character)
                
                # 生成NPC对话指导（基于玩家对话和角色设定）
                guide_message = director.guide_actor_from_player_speech(dialogue, selected_character)
                
                # 获取目标角色实例并执行对话
                actor_instance = director.actors.get(selected_character)
                if actor_instance:
                    # 调用AI生成NPC回应
                    response = player.talk_to_actor(actor_instance, dialogue, guide_message)
                    last_interaction += f"\n{selected_character}回应：{response}"
                    
                    # 记录NPC回应到历史
                    screenwriter.add_dialogue_record(selected_character, "对话", response, target=player.name)
                    
                    print(f"\n{selected_character}: {response}")

                    # 判断场景是否应该结束（基于NPC回应内容）
                    if not director.is_scene_continuing(response):
                        print("当前场景结束")
                        scene_finished = True
                        # 获取下一个场景ID
                        next_scene = script_ids[current_scene_index + 1] if current_scene_index + 1 < len(script_ids) else None
                        # 生成场景结束描述
                        ending_description = screenwriter.end_scene(last_interaction, director, current_scene_id, next_scene)
                        print(f"\n{ending_description}")
                        # 记录场景转场
                        screenwriter.add_dialogue_record("旁白", "场景转场", f"从{current_scene_id}场景转场到{next_scene if next_scene else '故事结束'}")
                    else:
                        print("当前场景继续")
                else:
                    print(f"错误：找不到角色 {selected_character} 的实例")
            else:
                print("无效的选择，请重新输入。")
        
        # 处理环境互动操作
        elif action == "2":
            print("\n==== 请输入互动内容 ====")
            interaction = input("请输入互动内容: ")
            last_interaction = f"{player.name}与环境互动：{interaction}"
            
            # 记录玩家互动
            screenwriter.add_dialogue_record(player.name, "环境互动", interaction)
            
            # 生成环境响应（由AI根据场景和互动内容生成）
            action_response = screenwriter.transform_scene(current_scene_id, interaction)
            last_interaction += f"\n环境响应：{action_response}"

            print(f"\n{action_response}")

            # 判断场景是否应该结束
            if not director.is_scene_continuing(action_response):
                print("当前场景结束")
                scene_finished = True
                next_scene = script_ids[current_scene_index + 1] if current_scene_index + 1 < len(script_ids) else None
                ending_description = screenwriter.end_scene(last_interaction, director, current_scene_id, next_scene)
                print(f"\n{ending_description}")
                screenwriter.add_dialogue_record("旁白", "场景转场", f"从{current_scene_id}场景转场到{next_scene if next_scene else '故事结束'}")
            else:
                print("当前场景继续")
        
        # 处理手动切换场景
        elif action.lower() == "next":
            print("手动结束当前场景，进入下一个场景")
            scene_finished = True
            
            next_scene = script_ids[current_scene_index + 1] if current_scene_index + 1 < len(script_ids) else None
            
            if next_scene:
                ending_description = screenwriter.end_scene(last_interaction or "玩家选择跳过当前场景", director, current_scene_id, next_scene)
                print(f"\n{ending_description}")
                screenwriter.add_dialogue_record("旁白", "场景转场", f"从{current_scene_id}场景转场到{next_scene}")
            else:
                print("\n没有更多场景，故事结束")
                should_exit_game = True
        
        # 处理退出游戏
        elif action.lower() == "esc":
            should_exit_game = True
            break
        
        # 无效输入处理
        else:
            print("无效的选择，请重新输入。")
    
    # ==================== 场景机会用尽处理 ====================
    # 当玩家用尽交互机会但场景未正常结束时，执行强制结束逻辑
    if not scene_finished and not should_exit_game:
        print("\n==== 场景对话机会已用完，生成场景强制结束 ====")
        
        next_scene = script_ids[current_scene_index + 1] if current_scene_index + 1 < len(script_ids) else None
        
        # 判断是否需要根据情节发展插入新场景
        should_generate = director.should_generate_new_script(screenwriter, current_scene_id, next_scene)
        
        # 如果需要生成且未达到插入次数限制，则动态生成新场景
        if should_generate and inserted_scene_count < max_inserted_scenes:
            print(f"\n==== 根据情节发展，需要插入新场景 ({inserted_scene_count+1}/{max_inserted_scenes}) ====")
            
            # 使用AI生成新场景剧本
            new_script = screenwriter.generate_new_script(current_scene_id, dialogue_history=screenwriter.get_dialogue_history())
            
            if "error" not in new_script:
                # 更新导演剧本
                director.load_script(screenwriter.initial_script)
                
                # 更新场景列表并获取新场景ID
                script_ids = list(screenwriter.initial_script.keys())
                new_scene_id = list(new_script.keys())[0]
                print(f"\n成功生成新场景: {new_scene_id}")
                
                # 将新场景设为下一个场景
                next_scene = new_scene_id
                inserted_scene_count += 1
                
                # 检查并创建新场景中的角色
                director.check_and_create_new_characters(script_ids, script_ids.index(new_scene_id), player.name)
            else:
                print(f"\n生成新场景失败: {new_script.get('error')}")
        elif should_generate and inserted_scene_count >= max_inserted_scenes:
            print(f"\n==== 已达到插入新场景次数限制({max_inserted_scenes}次)，继续使用原有剧本 ====")
        
        # 生成场景结束描述
        ending_description = screenwriter.end_scene(last_interaction, director, current_scene_id, next_scene)
        print(f"\n{ending_description}")
        screenwriter.add_dialogue_record("旁白", "场景转场", f"从{current_scene_id}场景转场到{next_scene if next_scene else '故事结束'}")
        scene_finished = True
    
    # 检查是否需要退出游戏
    if should_exit_game:
        print("退出戏剧")
        break
    
    # 更新场景列表（可能包含新生成的场景）
    script_ids = list(screenwriter.initial_script.keys())
    
    # ==================== 动态场景生成逻辑 ====================
    # 当所有预定义场景完成后，尝试动态生成新场景或结局
    if current_scene_index + 1 >= len(script_ids):
        # 检查是否已达到场景生成次数限制
        if new_scene_generation_count >= max_new_scene_generations:
            print(f"\n==== 已达到场景生成次数限制({max_new_scene_generations}次)，准备结束故事 ====")
            # 生成结局场景提示词
            ending_prompt = "这是故事的结局场景。请根据之前的剧情走向，提供一个令人满意、符合逻辑且有情感冲击力的结束。确保所有主要情节线索都得到适当的解决。"
            
            # 生成结局场景
            new_script = screenwriter.generate_new_script(current_scene_id, ending_prompt, dialogue_history=screenwriter.get_dialogue_history())
            
            if "error" not in new_script:
                script_ids = list(screenwriter.initial_script.keys())
                ending_scene_id = list(new_script.keys())[0]
                print(f"\n成功生成结局场景: {ending_scene_id}")
                
                director.load_script(screenwriter.initial_script)
                
                # 调整场景索引指向结局场景
                current_scene_index = script_ids.index(current_scene_id)
                current_scene_index += 1
                # 检查并创建结局场景中的角色
                director.check_and_create_new_characters(script_ids, current_scene_index, player.name)
            else:
                print(f"\n生成结局场景失败: {new_script.get('error')}")
                break
        else:
            # 生成新场景（非结局）
            print(f"\n==== 所有计划场景已完成，尝试生成新场景 ({new_scene_generation_count+1}/{max_new_scene_generations}) ====")
            new_script = screenwriter.generate_new_script(current_scene_id, dialogue_history=screenwriter.get_dialogue_history())
            
            if "error" not in new_script:
                script_ids = list(screenwriter.initial_script.keys())
                new_scene_id = list(new_script.keys())[0]
                print(f"\n成功生成新场景: {new_scene_id}")
                
                director.load_script(screenwriter.initial_script)
                
                new_scene_generation_count += 1
                # 检查并创建新场景中的角色
                director.check_and_create_new_characters(script_ids, current_scene_index + 1, player.name)
            else:
                print(f"\n生成新场景失败: {new_script.get('error')}")
                break
    
    # 移动到下一个场景
    current_scene_index += 1
    
    # 检查是否为最后一个场景
    if current_scene_index < len(script_ids):
        print(f"\n==== 进入下一个场景: {script_ids[current_scene_index]} ====")
    else:
        print("\n==== 故事结束 ====")
        break

print("\n==== 表演结束 ====")


# ==================== 对话历史导出 ====================
# 将游戏过程中的所有对话记录导出到JSON文件，用于后续分析和测评
import json
import datetime

# 生成带时间戳的文件名，避免覆盖
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
dialogue_history_file = f"dialogue_history_{timestamp}.json"

# 获取完整对话历史
dialogue_history = screenwriter.get_all_dialogue_history()

# 保存到文件（UTF-8编码，格式化输出）
with open(dialogue_history_file, "w", encoding="utf-8") as f:
    json.dump(dialogue_history, f, ensure_ascii=False, indent=2)

print(f"\n对话历史已导出到文件: {dialogue_history_file}")




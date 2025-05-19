from role import Actor, Player, Screenwriter, Director
from config import max_inserted_scenes,max_inserted_scenes,try_chance

# 初始化角色
# 创建调查员(玩家角色)
player = Player("霍华德", 25, "男")

# 创建NPC角色
librarian = Actor("玛莎·迪尔", 57, "女")
librarian.add_memory("我是海港镇图书馆的管理员，已经工作了30年")
librarian.add_memory("最近镇上发生了一些奇怪的事情，特别是靠近海边的居民")
librarian.add_memory("我收藏了一些关于古老神话的禁忌书籍，包括《伊波恩之书》")
librarian.add_relationship(player.name, "谨慎", "知道他是个调查员，但不确定能否信任他")
librarian.add_trait("谨慎小心")
librarian.add_trait("博学多识")
librarian.add_trait("对神秘学知识有着复杂的好奇和恐惧")

professor = Actor("威廉·阿克雷", 68, "男")
professor.add_memory("我是密斯卡托尼克大学的前教授，研究古代文明和神话")
professor.add_memory("我见证了十年前的海港镇事件，那些不可名状的存在")
professor.add_memory("我的理智已经受到了损害，但我仍然试图阻止即将到来的灾难")
professor.add_relationship(player.name, "盟友", "认为他可能是阻止仪式的关键人物")
professor.add_relationship("玛莎·迪尔", "同谋", "她帮我保存了一些关键的古籍")
professor.add_trait("精神不稳定")
professor.add_trait("睿智但偏执")
professor.add_trait("勇敢但已伤痕累累")

cultist = Actor("约瑟夫·马什", 45, "男")
cultist.add_memory("我是深潜者的后裔，忠于达贡和海德拉")
cultist.add_memory("我表面上是普通渔民，实际负责监视镇上的外来者")
cultist.add_memory("我知道即将到来的仪式，我的血脉让我可以呼唤海中的存在")
cultist.add_relationship(player.name, "敌对", "怀疑他想干扰我们的仪式")
cultist.add_relationship("威廉·阿克雷", "仇恨", "他知道太多了，必须被处理掉")
cultist.add_trait("狂热")
cultist.add_trait("双面性格")
cultist.add_trait("残忍无情")

# 创建导演
director = Director()

# 添加演员到导演管理中
director.add_actor(librarian)
director.add_actor(professor)
director.add_actor(cultist)

# 创建编剧
screenwriter = Screenwriter()

# 加载CoC风格的剧本
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
        "characters": ["霍华德", "威廉·阿克雷","玛莎·迪尔"],
        "dialogues": [
            {"character": "威廉·阿克雷", "content": "(手微微颤抖，眼神飘忽)你找到《伊波恩之书》了吗？时间不多了，'它们'即将苏醒...(突然压低声音)你被跟踪了，小心那些'渔民'，他们不是人类..."},
        ]
    },
    "scene_4": {
        "description": "海港镇海滩，夜晚。月光被厚重的云层遮挡，只有零星的星光照亮沙滩。海浪拍打着岸边，发出低沉的声响。远处礁石上似乎站着几个人影，正在进行某种仪式。空气中弥漫着浓重的咸味和一种无法描述的异味。",
        "characters": ["霍华德", "约瑟夫·马什","威廉·阿克雷"],
        "dialogues": [
            {"character": "约瑟夫·马什", "content": "(站在祭坛旁边，双手举起一个奇怪的雕像)外来者，你不该来这里。这片海域属于伟大的存在，而我们即将得到祂们的祝福。"},
        ]
    }
}

# 获取剧本id列表
script_ids = list(coc_script.keys())

# 添加场景控制变量
current_scene_index = 0  # 当前场景索引
new_scene_generation_count = 0  # 新场景生成次数统计
max_new_scene_generations = 1  # 最大允许生成的新场景数
inserted_scene_count = 0  # 通过"根据情节发展，需要插入新场景"生成的场景次数
# max_inserted_scenes = 2  # 最大允许插入的新场景数

# 导演加载剧本
director.load_script(coc_script) 

# 编剧也加载相同的剧本
screenwriter.load_initial_script(coc_script)

# 设置当前场景
director.set_current_scene(script_ids[current_scene_index])

print("\n==== 表演开始 ====")

# 在while循环中，使用Director类中的方法
while current_scene_index < len(script_ids):
    # 获取当前场景编号
    current_scene_id = script_ids[current_scene_index]
    # 确保当前场景设置正确
    director.set_current_scene(current_scene_id)
    
    # 检查并创建当前场景中的新角色
    director.ensure_all_characters_exist(current_scene_id, player.name)

    print(f"\n当前场景编号: {current_scene_id}")

    detailed_scene = screenwriter.generate_scene_description(current_scene_id, director, player.get_player_name())
    print("\n当前场景描述:")
    print(detailed_scene)
    # 将生成的场景描述加入到dialogue_history中
    screenwriter.add_dialogue_record("旁白", "场景描述", detailed_scene)

    # 导演获取场景描述
    # !!有BUG，编剧修改的新描述没有更新到导演中!!
    # print("\n导演获取当前场景描述:")
    scene_desc = director.get_scene_description()
    # print(scene_desc)

    # 判断当前场景中是否有初始dialogues，如果有，则先让NPC角色表演
    scene_info = director.script.get(current_scene_id, {})
    initial_dialogues = scene_info.get("dialogues", [])
    
    if initial_dialogues:
        print("\n==== 对话开始 ====")
        for dialogue in initial_dialogues:
            character_name = dialogue.get("character")
            content = dialogue.get("content")
            
            # 跳过玩家角色的对话
            if character_name == player.get_player_name():
                continue
                
            # 显示NPC的对话
            print(f"\n{character_name}: {content}")
            
            # 记录对话到编剧的对话历史
            screenwriter.add_dialogue_record(character_name, "场景对话", content)
            
        # print("\n==== 初始对话结束 ====")

    # 显示当前场景可对话角色
    print("\n当前场景可对话角色:")
    characters = director.get_scene_characters(player=player)
    print(characters)


    # 场景循环，有x次机会
    scene_finished = False
    should_exit_game = False
    last_interaction = ""  # 记录最后一次对话或互动内容
    # try_chance = 2
    
    for i in range(try_chance):
        if scene_finished:
            break
            
        print("\n==== 请选择你的行动 ====")
        print("\n输入1与人物对话")
        print("\n输入2与环境互动")
        # print("\n输入next进入下一个场景")
        print("\n输入next进入下一个场景")
        print("\n输入esc退出戏剧")
        action = input("请输入你的选择:")
        if action == "1":
            # 与人物对话
            print("\n==== 请选择对话对象 ====")
            for j, character in enumerate(characters):
                print(f"\n{j+1}. {character}")
            choice = input("请选择对话对象:")
            if choice.isdigit() and 1 <= int(choice) <= len(characters):
                selected_character = characters[int(choice) - 1]
                print(f"\n==== 请输入对话内容 ====")
                dialogue = input("请输入对话内容:")
                # 记录最后一次对话
                last_interaction = f"{player.name}对{selected_character}说：{dialogue}"
                
                # 添加玩家对话记录
                screenwriter.add_dialogue_record(player.name, "对话", dialogue, target=selected_character)
                # 使用合并后的方法直接生成指导
                guide_message = director.guide_actor_from_player_speech(dialogue, selected_character)
                # 获取角色实例，而不仅仅是角色名称
                actor_instance = director.actors.get(selected_character)
                if actor_instance:
                    # 演员对话，使用Actor实例
                    response = player.talk_to_actor(actor_instance, dialogue, guide_message)
                    # 更新最后一次互动记录
                    last_interaction += f"\n{selected_character}回应：{response}"
                    
                    # 添加NPC对话记录
                    screenwriter.add_dialogue_record(selected_character, "对话", response, target=player.name)
                    
                    print(f"\n{selected_character}: {response}")

                    # 判断是否继续当前场景
                    if not director.is_scene_continuing(response):
                        print("当前场景结束")
                        scene_finished = True
                        # 获取下一个场景ID（如果有）
                        next_scene = script_ids[current_scene_index + 1] if current_scene_index + 1 < len(script_ids) else None
                        # 生成场景结束描述
                        ending_description = screenwriter.end_scene(last_interaction, director, current_scene_id, next_scene)
                        print(f"\n{ending_description}")
                        # 添加场景转场描述
                        screenwriter.add_dialogue_record("旁白", "场景转场", f"从{current_scene_id}场景转场到{next_scene if next_scene else '故事结束'}")
                    else:
                        print("当前场景继续")
                else:
                    print(f"错误：找不到角色 {selected_character} 的实例")
            else:
                print("无效的选择，请重新输入。")
        
        elif action == "2":
            # 与环境互动
            print("\n==== 请输入互动内容 ====")
            interaction = input("请输入互动内容:")
            # 记录最后一次互动
            last_interaction = f"{player.name}与环境互动：{interaction}"
            
            # 添加玩家互动记录
            screenwriter.add_dialogue_record(player.name, "环境互动", interaction)
            # 编剧处理玩家行动
            action_response = screenwriter.transform_scene(current_scene_id, interaction)
            # 更新最后一次互动记录
            last_interaction += f"\n环境响应：{action_response}"

            print(f"\n{action_response}")

            # 判断是否继续当前场景
            if not director.is_scene_continuing(action_response):
                print("当前场景结束")
                scene_finished = True
                # 获取下一个场景ID（如果有）
                next_scene = script_ids[current_scene_index + 1] if current_scene_index + 1 < len(script_ids) else None
                # 生成场景结束描述
                ending_description = screenwriter.end_scene(last_interaction, director, current_scene_id, next_scene)
                print(f"\n{ending_description}")
                # 添加场景转场描述
                screenwriter.add_dialogue_record("旁白", "场景转场", f"从{current_scene_id}场景转场到{next_scene if next_scene else '故事结束'}")
            else:
                print("当前场景继续")
        
        elif action.lower() == "next":
            # 手动进入下一个场景
            print("手动结束当前场景，进入下一个场景")
            scene_finished = True
            
            # 获取下一个场景ID（如果有）
            next_scene = script_ids[current_scene_index + 1] if current_scene_index + 1 < len(script_ids) else None
            
            if next_scene:
                # 创建简单的转场描述
                ending_description = screenwriter.end_scene(last_interaction or "玩家选择跳过当前场景", director, current_scene_id, next_scene)
                print(f"\n{ending_description}")
                # 添加场景转场描述
                screenwriter.add_dialogue_record("旁白", "场景转场", f"从{current_scene_id}场景转场到{next_scene}")
            else:
                print("\n没有更多场景，故事结束")
                should_exit_game = True
        
        elif action.lower() == "esc":
            # 手动退出整个戏剧
            # print("退出戏剧")
            should_exit_game = True
            break
        
        else:
            print("无效的选择，请重新输入。")
    
    # 如果10次机会用完但场景没有正常结束，则生成强制结束描述
    if not scene_finished and not should_exit_game:
        print("\n==== 场景对话机会已用完，生成场景强制结束 ====")
        
        # 获取下一个场景ID（如果有）
        next_scene = script_ids[current_scene_index + 1] if current_scene_index + 1 < len(script_ids) else None
        
        # 判断是否需要插入新场景
        should_generate = director.should_generate_new_script(screenwriter, current_scene_id, next_scene)
        
        if should_generate and inserted_scene_count < max_inserted_scenes:
            print(f"\n==== 根据情节发展，需要插入新场景 ({inserted_scene_count+1}/{max_inserted_scenes}) ====")
            
            # 直接生成新场景，不需要玩家反馈
            new_script = screenwriter.generate_new_script(current_scene_id, dialogue_history=screenwriter.get_dialogue_history())
            
            if "error" not in new_script:
                # 更新导演的剧本
                director.load_script(screenwriter.initial_script)
                
                # 重新获取并排序剧本ID列表
                script_ids = list(screenwriter.initial_script.keys())
                new_scene_id = list(new_script.keys())[0]
                print(f"\n成功生成新场景: {new_scene_id}")
                
                # 设置新场景为下一个场景
                next_scene = new_scene_id
                
                # 增加插入场景计数
                inserted_scene_count += 1
                
                # 检查并创建新角色
                director.check_and_create_new_characters(script_ids, script_ids.index(new_scene_id), player.name)
            else:
                print(f"\n生成新场景失败: {new_script.get('error')}")
        elif should_generate and inserted_scene_count >= max_inserted_scenes:
            print(f"\n==== 已达到插入新场景次数限制({max_inserted_scenes}次)，继续使用原有剧本 ====")
        
        # 生成场景结束描述，使用更新后的next_scene
        ending_description = screenwriter.end_scene(last_interaction, director, current_scene_id, next_scene)
        print(f"\n{ending_description}")
        
        # 添加场景转场描述
        screenwriter.add_dialogue_record("旁白", "场景转场", f"从{current_scene_id}场景转场到{next_scene if next_scene else '故事结束'}")
        
        # 设置场景为已结束
        scene_finished = True
    
    # 检查是否应该退出戏剧
    if should_exit_game:
        print("退出戏剧")
        break
    
    # 当前场景结束，更新剧本ID列表（可能有新场景生成）
    script_ids = list(screenwriter.initial_script.keys())
    
    # 检查是否需要生成全新场景（所有场景均已完成）
    # 生成全新场景有次数限制，比如3次，生成3次后，给剧本强制生成结尾描述
    if current_scene_index + 1 >= len(script_ids):
        # 检查是否已达到场景生成次数限制
        if new_scene_generation_count >= max_new_scene_generations:
            print(f"\n==== 已达到场景生成次数限制({max_new_scene_generations}次)，准备结束故事 ====")
            # 生成结尾场景，不需要玩家输入
            ending_prompt = "这是故事的结局场景。请根据之前的剧情走向，提供一个令人满意、符合逻辑且有情感冲击力的结束。确保所有主要情节线索都得到适当的解决。"
            
            # 使用特殊标记告诉编剧这是结尾
            new_script = screenwriter.generate_new_script(current_scene_id, ending_prompt, dialogue_history=screenwriter.get_dialogue_history())
            
            if "error" not in new_script:
                # 更新剧本ID列表
                script_ids = list(screenwriter.initial_script.keys())
                ending_scene_id = list(new_script.keys())[0]
                print(f"\n成功生成结局场景: {ending_scene_id}")
                
                # 更新导演的剧本
                director.load_script(screenwriter.initial_script)
                
                # 确保下一个场景是结局场景
                current_scene_index = script_ids.index(current_scene_id)
                current_scene_index += 1
                # 检查并创建新角色
                director.check_and_create_new_characters(script_ids, current_scene_index, player.name)
            else:
                print(f"\n生成结局场景失败: {new_script.get('error')}")
                break
            
            # 设置标志，表示这是最后一个场景
            is_final_scene = True
        else:
            print(f"\n==== 所有计划场景已完成，尝试生成新场景 ({new_scene_generation_count+1}/{max_new_scene_generations}) ====")
            # 生成新场景，不需要玩家输入
            new_script = screenwriter.generate_new_script(current_scene_id,dialogue_history=screenwriter.get_dialogue_history())
            
            if "error" not in new_script:
                # 更新剧本ID列表
                script_ids = list(screenwriter.initial_script.keys())
                new_scene_id = list(new_script.keys())[0]
                print(f"\n成功生成新场景: {new_scene_id}")
                
                # 更新导演的剧本
                director.load_script(screenwriter.initial_script)
                
                # 增加场景生成计数
                new_scene_generation_count += 1
                # 检查并创建新角色
                director.check_and_create_new_characters(script_ids, current_scene_index + 1, player.name)
            else:
                print(f"\n生成新场景失败: {new_script.get('error')}")
                break
    
    # 移动到下一个场景
    current_scene_index += 1
    
    # 检查是否是最后一个场景或已经没有更多场景
    if current_scene_index < len(script_ids):
        print(f"\n==== 进入下一个场景: {script_ids[current_scene_index]} ====")
    else:
        print("\n==== 故事结束 ====")
        break  # 确保故事结束后退出循环

print("\n==== 表演结束 ====")

# 导出对话历史到JSON文件供测评使用
import json
import datetime

# 创建带有时间戳的文件名
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
dialogue_history_file = f"dialogue_history_{timestamp}.json"

# 将对话历史转换为可序列化格式
dialogue_history = screenwriter.get_all_dialogue_history() # 获取全部对话历史

# 保存对话历史到文件
with open(dialogue_history_file, "w", encoding="utf-8") as f:
    json.dump(dialogue_history, f, ensure_ascii=False, indent=2)

print(f"\n对话历史已导出到文件: {dialogue_history_file}")




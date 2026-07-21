import tkinter as tk
from tkinter import scrolledtext, messagebox
import threading
import sys
import os
from openai import OpenAI
from config import ark_api_key, ark_base_url, ark_model, ark_embedding_model, ark_embedding_dim, openai_api_key, deepseek_api_key, qwen_api_key, http_proxy, https_proxy, use_model, try_chance, max_new_scene_generations, max_inserted_scenes
from role import Actor, Director, Player, Screenwriter, parse_ark_response, handle_stream_response, get_ark_client, get_client

# 设置代理环境变量
# 火山方舟作为国内API通常不需要代理
os.environ["http_proxy"] = http_proxy
os.environ["https_proxy"] = https_proxy
class StdoutRedirector:
    """将标准输出重定向到Tkinter文本组件，实现实时日志显示

    该类捕获stdout输出并显示在Tkinter Text组件中，
    实现GUI界面中的实时日志功能。

    Attributes:
        text_widget: 目标Tkinter Text组件
        buffer: 内部缓冲区，用于累积输出内容
    """

    def __init__(self, text_widget):
        """使用目标文本组件初始化重定向器

        Args:
            text_widget: 用于显示stdout的Tkinter Text组件
        """
        self.text_widget = text_widget
        self.buffer = ""

    def write(self, string):
        """将字符串写入文本组件并自动滚动到末尾

        Args:
            string: 要写入的文本
        """
        self.buffer += string
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)

    def flush(self):
        """文件类接口所需的flush方法"""
        pass

class SettingsDialog:
    """游戏设置对话框，用于配置模型和场景参数

    该对话框允许用户配置：
    - AI模型选择（OpenAI、DeepSeek、Qwen）
    - 每场景尝试次数
    - 最大插入场景数
    - 最大新场景生成数

    Attributes:
        parent: 父Tkinter窗口
        dialog: Toplevel对话框窗口
        model_var: 选中模型的StringVar
        try_chance_var: 场景尝试次数的IntVar
        max_inserted_scenes_var: 最大插入场景数的IntVar
        max_new_scene_generations_var: 最大新场景生成数的IntVar
        result: 存储用户设置的字典
    """

    def __init__(self, parent):
        """初始化设置对话框

        Args:
            parent: 父Tkinter窗口
        """
        self.parent = parent
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("游戏设置")
        self.dialog.geometry("500x450")
        self.dialog.configure(bg="#1e1e2e")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # 使用当前值初始化设置变量
        self.model_var = tk.StringVar(value=use_model)
        self.try_chance_var = tk.IntVar(value=try_chance)
        self.max_inserted_scenes_var = tk.IntVar(value=max_inserted_scenes)
        self.max_new_scene_generations_var = tk.IntVar(value=max_new_scene_generations)
        
        # 创建UI组件
        self.create_widgets()
        
        # 使用默认值初始化结果
        self.result = {
            "model": use_model,
            "try_chance": try_chance,
            "max_inserted_scenes": max_inserted_scenes,
            "max_new_scene_generations": max_new_scene_generations
        }
        
        # 使对话框居中显示在父窗口上
        self.dialog.update_idletasks()
        width = self.dialog.winfo_width()
        height = self.dialog.winfo_height()
        x = parent.winfo_rootx() + (parent.winfo_width() - width) // 2
        y = parent.winfo_rooty() + (parent.winfo_height() - height) // 2
        self.dialog.geometry(f'{width}x{height}+{x}+{y}')
        
        # 强制窗口刷新
        self.dialog.update()
        
        # 设置窗口最小大小
        self.dialog.minsize(400, 300)
        
        # 处理窗口关闭事件
        self.dialog.protocol("WM_DELETE_WINDOW", self.on_cancel)
        
    def create_widgets(self):
        main_frame = tk.Frame(self.dialog, bg="#1e1e2e", padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 模型选择
        model_frame = tk.LabelFrame(main_frame, text="模型设置", bg="#2a2a3e", fg="#ffffff", 
                                  font=("微软雅黑", 12, "bold"), padx=10, pady=10)
        model_frame.pack(fill=tk.X, pady=5)
        
        models = [("OpenAI (gpt-4o-mini)", "gpt-4o-mini"), 
                 ("DeepSeek (deepseek-chat)", "deepseek-chat"), 
                 ("Qianwen (qwen3-235b-a22b)", "qwen3-235b-a22b")]
        
        for i, (text, value) in enumerate(models):
            radio = tk.Radiobutton(model_frame, text=text, value=value, variable=self.model_var,
                          bg="#2a2a3e", fg="#ffffff", selectcolor="#3d3d60", 
                          activebackground="#2a2a3e", activeforeground="#ffffff",
                          font=("微软雅黑", 10))
            radio.pack(anchor=tk.W, pady=2)
            # 确保按钮立即显示
            radio.update()
        
        # 场景设置
        scene_frame = tk.LabelFrame(main_frame, text="场景设置", bg="#2a2a3e", fg="#ffffff", 
                                  font=("微软雅黑", 12, "bold"), padx=10, pady=10)
        scene_frame.pack(fill=tk.X, pady=10)
        
        # 使用Grid布局以确保对齐
        scene_frame.grid_columnconfigure(0, weight=3)
        scene_frame.grid_columnconfigure(1, weight=1)
        
        # 尝试机会
        row = 0
        tk.Label(scene_frame, text="每场景尝试机会:", bg="#2a2a3e", fg="#ffffff", 
               font=("微软雅黑", 10)).grid(row=row, column=0, sticky=tk.W, pady=5)
        
        try_chance_spinbox = tk.Spinbox(scene_frame, from_=1, to=5, textvariable=self.try_chance_var,
                                     width=5, bg="#2d2d40", fg="#ffffff", font=("微软雅黑", 10))
        try_chance_spinbox.grid(row=row, column=1, sticky=tk.W, padx=10)
        
        # 最大插入场景数
        row += 1
        tk.Label(scene_frame, text="最大插入场景数:", bg="#2a2a3e", fg="#ffffff", 
               font=("微软雅黑", 10)).grid(row=row, column=0, sticky=tk.W, pady=5)
        
        max_inserted_spinbox = tk.Spinbox(scene_frame, from_=0, to=3, textvariable=self.max_inserted_scenes_var,
                                       width=5, bg="#2d2d40", fg="#ffffff", font=("微软雅黑", 10))
        max_inserted_spinbox.grid(row=row, column=1, sticky=tk.W, padx=10)
        
        # 最大新场景生成数
        row += 1
        tk.Label(scene_frame, text="最大新场景生成数:", bg="#2a2a3e", fg="#ffffff", 
               font=("微软雅黑", 10)).grid(row=row, column=0, sticky=tk.W, pady=5)
        
        max_new_spinbox = tk.Spinbox(scene_frame, from_=0, to=3, textvariable=self.max_new_scene_generations_var,
                                   width=5, bg="#2d2d40", fg="#ffffff", font=("微软雅黑", 10))
        max_new_spinbox.grid(row=row, column=1, sticky=tk.W, padx=10)
        
        # 按钮区域
        button_frame = tk.Frame(main_frame, bg="#1e1e2e", pady=10)
        button_frame.pack(fill=tk.X)
        
        # 添加确定按钮
        ok_button = tk.Button(button_frame, text="确定", command=self.on_ok, 
                bg="#3d3d60", fg="#ffffff", font=("微软雅黑", 10),
                width=10)
        ok_button.pack(side=tk.RIGHT, padx=5)
        
        # 添加取消按钮
        cancel_button = tk.Button(button_frame, text="取消", command=self.on_cancel,
                bg="#3d3d60", fg="#ffffff", font=("微软雅黑", 10),
                width=10)
        cancel_button.pack(side=tk.RIGHT, padx=5)
        
        # 强制更新所有小部件
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
        # 当取消时，将结果设置为None
        self.result = None
        self.dialog.destroy()
    
    def show(self):
        # 在主窗口中等待对话框
        self.parent.wait_window(self.dialog)
        return self.result

class CoCGameGUI:
    """克苏鲁的呼唤互动游戏主GUI类

    该类管理整个游戏界面和游戏流程，包括：
    - 显示设置对话框进行配置
    - 创建UI元素（场景描述、角色列表、动作按钮）
    - 处理玩家交互（对话、环境互动）
    - 管理场景转换和进度
    - 退出时导出对话历史

    Attributes:
        root: 主Tkinter窗口
        settings: 游戏配置设置
        current_scene_index: 当前场景在script_ids中的索引
        script_ids: 场景ID列表
        scene_finished: 表示当前场景是否结束的标志
        should_exit_game: 表示游戏是否应该退出的标志
        last_interaction: 最后一次互动记录
        inserted_scene_count: 插入场景计数
        new_scene_generation_count: 新场景生成计数
        characters: 当前场景可用角色列表
        player: Player对象
        director: Director对象
        screenwriter: Screenwriter对象
        scene_text: 场景显示的ScrolledText组件
        character_listbox: 角色选择的Listbox组件
        input_text: 玩家输入的Text组件
        log_text: 游戏日志的ScrolledText组件
        stdout_redirector: 日志显示的StdoutRedirector实例
    """

    def __init__(self, root):
        """初始化游戏GUI

        Args:
            root: 主Tkinter窗口
        """
        self.root = root
        self.root.title("克苏鲁的呼唤 - 互动游戏")
        self.root.geometry("1200x800")
        self.root.configure(bg="#1e1e2e")
        
        # 在游戏初始化前显示设置对话框
        self.show_settings_dialog()
        
        # 初始化游戏变量
        self.setup_variables()
        
        # 创建UI元素
        self.create_ui()
        
        # 在后台线程中启动游戏初始化
        self.initialize_game()
        
    def show_settings_dialog(self):
        # 显示设置对话框前，先确保根窗口已经完全初始化
        self.root.update_idletasks()
        
        # 创建并显示设置对话框
        dialog = SettingsDialog(self.root)
        settings = dialog.show()
        
        # 检查用户是否取消了设置
        if not settings:
            # 如果取消，使用默认值
            settings = {
                "model": use_model,
                "try_chance": try_chance,
                "max_inserted_scenes": max_inserted_scenes,
                "max_new_scene_generations": max_new_scene_generations
            }
            print("\n用户取消了设置，将使用默认配置")
        
        # 保存设置
        self.settings = settings
        
        # 打印当前配置
        print(f"\n===== 游戏配置 =====")
        print(f"使用模型: {settings['model']}")
        print(f"场景尝试机会: {settings['try_chance']}")
        print(f"最大插入场景数: {settings['max_inserted_scenes']}")
        print(f"最大新场景生成数: {settings['max_new_scene_generations']}")
        print(f"====================\n")
        
    def setup_variables(self):
        # 游戏状态变量
        self.current_scene_index = 0
        self.script_ids = []
        self.scene_finished = False
        self.should_exit_game = False
        self.last_interaction = ""
        self.inserted_scene_count = 0
        self.new_scene_generation_count = 0
        self.characters = []
        
        # 角色与导演
        self.player = None
        self.director = None
        self.screenwriter = None

    def create_ui(self):
        # 创建主框架
        main_frame = tk.Frame(self.root, bg="#1e1e2e")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 分割窗口
        top_frame = tk.Frame(main_frame, bg="#1e1e2e")
        top_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        bottom_frame = tk.Frame(main_frame, bg="#1e1e2e", height=200)
        bottom_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 场景和对话显示区
        left_frame = tk.Frame(top_frame, bg="#1e1e2e", width=800)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 场景描述区域
        scene_frame = tk.LabelFrame(left_frame, text="场景描述", bg="#2a2a3e", fg="#ffffff", font=("微软雅黑", 12, "bold"))
        scene_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.scene_text = scrolledtext.ScrolledText(scene_frame, wrap=tk.WORD, bg="#2d2d40", fg="#ffffff", font=("微软雅黑", 11))
        self.scene_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 角色和动作区
        right_frame = tk.Frame(top_frame, bg="#1e1e2e", width=300)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5, pady=5)
        
        # 角色选择区域
        character_frame = tk.LabelFrame(right_frame, text="角色", bg="#2a2a3e", fg="#ffffff", font=("微软雅黑", 12, "bold"))
        character_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.character_listbox = tk.Listbox(character_frame, bg="#2d2d40", fg="#ffffff", font=("微软雅黑", 11), selectbackground="#3d3d60")
        self.character_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 动作按钮区域
        action_frame = tk.LabelFrame(right_frame, text="动作", bg="#2a2a3e", fg="#ffffff", font=("微软雅黑", 12, "bold"))
        action_frame.pack(fill=tk.BOTH, padx=5, pady=5)
        
        # 对话按钮
        self.talk_btn = tk.Button(action_frame, text="与角色对话", bg="#3d3d60", fg="#ffffff", font=("微软雅黑", 11),
                               command=self.talk_to_character)
        self.talk_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # 互动按钮
        self.interact_btn = tk.Button(action_frame, text="与环境互动", bg="#3d3d60", fg="#ffffff", font=("微软雅黑", 11),
                                  command=self.interact_with_environment)
        self.interact_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # 下一场景按钮
        self.next_btn = tk.Button(action_frame, text="下一场景", bg="#3d3d60", fg="#ffffff", font=("微软雅黑", 11),
                               command=self.go_to_next_scene)
        self.next_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # 退出按钮
        self.exit_btn = tk.Button(action_frame, text="退出游戏", bg="#3d3d60", fg="#ffffff", font=("微软雅黑", 11),
                               command=self.exit_game)
        self.exit_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # 对话输入区域
        input_frame = tk.LabelFrame(bottom_frame, text="输入", bg="#2a2a3e", fg="#ffffff", font=("微软雅黑", 12, "bold"))
        input_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.input_text = tk.Text(input_frame, height=3, bg="#2d2d40", fg="#ffffff", font=("微软雅黑", 11))
        self.input_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 添加提示文本
        tip_label = tk.Label(input_frame, text="请输入内容后点击'与角色对话'或'与环境互动'按钮", 
                           bg="#2a2a3e", fg="#b0b0c0", font=("微软雅黑", 9, "italic"))
        tip_label.pack(side=tk.RIGHT, padx=5, pady=2)
        
        # 游戏日志区域
        log_frame = tk.LabelFrame(main_frame, text="游戏日志", bg="#2a2a3e", fg="#ffffff", font=("微软雅黑", 12, "bold"))
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, bg="#2d2d40", fg="#ffffff", font=("微软雅黑", 10))
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 重定向标准输出到日志区域
        self.stdout_redirector = StdoutRedirector(self.log_text)
        sys.stdout = self.stdout_redirector

    def initialize_game(self):
        # 启动游戏初始化线程，避免UI卡顿
        threading.Thread(target=self._initialize_game_thread, daemon=True).start()
    
    def _initialize_game_thread(self):
        # 修改全局变量以应用设置
        global use_model, try_chance, max_inserted_scenes, max_new_scene_generations
        
        # 保存当前设置值
        selected_model = self.settings['model']
        selected_try_chance = self.settings['try_chance']
        selected_max_inserted_scenes = self.settings['max_inserted_scenes']
        selected_max_new_scene_generations = self.settings['max_new_scene_generations']
        
        # 直接修改全局变量
        use_model = selected_model
        try_chance = selected_try_chance
        max_inserted_scenes = selected_max_inserted_scenes
        max_new_scene_generations = selected_max_new_scene_generations
        
        print(f"\n===== 确认游戏配置已生效 =====")
        print(f"使用模型: {use_model}")
        print(f"场景尝试机会: {try_chance}")
        print(f"最大插入场景数: {max_inserted_scenes}")
        print(f"最大新场景生成数: {max_new_scene_generations}")
        print(f"============================\n")
        
        # 创建一个函数以强制使用选定的模型设置
        def create_actor_with_model(name, age, gender):
            actor = Actor(name, age, gender)
            # 根据用户选择的模型重新设置talk_client
            if selected_model == "ark":
                actor.talk_client = get_ark_client()
            elif selected_model == "gpt-4o-mini":
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
        
        # 初始化角色
        # 创建调查员(玩家角色)
        self.player = Player("霍华德", 25, "男")
        
        # 使用自定义函数创建NPC角色，确保使用正确的模型
        librarian = create_actor_with_model("玛莎·迪尔", 57, "女")
        librarian.add_memory("我是海港镇图书馆的管理员，已经工作了30年")
        librarian.add_memory("最近镇上发生了一些奇怪的事情，特别是靠近海边的居民")
        librarian.add_memory("我收藏了一些关于古老神话的禁忌书籍，包括《伊波恩之书》")
        librarian.add_relationship(self.player.name, "谨慎", "知道他是个调查员，但不确定能否信任他")
        librarian.add_trait("谨慎小心")
        librarian.add_trait("博学多识")
        librarian.add_trait("对神秘学知识有着复杂的好奇和恐惧")

        professor = create_actor_with_model("威廉·阿克雷", 68, "男")
        professor.add_memory("我是密斯卡托尼克大学的前教授，研究古代文明和神话")
        professor.add_memory("我见证了十年前的海港镇事件，那些不可名状的存在")
        professor.add_memory("我的理智已经受到了损害，但我仍然试图阻止即将到来的灾难")
        professor.add_relationship(self.player.name, "盟友", "认为他可能是阻止仪式的关键人物")
        professor.add_relationship("玛莎·迪尔", "同谋", "她帮我保存了一些关键的古籍")
        professor.add_trait("精神不稳定")
        professor.add_trait("睿智但偏执")
        professor.add_trait("勇敢但已伤痕累累")

        cultist = create_actor_with_model("约瑟夫·马什", 45, "男")
        cultist.add_memory("我是深潜者的后裔，忠于达贡和海德拉")
        cultist.add_memory("我表面上是普通渔民，实际负责监视镇上的外来者")
        cultist.add_memory("我知道即将到来的仪式，我的血脉让我可以呼唤海中的存在")
        cultist.add_relationship(self.player.name, "敌对", "怀疑他想干扰我们的仪式")
        cultist.add_relationship("威廉·阿克雷", "仇恨", "他知道太多了，必须被处理掉")
        cultist.add_trait("狂热")
        cultist.add_trait("双面性格")
        cultist.add_trait("残忍无情")

        # 创建导演
        self.director = Director()

        # 添加演员到导演管理中
        self.director.add_actor(librarian)
        self.director.add_actor(professor)
        self.director.add_actor(cultist)

        # 创建编剧
        self.screenwriter = Screenwriter()

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
        self.script_ids = list(coc_script.keys())

        # 导演加载剧本
        self.director.load_script(coc_script) 

        # 编剧也加载相同的剧本
        self.screenwriter.load_initial_script(coc_script)

        # 设置当前场景
        self.current_scene_index = 0
        self.director.set_current_scene(self.script_ids[self.current_scene_index])

        # 输出实际使用的模型
        print(f"\n===== 模型确认 =====")
        print(f"玛莎·迪尔使用的模型端点: {librarian.talk_client.base_url}")
        print(f"威廉·阿克雷使用的模型端点: {professor.talk_client.base_url}")
        print(f"约瑟夫·马什使用的模型端点: {cultist.talk_client.base_url}")
        print(f"====================\n")
        
        # 启动第一个场景
        self.root.after(0, self.start_scene)

    def start_scene(self):
        # 获取当前场景编号
        current_scene_id = self.script_ids[self.current_scene_index]
        # 确保当前场景设置正确
        self.director.set_current_scene(current_scene_id)
        
        # 检查并创建当前场景中的新角色
        self.director.ensure_all_characters_exist(current_scene_id, self.player.name)

        print(f"\n当前场景编号: {current_scene_id}")

        # 生成场景描述
        detailed_scene = self.screenwriter.generate_scene_description(current_scene_id, self.director, self.player.get_player_name())
        self.scene_text.delete(1.0, tk.END)
        self.scene_text.insert(tk.END, detailed_scene)
        
        # 将生成的场景描述加入到dialogue_history中
        self.screenwriter.add_dialogue_record("旁白", "场景描述", detailed_scene)

        # 判断当前场景中是否有初始dialogues，如果有，则先让NPC角色表演
        scene_info = self.director.script.get(current_scene_id, {})
        initial_dialogues = scene_info.get("dialogues", [])
        
        if initial_dialogues:
            print("\n==== 对话开始 ====")
            for dialogue in initial_dialogues:
                character_name = dialogue.get("character")
                content = dialogue.get("content")
                
                # 跳过玩家角色的对话
                if character_name == self.player.get_player_name():
                    continue
                    
                # 显示NPC的对话
                print(f"\n{character_name}: {content}")
                self.scene_text.insert(tk.END, f"\n\n{character_name}: {content}")
                
                # 记录对话到编剧的对话历史
                self.screenwriter.add_dialogue_record(character_name, "场景对话", content)

        # 更新可对话角色列表
        self.update_character_list()
        
        # 重置场景状态
        self.scene_finished = False
        self.last_interaction = ""
        
        # 初始化场景尝试次数计数器
        self.scene_try_count = 0
        
        # 显示场景尝试次数上限
        print(f"\n==== 当前场景尝试机会: {self.settings['try_chance']} 次 ====")

    def update_character_list(self):
        # 获取当前场景可对话角色
        self.characters = self.director.get_scene_characters(player=self.player)
        
        # 更新列表框
        self.character_listbox.delete(0, tk.END)
        for character in self.characters:
            self.character_listbox.insert(tk.END, character)

    def talk_to_character(self):
        if self.scene_finished:
            messagebox.showinfo("提示", "当前场景已结束，请进入下一个场景")
            return
            
        # 检查是否还有尝试次数
        if self.scene_try_count >= self.settings['try_chance'] and not self.scene_finished:
            messagebox.showinfo("提示", f"已达到场景尝试次数上限({self.settings['try_chance']}次)")
            self.handle_scene_timeout()
            return
            
        # 检查是否选择了角色
        selected_idx = self.character_listbox.curselection()
        if not selected_idx:
            messagebox.showinfo("提示", "请先选择一个角色")
            return
            
        selected_character = self.characters[selected_idx[0]]
        
        # 获取对话内容
        dialogue = self.input_text.get(1.0, tk.END).strip()
        if not dialogue:
            messagebox.showinfo("提示", "请输入对话内容")
            return
        
        # 增加场景尝试次数
        self.scene_try_count += 1
        print(f"\n==== 尝试次数: {self.scene_try_count}/{self.settings['try_chance']} ====")
            
        # 记录最后一次对话
        self.last_interaction = f"{self.player.name}对{selected_character}说：{dialogue}"
        
        # 添加玩家对话记录
        self.screenwriter.add_dialogue_record(self.player.name, "对话", dialogue, target=selected_character)
        
        # 在场景文本中添加玩家对话
        self.scene_text.insert(tk.END, f"\n\n{self.player.name}: {dialogue}")
        
        # 使用合并后的方法直接生成指导
        guide_message = self.director.guide_actor_from_player_speech(dialogue, selected_character)
        
        # 获取角色实例，而不仅仅是角色名称
        actor_instance = self.director.actors.get(selected_character)
        
        if actor_instance:
            # 演员对话，使用Actor实例
            response = self.player.talk_to_actor(actor_instance, dialogue, guide_message)
            
            # 更新最后一次互动记录
            self.last_interaction += f"\n{selected_character}回应：{response}"
            
            # 添加NPC对话记录
            self.screenwriter.add_dialogue_record(selected_character, "对话", response, target=self.player.name)
            
            # 在场景文本中添加NPC回应
            self.scene_text.insert(tk.END, f"\n\n{selected_character}: {response}")
            self.scene_text.see(tk.END)
            
            print(f"\n{selected_character}: {response}")

            # 判断是否继续当前场景
            if not self.director.is_scene_continuing(response):
                print("当前场景结束")
                self.scene_finished = True
                
                # 获取下一个场景ID（如果有）
                next_scene = self.script_ids[self.current_scene_index + 1] if self.current_scene_index + 1 < len(self.script_ids) else None
                
                # 判断是否需要插入新场景
                should_generate = self.director.should_generate_new_script(self.screenwriter, 
                                                                         self.script_ids[self.current_scene_index], 
                                                                         next_scene)
                
                if should_generate and self.inserted_scene_count < self.settings['max_inserted_scenes']:
                    print(f"\n==== 根据情节发展，需要插入新场景 ({self.inserted_scene_count+1}/{self.settings['max_inserted_scenes']}) ====")
                    
                    # 生成新场景
                    new_script = self.screenwriter.generate_new_script(self.script_ids[self.current_scene_index], 
                                                                   dialogue_history=self.screenwriter.get_dialogue_history())
                    
                    if "error" not in new_script:
                        # 更新导演的剧本
                        self.director.load_script(self.screenwriter.initial_script)
                        
                        # 重新获取并排序剧本ID列表
                        self.script_ids = list(self.screenwriter.initial_script.keys())
                        new_scene_id = list(new_script.keys())[0]
                        print(f"\n成功生成新场景: {new_scene_id}")
                        
                        # 设置新场景为下一个场景
                        next_scene = new_scene_id
                        
                        # 增加插入场景计数
                        self.inserted_scene_count += 1
                        
                        # 检查并创建新角色
                        self.director.check_and_create_new_characters(self.script_ids, 
                                                                   self.script_ids.index(new_scene_id), 
                                                                   self.player.name)
                    else:
                        print(f"\n生成新场景失败: {new_script.get('error')}")
                elif should_generate and self.inserted_scene_count >= self.settings['max_inserted_scenes']:
                    print(f"\n==== 已达到插入新场景次数限制({self.settings['max_inserted_scenes']}次)，继续使用原有剧本 ====")
                
                # 生成场景结束描述
                ending_description = self.screenwriter.end_scene(self.last_interaction, self.director, 
                                                              self.script_ids[self.current_scene_index], next_scene)
                
                # 显示场景结束描述
                self.scene_text.insert(tk.END, f"\n\n{ending_description}")
                self.scene_text.see(tk.END)
                print(f"\n{ending_description}")
                
                # 添加场景转场描述
                self.screenwriter.add_dialogue_record("旁白", "场景转场", 
                                                   f"从{self.script_ids[self.current_scene_index]}场景转场到{next_scene if next_scene else '故事结束'}")
                
                # 提示用户进入下一场景
                messagebox.showinfo("场景结束", "当前场景已结束，请点击'下一场景'按钮继续")
            else:
                print("当前场景继续")
                
                # 检查是否已达到尝试次数上限
                if self.scene_try_count >= self.settings['try_chance'] and not self.scene_finished:
                    print(f"\n==== 场景尝试机会已用完 ({self.scene_try_count}/{self.settings['try_chance']})，强制结束场景 ====")
                    self.handle_scene_timeout()
        else:
            print(f"错误：找不到角色 {selected_character} 的实例")
            
        # 清空输入框
        self.input_text.delete(1.0, tk.END)

    def interact_with_environment(self):
        if self.scene_finished:
            messagebox.showinfo("提示", "当前场景已结束，请进入下一个场景")
            return
            
        # 检查是否还有尝试次数
        if self.scene_try_count >= self.settings['try_chance'] and not self.scene_finished:
            messagebox.showinfo("提示", f"已达到场景尝试次数上限({self.settings['try_chance']}次)")
            self.handle_scene_timeout()
            return
            
        # 获取互动内容
        interaction = self.input_text.get(1.0, tk.END).strip()
        if not interaction:
            messagebox.showinfo("提示", "请输入互动内容")
            return
            
        # 增加场景尝试次数
        self.scene_try_count += 1
        print(f"\n==== 尝试次数: {self.scene_try_count}/{self.settings['try_chance']} ====")
        
        # 记录最后一次互动
        self.last_interaction = f"{self.player.name}与环境互动：{interaction}"
        
        # 添加玩家互动记录
        self.screenwriter.add_dialogue_record(self.player.name, "环境互动", interaction)
        
        # 在场景文本中添加玩家互动
        self.scene_text.insert(tk.END, f"\n\n{self.player.name} 行动: {interaction}")
        
        # 编剧处理玩家行动
        action_response = self.screenwriter.transform_scene(self.script_ids[self.current_scene_index], interaction)
        
        # 更新最后一次互动记录
        self.last_interaction += f"\n环境响应：{action_response}"

        # 在场景文本中添加环境响应
        self.scene_text.insert(tk.END, f"\n\n{action_response}")
        self.scene_text.see(tk.END)
        
        print(f"\n{action_response}")

        # 判断是否继续当前场景
        if not self.director.is_scene_continuing(action_response):
            print("当前场景结束")
            self.scene_finished = True
            
            # 获取下一个场景ID（如果有）
            next_scene = self.script_ids[self.current_scene_index + 1] if self.current_scene_index + 1 < len(self.script_ids) else None
            
            # 判断是否需要插入新场景
            should_generate = self.director.should_generate_new_script(self.screenwriter, 
                                                                     self.script_ids[self.current_scene_index], 
                                                                     next_scene)
            
            if should_generate and self.inserted_scene_count < self.settings['max_inserted_scenes']:
                print(f"\n==== 根据情节发展，需要插入新场景 ({self.inserted_scene_count+1}/{self.settings['max_inserted_scenes']}) ====")
                
                # 生成新场景
                new_script = self.screenwriter.generate_new_script(self.script_ids[self.current_scene_index], 
                                                               dialogue_history=self.screenwriter.get_dialogue_history())
                
                if "error" not in new_script:
                    # 更新导演的剧本
                    self.director.load_script(self.screenwriter.initial_script)
                    
                    # 重新获取并排序剧本ID列表
                    self.script_ids = list(self.screenwriter.initial_script.keys())
                    new_scene_id = list(new_script.keys())[0]
                    print(f"\n成功生成新场景: {new_scene_id}")
                    
                    # 设置新场景为下一个场景
                    next_scene = new_scene_id
                    
                    # 增加插入场景计数
                    self.inserted_scene_count += 1
                    
                    # 检查并创建新角色
                    self.director.check_and_create_new_characters(self.script_ids, 
                                                               self.script_ids.index(new_scene_id), 
                                                               self.player.name)
                else:
                    print(f"\n生成新场景失败: {new_script.get('error')}")
            elif should_generate and self.inserted_scene_count >= self.settings['max_inserted_scenes']:
                print(f"\n==== 已达到插入新场景次数限制({self.settings['max_inserted_scenes']}次)，继续使用原有剧本 ====")
            
            # 生成场景结束描述
            ending_description = self.screenwriter.end_scene(self.last_interaction, self.director, 
                                                          self.script_ids[self.current_scene_index], next_scene)
            
            # 显示场景结束描述
            self.scene_text.insert(tk.END, f"\n\n{ending_description}")
            self.scene_text.see(tk.END)
            print(f"\n{ending_description}")
            
            # 添加场景转场描述
            self.screenwriter.add_dialogue_record("旁白", "场景转场", 
                                               f"从{self.script_ids[self.current_scene_index]}场景转场到{next_scene if next_scene else '故事结束'}")
            
            # 提示用户进入下一场景
            messagebox.showinfo("场景结束", "当前场景已结束，请点击'下一场景'按钮继续")
        else:
            print("当前场景继续")
            
            # 检查是否已达到尝试次数上限
            if self.scene_try_count >= self.settings['try_chance'] and not self.scene_finished:
                print(f"\n==== 场景尝试机会已用完 ({self.scene_try_count}/{self.settings['try_chance']})，强制结束场景 ====")
                self.handle_scene_timeout()
            
        # 清空输入框
        self.input_text.delete(1.0, tk.END)

    def go_to_next_scene(self):
        # 获取当前场景ID
        current_scene_id = self.script_ids[self.current_scene_index]
        
        if not self.scene_finished:
            # 如果场景未结束，询问用户是否确定跳过
            if not messagebox.askyesno("确认", "当前场景尚未结束，确定要跳过吗？"):
                return
                
            # 手动结束当前场景
            print("手动结束当前场景，进入下一个场景")
            
            # 获取下一个场景ID（如果有）
            next_scene = self.script_ids[self.current_scene_index + 1] if self.current_scene_index + 1 < len(self.script_ids) else None
            
            if next_scene:
                # 创建简单的转场描述
                ending_description = self.screenwriter.end_scene(self.last_interaction or "玩家选择跳过当前场景", 
                                                              self.director, current_scene_id, next_scene)
                                                              
                # 显示场景结束描述
                self.scene_text.insert(tk.END, f"\n\n{ending_description}")
                self.scene_text.see(tk.END)
                print(f"\n{ending_description}")
                
                # 添加场景转场描述
                self.screenwriter.add_dialogue_record("旁白", "场景转场", f"从{current_scene_id}场景转场到{next_scene}")
            else:
                print("\n没有更多场景，故事结束")
                messagebox.showinfo("游戏结束", "恭喜您完成了所有场景！")
                return
                
        # 移动到下一个场景
        self.current_scene_index += 1
        
        # 检查是否还有场景
        if self.current_scene_index < len(self.script_ids):
            print(f"\n==== 进入下一个场景: {self.script_ids[self.current_scene_index]} ====")
            # 开始新场景
            self.start_scene()
        else:
            # 更新剧本ID列表（可能有新场景生成）
            self.script_ids = list(self.screenwriter.initial_script.keys())
            
            # 检查是否需要生成全新场景（所有场景均已完成）
            if self.current_scene_index >= len(self.script_ids):
                # 检查是否已达到场景生成次数限制
                if self.new_scene_generation_count >= self.settings['max_new_scene_generations']:
                    print(f"\n==== 已达到场景生成次数限制({self.settings['max_new_scene_generations']}次)，准备结束故事 ====")
                    # 生成结尾场景
                    ending_prompt = "这是故事的结局场景。请根据之前的剧情走向，提供一个令人满意、符合逻辑且有情感冲击力的结束。确保所有主要情节线索都得到适当的解决。"
                    
                    # 使用特殊标记告诉编剧这是结尾
                    new_script = self.screenwriter.generate_new_script(current_scene_id, ending_prompt, 
                                                                    dialogue_history=self.screenwriter.get_dialogue_history())
                    
                    if "error" not in new_script:
                        # 更新剧本ID列表
                        self.script_ids = list(self.screenwriter.initial_script.keys())
                        ending_scene_id = list(new_script.keys())[0]
                        print(f"\n成功生成结局场景: {ending_scene_id}")
                        
                        # 更新导演的剧本
                        self.director.load_script(self.screenwriter.initial_script)
                        
                        # 确保下一个场景是结局场景
                        self.current_scene_index = self.script_ids.index(current_scene_id) + 1
                        # 检查并创建新角色
                        self.director.check_and_create_new_characters(self.script_ids, self.current_scene_index, self.player.name)
                        
                        # 开始新场景
                        self.start_scene()
                    else:
                        print(f"\n生成结局场景失败: {new_script.get('error')}")
                        messagebox.showinfo("游戏结束", "故事已结束，感谢您的参与！")
                else:
                    print(f"\n==== 所有计划场景已完成，尝试生成新场景 ({self.new_scene_generation_count+1}/{self.settings['max_new_scene_generations']}) ====")
                    # 生成新场景
                    new_script = self.screenwriter.generate_new_script(current_scene_id, 
                                                                    dialogue_history=self.screenwriter.get_dialogue_history())
                    
                    if "error" not in new_script:
                        # 更新剧本ID列表
                        self.script_ids = list(self.screenwriter.initial_script.keys())
                        new_scene_id = list(new_script.keys())[0]
                        print(f"\n成功生成新场景: {new_scene_id}")
                        
                        # 更新导演的剧本
                        self.director.load_script(self.screenwriter.initial_script)
                        
                        # 增加场景生成计数
                        self.new_scene_generation_count += 1
                        
                        # 检查并创建新角色
                        self.director.check_and_create_new_characters(self.script_ids, self.current_scene_index, self.player.name)
                        
                        # 开始新场景
                        self.start_scene()
                    else:
                        print(f"\n生成新场景失败: {new_script.get('error')}")
                        messagebox.showinfo("游戏结束", "故事已结束，感谢您的参与！")
            else:
                # 还有场景，开始新场景
                self.start_scene()

    def exit_game(self):
        if messagebox.askyesno("确认退出", "确定要退出游戏吗？"):
            # 导出对话历史到JSON文件供测评使用
            import json
            import datetime

            # 创建带有时间戳的文件名
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            dialogue_history_file = f"dialogue_history_{timestamp}.json"

            # 将对话历史转换为可序列化格式
            dialogue_history = self.screenwriter.get_all_dialogue_history() # 获取全部对话历史

            # 保存对话历史到文件
            with open(dialogue_history_file, "w", encoding="utf-8") as f:
                json.dump(dialogue_history, f, ensure_ascii=False, indent=2)

            print(f"\n对话历史已导出到文件: {dialogue_history_file}")
            
            # 退出应用
            self.root.destroy()

    def handle_scene_timeout(self):
        # 获取当前场景ID
        current_scene_id = self.script_ids[self.current_scene_index]
        
        # 获取下一个场景ID（如果有）
        next_scene = self.script_ids[self.current_scene_index + 1] if self.current_scene_index + 1 < len(self.script_ids) else None
        
        # 判断是否需要插入新场景
        should_generate = self.director.should_generate_new_script(self.screenwriter, current_scene_id, next_scene)
        
        if should_generate and self.inserted_scene_count < self.settings['max_inserted_scenes']:
            print(f"\n==== 根据情节发展，需要插入新场景 ({self.inserted_scene_count+1}/{self.settings['max_inserted_scenes']}) ====")
            
            # 生成新场景
            new_script = self.screenwriter.generate_new_script(current_scene_id, 
                                                           dialogue_history=self.screenwriter.get_dialogue_history())
            
            if "error" not in new_script:
                # 更新导演的剧本
                self.director.load_script(self.screenwriter.initial_script)
                
                # 重新获取并排序剧本ID列表
                self.script_ids = list(self.screenwriter.initial_script.keys())
                new_scene_id = list(new_script.keys())[0]
                print(f"\n成功生成新场景: {new_scene_id}")
                
                # 设置新场景为下一个场景
                next_scene = new_scene_id
                
                # 增加插入场景计数
                self.inserted_scene_count += 1
                
                # 检查并创建新角色
                self.director.check_and_create_new_characters(self.script_ids, 
                                                           self.script_ids.index(new_scene_id), 
                                                           self.player.name)
            else:
                print(f"\n生成新场景失败: {new_script.get('error')}")
        elif should_generate and self.inserted_scene_count >= self.settings['max_inserted_scenes']:
            print(f"\n==== 已达到插入新场景次数限制({self.settings['max_inserted_scenes']}次)，继续使用原有剧本 ====")
        
        # 生成场景结束描述
        ending_description = self.screenwriter.end_scene(self.last_interaction or "场景尝试次数已用完", 
                                                      self.director, current_scene_id, next_scene)
        
        # 显示场景结束描述
        self.scene_text.insert(tk.END, f"\n\n{ending_description}")
        self.scene_text.see(tk.END)
        print(f"\n{ending_description}")
        
        # 添加场景转场描述
        self.screenwriter.add_dialogue_record("旁白", "场景转场", 
                                           f"从{current_scene_id}场景转场到{next_scene if next_scene else '故事结束'}")
        
        # 设置场景为已结束
        self.scene_finished = True
        
        # 提示用户进入下一场景
        messagebox.showinfo("场景结束", "场景尝试次数已用完，请点击'下一场景'按钮继续")

    # 添加回车键处理函数
    def on_enter_pressed(self, event):
        # 阻止回车键在文本框中插入换行符
        self.submit_input()
        return "break"  # 阻止默认行为

if __name__ == "__main__":
    root = tk.Tk()
    app = CoCGameGUI(root)
    root.mainloop() 
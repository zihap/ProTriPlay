# API密钥配置
openai_api_key = "<Your-API-KEY>"
deepseek_api_key = "<Your-API-KEY>"
qwen_api_key = "<Your-API-KEY>"

# 代理配置
http_proxy = "http://localhost:7890"
https_proxy = "http://localhost:7890"

# 模型配置
use_model = "deepseek-chat"  # 可选: "gpt-4o-mini", "deepseek-chat", "qwen3-235b-a22b" 

# 测试配置
try_chance = 2  # 场景循环次数/尝试机会
max_new_scene_generations = 1  # 最大允许生成的新场景数 
max_inserted_scenes = 1  # 最大允许插入的新场景数
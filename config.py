import os
from dotenv import load_dotenv

# 加载环境变量配置文件
# 从项目根目录的.env文件中读取敏感配置信息，避免硬编码密钥
load_dotenv()

# ==================== 火山方舟 API 配置 ====================
# 火山方舟API密钥，用于认证访问火山方舟平台
# 默认值为占位符，实际使用时需在.env文件中配置ARK_API_KEY
ark_api_key = os.getenv("ARK_API_KEY", "<Your-Ark-API-KEY>")

# 火山方舟API基础URL，指定API请求的端点地址
# 默认使用北京区域的火山方舟API地址
ark_base_url = os.getenv("API_URL", "https://ark.cn-beijing.volces.com/api/plan/v3")

# 火山方舟对话模型名称，使用deepseek-v4-pro进行对话生成
ark_model = "deepseek-v4-pro"

# 火山方舟向量模型名称，使用doubao-embedding-vision进行文本向量化
ark_embedding_model = "doubao-embedding-vision"

# 火山方舟向量模型的维度，doubao-embedding-vision返回2048维向量
ark_embedding_dim = 2048

# ==================== 其他模型 API 配置 ====================
# OpenAI API密钥（备用）
openai_api_key = "<Your-API-KEY>"

# Deepseek API密钥（备用）
deepseek_api_key = "<Your-API-KEY>"

# Qwen API密钥（备用）
qwen_api_key = "<Your-API-KEY>"

# ==================== 网络代理配置 ====================
# HTTP代理地址，为空字符串表示不使用代理
http_proxy = ""

# HTTPS代理地址，为空字符串表示不使用代理
https_proxy = ""

# ==================== 运行时配置 ====================
# 当前使用的模型类型，可选值："ark"、"gpt-4o-mini"、"deepseek-chat"、"qwen3-235b-a22b"
# 默认为"ark"，即使用火山方舟模型
use_model = "ark"

# 场景循环尝试次数，每个场景允许玩家进行多少次交互
try_chance = 7

# 最大允许生成的新场景数（所有场景完成后）
max_new_scene_generations = 1

# 最大允许插入的新场景数（场景间插入）
max_inserted_scenes = 1
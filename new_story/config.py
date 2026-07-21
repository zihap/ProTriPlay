import os
from dotenv import load_dotenv

load_dotenv()

ark_api_key = os.getenv("ARK_API_KEY", "<Your-Ark-API-KEY>")
ark_base_url = os.getenv("ARK_API_URL", "https://ark.cn-beijing.volces.com/api/plan/v3")
ark_model = "deepseek-v4-pro"
ark_embedding_model = "doubao-embedding-vision"
ark_embedding_dim = 2048

openai_api_key = os.getenv("OPENAI_API_KEY", "<Your-OpenAI-API-KEY>")
openai_base_url = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1")
openai_model = ""
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "<Your-DeepSeek-API-KEY>")
deepseek_base_url = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1")
deepseek_model = ""
qwen_api_key = os.getenv("QWEN_API_KEY", "<Your-Qwen-API-KEY>")
qwen_base_url = os.getenv("QWEN_API_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
qwen_model = ""

http_proxy = os.getenv("HTTP_PROXY", "")
https_proxy = os.getenv("HTTPS_PROXY", "")

use_model = "ark"  # "ark" or "gpt" or "deepseek" or "qwen"

try_chance = 9
max_new_scene_generations = 7
max_inserted_scenes = 7

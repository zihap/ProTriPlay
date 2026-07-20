import os
from dotenv import load_dotenv

load_dotenv()

ark_api_key = os.getenv("ARK_API_KEY", "<Your-Ark-API-KEY>")
ark_base_url = os.getenv("API_URL", "https://ark.cn-beijing.volces.com/api/plan/v3")
ark_model = "deepseek-v4-pro"
ark_embedding_model = "doubao-embedding-vision"
ark_embedding_dim = 2048

openai_api_key = "<Your-API-KEY>"
deepseek_api_key = "<Your-API-KEY>"
qwen_api_key = "<Your-API-KEY>"

http_proxy = ""
https_proxy = ""

use_model = "ark"

try_chance = 2
max_new_scene_generations = 1
max_inserted_scenes = 1
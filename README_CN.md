该仓库包含了 ProTriPlay（A Trinity Framework for Professional Interactive Drama Based on LLMs）的一系列 Python 脚本，用于与多个 AI 模型进行交互、管理戏剧角色并在叙事结构中模拟对话。脚本集成了如 OpenAI、DeepSeek 和 Qwen 等 AI 功能，用于生成响应并保持角色的记忆。

## 文件概述

### 1. `config.py`
该配置文件包含连接不同 AI 服务所需的 API 密钥和模型设置。

#### 关键变量：
- **openai_api_key**：您的 OpenAI API 密钥。
- **deepseek_api_key**：您的 DeepSeek API 密钥。
- **qwen_api_key**：您的 Qwen API 密钥。
- **http_proxy, https_proxy**：代理配置。
- **use_model**：使用的 AI 模型（`deepseek-chat`、`gpt-4o-mini`、`qwen3-235b-a22b`）。
- **try_chance, max_new_scene_generations, max_inserted_scenes**：与测试和场景生成相关的配置。

### 2. `role.py`
该文件定义了演员互动、记忆管理和场景动态的核心逻辑。包括 `Actor`、`Player` 和 `Director` 类，用于管理角色状态和叙事。

#### 关键类：
- **Actor**：演员Agent，管理角色的属性、记忆、关系和性格特征。与 AI 模型互动生成对话和响应。
- **Player**：代表玩家角色，可以与 NPC（演员）进行互动。
- **Director**：导演Agent管理整个叙事，控制场景的流动以及角色之间的互动。
- **Screenwriter**：编剧Agent，负责剧本生成和基于预定义剧本创建对话。

### 3. `my_test.py`
该脚本模拟了一个具有记忆和特征管理的角色场景，其中 AI 驱动的玩家角色与 NPC 互动。

#### 主要功能：
- **角色管理**：定义了 `Actor` 和 `Player` 类，用于管理角色的属性、记忆和关系。
- **场景模拟**：使用 `Director` 类管理场景、角色和对话，模拟互动并推动故事情节的发展。
- **场景生成**：基于角色互动自动生成新的场景。

### 4. `gui_test.py`
该脚本提供了一个基于 Tkinter 的图形用户界面（GUI），使用户能够与 AI 系统进行交互，发送和接收消息。

#### 主要功能：
- **API 密钥管理**：允许配置 OpenAI、DeepSeek 和 Qwen 的 API 密钥。
- **代理配置**：允许设置 HTTP/HTTPS 代理。
- **模型选择**：用户可以选择不同的 AI 模型（DeepSeek、GPT-4、Qwen）。
- **记忆与响应处理**：支持处理流式和非流式 AI 响应。


## 依赖要求

- Python 3.10.0
- openai=1.37.1
- faiss=1.7.4

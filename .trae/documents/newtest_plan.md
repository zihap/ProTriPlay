# 多Agent合作叙事系统 - 实现计划

## 一、项目概述

本项目旨在从零开始创建一个多Agent合作叙事系统，包含四个核心模块：导演Agent、编剧Agent、角色Agent和主角Agent。系统采用模块化设计原则，确保各模块低耦合高内聚。

## 二、原项目分析结论

### 2.1 核心架构
| 模块 | 原项目类 | 职责 |
|------|---------|------|
| Actor | Actor | 角色数据模型、记忆管理、对话生成 |
| Director | Director | 剧本管理、场景切换、角色调度 |
| Player | Player | 玩家交互接口 |
| Screenwriter | Screenwriter | 剧本生成、场景描述、对话历史 |

### 2.2 技术栈
- Python 3.10+
- 火山方舟API (deepseek-v4-pro + doubao-embedding-vision)
- FAISS (向量检索)
- NumPy
- pytest (测试框架)

### 2.3 关键设计模式
- 配置集中化：所有API参数在config.py中统一管理
- 向量记忆系统：使用FAISS实现记忆的快速检索和去重
- 对话生成模式：结合记忆、关系、性格特征生成对话
- 动态场景生成：基于对话历史自动生成新场景

## 三、新项目设计方案

### 3.1 目录结构
```
newtest/
├── .env                    # 环境变量配置
├── pyproject.toml          # uv项目配置
├── config.py               # 集中化配置模块
├── api_client.py           # API客户端封装
├── memory/                 # 记忆系统模块
│   ├── __init__.py
│   ├── vector_store.py     # 向量存储与检索
│   └── memory_manager.py   # 记忆管理
├── agents/                 # Agent模块
│   ├── __init__.py
│   ├── base_agent.py       # 基础Agent类
│   ├── director_agent.py   # 导演Agent
│   ├── screenwriter_agent.py # 编剧Agent
│   ├── role_agent.py       # 角色Agent
│   └── protagonist_agent.py # 主角Agent
├── models/                 # 数据模型
│   ├── __init__.py
│   ├── story.py            # 剧情大纲模型
│   ├── scene.py            # 场景模型
│   └── character.py        # 角色模型
├── tests/                  # 测试模块
│   ├── __init__.py
│   ├── test_memory.py      # 记忆系统测试
│   ├── test_director.py    # 导演Agent测试
│   ├── test_screenwriter.py # 编剧Agent测试
│   ├── test_role.py        # 角色Agent测试
│   ├── test_protagonist.py # 主角Agent测试
│   └── test_integration.py # 集成测试
├── main.py                 # 主程序入口
└── story_output/           # 故事输出目录
```

### 3.2 核心模块设计

#### 3.2.1 导演Agent模块
**职责**：维护剧情大纲、控制场景切换、角色调度、剧情引导

**数据结构**：
- `StoryOutline`: 故事背景、主线节点、分支条件
- `SceneTransition`: 场景切换规则和条件
- `CharacterSchedule`: 角色出场配置

**核心方法**：
- `load_story_outline()`: 加载剧情大纲
- `switch_scene()`: 场景切换控制
- `schedule_characters()`: 角色调度
- `provide_guidance()`: 剧情引导提示

#### 3.2.2 编剧Agent模块
**职责**：场景布置生成、剧本生成、情境创建

**核心方法**：
- `generate_scene_setting()`: 场景环境描述生成
- `generate_script()`: 场景剧本生成
- `validate_script()`: 剧本内容验证

#### 3.2.3 角色Agent模块
**职责**：角色数据模型、行为决策、对话生成、记忆管理、关系机制

**数据结构**：
- `CharacterProfile`: 目标设定、性格特征、记忆存储
- `Relationship`: 角色间关系（动态变化）

**核心方法**：
- `make_decision()`: 行为决策系统
- `generate_dialogue()`: 对话生成
- `update_memory()`: 记忆更新
- `update_relationship()`: 关系动态调整

#### 3.2.4 主角Agent模块
**职责**：主角特殊属性、剧情观测、驱动系统、成长弧线

**核心方法**：
- `observe_scene()`: 剧情观测机制
- `drive_story()`: 主角驱动系统
- `update_growth()`: 成长弧线更新

## 四、实施步骤

### 步骤1：项目初始化
- 创建项目目录结构
- 使用uv初始化Python虚拟环境
- 安装依赖包（openai, faiss-cpu, numpy, pytest）

### 步骤2：配置模块
- 创建config.py，集中管理API配置
- 创建.env文件模板
- 创建api_client.py，封装API调用逻辑

### 步骤3：记忆系统模块
- 实现vector_store.py：FAISS向量索引封装
- 实现memory_manager.py：记忆添加、检索、去重逻辑

### 步骤4：数据模型模块
- 实现story.py：剧情大纲数据结构
- 实现scene.py：场景数据结构
- 实现character.py：角色数据模型（含关系机制）

### 步骤5：基础Agent类
- 实现base_agent.py：通用Agent基类

### 步骤6：导演Agent
- 实现director_agent.py：剧情大纲维护、场景切换、角色调度、剧情引导

### 步骤7：编剧Agent
- 实现screenwriter_agent.py：场景布置、剧本生成、情境创建

### 步骤8：角色Agent
- 实现role_agent.py：角色模型、行为决策、对话生成、记忆管理、关系机制

### 步骤9：主角Agent
- 实现protagonist_agent.py：特殊属性扩展、剧情观测、驱动系统、成长弧线

### 步骤10：测试模块
- 编写单元测试（各模块独立测试）
- 编写集成测试（模块间交互测试）
- 编写端到端测试（完整故事生成流程）

### 步骤11：主程序入口
- 实现main.py：完整故事生成工作流
- 实现故事输出功能

## 五、依赖与风险

### 5.1 依赖列表
| 依赖 | 版本 | 用途 |
|------|------|------|
| openai | >=1.0.0 | API客户端 |
| volcenginesdkarkruntime | - | 火山方舟SDK |
| faiss-cpu | >=1.7.4 | 向量检索 |
| numpy | >=1.24.0 | 数值计算 |
| pytest | >=7.0.0 | 测试框架 |
| python-dotenv | >=1.0.0 | 环境变量加载 |

### 5.2 风险处理
1. **API调用失败**：实现重试机制和降级方案（随机向量替代）
2. **对话循环**：实现记忆去重和对话内容检测
3. **场景生成失败**：实现最大重试次数限制
4. **性能问题**：限制记忆数量（最多20条），优化向量检索

## 六、测试计划

### 6.1 单元测试
- 记忆系统：添加、检索、去重功能
- 导演Agent：场景切换、角色调度逻辑
- 编剧Agent：场景生成、剧本验证
- 角色Agent：对话生成、关系管理
- 主角Agent：行为决策、成长弧线

### 6.2 集成测试
- 导演与编剧协作
- 角色与主角互动
- 场景切换流程
- 动态场景生成

### 6.3 端到端测试
- 完整故事生成流程（4场景）
- 对话历史保存与导出
- 角色关系动态变化验证

## 七、输出规范

故事输出文件格式：
- JSON格式，包含完整对话历史
- 字段：场景ID、场景描述、角色列表、对话记录、场景转场
- 编码：UTF-8
- 文件名：`story_output/story_YYYYMMDD_HHMMSS.json`

# lab04
## 概述

本项目完成了基于大语言模型的智能化需求建模实验，通过三种不同方式实现了对AI全景系统的需求建模：
基于纯Restful API的建模、基于OpenAI SDK的建模以及基于LLM Agent的MultiAgent Workflow自动化建模。
实验旨在掌握大语言模型辅助的建模方法，构建自动化需求建模流程。

## 实验内容

### 任务1：基于纯Restful API的智能化需求建模

#### Prompt设计

```plaintext
I would like to model an AI panoramic system. The users mainly include panorama image collectors, AI maintainers, administrators, and general users. The system should allow users to log in to the website to view panoramic images and ask questions to the AI. Panorama image collectors should be able to gather panoramic images and upload them to the website's database. AI maintainers are responsible for deploying interactive AI, and administrators manage other users.
```

#### 代码实现

```bash
# 原始curl命令（Windows需调整）
curl 'BASE_URL/chat/completions' \
-H "Content-Type: application/json" \
-H "Authorization: Bearer API_KEY" \
-d '{"model":"gpt-4o","messages":[{"role":"developer","content":"You are a helpful assistant."},{"role":"user","content":"I would like to model an AI panoramic system..."}]}'
```

#### 输出格式

期望输出包含需求模型的结构化描述，如用户角色、系统功能、数据流程等，格式为JSON。

### 基于OpenAI SDK的智能化需求建模

#### Prompt设计

与任务1相同，使用相同的用户需求描述。

#### 代码实现（2.py）

```python
import requests

API_URL = "https://api.chatfire.cn/v1/chat/completions"
API_KEY = "sk-zO8exlBicZh7nJeZn5GuC5X9SPuVrZzXoGyOW0i9BFvN62ON"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}
data = {
    "model": "gpt-4o",
    "messages": [
        {"role": "developer", "content": "You are a helpful assistant."},
        {"role": "user", "content": "I would like to model an AI panoramic system..."}
    ]
}
response = requests.post(API_URL, json=data, headers=headers)
print(response.json())
```

#### 输出格式

同任务1，使用JSON格式返回需求模型的结构化描述。

### 基于LLM Agent的智能化需求建模

#### MultiAgent Workflow设计

本任务采用双Agent协同工作流：
1. **UML模型生成器（uml_model_generator）**：根据用户输入生成UML模型
2. **评估器（evaluator）**：对生成的模型进行评估并提供反馈

工作流流程：
- 生成器生成初始UML模型
- 评估器对模型评分并提供改进建议
- 生成器根据反馈优化模型，循环直至评估通过

#### Agent定义

**生成器Agent**：
- 名称：uml_model_generator
- 模型：gpt-4o
- 指令："Generate a UML model based on the user's input. Include use case diagram, sequence diagrams, and class diagrams."
- 输出格式：UMLModelOutputSchema

**评估器Agent**：
- 名称：evaluator
- 模型：gpt-4o
- 指令："You evaluate a UML model and decide if it's good enough. If it's not good enough, you provide feedback on what needs to be improved. Never give it a pass on the first try."
- 输出格式：EvaluationFeedback

#### 输出格式DSL

定义UML模型的完整数据结构，包含以下组件：

```python
@dataclass
class UMLModelOutput:
    user_stories: UserStories                # 用户故事
    use_case_descriptions: UseCaseDescriptions  # 用例描述
    use_case_diagram: UseCaseDiagram          # 用例图
    sequence_diagrams: SequenceDiagrams      # 序列图
    class_diagrams: ClassDiagrams            # 类图
```

**用户故事结构**：
```python
@dataclass
class UserStory:
    id: str
    description: str
```

**用例描述结构**：
```python
@dataclass
class UseCaseDescription:
    num: str
    description: str
    basic_flow: List[BasicFlowStep]
```

**类图结构**：
```python
@dataclass
class ClassInfo:
    name: str
    attributes: List[Attribute]
    methods: List[Method]
    associations: List[Association]
```

#### 代码实现（3.py）

```python
import asyncio
from dataclasses import dataclass, asdict, fields
from typing import List, Dict, Literal, Optional, Any, Type, get_type_hints
from agents import Agent, Runner, OpenAIProvider, RunConfig, trace, RunResult
from openai import AsyncOpenAI
import json

# 定义UML模型相关数据类（完整代码见3.py）

# 输出格式验证类
class UMLModelGeneratorOutputSchema:
    # 实现JSON模式验证和数据转换（完整代码见3.py）

# 配置OpenAIProvider
provider = OpenAIProvider(
    openai_client=AsyncOpenAI(
        base_url="https://api.chatfire.cn/v1",
        api_key="sk-zO8exlBicZh7nJeZn5GuC5X9SPuVrZzXoGyOW0i9BFvN62ON",
    ),
    use_responses=False,
)

async def main() -> None:
    # 用户需求输入
    msg = ("我想要对一个AI全景系统建模，用户主要有全景图片采集者、AI维护者、管理者和用户，功能是用户登录网站查看全景图片并且可以向AI提问,"
           "全景图片采集者可以采集全景图片并放入网站数据，AI维护者部署可交互AI，管理者管理其他用户")
    input_items = [{"content": msg, "role": "user"}]
    uml_model = None
    
    # 循环生成-评估-优化流程
    with trace("LLM as a judge"):
        while True:
            # 生成UML模型
            uml_model_generator = Agent(
                name="uml_model_generator",
                model="gpt-4o",
                instructions="Generate a UML model based on the user's input...",
                output_type=UMLModelGeneratorOutputSchema
            )
            uml_model_result = await Runner.run(
                uml_model_generator,
                input_items,
                run_config=RunConfig(model_provider=provider)
            )
            
            # 评估模型
            evaluator = Agent(
                name="evaluator",
                model="gpt-4o",
                instructions="You evaluate a UML model and decide if it's good enough...",
                output_type=EvaluationFeedback
            )
            evaluator_result = await Runner.run(
                evaluator, input_items, run_config=RunConfig(model_provider=provider)
            )
            result = evaluator_result.final_output
            
            # 检查评估结果，不通过则添加反馈继续优化
            if result.score == "pass":
                break
            input_items.append({"content": f"Feedback: {result.feedback}", "role": "user"})
    
    # 输出最终模型（完整代码见3.py）

if __name__ == "__main__":
    asyncio.run(main())
```

### 配置说明

- 在代码中替换`API_URL`和`API_KEY`为实验提供的参数：
  - BASE_URL: https://api.chatfire.cn/v1
  - API_KEY: sk-zO8exlBicZh7nJeZn5GuC5X9SPuVrZzXoGyOW0i9BFvN62ON

## 生成的需求模型说明（为了减少迭代优化次数，只说明简单需求，详细需求参考lab01的设计）

### 用户角色定义
- **全景图片采集者**：负责采集全景图片并上传至数据库
- **AI维护者**：部署和维护交互式AI系统
- **管理员**：管理系统用户账号和权限
- **普通用户**：登录系统查看全景图片并向AI提问

### 系统功能模块
1. **用户认证模块**：处理用户登录和权限管理
2. **全景图片管理模块**：存储、检索和展示全景图片
3. **AI交互模块**：提供用户与AI的问答接口
4. **系统管理模块**：支持管理员对系统和用户的管理

### UML模型组件
- **用例图**：包含用户登录、查看图片、提问AI、上传图片等用例
- **序列图**：展示用户与系统交互的流程，如登录流程、图片上传流程
- **类图**：定义系统核心类，如User、PanoramaImage、AIAssistant等，包含属性和方法定义

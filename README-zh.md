# LangGraph 101

欢迎来到 LangGraph 101！

## 简介
本仓库包含用于学习 LangChain、LangGraph 和 Deep Agents 的动手教程，分为两个学习轨道：

- **101**：使用 LangChain 和 LangGraph 构建 Agent 的基础知识
- **201**：高级模式，包括多 Agent 系统、Deep Agents 和生产级工作流

要了解这些框架之间的关系，请参阅 [LangChain vs LangGraph vs Deep Agents](https://docs.langchain.com/oss/python/concepts/products)。

这是 LangChain Academy 的浓缩版本，旨在与 LangChain 工程师一起在会话中运行。如果您有兴趣深入学习，或自行完成教程，请查看 [LangChain Academy](https://academy.langchain.com/courses/intro-to-langgraph)！LangChain Academy 提供了我们 LangChain 工程师录制的实用视频。

## 内容概览

### 101 - 基础知识（`notebooks/101/`）
- **101_langchain_langgraph.ipynb**：使用模型、工具、记忆和流式传输构建您的第一个 Agent
- **102_middleware.ipynb**：中间件、人机协作模式和安全护栏

### 201 - 生产级模式（`notebooks/201/`）
- **email_agent.ipynb**：构建有状态的电子邮件分类与回复 Agent
- **multi_agent.ipynb**：包含监督者和专门子 Agent 的多 Agent 系统
- **research_agent.ipynb**：带有并行子研究员的深度研究 Agent
- **deepagents.ipynb**：使用 DeepAgents 从头构建研究 Agent——涵盖 AGENTS.md、技能、后端、长期记忆、HITL 等

`research_agent.ipynb`、`deepagents.ipynb` 以及 `agents/researcher/`、`agents/deep_agent/` 的实现都需要在 `.env` 中配置 `TAVILY_API_KEY` 才能进行网页搜索。

### Agents（`agents/`）
通过 `langgraph dev` 在 LangGraph Studio 中运行的独立 Agent 实现：
- **`agents/101/`** - 来自 101 笔记本的简单天气 Agent
- **`agents/email_agent/`** - 电子邮件分类 Agent
- **`agents/music_store/`** - 多 Agent 音乐商店（监督者 + 子 Agent）
- **`agents/researcher/`** - 带有并行子研究员的深度研究 Agent
- **`agents/deep_agent/`** - DeepAgents 研究 Agent，包含 AGENTS.md、技能（LinkedIn 发帖、Twitter/X 发帖）、长期记忆和 HITL

所有笔记本均使用最新的 **LangChain**、**LangGraph** 和 **DeepAgents** 原语，包括 `create_agent()`、`create_deep_agent()`、中间件和中断模式。

## 项目结构

```
langgraph-101/
├── notebooks/
│   ├── 101/                          # 基础知识
│   │   ├── 101_langchain_langgraph.ipynb
│   │   └── 102_middleware.ipynb
│   └── 201/                          # 生产级模式
│       └── deep_agents.ipynb
│       ├── email_agent.ipynb
│       ├── multi_agent.ipynb
│       ├── research_agent.ipynb

├── agents/                           # LangGraph Studio 的独立 Agent
│   ├── 101/agent.py
│   ├── email_agent/graph.py
│   ├── music_store/                  # 多 Agent 监督者 + 子 Agent
│   ├── researcher/                   # 深度研究 Agent
│   └── deep_agent/                   # DeepAgents 研究 Agent
│       ├── agent.py                  # Agent 定义
│       ├── AGENTS.md                 # Agent 身份与指令
│       └── skills/                   # 按需能力
│           ├── linkedin-post/SKILL.md
│           └── twitter-post/SKILL.md
├── utils/
│   ├── models.py                     # 集中式模型配置
│   └── utils.py                      # 共享工具函数
├── langgraph.json                    # langgraph dev 的 Agent 注册表
└── .env                              # API 密钥（不提交）
```

## 背景

在 LangChain，我们的目标是让构建 LLM 应用程序变得简单。您可以构建的一种 LLM 应用程序是 Agent。围绕构建 Agent 有很多兴奋点，因为它们可以自动化以前无法完成的广泛任务。

然而在实践中，构建能够可靠执行这些任务的系统极其困难。在与用户合作将 Agent 投入生产的过程中，我们了解到通常需要更多的控制。您可能需要 Agent 始终先调用特定工具，或根据其状态使用不同的提示词。

为了解决这个问题，我们构建了 [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview)——一个用于构建 Agent 和多 Agent 应用程序的框架。与 LangChain 包分离，LangGraph 的核心设计理念是帮助开发者为 Agent 工作流添加更高的精准度和控制力，以适应现实世界系统的复杂性。

对于需要规划、文件系统访问和任务委派的复杂多步骤任务，我们构建了 [Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview)——一个构建在 LangGraph 之上的 Agent 工具包，开箱即用地提供内置工具、上下文管理和技能。

## 准备工作

### 克隆 LangGraph 101 仓库
```
git clone https://github.com/langchain-ai/langgraph-101.git
```


### 创建环境
确保您安装了最新版本的 pip 和 python
```
$ cd langgraph-101
# 将 .env.example 文件复制为 .env
cp .env.example .env
```
请在 `.env` 中填入你要使用的 API 密钥。特别是研究类 Agent 需要 `TAVILY_API_KEY` 才能执行网页搜索。

如果您在设置 Python 环境或获取必要的 API 密钥时遇到问题（例如公司政策限制），请联系您的 LangChain 代表，我们会找到解决方案！

### 安装包
确保您安装了最新版本的 pip 和 python
```
# 如果尚未安装 uv，请先安装
pip install uv

# 安装包，允许预发布版本
uv sync

# 激活虚拟环境
source .venv/bin/activate
```

### 本地运行 Agent

您可以使用 `langgraph dev` 在本地运行本仓库中的 Agent。这将为您提供：
- 用于 Agent 的本地 API 服务器
- 用于测试和调试的 LangGraph Studio UI
- 开发过程中的热重载

```bash
# 从根目录启动 LangGraph 开发服务器
langgraph dev

# 这将启动一个本地服务器并提供：
# - Agent 的 API 端点（通常为 http://localhost:2024）
# - LangGraph Studio UI 链接
```

`langgraph.json` 配置文件定义了哪些 Agent 可用。您可以通过 API 或 LangGraph Studio 的可视化界面与 Agent 交互。

有关更多详细信息，请参阅 [LangGraph CLI 文档](https://docs.langchain.com/langsmith/cli#langgraph-cli)。

### 模型配置

本仓库使用**集中式 utils 模块**（`utils/`）来避免代码重复。所有模型配置和共享工具函数均在此定义：
- **`utils/models.py`** - LLM 模型初始化（OpenAI、Anthropic、Azure、Bedrock、Vertex AI）
- **`utils/utils.py`** - 共享工具函数（`show_graph`、`get_engine_for_chinook_db`）

**默认**：使用 `o3-mini` 模型的 OpenAI。要切换提供商，请按以下说明编辑 `utils/models.py`。

**注意**：笔记本会自动将项目根目录添加到 Python 路径中，因此无论它们位于哪个子目录，都可以从 `utils` 导入。

### Azure OpenAI 说明

如果您使用 Azure OpenAI 而非 OpenAI，请按以下步骤操作：

1. **在 `.env` 文件中设置环境变量**：
   ```
   AZURE_OPENAI_API_KEY=your_api_key
   AZURE_OPENAI_ENDPOINT=your_endpoint
   AZURE_OPENAI_API_VERSION=2024-03-01-preview
   ```

2. **更新 `utils/models.py`**：
   - 注释掉“默认模型”部分（第 20-28 行）
   - 取消注释“AZURE OpenAI 版本”部分（第 31-57 行）
   - 将 `azure_deployment` 名称配置为与您的部署匹配

3. **完成！** 所有 Agent 和笔记本将自动使用 Azure 模型

### AWS Bedrock 说明

如果您使用 AWS Bedrock 而非 OpenAI，请按以下步骤操作：

1. **在 `.env` 文件中设置环境变量**：
   ```
   AWS_ACCESS_KEY_ID=your_access_key
   AWS_SECRET_ACCESS_KEY=your_secret_key
   AWS_REGION_NAME=us-east-1
   AWS_MODEL_ARN=your_model_arn
   ```

2. **更新 `utils/models.py`**：
   - 注释掉“默认模型”部分（第 20-28 行）
   - 取消注释“Bedrock 版本”部分（第 60-78 行）
   - 根据需要配置模型设置

3. **完成！** 所有 Agent 和笔记本将自动使用 Bedrock 模型

### Google Vertex AI 说明

如果您使用 Google Vertex AI 而非 OpenAI，请按以下步骤操作：

1. **设置 Google Cloud 凭证**
   - 在您的 Google Cloud 项目中创建一个具有 Vertex AI 权限的服务账号
   - 下载服务账号 JSON 密钥文件
   - 将其保存为项目根目录下的 `vertexCred.json`

2. **在 `.env` 文件中配置环境变量**：
   ```
   GOOGLE_APPLICATION_CREDENTIALS=./vertexCred.json
   ```

3. **更新 `utils/models.py`**：
   - 注释掉“默认模型”部分（第 20-28 行）
   - 取消注释“Google Vertex AI 版本”部分（第 81-100 行）
   - 设置程序使用 `Path(__file__)` 自动处理凭证路径

4. **完成！** 所有 Agent 和笔记本将自动使用 Vertex AI 模型

**注意：** 请确保将 `vertexCred.json` 添加到您的 `.gitignore` 中，以避免提交凭证。

## 入门指南

### 推荐学习路径

1. **从 101 开始** - `notebooks/101/`
   - 从 `101_langchain_langgraph.ipynb` 开始，学习 LangChain + LangGraph 基础知识
   - 继续学习 `102_middleware.ipynb`，了解中间件和人机协作模式

2. **进阶到 201** - `notebooks/201/`
   - 通过 `email_agent.ipynb` 探索完整的有状态 Agent 示例
   - 使用 `multi_agent.ipynb` 构建多 Agent 系统
   - 通过 `deepagents.ipynb` 学习 Deep Agents——逐步构建一个包含 AGENTS.md、技能、后端、记忆和 HITL 的研究 Agent

3. **在 Studio 中运行 Agent**
   - 使用 `langgraph dev` 在 LangGraph Studio 中启动所有 Agent
   - 尝试 Deep Agent（`agents/deep_agent/`）——让它研究一个主题并撰写 LinkedIn 帖子

### 资源

- **[LangChain 文档](https://docs.langchain.com/oss/python/langchain/overview)** - 完整的 LangChain 参考
- **[LangGraph 文档](https://docs.langchain.com/oss/python/langgraph/overview)** - LangGraph 指南和 API 参考
- **[Deep Agents 文档](https://docs.langchain.com/oss/python/deepagents/)** - Deep Agents 工具包参考
- **[LangChain vs LangGraph vs Deep Agents](https://docs.langchain.com/oss/python/concepts/products)** - 框架之间的关系
- **[LangChain Academy](https://academy.langchain.com/)** - 包含视频教程的免费课程
- **[LangSmith](https://smith.langchain.com)** - LLM 应用程序的调试和监控

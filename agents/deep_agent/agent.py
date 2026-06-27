"""用于 LangGraph Studio 的 Deep Agent。

这是一个基于 DeepAgents 构建的研究 Agent，用于演示：
- 使用 AGENTS.md 定义 Agent 身份和指令（替代硬编码 system_prompt）
- 使用 Skills 提供按需能力（LinkedIn 帖子、Twitter/X 帖子）
- 自定义工具（Tavily 搜索 + 战略思考）
- 用研究子 Agent 承接委派任务
- 通过 CompositeBackend 实现长期记忆（/memories/ -> StoreBackend）
- 文件写入时使用人在回路

通过 `langgraph dev` 运行时，平台会自动提供 store 和 checkpointer。
"""

import os
from datetime import datetime

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StoreBackend
from langchain_core.tools import tool
from tavily import TavilyClient

from utils.models import model

AGENT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- 工具 ---

tavily_client = TavilyClient()


@tool(parse_docstring=True)
def tavily_search(query: str) -> str:
    """针对给定查询搜索网页信息。

    参数：
        query: 要执行的搜索查询
    """
    search_results = tavily_client.search(query, max_results=3, topic="general")

    result_texts = []
    for result in search_results.get("results", []):
        url = result["url"]
        title = result["title"]
        content = result.get("content", "没有可用内容")
        result_text = f"## {title}\n**URL:** {url}\n\n{content}\n\n---\n"
        result_texts.append(result_text)

    return f"为“{query}”找到 {len(result_texts)} 条结果：\n\n{''.join(result_texts)}"

# --- 研究子 Agent ---

current_date = datetime.now().strftime("%Y-%m-%d")

RESEARCHER_INSTRUCTIONS = f"""你是一位正在执行研究任务的研究助手。今天的日期是 {current_date}。

<Task>
使用工具收集研究主题相关信息。
</Task>

<Hard Limits>
- 简单查询：最多使用 2-3 次搜索工具调用
- 复杂查询：最多使用 5 次搜索工具调用
- 每次搜索后，使用 think_tool 反思发现
</Hard Limits>

<Output Format>
请按以下结构组织发现：
- 清晰标题
- 行内引用 [1]、[2]、[3]
- 末尾包含 Sources 部分
</Output Format>
"""

research_subagent = {
    "name": "research-agent",
    "description": "委派研究任务。一次只给一个主题。",
    "system_prompt": RESEARCHER_INSTRUCTIONS,
    "tools": [tavily_search],
}


# --- 后端 ---

# FilesystemBackend 用于磁盘访问（skills、AGENTS.md），/memories/ 路由到 StoreBackend。
composite_backend = CompositeBackend(
    default=FilesystemBackend(root_dir=AGENT_DIR, virtual_mode=True),
    routes={
        # 记忆会存储在 langgraph store 中，可在 studio 点击 "memory" 按钮查看。
        "/memories/": StoreBackend(),
    },
)


# --- Agent ---

agent = create_deep_agent(
    model=model,
    tools=[tavily_search],
    system_prompt="你是一位专家级研究助手。",
    memory=["./AGENTS.md"],
    skills=["./skills/"],
    subagents=[research_subagent],
    backend=composite_backend,
    interrupt_on={
        "write_file": True,
        "edit_file": True,
    },
)

# 在 studio 的 interrupt 输入框中输入 {"decisions": [{"type": "approve"}]} 可批准操作。

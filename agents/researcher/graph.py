"""用于教学目的的简化版 Deep Research Agent 实现。

这是研究 Agent 的精简版本，使用硬编码配置，便于在教学场景中理解和使用。
"""

import asyncio
import os
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from agents.researcher.prompts import (
    clarify_with_user_instructions,
    compress_research_simple_human_message,
    compress_research_system_prompt,
    final_report_generation_prompt,
    lead_researcher_prompt,
    research_system_prompt,
    transform_messages_into_research_topic_prompt,
)
from agents.researcher.models import (
    AgentInputState,
    AgentState,
    ClarifyWithUser,
    ConductResearch,
    ResearchComplete,
    ResearcherOutputState,
    ResearcherState,
    ResearchQuestion,
    SupervisorState,
)
from agents.researcher.utils import (
    get_all_tools,
    get_notes_from_tool_calls,
    get_today_str,
    openai_websearch_called,
    think_tool,
)

from dotenv import load_dotenv

load_dotenv("../../.env")

# ===== 硬编码配置 =====
# 为了让教学示例更简单，这些值采用硬编码
RESEARCH_MODEL = "openai:gpt-4.1-mini"
MAX_RESEARCHER_ITERATIONS = 3  # 研究主管迭代次数
MAX_REACT_TOOL_CALLS = 10  # 每个研究员的最大工具调用次数
MAX_CONCURRENT_RESEARCH_UNITS = 5  # 最大并行研究单元数
MAX_STRUCTURED_OUTPUT_RETRIES = 3  # 结构化输出重试次数
MAX_OUTPUT_TOKENS = 10000  # 模型输出最大 token 数


def get_model():
    """获取或创建模型实例。"""
    return init_chat_model(
        model=RESEARCH_MODEL,
        max_tokens=MAX_OUTPUT_TOKENS,
        api_key=os.getenv("OPENAI_API_KEY"),
        use_responses_api=True
    )


async def clarify_with_user(state: AgentState, config: RunnableConfig):
    """必要时使用人在回路向用户提出澄清问题。"""

    messages = state["messages"]

    # 配置用于结构化澄清分析的模型
    clarification_model = (
        get_model()
        .with_structured_output(ClarifyWithUser)
        .with_retry(stop_after_attempt=MAX_STRUCTURED_OUTPUT_RETRIES)
    )

    # 分析是否需要澄清
    prompt_content = clarify_with_user_instructions.format(
        messages=get_buffer_string(messages),
        date=get_today_str()
    )
    response = await clarification_model.ainvoke([HumanMessage(content=prompt_content)])
    # 如果需要澄清，使用 interrupt 暂停并等待用户输入
    if response.need_clarification:
        return {"messages": [AIMessage(content=response.question)], "need_elaboration": True}
    else:
        # 不需要澄清
        return {"messages": [AIMessage(content=response.verification)], "need_elaboration": False}


async def human_input(state: AgentState, config):
    ai_question = state["messages"][-1].content
    user_response = interrupt(ai_question)
    return {"messages": [HumanMessage(content=user_response)], "need_elaboration": False}


def human_feedback_needed(state: AgentState):
    need_elaboration = state.get("need_elaboration", False)
    if need_elaboration:
        return "human_input"
    else:
        return "write_research_brief"



async def write_research_brief(state: AgentState, config: RunnableConfig) -> Command[Literal["research_supervisor"]]:
    """将用户消息转换为结构化研究简报，并初始化主管。"""

    # 配置用于生成结构化研究问题的模型
    research_model = (
        get_model()
        .with_structured_output(ResearchQuestion)
        .with_retry(stop_after_attempt=MAX_STRUCTURED_OUTPUT_RETRIES)
    )

    # 从用户消息生成结构化研究简报
    prompt_content = transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(state.get("messages", [])),
        date=get_today_str()
    )
    response = await research_model.ainvoke([HumanMessage(content=prompt_content)])

    # 使用研究简报和指令初始化主管
    supervisor_system_prompt = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=MAX_CONCURRENT_RESEARCH_UNITS,
        max_researcher_iterations=MAX_RESEARCHER_ITERATIONS
    )

    return Command(
        goto="research_supervisor",
        update={
            "research_brief": response.research_brief,
            "supervisor_messages": {
                "type": "override",
                "value": [
                    SystemMessage(content=supervisor_system_prompt),
                    HumanMessage(content=response.research_brief)
                ]
            }
        }
    )


async def supervisor(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor_tools"]]:
    """研究主管，负责规划研究策略并委派给研究员。"""

    # 可用工具：研究委派、完成信号和战略思考
    lead_researcher_tools = [ConductResearch, ResearchComplete, think_tool]

    # 配置带工具和重试逻辑的模型
    research_model = (
        get_model()
        .bind_tools(lead_researcher_tools)
        .with_retry(stop_after_attempt=MAX_STRUCTURED_OUTPUT_RETRIES)
    )

    # 基于当前上下文生成主管回复
    supervisor_messages = state.get("supervisor_messages", [])
    response = await research_model.ainvoke(supervisor_messages)

    # 更新状态并进入工具执行
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1
        }
    )


async def supervisor_tools(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor", "__end__"]]:
    """执行主管调用的工具。"""

    # 提取当前状态并检查退出条件
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]

    # 定义研究阶段退出条件
    exceeded_allowed_iterations = research_iterations > MAX_RESEARCHER_ITERATIONS
    no_tool_calls = not most_recent_message.tool_calls
    research_complete_tool_call = any(
        tool_call["name"] == "ResearchComplete"
        for tool_call in most_recent_message.tool_calls
    )

    # 如果满足任何终止条件，则退出
    if exceeded_allowed_iterations or no_tool_calls or research_complete_tool_call:
        return Command(
            goto=END,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", "")
            }
        )

    # 一起处理所有工具调用（think_tool 和 ConductResearch）
    all_tool_messages = []
    update_payload = {"supervisor_messages": []}

    # 处理 think_tool 调用（战略反思）
    think_tool_calls = [
        tool_call for tool_call in most_recent_message.tool_calls
        if tool_call["name"] == "think_tool"
    ]

    for tool_call in think_tool_calls:
        reflection_content = tool_call["args"]["reflection"]
        all_tool_messages.append(ToolMessage(
            content=f"已记录反思：{reflection_content}",
            name="think_tool",
            tool_call_id=tool_call["id"]
        ))

    # 处理 ConductResearch 调用（研究委派）
    conduct_research_calls = [
        tool_call for tool_call in most_recent_message.tool_calls
        if tool_call["name"] == "ConductResearch"
    ]

    if conduct_research_calls:
        try:
            # 限制并发研究单元数
            allowed_conduct_research_calls = conduct_research_calls[:MAX_CONCURRENT_RESEARCH_UNITS]
            overflow_conduct_research_calls = conduct_research_calls[MAX_CONCURRENT_RESEARCH_UNITS:]

            # 并行执行研究任务
            research_tasks = [
                researcher_subgraph.ainvoke({
                    "researcher_messages": [
                        HumanMessage(content=tool_call["args"]["research_topic"])
                    ],
                    "research_topic": tool_call["args"]["research_topic"]
                }, config)
                for tool_call in allowed_conduct_research_calls
            ]

            tool_results = await asyncio.gather(*research_tasks)

            # 用研究结果创建工具消息
            for observation, tool_call in zip(tool_results, allowed_conduct_research_calls):
                all_tool_messages.append(ToolMessage(
                    content=observation.get("compressed_research", "综合研究报告时出错"),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                ))

            # 用错误消息处理超出并发限制的研究调用
            for overflow_call in overflow_conduct_research_calls:
                all_tool_messages.append(ToolMessage(
                    content=f"错误：超过最大并发研究单元数（{MAX_CONCURRENT_RESEARCH_UNITS}）",
                    name="ConductResearch",
                    tool_call_id=overflow_call["id"]
                ))

            # 汇总所有研究结果中的原始笔记
            raw_notes_concat = "\n".join([
                "\n".join(observation.get("raw_notes", []))
                for observation in tool_results
            ])

            if raw_notes_concat:
                update_payload["raw_notes"] = [raw_notes_concat]

        except Exception as e:
            # 发生错误时结束研究阶段
            return Command(
                goto=END,
                update={
                    "notes": get_notes_from_tool_calls(supervisor_messages),
                    "research_brief": state.get("research_brief", "")
                }
            )

    # 返回包含所有工具结果的命令
    update_payload["supervisor_messages"] = all_tool_messages
    return Command(
        goto="supervisor",
        update=update_payload
    )


# 构建主管子图
supervisor_builder = StateGraph(SupervisorState)
supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)
supervisor_builder.add_edge(START, "supervisor")
supervisor_subgraph = supervisor_builder.compile()


async def researcher(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher_tools"]]:
    """针对具体主题开展聚焦研究的单个研究员。"""

    researcher_messages = state.get("researcher_messages", [])

    # 获取所有可用研究工具
    tools = await get_all_tools()
    if len(tools) == 0:
        raise ValueError("未找到研究工具。请配置搜索 API。")

    # 准备系统提示词
    researcher_prompt = research_system_prompt.format(
        mcp_prompt="",  # 简化版本：不支持 MCP
        date=get_today_str()
    )

    # 配置带工具和重试逻辑的模型
    research_model = (
        get_model()
        .bind_tools(tools)
        .with_retry(stop_after_attempt=MAX_STRUCTURED_OUTPUT_RETRIES)
    )

    # 生成研究员回复
    messages = [SystemMessage(content=researcher_prompt)] + researcher_messages
    response = await research_model.ainvoke(messages)

    # 更新状态并进入工具执行
    return Command(
        goto="researcher_tools",
        update={
            "researcher_messages": [response],
            "tool_call_iterations": state.get("tool_call_iterations", 0) + 1
        }
    )


async def execute_tool_safely(tool, args):
    """安全执行工具并处理错误。"""
    try:
        return await tool.ainvoke(args)
    except Exception as e:
        return f"执行工具时出错：{str(e)}"


async def researcher_tools(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher", "compress_research"]]:
    """执行研究员调用的工具。"""

    researcher_messages = state.get("researcher_messages", [])
    most_recent_message = researcher_messages[-1]

    # 如果没有工具调用，则提前退出
    has_tool_calls = bool(most_recent_message.tool_calls)
    has_native_search = (
        openai_websearch_called(most_recent_message)
    )

    if not has_tool_calls and not has_native_search:
        return Command(goto="compress_research")

    # 执行所有工具调用
    tools = await get_all_tools()
    tools_by_name = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search"): tool
        for tool in tools
    }

    tool_calls = most_recent_message.tool_calls
    tool_execution_tasks = [
        execute_tool_safely(tools_by_name[tool_call["name"]], tool_call["args"])
        for tool_call in tool_calls
    ]
    observations = await asyncio.gather(*tool_execution_tasks)

    # 创建工具消息
    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        )
        for observation, tool_call in zip(observations, tool_calls)
    ]

    # 检查退出条件
    exceeded_iterations = state.get("tool_call_iterations", 0) >= MAX_REACT_TOOL_CALLS
    research_complete_called = any(
        tool_call["name"] == "ResearchComplete"
        for tool_call in most_recent_message.tool_calls
    )

    if exceeded_iterations or research_complete_called:
        return Command(
            goto="compress_research",
            update={"researcher_messages": tool_outputs}
        )

    # 继续研究循环
    return Command(
        goto="researcher",
        update={"researcher_messages": tool_outputs}
    )


async def compress_research(state: ResearcherState, config: RunnableConfig):
    """将研究发现压缩并综合为简洁摘要。"""

    researcher_messages = state.get("researcher_messages", [])

    # 添加压缩指令
    researcher_messages.append(HumanMessage(content=compress_research_simple_human_message))

    # 创建压缩提示词
    compression_prompt = compress_research_system_prompt.format(date=get_today_str())
    messages = [SystemMessage(content=compression_prompt)] + researcher_messages

    # 执行压缩
    response = await get_model().ainvoke(messages)

    # 提取原始笔记
    raw_notes_content = "\n".join([
        str(message.content)
        for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
    ])

    return {
        "compressed_research": str(response.content),
        "raw_notes": [raw_notes_content]
    }


# 构建研究员子图
researcher_builder = StateGraph(
    ResearcherState,
    output=ResearcherOutputState
)
researcher_builder.add_node("researcher", researcher)
researcher_builder.add_node("researcher_tools", researcher_tools)
researcher_builder.add_node("compress_research", compress_research)
researcher_builder.add_edge(START, "researcher")
researcher_builder.add_edge("compress_research", END)
researcher_subgraph = researcher_builder.compile()


async def final_report_generation(state: AgentState, config: RunnableConfig):
    """生成最终的综合研究报告。"""

    # 提取研究发现
    notes = state.get("notes", [])
    cleared_state = {"notes": {"type": "override", "value": []}}
    findings = "\n".join(notes)

    # 尝试生成报告
    max_retries = 3
    current_retry = 0

    while current_retry <= max_retries:
        try:
            # 创建综合提示词
            final_report_prompt = final_report_generation_prompt.format(
                research_brief=state.get("research_brief", ""),
                messages=get_buffer_string(state.get("messages", [])),
                findings=findings,
                date=get_today_str()
            )

            # 生成最终报告
            final_report = await get_model().ainvoke([HumanMessage(content=final_report_prompt)])

            return {
                "final_report": final_report.content,
                "messages": [final_report],
                **cleared_state
            }

        except Exception as e:
            current_retry += 1
            continue

    # 返回失败结果
    return {
        "final_report": "生成最终报告时出错：超过最大重试次数",
        "messages": [AIMessage(content="报告生成失败")],
        **cleared_state
    }


# 构建完整研究工作流
deep_researcher_builder = StateGraph(
    AgentState,
    input_schema=AgentInputState
)

# 添加节点
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
deep_researcher_builder.add_node("human_input", human_input)
deep_researcher_builder.add_node("write_research_brief", write_research_brief)
deep_researcher_builder.add_node("research_supervisor", supervisor_subgraph)
deep_researcher_builder.add_node("final_report_generation", final_report_generation)

# 添加边
deep_researcher_builder.add_edge(START, "clarify_with_user")
deep_researcher_builder.add_conditional_edges(
    "clarify_with_user",
    human_feedback_needed,
    {
        "human_input": "human_input",
        "write_research_brief": "write_research_brief",
    },
)
deep_researcher_builder.add_edge("human_input", "clarify_with_user")
deep_researcher_builder.add_edge("write_research_brief", "research_supervisor")
deep_researcher_builder.add_edge("research_supervisor", "final_report_generation")
deep_researcher_builder.add_edge("final_report_generation", END)

graph = deep_researcher_builder.compile()

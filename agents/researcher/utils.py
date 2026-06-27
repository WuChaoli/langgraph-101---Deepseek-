"""Deep Research Agent 的简化工具函数。"""

import logging
from datetime import datetime

from langchain_core.messages import (
    MessageLikeRepresentation,
    filter_messages,
)
from langchain_core.tools import tool

from agents.researcher.models import ResearchComplete

##########################
# 反思工具相关函数
##########################

@tool(description="用于研究规划的战略反思工具")
def think_tool(reflection: str) -> str:
    """用于对研究进展进行战略反思的工具。

    每次搜索后使用此工具分析结果并规划下一步。

    参数：
        reflection: 对研究进展和下一步计划的详细反思

    返回：
        反思已记录的确认信息
    """
    return f"已记录反思：{reflection}"


##########################
# 工具相关函数
##########################

async def get_all_tools():
    """组装研究操作所需的完整工具集。

    返回包含 OpenAI 原生网页搜索在内的工具。
    """
    # 核心研究工具
    tools = [tool(ResearchComplete), think_tool]

    # 添加 OpenAI 原生网页搜索
    # 这是 OpenAI 模型能够识别的特殊工具定义
    tools.append({"type": "web_search_preview"})

    return tools


def get_notes_from_tool_calls(messages: list[MessageLikeRepresentation]):
    """从工具调用消息中提取笔记。"""
    return [tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")]


##########################
# 模型供应商原生网页搜索相关函数
##########################

def anthropic_websearch_called(response):
    """检测是否使用了 Anthropic 原生网页搜索。"""
    try:
        usage = response.response_metadata.get("usage")
        if not usage:
            return False

        server_tool_use = usage.get("server_tool_use")
        if not server_tool_use:
            return False

        web_search_requests = server_tool_use.get("web_search_requests")
        if web_search_requests is None:
            return False

        return web_search_requests > 0

    except (AttributeError, TypeError):
        return False


def openai_websearch_called(response):
    """检测是否使用了 OpenAI 网页搜索。"""
    try:
        tool_outputs = response.additional_kwargs.get("tool_outputs")
        if not tool_outputs:
            return False

        for tool_output in tool_outputs:
            if tool_output.get("type") == "web_search_call":
                return True

        return False
    except (AttributeError, TypeError):
        return False


async def execute_tool_safely(tool, args):
    """安全执行工具并处理错误。"""
    try:
        return await tool.ainvoke(args)
    except Exception as e:
        return f"执行工具时出错：{str(e)}"


##########################
# 其他工具函数
##########################

def get_today_str() -> str:
    """获取用于展示的当前日期字符串。"""
    now = datetime.now()
    return f"{now:%a} {now:%b} {now.day}, {now:%Y}"

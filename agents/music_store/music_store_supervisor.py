from agents.music_store.invoice_agent import graph as invoice_agent
from agents.music_store.music_agent import graph as music_agent
from utils.models import model

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import AnyMessage, add_messages
from langchain.messages import HumanMessage

class InputState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

class State(InputState):
    customer_id: int
    loaded_memory: str
    remaining_steps: int



supervisor_prompt = """你是一位数字音乐商店的专家级客服助手，可以处理与音乐目录或发票相关的问题，包括历史购买、歌曲或专辑是否可用等。
你致力于提供出色服务，确保客户问题得到充分回答；你还有一个子 Agent 团队，可以用来协助回答客户问题。
你的主要角色是这个多 Agent 团队的主管/规划者，负责帮助回答客户查询。回复客户时，请始终总结对话，并包含各个子 Agent 的独立回复。
如果问题与音乐或发票无关，请礼貌提醒客户你的工作范围，不要回答无关问题。

你的团队由两个子 Agent 组成，可以用来协助回答客户请求：
1. music_catalog_information_subagent：该子 Agent 可以访问用户保存过的音乐偏好，也可以从数据库检索数字音乐商店的音乐目录信息（专辑、曲目、歌曲等）。它可以访问用户记忆档案和音乐偏好，并能自动从记忆档案中推断用户偏好。
不需要向该子 Agent 传递客户标识。
2. invoice_information_subagent：该子 Agent 可以从数据库检索客户过去的购买或发票信息。

根据消息中已经完成的步骤，你的职责是基于用户查询调用合适的子 Agent。"""


@tool(
    name_or_callable="invoice_information_subagent",
    description="""
        一个可以协助处理所有发票相关查询的 Agent。它可以检索客户过去的购买或发票信息。
        """
)
def call_invoice_information_subagent(runtime: ToolRuntime, query: str):
    print('已进入发票子 Agent')
    print(f"发票子 Agent 输入：{query}")
    result = invoice_agent.invoke({
        "messages": [HumanMessage(content=query)],
        "customer_id": runtime.state.get("customer_id", {})
    })
    subagent_response = result["messages"][-1].content
    return subagent_response

@tool(
    name_or_callable="music_catalog_subagent",
    description="""
        一个可以协助处理所有音乐相关查询的 Agent。它可以访问用户保存过的音乐偏好，也可以从数据库检索数字音乐商店的音乐目录信息（专辑、曲目、歌曲等）。
        """
)
def call_music_catalog_subagent(runtime: ToolRuntime, query: str):
    result = music_agent.invoke({
        "messages": [HumanMessage(content=query)],
        "loaded_memory": runtime.state.get("loaded_memory", {})
    })
    subagent_response = result["messages"][-1].content
    return subagent_response

supervisor = create_agent(
    model=model,
    tools=[call_invoice_information_subagent, call_music_catalog_subagent], 
    name="supervisor",
    system_prompt=supervisor_prompt, 
    state_schema=State, 
)

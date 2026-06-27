"""Deep Research Agent 的图状态定义和数据结构。"""

import operator
from typing import Annotated, Optional

from langchain_core.messages import MessageLikeRepresentation
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


###################
# 结构化输出
###################
class ConductResearch(BaseModel):
    """调用此工具针对特定主题开展研究。"""
    research_topic: str = Field(
        description="要研究的主题。应是单一主题，并且要非常详细地描述（至少一段）。",
    )

class ResearchComplete(BaseModel):
    """调用此工具表示研究已完成。"""

class Summary(BaseModel):
    """包含关键发现的研究摘要。"""
    
    summary: str
    key_excerpts: str

class ClarifyWithUser(BaseModel):
    """用户澄清请求模型。"""
    
    need_clarification: bool = Field(
        description="是否需要向用户提出澄清问题。",
    )
    question: str = Field(
        description="用于澄清报告范围的用户问题",
    )
    verification: str = Field(
        description="确认消息：用户提供必要信息后，我们将开始研究。",
    )

class ResearchQuestion(BaseModel):
    """用于指导研究的研究问题和简报。"""
    
    research_brief: str = Field(
        description="将用于指导研究的研究问题。",
    )


###################
# 状态定义
###################

def override_reducer(current_value, new_value):
    """允许覆盖状态值的 reducer 函数。"""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    else:
        return operator.add(current_value, new_value)
    
class AgentInputState(MessagesState):
    """InputState 只包含 'messages'。"""

class AgentState(MessagesState):
    """主 Agent 状态，包含消息和研究数据。"""
    
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: Optional[str]
    need_elaboration: bool
    raw_notes: Annotated[list[str], override_reducer] = []
    notes: Annotated[list[str], override_reducer] = []
    final_report: str

class SupervisorState(TypedDict):
    """管理研究任务的主管状态。"""
    
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: str
    notes: Annotated[list[str], override_reducer] = []
    research_iterations: int = 0
    raw_notes: Annotated[list[str], override_reducer] = []

class ResearcherState(TypedDict):
    """执行研究的单个研究员状态。"""
    
    researcher_messages: Annotated[list[MessageLikeRepresentation], operator.add]
    tool_call_iterations: int = 0
    research_topic: str
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []

class ResearcherOutputState(BaseModel):
    """单个研究员的输出状态。"""
    
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []

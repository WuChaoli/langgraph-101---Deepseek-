from agents.music_store.music_store_supervisor import supervisor
from utils.models import model
from utils.utils import get_engine_for_chinook_db

from langgraph.graph import StateGraph, START, END
from typing import Annotated, Optional, NotRequired
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.managed.is_last_step import RemainingSteps
from typing_extensions import TypedDict
from langchain.messages import SystemMessage, HumanMessage, AIMessage
import ast
from langchain_community.utilities.sql_database import SQLDatabase

engine = get_engine_for_chinook_db()
db = SQLDatabase(engine)

class InputState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

class State(InputState):
    customer_id: NotRequired[str]
    loaded_memory: NotRequired[str]
    remaining_steps: NotRequired[RemainingSteps]

from pydantic import BaseModel, Field

class UserInput(BaseModel):
    """用于解析用户提供的账户信息的 schema。"""
    identifier: str = Field(description = "标识符，可以是客户 ID、邮箱或电话号码。")


structured_llm = model.with_structured_output(schema=UserInput)
structured_system_prompt = """你是一位客服代表，负责提取客户标识符。\n
只从消息历史中提取客户账户信息。
如果客户尚未提供该信息，请为该字段返回空字符串。"""

from typing import Optional

# 辅助函数
def get_customer_id_from_identifier(identifier: str) -> Optional[int]:
    """
    使用标识符查询客户 ID。标识符可以是客户 ID、邮箱或电话号码。
    
    参数：
        identifier (str): 可以是客户 ID、邮箱或电话号码。
    
    返回：
        Optional[int]: 如果找到则返回 CustomerId，否则返回 None。
    """
    if identifier.isdigit():
        return int(identifier)
    elif identifier[0] == "+":
        query = f"SELECT CustomerId FROM Customer WHERE Phone = '{identifier}';"
        result = db.run(query)
        formatted_result = ast.literal_eval(result)
        if formatted_result:
            return formatted_result[0][0]
    elif "@" in identifier:
        query = f"SELECT CustomerId FROM Customer WHERE Email = '{identifier}';"
        result = db.run(query)
        formatted_result = ast.literal_eval(result)
        if formatted_result:
            return formatted_result[0][0]
    return None 

# 节点

def verify_info(state: State):
    """解析客户输入并与数据库匹配，以验证客户账户。"""

    if state.get("customer_id") is None: 
        system_instructions = """你是音乐商店 Agent，当前正在将验证客户身份作为客服流程的第一步。
        只有客户账户通过验证后，你才能协助他们解决问题。
        为了验证身份，客户需要提供客户 ID、邮箱或电话号码之一。
        如果客户尚未提供标识符，请向他们索要。
        如果客户已提供标识符但无法找到，请让他们修改后重新提供。"""

        user_input = state["messages"][-1] 
    
        # 解析客户 ID
        parsed_info = structured_llm.invoke([SystemMessage(content=structured_system_prompt)] + [user_input])
    
        # 提取详情
        identifier = parsed_info.identifier
    
        customer_id = ""
        # 尝试查找客户 ID
        if (identifier):
            customer_id = get_customer_id_from_identifier(identifier)
    
        if customer_id != "":
            intent_message = AIMessage(
                content= f"感谢你提供信息！我已经成功验证你的账户，客户 ID 为 {customer_id}。"
            )
            return {
                  "customer_id": customer_id,
                  "messages" : [intent_message]
                  }
        else:
          response = model.invoke([SystemMessage(content=system_instructions)]+state['messages'])
          return {"messages": [response]}

    else: 
        pass

from langgraph.types import interrupt
# 节点
def human_input(state: State):
    """应在此处被 interrupt 的空操作节点。"""
    user_input = interrupt("请提供输入。")
    return {"messages": [HumanMessage(content=user_input)]}


# 条件边
def should_interrupt(state: State):
    if state.get("customer_id") is not None:
        return "continue"
    else:
        return "interrupt"



# 添加节点
multi_agent_verify = StateGraph(State, input_schema = InputState)
multi_agent_verify.add_node("verify_info", verify_info)
multi_agent_verify.add_node("human_input", human_input)
multi_agent_verify.add_node("supervisor", supervisor)

multi_agent_verify.add_edge(START, "verify_info")
multi_agent_verify.add_conditional_edges(
    "verify_info",
    should_interrupt,
    {
        "continue": "supervisor",
        "interrupt": "human_input",
    },
)
multi_agent_verify.add_edge("human_input", "verify_info")
multi_agent_verify.add_edge("supervisor", END)
graph = multi_agent_verify.compile(name="multi_agent_verify")

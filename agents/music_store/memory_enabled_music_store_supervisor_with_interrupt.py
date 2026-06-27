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
from langgraph.store.base import BaseStore
from typing import Optional, List

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

# 用于整理记忆的辅助函数
def format_user_memory(user_data):
    """如果可用，则格式化用户音乐偏好。"""
    profile = user_data['memory']
    result = ""
    
    # 同时处理 Pydantic 模型（属性）和 dict（键）两种表示
    if isinstance(profile, dict):
        music_prefs = profile.get('music_preferences', [])
    else:
        music_prefs = getattr(profile, 'music_preferences', [])
    
    if music_prefs:
        result += f"音乐偏好：{', '.join(music_prefs)}"
    return result.strip()

# 节点
def load_memory(state: State, store: BaseStore):
    """如果可用，则加载用户音乐偏好。"""
    
    user_id = str(state["customer_id"])  # 转为字符串，以匹配 create_memory
    namespace = ("memory_profile", user_id)
    existing_memory = store.get(namespace, "user_memory")
    formatted_memory = ""
    if existing_memory and existing_memory.value:
        formatted_memory = format_user_memory(existing_memory.value)

    return {"loaded_memory" : formatted_memory}

# 用于创建记忆的用户档案结构

class UserProfile(BaseModel):
    customer_id: str = Field(
        description="客户的客户 ID"
    )
    music_preferences: List[str] = Field(
        description="客户的音乐偏好"
    )

create_memory_prompt = """你是一位专家分析师，正在观察客户与客服助手之间发生的一段对话。该客服助手服务于一家数字音乐商店，并使用多 Agent 团队回答客户请求。
你的任务是分析客户与客服助手之间的对话，并更新与该客户关联的记忆档案。
你尤其关注保存客户在对话中透露的任何音乐兴趣，特别是他们的音乐偏好，并写入记忆档案。

<core_instructions>
1. 记忆档案可能为空。如果为空，你应始终为客户创建新的记忆档案。
2. 你应识别客户在对话中表达的任何音乐兴趣，并在尚未存在时添加到记忆档案中。
3. 对记忆档案中的每个键，如果没有新信息，不要更新该值，保持原值不变。
4. 只有存在新信息时，才更新记忆档案中的值。
</core_instructions>

<expected_format>
客户的记忆档案应包含以下字段：
- customer_id：客户的客户 ID
- music_preferences：客户的音乐偏好

重要：确保你的回复是一个包含这些字段的对象。
</expected_format>


<important_context>
**重要上下文如下**
为帮助你完成任务，下面附上客户与客服助手之间的对话，以及与该客户关联的现有记忆档案。你应基于这些信息更新或创建记忆档案。

你需要分析的客户与客服助手对话如下：
{conversation}

你需要基于对话更新或创建的现有客户记忆档案如下：
{memory_profile}

</important_context>

提醒：深呼吸，仔细思考后再回复。
"""

# 节点
def create_memory(state: State, store: BaseStore):
    user_id = str(state["customer_id"])
    namespace = ("memory_profile", user_id)
    formatted_memory = state["loaded_memory"]
    formatted_system_message = SystemMessage(content=create_memory_prompt.format(conversation=state["messages"], memory_profile=formatted_memory))
    # Anthropic 要求除了 system message 之外至少有一条 user message
    user_prompt = HumanMessage(content="请根据指令分析这段对话，并更新客户的记忆档案。")
    updated_memory = model.with_structured_output(UserProfile).invoke([formatted_system_message, user_prompt])
    key = "user_memory"
    # 将 Pydantic 模型转为 dict，避免重启时出现 pickle 序列化问题
    store.put(namespace, key, {"memory": updated_memory.model_dump()})


multi_agent_final = StateGraph(State, input_schema = InputState) 
multi_agent_final.add_node("verify_info", verify_info)
multi_agent_final.add_node("human_input", human_input)
multi_agent_final.add_node("load_memory", load_memory)
multi_agent_final.add_node("supervisor", supervisor)
multi_agent_final.add_node("create_memory", create_memory)

multi_agent_final.add_edge(START, "verify_info")
multi_agent_final.add_conditional_edges(
    "verify_info",
    should_interrupt,
    {
        "continue": "load_memory",
        "interrupt": "human_input",
    },
)
multi_agent_final.add_edge("human_input", "verify_info")
multi_agent_final.add_edge("load_memory", "supervisor")
multi_agent_final.add_edge("supervisor", "create_memory")
multi_agent_final.add_edge("create_memory", END)

agent = multi_agent_final.compile(name="multi_agent_verify")

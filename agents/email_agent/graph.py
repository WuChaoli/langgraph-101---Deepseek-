from typing import Literal, TypedDict
from pydantic import BaseModel, Field
from datetime import datetime

from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Command
from dotenv import load_dotenv
from utils.models import model

load_dotenv("../.env")

class RouterSchema(BaseModel):
    """分析未读邮件，并根据内容进行路由。"""

    reasoning: str = Field(
        description="分类背后的逐步推理。"
    )
    classification: Literal["ignore", "respond", "notify"] = Field(
        description="邮件分类：'ignore' 表示无关邮件，"
        "'notify' 表示重要但不需要回复的信息，"
        "'respond' 表示需要回复的邮件",
    )

llm_router = model.with_structured_output(RouterSchema) 

# 工具
@tool
def schedule_meeting(
    attendees: list[str], subject: str, duration_minutes: int, preferred_day: datetime, start_time: int
) -> str:
    """安排日历会议。"""
    # 占位回复：真实应用中会检查日历并安排会议
    date_str = preferred_day.strftime("%A, %B %d, %Y")
    return f"已在 {date_str} {start_time} 安排“{subject}”会议，时长 {duration_minutes} 分钟，参会人数 {len(attendees)} 人"

@tool
def check_calendar_availability(day: str) -> str:
    """检查指定日期的日历空闲时间。"""
    # 占位回复：真实应用中会检查实际日历
    return f"{day} 可用时间：上午 9:00、下午 2:00、下午 4:00"


@tool
def write_email(to: str, subject: str, content: str) -> str:
    """编写并发送邮件。"""
    # 占位回复：真实应用中会发送邮件
    return f"已向 {to} 发送主题为“{subject}”的邮件，内容：{content}"

@tool
class Done(BaseModel):
    """邮件已发送。"""
    done: bool

tools = [schedule_meeting, check_calendar_availability, write_email, Done]
tools_by_name = {tool.name: tool for tool in tools}

llm_with_tools = model.bind_tools(tools, tool_choice="any", parallel_tool_calls=False)


# 状态定义
class StateInput(TypedDict):
    # 这是状态的输入
    email_input: dict

class State(MessagesState):
    # 这个状态类内置 messages 键
    email_input: dict
    classification_decision: Literal["ignore", "respond", "notify"]


# ------------------------------------------------------------
# 邮件 Agent
# ------------------------------------------------------------
action_instructions = """
< Role >
你是一位顶尖的高管助理，重视帮助你的高管尽可能高效地完成工作。
</ Role >

< Tools >
你可以使用以下工具来管理沟通和日程：

1. write_email(to, subject, content) - 向指定收件人发送邮件
2. schedule_meeting(attendees, subject, duration_minutes, preferred_day, start_time) - 安排日历会议
3. check_calendar_availability(day) - 检查指定日期的可用时间段
4. Done - 邮件已发送

注意：对每次输入，只能调用一个工具
</ Tools >

< Instructions >
处理邮件时，请遵循以下步骤：
1. 仔细分析邮件内容和目的
3. 如果需要回复邮件，请使用 write_email 工具起草回复邮件
4. 如果是会议请求，请使用 check_calendar_availability 工具查找空闲时间段
5. 如果要安排会议，请使用 schedule_meeting 工具，并为 preferred_day 参数传入 datetime 对象
   - 今天的日期是 {today}，请用它来准确安排会议
6. 如果已经安排会议，请使用 write_email 工具起草一封简短回复邮件
7. 使用 write_email 工具后，任务完成
8. 如果邮件已经发送，请使用 Done 工具表示任务完成
</ Instructions >

< Background >
我是 Robert，LangChain 的软件工程师。
</ Background >

< Response Preferences >
使用专业且简洁的语言。如果邮件提到截止日期，请在回复中明确确认并引用该截止日期。

回复需要调查的技术问题时：
- 明确说明你会调查，或会向谁询问
- 提供预计何时能获得更多信息或完成任务的时间

回复活动或会议邀请时：
- 始终确认任何提到的截止日期（尤其是注册截止日期）
- 如果提到工作坊或具体主题，请询问更具体的信息
- 如果提到折扣（团体或早鸟），请明确请求相关信息
- 不要直接承诺参加

回复协作或项目相关请求时：
- 确认对方提到的任何现有工作或材料（草稿、幻灯片、文档等）
- 明确说明会在会议前或会议中查看这些材料
- 安排会议时，清楚说明建议的具体星期、日期和时间。

回复会议安排请求时：
- 如果对方要求确认某个会议时间，请检查原邮件中提到的所有时间段，并根据你的可用时间安排其中一个建议时间；或者说明你无法参加对方建议的时间。
- 如果对方询问你的可用时间，请检查日历并发送邮件提出多个可选时间。不要直接安排会议。
- 在回复中提及会议时长，以确认你已正确记录。
- 在回复中引用会议目的。
</ Response Preferences >

< Calendar Preferences >
优先安排 30 分钟会议，但 15 分钟会议也可以接受。
更偏好一天中较晚的时间。
</ Calendar Preferences >
"""

# 节点
def llm_call(state: State):
    """由 LLM 判断是否调用工具。"""
    agent_system_prompt = action_instructions
    return {
        "messages": [
            llm_with_tools.invoke([
                {"role": "system", "content": agent_system_prompt.format(today=datetime.now().strftime("%Y-%m-%d"))}
            ] + state["messages"])
        ]
    }

def tool_node(state: State):
    """执行工具调用。"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append({"role": "tool", "content" : observation, "tool_call_id": tool_call["id"]})
    return {"messages": result}

# 条件边函数
def should_continue(state: State) -> Literal["Action", "__end__"]:
    """路由到 Action；如果调用了 Done 工具，则结束。"""
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        for tool_call in last_message.tool_calls: 
            if tool_call["name"] == "Done":
                return END
            else:
                return "Action"

# 构建工作流
agent_builder = StateGraph(State)

# 添加节点
agent_builder.add_node("agent", llm_call)
agent_builder.add_node("tools", tool_node)

# 添加边来连接节点
agent_builder.add_edge(START, "agent")
agent_builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        # should_continue 返回的名称：下一步要访问的节点名称
        "Action": "tools",
        END: END,
    },
)
agent_builder.add_edge("tools", "agent")

# 编译 Agent
agent = agent_builder.compile()

# ------------------------------------------------------------
# 分诊路由器
# ------------------------------------------------------------
def parse_email(email_input: dict) -> dict:
    """解析邮件输入字典。

    参数：
        email_input (dict): 包含邮件字段的字典：
            - author: 发件人姓名和邮箱
            - to: 收件人姓名和邮箱
            - subject: 邮件主题
            - email_thread: 完整邮件内容

    返回：
        tuple[str, str, str, str]: 包含以下内容的元组：
            - author: 发件人姓名和邮箱
            - to: 收件人姓名和邮箱
            - subject: 邮件主题
            - email_thread: 完整邮件内容
    """
    return (
        email_input["author"],
        email_input["to"],
        email_input["subject"],
        email_input["email_thread"],
    )

def format_email_markdown(subject, author, to, email_thread, email_id=None):
    """将邮件详情格式化为便于展示的 markdown 字符串。
    
    参数：
        subject: 邮件主题
        author: 邮件发件人
        to: 邮件收件人
        email_thread: 邮件内容
        email_id: 可选邮件 ID（用于 Gmail API）
    """
    id_section = f"\n**ID**: {email_id}" if email_id else ""
    
    return f"""

**主题**: {subject}
**发件人**: {author}
**收件人**: {to}{id_section}

{email_thread}

---
"""

triage_instructions = """
< Role >
你的角色是根据下面的指令和背景信息，对收到的邮件进行分诊。
</ Role >

< Background >
我是 Robert，LangChain 的软件工程师。
</ Background >

< Instructions >
将每封邮件归入以下三类之一：
1. IGNORE - 不值得回复或跟踪的邮件
2. NOTIFY - 值得通知的重要信息，但不需要回复
3. RESPOND - 需要直接回复的邮件
请把下面的邮件分类到其中一类。
</ Instructions >

< Rules >
不值得回复的邮件：
- 营销通讯和促销邮件
- 垃圾邮件或可疑邮件
- 被抄送的 FYI 线程，且没有直接问题

还有一些事项需要知晓，但不需要邮件回复。对于这些邮件，应进行通知（使用 `notify` 响应）。示例包括：
- 团队成员病假或休假
- 构建系统通知或部署
- 没有行动项的项目状态更新
- 重要公司公告
- 包含当前项目相关信息的 FYI 邮件
- HR 部门截止日期提醒
- GitHub 通知

值得回复的邮件：
- 团队成员提出、需要专业知识回答的直接问题
- 需要确认的会议请求
- 与团队项目相关的关键 bug 报告
- 管理层要求确认收到的请求
- 客户关于项目状态或功能的询问
- 关于文档、代码或 API 的技术问题（尤其是关于缺失端点或功能的问题）
- 与家庭相关的个人提醒（妻子/女儿）
- 与自我照顾相关的个人提醒（医生预约等）
</ Rules >
"""

def triage_router(state: State) -> Command[Literal["response_agent", "__end__"]]:
    """
    分析邮件内容，决定应该回复、通知还是忽略。
    """
    author, to, subject, email_thread = parse_email(state["email_input"])
    system_prompt = triage_instructions

    user_prompt = """
请判断如何处理下面的邮件线程：

发件人：{author}
收件人：{to}
主题：{subject}
{email_thread}""".format(
        author=author, to=to, subject=subject, email_thread=email_thread
    )

    # 为 Agent Inbox 创建邮件 markdown，以便通知时使用
    email_markdown = format_email_markdown(subject, author, to, email_thread)
    # 运行路由 LLM
    result = llm_router.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    # 决策
    classification = result.classification

    if classification == "respond":
        goto = "response_agent"
        # 将邮件添加到 messages
        update = {
            "classification_decision": result.classification,
            "messages": [{"role": "user",
                            "content": f"回复这封邮件：{email_markdown}"
                        }],
        }
    elif result.classification == "ignore":
        update =  { "classification_decision": result.classification}
        goto = END
    elif result.classification == "notify":
        update = { "classification_decision": result.classification}
        goto = END
    else:
        raise ValueError(f"无效分类：{result.classification}")
    return Command(goto=goto, update=update)

# 构建工作流
overall_workflow = (
    StateGraph(State, input=StateInput)
    .add_node(triage_router)
    .add_node("response_agent", agent)
    .add_edge(START, "triage_router")
)
graph = overall_workflow.compile()

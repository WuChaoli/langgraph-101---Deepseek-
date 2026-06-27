from utils.models import model
from utils.utils import get_engine_for_chinook_db
from langchain_community.utilities.sql_database import SQLDatabase
from typing_extensions import TypedDict
from typing import Annotated, NotRequired
from langgraph.graph.message import AnyMessage, add_messages
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime

engine = get_engine_for_chinook_db()
db = SQLDatabase(engine)

class InputState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

class State(InputState):
    customer_id: NotRequired[int]
    loaded_memory: NotRequired[str]

@tool 
def get_invoices_by_customer_sorted_by_date(runtime: ToolRuntime) -> list[dict]:
    """
    使用客户 ID 查询该客户的所有发票。客户 ID 存在状态变量中，因此不会出现在消息历史里。
    发票会按发票日期倒序排列，便于客户查看最近/最早的发票，或查看指定日期范围内的发票。
    
    返回：
        list[dict]: 该客户的发票列表。
    """
    # customer_id = state.get("customer_id", "未知用户")
    customer_id = runtime.state.get("customer_id", {})
    return db.run(f"SELECT * FROM Invoice WHERE CustomerId = {customer_id} ORDER BY InvoiceDate DESC;")


@tool 
def get_invoices_sorted_by_unit_price(runtime: ToolRuntime) -> list[dict]:
    """
    当客户想根据发票的单价/费用了解某张发票详情时使用此工具。
    该工具会查询客户的所有发票，并按单价从高到低排序。为了找到该客户关联的发票，
    需要使用客户 ID。客户 ID 存在状态变量中，因此不会出现在消息历史里。

    返回：
        list[dict]: 按单价排序的发票列表。
    """
    # customer_id = state.get("customer_id", "未知用户")
    query = f"""
        SELECT Invoice.*, InvoiceLine.UnitPrice
        FROM Invoice
        JOIN InvoiceLine ON Invoice.InvoiceId = InvoiceLine.InvoiceId
        WHERE Invoice.CustomerId = {customer_id}
        ORDER BY InvoiceLine.UnitPrice DESC;
    """
    customer_id = runtime.state.get("customer_id", {})
    return db.run(query)


@tool
def get_employee_by_invoice_and_customer(runtime: ToolRuntime, invoice_id: int) -> dict:
    """
    接收发票 ID 和客户 ID，并返回与该发票关联的员工信息。
    客户 ID 存在状态变量中，因此不会出现在消息历史里。
    参数：
        invoice_id (int): 指定发票的 ID。

    返回：
        dict: 与该发票关联的员工信息。
    """
    # customer_id = state.get("customer_id", "未知用户")
    customer_id = runtime.state.get("customer_id", {})
    query = f"""
        SELECT Employee.FirstName, Employee.Title, Employee.Email
        FROM Employee
        JOIN Customer ON Customer.SupportRepId = Employee.EmployeeId
        JOIN Invoice ON Invoice.CustomerId = Customer.CustomerId
        WHERE Invoice.InvoiceId = ({invoice_id}) AND Invoice.CustomerId = ({customer_id});
    """
    
    employee_info = db.run(query, include_columns=True)
    
    if not employee_info:
        return f"未找到发票 ID {invoice_id} 和客户标识 {customer_id} 对应的员工。"
    return employee_info

invoice_tools = [get_invoices_by_customer_sorted_by_date, get_invoices_sorted_by_unit_price, get_employee_by_invoice_and_customer]

invoice_subagent_prompt = """
    你是助手团队中的一个子 Agent，专门负责检索和处理发票信息。只有问题中与发票相关的部分才会路由给你，因此只回复这部分内容。

    你可以使用三个工具，用于从数据库检索和处理发票信息：
    - get_invoices_by_customer_sorted_by_date：查询某位客户的所有发票，并按发票日期排序。
    - get_invoices_sorted_by_unit_price：查询某位客户的所有发票，并按单价排序。
    - get_employee_by_invoice_and_customer：查询与某张发票和某位客户关联的员工信息。
    
    如果无法检索到发票信息，请告知客户暂时无法获取，并询问他们是否想查询其他内容。
    
    核心职责：
    - 从数据库检索并处理发票信息
    - 在客户询问时提供发票详情，包括客户信息、发票日期、总金额、与发票关联的员工等
    - 始终保持专业、友好和耐心
    
    你可能会收到额外上下文，请用它来帮助回答客户问题。上下文如下：
    """

# 定义子 Agent
graph = create_agent(model, tools=invoice_tools, name="invoice_information_subagent", system_prompt=invoice_subagent_prompt, state_schema=State)

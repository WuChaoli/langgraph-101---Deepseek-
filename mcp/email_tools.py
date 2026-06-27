"""通过 stdio transport 暴露邮件和日历工具的 MCP 服务器。

直接运行：python -u mcp/email_tools.py
notebook 会通过 langchain-mcp-adapters 以子进程方式使用它。
"""

from mcp.server import FastMCP

mcp = FastMCP("邮件工具")


@mcp.tool(description="编写并发送邮件。")
def write_email(to: str, subject: str, content: str) -> str:
    """编写并发送邮件。"""
    return f"已向 {to} 发送主题为“{subject}”的邮件"


@mcp.tool(description="检查指定日期的日历空闲时间。")
def check_calendar_availability(day: str) -> str:
    """检查指定日期的日历空闲时间。"""
    return f"{day} 可用时间：上午 9:00、下午 2:00、下午 4:00"


@mcp.tool(description="安排日历会议。")
def schedule_meeting(attendees: str, subject: str, day: str, time: str) -> str:
    """安排会议。"""
    return f"已在 {day} {time} 为 {attendees} 安排“{subject}”会议"


if __name__ == "__main__":
    mcp.run(transport="stdio")

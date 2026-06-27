import sqlite3
import requests
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

def show_graph(graph, xray=False):
    """显示 LangGraph Mermaid 图，失败时回退为 ASCII 图。
    
    参数：
        graph: 拥有 get_graph() 方法的 LangGraph 对象
        xray: 是否显示图的内部结构
    """
    from IPython.display import Image
    try:
        return Image(graph.get_graph(xray=xray).draw_mermaid_png())
    except Exception as e:
        print(f"⚠️  图片渲染失败：{e}")
        print("\n📊 改为显示 ASCII 图：\n")
        ascii_diagram = graph.get_graph(xray=xray).draw_ascii()
        print(ascii_diagram)
        return None

def get_engine_for_chinook_db():
    """拉取 SQL 文件，填充内存数据库，并创建 engine。"""
    url = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
    response = requests.get(url)
    sql_script = response.text

    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.executescript(sql_script)
    return create_engine(
        "sqlite://",
        creator=lambda: connection,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )

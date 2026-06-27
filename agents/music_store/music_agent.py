from utils.models import model
from utils.utils import get_engine_for_chinook_db
from langchain_community.utilities.sql_database import SQLDatabase
from typing_extensions import TypedDict
from typing import Annotated, NotRequired
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.managed.is_last_step import RemainingSteps
from langgraph.prebuilt import ToolNode
from langchain.messages import SystemMessage
from langgraph.graph import StateGraph, START, END
from langchain.tools import tool
import ast

engine = get_engine_for_chinook_db()
db = SQLDatabase(engine)

class InputState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    loaded_memory: NotRequired[str]

class State(InputState):
    customer_id: NotRequired[str]
    remaining_steps: NotRequired[RemainingSteps]



@tool
def get_albums_by_artist(artist: str):
    """按艺术家查询专辑。"""
    return db.run(
        f"""
        SELECT Album.Title, Artist.Name 
        FROM Album 
        JOIN Artist ON Album.ArtistId = Artist.ArtistId 
        WHERE Artist.Name LIKE '%{artist}%';
        """,
        include_columns=True
    )

@tool
def get_tracks_by_artist(artist: str):
    """按艺术家（或相似艺术家）查询歌曲。"""
    return db.run(
        f"""
        SELECT Track.Name as SongName, Artist.Name as ArtistName 
        FROM Album 
        LEFT JOIN Artist ON Album.ArtistId = Artist.ArtistId 
        LEFT JOIN Track ON Track.AlbumId = Album.AlbumId 
        WHERE Artist.Name LIKE '%{artist}%';
        """,
        include_columns=True
    )

@tool
def get_songs_by_genre(genre: str):
    """
    从数据库中获取匹配指定流派的歌曲。
    
    参数：
        genre (str): 要查询的歌曲流派。
    
    返回：
        list[dict]: 匹配指定流派的歌曲列表。
    """
    genre_id_query = f"SELECT GenreId FROM Genre WHERE Name LIKE '%{genre}%'"
    genre_ids = db.run(genre_id_query)
    if not genre_ids:
        return f"未找到流派为 {genre} 的歌曲"
    genre_ids = ast.literal_eval(genre_ids)
    genre_id_list = ", ".join(str(gid[0]) for gid in genre_ids)

    songs_query = f"""
        SELECT Track.Name as SongName, Artist.Name as ArtistName
        FROM Track
        LEFT JOIN Album ON Track.AlbumId = Album.AlbumId
        LEFT JOIN Artist ON Album.ArtistId = Artist.ArtistId
        WHERE Track.GenreId IN ({genre_id_list})
        GROUP BY Artist.Name
        LIMIT 8;
    """
    songs = db.run(songs_query, include_columns=True)
    if not songs:
        return f"未找到流派为 {genre} 的歌曲"
    formatted_songs = ast.literal_eval(songs)
    return [
        {"Song": song["SongName"], "Artist": song["ArtistName"]}
        for song in formatted_songs
    ]

@tool
def check_for_songs(song_title):
    """按歌曲名检查歌曲是否存在。"""
    return db.run(
        f"""
        SELECT * FROM Track WHERE Name LIKE '%{song_title}%';
        """,
        include_columns=True
    )

music_tools = [get_albums_by_artist, get_tracks_by_artist, get_songs_by_genre, check_for_songs]
llm_with_music_tools = model.bind_tools(music_tools)


# 节点
music_tool_node = ToolNode(music_tools)

# 节点
def music_assistant(state: State): 

    # 获取长期记忆
    memory = "None" 
    if "loaded_memory" in state: 
        memory = state["loaded_memory"]

    # Agent 指令
    music_assistant_prompt = f"""
    你是助手团队的一员，专门帮助客户发现和了解数字音乐商店目录中的音乐。
    如果找不到某位艺术家关联的播放列表、歌曲或专辑，这是可以接受的。
    只需要告知客户：目录中没有与该艺术家关联的播放列表、歌曲或专辑。
    你还会获得用户保存过的偏好上下文，请用它来定制回复。
    
    核心职责：
    - 搜索并提供有关歌曲、专辑、艺术家和播放列表的准确信息
    - 根据客户兴趣提供相关推荐
    - 细致处理与音乐相关的问题
    - 帮助客户发现他们可能喜欢的新音乐
    - 只有与音乐目录相关的问题才会路由给你；请忽略其他问题。
    
    搜索指南：
    1. 在判断某项内容不可用之前，始终先进行充分搜索
    2. 如果找不到精确匹配，请尝试：
       - 检查其他拼写
       - 查找相似的艺术家名称
       - 使用部分匹配搜索
       - 检查不同版本或混音版本
    3. 提供歌曲列表时：
       - 每首歌都包含艺术家名称
       - 相关时提及专辑
       - 如果属于某个播放列表，请说明
       - 如果存在多个版本，请指出
    
    下面提供额外上下文：

    之前保存的用户偏好：{memory}
    
    消息历史也已附带。
    """

    # 调用模型
    response = llm_with_music_tools.invoke([SystemMessage(music_assistant_prompt)] + state["messages"])
    
    # 更新状态
    return {"messages": [response]}

# 条件边：判断是否继续执行
def should_continue(state: State):
    messages = state["messages"]
    last_message = messages[-1]
    
    # 如果没有函数调用，则流程结束
    if not last_message.tool_calls:
        return "end"
    # 否则继续执行
    else:
        return "continue"

music_workflow = StateGraph(State, input_schema = InputState)

# 添加节点
music_workflow.add_node("music_assistant", music_assistant)
music_workflow.add_node("music_tool_node", music_tool_node)


# 添加边
# 首先定义起始节点。用户查询总是先路由到子 Agent 节点。
music_workflow.add_edge(START, "music_assistant")

# 添加条件边
music_workflow.add_conditional_edges(
    "music_assistant",
    # 表示条件边的函数
    should_continue,
    {
        # 如果返回 `continue`，则调用工具节点。
        "continue": "music_tool_node",
        # 否则流程结束。
        "end": END,
    },
)



music_workflow.add_edge("music_tool_node", "music_assistant")

graph = music_workflow.compile(name="music_catalog_subagent")

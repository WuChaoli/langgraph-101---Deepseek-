from langchain_core.tools import tool
from langchain.agents import create_agent
import requests
import json

from utils.models import model

@tool
def get_weather(latitude: float, longitude: float) -> str:
    """根据给定坐标获取当前华氏温度和天气代码。

    参数：
        latitude: 纬度坐标
        longitude: 经度坐标

    返回：
        包含 temperature_fahrenheit 和 weather_code 的 JSON 字符串（回复用户时不要直接展示天气代码，请解释成自然语言）
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,weather_code",
        "temperature_unit": "fahrenheit"
    }

    weather = requests.get(url, params=params).json()["current"]
    temperature = weather["temperature_2m"]
    weather_code = weather["weather_code"]
    result = {
        "temperature_fahrenheit": temperature,
        "weather_code": weather_code
    }

    return json.dumps(result)

@tool
def get_user_preferences(user_id: str) -> str:
    """获取用户保存过的偏好。"""
    # 模拟用户数据库
    preferences = {
        "alice": "喜欢科幻电影，偏好温暖天气的旅行目的地",
        "bob": "喜欢喜剧电影，旅行时偏好寒冷气候"
    }
    return preferences.get(user_id.lower(), "没有找到偏好记录")

@tool
def book_recommendation(genre: str, user_preferences: str = "") -> str:
    """根据类型和用户偏好获取个性化电影推荐。"""
    recommendations = {
        "sci-fi": "根据你的偏好，可以试试：《降临》《机械姬》或《火星救援》",
        "comedy": "根据你的偏好，可以试试：《谋杀绿脚趾》《王牌播音员》或《伴娘》"
    }
    return recommendations.get(genre.lower(), "暂无可用推荐")

# 创建一个乐于助人的个人助手 Agent
agent = create_agent(
    model=model,
    tools=[get_weather, get_user_preferences, book_recommendation],
    system_prompt="""你是一位乐于助人的个人助手。
    
    你可以：
    - 查询任意城市的天气
    - 查询用户偏好
    - 根据偏好推荐电影
    
    请始终保持友好，并根据用户偏好提供个性化回复。"""
)

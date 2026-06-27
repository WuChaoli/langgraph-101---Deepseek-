"""
模型初始化文件

配置整个工作坊 notebook 使用的 LLM 模型。

默认：Anthropic（claude-haiku-4-5）。

切换供应商：
  1. 注释掉下面的默认模型部分。
  2. 取消注释你想使用的供应商部分。
  3. 按照内联配置说明完成设置。

已包含的供应商配置（默认注释）：
  - Azure OpenAI  （需要 AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT）
  - AWS Bedrock   （需要 AWS 凭据 + AWS_MODEL_ARN）
  - Google Vertex （需要 GOOGLE_APPLICATION_CREDENTIALS）
"""

from dotenv import load_dotenv
load_dotenv(override=True)
from langchain.chat_models import init_chat_model


# ---- 默认模型 -------------------------------------------------------------
# model = init_chat_model("openai:gpt-4.1-mini")
model = init_chat_model("deepseek:deepseek-v4-flash")

# 默认使用 Anthropic
# model = init_chat_model("anthropic:claude-haiku-4-5")


# ---- Azure OpenAI ---------------------------------------------------------
# from langchain_openai import AzureChatOpenAI
# from azure.identity import InteractiveBrowserCredential

# credential = InteractiveBrowserCredential()

# def get_token():
#     token = credential.get_token("https://cognitiveservices.azure.com/.default")
#     return token.token

# 确保已设置 AZURE_OPENAI_API_KEY 和 AZURE_OPENAI_ENDPOINT。

# Azure OpenAI：使用环境变量
# model = AzureChatOpenAI(
#     azure_deployment="gpt-4o",
#     streaming=True,
# )

# Azure OpenAI：使用 Azure AD
# model = AzureChatOpenAI(
#     api_version="2024-03-01-preview",
#     azure_endpoint="https://deployment.openai.azure.com/",
#     azure_deployment="gpt-4o",
#     azure_ad_token_provider=get_token,
# )


# ---- AWS Bedrock ----------------------------------------------------------
# import os
# from langchain_aws import ChatBedrockConverse

# AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
# AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
# AWS_REGION_NAME = os.getenv("AWS_REGION_NAME")
# AWS_MODEL_ARN = os.getenv("AWS_MODEL_ARN")

# model = ChatBedrockConverse(
#     aws_access_key_id=AWS_ACCESS_KEY_ID,
#     aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
#     region_name=AWS_REGION_NAME,
#     provider="anthropic",
#     model_id=AWS_MODEL_ARN,
# )


# ---- Google Vertex AI -----------------------------------------------------
# 确保已配置 Vertex AI 凭据，并让 GOOGLE_APPLICATION_CREDENTIALS
# 指向对应的 JSON 文件。

# import os
# from pathlib import Path
# from langchain.chat_models import init_chat_model

# # 解析项目根目录并加载 .env（utils/ -> 项目根目录向上一级）
# project_root = Path(__file__).resolve().parent.parent
# load_dotenv(dotenv_path=project_root / ".env", override=True)

# # 如果凭据路径是相对路径，则转换为绝对路径
# if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
#     cred_path = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
#     if not os.path.isabs(cred_path):
#         os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(project_root / cred_path.lstrip("./"))

# model = init_chat_model("google_vertexai:gemini-2.5-flash")

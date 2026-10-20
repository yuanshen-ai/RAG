from langchain_openai import ChatOpenAI
from conf import app_config


llm_client = ChatOpenAI(
    model_name=app_config.llm.model_name,
    base_url=app_config.llm.base_url,
    api_key=app_config.llm.api_key
)

vlm_client = ChatOpenAI(
    model_name=app_config.vlm.model_name,
    base_url=app_config.vlm.base_url,
    api_key=app_config.vlm.api_key
)
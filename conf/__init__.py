from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict
from pathlib import Path

class FileLoggingConfig(BaseModel):
    enable: bool
    level: str
    path: str
    rotation: str
    retention: str 
    model_config = ConfigDict(from_attributes=True)


class ConsoleLoggingConfig(BaseModel):
    enable: bool
    level: str
    model_config = ConfigDict(from_attributes=True)


class LoggingConfig(BaseModel):
    file: FileLoggingConfig
    console: ConsoleLoggingConfig
    model_config = ConfigDict(from_attributes=True)


class MineruConfig(BaseModel):
    base_url: str
    token: str
    model_config = ConfigDict(from_attributes=True)


class MinioConfig(BaseModel):
    endpoint: str
    access_key: str
    secret_key: str
    bucket_name: str
    secure: bool
    model_config = ConfigDict(from_attributes=True)


class MilvusConfig(BaseModel):
    host: str
    port: int
    user: str
    password: str
    model_config = ConfigDict(from_attributes=True)


class PostgreConfig(BaseModel):
    host: str
    port: int
    user: str
    password: str
    db_name: str
    model_config = ConfigDict(from_attributes=True)


class LMBaseConfig(BaseModel):
    model_name: str
    api_key: str
    base_url: str
    model_config = ConfigDict(from_attributes=True)


class LLMConfig(LMBaseConfig):
    pass

class VLMConfig(LMBaseConfig):
    pass

class BailianConfig(BaseModel):
    base_url: str
    api_key: str
    model_config = ConfigDict(from_attributes=True)

class QueryAgentConfig(BaseModel):
    web_search_enable: bool
    model_config = ConfigDict(from_attributes=True)

class RAGAppConfig(BaseModel):
    mineru: MineruConfig
    logging: LoggingConfig
    minio: MinioConfig
    milvus: MilvusConfig
    llm: LLMConfig
    vlm: VLMConfig
    bailian: BailianConfig
    postgre: PostgreConfig
    query_agent: QueryAgentConfig
    model_config = ConfigDict(from_attributes=True)


config_path = Path(__file__).parent / "app_config.yaml"
context = OmegaConf.load(config_path)
app_config: RAGAppConfig = RAGAppConfig.model_validate(context)


if __name__ == "__main__":
    print(app_config)
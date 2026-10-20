from pydantic import BaseModel, Field, ConfigDict
from typing import Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing import Literal
from repositories.milvus import MilvusChunkRepository, MilvusEntityRepository
from repositories.milvus import MilvusSearchEntity, MilvusSearchChunk
from dtos.rerank import ReRankedChunk

class QueryWebSearchChunk(BaseModel):
    title: str
    content: str
    url: str

class QueryGraphState(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    rewritten_query: str | None = None
    entities: list[MilvusSearchEntity] | None = None
    embedding_chunks: list[MilvusSearchChunk] | None = None
    hyde_chunks: list[MilvusSearchChunk] | None = None
    web_chunks: list[QueryWebSearchChunk] | None = None
    rrf_chunks: list[MilvusSearchChunk] | None = None
    reranked_chunks: list[ReRankedChunk] | None = None
    should_continue: bool = True
    error: str | None = None


class QueryGraphContext(BaseModel):
    milvus_entity_repo: MilvusEntityRepository
    milvus_chunk_repo: MilvusChunkRepository

    model_config = ConfigDict(arbitrary_types_allowed=True)


class QueryGraphStepInfo(BaseModel):
    name: str
    status: Literal['running', 'success', 'failed']
    error: str | None = None


class QuerySourceInfo(BaseModel):
    # 标题：如果是file，此时title=file_name，如果是web，此时title=web title
    title: str
    url: str
    source: Literal['web', 'file']
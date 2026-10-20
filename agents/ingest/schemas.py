from pydantic import BaseModel, ConfigDict
from pathlib import Path
from typing import Literal
from repositories.milvus import MilvusEntityRepository, MilvusChunkRepository


class IngestMarkdownChunk(BaseModel):
    file_name: str
    title: str
    content: str
    header_chunk_index: int

class IngestGraphState(BaseModel):
    file_url: str
    file_path: Path
    markdown_dir: Path
    markdown_path: Path| None = None
    markdown_content: str | None = None
    markdown_chunks: list[IngestMarkdownChunk] | None = None
    entity_name: str | None = None

    should_continue: bool = True
    error: str | None = None


class IngestGraphContext(BaseModel):
    milvus_entity_repo: MilvusEntityRepository
    milvus_chunk_repo: MilvusChunkRepository

    model_config = ConfigDict(arbitrary_types_allowed=True)


class IngestGraphStepInfo(BaseModel):
    name: str
    status: Literal['running','success', 'failed']
    error: str | None = None

    
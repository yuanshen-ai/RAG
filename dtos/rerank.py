from pydantic import BaseModel, Field
from typing import Literal, Annotated


class ReRankChunk(BaseModel):
    title: str
    content: str
    # 仅适用于Web中的chunk
    url: str | None = None
    # 仅适用于向量数据库中的chunk
    file_name: str | None = None
    # 仅适用于向量数据库中的chunk
    file_url: str | None = None
    # 仅适用于向量数据库中的chunk
    chunk_id: int | None = None
    source: Literal['web', 'embedding']


class ReRankedChunk(ReRankChunk):
    score: float = Field(..., description="重排序模型计算的得分")
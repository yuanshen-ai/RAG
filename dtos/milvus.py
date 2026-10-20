from pydantic import BaseModel

class MilvusInsertEntity(BaseModel):
    file_name: str
    entity_name: str
    dense_vector: list[float] | None = None
    # 格式：{维度索引: 特征值}
    # sparse_vector = {
    #     0: 0.5,      # 第0个特征的值为0.5
    #     5: 1.2,      # 第5个特征的值为1.2
    #     100: -0.8    # 第100个特征的值为-0.8
    # }
    sparse_vector: dict[int, float] | None = None


class MilvusSearchEntity(BaseModel):
    id: int
    distance: float
    entity_name: str


class MilvusInsertChunk(BaseModel):
    file_url: str
    file_name: str
    title: str
    content: str
    header_chunk_index: int
    entity_name: str
    dense_vector: list[float] | None = None
    sparse_vector: dict[int, float] | None = None


class MilvusSearchChunk(BaseModel):
    id: int
    distance: float
    title: str
    content: str
    entity_name: str
    file_name: str
    file_url: str
from pymilvus import AsyncMilvusClient, DataType, AnnSearchRequest, WeightedRanker
from dtos.milvus import MilvusInsertEntity, MilvusSearchEntity, MilvusInsertChunk, MilvusSearchChunk
from utils.l2_normalize import l2_normalize
import json


class MilvusEntityRepository:
    collection_name = "entity_collection"

    def __init__(self, client: AsyncMilvusClient) -> None:
        self.client = client

    async def clear_collection(self):
        has_collection = await self.client.has_collection(self.collection_name)
        if has_collection:
            await self.client.drop_collection(self.collection_name)
        
    async def ensure_collection(self):
        has_collection = await self.client.has_collection(self.collection_name)
        if has_collection:
            return
        
        # 1. 创建字段
        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=True)
        # 主键
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        # 文件名
        schema.add_field(field_name="file_name", datatype=DataType.VARCHAR, max_length=255)
        # 实体名
        schema.add_field(field_name="entity_name", datatype=DataType.VARCHAR, max_length=255)
        # 稠密向量
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        # 稀疏向量
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        # 2. 创建索引
        index_params = self.client.prepare_index_params()
        # 为稠密向量添加索引
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200}
        )
        # 为稀疏向量添加索引
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            # IP：内积
            metric_type="IP",
            params = {"inverted_index_algo": "DAAT_MAXSCORE", "normalize": True, "quantization": "none"}
        )

        # 3. 创建集合
        await self.client.create_collection(collection_name=self.collection_name, schema=schema, index_params=index_params)
    
    async def add_entity(self, entity: MilvusInsertEntity):
        await self.ensure_collection()
        # 先删除同名实体
        await self.client.delete(
            self.collection_name, 
            filter=f'file_name == "{entity.file_name}" && entity_name == "{entity.entity_name}"'
        )
        # 对稀疏向量做归一化处理
        entity.sparse_vector = {
            k: v 
            for k, v in zip(entity.sparse_vector.keys(), l2_normalize(list(entity.sparse_vector.values())))
        }
        # 将entity添加到Milvus中
        await self.client.insert(collection_name=self.collection_name, data=[entity.model_dump()])
        # 强制加载
        await self.client.load_collection(self.collection_name)

    async def search_entity(
        self,
        dense_vectors: list[list[float]],
        sparse_vectors: list[dict[int, float]],
        limit: int = 5
    ) -> list[MilvusSearchEntity]:
        await self.ensure_collection()
        # 构建request对象
        dense_request = AnnSearchRequest(
            data=dense_vectors,
            anns_field="dense_vector",
            param={"metric_type": "COSINE"},
            limit=limit
        )
        sparse_request = AnnSearchRequest(
            data=sparse_vectors,
            anns_field="sparse_vector",
            param={"metric_type": "IP"},
            limit=limit
        )
        # 构建排序对象
        ranker = WeightedRanker(0.8, 0.2, norm_score=True)

        # 执行混合搜索
        result = await self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_request, sparse_request],
            ranker=ranker,
            limit=limit,
            output_fields=['entity_name']
        )

        items: list[MilvusSearchEntity] = []
        for entities in result:
            for entity in entities:
                items.append(MilvusSearchEntity(
                    id=entity['id'],
                    distance=entity['distance'],
                    entity_name=entity['entity']['entity_name']
                ))
        return items


class MilvusChunkRepository:
    collection_name = "chunk_collection"
    def __init__(self, client: AsyncMilvusClient):
        self.client = client

    async def ensure_collection(self):
        has_collection = await self.client.has_collection(self.collection_name)
        if has_collection:
            return
        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=True)
        # 主键
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        # 文件url
        schema.add_field(field_name="file_url", datatype=DataType.VARCHAR, max_length=1000)
        # 文件名
        schema.add_field(field_name="file_name", datatype=DataType.VARCHAR, max_length=255)
        # 标题
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=65535)
        # 内容
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        # chunk索引号
        schema.add_field(field_name="header_chunk_index", datatype=DataType.INT8)
        # entity_name
        schema.add_field(field_name="entity_name", datatype=DataType.VARCHAR, max_length=255)
        # 稠密向量
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        # 稀疏向量
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        # 创建索引
        index_params = self.client.prepare_index_params()
        # 为稠密向量构建索引
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200}
        )
        # 为稀疏向量构建索引
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            # IP：内积
            metric_type="IP",
            params = {"inverted_index_algo": "DAAT_MAXSCORE", "normalize": True, "quantization": "none"}
        )

        await self.client.create_collection(collection_name=self.collection_name, schema=schema, index_params=index_params)
    
    async def clear_collection(self):
        has_collection = await self.client.has_collection(self.collection_name)
        if has_collection:
            await self.client.drop_collection(self.collection_name)

    async def add_chunks(self, chunks: list[MilvusInsertChunk]):
        await self.ensure_collection()
        # 批量插入chunks
        chunk_size: int = 200
        for index in range(0, len(chunks), chunk_size):
            batch_chunks = chunks[index:index+chunk_size]
            chunk_dicts = [chunk.model_dump() for chunk in batch_chunks]
            await self.client.insert(collection_name=self.collection_name, data=chunk_dicts)
        await self.client.load_collection(self.collection_name)

    async def search_chunks(self, 
        dense_vectors: list[list[float]], 
        sparse_vectors: list[dict[int, float]],
        entity_names: list[str],
        limit: int=10
    ):
        await self.ensure_collection()
        expr = f'entity_name in {json.dumps(entity_names, ensure_ascii=False)}'
        # 构建稠密向量搜索请求
        dense_request = AnnSearchRequest(
            data=dense_vectors,
            anns_field="dense_vector",
            param={"metric_type": "COSINE"},
            expr=expr,
            limit=limit
        )
        # 构建稀疏向量搜索请求
        sparse_request = AnnSearchRequest(
            data=sparse_vectors,
            anns_field="sparse_vector",
            param={"metric_type": "IP"},
            expr=expr,
            limit=limit
        )

        ranker = WeightedRanker(0.5, 0.5, norm_score=True)
        result = await self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_request, sparse_request],
            ranker=ranker,
            limit=limit,
            output_fields=['id', "title", "content", "entity_name", "file_name", "file_url"]
        )
        items: list[MilvusSearchChunk] = []
        for chunks in result:
            for chunk in chunks:
                items.append(
                    MilvusSearchChunk(
                        id=chunk['id'],
                        distance=chunk['distance'],
                        title=chunk['entity']['title'],
                        content=chunk['entity']['content'],
                        entity_name=chunk['entity']['entity_name'],
                        file_name=chunk['entity']['file_name'],
                        file_url=chunk['entity']['file_url']
                    )
                )
        return items

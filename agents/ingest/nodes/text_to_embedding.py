import asyncio

from agents.ingest.schemas import IngestGraphState, IngestGraphContext, IngestGraphStepInfo
from langgraph.runtime import Runtime
from integrations.embedding import generate_texts_embeddings
from dtos.milvus import MilvusInsertChunk
from repositories.milvus import MilvusChunkRepository


# text-embedding-v4 官方限制为单次最多 10 条，通过并发批次提速。
EMBEDDING_BATCH_SIZE = 10
EMBEDDING_CONCURRENCY = 4


async def text_to_embedding(state: IngestGraphState, runtime: Runtime[IngestGraphContext]):
    writer = runtime.stream_writer
    writer(IngestGraphStepInfo(name="text_to_embedding", status="running"))

    try:
        # 1. 处理markdown_chunks
        assert state.markdown_chunks is not None, "markdown chunks is None"
        texts = [f"实体名：{state.entity_name}，介绍：{chunk.content}" for chunk in state.markdown_chunks]

        # 2. 生成嵌入向量
        semaphore = asyncio.Semaphore(EMBEDDING_CONCURRENCY)

        async def embed_batch(batch_texts: list[str]):
            async with semaphore:
                batch_embeddings = await generate_texts_embeddings(batch_texts)
            if batch_embeddings is None or len(batch_embeddings) != len(batch_texts):
                raise RuntimeError(
                    f"向量接口返回数量异常：请求 {len(batch_texts)} 条，"
                    f"返回 {len(batch_embeddings or [])} 条"
                )
            return batch_embeddings

        batches = [
            texts[index:index + EMBEDDING_BATCH_SIZE]
            for index in range(0, len(texts), EMBEDDING_BATCH_SIZE)
        ]
        batch_results = await asyncio.gather(*(embed_batch(batch) for batch in batches))
        embeddings: list[tuple[list[float], dict[int, float]]] = []
        for batch_embeddings in batch_results:
            for vector in batch_embeddings:
                embeddings.append((vector.get("dense"), vector.get("sparse"))) #type: ignore
        
        # 3. 将向量写入milvus数据库
        milvus_chunk_repo: MilvusChunkRepository = runtime.context.milvus_chunk_repo
        await milvus_chunk_repo.add_chunks(
            [
                MilvusInsertChunk(
                    file_name=state.file_path.name,
                    file_url=state.file_url,
                    title=chunk.title,
                    content=chunk.content,
                    header_chunk_index=chunk.header_chunk_index,
                    entity_name=state.entity_name, #type: ignore
                    dense_vector=embedding[0],
                    sparse_vector=embedding[1]
                )
                for chunk, embedding in zip(state.markdown_chunks, embeddings)
            ]
        )
    except Exception as e:
        writer(IngestGraphStepInfo(name="text_to_embedding", status="failed"))
        return {"should_continue": False, "error": str(e)}

    writer(IngestGraphStepInfo(name="text_to_embedding", status="success"))

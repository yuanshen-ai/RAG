from agents.query.schemas import QueryGraphState, QueryGraphContext, QueryGraphStepInfo
from langgraph.runtime import Runtime
from dtos.milvus import MilvusSearchChunk
from core.log import logger
from agents.query.stream_events import step_event


async def rrf_merge(state: QueryGraphState, runtime: Runtime[QueryGraphContext]):
    writer = runtime.stream_writer
    writer(step_event(QueryGraphStepInfo(name="rrf_merge", status="running")))

    embedding_chunks = state.embedding_chunks
    hyde_chunks = state.hyde_chunks

    assert embedding_chunks is not None, "embeddnig_chunks is None"
    assert hyde_chunks is not None, "hyde_chunks is None"

    try:
        # 1. 对原始chunk进行排序
        embedding_chunks.sort(key=lambda chunk: chunk.distance, reverse=True)
        hyde_chunks.sort(key=lambda chunk: chunk.distance, reverse=True)

        rrf_chunks: dict[int, tuple[MilvusSearchChunk, float]] = {}
        k = 60
        weights = (1.0, 1.0)

        # 2. 计算每个chunk的rrf_score
        for index, embedding_chunk in enumerate(embedding_chunks, start=1):
            rrf_score = (1 / (k + index)) * weights[0]
            rrf_chunks[embedding_chunk.id] = (embedding_chunk, rrf_score)
        
        for index, hyde_chunk in enumerate(hyde_chunks, start=1):
            rrf_score = (1 / (k + index)) * weights[1]
            if hyde_chunk.id in rrf_chunks:
                existing_score = rrf_chunks[hyde_chunk.id][1]
                current_score = rrf_score + existing_score
                rrf_chunks[hyde_chunk.id] = (hyde_chunk, current_score)
            else:
                rrf_chunks[hyde_chunk.id] = (hyde_chunk, rrf_score)
        
        chunks = [chunk for chunk, _ in sorted(rrf_chunks.values(), key=lambda item: item[1], reverse=True)][:10]
    except Exception as e:
        writer(step_event(QueryGraphStepInfo(name="rrf_merge", status="failed", error=str(e))))
        return {"should_continue": False, "error": str(e)}
    
    # logger.info(f"rrf chunks: {chunks}")

    writer(step_event(QueryGraphStepInfo(name="rrf_merge", status="success")))
    return {"rrf_chunks": chunks}
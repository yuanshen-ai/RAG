from agents.query.schemas import QueryGraphState, QueryGraphContext, QueryGraphStepInfo
from langgraph.runtime import Runtime
from dtos.rerank import ReRankChunk, ReRankedChunk
from integrations.rerank import rerank_documents
from core.log import logger
from agents.query.stream_events import step_event


async def chunks_rerank(state: QueryGraphState, runtime: Runtime[QueryGraphContext]):
    writer = runtime.stream_writer
    writer(step_event(QueryGraphStepInfo(name="chunks_rerank", status="running")))

    # 1. 将网络搜索和embedding的chunk合并起来
    web_chunks = state.web_chunks
    rrf_chunks = state.rrf_chunks
    query = state.rewritten_query

    assert web_chunks is not None, "web_chunks is None"
    assert rrf_chunks is not None, "rrf_chunks is None"
    assert query is not None, "rewritten_query is None"

    merged_chunks: list[ReRankChunk] = []
    for web_chunk in web_chunks:
        merged_chunks.append(ReRankChunk(
            title=web_chunk.title,
            content=web_chunk.content,
            url=web_chunk.url,
            source="web"
        ))
    for rrf_chunk in rrf_chunks:
        merged_chunks.append(ReRankChunk(
            title=rrf_chunk.title,
            content=rrf_chunk.content,
            chunk_id=rrf_chunk.id,
            file_name=rrf_chunk.file_name,
            file_url=rrf_chunk.file_url,
            source="embedding"
        ))
    
    # 2. 获取所有chunk的重排序得分
    try:
        reranked_chunks: list[ReRankedChunk] | None = await rerank_documents(query, merged_chunks)
        if reranked_chunks is not None:
            for chunk in reranked_chunks:
                logger.info(f"content: {chunk.content}, score: {chunk.score}")
    except Exception as e:
        writer(step_event(QueryGraphStepInfo(name="chunks_rerank", status="failed", error=str(e))))
        return {'should_continue': False, "error": str(e)}
    
    # 3. 取topK: 结合固定上下限+断崖式阈值判断,避免机械取前N条,保留语义相关的连续文档集合
    max_topk = 10
    min_topk = 1
    # A: 180, B: 150
    # gap_abs = (A-B) = 30
    max_gap_abs = 0.5
    # gap_ratio = (A-B) / A = (180-150) / 180 = 30 / 180
    max_gap_ratio = 0.25
    # k: 保存取到前k个
    k = 0
    assert reranked_chunks is not None, "reranked_chunks is None"
    for index in range(0, len(reranked_chunks)-1):
        current_chunk = reranked_chunks[index]
        next_chunk = reranked_chunks[index+1]
        # 求绝对gap
        gap_abs = (current_chunk.score - next_chunk.score)
        gap_ratio = gap_abs / (current_chunk.score + 1e-6)
        if gap_abs > max_gap_abs:
            break
        # 求gap比例
        elif gap_ratio > max_gap_ratio:
            break
        else:
            k += 1
    
    final_chunks = reranked_chunks[0:min(max(k, min_topk), max_topk)]
    # logger.info(f"final chunks: {final_chunks}")

    writer(step_event(QueryGraphStepInfo(name="chunks_rerank", status="success")))
    return {"reranked_chunks": final_chunks}


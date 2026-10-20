from agents.query.schemas import QueryGraphState, QueryGraphContext, QueryGraphStepInfo
from langgraph.runtime import Runtime
from integrations.embedding import generate_texts_embeddings
from core.log import logger
from agents.query.stream_events import step_event


async def embedding_search(state: QueryGraphState, runtime: Runtime[QueryGraphContext]):
    writer = runtime.stream_writer
    writer(step_event(QueryGraphStepInfo(name="embedding_search", status="running")))

    query = state.rewritten_query
    entities = state.entities

    try:
        # 1. 将用户的问题转换成向量
        assert query is not None, "rewritten_query is None"
        query_embeddings = await generate_texts_embeddings([query])
        dense_vector = query_embeddings[0].get("dense") #type: ignore
        sparse_vector = query_embeddings[0].get("sparse") #type: ignore

        # 2. 从向量数据库中查找数据
        milvus_chunk_repo = runtime.context.milvus_chunk_repo
        chunks = await milvus_chunk_repo.search_chunks(
            dense_vectors=[dense_vector], #type: ignore
            sparse_vectors=[sparse_vector], #type: ignore
            entity_names=[entity.entity_name for entity in entities] #type: ignore
        )
    except Exception as e:
        writer(step_event(QueryGraphStepInfo(name="embedding_search", status="failed", error=str(e))))
        return {"should_continue": False, "error": str(e)}

    # logger.info(f"检索到的chunks：{chunks}")

    writer(step_event(QueryGraphStepInfo(name="embedding_search", status="success")))
    return {"embedding_chunks": chunks}




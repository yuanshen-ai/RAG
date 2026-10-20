from agents.query.schemas import QueryGraphState, QueryGraphContext, QueryGraphStepInfo
from langgraph.runtime import Runtime
from agents.ainvoke_llm import ainvoke_llm_str
from integrations.embedding import generate_texts_embeddings
from core.log import logger
from agents.query.stream_events import step_event


async def hyde_search(state: QueryGraphState, runtime: Runtime[QueryGraphContext]):
    writer = runtime.stream_writer
    writer(step_event(QueryGraphStepInfo(name="hyde_search", status="running")))

    query = state.rewritten_query
    entities = state.entities

    try:
        # 1. 生成hyde
        hyde = await ainvoke_llm_str("hyde_generate", {"query": query})

        # 2. 将hyde生成向量
        hyde_embeddings = await generate_texts_embeddings([hyde])
        dense_vector = hyde_embeddings[0].get("dense") #type: ignore
        sparse_vector = hyde_embeddings[0].get("sparse") #type: ignore

        # 3. 从向量数据库中检索数据
        milvus_chunk_repo = runtime.context.milvus_chunk_repo
        chunks = await milvus_chunk_repo.search_chunks(
            dense_vectors=[dense_vector],
            sparse_vectors=[sparse_vector],
            entity_names=[entity.entity_name for entity in entities] #type: ignore
        )
    except Exception as e:
        writer(step_event(QueryGraphStepInfo(name="hyde_search", status="failed", error=str(e))))
        return {"should_continue": False, "error": str(e)}
    
    # logger.info(f"hyde: {hyde}")
    # logger.info(f"chunk: {chunks}")

    writer(step_event(QueryGraphStepInfo(name="hyde_search", status="success")))
    return {"hyde_chunks": chunks}




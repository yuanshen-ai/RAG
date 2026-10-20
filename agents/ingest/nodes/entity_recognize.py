from agents.ingest.schemas import IngestGraphState, IngestGraphContext, IngestGraphStepInfo
from langgraph.runtime import Runtime
from agents.ainvoke_llm import ainvoke_llm_str
from integrations.embedding import generate_texts_embeddings
from dtos.milvus import MilvusInsertEntity
from core.log import logger


async def entity_recognize(state: IngestGraphState, runtime: Runtime[IngestGraphContext]):
    writer = runtime.stream_writer
    writer(IngestGraphStepInfo(name="entity_recognize", status="running"))

    try:
        # 提取entity_name
        top_chunks = state.markdown_chunks[:5]
        chunks_content = "\n\n".join(f"{chunk.title}\n{chunk.content}" for chunk in top_chunks)

        # 调用的提取实体
        entity_name = await ainvoke_llm_str("recognize_entity", {"file_name": state.file_path.name, "context": chunks_content})
        embedding_result = await generate_texts_embeddings(texts=[entity_name])

        dense_embedding = embedding_result[0].get('dense')
        sparse_embedding = embedding_result[0].get('sparse')

        milvus_entity_repo = runtime.context.milvus_entity_repo

        # 把eneity及其向量添加到Milvus中
        await milvus_entity_repo.add_entity(
            MilvusInsertEntity(
                file_name=state.file_path.name,
                entity_name=entity_name,
                dense_vector=dense_embedding,
                sparse_vector=sparse_embedding
            )
        )
    except Exception as e:
        writer(IngestGraphStepInfo(name="entity_recognize", status="failed", error=str(e)))
        return {"should_continue": False, "error": str(e)}
    
    logger.info(f"entity_name: {entity_name}")

    writer(IngestGraphStepInfo(name="entity_recognize", status="success"))
    return {"entity_name": entity_name}
    

    
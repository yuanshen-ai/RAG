from agents.query.schemas import QueryGraphState, QueryGraphContext, QueryGraphStepInfo
from langgraph.runtime import Runtime
from agents.ainvoke_llm import ainvoke_llm_json
from integrations.embedding import generate_texts_embeddings
from repositories.milvus import MilvusEntityRepository
from dtos.milvus import MilvusSearchEntity
from core.log import logger
from langchain.messages import AIMessage
from agents.query.stream_events import step_event


async def entity_confirm(state: QueryGraphState, runtime: Runtime[QueryGraphContext]):
    writer = runtime.stream_writer
    writer(step_event(QueryGraphStepInfo(name="entity_confirm", status="running")))

    history_messages = state.messages[0:-1]

    result = await ainvoke_llm_json(
        "entity_parse",
        {
            "history_text": [{"role": message.type, "content": message.content} for message in history_messages],
            "query": state.query
        },
    )

    entity_names = result['entity_names']
    rewritten_query = result['rewritten_query']

    if len(entity_names) == 0:
        error = "无法从您的描述中识别到产品名称！"
        writer(QueryGraphStepInfo(name="entity_confirm", status="failed", error=error))
        return {"should_continue": False, "error": error}

    # [{"dense": [1,0.2, ...], "sparse": {2, [123,.12.,]}}]
    entity_embeddings = await generate_texts_embeddings(entity_names)

    assert entity_embeddings is not None, "entity_embeddings is None"

    dense_vectors: list[list[float]] = []
    sparse_vectors: list[dict[int, float]] = []
    for entity_embedding in entity_embeddings:
        dense_vectors.append(entity_embedding.get("dense")) #type: ignore
        sparse_vectors.append(entity_embedding.get("sparse")) #type: ignore
    
    milvus_entity_repo: MilvusEntityRepository = runtime.context.milvus_entity_repo
    entities: list[MilvusSearchEntity] = await milvus_entity_repo.search_entity(
        dense_vectors=dense_vectors,
        sparse_vectors=sparse_vectors
    )

    # 1. 去重
    seen = set()
    for entity in entities[:]:
        if entity.entity_name not in seen:
            seen.add(entity.entity_name)
        else:
            entities.remove(entity)
    
    # 2. 根据distance进行排序
    # distance: 这个值越大，代表搜索的结果越准确
    entities.sort(key=lambda entity: entity.distance, reverse=True)

    # 3. 筛选
    entities = list(filter(lambda entity: entity.distance > 0.65, entities))

    logger.info(f"查找到的实体有：{entities}")

    if len(entities) == 0:
        error = "没有找到合适的产品！"
        writer(step_event(QueryGraphStepInfo(name="entity_confirm", status="failed", error=error)))
        return {"should_continue": False, "error": error}
    else:
        writer(step_event(QueryGraphStepInfo(name="entity_confirm", status="success")))
        return {
            "entities": entities,
            "rewritten_query": rewritten_query,
            "messages": [AIMessage(content=
                f"查找到的产品名称为：{[entity.model_dump() for entity in entities]}，改写后的问题为：{rewritten_query}"
            )]
        }

    




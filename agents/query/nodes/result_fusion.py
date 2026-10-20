from agents.query.schemas import QueryGraphState, QueryGraphContext, QueryGraphStepInfo, QuerySourceInfo
from langgraph.runtime import Runtime
from agents.ainvoke_llm import astream_llm_str
from agents.query.stream_events import step_event, answer_delta_event, sources_event
from langchain.messages import AIMessage
from core.log import logger

async def result_fusion(state: QueryGraphState, runtime: Runtime[QueryGraphContext]):
    writer = runtime.stream_writer
    writer(step_event(QueryGraphStepInfo(name="result_fusion", status="running")))

    reranked_chunks = state.reranked_chunks
    entities = state.entities
    messages = state.messages

    assert reranked_chunks is not None
    assert entities is not None

    sources: list[QuerySourceInfo] = [
        QuerySourceInfo(
            title=(chunk.title if chunk.source == 'web' else chunk.file_name) or "",
            url=(chunk.url if chunk.source == "web" else chunk.file_url) or "",
            source="web" if chunk.source == 'web' else 'file'
        )
        for chunk in reranked_chunks
    ]
    writer(sources_event(sources))

    final_answer = ""
    async for delta in astream_llm_str(
        "result_fusion",
        {
            "chunks": [chunk.model_dump() for chunk in reranked_chunks],
            "messages": [{"role": message.type, "content": message.content} for message in messages],
            "entity_names": [entity.entity_name for entity in entities],
            "query": state.rewritten_query
        }
    ):
        final_answer += delta
        writer(answer_delta_event(delta=delta))

    writer(step_event(QueryGraphStepInfo(name="result_fusion", status="success")))

    logger.info(f"最終的回復：{final_answer}")

    return {"messages": [AIMessage(content=final_answer)]}


from agents.query.schemas import QueryGraphState, QueryGraphContext, QueryGraphStepInfo
from langgraph.runtime import Runtime
from agents.ainvoke_llm import ainvoke_llm_json
from core.log import logger
from agents.query.stream_events import step_event

async def intention_parse(state: QueryGraphState, runtime: Runtime[QueryGraphContext]):
    writer = runtime.stream_writer
    writer(step_event(QueryGraphStepInfo(name="intention_parse", status="running")))

    history_messages = state.messages[0:-1]

    try:
        result = await ainvoke_llm_json("intention_parse", {
            "messages": [{"role": message.type, "content": message.content} for message in history_messages],
            "query": state.query
        })
    except Exception as e:
        writer(step_event(QueryGraphStepInfo(name="intention_parse", status="failed")))
        return {"should_continue": False, "error": str(e)}

    logger.info(result)
    
    writer(step_event(QueryGraphStepInfo(name="intention_parse", status="success")))
    return {"should_continue": result['should_continue']}
    

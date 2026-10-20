from agents.query.schemas import QueryGraphState, QueryGraphContext, QueryGraphStepInfo, QueryWebSearchChunk
from langgraph.runtime import Runtime
from langchain_mcp_adapters.client import MultiServerMCPClient, StreamableHttpConnection
from typing import cast
from conf import app_config
from core.log import logger
import json
from agents.query.stream_events import step_event


async def web_search(state: QueryGraphState, runtime: Runtime[QueryGraphContext]):
    writer = runtime.stream_writer
    writer(step_event(QueryGraphStepInfo(name="web_search", status="running")))

    if not app_config.query_agent.web_search_enable:
        writer(step_event(QueryGraphStepInfo(name="web_search", status="success")))
        return {"should_continue": True, "web_chunks": []}

    query = state.rewritten_query

    try:
        client = MultiServerMCPClient(
            {
                "WebSearch": cast(
                    StreamableHttpConnection,
                    {
                        "transport": "http",
                        "url": "https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/mcp",
                        "headers": {
                            "Authorization": f"Bearer {app_config.bailian.api_key}"
                        }
                    }
                ),
            }
        )
        tools = await client.get_tools()
        web_chunks: list[QueryWebSearchChunk] = []
        for tool in tools:
            if tool.name == "bailian_web_search":
                search_results = await tool.ainvoke({"query": query})
                search_result = search_results[0]
                pages = json.loads(search_result['text']).get("pages")
                for page in pages:
                    title = page.get("title")
                    content = page.get("snippet")
                    url = page.get("url")
                    web_chunks.append(
                        QueryWebSearchChunk(
                            title=title,
                            content=content,
                            url=url
                        )
                    )
    except Exception as e:
        writer(step_event(QueryGraphStepInfo(name="web_search", status="failed", error=str(e))))
        return {"should_continue": False, "error": str(e)}
                
    # logger.info(f"web chunks: {web_chunks}")

    writer(step_event(QueryGraphStepInfo(name="web_search", status="success")))
    return {"web_chunks": web_chunks}
